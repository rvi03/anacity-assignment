"""The scorekeeper — one definition of "correct", used by both tracks.

Four outputs are predicted per record, and each is matched by its own
rule:

    facility       exact
    usage weekday  exact
    usage hour     exact
    notification   within a symmetric multiplicative tolerance,
                   `|log((pred+1)/(actual+1))| <= log(ratio)`

The notification rule is relative because that head predicts a
multi-day interval. A fixed +/-30-minute window would be false almost
everywhere and would quietly turn a four-part score into a three-part
one. Mean absolute error stays the primary notification metric; the
fixed-minute rates beside it are diagnostics and never select anything.

Matches roll up per record and over the holdout:

    SCORE          components matched, 0..4
    OVERALL        mean(SCORE) / 4
    STRICT         share of records matching all four
    AT LEAST THREE share matching three or more
    lift           model rate against a baseline rate, relative and
                   absolute; relative is undefined, never divided, when
                   the baseline denominator is zero

The match uses the submitted non-negative operational delay. Raw delay
metrics and the clamp rate are reported beside it, so clamping cannot
hide an unstable regression output.

Leakage contract: every function here reads the truth of a row that has
already been predicted. Nothing it computes returns to a feature, a
fit, or a prediction, and no function reads a row other than the one it
is scoring.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import math
from typing import Any

import numpy as np
import pandas as pd

FACILITY = "facility"
USAGE_WEEKDAY = "usage_weekday"
USAGE_HOUR = "usage_hour"
NOTIFICATION = "notification"
COMPONENTS = (FACILITY, USAGE_WEEKDAY, USAGE_HOUR, NOTIFICATION)

TARGET_COLUMNS = {
    FACILITY: "target_facility_id",
    USAGE_WEEKDAY: "target_usage_weekday",
    USAGE_HOUR: "target_usage_hour",
    NOTIFICATION: "notification_delay_minutes",
}
PREDICTED_COLUMNS = {
    FACILITY: "predicted_facility_id",
    USAGE_WEEKDAY: "predicted_usage_weekday",
    USAGE_HOUR: "predicted_usage_hour",
    NOTIFICATION: "predicted_delay_minutes",
}

SCORE_COLUMN = "score"

_HOURS_IN_DAY = 24
_HOUR_TOLERANCE = 1
_TOP_K = 3
_P90 = 90
_LOG_RATIO_OFFSET = 1.0
_STRICT_SCORE = len(COMPONENTS)
_NEAR_MISS_SCORE = _STRICT_SCORE - 1


@dataclasses.dataclass(frozen=True)
class Comparison:
    """A model quantity set against the baseline's.

    Attributes:
        model: The model's value.
        baseline: The baseline's value.
        absolute: Model minus baseline, always defined.
        relative: The relative change, or None when the baseline
            denominator is zero and the ratio is undefined.
    """

    model: float
    baseline: float
    absolute: float
    relative: float | None


def _require_rows(frame: pd.DataFrame | pd.Series, what: str) -> None:
    """Raise when there is nothing to score.

    An empty input would produce a NaN that reads like a measured
    zero.

    Args:
        frame: The rows to score.
        what: What was empty, for the message.

    Raises:
        ValueError: If ``frame`` has no rows.
    """
    if len(frame) == 0:
        msg = f"cannot evaluate an empty {what}"
        raise ValueError(msg)


def _labels(actual: pd.Series, predicted: pd.Series) -> list[Any]:
    """Return the label set metrics are averaged over.

    Both sides contribute: a class the model invents but never occurs
    still costs precision.

    Args:
        actual: Observed labels.
        predicted: Predicted labels.

    Returns:
        The sorted union of the labels seen on either side.
    """
    seen = set(actual.dropna().unique()) | set(predicted.dropna().unique())
    return sorted(seen)


def accuracy(actual: pd.Series, predicted: pd.Series) -> float:
    """Return the share of rows whose label matches exactly.

    Args:
        actual: Observed labels.
        predicted: Predicted labels.

    Returns:
        Share in ``[0, 1]``.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(actual, "prediction set")
    return float((actual.to_numpy() == predicted.to_numpy()).mean())


def per_class_recall(
    actual: pd.Series, predicted: pd.Series
) -> dict[Any, float]:
    """Return recall for every class that actually occurs.

    A class with no observed rows has no recall to report, so it is
    absent rather than recorded as zero.

    Args:
        actual: Observed labels.
        predicted: Predicted labels.

    Returns:
        Class label to recall in ``[0, 1]``.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(actual, "prediction set")
    hit = actual.to_numpy() == predicted.to_numpy()
    recalls = {}
    for label in sorted(set(actual.dropna().unique())):
        occurs = actual.to_numpy() == label
        recalls[label] = float(hit[occurs].mean())
    return recalls


def macro_f1(actual: pd.Series, predicted: pd.Series) -> float:
    """Return the unweighted mean F1 over every label seen.

    A label with a zero denominator scores zero rather than being
    dropped, so a model that never predicts a rare class is penalised
    for it.

    Args:
        actual: Observed labels.
        predicted: Predicted labels.

    Returns:
        Macro-F1 in ``[0, 1]``.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(actual, "prediction set")
    actual_values = actual.to_numpy()
    predicted_values = predicted.to_numpy()
    scores = []
    for label in _labels(actual, predicted):
        true_positive = float(
            ((predicted_values == label) & (actual_values == label)).sum()
        )
        predicted_positive = float((predicted_values == label).sum())
        actual_positive = float((actual_values == label).sum())
        denominator = predicted_positive + actual_positive
        scores.append(
            0.0 if denominator == 0 else 2.0 * true_positive / denominator
        )
    return float(np.mean(scores))


def top_k_accuracy(
    actual: pd.Series, probabilities: pd.DataFrame, k: int = _TOP_K
) -> float:
    """Return the share of rows whose label is among the ``k`` likeliest.

    Args:
        actual: Observed labels.
        probabilities: One column per class label, one row per record,
            in the same row order as ``actual``.
        k: How many classes count as a hit.

    Returns:
        Share in ``[0, 1]``.

    Raises:
        ValueError: If there are no rows, if the shapes disagree, or if
            an observed label has no column.
    """
    _require_rows(actual, "prediction set")
    if len(actual) != len(probabilities):
        msg = (
            f"{len(actual)} labels against {len(probabilities)} probability "
            "rows"
        )
        raise ValueError(msg)
    unknown = set(actual.dropna().unique()) - set(probabilities.columns)
    if unknown:
        msg = f"no probability column for observed labels {sorted(unknown)}"
        raise ValueError(msg)

    columns = list(probabilities.columns)
    values = probabilities.to_numpy()
    # stable order so equal probabilities break ties by column order,
    # which keeps a rerun's number identical
    ranked = np.argsort(-values, axis=1, kind="stable")[:, :k]
    positions = {label: index for index, label in enumerate(columns)}
    wanted = np.array([positions[label] for label in actual])
    return float((ranked == wanted[:, None]).any(axis=1).mean())


def confusion_matrix(actual: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    """Return observed labels against predicted ones, counts in cells.

    Args:
        actual: Observed labels.
        predicted: Predicted labels.

    Returns:
        A square frame indexed by observed label, columns predicted,
        covering every label seen on either side.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(actual, "prediction set")
    labels = _labels(actual, predicted)
    counted = pd.crosstab(
        pd.Series(actual.to_numpy(), name="actual"),
        pd.Series(predicted.to_numpy(), name="predicted"),
    )
    return counted.reindex(index=labels, columns=labels, fill_value=0)


def classification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    probabilities: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the metrics reported for one classification head.

    Top-3 accuracy appears only when probabilities are supplied.

    Args:
        actual: Observed labels.
        predicted: Predicted labels.
        probabilities: Per-class probabilities, when the head has them.

    Returns:
        Metric name to value; ``per_class_recall`` is itself a mapping.

    Raises:
        ValueError: If there are no rows.
    """
    metrics: dict[str, Any] = {
        "accuracy": accuracy(actual, predicted),
        "macro_f1": macro_f1(actual, predicted),
        "per_class_recall": per_class_recall(actual, predicted),
    }
    if probabilities is not None:
        metrics[f"top_{_TOP_K}_accuracy"] = top_k_accuracy(
            actual, probabilities
        )
    return metrics


def circular_hour_error(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """Return the hour distance the short way around the clock.

    Hour 23 and hour 0 are one hour apart, not twenty-three, so the
    error wraps.

    Args:
        actual: Observed hours in ``[0, 24)``.
        predicted: Predicted hours in ``[0, 24)``.

    Returns:
        Absolute circular error per row, in hours.
    """
    difference = (predicted.to_numpy() - actual.to_numpy()) % _HOURS_IN_DAY
    wrapped = np.minimum(difference, _HOURS_IN_DAY - difference)
    return pd.Series(wrapped, index=actual.index, name="circular_hour_error")


def hour_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    probabilities: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return the hour head's metrics, including its circular ones.

    Args:
        actual: Observed hours.
        predicted: Predicted hours.
        probabilities: Per-hour probabilities, when the head has them.

    Returns:
        Metric name to value.

    Raises:
        ValueError: If there are no rows.
    """
    metrics = classification_metrics(actual, predicted, probabilities)
    error = circular_hour_error(actual, predicted)
    metrics["circular_mae_hours"] = float(error.mean())
    metrics[f"within_{_HOUR_TOLERANCE}_hour"] = float(
        (error <= _HOUR_TOLERANCE).mean()
    )
    return metrics


def clamp_delays(predicted: pd.Series) -> tuple[pd.Series, float]:
    """Return the submitted non-negative delay and how often it clamped.

    A negative predicted wait is not submittable, so it is raised to
    zero and counted.

    Args:
        predicted: Predicted delays in minutes, possibly negative.

    Returns:
        The clamped delays, and the share of rows that were clamped.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(predicted, "prediction set")
    clamped = predicted.clip(lower=0.0)
    rate = float((predicted < 0.0).mean())
    return clamped, rate


def notification_metrics(
    actual: pd.Series,
    predicted: pd.Series,
    support_minutes: Sequence[int],
) -> dict[str, float]:
    """Return the notification head's error metrics.

    Mean absolute error is the primary figure. The fixed-minute rates
    are diagnostics and never select a model.

    Args:
        actual: Observed delays in minutes.
        predicted: Predicted delays in minutes.
        support_minutes: Absolute tolerances reported alongside.

    Returns:
        Metric name to value, in minutes where a metric has units.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(actual, "prediction set")
    error = predicted.to_numpy() - actual.to_numpy()
    absolute = np.abs(error)
    metrics = {
        "mae_minutes": float(absolute.mean()),
        "median_ae_minutes": float(np.median(absolute)),
        "p90_ae_minutes": float(np.percentile(absolute, _P90)),
        "signed_bias_minutes": float(error.mean()),
        "median_abs_log_ratio": float(
            np.median(np.abs(_log_ratio(actual, predicted)))
        ),
    }
    for minutes in support_minutes:
        metrics[f"within_{minutes}_minutes"] = float(
            (absolute <= minutes).mean()
        )
    return metrics


def _log_ratio(actual: pd.Series, predicted: pd.Series) -> np.ndarray:
    """Return the log ratio the notification rule is written in.

    The offset keeps very small positive delays finite and makes
    over- and under-prediction symmetric.

    Args:
        actual: Observed delays in minutes.
        predicted: Predicted delays in minutes, already non-negative.

    Returns:
        ``log((predicted + 1) / (actual + 1))`` per row.

    Raises:
        ValueError: If either side carries a negative delay, for which
            the ratio is not defined.
    """
    actual_values = actual.to_numpy(dtype=float)
    predicted_values = predicted.to_numpy(dtype=float)
    if (actual_values < 0).any() or (predicted_values < 0).any():
        msg = "delays must be non-negative before the log ratio is taken"
        raise ValueError(msg)
    return np.log(
        (predicted_values + _LOG_RATIO_OFFSET)
        / (actual_values + _LOG_RATIO_OFFSET)
    )


def notification_match(
    actual: pd.Series, predicted: pd.Series, match_ratio: float
) -> pd.Series:
    """Return whether each delay falls inside the relative tolerance.

    Args:
        actual: Observed delays in minutes.
        predicted: Submitted non-negative delays in minutes.
        match_ratio: Symmetric multiplicative tolerance; ``1.25`` is
            ``+/-25%``.

    Returns:
        One boolean per row.

    Raises:
        ValueError: If ``match_ratio`` is not above one, or a delay is
            negative.
    """
    if match_ratio <= 1.0:
        msg = f"match ratio must be above 1.0, got {match_ratio}"
        raise ValueError(msg)
    inside = np.abs(_log_ratio(actual, predicted)) <= math.log(match_ratio)
    return pd.Series(inside, index=actual.index, name=NOTIFICATION)


def component_matches(frame: pd.DataFrame, match_ratio: float) -> pd.DataFrame:
    """Return the four component matches, one column each.

    The notification column uses the submitted non-negative delay, the
    value an operator would have seen.

    Args:
        frame: Joined rows carrying every target and predicted column.
        match_ratio: Symmetric multiplicative notification tolerance.

    Returns:
        One boolean column per component, in component order.

    Raises:
        ValueError: If a required column is missing or there are no
            rows.
    """
    _require_rows(frame, "prediction set")
    missing = [
        column
        for column in list(TARGET_COLUMNS.values())
        + list(PREDICTED_COLUMNS.values())
        if column not in frame.columns
    ]
    if missing:
        msg = f"missing columns {missing}"
        raise ValueError(msg)

    clamped, _ = clamp_delays(frame[PREDICTED_COLUMNS[NOTIFICATION]])
    matches = {}
    for component in (FACILITY, USAGE_WEEKDAY, USAGE_HOUR):
        actual = frame[TARGET_COLUMNS[component]].to_numpy()
        predicted = frame[PREDICTED_COLUMNS[component]].to_numpy()
        matches[component] = pd.Series(
            actual == predicted, index=frame.index, name=component
        )
    matches[NOTIFICATION] = notification_match(
        frame[TARGET_COLUMNS[NOTIFICATION]], clamped, match_ratio
    )
    return pd.DataFrame(matches, columns=list(COMPONENTS))


def score_rows(matches: pd.DataFrame) -> pd.Series:
    """Return how many of the four components each row matched.

    Args:
        matches: Boolean component columns.

    Returns:
        An integer score in ``[0, 4]`` per row.
    """
    return matches[list(COMPONENTS)].sum(axis=1).rename(SCORE_COLUMN)


def match_summary(matches: pd.DataFrame) -> dict[str, float]:
    """Return the per-component and overall match rates.

    Args:
        matches: Boolean component columns.

    Returns:
        Per-component rates, the mean score, the mean score as a share
        of four, the all-four rate, and the three-or-more rate.

    Raises:
        ValueError: If there are no rows.
    """
    _require_rows(matches, "match set")
    scores = score_rows(matches)
    summary = {
        component: float(matches[component].mean()) for component in COMPONENTS
    }
    summary["score_mean"] = float(scores.mean())
    summary["overall"] = float(scores.mean()) / _STRICT_SCORE
    summary["strict"] = float((scores == _STRICT_SCORE).mean())
    summary["at_least_three"] = float((scores >= _NEAR_MISS_SCORE).mean())
    return summary


def match_lift(model: float, baseline: float) -> Comparison:
    """Return a match rate against the baseline's, higher being better.

    Args:
        model: The model's match rate.
        baseline: The baseline's match rate.

    Returns:
        The absolute difference always, and the relative lift when the
        baseline rate is non-zero.
    """
    relative = None if baseline == 0.0 else (model - baseline) / baseline
    return Comparison(
        model=model,
        baseline=baseline,
        absolute=model - baseline,
        relative=relative,
    )


def mae_reduction(model: float, baseline: float) -> Comparison:
    """Return the notification error against the baseline's.

    Positive means improvement, as it does for a match lift.

    Args:
        model: The model's mean absolute error, in minutes.
        baseline: The baseline's mean absolute error, in minutes.

    Returns:
        The absolute reduction in minutes always, and the relative
        reduction when the baseline error is non-zero.
    """
    relative = None if baseline == 0.0 else 1.0 - model / baseline
    return Comparison(
        model=model,
        baseline=baseline,
        absolute=baseline - model,
        relative=relative,
    )


def evaluate_predictions(
    frame: pd.DataFrame,
    match_ratio: float,
    support_minutes: Sequence[int],
    probabilities: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Score one model's predictions against the rows it predicted.

    Args:
        frame: Joined rows carrying every target and predicted column.
        match_ratio: Symmetric multiplicative notification tolerance.
        support_minutes: Absolute notification tolerances to report.
        probabilities: Per-class probabilities keyed by component, for
            the three classification heads. Supply all three or none.

    Returns:
        Row count, per-head metrics, the notification clamp rate, and
        the match summary.

    Raises:
        ValueError: If there are no rows, a column is missing, or
            probabilities cover only some classification heads.
    """
    _require_rows(frame, "prediction set")
    given = _classifier_probabilities(probabilities)
    clamped, clamp_rate = clamp_delays(frame[PREDICTED_COLUMNS[NOTIFICATION]])

    heads: dict[str, Any] = {}
    for component in (FACILITY, USAGE_WEEKDAY):
        heads[component] = classification_metrics(
            frame[TARGET_COLUMNS[component]],
            frame[PREDICTED_COLUMNS[component]],
            given.get(component),
        )
    heads[USAGE_HOUR] = hour_metrics(
        frame[TARGET_COLUMNS[USAGE_HOUR]],
        frame[PREDICTED_COLUMNS[USAGE_HOUR]],
        given.get(USAGE_HOUR),
    )
    actual_delay = frame[TARGET_COLUMNS[NOTIFICATION]]
    heads[NOTIFICATION] = {
        "submitted": notification_metrics(
            actual_delay, clamped, support_minutes
        ),
        "raw": notification_metrics(
            actual_delay,
            frame[PREDICTED_COLUMNS[NOTIFICATION]].clip(lower=0.0),
            support_minutes,
        ),
        "clamp_rate": clamp_rate,
    }
    return {
        "rows": len(frame),
        "heads": heads,
        "matches": match_summary(component_matches(frame, match_ratio)),
    }


def json_ready(value: Any) -> Any:
    """Render a report in types JSON can carry, without losing a number.

    Class labels arrive as NumPy scalars, and a NumPy integer is not a
    key JSON accepts. Rendering happens once, on the way out, so the
    metric functions keep returning the labels their callers index by.

    Args:
        value: A report, or any part of one.

    Returns:
        The same structure with mapping keys as text and NumPy scalars
        as their Python equivalents.
    """
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def selection_score(report: Mapping[str, Any]) -> float:
    """Return the one score that selects a model and headlines it.

    Selecting on one metric and reporting another picks models that
    score worse on the number the report leads with, so the mean share
    of matched components does both jobs.

    Args:
        report: A report from :func:`evaluate_predictions`.

    Returns:
        The mean matched components as a share of four.

    Raises:
        ValueError: If the report carries no match summary.
    """
    matches = report.get("matches")
    if not isinstance(matches, Mapping) or "overall" not in matches:
        msg = "this report carries no match summary to select on"
        raise ValueError(msg)
    return float(matches["overall"])


def verdict(model: float, baseline: float) -> str:
    """Name the outcome of a comparison, including a loss.

    Args:
        model: The model's selection score.
        baseline: The baseline's selection score.

    Returns:
        ``beats``, ``ties``, or ``loses`` — reported as it comes out.
    """
    if model > baseline:
        return "beats"
    if model < baseline:
        return "loses"
    return "ties"


def compare_reports(
    model: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    """Set one model's report against the baseline's, component by component.

    Args:
        model: The model's report from :func:`evaluate_predictions`.
        baseline: The baseline's report over the same rows.

    Returns:
        The two selection scores and the verdict between them, a match
        lift per component and overall, and the notification error
        reduction.

    Raises:
        ValueError: If either report is missing its match summary or its
            notification metrics.
    """
    model_score = selection_score(model)
    baseline_score = selection_score(baseline)
    lifts = {
        component: dataclasses.asdict(
            match_lift(
                float(model["matches"][component]),
                float(baseline["matches"][component]),
            )
        )
        for component in (*COMPONENTS, "overall", "strict")
    }
    reduction = mae_reduction(
        float(model["heads"][NOTIFICATION]["submitted"]["mae_minutes"]),
        float(baseline["heads"][NOTIFICATION]["submitted"]["mae_minutes"]),
    )
    return {
        "selection_score": {
            "model": model_score,
            "baseline": baseline_score,
            "verdict": verdict(model_score, baseline_score),
        },
        "match_lift": lifts,
        "mae_reduction": dataclasses.asdict(reduction),
    }


def _classifier_probabilities(
    probabilities: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    """Return the per-head probabilities, all three or none.

    Top-3 accuracy is reported for every classification head or for
    none of them.

    Args:
        probabilities: Per-class probabilities keyed by component.

    Returns:
        The mapping, empty when none was supplied.

    Raises:
        ValueError: If it covers some classification heads but not all,
            or names something that is not one.
    """
    if not probabilities:
        return {}
    classifiers = {FACILITY, USAGE_WEEKDAY, USAGE_HOUR}
    supplied = set(probabilities)
    unknown = supplied - classifiers
    if unknown:
        msg = f"{sorted(unknown)} are not classification heads"
        raise ValueError(msg)
    if supplied != classifiers:
        msg = (
            "top-k accuracy is reported for every classification head or "
            f"none; got {sorted(supplied)}"
        )
        raise ValueError(msg)
    return probabilities
