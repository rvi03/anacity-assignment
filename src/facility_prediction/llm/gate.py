"""Decides whether training this model is worth the remaining compute.

Two passes answered the same rows: the base model alone, and the base
model with the pilot adapter. This module marks both against what the
residents actually did, marks two deliberately trivial references
against the same rows, and asks one question — is the trained pass
better by more than sampling noise?

    zero-shot answers ─┐
                       ├─> same rows, same marking ─> paired difference
    pilot answers ─────┘                                    │
                                                   interval above zero?
                                                            │
                                            ┌───────────────┴────────┐
                                       train at size            stop here
                                                            and say so

The difference is measured row by row on the same rows, because a
difference between two independent samples of two hundred rows would
be mostly noise. A row whose answer never parsed stays in the count
with every component wrong.

Leakage contract: reads validation targets only, and only after the
answers exist. No holdout target is read, and no answer is changed by
what the target turned out to be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.evaluation import evaluate
from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import llm_predict
from facility_prediction.llm import settings as settings_module

GATE = "the trained pilot beats zero-shot by more than sampling noise"
RUN_NAME = "learnability-gate"

BUCKET_ACCURACY = "bucket_accuracy"
NOTIFICATION_MATCH = "notification_match"

INVALID_LABEL = "__INVALID__"

EVIDENCE = "measured"


class GateError(Exception):
    """Raised when the branch may not spend its remaining compute."""


def actual_labels(
    samples: pd.DataFrame, ladder: Sequence[buckets_module.Bucket]
) -> pd.Series:
    """Returns the label each row's real delay falls in.

    Leakage contract: reads validation targets, which are readable
    before this branch's single holdout pass.

    Args:
        samples: Rows carrying their notification delay.
        ladder: The frozen delay buckets.

    Returns:
        The label per row, indexed by sample identifier.
    """
    delays = samples[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]]
    labels = buckets_module.assign(delays, list(ladder))
    return pd.Series(
        labels.to_numpy(),
        index=samples[llm_predict.SAMPLE_ID].astype(str),
        name="actual_bucket",
    )


def matched_components(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    match_ratio: float,
) -> pd.DataFrame:
    """Returns the four component matches, failures counted as wrong.

    Args:
        predictions: Predicted columns indexed by sample identifier.
        targets: Target columns indexed by sample identifier.
        match_ratio: Symmetric multiplicative notification tolerance.

    Returns:
        One boolean column per component, every row of ``predictions``
        present.
    """
    usable = predictions.loc[predictions[llm_predict.VALID_COLUMN]]
    if usable.empty:
        return pd.DataFrame(
            False, index=predictions.index, columns=list(evaluate.COMPONENTS)
        )
    joined = usable.join(targets, how="inner")
    matches = evaluate.component_matches(joined, match_ratio)
    matches.index = joined.index
    return matches.reindex(predictions.index, fill_value=False)


def arm_metrics(
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    labels: pd.Series,
    config: config_module.Config,
    ceiling: float,
) -> dict[str, Any]:
    """Marks one pass against the rows it answered.

    Args:
        predictions: Predicted columns indexed by sample identifier.
        targets: Target columns indexed by sample identifier.
        labels: The real label per row.
        config: Validated shared configuration.
        ceiling: The representation ceiling the labels can reach.

    Returns:
        The pass's rates, its per-row component matches, and the
        confusion between real and predicted labels.
    """
    match_ratio = config.evaluation.notification_match_ratio
    matches = matched_components(predictions, targets, match_ratio)
    predicted = (
        predictions[llm_predict.BUCKET_COLUMN]
        .fillna(INVALID_LABEL)
        .reindex(labels.index)
    )
    correct = predicted == labels
    summary = evaluate.match_summary(matches)
    notification = float(matches[evaluate.NOTIFICATION].mean())
    return {
        "rates": {
            **summary,
            BUCKET_ACCURACY: float(correct.mean()),
            "bucket_macro_f1": evaluate.macro_f1(labels, predicted),
            "match_over_ceiling": notification / ceiling,
            "valid": float(predictions[llm_predict.VALID_COLUMN].mean()),
        },
        "per_row": pd.DataFrame(
            {
                BUCKET_ACCURACY: correct,
                NOTIFICATION_MATCH: matches[evaluate.NOTIFICATION].reindex(
                    labels.index
                ),
            }
        ),
        "confusion": _confusion(labels, predicted),
        "support": {
            str(label): int(count)
            for label, count in labels.value_counts().sort_index().items()
        },
    }


def _confusion(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    """Returns the confusion between real and predicted labels.

    Args:
        actual: The real label per row.
        predicted: The predicted label per row.

    Returns:
        Real label to predicted label to count, zero cells dropped.
    """
    frame = evaluate.confusion_matrix(actual, predicted)
    return {
        str(row): {
            str(column): int(value)
            for column, value in frame.loc[row].items()
            if value
        }
        for row in frame.index
    }


def reference_metrics(
    train_delays: pd.Series,
    labels: pd.Series,
    targets: pd.DataFrame,
    ladder: Sequence[buckets_module.Bucket],
    config: config_module.Config,
) -> dict[str, Any]:
    """Marks the two deliberately trivial references on the same rows.

    One always answers the label most training rows carry; the other
    always answers the training median delay. A trained model that
    cannot beat these has learned nothing worth the compute.

    Leakage contract: both references are fitted on training rows only.

    Args:
        train_delays: Notification delays of the training split.
        labels: The real label per gate row.
        targets: Target columns indexed by sample identifier.
        ladder: The frozen delay buckets.
        config: Validated shared configuration.

    Returns:
        The majority-label accuracy and the median-delay match rate.
    """
    majority = max(ladder, key=lambda bucket: bucket.train_rows).label
    median = float(train_delays.median())
    actual = targets.loc[
        labels.index, evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]
    ]
    matched = evaluate.notification_match(
        actual,
        pd.Series(median, index=actual.index),
        config.evaluation.notification_match_ratio,
    )
    return {
        f"majority_{BUCKET_ACCURACY}": float((labels == majority).mean()),
        f"median_delay_{NOTIFICATION_MATCH}": float(matched.mean()),
        "majority_label": majority,
        "median_delay_minutes": median,
    }


def paired_bootstrap(
    base: pd.Series,
    trained: pd.Series,
    gate: settings_module.Gate,
) -> dict[str, float]:
    """Returns the interval for one pass's improvement over the other.

    The two passes answered the same rows, so a resample takes the same
    rows from both and the difference keeps its pairing.

    Args:
        base: Per-row outcome of the untrained pass.
        trained: Per-row outcome of the trained pass, same rows.
        gate: The gate's settings.

    Returns:
        The observed difference and the interval's bounds.

    Raises:
        GateError: If the two passes did not answer the same rows.
    """
    if list(base.index) != list(trained.index):
        msg = "the two passes did not answer the same rows, so no paired"
        raise GateError(f"{msg} difference exists")
    difference = (
        trained.astype(float).to_numpy() - base.astype(float).to_numpy()
    )
    rng = np.random.default_rng(gate.sample_seed)
    draws = rng.integers(
        0, len(difference), size=(gate.bootstrap_resamples, len(difference))
    )
    means = difference[draws].mean(axis=1)
    tail = (1.0 - gate.bootstrap_confidence) / 2.0
    return {
        "difference": float(difference.mean()),
        "low": float(np.quantile(means, tail)),
        "high": float(np.quantile(means, 1.0 - tail)),
        "resamples": float(gate.bootstrap_resamples),
        "confidence": gate.bootstrap_confidence,
    }


def build_report(
    arms: Mapping[str, Mapping[str, Any]],
    intervals: Mapping[str, Mapping[str, float]],
    references: Mapping[str, float],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Assembles the record the gate is decided from.

    Args:
        arms: Arm name to its marked results.
        intervals: Metric name to its paired interval.
        references: The trivial references' rates.
        context: What was run, on which rows, with which artifacts.

    Returns:
        The report payload, ready to serialise.
    """
    return {
        "evidence": EVIDENCE,
        "gate": GATE,
        "context": dict(context),
        "references": dict(references),
        "arms": {
            name: {
                "rates": dict(arm["rates"]),
                "operational": dict(arm["operational"]),
                "support": dict(arm["support"]),
                "confusion": dict(arm["confusion"]),
            }
            for name, arm in arms.items()
        },
        "paired_intervals": {
            name: dict(interval) for name, interval in intervals.items()
        },
    }


def check_gate(report: Mapping[str, Any], pilot: Mapping[str, Any]) -> None:
    """Refuses the full training unless the pilot earned it.

    Args:
        report: The assembled report.
        pilot: The pilot training record.

    Raises:
        GateError: On the first condition the evidence does not meet.
    """
    initial = pilot["loss"]["initial_validation"]
    final = pilot["loss"]["final_validation"]
    if initial is None or final is None or final >= initial:
        msg = f"the pilot's validation loss did not fall: {initial} to {final}"
        raise GateError(msg)

    rates = report["arms"][llm_predict.PILOT_ADAPTER]["rates"]
    references = report["references"]
    for metric, reference in (
        (BUCKET_ACCURACY, f"majority_{BUCKET_ACCURACY}"),
        (NOTIFICATION_MATCH, f"median_delay_{NOTIFICATION_MATCH}"),
    ):
        value = _rate(rates, metric)
        if value < references[reference]:
            msg = (
                f"the trained pass scores {value:.4f} on {metric}, below the "
                f"{references[reference]:.4f} of a rule that ignores the "
                "prompt entirely"
            )
            raise GateError(msg)

    improved = [
        name
        for name, interval in report["paired_intervals"].items()
        if interval["low"] > 0.0
    ]
    if not improved:
        margins = ", ".join(
            f"{name} {interval['low']:+.4f} to {interval['high']:+.4f}"
            for name, interval in report["paired_intervals"].items()
        )
        msg = (
            "no interval for the trained pass's improvement clears zero "
            f"({margins}), so training is not distinguishable from noise"
        )
        raise GateError(msg)


def _rate(rates: Mapping[str, float], metric: str) -> float:
    """Returns one arm's rate for a gated metric.

    Args:
        rates: One arm's rates.
        metric: The gated metric's name.

    Returns:
        Its value.
    """
    if metric == NOTIFICATION_MATCH:
        return float(rates[evaluate.NOTIFICATION])
    return float(rates[metric])


def run_params(report: Mapping[str, Any]) -> dict[str, str]:
    """Returns the run parameters for this gate pass.

    Args:
        report: The assembled report.

    Returns:
        Parameter name to value, as text.
    """
    params = {str(key): str(value) for key, value in report["context"].items()}
    params["evidence"] = str(report["evidence"])
    params["majority_label"] = str(report["references"]["majority_label"])
    return params


def run_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Returns the run metrics for this gate pass.

    Args:
        report: The assembled report.

    Returns:
        Metric name to value.
    """
    metrics = {
        f"{name}_{key}": float(value)
        for name, arm in report["arms"].items()
        for key, value in {**arm["rates"], **arm["operational"]}.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    metrics.update(
        {
            f"interval_{name}_{key}": float(value)
            for name, interval in report["paired_intervals"].items()
            for key, value in interval.items()
        }
    )
    metrics.update(
        {
            f"reference_{key}": float(value)
            for key, value in report["references"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return metrics
