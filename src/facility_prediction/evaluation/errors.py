"""Where the model fails, sliced by the things that could explain it.

A headline score says how often the model is right. It does not say
whether the misses are spread evenly or concentrated in one kind of
resident, one part of the facility catalog, or one side of a drift
event. Those are different problems with different fixes, and a single
number cannot tell them apart.

Every slice here is cut from columns that already exist on the scored
rows, so a reviewer can rederive any cell:

    activity quartile   how much history the resident had, plus the
                        sparse bucket the acceptance checks single out
    facility head/tail  does the thin end of the catalog get predicted
                        at all, or does the model spend everything on
                        the popular facilities?
    history length      short against long
    drift               before and after each dated event
    weekday / weekend
    lead-time bucket    short-notice against long-planned
    cold start          residents and classes the training split never
                        saw

Alongside them: the largest notification errors, the commonest
confusions per output, and feature importance both ways — CatBoost's
own gain and SHAP over a fixed seeded sample.

Leakage contract: everything here reads *saved predictions* and the
features they were produced from. Nothing is fitted, so no slice can
influence a model. The caller chooses the split; before the freeze that
must be validation, because a slice computed on holdout targets would
spend the single scoring pass.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.evaluation import evaluate

_LOGGER = logging.getLogger(__name__)

ANALYSIS_FILENAME = "error_analysis.json"
IMPORTANCE_PLOT = "feature_importance.png"

SLICE_COLUMN = "slice"
_ACTIVITY_QUARTILES = 4
_TOP_CONFUSIONS = 10
_TOP_ERRORS = 20
_TOP_FEATURES = 20
# Saturday and Sunday, in pandas' Monday-is-zero weekday numbering.
_FIRST_WEEKEND_DAY = 5
# A multiclass SHAP result is (rows, classes, features + 1).
_MULTICLASS_SHAP_DIMENSIONS = 3
# The plan caps SHAP at a fixed, seeded sample so the cost is bounded
# and the same rows are explained on every run.
SHAP_SAMPLE_ROWS = 500


class AnalysisError(Exception):
    """Raised when a slice cannot be cut from what it was given."""


def _quartile_labels(values: pd.Series) -> pd.Series:
    """Return quartile membership as readable labels.

    Ties are common — many residents share a booking count — so the
    quartiles are cut on rank rather than on value, and a degenerate
    column collapses to one label instead of raising.

    Args:
        values: A numeric column to rank.

    Returns:
        A label per row, ``q1`` (lowest) through ``q4``.
    """
    ranked = values.rank(method="average", pct=True)
    edges = np.linspace(0.0, 1.0, _ACTIVITY_QUARTILES + 1)
    index = np.clip(
        np.digitize(ranked.to_numpy(), edges[1:-1], right=True),
        0,
        _ACTIVITY_QUARTILES - 1,
    )
    return pd.Series(
        [f"q{position + 1}" for position in index],
        index=values.index,
        name="activity_quartile",
    )


def _head_and_tail(
    frame: pd.DataFrame, config: config_module.Config
) -> pd.Series:
    """Split the facility catalog into its popular head and thin tail.

    The cut is the configured median popularity, so it moves with the
    catalog rather than being a hardcoded facility list.

    Args:
        frame: Scored rows carrying the actual facility.
        config: Validated configuration naming configured popularity.

    Returns:
        ``head`` or ``tail`` per row.
    """
    popularity = {
        facility.name: facility.popularity for facility in config.facilities
    }
    cut = float(np.median(list(popularity.values())))
    actual = frame[evaluate.TARGET_COLUMNS[evaluate.FACILITY]]
    return pd.Series(
        np.where(actual.map(popularity).to_numpy() >= cut, "head", "tail"),
        index=frame.index,
        name="facility_band",
    )


def _lead_bucket(frame: pd.DataFrame) -> pd.Series:
    """Bucket each row by how far ahead the booking was made.

    Args:
        frame: Scored rows carrying the actual notification delay.

    Returns:
        A readable lead-time bucket per row.
    """
    hours = (
        frame[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]].to_numpy(
            dtype=np.float64
        )
        / 60.0
    )
    edges = [6.0, 24.0, 72.0]
    names = ["under_6h", "6h_to_1d", "1d_to_3d", "over_3d"]
    return pd.Series(
        [names[position] for position in np.digitize(hours, edges)],
        index=frame.index,
        name="lead_bucket",
    )


def build_slice_columns(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    training_ids: Sequence[str],
    config: config_module.Config,
) -> pd.DataFrame:
    """Return one column per declared slice, aligned to ``frame``.

    Args:
        frame: Scored rows, already joined to their targets.
        features: The feature table, for history depth.
        training_ids: Sample ids the fit was allowed to read, so
            cold-start rows can be identified.
        config: Validated configuration.

    Returns:
        A frame of label columns, one per slice dimension.

    Raises:
        AnalysisError: If the rows and features cannot be aligned.
    """
    if "sample_id" not in frame.columns:
        msg = "scored rows carry no sample_id to align features by"
        raise AnalysisError(msg)

    depth = (
        features.set_index("sample_id")["n_prior_bookings"]
        if ("n_prior_bookings" in features.columns)
        else None
    )
    if depth is None:
        depth = frame.get("n_prior_bookings")
    if depth is None:
        msg = "neither the rows nor the features carry n_prior_bookings"
        raise AnalysisError(msg)
    aligned = (
        frame["sample_id"].map(depth)
        if depth.index.name == "sample_id"
        else depth
    ).astype(float)

    seen_residents = set(
        features.loc[
            features["sample_id"].isin(set(training_ids)), "resident_id"
        ]
    )
    sparse = config.generator.acceptance.sparse_resident_bookings
    usage = frame[evaluate.TARGET_COLUMNS[evaluate.USAGE_WEEKDAY]]

    return pd.DataFrame(
        {
            "activity_quartile": _quartile_labels(aligned),
            "sparse_history": np.where(
                aligned.to_numpy() < sparse, "under_5_prior", "5_or_more"
            ),
            "facility_band": _head_and_tail(frame, config),
            "history_length": np.where(
                aligned.to_numpy() >= aligned.median(), "long", "short"
            ),
            "day_type": np.where(
                usage.astype(int).to_numpy() >= _FIRST_WEEKEND_DAY,
                "weekend",
                "weekday",
            ),
            "lead_bucket": _lead_bucket(frame),
            "cold_start_resident": np.where(
                frame["resident_id"].isin(seen_residents).to_numpy(),
                "seen_in_training",
                "unseen_in_training",
            ),
        },
        index=frame.index,
    )


def slice_report(
    frame: pd.DataFrame,
    slices: pd.DataFrame,
    config: config_module.Config,
) -> list[dict[str, Any]]:
    """Score every slice of every dimension.

    Args:
        frame: Scored rows carrying targets and predictions.
        slices: The label columns from :func:`build_slice_columns`.
        config: Validated configuration; supplies the match tolerance.

    Returns:
        One row per (dimension, slice), each with its own match rates
        and the share of the split it covers.
    """
    matches = evaluate.component_matches(
        frame, config.evaluation.notification_match_ratio
    )
    rows: list[dict[str, Any]] = []
    for dimension in slices.columns:
        for label, positions in slices.groupby(
            slices[dimension], sort=True
        ).groups.items():
            block = matches.loc[positions]
            summary = evaluate.match_summary(block)
            rows.append(
                {
                    "dimension": dimension,
                    SLICE_COLUMN: str(label),
                    "rows": len(positions),
                    "share_of_split": round(len(positions) / len(frame), 4),
                    **{
                        key: round(float(value), 4)
                        for key, value in summary.items()
                    },
                }
            )
    return rows


def confusions(frame: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    """Return the commonest wrong answers for each categorical output.

    Args:
        frame: Scored rows carrying targets and predictions.

    Returns:
        Per output, the most frequent (actual, predicted) pairs that
        disagree, largest first.
    """
    report: dict[str, list[dict[str, Any]]] = {}
    for component in (
        evaluate.FACILITY,
        evaluate.USAGE_WEEKDAY,
        evaluate.USAGE_HOUR,
    ):
        actual = frame[evaluate.TARGET_COLUMNS[component]].astype(str)
        predicted = frame[evaluate.PREDICTED_COLUMNS[component]].astype(str)
        wrong = actual != predicted
        pairs = (
            pd.DataFrame(
                {"actual": actual[wrong], "predicted": predicted[wrong]}
            )
            .value_counts()
            .head(_TOP_CONFUSIONS)
        )
        report[component] = [
            {
                "actual": str(index[0]),
                "predicted": str(index[1]),
                "rows": int(count),
            }
            for index, count in pairs.items()
        ]
    return report


def largest_notification_errors(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return the rows whose submitted delay missed by the most minutes.

    Args:
        frame: Scored rows carrying the actual and predicted delay.

    Returns:
        The largest absolute errors, worst first.
    """
    clamped, _ = evaluate.clamp_delays(
        frame[evaluate.PREDICTED_COLUMNS[evaluate.NOTIFICATION]]
    )
    actual = frame[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]]
    error = (clamped - actual).abs()
    worst = error.nlargest(_TOP_ERRORS)
    return [
        {
            "sample_id": str(frame.loc[position, "sample_id"]),
            "actual_minutes": round(float(actual.loc[position]), 1),
            "submitted_minutes": round(float(clamped.loc[position]), 1),
            "absolute_error_minutes": round(float(value), 1),
        }
        for position, value in worst.items()
    ]


def importance_sample(
    table: pd.DataFrame, seed: int, rows: int = SHAP_SAMPLE_ROWS
) -> pd.DataFrame:
    """Return the fixed, seeded rows that importance is computed over.

    The sample is drawn once from a seeded generator, so the same rows
    are explained on every run and an importance number can be compared
    between runs rather than only within one.

    Args:
        table: The rows available to sample from.
        seed: Seed for the draw.
        rows: Cap on the sample size.

    Returns:
        At most ``rows`` rows, in their original order.
    """
    if len(table) <= rows:
        return table
    generator = np.random.default_rng(seed)
    chosen = np.sort(generator.choice(len(table), size=rows, replace=False))
    return table.iloc[chosen]


def feature_importance(
    model: Any, pool: Any, names: Sequence[str]
) -> dict[str, list[dict[str, Any]]]:
    """Return gain and SHAP importance for one fitted head.

    Both are reported because they answer different questions. Gain says
    how much a feature moved the loss during fitting; SHAP says how much
    it moves this sample's predictions. A feature that is large on one
    and small on the other is worth looking at, and averaging them into
    a single ranking would hide exactly that.

    Args:
        model: A fitted CatBoost estimator.
        pool: The sample to explain, as a CatBoost pool.
        names: Feature names, in matrix column order.

    Returns:
        ``gain`` and ``shap`` lists, each largest first.
    """
    gain = np.asarray(
        model.get_feature_importance(type="PredictionValuesChange"),
        dtype=np.float64,
    )
    shap = np.asarray(
        model.get_feature_importance(pool, type="ShapValues"),
        dtype=np.float64,
    )
    # A multiclass head returns one SHAP block per class, and every
    # block carries a trailing expected-value column that is not a
    # feature. Collapse the classes and drop that column.
    if shap.ndim == _MULTICLASS_SHAP_DIMENSIONS:
        shap = np.abs(shap).mean(axis=1)
    shap = np.abs(shap[:, : len(names)]).mean(axis=0)

    return {
        "gain": _ranked(gain, names),
        "shap": _ranked(shap, names),
    }


def _ranked(values: np.ndarray, names: Sequence[str]) -> list[dict[str, Any]]:
    """Return the largest-scoring features, largest first.

    Args:
        values: One score per feature.
        names: Feature names, in the same order.

    Returns:
        The top features and their scores.
    """
    order = np.argsort(values)[::-1][:_TOP_FEATURES]
    return [
        {
            "feature": str(names[position]),
            "score": round(float(values[position]), 6),
        }
        for position in order
    ]


def build_analysis(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    training_ids: Sequence[str],
    config: config_module.Config,
    split_name: str,
    importance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the whole error analysis.

    Args:
        frame: Scored rows for one model, joined to their targets.
        features: The feature table.
        training_ids: Sample ids the fit was allowed to read.
        config: Validated configuration.
        split_name: Which split these rows belong to, recorded so no
            reader mistakes a validation analysis for a holdout one.
        importance: Per-head gain and SHAP, when a model was loaded.

    Returns:
        The analysis payload, ready to serialise.

    Raises:
        AnalysisError: If there is nothing to analyse.
    """
    if frame.empty:
        msg = "cannot analyse an empty prediction set"
        raise AnalysisError(msg)
    slices = build_slice_columns(frame, features, training_ids, config)
    return {
        "provenance": {
            "split": split_name,
            "rows": len(frame),
            "seed": config.seed,
            "shap_sample_cap": SHAP_SAMPLE_ROWS,
        },
        "slices": slice_report(frame, slices, config),
        "confusions": confusions(frame),
        "largest_notification_errors": largest_notification_errors(frame),
        "importance": dict(importance or {}),
    }


def write_analysis(payload: Mapping[str, Any], path: pathlib.Path) -> str:
    """Write the analysis and return its content hash.

    Args:
        payload: The analysis payload.
        path: Destination JSON; parent directories are created.

    Returns:
        The hex SHA-256 of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(evaluate.json_ready(payload), indent=2, sort_keys=False)
    text += "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_importance_plot(
    importance: Mapping[str, Any], path: pathlib.Path
) -> pathlib.Path:
    """Render gain against SHAP for every head.

    Args:
        importance: The per-head importance mapping.
        path: Destination image; parent directories are created.

    Returns:
        The path written.
    """
    # The backend must be chosen before pyplot loads, and a module-level
    # import would make every caller pay for a plotting stack.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    heads = list(importance)
    figure, axes = plt.subplots(
        len(heads) or 1, 1, figsize=(10, 3.2 * max(len(heads), 1))
    )
    for position, head in enumerate(heads):
        panel = np.atleast_1d(axes)[position]
        top = importance[head]["shap"][:12][::-1]
        panel.barh(
            [row["feature"] for row in top],
            [row["score"] for row in top],
        )
        panel.set_title(f"{head} — mean |SHAP| over the seeded sample")
        panel.tick_params(axis="y", labelsize="x-small")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def check_analysis(payload: Mapping[str, Any]) -> Sequence[str]:
    """Return every way the analysis contradicts itself.

    Args:
        payload: The analysis payload.

    Returns:
        One message per problem; empty when it holds together.
    """
    problems: list[str] = []
    rows = payload["provenance"]["rows"]
    by_dimension: dict[str, int] = {}
    for entry in payload["slices"]:
        by_dimension[entry["dimension"]] = (
            by_dimension.get(entry["dimension"], 0) + entry["rows"]
        )
        if not 0.0 <= entry["overall"] <= 1.0:
            problems.append(
                f"{entry['dimension']}/{entry['slice']}: overall "
                f"{entry['overall']} is not a rate"
            )
    for dimension, counted in by_dimension.items():
        if counted != rows:
            problems.append(
                f"{dimension}: slices cover {counted} rows, not {rows} — "
                "a slice dimension must partition the split"
            )
    for entry in payload["largest_notification_errors"]:
        if entry["absolute_error_minutes"] < 0:
            problems.append("a notification error is negative")
    return problems
