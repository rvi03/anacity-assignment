"""Imports the shared pipeline's measured outputs for the LLM track.

Reads the booking, sample, split, and prediction tables back from the
store, verifies them against the manifests committed beside them, and
builds one record of the counts, split boundaries, and baseline scores
the LLM track starts from. Any mismatch between a table and its
manifest raises :class:`SliceImportError`.

Leakage contract: the comparator is scored on the split named by the
caller, and rows are restricted to that split before any target column
is read. Split labels are read; holdout targets are not.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import math
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction import llm
from facility_prediction.data import digest as digest_module
from facility_prediction.data import generate
from facility_prediction.data import samples as samples_module
from facility_prediction.data import split as split_module
from facility_prediction.evaluation import evaluate
from facility_prediction.models import baselines

GATE = "every inherited quantity is measured, and its sources agree"
RUN_NAME = "slice-import"
EVIDENCE = "measured"

TRACK_COLUMN = "track"
MODEL_COLUMN = "model"

_SPLITS_SORT = (split_module.SAMPLE_ID_COLUMN,)
_PREDICTIONS_SORT = (
    TRACK_COLUMN,
    MODEL_COLUMN,
    split_module.SAMPLE_ID_COLUMN,
)


class SliceImportError(Exception):
    """Raised when an imported value is absent or disagrees."""


@dataclasses.dataclass(frozen=True)
class Spine:
    """The shared pipeline's tables, as read back from the store.

    Attributes:
        bookings: Every generated booking.
        samples: Every rolling-origin sample, with its targets.
        splits: One split label per sample.
        predictions: Every track's stored predictions.
    """

    bookings: pd.DataFrame
    samples: pd.DataFrame
    splits: pd.DataFrame
    predictions: pd.DataFrame


@dataclasses.dataclass(frozen=True)
class Manifests:
    """The manifests the shared pipeline committed beside them.

    Attributes:
        generation: The generation summary payload.
        sample: The sample summary payload.
        split: The frozen split manifest payload.
        comparison: The holdout comparison manifest payload.
    """

    generation: Mapping[str, Any]
    sample: Mapping[str, Any]
    split: Mapping[str, Any]
    comparison: Mapping[str, Any]


def file_digest(path: pathlib.Path) -> str:
    """Returns the SHA-256 of a committed file.

    Args:
        path: The file to hash.

    Returns:
        The hex SHA-256 of its bytes.

    Raises:
        SliceImportError: If the file does not exist.
    """
    if not path.is_file():
        msg = f"cannot import the shared slice: {path} does not exist"
        raise SliceImportError(msg)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_digests(spine: Spine) -> dict[str, str]:
    """Recomputes the canonical digest of every imported table.

    Args:
        spine: The tables read back from the store.

    Returns:
        Digest name to hex SHA-256, one entry per table.
    """
    return {
        "bookings_digest": generate.bookings_digest(spine.bookings),
        "samples_digest": samples_module.samples_digest(spine.samples),
        "splits_digest": digest_module.canonical_digest(
            spine.splits, sort_by=_SPLITS_SORT
        ),
        "predictions_digest": digest_module.canonical_digest(
            spine.predictions, sort_by=_PREDICTIONS_SORT
        ),
    }


def _lookup(payload: Mapping[str, Any], *keys: str) -> Any:
    """Reads a nested manifest value.

    Args:
        payload: A manifest payload.
        keys: The path to the value, outermost key first.

    Returns:
        The value found at that path.

    Raises:
        SliceImportError: If any key on the path is absent.
    """
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            path = " -> ".join(keys)
            msg = f"cannot import the shared slice: no {path} was recorded"
            raise SliceImportError(msg)
        value = value[key]
    return value


def _agree(what: str, recorded: Any, live: Any) -> None:
    """Requires a manifest value to equal the recomputed one.

    Args:
        what: What is being compared, for the message.
        recorded: The value a committed manifest carries.
        live: The value recomputed from the store.

    Raises:
        SliceImportError: If the two differ, which means the spine
            moved after the manifest was written.
    """
    if recorded != live:
        msg = (
            f"cannot import the shared slice: {what} is recorded as "
            f"{recorded!r} but the store holds {live!r}"
        )
        raise SliceImportError(msg)


def check_agreement(spine: Spine, manifests: Manifests) -> None:
    """Verifies every manifest value against the live tables.

    Leakage contract: reads digests, counts, and split labels only. No
    target column is read here.

    Args:
        spine: The tables read back from the store.
        manifests: The committed manifests written beside them.

    Raises:
        SliceImportError: On the first disagreement, naming it.
    """
    live = table_digests(spine)
    _agree(
        "the booking digest in the generation summary",
        _lookup(manifests.generation, "provenance", "bookings_digest"),
        live["bookings_digest"],
    )
    _agree(
        "the booking digest in the sample summary",
        _lookup(manifests.sample, "provenance", "bookings_digest"),
        live["bookings_digest"],
    )
    _agree(
        "the sample digest in the sample summary",
        _lookup(manifests.sample, "provenance", "samples_digest"),
        live["samples_digest"],
    )
    _agree(
        "the sample digest in the split manifest",
        _lookup(manifests.split, "provenance", "samples_digest"),
        live["samples_digest"],
    )
    _agree(
        "the booking count",
        _lookup(manifests.generation, "counts", "bookings"),
        len(spine.bookings),
    )
    _agree(
        "the sample count",
        _lookup(manifests.sample, "counts", "samples"),
        len(spine.samples),
    )
    _agree(
        "the split's sample count",
        _lookup(manifests.split, "samples"),
        len(spine.splits),
    )

    labelled = spine.splits[split_module.SPLIT_COLUMN].value_counts()
    for name in split_module.SPLIT_NAMES:
        _agree(
            f"the {name} row count",
            _lookup(manifests.split, "splits", name, "rows"),
            int(labelled.get(name, 0)),
        )

    comparison = _lookup(manifests.comparison, "sample_ids")
    _agree(
        "the comparison row count",
        _lookup(manifests.split, "settings", "comparison_rows"),
        len(comparison),
    )
    _check_comparison_is_holdout(spine, comparison)


def _check_comparison_is_holdout(spine: Spine, sample_ids: list[str]) -> None:
    """Verifies that every comparison identifier is a holdout row.

    Both tracks score the same sealed rows, so an identifier outside
    the holdout would widen the comparison.

    Leakage contract: compares identifiers against split labels. No
    target is read.

    Args:
        spine: The tables read back from the store.
        sample_ids: The identifiers the comparison manifest carries.

    Raises:
        SliceImportError: If any identifier lies outside the holdout.
    """
    holdout = set(
        spine.splits.loc[
            spine.splits[split_module.SPLIT_COLUMN] == split_module.TEST,
            split_module.SAMPLE_ID_COLUMN,
        ]
    )
    stray = sorted(set(sample_ids) - holdout)
    if stray:
        msg = (
            "cannot import the shared slice: the comparison manifest names "
            f"{len(stray)} row(s) outside the holdout, first {stray[0]!r}"
        )
        raise SliceImportError(msg)


def imported_counts(spine: Spine, manifests: Manifests) -> dict[str, int]:
    """Returns the row counts imported from the store.

    Args:
        spine: The tables read back from the store.
        manifests: The committed manifests written beside them.

    Returns:
        Count name to value.
    """
    return {
        "bookings": len(spine.bookings),
        "samples": len(spine.samples),
        "residents_with_bookings": int(spine.bookings["resident_id"].nunique()),
        "residents_with_samples": int(spine.samples["resident_id"].nunique()),
        "residents_excluded_no_prior_history": int(
            _lookup(
                manifests.sample,
                "counts",
                "residents_excluded_no_prior_history",
            )
        ),
        "comparison_rows": len(_lookup(manifests.comparison, "sample_ids")),
        "stored_predictions": len(spine.predictions),
    }


def comparator_scores(
    spine: Spine, config: config_module.Config, split_name: str
) -> dict[str, Any]:
    """Scores the shared baseline on one named split.

    Leakage contract: rows are restricted to ``split_name`` before any
    target column is read.

    Args:
        spine: The tables read back from the store.
        config: Validated configuration; supplies the notification
            tolerances, which are never redefined here.
        split_name: The split to score.

    Returns:
        The comparator's identity and its per-output figures.

    Raises:
        SliceImportError: If the comparator scored no row of that
            split.
    """
    labels = spine.splits.loc[
        spine.splits[split_module.SPLIT_COLUMN] == split_name,
        split_module.SAMPLE_ID_COLUMN,
    ]
    scored = spine.samples.loc[
        spine.samples[split_module.SAMPLE_ID_COLUMN].isin(set(labels))
    ]
    comparator = spine.predictions.loc[
        (spine.predictions[TRACK_COLUMN] == baselines.TRACK)
        & (spine.predictions[MODEL_COLUMN] == baselines.MODEL_NAME)
    ]
    joined = comparator.merge(
        scored, on=split_module.SAMPLE_ID_COLUMN, how="inner"
    )
    if joined.empty:
        msg = (
            "cannot import the shared slice: the comparator scored no "
            f"{split_name} row"
        )
        raise SliceImportError(msg)

    report = evaluate.evaluate_predictions(
        joined,
        config.evaluation.notification_match_ratio,
        config.evaluation.notification_support_minutes,
    )
    heads = report["heads"]
    matches = report["matches"]
    notification = heads[evaluate.NOTIFICATION]
    return {
        TRACK_COLUMN: baselines.TRACK,
        MODEL_COLUMN: baselines.MODEL_NAME,
        "split": split_name,
        "rows": int(report["rows"]),
        "facility_accuracy": heads[evaluate.FACILITY]["accuracy"],
        "usage_weekday_accuracy": heads[evaluate.USAGE_WEEKDAY]["accuracy"],
        "usage_hour_accuracy": heads[evaluate.USAGE_HOUR]["accuracy"],
        "notification_mae_minutes": notification["submitted"]["mae_minutes"],
        "notification_clamp_rate": notification["clamp_rate"],
        "score_mean": matches["score_mean"],
        "overall": matches["overall"],
        "strict": matches["strict"],
        "at_least_three": matches["at_least_three"],
    }


def build_record(
    spine: Spine,
    manifests: Manifests,
    sources: Mapping[str, str],
    config: config_module.Config,
    split_name: str,
) -> dict[str, Any]:
    """Assembles the import record.

    Args:
        spine: The tables read back from the store.
        manifests: The committed manifests written beside them.
        sources: Content hash per committed source file.
        config: Validated configuration.
        split_name: The split the comparator is scored on.

    Returns:
        The record payload, ready to serialise.
    """
    return {
        TRACK_COLUMN: llm.TRACK,
        "branch_state": llm.BRANCH_NOT_STARTED,
        "evidence": EVIDENCE,
        "provenance": {
            "seed": config.seed,
            "timezone": config.timezone,
            **table_digests(spine),
            **dict(sources),
        },
        "counts": imported_counts(spine, manifests),
        "split": {
            "basis": _lookup(manifests.split, "settings", "split_basis"),
            "cutoffs": dict(_lookup(manifests.split, "cutoffs")),
            "shape": dict(_lookup(manifests.split, "splits")),
        },
        "comparator": comparator_scores(spine, config, split_name),
    }


def check_measured(record: Mapping[str, Any], path: str = "") -> None:
    """Verifies that every value in a record is finite and present.

    Args:
        record: The record payload, or one of its nested mappings.
        path: Dotted prefix of the mapping being checked, for the
            message.

    Raises:
        SliceImportError: On the first entry that is not a measured
            value.
    """
    for key, value in record.items():
        where = f"{path}{key}"
        if isinstance(value, Mapping):
            check_measured(value, f"{where}.")
        elif isinstance(value, str):
            if not value.strip():
                msg = f"cannot import the shared slice: {where} is empty"
                raise SliceImportError(msg)
        elif isinstance(value, bool) or value is None:
            msg = (
                f"cannot import the shared slice: {where} is {value!r}, "
                "which measures nothing"
            )
            raise SliceImportError(msg)
        elif isinstance(value, int | float):
            if not math.isfinite(value):
                msg = f"cannot import the shared slice: {where} is {value!r}"
                raise SliceImportError(msg)
        else:
            msg = (
                f"cannot import the shared slice: {where} holds a "
                f"{type(value).__name__}, which is not a measured value"
            )
            raise SliceImportError(msg)


def run_params(record: Mapping[str, Any]) -> dict[str, str]:
    """Returns the run parameters for this import.

    Args:
        record: The record payload.

    Returns:
        Parameter name to value, as text.
    """
    params = {key: str(value) for key, value in record["provenance"].items()}
    params["branch_state"] = str(record["branch_state"])
    params["evidence"] = str(record["evidence"])
    params["comparator_split"] = str(record["comparator"]["split"])
    params["comparator_model"] = str(record["comparator"][MODEL_COLUMN])
    return params


def run_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """Returns the run metrics for this import.

    Args:
        record: The record payload.

    Returns:
        Metric name to value.
    """
    metrics = {
        f"count_{key}": float(value) for key, value in record["counts"].items()
    }
    metrics.update(
        {
            f"comparator_{key}": float(value)
            for key, value in record["comparator"].items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
    )
    return metrics
