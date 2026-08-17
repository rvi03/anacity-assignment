"""Chronological split, frozen before any model exists.

The split is drawn on the **elapsed target-booking time span**, never on
sorted row positions or row-count quantiles: a busy month would
otherwise pull a boundary towards itself and let volume, rather than
chronology, decide what counts as the future.

    start = min(target)      span = max(target) - start
    train_cut = start + train_frac * span
    val_cut   = start + (train_frac + val_frac) * span

    train        [start, train_cut)
    validation   [train_cut, val_cut)
    test         [val_cut, end]

Two artifacts are written before a model is fitted, and neither is ever
redrawn: the split manifest, which carries the timezone-aware cutoffs
and the realised counts and date ranges, and the comparison manifest —
a seeded uniform draw of holdout sample ids, capped by configuration,
that both tracks are scored on.

Leakage contract: reads ``target_booking_timestamp`` only, to place the
two boundaries. It reads no label, no feature, and no model output. The
comparison draw is uniform over sorted sample ids and never stratified
on a target, so it cannot be nudged by a result nobody has seen yet.
Samples sharing an identical target timestamp always land in the same
partition, because the boundary is a time, not a position.

The frozen labels are also what every fit call is audited against: rows
go into a fit only if this module placed them in an allowed partition,
and a rebuilt manifest that moves a boundary is refused rather than
quietly adopted.
"""

from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import itertools
import json
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.data import samples as samples_module

_LOGGER = logging.getLogger(__name__)

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
SPLIT_NAMES = (TRAIN, VALIDATION, TEST)

SPLIT_COLUMN = "split"
TARGET_TIME_COLUMN = "target_booking_timestamp"
SAMPLE_ID_COLUMN = "sample_id"


@dataclasses.dataclass(frozen=True)
class Cutoffs:
    """The two frozen boundaries, and the span they were cut from.

    Attributes:
        start: Earliest target booking timestamp observed.
        train_cut: First instant that is no longer training.
        val_cut: First instant that is holdout.
        end: Latest target booking timestamp observed.
    """

    start: pd.Timestamp
    train_cut: pd.Timestamp
    val_cut: pd.Timestamp
    end: pd.Timestamp

    def to_dict(self) -> dict[str, str]:
        """Return the cutoffs as ISO 8601 strings with their offset."""
        return {
            "start": self.start.isoformat(),
            "train_cut": self.train_cut.isoformat(),
            "val_cut": self.val_cut.isoformat(),
            "end": self.end.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str]) -> Cutoffs:
        """Rebuild cutoffs written by :meth:`to_dict`.

        Args:
            payload: The ``cutoffs`` mapping from a split manifest.

        Returns:
            The reconstructed, timezone-aware :class:`Cutoffs`.
        """
        return cls(
            start=pd.Timestamp(payload["start"]),
            train_cut=pd.Timestamp(payload["train_cut"]),
            val_cut=pd.Timestamp(payload["val_cut"]),
            end=pd.Timestamp(payload["end"]),
        )


def _check_samples(frame: pd.DataFrame) -> None:
    """Reject a sample table the splitter cannot read safely.

    Args:
        frame: Candidate sample table.

    Raises:
        ValueError: If it is empty, misses a required column, or carries
            a timezone-naive target timestamp.
    """
    if frame.empty:
        msg = "cannot split an empty sample table"
        raise ValueError(msg)

    missing = [
        column
        for column in (SAMPLE_ID_COLUMN, TARGET_TIME_COLUMN)
        if column not in frame.columns
    ]
    if missing:
        msg = f"samples is missing required columns: {missing}"
        raise ValueError(msg)

    if not isinstance(frame[TARGET_TIME_COLUMN].dtype, pd.DatetimeTZDtype):
        msg = (
            f"{TARGET_TIME_COLUMN} must be timezone-aware, got dtype "
            f"{frame[TARGET_TIME_COLUMN].dtype}"
        )
        raise ValueError(msg)


def compute_cutoffs(
    frame: pd.DataFrame, config: config_module.Config
) -> Cutoffs:
    """Place the two boundaries on the elapsed target-time span.

    Leakage contract: reads ``target_booking_timestamp`` only. The
    boundaries are instants, so identical target timestamps cannot be
    separated by them.

    Args:
        frame: Samples as returned by the sampler.
        config: Validated configuration; supplies the two fractions and
            the embargo setting.

    Returns:
        The frozen :class:`Cutoffs`.

    Raises:
        ValueError: If the sample table is unusable, or a non-zero
            embargo is configured — that is a validation-only
            sensitivity and is not part of the frozen split.
    """
    _check_samples(frame)
    if config.split.embargo_days != 0:
        msg = (
            "embargo_days is a validation-only sensitivity and is not "
            f"applied to the frozen split, got {config.split.embargo_days}"
        )
        raise ValueError(msg)

    target = frame[TARGET_TIME_COLUMN]
    start = target.min()
    end = target.max()
    span = end - start
    val_fraction = config.split.train_frac + config.split.val_frac
    return Cutoffs(
        start=start,
        train_cut=start + span * config.split.train_frac,
        val_cut=start + span * val_fraction,
        end=end,
    )


def assign_split(frame: pd.DataFrame, cutoffs: Cutoffs) -> pd.DataFrame:
    """Label every sample train, validation, or test.

    Leakage contract: membership depends on the target booking time and
    the frozen cutoffs alone.

    Args:
        frame: Samples as returned by the sampler.
        cutoffs: The frozen boundaries.

    Returns:
        A copy of ``frame`` with a ``split`` column appended.
    """
    _check_samples(frame)
    target = frame[TARGET_TIME_COLUMN]
    labelled = frame.copy()
    labelled[SPLIT_COLUMN] = np.where(
        target < cutoffs.train_cut,
        TRAIN,
        np.where(target < cutoffs.val_cut, VALIDATION, TEST),
    )
    return labelled


def check_boundaries(frame: pd.DataFrame) -> None:
    """Assert the partitions do not overlap in time.

    Args:
        frame: A labelled frame from :func:`assign_split`.

    Raises:
        ValueError: If any train target is not strictly earlier than
            every validation target, or any validation target is not
            strictly earlier than every test target.
    """
    bounds = frame.groupby(SPLIT_COLUMN)[TARGET_TIME_COLUMN].agg(["min", "max"])
    ordered = [name for name in SPLIT_NAMES if name in bounds.index]
    for earlier, later in itertools.pairwise(ordered):
        if bounds.loc[earlier, "max"] >= bounds.loc[later, "min"]:
            msg = (
                f"{earlier} must end strictly before {later} begins: "
                f"{bounds.loc[earlier, 'max']} >= {bounds.loc[later, 'min']}"
            )
            raise ValueError(msg)


def _realised_shape(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarise each partition's realised size and date range.

    Args:
        frame: A labelled frame from :func:`assign_split`.

    Returns:
        Split name to row count and first/last target timestamp. An
        empty partition reports zero rows and null bounds.
    """
    shape: dict[str, dict[str, Any]] = {}
    for name in SPLIT_NAMES:
        rows = frame.loc[frame[SPLIT_COLUMN] == name, TARGET_TIME_COLUMN]
        shape[name] = {
            "rows": len(rows),
            "share": round(len(rows) / len(frame), 6),
            "first": rows.min().isoformat() if len(rows) else None,
            "last": rows.max().isoformat() if len(rows) else None,
        }
    return shape


def build_split_manifest(
    frame: pd.DataFrame,
    cutoffs: Cutoffs,
    config: config_module.Config,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the manifest that freezes this split.

    Args:
        frame: A labelled frame from :func:`assign_split`.
        cutoffs: The frozen boundaries.
        config: Validated configuration; its fractions are recorded so
            the manifest explains itself without the config file.
        provenance: Seed, timezone, and input hashes.

    Returns:
        The manifest payload, ready to serialise.
    """
    return {
        "provenance": provenance,
        "settings": {
            "train_frac": config.split.train_frac,
            "val_frac": config.split.val_frac,
            "embargo_days": config.split.embargo_days,
            "comparison_rows": config.split.comparison_rows,
            "split_basis": TARGET_TIME_COLUMN,
        },
        "cutoffs": cutoffs.to_dict(),
        "splits": _realised_shape(frame),
        "samples": len(frame),
    }


def draw_comparison_manifest(
    frame: pd.DataFrame, config: config_module.Config
) -> tuple[str, ...]:
    """Draw the seeded holdout rows both tracks are compared on.

    Leakage contract: a uniform draw over sorted holdout sample ids. No
    target, prediction, or feature takes part, so the draw is fixed
    before any result exists and is never redrawn afterwards.

    Args:
        frame: A labelled frame from :func:`assign_split`.
        config: Validated configuration; supplies the seed and the cap.

    Returns:
        Sample ids in ascending order. A holdout no larger than the cap
        is returned whole.
    """
    holdout = sorted(frame.loc[frame[SPLIT_COLUMN] == TEST, SAMPLE_ID_COLUMN])
    if len(holdout) <= config.split.comparison_rows:
        return tuple(holdout)

    rng = np.random.default_rng(config.seed)
    chosen = rng.choice(
        np.array(holdout),
        size=config.split.comparison_rows,
        replace=False,
    )
    return tuple(sorted(chosen.tolist()))


FIT_SPLITS = (TRAIN,)

# What the manifest asserts about membership, as opposed to how it came
# to be made. A rebuild may legitimately record new provenance; it may
# never move a boundary or change who is on which side of one.
MEMBERSHIP_FIELDS = ("settings", "cutoffs", "splits", "samples")


def check_fit_membership(
    sample_ids: Sequence[str],
    splits: pd.DataFrame,
    allowed: Sequence[str] = FIT_SPLITS,
) -> int:
    """Audit the rows handed to a fit call before it happens.

    A model may only be fitted on rows the frozen split placed in an
    allowed partition. Calling this immediately before ``fit`` turns
    "we only trained on training rows" from a claim into a check.

    Leakage contract: reads split labels only. A row's partition is not
    a target, so this audit can run before any scoring pass without
    touching a sealed label.

    Args:
        sample_ids: The sample ids about to be fitted on.
        splits: The frozen labels, carrying ``sample_id`` and ``split``.
        allowed: Partitions a fit may legitimately read.

    Returns:
        The number of rows audited.

    Raises:
        ValueError: If any id is absent from the frozen labels, or
            belongs to a partition outside ``allowed``.
    """
    labels = dict(
        zip(splits[SAMPLE_ID_COLUMN], splits[SPLIT_COLUMN], strict=True)
    )
    unknown = sorted({str(name) for name in sample_ids if name not in labels})
    if unknown:
        msg = (
            f"{len(unknown)} rows in this fit are absent from the frozen "
            f"split, first: {unknown[:5]}"
        )
        raise ValueError(msg)

    permitted = set(allowed)
    intruders = sorted(
        {
            f"{name} ({labels[name]})"
            for name in sample_ids
            if labels[name] not in permitted
        }
    )
    if intruders:
        msg = (
            f"{len(intruders)} rows outside {sorted(permitted)} reached a "
            f"fit call, first: {intruders[:5]}"
        )
        raise ValueError(msg)
    return len(sample_ids)


def check_manifest_unchanged(
    rebuilt: dict[str, Any], frozen: dict[str, Any]
) -> None:
    """Reject a split that has moved since it was frozen.

    Args:
        rebuilt: A manifest built from the current samples.
        frozen: The manifest committed when the split was drawn.

    Raises:
        ValueError: If any membership-defining field differs.
    """
    moved = [
        name
        for name in MEMBERSHIP_FIELDS
        if rebuilt.get(name) != frozen.get(name)
    ]
    if moved:
        msg = f"the frozen split has moved; these fields differ: {moved}"
        raise ValueError(msg)


def write_json(payload: dict[str, Any], path: pathlib.Path) -> str:
    """Write a manifest and return its content hash.

    Args:
        payload: The manifest payload.
        path: Destination JSON; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _LOGGER.info("wrote %s (sha256=%s)", path, digest)
    return digest


def load_manifest(path: pathlib.Path) -> dict[str, Any]:
    """Read a manifest written by :func:`write_json`.

    Args:
        path: Path to the manifest.

    Returns:
        The parsed payload.
    """
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def load_samples(
    path: pathlib.Path, config: config_module.Config
) -> pd.DataFrame:
    """Read the sample CSV back as a timezone-aware frame.

    Args:
        path: The CSV written by the sampler.
        config: Validated configuration; supplies the timezone the
            timestamps are converted back into.

    Returns:
        The sample table with every timestamp column timezone-aware.
    """
    frame = pd.read_csv(path)
    for column in samples_module.TIMESTAMP_COLUMNS:
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(config.tzinfo)
    return frame
