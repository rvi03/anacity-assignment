"""Builds the notification delay ladder and its representation ceiling.

The notification output is generated as one label from a finite set, so
the delays have to be cut into buckets and each bucket given a single
minute value the model's answer resolves to.

    delays ──> geometric ladder ──> merge sparse ──> representatives
                                                          │
                              training coverage of each  ─┘
                              representative ──> CEILING

The cut is geometric on ``delay_minutes + 1`` because the shared
notification match is a symmetric multiplicative test: a bucket whose
upper bound is at most the match ratio squared above its lower bound is
covered end to end by its geometric midpoint. Wider buckets, and the
two unbounded ends, fall back to the training median and report
whatever coverage they actually reach.

CEILING is the notification match rate a perfect bucket classifier
would score with the frozen representatives. It is arithmetic over
training delays; no model is involved.

Leakage contract: every quantity here is computed from training rows
only. The caller selects those rows; no function reads a validation or
holdout target.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction.evaluation import evaluate
from facility_prediction.llm import settings as settings_module

# The shared match works on ``delay_minutes + 1``; the ladder is cut in
# the same transformed space so a bucket's width is the width the match
# rule sees. tests/llm/test_buckets.py fails if the two disagree.
TRANSFORM_OFFSET = 1.0

MINUTES_PER_DAY = 1440

GEOMETRIC = "geometric_midpoint"
EMPIRICAL = "training_median"

GATE = "the representation ceiling clears its configured floor"
RUN_NAME = "notification-buckets"


class BucketError(Exception):
    """Raised when the ladder cannot be built or fails its gate."""


@dataclasses.dataclass(frozen=True)
class Bucket:
    """One delay bucket.

    Attributes:
        label: The label a model generates for this bucket.
        lower: Inclusive lower bound in transformed space.
        upper: Exclusive upper bound in transformed space, or None for
            the unbounded top bucket.
        representative: The delay in minutes this label resolves to, or
            None before representatives are frozen.
        source: How the representative was chosen.
        train_rows: Training rows the bucket holds.
    """

    label: str
    lower: float
    upper: float | None
    representative: float | None = None
    source: str | None = None
    train_rows: int = 0

    @property
    def width(self) -> float | None:
        """Returns the bucket's multiplicative width, or None if open.

        Returns:
            ``upper / lower`` for a bounded bucket, otherwise None.
        """
        if self.upper is None:
            return None
        return self.upper / self.lower


def _to_delay(transformed: float) -> float:
    """Converts a transformed bound back to delay minutes.

    Args:
        transformed: A bound in transformed space.

    Returns:
        The same bound in minutes.
    """
    return transformed - TRANSFORM_OFFSET


def _label(lower: float, upper: float | None, floor: float) -> str:
    """Names a bucket after its bounds in whole minutes.

    Args:
        lower: Inclusive lower bound in transformed space.
        upper: Exclusive upper bound in transformed space, or None.
        floor: Transformed lower bound of the whole ladder.

    Returns:
        The label, derived from the bounds and nothing else.
    """
    if upper is None:
        return f"OVER_{round(_to_delay(lower))}M"
    if lower < floor:
        return f"UNDER_{round(_to_delay(upper))}M"
    return f"M{round(_to_delay(lower))}_{round(_to_delay(upper))}"


def build_ladder(buckets: settings_module.NotificationBuckets) -> list[Bucket]:
    """Cuts the delay range into a geometric ladder.

    The finite steps share one width, at most ``ratio``, so the ladder
    lands exactly on the ceiling bound instead of overshooting it.

    Args:
        buckets: The configured ladder settings.

    Returns:
        Buckets in ascending order: one floor bucket, the finite
        ladder, one unbounded ceiling bucket.

    Raises:
        BucketError: If the floor is not below the ceiling.
    """
    floor = buckets.floor_minutes + TRANSFORM_OFFSET
    ceiling = buckets.ceiling_days * MINUTES_PER_DAY + TRANSFORM_OFFSET
    if floor >= ceiling:
        msg = (
            f"floor {buckets.floor_minutes}m is not below ceiling "
            f"{buckets.ceiling_days}d"
        )
        raise BucketError(msg)

    steps = math.ceil(math.log(ceiling / floor) / math.log(buckets.ratio))
    edges = [
        floor * (ceiling / floor) ** (index / steps) for index in range(steps)
    ]
    edges.append(ceiling)

    ladder = [
        Bucket(
            label=_label(TRANSFORM_OFFSET, floor, floor),
            lower=TRANSFORM_OFFSET,
            upper=floor,
        )
    ]
    ladder += [
        Bucket(label=_label(low, high, floor), lower=low, upper=high)
        for low, high in itertools.pairwise(edges)
    ]
    ladder.append(
        Bucket(label=_label(ceiling, None, floor), lower=ceiling, upper=None)
    )
    return ladder


def assign(delays: pd.Series, ladder: list[Bucket]) -> pd.Series:
    """Assigns each delay to exactly one bucket label.

    Args:
        delays: Delays in minutes.
        ladder: Buckets in ascending order.

    Returns:
        The label of the bucket each delay falls in.

    Raises:
        BucketError: If a delay is negative, which the ladder does not
            cover.
    """
    if (delays < 0).any():
        msg = "a negative delay has no bucket"
        raise BucketError(msg)
    bounds = np.array([bucket.lower for bucket in ladder[1:]])
    index = np.searchsorted(
        bounds, delays.to_numpy() + TRANSFORM_OFFSET, side="right"
    )
    labels = np.array([bucket.label for bucket in ladder])
    return pd.Series(
        labels[index], index=delays.index, name="notification_bucket"
    )


def count_rows(ladder: list[Bucket], delays: pd.Series) -> list[Bucket]:
    """Records how many training rows each bucket holds.

    Args:
        ladder: Buckets in ascending order.
        delays: Training delays in minutes.

    Returns:
        The same ladder with ``train_rows`` filled in.
    """
    counts = assign(delays, ladder).value_counts()
    return [
        dataclasses.replace(bucket, train_rows=int(counts.get(bucket.label, 0)))
        for bucket in ladder
    ]


def merge_sparse(
    ladder: list[Bucket], delays: pd.Series, min_train_rows: int
) -> tuple[list[Bucket], list[dict[str, Any]]]:
    """Merges buckets that hold too few training rows.

    A sparse bucket joins its lower neighbour; the floor bucket has no
    lower neighbour and joins its upper one. Merging repeats until
    every bucket clears the threshold or one bucket is left.

    Args:
        ladder: Buckets in ascending order, with row counts.
        delays: Training delays in minutes.
        min_train_rows: Rows a bucket must hold.

    Returns:
        The merged ladder and one log entry per merge, in order.
    """
    current = count_rows(ladder, delays)
    log: list[dict[str, Any]] = []
    while len(current) > 1:
        sparse = next(
            (
                position
                for position, bucket in enumerate(current)
                if bucket.train_rows < min_train_rows
            ),
            None,
        )
        if sparse is None:
            break
        lower = sparse - 1 if sparse > 0 else sparse
        upper = lower + 1
        merged = Bucket(
            label=_label(
                current[lower].lower, current[upper].upper, current[1].lower
            ),
            lower=current[lower].lower,
            upper=current[upper].upper,
        )
        log.append(
            {
                "merged": [current[lower].label, current[upper].label],
                "into": merged.label,
                "train_rows_before": [
                    current[lower].train_rows,
                    current[upper].train_rows,
                ],
            }
        )
        current = [*current[:lower], merged, *current[upper + 1 :]]
        current = count_rows(current, delays)
    return current, log


def _representative(
    bucket: Bucket, delays: pd.Series, match_ratio: float, decimals: int
) -> tuple[float, str]:
    """Chooses the minute value a bucket's label resolves to.

    A bounded bucket no wider than ``match_ratio`` squared is covered
    end to end by its geometric midpoint. Anything wider, and both
    unbounded ends, use the training median instead.

    Args:
        bucket: The bucket to resolve.
        delays: Training delays in minutes falling in this bucket.
        match_ratio: The shared notification match ratio.
        decimals: Decimal places the representative is rounded to.

    Returns:
        The representative in minutes, and how it was chosen.

    Raises:
        BucketError: If a wide or unbounded bucket holds no training
            row to take a median from.
    """
    width = bucket.width
    narrow = width is not None and width <= match_ratio**2
    if bucket.upper is not None and narrow:
        midpoint = math.sqrt(bucket.lower * bucket.upper)
        return round(_to_delay(midpoint), decimals), GEOMETRIC
    if delays.empty:
        msg = f"bucket {bucket.label} has no training row to resolve"
        raise BucketError(msg)
    return round(float(delays.median()), decimals), EMPIRICAL


def resolve(
    ladder: list[Bucket],
    delays: pd.Series,
    match_ratio: float,
    decimals: int,
) -> list[Bucket]:
    """Freezes one representative minute value per bucket.

    Args:
        ladder: Buckets in ascending order.
        delays: Training delays in minutes.
        match_ratio: The shared notification match ratio.
        decimals: Decimal places a representative is rounded to.

    Returns:
        The ladder with representatives frozen.
    """
    labels = assign(delays, ladder)
    resolved = []
    for bucket in ladder:
        inside = delays.loc[labels == bucket.label]
        value, source = _representative(bucket, inside, match_ratio, decimals)
        resolved.append(
            dataclasses.replace(
                bucket,
                representative=value,
                source=source,
                train_rows=len(inside),
            )
        )
    return resolved


def coverage(delays: pd.Series, representative: float, ratio: float) -> float:
    """Returns the share of delays the representative matches.

    Uses the shared match rule, so a bucket's coverage is measured the
    way the scored metric measures it.

    Args:
        delays: Training delays in minutes.
        representative: The minute value the bucket resolves to.
        ratio: The shared notification match ratio.

    Returns:
        The matched share, or 0.0 for an empty bucket.
    """
    if delays.empty:
        return 0.0
    predicted = pd.Series(representative, index=delays.index)
    return float(evaluate.notification_match(delays, predicted, ratio).mean())


def representation_ceiling(
    ladder: list[Bucket], delays: pd.Series, match_ratio: float
) -> tuple[float, dict[str, float]]:
    """Returns the best notification rate this ladder can reach.

    Args:
        ladder: Buckets with frozen representatives.
        delays: Training delays in minutes.
        match_ratio: The shared notification match ratio.

    Returns:
        The ceiling, and the per-bucket coverage behind it.

    Raises:
        BucketError: If there are no training delays, or a
            representative is unresolved.
    """
    if delays.empty:
        msg = "cannot compute a ceiling without training delays"
        raise BucketError(msg)
    labels = assign(delays, ladder)
    per_bucket: dict[str, float] = {}
    ceiling = 0.0
    for bucket in ladder:
        if bucket.representative is None:
            msg = f"bucket {bucket.label} has no representative"
            raise BucketError(msg)
        inside = delays.loc[labels == bucket.label]
        covered = coverage(inside, bucket.representative, match_ratio)
        per_bucket[bucket.label] = covered
        ceiling += covered * len(inside) / len(delays)
    return ceiling, per_bucket


def check_partition(ladder: list[Bucket]) -> None:
    """Requires the ladder to partition every non-negative delay.

    Args:
        ladder: Buckets in ascending order.

    Raises:
        BucketError: If the ladder does not start at zero, leaves a gap
            or an overlap, or is not open at the top.
    """
    if not ladder:
        msg = "the ladder is empty"
        raise BucketError(msg)
    if ladder[0].lower != TRANSFORM_OFFSET:
        msg = f"the ladder starts above zero, at {_to_delay(ladder[0].lower)}"
        raise BucketError(msg)
    if ladder[-1].upper is not None:
        msg = "the ladder's top bucket is bounded, so long delays fall out"
        raise BucketError(msg)
    for lower, upper in itertools.pairwise(ladder):
        if lower.upper != upper.lower:
            msg = (
                f"{lower.label} ends at {lower.upper} but {upper.label} "
                f"starts at {upper.lower}"
            )
            raise BucketError(msg)


def check_widths(ladder: list[Bucket], ratio: float) -> None:
    """Requires every ladder step to be no wider than configured.

    The floor bucket is exempt: it runs down to zero delay, so its
    multiplicative width is not a property of the ladder.

    Args:
        ladder: Buckets in ascending order, before merging.
        ratio: The configured multiplicative width.

    Raises:
        BucketError: If a step of the ladder is wider than ``ratio``.
    """
    for bucket in ladder[1:]:
        width = bucket.width
        if width is not None and width > ratio:
            msg = f"{bucket.label} is {width:.4f} wide, above {ratio}"
            raise BucketError(msg)


def check_ratio(
    buckets: settings_module.NotificationBuckets, match_ratio: float
) -> None:
    """Requires the configured width to fit the shared match rule.

    Args:
        buckets: The configured ladder settings.
        match_ratio: The shared notification match ratio.

    Raises:
        BucketError: If a bucket can be wider than the tolerance its
            representative has to cover.
    """
    limit = match_ratio**2
    if buckets.ratio > limit:
        msg = (
            f"bucket ratio {buckets.ratio} exceeds the match ratio squared "
            f"({limit})"
        )
        raise BucketError(msg)


def build(
    buckets: settings_module.NotificationBuckets,
    delays: pd.Series,
    match_ratio: float,
) -> tuple[list[Bucket], list[dict[str, Any]]]:
    """Builds, merges, and freezes the ladder in one pass.

    Leakage contract: ``delays`` must be training rows only.

    Args:
        buckets: The configured ladder settings.
        delays: Training delays in minutes.
        match_ratio: The shared notification match ratio.

    Returns:
        The frozen ladder and its merge log.

    Raises:
        BucketError: If the configuration or the resulting ladder is
            not usable.
    """
    check_ratio(buckets, match_ratio)
    ladder = build_ladder(buckets)
    check_partition(ladder)
    check_widths(ladder, buckets.ratio)
    merged, log = merge_sparse(ladder, delays, buckets.min_train_rows)
    check_partition(merged)
    resolved = resolve(
        merged, delays, match_ratio, buckets.representative_decimals
    )
    return resolved, log


def load_ladder(manifest: dict[str, Any]) -> list[Bucket]:
    """Rebuilds the frozen ladder from its manifest.

    The manifest, not the database, is what a later step reads: the
    ladder is frozen once and every prompt after that resolves against
    the same table.

    Args:
        manifest: The payload written by :func:`build_manifest`.

    Returns:
        The buckets, ascending.

    Raises:
        BucketError: If the manifest carries no buckets.
    """
    recorded = manifest.get("buckets")
    if not recorded:
        msg = "the bucket manifest carries no buckets"
        raise BucketError(msg)
    ladder = [
        Bucket(
            label=str(entry["label"]),
            lower=entry["lower_minutes"] + TRANSFORM_OFFSET,
            upper=(
                None
                if entry["upper_minutes"] is None
                else entry["upper_minutes"] + TRANSFORM_OFFSET
            ),
            representative=entry["representative_minutes"],
            source=entry["representative_source"],
            train_rows=int(entry["train_rows"]),
        )
        for entry in recorded
    ]
    check_partition(ladder)
    return ladder


def check_ceiling(ceiling: float, gate: float) -> None:
    """Requires the ceiling to clear its configured floor.

    Args:
        ceiling: The computed representation ceiling.
        gate: The lowest acceptable value.

    Raises:
        BucketError: If the ceiling is below the gate.
    """
    if ceiling < gate:
        msg = (
            f"representation ceiling {ceiling:.4f} is below the required {gate}"
        )
        raise BucketError(msg)


def build_manifest(
    ladder: list[Bucket],
    log: list[dict[str, Any]],
    ceiling: float,
    per_bucket: dict[str, float],
    buckets: settings_module.NotificationBuckets,
    match_ratio: float,
) -> dict[str, Any]:
    """Assembles the record of the frozen ladder.

    Args:
        ladder: Buckets with frozen representatives.
        log: The merge log.
        ceiling: The representation ceiling.
        per_bucket: Coverage per bucket label.
        buckets: The configured ladder settings.
        match_ratio: The shared notification match ratio.

    Returns:
        The manifest payload, ready to serialise.
    """
    return {
        "settings": {
            **buckets.model_dump(),
            "notification_match_ratio": match_ratio,
        },
        "labels": [bucket.label for bucket in ladder],
        "buckets": [
            {
                "label": bucket.label,
                "lower_minutes": _to_delay(bucket.lower),
                "upper_minutes": (
                    None if bucket.upper is None else _to_delay(bucket.upper)
                ),
                "representative_minutes": bucket.representative,
                "representative_source": bucket.source,
                "train_rows": bucket.train_rows,
                "training_coverage": per_bucket[bucket.label],
            }
            for bucket in ladder
        ],
        "merges": log,
        "representation_ceiling": ceiling,
        "ceiling_gate": buckets.ceiling_gate,
        "gate_passed": ceiling >= buckets.ceiling_gate,
    }
