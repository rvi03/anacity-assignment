"""Frequency and recency baselines — the number to beat.

Five rules, no learning. Each prediction is a count or a median taken
from the resident's own past, with a community-wide fallback for
whoever has no relevant history yet:

    facility       resident's most frequent prior facility
                   -> community mode
    usage weekday  resident's most frequent prior usage weekday
                   -> community mode
    usage hour     resident's most frequent prior usage hour
                   -> mode for the predicted facility -> community mode
    notification   resident's median prior booking-to-booking interval
                   -> community median
    transition     next facility from the resident's own order-1
                   transition counts, keyed on their last facility

The notification rule is the median of the target itself. No lead time
is subtracted: the inter-booking interval already lands on the next
booking moment, so subtracting one would be dimensionally wrong.

Ties are broken by the most recent occurrence, then by name or numeric
order, so a rerun predicts identically.

Leakage contract: fitting reads training rows only. Predicting for a
sample reads that resident's bookings with ``booking_timestamp <=
origin`` and nothing later; community fallbacks are the fitted
training-set quantities, never recomputed over the rows being scored.
No target of a scored row takes part in its own prediction.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import logging
from typing import Any

import pandas as pd

_LOGGER = logging.getLogger(__name__)

PREDICTION_COLUMNS = (
    "sample_id",
    "predicted_facility_id",
    "predicted_usage_weekday",
    "predicted_usage_hour",
    "predicted_delay_minutes",
    "predicted_transition_facility_id",
)

TRACK = "baseline"
MODEL_NAME = "frequency_recency"

_ORIGIN_COLUMN = "origin"
_RESIDENT_COLUMN = "resident_id"
_SAMPLE_ID_COLUMN = "sample_id"


@dataclasses.dataclass(frozen=True)
class CommunityFallbacks:
    """What to predict for a resident with no usable history.

    Every value is fitted on training rows only.

    Attributes:
        facility_id: Most booked facility across the training split.
        usage_weekday: Most used weekday.
        usage_hour: Most used hour of day.
        hour_by_facility: Most used hour per facility, for residents who
            have no hour history of their own.
        delay_minutes: Median booking-to-booking interval.
        transition_facility_id: Most common facility to follow each
            facility, keyed on the previous facility.
    """

    facility_id: str
    usage_weekday: int
    usage_hour: int
    hour_by_facility: dict[str, int]
    delay_minutes: float
    transition_facility_id: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return the fallbacks as a JSON-serialisable mapping."""
        return dataclasses.asdict(self)


def _mode(values: pd.Series) -> Any:
    """Return the most frequent value, ties broken by recency.

    ``values`` is expected in chronological order, so the last-seen of
    two equally frequent values wins.

    Args:
        values: Observations in chronological order.

    Returns:
        The winning value, or None when ``values`` is empty.
    """
    if values.empty:
        return None
    counts: collections.Counter[Any] = collections.Counter()
    last_seen: dict[Any, int] = {}
    for position, value in enumerate(values):
        counts[value] += 1
        last_seen[value] = position
    return max(counts, key=lambda value: (counts[value], last_seen[value]))


def _transition_targets(history: pd.DataFrame) -> dict[str, str]:
    """Count which facility follows which, over one booking sequence.

    Args:
        history: Bookings in chronological order, carrying
            ``resident_id`` and ``facility_id``.

    Returns:
        Previous facility to its most frequent successor.
    """
    followers: dict[str, list[str]] = collections.defaultdict(list)
    for _, group in history.groupby(_RESIDENT_COLUMN, sort=False):
        facilities = list(group["facility_id"])
        for previous, following in itertools.pairwise(facilities):
            followers[previous].append(following)
    return {
        previous: _mode(pd.Series(values))
        for previous, values in followers.items()
    }


def fit(
    bookings: pd.DataFrame, train_samples: pd.DataFrame
) -> CommunityFallbacks:
    """Fit the community fallbacks on training rows only.

    Leakage contract: reads bookings whose ``booking_timestamp`` is at
    or before the latest training origin, and the training samples'
    delays. No validation or holdout row takes part.

    Args:
        bookings: The full booking table.
        train_samples: Samples labelled as the training split.

    Returns:
        The fitted :class:`CommunityFallbacks`.

    Raises:
        ValueError: If there are no training samples to fit on.
    """
    if train_samples.empty:
        msg = "cannot fit baselines without training samples"
        raise ValueError(msg)

    horizon = train_samples[_ORIGIN_COLUMN].max()
    history = bookings.loc[
        bookings["booking_timestamp"] <= horizon
    ].sort_values(["booking_timestamp", "booking_id"], kind="mergesort")

    hours = history["usage_timestamp"].dt.hour
    hour_by_facility = {
        str(facility): int(_mode(hours.loc[group.index]))
        for facility, group in history.groupby("facility_id", sort=False)
    }
    return CommunityFallbacks(
        facility_id=str(_mode(history["facility_id"])),
        usage_weekday=int(_mode(history["usage_timestamp"].dt.weekday)),
        usage_hour=int(_mode(hours)),
        hour_by_facility=hour_by_facility,
        delay_minutes=float(
            train_samples["notification_delay_minutes"].median()
        ),
        transition_facility_id=_transition_targets(history),
    )


def _resident_histories(
    bookings: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Index bookings by resident, chronologically.

    Args:
        bookings: The full booking table.

    Returns:
        Resident id to that resident's bookings, sorted by creation
        time.
    """
    ordered = bookings.sort_values(
        ["booking_timestamp", "booking_id"], kind="mergesort"
    )
    return {
        str(resident): group
        for resident, group in ordered.groupby(_RESIDENT_COLUMN, sort=False)
    }


def _predict_one(
    history: pd.DataFrame, fallbacks: CommunityFallbacks
) -> dict[str, Any]:
    """Apply the five rules to one resident's past-only history.

    Leakage contract: ``history`` must already be truncated to
    ``booking_timestamp <= origin``. Nothing here re-reads the frame it
    is scoring.

    Args:
        history: That resident's bookings at or before the origin, in
            chronological order.
        fallbacks: The fitted community fallbacks.

    Returns:
        The five predictions for one sample.
    """
    facility = _mode(history["facility_id"]) or fallbacks.facility_id
    weekday = _mode(history["usage_timestamp"].dt.weekday)
    hour = _mode(history["usage_timestamp"].dt.hour)
    if hour is None:
        hour = fallbacks.hour_by_facility.get(
            str(facility), fallbacks.usage_hour
        )

    intervals = history["booking_timestamp"].diff().dropna()
    delay = (
        float(intervals.dt.total_seconds().median() / 60.0)
        if not intervals.empty
        else fallbacks.delay_minutes
    )

    last_facility = str(history["facility_id"].iloc[-1])
    own_transitions = _transition_targets(history)
    transition = own_transitions.get(
        last_facility,
        fallbacks.transition_facility_id.get(
            last_facility, fallbacks.facility_id
        ),
    )
    return {
        "predicted_facility_id": str(facility),
        "predicted_usage_weekday": int(
            weekday if weekday is not None else fallbacks.usage_weekday
        ),
        "predicted_usage_hour": int(hour),
        "predicted_delay_minutes": delay,
        "predicted_transition_facility_id": str(transition),
    }


def predict(
    samples: pd.DataFrame,
    bookings: pd.DataFrame,
    fallbacks: CommunityFallbacks,
) -> pd.DataFrame:
    """Predict all four outputs, plus the transition rule, per sample.

    Leakage contract: for each sample, the resident's history is
    truncated to ``booking_timestamp <= origin`` before any rule sees
    it. Community fallbacks come from :func:`fit` and are not
    recomputed over the rows being scored.

    Args:
        samples: Samples to score; needs ``sample_id``, ``resident_id``,
            and ``origin``.
        bookings: The full booking table.
        fallbacks: The fitted community fallbacks.

    Returns:
        One prediction row per sample, in the input's order.
    """
    histories = _resident_histories(bookings)
    rows = []
    for sample_id, resident, origin in zip(
        samples["sample_id"],
        samples[_RESIDENT_COLUMN],
        samples[_ORIGIN_COLUMN],
        strict=True,
    ):
        history = histories[str(resident)]
        past = history.loc[history["booking_timestamp"] <= origin]
        rows.append({"sample_id": sample_id, **_predict_one(past, fallbacks)})
    return pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
