"""Past-only summaries the prompt needs and the shared table lacks.

Everything the shared feature table already computes is read from it,
not recomputed. Three things it has no column for are computed here:
the recent-event sequence the prompt renders, the resident's commonest
prior usage hour, and a behaviour-change indicator.

Leakage contract: every function takes a resident's bookings and an
origin, and reads only rows with ``booking_timestamp <= origin``. A
later row cannot change any value returned here.
"""

from __future__ import annotations

import collections
import dataclasses
import math
from typing import Any

import pandas as pd

MINUTES_PER_HOUR = 60.0
HOURS_PER_DAY = 24.0

BOOKING_TIME = "booking_timestamp"
USAGE_TIME = "usage_timestamp"
FACILITY = "facility_id"
RESIDENT = "resident_id"

INSUFFICIENT = "change_history_insufficient"
CHANGED = "recent_behaviour_change"


@dataclasses.dataclass(frozen=True)
class Event:
    """One prior booking, as the prompt renders it.

    Attributes:
        facility: Facility name.
        booked_weekday: Weekday the booking was made, 0 is Monday.
        booked_hour: Hour the booking was made.
        booked_minute: Minute the booking was made.
        used_weekday: Weekday of usage, 0 is Monday.
        used_hour: Hour of usage.
        lead_hours: Hours between booking and usage.
        gap_days: Days since the previous booking, or None for the
            first event in the window.
    """

    facility: str
    booked_weekday: int
    booked_hour: int
    booked_minute: int
    used_weekday: int
    used_hour: int
    lead_hours: float
    gap_days: float | None


def past_bookings(
    bookings: pd.DataFrame, resident: str, origin: pd.Timestamp
) -> pd.DataFrame:
    """Returns one resident's bookings at or before the origin.

    Leakage contract: the bound is ``booking_timestamp <= origin``.

    Args:
        bookings: The full booking table.
        resident: The resident to select.
        origin: The prediction origin, timezone-aware.

    Returns:
        That resident's prior bookings, oldest first.

    Raises:
        ValueError: If ``origin`` is timezone-naive.
    """
    if origin.tzinfo is None:
        msg = f"origin must be timezone-aware, got {origin!r}"
        raise ValueError(msg)
    mine = bookings.loc[
        (bookings[RESIDENT] == resident) & (bookings[BOOKING_TIME] <= origin)
    ]
    return mine.sort_values([BOOKING_TIME, "booking_id"], kind="mergesort")


def recent_events(history: pd.DataFrame, depth: int) -> list[Event]:
    """Returns the last ``depth`` prior bookings, oldest first.

    Truncation drops the oldest rows only, so the newest behaviour is
    never the part that falls out of the window.

    Args:
        history: One resident's prior bookings, oldest first.
        depth: How many events the prompt carries.

    Returns:
        The events, oldest first, at most ``depth`` of them.
    """
    window = history.tail(depth)
    gaps = window[BOOKING_TIME].diff().dt.total_seconds() / (
        MINUTES_PER_HOUR * MINUTES_PER_HOUR * HOURS_PER_DAY
    )
    events = []
    for (_, row), gap in zip(window.iterrows(), gaps, strict=True):
        booked = row[BOOKING_TIME]
        used = row[USAGE_TIME]
        events.append(
            Event(
                facility=str(row[FACILITY]),
                booked_weekday=int(booked.weekday()),
                booked_hour=int(booked.hour),
                booked_minute=int(booked.minute),
                used_weekday=int(used.weekday()),
                used_hour=int(used.hour),
                lead_hours=(used - booked).total_seconds()
                / (MINUTES_PER_HOUR * MINUTES_PER_HOUR),
                gap_days=None if pd.isna(gap) else float(gap),
            )
        )
    return events


def preferred_usage_hour(history: pd.DataFrame) -> int | None:
    """Returns the resident's commonest prior usage hour.

    Ties go to the most recent hour, so the value does not depend on
    numeric order.

    Args:
        history: One resident's prior bookings, oldest first.

    Returns:
        The hour, or None when there is no prior booking.
    """
    hours = [int(stamp.hour) for stamp in history[USAGE_TIME]]
    if not hours:
        return None
    counts = collections.Counter(hours)
    best = max(counts.values())
    tied = [hour for hour, count in counts.items() if count == best]
    return next(hour for hour in reversed(hours) if hour in tied)


def _distribution(values: list[str], levels: tuple[str, ...]) -> list[float]:
    """Returns the share of each level in ``values``.

    Args:
        values: Observed levels.
        levels: The level set, in a fixed order.

    Returns:
        One share per level, summing to one, or zeros when empty.
    """
    if not values:
        return [0.0 for _ in levels]
    counts = collections.Counter(values)
    return [counts.get(level, 0) / len(values) for level in levels]


def jensen_shannon_distance(left: list[float], right: list[float]) -> float:
    """Returns the Jensen-Shannon distance between two distributions.

    Args:
        left: A distribution over the same ordered levels as ``right``.
        right: The other distribution.

    Returns:
        The distance in ``[0, 1]``, base-2 logarithms.

    Raises:
        ValueError: If the two distributions have different lengths.
    """
    if len(left) != len(right):
        msg = f"distributions differ in length: {len(left)} and {len(right)}"
        raise ValueError(msg)
    divergence = 0.0
    for first, second in zip(left, right, strict=True):
        middle = (first + second) / 2.0
        if middle == 0.0:
            continue
        for value in (first, second):
            if value > 0.0:
                divergence += 0.5 * value * math.log2(value / middle)
    return math.sqrt(max(divergence, 0.0))


def behaviour_change(
    history: pd.DataFrame,
    levels: tuple[str, ...],
    minimum_priors: int,
    recent_depth: int,
    threshold: float,
) -> dict[str, int]:
    """Flags a resident whose recent facility mix has shifted.

    Leakage contract: reads the prior bookings it is handed and nothing
    else, so the flag is a property of the past alone.

    Args:
        history: One resident's prior bookings, oldest first.
        levels: The facility catalog, in a fixed order.
        minimum_priors: Prior bookings needed before the comparison is
            attempted.
        recent_depth: How many of the newest bookings count as recent.
        threshold: Distance at or above which the flag is raised.

    Returns:
        The change flag and the insufficient-history flag, one of which
        is always zero.
    """
    facilities = [str(value) for value in history[FACILITY]]
    if len(facilities) < minimum_priors:
        return {CHANGED: 0, INSUFFICIENT: 1}
    recent = _distribution(facilities[-recent_depth:], levels)
    earlier = _distribution(facilities[:-recent_depth], levels)
    distance = jensen_shannon_distance(recent, earlier)
    return {CHANGED: int(distance >= threshold), INSUFFICIENT: 0}


def summarise(
    history: pd.DataFrame,
    levels: tuple[str, ...],
    minimum_priors: int,
    recent_depth: int,
    threshold: float,
) -> dict[str, Any]:
    """Returns the LLM-only summary values for one sample.

    Leakage contract: every value comes from ``history``, which the
    caller has already bounded at the origin.

    Args:
        history: One resident's prior bookings, oldest first.
        levels: The facility catalog, in a fixed order.
        minimum_priors: Prior bookings needed for the change flag.
        recent_depth: How many of the newest bookings count as recent.
        threshold: Distance at or above which the flag is raised.

    Returns:
        Summary name to value.
    """
    return {
        "preferred_usage_hour": preferred_usage_hour(history),
        **behaviour_change(
            history, levels, minimum_priors, recent_depth, threshold
        ),
    }
