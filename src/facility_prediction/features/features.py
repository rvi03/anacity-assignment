"""Feature table — what was knowable at each prediction origin.

Every column here answers one question: at the moment the origin
booking was created, what did the past say about this booking? The
target booking never contributes to its own features, and neither does
any event created after the origin.

    this resident's bookings with     calendar facts of the origin
    booking_timestamp <= origin  -->  counts, rhythm, preferences,   -->  one
                                      timing                              row
    every resident's bookings with    what the community was doing
    booking_timestamp <  origin  -->  before this booking existed

The two bounds differ by design. A resident's own history includes the
origin event, because the origin *is* one of their bookings and its
existence is the thing that starts the prediction. The community's
history excludes it, and everything simultaneous with it, so that no row
can read an aggregate it helped to form.

Two conventions make the table safe to hand to a gradient-boosted
model without a preprocessing step in between:

    categoricals  always text, never null, never a float. An absent
                  slot carries the single configured token, so there is
                  exactly one spelling of "not known yet".
    numerics      always float64, missing as NaN. A count column that
                  is integral in one split and float in another is the
                  classic train/evaluation dtype drift, so every numeric
                  column is float everywhere by construction.

Leakage contract: a resident's history is truncated by the timestamp
bound ``booking_timestamp <= origin`` — derived here from the origin
itself, not inherited from the sampler's bookkeeping. The prefix length
is cross-checked against the sample's recorded prior-booking count, and
a disagreement is an error rather than a silent choice. Community
aggregates are truncated by the stricter bound ``booking_timestamp <
origin``; every window over them is left-closed and right-open, so the
origin instant is never inside one. No label and no later event is read.
"""

from __future__ import annotations

import collections
from collections.abc import Sequence
import dataclasses
import json
import logging
import pathlib
import re
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandera.pandas as pa

from facility_prediction import config as config_module
from facility_prediction.data import digest as digest_module
from facility_prediction.data import generate
from facility_prediction.data import samples as samples_module

_LOGGER = logging.getLogger(__name__)

IDENTIFIER_COLUMNS = ("sample_id", "resident_id", "booking_id")
_RESIDENT_PROFILE_COLUMN = "resident_profile_id"

# The labels themselves, taken from the sampler's contract rather than
# retyped, so renaming a target renames what may not be a feature.
TARGET_COLUMNS = tuple(
    name
    for name in samples_module.SAMPLE_COLUMNS
    if name.startswith("target_") or name == "notification_delay_minutes"
)

# A target-derived column rarely arrives under the target's own name; it
# arrives as `target_facility_share` or `delay_label`. Any feature whose
# name carries one of these fragments is refused without argument.
DENYLIST_FRAGMENTS = ("target", "notification_delay", "label")

MODEL_EXCLUSIONS = {
    "sample_id": "a row key, not an observation of the resident",
    "booking_id": "a row key; it identifies the target, so it must not "
    "reach a model that predicts the target",
    "resident_id": "the history summaries already carry personalisation; "
    "the raw identifier is kept only so a reviewer can trace a row",
}

_CALENDAR_FEATURES = (
    "origin_weekday",
    "origin_hour",
    "origin_month",
    "origin_week_of_year",
    "origin_hour_sin",
    "origin_hour_cos",
    "origin_weekday_sin",
    "origin_weekday_cos",
    "origin_month_sin",
    "origin_month_cos",
)
_RECENCY_FEATURES = (
    "days_since_previous_booking",
    "days_since_first_booking",
)
_INTERVAL_FEATURES = (
    "inter_booking_interval_mean_days",
    "inter_booking_interval_median_days",
    "inter_booking_interval_std_days",
)
_LAST_USAGE_FEATURES = ("last_usage_weekday", "last_usage_hour")
_BOOKING_ROUTINE_FEATURES = (
    "last_booking_weekday",
    "last_booking_hour",
    "booking_weekday_mode",
    "days_until_booking_weekday_mode",
    "cadence_progress_ratio",
)
_LEAD_FEATURES = (
    "lead_minutes_mean",
    "lead_minutes_median",
    "lead_minutes_recent",
)
_ENTROPY_FEATURES = (
    "facility_entropy_bits",
    "facility_entropy_normalised",
)
_TRANSITION_TOTAL = "transitions_from_last_total"

_ORIGIN_COLUMN = "origin"
_RESIDENT_COLUMN = "resident_id"
_TARGET_BOOKING_COLUMN = "target_booking_id"
_TARGET_TIME_COLUMN = "target_booking_timestamp"
_PRIOR_COUNT_COLUMN = "n_prior_bookings"

_HOURS_IN_DAY = 24
_DAYS_IN_WEEK = 7
_MONTHS_IN_YEAR = 12
_MINUTES_IN_HOUR = 60.0
_NANOSECONDS_IN_DAY = 86_400_000_000_000.0
_MIN_INTERVALS_FOR_SPREAD = 2
_HALVING = 0.5

# Missingness spelt as text is the failure this table exists to prevent:
# a categorical column holding "nan" reads as a level, not as a gap.
_FORBIDDEN_TEXT = frozenset(
    {"", "nan", "none", "null", "na", "<na>", "n/a", "nat"}
)


class FeatureSchemaError(Exception):
    """Raised when a feature table violates its declared schema."""


def _slug(name: str) -> str:
    """Return a column-safe form of a catalog name.

    Args:
        name: A facility or time-band name as written in configuration.

    Returns:
        Lowercase text with every run of non-alphanumeric characters
        replaced by a single underscore.
    """
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def categorical_feature_names(
    config: config_module.Config,
) -> tuple[str, ...]:
    """Return the text-valued feature columns, in table order.

    Args:
        config: Validated configuration; supplies how many recent
            facility slots and which rolling-preference depths to keep.

    Returns:
        Column names, every one of which is a string column carrying the
        configured missing token where the history is too short.
    """
    slots = config.features.history_facility_slots
    return (
        _RESIDENT_PROFILE_COLUMN,
        *(f"last_{slot}_facility" for slot in range(1, slots + 1)),
        *(
            f"rolling_top_facility_last_{window}"
            for window in config.features.rolling_preference_bookings
        ),
    )


def numeric_feature_names(config: config_module.Config) -> tuple[str, ...]:
    """Return the float-valued feature columns, in table order.

    Args:
        config: Validated configuration; supplies the facility catalog,
            the prior windows, the trend pair, the time bands, the
            rolling-preference depths, and the community windows.

    Returns:
        Column names, every one of which is a float64 column. The
        resident's own history comes first, then what the community was
        doing before the origin.
    """
    short_days, long_days = config.features.trend_windows_days
    facilities = tuple(_slug(name) for name in config.facility_names)
    bands = tuple(_slug(band.name) for band in config.features.time_bands)
    return (
        _PRIOR_COUNT_COLUMN,
        *_CALENDAR_FEATURES,
        *_RECENCY_FEATURES,
        *(
            f"bookings_prior_{days}d"
            for days in config.features.prior_windows_days
        ),
        f"booking_rate_ratio_{short_days}d_{long_days}d",
        *_INTERVAL_FEATURES,
        *_BOOKING_ROUTINE_FEATURES,
        *_LAST_USAGE_FEATURES,
        *(f"facility_count_{slug}" for slug in facilities),
        *(f"facility_share_{slug}" for slug in facilities),
        *(f"usage_weekday_count_{day}" for day in range(_DAYS_IN_WEEK)),
        *(f"usage_weekday_share_{day}" for day in range(_DAYS_IN_WEEK)),
        *(f"time_band_count_{slug}" for slug in bands),
        *(f"time_band_share_{slug}" for slug in bands),
        *(
            f"rolling_top_facility_share_last_{window}"
            for window in config.features.rolling_preference_bookings
        ),
        *_LEAD_FEATURES,
        *(f"ewma_facility_share_{slug}" for slug in facilities),
        *(f"ewma_time_band_share_{slug}" for slug in bands),
        *_ENTROPY_FEATURES,
        _TRANSITION_TOTAL,
        *(f"transition_count_from_last_{slug}" for slug in facilities),
        *(f"transition_share_from_last_{slug}" for slug in facilities),
        *(f"community_facility_share_{slug}" for slug in facilities),
        *(
            f"community_facility_share_origin_weekday_{slug}"
            for slug in facilities
        ),
        *(
            f"community_facility_share_origin_band_{slug}"
            for slug in facilities
        ),
        *(
            f"community_facility_share_{days}d_{slug}"
            for days in config.features.community_windows_days
            for slug in facilities
        ),
        *(f"community_lead_minutes_mean_{slug}" for slug in facilities),
    )


def feature_columns(config: config_module.Config) -> tuple[str, ...]:
    """Return every modelling column, identifiers excluded.

    Args:
        config: Validated configuration.

    Returns:
        Categorical columns followed by numeric columns.
    """
    return (
        *categorical_feature_names(config),
        *numeric_feature_names(config),
    )


def denylist_violations(columns: Sequence[str]) -> tuple[str, ...]:
    """Return the columns that name a target or something derived from one.

    Args:
        columns: Candidate model-input column names.

    Returns:
        The offending names in sorted order; empty when none offend.
    """
    denied = {
        name
        for name in columns
        if name in TARGET_COLUMNS
        or any(fragment in name.lower() for fragment in DENYLIST_FRAGMENTS)
    }
    return tuple(sorted(denied))


def check_denylist(columns: Sequence[str]) -> None:
    """Reject a model-input column set that carries a target.

    Args:
        columns: Candidate model-input column names.

    Raises:
        ValueError: If any column names a target or is derived from one.
    """
    denied = denylist_violations(columns)
    if denied:
        msg = (
            "these columns are targets or target-derived and must not be "
            f"model inputs: {list(denied)}"
        )
        raise ValueError(msg)


def table_columns(config: config_module.Config) -> tuple[str, ...]:
    """Return the full feature-table column order.

    Args:
        config: Validated configuration.

    Returns:
        Identifier columns followed by :func:`feature_columns`.
    """
    return (*IDENTIFIER_COLUMNS, *feature_columns(config))


@dataclasses.dataclass(frozen=True, eq=False)
class _History:
    """One resident's bookings as parallel arrays, oldest first.

    Attributes:
        booking_ts: Creation instants, strictly increasing.
        facility: Facility booked, aligned with ``booking_ts``.
        facility_index: The same facility as its position in the
            configured catalog, so counts can be taken with a bincount.
        usage_weekday: Weekday of the usage instant, 0 is Monday.
        usage_hour: Hour of day of the usage instant.
        usage_band: Index into the configured time bands.
        lead_minutes: Minutes from creation to usage.
    """

    booking_ts: pd.DatetimeIndex
    facility: tuple[str, ...]
    facility_index: npt.NDArray[np.int64]
    usage_weekday: npt.NDArray[np.int64]
    usage_hour: npt.NDArray[np.int64]
    usage_band: npt.NDArray[np.int64]
    lead_minutes: npt.NDArray[np.float64]


def _band_lookup(config: config_module.Config) -> npt.NDArray[np.int64]:
    """Return an hour-of-day to time-band index table.

    Args:
        config: Validated configuration; its bands partition the day.

    Returns:
        A length-24 array mapping each hour to its band index.
    """
    lookup = np.zeros(_HOURS_IN_DAY, dtype=np.int64)
    for index, band in enumerate(config.features.time_bands):
        lookup[band.open_hour : band.close_hour] = index
    return lookup


def _facility_indices(
    values: pd.Series, config: config_module.Config
) -> npt.NDArray[np.int64]:
    """Map facility names onto their position in the catalog.

    Args:
        values: Facility names as booked.
        config: Validated configuration; its catalog is the enum.

    Returns:
        One catalog position per input value.

    Raises:
        ValueError: If any value is not in the configured catalog. A
            facility the catalog does not know would silently drop out
            of every count that is taken by position.
    """
    positions = {
        name: index for index, name in enumerate(config.facility_names)
    }
    unknown = sorted(set(values.astype(str)) - set(positions))
    if unknown:
        msg = f"these facilities are not in the configured catalog: {unknown}"
        raise ValueError(msg)
    return values.astype(str).map(positions).to_numpy(dtype=np.int64)


def _resident_histories(
    bookings: pd.DataFrame, config: config_module.Config
) -> dict[str, _History]:
    """Index the booking table by resident, chronologically.

    Args:
        bookings: The full booking table.
        config: Validated configuration; supplies the time bands.

    Returns:
        Resident id to that resident's :class:`_History`.
    """
    ordered = bookings.sort_values(
        [_RESIDENT_COLUMN, "booking_timestamp", "booking_id"], kind="mergesort"
    )
    lookup = _band_lookup(config)
    histories: dict[str, _History] = {}
    for resident, group in ordered.groupby(_RESIDENT_COLUMN, sort=False):
        booking = group["booking_timestamp"]
        usage = group["usage_timestamp"]
        hours = usage.dt.hour.to_numpy(dtype=np.int64)
        histories[str(resident)] = _History(
            booking_ts=pd.DatetimeIndex(booking),
            facility=tuple(str(value) for value in group["facility_id"]),
            facility_index=_facility_indices(group["facility_id"], config),
            usage_weekday=usage.dt.weekday.to_numpy(dtype=np.int64),
            usage_hour=hours,
            usage_band=lookup[hours],
            lead_minutes=(
                (usage - booking).dt.total_seconds() / _MINUTES_IN_HOUR
            ).to_numpy(dtype=np.float64),
        )
    return histories


def _most_frequent(values: Sequence[str]) -> str | None:
    """Return the commonest value, ties broken by the latest occurrence.

    ``values`` is expected oldest-first, so between two equally
    frequent facilities the more recently used one wins.

    Args:
        values: Observations in chronological order.

    Returns:
        The winning value, or None when ``values`` is empty.
    """
    if not values:
        return None
    counts: collections.Counter[str] = collections.Counter(values)
    last_seen = {value: position for position, value in enumerate(values)}
    return max(counts, key=lambda value: (counts[value], last_seen[value]))


def _cyclical(values: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    """Return sine and cosine encodings of a periodic quantity.

    Args:
        values: Zero-based positions within the cycle.
        period: Length of the cycle.

    Returns:
        The sine and cosine components, so that the last position of a
        cycle sits next to the first rather than at the far end of a
        line.
    """
    angle = 2.0 * np.pi * values.astype("float64") / period
    return np.sin(angle), np.cos(angle)


def _calendar_features(origins: pd.Series) -> pd.DataFrame:
    """Describe the prediction origin as calendar quantities.

    Leakage contract: reads the origin instant and nothing else. The
    origin is by construction an event that has already happened, so no
    bound is needed beyond using the origin itself.

    Args:
        origins: Timezone-aware origin instants, one per sample.

    Returns:
        One row per origin, carrying the plain calendar parts and the
        cyclical encodings of hour, weekday, and month.
    """
    weekday = origins.dt.weekday
    hour = origins.dt.hour
    month = origins.dt.month
    hour_sin, hour_cos = _cyclical(hour, _HOURS_IN_DAY)
    weekday_sin, weekday_cos = _cyclical(weekday, _DAYS_IN_WEEK)
    month_sin, month_cos = _cyclical(month - 1, _MONTHS_IN_YEAR)
    return pd.DataFrame(
        {
            "origin_weekday": weekday,
            "origin_hour": hour,
            "origin_month": month,
            "origin_week_of_year": origins.dt.isocalendar().week,
            "origin_hour_sin": hour_sin,
            "origin_hour_cos": hour_cos,
            "origin_weekday_sin": weekday_sin,
            "origin_weekday_cos": weekday_cos,
            "origin_month_sin": month_sin,
            "origin_month_cos": month_cos,
        }
    )


def _recency_features(
    history: _History, origin: pd.Timestamp, prior: int
) -> dict[str, float]:
    """Measure how long the resident has been quiet, and how long known.

    Leakage contract: reads ``history`` positions ``< prior`` only, all
    of which satisfy ``booking_timestamp <= origin``.

    Args:
        history: The resident's full history.
        origin: The prediction origin.
        prior: Number of leading history rows at or before the origin.

    Returns:
        Days from the booking before the origin to the origin, and days
        from the resident's first observed booking to the origin. The
        first is NaN when the origin booking is the only one.
    """
    previous = (
        (origin - history.booking_ts[prior - 2]).total_seconds()
        if prior >= _MIN_INTERVALS_FOR_SPREAD
        else float("nan")
    )
    first = (origin - history.booking_ts[0]).total_seconds()
    seconds_in_day = _NANOSECONDS_IN_DAY / 1e9
    return {
        "days_since_previous_booking": float(previous) / seconds_in_day,
        "days_since_first_booking": float(first) / seconds_in_day,
    }


def _window_counts(
    history: _History,
    origin: pd.Timestamp,
    prior: int,
    config: config_module.Config,
) -> dict[str, float]:
    """Count recent bookings, and compare a short window to a long one.

    Leakage contract: every window is the half-open interval
    ``(origin - window, origin]``, so it can only ever contain events
    that had already happened.

    Args:
        history: The resident's full history.
        origin: The prediction origin.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration; supplies the window lengths and
            the trend pair.

    Returns:
        One count per configured window, plus the rate ratio of the
        short window to the long one. The ratio is NaN when the long
        window is empty and there is nothing to compare against.
    """
    prefix = history.booking_ts[:prior]
    counts: dict[str, float] = {}
    for days in config.features.prior_windows_days:
        start = origin - pd.Timedelta(days=days)
        earlier = int(prefix.searchsorted(start, side="right"))
        counts[f"bookings_prior_{days}d"] = float(prior - earlier)

    short_days, long_days = config.features.trend_windows_days
    short_count = counts[f"bookings_prior_{short_days}d"]
    long_count = counts[f"bookings_prior_{long_days}d"]
    ratio = (
        (short_count / short_days) / (long_count / long_days)
        if long_count > 0
        else float("nan")
    )
    counts[f"booking_rate_ratio_{short_days}d_{long_days}d"] = ratio
    return counts


def _interval_features(history: _History, prior: int) -> dict[str, float]:
    """Summarise the resident's booking-to-booking rhythm.

    The interval measured is ``booking_timestamp`` to
    ``booking_timestamp``, the quantity the notification output
    predicts. A usage-to-usage gap is never substituted.

    Leakage contract: reads ``history`` positions ``< prior`` only.

    Args:
        history: The resident's full history.
        prior: Number of leading history rows at or before the origin.

    Returns:
        Mean, median, and sample standard deviation of the intervals in
        days. All three are NaN with fewer than two bookings; the spread
        is NaN with fewer than three, where one interval cannot have
        one.
    """
    if prior < _MIN_INTERVALS_FOR_SPREAD:
        return dict.fromkeys(_INTERVAL_FEATURES, float("nan"))

    gaps = np.diff(history.booking_ts[:prior].asi8) / _NANOSECONDS_IN_DAY
    spread = (
        float(gaps.std(ddof=1))
        if gaps.size >= _MIN_INTERVALS_FOR_SPREAD
        else float("nan")
    )
    return {
        "inter_booking_interval_mean_days": float(gaps.mean()),
        "inter_booking_interval_median_days": float(np.median(gaps)),
        "inter_booking_interval_std_days": spread,
    }


def _booking_routine_features(
    history: _History, origin: pd.Timestamp, prior: int
) -> dict[str, float]:
    """Describe the resident's observed booking rhythm.

    The mode and cadence are computed from booking creation times, not
    usage times: they are directly relevant to the next-booking and
    notification targets.  A resident profile is deliberately not
    inferred here; this function reads only their already-observed
    events.

    Leakage contract: reads ``history`` positions ``< prior`` only.

    Args:
        history: The resident's full booking history.
        origin: Prediction instant, equal to the last known booking.
        prior: Number of known bookings in the history prefix.

    Returns:
        Latest booking calendar values, the resident's modal booking
        weekday, days until that weekday recurs, and elapsed cadence as
        a fraction of the resident's median historical interval.
    """
    booking = history.booking_ts[:prior]
    weekdays = booking.weekday
    counts = np.bincount(weekdays, minlength=_DAYS_IN_WEEK)
    last_seen = {
        int(weekday): position for position, weekday in enumerate(weekdays)
    }
    mode = max(
        range(_DAYS_IN_WEEK),
        key=lambda weekday: (int(counts[weekday]), last_seen.get(weekday, -1)),
    )
    days_until = (mode - int(origin.weekday())) % _DAYS_IN_WEEK
    if days_until == 0:
        days_until = _DAYS_IN_WEEK

    cadence_progress = float("nan")
    if prior >= _MIN_INTERVALS_FOR_SPREAD:
        gaps = np.diff(booking.asi8) / _NANOSECONDS_IN_DAY
        median_gap = float(np.median(gaps))
        elapsed = (origin - booking[prior - 2]).total_seconds() / 86_400.0
        cadence_progress = (
            elapsed / median_gap if median_gap > 0 else float("nan")
        )

    latest = booking[prior - 1]
    return {
        "last_booking_weekday": float(latest.weekday()),
        "last_booking_hour": float(latest.hour),
        "booking_weekday_mode": float(mode),
        "days_until_booking_weekday_mode": float(days_until),
        "cadence_progress_ratio": cadence_progress,
    }


def _distribution_features(
    history: _History, prior: int, config: config_module.Config
) -> dict[str, float]:
    """Count where, when, and in which part of the day the resident goes.

    Shares are reported beside counts: a count says how active somebody
    is, a share says what they prefer.

    Leakage contract: reads ``history`` positions ``< prior`` only.

    Args:
        history: The resident's full history.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration; supplies the facility catalog
            and the time bands.

    Returns:
        Per-facility, per-weekday, and per-time-band counts and shares.
    """
    total = float(prior)
    features: dict[str, float] = {}

    facilities = collections.Counter(history.facility[:prior])
    for name in config.facility_names:
        count = float(facilities.get(name, 0))
        features[f"facility_count_{_slug(name)}"] = count
        features[f"facility_share_{_slug(name)}"] = count / total

    weekdays = np.bincount(
        history.usage_weekday[:prior], minlength=_DAYS_IN_WEEK
    )
    for day in range(_DAYS_IN_WEEK):
        features[f"usage_weekday_count_{day}"] = float(weekdays[day])
        features[f"usage_weekday_share_{day}"] = float(weekdays[day]) / total

    bands = np.bincount(
        history.usage_band[:prior], minlength=len(config.features.time_bands)
    )
    for index, band in enumerate(config.features.time_bands):
        features[f"time_band_count_{_slug(band.name)}"] = float(bands[index])
        features[f"time_band_share_{_slug(band.name)}"] = (
            float(bands[index]) / total
        )
    return features


def _preference_features(
    history: _History, prior: int, config: config_module.Config
) -> dict[str, Any]:
    """Name the recent facilities, and the favourite of each recent run.

    Leakage contract: reads ``history`` positions ``< prior`` only.

    Args:
        history: The resident's full history.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration; supplies the missing token, the
            number of recent slots, and the rolling depths.

    Returns:
        The recent facility slots and rolling favourites as text — the
        configured token where the history is too short — beside the
        favourite's share of each rolling window as a float.
    """
    token = config.features.categorical_missing_token
    features: dict[str, Any] = {}
    for slot in range(1, config.features.history_facility_slots + 1):
        features[f"last_{slot}_facility"] = (
            history.facility[prior - slot] if prior >= slot else token
        )

    for window in config.features.rolling_preference_bookings:
        recent = history.facility[max(0, prior - window) : prior]
        favourite = _most_frequent(recent)
        features[f"rolling_top_facility_last_{window}"] = (
            favourite if favourite is not None else token
        )
        features[f"rolling_top_facility_share_last_{window}"] = (
            float(recent.count(favourite)) / len(recent)
            if favourite is not None
            else float("nan")
        )
    return features


def _timing_features(history: _History, prior: int) -> dict[str, float]:
    """Describe when the resident uses a booking, and how far ahead.

    Leakage contract: reads ``history`` positions ``< prior`` only.

    Args:
        history: The resident's full history.
        prior: Number of leading history rows at or before the origin.

    Returns:
        The weekday and hour of the most recent usage, and the mean,
        median, and most recent booking lead time in minutes.
    """
    leads = history.lead_minutes[:prior]
    return {
        "last_usage_weekday": float(history.usage_weekday[prior - 1]),
        "last_usage_hour": float(history.usage_hour[prior - 1]),
        "lead_minutes_mean": float(leads.mean()),
        "lead_minutes_median": float(np.median(leads)),
        "lead_minutes_recent": float(leads[-1]),
    }


def _ewma_features(
    history: _History,
    origin: pd.Timestamp,
    prior: int,
    config: config_module.Config,
) -> dict[str, float]:
    """Weight the resident's past so that recent bookings count most.

    A plain share treats a booking from a year ago as evidence of what
    somebody wants today. These columns halve an event's weight for
    every configured half-life it sits before the origin, which is what
    makes a changed habit visible while it is still changing.

    Leakage contract: reads ``history`` positions ``< prior`` only, and
    weights them by their distance back from the origin, so no future
    event can carry weight.

    Args:
        history: The resident's full history.
        origin: The prediction origin.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration; supplies the catalog, the time
            bands, and the half-life.

    Returns:
        Decayed shares over the facility catalog and over the time
        bands. Each group sums to one, or is NaN when every weight has
        decayed to zero.
    """
    ages = (
        origin.value - history.booking_ts[:prior].asi8
    ) / _NANOSECONDS_IN_DAY
    weights = np.power(
        _HALVING, ages / config.features.ewma_halflife_days, dtype=np.float64
    )
    facilities = config.facility_names
    bands = config.features.time_bands
    facility_mass = np.bincount(
        history.facility_index[:prior],
        weights=weights,
        minlength=len(facilities),
    )
    band_mass = np.bincount(
        history.usage_band[:prior], weights=weights, minlength=len(bands)
    )
    total = float(weights.sum())
    features: dict[str, float] = {}
    for index, name in enumerate(facilities):
        features[f"ewma_facility_share_{_slug(name)}"] = (
            float(facility_mass[index]) / total if total > 0 else float("nan")
        )
    for index, band in enumerate(bands):
        features[f"ewma_time_band_share_{_slug(band.name)}"] = (
            float(band_mass[index]) / total if total > 0 else float("nan")
        )
    return features


def _entropy_features(
    history: _History, prior: int, config: config_module.Config
) -> dict[str, float]:
    """Say whether the resident is habitual or exploratory.

    Two residents with the same favourite facility are not the same
    resident if one of them books nothing else and the other books
    everything. Entropy is the one number that separates them.

    Leakage contract: reads ``history`` positions ``< prior`` only.

    Args:
        history: The resident's full history.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration; supplies the catalog, whose
            size sets the maximum entropy.

    Returns:
        Shannon entropy of the resident's facility distribution in
        bits, and the same value as a fraction of the most a catalog
        this size allows. The fraction is NaN for a one-facility
        catalog, where there is no spread to measure.
    """
    counts = np.bincount(
        history.facility_index[:prior], minlength=len(config.facility_names)
    )
    shares = counts[counts > 0] / float(prior)
    entropy = float(-(shares * np.log2(shares)).sum())
    ceiling = np.log2(len(config.facility_names))
    return {
        "facility_entropy_bits": entropy,
        "facility_entropy_normalised": (
            entropy / float(ceiling) if ceiling > 0 else float("nan")
        ),
    }


def _transition_features(
    history: _History, prior: int, config: config_module.Config
) -> dict[str, float]:
    """Count where this resident has gone after the facility they just used.

    Leakage contract: reads ``history`` positions ``< prior`` only. The
    transitions counted all ended at or before the origin.

    Args:
        history: The resident's full history.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration; supplies the catalog.

    Returns:
        The number of past transitions out of the most recent facility,
        the count into each catalog facility, and each count's share.
        Shares are NaN until such a transition has happened at all.
    """
    catalog = config.facility_names
    prefix = history.facility_index[:prior]
    counts = np.zeros(len(catalog), dtype=np.float64)
    if prior > 1:
        sources, destinations = prefix[:-1], prefix[1:]
        from_last = destinations[sources == prefix[-1]]
        counts = np.bincount(from_last, minlength=len(catalog)).astype(
            np.float64
        )

    total = float(counts.sum())
    features: dict[str, float] = {_TRANSITION_TOTAL: total}
    for index, name in enumerate(catalog):
        features[f"transition_count_from_last_{_slug(name)}"] = float(
            counts[index]
        )
        features[f"transition_share_from_last_{_slug(name)}"] = (
            float(counts[index]) / total if total > 0 else float("nan")
        )
    return features


def _history_row(
    history: _History,
    origin: pd.Timestamp,
    prior: int,
    config: config_module.Config,
) -> dict[str, Any]:
    """Summarise one resident's past as of one prediction origin.

    Leakage contract: reads only the leading ``prior`` history rows, all
    of which satisfy ``booking_timestamp <= origin``. Nothing later, and
    no other resident's events, take part.

    Args:
        history: The resident's full history.
        origin: The prediction origin, timezone-aware.
        prior: Number of leading history rows at or before the origin.
        config: Validated configuration.

    Returns:
        Feature name to value for one sample, identifiers excluded.
    """
    return {
        _PRIOR_COUNT_COLUMN: float(prior),
        **_recency_features(history, origin, prior),
        **_window_counts(history, origin, prior, config),
        **_interval_features(history, prior),
        **_booking_routine_features(history, origin, prior),
        **_timing_features(history, prior),
        **_distribution_features(history, prior, config),
        **_preference_features(history, prior, config),
        **_ewma_features(history, origin, prior, config),
        **_entropy_features(history, prior, config),
        **_transition_features(history, prior, config),
    }


@dataclasses.dataclass(frozen=True, eq=False)
class _Community:
    """Every resident's bookings as prefix sums, oldest first.

    A prefix sum is what makes "the community as of this instant"
    affordable: the aggregate over any window is one subtraction, so a
    row never rescans the table and never has the chance to read past
    its own bound.

    Attributes:
        booking_ts: Creation instants of every booking, ascending.
        cum_count: Bookings per facility up to each position, with a
            leading row of zeros so position ``i`` means "the first
            ``i`` events".
        cum_lead: The same running sum over lead minutes.
        cum_count_weekday: ``cum_count`` restricted to each weekday of
            the booking instant, indexed weekday-first.
        cum_count_band: ``cum_count`` restricted to each time band of
            the booking instant, indexed band-first.
    """

    booking_ts: pd.DatetimeIndex
    cum_count: npt.NDArray[np.float64]
    cum_lead: npt.NDArray[np.float64]
    cum_count_weekday: npt.NDArray[np.float64]
    cum_count_band: npt.NDArray[np.float64]


def _prefix_sums(
    values: npt.NDArray[np.float64], mask: npt.NDArray[np.bool_] | None = None
) -> npt.NDArray[np.float64]:
    """Return running totals with a leading zero row.

    Args:
        values: One row per event, one column per facility.
        mask: Optional per-event filter; masked-out events contribute
            nothing.

    Returns:
        An array one row longer than ``values``, whose row ``i`` totals
        the first ``i`` events.
    """
    selected = values if mask is None else values * mask[:, None]
    totals = np.zeros((values.shape[0] + 1, values.shape[1]), dtype=np.float64)
    totals[1:] = np.cumsum(selected, axis=0)
    return totals


def _community_index(
    bookings: pd.DataFrame, config: config_module.Config
) -> _Community:
    """Index every booking in the community by creation instant.

    Leakage contract: this builds prefix sums only. Nothing here selects
    a window; :func:`_window_bounds` is the single place a bound is
    chosen, so there is one implementation of "before the origin" rather
    than one per feature.

    Args:
        bookings: The full booking table.
        config: Validated configuration; supplies the catalog and the
            time bands.

    Returns:
        The community index, sorted ascending by creation instant.
    """
    ordered = bookings.sort_values(
        ["booking_timestamp", "booking_id"], kind="mergesort"
    )
    booking = ordered["booking_timestamp"]
    facility = _facility_indices(ordered["facility_id"], config)
    weekday = booking.dt.weekday.to_numpy(dtype=np.int64)
    band = _band_lookup(config)[booking.dt.hour.to_numpy(dtype=np.int64)]
    lead = (
        (ordered["usage_timestamp"] - booking).dt.total_seconds()
        / _MINUTES_IN_HOUR
    ).to_numpy(dtype=np.float64)

    facilities = len(config.facility_names)
    one_hot = np.zeros((len(ordered), facilities), dtype=np.float64)
    one_hot[np.arange(len(ordered)), facility] = 1.0
    bands = len(config.features.time_bands)
    return _Community(
        booking_ts=pd.DatetimeIndex(booking),
        cum_count=_prefix_sums(one_hot),
        cum_lead=_prefix_sums(one_hot * lead[:, None]),
        cum_count_weekday=np.stack(
            [
                _prefix_sums(one_hot, weekday == day)
                for day in range(_DAYS_IN_WEEK)
            ]
        ),
        cum_count_band=np.stack(
            [_prefix_sums(one_hot, band == index) for index in range(bands)]
        ),
    )


def _window_bounds(
    community: _Community, origin: pd.Timestamp, span_days: int | None = None
) -> tuple[int, int]:
    """Return the half-open event range a community window covers.

    Leakage contract: both ends are located with ``side='left'``, so the
    range is ``[origin - span, origin)``. It is closed on the left and
    open on the right, which is what keeps the origin instant — and
    every event simultaneous with it — outside every community
    aggregate. This is the only place a community bound is computed.

    Args:
        community: The community index.
        origin: The prediction origin.
        span_days: Window length in days, or None for everything known
            before the origin.

    Returns:
        The first and last-plus-one positions in the index.
    """
    end = int(community.booking_ts.searchsorted(origin, side="left"))
    if span_days is None:
        return 0, end
    start = origin - pd.Timedelta(days=span_days)
    return int(community.booking_ts.searchsorted(start, side="left")), end


def _check_community_bound(
    community: _Community,
    origin: pd.Timestamp,
    end: int,
    sample_id: str,
) -> None:
    """Assert the community aggregate stops before the origin.

    The row-level statement the community columns rest on, checked on
    every row rather than argued for once. It is an ``if … raise`` and
    not an ``assert`` so that running with optimisations cannot switch
    the guarantee off.

    Args:
        community: The community index.
        origin: The prediction origin.
        end: Last-plus-one position the aggregate reads.
        sample_id: Identifier used in the error message.

    Raises:
        ValueError: If the newest community event read was created at or
            after the origin.
    """
    if end > 0:
        latest = community.booking_ts[end - 1]
        if latest >= origin:
            msg = (
                f"sample {sample_id} reads a community booking created at "
                f"{latest.isoformat()}, not before its origin "
                f"{origin.isoformat()}"
            )
            raise ValueError(msg)


def _shares(
    counts: npt.NDArray[np.float64], names: Sequence[str], prefix: str
) -> dict[str, float]:
    """Turn per-facility counts into named shares.

    Args:
        counts: One count per catalog facility.
        names: The catalog, in the same order.
        prefix: Column-name prefix the facility slug is appended to.

    Returns:
        One column per facility. Every share is NaN when the counts are
        all zero, because no proportion exists to report.
    """
    total = float(counts.sum())
    return {
        f"{prefix}{_slug(name)}": (
            float(counts[index]) / total if total > 0 else float("nan")
        )
        for index, name in enumerate(names)
    }


def _community_features(
    community: _Community,
    origin: pd.Timestamp,
    end: int,
    config: config_module.Config,
) -> dict[str, float]:
    """Describe what the whole community was doing before this booking.

    This is the information a per-resident history cannot hold: which
    facilities the community favours, how that has moved recently, and
    how far ahead people book each one. A facility that opened last
    month, or a season that is turning, shows up here and nowhere else.

    Every column is a proportion or a per-booking average. The running
    event counts they are built from are deliberately *not* columns: a
    count that only ever grows is a clock, and under a chronological
    split every evaluation row would carry a value no training row ever
    had. A model cannot split on a clock it has not seen.

    Leakage contract: every window comes from :func:`_window_bounds` and
    is therefore left-closed and right-open at the origin. The origin's
    own booking is outside all of them.

    Args:
        community: The community index.
        origin: The prediction origin.
        end: Last-plus-one position of the events before the origin.
        config: Validated configuration; supplies the catalog and the
            community windows.

    Returns:
        Overall popularity, popularity on the origin's weekday and in
        the origin's time band, popularity over each configured window,
        and mean lead time per facility. Every share is NaN while the
        aggregate it would describe is empty.
    """
    names = config.facility_names
    counts = community.cum_count[end]
    weekday = community.cum_count_weekday[origin.weekday()][end]
    band = community.cum_count_band[_band_lookup(config)[origin.hour]][end]
    lead = community.cum_lead[end]

    features: dict[str, float] = {}
    features.update(_shares(counts, names, "community_facility_share_"))
    features.update(
        _shares(weekday, names, "community_facility_share_origin_weekday_")
    )
    features.update(
        _shares(band, names, "community_facility_share_origin_band_")
    )
    for days in config.features.community_windows_days:
        start, _ = _window_bounds(community, origin, days)
        windowed = counts - community.cum_count[start]
        features.update(
            _shares(windowed, names, f"community_facility_share_{days}d_")
        )
    for index, name in enumerate(names):
        features[f"community_lead_minutes_mean_{_slug(name)}"] = (
            float(lead[index]) / float(counts[index])
            if counts[index] > 0
            else float("nan")
        )
    return features


def _check_inputs(samples: pd.DataFrame, bookings: pd.DataFrame) -> None:
    """Reject inputs the feature builder cannot read safely.

    Args:
        samples: Candidate sample table.
        bookings: Candidate booking table.

    Raises:
        ValueError: If a contract column is missing or a timestamp
            column is timezone-naive.
    """
    required = (
        "sample_id",
        _RESIDENT_COLUMN,
        _ORIGIN_COLUMN,
        _TARGET_BOOKING_COLUMN,
        _TARGET_TIME_COLUMN,
        _PRIOR_COUNT_COLUMN,
    )
    missing = [name for name in required if name not in samples.columns]
    if missing:
        msg = f"samples is missing required columns: {missing}"
        raise ValueError(msg)

    absent = [name for name in generate.BOOKING_COLUMNS if name not in bookings]
    if absent:
        msg = f"bookings is missing required columns: {absent}"
        raise ValueError(msg)

    naive = [
        name
        for name, series in (
            (_ORIGIN_COLUMN, samples[_ORIGIN_COLUMN]),
            (_TARGET_TIME_COLUMN, samples[_TARGET_TIME_COLUMN]),
            ("booking_timestamp", bookings["booking_timestamp"]),
            ("usage_timestamp", bookings["usage_timestamp"]),
        )
        if not isinstance(series.dtype, pd.DatetimeTZDtype)
    ]
    if naive:
        msg = f"these columns must be timezone-aware: {naive}"
        raise ValueError(msg)


def _prefix_length(
    history: _History,
    origin: pd.Timestamp,
    recorded: int,
    sample_id: str,
) -> int:
    """Return how many of a resident's bookings precede the origin.

    Leakage contract: the length is derived from the bound
    ``booking_timestamp <= origin`` alone. The sampler's recorded count
    is used only to detect disagreement, never to widen the window.

    Args:
        history: The resident's full history.
        origin: The prediction origin.
        recorded: The prior-booking count the sampler wrote.
        sample_id: Identifier used in the error message.

    Returns:
        The number of leading history rows at or before the origin.

    Raises:
        ValueError: If no booking precedes the origin, or the derived
            length disagrees with the recorded count.
    """
    prior = int(history.booking_ts.searchsorted(origin, side="right"))
    if prior < 1:
        msg = (
            f"sample {sample_id} has no booking at or before its origin "
            f"{origin.isoformat()}"
        )
        raise ValueError(msg)
    if prior != recorded:
        msg = (
            f"sample {sample_id} records {recorded} prior bookings but the "
            f"origin bound admits {prior}"
        )
        raise ValueError(msg)
    return prior


def _check_as_of(
    history: _History,
    origin: pd.Timestamp,
    prior: int,
    target_ts: pd.Timestamp,
    sample_id: str,
) -> None:
    """Assert one row's history is past and its target is future.

    This is the row-level statement the whole table rests on, checked on
    every row rather than argued for once in a docstring. It is an
    ``if … raise`` and not an ``assert`` so that running with
    optimisations cannot switch the guarantee off.

    Leakage contract: the newest history event read must satisfy
    ``booking_timestamp <= origin``, and the target booking must satisfy
    ``booking_timestamp > origin``. Both bounds are checked here.

    Args:
        history: The resident's full history.
        origin: The prediction origin.
        prior: Number of leading history rows the features were built
            from.
        target_ts: Creation instant of the target booking.
        sample_id: Identifier used in the error message.

    Raises:
        ValueError: If the newest event used is later than the origin,
            or the target booking was not created after it.
    """
    latest = history.booking_ts[prior - 1]
    if latest > origin:
        msg = (
            f"sample {sample_id} reads a booking created at "
            f"{latest.isoformat()}, after its origin {origin.isoformat()}"
        )
        raise ValueError(msg)
    if target_ts <= origin:
        msg = (
            f"sample {sample_id} has a target created at "
            f"{target_ts.isoformat()}, not after its origin "
            f"{origin.isoformat()}"
        )
        raise ValueError(msg)


def build_features(
    samples: pd.DataFrame,
    bookings: pd.DataFrame,
    config: config_module.Config,
) -> pd.DataFrame:
    """Build one feature row per prediction sample.

    Leakage contract: for each sample the resident's history is
    truncated to ``booking_timestamp <= origin`` and the community's to
    ``booking_timestamp < origin`` before any summary is taken. No
    label and no later event is read, so a row's features are identical
    whether or not the future exists in the input.

    Args:
        samples: Rolling-origin samples carrying ``sample_id``,
            ``resident_id``, ``origin``, ``target_booking_id``, and
            ``n_prior_bookings``.
        bookings: The full booking table.
        config: Validated configuration.

    Returns:
        The feature table in :func:`table_columns` order, one row per
        sample in the input's order, categoricals as text and every
        other feature as float64.

    Raises:
        ValueError: If a contract column is missing, a timestamp is
            naive, a declared feature column names a target, a facility
            is outside the catalog, a resident is absent from the
            booking table, a sample's recorded history length disagrees
            with its origin, or any row reads an event later than its
            origin, a community event at or after it, or a target that
            is not after it.
    """
    _check_inputs(samples, bookings)
    check_denylist(feature_columns(config))
    histories = _resident_histories(bookings, config)
    community = _community_index(bookings, config)

    rows = []
    events = 0
    community_events = 0
    for sample_id, resident, origin, target_ts, recorded in zip(
        samples["sample_id"],
        samples[_RESIDENT_COLUMN],
        samples[_ORIGIN_COLUMN],
        samples[_TARGET_TIME_COLUMN],
        samples[_PRIOR_COUNT_COLUMN],
        strict=True,
    ):
        history = histories.get(str(resident))
        if history is None:
            msg = f"sample {sample_id} names unknown resident {resident}"
            raise ValueError(msg)
        prior = _prefix_length(history, origin, int(recorded), str(sample_id))
        _check_as_of(history, origin, prior, target_ts, str(sample_id))
        _, end = _window_bounds(community, origin)
        _check_community_bound(community, origin, end, str(sample_id))
        events += prior
        community_events += end
        rows.append(
            {
                **_history_row(history, origin, prior, config),
                **_community_features(community, origin, end, config),
            }
        )

    _LOGGER.info(
        "as-of check passed on %d rows over %d history events and %d "
        "community events",
        len(rows),
        events,
        community_events,
    )

    identifiers = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].astype(str).to_numpy(),
            "resident_id": samples[_RESIDENT_COLUMN].astype(str).to_numpy(),
            "booking_id": samples[_TARGET_BOOKING_COLUMN]
            .astype(str)
            .to_numpy(),
            _RESIDENT_PROFILE_COLUMN: samples[_RESIDENT_COLUMN]
            .astype(str)
            .to_numpy(),
        }
    )
    calendar = _calendar_features(
        samples[_ORIGIN_COLUMN].reset_index(drop=True)
    )
    frame = pd.concat(
        [identifiers, calendar, pd.DataFrame(rows)], axis=1
    ).reset_index(drop=True)
    return _cast(frame, config)


def _cast(frame: pd.DataFrame, config: config_module.Config) -> pd.DataFrame:
    """Fix every column's dtype so two splits cannot disagree on one.

    Args:
        frame: The assembled feature table.
        config: Validated configuration.

    Returns:
        The same table in :func:`table_columns` order, text columns as
        `object` holding `str`, every feature column as float64.
    """
    typed = frame.copy()
    text = (*IDENTIFIER_COLUMNS, *categorical_feature_names(config))
    for name in text:
        typed[name] = typed[name].astype(str)
    for name in numeric_feature_names(config):
        typed[name] = typed[name].astype("float64")
    return typed[list(table_columns(config))]


def _is_text(series: pd.Series) -> pd.Series:
    """Return whether each value really is a string object.

    Args:
        series: A column expected to hold text.

    Returns:
        A boolean mask, False wherever a float, None, or other non-text
        object has been written into a categorical slot.
    """
    return series.map(lambda value: isinstance(value, str))


def _is_not_missing_text(series: pd.Series) -> pd.Series:
    """Return whether each value avoids spelling missingness as a level.

    Args:
        series: A column expected to hold text.

    Returns:
        A boolean mask, False wherever a value is empty or reads as a
        rendered null.
    """
    return ~series.astype(str).str.strip().str.lower().isin(_FORBIDDEN_TEXT)


def feature_schema(config: config_module.Config) -> pa.DataFrameSchema:
    """Return the declarative contract the feature table must satisfy.

    Args:
        config: Validated configuration; the column set is derived from
            it, so a config change cannot leave the schema behind.

    Returns:
        A strict, ordered schema: identifiers and categoricals are
        non-null text that never spells missingness, and every feature
        column is nullable float64.
    """
    text_checks = [
        pa.Check(_is_text, element_wise=False, name="values_are_text"),
        pa.Check(
            _is_not_missing_text,
            element_wise=False,
            name="no_missingness_as_text",
        ),
    ]
    columns: dict[str, pa.Column] = {
        name: pa.Column(str, nullable=False, checks=text_checks)
        for name in (*IDENTIFIER_COLUMNS, *categorical_feature_names(config))
    }
    for name in numeric_feature_names(config):
        columns[name] = pa.Column("float64", nullable=True)
    return pa.DataFrameSchema(
        {name: columns[name] for name in table_columns(config)},
        strict=True,
        ordered=True,
        coerce=False,
        name="feature_table",
    )


def validate_features(
    frame: pd.DataFrame, config: config_module.Config
) -> pd.DataFrame:
    """Check a feature table against :func:`feature_schema`.

    Args:
        frame: The table to check.
        config: Validated configuration.

    Returns:
        The same table, unchanged.

    Raises:
        FeatureSchemaError: If any column is missing, out of order,
            wrongly typed, null where it must not be, or carries
            missingness written as text.
    """
    try:
        return feature_schema(config).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as error:
        msg = f"feature table violates its schema:\n{error.failure_cases}"
        raise FeatureSchemaError(msg) from error
    except pa.errors.SchemaError as error:
        msg = f"feature table violates its schema: {error}"
        raise FeatureSchemaError(msg) from error


def check_consistent_dtypes(
    train: pd.DataFrame, evaluation: pd.DataFrame
) -> None:
    """Reject a train/evaluation pair that disagrees on any column.

    A count column that is integral in training and float at evaluation
    time changes what a categorical boundary means.

    Args:
        train: Features for the training rows.
        evaluation: Features for the rows to be scored.

    Raises:
        FeatureSchemaError: If the column order differs, or any column's
            dtype differs between the two frames.
    """
    if list(train.columns) != list(evaluation.columns):
        msg = (
            "train and evaluation feature tables have different columns: "
            f"{sorted(set(train.columns) ^ set(evaluation.columns))}"
        )
        raise FeatureSchemaError(msg)

    mismatched = {
        name: (str(train[name].dtype), str(evaluation[name].dtype))
        for name in train.columns
        if train[name].dtype != evaluation[name].dtype
    }
    if mismatched:
        msg = f"train and evaluation dtypes differ: {mismatched}"
        raise FeatureSchemaError(msg)


def features_digest(frame: pd.DataFrame) -> str:
    """Return the canonical digest of a feature table.

    Args:
        frame: A table as returned by :func:`build_features`.

    Returns:
        The hex SHA-256 over the canonically rendered rows.
    """
    return digest_module.canonical_digest(
        frame, sort_by=("sample_id",), columns=tuple(frame.columns)
    )


def build_manifest(
    frame: pd.DataFrame,
    config: config_module.Config,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write down what the feature table contains and how it was made.

    The declared columns are diffed against the target denylist here, so
    a manifest that records a target as a feature cannot be written at
    all.

    Args:
        frame: The validated feature table.
        config: Validated configuration.
        provenance: Seed, timezone, and input digests.

    Returns:
        The manifest payload, ready to serialise.

    Raises:
        ValueError: If any declared feature column names a target or is
            derived from one.
    """
    categorical = list(categorical_feature_names(config))
    numeric = list(numeric_feature_names(config))
    check_denylist([*categorical, *numeric])
    return {
        "provenance": provenance,
        "conventions": {
            "categorical_missing_token": (
                config.features.categorical_missing_token
            ),
            "history_facility_slots": config.features.history_facility_slots,
            "prior_windows_days": list(config.features.prior_windows_days),
            "trend_windows_days": list(config.features.trend_windows_days),
            "rolling_preference_bookings": list(
                config.features.rolling_preference_bookings
            ),
            "ewma_halflife_days": config.features.ewma_halflife_days,
            "community_windows_days": list(
                config.features.community_windows_days
            ),
            "time_bands": {
                band.name: list(band.hours)
                for band in config.features.time_bands
            },
            "inter_booking_interval": "booking_timestamp to "
            "booking_timestamp, the same quantity the notification "
            "output predicts",
            "history_bound": "booking_timestamp <= origin",
            "community_bound": "booking_timestamp < origin; every "
            "community window is left-closed and right-open, so the "
            "origin instant is outside all of them",
        },
        "identifier_columns": list(IDENTIFIER_COLUMNS),
        "excluded_from_model": MODEL_EXCLUSIONS,
        "target_denylist": {
            "columns": list(TARGET_COLUMNS),
            "name_fragments": list(DENYLIST_FRAGMENTS),
            "violations": list(denylist_violations([*categorical, *numeric])),
        },
        "categorical_features": categorical,
        "numeric_features": numeric,
        "counts": {
            "rows": len(frame),
            "features": len(categorical) + len(numeric),
            "categorical": len(categorical),
            "numeric": len(numeric),
        },
    }


def write_manifest(payload: dict[str, Any], path: pathlib.Path) -> None:
    """Write the feature manifest.

    Args:
        payload: The manifest payload.
        path: Destination JSON; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
