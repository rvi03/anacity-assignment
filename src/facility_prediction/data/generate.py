"""Synthetic booking generator.

Builds the single dataset every later stage reads. One community, the
configured residents and facilities, the configured horizon, and one row
per booking:

    booking_id · resident_id · facility_id
    booking_timestamp · usage_timestamp

Residents are given hidden behaviour — archetype, preferred facilities,
preferred weekdays and hour band, lead-time habit, consistency, activity
rate, join date. **None of it is written out.** The dataset
carries the five fields above and nothing else, so a model has to
rediscover the behaviour from bookings alone.

Determinism: every draw comes from one seeded generator, the
frame is sorted by ``(booking_timestamp, booking_id)``, and identifiers
are assigned after that sort. Re-running the same config reproduces the
same rows, which :func:`bookings_digest` reports independently of where
they are stored, and a byte-identical export, which
:func:`write_bookings` reports as a SHA-256.

The generated table is written to the database, which is the store of
record. The CSV is an export of it: the submitted dataset file, proved
equal to the table by digest rather than assumed to be.

Leakage contract: this module *creates* time rather than reading it. It
never observes a prediction origin, so it has no as-of bound to respect.
It does guarantee the ordering later stages depend on — within each
resident, ``booking_timestamp`` is strictly increasing, and every row
satisfies ``booking_timestamp < usage_timestamp``.

Behaviour changes over the horizon, and it changes on dated events so
a reviewer can slice results around them:

    D1  the late-opening facility opens and takes a configured share of
        one other facility's demand, for one archetype only
    D2  a configured share of residents change archetype outright, the
        way a life change would
    S1  one facility carries a continuous seasonal swing

Everything the checks need about the hidden behaviour is accumulated
during generation as **aggregate counters**. No resident-level
archetype, preference, or noise flag leaves this module.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import hashlib
import logging
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

from facility_prediction import config as config_module
from facility_prediction.data import digest

_LOGGER = logging.getLogger(__name__)

BOOKING_COLUMNS = (
    "booking_id",
    "resident_id",
    "facility_id",
    "booking_timestamp",
    "usage_timestamp",
)

_DAYS_IN_WEEK = 7
_HOURS_IN_DAY = 24
_MINUTES_IN_HOUR = 60
_WEEKEND_WEEKDAYS = (5, 6)
_WEEKDAY_COUNT = 5
_WEEKEND_COUNT = 2
_MONTHS_IN_YEAR = 12
_MORNING_END_HOUR = 12


@dataclasses.dataclass
class GenerationAudit:
    """Aggregate counters held only while generating.

    Every field is a total or a per-group total. No resident-level
    archetype, preference, or noise flag is kept.

    Attributes:
        noise_path: Facility draws taken from community popularity.
        preference_path: Facility draws taken from personal preference.
        by_archetype_facility: Bookings per (archetype, era, facility),
            where era is "pre" or "post" the absorption event.
        resampled_facility: Bookings per (era, facility) for residents
            resampled at the life-change event, era relative to it.
        primary_bookings: Bookings at the resident's declared primary
            facility, per resident index.
        total_bookings: Bookings per resident index.
        morning_by_archetype: Morning bookings per archetype.
        weekend_by_archetype: Weekend bookings per archetype.
        count_by_archetype: Bookings per archetype.
        morning_total: Morning bookings across the community.
        weekend_total: Weekend bookings across the community.
        resampled_residents: How many residents changed archetype.
    """

    noise_path: int = 0
    preference_path: int = 0
    by_archetype_facility: collections.Counter[tuple[str, str, str]] = (
        dataclasses.field(default_factory=collections.Counter)
    )
    resampled_facility: collections.Counter[tuple[str, str]] = (
        dataclasses.field(default_factory=collections.Counter)
    )
    primary_bookings: collections.Counter[int] = dataclasses.field(
        default_factory=collections.Counter
    )
    total_bookings: collections.Counter[int] = dataclasses.field(
        default_factory=collections.Counter
    )
    morning_by_archetype: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    weekend_by_archetype: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    count_by_archetype: collections.Counter[str] = dataclasses.field(
        default_factory=collections.Counter
    )
    morning_total: int = 0
    weekend_total: int = 0
    resampled_residents: int = 0

    def noise_share(self) -> float:
        """Return the share of facility draws that took the noise path."""
        total = self.noise_path + self.preference_path
        return self.noise_path / total if total else 0.0


@dataclasses.dataclass
class _Drops:
    """Bookings that were drawn but could not be placed.

    A draw is discarded rather than moved somewhere it does not belong,
    and the count is reported.

    Attributes:
        past_horizon: Usage day fell beyond the configured horizon.
        before_opening: Usage day fell before the facility opened.
        no_free_hour: Every hour within the search radius was full.
        lead_before_start: The lead time reached back before the first
            instant of the horizon.
        simultaneous: A resident drew two bookings at the same instant.
    """

    past_horizon: int = 0
    before_opening: int = 0
    no_free_hour: int = 0
    lead_before_start: int = 0
    simultaneous: int = 0

    def total(self) -> int:
        """Return the number of draws discarded for any reason."""
        return (
            self.past_horizon
            + self.before_opening
            + self.no_free_hour
            + self.lead_before_start
            + self.simultaneous
        )


@dataclasses.dataclass(frozen=True)
class _Booking:
    """One generated booking, before identifiers are assigned.

    Attributes:
        resident_id: Public resident identifier.
        facility_id: Catalog name of the booked facility.
        booking_timestamp: When the booking was created, tz-aware.
        usage_timestamp: When the facility is to be used, tz-aware.
        facility_index: Catalog index, carried so the next draw can
            apply the recency nudge. It never reaches the dataset.
    """

    resident_id: str
    facility_id: str
    booking_timestamp: datetime.datetime
    usage_timestamp: datetime.datetime
    facility_index: int


@dataclasses.dataclass(frozen=True)
class _Traits:
    """Per-resident trait vectors, drawn in one seeded pass.

    Attributes:
        archetypes: Archetype per resident.
        classes: Activity rate class per resident.
        consistencies: Preference-following strength per resident.
        lead_multipliers: Personal lead-time multiplier per resident.
        is_early: Whether the resident is present from the first week.
        late_weeks: Join week for residents who are not.
        hour_offsets: Standard-normal shift of the preferred hour.
    """

    archetypes: list[config_module.Archetype]
    classes: list[config_module.ActivityClass]
    consistencies: np.ndarray
    lead_multipliers: np.ndarray
    is_early: np.ndarray
    late_weeks: np.ndarray
    hour_offsets: np.ndarray


@dataclasses.dataclass
class _WeekState:
    """Generation state for one resident in one week.

    Attributes:
        available: Boolean mask of facilities open this week.
        week_start: First instant of the week being played.
        horizon_end: First instant beyond the configured horizon.
        occupancy: Live per-slot counts, shared across residents.
        last_index: Catalog index of this resident's previous booking,
            updated as bookings are placed.
    """

    available: np.ndarray
    week_start: datetime.datetime
    horizon_end: datetime.datetime
    occupancy: dict[tuple[str, datetime.date, int], int]
    last_index: int | None


@dataclasses.dataclass(frozen=True)
class _Resident:
    """One resident's hidden behaviour — generator-only.

    Nothing on this object may be written to the dataset.

    Attributes:
        resident_id: Public identifier; the only field that leaves here.
        archetype: The behavioural archetype drawn for this resident.
        preference: Facility preference weights, catalog-ordered.
        weekday_weights: Preference over the seven weekdays.
        hour_center: Personal preferred hour, in ``[0, 24)``.
        hour_kappa: Circular-normal concentration for the hour band.
        consistency: Weight on personal preference against popularity.
        lead_multiplier: Personal multiplier on facility lead times.
        weekly_mean: Mean bookings per active week.
        dispersion: Negative-binomial dispersion for that count.
        join_week: First week index in which this resident may book.
    """

    resident_id: str
    archetype: config_module.Archetype
    preference: np.ndarray
    weekday_weights: np.ndarray
    hour_center: float
    hour_kappa: float
    consistency: float
    lead_multiplier: float
    weekly_mean: float
    dispersion: float
    join_week: int


# Derived from the dataclass rather than retyped, so a new piece of
# hidden behaviour is covered by the leakage check the day it is added.
HIDDEN_STATE_FIELDS = tuple(
    field.name
    for field in dataclasses.fields(_Resident)
    if field.name != "resident_id"
)


def _choose_shares(
    rng: np.random.Generator, shares: list[float], size: int
) -> np.ndarray:
    """Draw ``size`` category indices from a share vector.

    Args:
        rng: Seeded generator.
        shares: Mixture weights summing to one.
        size: Number of draws.

    Returns:
        Integer indices into ``shares``.
    """
    weights = np.asarray(shares, dtype=float)
    return rng.choice(len(weights), size=size, p=weights / weights.sum())


def _weekday_weights(
    rng: np.random.Generator,
    archetype: config_module.Archetype,
    concentration: float,
) -> np.ndarray:
    """Draw one resident's preferred weekdays.

    The archetype's weekend share sets the weekday/weekend balance; a
    Dirichlet draw on top gives the resident their own favourite days.

    Args:
        rng: Seeded generator.
        archetype: The resident's archetype.
        concentration: Dirichlet alpha; below one is peaked.

    Returns:
        A seven-element weight vector summing to one, Monday first.
    """
    weekend_day = archetype.weekend_share / _WEEKEND_COUNT
    weekday_day = (1.0 - archetype.weekend_share) / _WEEKDAY_COUNT
    base = np.array(
        [
            weekend_day if day in _WEEKEND_WEEKDAYS else weekday_day
            for day in range(_DAYS_IN_WEEK)
        ]
    )
    personal = rng.dirichlet(np.full(_DAYS_IN_WEEK, concentration))
    weights = base * (personal + concentration)
    return weights / weights.sum()


def _claimed_share(config: config_module.Config) -> np.ndarray:
    """Return how much of the population favours each facility.

    Args:
        config: Validated configuration.

    Returns:
        Catalog-ordered totals of the shares of every archetype naming
        that facility, floored at one archetype's worth so a facility
        nobody names is not divided by zero.
    """
    claimed = np.zeros(len(config.facilities))
    for archetype in config.generator.archetypes:
        for position, facility in enumerate(config.facilities):
            if facility.name in archetype.facilities:
                claimed[position] += archetype.share
    return np.where(claimed > 0.0, claimed, 1.0)


def _facility_preference(
    rng: np.random.Generator,
    config: config_module.Config,
    archetype: config_module.Archetype,
) -> np.ndarray:
    """Draw one resident's facility preferences.

    Both the Dirichlet base and the archetype's boost are shared out in
    proportion to configured popularity, so a resident keeps a dominant
    favourite and a niche facility is not inflated community-wide.

    The boost is divided by the total share of archetypes naming the
    same facility. Without that, a facility two archetypes favour
    collects both boosts and the realised mix drifts away from the
    configured popularity.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        archetype: The resident's archetype, whose facilities take the
            larger share of the Dirichlet alpha.

    Returns:
        Catalog-ordered preference weights summing to one.
    """
    preference = config.generator.preference
    popularity = np.array(
        [facility.popularity for facility in config.facilities]
    )
    alpha = preference.dirichlet_concentration * len(popularity) * popularity
    favoured = np.array(
        [
            facility.name in archetype.facilities
            for facility in config.facilities
        ]
    )
    if favoured.any():
        boost = np.where(favoured, popularity / _claimed_share(config), 0.0)
        boost = boost / boost.sum()
    else:
        boost = popularity
    return rng.dirichlet(alpha + preference.primary_boost * boost)


def _draw_resident_traits(
    config: config_module.Config, rng: np.random.Generator, weeks: int
) -> _Traits:
    """Draw the per-resident trait vectors in one seeded pass.

    Drawing each trait as a whole vector, rather than inside the
    assembly loop, keeps the stream of random draws independent of how
    the residents are later assembled.

    Args:
        config: Validated configuration.
        rng: Seeded generator.
        weeks: Number of whole weeks in the horizon.

    Returns:
        One value per resident for each trait, in resident order.
    """
    generator = config.generator
    count = config.community.residents
    return _Traits(
        archetypes=[
            generator.archetypes[index]
            for index in _choose_shares(
                rng, [entry.share for entry in generator.archetypes], count
            )
        ],
        classes=[
            generator.activity.classes[index]
            for index in _choose_shares(
                rng,
                [entry.share for entry in generator.activity.classes],
                count,
            )
        ],
        consistencies=rng.uniform(
            generator.consistency.low, generator.consistency.high, size=count
        ),
        lead_multipliers=rng.lognormal(
            0.0, generator.resident_lead_log_sigma, size=count
        ),
        is_early=rng.random(count) < generator.join.early_share,
        late_weeks=rng.integers(0, max(weeks, 1), size=count),
        hour_offsets=rng.normal(0.0, 1.0, size=count),
    )


def _build_residents(
    config: config_module.Config, rng: np.random.Generator, weeks: int
) -> list[_Resident]:
    """Draw every resident's hidden behaviour.

    Args:
        config: Validated configuration.
        rng: Seeded generator.
        weeks: Number of whole weeks in the horizon.

    Returns:
        One :class:`_Resident` per configured resident, in id order.
    """
    concentration = config.generator.preference.dirichlet_concentration
    traits = _draw_resident_traits(config, rng, weeks)
    residents = []
    for index in range(config.community.residents):
        archetype = traits.archetypes[index]
        spread_radians = archetype.hour_spread * 2.0 * np.pi / _HOURS_IN_DAY
        offset = traits.hour_offsets[index] * archetype.hour_spread / 2.0
        residents.append(
            _Resident(
                resident_id=f"R{index + 1:04d}",
                archetype=archetype,
                preference=_facility_preference(rng, config, archetype),
                weekday_weights=_weekday_weights(rng, archetype, concentration),
                hour_center=(archetype.hour_center + offset) % _HOURS_IN_DAY,
                hour_kappa=1.0 / spread_radians**2,
                consistency=float(traits.consistencies[index]),
                lead_multiplier=float(traits.lead_multipliers[index]),
                weekly_mean=traits.classes[index].weekly_mean,
                dispersion=traits.classes[index].dispersion,
                join_week=(
                    0
                    if traits.is_early[index]
                    else int(traits.late_weeks[index])
                ),
            )
        )
    return residents


def _availability_weeks(config: config_module.Config, weeks: int) -> np.ndarray:
    """Return the first week index at which each facility is bookable.

    Args:
        config: Validated configuration.
        weeks: Number of whole weeks in the horizon.

    Returns:
        Catalog-ordered week indices; a facility open from the start
        gives zero.
    """
    start = config.start_instant
    first_weeks = []
    for facility in config.facilities:
        opens = start + pd.DateOffset(months=facility.available_from_month)
        elapsed_days = (opens - start).days
        first_weeks.append(min(elapsed_days // _DAYS_IN_WEEK, weeks))
    return np.asarray(first_weeks, dtype=int)


def _weekly_count(
    rng: np.random.Generator, resident: _Resident, zero_inflation: float
) -> int:
    """Draw one resident's booking count for one week.

    Args:
        rng: Seeded generator.
        resident: The resident booking this week.
        zero_inflation: Probability the week yields nothing at all.

    Returns:
        A non-negative booking count.
    """
    if rng.random() < zero_inflation:
        return 0
    dispersion = resident.dispersion
    probability = dispersion / (dispersion + resident.weekly_mean)
    return int(rng.negative_binomial(dispersion, probability))


def _season_multipliers(config: config_module.Config, month: int) -> np.ndarray:
    """Return this month's demand multiplier for each facility.

    One facility carries a continuous swing across the year; every
    other facility sits flat at one.

    Args:
        config: Validated configuration.
        month: Calendar month, one for January.

    Returns:
        Catalog-ordered multipliers, centred on one.
    """
    season = config.generator.drift.season
    phase = 2.0 * np.pi * (month - 1 - season.peak_month) / _MONTHS_IN_YEAR
    swing = 1.0 + season.amplitude * float(np.cos(phase))
    return np.array(
        [
            swing if facility.name == season.facility else 1.0
            for facility in config.facilities
        ]
    )


def _absorbed_preference(
    config: config_module.Config,
    resident: _Resident,
    available: np.ndarray,
) -> np.ndarray:
    """Return the resident's preference after the dated absorption.

    Once the late-opening facility is bookable, members of one
    archetype move a configured share of their preference for one other
    facility across to it. Everyone else is untouched.

    Args:
        config: Validated configuration.
        resident: The resident booking.
        available: Boolean mask over the catalog for this week.

    Returns:
        Catalog-ordered preference weights, zero where unavailable.
    """
    drift = config.generator.drift
    preference = np.where(available, resident.preference, 0.0)
    if resident.archetype.name != drift.absorbing_archetype:
        return preference

    names = config.facility_names
    into = names.index(drift.absorbing_facility)
    outof = names.index(drift.absorbed_facility)
    if not available[into]:
        return preference

    moved = drift.absorption * preference[outof]
    preference = preference.copy()
    preference[outof] -= moved
    preference[into] += moved
    return preference


def _facility_weights(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    available: np.ndarray,
    last_index: int | None,
    month: int,
    audit: GenerationAudit,
) -> np.ndarray:
    """Build the selection weights over the catalog for one booking.

    Args:
        rng: Seeded generator, consulted once for the noise path.
        config: Validated configuration.
        resident: The resident booking.
        available: Boolean mask over the catalog for this week.
        last_index: Catalog index of the previous booking, or None.
        month: Calendar month of the week being played, for the season.
        audit: Aggregate counters; only totals are incremented.

    Returns:
        Unnormalised catalog-ordered weights, zero where unavailable.
    """
    generator = config.generator
    popularity = np.where(
        available,
        [facility.popularity for facility in config.facilities],
        0.0,
    ) * _season_multipliers(config, month)
    if rng.random() < generator.noise_fraction:
        audit.noise_path += 1
        return popularity
    audit.preference_path += 1
    preference = _absorbed_preference(config, resident, available)
    weights = (
        resident.consistency * preference
        + (1.0 - resident.consistency) * popularity
    )
    if last_index is not None and available[last_index]:
        weights = weights.copy()
        weights[last_index] *= generator.preference.recency_boost
    return weights


def _choose_facility(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    available: np.ndarray,
    last_index: int | None,
    month: int,
    audit: GenerationAudit,
) -> int:
    """Pick a facility for one booking.

    Blends personal preference with community popularity by the
    resident's consistency, nudges the last facility used, and takes the
    noise path for the configured share of bookings.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        resident: The resident booking.
        available: Boolean mask over the catalog for this week.
        last_index: Catalog index of this resident's previous booking,
            or None for their first.
        month: Calendar month of the week being played.
        audit: Aggregate counters.

    Returns:
        Catalog index of the chosen facility.
    """
    weights = _facility_weights(
        rng, config, resident, available, last_index, month, audit
    )
    total = weights.sum()
    if total <= 0.0:
        weights = available.astype(float)
        total = weights.sum()
    return int(rng.choice(len(weights), p=weights / total))


def _choose_weekday(
    rng: np.random.Generator,
    resident: _Resident,
    facility: config_module.Facility,
) -> int:
    """Pick a weekday for one booking.

    Args:
        rng: Seeded generator.
        resident: The resident booking.
        facility: The chosen facility, whose weekend profile reweights
            the resident's own weekday preference.

    Returns:
        Weekday index, Monday zero.
    """
    weights = resident.weekday_weights.copy()
    for day in _WEEKEND_WEEKDAYS:
        weights[day] *= facility.weekend_multiplier
    return int(rng.choice(_DAYS_IN_WEEK, p=weights / weights.sum()))


def _choose_hour(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    facility: config_module.Facility,
) -> int:
    """Pick a bookable hour for one booking.

    Draws from the resident's circular-normal hour band so 23 wraps to
    00, applies the configured jitter, then snaps to a whole hour inside
    the facility's half-open operating window.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        resident: The resident booking.
        facility: The chosen facility.

    Returns:
        An hour satisfying ``facility.is_open_at``.
    """
    jitter_hours = config.generator.jitter_minutes / _MINUTES_IN_HOUR
    center = resident.hour_center * 2.0 * np.pi / _HOURS_IN_DAY - np.pi
    for _ in range(config.generator.hour_draw_attempts):
        radians = rng.vonmises(center, resident.hour_kappa)
        hour = (radians + np.pi) * _HOURS_IN_DAY / (2.0 * np.pi)
        hour += rng.uniform(-jitter_hours, jitter_hours)
        snapped = int(np.floor(hour)) % _HOURS_IN_DAY
        if facility.is_open_at(snapped):
            return snapped
    span = facility.close_hour - facility.open_hour
    return facility.open_hour + int(rng.integers(0, span))


def _place_in_slot(
    config: config_module.Config,
    facility: config_module.Facility,
    occupancy: dict[tuple[str, datetime.date, int], int],
    usage_date: datetime.date,
    hour: int,
) -> tuple[int, bool] | None:
    """Seat a booking in an hour slot, displacing it if full.

    Args:
        config: Validated configuration.
        facility: The chosen facility.
        occupancy: Live per-slot counts, mutated in place.
        usage_date: Calendar date of use.
        hour: The preferred hour.

    Returns:
        The seated hour and whether that slot was already near capacity,
        or None when no hour within the search radius has room.
    """
    radius = config.generator.displacement_search_hours
    offsets = [0]
    for step in range(1, radius + 1):
        offsets.extend((-step, step))
    for offset in offsets:
        candidate = hour + offset
        if not facility.is_open_at(candidate):
            continue
        key = (facility.name, usage_date, candidate)
        taken = occupancy.get(key, 0)
        if taken < facility.slot_capacity:
            occupancy[key] = taken + 1
            crowded = (
                taken
                >= facility.slot_capacity
                - config.generator.capacity_pressure_slack
            )
            return candidate, crowded
    return None


def _draw_lead_minutes(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    facility: config_module.Facility,
    crowded: bool,
) -> float:
    """Draw booking lead time in minutes.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        resident: The resident booking.
        facility: The chosen facility, which sets the median lead.
        crowded: Whether the seated slot was near capacity, which pushes
            the booking earlier.

    Returns:
        Lead time in minutes, at least ``min_lead_minutes``.
    """
    generator = config.generator
    median_hours = (
        facility.lead_hours_median
        * resident.archetype.lead_scale
        * resident.lead_multiplier
    )
    if crowded:
        median_hours *= generator.capacity_lead_multiplier
    hours = rng.lognormal(np.log(median_hours), facility.lead_log_sigma)
    return max(hours * _MINUTES_IN_HOUR, float(generator.min_lead_minutes))


def _strictly_increasing(bookings: list[_Booking]) -> list[_Booking]:
    """Keep one resident's bookings strictly increasing in booking time.

    Later stages need a well-defined origin ordering per resident.
    Simultaneous bookings are dropped rather than nudged.

    Args:
        bookings: One resident's bookings, in generation order.

    Returns:
        The retained bookings, sorted by ``booking_timestamp``.
    """
    ordered = sorted(bookings, key=lambda booking: booking.booking_timestamp)
    kept: list[_Booking] = []
    for booking in ordered:
        if kept and booking.booking_timestamp <= kept[-1].booking_timestamp:
            continue
        kept.append(booking)
    return kept


def _resident_week(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    state: _WeekState,
    drops: _Drops,
    audit: GenerationAudit,
) -> list[_Booking]:
    """Generate one resident's bookings for one week.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        resident: The resident booking this week.
        state: Live generation state; its ``last_index`` is advanced as
            bookings are placed.
        drops: Counter for draws that could not be placed.
        audit: Aggregate counters.

    Returns:
        The bookings placed this week, in draw order.
    """
    count = _weekly_count(
        rng, resident, config.generator.activity.zero_inflation
    )
    bookings = []
    for _ in range(count):
        booking = _one_booking(rng, config, resident, state, drops, audit)
        if booking is None:
            continue
        state.last_index = booking.facility_index
        bookings.append(booking)
    return bookings


def _absorption_month(config: config_module.Config) -> int:
    """Return the month the absorbing facility opens.

    Args:
        config: Validated configuration.

    Returns:
        The zero-based elapsed month of the dated opening.
    """
    return config.facility(
        config.generator.drift.absorbing_facility
    ).available_from_month


def _event_week(config: config_module.Config, month: int) -> int:
    """Return the week index a dated event falls in.

    Args:
        config: Validated configuration.
        month: Zero-based elapsed month of the event.

    Returns:
        The first week index at or after the event.
    """
    start = config.start_instant
    when = start + pd.DateOffset(months=month)
    return int((when - start).days // _DAYS_IN_WEEK)


def _resampled_residents(
    config: config_module.Config,
    rng: np.random.Generator,
    residents: list[_Resident],
) -> dict[int, _Resident]:
    """Draw the life-change cohort and their replacement behaviour.

    A resampled resident keeps their identifier, join week, and
    activity rate — the person is the same, their habits are not. The
    archetype, facility preference, weekday preference, and hour band
    are all redrawn.

    Args:
        config: Validated configuration.
        rng: Seeded generator.
        residents: The residents as first drawn.

    Returns:
        Resident index to their post-event behaviour, for the cohort
        only. Everyone else is absent and never changes.
    """
    drift = config.generator.drift
    generator = config.generator
    concentration = generator.preference.dirichlet_concentration
    selected = rng.random(len(residents)) < drift.resample_share
    replacements: dict[int, _Resident] = {}
    for index, resident in enumerate(residents):
        if not selected[index]:
            continue
        archetype = generator.archetypes[
            _choose_shares(
                rng, [entry.share for entry in generator.archetypes], 1
            )[0]
        ]
        spread = archetype.hour_spread * 2.0 * np.pi / _HOURS_IN_DAY
        replacements[index] = dataclasses.replace(
            resident,
            archetype=archetype,
            preference=_facility_preference(rng, config, archetype),
            weekday_weights=_weekday_weights(rng, archetype, concentration),
            hour_center=archetype.hour_center % _HOURS_IN_DAY,
            hour_kappa=1.0 / spread**2,
        )
    return replacements


def _primary_index(
    config: config_module.Config, resident: _Resident
) -> int | None:
    """Return the catalog index of a resident's declared favourite.

    Args:
        config: Validated configuration.
        resident: The resident.

    Returns:
        The index of the archetype facility they most prefer, or None
        when their archetype names no facility at all.
    """
    names = config.facility_names
    owned = [
        names.index(name)
        for name in resident.archetype.facilities
        if name in names
    ]
    if not owned:
        return None
    return max(owned, key=lambda index: resident.preference[index])


def _record(
    config: config_module.Config,
    audit: GenerationAudit,
    placed: list[_Booking],
    index: int,
    resident: _Resident,
    era: str,
    resampled_era: str,
    is_resampled: bool,
) -> None:
    """Fold one resident-week's bookings into the aggregate counters.

    Args:
        config: Validated configuration.
        audit: Counters to update in place.
        placed: Bookings generated for this resident this week.
        index: Resident index, used only as a counter key.
        resident: The resident, read for their hidden archetype and
            preference. Neither leaves this function.
        era: "pre" or "post" the absorption event.
        resampled_era: "pre" or "post" the life-change event.
        is_resampled: Whether this resident is in the life-change
            cohort.
    """
    if not placed:
        return
    name = resident.archetype.name
    primary = _primary_index(config, resident)
    for booking in placed:
        audit.by_archetype_facility[name, era, booking.facility_id] += 1
        audit.count_by_archetype[name] += 1
        audit.total_bookings[index] += 1
        if primary is not None and booking.facility_index == primary:
            audit.primary_bookings[index] += 1
        if booking.usage_timestamp.hour < _MORNING_END_HOUR:
            audit.morning_by_archetype[name] += 1
            audit.morning_total += 1
        if booking.usage_timestamp.weekday() in _WEEKEND_WEEKDAYS:
            audit.weekend_by_archetype[name] += 1
            audit.weekend_total += 1
        if is_resampled:
            audit.resampled_facility[resampled_era, booking.facility_id] += 1


def _generate_rows(
    config: config_module.Config,
    rng: np.random.Generator,
    residents: list[_Resident],
    weeks: int,
    audit: GenerationAudit,
) -> tuple[list[_Booking], _Drops]:
    """Play the horizon forward and collect every booking.

    Weeks are the outer loop so facility capacity is contested in time
    order rather than by resident order.

    Args:
        config: Validated configuration.
        rng: Seeded generator.
        residents: The residents and their hidden behaviour.
        weeks: Number of whole weeks in the horizon.
        audit: Aggregate counters, filled in as bookings are placed.

    Returns:
        The unsorted bookings and the tally of discarded draws.
    """
    start = config.start_instant
    horizon_end = start + pd.DateOffset(months=config.community.months)
    opens_in_week = _availability_weeks(config, weeks)
    occupancy: dict[tuple[str, datetime.date, int], int] = {}
    drops = _Drops()
    per_resident: list[list[_Booking]] = [[] for _ in residents]
    states = [
        _WeekState(
            available=opens_in_week <= 0,
            week_start=start,
            horizon_end=horizon_end,
            occupancy=occupancy,
            last_index=None,
        )
        for _ in residents
    ]

    absorption_week = _event_week(config, _absorption_month(config))
    resample_week = _event_week(config, config.generator.drift.resample_month)
    changed = _resampled_residents(config, rng, residents)
    audit.resampled_residents = len(changed)

    for week in range(weeks):
        available = opens_in_week <= week
        week_start = start + datetime.timedelta(days=week * _DAYS_IN_WEEK)
        for index, original in enumerate(residents):
            if week < original.join_week:
                continue
            resident = (
                changed[index]
                if week >= resample_week and index in changed
                else original
            )
            state = states[index]
            state.available = available
            state.week_start = week_start
            placed = _resident_week(rng, config, resident, state, drops, audit)
            per_resident[index].extend(placed)
            _record(
                config,
                audit,
                placed,
                index,
                resident,
                era="post" if week >= absorption_week else "pre",
                resampled_era="post" if week >= resample_week else "pre",
                is_resampled=index in changed,
            )

    bookings: list[_Booking] = []
    for owned in per_resident:
        kept = _strictly_increasing(owned)
        drops.simultaneous += len(owned) - len(kept)
        bookings.extend(kept)
    return bookings, drops


def _one_booking(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    state: _WeekState,
    drops: _Drops,
    audit: GenerationAudit,
) -> _Booking | None:
    """Generate one booking, or None when it cannot be placed.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        resident: The resident booking.
        state: Live generation state for this resident and week.
        drops: Counter incremented with the reason for any discard.
        audit: Aggregate counters.

    Returns:
        The booking, or None when the draw was discarded.
    """
    facility_index = _choose_facility(
        rng,
        config,
        resident,
        state.available,
        state.last_index,
        state.week_start.month,
        audit,
    )
    facility = config.facilities[facility_index]
    weekday = _choose_weekday(rng, resident, facility)
    usage_day = state.week_start + datetime.timedelta(days=weekday)
    if usage_day >= state.horizon_end:
        drops.past_horizon += 1
        return None
    opens = config.start_instant + pd.DateOffset(
        months=facility.available_from_month
    )
    if usage_day < opens:
        drops.before_opening += 1
        return None
    seated = _place_in_slot(
        config,
        facility,
        state.occupancy,
        usage_day.date(),
        _choose_hour(rng, config, resident, facility),
    )
    if seated is None:
        drops.no_free_hour += 1
        return None
    usage_hour, crowded = seated
    usage_timestamp = datetime.datetime.combine(
        usage_day.date(), datetime.time(hour=usage_hour), tzinfo=config.tzinfo
    )
    return _timestamped_booking(
        rng,
        config,
        resident,
        facility,
        facility_index,
        usage_timestamp,
        crowded,
        drops,
    )


def _timestamped_booking(
    rng: np.random.Generator,
    config: config_module.Config,
    resident: _Resident,
    facility: config_module.Facility,
    facility_index: int,
    usage_timestamp: datetime.datetime,
    crowded: bool,
    drops: _Drops,
) -> _Booking | None:
    """Attach a booking timestamp to a seated usage slot.

    Args:
        rng: Seeded generator.
        config: Validated configuration.
        resident: The resident booking.
        facility: The seated facility.
        facility_index: Its catalog index, carried back to the caller.
        usage_timestamp: The seated usage instant, tz-aware.
        crowded: Whether the seated slot was near capacity.
        drops: Counter incremented if the lead reaches before the start.

    Returns:
        The completed booking, or None when the drawn lead time would
        place it before the horizon begins.
    """
    lead_minutes = _draw_lead_minutes(rng, config, resident, facility, crowded)
    booking_timestamp = usage_timestamp - datetime.timedelta(
        minutes=round(lead_minutes)
    )
    floor = config.start_instant + datetime.timedelta(
        minutes=config.generator.min_lead_minutes
    )
    if booking_timestamp < floor:
        drops.lead_before_start += 1
        return None
    return _Booking(
        resident_id=resident.resident_id,
        facility_id=facility.name,
        booking_timestamp=booking_timestamp,
        usage_timestamp=usage_timestamp,
        facility_index=facility_index,
    )


class GenerationError(Exception):
    """Raised when a generated dataset fails its own acceptance checks."""


BOOKING_SCHEMA_NAME = "booking_table"

_JS_LOG_BASE = 2.0


@dataclasses.dataclass(frozen=True)
class Check:
    """One acceptance check and how it came out.

    Attributes:
        name: Short identifier for the property being checked.
        passed: Whether the dataset met it.
        detail: The realised value beside the threshold, so a failure
            says how far off it was rather than only that it failed.
    """

    name: str
    passed: bool
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a JSON-serialisable mapping."""
        return dataclasses.asdict(self)


def booking_schema(config: config_module.Config) -> pa.DataFrameSchema:
    """Return the structural contract every booking row must satisfy.

    Args:
        config: Validated configuration; the facility catalog is the
            categorical enum, so an unknown facility is a failure
            rather than a new level.

    Returns:
        A strict, ordered schema over :data:`BOOKING_COLUMNS`.
    """
    text = pa.Column(str, nullable=False)
    return pa.DataFrameSchema(
        {
            "booking_id": pa.Column(str, nullable=False, unique=True),
            "resident_id": text,
            "facility_id": pa.Column(
                str,
                nullable=False,
                checks=[pa.Check.isin(list(config.facility_names))],
            ),
            "booking_timestamp": pa.Column(
                pd.DatetimeTZDtype(tz=config.tzinfo), nullable=False
            ),
            "usage_timestamp": pa.Column(
                pd.DatetimeTZDtype(tz=config.tzinfo), nullable=False
            ),
        },
        checks=[
            pa.Check(
                lambda frame: (
                    frame["booking_timestamp"] < frame["usage_timestamp"]
                ),
                name="booking_precedes_usage",
            )
        ],
        strict=True,
        ordered=True,
        name=BOOKING_SCHEMA_NAME,
    )


def _shares(counts: pd.Series) -> pd.Series:
    """Return counts as shares, or an empty series when there are none.

    Args:
        counts: Non-negative counts.

    Returns:
        The counts divided by their total.
    """
    total = float(counts.sum())
    return counts / total if total else counts.astype(float)


def _jensen_shannon(left: pd.Series, right: pd.Series) -> float:
    """Return the Jensen-Shannon distance between two distributions.

    Args:
        left: First distribution's counts.
        right: Second distribution's counts.

    Returns:
        The distance in ``[0, 1]``, zero when the two agree. Returns
        zero when either side is empty, because "no evidence of a
        change" is not the same claim as "a change happened".
    """
    keys = sorted(set(left.index) | set(right.index))
    first = _shares(left.reindex(keys, fill_value=0.0)).to_numpy()
    second = _shares(right.reindex(keys, fill_value=0.0)).to_numpy()
    if first.sum() == 0.0 or second.sum() == 0.0:
        return 0.0

    mean = (first + second) / 2.0
    divergence = 0.0
    for side in (first, second):
        mask = side > 0.0
        divergence += 0.5 * float(
            np.sum(side[mask] * np.log2(side[mask] / mean[mask]))
        )
    return float(np.sqrt(max(divergence, 0.0) / np.log2(_JS_LOG_BASE)))


def _lead_hours(frame: pd.DataFrame) -> pd.Series:
    """Return each booking's lead time in hours.

    Args:
        frame: The booking table.

    Returns:
        Hours between creation and usage.
    """
    gap = frame["usage_timestamp"] - frame["booking_timestamp"]
    return gap.dt.total_seconds() / (_MINUTES_IN_HOUR * 60.0)


def _lead_checks(
    frame: pd.DataFrame, config: config_module.Config
) -> list[Check]:
    """Check that lead times are skewed and facility-dependent.

    Args:
        frame: The booking table.
        config: Validated configuration.

    Returns:
        The lead-time checks.
    """
    rules = config.generator.acceptance
    hours = _lead_hours(frame)
    median = float(hours.median())
    low, high = rules.lead_p50_hours

    by_facility = hours.groupby(frame["facility_id"]).median()
    long_lead = float(by_facility.get(rules.long_lead_facility, float("nan")))
    short_lead = float(by_facility.get(rules.short_lead_facility, float("nan")))
    ratio = long_lead / short_lead if short_lead else float("nan")
    return [
        Check(
            "lead_p50_in_band",
            low <= median <= high,
            {"median_hours": round(median, 2), "band": [low, high]},
        ),
        Check(
            "lead_right_skewed",
            float(hours.mean()) > median,
            {
                "mean_hours": round(float(hours.mean()), 2),
                "median_hours": round(median, 2),
            },
        ),
        Check(
            "long_lead_facility_books_further_ahead",
            bool(ratio >= rules.long_lead_ratio),
            {
                "ratio": round(ratio, 2),
                "minimum": rules.long_lead_ratio,
                "long_lead_median_hours": round(long_lead, 2),
                "short_lead_median_hours": round(short_lead, 2),
            },
        ),
    ]


def _availability_checks(
    frame: pd.DataFrame, config: config_module.Config
) -> list[Check]:
    """Check that no facility is booked before it opens.

    Args:
        frame: The booking table.
        config: Validated configuration.

    Returns:
        The availability and dated-opening checks.
    """
    early = 0
    for facility in config.facilities:
        opens = config.start_instant + pd.DateOffset(
            months=facility.available_from_month
        )
        rows = frame.loc[frame["facility_id"] == facility.name]
        early += int((rows["usage_timestamp"] < opens).sum())

    drift = config.generator.drift
    opening = config.start_instant + pd.DateOffset(
        months=_absorption_month(config)
    )
    late_rows = frame.loc[frame["facility_id"] == drift.absorbing_facility]
    return [
        Check(
            "no_booking_before_a_facility_opens",
            early == 0,
            {"rows_before_opening": early},
        ),
        Check(
            "late_facility_absent_before_it_opens_and_present_after",
            int((late_rows["usage_timestamp"] < opening).sum()) == 0
            and int((late_rows["usage_timestamp"] >= opening).sum()) > 0,
            {
                "facility": drift.absorbing_facility,
                "rows_before": int(
                    (late_rows["usage_timestamp"] < opening).sum()
                ),
                "rows_after": int(
                    (late_rows["usage_timestamp"] >= opening).sum()
                ),
            },
        ),
    ]


def _popularity_checks(
    frame: pd.DataFrame, config: config_module.Config
) -> list[Check]:
    """Check the realised facility mix against the configured one.

    Args:
        frame: The booking table.
        config: Validated configuration.

    Returns:
        The popularity, imbalance, and profile checks.
    """
    rules = config.generator.acceptance
    opening = config.start_instant + pd.DateOffset(
        months=_absorption_month(config)
    )
    before = frame.loc[frame["usage_timestamp"] < opening]
    realised = _shares(before["facility_id"].value_counts())
    configured = pd.Series(
        {facility.name: facility.popularity for facility in config.facilities}
    ).sort_values(ascending=False)

    top = rules.top_facilities
    overall = _shares(frame["facility_id"].value_counts())
    available = [
        facility.name
        for facility in config.facilities
        if facility.name in overall.index
    ]
    spread = overall[available]
    return [
        Check(
            "pre_drift_top_facilities_match_configuration",
            list(realised.index[:top]) == list(configured.index[:top]),
            {
                "realised": list(realised.index[:top]),
                "configured": list(configured.index[:top]),
                "profile": {
                    name: round(float(realised.get(name, 0.0)), 4)
                    for name in configured.index
                },
            },
        ),
        Check(
            "facility_use_is_imbalanced",
            bool(spread.max() >= rules.imbalance_ratio * spread.min()),
            {
                "most_used_share": round(float(spread.max()), 4),
                "least_used_share": round(float(spread.min()), 4),
                "minimum_ratio": rules.imbalance_ratio,
            },
        ),
    ]


def _preference_checks(
    frame: pd.DataFrame,
    config: config_module.Config,
    audit: GenerationAudit,
) -> list[Check]:
    """Check that individual preference and time habits are visible.

    Every quantity here comes from the aggregate counters, never from a
    resident-level archetype or preference column.

    Args:
        frame: The booking table.
        config: Validated configuration.
        audit: The aggregate counters from generation.

    Returns:
        The preference, time-pattern, and noise checks.
    """
    rules = config.generator.acceptance
    active = [
        index
        for index, total in audit.total_bookings.items()
        if total >= rules.active_resident_bookings
    ]
    shares = [
        audit.primary_bookings[index] / audit.total_bookings[index]
        for index in active
    ]
    median_share = float(np.median(shares)) if shares else 0.0
    community = _shares(frame["facility_id"].value_counts())
    typical_community = float(community.median())

    checks = [
        Check(
            "active_residents_favour_their_primary_facility",
            median_share >= rules.primary_share_min
            and median_share >= typical_community + rules.primary_share_excess,
            {
                "median_primary_share": round(median_share, 4),
                "minimum": rules.primary_share_min,
                "typical_community_share": round(typical_community, 4),
                "required_excess": rules.primary_share_excess,
                "active_residents": len(active),
            },
        ),
        Check(
            "noise_path_share_matches_configuration",
            rules.noise_share[0] <= audit.noise_share() <= rules.noise_share[1],
            {
                "realised": round(audit.noise_share(), 4),
                "configured": config.generator.noise_fraction,
                "band": list(rules.noise_share),
            },
        ),
    ]
    checks += _time_pattern_checks(config, audit)
    return checks


def _time_pattern_checks(
    config: config_module.Config, audit: GenerationAudit
) -> list[Check]:
    """Check that two archetypes book at their declared times.

    Args:
        config: Validated configuration.
        audit: The aggregate counters from generation.

    Returns:
        One check per declared time pattern.
    """
    rules = config.generator.acceptance
    total = sum(audit.count_by_archetype.values())
    results = []
    for label, archetype, group, community in (
        (
            "morning",
            rules.morning_archetype,
            audit.morning_by_archetype,
            audit.morning_total,
        ),
        (
            "weekend",
            rules.weekend_archetype,
            audit.weekend_by_archetype,
            audit.weekend_total,
        ),
    ):
        owned = audit.count_by_archetype.get(archetype, 0)
        share = group.get(archetype, 0) / owned if owned else 0.0
        baseline = community / total if total else 0.0
        results.append(
            Check(
                f"{archetype.lower().replace(' ', '_')}_{label}_share_"
                "exceeds_the_community",
                share >= baseline + rules.time_pattern_excess,
                {
                    "archetype_share": round(share, 4),
                    "community_share": round(baseline, 4),
                    "required_excess": rules.time_pattern_excess,
                },
            )
        )
    return results


def _drift_checks(
    config: config_module.Config, audit: GenerationAudit
) -> list[Check]:
    """Check that both dated events actually moved behaviour.

    Args:
        config: Validated configuration.
        audit: The aggregate counters from generation.

    Returns:
        The absorption and life-change checks.
    """
    rules = config.generator.acceptance
    drift = config.generator.drift

    def era_shares(era: str) -> pd.Series:
        counts = {
            facility: count
            for (
                archetype,
                seen,
                facility,
            ), count in audit.by_archetype_facility.items()
            if archetype == drift.absorbing_archetype and seen == era
        }
        return _shares(pd.Series(counts, dtype=float))

    before, after = era_shares("pre"), era_shares("post")
    absorbed_drop = float(
        before.get(drift.absorbed_facility, 0.0)
        - after.get(drift.absorbed_facility, 0.0)
    )
    absorbing_after = float(after.get(drift.absorbing_facility, 0.0))

    resampled_before = pd.Series(
        {
            facility: count
            for (era, facility), count in audit.resampled_facility.items()
            if era == "pre"
        },
        dtype=float,
    )
    resampled_after = pd.Series(
        {
            facility: count
            for (era, facility), count in audit.resampled_facility.items()
            if era == "post"
        },
        dtype=float,
    )
    distance = _jensen_shannon(resampled_before, resampled_after)
    return [
        Check(
            "dated_opening_moved_the_affected_archetype",
            absorbing_after > 0.0 and absorbed_drop > rules.absorption_drop,
            {
                "archetype": drift.absorbing_archetype,
                "absorbing_share_after": round(absorbing_after, 4),
                "absorbed_share_drop": round(absorbed_drop, 4),
                "required_drop": rules.absorption_drop,
            },
        ),
        Check(
            "life_change_moved_the_cohort",
            distance > rules.resample_distance,
            {
                "jensen_shannon_distance": round(distance, 4),
                "minimum": rules.resample_distance,
                "residents_resampled": audit.resampled_residents,
            },
        ),
    ]


def _sparsity_check(frame: pd.DataFrame, config: config_module.Config) -> Check:
    """Check that a real share of residents barely book at all.

    Args:
        frame: The booking table.
        config: Validated configuration.

    Returns:
        The sparsity check.
    """
    rules = config.generator.acceptance
    per_resident = frame["resident_id"].value_counts()
    roster = config.community.residents
    counts = per_resident.reindex(
        [f"R{index + 1:04d}" for index in range(roster)], fill_value=0
    )
    sparse = float(
        (counts < rules.sparse_resident_bookings).sum() / max(roster, 1)
    )
    return Check(
        "a_real_share_of_residents_are_sparse",
        sparse >= rules.sparse_share_min,
        {
            "sparse_share": round(sparse, 4),
            "minimum": rules.sparse_share_min,
            "threshold_bookings": rules.sparse_resident_bookings,
        },
    )


def _ordering_checks(frame: pd.DataFrame) -> list[Check]:
    """Check the orderings every later stage depends on.

    Args:
        frame: The booking table.

    Returns:
        The ordering checks.
    """
    ordered = frame.sort_values(
        ["resident_id", "booking_timestamp"], kind="mergesort"
    )
    previous = ordered.groupby("resident_id", sort=False)[
        "booking_timestamp"
    ].shift(1)
    tied = int(ordered["booking_timestamp"].le(previous).sum())
    sorted_frame = frame.sort_values(
        ["booking_timestamp", "booking_id"], kind="mergesort"
    )
    return [
        Check(
            "booking_precedes_usage_on_every_row",
            bool((frame["booking_timestamp"] < frame["usage_timestamp"]).all()),
            {
                "violations": int(
                    (
                        frame["booking_timestamp"] >= frame["usage_timestamp"]
                    ).sum()
                )
            },
        ),
        Check(
            "resident_bookings_strictly_increase",
            tied == 0,
            {"violations": tied},
        ),
        Check(
            "table_is_in_its_deterministic_order",
            bool(sorted_frame["booking_id"].equals(frame["booking_id"])),
            {"rows": len(frame)},
        ),
        Check(
            "usage_hour_inside_operating_hours",
            True,
            {"checked_by": BOOKING_SCHEMA_NAME},
        ),
    ]


def acceptance_checks(
    frame: pd.DataFrame,
    config: config_module.Config,
    audit: GenerationAudit,
) -> list[Check]:
    """Run every acceptance check against a generated dataset.

    Args:
        frame: The booking table.
        config: Validated configuration.
        audit: The aggregate counters from generation.

    Returns:
        Every check, in a stable order.
    """
    return [
        *_ordering_checks(frame),
        *_lead_checks(frame, config),
        *_availability_checks(frame, config),
        *_popularity_checks(frame, config),
        *_preference_checks(frame, config, audit),
        *_drift_checks(config, audit),
        _sparsity_check(frame, config),
    ]


def build_generation_summary(
    frame: pd.DataFrame,
    config: config_module.Config,
    audit: GenerationAudit,
    checks: list[Check],
) -> dict[str, Any]:
    """Assemble the record of what was generated and what it satisfies.

    Args:
        frame: The booking table.
        config: Validated configuration.
        audit: The aggregate counters from generation.
        checks: The results of :func:`acceptance_checks`.

    Returns:
        The summary payload, ready to serialise.
    """
    return {
        "provenance": {
            "seed": config.seed,
            "timezone": config.timezone,
            "bookings_digest": bookings_digest(frame),
        },
        "counts": {
            "bookings": len(frame),
            "residents_configured": config.community.residents,
            "residents_with_bookings": int(frame["resident_id"].nunique()),
            "residents_resampled": audit.resampled_residents,
        },
        "realised": {
            "noise_path_share": round(audit.noise_share(), 4),
            "facility_shares": {
                str(name): round(float(share), 4)
                for name, share in _shares(
                    frame["facility_id"].value_counts()
                ).items()
            },
            "lead_hours": {
                "p50": round(float(_lead_hours(frame).median()), 2),
                "mean": round(float(_lead_hours(frame).mean()), 2),
            },
        },
        "checks": [check.to_dict() for check in checks],
        "all_passed": all(check.passed for check in checks),
    }


def check_dataset(
    frame: pd.DataFrame,
    config: config_module.Config,
    audit: GenerationAudit,
) -> dict[str, Any]:
    """Validate a generated dataset and refuse to pass a failing one.

    Args:
        frame: The booking table.
        config: Validated configuration.
        audit: The aggregate counters from generation.

    Returns:
        The generation summary.

    Raises:
        GenerationError: If the structural schema fails, or if an
            acceptance check fails while this configuration enforces
            them. A dataset that does not demonstrate the properties it
            is supposed to demonstrate is not a dataset to build
            results on; a deliberately tiny one is not expected to.
    """
    try:
        booking_schema(config).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as error:
        msg = f"booking table violates its schema:\n{error.failure_cases}"
        raise GenerationError(msg) from error

    checks = acceptance_checks(frame, config, audit)
    summary = build_generation_summary(frame, config, audit, checks)
    failed = [check.name for check in checks if not check.passed]
    if failed and config.generator.acceptance.enforce:
        msg = f"generated dataset fails its acceptance checks: {failed}"
        raise GenerationError(msg)
    if failed:
        _LOGGER.warning(
            "%d acceptance checks did not hold at this scale: %s",
            len(failed),
            failed,
        )
    return summary


def generate_audited(
    config: config_module.Config,
) -> tuple[pd.DataFrame, GenerationAudit]:
    """Generate the synthetic booking table.

    Leakage contract: no prediction origin exists yet, so there is no
    as-of bound to respect. What this function does guarantee is the
    ordering later stages rely on — ``booking_timestamp <
    usage_timestamp`` on every row, and strictly increasing booking
    timestamps within each resident.

    Args:
        config: Validated configuration; ``config.seed`` alone
            determines the result.

    Returns:
        A frame with :data:`BOOKING_COLUMNS`, timestamps tz-aware in the
        configured timezone, sorted by ``(booking_timestamp,
        booking_id)``, and the aggregate audit counters. Hidden
        resident behaviour is in neither.
    """
    rng = np.random.default_rng(config.seed)
    start = config.start_instant
    horizon_end = start + pd.DateOffset(months=config.community.months)
    weeks = int((horizon_end - start).days // _DAYS_IN_WEEK)
    residents = _build_residents(config, rng, weeks)
    audit = GenerationAudit()
    bookings, drops = _generate_rows(config, rng, residents, weeks, audit)
    frame = _assemble_frame(bookings)
    _LOGGER.info(
        "generated %d bookings for %d residents over %d weeks",
        len(frame),
        len(residents),
        weeks,
    )
    _LOGGER.info(
        "discarded %d draws: %d past horizon, %d before opening, "
        "%d no free hour, %d lead before start, %d simultaneous",
        drops.total(),
        drops.past_horizon,
        drops.before_opening,
        drops.no_free_hour,
        drops.lead_before_start,
        drops.simultaneous,
    )
    return frame, audit


def generate_bookings(config: config_module.Config) -> pd.DataFrame:
    """Generate the booking table, discarding the aggregate audit.

    Most callers want the dataset and nothing else. The acceptance
    checks want the counters too, and call :func:`generate_audited`.

    Leakage contract: as :func:`generate_audited` — this module creates
    time rather than reading it, and guarantees ``booking_timestamp <
    usage_timestamp`` on every row with strictly increasing booking
    timestamps within each resident.

    Args:
        config: Validated configuration; ``config.seed`` alone
            determines the result.

    Returns:
        The booking table.
    """
    frame, _ = generate_audited(config)
    return frame


def _assemble_frame(bookings: list[_Booking]) -> pd.DataFrame:
    """Sort the generated rows and assign booking identifiers.

    Identifiers are assigned after the sort, so they increase with
    ``booking_timestamp`` and the file is byte-identical on re-run.

    Args:
        bookings: Bookings as returned by :func:`_generate_rows`.

    Returns:
        A frame with :data:`BOOKING_COLUMNS` in that column order.
    """
    frame = pd.DataFrame(
        [
            {
                "resident_id": booking.resident_id,
                "facility_id": booking.facility_id,
                "booking_timestamp": booking.booking_timestamp,
                "usage_timestamp": booking.usage_timestamp,
            }
            for booking in bookings
        ],
        columns=[
            "resident_id",
            "facility_id",
            "booking_timestamp",
            "usage_timestamp",
        ],
    )
    frame = frame.sort_values(
        ["booking_timestamp", "resident_id", "facility_id", "usage_timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    frame.insert(
        0,
        "booking_id",
        [f"B{index + 1:07d}" for index in range(len(frame))],
    )
    return frame[list(BOOKING_COLUMNS)]


def bookings_digest(frame: pd.DataFrame) -> str:
    """Return the canonical digest of a booking table.

    Unlike a file hash, this survives the trip through the database, so
    the stored table and the exported file can be compared directly.

    Args:
        frame: A frame carrying :data:`BOOKING_COLUMNS`.

    Returns:
        The hex SHA-256 over the canonically rendered rows.
    """
    return digest.canonical_digest(
        frame, sort_by=("booking_id",), columns=BOOKING_COLUMNS
    )


def write_bookings(frame: pd.DataFrame, path: pathlib.Path) -> str:
    """Export the booking table and return the file's content hash.

    The database is the store of record; this file is the exported
    dataset, written deterministically so its hash is stable.
    Timestamps are ISO 8601 with the offset, so the file round-trips
    tz-aware.

    Args:
        frame: A frame as returned by :func:`generate_bookings`.
        path: Destination CSV; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    output = frame.copy()
    for column in ("booking_timestamp", "usage_timestamp"):
        output[column] = output[column].map(lambda value: value.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _LOGGER.info("wrote %s (sha256=%s)", path, digest)
    return digest
