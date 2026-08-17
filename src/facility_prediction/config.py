"""Typed configuration — one file, one source of truth.

`configs/default.yaml` is the sole definition of dataset shape. This
module turns it into a frozen, validated Pydantic v2 object, so these
conventions are enforced at load time instead of by scattered asserts:

- ``available_from_month`` is a zero-based elapsed-month index
  counted from ``community.start_date``, and must fall inside the
  configured horizon.
- ``hours`` is the half-open interval ``[open_hour, close_hour)``.
- Naive datetimes are rejected anywhere in the file, by the loader
  and by the model itself.
- The facility catalog is the categorical enum: downstream code reads
  :attr:`Config.facility_names` and never retypes the list.

Leakage contract: this module is time-independent. It reads no event
data, so it has no origin/target ordering to preserve. :meth:`Config.
start_instant` derives a timezone-aware instant from configuration only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
from typing import Any, ClassVar, Literal, Self
import zoneinfo

import pydantic
import yaml

_HOURS_IN_DAY = 24
_MONTHS_IN_YEAR = 12
_POPULARITY_SUM = 1.0
_POPULARITY_SUM_TOLERANCE = 1e-6
_SHARE_SUM = 1.0
_SHARE_SUM_TOLERANCE = 1e-6
_MAX_CATBOOST_DEPTH = 16


class ConfigError(Exception):
    """Raised when a configuration file cannot be loaded or validated."""


class _StrictModel(pydantic.BaseModel):
    """Base for every config model: frozen, and unknown keys are errors."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )


class Facility(_StrictModel):
    """One bookable facility.

    Attributes:
        name: Catalog name; also the categorical level used downstream.
        popularity: Share of community demand, in ``(0, 1]``.
        hours: Half-open operating interval ``[open_hour, close_hour)``.
        available_from_month: Zero-based elapsed-month index at which the
            facility opens, counted from ``community.start_date``.
        weekend_multiplier: Weekend demand relative to a weekday.
        lead_hours_median: Median booking lead time in hours. Halls
            book far further ahead than the gym.
        lead_log_sigma: Log-scale spread of that lead distribution.
        slot_capacity: Bookings one facility-hour absorbs before a
            resident is displaced to another hour.
    """

    name: str = pydantic.Field(min_length=1)
    popularity: float = pydantic.Field(gt=0.0, le=1.0)
    hours: tuple[int, int]
    available_from_month: int = pydantic.Field(ge=0)
    weekend_multiplier: float = pydantic.Field(gt=0.0)
    lead_hours_median: float = pydantic.Field(gt=0.0)
    lead_log_sigma: float = pydantic.Field(gt=0.0)
    slot_capacity: pydantic.PositiveInt

    @pydantic.field_validator("hours")
    @classmethod
    def _check_half_open(cls, value: tuple[int, int]) -> tuple[int, int]:
        open_hour, close_hour = value
        if not 0 <= open_hour < close_hour <= _HOURS_IN_DAY:
            msg = (
                "hours must be a half-open interval "
                f"0 <= open < close <= {_HOURS_IN_DAY}, got {value}"
            )
            raise ValueError(msg)
        return value

    @property
    def open_hour(self) -> int:
        """Return the first bookable hour."""
        return self.hours[0]

    @property
    def close_hour(self) -> int:
        """Return the first hour that is no longer bookable."""
        return self.hours[1]

    def is_open_at(self, hour: int) -> bool:
        """Return whether ``hour`` lies in the half-open operating window.

        Args:
            hour: Hour of day in ``[0, 24)``.

        Returns:
            True when ``open_hour <= hour < close_hour``.
        """
        return self.open_hour <= hour < self.close_hour


class Community(_StrictModel):
    """Scale and horizon of the simulated community.

    Attributes:
        residents: Number of residents generated.
        months: Length of the horizon in whole months.
        start_date: Calendar date of month index 0, in the configured
            timezone. A datetime is rejected; the instant is derived by
            :meth:`Config.start_instant`, never carried naively.
    """

    residents: pydantic.PositiveInt
    months: pydantic.PositiveInt
    start_date: datetime.date

    @pydantic.field_validator("start_date", mode="before")
    @classmethod
    def _reject_datetime(cls, value: Any) -> Any:
        if isinstance(value, datetime.datetime):
            kind = "naive" if value.tzinfo is None else "aware"
            msg = (
                f"start_date must be a calendar date, got a {kind} "
                f"datetime {value!r}; the timezone comes from `timezone`"
            )
            raise ValueError(msg)
        return value


class Split(_StrictModel):
    """Chronological split fractions of elapsed target time.

    Attributes:
        train_frac: Share of the elapsed target-time span used to train.
        val_frac: Share used to validate.
        embargo_days: Gap held out at each boundary; zero by default.
        comparison_rows: Cap on the seeded holdout sample both tracks are
            compared on. A smaller holdout is used whole.
    """

    train_frac: float = pydantic.Field(gt=0.0, lt=1.0)
    val_frac: float = pydantic.Field(gt=0.0, lt=1.0)
    embargo_days: int = pydantic.Field(ge=0)
    comparison_rows: pydantic.PositiveInt

    @pydantic.model_validator(mode="after")
    def _check_leaves_test_span(self) -> Self:
        if self.train_frac + self.val_frac >= 1.0:
            msg = (
                "train_frac + val_frac must leave a test span, got "
                f"{self.train_frac} + {self.val_frac}"
            )
            raise ValueError(msg)
        return self

    @property
    def test_frac(self) -> float:
        """Return the residual share of elapsed target time held for test."""
        return 1.0 - self.train_frac - self.val_frac


class Storage(_StrictModel):
    """How to reach the database that holds every generated table.

    The password is absent: it is read from the environment variable
    named here, never committed with these settings.

    Attributes:
        driver: SQLAlchemy driver name.
        host: Database host.
        port: Database port.
        user: Role to connect as.
        database: Database holding the pipeline's tables.
        password_env: Environment variable carrying the password.
    """

    driver: str = pydantic.Field(min_length=1)
    host: str = pydantic.Field(min_length=1)
    port: int = pydantic.Field(gt=0, le=65535)
    user: str = pydantic.Field(min_length=1)
    database: str = pydantic.Field(min_length=1)
    password_env: str = pydantic.Field(min_length=1)


class Tracking(_StrictModel):
    """Where runs, prompts, and call traces are recorded.

    Tracking mirrors the pipeline. Turning it off changes no artifact,
    so a replay needs no server.

    Attributes:
        enabled: Whether to record at all. False makes every tracking
            call a no-op instead of an error.
        experiment: The one experiment both tracks write runs to.
        tracking_uri_env: Environment variable carrying the server URI,
            so the address lives beside the database credentials rather
            than in two places.
        prompt_alias: Alias naming the live registered prompt version.
    """

    enabled: bool
    experiment: str = pydantic.Field(min_length=1)
    tracking_uri_env: str = pydantic.Field(min_length=1)
    prompt_alias: str = pydantic.Field(min_length=1)


class Evaluation(_StrictModel):
    """Scored-metric settings.

    Attributes:
        notification_match_ratio: Symmetric multiplicative tolerance for a
            notification-time match; ``1.25`` means +/-25%.
        notification_support_minutes: Supporting absolute tolerances, in
            strictly increasing minutes.
        min_prior_bookings: Prior bookings a resident needs before a
            sample is eligible.
    """

    notification_match_ratio: float = pydantic.Field(gt=1.0)
    notification_support_minutes: tuple[int, ...] = pydantic.Field(min_length=1)
    min_prior_bookings: int = pydantic.Field(ge=0)

    @pydantic.field_validator("notification_support_minutes")
    @classmethod
    def _check_increasing(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(minutes <= 0 for minutes in value):
            msg = f"support tolerances must be positive, got {value}"
            raise ValueError(msg)
        if list(value) != sorted(set(value)):
            msg = f"support tolerances must strictly increase, got {value}"
            raise ValueError(msg)
        return value


class Application(_StrictModel):
    """Display-only application settings; never scored.

    Attributes:
        notification_lead_minutes: Lead used to render ``suggested_send``.
    """

    notification_lead_minutes: int = pydantic.Field(ge=0)


class Review(_StrictModel):
    """Prediction-review workbook settings.

    Attributes:
        history_rows: Depth of the "Past bookings" column; display only.
    """

    history_rows: int = pydantic.Field(ge=0)


class TimeBand(_StrictModel):
    """One named part of the day.

    Attributes:
        name: Band label; also the suffix of the columns it produces.
        hours: Half-open interval ``[open_hour, close_hour)``.
    """

    name: str = pydantic.Field(min_length=1)
    hours: tuple[int, int]

    @pydantic.field_validator("hours")
    @classmethod
    def _check_half_open(cls, value: tuple[int, int]) -> tuple[int, int]:
        open_hour, close_hour = value
        if not 0 <= open_hour < close_hour <= _HOURS_IN_DAY:
            msg = (
                "hours must be a half-open interval "
                f"0 <= open < close <= {_HOURS_IN_DAY}, got {value}"
            )
            raise ValueError(msg)
        return value

    @property
    def open_hour(self) -> int:
        """Return the first hour inside the band."""
        return self.hours[0]

    @property
    def close_hour(self) -> int:
        """Return the first hour that is no longer in the band."""
        return self.hours[1]


class Features(_StrictModel):
    """Feature-table conventions.

    Attributes:
        categorical_missing_token: The single sentinel written into every
            missing categorical slot.
        history_facility_slots: How many of the most recent facilities
            are kept as their own categorical columns.
        prior_windows_days: Lookback windows, in days, that recent
            bookings are counted over. Strictly increasing.
        trend_windows_days: The short and long window whose booking
            rates form the trend ratio. Both must be counted windows.
        rolling_preference_bookings: Depths, in bookings, that a recent
            favourite facility is taken over. Strictly increasing.
        ewma_halflife_days: Half-life, in days, of the exponentially
            weighted preference features: an event this many days before
            the origin counts half as much as one at the origin.
        community_windows_days: Lookback windows, in days, that
            community-wide facility popularity is measured over.
            Strictly increasing.
        time_bands: Named parts of the day; they must tile the whole day
            without a gap or an overlap, or usage would be counted twice
            or not at all.
    """

    categorical_missing_token: str = pydantic.Field(min_length=1)
    history_facility_slots: pydantic.PositiveInt
    prior_windows_days: tuple[int, ...] = pydantic.Field(min_length=1)
    trend_windows_days: tuple[int, int]
    rolling_preference_bookings: tuple[int, ...] = pydantic.Field(min_length=1)
    ewma_halflife_days: pydantic.PositiveFloat
    community_windows_days: tuple[int, ...] = pydantic.Field(min_length=1)
    time_bands: tuple[TimeBand, ...] = pydantic.Field(min_length=1)

    @pydantic.field_validator(
        "prior_windows_days",
        "rolling_preference_bookings",
        "community_windows_days",
    )
    @classmethod
    def _check_increasing(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(entry <= 0 for entry in value):
            msg = f"window sizes must be positive, got {value}"
            raise ValueError(msg)
        if list(value) != sorted(set(value)):
            msg = f"window sizes must strictly increase, got {value}"
            raise ValueError(msg)
        return value

    @pydantic.model_validator(mode="after")
    def _check_trend_pair(self) -> Self:
        short_days, long_days = self.trend_windows_days
        if short_days >= long_days:
            msg = (
                "trend_windows_days must be (short, long) with short < "
                f"long, got {self.trend_windows_days}"
            )
            raise ValueError(msg)
        unknown = [
            days
            for days in self.trend_windows_days
            if days not in self.prior_windows_days
        ]
        if unknown:
            msg = (
                "trend_windows_days must name windows that are counted; "
                f"{unknown} are not in prior_windows_days"
            )
            raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def _check_bands_tile_the_day(self) -> Self:
        names = [band.name for band in self.time_bands]
        if len(set(names)) != len(names):
            msg = f"time band names must be unique, got {names}"
            raise ValueError(msg)

        ordered = sorted(self.time_bands, key=lambda band: band.open_hour)
        boundary = 0
        for band in ordered:
            if band.open_hour != boundary:
                msg = (
                    "time bands must tile [0, 24) with no gap or overlap; "
                    f"expected a band opening at {boundary}, got "
                    f"{band.name!r} opening at {band.open_hour}"
                )
                raise ValueError(msg)
            boundary = band.close_hour
        if boundary != _HOURS_IN_DAY:
            msg = f"time bands must reach hour {_HOURS_IN_DAY}, got {boundary}"
            raise ValueError(msg)
        return self


class BootstrapSetting(_StrictModel):
    """One CatBoost bootstrap configuration, pinned per head.

    Attributes:
        type: CatBoost ``bootstrap_type``.
        subsample: Sampling rate, where the bootstrap type takes one.
    """

    type: Literal["MVS", "Bernoulli", "Bayesian", "Poisson", "No"]
    subsample: float | None = pydantic.Field(default=None, gt=0.0, le=1.0)


class Bootstrap(_StrictModel):
    """Per-head bootstrap settings — not one global default.

    Attributes:
        notification: Setting for the notification-bucket classifier.
        classifiers: Setting shared by the facility/day/hour heads.
    """

    notification: BootstrapSetting
    classifiers: BootstrapSetting


class CatBoost(_StrictModel):
    """Resolved CatBoost settings, fixed before any fit.

    Attributes:
        task_type: CatBoost ``task_type``.
        thread_count: Worker threads per fit.
        random_seed_from_root: Derive each head's seed from ``seed``.
        iterations: Iteration count used when the inner search is off,
            and the ceiling a searched count is clipped to.
        iteration_search: Choose each head's iteration count on an inner
            fold cut from the training rows, then refit on all of them.
            No validation or holdout row takes part.
        inner_validation_frac: Share of training rows, the latest ones,
            held out as that inner fold.
        inner_od_wait: Iterations without improvement before the inner
            search stops looking.
        depth: Tree depth.
        learning_rate: Learning rate.
        l2_leaf_reg: L2 regularisation on leaf values.
        use_best_model: Must stay false — no early stopping, so the
            iteration count is the one written in this file.
        allow_writing_files: Keep CatBoost from writing side artifacts.
        stretch_fit_budget: Validation-only fits allowed after the
            primary heads freeze; the ceiling a stretch run stops at.
        notification_bucket_ratio: Width of one finite notification
            bucket on a multiplicative scale.
        notification_bucket_floor_minutes: Upper edge of the first
            notification bucket.
        notification_bucket_ceiling_days: Lower edge of the final,
            open-ended notification bucket.
        notification_decode: How a notification bucket distribution
            becomes one delay. ``window`` submits the delay whose scored
            tolerance window carries the most predicted mass; ``argmax``
            submits the most likely bucket's geometric representative.
        notification_framing: What the notification head predicts.
            ``absolute`` buckets the delay itself; ``cadence`` buckets
            the residual around the resident's own median booking gap,
            so the personalisation arrives as an offset rather than
            something the model has to relearn per resident.
        notification_residual_span: Half-width of the modelled residual
            range, in natural-log units, under the ``cadence`` framing.
        notification_residual_step: Width of one residual bucket, in
            natural-log units.
        notification_residual_blend: Weight on the model's own residual
            density against the training marginal at decode time; one
            uses the model alone.
        facility_framing: What the facility head predicts. ``multiclass``
            scores eight classes from one wide row; ``ranking`` scores
            one row per candidate facility and takes the top of the
            ranking.
        facility_ranking_loss: CatBoost ranking objective used when the
            facility head is framed as ranking.
        bootstrap: Per-head bootstrap settings.
    """

    task_type: Literal["CPU", "GPU"]
    thread_count: pydantic.PositiveInt
    random_seed_from_root: bool
    iterations: pydantic.PositiveInt
    iteration_search: bool = False
    inner_validation_frac: float = pydantic.Field(default=0.15, gt=0.0, lt=0.5)
    inner_od_wait: pydantic.PositiveInt = 100
    depth: int = pydantic.Field(ge=1, le=_MAX_CATBOOST_DEPTH)
    learning_rate: float = pydantic.Field(gt=0.0, le=1.0)
    l2_leaf_reg: float = pydantic.Field(gt=0.0)
    use_best_model: bool
    allow_writing_files: bool
    stretch_fit_budget: pydantic.NonNegativeInt
    notification_bucket_ratio: float = pydantic.Field(gt=1.0)
    notification_bucket_floor_minutes: pydantic.PositiveInt
    notification_bucket_ceiling_days: pydantic.PositiveInt
    notification_decode: Literal["window", "argmax"] = "argmax"
    notification_framing: Literal["absolute", "cadence"] = "absolute"
    notification_residual_span: float = pydantic.Field(default=4.5, gt=0.0)
    notification_residual_step: float = pydantic.Field(default=0.15, gt=0.0)
    notification_residual_blend: float = pydantic.Field(
        default=1.0, gt=0.0, le=1.0
    )
    facility_framing: Literal["multiclass", "ranking"] = "multiclass"
    facility_ranking_loss: Literal["YetiRank", "QuerySoftMax", "PairLogit"] = (
        "YetiRank"
    )
    bootstrap: Bootstrap

    @pydantic.model_validator(mode="after")
    def _check_residual_grid(self) -> CatBoost:
        """Reject a residual grid that would not tile the modelled span.

        Returns:
            The validated settings.

        Raises:
            ValueError: If one bucket is wider than the whole span.
        """
        if self.notification_residual_step > self.notification_residual_span:
            msg = (
                "notification_residual_step must fit inside "
                f"notification_residual_span, got "
                f"{self.notification_residual_step} in "
                f"{self.notification_residual_span}"
            )
            raise ValueError(msg)
        return self


def _check_shares_sum_to_one(shares: list[float], label: str) -> None:
    """Raise when a mixture's shares do not sum to one.

    Args:
        shares: The mixture weights.
        label: Name of the mixture, used in the error message.

    Raises:
        ValueError: If the shares miss 1.0 by more than the tolerance.
    """
    total = sum(shares)
    if abs(total - _SHARE_SUM) > _SHARE_SUM_TOLERANCE:
        msg = f"{label} shares must sum to 1.0, got {total}"
        raise ValueError(msg)


class ActivityClass(_StrictModel):
    """One resident activity rate class.

    Attributes:
        name: Class label; generator-only, never written to the dataset.
        share: Fraction of residents assigned to this class.
        weekly_mean: Mean bookings per active week.
        dispersion: Negative-binomial dispersion; small is burstier.
    """

    name: str = pydantic.Field(min_length=1)
    share: float = pydantic.Field(gt=0.0, le=1.0)
    weekly_mean: float = pydantic.Field(gt=0.0)
    dispersion: float = pydantic.Field(gt=0.0)


class Activity(_StrictModel):
    """The zero-inflated negative-binomial booking process.

    Attributes:
        zero_inflation: Probability that an active week yields nothing.
        classes: The activity rate classes; shares sum to one.
    """

    zero_inflation: float = pydantic.Field(ge=0.0, lt=1.0)
    classes: tuple[ActivityClass, ...] = pydantic.Field(min_length=1)

    @pydantic.field_validator("classes")
    @classmethod
    def _check_shares(
        cls, value: tuple[ActivityClass, ...]
    ) -> tuple[ActivityClass, ...]:
        _check_shares_sum_to_one(
            [entry.share for entry in value], "activity class"
        )
        return value


class Archetype(_StrictModel):
    """One behavioural archetype.

    Every attribute is generator-only state. None of it reaches the
    dataset or the modelling table.

    Attributes:
        name: Archetype label.
        share: Fraction of residents drawn from this archetype.
        facilities: Catalog names this archetype gravitates towards; an
            empty tuple means no facility affinity.
        hour_center: Preferred hour of day, in ``[0, 24)``.
        hour_spread: Spread of the circular-normal hour band, in hours.
        weekend_share: Fraction of this archetype's usage on weekends.
        lead_scale: Multiplier on the facility's median lead time.
    """

    name: str = pydantic.Field(min_length=1)
    share: float = pydantic.Field(gt=0.0, le=1.0)
    facilities: tuple[str, ...]
    hour_center: float = pydantic.Field(ge=0.0, lt=float(_HOURS_IN_DAY))
    hour_spread: float = pydantic.Field(gt=0.0)
    weekend_share: float = pydantic.Field(gt=0.0, lt=1.0)
    lead_scale: float = pydantic.Field(gt=0.0)


class Preference(_StrictModel):
    """Shape of a resident's facility preferences.

    Attributes:
        dirichlet_concentration: Base Dirichlet alpha; below one gives
            each resident one or two dominant facilities.
        primary_boost: Alpha added for the archetype's own facilities.
        recency_boost: Weight multiplier on the facility a resident
            used last, so recent behaviour carries forward.
    """

    dirichlet_concentration: float = pydantic.Field(gt=0.0)
    primary_boost: float = pydantic.Field(ge=0.0)
    recency_boost: float = pydantic.Field(ge=1.0)


class Consistency(_StrictModel):
    """Range of preference-following strength drawn per resident.

    Attributes:
        low: Lowest consistency drawn, as a weight on personal
            preference against community popularity.
        high: Highest consistency drawn.
    """

    low: float = pydantic.Field(gt=0.0, le=1.0)
    high: float = pydantic.Field(gt=0.0, le=1.0)

    @pydantic.model_validator(mode="after")
    def _check_ordered(self) -> Self:
        if self.low >= self.high:
            msg = f"consistency low must be < high, got {self.low}, {self.high}"
            raise ValueError(msg)
        return self


class Acceptance(_StrictModel):
    """Thresholds the generated dataset must meet to be accepted.

    Each field is a threshold a generated dataset must meet.

    Attributes:
        enforce: Whether a failed check stops the pipeline. A tiny
            configuration cannot demonstrate a population-level
            property, so a smoke run records the results without
            being blocked by them. The real dataset always enforces.
        lead_p50_hours: Allowed range for the median booking lead time.
        long_lead_ratio: How many times the short-lead facility's median
            lead the long-lead facility must reach.
        long_lead_facility: The facility expected to book furthest ahead.
        short_lead_facility: The one expected to book most immediately.
        active_resident_bookings: Bookings that make a resident "active"
            for the preference checks.
        primary_share_min: Median share at an active resident's declared
            favourite facility.
        primary_share_excess: How far that must exceed the facility's
            own community share.
        time_pattern_excess: How far an archetype's morning or weekend
            share must exceed the community's.
        morning_archetype: Archetype expected to book mornings.
        weekend_archetype: Archetype expected to book weekends.
        imbalance_ratio: Most-used facility share against least-used.
        noise_share: Allowed range for the realised noise-path share.
        absorption_drop: How far the absorbed facility's share must fall
            for the affected archetype after the dated opening.
        resample_distance: Jensen-Shannon distance the life-change
            cohort's facility mix must move.
        sparse_resident_bookings: Bookings below which a resident counts
            as sparse.
        sparse_share_min: Share of residents that must be sparse.
        top_facilities: How many facilities the realised and configured
            popularity rankings must agree on, in order.
    """

    enforce: bool
    lead_p50_hours: tuple[float, float]
    long_lead_ratio: float = pydantic.Field(gt=1.0)
    long_lead_facility: str = pydantic.Field(min_length=1)
    short_lead_facility: str = pydantic.Field(min_length=1)
    active_resident_bookings: pydantic.PositiveInt
    primary_share_min: float = pydantic.Field(gt=0.0, lt=1.0)
    primary_share_excess: float = pydantic.Field(gt=0.0, lt=1.0)
    time_pattern_excess: float = pydantic.Field(gt=0.0, lt=1.0)
    morning_archetype: str = pydantic.Field(min_length=1)
    weekend_archetype: str = pydantic.Field(min_length=1)
    imbalance_ratio: float = pydantic.Field(gt=1.0)
    noise_share: tuple[float, float]
    absorption_drop: float = pydantic.Field(gt=0.0, lt=1.0)
    resample_distance: float = pydantic.Field(gt=0.0, lt=1.0)
    sparse_resident_bookings: pydantic.PositiveInt
    sparse_share_min: float = pydantic.Field(gt=0.0, lt=1.0)
    top_facilities: pydantic.PositiveInt

    @pydantic.field_validator("lead_p50_hours", "noise_share")
    @classmethod
    def _check_band(cls, value: tuple[float, float]) -> tuple[float, float]:
        low, high = value
        if not 0.0 <= low < high:
            msg = f"a band must be (low, high) with low < high, got {value}"
            raise ValueError(msg)
        return value


class Season(_StrictModel):
    """A continuous seasonal swing on one facility's demand.

    Attributes:
        facility: Catalog name the swing applies to.
        amplitude: Peak deviation from the flat multiplier of one, so
            0.35 means demand ranges over roughly 0.65 to 1.35.
        peak_month: Zero-based calendar month at the top of the swing.
    """

    facility: str = pydantic.Field(min_length=1)
    amplitude: float = pydantic.Field(gt=0.0, lt=1.0)
    peak_month: int = pydantic.Field(ge=0, lt=_MONTHS_IN_YEAR)


class Drift(_StrictModel):
    """Dated behaviour changes, so "changing behaviour" can be sliced.

    Each field names a dated event, so the drift can be tested for and
    shown as a slice.

    Attributes:
        absorbing_facility: Facility that opens partway through and
            takes demand from another.
        absorbed_facility: Facility whose demand it takes.
        absorbing_archetype: Archetype whose members move; others are
            unaffected.
        absorption: Share of the absorbed facility's preference weight
            that moves, once the absorbing facility is open.
        resample_month: Zero-based elapsed month at which some residents
            change archetype, as a life change would.
        resample_share: Fraction of residents resampled then.
        season: The continuous seasonal swing.
    """

    absorbing_facility: str = pydantic.Field(min_length=1)
    absorbed_facility: str = pydantic.Field(min_length=1)
    absorbing_archetype: str = pydantic.Field(min_length=1)
    absorption: float = pydantic.Field(gt=0.0, lt=1.0)
    resample_month: int = pydantic.Field(ge=1)
    resample_share: float = pydantic.Field(gt=0.0, lt=1.0)
    season: Season

    @pydantic.model_validator(mode="after")
    def _check_distinct(self) -> Self:
        if self.absorbing_facility == self.absorbed_facility:
            msg = (
                "a facility cannot absorb its own demand, got "
                f"{self.absorbing_facility!r} for both"
            )
            raise ValueError(msg)
        return self


class Join(_StrictModel):
    """When residents enter the community.

    Attributes:
        early_share: Fraction present from the first week; the rest join
            uniformly across the horizon, giving short histories.
    """

    early_share: float = pydantic.Field(gt=0.0, le=1.0)


class Generator(_StrictModel):
    """Settings for the synthetic booking generator.

    Attributes:
        noise_fraction: Share of bookings drawn from community
            popularity instead of personal preference.
        jitter_minutes: Symmetric jitter applied before an hour is
            snapped to a bookable slot.
        min_lead_minutes: Floor on booking-to-usage lead, so
            ``booking_timestamp < usage_timestamp`` always holds.
        displacement_search_hours: How far either side of a full slot to
            look for a free hour before dropping the booking.
        capacity_pressure_slack: How many free seats still count as
            "nearly full", and so still push the booking earlier.
        hour_draw_attempts: Draws allowed from a resident's preferred
            hour band before falling back to a uniform open hour.
        capacity_lead_multiplier: Extra lead applied when the chosen
            slot was already near capacity.
        resident_lead_log_sigma: Spread of the per-resident lead-time
            multiplier.
        acceptance: Thresholds the generated dataset must meet.
        drift: Dated behaviour changes and the seasonal swing.
        join: Resident join-date settings.
        activity: The weekly booking-count process.
        consistency: Preference-following strength range.
        preference: Facility-preference shape.
        archetypes: The behavioural archetypes; shares sum to one.
    """

    noise_fraction: float = pydantic.Field(ge=0.0, lt=1.0)
    jitter_minutes: int = pydantic.Field(ge=0)
    min_lead_minutes: pydantic.PositiveInt
    displacement_search_hours: int = pydantic.Field(ge=0)
    capacity_pressure_slack: int = pydantic.Field(ge=0)
    hour_draw_attempts: pydantic.PositiveInt
    capacity_lead_multiplier: float = pydantic.Field(ge=1.0)
    resident_lead_log_sigma: float = pydantic.Field(gt=0.0)
    acceptance: Acceptance
    drift: Drift
    join: Join
    activity: Activity
    consistency: Consistency
    preference: Preference
    archetypes: tuple[Archetype, ...] = pydantic.Field(min_length=1)

    @pydantic.field_validator("archetypes")
    @classmethod
    def _check_shares(
        cls, value: tuple[Archetype, ...]
    ) -> tuple[Archetype, ...]:
        names = [entry.name for entry in value]
        if len(set(names)) != len(names):
            msg = f"archetype names must be unique, got {names}"
            raise ValueError(msg)
        _check_shares_sum_to_one([entry.share for entry in value], "archetype")
        return value


class Config(_StrictModel):
    """The whole configuration file, validated.

    Attributes:
        seed: Root seed recorded in every artifact.
        timezone: IANA timezone name for all generated timestamps.
        community: Scale and horizon.
        facilities: The facility catalog, and therefore the categorical
            enum used downstream.
        generator: Synthetic-generator settings; generator-only.
        storage: Where the generated tables live.
        tracking: Where runs, prompts, and traces are mirrored.
        split: Chronological split fractions.
        evaluation: Scored-metric settings.
        application: Display-only application settings.
        review: Workbook display settings.
        features: Feature-table conventions.
        catboost: Resolved model settings.
    """

    seed: int
    timezone: str
    community: Community
    facilities: tuple[Facility, ...] = pydantic.Field(min_length=1)
    generator: Generator
    storage: Storage
    tracking: Tracking
    split: Split
    evaluation: Evaluation
    application: Application
    review: Review
    features: Features
    catboost: CatBoost

    @pydantic.field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        try:
            zoneinfo.ZoneInfo(value)
        except (zoneinfo.ZoneInfoNotFoundError, ValueError) as error:
            msg = f"unknown IANA timezone {value!r}"
            raise ValueError(msg) from error
        return value

    @pydantic.model_validator(mode="after")
    def _check_catalog(self) -> Self:
        names = [facility.name for facility in self.facilities]
        if len(set(names)) != len(names):
            msg = f"facility names must be unique, got {names}"
            raise ValueError(msg)

        total = sum(facility.popularity for facility in self.facilities)
        if abs(total - _POPULARITY_SUM) > _POPULARITY_SUM_TOLERANCE:
            msg = f"facility popularity must sum to 1.0, got {total}"
            raise ValueError(msg)

        horizon = self.community.months
        beyond = [
            facility.name
            for facility in self.facilities
            if facility.available_from_month >= horizon
        ]
        if beyond:
            msg = (
                "available_from_month is a zero-based index and must be "
                f"< community.months ({horizon}); beyond horizon: {beyond}"
            )
            raise ValueError(msg)
        return self

    @pydantic.model_validator(mode="after")
    def _check_archetype_facilities(self) -> Self:
        known = {facility.name for facility in self.facilities}
        for archetype in self.generator.archetypes:
            unknown = [
                name for name in archetype.facilities if name not in known
            ]
            if unknown:
                msg = (
                    f"archetype {archetype.name!r} names facilities absent "
                    f"from the catalog: {unknown}"
                )
                raise ValueError(msg)
        return self

    @property
    def tzinfo(self) -> zoneinfo.ZoneInfo:
        """Return the configured timezone as a `ZoneInfo`."""
        return zoneinfo.ZoneInfo(self.timezone)

    @property
    def facility_names(self) -> tuple[str, ...]:
        """Return the facility catalog: the one categorical enum."""
        return tuple(facility.name for facility in self.facilities)

    @property
    def start_instant(self) -> datetime.datetime:
        """Return midnight of ``start_date``, timezone-aware.

        Leakage contract: derived from configuration only; it observes no
        event and therefore cannot leak one.
        """
        return datetime.datetime.combine(
            self.community.start_date,
            datetime.time.min,
            tzinfo=self.tzinfo,
        )

    def facility(self, name: str) -> Facility:
        """Return the catalog entry for ``name``.

        Args:
            name: Facility name as it appears in the config catalog.

        Returns:
            The matching :class:`Facility`.

        Raises:
            KeyError: If ``name`` is not in the catalog.
        """
        for entry in self.facilities:
            if entry.name == name:
                return entry
        raise KeyError(name)


def _reject_naive_datetimes(node: Any, path: str) -> None:
    """Raise on any naive datetime found in a parsed YAML node.

    Args:
        node: Parsed YAML value.
        path: Dotted location of ``node``, used in the error message.

    Raises:
        ConfigError: If a `datetime.datetime` without a UTC offset is
            found anywhere beneath ``node``.
    """
    if isinstance(node, datetime.datetime):
        if node.tzinfo is None or node.tzinfo.utcoffset(node) is None:
            msg = f"naive datetime at {path}: {node!r}"
            raise ConfigError(msg)
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _reject_naive_datetimes(value, f"{path}.{key}")
        return
    if isinstance(node, list):
        for index, value in enumerate(node):
            _reject_naive_datetimes(value, f"{path}[{index}]")


def parse_config(raw: Any) -> Config:
    """Validate an already-parsed configuration mapping.

    Args:
        raw: Mapping as produced by `yaml.safe_load`.

    Returns:
        The validated, frozen :class:`Config`.

    Raises:
        ConfigError: If ``raw`` is not a mapping, contains a naive
            datetime, or fails schema validation.
    """
    if not isinstance(raw, dict):
        msg = f"configuration must be a mapping, got {type(raw).__name__}"
        raise ConfigError(msg)
    _reject_naive_datetimes(raw, "config")
    try:
        return Config.model_validate(raw)
    except pydantic.ValidationError as error:
        raise ConfigError(str(error)) from error


def load_config(path: pathlib.Path) -> Config:
    """Load and validate a YAML configuration file.

    Args:
        path: Path to a config file such as ``configs/default.yaml``.

    Returns:
        The validated, frozen :class:`Config`.

    Raises:
        ConfigError: If the file is unreadable, is not valid YAML, or
            fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"cannot read config {path}: {error}"
        raise ConfigError(msg) from error
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        msg = f"cannot parse config {path}: {error}"
        raise ConfigError(msg) from error
    return parse_config(raw)


def config_hash(config: Config) -> str:
    """Return a stable fingerprint of a resolved configuration.

    The hash is taken over the JSON rendering with keys sorted, so it
    depends on the settings themselves and not on the order they were
    written in. It is what freezes a run: a scoring pass records the
    hash it was configured with, and a later hash that differs means the
    settings moved after the freeze.

    Leakage contract: this function reads configuration only. It sees no
    event, no label, and no model output, so nothing it returns can
    carry information about the future.

    Args:
        config: The validated configuration to fingerprint.

    Returns:
        The hex SHA-256 of the canonical JSON rendering.
    """
    payload = json.dumps(
        config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dump_config(config: Config, path: pathlib.Path) -> None:
    """Write ``config`` back to YAML without loss.

    Round-tripping through this function and :func:`load_config` yields
    an equal :class:`Config`.

    Args:
        config: The configuration to serialise.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="json")
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )
