"""Rolling-origin prediction samples.

Turns the booking table into the prediction problems every later stage
is scored on. Within a resident's chronological sequence, each booking
that already has enough prior bookings becomes one sample:

    history b1 … b(i-1)   ->   origin = b(i-1).booking_timestamp
                          ->   target = bi, which supplies four labels

The labels are the target booking's facility, the weekday and hour of
its usage, and ``notification_delay_minutes`` — the gap from the origin
to the moment the target booking was created. The origin is anchored on
a booking event, never on a usage time, so the notification label does
not inherit the day and hour labels' errors.

Residents are never silently dropped. A resident whose whole history is
one booking cannot produce a sample by definition; that count is
reported rather than discarded. So is the right-censoring: the last
booking of every resident has no observed successor, so it is a censored
observation, not a negative example.

Leakage contract: a sample records the origin, and the labels of the
target booking. Nothing else about the target is carried forward.
Features built later must read the resident's own events with
``booking_timestamp <= origin`` and community events with
``booking_timestamp < origin``; this module reads the target booking
solely to write labels.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.data import digest as digest_module
from facility_prediction.data import generate

_LOGGER = logging.getLogger(__name__)

SAMPLE_COLUMNS = (
    "sample_id",
    "resident_id",
    "origin",
    "origin_booking_id",
    "n_prior_bookings",
    "target_booking_id",
    "target_booking_timestamp",
    "target_usage_timestamp",
    "target_facility_id",
    "target_usage_weekday",
    "target_usage_hour",
    "notification_delay_minutes",
)

TIMESTAMP_COLUMNS = (
    "origin",
    "target_booking_timestamp",
    "target_usage_timestamp",
)

_SECONDS_IN_MINUTE = 60.0


@dataclasses.dataclass(frozen=True)
class SampleCounts:
    """What the sampler kept, and what it could not use.

    Attributes:
        residents_configured: Residents the configuration asked for.
        residents_with_bookings: Residents that appear in the dataset.
        residents_without_bookings: Residents that made no booking at
            all, and so appear nowhere in the dataset.
        residents_excluded_no_prior_history: Residents that booked, but
            never accumulated enough prior bookings to form a sample.
        residents_with_samples: Residents contributing at least one
            sample.
        bookings: Rows in the source booking table.
        samples: Rows in the sample table.
        censored_bookings: Bookings whose successor was never observed —
            one per resident with at least one booking.
        observation_start: Earliest booking creation time observed.
        observation_end: Latest usage time observed.
        min_prior_bookings: Prior bookings a target booking needed.
    """

    residents_configured: int
    residents_with_bookings: int
    residents_without_bookings: int
    residents_excluded_no_prior_history: int
    residents_with_samples: int
    bookings: int
    samples: int
    censored_bookings: int
    observation_start: str
    observation_end: str
    min_prior_bookings: int

    def to_dict(self) -> dict[str, Any]:
        """Return the counts as a JSON-serialisable mapping."""
        return dataclasses.asdict(self)


def _check_bookings(bookings: pd.DataFrame) -> None:
    """Reject a booking table the sampler cannot read safely.

    Args:
        bookings: Candidate booking table.

    Raises:
        ValueError: If a contract column is missing or a timestamp
            column is timezone-naive.
    """
    missing = [
        column
        for column in generate.BOOKING_COLUMNS
        if column not in bookings.columns
    ]
    if missing:
        msg = f"bookings is missing required columns: {missing}"
        raise ValueError(msg)

    for column in ("booking_timestamp", "usage_timestamp"):
        if not isinstance(bookings[column].dtype, pd.DatetimeTZDtype):
            msg = (
                f"{column} must be timezone-aware, got dtype "
                f"{bookings[column].dtype}"
            )
            raise ValueError(msg)


def _check_ordering(ordered: pd.DataFrame) -> None:
    """Reject bookings whose within-resident order is ambiguous.

    Two bookings created by one resident at the same instant leave no
    defensible choice of origin, so the sampler stops.

    Args:
        ordered: Bookings sorted by resident then booking timestamp.

    Raises:
        ValueError: If a resident has two bookings sharing a creation
            time.
    """
    grouped = ordered.groupby("resident_id", sort=False)
    previous = grouped["booking_timestamp"].shift(1)
    tied = ordered["booking_timestamp"].eq(previous)
    if bool(tied.any()):
        residents = sorted(set(ordered.loc[tied, "resident_id"]))
        msg = (
            "booking timestamps must strictly increase within a "
            f"resident; ties found for {residents[:5]}"
        )
        raise ValueError(msg)


def _origin_anchored_frame(
    ordered: pd.DataFrame, min_prior_bookings: int
) -> pd.DataFrame:
    """Attach each booking's origin and keep the eligible rows.

    Leakage contract: the origin is the resident's immediately previous
    booking creation time, so every origin is strictly earlier than the
    target booking it anchors.

    Args:
        ordered: Bookings sorted by resident then booking timestamp.
        min_prior_bookings: Prior bookings a target booking needs.

    Returns:
        One row per eligible target booking, carrying its origin,
        origin booking id, and prior-booking count.
    """
    grouped = ordered.groupby("resident_id", sort=False)
    n_prior = grouped.cumcount()
    frame = pd.DataFrame(
        {
            "resident_id": ordered["resident_id"],
            "origin": grouped["booking_timestamp"].shift(1),
            "origin_booking_id": grouped["booking_id"].shift(1),
            "n_prior_bookings": n_prior,
            "target_booking_id": ordered["booking_id"],
            "target_booking_timestamp": ordered["booking_timestamp"],
            "target_usage_timestamp": ordered["usage_timestamp"],
            "target_facility_id": ordered["facility_id"],
        }
    )
    return frame.loc[n_prior >= min_prior_bookings].reset_index(drop=True)


def _add_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the four labels from the target booking.

    Args:
        frame: Origin-anchored rows from :func:`_origin_anchored_frame`.

    Returns:
        The same rows with the weekday, hour, and delay labels added.

    Raises:
        ValueError: If any delay is not strictly positive, which would
            mean the origin did not precede the target booking.
    """
    labelled = frame.copy()
    usage = labelled["target_usage_timestamp"]
    labelled["target_usage_weekday"] = usage.dt.weekday
    labelled["target_usage_hour"] = usage.dt.hour
    gap = labelled["target_booking_timestamp"] - labelled["origin"]
    labelled["notification_delay_minutes"] = (
        gap.dt.total_seconds() / _SECONDS_IN_MINUTE
    )

    invalid = labelled["notification_delay_minutes"] <= 0
    if bool(invalid.any()):
        offenders = list(labelled.loc[invalid, "target_booking_id"][:5])
        msg = (
            "every origin must strictly precede its target booking; "
            f"non-positive delay for {offenders}"
        )
        raise ValueError(msg)
    return labelled


def build_samples(
    bookings: pd.DataFrame, config: config_module.Config
) -> pd.DataFrame:
    """Build one rolling-origin sample per eligible target booking.

    Leakage contract: reads the resident's booking sequence only to
    place the origin at ``b(i-1).booking_timestamp``, and the target
    booking only to write its labels. Every origin satisfies
    ``origin < target_booking_timestamp``.

    Args:
        bookings: The booking table, with timezone-aware timestamps.
        config: Validated configuration; supplies the minimum number of
            prior bookings a target needs.

    Returns:
        Samples ordered by ``(origin, target_booking_id)``, with a
        stable ``sample_id`` assigned after that sort.

    Raises:
        ValueError: If the configured minimum is below one, a contract
            column is missing, a timestamp is naive, a resident has tied
            booking times, or an origin fails to precede its target.
    """
    min_prior_bookings = config.evaluation.min_prior_bookings
    if min_prior_bookings < 1:
        msg = (
            "min_prior_bookings must be at least 1: a sample without a "
            f"prior booking has no origin, got {min_prior_bookings}"
        )
        raise ValueError(msg)

    _check_bookings(bookings)
    ordered = bookings.sort_values(
        ["resident_id", "booking_timestamp", "booking_id"], kind="mergesort"
    ).reset_index(drop=True)
    _check_ordering(ordered)

    frame = _add_labels(_origin_anchored_frame(ordered, min_prior_bookings))
    frame = frame.sort_values(
        ["origin", "target_booking_id"], kind="mergesort"
    ).reset_index(drop=True)
    frame.insert(
        0,
        "sample_id",
        [f"S{index + 1:07d}" for index in range(len(frame))],
    )
    return frame[list(SAMPLE_COLUMNS)]


def summarise_samples(
    bookings: pd.DataFrame,
    samples: pd.DataFrame,
    config: config_module.Config,
) -> SampleCounts:
    """Count what became a sample and what could not.

    Args:
        bookings: The booking table the samples were built from.
        samples: The frame returned by :func:`build_samples`.
        config: Validated configuration; supplies the roster size and
            the minimum prior-booking threshold.

    Returns:
        The reconciled :class:`SampleCounts`.
    """
    per_resident = bookings.groupby("resident_id", sort=False).size()
    min_prior_bookings = config.evaluation.min_prior_bookings
    residents_with_bookings = len(per_resident)
    residents_with_samples = int(samples["resident_id"].nunique())

    return SampleCounts(
        residents_configured=config.community.residents,
        residents_with_bookings=residents_with_bookings,
        residents_without_bookings=(
            config.community.residents - residents_with_bookings
        ),
        residents_excluded_no_prior_history=int(
            (per_resident <= min_prior_bookings).sum()
        ),
        residents_with_samples=residents_with_samples,
        bookings=len(bookings),
        samples=len(samples),
        censored_bookings=residents_with_bookings,
        observation_start=bookings["booking_timestamp"].min().isoformat(),
        observation_end=bookings["usage_timestamp"].max().isoformat(),
        min_prior_bookings=min_prior_bookings,
    )


def samples_digest(frame: pd.DataFrame) -> str:
    """Return the canonical digest of a sample table.

    Args:
        frame: A frame carrying :data:`SAMPLE_COLUMNS`.

    Returns:
        The hex SHA-256 over the canonically rendered rows.
    """
    return digest_module.canonical_digest(
        frame, sort_by=("sample_id",), columns=SAMPLE_COLUMNS
    )


def load_bookings(
    path: pathlib.Path, config: config_module.Config
) -> pd.DataFrame:
    """Read an exported booking CSV back as a timezone-aware frame.

    The pipeline reads bookings from the store; this reads the exported
    file, for a reviewer checking the submitted dataset on its own.

    Args:
        path: The CSV written by the generator.
        config: Validated configuration; supplies the timezone the
            timestamps are converted back into.

    Returns:
        The booking table with both timestamp columns timezone-aware.
    """
    frame = pd.read_csv(path)
    for column in ("booking_timestamp", "usage_timestamp"):
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(config.tzinfo)
    return frame


def write_samples(frame: pd.DataFrame, path: pathlib.Path) -> str:
    """Write the sample table and return its content hash.

    Timestamps are written in ISO 8601 with the offset, so the file
    round-trips timezone-aware.

    Args:
        frame: A frame as returned by :func:`build_samples`.
        path: Destination CSV; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    output = frame.copy()
    for column in TIMESTAMP_COLUMNS:
        output[column] = output[column].map(lambda value: value.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _LOGGER.info("wrote %s (sha256=%s)", path, digest)
    return digest


def write_summary(
    counts: SampleCounts,
    provenance: dict[str, Any],
    path: pathlib.Path,
) -> None:
    """Write the counts beside the provenance that produced them.

    Args:
        counts: The reconciled counts.
        provenance: Seed, timezone, and input/output hashes.
        path: Destination JSON; parent directories are created.
    """
    payload = {"provenance": provenance, "counts": counts.to_dict()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
