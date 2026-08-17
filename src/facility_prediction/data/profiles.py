"""Generator diagnosis in depth: quantiles, intervals, and plots.

The acceptance checks answer one question per property — is it present?
That is enough to accept a dataset and not enough to understand one. A
check that reports `popularity_ranking: pass` hides how close the run
came to failing, and a median lead time reported as a single number
hides whether the distribution behind it is the right shape.

This module reports the distributions the checks reduce to booleans:

    realised vs configured   every facility, both shares, the signed
                             deviation, and a bootstrap interval on the
                             realised one — so "close enough" is a
                             measured distance, not an impression
    lead-time quantiles      p05 … p95 overall and per facility, plus
                             the skew the p50 check assumes exists
    per-resident activity    the booking-count distribution the sparsity
                             check reduces to a single share
    drift over time          monthly facility shares, so the two dated
                             events are visible as shapes rather than
                             as two passing assertions

Every interval is a seeded bootstrap: the same dataset and the same seed
give the same bounds, so a reported interval is reproducible evidence
rather than a number that moves when the file is read again.

Leakage contract: this module reads the generated booking table only. It
sees no sample, no split, no feature, and no model output, so nothing it
computes can reach a fit. It is diagnosis of the data, run beside the
pipeline and never inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
import logging
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from facility_prediction import config as config_module

_LOGGER = logging.getLogger(__name__)

PROFILE_FILENAME = "generation_profile.json"
PLOT_FILENAMES = (
    "popularity_realised_vs_configured.png",
    "lead_time_distribution.png",
    "bookings_per_resident.png",
    "facility_share_by_month.png",
)

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_HOURS_PER_MINUTE = 1.0 / 60.0
# Shares are rounded to four places before they are summed, so an
# exact 1.0 is not available to compare against.
_SHARE_SUM_TOLERANCE = 0.01


class ProfileError(Exception):
    """Raised when a profile cannot be built from what it was given."""


def _lead_hours(frame: pd.DataFrame) -> npt.NDArray[np.float64]:
    """Return every booking's lead time in hours.

    Args:
        frame: The booking table.

    Returns:
        One lead time per row, in hours.
    """
    delta = frame["usage_timestamp"] - frame["booking_timestamp"]
    return (
        delta.dt.total_seconds().to_numpy(dtype=np.float64) / 60.0
    ) * _HOURS_PER_MINUTE


def bootstrap_interval(
    values: npt.ArrayLike,
    seed: int,
    resamples: int = 2000,
    level: float = 0.95,
) -> tuple[float, float]:
    """Return a seeded bootstrap interval for a sample mean.

    The interval is percentile-based and its generator is seeded from
    configuration, so re-running reports the identical bounds. An
    interval that moved between runs would be a description of the
    random draw rather than of the data.

    Args:
        values: The observations to resample.
        seed: Seed for the resampling generator.
        resamples: How many resamples to draw.
        level: Coverage of the reported interval.

    Returns:
        The lower and upper bound of the interval.

    Raises:
        ProfileError: If there is nothing to resample.
    """
    sample = np.asarray(values, dtype=np.float64)
    if not sample.size:
        msg = "cannot bootstrap an empty sample"
        raise ProfileError(msg)
    generator = np.random.default_rng(seed)
    draws = generator.choice(
        sample, size=(resamples, sample.size), replace=True
    ).mean(axis=1)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return float(lower), float(upper)


def quantiles(values: npt.ArrayLike) -> dict[str, float]:
    """Return the declared quantiles of a sample.

    Args:
        values: The observations.

    Returns:
        One entry per declared quantile, keyed ``p05``-style.

    Raises:
        ProfileError: If there is nothing to summarise.
    """
    sample = np.asarray(values, dtype=np.float64)
    if not sample.size:
        msg = "cannot take quantiles of an empty sample"
        raise ProfileError(msg)
    computed = np.quantile(sample, _QUANTILES)
    return {
        f"p{round(q * 100):02d}": round(float(value), 4)
        for q, value in zip(_QUANTILES, computed, strict=True)
    }


def popularity_profile(
    frame: pd.DataFrame, config: config_module.Config
) -> list[dict[str, Any]]:
    """Return realised against configured popularity, facility by facility.

    The acceptance check compares the top three by rank. This reports
    every facility, the signed deviation, and an interval on the
    realised share, which is what says whether a deviation is a finding
    or a sample size.

    Args:
        frame: The booking table.
        config: Validated configuration naming the configured shares.

    Returns:
        One row per facility, ordered by configured share descending.
    """
    total = len(frame)
    counts = frame["facility_id"].value_counts()
    rows: list[dict[str, Any]] = []
    for facility in sorted(
        config.facilities, key=lambda item: -item.popularity
    ):
        indicator = (frame["facility_id"] == facility.name).to_numpy(
            dtype=np.float64
        )
        realised = float(counts.get(facility.name, 0)) / total
        lower, upper = bootstrap_interval(indicator, seed=config.seed)
        rows.append(
            {
                "facility": facility.name,
                "configured": round(facility.popularity, 4),
                "realised": round(realised, 4),
                "deviation": round(realised - facility.popularity, 4),
                "realised_interval": [round(lower, 4), round(upper, 4)],
                "bookings": int(counts.get(facility.name, 0)),
                "available_from_month": facility.available_from_month,
            }
        )
    return rows


def lead_time_profile(
    frame: pd.DataFrame, config: config_module.Config
) -> dict[str, Any]:
    """Return the lead-time distribution overall and per facility.

    Args:
        frame: The booking table.
        config: Validated configuration; supplies the bootstrap seed.

    Returns:
        Overall quantiles, the mean and its interval, the skew the p50
        check assumes, and per-facility quantiles.
    """
    hours = _lead_hours(frame)
    lower, upper = bootstrap_interval(hours, seed=config.seed)
    per_facility = {}
    for facility, group in frame.groupby("facility_id", sort=True):
        per_facility[str(facility)] = quantiles(_lead_hours(group))
    return {
        "overall": quantiles(hours),
        "mean": round(float(hours.mean()), 4),
        "mean_interval": [round(lower, 4), round(upper, 4)],
        # A right-skewed distribution has its mean above its median.
        # Reporting the ratio makes the assumption falsifiable.
        "mean_over_median": round(float(hours.mean() / np.median(hours)), 4),
        "per_facility": per_facility,
    }


def activity_profile(
    frame: pd.DataFrame, config: config_module.Config
) -> dict[str, Any]:
    """Return the per-resident booking-count distribution.

    Args:
        frame: The booking table.
        config: Validated configuration naming the configured resident
            count, so residents with no booking at all are counted
            rather than silently absent.

    Returns:
        Quantiles of the booking count, and the sparse share the
        acceptance check reduces to one number.
    """
    counts = frame["resident_id"].value_counts()
    complete = np.zeros(config.community.residents, dtype=np.float64)
    complete[: len(counts)] = counts.to_numpy(dtype=np.float64)
    threshold = config.generator.acceptance.sparse_resident_bookings
    return {
        "residents_configured": config.community.residents,
        "residents_with_bookings": int((complete > 0).sum()),
        "bookings_per_resident": quantiles(complete),
        "sparse_threshold_bookings": threshold,
        "sparse_share": round(float((complete < threshold).mean()), 4),
    }


def monthly_share_profile(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return each facility's share of bookings, month by month.

    The two dated drift events are assertions elsewhere. Here they are a
    shape: a facility that opens mid-series starts at zero, and one that
    loses demand to it declines from the same month.

    Args:
        frame: The booking table.

    Returns:
        One row per month, carrying every facility's share that month.
    """
    # Periods carry no zone, so the conversion is done on the local
    # wall clock deliberately: a booking belongs to the month a
    # resident experienced, not to the month it fell in at UTC.
    months = frame["usage_timestamp"].dt.tz_localize(None).dt.to_period("M")
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby(months, sort=True):
        shares = group["facility_id"].value_counts(normalize=True)
        rows.append(
            {
                "month": str(month),
                "bookings": len(group),
                "shares": {
                    str(name): round(float(value), 4)
                    for name, value in sorted(shares.items())
                },
            }
        )
    return rows


def build_profile(
    frame: pd.DataFrame, config: config_module.Config
) -> dict[str, Any]:
    """Assemble the whole rigour profile.

    Args:
        frame: The booking table.
        config: Validated configuration.

    Returns:
        The profile payload, ready to serialise.

    Raises:
        ProfileError: If the table is empty.
    """
    if frame.empty:
        msg = "cannot profile an empty booking table"
        raise ProfileError(msg)
    return {
        "provenance": {
            "seed": config.seed,
            "timezone": config.timezone,
            "bookings": len(frame),
        },
        "popularity": popularity_profile(frame, config),
        "lead_time_hours": lead_time_profile(frame, config),
        "activity": activity_profile(frame, config),
        "monthly_shares": monthly_share_profile(frame),
    }


def write_profile(payload: dict[str, Any], path: pathlib.Path) -> str:
    """Write the profile and return its content hash.

    Args:
        payload: The profile payload.
        path: Destination JSON; parent directories are created.

    Returns:
        The hex SHA-256 of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_plots(
    frame: pd.DataFrame,
    profile: dict[str, Any],
    directory: pathlib.Path,
) -> tuple[pathlib.Path, ...]:
    """Render the profile as figures beside its JSON.

    Plots are imported lazily and drawn on a non-interactive backend, so
    importing this module never opens a window or requires a display.

    Args:
        frame: The booking table.
        profile: The payload from :func:`build_profile`.
        directory: Destination directory; created if absent.

    Returns:
        The paths written, in :data:`PLOT_FILENAMES` order.
    """
    # Imported here, not at module scope: the backend must be selected
    # before pyplot is first imported, and a module-level import would
    # make every caller of this file pay for a plotting stack it does
    # not use.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    directory.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []

    popularity = profile["popularity"]
    names = [row["facility"] for row in popularity]
    figure, axes = plt.subplots(figsize=(9, 4.5))
    positions = np.arange(len(names))
    axes.bar(
        positions - 0.2,
        [row["configured"] for row in popularity],
        width=0.4,
        label="configured",
    )
    axes.bar(
        positions + 0.2,
        [row["realised"] for row in popularity],
        width=0.4,
        label="realised",
    )
    axes.set_xticks(positions, names, rotation=30, ha="right")
    axes.set_ylabel("share of bookings")
    axes.set_title("Realised against configured popularity")
    axes.legend()
    written.append(_save(figure, directory / PLOT_FILENAMES[0], plt))

    figure, axes = plt.subplots(figsize=(9, 4.5))
    axes.hist(_lead_hours(frame), bins=60)
    axes.set_xscale("log")
    axes.set_xlabel("lead time (hours, log scale)")
    axes.set_ylabel("bookings")
    axes.set_title("Booking lead time")
    written.append(_save(figure, directory / PLOT_FILENAMES[1], plt))

    figure, axes = plt.subplots(figsize=(9, 4.5))
    axes.hist(frame["resident_id"].value_counts().to_numpy(), bins=40)
    axes.set_xlabel("bookings per resident")
    axes.set_ylabel("residents")
    axes.set_title("Activity distribution")
    written.append(_save(figure, directory / PLOT_FILENAMES[2], plt))

    monthly = profile["monthly_shares"]
    # Months are plotted at integer positions with string labels. Handing
    # matplotlib the label strings directly makes it guess whether they
    # are numbers, dates, or categories, and it says so on every series.
    months = np.arange(len(monthly))
    figure, axes = plt.subplots(figsize=(10, 5))
    for facility in names:
        axes.plot(
            months,
            [row["shares"].get(facility, 0.0) for row in monthly],
            label=facility,
        )
    every = max(1, len(monthly) // 12)
    axes.set_xticks(
        months[::every],
        [row["month"] for row in monthly][::every],
        rotation=60,
    )
    axes.set_ylabel("share of that month's bookings")
    axes.set_title("Facility share by month")
    axes.legend(fontsize="small", ncol=2)
    written.append(_save(figure, directory / PLOT_FILENAMES[3], plt))

    return tuple(written)


def _save(figure: Any, path: pathlib.Path, plt: Any) -> pathlib.Path:
    """Write one figure and release it.

    Args:
        figure: The figure to write.
        path: Destination file.
        plt: The pyplot module, passed in so the import stays lazy.

    Returns:
        The path written.
    """
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return path


def check_profile(profile: dict[str, Any]) -> Sequence[str]:
    """Return every way a profile contradicts what it should describe.

    This is the rigour gate. It is deliberately about internal
    consistency rather than about thresholds: the thresholds are the
    acceptance checks' job, and restating them here would make one
    failure look like two.

    Args:
        profile: The payload from :func:`build_profile`.

    Returns:
        One message per problem found; empty when the profile holds
        together.
    """
    problems: list[str] = []

    shares = sum(row["realised"] for row in profile["popularity"])
    if abs(shares - 1.0) > _SHARE_SUM_TOLERANCE:
        problems.append(
            f"realised popularity shares sum to {shares:.4f}, not 1"
        )

    for row in profile["popularity"]:
        lower, upper = row["realised_interval"]
        if not lower <= row["realised"] <= upper:
            problems.append(
                f"{row['facility']}: realised share {row['realised']} sits "
                f"outside its own interval [{lower}, {upper}]"
            )

    lead = profile["lead_time_hours"]["overall"]
    ordered = [lead[key] for key in sorted(lead)]
    if ordered != sorted(ordered):
        problems.append("lead-time quantiles are not monotonic")

    if profile["lead_time_hours"]["mean_over_median"] <= 1.0:
        problems.append(
            "lead time is not right-skewed: its mean does not exceed its "
            "median, which the p50 acceptance check assumes"
        )

    activity = profile["activity"]
    if activity["residents_with_bookings"] > activity["residents_configured"]:
        problems.append("more residents booked than were configured")

    for row in profile["monthly_shares"]:
        total = sum(row["shares"].values())
        if abs(total - 1.0) > _SHARE_SUM_TOLERANCE:
            problems.append(
                f"{row['month']}: facility shares sum to {total:.4f}, not 1"
            )

    return problems
