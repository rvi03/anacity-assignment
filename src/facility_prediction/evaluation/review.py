"""The prediction review workbook — one file, every track.

A reviewer should be able to open one spreadsheet and see, row by row,
what was predicted, what actually happened, and which of the four
components matched. Three sheets do that:

    Predictions   one row per (track, model, scored record)
    Summary       split shape, per-model metrics, provenance, and the
                  branch status of each track
    Definitions   what every column means, in words

Both tracks write here. Rows carry a ``track`` column and are rendered
from the shared prediction table, so neither track can drop the
other's rows by rebuilding the file.

**Which rows appear.** The holdout split is sealed until the one scoring
pass each track is allowed, so this workbook renders the rows whose
targets may currently be read. The Summary states which split that was
and how many rows it held, rather than leaving a reader to assume the
table is the holdout.

Leakage contract: this module reads targets of rows that have already
been predicted, and it is handed the frame to render. Nothing it
computes returns to a feature, a fit, or a prediction. It applies no
as-of bound of its own; the caller decides which split it may see, and
passing it holdout rows before that split is unsealed is the caller's
defect, not something this module can detect.
"""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.evaluation import evaluate

PREDICTIONS_SHEET = "Predictions"
SUMMARY_SHEET = "Summary"
DEFINITIONS_SHEET = "Definitions"

MATCH_TRUE = "✓"
MATCH_FALSE = "✗"

_MINUTES_IN_HOUR = 60.0
_TIMESTAMP_FORMAT = "%a %Y-%m-%d %H:%M"
HISTORY_SEPARATOR = " ⏎ "

PREDICTION_SHEET_COLUMNS = (
    "track",
    "model",
    "record",
    "resident",
    "origin",
    "past_bookings",
    "predicted_facility",
    "predicted_usage_weekday",
    "predicted_usage_hour",
    "predicted_booking_time",
    "suggested_send",
    "actual_facility",
    "actual_usage_weekday",
    "actual_usage_hour",
    "actual_booked",
    "match_facility",
    "match_usage_weekday",
    "match_usage_hour",
    "match_notification",
    "score",
    "in_comparison_manifest",
    "cold_start",
    "semantically_valid",
    "circular_hour_error",
    "predicted_delay_minutes_raw",
    "predicted_delay_minutes_submitted",
    "delay_clamped",
    "notification_abs_error_minutes",
    "notification_log_ratio",
)

COLUMN_DEFINITIONS = {
    "track": "Which system produced the row: baseline, traditional, or llm.",
    "model": "The named model inside that track.",
    "record": "The booking being predicted. Never given to any model.",
    "resident": "Whose history the prediction was made from.",
    "origin": (
        "The moment the prediction was made: the creation time of the "
        "resident's previous booking. Nothing after it was readable."
    ),
    "past_bookings": (
        "The most recent prior bookings, newest first, as "
        "facility / usage weekday / usage time / booked-at. Display only; "
        "it never affects a feature, a prediction, or a metric."
    ),
    "predicted_booking_time": (
        "Origin plus the predicted notification delay — when the model "
        "expects the next booking to be created."
    ),
    "suggested_send": (
        "The predicted booking time minus the configured notification "
        "lead. Display only; it is never scored."
    ),
    "actual_booked": "When the next booking was actually created.",
    "match_notification": (
        "Whether the submitted delay falls inside the symmetric "
        "multiplicative tolerance around the actual delay."
    ),
    "score": "How many of the four components matched, 0 to 4.",
    "in_comparison_manifest": (
        "Whether this row is in the seeded holdout subset the tracks are "
        "compared on. Rows outside it still count in a single track's own "
        "metrics."
    ),
    "cold_start": (
        "Whether the resident had only the minimum history required for a "
        "sample at this origin."
    ),
    "semantically_valid": (
        "Whether the predicted facility is actually open at the predicted "
        "hour. An invalid combination is reported, never corrected."
    ),
    "circular_hour_error": (
        "Hours between predicted and actual usage hour, measured around "
        "the clock, so 23:00 and 01:00 are two hours apart."
    ),
    "predicted_delay_minutes_raw": (
        "The model's delay output before any clamp."
    ),
    "predicted_delay_minutes_submitted": (
        "The delay after clamping negatives to zero — the number the "
        "match uses, because it is what an operator would have acted on."
    ),
    "delay_clamped": "Whether the raw delay was negative and was clamped.",
    "notification_log_ratio": (
        "log((predicted+1)/(actual+1)). Zero is exact; the match "
        "threshold is the log of the configured ratio."
    ),
}


@dataclasses.dataclass(frozen=True)
class Coverage:
    """Which rows the workbook was built from, and why those.

    Attributes:
        split: Name of the split rendered.
        rows: Rows rendered, across every track.
        reason: Plain-English statement of why this split and not
            another, written into the Summary so a reader never has to
            infer it.
    """

    split: str
    rows: int
    reason: str


def _format_timestamp(value: pd.Timestamp | None) -> str:
    """Render one instant for display, or blank when there is none.

    Args:
        value: A timezone-aware instant, or a missing value.

    Returns:
        The formatted instant, or an empty string.
    """
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime(_TIMESTAMP_FORMAT)


def _weekday_name(value: float | int | None) -> str:
    """Render a weekday index as its short name.

    Args:
        value: Weekday index where 0 is Monday, or a missing value.

    Returns:
        The three-letter name, or an empty string.
    """
    if value is None or pd.isna(value):
        return ""
    return calendar.day_abbr[int(value)]


def _hour_label(value: float | int | None) -> str:
    """Render an hour of day as a clock time.

    Args:
        value: Hour in ``[0, 24)``, or a missing value.

    Returns:
        The hour as ``HH:00``, or an empty string.
    """
    if value is None or pd.isna(value):
        return ""
    return f"{int(value):02d}:00"


def _mark(value: bool) -> str:
    """Render a match as a tick or a cross.

    Args:
        value: Whether the component matched.

    Returns:
        The tick or cross glyph.
    """
    return MATCH_TRUE if bool(value) else MATCH_FALSE


def history_strings(
    samples: pd.DataFrame, bookings: pd.DataFrame, config: config_module.Config
) -> pd.Series:
    """Render each sample's most recent prior bookings for display.

    Leakage contract: reads only that resident's bookings with
    ``booking_timestamp <= origin``. The rendered text is display only
    and reaches no feature, model, or metric.

    Args:
        samples: Rows carrying ``sample_id``, ``resident_id``, and
            ``origin``.
        bookings: The full booking table.
        config: Validated configuration; supplies how many bookings to
            show.

    Returns:
        One string per sample, indexed by ``sample_id``, newest first.
    """
    depth = config.review.history_rows
    ordered = bookings.sort_values(
        ["resident_id", "booking_timestamp"], kind="mergesort"
    )
    by_resident = {
        str(resident): group
        for resident, group in ordered.groupby("resident_id", sort=False)
    }

    rendered = []
    for resident, origin in zip(
        samples["resident_id"], samples["origin"], strict=True
    ):
        history = by_resident[str(resident)]
        past = history.loc[history["booking_timestamp"] <= origin]
        recent = past.iloc[-depth:][::-1] if depth else past.iloc[:0]
        rendered.append(
            HISTORY_SEPARATOR.join(
                f"{row.facility_id} / "
                f"{_weekday_name(row.usage_timestamp.weekday())} / "
                f"{_hour_label(row.usage_timestamp.hour)} / "
                f"booked {_format_timestamp(row.booking_timestamp)}"
                for row in recent.itertuples()
            )
        )
    return pd.Series(rendered, index=samples["sample_id"], name="past_bookings")


def _semantic_validity(
    facilities: pd.Series, hours: pd.Series, config: config_module.Config
) -> pd.Series:
    """Return whether each facility/hour pair is inside opening hours.

    Args:
        facilities: Predicted facility names.
        hours: Predicted usage hours.
        config: Validated configuration; supplies the catalog's hours.

    Returns:
        A boolean series, False where the combination could not happen.
    """
    open_hours = {facility.name: facility for facility in config.facilities}
    return pd.Series(
        [
            (
                bool(open_hours[str(name)].is_open_at(int(hour)))
                if str(name) in open_hours and not pd.isna(hour)
                else False
            )
            for name, hour in zip(facilities, hours, strict=True)
        ],
        index=facilities.index,
    )


def build_predictions_sheet(
    joined: pd.DataFrame,
    history: pd.Series,
    comparison_ids: set[str],
    config: config_module.Config,
) -> pd.DataFrame:
    """Assemble one display row per scored record.

    Leakage contract: reads the targets of rows already predicted, to
    show what happened. Nothing computed here returns to a model.

    Args:
        joined: Predictions joined to their samples, carrying every
            target and predicted column plus ``origin``,
            ``resident_id``, ``target_booking_id``, ``track``,
            ``model``, and ``n_prior_bookings``.
        history: Rendered past-booking text indexed by ``sample_id``.
        comparison_ids: Sample ids in the frozen cross-track manifest.
        config: Validated configuration.

    Returns:
        The Predictions sheet, in :data:`PREDICTION_SHEET_COLUMNS` order.

    Raises:
        ValueError: If ``joined`` has no rows to render.
    """
    if joined.empty:
        msg = "cannot build a workbook from zero scored rows"
        raise ValueError(msg)

    matches = evaluate.component_matches(
        joined, config.evaluation.notification_match_ratio
    )
    submitted, _ = evaluate.clamp_delays(joined["predicted_delay_minutes"])
    raw = joined["predicted_delay_minutes"]
    actual_delay = joined["notification_delay_minutes"]
    lead = pd.Timedelta(minutes=config.application.notification_lead_minutes)
    predicted_booking = joined["origin"] + pd.to_timedelta(submitted, unit="m")

    sheet = pd.DataFrame(
        {
            "track": joined["track"].astype(str),
            "model": joined["model"].astype(str),
            "record": joined["target_booking_id"].astype(str),
            "resident": joined["resident_id"].astype(str),
            "origin": joined["origin"].map(_format_timestamp),
            "past_bookings": joined["sample_id"].map(history),
            "predicted_facility": joined["predicted_facility_id"].astype(str),
            "predicted_usage_weekday": joined["predicted_usage_weekday"].map(
                _weekday_name
            ),
            "predicted_usage_hour": joined["predicted_usage_hour"].map(
                _hour_label
            ),
            "predicted_booking_time": predicted_booking.map(_format_timestamp),
            "suggested_send": (predicted_booking - lead).map(_format_timestamp),
            "actual_facility": joined["target_facility_id"].astype(str),
            "actual_usage_weekday": joined["target_usage_weekday"].map(
                _weekday_name
            ),
            "actual_usage_hour": joined["target_usage_hour"].map(_hour_label),
            "actual_booked": joined["target_booking_timestamp"].map(
                _format_timestamp
            ),
            "match_facility": matches[evaluate.FACILITY].map(_mark),
            "match_usage_weekday": matches[evaluate.USAGE_WEEKDAY].map(_mark),
            "match_usage_hour": matches[evaluate.USAGE_HOUR].map(_mark),
            "match_notification": matches[evaluate.NOTIFICATION].map(_mark),
            "score": evaluate.score_rows(matches),
            "in_comparison_manifest": joined["sample_id"].isin(comparison_ids),
            "cold_start": joined["n_prior_bookings"]
            <= config.evaluation.min_prior_bookings,
            "semantically_valid": _semantic_validity(
                joined["predicted_facility_id"],
                joined["predicted_usage_hour"],
                config,
            ),
            "circular_hour_error": evaluate.circular_hour_error(
                joined["target_usage_hour"], joined["predicted_usage_hour"]
            ),
            "predicted_delay_minutes_raw": raw.round(1),
            "predicted_delay_minutes_submitted": submitted.round(1),
            "delay_clamped": raw < 0,
            "notification_abs_error_minutes": (
                (submitted - actual_delay).abs().round(1)
            ),
            "notification_log_ratio": np.log(
                (submitted + 1.0) / (actual_delay + 1.0)
            ).round(4),
        }
    )
    return sheet[list(PREDICTION_SHEET_COLUMNS)].reset_index(drop=True)


def _split_shape(
    joined: pd.DataFrame, coverage: Coverage
) -> list[tuple[str, Any]]:
    """Describe the rendered rows as Summary key/value pairs.

    Args:
        joined: The scored rows the workbook was built from.
        coverage: Which split was rendered and why.

    Returns:
        Ordered key/value pairs.
    """
    return [
        ("Rows rendered", coverage.rows),
        ("Split rendered", coverage.split),
        ("Why this split", coverage.reason),
        ("Records", int(joined["target_booking_id"].nunique())),
        ("Residents", int(joined["resident_id"].nunique())),
        ("First origin", _format_timestamp(joined["origin"].min())),
        ("Last origin", _format_timestamp(joined["origin"].max())),
    ]


def _model_metrics(
    joined: pd.DataFrame, config: config_module.Config
) -> list[tuple[str, Any]]:
    """Score every (track, model) present and flatten it for the sheet.

    Args:
        joined: The scored rows the workbook was built from.
        config: Validated configuration; supplies the tolerances.

    Returns:
        Ordered key/value pairs, one block per model.
    """
    rows: list[tuple[str, Any]] = []
    for (track, model), group in joined.groupby(["track", "model"], sort=True):
        result = evaluate.evaluate_predictions(
            group,
            config.evaluation.notification_match_ratio,
            config.evaluation.notification_support_minutes,
        )
        prefix = f"{track} / {model}"
        heads = result["heads"]
        summary = result["matches"]
        rows += [
            (f"{prefix} — rows", result["rows"]),
            (
                f"{prefix} — facility accuracy",
                round(heads[evaluate.FACILITY]["accuracy"], 4),
            ),
            (
                f"{prefix} — weekday accuracy",
                round(heads[evaluate.USAGE_WEEKDAY]["accuracy"], 4),
            ),
            (
                f"{prefix} — hour accuracy",
                round(heads[evaluate.USAGE_HOUR]["accuracy"], 4),
            ),
            (
                f"{prefix} — delay MAE (minutes)",
                round(
                    heads[evaluate.NOTIFICATION]["submitted"]["mae_minutes"],
                    1,
                ),
            ),
            (
                f"{prefix} — hour within 1 hour",
                round(heads[evaluate.USAGE_HOUR]["within_1_hour"], 4),
            ),
            (
                f"{prefix} — delay clamp rate",
                round(heads[evaluate.NOTIFICATION]["clamp_rate"], 4),
            ),
            (f"{prefix} — SCORE of 4", round(summary["score_mean"], 4)),
            (f"{prefix} — OVERALL", round(summary["overall"], 4)),
            (f"{prefix} — STRICT", round(summary["strict"], 4)),
            (
                f"{prefix} — at least 3 of 4",
                round(summary["at_least_three"], 4),
            ),
        ]
    return rows


def track_states(predictions: pd.DataFrame) -> dict[str, str]:
    """Describe each track from the rows it actually wrote.

    Reading the state off the stored rows keeps the Summary from
    claiming a track is absent after it has written, or present before
    it has. A hand-written status line goes stale the moment a track
    advances; this one cannot.

    Args:
        predictions: Every stored prediction row, of every track.

    Returns:
        Summary key/value pairs naming what each track contributed.
    """
    written = {
        str(track): sorted({str(name) for name in group})
        for track, group in predictions.groupby("track")["model"]
    }
    absent = "no rows written yet"
    return {
        "Baseline": ", ".join(written.get("baseline", [])) or absent,
        "Traditional track": ", ".join(written.get("traditional", []))
        or absent,
        "LLM track": ", ".join(written.get("llm", [])) or absent,
        "Evidence state": (
            "measured — every number here was computed from saved predictions."
        ),
    }


def build_summary_sheet(
    joined: pd.DataFrame,
    coverage: Coverage,
    provenance: dict[str, Any],
    status: dict[str, str],
    config: config_module.Config,
    slices: Sequence[Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Assemble the Summary sheet.

    Args:
        joined: The scored rows the workbook was built from.
        coverage: Which split was rendered and why.
        provenance: Seed, digests, and versions.
        status: Each track's branch status and evidence state.
        config: Validated configuration.
        slices: Error-analysis slices, when they have been computed for
            the split being rendered.

    Returns:
        A two-column key/value sheet.
    """
    rows: list[tuple[str, Any]] = [("— Coverage —", "")]
    rows += _split_shape(joined, coverage)
    rows += [("", ""), ("— Metrics —", "")]
    rows += _model_metrics(joined, config)
    rows += [("", ""), ("— Settings —", "")]
    rows += [
        ("Seed", config.seed),
        ("Timezone", config.timezone),
        (
            "Notification match rule",
            f"within x{config.evaluation.notification_match_ratio} "
            "of the actual delay, both directions",
        ),
        (
            "Suggested-send lead (minutes, unscored)",
            config.application.notification_lead_minutes,
        ),
        ("Past-bookings depth shown", config.review.history_rows),
    ]
    if slices:
        rows += [("", ""), ("— Where it fails —", "")]
        rows += _slice_rows(slices)
    rows += [("", ""), ("— Status —", "")]
    rows += sorted(status.items())
    rows += [("", ""), ("— Provenance —", "")]
    rows += sorted(provenance.items())
    rows += [
        ("Python", sys.version.split()[0]),
        ("pandas", pd.__version__),
        ("numpy", np.__version__),
    ]
    return pd.DataFrame(rows, columns=["Item", "Value"])


def _slice_rows(
    slices: Sequence[Mapping[str, Any]],
) -> list[tuple[str, Any]]:
    """Render the error-analysis slices as Summary key/value pairs.

    Only the overall rate and the row count are shown. A reviewer who
    wants the per-component breakdown has `error_analysis.json`, and
    repeating it here would make the sheet unreadable without making it
    more complete.

    Args:
        slices: Rows from the error analysis.

    Returns:
        One pair per slice, grouped by dimension.
    """
    rendered: list[tuple[str, Any]] = []
    for dimension in dict.fromkeys(entry["dimension"] for entry in slices):
        for entry in slices:
            if entry["dimension"] != dimension:
                continue
            rendered.append(
                (
                    f"{dimension} / {entry['slice']}",
                    f"{entry['overall']:.4f} over {entry['rows']} rows",
                )
            )
    return rendered


def build_definitions_sheet(config: config_module.Config) -> pd.DataFrame:
    """Assemble the Definitions sheet.

    Args:
        config: Validated configuration; its tolerances are quoted so
            the sheet explains itself without the config file.

    Returns:
        A two-column sheet describing every displayed column and the
        conventions behind them.
    """
    rows = [
        (
            "Origin",
            "The creation time of the resident's previous booking. Every "
            "feature and prediction reads only events at or before it.",
        ),
        (
            "Inter-booking interval",
            "Booking creation to booking creation — the same quantity the "
            "notification output predicts. Usage-to-usage gaps are a "
            "different number and are named separately.",
        ),
        (
            "Notification match",
            "The submitted delay matches when it is within "
            f"x{config.evaluation.notification_match_ratio} of the actual "
            "delay in either direction, measured on log ratio.",
        ),
        (
            "Supporting tolerances",
            "Absolute minute windows reported beside the match rate as "
            "diagnostics: "
            f"{list(config.evaluation.notification_support_minutes)}. They "
            "never select a model.",
        ),
        (
            "Predicted booking time vs suggested send",
            "The first is when the booking is expected to be created; the "
            "second subtracts the configured lead so a notification "
            "arrives before it. Only the first is scored.",
        ),
        (
            "Rolling origin",
            "Each booking with enough prior history becomes one prediction "
            "problem, anchored on the previous booking rather than on a "
            "calendar grid.",
        ),
        (
            "Clamping",
            "A negative predicted delay is submitted as zero and flagged. "
            "The raw value is shown beside it, so clamping cannot hide an "
            "unstable output.",
        ),
    ]
    rows += sorted(COLUMN_DEFINITIONS.items())
    return pd.DataFrame(rows, columns=["Term", "Meaning"])


def write_workbook(sheets: dict[str, pd.DataFrame], path: pathlib.Path) -> None:
    """Write the sheets into one workbook.

    Args:
        sheets: Sheet name to frame, written in the given order.
        path: Destination ``.xlsx``; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)


def write_csv(frame: pd.DataFrame, path: pathlib.Path) -> str:
    """Write the prediction rows as CSV and return their content hash.

    Args:
        frame: The Predictions sheet.
        path: Destination CSV; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()
