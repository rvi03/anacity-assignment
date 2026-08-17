"""Tests the review workbook's contract.

Covers what a reviewer relies on: every displayed value recomputes from
the saved predictions, the history column never shows an event later
than the origin, an impossible facility/hour pair is flagged rather than
corrected, and the CSV carries the same rows as the sheet.
"""

from __future__ import annotations

import pathlib
import zoneinfo

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate
from facility_prediction.evaluation import evaluate, review

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


def stamp(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_convert(KOLKATA)


@pytest.fixture(scope="module")
def config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture
def bookings() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            (
                "B01",
                "R001",
                "Gym",
                "2024-01-01T08:00:00+05:30",
                "2024-01-02T07:00:00+05:30",
            ),
            (
                "B02",
                "R001",
                "Pool",
                "2024-01-03T08:00:00+05:30",
                "2024-01-04T18:00:00+05:30",
            ),
            (
                "B03",
                "R001",
                "Gym",
                "2024-01-11T08:00:00+05:30",
                "2024-01-12T09:00:00+05:30",
            ),
        ],
        columns=list(generate.BOOKING_COLUMNS),
    )
    for column in ("booking_timestamp", "usage_timestamp"):
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(KOLKATA)
    return frame


@pytest.fixture
def joined() -> pd.DataFrame:
    """One correct row and one wrong row, joined and ready to render."""
    return pd.DataFrame(
        {
            "track": ["baseline", "baseline"],
            "model": ["frequency_recency", "frequency_recency"],
            "sample_id": ["S0000001", "S0000002"],
            "resident_id": ["R001", "R001"],
            "origin": [
                stamp("2024-01-03T08:00:00+05:30"),
                stamp("2024-01-11T08:00:00+05:30"),
            ],
            "target_booking_id": ["B03", "B04"],
            "target_booking_timestamp": [
                stamp("2024-01-11T08:00:00+05:30"),
                stamp("2024-02-20T08:00:00+05:30"),
            ],
            "n_prior_bookings": [2, 3],
            "target_facility_id": ["Gym", "Pool"],
            "target_usage_weekday": [4, 1],
            "target_usage_hour": [9, 18],
            "notification_delay_minutes": [11520.0, 57600.0],
            "predicted_facility_id": ["Gym", "Yoga Room"],
            "predicted_usage_weekday": [4, 3],
            "predicted_usage_hour": [9, 23],
            "predicted_delay_minutes": [11520.0, -60.0],
        }
    )


@pytest.fixture
def history(joined, bookings, config) -> pd.Series:
    return review.history_strings(joined, bookings, config)


@pytest.fixture
def sheet(joined, history, config) -> pd.DataFrame:
    return review.build_predictions_sheet(joined, history, {"S0000002"}, config)


# --- the row a reviewer reads ----------------------------------------


def test_the_sheet_has_one_row_per_prediction(sheet, joined):
    assert len(sheet) == len(joined)
    assert list(sheet.columns) == list(review.PREDICTION_SHEET_COLUMNS)


def test_a_fully_correct_row_scores_four(sheet):
    first = sheet.iloc[0]

    assert first["score"] == 4
    assert first["match_facility"] == review.MATCH_TRUE
    assert first["match_notification"] == review.MATCH_TRUE


def test_score_equals_the_ticks_shown(sheet):
    marks = sheet[
        [
            "match_facility",
            "match_usage_weekday",
            "match_usage_hour",
            "match_notification",
        ]
    ]

    recomputed = marks.eq(review.MATCH_TRUE).sum(axis=1)
    assert list(recomputed) == list(sheet["score"])


def test_matches_agree_with_the_scorekeeper(sheet, joined, config):
    matches = evaluate.component_matches(
        joined, config.evaluation.notification_match_ratio
    )

    assert list(sheet["score"]) == list(evaluate.score_rows(matches))


def test_predicted_booking_time_is_origin_plus_the_submitted_delay(
    sheet, joined
):
    expected = joined["origin"].iloc[0] + pd.Timedelta(minutes=11520.0)

    assert sheet.iloc[0]["predicted_booking_time"] == expected.strftime(
        "%a %Y-%m-%d %H:%M"
    )


def test_suggested_send_subtracts_the_configured_lead(sheet, config):
    booked = pd.Timestamp(sheet.iloc[0]["predicted_booking_time"])
    send = pd.Timestamp(sheet.iloc[0]["suggested_send"])

    assert booked - send == pd.Timedelta(
        minutes=config.application.notification_lead_minutes
    )


# --- flags, never silent repairs -------------------------------------


def test_a_negative_delay_is_clamped_and_flagged(sheet):
    second = sheet.iloc[1]

    assert second["predicted_delay_minutes_raw"] == -60.0
    assert second["predicted_delay_minutes_submitted"] == 0.0
    assert bool(second["delay_clamped"])


def test_an_impossible_facility_hour_is_flagged_not_corrected(sheet):
    second = sheet.iloc[1]

    assert second["predicted_facility"] == "Yoga Room"
    assert second["predicted_usage_hour"] == "23:00"
    assert not bool(second["semantically_valid"])


def test_a_possible_facility_hour_passes(sheet):
    assert bool(sheet.iloc[0]["semantically_valid"])


def test_cold_start_marks_the_shortest_histories(sheet, config):
    assert list(sheet["cold_start"]) == [False, False]
    assert config.evaluation.min_prior_bookings == 1


def test_comparison_membership_comes_from_the_manifest(sheet):
    assert list(sheet["in_comparison_manifest"]) == [False, True]


# --- history is display only, and past only --------------------------


def test_history_never_shows_an_event_after_the_origin(history):
    assert "2024-01-11" not in history.iloc[0]
    assert "2024-01-03" in history.iloc[0]


def test_history_is_newest_first_and_config_deep(history, config):
    entries = history.iloc[1].split(" ⏎ ")

    assert len(entries) == config.review.history_rows
    assert entries[0].startswith("Gym")


def test_history_respects_a_zero_depth(joined, bookings, config):
    shallow = config.model_copy(
        update={"review": config.review.model_copy(update={"history_rows": 0})}
    )

    rendered = review.history_strings(joined, bookings, shallow)

    assert list(rendered) == ["", ""]


# --- the workbook itself ---------------------------------------------


def test_zero_rows_is_an_error_not_an_empty_workbook(history, config):
    with pytest.raises(ValueError, match="zero scored rows"):
        review.build_predictions_sheet(
            pd.DataFrame(columns=["track"]), history, set(), config
        )


def test_track_states_name_the_models_each_track_wrote():
    predictions = pd.DataFrame(
        {
            "track": ["baseline", "traditional", "traditional"],
            "model": ["frequency_recency", "catboost", "selected_heads"],
        }
    )

    states = review.track_states(predictions)

    assert states["Baseline"] == "frequency_recency"
    assert states["Traditional track"] == "catboost, selected_heads"


def test_a_track_that_wrote_nothing_is_reported_as_absent():
    predictions = pd.DataFrame(
        {"track": ["baseline"], "model": ["frequency_recency"]}
    )

    states = review.track_states(predictions)

    assert states["LLM track"] == "no rows written yet"
    assert states["Traditional track"] == "no rows written yet"


def test_summary_states_which_split_was_rendered(joined, config):
    coverage = review.Coverage(
        split="validation", rows=2, reason="holdout is still sealed"
    )

    summary = review.build_summary_sheet(joined, coverage, {}, {}, config)

    items = dict(zip(summary["Item"], summary["Value"], strict=True))
    assert items["Split rendered"] == "validation"
    assert items["Why this split"] == "holdout is still sealed"
    assert items["Rows rendered"] == 2


def test_summary_metrics_recompute_from_the_predictions(joined, config):
    coverage = review.Coverage(split="validation", rows=2, reason="test")

    summary = review.build_summary_sheet(joined, coverage, {}, {}, config)

    items = dict(zip(summary["Item"], summary["Value"], strict=True))
    scored = evaluate.evaluate_predictions(
        joined,
        config.evaluation.notification_match_ratio,
        config.evaluation.notification_support_minutes,
    )
    key = "baseline / frequency_recency — OVERALL"
    assert items[key] == round(scored["matches"]["overall"], 4)


def test_definitions_cover_every_displayed_column(config):
    definitions = review.build_definitions_sheet(config)

    described = set(definitions["Term"])
    assert set(review.COLUMN_DEFINITIONS) <= described


def test_the_workbook_has_the_three_sheets(sheet, config, tmp_path):
    path = tmp_path / "review.xlsx"
    coverage = review.Coverage(split="validation", rows=len(sheet), reason="t")

    review.write_workbook(
        {
            review.PREDICTIONS_SHEET: sheet,
            review.SUMMARY_SHEET: review.build_summary_sheet(
                _summary_source(), coverage, {}, {}, config
            ),
            review.DEFINITIONS_SHEET: review.build_definitions_sheet(config),
        },
        path,
    )

    assert pd.ExcelFile(path).sheet_names == [
        review.PREDICTIONS_SHEET,
        review.SUMMARY_SHEET,
        review.DEFINITIONS_SHEET,
    ]


def test_the_csv_carries_the_same_rows(sheet, tmp_path):
    path = tmp_path / "review.csv"

    digest = review.write_csv(sheet, path)

    reread = pd.read_csv(path)
    assert len(reread) == len(sheet)
    assert list(reread.columns) == list(sheet.columns)
    assert len(digest) == 64


def _summary_source() -> pd.DataFrame:
    """A minimal joined frame the Summary can score."""
    return pd.DataFrame(
        {
            "track": ["baseline"],
            "model": ["frequency_recency"],
            "resident_id": ["R001"],
            "target_booking_id": ["B03"],
            "origin": [stamp("2024-01-03T08:00:00+05:30")],
            "target_facility_id": ["Gym"],
            "target_usage_weekday": [4],
            "target_usage_hour": [9],
            "notification_delay_minutes": [11520.0],
            "predicted_facility_id": ["Gym"],
            "predicted_usage_weekday": [4],
            "predicted_usage_hour": [9],
            "predicted_delay_minutes": [11520.0],
        }
    )
