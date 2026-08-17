"""Tests the cross-track comparison query.

The registry itself is thin — it attaches files to a run and tags them.
What needs testing is the query that reads every track at once, because
that is where a track can silently disappear: an inner join over a
manifest drops whatever has no rows, and a missing track then looks
identical to a track that scored zero.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.cli.commands import analysis

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture
def samples_table(config) -> pd.DataFrame:
    names = config.facility_names
    return pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "target_facility_id": [names[0], names[1], names[0]],
            "target_usage_weekday": [1, 2, 3],
            "target_usage_hour": [7, 8, 9],
            "notification_delay_minutes": [600.0, 700.0, 800.0],
        }
    )


def _predictions(
    track: str, model: str, ids: list[str], config
) -> pd.DataFrame:
    names = config.facility_names
    return pd.DataFrame(
        {
            "track": [track] * len(ids),
            "model": [model] * len(ids),
            "sample_id": ids,
            "predicted_facility_id": [names[0]] * len(ids),
            "predicted_usage_weekday": [1] * len(ids),
            "predicted_usage_hour": [7] * len(ids),
            "predicted_delay_minutes": [600.0] * len(ids),
        }
    )


@pytest.fixture
def stored(config) -> pd.DataFrame:
    return pd.concat(
        [
            _predictions("baseline", "frequency_recency", ["S1", "S2"], config),
            _predictions("traditional", "catboost", ["S1", "S2"], config),
            _predictions("llm", "adapter", ["S1"], config),
        ],
        ignore_index=True,
    )


def test_one_query_returns_every_track(stored, samples_table, config):
    rows = analysis.cross_track_comparison(
        stored, samples_table, {"S1", "S2"}, config
    )

    assert {row["track"] for row in rows} == {
        "baseline",
        "traditional",
        "llm",
    }


def test_each_row_names_its_model(stored, samples_table, config):
    rows = analysis.cross_track_comparison(
        stored, samples_table, {"S1", "S2"}, config
    )

    assert {(row["track"], row["model"]) for row in rows} == {
        ("baseline", "frequency_recency"),
        ("traditional", "catboost"),
        ("llm", "adapter"),
    }


def test_the_manifest_bounds_which_rows_are_compared(
    stored, samples_table, config
):
    rows = analysis.cross_track_comparison(
        stored, samples_table, {"S1"}, config
    )

    assert {row["rows_on_manifest"] for row in rows} == {1}


def test_a_track_with_fewer_rows_is_reported_not_dropped(
    stored, samples_table, config
):
    # The LLM track covers S1 only. It must appear with a row count of
    # one rather than vanish from the comparison.
    rows = analysis.cross_track_comparison(
        stored, samples_table, {"S1", "S2"}, config
    )

    llm = next(row for row in rows if row["track"] == "llm")
    assert llm["rows_on_manifest"] == 1


def test_a_track_with_no_rows_on_the_manifest_still_appears(
    stored, samples_table, config
):
    rows = analysis.cross_track_comparison(
        stored, samples_table, {"S3"}, config
    )

    llm = next(row for row in rows if row["track"] == "llm")
    assert llm["rows_on_manifest"] == 0
    assert "overall" not in llm


def test_a_perfect_track_scores_one(stored, samples_table, config):
    # S1's targets are exactly what every fixture row predicts.
    rows = analysis.cross_track_comparison(
        stored, samples_table, {"S1"}, config
    )

    assert {row["overall"] for row in rows} == {1.0}


def test_rows_come_back_in_a_stable_order(stored, samples_table, config):
    first = analysis.cross_track_comparison(
        stored, samples_table, {"S1", "S2"}, config
    )
    second = analysis.cross_track_comparison(
        stored, samples_table, {"S1", "S2"}, config
    )

    assert first == second
