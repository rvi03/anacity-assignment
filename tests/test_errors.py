"""Tests the error-analysis slices.

Two things have to hold for a slice table to be worth reading. Every
dimension must *partition* the split — if the slices of one dimension do
not sum back to the whole, a reader comparing two of them is comparing
overlapping or incomplete sets. And the consistency gate must fire on a
table that breaks that, or it is decoration.

Everything here is built on hand-written rows rather than on a generated
dataset, so each expected number is countable by eye.
"""

from __future__ import annotations

import copy
import pathlib
import zoneinfo

import numpy as np
import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.evaluation import errors, evaluate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture
def scored(config) -> pd.DataFrame:
    """Eight rows: four exactly right, four wrong in a known way."""
    names = config.facility_names
    return pd.DataFrame(
        {
            "sample_id": [f"S{index:07d}" for index in range(1, 9)],
            "resident_id": ["R001", "R001", "R002", "R002"] * 2,
            "target_facility_id": [names[0]] * 4 + [names[-1]] * 4,
            "predicted_facility_id": [names[0]] * 4 + [names[0]] * 4,
            "target_usage_weekday": [0, 1, 5, 6, 0, 1, 5, 6],
            "predicted_usage_weekday": [0, 1, 5, 6, 3, 3, 3, 3],
            "target_usage_hour": [7, 8, 9, 10, 7, 8, 9, 10],
            "predicted_usage_hour": [7, 8, 9, 10, 20, 20, 20, 20],
            "notification_delay_minutes": [
                120.0,
                600.0,
                2000.0,
                6000.0,
                120.0,
                600.0,
                2000.0,
                6000.0,
            ],
            "predicted_delay_minutes": [
                120.0,
                600.0,
                2000.0,
                6000.0,
                99000.0,
                99000.0,
                99000.0,
                99000.0,
            ],
        }
    )


@pytest.fixture
def feature_rows(scored) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": scored["sample_id"],
            "resident_id": scored["resident_id"],
            "n_prior_bookings": [1, 2, 3, 4, 9, 10, 11, 12],
        }
    )


# --- slice construction ------------------------------------------------


def test_every_dimension_partitions_the_split(scored, feature_rows, config):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )

    for dimension in slices.columns:
        assert slices[dimension].notna().all()
        assert len(slices[dimension]) == len(scored)


def test_the_sparse_bucket_uses_the_configured_threshold(
    scored, feature_rows, config
):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )

    threshold = config.generator.acceptance.sparse_resident_bookings
    expected = [
        "under_5_prior" if depth < threshold else "5_or_more"
        for depth in feature_rows["n_prior_bookings"]
    ]
    assert list(slices["sparse_history"]) == expected


def test_weekend_rows_are_labelled_weekend(scored, feature_rows, config):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )

    assert (
        list(slices["day_type"])
        == [
            "weekday",
            "weekday",
            "weekend",
            "weekend",
        ]
        * 2
    )


def test_a_resident_absent_from_training_is_cold_start(
    scored, feature_rows, config
):
    seen_only = [
        sample
        for sample, resident in zip(
            scored["sample_id"], scored["resident_id"], strict=True
        )
        if resident == "R001"
    ]

    slices = errors.build_slice_columns(scored, feature_rows, seen_only, config)

    labels = dict(
        zip(
            scored["resident_id"],
            slices["cold_start_resident"],
            strict=True,
        )
    )
    assert labels["R001"] == "seen_in_training"
    assert labels["R002"] == "unseen_in_training"


def test_the_facility_band_splits_on_configured_popularity(
    scored, feature_rows, config
):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )

    # The first facility is the most popular and the last the least.
    assert set(slices["facility_band"].iloc[:4]) == {"head"}
    assert set(slices["facility_band"].iloc[4:]) == {"tail"}


def test_rows_without_a_sample_id_are_rejected(scored, feature_rows, config):
    with pytest.raises(errors.AnalysisError, match="no sample_id"):
        errors.build_slice_columns(
            scored.drop(columns=["sample_id"]),
            feature_rows,
            [],
            config,
        )


# --- the slice report --------------------------------------------------


def test_each_dimension_covers_every_row(scored, feature_rows, config):
    payload = errors.build_analysis(
        scored, feature_rows, list(scored["sample_id"]), config, "validation"
    )

    counted: dict[str, int] = {}
    for entry in payload["slices"]:
        counted[entry["dimension"]] = (
            counted.get(entry["dimension"], 0) + entry["rows"]
        )
    assert set(counted.values()) == {len(scored)}


def test_the_perfect_half_scores_one_on_every_component(
    scored, feature_rows, config
):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )
    report = errors.slice_report(scored, slices, config)

    head = next(
        entry
        for entry in report
        if entry["dimension"] == "facility_band" and entry["slice"] == "head"
    )
    assert head["overall"] == 1.0


def test_the_wrong_half_never_matches_the_facility(
    scored, feature_rows, config
):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )
    report = errors.slice_report(scored, slices, config)

    tail = next(
        entry
        for entry in report
        if entry["dimension"] == "facility_band" and entry["slice"] == "tail"
    )
    assert tail["facility"] == 0.0


def test_shares_of_the_split_sum_to_one_per_dimension(
    scored, feature_rows, config
):
    slices = errors.build_slice_columns(
        scored, feature_rows, list(scored["sample_id"]), config
    )
    report = errors.slice_report(scored, slices, config)

    for dimension in slices.columns:
        total = sum(
            entry["share_of_split"]
            for entry in report
            if entry["dimension"] == dimension
        )
        assert total == pytest.approx(1.0, abs=0.001)


# --- confusions and worst errors ---------------------------------------


def test_only_disagreements_are_reported_as_confusions(scored):
    report = errors.confusions(scored)

    for component, pairs in report.items():
        del component
        for pair in pairs:
            assert pair["actual"] != pair["predicted"]


def test_the_commonest_confusion_is_first(scored):
    report = errors.confusions(scored)

    counts = [pair["rows"] for pair in report[evaluate.USAGE_HOUR]]
    assert counts == sorted(counts, reverse=True)


def test_the_largest_notification_errors_are_worst_first(scored):
    worst = errors.largest_notification_errors(scored)

    values = [row["absolute_error_minutes"] for row in worst]
    assert values == sorted(values, reverse=True)
    assert values[0] > 0


# --- the importance sample ---------------------------------------------


def test_the_importance_sample_is_capped_and_seeded():
    table = pd.DataFrame({"a": np.arange(1000)})

    first = errors.importance_sample(table, seed=5, rows=100)
    second = errors.importance_sample(table, seed=5, rows=100)

    assert len(first) == 100
    pd.testing.assert_frame_equal(first, second)


def test_a_small_table_is_sampled_whole():
    table = pd.DataFrame({"a": np.arange(10)})

    assert len(errors.importance_sample(table, seed=5, rows=100)) == 10


def test_the_sample_keeps_the_original_row_order():
    table = pd.DataFrame({"a": np.arange(1000)})

    sample = errors.importance_sample(table, seed=5, rows=100)

    assert list(sample["a"]) == sorted(sample["a"])


# --- the consistency gate ----------------------------------------------


@pytest.fixture
def analysis(scored, feature_rows, config) -> dict:
    return errors.build_analysis(
        scored, feature_rows, list(scored["sample_id"]), config, "validation"
    )


def test_a_consistent_analysis_reports_no_problems(analysis):
    assert list(errors.check_analysis(analysis)) == []


def test_the_gate_catches_a_dimension_that_does_not_partition(analysis):
    broken = copy.deepcopy(analysis)
    broken["slices"][0]["rows"] += 3

    problems = errors.check_analysis(broken)

    assert any("must partition the split" in item for item in problems)


def test_the_gate_catches_a_rate_outside_zero_to_one(analysis):
    broken = copy.deepcopy(analysis)
    broken["slices"][0]["overall"] = 1.5

    problems = errors.check_analysis(broken)

    assert any("is not a rate" in item for item in problems)


def test_analysing_nothing_is_rejected(scored, feature_rows, config):
    with pytest.raises(errors.AnalysisError, match="empty prediction set"):
        errors.build_analysis(
            scored.iloc[0:0], feature_rows, [], config, "validation"
        )


def test_the_analysis_records_which_split_it_read(analysis):
    assert analysis["provenance"]["split"] == "validation"


def test_writing_the_same_analysis_twice_gives_the_same_hash(
    analysis, tmp_path
):
    first = errors.write_analysis(analysis, tmp_path / "a.json")
    second = errors.write_analysis(analysis, tmp_path / "b.json")

    assert first == second
