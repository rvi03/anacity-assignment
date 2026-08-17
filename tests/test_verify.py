"""Tests that verification can actually fail.

`make verify` passing is only evidence if it would have caught a change.
Every test here breaks one committed value and asserts the matching
check reports it — and a companion test asserts the same check stays
quiet when nothing moved, so the checks are not simply always-on.
"""

from __future__ import annotations

import hashlib
import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples
from facility_prediction.evaluation import verify
from facility_prediction.features import features

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def bookings(config) -> pd.DataFrame:
    return generate.generate_bookings(config)


@pytest.fixture(scope="module")
def sample_table(bookings, config) -> pd.DataFrame:
    return samples.build_samples(bookings, config)


@pytest.fixture(scope="module")
def feature_table(sample_table, bookings, config):
    return features.build_features(sample_table, bookings, config)


@pytest.fixture
def committed(config, bookings, sample_table, feature_table) -> dict:
    return {
        "provenance": {
            "config_hash": config_module.config_hash(config),
            "bookings_digest": generate.bookings_digest(bookings),
            "samples_digest": samples.samples_digest(sample_table),
            "features_digest": features.features_digest(feature_table),
        },
        "validation": {"model": {"matches": {"overall": 0.25}}},
    }


# --- digests ------------------------------------------------------------


def test_matching_digests_report_nothing(
    committed, bookings, sample_table, feature_table
):
    assert (
        verify.check_digests(committed, bookings, sample_table, feature_table)
        == []
    )


def test_a_moved_bookings_digest_is_reported(
    committed, bookings, sample_table, feature_table
):
    committed["provenance"]["bookings_digest"] = "0" * 64

    problems = verify.check_digests(
        committed, bookings, sample_table, feature_table
    )

    assert any("bookings_digest" in item for item in problems)


def test_a_moved_features_digest_is_reported(
    committed, bookings, sample_table, feature_table
):
    committed["provenance"]["features_digest"] = "0" * 64

    problems = verify.check_digests(
        committed, bookings, sample_table, feature_table
    )

    assert any("features_digest" in item for item in problems)


def test_a_changed_dataset_is_reported(
    committed, bookings, sample_table, feature_table
):
    # Not a doctored record this time — a doctored dataset. The digest
    # has to move because the rows did.
    changed = bookings.copy()
    changed.loc[changed.index[0], "facility_id"] = "Somewhere Else"

    problems = verify.check_digests(
        committed, changed, sample_table, feature_table
    )

    assert any("bookings_digest" in item for item in problems)


# --- config hash --------------------------------------------------------


def test_an_unchanged_config_reports_nothing(committed, config):
    assert verify.check_config_hash(committed, config) == []


def test_a_changed_setting_is_reported(committed, config):
    moved = config.model_copy(update={"seed": config.seed + 1})

    problems = verify.check_config_hash(committed, moved)

    assert any("config_hash" in item for item in problems)


# --- scored metrics -----------------------------------------------------


def test_identical_metrics_report_nothing(committed):
    assert (
        verify.check_scored_metrics(
            committed, "validation", committed["validation"]
        )
        == []
    )


def test_a_moved_metric_is_reported(committed):
    recomputed = {"model": {"matches": {"overall": 0.26}}}

    problems = verify.check_scored_metrics(committed, "validation", recomputed)

    assert any("overall" in item for item in problems)


def test_a_float_representation_difference_is_tolerated(committed):
    recomputed = {"model": {"matches": {"overall": 0.25 + 1e-15}}}

    assert (
        verify.check_scored_metrics(committed, "validation", recomputed) == []
    )


def test_a_split_with_no_committed_block_is_reported(committed):
    problems = verify.check_scored_metrics(committed, "test", {})

    assert any("no committed block" in item for item in problems)


def test_only_numbers_are_compared(committed):
    # A string that changed is not a metric that moved; comparing them
    # would make every rename look like a regression.
    recomputed = {"model": {"matches": {"overall": 0.25}, "note": "different"}}

    assert (
        verify.check_scored_metrics(committed, "validation", recomputed) == []
    )


# --- the workbook -------------------------------------------------------


def test_an_unchanged_workbook_reports_nothing(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    assert verify.check_workbook(path, digest) == []


def test_a_changed_workbook_is_reported(tmp_path):
    path = tmp_path / "review.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text("a,b\n1,3\n", encoding="utf-8")

    problems = verify.check_workbook(path, digest)

    assert any("review_csv_sha256" in item for item in problems)


def test_a_missing_workbook_is_reported(tmp_path):
    problems = verify.check_workbook(tmp_path / "absent.csv", "0" * 64)

    assert any("absent" in item for item in problems)


def test_no_recorded_hash_means_nothing_to_check(tmp_path):
    assert verify.check_workbook(tmp_path / "absent.csv", None) == []


# --- tracking parity ----------------------------------------------------


def test_matching_tracking_metrics_report_nothing():
    assert verify.check_tracking_parity({"score": 0.25}, {"score": 0.25}) == []


def test_a_tracker_that_disagrees_is_reported():
    problems = verify.check_tracking_parity({"score": 0.30}, {"score": 0.25})

    assert any("tracking.score" in item for item in problems)


def test_a_metric_the_tracker_never_logged_is_reported():
    problems = verify.check_tracking_parity({}, {"score": 0.25})

    assert any("no logged score" in item for item in problems)


def test_an_unreachable_tracker_is_not_a_failure():
    # The tracking database is excluded from reproduction by design;
    # its absence must not be reported as a wrong number.
    assert verify.check_tracking_parity(None, {"score": 0.25}) == []


# --- reporting ----------------------------------------------------------


def test_a_clean_run_says_so():
    assert "reproduced" in verify.report([])


def test_a_dirty_run_names_every_difference():
    text = verify.report(["a: x", "b: y"])

    assert "2 value(s) disagree" in text
    assert "a: x" in text
    assert "b: y" in text


def test_missing_metrics_are_refused(tmp_path):
    with pytest.raises(verify.VerificationError, match="no committed metrics"):
        verify.load_metrics(tmp_path / "metrics.json")
