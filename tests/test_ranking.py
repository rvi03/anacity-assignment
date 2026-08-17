"""Tests the candidate reshape behind the ranked facility head.

Covers what the reshape promises: the per-facility families are found
from the catalog rather than listed, every sample becomes one contiguous
block of candidate rows, a candidate's matched columns are that
facility's own values and no other's, the labels mark the facility that
was booked, and decoding a score grid returns the top of each block.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples
from facility_prediction.features import features
from facility_prediction.models import ranking

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def smoke_features(smoke_config):
    bookings = generate.generate_bookings(smoke_config)
    frame = samples.build_samples(bookings, smoke_config)
    return features.build_features(frame, bookings, smoke_config)


@pytest.fixture(scope="module")
def rows(smoke_features):
    return smoke_features.head(40).reset_index(drop=True)


@pytest.fixture(scope="module")
def long(rows, smoke_config):
    return ranking.build_candidates(rows, smoke_config)


# --- what the reshape discovers ---------------------------------------


def test_every_family_covers_the_whole_catalog(smoke_config):
    slugs = [ranking._slug(name) for name in smoke_config.facility_names]
    declared = set(features.numeric_feature_names(smoke_config))
    for family in ranking.facility_families(smoke_config):
        for slug in slugs:
            assert f"{family}_{slug}" in declared


def test_shared_columns_name_no_facility(smoke_config):
    slugs = [ranking._slug(name) for name in smoke_config.facility_names]
    families = ranking.facility_families(smoke_config)
    for column in ranking.shared_columns(smoke_config):
        for family in families:
            for slug in slugs:
                assert column != f"{family}_{slug}"


def test_shared_and_per_facility_columns_partition_the_features(smoke_config):
    slugs = [ranking._slug(name) for name in smoke_config.facility_names]
    per_facility = {
        f"{family}_{slug}"
        for family in ranking.facility_families(smoke_config)
        for slug in slugs
    }
    shared = set(ranking.shared_columns(smoke_config))
    assert shared | per_facility == set(features.feature_columns(smoke_config))
    assert not shared & per_facility


def test_the_candidate_name_is_categorical(smoke_config):
    assert ranking.CANDIDATE_COLUMN in ranking.candidate_categoricals(
        smoke_config
    )


# --- the shape of the long table --------------------------------------


def test_one_row_per_sample_and_facility(long, rows, smoke_config):
    assert len(long) == len(rows) * len(smoke_config.facility_names)


def test_each_sample_is_one_contiguous_block(long, smoke_config):
    width = len(smoke_config.facility_names)
    groups = long[ranking.GROUP_COLUMN].to_numpy()
    assert np.array_equal(
        groups, np.repeat(np.arange(len(long) // width), width)
    )


def test_every_block_lists_the_catalog_in_order(long, smoke_config):
    width = len(smoke_config.facility_names)
    names = long[ranking.CANDIDATE_COLUMN].to_numpy().reshape(-1, width)
    for block in names:
        assert list(block) == list(smoke_config.facility_names)


def test_the_long_table_carries_every_declared_column(long, smoke_config):
    for column in ranking.candidate_columns(smoke_config):
        assert column in long.columns


def test_no_declared_column_is_a_target(smoke_config):
    features.check_denylist(
        [
            column
            for column in ranking.candidate_columns(smoke_config)
            if column != ranking.CANDIDATE_COLUMN
        ]
    )


# --- the matched values -----------------------------------------------


def test_a_matched_column_takes_that_facility_and_no_other(
    long, rows, smoke_config
):
    width = len(smoke_config.facility_names)
    for position, name in enumerate(smoke_config.facility_names):
        slug = ranking._slug(name)
        wide = rows[f"facility_share_{slug}"].to_numpy()
        matched = long["candidate_facility_share"].to_numpy()[position::width]
        assert np.allclose(wide, matched)


def test_the_top_ranked_candidate_has_rank_zero(long, smoke_config):
    width = len(smoke_config.facility_names)
    rank = long["candidate_rank_facility_share"].to_numpy().reshape(-1, width)
    share = long["candidate_facility_share"].to_numpy().reshape(-1, width)
    assert np.array_equal(rank.argmin(axis=1), share.argmax(axis=1))


def test_the_gap_to_the_top_is_zero_for_the_top(long, smoke_config):
    width = len(smoke_config.facility_names)
    gap = long["candidate_gap_to_top_facility_share"].to_numpy()
    assert np.isclose(gap.reshape(-1, width).min(axis=1), 0.0).all()
    assert (gap >= 0.0).all()


def test_the_recency_flag_marks_exactly_the_last_facility(
    long, rows, smoke_config
):
    width = len(smoke_config.facility_names)
    flag = long["candidate_is_last_1_facility"].to_numpy().reshape(-1, width)
    names = long[ranking.CANDIDATE_COLUMN].to_numpy().reshape(-1, width)
    for row, (marks, block) in enumerate(zip(flag, names, strict=True)):
        last = rows["last_1_facility"].iloc[row]
        expected = [1.0 if name == last else 0.0 for name in block]
        assert list(marks) == expected


def test_an_unused_facility_is_flagged_unseen(long):
    unseen = long["candidate_is_unseen"].to_numpy()
    counts = long["candidate_facility_count"].to_numpy()
    assert np.array_equal(unseen == 1.0, counts <= 0.0)


# --- labels and decoding ----------------------------------------------


def test_exactly_one_candidate_per_sample_is_relevant(long, smoke_config):
    width = len(smoke_config.facility_names)
    names = long[ranking.CANDIDATE_COLUMN].to_numpy().reshape(-1, width)
    booked = pd.Series([block[0] for block in names], dtype=object)
    labels = ranking.candidate_labels(long, booked, smoke_config)
    assert labels.reshape(-1, width).sum(axis=1).tolist() == [1] * len(names)


def test_the_relevant_candidate_is_the_booked_one(long, smoke_config):
    width = len(smoke_config.facility_names)
    names = long[ranking.CANDIDATE_COLUMN].to_numpy().reshape(-1, width)
    booked = pd.Series([block[2] for block in names], dtype=object)
    labels = ranking.candidate_labels(long, booked, smoke_config)
    assert np.array_equal(
        labels.reshape(-1, width).argmax(axis=1), [2] * len(names)
    )


def test_labels_reject_a_mismatched_target_count(long, smoke_config):
    with pytest.raises(ranking.RankingError, match="do not cover"):
        ranking.candidate_labels(
            long, pd.Series(["Gym"], dtype=object), smoke_config
        )


def test_ranking_returns_the_highest_scoring_candidate(long, smoke_config):
    width = len(smoke_config.facility_names)
    rng = np.random.default_rng(20260811)
    scores = rng.normal(size=len(long))
    chosen = ranking.rank_candidates(long, scores, smoke_config)
    names = long[ranking.CANDIDATE_COLUMN].to_numpy().reshape(-1, width)
    expected = names[
        np.arange(len(names)), scores.reshape(-1, width).argmax(axis=1)
    ]
    assert chosen.tolist() == list(expected)


def test_ranking_rejects_a_score_count_that_does_not_fit(long, smoke_config):
    with pytest.raises(ranking.RankingError, match="scores for"):
        ranking.rank_candidates(long, np.zeros(3), smoke_config)


def test_probabilities_sum_to_one_per_sample(long, smoke_config):
    rng = np.random.default_rng(20260812)
    sheet = ranking.candidate_probabilities(
        long, rng.normal(size=len(long)), smoke_config
    )
    assert np.allclose(sheet.to_numpy().sum(axis=1), 1.0)


def test_probabilities_keep_the_ranking_order(long, smoke_config):
    rng = np.random.default_rng(20260813)
    scores = rng.normal(size=len(long))
    sheet = ranking.candidate_probabilities(long, scores, smoke_config)
    chosen = ranking.rank_candidates(long, scores, smoke_config)
    top = sheet.columns[sheet.to_numpy().argmax(axis=1)]
    assert list(top) == chosen.tolist()


def test_probability_columns_are_the_catalog(long, smoke_config):
    sheet = ranking.candidate_probabilities(
        long, np.zeros(len(long)), smoke_config
    )
    assert list(sheet.columns) == list(smoke_config.facility_names)


def test_a_table_without_a_sample_id_is_rejected(rows, smoke_config):
    with pytest.raises(ranking.RankingError, match="sample_id"):
        ranking.build_candidates(rows.drop(columns="sample_id"), smoke_config)


def test_a_table_missing_a_family_column_is_rejected(rows, smoke_config):
    slug = ranking._slug(smoke_config.facility_names[0])
    with pytest.raises(ranking.RankingError, match="candidate columns"):
        ranking.build_candidates(
            rows.drop(columns=f"facility_count_{slug}"), smoke_config
        )
