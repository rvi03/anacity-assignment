"""Tests the generator rigour profile.

The profile's job is to be believable without being re-derived by hand,
so what is tested here is that its numbers are what they claim: the
intervals are seeded and repeatable, the quantiles are monotonic, the
shares are shares, and the consistency gate actually fires on a profile
that contradicts itself. A gate that passes on anything is not a gate.
"""

from __future__ import annotations

import copy
import hashlib
import pathlib
import zoneinfo

import numpy as np
import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate, profiles

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def smoke_bookings(smoke_config) -> pd.DataFrame:
    return generate.generate_bookings(smoke_config)


@pytest.fixture(scope="module")
def profile(smoke_bookings, smoke_config) -> dict:
    return profiles.build_profile(smoke_bookings, smoke_config)


# --- intervals --------------------------------------------------------


def test_a_bootstrap_interval_is_repeatable_for_a_seed():
    values = np.linspace(0.0, 10.0, 200)

    first = profiles.bootstrap_interval(values, seed=7)
    second = profiles.bootstrap_interval(values, seed=7)

    assert first == second


def test_a_different_seed_gives_a_different_interval():
    values = np.linspace(0.0, 10.0, 200)

    assert profiles.bootstrap_interval(
        values, seed=7
    ) != profiles.bootstrap_interval(values, seed=8)


def test_an_interval_brackets_the_sample_mean():
    values = np.concatenate([np.zeros(100), np.ones(100)])

    lower, upper = profiles.bootstrap_interval(values, seed=11)

    assert lower <= values.mean() <= upper


def test_a_wider_level_gives_a_wider_interval():
    values = np.linspace(0.0, 10.0, 200)

    narrow = profiles.bootstrap_interval(values, seed=3, level=0.50)
    wide = profiles.bootstrap_interval(values, seed=3, level=0.99)

    assert wide[0] <= narrow[0]
    assert wide[1] >= narrow[1]


def test_bootstrapping_nothing_is_rejected():
    with pytest.raises(profiles.ProfileError, match="empty sample"):
        profiles.bootstrap_interval([], seed=1)


# --- quantiles --------------------------------------------------------


def test_quantiles_are_monotonic():
    values = np.random.default_rng(20260816).lognormal(size=500)

    computed = profiles.quantiles(values)

    ordered = [computed[key] for key in sorted(computed)]
    assert ordered == sorted(ordered)


def test_quantiles_of_a_constant_sample_are_that_constant():
    computed = profiles.quantiles(np.full(50, 3.5))

    assert set(computed.values()) == {3.5}


def test_quantiles_of_nothing_are_rejected():
    with pytest.raises(profiles.ProfileError, match="empty sample"):
        profiles.quantiles([])


# --- the profile itself ------------------------------------------------


def test_every_configured_facility_appears_in_the_popularity_profile(
    profile, smoke_config
):
    named = {row["facility"] for row in profile["popularity"]}

    assert named == set(smoke_config.facility_names)


def test_realised_shares_sum_to_one(profile):
    total = sum(row["realised"] for row in profile["popularity"])

    assert total == pytest.approx(1.0, abs=0.01)


def test_the_deviation_is_realised_minus_configured(profile):
    for row in profile["popularity"]:
        assert row["deviation"] == pytest.approx(
            row["realised"] - row["configured"], abs=1e-4
        )


def test_lead_times_are_reported_per_facility(profile, smoke_config):
    named = set(profile["lead_time_hours"]["per_facility"])

    assert named.issubset(set(smoke_config.facility_names))
    assert named


def test_the_activity_profile_counts_residents_who_never_booked(
    profile, smoke_config
):
    activity = profile["activity"]

    assert activity["residents_configured"] == smoke_config.community.residents
    assert (
        activity["residents_with_bookings"]
        <= (activity["residents_configured"])
    )


def test_monthly_shares_are_shares(profile):
    for row in profile["monthly_shares"]:
        assert sum(row["shares"].values()) == pytest.approx(1.0, abs=0.01)


def test_the_profile_is_deterministic(smoke_bookings, smoke_config):
    first = profiles.build_profile(smoke_bookings, smoke_config)
    second = profiles.build_profile(smoke_bookings, smoke_config)

    assert first == second


def test_profiling_nothing_is_rejected(smoke_bookings, smoke_config):
    with pytest.raises(profiles.ProfileError, match="empty booking table"):
        profiles.build_profile(smoke_bookings.iloc[0:0], smoke_config)


# --- the consistency gate ----------------------------------------------


def test_a_consistent_profile_reports_no_problems(profile):
    assert list(profiles.check_profile(profile)) == []


def test_the_gate_catches_shares_that_do_not_sum_to_one(profile):
    broken = copy.deepcopy(profile)
    broken["popularity"][0]["realised"] += 0.5

    problems = profiles.check_profile(broken)

    assert any("sum to" in message for message in problems)


def test_the_gate_catches_a_share_outside_its_own_interval(profile):
    broken = copy.deepcopy(profile)
    broken["popularity"][0]["realised_interval"] = [0.99, 1.0]

    problems = profiles.check_profile(broken)

    assert any("outside its own interval" in message for message in problems)


def test_the_gate_catches_non_monotonic_quantiles(profile):
    broken = copy.deepcopy(profile)
    overall = broken["lead_time_hours"]["overall"]
    overall["p05"], overall["p95"] = overall["p95"], overall["p05"]

    problems = profiles.check_profile(broken)

    assert any("not monotonic" in message for message in problems)


def test_the_gate_catches_a_distribution_that_is_not_right_skewed(profile):
    broken = copy.deepcopy(profile)
    broken["lead_time_hours"]["mean_over_median"] = 0.9

    problems = profiles.check_profile(broken)

    assert any("not right-skewed" in message for message in problems)


def test_the_gate_catches_more_residents_booking_than_exist(profile):
    broken = copy.deepcopy(profile)
    broken["activity"]["residents_with_bookings"] = (
        broken["activity"]["residents_configured"] + 1
    )

    problems = profiles.check_profile(broken)

    assert any("than were configured" in message for message in problems)


def test_the_gate_catches_a_month_whose_shares_do_not_sum_to_one(profile):
    broken = copy.deepcopy(profile)
    first = next(iter(broken["monthly_shares"][0]["shares"]))
    broken["monthly_shares"][0]["shares"][first] += 0.5

    problems = profiles.check_profile(broken)

    assert any("facility shares sum to" in message for message in problems)


# --- artifacts ---------------------------------------------------------


def test_the_written_profile_hashes_its_own_bytes(profile, tmp_path):
    path = tmp_path / "profile.json"

    digest = profiles.write_profile(profile, path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_writing_the_same_profile_twice_gives_the_same_hash(profile, tmp_path):
    first = profiles.write_profile(profile, tmp_path / "a.json")
    second = profiles.write_profile(profile, tmp_path / "b.json")

    assert first == second


def test_every_declared_plot_is_written(smoke_bookings, profile, tmp_path):
    written = profiles.write_plots(smoke_bookings, profile, tmp_path)

    assert tuple(path.name for path in written) == profiles.PLOT_FILENAMES
    assert all(path.stat().st_size > 0 for path in written)
