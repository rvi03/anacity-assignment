"""Tests the synthetic generator's contract.

Covers what the generator promises its callers: an identical file hash
on re-run, ``booking_timestamp < usage_timestamp`` on every row, and
the guarantees later stages depend on — timezone-aware timestamps, rows
sorted by ``(booking_timestamp, booking_id)``, strictly increasing
booking times within a resident, usage hours inside the half-open
operating window, and no hidden resident behaviour in the dataset.
"""

from __future__ import annotations

import copy
import datetime
import pathlib
from typing import Any
import zoneinfo

import pandas as pd
import pytest
import yaml

from facility_prediction import config as config_module
from facility_prediction.data import generate

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def smoke_frame(smoke_config) -> pd.DataFrame:
    return generate.generate_bookings(smoke_config)


def raw_smoke() -> dict[str, Any]:
    """A fresh mutable copy of the parsed smoke config."""
    return copy.deepcopy(yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8")))


# --- the file is the reproducibility claim ------


def test_rerun_produces_an_identical_file_hash(smoke_config, tmp_path):
    first = generate.write_bookings(
        generate.generate_bookings(smoke_config), tmp_path / "first.csv"
    )

    second = generate.write_bookings(
        generate.generate_bookings(smoke_config), tmp_path / "second.csv"
    )

    assert first == second
    assert (tmp_path / "first.csv").read_bytes() == (
        tmp_path / "second.csv"
    ).read_bytes()


def test_a_different_seed_produces_a_different_file(tmp_path):
    raw = raw_smoke()
    raw["seed"] = raw["seed"] + 1
    other = config_module.parse_config(raw)

    baseline = generate.write_bookings(
        generate.generate_bookings(config_module.load_config(SMOKE_PATH)),
        tmp_path / "baseline.csv",
    )
    shifted = generate.write_bookings(
        generate.generate_bookings(other), tmp_path / "shifted.csv"
    )

    assert baseline != shifted


def test_written_file_round_trips_timezone_aware(smoke_frame, tmp_path):
    path = tmp_path / "bookings.csv"
    generate.write_bookings(smoke_frame, path)

    reloaded = pd.read_csv(
        path, parse_dates=["booking_timestamp", "usage_timestamp"]
    )

    offset = datetime.timedelta(hours=5, minutes=30)
    for column in ("booking_timestamp", "usage_timestamp"):
        assert reloaded[column].dt.tz is not None
        assert reloaded[column].iloc[0].utcoffset() == offset
        assert (
            reloaded[column].dt.tz_convert(KOLKATA).to_list()
            == smoke_frame[column].to_list()
        )
    assert len(reloaded) == len(smoke_frame)


# --- the prediction contract -----------------------------


def test_frame_carries_exactly_the_contract_columns(smoke_frame):
    assert tuple(smoke_frame.columns) == generate.BOOKING_COLUMNS


def test_no_hidden_resident_state_reaches_the_dataset(smoke_frame):
    leaked = {
        "archetype",
        "consistency",
        "preference",
        "activity",
        "activity_class",
        "join_week",
        "lead_multiplier",
        "hour_center",
        "is_noise",
        "weekday_weights",
    }

    assert leaked.isdisjoint(set(smoke_frame.columns))


def test_booking_precedes_usage_on_every_row(smoke_frame):
    assert (
        smoke_frame["booking_timestamp"] < smoke_frame["usage_timestamp"]
    ).all()


def test_lead_time_respects_the_configured_floor(smoke_frame, smoke_config):
    lead = smoke_frame["usage_timestamp"] - smoke_frame["booking_timestamp"]

    floor = datetime.timedelta(minutes=smoke_config.generator.min_lead_minutes)
    assert lead.min() >= floor


def test_every_timestamp_is_timezone_aware(smoke_frame):
    for column in ("booking_timestamp", "usage_timestamp"):
        series = smoke_frame[column]

        assert series.dt.tz is not None
        assert series.dt.tz.key == "Asia/Kolkata"


def test_identifiers_are_unique_and_non_null(smoke_frame):
    assert smoke_frame["booking_id"].is_unique
    assert smoke_frame["resident_id"].notna().all()
    assert smoke_frame["facility_id"].notna().all()


# --- ordering later stages depend on ---------------------


def test_rows_are_sorted_by_booking_time_then_id(smoke_frame):
    ordered = smoke_frame.sort_values(
        ["booking_timestamp", "booking_id"], kind="mergesort"
    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(smoke_frame, ordered)


def test_booking_times_strictly_increase_within_each_resident(smoke_frame):
    for _, rows in smoke_frame.groupby("resident_id", sort=True):
        stamps = rows["booking_timestamp"].sort_values()

        assert stamps.is_monotonic_increasing
        assert stamps.is_unique


# --- facility rules ----------------------------------


def test_usage_hour_is_inside_half_open_operating_hours(
    smoke_frame, smoke_config
):
    hours = smoke_frame["usage_timestamp"].dt.hour

    for name, rows in smoke_frame.groupby("facility_id", sort=True):
        facility = smoke_config.facility(str(name))
        used = hours.loc[rows.index]

        assert used.min() >= facility.open_hour
        assert used.max() < facility.close_hour


def test_facility_ids_come_from_the_catalog(smoke_frame, smoke_config):
    assert set(smoke_frame["facility_id"]) <= set(smoke_config.facility_names)


def test_no_booking_uses_a_facility_before_it_opens(smoke_frame, smoke_config):
    start = smoke_config.start_instant

    for name, rows in smoke_frame.groupby("facility_id", sort=True):
        facility = smoke_config.facility(str(name))
        opens = start + pd.DateOffset(months=facility.available_from_month)

        assert rows["usage_timestamp"].min() >= opens


def test_yoga_room_has_no_rows_before_its_opening_month(
    smoke_frame, smoke_config
):
    opens = smoke_config.start_instant + pd.DateOffset(
        months=smoke_config.facility("Yoga Room").available_from_month
    )
    yoga = smoke_frame[smoke_frame["facility_id"] == "Yoga Room"]

    assert (yoga["usage_timestamp"] >= opens).all()


def test_usage_stays_inside_the_configured_horizon(smoke_frame, smoke_config):
    start = smoke_config.start_instant
    end = start + pd.DateOffset(months=smoke_config.community.months)

    assert smoke_frame["usage_timestamp"].min() >= start
    assert smoke_frame["usage_timestamp"].max() < end


def test_slot_capacity_is_never_exceeded(smoke_frame, smoke_config):
    seats = smoke_frame.assign(
        usage_date=smoke_frame["usage_timestamp"].dt.date,
        usage_hour=smoke_frame["usage_timestamp"].dt.hour,
    )

    counts = seats.groupby(
        ["facility_id", "usage_date", "usage_hour"], sort=True
    ).size()
    for (name, _, _), taken in counts.items():
        assert taken <= smoke_config.facility(str(name)).slot_capacity


# --- scale and behaviour ---------------------------


def test_smoke_config_stays_small_enough_for_ci(smoke_frame, smoke_config):
    assert smoke_config.community.residents < 100
    assert 0 < len(smoke_frame) < 10_000


def test_residents_have_varied_history_lengths(smoke_frame):
    per_resident = smoke_frame.groupby("resident_id").size()

    assert per_resident.min() < per_resident.max()


def test_a_shorter_horizon_produces_fewer_bookings():
    raw = raw_smoke()
    raw["community"]["months"] = 12
    shorter = config_module.parse_config(raw)

    assert len(generate.generate_bookings(shorter)) < len(
        generate.generate_bookings(config_module.load_config(SMOKE_PATH))
    )


def test_default_config_generates_the_planned_scale():
    config = config_module.load_config(DEFAULT_PATH)

    frame = generate.generate_bookings(config)

    booked = set(frame["resident_id"])
    expected = {
        f"R{index + 1:04d}" for index in range(config.community.residents)
    }

    assert 20_000 <= len(frame) <= 30_000
    assert booked <= expected
    assert len(booked) > len(expected) // 2


# --- dated drift and acceptance checks --------------------------------


def smoke_audited(config):
    """Generate the smoke dataset together with its aggregate counters."""
    return generate.generate_audited(config)


def test_every_acceptance_check_passes_on_the_real_dataset():
    config = config_module.load_config(REPO_ROOT / "configs" / "default.yaml")
    frame, audit = generate.generate_audited(config)

    failed = [
        check.name
        for check in generate.acceptance_checks(frame, config, audit)
        if not check.passed
    ]

    assert failed == []


def test_structural_checks_hold_even_at_smoke_scale(smoke_config):
    frame, audit = smoke_audited(smoke_config)
    structural = {
        "booking_precedes_usage_on_every_row",
        "resident_bookings_strictly_increase",
        "table_is_in_its_deterministic_order",
        "no_booking_before_a_facility_opens",
    }

    failed = [
        check.name
        for check in generate.acceptance_checks(frame, smoke_config, audit)
        if check.name in structural and not check.passed
    ]

    assert failed == []


def test_a_tiny_configuration_records_failures_without_stopping(
    smoke_config,
):
    frame, audit = smoke_audited(smoke_config)

    summary = generate.check_dataset(frame, smoke_config, audit)

    assert not smoke_config.generator.acceptance.enforce
    assert summary["checks"]


def test_the_booking_schema_accepts_the_generated_table(smoke_config):
    frame, _ = smoke_audited(smoke_config)

    validated = generate.booking_schema(smoke_config).validate(frame)

    assert len(validated) == len(frame)


def test_the_schema_rejects_a_facility_outside_the_catalog(smoke_config):
    frame, _ = smoke_audited(smoke_config)
    broken = frame.copy()
    broken.loc[broken.index[0], "facility_id"] = "Rooftop Bar"

    with pytest.raises(Exception, match="isin"):
        generate.booking_schema(smoke_config).validate(broken)


def test_a_failing_dataset_stops_a_pipeline_that_enforces(smoke_config):
    frame, _ = smoke_audited(smoke_config)
    enforcing = smoke_config.model_copy(
        update={
            "generator": smoke_config.generator.model_copy(
                update={
                    "acceptance": (
                        smoke_config.generator.acceptance.model_copy(
                            update={"enforce": True}
                        )
                    )
                }
            )
        }
    )

    with pytest.raises(generate.GenerationError, match="acceptance checks"):
        generate.check_dataset(frame, enforcing, generate.GenerationAudit())


def test_the_summary_records_every_check_and_the_verdict(smoke_config):
    frame, audit = smoke_audited(smoke_config)

    summary = generate.check_dataset(frame, smoke_config, audit)

    assert len(summary["checks"]) == len(
        generate.acceptance_checks(frame, smoke_config, audit)
    )
    assert summary["counts"]["bookings"] == len(frame)
    assert summary["all_passed"] == all(
        check["passed"] for check in summary["checks"]
    )


def test_the_late_facility_appears_only_after_it_opens(smoke_config):
    frame, _ = smoke_audited(smoke_config)
    drift = smoke_config.generator.drift
    opens = smoke_config.start_instant + pd.DateOffset(
        months=smoke_config.facility(
            drift.absorbing_facility
        ).available_from_month
    )
    rows = frame.loc[frame["facility_id"] == drift.absorbing_facility]

    assert (rows["usage_timestamp"] >= opens).all()
    assert not rows.empty


def test_the_life_change_cohort_is_a_configured_share(smoke_config):
    _, audit = smoke_audited(smoke_config)
    expected = (
        smoke_config.community.residents
        * smoke_config.generator.drift.resample_share
    )

    assert abs(audit.resampled_residents - expected) <= expected


def test_the_season_swings_one_facility_only(smoke_config):
    season = smoke_config.generator.drift.season
    peak = generate._season_multipliers(smoke_config, season.peak_month + 1)
    names = smoke_config.facility_names
    others = [
        peak[index]
        for index, name in enumerate(names)
        if name != season.facility
    ]

    assert peak[names.index(season.facility)] > 1.0
    assert set(others) == {1.0}


def test_the_audit_holds_no_resident_level_state(smoke_config):
    _, audit = smoke_audited(smoke_config)

    assert audit.noise_path + audit.preference_path > 0
    assert set(audit.total_bookings) <= set(
        range(smoke_config.community.residents)
    )
