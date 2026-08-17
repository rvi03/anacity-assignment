"""Tests the rolling-origin sampler's contract.

Covers what a sample promises: its origin is the resident's immediately
previous booking creation time, it never reads past the target booking,
its four labels come from that target, and residents who cannot produce
a sample are counted instead of dropped.
"""

from __future__ import annotations

import pathlib
import zoneinfo

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


def bookings_frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """A tiny booking table from ISO strings carrying an offset."""
    frame = pd.DataFrame(rows, columns=list(generate.BOOKING_COLUMNS))
    for column in ("booking_timestamp", "usage_timestamp"):
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(KOLKATA)
    return frame


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def smoke_bookings(smoke_config) -> pd.DataFrame:
    return generate.generate_bookings(smoke_config)


@pytest.fixture(scope="module")
def smoke_samples(smoke_bookings, smoke_config) -> pd.DataFrame:
    return samples.build_samples(smoke_bookings, smoke_config)


@pytest.fixture
def three_bookings() -> pd.DataFrame:
    return bookings_frame(
        [
            (
                "B0000001",
                "R001",
                "Gym",
                "2024-01-01T08:00:00+05:30",
                "2024-01-02T07:00:00+05:30",
            ),
            (
                "B0000002",
                "R001",
                "Pool",
                "2024-01-03T09:30:00+05:30",
                "2024-01-04T18:00:00+05:30",
            ),
            (
                "B0000003",
                "R001",
                "Gym",
                "2024-01-06T20:15:00+05:30",
                "2024-01-07T06:00:00+05:30",
            ),
            (
                "B0000004",
                "R002",
                "Tennis",
                "2024-01-02T10:00:00+05:30",
                "2024-01-03T17:00:00+05:30",
            ),
        ]
    )


# --- the origin is the previous booking, nothing else ------


def test_origin_is_the_residents_immediately_previous_booking(three_bookings):
    frame = samples.build_samples(
        three_bookings, config_module.load_config(SMOKE_PATH)
    )

    by_target = frame.set_index("target_booking_id")

    assert by_target.loc["B0000002", "origin_booking_id"] == "B0000001"
    assert by_target.loc["B0000003", "origin_booking_id"] == "B0000002"
    assert by_target.loc["B0000002", "origin"] == pd.Timestamp(
        "2024-01-01T08:00:00+05:30"
    )


def test_origin_strictly_precedes_every_target_booking(smoke_samples):
    later = smoke_samples["target_booking_timestamp"] > smoke_samples["origin"]

    assert bool(later.all())


def test_origin_matches_the_previous_booking_across_the_dataset(
    smoke_bookings, smoke_samples
):
    ordered = smoke_bookings.sort_values(
        ["resident_id", "booking_timestamp", "booking_id"], kind="mergesort"
    )
    expected = dict(
        zip(
            ordered["booking_id"],
            ordered.groupby("resident_id", sort=False)["booking_id"].shift(1),
            strict=True,
        )
    )

    actual = dict(
        zip(
            smoke_samples["target_booking_id"],
            smoke_samples["origin_booking_id"],
            strict=True,
        )
    )

    assert all(expected[target] == origin for target, origin in actual.items())


def test_no_sample_is_built_from_a_later_booking(smoke_bookings, smoke_samples):
    creation = dict(
        zip(
            smoke_bookings["booking_id"],
            smoke_bookings["booking_timestamp"],
            strict=True,
        )
    )

    origins = smoke_samples["origin_booking_id"].map(creation)

    assert bool(origins.eq(smoke_samples["origin"]).all())


# --- who is kept, who is counted ------


def test_resident_with_n_bookings_yields_n_minus_one_samples(three_bookings):
    frame = samples.build_samples(
        three_bookings, config_module.load_config(SMOKE_PATH)
    )

    counts = frame["resident_id"].value_counts().to_dict()

    assert counts == {"R001": 2}


def test_single_booking_resident_yields_nothing_but_is_counted(
    three_bookings,
):
    config = config_module.load_config(SMOKE_PATH)
    frame = samples.build_samples(three_bookings, config)

    counts = samples.summarise_samples(three_bookings, frame, config)

    assert "R002" not in set(frame["resident_id"])
    assert counts.residents_excluded_no_prior_history == 1


def test_counts_reconcile_against_the_configured_roster(
    smoke_bookings, smoke_samples, smoke_config
):
    counts = samples.summarise_samples(
        smoke_bookings, smoke_samples, smoke_config
    )

    total = (
        counts.residents_without_bookings
        + counts.residents_excluded_no_prior_history
        + counts.residents_with_samples
    )

    assert total == counts.residents_configured
    assert counts.residents_without_bookings >= 0


def test_sample_count_equals_bookings_minus_one_per_booking_resident(
    smoke_bookings, smoke_samples, smoke_config
):
    counts = samples.summarise_samples(
        smoke_bookings, smoke_samples, smoke_config
    )

    assert counts.samples == counts.bookings - counts.residents_with_bookings


def test_every_resident_final_booking_is_reported_as_censored(
    smoke_bookings, smoke_samples, smoke_config
):
    counts = samples.summarise_samples(
        smoke_bookings, smoke_samples, smoke_config
    )

    assert counts.censored_bookings == counts.residents_with_bookings


def test_observation_horizon_spans_the_dataset(
    smoke_bookings, smoke_samples, smoke_config
):
    counts = samples.summarise_samples(
        smoke_bookings, smoke_samples, smoke_config
    )

    assert (
        counts.observation_start
        == smoke_bookings["booking_timestamp"].min().isoformat()
    )
    assert (
        counts.observation_end
        == smoke_bookings["usage_timestamp"].max().isoformat()
    )


# --- the four labels come from the target booking ------


def test_labels_are_taken_from_the_target_booking(three_bookings):
    frame = samples.build_samples(
        three_bookings, config_module.load_config(SMOKE_PATH)
    )

    row = frame.set_index("target_booking_id").loc["B0000003"]

    assert row["target_facility_id"] == "Gym"
    assert row["target_usage_weekday"] == 6
    assert row["target_usage_hour"] == 6


def test_delay_is_the_gap_from_origin_to_target_booking_creation(
    three_bookings,
):
    frame = samples.build_samples(
        three_bookings, config_module.load_config(SMOKE_PATH)
    )

    row = frame.set_index("target_booking_id").loc["B0000002"]

    assert row["notification_delay_minutes"] == pytest.approx(2 * 24 * 60 + 90)


def test_delay_is_strictly_positive_everywhere(smoke_samples):
    assert bool((smoke_samples["notification_delay_minutes"] > 0).all())


def test_labels_match_the_target_row_across_the_dataset(
    smoke_bookings, smoke_samples
):
    targets = smoke_bookings.set_index("booking_id")[
        ["facility_id", "usage_timestamp"]
    ]
    joined = smoke_samples.join(targets, on="target_booking_id")

    assert bool(joined["target_facility_id"].eq(joined["facility_id"]).all())
    assert bool(
        joined["target_usage_hour"].eq(joined["usage_timestamp"].dt.hour).all()
    )


# --- shape, ordering, determinism ------


def test_frame_carries_exactly_the_contract_columns(smoke_samples):
    assert tuple(smoke_samples.columns) == samples.SAMPLE_COLUMNS


def test_every_timestamp_is_timezone_aware(smoke_samples):
    for column in samples.TIMESTAMP_COLUMNS:
        assert isinstance(smoke_samples[column].dtype, pd.DatetimeTZDtype)
        assert str(smoke_samples[column].dt.tz) == "Asia/Kolkata"


def test_samples_are_ordered_by_origin_then_target_id(smoke_samples):
    ordered = smoke_samples.sort_values(
        ["origin", "target_booking_id"], kind="mergesort"
    )

    assert list(ordered["sample_id"]) == list(smoke_samples["sample_id"])


def test_sample_ids_are_unique(smoke_samples):
    assert smoke_samples["sample_id"].is_unique


def test_shuffled_input_produces_identical_samples(
    smoke_bookings, smoke_samples, smoke_config
):
    shuffled = smoke_bookings.sample(
        frac=1.0, random_state=smoke_config.seed
    ).reset_index(drop=True)

    rebuilt = samples.build_samples(shuffled, smoke_config)

    pd.testing.assert_frame_equal(rebuilt, smoke_samples)


def test_rerun_produces_an_identical_file_hash(smoke_samples, tmp_path):
    first = samples.write_samples(smoke_samples, tmp_path / "first.csv")

    second = samples.write_samples(smoke_samples, tmp_path / "second.csv")

    assert first == second


def test_written_file_round_trips_timezone_aware(smoke_samples, tmp_path):
    path = tmp_path / "samples.csv"
    samples.write_samples(smoke_samples, path)

    reloaded = pd.read_csv(path)
    parsed = pd.to_datetime(reloaded["origin"], format="ISO8601", utc=True)

    assert bool(parsed.dt.tz_convert(KOLKATA).eq(smoke_samples["origin"]).all())


def test_summary_is_written_with_its_provenance(
    smoke_bookings, smoke_samples, smoke_config, tmp_path
):
    counts = samples.summarise_samples(
        smoke_bookings, smoke_samples, smoke_config
    )
    path = tmp_path / "sample_summary.json"

    samples.write_summary(counts, {"seed": smoke_config.seed}, path)

    payload = pd.read_json(path, typ="series")
    assert payload["provenance"]["seed"] == smoke_config.seed
    assert payload["counts"]["samples"] == counts.samples


# --- the sampler refuses input it cannot read ------


def test_minimum_of_zero_prior_bookings_is_rejected(three_bookings):
    raw = config_module.load_config(SMOKE_PATH).model_dump()
    raw["evaluation"]["min_prior_bookings"] = 0
    config = config_module.parse_config(raw)

    with pytest.raises(ValueError, match="min_prior_bookings"):
        samples.build_samples(three_bookings, config)


def test_naive_timestamps_are_rejected(three_bookings):
    naive = three_bookings.copy()
    naive["booking_timestamp"] = naive["booking_timestamp"].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        samples.build_samples(naive, config_module.load_config(SMOKE_PATH))


def test_missing_contract_column_is_rejected(three_bookings):
    without = three_bookings.drop(columns=["facility_id"])

    with pytest.raises(ValueError, match="missing required columns"):
        samples.build_samples(without, config_module.load_config(SMOKE_PATH))


def test_simultaneous_bookings_by_one_resident_are_rejected(three_bookings):
    tied = three_bookings.copy()
    tied.loc[1, "booking_timestamp"] = tied.loc[0, "booking_timestamp"]

    with pytest.raises(ValueError, match="strictly increase"):
        samples.build_samples(tied, config_module.load_config(SMOKE_PATH))


def test_a_higher_minimum_history_drops_the_thinnest_residents(
    smoke_bookings, smoke_samples
):
    raw = config_module.load_config(SMOKE_PATH).model_dump()
    raw["evaluation"]["min_prior_bookings"] = 3
    config = config_module.parse_config(raw)

    stricter = samples.build_samples(smoke_bookings, config)

    assert len(stricter) < len(smoke_samples)
    assert bool((stricter["n_prior_bookings"] >= 3).all())
