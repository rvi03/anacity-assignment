"""Tests the feature table's contract.

Covers what the table promises: every column is derived from events at
or before the origin, categoricals are always text and never a rendered
null, numerics are float everywhere so two splits cannot disagree on a
dtype, and the same inputs always produce the same table.
"""

from __future__ import annotations

import copy
import math
import pathlib
import zoneinfo

import pandas as pd
import pytest
import yaml

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples
from facility_prediction.features import features

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"

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


@pytest.fixture(scope="module")
def smoke_features(smoke_samples, smoke_bookings, smoke_config):
    return features.build_features(smoke_samples, smoke_bookings, smoke_config)


@pytest.fixture
def four_bookings() -> pd.DataFrame:
    """One resident, four bookings, hand-checkable spacings."""
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
                "2024-01-03T08:00:00+05:30",
                "2024-01-04T18:00:00+05:30",
            ),
            (
                "B0000003",
                "R001",
                "Gym",
                "2024-01-11T08:00:00+05:30",
                "2024-01-12T09:00:00+05:30",
            ),
            (
                "B0000004",
                "R001",
                "Tennis",
                "2024-02-20T08:00:00+05:30",
                "2024-02-21T20:00:00+05:30",
            ),
        ]
    )


@pytest.fixture
def four_booking_features(four_bookings, smoke_config) -> pd.DataFrame:
    frame = samples.build_samples(four_bookings, smoke_config)
    return features.build_features(frame, four_bookings, smoke_config)


def load_raw(path: pathlib.Path) -> dict:
    """Read a config file as a plain mapping, for mutation in a test."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# --- schema -----------------------------------------------------------


def test_schema_accepts_the_table_it_describes(smoke_features, smoke_config):
    validated = features.validate_features(smoke_features, smoke_config)

    assert list(validated.columns) == list(features.table_columns(smoke_config))


def test_every_numeric_feature_is_float(smoke_features, smoke_config):
    dtypes = {
        name: str(smoke_features[name].dtype)
        for name in features.numeric_feature_names(smoke_config)
    }

    assert set(dtypes.values()) == {"float64"}


def test_categoricals_never_hold_a_float(smoke_features, smoke_config):
    for name in features.categorical_feature_names(smoke_config):
        values = smoke_features[name]

        assert values.map(lambda value: isinstance(value, str)).all()


def test_categoricals_never_hold_a_rendered_null(smoke_features, smoke_config):
    for name in features.categorical_feature_names(smoke_config):
        lowered = smoke_features[name].str.strip().str.lower()

        assert not lowered.isin(features._FORBIDDEN_TEXT).any()


def test_missing_categorical_slots_use_the_configured_token(
    four_booking_features, smoke_config
):
    token = smoke_config.features.categorical_missing_token
    first = four_booking_features.iloc[0]

    assert first["last_1_facility"] == "Gym"
    assert first["last_2_facility"] == token
    assert first["last_3_facility"] == token


def test_schema_rejects_a_float_in_a_categorical(smoke_features, smoke_config):
    broken = smoke_features.copy()
    broken.loc[broken.index[0], "last_1_facility"] = float("nan")

    with pytest.raises(features.FeatureSchemaError):
        features.validate_features(broken, smoke_config)


def test_schema_rejects_missingness_spelt_as_text(smoke_features, smoke_config):
    broken = smoke_features.copy()
    broken.loc[broken.index[0], "last_2_facility"] = "nan"

    with pytest.raises(features.FeatureSchemaError):
        features.validate_features(broken, smoke_config)


def test_schema_rejects_an_unexpected_column(smoke_features, smoke_config):
    broken = smoke_features.copy()
    broken["surprise"] = 1.0

    with pytest.raises(features.FeatureSchemaError):
        features.validate_features(broken, smoke_config)


def test_schema_rejects_a_reordered_table(smoke_features, smoke_config):
    reordered = smoke_features[list(reversed(smoke_features.columns))]

    with pytest.raises(features.FeatureSchemaError):
        features.validate_features(reordered, smoke_config)


# --- the gate: train and evaluation agree on every dtype --------------


def test_train_and_evaluation_dtypes_are_identical(
    smoke_features, smoke_config
):
    half = len(smoke_features) // 2
    train = smoke_features.iloc[:half]
    evaluation = smoke_features.iloc[half:]

    features.check_consistent_dtypes(train, evaluation)

    assert features.validate_features(train, smoke_config) is not None
    assert features.validate_features(evaluation, smoke_config) is not None


def test_dtype_check_catches_an_integer_count_column(smoke_features):
    half = len(smoke_features) // 2
    train = smoke_features.iloc[:half].copy()
    evaluation = smoke_features.iloc[half:].copy()
    train["n_prior_bookings"] = train["n_prior_bookings"].astype("int64")

    with pytest.raises(features.FeatureSchemaError, match="dtypes differ"):
        features.check_consistent_dtypes(train, evaluation)


def test_dtype_check_catches_a_missing_column(smoke_features):
    half = len(smoke_features) // 2

    with pytest.raises(features.FeatureSchemaError, match="different columns"):
        features.check_consistent_dtypes(
            smoke_features.iloc[:half].drop(columns=["last_1_facility"]),
            smoke_features.iloc[half:],
        )


# --- leakage ----------------------------------------------------------


def test_features_ignore_events_after_the_origin(
    four_bookings, four_booking_features, smoke_config
):
    truncated = four_bookings.iloc[:2]
    frame = samples.build_samples(truncated, smoke_config)
    from_truncated = features.build_features(frame, truncated, smoke_config)

    pd.testing.assert_frame_equal(
        from_truncated, four_booking_features.iloc[:1]
    )


def test_a_changed_future_cannot_change_a_past_row(
    four_bookings, four_booking_features, smoke_config
):
    perturbed = four_bookings.copy()
    perturbed.loc[3, "facility_id"] = "Pool"
    perturbed.loc[3, "usage_timestamp"] += pd.Timedelta(days=5)
    frame = samples.build_samples(perturbed, smoke_config)
    rebuilt = features.build_features(frame, perturbed, smoke_config)

    pd.testing.assert_frame_equal(
        rebuilt.iloc[:2], four_booking_features.iloc[:2]
    )


def test_history_length_disagreeing_with_the_origin_is_an_error(
    four_bookings, smoke_config
):
    frame = samples.build_samples(four_bookings, smoke_config)
    frame.loc[0, "n_prior_bookings"] = 3

    with pytest.raises(ValueError, match="prior bookings"):
        features.build_features(frame, four_bookings, smoke_config)


def test_an_origin_before_every_booking_is_an_error(
    four_bookings, smoke_config
):
    frame = samples.build_samples(four_bookings, smoke_config)
    frame.loc[0, "origin"] = frame.loc[0, "origin"] - pd.Timedelta(days=365)
    frame.loc[0, "n_prior_bookings"] = 0

    with pytest.raises(ValueError, match="no booking at or before"):
        features.build_features(frame, four_bookings, smoke_config)


def test_naive_timestamps_are_rejected(four_bookings, smoke_config):
    frame = samples.build_samples(four_bookings, smoke_config)
    frame["origin"] = frame["origin"].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        features.build_features(frame, four_bookings, smoke_config)


# --- hand-checked values ----------------------------------------------


def test_prior_counts_and_windows_are_hand_checkable(four_booking_features):
    third = four_booking_features.iloc[2]

    assert third["n_prior_bookings"] == pytest.approx(3.0)
    assert third["bookings_prior_7d"] == pytest.approx(1.0)
    assert third["bookings_prior_30d"] == pytest.approx(3.0)
    assert third["bookings_prior_180d"] == pytest.approx(3.0)


def test_recency_is_measured_from_the_origin(four_booking_features):
    third = four_booking_features.iloc[2]

    assert third["days_since_previous_booking"] == pytest.approx(8.0)
    assert third["days_since_first_booking"] == pytest.approx(10.0)


def test_recency_is_missing_on_a_one_booking_history(four_booking_features):
    first = four_booking_features.iloc[0]

    assert pd.isna(first["days_since_previous_booking"])
    assert first["days_since_first_booking"] == pytest.approx(0.0)


def test_intervals_are_booking_to_booking(four_booking_features):
    third = four_booking_features.iloc[2]

    assert third["inter_booking_interval_mean_days"] == pytest.approx(5.0)
    assert third["inter_booking_interval_median_days"] == pytest.approx(5.0)


def test_spread_needs_two_intervals(four_booking_features):
    first, third = (
        four_booking_features.iloc[0],
        four_booking_features.iloc[2],
    )

    assert pd.isna(first["inter_booking_interval_std_days"])
    assert third["inter_booking_interval_std_days"] == pytest.approx(
        4.242640687, rel=1e-6
    )


def test_facility_counts_and_shares_agree(four_booking_features):
    third = four_booking_features.iloc[2]

    assert third["facility_count_gym"] == pytest.approx(2.0)
    assert third["facility_share_gym"] == pytest.approx(2.0 / 3.0)
    assert third["facility_count_tennis"] == pytest.approx(0.0)


def test_rolling_favourite_breaks_ties_by_recency(four_booking_features):
    second = four_booking_features.iloc[1]

    assert second["rolling_top_facility_last_3"] == "Pool"
    assert second["rolling_top_facility_share_last_3"] == pytest.approx(0.5)


def test_lead_times_are_usage_minus_booking(four_booking_features):
    first = four_booking_features.iloc[0]

    assert first["lead_minutes_recent"] == pytest.approx(23.0 * 60.0)
    assert first["lead_minutes_mean"] == pytest.approx(23.0 * 60.0)


def test_time_band_shares_sum_to_one(smoke_features, smoke_config):
    columns = [
        f"time_band_share_{features._slug(band.name)}"
        for band in smoke_config.features.time_bands
    ]

    assert smoke_features[columns].sum(axis=1).round(9).eq(1.0).all()


def test_rate_ratio_compares_a_short_window_to_a_long_one(
    four_booking_features,
):
    third = four_booking_features.iloc[2]

    assert third["bookings_prior_30d"] == pytest.approx(3.0)
    assert third["bookings_prior_90d"] == pytest.approx(3.0)
    assert third["booking_rate_ratio_30d_90d"] == pytest.approx(3.0)


# --- decayed preference, entropy, transitions -------------------------


def test_ewma_weights_a_recent_booking_above_an_old_one(
    four_booking_features,
):
    third = four_booking_features.iloc[2]
    weights = (0.5 ** (10.0 / 30.0), 0.5 ** (8.0 / 30.0), 1.0)
    gym = weights[0] + weights[2]

    assert third["ewma_facility_share_gym"] == pytest.approx(gym / sum(weights))
    assert third["ewma_facility_share_gym"] > third["facility_share_gym"]


def test_ewma_shares_sum_to_one_within_each_group(smoke_features, smoke_config):
    facilities = [
        f"ewma_facility_share_{features._slug(name)}"
        for name in smoke_config.facility_names
    ]
    bands = [
        f"ewma_time_band_share_{features._slug(band.name)}"
        for band in smoke_config.features.time_bands
    ]

    assert smoke_features[facilities].sum(axis=1).round(9).eq(1.0).all()
    assert smoke_features[bands].sum(axis=1).round(9).eq(1.0).all()


def test_entropy_separates_a_habitual_row_from_a_mixed_one(
    four_booking_features, smoke_config
):
    first, third = (
        four_booking_features.iloc[0],
        four_booking_features.iloc[2],
    )
    mixed = -(2 / 3 * math.log2(2 / 3) + 1 / 3 * math.log2(1 / 3))

    assert first["facility_entropy_bits"] == pytest.approx(0.0)
    assert third["facility_entropy_bits"] == pytest.approx(mixed)
    assert third["facility_entropy_normalised"] == pytest.approx(
        mixed / math.log2(len(smoke_config.facility_names))
    )


def test_transitions_are_counted_out_of_the_most_recent_facility(
    four_booking_features,
):
    third = four_booking_features.iloc[2]

    assert third["last_1_facility"] == "Gym"
    assert third["transitions_from_last_total"] == pytest.approx(1.0)
    assert third["transition_count_from_last_pool"] == pytest.approx(1.0)
    assert third["transition_share_from_last_pool"] == pytest.approx(1.0)
    assert third["transition_count_from_last_gym"] == pytest.approx(0.0)


def test_a_one_booking_history_has_no_transition_to_report(
    four_booking_features,
):
    first = four_booking_features.iloc[0]

    assert first["transitions_from_last_total"] == pytest.approx(0.0)
    assert pd.isna(first["transition_share_from_last_gym"])


# --- community history ------------------------------------------------


def test_no_community_column_is_a_running_count(smoke_config):
    community = [
        name
        for name in features.numeric_feature_names(smoke_config)
        if name.startswith("community_")
    ]

    assert community
    assert all(
        name.startswith("community_facility_share_")
        or name.startswith("community_lead_minutes_mean_")
        for name in community
    )


def test_community_shares_describe_only_earlier_bookings(
    four_booking_features,
):
    third = four_booking_features.iloc[2]

    assert third["community_facility_share_gym"] == pytest.approx(0.5)
    assert third["community_facility_share_pool"] == pytest.approx(0.5)
    assert third["community_facility_share_tennis"] == pytest.approx(0.0)


def test_community_shares_are_missing_before_any_booking_exists(
    four_booking_features,
):
    first = four_booking_features.iloc[0]

    assert pd.isna(first["community_facility_share_gym"])
    assert pd.isna(first["community_lead_minutes_mean_gym"])


def test_community_popularity_is_keyed_by_the_origins_own_slot(
    four_booking_features,
):
    third = four_booking_features.iloc[2]

    assert pd.isna(third["community_facility_share_origin_weekday_gym"])
    assert third["community_facility_share_origin_band_gym"] == pytest.approx(
        0.5
    )


def test_community_lead_time_is_averaged_per_facility(four_booking_features):
    third = four_booking_features.iloc[2]

    assert third["community_lead_minutes_mean_gym"] == pytest.approx(
        23.0 * 60.0
    )
    assert third["community_lead_minutes_mean_pool"] == pytest.approx(
        34.0 * 60.0
    )
    assert pd.isna(third["community_lead_minutes_mean_tennis"])


def test_a_window_sees_only_the_bookings_inside_it(four_bookings, smoke_config):
    older = four_bookings.copy()
    older.loc[0, "booking_timestamp"] -= pd.Timedelta(days=60)
    older.loc[0, "usage_timestamp"] -= pd.Timedelta(days=60)
    frame = samples.build_samples(older, smoke_config)

    third = features.build_features(frame, older, smoke_config).iloc[2]

    assert third["community_facility_share_30d_pool"] == pytest.approx(1.0)
    assert third["community_facility_share_90d_pool"] == pytest.approx(0.5)


# --- determinism and identifiers --------------------------------------


def test_building_twice_gives_the_same_digest(
    smoke_samples, smoke_bookings, smoke_config, smoke_features
):
    again = features.build_features(smoke_samples, smoke_bookings, smoke_config)

    assert features.features_digest(again) == features.features_digest(
        smoke_features
    )


def test_identifiers_are_kept_but_flagged_as_non_features(
    smoke_features, smoke_config
):
    assert list(smoke_features.columns[:3]) == list(features.IDENTIFIER_COLUMNS)
    assert set(features.MODEL_EXCLUSIONS) >= set(features.IDENTIFIER_COLUMNS)
    assert not set(features.IDENTIFIER_COLUMNS) & set(
        features.feature_columns(smoke_config)
    )


def test_one_row_per_sample_in_input_order(smoke_features, smoke_samples):
    assert len(smoke_features) == len(smoke_samples)
    assert list(smoke_features["sample_id"]) == list(smoke_samples["sample_id"])


def test_manifest_lists_every_produced_column(smoke_features, smoke_config):
    manifest = features.build_manifest(smoke_features, smoke_config, {})

    listed = (
        manifest["identifier_columns"]
        + manifest["categorical_features"]
        + manifest["numeric_features"]
    )
    assert listed == list(smoke_features.columns)
    assert manifest["counts"]["rows"] == len(smoke_features)


# --- configuration ----------------------------------------------------


def test_default_and_smoke_declare_the_same_columns():
    default = config_module.load_config(DEFAULT_PATH)
    smoke = config_module.load_config(SMOKE_PATH)

    assert features.table_columns(default) == features.table_columns(smoke)


def test_time_bands_must_tile_the_day():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["time_bands"] = [
        {"name": "morning", "hours": [6, 12]},
        {"name": "rest", "hours": [12, 24]},
    ]

    with pytest.raises(config_module.ConfigError, match="tile"):
        config_module.parse_config(raw)


def test_time_bands_may_not_overlap():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["time_bands"] = [
        {"name": "early", "hours": [0, 13]},
        {"name": "late", "hours": [12, 24]},
    ]

    with pytest.raises(config_module.ConfigError, match="tile"):
        config_module.parse_config(raw)


def test_trend_windows_must_be_counted_windows():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["trend_windows_days"] = [14, 90]

    with pytest.raises(config_module.ConfigError, match="prior_windows_days"):
        config_module.parse_config(raw)


def test_trend_windows_must_be_short_then_long():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["trend_windows_days"] = [90, 30]

    with pytest.raises(config_module.ConfigError, match="short < "):
        config_module.parse_config(raw)


def test_prior_windows_must_strictly_increase():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["prior_windows_days"] = [30, 7, 90, 180]

    with pytest.raises(config_module.ConfigError, match="strictly increase"):
        config_module.parse_config(raw)


def test_community_windows_must_strictly_increase():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["community_windows_days"] = [90, 30]

    with pytest.raises(config_module.ConfigError, match="strictly increase"):
        config_module.parse_config(raw)


def test_the_ewma_half_life_must_be_positive():
    raw = load_raw(SMOKE_PATH)
    raw["features"]["ewma_halflife_days"] = 0

    with pytest.raises(config_module.ConfigError, match="ewma_halflife_days"):
        config_module.parse_config(raw)


def test_column_names_follow_the_configured_windows():
    raw = copy.deepcopy(load_raw(SMOKE_PATH))
    raw["features"]["prior_windows_days"] = [14, 28]
    raw["features"]["trend_windows_days"] = [14, 28]
    config = config_module.parse_config(raw)

    names = features.numeric_feature_names(config)

    assert "bookings_prior_14d" in names
    assert "booking_rate_ratio_14d_28d" in names
    assert "bookings_prior_7d" not in names


def test_community_column_names_follow_the_configured_windows():
    raw = copy.deepcopy(load_raw(SMOKE_PATH))
    raw["features"]["community_windows_days"] = [14]
    config = config_module.parse_config(raw)

    names = features.numeric_feature_names(config)

    assert "community_facility_share_14d_gym" in names
    assert "community_facility_share_30d_gym" not in names
