"""The numbered leakage controls, each with a test that can fail.

Five controls are checked here. Each has at least one test that passes
only because the control is present: delete the guard and the named test
goes red, which is the difference between a test of a guarantee and a
restatement of it.

    L1  a row's history stops at its origin; its target comes after it
    L2  no target, and nothing derived from one, is a model input
    L3  community aggregates stop strictly before the origin, and every
        window over them is closed on the left and open on the right
    L4  the split is chronological and the partitions cannot overlap
    L7  membership is frozen, fits see training rows only, and the
        settings a run used are identified by a hash
    L8  the generator's hidden behaviour never reaches the model table

One of the eight controls is deliberately absent. L6 governs chained
models, which do not exist, so there is nothing here to test and nothing
here is claimed about it.
"""

from __future__ import annotations

import copy
import pathlib
import zoneinfo

import pandas as pd
import pytest
import yaml

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples, split
from facility_prediction.features import features

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")

SHIPPED_CONTROLS = ("l1", "l2", "l3", "l4", "l7", "l8")


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


@pytest.fixture(scope="module")
def smoke_splits(smoke_samples, smoke_config) -> pd.DataFrame:
    cutoffs = split.compute_cutoffs(smoke_samples, smoke_config)
    labelled = split.assign_split(smoke_samples, cutoffs)
    return labelled[[split.SAMPLE_ID_COLUMN, split.SPLIT_COLUMN]]


@pytest.fixture(scope="module")
def smoke_manifest(smoke_samples, smoke_config) -> dict:
    cutoffs = split.compute_cutoffs(smoke_samples, smoke_config)
    labelled = split.assign_split(smoke_samples, cutoffs)
    return split.build_split_manifest(labelled, cutoffs, smoke_config, {})


def two_bookings() -> pd.DataFrame:
    """One resident, two bookings, so exactly one sample exists."""
    frame = pd.DataFrame(
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
                "2024-01-05T08:00:00+05:30",
                "2024-01-06T18:00:00+05:30",
            ),
        ],
        columns=list(generate.BOOKING_COLUMNS),
    )
    for column in ("booking_timestamp", "usage_timestamp"):
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(KOLKATA)
    return frame


ORIGIN_INSTANT = pd.Timestamp("2024-01-01T08:00:00+05:30").tz_convert(KOLKATA)


def community_bookings() -> pd.DataFrame:
    """Two residents, so one row has a community history it did not make.

    R001's only origin is :data:`ORIGIN_INSTANT`. Exactly two bookings
    exist before it, both R002's, and the earlier of the two sits
    exactly thirty days back — on the left edge of the shortest
    configured community window.
    """
    frame = pd.DataFrame(
        [
            (
                "B0000001",
                "R002",
                "Pool",
                "2023-12-02T08:00:00+05:30",
                "2023-12-03T09:00:00+05:30",
            ),
            (
                "B0000002",
                "R002",
                "Gym",
                "2023-12-20T08:00:00+05:30",
                "2023-12-21T09:00:00+05:30",
            ),
            (
                "B0000003",
                "R001",
                "Gym",
                "2024-01-01T08:00:00+05:30",
                "2024-01-02T07:00:00+05:30",
            ),
            (
                "B0000004",
                "R001",
                "Pool",
                "2024-01-05T08:00:00+05:30",
                "2024-01-06T18:00:00+05:30",
            ),
        ],
        columns=list(generate.BOOKING_COLUMNS),
    )
    for column in ("booking_timestamp", "usage_timestamp"):
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(KOLKATA)
    return frame


def with_extra_booking(
    bookings: pd.DataFrame, booked_at: pd.Timestamp, facility: str
) -> pd.DataFrame:
    """Add a third resident's single booking at a chosen instant.

    One booking makes no sample of its own, so the sample table is
    unchanged and any difference in the feature table comes from the
    community aggregates alone.
    """
    extra = pd.DataFrame(
        [
            {
                "booking_id": "B0000009",
                "resident_id": "R003",
                "facility_id": facility,
                "booking_timestamp": booked_at,
                "usage_timestamp": booked_at + pd.Timedelta(hours=12),
            }
        ]
    )
    return pd.concat([bookings, extra], ignore_index=True)


def config_with_seed(seed: int) -> config_module.Config:
    """The smoke config with a different root seed."""
    raw = copy.deepcopy(yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8")))
    raw["seed"] = seed
    return config_module.parse_config(raw)


# --- L1 · history stops at the origin, the target comes after ---------


def test_l1_every_row_reads_only_events_at_or_before_its_origin(
    smoke_samples, smoke_bookings
):
    latest = {}
    for booking_id, timestamp in zip(
        smoke_bookings["booking_id"],
        smoke_bookings["booking_timestamp"],
        strict=True,
    ):
        latest[booking_id] = timestamp

    origins = smoke_samples["origin"]
    origin_events = smoke_samples["origin_booking_id"].map(latest)

    assert (origin_events <= origins).all()


def test_l1_every_target_is_created_after_its_origin(smoke_samples):
    later = smoke_samples["target_booking_timestamp"] > smoke_samples["origin"]

    assert later.all()


def test_l1_a_target_at_the_origin_is_rejected(smoke_config):
    bookings = two_bookings()
    frame = samples.build_samples(bookings, smoke_config)
    frame["target_booking_timestamp"] = frame["origin"]

    with pytest.raises(ValueError, match="not after its origin"):
        features.build_features(frame, bookings, smoke_config)


def test_l1_a_history_longer_than_the_origin_admits_is_rejected(smoke_config):
    bookings = two_bookings()
    frame = samples.build_samples(bookings, smoke_config)
    frame["n_prior_bookings"] = frame["n_prior_bookings"] + 1

    with pytest.raises(ValueError, match="the origin bound admits"):
        features.build_features(frame, bookings, smoke_config)


def test_l1_the_feature_table_ignores_events_after_the_origin(
    smoke_config, smoke_features, smoke_samples, smoke_bookings
):
    cut = smoke_samples["origin"].median()
    early = smoke_samples.loc[smoke_samples["origin"] <= cut]
    truncated = smoke_bookings.loc[smoke_bookings["booking_timestamp"] <= cut]

    rebuilt = features.build_features(early, truncated, smoke_config)

    pd.testing.assert_frame_equal(
        rebuilt.reset_index(drop=True),
        smoke_features.head(len(early)).reset_index(drop=True),
    )


# --- L2 · no target, and nothing derived from one, is an input --------


def test_l2_no_feature_column_names_a_target(smoke_config):
    columns = features.feature_columns(smoke_config)

    denied = features.denylist_violations(columns)

    assert denied == ()


def test_l2_the_denylist_catches_a_target_column():
    denied = features.denylist_violations(
        ["origin_hour", "target_facility_id", "n_prior_bookings"]
    )

    assert denied == ("target_facility_id",)


def test_l2_the_denylist_catches_a_target_derived_column():
    denied = features.denylist_violations(
        ["origin_hour", "notification_delay_bucket", "delay_label_mean"]
    )

    assert denied == ("delay_label_mean", "notification_delay_bucket")


def test_l2_every_sampler_target_is_on_the_denylist():
    targets = [
        name
        for name in samples.SAMPLE_COLUMNS
        if name.startswith("target_") or name == "notification_delay_minutes"
    ]

    assert features.denylist_violations(targets) == tuple(sorted(targets))


def test_l2_a_manifest_cannot_record_a_target_as_a_feature(
    smoke_features, smoke_config, monkeypatch
):
    def leaking_names(_: config_module.Config) -> tuple[str, ...]:
        return ("last_1_facility", "target_facility_id")

    monkeypatch.setattr(features, "categorical_feature_names", leaking_names)

    with pytest.raises(ValueError, match="must not be model inputs"):
        features.build_manifest(smoke_features, smoke_config, {})


def test_l2_the_manifest_records_the_denylist_it_was_checked_against(
    smoke_features, smoke_config
):
    manifest = features.build_manifest(smoke_features, smoke_config, {})

    assert manifest["target_denylist"]["violations"] == []
    assert "target_facility_id" in manifest["target_denylist"]["columns"]


def test_l2_the_identifier_of_the_target_is_never_a_model_input(smoke_config):
    assert "booking_id" not in features.feature_columns(smoke_config)
    assert "booking_id" in features.MODEL_EXCLUSIONS


# --- L3 · community aggregates stop strictly before the origin --------


def test_l3_a_community_aggregate_reads_only_earlier_bookings(smoke_config):
    bookings = community_bookings()
    frame = samples.build_samples(bookings, smoke_config)

    built = features.build_features(frame, bookings, smoke_config)
    row = built.loc[built["resident_id"] == "R001"].iloc[0]

    assert row["community_facility_share_gym"] == pytest.approx(0.5)
    assert row["community_facility_share_pool"] == pytest.approx(0.5)
    assert row["community_facility_share_tennis"] == pytest.approx(0.0)


def test_l3_a_booking_made_at_the_origin_instant_is_excluded(smoke_config):
    bookings = community_bookings()
    frame = samples.build_samples(bookings, smoke_config)
    baseline = features.build_features(frame, bookings, smoke_config)
    simultaneous = with_extra_booking(bookings, ORIGIN_INSTANT, "Tennis")

    rebuilt = features.build_features(
        samples.build_samples(simultaneous, smoke_config),
        simultaneous,
        smoke_config,
    )

    pd.testing.assert_frame_equal(rebuilt, baseline)


def test_l3_a_later_community_booking_cannot_change_an_earlier_row(
    smoke_config,
):
    bookings = community_bookings()
    frame = samples.build_samples(bookings, smoke_config)
    baseline = features.build_features(frame, bookings, smoke_config)
    later = with_extra_booking(
        bookings, ORIGIN_INSTANT + pd.Timedelta(days=7), "Tennis"
    )

    rebuilt = features.build_features(
        samples.build_samples(later, smoke_config), later, smoke_config
    )

    pd.testing.assert_frame_equal(rebuilt, baseline)


def test_l3_a_community_window_is_closed_on_the_left(smoke_config):
    bookings = community_bookings()
    frame = samples.build_samples(bookings, smoke_config)

    built = features.build_features(frame, bookings, smoke_config)
    row = built.loc[built["resident_id"] == "R001"].iloc[0]

    assert row["community_facility_share_30d_pool"] == pytest.approx(0.5)


def test_l3_the_bound_is_asserted_on_every_row(smoke_config):
    bookings = community_bookings()
    community = features._community_index(bookings, smoke_config)
    reach_past_the_origin = len(bookings)

    with pytest.raises(ValueError, match="not before its origin"):
        features._check_community_bound(
            community, ORIGIN_INSTANT, reach_past_the_origin, "S0000001"
        )


# --- L4 · the split is chronological ----------------------------------


def test_l4_the_partitions_never_overlap_in_time(smoke_samples, smoke_config):
    cutoffs = split.compute_cutoffs(smoke_samples, smoke_config)
    labelled = split.assign_split(smoke_samples, cutoffs)

    split.check_boundaries(labelled)

    bounds = labelled.groupby(split.SPLIT_COLUMN)[split.TARGET_TIME_COLUMN].agg(
        ["min", "max"]
    )
    assert bounds.loc[split.TRAIN, "max"] < bounds.loc[split.VALIDATION, "min"]
    assert bounds.loc[split.VALIDATION, "max"] < bounds.loc[split.TEST, "min"]


def test_l4_an_overlapping_split_is_rejected(smoke_samples, smoke_config):
    cutoffs = split.compute_cutoffs(smoke_samples, smoke_config)
    labelled = split.assign_split(smoke_samples, cutoffs)
    moved = labelled.copy()
    moved.iloc[0, moved.columns.get_loc(split.SPLIT_COLUMN)] = split.TEST

    with pytest.raises(ValueError, match="must end strictly before"):
        split.check_boundaries(moved)


def test_l4_the_frozen_split_carries_no_embargo(smoke_config):
    assert smoke_config.split.embargo_days == 0


# --- L7 · membership is frozen and cannot reach a fit -----------------


def test_l7_a_fit_on_training_rows_is_audited_and_passes(smoke_splits):
    training = smoke_splits.loc[
        smoke_splits[split.SPLIT_COLUMN] == split.TRAIN,
        split.SAMPLE_ID_COLUMN,
    ].tolist()

    audited = split.check_fit_membership(training, smoke_splits)

    assert audited == len(training)


def test_l7_a_holdout_row_in_a_fit_call_is_rejected(smoke_splits):
    training = smoke_splits.loc[
        smoke_splits[split.SPLIT_COLUMN] == split.TRAIN,
        split.SAMPLE_ID_COLUMN,
    ].tolist()
    holdout = smoke_splits.loc[
        smoke_splits[split.SPLIT_COLUMN] == split.TEST,
        split.SAMPLE_ID_COLUMN,
    ].tolist()

    with pytest.raises(ValueError, match=f"outside \\['{split.TRAIN}'\\]"):
        split.check_fit_membership([*training, holdout[0]], smoke_splits)


def test_l7_a_validation_row_in_a_fit_call_is_rejected(smoke_splits):
    validation = smoke_splits.loc[
        smoke_splits[split.SPLIT_COLUMN] == split.VALIDATION,
        split.SAMPLE_ID_COLUMN,
    ].tolist()

    with pytest.raises(ValueError, match="reached a fit call"):
        split.check_fit_membership(validation[:1], smoke_splits)


def test_l7_a_row_absent_from_the_frozen_split_is_rejected(smoke_splits):
    with pytest.raises(ValueError, match="absent from the frozen split"):
        split.check_fit_membership(["S9999999"], smoke_splits)


def test_l7_selection_may_read_validation_when_it_says_so(smoke_splits):
    validation = smoke_splits.loc[
        smoke_splits[split.SPLIT_COLUMN] == split.VALIDATION,
        split.SAMPLE_ID_COLUMN,
    ].tolist()

    audited = split.check_fit_membership(
        validation, smoke_splits, allowed=(split.VALIDATION,)
    )

    assert audited == len(validation)


def test_l7_rebuilding_the_split_reproduces_the_frozen_manifest(
    smoke_samples, smoke_config, smoke_manifest
):
    cutoffs = split.compute_cutoffs(smoke_samples, smoke_config)
    labelled = split.assign_split(smoke_samples, cutoffs)
    rebuilt = split.build_split_manifest(labelled, cutoffs, smoke_config, {})

    split.check_manifest_unchanged(rebuilt, smoke_manifest)


def test_l7_a_moved_cutoff_is_refused(smoke_manifest):
    moved = copy.deepcopy(smoke_manifest)
    moved["cutoffs"]["val_cut"] = moved["cutoffs"]["end"]

    with pytest.raises(ValueError, match="the frozen split has moved"):
        split.check_manifest_unchanged(moved, smoke_manifest)


def test_l7_a_changed_row_count_is_refused(smoke_manifest):
    moved = copy.deepcopy(smoke_manifest)
    moved["splits"][split.TEST]["rows"] += 1

    with pytest.raises(ValueError, match="splits"):
        split.check_manifest_unchanged(moved, smoke_manifest)


def test_l7_new_provenance_alone_is_not_a_moved_split(smoke_manifest):
    reproduced = copy.deepcopy(smoke_manifest)
    reproduced["provenance"] = {"seed": 999}

    split.check_manifest_unchanged(reproduced, smoke_manifest)


def test_l7_the_config_hash_is_stable_across_a_reload(smoke_config):
    again = config_module.load_config(SMOKE_PATH)

    assert config_module.config_hash(again) == config_module.config_hash(
        smoke_config
    )


def test_l7_the_config_hash_survives_a_round_trip(smoke_config, tmp_path):
    path = tmp_path / "round_trip.yaml"
    config_module.dump_config(smoke_config, path)

    reloaded = config_module.load_config(path)

    assert config_module.config_hash(reloaded) == config_module.config_hash(
        smoke_config
    )


def test_l7_the_config_hash_changes_when_a_setting_changes(smoke_config):
    other = config_with_seed(smoke_config.seed + 1)

    assert config_module.config_hash(other) != config_module.config_hash(
        smoke_config
    )


# --- L8 · generator hidden state stays in the generator ---------------


def test_l8_no_hidden_generator_state_reaches_the_feature_table(
    smoke_features,
):
    columns = {name.lower() for name in smoke_features.columns}
    leaked = {name for name in generate.HIDDEN_STATE_FIELDS if name in columns}

    assert leaked == set()


def test_l8_no_hidden_generator_state_reaches_the_sample_table(smoke_samples):
    columns = {name.lower() for name in smoke_samples.columns}

    assert columns.isdisjoint(set(generate.HIDDEN_STATE_FIELDS))


def test_l8_the_hidden_state_list_names_the_behaviour_it_must_hide():
    hidden = set(generate.HIDDEN_STATE_FIELDS)

    assert {"archetype", "preference", "consistency"} <= hidden
    assert "resident_id" not in hidden


def test_l8_a_leaked_hidden_field_would_be_caught(smoke_features):
    doctored = smoke_features.copy()
    doctored["consistency"] = 1.0

    columns = {name.lower() for name in doctored.columns}
    leaked = {name for name in generate.HIDDEN_STATE_FIELDS if name in columns}

    assert leaked == {"consistency"}


# --- every shipped control is represented here -------------------------


def test_every_shipped_control_is_named_by_a_test():
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    named = {
        control
        for control in SHIPPED_CONTROLS
        if f"def test_{control}_" in source
    }

    assert named == set(SHIPPED_CONTROLS)
