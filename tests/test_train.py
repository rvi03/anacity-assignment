"""Tests the four CatBoost heads' contract.

Covers what training promises: every head resolves the settings the
configuration declares, the pinned bootstrap reaches the head it was
pinned for, the iteration count is the one that runs — no evaluation
set, no checkpoint, no early stop — a saved head reloads and predicts
identically, and the metrics record carries what a reviewer needs to
reproduce the fit.

The smoke configuration's twenty iterations are used throughout: this
file tests the training contract, never model quality.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import zoneinfo

import catboost
import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples, split
from facility_prediction.features import features
from facility_prediction.models import train

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


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
def training_rows(smoke_samples, smoke_features, smoke_config):
    cutoffs = split.compute_cutoffs(smoke_samples, smoke_config)
    labelled = split.assign_split(smoke_samples, cutoffs)
    keep = set(
        labelled.loc[
            labelled[split.SPLIT_COLUMN] == split.TRAIN,
            split.SAMPLE_ID_COLUMN,
        ]
    )
    return smoke_features.loc[smoke_features["sample_id"].isin(keep)]


@pytest.fixture(scope="module")
def heads(training_rows, smoke_samples, smoke_config):
    return train.fit(training_rows, smoke_samples, smoke_config)


# --- resolved settings ------------------------------------------------


def test_every_declared_head_is_fitted(heads):
    assert tuple(heads) == train.HEAD_NAMES


def test_the_notification_classifier_takes_its_own_bootstrap(smoke_config):
    params = train.head_params(smoke_config, train.NOTIFICATION)

    assert params["bootstrap_type"] == (
        smoke_config.catboost.bootstrap.notification.type
    )


def test_every_classifier_takes_the_classifier_bootstrap(smoke_config):
    declared = smoke_config.catboost.bootstrap.classifiers

    for name in (train.FACILITY, train.USAGE_WEEKDAY, train.USAGE_HOUR):
        params = train.head_params(smoke_config, name)

        assert params["bootstrap_type"] == declared.type
        assert params["subsample"] == declared.subsample


def test_a_bootstrap_without_a_subsample_declares_none(smoke_config):
    params = train.head_params(smoke_config, train.NOTIFICATION)

    assert "subsample" not in params


def test_every_head_is_a_multiclass_classifier(smoke_config):
    losses = {
        name: train.head_params(smoke_config, name)["loss_function"]
        for name in train.HEAD_NAMES
    }

    assert set(losses.values()) == {"MultiClass"}


def test_the_default_configuration_searches_iterations_under_a_ceiling():
    config = config_module.load_config(DEFAULT_PATH)

    assert config.catboost.iteration_search is True
    assert config.catboost.iterations == 1500
    assert 0.0 < config.catboost.inner_validation_frac < 0.5
    assert config.catboost.use_best_model is False


def test_each_head_takes_its_own_seed(smoke_config):
    seeds = {train.head_seed(smoke_config, name) for name in train.HEAD_NAMES}

    assert len(seeds) == len(train.HEAD_NAMES)


def test_seeds_collapse_to_the_root_when_configuration_says_so(smoke_config):
    fixed = smoke_config.model_copy(
        update={
            "catboost": smoke_config.catboost.model_copy(
                update={"random_seed_from_root": False}
            )
        }
    )

    seeds = {train.head_seed(fixed, name) for name in train.HEAD_NAMES}

    assert seeds == {fixed.seed}


def test_an_unknown_head_is_rejected(smoke_config):
    with pytest.raises(train.TrainingError, match="unknown head"):
        train.head_params(smoke_config, "weather")


def test_notification_bucket_representative_matches_finite_bucket(
    smoke_config,
):
    edges = train.notification_bucket_edges(smoke_config)
    labels = train.notification_bucket_labels(
        pd.Series([edges[1], edges[2] - 0.01]), smoke_config
    )
    predicted = train.notification_bucket_representatives(
        labels.to_numpy(), smoke_config
    )

    ratio = smoke_config.evaluation.notification_match_ratio
    assert predicted[0] / edges[1] <= ratio
    assert edges[2] / predicted[1] <= ratio


# --- no early stopping ------------------------------------------------


def test_every_head_keeps_the_configured_iteration_count(heads, smoke_config):
    counts = {name: int(head.model.tree_count_) for name, head in heads.items()}

    assert set(counts.values()) == {smoke_config.catboost.iterations}


def test_a_head_that_stopped_early_is_rejected(heads, smoke_config):
    fitted = heads[train.FACILITY]
    claimed = dataclasses.replace(
        fitted,
        params={**fitted.params, "iterations": fitted.params["iterations"] + 1},
    )

    with pytest.raises(train.TrainingError, match="stop early"):
        train.check_no_early_stopping(claimed, smoke_config)


def test_no_head_is_allowed_to_use_a_best_model(heads):
    chosen = {head.params["use_best_model"] for head in heads.values()}

    assert chosen == {False}


def test_fitting_without_rows_is_rejected(
    smoke_features, smoke_samples, smoke_config
):
    empty = train.build_design(
        train.FACILITY,
        smoke_features.iloc[0:0],
        train.align_targets(smoke_features.iloc[0:0], smoke_samples),
        smoke_config,
    )

    with pytest.raises(train.TrainingError, match="without training rows"):
        train.fit_head(train.FACILITY, empty, smoke_config)


# --- what reaches the model -------------------------------------------


def test_identifiers_never_reach_the_model(smoke_features, smoke_config):
    matrix = train.model_matrix(smoke_features, smoke_config)

    assert set(features.IDENTIFIER_COLUMNS).isdisjoint(matrix.columns)


def test_safe_resident_profile_reaches_the_model(smoke_features, smoke_config):
    matrix = train.model_matrix(smoke_features, smoke_config)

    assert "resident_profile_id" in matrix.columns


def test_no_target_column_reaches_the_model(smoke_features, smoke_config):
    matrix = train.model_matrix(smoke_features, smoke_config)

    assert features.denylist_violations(matrix.columns) == ()


def test_a_feature_table_missing_a_declared_column_is_rejected(
    smoke_features, smoke_config
):
    short = smoke_features.drop(columns=["origin_hour"])

    with pytest.raises(train.TrainingError, match="missing declared columns"):
        train.model_matrix(short, smoke_config)


def test_categoricals_reach_catboost_as_text(smoke_features, smoke_config):
    matrix = train.model_matrix(smoke_features, smoke_config)

    for name in features.categorical_feature_names(smoke_config):
        assert matrix[name].map(lambda value: isinstance(value, str)).all()


def test_targets_are_aligned_to_the_feature_rows(smoke_features, smoke_samples):
    shuffled = smoke_samples.sample(frac=1.0, random_state=0)

    aligned = train.align_targets(smoke_features, shuffled)

    expected = smoke_samples.set_index("sample_id").loc[
        smoke_features["sample_id"], "target_facility_id"
    ]
    assert list(aligned["target_facility_id"]) == list(expected)


def test_a_feature_row_without_a_target_is_rejected(
    smoke_features, smoke_samples
):
    short = smoke_samples.iloc[1:]

    with pytest.raises(train.TrainingError, match="no target row"):
        train.align_targets(smoke_features, short)


def test_a_sample_table_missing_a_target_is_rejected(
    smoke_features, smoke_samples
):
    short = smoke_samples.drop(columns=["target_usage_hour"])

    with pytest.raises(train.TrainingError, match="missing target columns"):
        train.align_targets(smoke_features, short)


# --- predictions ------------------------------------------------------


def test_predictions_cover_every_row_with_every_output(
    heads, smoke_features, smoke_config
):
    predicted = train.predict(
        train.estimators(heads), smoke_features, smoke_config
    )

    assert len(predicted) == len(smoke_features)
    assert set(predicted.columns) == {
        "sample_id",
        *train.PREDICTION_COLUMNS.values(),
    }


def test_select_head_sources_uses_candidate_order_to_break_ties():
    reports = {
        "baseline": {
            "matches": {
                train.FACILITY: 0.3,
                train.USAGE_WEEKDAY: 0.2,
                train.USAGE_HOUR: 0.2,
                train.NOTIFICATION: 0.1,
            }
        },
        "catboost": {
            "matches": {
                train.FACILITY: 0.3,
                train.USAGE_WEEKDAY: 0.1,
                train.USAGE_HOUR: 0.4,
                train.NOTIFICATION: 0.05,
            }
        },
    }

    selected = train.select_head_sources(reports, ("baseline", "catboost"))

    assert selected == {
        train.FACILITY: "baseline",
        train.USAGE_WEEKDAY: "baseline",
        train.USAGE_HOUR: "catboost",
        train.NOTIFICATION: "baseline",
    }


def test_compose_selected_predictions_takes_each_selected_head():
    baseline = pd.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "predicted_facility_id": ["Gym", "Pool"],
            "predicted_usage_weekday": [1, 2],
            "predicted_usage_hour": [7, 8],
            "predicted_delay_minutes": [90.0, 120.0],
        }
    )
    catboost = baseline.assign(
        predicted_facility_id=["Pool", "Gym"],
        predicted_usage_weekday=[3, 4],
        predicted_usage_hour=[9, 10],
        predicted_delay_minutes=[180.0, 240.0],
    )

    selected = train.compose_selected_predictions(
        {"baseline": baseline, "catboost": catboost},
        {
            train.FACILITY: "baseline",
            train.USAGE_WEEKDAY: "catboost",
            train.USAGE_HOUR: "baseline",
            train.NOTIFICATION: "catboost",
        },
    )

    assert selected.to_dict("list") == {
        "sample_id": ["S1", "S2"],
        "predicted_facility_id": ["Gym", "Pool"],
        "predicted_usage_weekday": [3, 4],
        "predicted_usage_hour": [7, 8],
        "predicted_delay_minutes": [180.0, 240.0],
    }


def test_predicted_facilities_come_from_the_catalog(
    heads, smoke_features, smoke_config
):
    predicted = train.predict(
        train.estimators(heads), smoke_features, smoke_config
    )

    assert set(predicted["predicted_facility_id"]) <= set(
        smoke_config.facility_names
    )


def test_predicted_calendar_values_are_in_range(
    heads, smoke_features, smoke_config
):
    predicted = train.predict(
        train.estimators(heads), smoke_features, smoke_config
    )

    assert predicted["predicted_usage_weekday"].between(0, 6).all()
    assert predicted["predicted_usage_hour"].between(0, 23).all()


def test_predicting_twice_gives_the_same_answers(
    heads, smoke_features, smoke_config
):
    first = train.predict(train.estimators(heads), smoke_features, smoke_config)
    second = train.predict(
        train.estimators(heads), smoke_features, smoke_config
    )

    pd.testing.assert_frame_equal(first, second)


def test_predicting_without_a_head_is_rejected(
    heads, smoke_features, smoke_config
):
    partial = {
        name: head for name, head in heads.items() if name != train.NOTIFICATION
    }

    with pytest.raises(train.TrainingError, match="no fitted head"):
        train.predict(partial, smoke_features, smoke_config)


# --- saving and reloading ---------------------------------------------


def test_a_saved_head_reloads_and_predicts_identically(
    heads, smoke_features, smoke_config, tmp_path
):
    train.save(heads, tmp_path)
    reloaded = train.load(tmp_path, smoke_config)

    matrix = train.model_matrix(smoke_features, smoke_config)
    pool = catboost.Pool(
        data=matrix,
        cat_features=list(features.categorical_feature_names(smoke_config)),
    )
    for name, head in heads.items():
        assert (reloaded[name].predict(pool) == head.model.predict(pool)).all()


def test_a_missing_saved_head_is_reported(smoke_config, tmp_path):
    with pytest.raises(train.TrainingError, match="no saved model"):
        train.load(tmp_path, smoke_config)


# --- the metrics record -----------------------------------------------


def test_the_metrics_record_carries_every_resolved_setting(heads, smoke_config):
    payload = train.build_metrics(heads, smoke_config, {}, len(heads))

    for name in train.HEAD_NAMES:
        params = payload["training"]["heads"][name]["params"]
        assert params["iterations"] == smoke_config.catboost.iterations
        assert params["use_best_model"] is False
        assert params["thread_count"] == smoke_config.catboost.thread_count
        assert params["random_seed"] == train.head_seed(smoke_config, name)


def test_the_metrics_record_names_the_catboost_version(heads, smoke_config):
    payload = train.build_metrics(heads, smoke_config, {}, len(heads))

    assert payload["versions"]["catboost"] == catboost.__version__


def test_the_metrics_record_states_that_nothing_stopped_early(
    heads, smoke_config
):
    training = train.build_metrics(heads, smoke_config, {}, 0)["training"]

    assert training["early_stopping"] is False
    assert training["evaluation_set_passed"] is False
    assert training["use_best_model"] is False


def test_the_metrics_record_reports_the_stretch_fit_budget(heads, smoke_config):
    payload = train.build_metrics(heads, smoke_config, {}, 0)

    assert payload["fits"]["primary"] == len(train.HEAD_NAMES)
    assert (
        payload["fits"]["stretch_budget"]
        == smoke_config.catboost.stretch_fit_budget
    )
    assert payload["fits"]["stretch_completed"] == 0


def test_the_metrics_record_round_trips_as_json(heads, smoke_config, tmp_path):
    path = tmp_path / "metrics.json"
    payload = train.build_metrics(heads, smoke_config, {"seed": 1}, 10)

    train.write_metrics(payload, path)

    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_flat_params_name_every_head_and_setting(heads):
    flat = train.flat_params(heads)

    assert (
        flat[f"{train.FACILITY}.iterations"]
        == (heads[train.FACILITY].params["iterations"])
    )
    assert all("." in key for key in flat)
