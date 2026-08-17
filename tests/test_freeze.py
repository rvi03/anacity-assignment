"""Tests the freeze and the holdout seal.

These two records are the only thing standing between "scored once" and
"scored until the number looked good", so the tests that matter are the
refusals. Each one below breaks the protocol in a different way and
asserts that the break is refused rather than warned about.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from facility_prediction import config as config_module
from facility_prediction.evaluation import freeze

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

SOURCES = {
    "facility": "catboost",
    "usage_weekday": "frequency_recency",
    "usage_hour": "catboost",
    "notification": "catboost",
}


@pytest.fixture(scope="module")
def config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture
def frozen(config) -> dict:
    return freeze.build_freeze(config, SOURCES, {"features_digest": "abc"})


# --- what a freeze records ---------------------------------------------


def test_a_freeze_records_the_config_hash(frozen, config):
    assert frozen["config_hash"] == config_module.config_hash(config)


def test_a_freeze_records_every_output_source(frozen):
    assert frozen["head_sources"] == SOURCES


def test_a_freeze_without_sources_is_refused(config):
    with pytest.raises(freeze.FreezeError, match="name the source"):
        freeze.build_freeze(config, {}, {})


def test_a_freeze_round_trips_through_disk(frozen, tmp_path):
    path = tmp_path / freeze.FREEZE_FILENAME

    freeze.write_freeze(frozen, path)

    assert freeze.read_freeze(path) == frozen


def test_reading_a_freeze_that_was_never_written_is_refused(tmp_path):
    with pytest.raises(freeze.FreezeError, match="no freeze at"):
        freeze.read_freeze(tmp_path / freeze.FREEZE_FILENAME)


# --- the refusals that matter ------------------------------------------


def test_a_configuration_that_moved_after_the_freeze_is_refused(frozen, config):
    moved = config.model_copy(update={"seed": config.seed + 1})

    with pytest.raises(freeze.FreezeError, match="changed after the freeze"):
        freeze.check_freeze_matches(frozen, moved)


def test_an_unchanged_configuration_passes(frozen, config):
    freeze.check_freeze_matches(frozen, config)


def test_a_second_scoring_is_refused(frozen, tmp_path):
    path = tmp_path / freeze.SEAL_FILENAME
    freeze.write_seal(
        freeze.build_seal(frozen, rows=100, headline={"selection_score": 0.2}),
        path,
    )

    with pytest.raises(freeze.FreezeError, match="already scored"):
        freeze.require_unsealed(path)


def test_an_unspent_holdout_may_be_scored(tmp_path):
    freeze.require_unsealed(tmp_path / freeze.SEAL_FILENAME)


def test_the_refusal_names_what_deleting_the_seal_would_cost(frozen, tmp_path):
    # The message is the control. If it does not say what is lost, the
    # seal is a speed bump rather than a rule.
    path = tmp_path / freeze.SEAL_FILENAME
    freeze.write_seal(freeze.build_seal(frozen, 100, {}), path)

    with pytest.raises(freeze.FreezeError) as caught:
        freeze.require_unsealed(path)

    assert "best-of-N" in str(caught.value)


# --- what a seal records -----------------------------------------------


def test_a_seal_carries_the_freeze_it_scored_against(frozen):
    seal = freeze.build_seal(frozen, rows=3596, headline={"score": 0.24})

    assert seal["config_hash"] == frozen["config_hash"]
    assert seal["head_sources"] == SOURCES
    assert seal["rows_scored"] == 3596


def test_a_seal_is_readable_json(frozen, tmp_path):
    path = tmp_path / freeze.SEAL_FILENAME
    seal = freeze.build_seal(frozen, 3596, {"score": 0.24})

    freeze.write_seal(seal, path)

    assert json.loads(path.read_text(encoding="utf-8")) == seal


def test_is_sealed_is_false_before_and_true_after(frozen, tmp_path):
    path = tmp_path / freeze.SEAL_FILENAME
    assert not freeze.is_sealed(path)

    freeze.write_seal(freeze.build_seal(frozen, 1, {}), path)

    assert freeze.is_sealed(path)
