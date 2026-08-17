"""The size decision: its arithmetic, its blindness, and its seal.

The decision this covers is the one that cannot be revisited, so most
of these tests are about refusing: an iteration count that does not
divide into whole updates, a first step that is not the declared full
training, a payload that carries a quality measure, a second decision
written over a sealed one.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from facility_prediction.llm import ladder
from facility_prediction.llm import settings as settings_module

LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

SECONDS_PER_ITERATION = 0.5
LATENCY_SECONDS = 4.0
RETRY_RESERVE = 0.05

# 1000 iterations at the rate above, so the third step is the first
# that fits and the two larger ones do not.
CAP_HOURS = 1200.0 / ladder.SECONDS_PER_HOUR


@pytest.fixture
def settings():
    return settings_module.load_settings(LLM_CONFIG)


def _with_cap(settings, cap_hours):
    fields = settings.model_dump()
    fields["compute"] = {**fields["compute"], "cap_hours": cap_hours}
    return settings_module.Settings(**fields)


def _projections(settings, prior_seconds=0.0):
    return ladder.project(
        ladder.build_steps(settings),
        prior_seconds=prior_seconds,
        seconds_per_iteration=SECONDS_PER_ITERATION,
        generation=0.0,
    )


def _decision(settings, cap_hours):
    capped = _with_cap(settings, cap_hours)
    projections = _projections(capped)
    selected = ladder.decide(projections, cap_hours * ladder.SECONDS_PER_HOUR)
    return ladder.build_decision(
        projections,
        selected,
        capped,
        {
            "seconds_per_iteration": SECONDS_PER_ITERATION,
            "seconds_per_row": LATENCY_SECONDS,
            "retry_reserve": RETRY_RESERVE,
        },
    )


def test_iteration_count_follows_rows_and_passes():
    counts = ladder.iters_for(2000, 3, 1)

    assert counts == 6000


def test_step_carries_its_optimizer_update_count(settings):
    step = ladder.build_step(1, 2000, 3, settings.tuning)

    assert step.iters == 6000
    assert step.optimizer_updates == 1500


def test_step_refuses_a_partial_optimizer_update(settings):
    fields = settings.tuning.model_dump()
    fields["grad_accumulation_steps"] = 7
    tuning = settings_module.Tuning(**fields)

    with pytest.raises(ladder.DecisionError, match="whole optimizer"):
        ladder.build_step(1, 100, 1, tuning)


def test_configured_ladder_starts_at_the_declared_full_training(settings):
    steps = ladder.build_steps(settings)

    assert steps[0].iters == settings.tier_b.iters
    assert steps[0].train_rows == settings.tier_b.train_rows


def test_ladder_sizes_never_grow(settings):
    iterations = [step.iters for step in ladder.build_steps(settings)]

    assert iterations == sorted(iterations, reverse=True)


def test_first_step_must_be_the_declared_full_training(settings):
    smaller = ladder.build_step(1, 1000, 1, settings.tuning)

    with pytest.raises(ladder.DecisionError, match="declared full"):
        ladder.check_declared(smaller, settings.tier_b, settings.tuning)


def test_declared_iteration_count_must_match_its_arithmetic(settings):
    fields = settings.tier_b.model_dump()
    fields["iters"] = 5999
    tier_b = settings_module.TierB(**fields)

    with pytest.raises(ladder.DecisionError, match="5999 iterations"):
        ladder.check_declared(
            ladder.build_step(1, 2000, 3, settings.tuning),
            tier_b,
            settings.tuning,
        )


def test_scored_generations_count_every_remaining_pass(settings):
    total = ladder.scored_generations(settings.compute, 500)

    assert total == 900


def test_generation_cost_carries_the_retry_allowance():
    seconds = ladder.generation_seconds(900, 4.0, 0.05)

    assert seconds == pytest.approx(900 * 4.0 * 1.05)


def test_the_first_step_that_fits_is_chosen(settings):
    selected = ladder.decide(_projections(settings), cap_seconds=1200.0)

    assert selected.step.position == 3
    assert selected.total_seconds <= 1200.0


def test_no_step_fitting_stops_the_branch(settings):
    with pytest.raises(ladder.LadderError, match="smallest pre-declared"):
        ladder.decide(_projections(settings), cap_seconds=10.0)


def test_prior_spend_can_exhaust_the_cap_on_its_own(settings):
    cap = settings.compute.cap_hours * ladder.SECONDS_PER_HOUR
    projections = _projections(settings, prior_seconds=cap)

    with pytest.raises(ladder.LadderError):
        ladder.decide(projections, cap_seconds=cap)


def test_decision_records_every_step_and_whether_it_fits(settings):
    payload = _decision(settings, CAP_HOURS)

    assert len(payload["steps"]) == len(settings.ladder)
    assert [step["fits"] for step in payload["steps"]] == [
        False,
        False,
        True,
        True,
    ]
    assert payload["selected"]["position"] == 3


def test_decision_refuses_a_quality_measure(settings):
    payload = _decision(settings, CAP_HOURS)
    payload["inputs"]["validation_loss"] = 1.0

    with pytest.raises(ladder.DecisionError, match="'loss'"):
        ladder.check_blind(payload)


def test_decision_accepts_a_hash_that_merely_reads_like_one(settings):
    payload = _decision(settings, CAP_HOURS)
    payload["inputs"]["pilot_record_sha256"] = "f1accf5c0" * 7

    ladder.check_blind(payload)


def test_sealed_decision_survives_an_identical_rewrite(settings, tmp_path):
    payload = _decision(settings, CAP_HOURS)
    path = tmp_path / "ladder_decision.json"

    first = ladder.write_once(payload, path)
    second = ladder.write_once(payload, path)

    assert first == second


def test_sealed_decision_refuses_a_different_one(settings, tmp_path):
    path = tmp_path / "ladder_decision.json"
    ladder.write_once(_decision(settings, CAP_HOURS), path)

    with pytest.raises(ladder.DecisionError, match="already sealed"):
        ladder.write_once(_decision(settings, 4.0), path)


def test_sealed_decision_keeps_the_first_selection(settings, tmp_path):
    path = tmp_path / "ladder_decision.json"
    ladder.write_once(_decision(settings, CAP_HOURS), path)

    sealed = json.loads(path.read_text(encoding="utf-8"))

    assert sealed["selected"]["position"] == 3


def test_the_decision_never_reads_the_training_run_it_measures():
    source = pathlib.Path(ladder.__file__).read_text(encoding="utf-8")

    assert "llm_train" not in source
    assert "val_loss" not in source
