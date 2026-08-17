"""Tests the stretch-variant budget and its report.

The variants themselves are just refits. What has to be right is the
accounting around them: the budget is a ceiling that stops the run at a
declared point, the report says which variants were never reached and
which were never built, and nothing here claims to have touched the
holdout or changed the shipped model.
"""

from __future__ import annotations

import pathlib

import pytest

from facility_prediction import config as config_module
from facility_prediction.evaluation import ablations

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


def _variant(priority: int, fits: int) -> ablations.Variant:
    return ablations.Variant(
        priority=priority,
        name=f"v{priority}",
        question="does it?",
        fits=fits,
        overrides={"depth": 4},
        heads=("facility",),
    )


# --- the budget ---------------------------------------------------------


def test_a_budget_that_covers_everything_runs_everything():
    declared = (_variant(1, 1), _variant(2, 1))

    chosen, skipped, planned = ablations.plan_run(10, declared)

    assert len(chosen) == 2
    assert skipped == []
    assert planned == 2


def test_the_run_stops_at_the_first_item_that_would_exceed_the_budget():
    declared = (_variant(1, 2), _variant(2, 5), _variant(3, 1))

    chosen, skipped, planned = ablations.plan_run(4, declared)

    # Priority 3 costs only 1 and would fit, but taking it would mean
    # the stopping point depends on cost rather than on the declared
    # order. It is skipped with priority 2.
    assert [item.name for item in chosen] == ["v1"]
    assert [item.name for item in skipped] == ["v2", "v3"]
    assert planned == 2


def test_priority_order_governs_not_declaration_order():
    declared = (_variant(3, 1), _variant(1, 1), _variant(2, 1))

    chosen, _, _ = ablations.plan_run(2, declared)

    assert [item.priority for item in chosen] == [1, 2]


def test_a_zero_budget_runs_nothing():
    chosen, skipped, planned = ablations.plan_run(0, (_variant(1, 1),))

    assert chosen == []
    assert len(skipped) == 1
    assert planned == 0


def test_the_shipped_variant_list_is_in_priority_order():
    priorities = [variant.priority for variant in ablations.DECLARED]

    assert priorities == sorted(priorities)


def test_every_shipped_variant_costs_at_least_one_fit():
    assert all(variant.fits >= 1 for variant in ablations.DECLARED)


def test_every_shipped_variant_names_the_heads_it_refits():
    assert all(variant.heads for variant in ablations.DECLARED)


# --- applying a variant -------------------------------------------------


def test_an_override_changes_only_what_it_names(config):
    varied = ablations._with_overrides(config, {"depth": 8})

    assert varied.catboost.depth == 8
    assert varied.catboost.learning_rate == config.catboost.learning_rate
    assert varied.seed == config.seed


def test_applying_an_override_leaves_the_frozen_config_untouched(config):
    before = config.catboost.depth

    ablations._with_overrides(config, {"depth": 8})

    assert config.catboost.depth == before


# --- the report ---------------------------------------------------------


@pytest.fixture
def completed() -> list[dict]:
    return [
        {
            "name": "v1",
            "priority": 1,
            "question": "does it?",
            "fits": 1,
            "overrides": {"depth": 8},
            "heads": ["facility"],
            "matches": {"facility": 0.40},
            "seconds": 1.0,
        }
    ]


def test_the_report_states_that_no_holdout_row_was_read(completed):
    payload = ablations.build_report(
        completed, [], 25, 1, {"facility": 0.35}, "validation"
    )

    assert payload["holdout_read"] is False
    assert payload["primary_artifacts_changed"] is False
    assert payload["split"] == "validation"


def test_a_variant_that_beats_the_frozen_head_is_reported(completed):
    payload = ablations.build_report(
        completed, [], 25, 1, {"facility": 0.35}, "validation"
    )

    improved = payload["improved_on_frozen"]
    assert len(improved) == 1
    assert improved[0]["delta"] == pytest.approx(0.05)


def test_a_variant_that_loses_is_not_reported_as_an_improvement(completed):
    payload = ablations.build_report(
        completed, [], 25, 1, {"facility": 0.45}, "validation"
    )

    assert payload["improved_on_frozen"] == []


def test_the_report_counts_planned_against_completed_fits(completed):
    payload = ablations.build_report(
        completed, [_variant(2, 3)], 25, 4, {}, "validation"
    )

    assert payload["budget"]["declared_fits"] == 25
    assert payload["budget"]["planned_fits"] == 4
    assert payload["budget"]["completed_fits"] == 1


def test_the_report_names_the_variant_it_stopped_before(completed):
    payload = ablations.build_report(
        completed, [_variant(2, 3)], 25, 4, {}, "validation"
    )

    assert "v2" in payload["budget"]["stopping_point"]


def test_a_complete_run_says_so(completed):
    payload = ablations.build_report(completed, [], 25, 1, {}, "validation")

    assert "every declared variant ran" in payload["budget"]["stopping_point"]


def test_the_report_names_what_was_never_built(completed):
    payload = ablations.build_report(completed, [], 25, 1, {}, "validation")

    # A budget that only counts what was convenient is not a budget.
    assert payload["not_built"] == ablations.NOT_BUILT
    assert payload["not_built"]


def test_writing_the_same_report_twice_gives_the_same_hash(completed, tmp_path):
    payload = ablations.build_report(completed, [], 25, 1, {}, "validation")

    first = ablations.write_report(payload, tmp_path / "a.json")
    second = ablations.write_report(payload, tmp_path / "b.json")

    assert first == second


# --- the wall-clock stop rule -------------------------------------------


def test_the_first_variant_is_never_projected_past_the_budget():
    # Nothing has been measured yet, so there is no projection to make.
    assert not ablations.would_exceed_time(0.0, 0, 3, minutes_budget=1.0)


def test_the_first_variant_can_overrun_and_that_is_recorded():
    # Documented limitation, not an oversight: with nothing measured
    # there is nothing to project, so variant one runs to completion
    # however long it takes. The report carries elapsed_seconds so the
    # overrun is visible.
    assert not ablations.would_exceed_time(99_999.0, 0, 1, minutes_budget=1.0)


def test_a_cheap_run_keeps_going():
    # One fit in six seconds; three more project to 24s, well inside
    # a one-minute budget.
    assert not ablations.would_exceed_time(6.0, 1, 3, minutes_budget=1.0)


def test_an_expensive_run_stops_before_the_next_variant():
    # One fit took 40s; three more project to 160s, past one minute.
    assert ablations.would_exceed_time(40.0, 1, 3, minutes_budget=1.0)


def test_the_projection_uses_measured_cost_per_fit():
    # Same elapsed time, twice the fits, so half the cost each — and
    # the same next variant now fits.
    assert ablations.would_exceed_time(40.0, 1, 2, minutes_budget=1.5)
    assert not ablations.would_exceed_time(40.0, 2, 2, minutes_budget=1.5)


def test_the_time_budget_is_not_a_frozen_config_key(config):
    # The configuration hash is the freeze. A stretch-only ceiling added
    # to it would invalidate that hash without being able to change a
    # single shipped number, so it lives in this module instead.
    assert not hasattr(config.catboost, "stretch_minutes_budget")
    assert ablations.DEFAULT_MINUTES_BUDGET > 0


def test_the_report_carries_a_time_stop_reason(completed):
    payload = ablations.build_report(
        completed,
        [_variant(2, 3)],
        25,
        4,
        {},
        "validation",
        elapsed_seconds=90.0,
        minutes_budget=1.0,
        stopping_point="stopped before 'v2': projected past the budget",
    )

    assert "projected past the budget" in payload["budget"]["stopping_point"]
    assert payload["budget"]["elapsed_seconds"] == 90.0
    assert payload["budget"]["minutes_budget"] == 1.0
