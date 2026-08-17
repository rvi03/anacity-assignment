"""The one full training run: which size it may train, and its draw.

No model trains here. The sealed ladder decision and the rendered
prompts are written by hand, so each refusal can be aimed at directly:
a decision that seals nothing, a step whose own projection does not
fit, an iteration count that contradicts its own rows, and an adapter
that comes back off disk different from the one the trainer saved.
"""

from __future__ import annotations

import pathlib

import pytest

from facility_prediction.llm import buckets, tier_b
from facility_prediction.llm import settings as settings_module

LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

LOW = "UNDER_60M"
HIGH = "OVER_60M"

ROWS = 40
DRAW = 12


@pytest.fixture
def tuning():
    return settings_module.load_settings(LLM_CONFIG).tuning


@pytest.fixture
def ladder():
    return [
        buckets.Bucket(
            label=LOW, lower=1.0, upper=61.0, representative=30.0, train_rows=20
        ),
        buckets.Bucket(
            label=HIGH,
            lower=61.0,
            upper=None,
            representative=90.0,
            train_rows=20,
        ),
    ]


@pytest.fixture
def prompts():
    return [
        {"sample_id": f"S{index:04d}", "prompt": "p", "system": "s"}
        for index in range(ROWS)
    ]


@pytest.fixture
def labels():
    return {
        f"S{index:04d}": LOW if index % 2 else HIGH for index in range(ROWS)
    }


def step(train_rows=DRAW, epoch_equivalent=1, iters=DRAW, fits=True):
    return {
        "position": 4,
        "train_rows": train_rows,
        "epoch_equivalent": epoch_equivalent,
        "iters": iters,
        "fits": fits,
    }


def test_selected_step_is_returned_when_it_fits():
    assert tier_b.selected_step({"selected": step()})["train_rows"] == DRAW


def test_a_decision_that_seals_nothing_is_refused():
    with pytest.raises(tier_b.TierBError, match="selects no step"):
        tier_b.selected_step({"steps": [step()]})


def test_a_step_that_does_not_fit_the_cap_is_refused():
    with pytest.raises(tier_b.TierBError, match="does not fit"):
        tier_b.selected_step({"selected": step(fits=False)})


def test_iteration_count_must_follow_from_rows_and_passes(tuning):
    assert tier_b.declared_iters(step(), tuning) == DRAW


def test_an_iteration_count_that_contradicts_its_rows_is_refused(tuning):
    with pytest.raises(tier_b.TierBError, match="imply"):
        tier_b.declared_iters(step(iters=DRAW + 1), tuning)


def test_more_passes_multiply_the_iterations(tuning):
    assert (
        tier_b.declared_iters(step(epoch_equivalent=3, iters=DRAW * 3), tuning)
        == DRAW * 3
    )


def test_the_draw_takes_the_sealed_number_of_rows(prompts, labels):
    selection = tier_b.select(prompts, labels, step(), seed=1)
    assert len(selection.rows) == DRAW


def test_the_draw_is_seeded_and_repeats(prompts, labels):
    first = tier_b.select(prompts, labels, step(), seed=7)
    second = tier_b.select(prompts, labels, step(), seed=7)
    assert [row["sample_id"] for row in first.rows] == [
        row["sample_id"] for row in second.rows
    ]


def test_a_different_seed_draws_differently(prompts, labels):
    first = tier_b.select(prompts, labels, step(), seed=7)
    second = tier_b.select(prompts, labels, step(), seed=8)
    assert [row["sample_id"] for row in first.rows] != [
        row["sample_id"] for row in second.rows
    ]


def test_the_draw_takes_no_floor_per_label(prompts, labels):
    """The pilot guarantees a floor per label; this draw must not."""
    tiny = tier_b.select(prompts, labels, step(train_rows=1, iters=1), seed=3)
    assert len(tiny.support) == 1


def test_asking_for_more_rows_than_exist_is_refused(prompts, labels):
    with pytest.raises(tier_b.TierBError, match="cannot draw"):
        tier_b.select(
            prompts, labels, step(train_rows=ROWS + 1, iters=ROWS + 1), seed=1
        )


def test_the_draw_record_names_its_rows_and_hashes_them(
    prompts, labels, ladder
):
    selection = tier_b.select(prompts, labels, step(), seed=5)
    record = tier_b.draw_record(selection, step(), 5, ladder)
    assert record["train_rows"] == DRAW
    assert len(record["sample_ids"]) == DRAW
    assert record["min_rows_per_label"] == 0
    assert record["ladder_position"] == 4
    assert len(record["sample_ids_sha256"]) == 64


def test_the_record_hash_changes_with_the_rows(prompts, labels, ladder):
    one = tier_b.draw_record(
        tier_b.select(prompts, labels, step(), seed=5), step(), 5, ladder
    )
    other = tier_b.draw_record(
        tier_b.select(prompts, labels, step(), seed=6), step(), 6, ladder
    )
    assert one["sample_ids_sha256"] != other["sample_ids_sha256"]


def test_an_unrepresented_label_is_recorded_not_backfilled(
    prompts, labels, ladder
):
    selection = tier_b.select(
        prompts, labels, step(train_rows=1, iters=1), seed=3
    )
    record = tier_b.draw_record(selection, step(), 3, ladder)
    assert len(record["unrepresented"]) == 1
    assert len(record["sample_ids"]) == 1


def test_a_matching_reload_passes():
    saved = {"parameters": 224, "adapted_blocks": 16, "sha256": "a" * 64}
    tier_b.check_reload(dict(saved), saved)


@pytest.mark.parametrize(
    "key, value",
    [
        ("parameters", 1),
        ("adapted_blocks", 1),
        ("sha256", "b" * 64),
    ],
)
def test_a_reload_that_differs_is_refused(key, value):
    saved = {"parameters": 224, "adapted_blocks": 16, "sha256": "a" * 64}
    reloaded = {**saved, key: value}
    with pytest.raises(tier_b.TierBError, match=key):
        tier_b.check_reload(reloaded, saved)
