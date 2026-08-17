"""Prompt rendering, and the three things that must not happen to it.

Golden: the same input renders the same bytes.
Sealed: no holdout row reaches a training file.
Future-perturbation: a later booking cannot change an earlier prompt.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.llm import (
    buckets,
    llm_data,
    prompt_features,
    serialize,
)
from facility_prediction.llm import settings as settings_module

SMOKE_CONFIG = pathlib.Path("configs") / "smoke.yaml"
LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"
ZONE = "Asia/Kolkata"
ORIGIN = pd.Timestamp("2025-03-01 10:00", tz=ZONE)
LATER = pd.Timestamp("2025-06-01 10:00", tz=ZONE)


@pytest.fixture(scope="module")
def config():
    return config_module.load_config(SMOKE_CONFIG)


@pytest.fixture(scope="module")
def settings():
    return settings_module.load_settings(LLM_CONFIG)


@pytest.fixture
def ladder(settings):
    delays = pd.Series([30.0, 300.0, 3000.0, 30000.0, 300000.0] * 200)
    built, _ = buckets.build(settings.notification_buckets, delays, 1.25)
    return built


def _booking(index, resident, facility, booked, used):
    return {
        "booking_id": f"b-{index}",
        "resident_id": resident,
        "facility_id": facility,
        "booking_timestamp": pd.Timestamp(booked, tz=ZONE),
        "usage_timestamp": pd.Timestamp(used, tz=ZONE),
    }


@pytest.fixture
def bookings():
    return pd.DataFrame(
        [
            _booking(1, "r-1", "Gym", "2025-01-05 18:00", "2025-01-06 07:00"),
            _booking(2, "r-1", "Gym", "2025-01-12 18:30", "2025-01-13 07:00"),
            _booking(3, "r-1", "Pool", "2025-02-01 09:00", "2025-02-02 18:00"),
            # after the origin: never visible to a prompt built at it
            _booking(4, "r-1", "Tennis", "2025-04-01 09:00", "2025-04-02 8:00"),
            _booking(
                5, "r-2", "Yoga Room", "2025-01-09 07:00", "2025-01-10 6:00"
            ),
        ]
    )


def _context(bookings, ladder, config, settings, origin=ORIGIN):
    history = prompt_features.past_bookings(bookings, "r-1", origin)
    return serialize.PromptContext(
        origin=origin,
        facilities=config.facility_names,
        ladder=ladder,
        summary={
            "total_bookings": len(history),
            "days_since_last_booking": 7.5,
            "facility_counts": {"Gym": 2, "Pool": 1},
            "preferred_usage_hour": prompt_features.preferred_usage_hour(
                history
            ),
        },
        events=prompt_features.recent_events(
            history, settings.prompt.recent_events
        ),
        decimals=settings.prompt.float_decimals,
    )


def test_identical_input_renders_identical_bytes(
    bookings, ladder, config, settings
):
    first = serialize.render_prompt(
        _context(bookings, ladder, config, settings)
    )
    second = serialize.render_prompt(
        _context(bookings, ladder, config, settings)
    )

    assert first == second


def test_a_future_booking_cannot_change_an_earlier_prompt(
    bookings, ladder, config, settings
):
    before = serialize.render_prompt(
        _context(bookings, ladder, config, settings)
    )

    perturbed = bookings.copy()
    perturbed.loc[perturbed["booking_id"] == "b-4", "facility_id"] = (
        "Basketball"
    )
    perturbed.loc[perturbed["booking_id"] == "b-4", "usage_timestamp"] = (
        pd.Timestamp("2025-05-05 20:00", tz=ZONE)
    )
    extra = pd.DataFrame(
        [_booking(6, "r-1", "Clubhouse", "2025-09-01 09:00", "2025-09-02 9:00")]
    )
    perturbed = pd.concat([perturbed, extra], ignore_index=True)

    after = serialize.render_prompt(
        _context(perturbed, ladder, config, settings)
    )

    assert after == before


def test_history_stops_at_the_origin(bookings):
    history = prompt_features.past_bookings(bookings, "r-1", ORIGIN)

    assert (history["booking_timestamp"] <= ORIGIN).all()
    assert "b-4" not in set(history["booking_id"])


def test_a_naive_origin_is_refused(bookings):
    with pytest.raises(ValueError, match="timezone-aware"):
        prompt_features.past_bookings(
            bookings, "r-1", pd.Timestamp("2025-03-01 10:00")
        )


def test_truncation_drops_the_oldest_events_only(bookings):
    history = prompt_features.past_bookings(bookings, "r-1", LATER)

    kept = prompt_features.recent_events(history, 2)

    assert len(kept) == 2
    assert kept[-1].facility == "Tennis"
    assert kept[0].facility == "Pool"


def test_the_first_event_has_no_gap(bookings):
    history = prompt_features.past_bookings(bookings, "r-1", ORIGIN)

    events = prompt_features.recent_events(history, 16)

    assert events[0].gap_days is None
    assert events[1].gap_days == pytest.approx(7.0208, abs=1e-3)


def test_no_identifier_reaches_the_prompt(bookings, ladder, config, settings):
    prompt = serialize.render_prompt(
        _context(bookings, ladder, config, settings)
    )

    assert "r-1" not in prompt
    assert "b-1" not in prompt


def test_every_allowed_value_is_rendered_from_config(
    bookings, ladder, config, settings
):
    prompt = serialize.render_prompt(
        _context(bookings, ladder, config, settings)
    )

    for name in config.facility_names:
        assert name in prompt
    for bucket in ladder:
        assert bucket.label in prompt


def test_an_absent_value_renders_as_none_not_as_a_number(
    bookings, ladder, config, settings
):
    context = _context(bookings, ladder, config, settings)
    empty = serialize.PromptContext(
        origin=context.origin,
        facilities=context.facilities,
        ladder=context.ladder,
        summary={"days_since_last_booking": float("nan"), "median": None},
        events=context.events,
        decimals=context.decimals,
    )

    prompt = serialize.render_prompt(empty)

    assert "days_since_last_booking=none" in prompt
    assert "nan" not in prompt


def test_the_target_carries_the_four_fields_in_contract_order():
    rendered = serialize.render_target("Gym", 2, 7, "M287_424")

    assert rendered == (
        '{"facility": "Gym", "usage_day": "Wednesday", "usage_hour": 7, '
        '"notification_delay_bucket": "M287_424"}'
    )


def test_an_hour_outside_the_day_is_refused():
    with pytest.raises(ValueError, match="outside the day"):
        serialize.render_target("Gym", 2, 24, "M287_424")


def test_the_schema_enumerates_config_values_only(ladder, config):
    schema = serialize.output_schema(
        config.facility_names, [b.label for b in ladder]
    )

    properties = schema["properties"]
    assert properties["facility"]["enum"] == list(config.facility_names)
    assert properties["notification_delay_bucket"]["enum"] == [
        b.label for b in ladder
    ]
    assert schema["additionalProperties"] is False


def test_the_prompt_hash_changes_with_the_template(
    bookings, ladder, config, settings
):
    prompt = serialize.render_prompt(
        _context(bookings, ladder, config, settings)
    )

    same = serialize.template_hash(1, prompt)
    edited = serialize.template_hash(1, prompt + " ")
    bumped = serialize.template_hash(2, prompt)

    assert same == serialize.template_hash(1, prompt)
    assert edited != same
    assert bumped != same


def test_a_holdout_row_in_a_training_file_is_refused():
    splits = pd.DataFrame(
        {"sample_id": ["s-1", "s-2"], "split": ["train", "test"]}
    )
    rows = [{"sample_id": "s-1"}, {"sample_id": "s-2"}]

    with pytest.raises(llm_data.DataError, match="holdout"):
        llm_data.check_sealed(rows, splits)


def test_only_the_train_and_validation_splits_are_written():
    assert set(llm_data.SPLIT_FILES) == {"train", "validation"}


def test_an_exact_duplicate_pair_is_dropped_once():
    rows = [
        {"sample_id": "a", "prompt": "p", "target": "t"},
        {"sample_id": "b", "prompt": "p", "target": "t"},
        {"sample_id": "c", "prompt": "p", "target": "u"},
    ]

    kept, dropped = llm_data.deduplicate(rows)

    assert dropped == 1
    assert [row["sample_id"] for row in kept] == ["a", "c"]


def test_a_written_file_reproduces_its_hash(tmp_path):
    rows = [{"sample_id": "a", "prompt": "p", "target": "t"}]

    first = llm_data.write_jsonl(rows, tmp_path / "a.jsonl")
    second = llm_data.write_jsonl(rows, tmp_path / "b.jsonl")

    assert first == second


def test_the_change_flag_needs_enough_history():
    history = pd.DataFrame({"facility_id": ["Gym"] * 5})

    flags = prompt_features.behaviour_change(
        history, ("Gym", "Pool"), 20, 10, 0.25
    )

    assert flags[prompt_features.INSUFFICIENT] == 1
    assert flags[prompt_features.CHANGED] == 0


def test_a_switched_resident_is_flagged_as_changed():
    history = pd.DataFrame({"facility_id": ["Gym"] * 15 + ["Pool"] * 10})

    flags = prompt_features.behaviour_change(
        history, ("Gym", "Pool"), 20, 10, 0.25
    )

    assert flags[prompt_features.CHANGED] == 1
    assert flags[prompt_features.INSUFFICIENT] == 0


def test_a_steady_resident_is_not_flagged():
    history = pd.DataFrame({"facility_id": ["Gym"] * 25})

    flags = prompt_features.behaviour_change(
        history, ("Gym", "Pool"), 20, 10, 0.25
    )

    assert flags[prompt_features.CHANGED] == 0


def test_the_distance_between_identical_distributions_is_zero():
    assert (
        prompt_features.jensen_shannon_distance([0.5, 0.5], [0.5, 0.5]) == 0.0
    )


def test_the_distance_between_disjoint_distributions_is_one():
    assert prompt_features.jensen_shannon_distance(
        [1.0, 0.0], [0.0, 1.0]
    ) == pytest.approx(1.0)
