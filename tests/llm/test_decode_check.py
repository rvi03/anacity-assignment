"""The fixture pass: what it draws, what it counts, what stops it.

The gate this covers is the one that decides whether the branch may
generate at all, so the tests are mostly about refusing: a fixture that
never parsed, a repeat that came back different, a draw that could not
be filled. The decode itself is injected; no model is loaded.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from facility_prediction import config as config_module
from facility_prediction.llm import constrained_decode, decode_check, serialize
from facility_prediction.llm import settings as settings_module

CONFIG = pathlib.Path("configs") / "default.yaml"
LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

LABELS = ["UNDER_287M", "M287_424"]
ROWS = 40
FIXTURES = 8
REPEATS = 3
LATENCY = 4


@pytest.fixture
def config():
    return config_module.load_config(CONFIG)


@pytest.fixture
def decoding():
    fields = settings_module.load_settings(LLM_CONFIG).decoding.model_dump()
    fields.update(
        fixtures=FIXTURES,
        repeat_fixtures=REPEATS,
        latency_fixtures=LATENCY,
    )
    return settings_module.Decoding(**fields)


@pytest.fixture
def rows(config):
    return [
        {
            "sample_id": f"s{index:03d}",
            "system": serialize.SYSTEM_INSTRUCTION,
            "prompt": f"prompt {index}",
            "target": _answer(config),
        }
        for index in range(ROWS)
    ]


def _answer(config, hour=9):
    return json.dumps(
        {
            serialize.FACILITY_FIELD: config.facility_names[0],
            serialize.USAGE_DAY_FIELD: "Monday",
            serialize.USAGE_HOUR_FIELD: hour,
            serialize.BUCKET_FIELD: LABELS[0],
        }
    )


def _completion(text, seconds=1.0):
    return constrained_decode.Completion(
        text=text,
        prompt_tokens=700,
        completion_tokens=30,
        prefill_tokens_per_second=900.0,
        decode_tokens_per_second=40.0,
        peak_memory_gib=3.0,
        seconds=seconds,
        finish_reason="stop",
    )


def _outcome(text, status=constrained_decode.VALID, seconds=1.0):
    fields = json.loads(text) if status == constrained_decode.VALID else None
    return constrained_decode.Outcome(
        status=status,
        fields=fields,
        completion=_completion(text, seconds),
        attempts=1,
        failure_stage=None if fields else constrained_decode.PARSE_STAGE,
        failure_reason=None if fields else "not JSON",
        semantic_invalid_reason=None,
    )


def _steady(config):
    return lambda _fixture: _outcome(_answer(config))


def _report(config, decoding, decode=None):
    fixtures = [
        decode_check.Fixture(f"s{index:03d}", "system", f"prompt {index}")
        for index in range(decoding.fixtures)
    ]
    results = decode_check.run_fixtures(
        fixtures, decode or _steady(config), decoding.repeat_fixtures
    )
    return decode_check.build_report(
        results, {"runtime": {"outlines": "1.3.3"}}, decoding, {"file": "x"}
    )


def test_the_draw_is_the_same_on_a_second_run(rows, config):
    first = decode_check.select_fixtures(rows, FIXTURES, config.seed)
    second = decode_check.select_fixtures(rows, FIXTURES, config.seed)

    assert [item.sample_id for item in first] == [
        item.sample_id for item in second
    ]


def test_the_draw_is_ordered_by_identifier(rows, config):
    drawn = decode_check.select_fixtures(rows, FIXTURES, config.seed)

    ids = [item.sample_id for item in drawn]
    assert ids == sorted(ids)


def test_a_draw_larger_than_the_file_is_refused(rows, config):
    with pytest.raises(decode_check.FixtureError, match="cannot draw"):
        decode_check.select_fixtures(rows, ROWS + 1, config.seed)


def test_missing_prompt_file_is_refused(tmp_path):
    with pytest.raises(decode_check.FixtureError, match="does not exist"):
        decode_check.read_rows(tmp_path / "valid.jsonl")


def test_only_the_leading_fixtures_are_decoded_twice(config, decoding):
    calls = []

    def decode(fixture):
        calls.append(fixture.sample_id)
        return _outcome(_answer(config))

    _report(config, decoding, decode)

    assert len(calls) == decoding.fixtures + decoding.repeat_fixtures


def test_a_clean_pass_clears_the_gate(config, decoding):
    report = _report(config, decoding)

    decode_check.check_gate(report, decoding)

    assert report["counts"]["valid"] == decoding.fixtures
    assert report["counts"]["failed"] == 0


def test_one_unparsed_fixture_stops_the_gate(config, decoding):
    def decode(fixture):
        if fixture.sample_id.endswith("3"):
            return _outcome("{", status=constrained_decode.FAILED)
        return _outcome(_answer(config))

    report = _report(config, decoding, decode)

    with pytest.raises(decode_check.FixtureError, match="never"):
        decode_check.check_gate(report, decoding)


def test_a_repeat_that_differs_stops_the_gate(config, decoding):
    seen = {"calls": 0}

    def decode(_fixture):
        seen["calls"] += 1
        return _outcome(_answer(config, hour=9 + seen["calls"] % 2))

    report = _report(config, decoding, decode)

    with pytest.raises(decode_check.FixtureError, match="not deterministic"):
        decode_check.check_gate(report, decoding)


def test_a_short_fixture_set_stops_the_gate(config, decoding):
    report = _report(config, decoding)
    report["counts"]["fixtures"] -= 1

    with pytest.raises(decode_check.FixtureError, match="were configured"):
        decode_check.check_gate(report, decoding)


def test_timing_covers_the_declared_sample_only(config, decoding):
    seconds = iter(range(1, decoding.fixtures * 3 + 1))

    def decode(_fixture):
        return _outcome(_answer(config), seconds=float(next(seconds)))

    report = _report(config, decoding, decode)

    assert report["timing"]["sampled_fixtures"] == decoding.latency_fixtures


def test_the_report_carries_a_measured_retry_allowance(config, decoding):
    report = _report(config, decoding)

    assert report["retry_reserve"] >= decoding.retry_reserve_floor
    assert report["evidence"] == "measured"


def test_semantic_invalid_answers_are_counted_and_listed(config, decoding):
    shut = (config.facilities[0].open_hour - 1) % 24

    def decode(_fixture):
        outcome = _outcome(_answer(config, hour=shut))
        return constrained_decode.Outcome(
            status=outcome.status,
            fields=outcome.fields,
            completion=outcome.completion,
            attempts=outcome.attempts,
            failure_stage=None,
            failure_reason=None,
            semantic_invalid_reason=constrained_decode.semantic_invalid_reason(
                outcome.fields, config
            ),
        )

    report = _report(config, decoding, decode)

    assert report["semantic_invalid_rate"] == 1.0
    assert len(report["semantic_invalid"]) == decoding.fixtures
    decode_check.check_gate(report, decoding)
