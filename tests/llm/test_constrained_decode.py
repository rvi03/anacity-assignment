"""Shape validation, the retry path, and the typed failure row.

None of these tests loads a model. The generation call is injected, so
what is exercised here is the contract around it: what counts as a
usable answer, what happens to one that is not, and what a run has to
carry to be reproducible. The stack itself is checked by the fixture
pass, which needs a GPU and is not a test.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from facility_prediction import config as config_module
from facility_prediction.llm import constrained_decode, serialize
from facility_prediction.llm import settings as settings_module

CONFIG = pathlib.Path("configs") / "default.yaml"
LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

LABELS = ["UNDER_287M", "M287_424", "OVER_144000M"]


@pytest.fixture
def config():
    return config_module.load_config(CONFIG)


@pytest.fixture
def settings():
    return settings_module.load_settings(LLM_CONFIG)


@pytest.fixture
def schema(config):
    return serialize.output_schema(config.facility_names, LABELS)


def completion(text, seconds=1.0):
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


def answer(config, hour=9, label="M287_424"):
    return json.dumps(
        {
            serialize.FACILITY_FIELD: config.facility_names[0],
            serialize.USAGE_DAY_FIELD: "Monday",
            serialize.USAGE_HOUR_FIELD: hour,
            serialize.BUCKET_FIELD: label,
        }
    )


def replies(*texts):
    remaining = list(texts)

    def generate():
        return completion(remaining.pop(0))

    return generate


def test_a_well_formed_answer_parses_into_the_four_fields(config, schema):
    fields = constrained_decode.parse_fields(answer(config), schema)

    assert list(fields) == list(serialize.TARGET_FIELDS)


def test_text_that_is_not_json_is_rejected_at_the_parse_stage(schema):
    with pytest.raises(constrained_decode.DecodeError) as raised:
        constrained_decode.parse_fields("the gym, probably", schema)

    assert raised.value.stage == constrained_decode.PARSE_STAGE


def test_a_missing_field_is_rejected(config, schema):
    payload = json.loads(answer(config))
    del payload[serialize.USAGE_HOUR_FIELD]

    with pytest.raises(constrained_decode.DecodeError) as raised:
        constrained_decode.parse_fields(json.dumps(payload), schema)

    assert raised.value.stage == constrained_decode.SCHEMA_STAGE


def test_an_extra_field_is_rejected(config, schema):
    payload = json.loads(answer(config))
    payload["confidence"] = 0.9

    with pytest.raises(constrained_decode.DecodeError) as raised:
        constrained_decode.parse_fields(json.dumps(payload), schema)

    assert "confidence" in raised.value.reason


def test_a_facility_outside_the_catalog_is_rejected(config, schema):
    payload = json.loads(answer(config))
    payload[serialize.FACILITY_FIELD] = "Rooftop Bar"

    with pytest.raises(constrained_decode.DecodeError) as raised:
        constrained_decode.parse_fields(json.dumps(payload), schema)

    assert raised.value.stage == constrained_decode.SCHEMA_STAGE


def test_a_label_outside_the_frozen_ladder_is_rejected(config, schema):
    payload = json.loads(answer(config))
    payload[serialize.BUCKET_FIELD] = "M99999_100000"

    with pytest.raises(constrained_decode.DecodeError) as raised:
        constrained_decode.parse_fields(json.dumps(payload), schema)

    assert raised.value.stage == constrained_decode.SCHEMA_STAGE


def test_a_rejected_completion_is_retried_once_and_can_succeed(config, schema):
    outcome = constrained_decode.decode_one(
        replies("{", answer(config)), schema, config, retries=1
    )

    assert outcome.status == constrained_decode.VALID
    assert outcome.attempts == 2


def test_a_second_failure_becomes_a_typed_failure_row(config, schema):
    outcome = constrained_decode.decode_one(
        replies("{", "still not json"), schema, config, retries=1
    )

    assert outcome.status == constrained_decode.FAILED
    assert outcome.fields is None
    assert outcome.failure_stage == constrained_decode.PARSE_STAGE
    assert outcome.failure_reason


def test_a_failed_row_carries_no_substituted_answer(config, schema):
    outcome = constrained_decode.decode_one(
        replies("nope", "nope"), schema, config, retries=1
    )

    assert outcome.fields is None
    assert outcome.completion.text == "nope"


def test_an_hour_outside_the_opening_times_is_flagged_not_repaired(config):
    facility = config.facilities[0]
    shut = (facility.open_hour - 1) % 24
    fields = json.loads(answer(config, hour=shut))

    reason = constrained_decode.semantic_invalid_reason(fields, config)

    assert reason is not None
    assert fields[serialize.USAGE_HOUR_FIELD] == shut


def test_an_hour_inside_the_opening_times_is_not_flagged(config):
    facility = config.facilities[0]
    fields = json.loads(answer(config, hour=facility.open_hour))

    assert constrained_decode.semantic_invalid_reason(fields, config) is None


def test_a_valid_answer_may_still_be_semantically_invalid(config, schema):
    shut = (config.facilities[0].open_hour - 1) % 24

    outcome = constrained_decode.decode_one(
        replies(answer(config, hour=shut)), schema, config, retries=1
    )

    assert outcome.status == constrained_decode.VALID
    assert outcome.semantic_invalid_reason is not None


def test_zero_failures_still_reserve_a_non_zero_retry_allowance(settings):
    reserve = constrained_decode.retry_reserve(0, 100, settings.decoding)

    assert reserve == settings.decoding.retry_reserve_floor


def test_observed_failures_raise_the_allowance_above_the_floor(settings):
    reserve = constrained_decode.retry_reserve(20, 100, settings.decoding)

    assert reserve > settings.decoding.retry_reserve_floor


def test_the_bound_on_no_failures_is_the_exact_closed_form():
    bound = constrained_decode.binomial_upper_bound(0, 100, 0.95)

    assert bound == pytest.approx(1.0 - 0.05 ** (1 / 100), abs=1e-9)


def test_a_bound_is_refused_over_no_trials():
    with pytest.raises(ValueError, match="0 trials"):
        constrained_decode.binomial_upper_bound(0, 0, 0.95)


def test_the_identity_changes_when_the_schema_changes(settings):
    hashes = {"model.safetensors": "a" * 64}
    versions = {"outlines": "1.3.3"}

    first = constrained_decode.build_identity(
        settings, hashes, versions, "prompt", "schema-one"
    )
    second = constrained_decode.build_identity(
        settings, hashes, versions, "prompt", "schema-two"
    )

    assert first["identity_hash"] != second["identity_hash"]


def test_the_identity_changes_when_a_weight_file_changes(settings):
    versions = {"outlines": "1.3.3"}

    first = constrained_decode.build_identity(
        settings, {"model.safetensors": "a" * 64}, versions, "p", "s"
    )
    second = constrained_decode.build_identity(
        settings, {"model.safetensors": "b" * 64}, versions, "p", "s"
    )

    assert first["identity_hash"] != second["identity_hash"]


def test_the_identity_records_the_revision_that_generates(settings):
    identity = constrained_decode.build_identity(
        settings, {}, {}, "prompt", "schema"
    )

    assert identity["model_revision"] == settings.model.revision
    assert identity["tokenizer_revision"] == settings.model.revision


def test_a_moving_model_reference_is_refused_at_load(settings):
    fields = settings.model.model_dump()
    fields["revision"] = "main"

    with pytest.raises(ValueError, match="40-character commit hash"):
        settings_module.Model(**fields)


def test_a_latency_sample_larger_than_the_fixture_set_is_refused(settings):
    fields = settings.decoding.model_dump()
    fields["latency_fixtures"] = settings.decoding.fixtures + 1

    with pytest.raises(ValueError, match="fixtures it is drawn from"):
        settings_module.Decoding(**fields)


class _Tokenizer:
    """A chat template that puts fixed text before an answer."""

    def __init__(self, prefix):
        self.prefix = prefix

    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ):
        del tokenize
        rendered = "".join(
            f"<|{message['role']}|>{message['content']}" for message in messages
        )
        if add_generation_prompt:
            return f"{rendered}<|assistant|>"
        return rendered


class _Model:
    def __init__(self, tokenizer):
        self.mlx_tokenizer = tokenizer


class _Broken(_Tokenizer):
    def apply_chat_template(
        self, messages, tokenize=True, add_generation_prompt=False
    ):
        del messages, tokenize
        return "generation" if add_generation_prompt else "unrelated"


def test_the_prefix_is_read_out_of_the_chat_template():
    class _Thinking(_Tokenizer):
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False
        ):
            base = _Tokenizer.apply_chat_template(
                self,
                messages[:2],
                tokenize=tokenize,
                add_generation_prompt=True,
            )
            if add_generation_prompt:
                return base
            return base + self.prefix + messages[2]["content"]

    prefix = constrained_decode.derive_assistant_prefix(
        _Model(_Thinking("<think>\n\n</think>\n\n"))
    )

    assert prefix == "<think>\n\n</think>\n\n"


def test_a_template_inserting_nothing_gives_an_empty_prefix():
    class _Plain(_Tokenizer):
        def apply_chat_template(
            self, messages, tokenize=True, add_generation_prompt=False
        ):
            base = _Tokenizer.apply_chat_template(
                self,
                messages[:2],
                tokenize=tokenize,
                add_generation_prompt=True,
            )
            return (
                base if add_generation_prompt else base + messages[2]["content"]
            )

    assert constrained_decode.derive_assistant_prefix(_Model(_Plain(""))) == ""


def test_a_template_whose_turn_does_not_extend_its_prompt_is_refused():
    with pytest.raises(constrained_decode.StackError, match="cannot be shown"):
        constrained_decode.derive_assistant_prefix(_Model(_Broken("")))


def test_the_prefix_is_part_of_the_run_identity(settings):
    with_prefix = constrained_decode.build_identity(
        settings, {}, {}, "prompt", "schema", "none", "<think>\n\n</think>\n\n"
    )
    without = constrained_decode.build_identity(
        settings, {}, {}, "prompt", "schema", "none", ""
    )

    assert with_prefix["identity_hash"] != without["identity_hash"]
