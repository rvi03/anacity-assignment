"""Runs, prompt versions, and call traces against the real server.

These run against the MLflow service the compose stack starts. A stub
would exercise none of what is actually being tested: that a run is
retrievable with the tag that identifies its track, that an alias
resolves to the text it was registered with, and that a failed attempt
leaves a span carrying its reason.

When no server is reachable the tests skip with a named marker rather
than passing, so an absent server is visible in the run summary instead
of being mistaken for a green suite.
"""

from __future__ import annotations

import pathlib

import mlflow
import pytest

from facility_prediction import config as config_module
from facility_prediction import tracking
from facility_prediction.data import storage

SMOKE_CONFIG = pathlib.Path("configs") / "smoke.yaml"
PROMPT_NAME = "facility-usage-smoke-prompt"
TEMPLATE = "Resident {{resident_id}} booked {{facility}}. Next facility?"


@pytest.fixture(scope="module")
def smoke_config():
    return config_module.load_config(SMOKE_CONFIG)


@pytest.fixture(scope="module")
def server(smoke_config):
    if not tracking.is_reachable(smoke_config):
        pytest.skip("no tracking server reachable")
    return mlflow.tracking.MlflowClient(tracking.tracking_uri(smoke_config))


@pytest.fixture
def disabled(smoke_config):
    off = smoke_config.tracking.model_copy(update={"enabled": False})
    return smoke_config.model_copy(update={"tracking": off})


def test_a_run_carries_the_track_that_opened_it(smoke_config, server):
    with tracking.run(
        smoke_config, name="baselines", track="baseline"
    ) as logger:
        logger.log_params({"seed": smoke_config.seed})
        logger.log_metrics({"rows_scored": 12.0})
        run_id = logger.run_id

    recorded = server.get_run(run_id)
    assert recorded.data.tags[tracking.TRACK_TAG] == "baseline"
    assert recorded.data.params["seed"] == str(smoke_config.seed)
    assert recorded.data.metrics["rows_scored"] == 12.0


def test_a_run_keeps_an_artifact_it_was_given(smoke_config, server, tmp_path):
    path = tmp_path / "summary.json"
    path.write_text('{"rows": 12}', encoding="utf-8")

    with tracking.run(smoke_config, name="artifact", track="baseline") as log:
        log.log_artifact(path)
        run_id = log.run_id

    assert [item.path for item in server.list_artifacts(run_id)] == [
        "summary.json"
    ]


def test_a_gate_stop_is_recorded_as_a_failed_run(smoke_config, server):
    run_id = tracking.log_gate_stop(
        smoke_config,
        gate="representation ceiling",
        reason="ceiling 0.91 below the required 0.95",
        track="llm",
    )

    recorded = server.get_run(run_id)
    assert recorded.info.status == "FAILED"
    assert recorded.data.tags[tracking.OUTCOME_TAG] == tracking.STOPPED_AT_GATE
    assert recorded.data.tags[tracking.GATE_TAG] == "representation ceiling"
    assert "0.91" in recorded.data.tags[tracking.GATE_REASON_TAG]


def test_a_registered_prompt_resolves_by_alias_to_the_same_text(smoke_config):
    registered = tracking.register_prompt(
        smoke_config,
        name=PROMPT_NAME,
        template=TEMPLATE,
        commit_message="smoke fixture",
    )
    tracking.set_prompt_alias(
        smoke_config,
        name=PROMPT_NAME,
        alias=smoke_config.tracking.prompt_alias,
        version=registered.version,
    )

    resolved = tracking.resolve_prompt(smoke_config, name=PROMPT_NAME)

    assert resolved.version == registered.version
    assert resolved.template_hash == tracking.prompt_hash(TEMPLATE)
    assert resolved.alias == smoke_config.tracking.prompt_alias


def test_a_traced_call_produces_one_span_per_attempt(smoke_config):
    with tracking.trace_generation(
        smoke_config, track="llm", model="test-model@rev1"
    ) as trace:
        with trace.attempt(0, prompt=TEMPLATE) as attempt:
            attempt.record_failure(reason="schema parse failed")
        with trace.attempt(1, prompt=TEMPLATE) as attempt:
            attempt.record_success(
                completion='{"facility": "Gym"}', parse_outcome="valid"
            )
        trace_id = trace.trace_id

    tracking.flush_traces()
    spans = mlflow.get_trace(trace_id).data.spans

    assert [span.name for span in spans] == [
        "generation",
        "attempt-0",
        "attempt-1",
    ]


def test_a_failed_attempt_carries_its_reason(smoke_config):
    with tracking.trace_generation(
        smoke_config, track="llm", model="test-model@rev1"
    ) as trace:
        with trace.attempt(0, prompt=TEMPLATE) as attempt:
            attempt.record_failure(reason="empty completion")
        trace_id = trace.trace_id

    tracking.flush_traces()
    failed = next(
        span
        for span in mlflow.get_trace(trace_id).data.spans
        if span.name == "attempt-0"
    )

    assert failed.status.status_code.value == "ERROR"
    assert failed.attributes["failure_reason"] == "empty completion"
    assert failed.attributes["is_retry"] is False


def test_disabled_tracking_records_nothing_and_raises_nothing(
    disabled, tmp_path
):
    path = tmp_path / "summary.json"
    path.write_text("{}", encoding="utf-8")

    with tracking.run(disabled, name="baselines", track="baseline") as logger:
        logger.log_params({"seed": disabled.seed})
        logger.log_metrics({"rows_scored": 12.0})
        logger.log_artifact(path)
        logger.set_tags({"note": "ignored"})

    assert logger.run_id is None
    assert logger.recording is False


def test_a_disabled_trace_still_yields_a_usable_trace(disabled):
    with (
        tracking.trace_generation(
            disabled, track="llm", model="test-model@rev1"
        ) as trace,
        trace.attempt(0, prompt=TEMPLATE) as attempt,
    ):
        attempt.record_success(completion="{}", parse_outcome="valid")

    assert trace.trace_id is None


def test_the_prompt_registry_refuses_to_be_silent_when_disabled(disabled):
    with pytest.raises(tracking.TrackingError, match="disabled"):
        tracking.resolve_prompt(disabled, name=PROMPT_NAME)


def test_the_mirror_database_is_absent_from_the_comparison_set(smoke_config):
    compared = tracking.verify_comparison_databases(smoke_config)

    assert tracking.MIRROR_DATABASE not in compared
    assert smoke_config.storage.database in compared


def test_missing_tracking_uri_names_the_variable(smoke_config, monkeypatch):
    monkeypatch.delenv(smoke_config.tracking.tracking_uri_env, raising=False)
    monkeypatch.setattr(storage, "_ENV_FILE", pathlib.Path("absent.env"))

    with pytest.raises(tracking.TrackingError, match="MLFLOW_TRACKING_URI"):
        tracking.tracking_uri(smoke_config)
