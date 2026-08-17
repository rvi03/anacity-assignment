"""The learnability gate: marking, pairing, and refusing.

No model runs here. Answers are written by hand so each condition can
be aimed at directly: a failed row that must stay in the denominator, a
trained pass that only ties, a pass that loses to a rule ignoring the
prompt, and a replay that reproduces the marking from saved answers.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.llm import buckets, gate, llm_predict
from facility_prediction.llm import settings as settings_module

CONFIG = pathlib.Path("configs") / "default.yaml"
LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

LOW = "UNDER_60M"
MID = "M60_90"
HIGH = "OVER_90M"

CEILING = 0.95
ROWS = 40


@pytest.fixture
def config():
    return config_module.load_config(CONFIG)


@pytest.fixture
def gate_settings():
    return settings_module.load_settings(LLM_CONFIG).gate


@pytest.fixture
def ladder():
    return [
        buckets.Bucket(
            label=LOW,
            lower=1.0,
            upper=61.0,
            representative=30.0,
            train_rows=10,
        ),
        buckets.Bucket(
            label=MID,
            lower=61.0,
            upper=91.0,
            representative=75.0,
            train_rows=80,
        ),
        buckets.Bucket(
            label=HIGH,
            lower=91.0,
            upper=None,
            representative=120.0,
            train_rows=30,
        ),
    ]


@pytest.fixture
def identifiers():
    return [f"s{index:03d}" for index in range(ROWS)]


@pytest.fixture
def targets(identifiers):
    return pd.DataFrame(
        {
            "target_facility_id": ["Gym"] * ROWS,
            "target_usage_weekday": [1] * ROWS,
            "target_usage_hour": [9] * ROWS,
            "notification_delay_minutes": [75.0] * ROWS,
        },
        index=pd.Index(identifiers, name="sample_id"),
    )


@pytest.fixture
def labels(identifiers):
    return pd.Series(
        [MID] * ROWS, index=pd.Index(identifiers, name="sample_id")
    )


def _record(sample_id, arm, label=MID, valid=True, hour=9):
    fields = (
        {
            "facility": "Gym",
            "usage_day": "Tuesday",
            "usage_hour": hour,
            "notification_delay_bucket": label,
        }
        if valid
        else None
    )
    return {
        "sample_id": sample_id,
        "arm": arm,
        "status": "valid" if valid else "failed",
        "fields": fields,
        "attempts": 1,
        "failure_stage": None if valid else "parse",
        "failure_reason": None if valid else "not JSON",
        "semantic_invalid_reason": None,
        "seconds": 3.0,
        "prompt_tokens": 1400,
        "completion_tokens": 40,
        "identity_hash": "abc",
    }


def _records(identifiers, arm, correct, **kwargs):
    return [
        _record(
            identifier,
            arm,
            label=MID if index < correct else HIGH,
            **kwargs,
        )
        for index, identifier in enumerate(identifiers)
    ]


def _marked(records, targets, labels, config, ladder):
    return gate.arm_metrics(
        llm_predict.to_frame(records, ladder), targets, labels, config, CEILING
    )


def test_failed_row_stays_in_the_denominator_scoring_nothing(
    identifiers, targets, labels, config, ladder
):
    records = _records(identifiers, llm_predict.ZERO_SHOT, ROWS)
    records[0] = _record(identifiers[0], llm_predict.ZERO_SHOT, valid=False)

    marked = _marked(records, targets, labels, config, ladder)

    assert marked["rates"][gate.BUCKET_ACCURACY] == pytest.approx(
        (ROWS - 1) / ROWS
    )
    assert marked["rates"]["valid"] == pytest.approx((ROWS - 1) / ROWS)


def test_every_answer_failing_scores_zero_rather_than_raising(
    identifiers, targets, labels, config, ladder
):
    records = [
        _record(identifier, llm_predict.ZERO_SHOT, valid=False)
        for identifier in identifiers
    ]

    marked = _marked(records, targets, labels, config, ladder)

    assert marked["rates"][gate.BUCKET_ACCURACY] == 0.0
    assert marked["rates"]["notification"] == 0.0


def test_notification_match_reads_the_bucket_representative(
    identifiers, targets, labels, config, ladder
):
    records = _records(identifiers, llm_predict.ZERO_SHOT, ROWS)

    marked = _marked(records, targets, labels, config, ladder)

    assert marked["rates"]["notification"] == 1.0
    assert marked["rates"]["match_over_ceiling"] == pytest.approx(1 / CEILING)


def test_a_wrong_bucket_outside_tolerance_does_not_match(
    identifiers, targets, labels, config, ladder
):
    records = _records(identifiers, llm_predict.ZERO_SHOT, 0)

    marked = _marked(records, targets, labels, config, ladder)

    assert marked["rates"][gate.BUCKET_ACCURACY] == 0.0
    assert marked["rates"]["notification"] == 0.0


def test_an_answer_naming_an_unknown_bucket_is_refused(identifiers, ladder):
    records = [_record(identifiers[0], llm_predict.ZERO_SHOT, label="NOPE")]

    with pytest.raises(llm_predict.PredictionError, match="no representative"):
        llm_predict.to_frame(records, ladder)


def test_paired_interval_is_identical_under_its_seed(
    identifiers, gate_settings
):
    base = pd.Series([False] * ROWS, index=identifiers)
    trained = pd.Series([True] * 30 + [False] * 10, index=identifiers)

    first = gate.paired_bootstrap(base, trained, gate_settings)
    second = gate.paired_bootstrap(base, trained, gate_settings)

    assert first == second


def test_paired_interval_needs_the_same_rows(identifiers, gate_settings):
    base = pd.Series([True] * ROWS, index=identifiers)
    trained = pd.Series([True] * ROWS, index=list(reversed(identifiers)))

    with pytest.raises(gate.GateError, match="same rows"):
        gate.paired_bootstrap(base, trained, gate_settings)


def test_no_difference_leaves_an_interval_containing_zero(
    identifiers, gate_settings
):
    outcome = pd.Series([True] * 20 + [False] * 20, index=identifiers)

    interval = gate.paired_bootstrap(outcome, outcome, gate_settings)

    assert interval["difference"] == 0.0
    assert interval["low"] == 0.0


def test_references_come_from_training_rows_only(
    targets, labels, config, ladder
):
    train_delays = pd.Series([75.0] * 100)

    references = gate.reference_metrics(
        train_delays, labels, targets, ladder, config
    )

    assert references["majority_label"] == MID
    assert references["median_delay_minutes"] == 75.0
    assert references["median_delay_notification_match"] == 1.0


def test_gate_refuses_a_pilot_whose_loss_did_not_fall(_report):
    pilot = {"loss": {"initial_validation": 1.0, "final_validation": 1.5}}

    with pytest.raises(gate.GateError, match="did not fall"):
        gate.check_gate(_report, pilot)


def test_gate_refuses_an_interval_containing_zero(_report):
    _report["paired_intervals"][gate.BUCKET_ACCURACY]["low"] = -0.01
    _report["paired_intervals"][gate.NOTIFICATION_MATCH]["low"] = -0.02

    with pytest.raises(gate.GateError, match="not distinguishable"):
        gate.check_gate(_report, _pilot())


def test_gate_refuses_a_pass_below_the_rule_that_ignores_the_prompt(_report):
    _report["references"]["majority_bucket_accuracy"] = 0.99

    with pytest.raises(gate.GateError, match="ignores the prompt"):
        gate.check_gate(_report, _pilot())


def test_gate_passes_on_one_interval_clear_of_zero(_report):
    _report["paired_intervals"][gate.NOTIFICATION_MATCH]["low"] = -0.05

    gate.check_gate(_report, _pilot())


def test_marking_replays_from_saved_answers(
    identifiers, targets, labels, config, ladder, tmp_path
):
    records = _records(identifiers, llm_predict.PILOT_ADAPTER, 30)
    path = tmp_path / "answers.jsonl"
    llm_predict.write_records(records, path)

    replayed = llm_predict.arm_records(
        llm_predict.read_records(path), llm_predict.PILOT_ADAPTER
    )

    assert (
        _marked(replayed, targets, labels, config, ladder)["rates"]
        == (_marked(records, targets, labels, config, ladder)["rates"])
    )


def test_replay_refuses_an_arm_that_was_never_run(identifiers, tmp_path):
    path = tmp_path / "answers.jsonl"
    llm_predict.write_records(
        _records(identifiers, llm_predict.ZERO_SHOT, 10), path
    )

    with pytest.raises(llm_predict.PredictionError, match="no records"):
        llm_predict.arm_records(
            llm_predict.read_records(path), llm_predict.PILOT_ADAPTER
        )


def _pilot():
    return {"loss": {"initial_validation": 3.5, "final_validation": 0.2}}


@pytest.fixture
def _report(identifiers, targets, labels, config, ladder, gate_settings):
    zero_shot = _marked(
        _records(identifiers, llm_predict.ZERO_SHOT, 10),
        targets,
        labels,
        config,
        ladder,
    )
    pilot = _marked(
        _records(identifiers, llm_predict.PILOT_ADAPTER, 35),
        targets,
        labels,
        config,
        ladder,
    )
    for arm in (zero_shot, pilot):
        arm["operational"] = llm_predict.operational_metrics(
            _records(identifiers, llm_predict.ZERO_SHOT, 10)
        )
    return gate.build_report(
        {
            llm_predict.ZERO_SHOT: zero_shot,
            llm_predict.PILOT_ADAPTER: pilot,
        },
        {
            metric: gate.paired_bootstrap(
                zero_shot["per_row"][metric],
                pilot["per_row"][metric],
                gate_settings,
            )
            for metric in (gate.BUCKET_ACCURACY, gate.NOTIFICATION_MATCH)
        },
        gate.reference_metrics(
            # A training split whose commonest label and median delay
            # are both wrong here, so the references are beatable and
            # the test aims at the condition it names.
            pd.Series([30.0] * 100),
            labels,
            targets,
            [
                buckets.Bucket(
                    label=LOW,
                    lower=1.0,
                    upper=61.0,
                    representative=30.0,
                    train_rows=90,
                ),
                *ladder[1:],
            ],
            config,
        ),
        {"rows": ROWS, "split": "validation"},
    )
