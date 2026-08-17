"""The single scored pass: freeze first, then open the holdout.

No model runs here. Answers are written by hand so the one ordering
this step exists to guarantee can be attacked directly — scoring a file
that changed after it was sealed must fail, and a pass that covers rows
the frozen manifest never sealed must fail too.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import split as split_module
from facility_prediction.llm import buckets, final_pass, llm_predict

CONFIG = pathlib.Path("configs") / "default.yaml"

MID = "M60_90"
HIGH = "OVER_90M"

CEILING = 0.95
ROWS = 20


@pytest.fixture
def config():
    return config_module.load_config(CONFIG)


@pytest.fixture
def ladder():
    return [
        buckets.Bucket(
            label=MID, lower=1.0, upper=91.0, representative=75.0, train_rows=20
        ),
        buckets.Bucket(
            label=HIGH,
            lower=91.0,
            upper=None,
            representative=120.0,
            train_rows=20,
        ),
    ]


@pytest.fixture
def identifiers():
    return [f"h{index:03d}" for index in range(ROWS)]


@pytest.fixture
def prompts(identifiers):
    return [
        {"sample_id": identifier, "system": "s", "prompt": "p"}
        for identifier in identifiers
    ]


@pytest.fixture
def splits(identifiers):
    return pd.DataFrame(
        {
            split_module.SAMPLE_ID_COLUMN: [*identifiers, "v000"],
            split_module.SPLIT_COLUMN: [split_module.TEST] * ROWS
            + [split_module.VALIDATION],
        }
    )


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


def _record(sample_id, label=MID):
    return {
        "sample_id": sample_id,
        "arm": final_pass.TIER_B,
        "status": "valid",
        "fields": {
            "facility": "Gym",
            "usage_day": "Tuesday",
            "usage_hour": 9,
            "notification_delay_bucket": label,
        },
        "attempts": 1,
        "failure_stage": None,
        "failure_reason": None,
        "semantic_invalid_reason": None,
        "seconds": 3.0,
        "prompt_tokens": 1400,
        "completion_tokens": 40,
        "identity_hash": "abc",
    }


@pytest.fixture
def records(identifiers):
    return [_record(identifier) for identifier in identifiers]


def test_a_target_free_prompt_set_passes(prompts):
    final_pass.check_target_free(prompts)


def test_a_prompt_carrying_a_target_is_refused(prompts):
    prompts[3]["target"] = '{"facility": "Gym"}'
    with pytest.raises(final_pass.FinalPassError, match="carry a target"):
        final_pass.check_target_free(prompts)


def test_the_sealed_manifest_rows_are_accepted(prompts, identifiers, splits):
    final_pass.check_manifest(prompts, identifiers, splits)


def test_scoring_fewer_rows_than_the_manifest_seals_is_refused(
    prompts, identifiers, splits
):
    with pytest.raises(final_pass.FinalPassError, match="same rows"):
        final_pass.check_manifest(prompts[:-1], identifiers, splits)


def test_a_row_outside_the_holdout_is_refused(prompts, identifiers, splits):
    prompts.append({"sample_id": "v000", "system": "s", "prompt": "p"})
    with pytest.raises(final_pass.FinalPassError, match="not holdout rows"):
        final_pass.check_manifest(prompts, [*identifiers, "v000"], splits)


def test_the_freeze_records_the_file_it_sealed(tmp_path, records):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)

    assert freeze.rows == ROWS
    assert len(freeze.sha256) == 64
    assert freeze.to_dict()["predictions_sha256"] == freeze.sha256


def test_the_freeze_is_the_hash_of_what_is_on_disk(tmp_path, records):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)
    final_pass.check_freeze(freeze)


def test_answers_changed_after_the_freeze_cannot_be_scored(
    tmp_path, records, targets, labels, config, ladder
):
    """The one guarantee this step exists to make."""
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)
    llm_predict.write_records([_record(records[0]["sample_id"], HIGH)], path)

    with pytest.raises(final_pass.FinalPassError, match="after the freeze"):
        final_pass.score(
            freeze,
            llm_predict.to_frame(records, ladder),
            targets,
            labels,
            config,
            CEILING,
        )


def test_a_missing_prediction_file_cannot_be_scored(tmp_path, records):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)
    path.unlink()

    with pytest.raises(final_pass.FinalPassError, match="are gone"):
        final_pass.check_freeze(freeze)


def test_scoring_marks_the_frozen_answers(
    tmp_path, records, targets, labels, config, ladder
):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)

    marked = final_pass.score(
        freeze,
        llm_predict.to_frame(records, ladder),
        targets,
        labels,
        config,
        CEILING,
    )

    assert marked["rates"]["facility"] == pytest.approx(1.0)
    assert marked["rates"]["notification"] == pytest.approx(1.0)
    assert marked["support"][MID] == ROWS


def test_the_report_carries_the_freeze_and_the_rates(
    tmp_path, records, targets, labels, config, ladder
):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)
    marked = final_pass.score(
        freeze,
        llm_predict.to_frame(records, ladder),
        targets,
        labels,
        config,
        CEILING,
    )

    report = final_pass.build_report(
        freeze,
        marked,
        llm_predict.operational_metrics(records),
        {"rows": ROWS, "split": split_module.TEST, "replayed": False},
    )

    assert report["evidence"] == final_pass.EVIDENCE
    assert report["freeze"]["predictions_sha256"] == freeze.sha256
    assert report["context"]["split"] == split_module.TEST
    assert final_pass.run_params(report)["predictions_sha256"] == freeze.sha256
    assert final_pass.run_metrics(report)["facility"] == pytest.approx(1.0)


def test_a_failed_row_stays_in_the_denominator(
    tmp_path, records, targets, labels, config, ladder
):
    records[0] = {**records[0], "status": "failed", "fields": None}
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)

    marked = final_pass.score(
        freeze,
        llm_predict.to_frame(records, ladder),
        targets,
        labels,
        config,
        CEILING,
    )

    assert marked["rates"]["facility"] == pytest.approx((ROWS - 1) / ROWS)
    assert marked["rates"]["valid"] == pytest.approx((ROWS - 1) / ROWS)


def test_the_comparator_is_reported_beside_the_result(
    tmp_path, records, targets, labels, config, ladder
):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)
    marked = final_pass.score(
        freeze,
        llm_predict.to_frame(records, ladder),
        targets,
        labels,
        config,
        CEILING,
    )

    report = final_pass.build_report(
        freeze,
        marked,
        llm_predict.operational_metrics(records),
        {"rows": ROWS, "split": split_module.TEST, "replayed": False},
        {"overall": 0.25},
    )

    assert report["comparator"]["model"] == final_pass.COMPARATOR
    assert report["comparator"]["delta_overall"] == pytest.approx(
        report["rates"]["overall"] - 0.25
    )


def test_a_pass_with_no_comparator_omits_the_block(
    tmp_path, records, targets, labels, config, ladder
):
    path = tmp_path / "predictions.jsonl"
    freeze = final_pass.freeze_predictions(records, path)
    marked = final_pass.score(
        freeze,
        llm_predict.to_frame(records, ladder),
        targets,
        labels,
        config,
        CEILING,
    )

    report = final_pass.build_report(
        freeze,
        marked,
        llm_predict.operational_metrics(records),
        {"rows": ROWS, "split": split_module.TEST, "replayed": False},
    )

    assert "comparator" not in report
