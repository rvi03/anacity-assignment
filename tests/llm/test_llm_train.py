"""The pilot: what it draws, what it trains on, what it records.

No model is loaded here. The trainer's own measurements are injected,
so what is tested is the part this repository owns: the draw, the
conversation records, the flat configuration handed to the trainer, and
the refusals when a run did not do what it declared.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from facility_prediction.llm import buckets, llm_train, pilot_select
from facility_prediction.llm import settings as settings_module

LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

LABELS = ("UNDER_60M", "M60_90", "OVER_90M")
ROWS = 60
DRAW_ROWS = 12
MIN_PER_LABEL = 2
SEED = 20260811


@pytest.fixture
def settings():
    return settings_module.load_settings(LLM_CONFIG)


@pytest.fixture
def rows():
    return [
        {
            "sample_id": f"s{index:03d}",
            "system": "answer in json",
            "prompt": f"prompt {index}",
            "target": json.dumps({"notification_bucket": LABELS[index % 3]}),
        }
        for index in range(ROWS)
    ]


@pytest.fixture
def labels(rows):
    # The first label is deliberately rare, so the floor per label has
    # to do work the uniform fill would not.
    assigned = {}
    for index, row in enumerate(rows):
        name = LABELS[0] if index < MIN_PER_LABEL else LABELS[1 + index % 2]
        assigned[row["sample_id"]] = name
    return assigned


def _ladder():
    return [
        buckets.Bucket(label=LABELS[0], lower=1.0, upper=61.0, train_rows=10),
        buckets.Bucket(label=LABELS[1], lower=61.0, upper=91.0, train_rows=60),
        buckets.Bucket(label=LABELS[2], lower=91.0, upper=None, train_rows=30),
    ]


def _select(rows, labels, train_rows=DRAW_ROWS):
    return pilot_select.select(rows, labels, train_rows, MIN_PER_LABEL, SEED)


def test_draw_covers_every_label_present(rows, labels):
    selection = _select(rows, labels)

    assert set(selection.support) == set(labels.values())
    assert selection.support[LABELS[0]] >= MIN_PER_LABEL


def test_draw_takes_every_row_of_a_label_thinner_than_the_floor(rows, labels):
    labels[rows[0]["sample_id"]] = "ONLY_ONE"

    selection = _select(rows, labels)

    assert selection.support["ONLY_ONE"] == 1
    assert selection.available["ONLY_ONE"] == 1


def test_draw_is_the_declared_size(rows, labels):
    selection = _select(rows, labels)

    assert len(selection.rows) == DRAW_ROWS


def test_draw_repeats_exactly_under_the_same_seed(rows, labels):
    first = _select(rows, labels)
    second = _select(rows, labels)

    assert [row["sample_id"] for row in first.rows] == [
        row["sample_id"] for row in second.rows
    ]


def test_draw_is_ordered_by_identifier(rows, labels):
    selection = _select(rows, labels)

    identifiers = [row["sample_id"] for row in selection.rows]

    assert identifiers == sorted(identifiers)


def test_draw_refuses_more_rows_than_exist(rows, labels):
    with pytest.raises(pilot_select.PilotError, match="cannot draw"):
        _select(rows, labels, train_rows=ROWS + 1)


def test_draw_refuses_a_floor_it_cannot_fit(rows, labels):
    with pytest.raises(pilot_select.PilotError, match="more than the"):
        pilot_select.select(rows, labels, 3, MIN_PER_LABEL, SEED)


def test_draw_refuses_a_row_with_no_training_label(rows, labels):
    del labels[rows[0]["sample_id"]]

    with pytest.raises(pilot_select.PilotError, match="no training label"):
        _select(rows, labels)


def test_draw_record_declares_it_is_not_prevalence_representative(
    rows, labels, settings
):
    selection = _select(rows, labels)

    record = pilot_select.draw_record(selection, settings.pilot, _ladder())

    assert record["shares"]["prevalence_representative"] is False
    assert record["shares"]["max_absolute_share_deviation"] > 0.0


def test_draw_record_lists_the_rows_it_drew(rows, labels, settings):
    selection = _select(rows, labels)

    record = pilot_select.draw_record(selection, settings.pilot, _ladder())

    assert record["sample_ids"] == [row["sample_id"] for row in selection.rows]
    assert len(record["sample_ids_sha256"]) == 64


def test_training_labels_come_from_the_frozen_ladder():
    samples = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "notification_delay_minutes": [30.0, 120.0],
        }
    )

    assigned = pilot_select.training_labels(samples, _ladder())

    assert assigned == {"a": LABELS[0], "b": LABELS[2]}


def test_conversation_keeps_the_system_prompt_and_answer(rows):
    records = llm_train.chat_records(rows[:1])

    assert [message["role"] for message in records[0]["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert records[0]["messages"][2]["content"] == rows[0]["target"]


def test_conversation_refuses_a_row_with_no_answer(rows):
    rows[0]["target"] = ""

    with pytest.raises(llm_train.TrainingError, match="no answer"):
        llm_train.chat_records(rows[:1])


def test_written_conversations_are_byte_identical_twice(rows, tmp_path):
    records = llm_train.chat_records(rows)

    first = llm_train.write_jsonl(records, tmp_path / "a.jsonl")
    second = llm_train.write_jsonl(records, tmp_path / "b.jsonl")

    assert first == second


def test_trainer_config_saves_only_at_the_declared_iteration(
    settings, tmp_path
):
    flat = llm_train.flat_config(
        settings, "model", tmp_path, tmp_path, settings.pilot.iters
    )

    assert flat["save_every"] == settings.pilot.iters
    assert flat["iters"] == settings.pilot.iters


def test_trainer_config_masks_the_prompt_and_trains_adapters_only(
    settings, tmp_path
):
    flat = llm_train.flat_config(settings, "model", tmp_path, tmp_path, 400)

    assert flat["mask_prompt"] is True
    assert flat["fine_tune_type"] == "lora"
    assert flat["train"] is True
    assert flat["test"] is False


def test_command_line_names_the_documented_controls(settings, tmp_path):
    flat = llm_train.flat_config(settings, "model", tmp_path, tmp_path, 400)

    command = llm_train.command_line(flat, tmp_path / "mlx_config.yaml")

    for control in (
        "--iters 400",
        "--batch-size 1",
        "--grad-accumulation-steps 4",
        "--num-layers 16",
        "--mask-prompt",
    ):
        assert control in command


def test_declared_pilot_iterations_must_match_their_arithmetic(settings):
    fields = settings.model_dump()
    fields["pilot"] = {**fields["pilot"], "iters": 399}

    with pytest.raises(llm_train.TrainingError, match="399 iterations"):
        llm_train.declared_iters(settings_module.Settings(**fields))


def test_declared_pilot_iterations_must_fill_whole_updates(settings):
    fields = settings.model_dump()
    fields["pilot"] = {
        **fields["pilot"],
        "train_rows": 101,
        "epoch_equivalent": 1,
        "iters": 101,
    }

    with pytest.raises(llm_train.TrainingError, match="whole optimizer"):
        llm_train.declared_iters(settings_module.Settings(**fields))


def test_adapter_description_counts_blocks_and_projections():
    names = [
        "model.layers.20.self_attn.q_proj.lora_a",
        "model.layers.20.self_attn.v_proj.lora_b",
        "model.layers.21.self_attn.q_proj.lora_a",
    ]

    described = llm_train.describe_parameters(names, "abc")

    assert described["adapted_blocks"] == 2
    assert described["projections"] == ["q_proj", "v_proj"]
    assert described["parameters"] == 3


def test_record_separates_timing_from_loss(settings, tmp_path):
    record = _record(settings, tmp_path)

    assert set(record["timing"]) == {
        "seconds",
        "seconds_per_iteration",
        "trained_tokens",
        "peak_memory_gb",
    }
    assert record["loss"]["final_validation"] == 1.0


def test_record_refuses_an_adapter_with_no_parameters(settings, tmp_path):
    record = _record(settings, tmp_path)
    record["adapter"]["parameters"] = 0

    with pytest.raises(llm_train.TrainingError, match="no parameters"):
        llm_train.check_saved(record)


def test_record_refuses_a_run_that_stopped_early(settings, tmp_path):
    record = _record(settings, tmp_path)
    record["loss"]["training_reports"][-1]["iteration"] = 200

    with pytest.raises(llm_train.TrainingError, match="declared"):
        llm_train.check_saved(record)


def test_measured_seconds_per_iteration_is_the_run_divided_by_its_iters():
    result = llm_train.TrainingResult(
        seconds=200.0, iters=400, train_reports=[], val_reports=[]
    )

    assert result.seconds_per_iteration == pytest.approx(0.5)


def test_training_report_writes_pollable_progress_files(tmp_path):
    recorder = llm_train._Recorder(100, 2, tmp_path)

    recorder.on_train_loss_report(
        {
            "iteration": 25,
            "train_loss": 1.0,
            "learning_rate": 0.0001,
            "iterations_per_second": 2.0,
            "tokens_per_second": 3.0,
            "trained_tokens": 100.0,
            "peak_memory": 4.0,
        }
    )

    status = json.loads(
        (tmp_path / llm_train.TRAINING_STATUS_FILE).read_text(encoding="utf-8")
    )

    assert status["iteration"] == 25
    assert status["row_passes_completed"] == 50
    assert status["percent_complete"] == 25.0
    assert "progress iteration=25/100 rows_processed=50 percent=25.0" in (
        tmp_path / llm_train.TRAINING_LOG_FILE
    ).read_text(encoding="utf-8")


def _record(settings, tmp_path):
    result = llm_train.TrainingResult(
        seconds=200.0,
        iters=settings.pilot.iters,
        train_reports=[
            {
                "iteration": float(settings.pilot.iters),
                "trained_tokens": 100.0,
                "peak_memory": 4.0,
            }
        ],
        val_reports=[{"iteration": 0.0, "val_loss": 2.0}, {"val_loss": 1.0}],
    )
    flat = llm_train.flat_config(
        settings, "model", tmp_path, tmp_path, settings.pilot.iters
    )
    record = llm_train.build_record(
        result,
        flat,
        llm_train.describe_parameters(["a.0.q_proj.lora_a"], "abc"),
        {"train_rows": 200, "epoch_equivalent": 2},
        {"model_id": "pinned"},
    )
    llm_train.check_saved(record)
    return record
