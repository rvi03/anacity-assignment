"""Adapter training: the pilot, the size ladder, and the full run.

How large the full run may be is decided on timing alone, before any
quality number exists, and the sealed size is the only one the trainer
will accept.
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import pathlib
from typing import Any

from facility_prediction import config as config_module
from facility_prediction import llm, tracking
from facility_prediction.data import split, storage
from facility_prediction.llm import (
    buckets,
    constrained_decode,
    decode_check,
    ladder,
    llm_data,
    llm_train,
    pilot_select,
    slice_import,
    tier_b,
)
from facility_prediction.llm import settings as llm_settings

_LOGGER = logging.getLogger(__name__)


def _draw_pilot(
    config: config_module.Config,
    settings: llm_settings.Settings,
    frozen_ladder: list[buckets.Bucket],
    directory: pathlib.Path,
) -> pilot_select.Selection:
    """Draw the fixed set of training rows the pilot trains on.

    Leakage contract: labels come from training-split delays, and the
    prompts are the rendered training file. No later split is read.

    Args:
        config: Validated shared configuration.
        settings: The LLM configuration.
        frozen_ladder: The frozen delay buckets.
        directory: Directory holding the rendered prompt files.

    Returns:
        The drawn rows and the shape of the draw.
    """
    with storage.engine_scope(config) as engine:
        samples_table = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    labelled = samples_table.merge(
        splits, on=split.SAMPLE_ID_COLUMN, how="inner"
    )
    training = labelled.loc[labelled[split.SPLIT_COLUMN] == split.TRAIN]
    return pilot_select.select(
        decode_check.read_rows(directory / llm_data.TRAIN_FILE),
        pilot_select.training_labels(training, frozen_ladder),
        settings.pilot.train_rows,
        settings.pilot.min_rows_per_label,
        settings.pilot.sample_seed,
    )


def _write_pilot_data(
    selection: pilot_select.Selection,
    directory: pathlib.Path,
    destination: pathlib.Path,
) -> dict[str, object]:
    """Write the trainer's own copies of the drawn conversations.

    Args:
        selection: The drawn rows.
        directory: Directory holding the rendered prompt files.
        destination: Directory the trainer reads.

    Returns:
        What was written, with each file's hash.
    """
    validation = decode_check.read_rows(directory / llm_data.VALID_FILE)
    return {
        "train_file": llm_train.TRAIN_FILE,
        "train_rows": len(selection.rows),
        "train_sha256": llm_train.write_jsonl(
            llm_train.chat_records(selection.rows),
            destination / llm_train.TRAIN_FILE,
        ),
        "valid_file": llm_train.VALID_FILE,
        "valid_rows": len(validation),
        "valid_sha256": llm_train.write_jsonl(
            llm_train.chat_records(validation),
            destination / llm_train.VALID_FILE,
        ),
    }


def run_llm_pilot(
    config: config_module.Config,
    llm_config: pathlib.Path,
    buckets_manifest: pathlib.Path,
    directory: pathlib.Path,
    adapter: pathlib.Path,
    record: pathlib.Path,
) -> None:
    """Train the fixed pilot adapter and measure what it cost.

    Leakage contract: trains on drawn training rows and monitors loss
    on validation rows. No holdout row is read.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        buckets_manifest: Path to the frozen bucket manifest.
        directory: Directory holding the rendered prompt files.
        adapter: Directory the adapter and its inputs are written to.
        record: Destination of the pilot record.

    Raises:
        TrainingError: If the pilot cannot train on this host.
        PilotError: If the declared draw cannot be filled.
        StackError: If the pinned artifact cannot be fetched.
    """
    settings = llm_settings.load_settings(llm_config)
    frozen_ladder = buckets.load_ladder(split.load_manifest(buckets_manifest))
    data = adapter / "data"

    try:
        iters = llm_train.declared_iters(settings)
        selection = _draw_pilot(config, settings, frozen_ladder, directory)
        written = _write_pilot_data(selection, directory, data)
        flat = llm_train.flat_config(
            settings,
            str(constrained_decode.snapshot(settings)),
            data,
            adapter,
            iters,
        )
        provenance = {
            "model_id": settings.model.id,
            "model_revision": settings.model.revision,
            "config_sha256": llm_train.write_config(
                flat, adapter / llm_train.CONFIG_FILE
            ),
            "command": llm_train.command_line(
                flat, adapter / llm_train.CONFIG_FILE
            ),
            "trainer_help": llm_train.capture_help(),
            "data": written,
        }
        payload = llm_train.build_record(
            llm_train.train_adapter(flat),
            flat,
            llm_train.adapter_parameters(adapter / llm_train.ADAPTER_FILE),
            pilot_select.draw_record(selection, settings.pilot, frozen_ladder),
            provenance,
        )
        llm_train.check_saved(payload)
    except (
        llm_train.TrainingError,
        pilot_select.PilotError,
        constrained_decode.StackError,
    ) as error:
        tracking.log_gate_stop(
            config, gate=llm_train.GATE, reason=str(error), track=llm.TRACK
        )
        raise

    split.write_json(payload, record)
    with tracking.run(
        config, name=llm_train.RUN_NAME, track=llm.TRACK
    ) as logger:
        logger.log_params(llm_train.run_params(payload))
        logger.log_metrics(llm_train.run_metrics(payload))
        logger.log_artifact(record)

    _LOGGER.info(
        "trained %d iterations on %d rows in %.1fs (%.3fs per iteration), "
        "peak memory %.2f GB; adapter holds %d parameters over %d blocks",
        payload["arithmetic"]["iters"],
        payload["draw"]["train_rows"],
        payload["timing"]["seconds"],
        payload["timing"]["seconds_per_iteration"],
        payload["timing"]["peak_memory_gb"],
        payload["adapter"]["parameters"],
        payload["adapter"]["adapted_blocks"],
    )


def _draw_tier_b(
    config: config_module.Config,
    settings: llm_settings.Settings,
    frozen_ladder: list[buckets.Bucket],
    directory: pathlib.Path,
    step: Mapping[str, Any],
) -> pilot_select.Selection:
    """Draw the rows the one full training run trains on.

    Leakage contract: labels come from training-split delays, and the
    prompts are the rendered training file. No later split is read.

    Args:
        config: Validated shared configuration.
        settings: The LLM configuration.
        frozen_ladder: The frozen delay buckets.
        directory: Directory holding the rendered prompt files.
        step: The sealed ladder step.

    Returns:
        The drawn rows and the shape of the draw.
    """
    with storage.engine_scope(config) as engine:
        samples_table = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    labelled = samples_table.merge(
        splits, on=split.SAMPLE_ID_COLUMN, how="inner"
    )
    training = labelled.loc[labelled[split.SPLIT_COLUMN] == split.TRAIN]
    return tier_b.select(
        decode_check.read_rows(directory / llm_data.TRAIN_FILE),
        pilot_select.training_labels(training, frozen_ladder),
        step,
        settings.tier_b.sample_seed,
    )


def _reload_through_outlines(
    settings: llm_settings.Settings,
    manifest: pathlib.Path,
    adapter: pathlib.Path,
) -> dict[str, Any]:
    """Load the saved adapter back through the scored generation path.

    A trained adapter that the scored runtime cannot load is not a
    usable result, so the reload is part of the step rather than a
    later discovery.

    Args:
        settings: The LLM configuration.
        manifest: Path to the frozen prompt manifest.
        adapter: Directory holding the saved adapter.

    Returns:
        What the runtime read back, and the resulting run identity.

    Raises:
        StackError: If the pinned path cannot be assembled here.
    """
    prompt_manifest = split.load_manifest(manifest)
    runtime = constrained_decode.load_runtime(
        settings,
        prompt_manifest["schema"],
        str(prompt_manifest["prompt_hash"]),
        str(prompt_manifest["schema_hash"]),
        adapter=adapter,
    )
    parameters = llm_train.adapter_parameters(adapter / llm_train.ADAPTER_FILE)
    return {
        "identity_hash": str(runtime.identity["identity_hash"]),
        "adapter_sha256": str(runtime.identity["adapter"]),
        # The hash below is the one the RUNTIME recorded for the file it
        # loaded, not one recomputed here, so the check compares the
        # adapter that will generate against the adapter that trained.
        "read_back": {
            "parameters": parameters["parameters"],
            "adapted_blocks": parameters["adapted_blocks"],
            "sha256": str(runtime.identity["adapter"]),
        },
    }


def run_llm_tier_b(
    config: config_module.Config,
    llm_config: pathlib.Path,
    buckets_manifest: pathlib.Path,
    prompt_manifest: pathlib.Path,
    decision: pathlib.Path,
    directory: pathlib.Path,
    adapter: pathlib.Path,
    record: pathlib.Path,
) -> None:
    """Train the one full adapter at the sealed ladder size.

    Leakage contract: trains on drawn training rows and monitors loss
    on validation rows. No holdout row is read.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        buckets_manifest: Path to the frozen bucket manifest.
        prompt_manifest: Path to the frozen prompt manifest.
        decision: Path to the sealed ladder decision.
        directory: Directory holding the rendered prompt files.
        adapter: Directory the adapter and its inputs are written to.
        record: Destination of the training record.

    Raises:
        TierBError: If the sealed size cannot be trained as declared.
        TrainingError: If the run cannot train on this host.
        StackError: If the pinned artifact cannot be fetched.
    """
    settings = llm_settings.load_settings(llm_config)
    frozen_ladder = buckets.load_ladder(split.load_manifest(buckets_manifest))
    step = tier_b.selected_step(split.load_manifest(decision))
    data = adapter / "data"

    try:
        iters = tier_b.declared_iters(step, settings.tuning)
        selection = _draw_tier_b(
            config, settings, frozen_ladder, directory, step
        )
        written = _write_pilot_data(selection, directory, data)
        flat = llm_train.flat_config(
            settings,
            str(constrained_decode.snapshot(settings)),
            data,
            adapter,
            iters,
        )
        provenance = {
            "model_id": settings.model.id,
            "model_revision": settings.model.revision,
            "config_sha256": llm_train.write_config(
                flat, adapter / llm_train.CONFIG_FILE
            ),
            "command": llm_train.command_line(
                flat, adapter / llm_train.CONFIG_FILE
            ),
            "trainer_help": llm_train.capture_help(),
            "data": written,
        }
        payload = llm_train.build_record(
            llm_train.train_adapter(flat),
            flat,
            llm_train.adapter_parameters(adapter / llm_train.ADAPTER_FILE),
            tier_b.draw_record(
                selection, step, settings.tier_b.sample_seed, frozen_ladder
            ),
            provenance,
        )
        llm_train.check_saved(payload)
        reload = _reload_through_outlines(settings, prompt_manifest, adapter)
        tier_b.check_reload(reload["read_back"], payload["adapter"])
    except (
        tier_b.TierBError,
        llm_train.TrainingError,
        constrained_decode.StackError,
    ) as error:
        tracking.log_gate_stop(
            config, gate=tier_b.GATE, reason=str(error), track=llm.TRACK
        )
        raise

    payload["evidence"] = tier_b.EVIDENCE
    payload["gate"] = tier_b.GATE
    payload["ladder_step"] = step
    payload["reload"] = {
        "identity_hash": reload["identity_hash"],
        "adapter_sha256": reload["adapter_sha256"],
        "reloaded_through": "outlines",
    }
    split.write_json(payload, record)
    with tracking.run(config, name=tier_b.RUN_NAME, track=llm.TRACK) as logger:
        logger.log_params(llm_train.run_params(payload))
        logger.log_metrics(llm_train.run_metrics(payload))
        logger.log_artifact(record)

    _LOGGER.info(
        "trained %d iterations on %d rows in %.1fs (%.3fs per iteration), "
        "peak memory %.2f GB; adapter holds %d parameters over %d blocks "
        "and reloads through Outlines as %s",
        payload["arithmetic"]["iters"],
        payload["draw"]["train_rows"],
        payload["timing"]["seconds"],
        payload["timing"]["seconds_per_iteration"],
        payload["timing"]["peak_memory_gb"],
        payload["adapter"]["parameters"],
        payload["adapter"]["adapted_blocks"],
        reload["adapter_sha256"][:12],
    )


def run_llm_ladder(
    config: config_module.Config,
    llm_config: pathlib.Path,
    pilot: pathlib.Path,
    decode: pathlib.Path,
    decision: pathlib.Path,
) -> None:
    """Seal how large the full training run may be, on timing alone.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        pilot: Path to the pilot record.
        decode: Path to the fixture record.
        decision: Destination of the sealed decision.

    Raises:
        LadderError: If no pre-declared training size fits the cap.
        DecisionError: If the decision is malformed, or a different one
            is already sealed.
    """
    settings = llm_settings.load_settings(llm_config)
    pilot_record = split.load_manifest(pilot)
    decode_record = split.load_manifest(decode)
    timing = pilot_record["timing"]

    generations = ladder.scored_generations(
        settings.compute, config.split.comparison_rows
    )
    inputs = {
        "shared_pipeline_seconds": settings.compute.shared_pipeline_seconds,
        "fixture_seconds": decode_record["elapsed_seconds"],
        "pilot_seconds": timing["seconds"],
        "seconds_per_iteration": timing["seconds_per_iteration"],
        "seconds_per_row": decode_record["timing"]["latency_mean_seconds"],
        "retry_reserve": decode_record["retry_reserve"],
        "generations_remaining": generations,
        "comparison_rows": config.split.comparison_rows,
        "gate_rows": settings.compute.gate_rows,
        "gate_passes": settings.compute.gate_passes,
        "runtime_identity_hash": decode_record["identity"]["identity_hash"],
        "pilot_record_sha256": slice_import.file_digest(pilot),
        "decode_record_sha256": slice_import.file_digest(decode),
    }
    prior = (
        settings.compute.shared_pipeline_seconds
        + float(decode_record["elapsed_seconds"])
        + float(timing["seconds"])
    )
    projections = ladder.project(
        ladder.build_steps(settings),
        prior,
        float(timing["seconds_per_iteration"]),
        ladder.generation_seconds(
            generations,
            float(decode_record["timing"]["latency_mean_seconds"]),
            float(decode_record["retry_reserve"]),
        ),
    )

    try:
        selected = ladder.decide(
            projections, settings.compute.cap_hours * ladder.SECONDS_PER_HOUR
        )
    except ladder.LadderError as error:
        tracking.log_gate_stop(
            config, gate=ladder.GATE, reason=str(error), track=llm.TRACK
        )
        raise

    payload = ladder.build_decision(projections, selected, settings, inputs)
    ladder.check_blind(payload)
    ladder.write_once(payload, decision)
    with tracking.run(config, name=ladder.RUN_NAME, track=llm.TRACK) as logger:
        logger.log_params(ladder.run_params(payload))
        logger.log_metrics(ladder.run_metrics(payload))
        logger.log_artifact(decision)

    _LOGGER.info(
        "step %d of %d selected: %d rows over %d pass(es), %d iterations; "
        "projected total %.2fh against a cap of %.2fh",
        selected.step.position,
        len(projections),
        selected.step.train_rows,
        selected.step.epoch_equivalent,
        selected.step.iters,
        selected.total_hours,
        settings.compute.cap_hours,
    )
