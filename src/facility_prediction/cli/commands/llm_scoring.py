"""Generating answers under the schema, and scoring them once.

Every answer is written and hashed before any holdout target is
opened, and scoring refuses to run if that file no longer matches its
hash. A completion that fails twice becomes a typed failure that keeps
its place in every denominator; it is never replaced by a guess.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction import llm, tracking
from facility_prediction.cli.paths import (
    DEFAULT_PILOT_ADAPTER,
    LLM_MODEL_NAME,
    REVIEW_SPLIT,
)
from facility_prediction.data import samples, split, storage
from facility_prediction.evaluation import evaluate
from facility_prediction.features import features
from facility_prediction.llm import (
    buckets,
    constrained_decode,
    decode_check,
    final_pass,
    gate,
    gate_manifest,
    llm_data,
    llm_predict,
    slice_import,
)
from facility_prediction.llm import settings as llm_settings
from facility_prediction.models import baselines

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _FinalPaths:
    """Where the single scored pass reads from and writes to.

    Attributes:
        buckets: The frozen bucket manifest.
        prompts: The frozen prompt manifest.
        manifest: The frozen comparison manifest.
        adapter: Directory holding the frozen adapter.
        predictions: Destination of the per-row answers.
        report: Destination of the scored report.
    """

    buckets: pathlib.Path
    prompts: pathlib.Path
    manifest: pathlib.Path
    adapter: pathlib.Path
    predictions: pathlib.Path
    report: pathlib.Path


def _final_prompt_rows(
    config: config_module.Config,
    settings: llm_settings.Settings,
    frozen_ladder: list[buckets.Bucket],
    identifiers: list[str],
) -> list[dict[str, Any]]:
    """Render the target-free prompts for the sealed holdout rows.

    Leakage contract: reads each row's origin, its resident's earlier
    bookings, and the shared feature table. No target column is read.

    Args:
        config: Validated shared configuration.
        settings: The LLM configuration.
        frozen_ladder: The frozen delay buckets.
        identifiers: The frozen comparison manifest's identifiers.

    Returns:
        One target-free prompt row per identifier, in manifest order.
    """
    with storage.engine_scope(config) as engine:
        samples_table = storage.read_table(engine, storage.SAMPLES, config)
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        feature_table = features.build_features(samples_table, bookings, config)

    wanted = set(identifiers)
    chosen = samples_table.loc[
        samples_table[split.SAMPLE_ID_COLUMN].astype(str).isin(wanted)
    ]
    ordered = chosen.set_index(chosen[split.SAMPLE_ID_COLUMN].astype(str)).loc[
        identifiers
    ]
    rows = final_pass.prompt_rows(
        ordered.reset_index(drop=True),
        bookings,
        feature_table,
        frozen_ladder,
        config,
        settings,
    )
    final_pass.check_target_free(rows)
    return rows


@dataclasses.dataclass(frozen=True)
class _FinalTargets:
    """What the single scored pass marks its answers against.

    Attributes:
        targets: Target columns indexed by sample identifier.
        labels: The real notification label per row.
    """

    targets: pd.DataFrame
    labels: pd.Series


def _final_targets(
    config: config_module.Config,
    identifiers: list[str],
    frozen_ladder: list[buckets.Bucket],
) -> _FinalTargets:
    """Open the holdout targets for the sealed rows.

    Leakage contract: this reads HOLDOUT targets, which no other step
    in this track may do. It is called once, after the answers have
    been written and hashed, and never before.

    Args:
        config: Validated shared configuration.
        identifiers: The frozen comparison manifest's identifiers.
        frozen_ladder: The frozen delay buckets.

    Returns:
        The target columns and the real label per row, in manifest
        order.
    """
    with storage.engine_scope(config) as engine:
        samples_table = storage.read_table(engine, storage.SAMPLES, config)

    wanted = set(identifiers)
    chosen = samples_table.loc[
        samples_table[split.SAMPLE_ID_COLUMN].astype(str).isin(wanted)
    ]
    indexed = chosen.set_index(chosen[split.SAMPLE_ID_COLUMN].astype(str))
    return _FinalTargets(
        targets=indexed[list(evaluate.TARGET_COLUMNS.values())].loc[
            identifiers
        ],
        labels=gate.actual_labels(chosen, frozen_ladder).loc[identifiers],
    )


def _comparator_rates(
    config: config_module.Config,
    identifiers: list[str],
    opened: _FinalTargets,
) -> dict[str, float]:
    """Score the shared frequency baseline on the same sealed rows.

    The plan requires every track's result on the frozen manifest to be
    computed from identical definitions, so the comparator is marked
    here against the targets this pass has already opened rather than
    quoted from a different row set.

    Leakage contract: reads the baseline's stored predictions and the
    holdout targets already opened by this pass. It opens nothing new.

    Args:
        config: Validated shared configuration.
        identifiers: The frozen comparison manifest's identifiers.
        opened: The targets this pass opened after its freeze.

    Returns:
        The comparator's component rates on those rows.
    """
    with storage.engine_scope(config) as engine:
        stored = storage.read_table(engine, storage.PREDICTIONS, config)

    rows = stored.loc[stored["track"] == baselines.TRACK]
    indexed = rows.set_index(rows[split.SAMPLE_ID_COLUMN].astype(str))
    present = [name for name in identifiers if name in indexed.index]
    frame = indexed.loc[present, list(evaluate.PREDICTED_COLUMNS.values())]
    matches = evaluate.component_matches(
        frame.join(opened.targets, how="inner"),
        config.evaluation.notification_match_ratio,
    )
    return evaluate.match_summary(matches)


def _store_llm_predictions(
    config: config_module.Config,
    records: list[dict[str, Any]],
    frozen_ladder: list[buckets.Bucket],
) -> int:
    """Append this track's scored rows beside every other track's.

    Args:
        config: Validated shared configuration.
        records: The frozen answers.
        frozen_ladder: The frozen delay buckets.

    Returns:
        How many rows were written.
    """
    frame = llm_predict.to_frame(records, frozen_ladder).reset_index()
    frame = frame.drop(columns=[llm_predict.VALID_COLUMN])
    frame = frame.drop(columns=[llm_predict.BUCKET_COLUMN])
    frame["track"] = llm.TRACK
    frame["model"] = LLM_MODEL_NAME
    frame["predicted_transition_facility_id"] = None
    with storage.engine_scope(config) as engine:
        return storage.write_predictions(
            engine, frame, track=llm.TRACK, model=LLM_MODEL_NAME
        )


def run_llm_final(
    config: config_module.Config,
    llm_config: pathlib.Path,
    paths: _FinalPaths,
    from_cache: bool,
) -> None:
    """Score the frozen adapter once on the sealed holdout rows.

    Leakage contract: prompts carry no target, every answer is written
    and hashed before a holdout target is opened, and the join happens
    only against that frozen file.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        paths: The manifests, the adapter, and the outputs.
        from_cache: Recompute from saved answers instead of generating.

    Raises:
        FinalPassError: If the pass cannot be run or scored as declared.
        StackError: If the pinned path cannot be assembled here.
    """
    settings = llm_settings.load_settings(llm_config)
    frozen = split.load_manifest(paths.buckets)
    frozen_ladder = buckets.load_ladder(frozen)
    manifest = split.load_manifest(paths.manifest)
    identifiers = [str(value) for value in manifest["sample_ids"]]

    prompts = _final_prompt_rows(config, settings, frozen_ladder, identifiers)
    with storage.engine_scope(config) as engine:
        splits = storage.read_table(engine, storage.SPLITS, config)
    final_pass.check_manifest(prompts, identifiers, splits)

    try:
        if from_cache:
            records = llm_predict.read_records(paths.predictions)
            freeze = final_pass.Freeze(
                path=paths.predictions,
                sha256=slice_import.file_digest(paths.predictions),
                rows=len(records),
            )
        else:
            prompt_manifest = split.load_manifest(paths.prompts)
            schema = dict(prompt_manifest["schema"])
            runtime = constrained_decode.load_runtime(
                settings,
                schema,
                str(prompt_manifest["prompt_hash"]),
                str(prompt_manifest["schema_hash"]),
                paths.adapter,
            )
            records = llm_predict.run_arm(
                prompts,
                final_pass.TIER_B,
                functools.partial(
                    _decode_prompt,
                    config,
                    runtime,
                    schema,
                    settings.decoding.retries,
                ),
                runtime.identity,
            )
            tracking.flush_traces()
            freeze = final_pass.freeze_predictions(records, paths.predictions)
            _LOGGER.info(
                "froze %d answers at %s before reading any target",
                freeze.rows,
                freeze.sha256[:12],
            )

        # Every holdout target this branch ever reads is read below,
        # after the freeze above.
        opened = _final_targets(config, identifiers, frozen_ladder)
        marked = final_pass.score(
            freeze,
            llm_predict.to_frame(records, frozen_ladder),
            opened.targets,
            opened.labels,
            config,
            float(frozen["representation_ceiling"]),
        )
    except (
        final_pass.FinalPassError,
        llm_predict.PredictionError,
        constrained_decode.StackError,
    ) as error:
        tracking.log_gate_stop(
            config, gate=final_pass.GATE, reason=str(error), track=llm.TRACK
        )
        raise

    stored = _store_llm_predictions(config, records, frozen_ladder)
    comparator = _comparator_rates(config, identifiers, opened)
    report = final_pass.build_report(
        freeze,
        marked,
        llm_predict.operational_metrics(records),
        {
            "rows": len(prompts),
            "stored_rows": stored,
            "model": LLM_MODEL_NAME,
            "split": split.TEST,
            "manifest_sha256": slice_import.file_digest(paths.manifest),
            "adapter": str(paths.adapter),
            "replayed": from_cache,
        },
        comparator,
    )
    split.write_json(report, paths.report)

    with tracking.run(
        config, name=final_pass.RUN_NAME, track=llm.TRACK
    ) as logger:
        logger.log_params(final_pass.run_params(report))
        logger.log_metrics(final_pass.run_metrics(report))
        logger.log_artifact(paths.report)

    rates = report["rates"]
    _LOGGER.info(
        "scored %d holdout rows once: facility %.4f, weekday %.4f, "
        "hour %.4f, notification %.4f, overall %.4f; the shared baseline "
        "scores %.4f on the same rows (%+.4f)",
        report["context"]["rows"],
        rates[evaluate.FACILITY],
        rates[evaluate.USAGE_WEEKDAY],
        rates[evaluate.USAGE_HOUR],
        rates[evaluate.NOTIFICATION],
        rates["overall"],
        comparator["overall"],
        report["comparator"]["delta_overall"],
    )


def run_llm_gate_manifest(
    config: config_module.Config,
    llm_config: pathlib.Path,
    manifest: pathlib.Path,
) -> None:
    """Freeze the rows both development passes are judged on.

    Leakage contract: draws from the validation split only, and refuses
    to write if a holdout row reached the draw.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        manifest: Destination of the gate manifest.

    Raises:
        ManifestError: If the draw cannot be filled, or reached the
            sealed split.
    """
    settings = llm_settings.load_settings(llm_config)
    with storage.engine_scope(config) as engine:
        samples_table = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    labelled = samples_table.merge(
        splits, on=split.SAMPLE_ID_COLUMN, how="inner"
    )
    validation = labelled.loc[labelled[split.SPLIT_COLUMN] == REVIEW_SPLIT]
    holdout = set(
        labelled.loc[
            labelled[split.SPLIT_COLUMN] == split.TEST, split.SAMPLE_ID_COLUMN
        ].astype(str)
    )

    try:
        drawn = gate_manifest.draw(
            validation, settings.gate, settings.compute.gate_rows
        )
        payload = gate_manifest.build_manifest(
            drawn,
            settings.gate,
            {
                "validation_rows": len(validation),
                "samples_digest": samples.samples_digest(samples_table),
            },
        )
        gate_manifest.check_sealed(payload, holdout)
    except gate_manifest.ManifestError as error:
        tracking.log_gate_stop(
            config,
            gate="the gate manifest is drawn from unsealed rows only",
            reason=str(error),
            track=llm.TRACK,
        )
        raise

    split.write_json(payload, manifest)
    _LOGGER.info(
        "%d gate rows over %d strata, %d of them holding a single row",
        payload["rows"],
        len(payload["strata"]),
        payload["singleton_strata"],
    )


def _decode_prompt(
    config: config_module.Config,
    runtime: constrained_decode.Runtime,
    schema: dict[str, object],
    retries: int,
    prompt: dict[str, object],
) -> constrained_decode.Outcome:
    """Generate one answer for one manifest row, tracing the call.

    Args:
        config: Validated configuration.
        runtime: The loaded generation path.
        schema: The frozen output schema.
        retries: Deterministic retries after the first attempt.
        prompt: The rendered prompt row.

    Returns:
        The outcome, valid or typed failure.
    """
    with tracking.trace_generation(
        config,
        track=llm.TRACK,
        model=f"{runtime.identity['model_id']}@"
        f"{runtime.identity['model_revision']}",
        attributes=runtime.identity,
    ) as trace:
        return constrained_decode.decode_one(
            lambda: constrained_decode.generate_completion(
                runtime, str(prompt["system"]), str(prompt["prompt"])
            ),
            schema,
            config,
            retries,
            lambda index: trace.attempt(index, prompt=str(prompt["prompt"])),
        )


def _generate_arms(
    config: config_module.Config,
    settings: llm_settings.Settings,
    prompts: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Answer the manifest twice: without an adapter, then with one.

    Args:
        config: Validated configuration.
        settings: The LLM configuration.
        prompts: The rendered prompts, in manifest order.
        manifest: The frozen prompt manifest.

    Returns:
        Every record from both passes, the first pass first.
    """
    schema = dict(manifest["schema"])
    records: list[dict[str, Any]] = []
    for arm, adapter in (
        (llm_predict.ZERO_SHOT, None),
        (llm_predict.PILOT_ADAPTER, DEFAULT_PILOT_ADAPTER),
    ):
        runtime = constrained_decode.load_runtime(
            settings,
            schema,
            str(manifest["prompt_hash"]),
            str(manifest["schema_hash"]),
            adapter,
        )
        records.extend(
            llm_predict.run_arm(
                prompts,
                arm,
                functools.partial(
                    _decode_prompt,
                    config,
                    runtime,
                    schema,
                    settings.decoding.retries,
                ),
                runtime.identity,
            )
        )
        _LOGGER.info("%s: %d answers generated", arm, len(prompts))
    return records


def _mark_arm(
    records: list[dict[str, Any]], marking: _Marking
) -> dict[str, Any]:
    """Mark one pass against the rows it answered.

    Args:
        records: That pass's records.
        marking: What every pass is marked against.

    Returns:
        The pass's rates, per-row outcomes, support, confusion, and
        what the pass cost.
    """
    frame = llm_predict.to_frame(records, marking.ladder)
    marked = gate.arm_metrics(
        frame,
        marking.targets,
        marking.labels,
        marking.config,
        marking.ceiling,
    )
    marked["operational"] = llm_predict.operational_metrics(records)
    return marked


@dataclasses.dataclass(frozen=True)
class _GatePaths:
    """Where one gate pass reads from and writes to.

    Attributes:
        buckets: The frozen bucket manifest.
        prompts: The frozen prompt manifest.
        manifest: The frozen gate manifest.
        directory: Directory holding the rendered prompt files.
        pilot: The pilot training record.
        predictions: Destination of the per-row answers.
        report: Destination of the gate report.
    """

    buckets: pathlib.Path
    prompts: pathlib.Path
    manifest: pathlib.Path
    directory: pathlib.Path
    pilot: pathlib.Path
    predictions: pathlib.Path
    report: pathlib.Path


@dataclasses.dataclass(frozen=True)
class _Marking:
    """What every pass is marked against.

    Attributes:
        ladder: The frozen delay buckets.
        targets: Target columns indexed by sample identifier.
        labels: The real label per row.
        config: Validated shared configuration.
        ceiling: The representation ceiling the labels can reach.
    """

    ladder: list[buckets.Bucket]
    targets: pd.DataFrame
    labels: pd.Series
    config: config_module.Config
    ceiling: float


@dataclasses.dataclass(frozen=True)
class _GateInputs:
    """What the gate marks its answers against.

    Attributes:
        targets: Target columns indexed by sample identifier.
        labels: The real label per row.
        train_delays: Notification delays of the training split.
        identifiers: The manifest's rows, in order.
    """

    targets: pd.DataFrame
    labels: pd.Series
    train_delays: pd.Series
    identifiers: list[str]


def _gate_inputs(
    config: config_module.Config,
    manifest: dict[str, Any],
    ladder: list[buckets.Bucket],
) -> _GateInputs:
    """Read what the gate marks its answers against.

    Leakage contract: reads validation targets for the manifest rows
    and training delays for the references. No holdout row is read.

    Args:
        config: Validated configuration.
        manifest: The frozen gate manifest.
        ladder: The frozen delay buckets.

    Returns:
        The target frame, the real labels, the training delays, and the
        manifest's identifiers in order.
    """
    with storage.engine_scope(config) as engine:
        samples_table = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    labelled = samples_table.merge(
        splits, on=split.SAMPLE_ID_COLUMN, how="inner"
    )
    identifiers = [str(value) for value in manifest["sample_ids"]]
    chosen = labelled.loc[
        labelled[split.SAMPLE_ID_COLUMN].astype(str).isin(set(identifiers))
    ]
    targets = chosen.set_index(chosen[split.SAMPLE_ID_COLUMN].astype(str))[
        list(evaluate.TARGET_COLUMNS.values())
    ]
    training = labelled.loc[labelled[split.SPLIT_COLUMN] == split.TRAIN]
    return _GateInputs(
        targets=targets.loc[identifiers],
        labels=gate.actual_labels(chosen, ladder).loc[identifiers],
        train_delays=training[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]],
        identifiers=identifiers,
    )


def run_llm_gate(
    config: config_module.Config,
    llm_config: pathlib.Path,
    paths: _GatePaths,
    from_cache: bool,
) -> None:
    """Score both development passes and decide on the full training.

    Leakage contract: the prompts are target-free, targets are joined
    only after every answer exists, and the holdout is never read.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        paths: The manifests, the prompt directory, and the outputs.
        from_cache: Recompute from saved answers instead of generating.

    Raises:
        GateError: If the evidence does not justify the full training.
        StackError: If the pinned path cannot be assembled here.
    """
    settings = llm_settings.load_settings(llm_config)
    frozen = split.load_manifest(paths.buckets)
    frozen_ladder = buckets.load_ladder(frozen)
    manifest = split.load_manifest(paths.manifest)
    inputs = _gate_inputs(config, manifest, frozen_ladder)

    rows = {
        str(row["sample_id"]): row
        for row in decode_check.read_rows(paths.directory / llm_data.VALID_FILE)
    }
    prompts = [rows[identifier] for identifier in inputs.identifiers]

    if from_cache:
        records = llm_predict.read_records(paths.predictions)
    else:
        records = _generate_arms(
            config, settings, prompts, split.load_manifest(paths.prompts)
        )
        llm_predict.write_records(records, paths.predictions)
        tracking.flush_traces()

    marking = _Marking(
        ladder=frozen_ladder,
        targets=inputs.targets,
        labels=inputs.labels,
        config=config,
        ceiling=float(frozen["representation_ceiling"]),
    )
    arms = {
        arm: _mark_arm(llm_predict.arm_records(records, arm), marking)
        for arm in (llm_predict.ZERO_SHOT, llm_predict.PILOT_ADAPTER)
    }
    report = gate.build_report(
        arms,
        {
            metric: gate.paired_bootstrap(
                arms[llm_predict.ZERO_SHOT]["per_row"][metric],
                arms[llm_predict.PILOT_ADAPTER]["per_row"][metric],
                settings.gate,
            )
            for metric in (gate.BUCKET_ACCURACY, gate.NOTIFICATION_MATCH)
        },
        gate.reference_metrics(
            inputs.train_delays,
            inputs.labels,
            inputs.targets,
            frozen_ladder,
            config,
        ),
        {
            "rows": len(prompts),
            "split": REVIEW_SPLIT,
            "manifest_sha256": slice_import.file_digest(paths.manifest),
            "adapter": str(DEFAULT_PILOT_ADAPTER),
            "replayed": from_cache,
        },
    )
    split.write_json(report, paths.report)

    try:
        gate.check_gate(report, split.load_manifest(paths.pilot))
    except gate.GateError as error:
        tracking.log_gate_stop(
            config, gate=gate.GATE, reason=str(error), track=llm.TRACK
        )
        raise
    finally:
        _log_gate(report)

    with tracking.run(config, name=gate.RUN_NAME, track=llm.TRACK) as logger:
        logger.log_params(gate.run_params(report))
        logger.log_metrics(gate.run_metrics(report))
        logger.log_artifact(paths.report)
        logger.log_artifact(paths.predictions)


def _log_gate(report: dict[str, Any]) -> None:
    """Print both passes and the intervals between them.

    Args:
        report: The assembled report.
    """
    for name, arm in report["arms"].items():
        _LOGGER.info(
            "%-14s bucket %.4f · notification %.4f · facility %.4f · "
            "score %.4f of 4 · valid %.4f",
            name,
            arm["rates"][gate.BUCKET_ACCURACY],
            arm["rates"][evaluate.NOTIFICATION],
            arm["rates"][evaluate.FACILITY],
            arm["rates"]["score_mean"],
            arm["rates"]["valid"],
        )
    for metric, interval in report["paired_intervals"].items():
        _LOGGER.info(
            "%-18s trained minus zero-shot %+.4f, 95%% interval %+.4f to %+.4f",
            metric,
            interval["difference"],
            interval["low"],
            interval["high"],
        )
