"""Everything the LLM track needs before a model is trained.

Importing the shared slice, cutting the delay buckets, rendering the
prompts, and proving the pinned decoding stack answers a fixture set
byte-identically.
"""

from __future__ import annotations

import logging
import pathlib

from facility_prediction import config as config_module
from facility_prediction import llm, tracking
from facility_prediction.cli.paths import (
    DEFAULT_COMPARISON_MANIFEST,
    DEFAULT_GENERATION_SUMMARY,
    DEFAULT_SAMPLE_SUMMARY,
    DEFAULT_SPLIT_MANIFEST,
    REVIEW_SPLIT,
    SLICE_SOURCES,
)
from facility_prediction.data import split, storage
from facility_prediction.evaluation import evaluate
from facility_prediction.features import features
from facility_prediction.llm import (
    buckets,
    constrained_decode,
    decode_check,
    llm_data,
    slice_import,
)
from facility_prediction.llm import settings as llm_settings

_LOGGER = logging.getLogger(__name__)


def run_llm_import_slice(
    config: config_module.Config, record: pathlib.Path
) -> None:
    """Record what the LLM branch inherits from the shared spine.

    Reads the tables back, checks them against the manifests written
    beside them, and writes the record. A disagreement is logged as a
    gate stop rather than adopted.

    Leakage contract: the comparator is scored on
    :data:`REVIEW_SPLIT`, because the holdout stays sealed until each
    track's single scoring pass.

    Args:
        config: Validated configuration.
        record: Destination of the inherited-facts record.

    Raises:
        SliceImportError: If a source is missing, disagrees with the
            store, or carries a value that was never measured.
    """
    with storage.engine_scope(config) as engine:
        spine = slice_import.Spine(
            bookings=storage.read_table(engine, storage.BOOKINGS, config),
            samples=storage.read_table(engine, storage.SAMPLES, config),
            splits=storage.read_table(engine, storage.SPLITS, config),
            predictions=storage.read_table(engine, storage.PREDICTIONS, config),
        )

    try:
        sources = {
            name: slice_import.file_digest(path)
            for name, path in SLICE_SOURCES.items()
        }
        manifests = slice_import.Manifests(
            generation=split.load_manifest(DEFAULT_GENERATION_SUMMARY),
            sample=split.load_manifest(DEFAULT_SAMPLE_SUMMARY),
            split=split.load_manifest(DEFAULT_SPLIT_MANIFEST),
            comparison=split.load_manifest(DEFAULT_COMPARISON_MANIFEST),
        )
        slice_import.check_agreement(spine, manifests)
        payload = slice_import.build_record(
            spine, manifests, sources, config, REVIEW_SPLIT
        )
        slice_import.check_measured(payload)
    except slice_import.SliceImportError as error:
        tracking.log_gate_stop(
            config,
            gate=slice_import.GATE,
            reason=str(error),
            track=llm.TRACK,
        )
        raise

    digest = split.write_json(payload, record)
    with tracking.run(
        config, name=slice_import.RUN_NAME, track=llm.TRACK
    ) as logger:
        logger.log_params(slice_import.run_params(payload))
        logger.log_metrics(slice_import.run_metrics(payload))
        logger.log_artifact(record)

    comparator = payload["comparator"]
    _LOGGER.info(
        "imported %d bookings, %d samples and %d comparator rows on the %s "
        "split (sha256=%s)",
        payload["counts"]["bookings"],
        payload["counts"]["samples"],
        comparator["rows"],
        comparator["split"],
        digest,
    )


def run_llm_buckets(
    config: config_module.Config,
    llm_config: pathlib.Path,
    manifest: pathlib.Path,
) -> None:
    """Freeze the notification delay ladder and score its ceiling.

    Leakage contract: the ladder, the merges, and every representative
    are computed from training rows only. Validation and holdout
    delays are never read.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        manifest: Destination of the bucket manifest.

    Raises:
        BucketError: If the ladder cannot be built, or its ceiling is
            below the configured gate.
    """
    settings = llm_settings.load_settings(llm_config)
    with storage.engine_scope(config) as engine:
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    labelled = frame.merge(splits, on=split.SAMPLE_ID_COLUMN, how="inner")
    training = labelled.loc[labelled[split.SPLIT_COLUMN] == split.TRAIN]
    delays = training[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]]
    match_ratio = config.evaluation.notification_match_ratio

    try:
        ladder, merges = buckets.build(
            settings.notification_buckets, delays, match_ratio
        )
        ceiling, per_bucket = buckets.representation_ceiling(
            ladder, delays, match_ratio
        )
        payload = buckets.build_manifest(
            ladder,
            merges,
            ceiling,
            per_bucket,
            settings.notification_buckets,
            match_ratio,
        )
        split.write_json(payload, manifest)
        buckets.check_ceiling(
            ceiling, settings.notification_buckets.ceiling_gate
        )
    except buckets.BucketError as error:
        tracking.log_gate_stop(
            config, gate=buckets.GATE, reason=str(error), track=llm.TRACK
        )
        raise

    with tracking.run(config, name=buckets.RUN_NAME, track=llm.TRACK) as logger:
        logger.log_params(
            {str(key): str(value) for key, value in payload["settings"].items()}
        )
        logger.log_metrics(
            {
                "representation_ceiling": ceiling,
                "labels": float(len(ladder)),
                "merges": float(len(merges)),
                "train_rows": float(len(delays)),
            }
        )
        logger.log_artifact(manifest)

    _LOGGER.info(
        "%d labels after %d merges; representation ceiling %.4f (gate %.2f)",
        len(ladder),
        len(merges),
        ceiling,
        settings.notification_buckets.ceiling_gate,
    )


def run_build_llm_data(
    config: config_module.Config,
    llm_config: pathlib.Path,
    buckets_manifest: pathlib.Path,
    directory: pathlib.Path,
    manifest: pathlib.Path,
) -> None:
    """Render the training and validation prompt datasets.

    Leakage contract: only the training and validation splits are
    rendered, and the holdout check runs before anything is written.

    Args:
        config: Validated shared configuration.
        llm_config: Path to the LLM configuration.
        buckets_manifest: Path to the frozen bucket manifest.
        directory: Destination directory for the JSONL files.
        manifest: Destination of the prompt manifest.

    Raises:
        DataError: If a holdout row reaches a training file, or a
            sample has no shared feature row.
    """
    settings = llm_settings.load_settings(llm_config)
    frozen = split.load_manifest(buckets_manifest)
    ladder = buckets.load_ladder(frozen)

    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        samples_table = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    table = features.validate_features(
        features.build_features(samples_table, bookings, config), config
    )
    labelled = samples_table.merge(
        splits, on=split.SAMPLE_ID_COLUMN, how="inner"
    )

    written: dict[str, dict[str, float | int | str]] = {}
    example = ""
    for name, filename in llm_data.SPLIT_FILES.items():
        chosen = labelled.loc[labelled[split.SPLIT_COLUMN] == name]
        rows = llm_data.build_rows(
            chosen, bookings, table, ladder, config, settings
        )
        llm_data.check_sealed(rows, splits)
        kept, dropped = llm_data.deduplicate(rows)
        digest = llm_data.write_jsonl(kept, directory / filename)
        written[name] = {
            "file": filename,
            "rows": len(kept),
            "duplicates_dropped": dropped,
            "sha256": digest,
            **llm_data.length_percentiles(kept),
        }
        example = example or kept[0]["prompt"]
        _LOGGER.info(
            "%s: %d rows, %d exact duplicates dropped (sha256=%s)",
            filename,
            len(kept),
            dropped,
            digest,
        )

    payload = llm_data.build_manifest(
        written, ladder, config, settings, example
    )
    split.write_json(payload, manifest)

    with tracking.run(
        config, name=llm_data.RUN_NAME, track=llm.TRACK
    ) as logger:
        logger.log_params(
            {
                "prompt_version": str(payload["prompt_version"]),
                "prompt_hash": str(payload["prompt_hash"]),
                "schema_hash": str(payload["schema_hash"]),
                "labels": str(len(payload["labels"])),
            }
        )
        logger.log_metrics(
            {
                f"{name}_rows": float(str(shape["rows"]))
                for name, shape in written.items()
            }
        )
        logger.log_artifact(manifest)


def _decode_fixture(
    config: config_module.Config,
    runtime: constrained_decode.Runtime,
    schema: dict[str, object],
    retries: int,
    fixture: decode_check.Fixture,
) -> constrained_decode.Outcome:
    """Generate one answer for one fixture, tracing the call.

    Args:
        config: Validated configuration.
        runtime: The loaded generation path.
        schema: The frozen output schema.
        retries: Deterministic retries after the first attempt.
        fixture: The prompt to answer.

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
                runtime, fixture.system, fixture.prompt
            ),
            schema,
            config,
            retries,
            lambda index: trace.attempt(index, prompt=fixture.prompt),
        )


def run_llm_decode_check(
    config: config_module.Config,
    llm_config: pathlib.Path,
    prompt_manifest: pathlib.Path,
    directory: pathlib.Path,
    record: pathlib.Path,
) -> None:
    """Run the pinned decoding stack against its compatibility fixtures.

    Leakage contract: fixtures are drawn from the rendered validation
    prompts, which carry no target. The holdout is not read.

    Args:
        config: Validated configuration.
        llm_config: Path to the LLM configuration.
        prompt_manifest: Path to the frozen prompt manifest.
        directory: Directory holding the rendered prompt files.
        record: Destination of the fixture record.

    Raises:
        FixtureError: If the stack does not clear every fixture.
        StackError: If the pinned path cannot be assembled here.
    """
    settings = llm_settings.load_settings(llm_config)
    manifest = split.load_manifest(prompt_manifest)
    schema = dict(manifest["schema"])
    decoding = settings.decoding

    try:
        rows = decode_check.read_rows(directory / llm_data.VALID_FILE)
        fixtures = decode_check.select_fixtures(
            rows, decoding.fixtures, config.seed
        )
        runtime = constrained_decode.load_runtime(
            settings,
            schema,
            str(manifest["prompt_hash"]),
            str(manifest["schema_hash"]),
        )
        results = decode_check.run_fixtures(
            fixtures,
            lambda fixture: _decode_fixture(
                config, runtime, schema, decoding.retries, fixture
            ),
            decoding.repeat_fixtures,
        )
        payload = decode_check.build_report(
            results,
            runtime.identity,
            decoding,
            {
                "file": llm_data.VALID_FILE,
                "rows_available": len(rows),
                "seed": config.seed,
            },
        )
        split.write_json(payload, record)
        decode_check.check_gate(payload, decoding)
    except (
        decode_check.FixtureError,
        constrained_decode.StackError,
    ) as error:
        tracking.log_gate_stop(
            config,
            gate=decode_check.GATE,
            reason=str(error),
            track=llm.TRACK,
        )
        raise

    with tracking.run(
        config, name=decode_check.RUN_NAME, track=llm.TRACK
    ) as logger:
        logger.log_params(decode_check.run_params(payload))
        logger.log_metrics(decode_check.run_metrics(payload))
        logger.log_artifact(record)
    tracking.flush_traces()

    _LOGGER.info(
        "%d fixtures, %d valid, %d retried, %d repeated identically; "
        "latency p50 %.2fs, peak memory %.2f GiB",
        payload["counts"]["fixtures"],
        payload["counts"]["valid"],
        payload["counts"]["retried"],
        payload["counts"]["repeated_identical"],
        payload["timing"]["latency_p50_seconds"],
        payload["peak_memory_gib"],
    )
