"""Reporting that runs beside the pipeline rather than inside it.

The generator profile, the error slices, the post-freeze stretch
variants, the model registry, and the one table that puts every track's
result on the same rows.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction import tracking
from facility_prediction.cli.commands.verification import review_split
from facility_prediction.cli.paths import (
    DEFAULT_COMPARISON_MANIFEST,
    DEFAULT_SEAL,
    EVALUATION_SPLIT,
)
from facility_prediction.data import profiles, split, storage
from facility_prediction.evaluation import ablations, errors, evaluate
from facility_prediction.evaluation import freeze as freeze_module
from facility_prediction.features import features
from facility_prediction.models import train

_LOGGER = logging.getLogger(__name__)


def run_profile(
    config: config_module.Config,
    profile: pathlib.Path,
    plots: pathlib.Path,
) -> None:
    """Profile the generated dataset and render its figures.

    Leakage contract: reads the booking table and nothing downstream of
    it. No sample, split, feature, or prediction is touched, so this
    cannot influence any fit.

    Args:
        config: Validated configuration.
        profile: Destination JSON for the profile.
        plots: Destination directory for the figures.

    Raises:
        SystemExit: If the profile contradicts itself.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)

    payload = profiles.build_profile(bookings, config)
    problems = profiles.check_profile(payload)
    digest = profiles.write_profile(payload, profile)
    written = profiles.write_plots(bookings, payload, plots)

    _LOGGER.info("wrote %s (sha256=%s)", profile, digest)
    _LOGGER.info("wrote %d plots to %s", len(written), plots)
    if problems:
        for problem in problems:
            _LOGGER.error("profile check failed: %s", problem)
        raise SystemExit(1)
    _LOGGER.info("profile is internally consistent; no contradictions found")


def run_tune(
    config: config_module.Config,
    models: pathlib.Path,
    metrics: pathlib.Path,
    report: pathlib.Path,
    minutes: float = ablations.DEFAULT_MINUTES_BUDGET,
) -> None:
    """Run the declared stretch variants on validation, after the freeze.

    Leakage contract: fits on training rows and scores the validation
    split. No holdout row is read, and nothing the submission depends on
    is written — this writes one report and no model, prediction, or
    metric.

    Args:
        config: Validated configuration.
        models: Directory the frozen heads were written to.
        metrics: The committed metrics, for the frozen reference.
        report: Destination JSON for the ablation report.
        minutes: Wall-clock ceiling for the run. Deliberately not a
            configuration key: the configuration is frozen before the
            holdout is scored, and a key added afterwards would
            invalidate that freeze without changing a shipped number.

    Raises:
        SystemExit: If a variant cannot be run as declared.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    table = features.validate_features(
        features.build_features(frame, bookings, config), config
    )
    rows_of = {
        name: table.loc[
            table[split.SAMPLE_ID_COLUMN].isin(
                set(
                    splits.loc[
                        splits[split.SPLIT_COLUMN] == name,
                        split.SAMPLE_ID_COLUMN,
                    ]
                )
            )
        ]
        for name in (split.TRAIN, EVALUATION_SPLIT)
    }

    committed = json.loads(metrics.read_text(encoding="utf-8"))
    reference = {
        head: float(value)
        for head, value in committed[EVALUATION_SPLIT]["model"][
            "matches"
        ].items()
        if head in train.HEAD_NAMES
    }

    chosen, skipped, planned = ablations.plan_run(
        config.catboost.stretch_fit_budget
    )
    frozen_heads = train.load(models, config)
    completed: list[dict[str, Any]] = []
    started = time.monotonic()
    stopping_point = ""
    for index, variant in enumerate(chosen):
        elapsed = time.monotonic() - started
        spent = sum(int(entry["fits"]) for entry in completed)
        if ablations.would_exceed_time(
            elapsed,
            spent,
            variant.fits,
            minutes,
        ):
            stopping_point = (
                f"stopped before {variant.name!r}: at {elapsed / 60:.1f} "
                f"minutes over {spent} fit(s), its {variant.fits} fit(s) "
                f"project past the "
                f"{minutes:.0f}-minute budget"
            )
            _LOGGER.info("%s", stopping_point)
            skipped = list(chosen[index:]) + list(skipped)
            break
        try:
            result = ablations.run_variant(
                variant,
                rows_of[split.TRAIN],
                rows_of[EVALUATION_SPLIT],
                frame,
                config,
                frozen_heads,
            )
        except ablations.AblationError as error:
            _LOGGER.error("%s", error)
            raise SystemExit(1) from error
        completed.append(result)
        _LOGGER.info(
            "variant %s: %s (%.1fs)",
            variant.name,
            ", ".join(
                f"{head} {value:.4f} against "
                f"{reference.get(head, float('nan')):.4f}"
                for head, value in result["matches"].items()
            ),
            result["seconds"],
        )

    payload = ablations.build_report(
        completed,
        skipped,
        config.catboost.stretch_fit_budget,
        planned,
        reference,
        EVALUATION_SPLIT,
        elapsed_seconds=time.monotonic() - started,
        minutes_budget=minutes,
        stopping_point=stopping_point,
    )
    digest = ablations.write_report(payload, report)
    _LOGGER.info(
        "wrote %s (sha256=%s): %d of %d declared fits spent, %d variant(s) "
        "beat the frozen head. %s",
        report,
        digest,
        payload["budget"]["completed_fits"],
        payload["budget"]["declared_fits"],
        len(payload["improved_on_frozen"]),
        payload["budget"]["stopping_point"],
    )


def cross_track_comparison(
    stored: pd.DataFrame,
    samples_table: pd.DataFrame,
    manifest_ids: set[str],
    config: config_module.Config,
) -> list[dict[str, Any]]:
    """Score every track's rows over the shared comparison manifest.

    One query, every track. The manifest is the seeded holdout subset
    that bounds cross-track comparison, so a track with no rows appears
    with a zero count rather than vanishing — a track that has not run
    is a fact about the project, not an absence to hide.

    Args:
        stored: Every stored prediction row.
        samples_table: The sample table, carrying targets.
        manifest_ids: The comparison manifest's sample ids.
        config: Validated configuration.

    Returns:
        One row per (track, model) with its match rates on the manifest.
    """
    rows: list[dict[str, Any]] = []
    scoped = samples_table.loc[
        samples_table[split.SAMPLE_ID_COLUMN].isin(manifest_ids)
    ]
    for (track, model), group in stored.groupby(["track", "model"], sort=True):
        joined = scoped.merge(
            group, on=split.SAMPLE_ID_COLUMN, how="inner"
        ).reset_index(drop=True)
        entry: dict[str, Any] = {
            "track": str(track),
            "model": str(model),
            "rows_on_manifest": len(joined),
        }
        if not joined.empty:
            matches = evaluate.component_matches(
                joined, config.evaluation.notification_match_ratio
            )
            entry.update(
                {
                    key: round(float(value), 4)
                    for key, value in evaluate.match_summary(matches).items()
                }
            )
        rows.append(entry)
    return rows


def run_register(
    config: config_module.Config,
    models: pathlib.Path,
    record: pathlib.Path,
    registry: pathlib.Path,
) -> None:
    """Register the frozen heads and query every track at once.

    Leakage contract: registers files that already exist and scores
    rows that were already predicted. Nothing is fitted.

    Args:
        config: Validated configuration.
        models: Directory the fitted heads were written to.
        record: The freeze JSON, whose hash tags each version.
        registry: Destination JSON recording what was registered.

    Raises:
        SystemExit: If the configuration was never frozen.
    """
    try:
        frozen = freeze_module.read_freeze(record)
    except freeze_module.FreezeError as error:
        _LOGGER.error("%s", error)
        raise SystemExit(1) from error

    registered: dict[str, Any] = {}
    with tracking.run(
        config, name="model_registry", track=train.TRACK
    ) as logger:
        logger.log_params({"config_hash": frozen["config_hash"]})
        for name in train.HEAD_NAMES:
            path = train.model_path(models, name)
            if not path.is_file():
                _LOGGER.error("no saved head at %s", path)
                raise SystemExit(1)
            logger.log_artifact(path)
            registered[f"{train.MODEL_NAME}_{name}"] = tracking.register_model(
                config,
                name=f"{train.MODEL_NAME}_{name}",
                artifact=path,
                run_id=logger.run_id,
                tags={
                    "frozen": "true",
                    "config_hash": frozen["config_hash"],
                    "track": train.TRACK,
                },
            )

    with storage.engine_scope(config) as engine:
        frame = storage.read_table(engine, storage.SAMPLES, config)
        stored = storage.read_table(engine, storage.PREDICTIONS, config)

    manifest = split.load_manifest(DEFAULT_COMPARISON_MANIFEST)
    comparison = cross_track_comparison(
        stored, frame, set(manifest["sample_ids"]), config
    )
    resolved = {
        name: list(tracking.registered_versions(config, name))
        for name in registered
    }
    payload = {
        "config_hash": frozen["config_hash"],
        "head_sources": frozen["head_sources"],
        "registered": registered,
        "resolved_versions": resolved,
        "comparison_manifest_rows": len(manifest["sample_ids"]),
        "cross_track_comparison": comparison,
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    _LOGGER.info(
        "registered %d heads at config %s; one query returned %d "
        "(track, model) pairs over %d manifest rows",
        len(registered),
        frozen["config_hash"],
        len(comparison),
        len(manifest["sample_ids"]),
    )


def run_slices(
    config: config_module.Config,
    models: pathlib.Path,
    analysis: pathlib.Path,
    plots: pathlib.Path,
) -> None:
    """Slice the scored split and explain the heads that produced it.

    Leakage contract: reads saved predictions and the features they came
    from. Nothing is fitted. The split is :data:`EVALUATION_SPLIT`, so
    the holdout stays sealed until it is scored once.

    Args:
        config: Validated configuration.
        models: Directory the fitted heads were written to.
        analysis: Destination JSON for the analysis.
        plots: Destination directory for the importance figure.

    Raises:
        SystemExit: If the analysis contradicts itself.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)
        stored = storage.read_table(engine, storage.PREDICTIONS, config)

    table = features.validate_features(
        features.build_features(frame, bookings, config), config
    )
    sliced_split, _ = review_split(DEFAULT_SEAL)
    scored_ids = set(
        splits.loc[
            splits[split.SPLIT_COLUMN] == sliced_split,
            split.SAMPLE_ID_COLUMN,
        ]
    )
    training_ids = list(
        splits.loc[
            splits[split.SPLIT_COLUMN] == split.TRAIN, split.SAMPLE_ID_COLUMN
        ]
    )
    rows = stored.loc[
        (stored["model"] == train.CHAMPION_MODEL_NAME)
        & stored[split.SAMPLE_ID_COLUMN].isin(scored_ids)
    ].merge(frame, on=split.SAMPLE_ID_COLUMN, how="inner")

    heads = train.load(models, config)
    sample = errors.importance_sample(
        table.loc[table[split.SAMPLE_ID_COLUMN].isin(scored_ids)],
        seed=config.seed,
    )
    importance = {}
    for name, model in heads.items():
        design = train.build_design(
            name, sample, train.align_targets(sample, frame), config
        )
        importance[name] = errors.feature_importance(
            model, design.pool(), list(design.matrix.columns)
        )

    payload = errors.build_analysis(
        rows, table, training_ids, config, sliced_split, importance
    )
    problems = errors.check_analysis(payload)
    digest = errors.write_analysis(payload, analysis)
    errors.write_importance_plot(importance, plots / errors.IMPORTANCE_PLOT)

    _LOGGER.info(
        "wrote %s (sha256=%s): %d slices over %d %s rows, SHAP on %d",
        analysis,
        digest,
        len(payload["slices"]),
        len(rows),
        sliced_split,
        len(sample),
    )
    if problems:
        for problem in problems:
            _LOGGER.error("analysis check failed: %s", problem)
        raise SystemExit(1)


def _matching_slices(
    analysis: pathlib.Path, rendered_split: str
) -> list[dict[str, Any]]:
    """Return the error slices, but only if they describe this split.

    Args:
        analysis: The error-analysis JSON.
        rendered_split: The split the workbook is rendering.

    Returns:
        The slice rows, or nothing when the analysis is absent or
        describes a different split.
    """
    if not analysis.is_file():
        return []
    payload = json.loads(analysis.read_text(encoding="utf-8"))
    if payload.get("provenance", {}).get("split") != rendered_split:
        return []
    return list(payload.get("slices", []))
