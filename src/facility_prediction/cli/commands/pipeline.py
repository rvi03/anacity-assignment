"""The eight stages, in the order a run performs them.

generate -> samples -> split -> features -> baselines -> train ->
evaluate -> review. The split is drawn before features because the
feature table's train/evaluation dtype check needs to know which rows
are which.

Leakage contract: these functions order stages and move frames between
them. They compute no feature and read no label, so they hold no origin
of their own; each stage keeps its own contract.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction import tracking
from facility_prediction.cli.paths import (
    EVALUATION_SPLIT,
    HOLDOUT_SPLIT,
)
from facility_prediction.data import generate, samples, split, storage
from facility_prediction.evaluation import evaluate
from facility_prediction.features import features
from facility_prediction.models import baselines, train

_LOGGER = logging.getLogger(__name__)


def run_generate(
    config: config_module.Config,
    export: pathlib.Path,
    summary: pathlib.Path,
) -> None:
    """Generate the dataset, check it, store it, and export it.

    The acceptance checks run here, not only in the test suite, so a
    dataset that fails one stops the pipeline.

    Args:
        config: Validated configuration.
        export: Destination of the deterministic CSV export.
        summary: Destination of the generation summary.

    Raises:
        RuntimeError: If the stored rows do not reproduce the generated
            table's canonical digest, which would mean the store of
            record and the export disagree.
    """
    frame, audit = generate.generate_audited(config)
    report = generate.check_dataset(frame, config, audit)
    split.write_json(report, summary)
    _LOGGER.info("%d acceptance checks passed", len(report["checks"]))
    with storage.engine_scope(config) as engine:
        storage.write_table(engine, storage.BOOKINGS, frame)
        stored = storage.read_table(engine, storage.BOOKINGS, config)

    generate.write_bookings(frame, export)
    digest = generate.bookings_digest(frame)
    _LOGGER.info("bookings digest: %s", digest)
    if generate.bookings_digest(stored) != digest:
        msg = "the stored bookings differ from the generated ones"
        raise RuntimeError(msg)


def run_samples(config: config_module.Config, summary: pathlib.Path) -> None:
    """Build rolling-origin samples and report what could not be used.

    Args:
        config: Validated configuration.
        summary: Destination of the sample-count summary.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = samples.build_samples(bookings, config)
        counts = samples.summarise_samples(bookings, frame, config)
        storage.write_table(engine, storage.SAMPLES, frame)

    samples.write_summary(
        counts,
        {
            "seed": config.seed,
            "timezone": config.timezone,
            "bookings_digest": generate.bookings_digest(bookings),
            "samples_digest": samples.samples_digest(frame),
        },
        summary,
    )
    _LOGGER.info(
        "%d samples from %d bookings; %d residents excluded (no prior "
        "history), %d with no booking at all",
        counts.samples,
        counts.bookings,
        counts.residents_excluded_no_prior_history,
        counts.residents_without_bookings,
    )


def run_split(
    config: config_module.Config,
    manifest: pathlib.Path,
    comparison: pathlib.Path,
) -> None:
    """Freeze the chronological split and the comparison manifest.

    Args:
        config: Validated configuration.
        manifest: Destination of the split manifest.
        comparison: Destination of the holdout comparison manifest.
    """
    with storage.engine_scope(config) as engine:
        frame = storage.read_table(engine, storage.SAMPLES, config)
        cutoffs = split.compute_cutoffs(frame, config)
        labelled = split.assign_split(frame, cutoffs)
        split.check_boundaries(labelled)
        storage.write_table(
            engine,
            storage.SPLITS,
            labelled[[split.SAMPLE_ID_COLUMN, split.SPLIT_COLUMN]],
        )

    provenance = {
        "seed": config.seed,
        "timezone": config.timezone,
        "samples_digest": samples.samples_digest(frame),
    }
    payload = split.build_split_manifest(labelled, cutoffs, config, provenance)
    split.write_json(payload, manifest)

    holdout = split.draw_comparison_manifest(labelled, config)
    split.write_json(
        {"provenance": provenance, "sample_ids": list(holdout)}, comparison
    )

    for name, shape in payload["splits"].items():
        _LOGGER.info(
            "%-10s %6d rows  %s -> %s",
            name,
            shape["rows"],
            shape["first"],
            shape["last"],
        )
    _LOGGER.info("comparison manifest: %d holdout rows", len(holdout))


def run_features(config: config_module.Config, manifest: pathlib.Path) -> None:
    """Build the feature table, check its schema, and record its shape.

    Args:
        config: Validated configuration.
        manifest: Destination of the feature manifest.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    table = features.validate_features(
        features.build_features(frame, bookings, config), config
    )
    training = set(
        splits.loc[
            splits[split.SPLIT_COLUMN] == split.TRAIN,
            split.SAMPLE_ID_COLUMN,
        ]
    )
    is_training = table["sample_id"].isin(training)
    features.check_consistent_dtypes(
        table.loc[is_training], table.loc[~is_training]
    )

    features.write_manifest(
        features.build_manifest(
            table,
            config,
            {
                "seed": config.seed,
                "timezone": config.timezone,
                "bookings_digest": generate.bookings_digest(bookings),
                "samples_digest": samples.samples_digest(frame),
                "features_digest": features.features_digest(table),
            },
        ),
        manifest,
    )
    _LOGGER.info(
        "%d feature rows, %d columns (%d categorical), %d training rows",
        len(table),
        len(features.feature_columns(config)),
        len(features.categorical_feature_names(config)),
        int(is_training.sum()),
    )


def run_baselines(config: config_module.Config) -> None:
    """Fit the baselines on training rows and score every sample.

    Args:
        config: Validated configuration.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)
        labelled = frame.merge(splits, on=split.SAMPLE_ID_COLUMN, how="inner")
        train = labelled.loc[labelled[split.SPLIT_COLUMN] == split.TRAIN]
        audited = split.check_fit_membership(
            train[split.SAMPLE_ID_COLUMN].tolist(), splits
        )

        with tracking.run(
            config, name=baselines.MODEL_NAME, track=baselines.TRACK
        ) as logger:
            logger.log_params(
                {
                    "seed": config.seed,
                    "model": baselines.MODEL_NAME,
                    "min_prior_bookings": config.evaluation.min_prior_bookings,
                    "config_hash": config_module.config_hash(config),
                }
            )
            fallbacks = baselines.fit(bookings, train)
            predictions = baselines.predict(labelled, bookings, fallbacks)
            predictions.insert(0, "model", baselines.MODEL_NAME)
            predictions.insert(0, "track", baselines.TRACK)
            storage.write_predictions(
                engine,
                predictions,
                track=baselines.TRACK,
                model=baselines.MODEL_NAME,
            )
            logger.log_metrics(
                {
                    "train_rows": float(len(train)),
                    "scored_rows": float(len(predictions)),
                    "audited_fit_rows": float(audited),
                }
            )

    _LOGGER.info("community fallbacks: %s", fallbacks.to_dict())
    _LOGGER.info("fit audit: %d rows, all in the training split", audited)


def run_train(
    config: config_module.Config,
    models: pathlib.Path,
    metrics: pathlib.Path,
) -> None:
    """Fit the four CatBoost heads on training rows and save them.

    The rows handed to every fit are audited against the frozen split
    first, so a validation or holdout row cannot reach a model even by
    accident.

    Args:
        config: Validated configuration.
        models: Directory the fitted heads are written to.
        metrics: Destination of the metrics record.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

    table = features.validate_features(
        features.build_features(frame, bookings, config), config
    )
    training = set(
        splits.loc[
            splits[split.SPLIT_COLUMN] == split.TRAIN,
            split.SAMPLE_ID_COLUMN,
        ]
    )
    rows = table.loc[table["sample_id"].isin(training)]
    audited = split.check_fit_membership(rows["sample_id"].tolist(), splits)

    heads = train.fit(rows, frame, config)
    train.save(heads, models)

    provenance = {
        "seed": config.seed,
        "timezone": config.timezone,
        "config_hash": config_module.config_hash(config),
        "bookings_digest": generate.bookings_digest(bookings),
        "samples_digest": samples.samples_digest(frame),
        "features_digest": features.features_digest(table),
    }
    payload = train.build_metrics(heads, config, provenance, audited)
    # A holdout score is spent once and cannot be recomputed, so a
    # re-fit must not quietly drop it. Refitting is otherwise free to
    # replace everything it produced.
    if metrics.is_file():
        previous = json.loads(metrics.read_text(encoding="utf-8"))
        for block in (HOLDOUT_SPLIT, "review"):
            if block in previous:
                payload[block] = previous[block]
    train.write_metrics(payload, metrics)

    for name, head in heads.items():
        with tracking.run(
            config, name=f"{train.MODEL_NAME}_{name}", track=train.TRACK
        ) as logger:
            logger.log_params(
                {"head": name, "config_hash": provenance["config_hash"]}
                | dict(head.params)
            )
            logger.log_metrics(
                {
                    "train_rows": float(head.rows),
                    "trees": float(head.model.tree_count_),
                    "fit_seconds": head.fit_seconds,
                }
            )

    _LOGGER.info(
        "trained %d heads on %d audited rows in %.1fs total",
        len(heads),
        audited,
        sum(head.fit_seconds for head in heads.values()),
    )


def _scored_rows(
    predictions: pd.DataFrame,
    samples_table: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    track: str,
    model: str,
    split_name: str,
) -> pd.DataFrame:
    """Join one model's predictions to the truth of one split.

    Leakage contract: rows are restricted to ``split_name`` before any
    target column is read, so scoring one split cannot touch another's
    labels.

    Args:
        predictions: Every stored prediction row.
        samples_table: The sample table, carrying the targets.
        splits: The frozen split labels.
        track: Which track's rows to score.
        model: Which model's rows to score.
        split_name: The split to restrict to.

    Returns:
        One row per scored sample, carrying targets and predictions.

    Raises:
        SystemExit: Never; an empty result is returned as an empty
            frame and reported by the caller.
    """
    chosen = predictions.loc[
        (predictions["track"] == track) & (predictions["model"] == model)
    ]
    wanted = splits.loc[
        splits[split.SPLIT_COLUMN] == split_name, split.SAMPLE_ID_COLUMN
    ]
    return (
        samples_table.loc[samples_table["sample_id"].isin(set(wanted))]
        .merge(chosen, on="sample_id", how="inner")
        .reset_index(drop=True)
    )


def run_evaluate(
    config: config_module.Config,
    models: pathlib.Path,
    metrics: pathlib.Path,
) -> None:
    """Score the trained heads on validation, against the baseline.

    The holdout stays sealed: only the validation split is scored here,
    and the verdict is recorded as it comes out — a loss is reported as
    a loss.

    Args:
        config: Validated configuration.
        models: Directory the fitted heads were written to.
        metrics: The metrics record, read and rewritten in place.

    Raises:
        RuntimeError: If either model has no rows on the scored split.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)

        table = features.validate_features(
            features.build_features(frame, bookings, config), config
        )
        heads = train.load(models, config)
        calibration = train.load_calibration(models, config)
        predicted = train.predict(heads, table, config, calibration)
        predicted.insert(0, "model", train.MODEL_NAME)
        predicted.insert(0, "track", train.TRACK)
        storage.write_predictions(
            engine, predicted, track=train.TRACK, model=train.MODEL_NAME
        )
        stored = storage.read_table(engine, storage.PREDICTIONS, config)

    # Keyed by sample id, so a probability row cannot drift out of step
    # with the scored row it belongs to when the join reorders.
    probabilities = {
        component: sheet.set_index(table["sample_id"].to_numpy())
        for component, sheet in train.predict_probabilities(
            heads, table, config
        ).items()
    }
    scored = {
        train.MODEL_NAME: (train.TRACK, train.MODEL_NAME),
        baselines.MODEL_NAME: (baselines.TRACK, baselines.MODEL_NAME),
    }
    reports: dict[str, dict[str, Any]] = {}
    for name, (track, model) in scored.items():
        rows = _scored_rows(
            stored,
            frame,
            splits,
            track=track,
            model=model,
            split_name=EVALUATION_SPLIT,
        )
        if rows.empty:
            msg = f"{track}/{model} predicted no {EVALUATION_SPLIT} row"
            raise RuntimeError(msg)
        chosen = (
            {
                component: sheet.loc[rows["sample_id"]].reset_index(drop=True)
                for component, sheet in probabilities.items()
            }
            if name == train.MODEL_NAME
            else None
        )
        reports[name] = evaluate.evaluate_predictions(
            rows,
            config.evaluation.notification_match_ratio,
            config.evaluation.notification_support_minutes,
            chosen,
        )

    catboost_comparison = evaluate.compare_reports(
        reports[train.MODEL_NAME], reports[baselines.MODEL_NAME]
    )
    candidate_order = (baselines.MODEL_NAME, train.MODEL_NAME)
    sources = train.select_head_sources(reports, candidate_order)
    candidate_predictions = {
        name: stored.loc[
            (stored["track"] == track) & (stored["model"] == model),
            ["sample_id", *train.PREDICTION_COLUMNS.values()],
        ]
        .sort_values("sample_id", kind="mergesort")
        .reset_index(drop=True)
        for name, (track, model) in scored.items()
    }
    selected = train.compose_selected_predictions(
        candidate_predictions, sources
    )
    selected.insert(0, "model", train.CHAMPION_MODEL_NAME)
    selected.insert(0, "track", train.TRACK)
    with storage.engine_scope(config) as engine:
        storage.write_predictions(
            engine,
            selected,
            track=train.TRACK,
            model=train.CHAMPION_MODEL_NAME,
        )

    selected_rows = frame.loc[
        frame["sample_id"].isin(
            set(
                splits.loc[
                    splits[split.SPLIT_COLUMN] == EVALUATION_SPLIT,
                    split.SAMPLE_ID_COLUMN,
                ]
            )
        )
    ].merge(selected, on="sample_id", how="inner")
    reports[train.CHAMPION_MODEL_NAME] = evaluate.evaluate_predictions(
        selected_rows,
        config.evaluation.notification_match_ratio,
        config.evaluation.notification_support_minutes,
    )
    comparison = evaluate.compare_reports(
        reports[train.CHAMPION_MODEL_NAME], reports[baselines.MODEL_NAME]
    )
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload[EVALUATION_SPLIT] = evaluate.json_ready(
        {
            "split": EVALUATION_SPLIT,
            "holdout_scored": False,
            "rows": reports[train.MODEL_NAME]["rows"],
            "model": reports[train.MODEL_NAME],
            "baseline": reports[baselines.MODEL_NAME],
            "champion": {
                "model": train.CHAMPION_MODEL_NAME,
                "head_sources": sources,
                "candidate_order": list(candidate_order),
                "selection_split": EVALUATION_SPLIT,
                "report": reports[train.CHAMPION_MODEL_NAME],
            },
            "comparison": comparison,
            "catboost_comparison": catboost_comparison,
        }
    )
    train.write_metrics(payload, metrics)

    with tracking.run(
        config, name=f"{train.MODEL_NAME}_{EVALUATION_SPLIT}", track=train.TRACK
    ) as logger:
        logger.log_params(
            {
                "split": EVALUATION_SPLIT,
                "config_hash": config_module.config_hash(config),
            }
        )
        logger.log_metrics(
            {
                "selection_score": comparison["selection_score"]["model"],
                "catboost_selection_score": catboost_comparison[
                    "selection_score"
                ]["model"],
                "baseline_selection_score": (
                    comparison["selection_score"]["baseline"]
                ),
                "scored_rows": float(reports[train.MODEL_NAME]["rows"]),
            }
        )

    _LOGGER.info(
        "%s scoring on %d rows: selected heads %.4f against baseline %.4f — %s",
        EVALUATION_SPLIT,
        reports[train.MODEL_NAME]["rows"],
        comparison["selection_score"]["model"],
        comparison["selection_score"]["baseline"],
        comparison["selection_score"]["verdict"],
    )
