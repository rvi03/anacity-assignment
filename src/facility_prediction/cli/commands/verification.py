"""Freezing, scoring the holdout once, and verifying the result.

The holdout is scored once, ever. Two records enforce it: a freeze
written before any holdout row is read, and a seal written after.
Scoring refuses if the configuration moved after the freeze, and
refuses outright if the seal already exists.
"""

from __future__ import annotations

import json
import logging
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction import tracking
from facility_prediction.cli.commands.pipeline import _scored_rows
from facility_prediction.cli.paths import (
    DEFAULT_SEAL,
    EVALUATION_SPLIT,
    HOLDOUT_SPLIT,
    SCORED_REASON,
    SEALED_REASON,
)
from facility_prediction.data import split, storage
from facility_prediction.evaluation import evaluate
from facility_prediction.evaluation import freeze as freeze_module
from facility_prediction.evaluation import verify as verify_module
from facility_prediction.features import features
from facility_prediction.models import baselines, train

_LOGGER = logging.getLogger(__name__)


def review_split(seal: pathlib.Path) -> tuple[str, str]:
    """Return the split the workbook may render, and why.

    Derived from the seal rather than hardcoded, so the workbook cannot
    claim the holdout is sealed after it has been scored, nor render
    holdout targets before it has.

    Args:
        seal: The holdout seal path.

    Returns:
        The split name and the sentence the Summary states.
    """
    if freeze_module.is_sealed(seal):
        return HOLDOUT_SPLIT, SCORED_REASON
    return split.VALIDATION, SEALED_REASON


def _logged_headline(
    config: config_module.Config, run_name: str
) -> dict[str, float] | None:
    """Read one run's metrics back from the tracker.

    Args:
        config: Validated configuration.
        run_name: The run to look for.

    Returns:
        Its metrics, or None when the tracker is unreachable or the run
        is absent. An absent tracker is not a wrong number.
    """
    if not tracking.is_reachable(config):
        return None
    try:
        import mlflow  # noqa: PLC0415

        tracking.configure(config)
        found = mlflow.search_runs(
            experiment_names=[config.tracking.experiment],
            filter_string=f"attributes.run_name = '{run_name}'",
            max_results=1,
            output_format="pandas",
        )
        table = pd.DataFrame(found)
        if table.empty:
            return None
        return {
            str(name).removeprefix("metrics."): float(value)
            for name, value in table.iloc[0].items()
            if str(name).startswith("metrics.") and pd.notna(value)
        }
    except Exception:
        _LOGGER.info("could not read tracking metrics; parity not checked")
        return None


def run_verify(
    config: config_module.Config,
    metrics: pathlib.Path,
    csv: pathlib.Path,
) -> None:
    """Recompute every committed value and report what disagrees.

    Leakage contract: re-scores splits that were already scored, from
    prediction rows that already exist. It creates no prediction and
    fits nothing, so it cannot spend a scoring of the holdout.

    Args:
        config: Validated configuration.
        metrics: The committed metrics record.
        csv: The reviewer's CSV, rehashed.

    Raises:
        SystemExit: If any committed value fails to reproduce.
    """
    try:
        payload = verify_module.load_metrics(metrics)
    except verify_module.VerificationError as error:
        _LOGGER.error("%s", error)
        raise SystemExit(1) from error

    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)
        stored = storage.read_table(engine, storage.PREDICTIONS, config)

    table = features.build_features(frame, bookings, config)
    problems = [
        *verify_module.check_config_hash(payload, config),
        *verify_module.check_digests(payload, bookings, frame, table),
    ]

    sources = {
        "champion": (train.TRACK, train.CHAMPION_MODEL_NAME),
        "model": (train.TRACK, train.MODEL_NAME),
        "baseline": (baselines.TRACK, baselines.MODEL_NAME),
    }
    for split_name in (EVALUATION_SPLIT, HOLDOUT_SPLIT):
        if split_name not in payload:
            continue
        try:
            reports = verify_module.rescore(
                config, stored, frame, splits, split_name, sources
            )
        except verify_module.VerificationError as error:
            problems.append(str(error))
            continue
        problems += verify_module.check_scored_metrics(
            payload, split_name, evaluate.json_ready(reports)
        )

    problems += verify_module.check_workbook(
        csv, payload.get("review", {}).get("csv_sha256")
    )

    if HOLDOUT_SPLIT in payload:
        holdout = payload[HOLDOUT_SPLIT]
        problems += verify_module.check_tracking_parity(
            _logged_headline(
                config, f"{train.CHAMPION_MODEL_NAME}_{HOLDOUT_SPLIT}"
            ),
            {
                "selection_score": holdout["comparison"]["selection_score"][
                    "model"
                ],
                "baseline_selection_score": holdout["comparison"][
                    "selection_score"
                ]["baseline"],
            },
        )

    _LOGGER.info("%s", verify_module.report(problems))
    if problems:
        raise SystemExit(1)


def run_freeze(
    config: config_module.Config,
    metrics: pathlib.Path,
    record: pathlib.Path,
) -> None:
    """Lock the configuration and the chosen sources before scoring.

    Leakage contract: reads configuration, the validation metrics, and
    the input digests. No holdout row is touched.

    Args:
        config: Validated configuration.
        metrics: The metrics record, for the validation-chosen sources.
        record: Destination JSON for the freeze.

    Raises:
        SystemExit: If validation has not chosen its sources yet, or the
            holdout was already scored.
    """
    freeze_module.require_unsealed(DEFAULT_SEAL)
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    validation = payload.get(EVALUATION_SPLIT, {})
    sources = validation.get("champion", {}).get("head_sources")
    if not sources:
        _LOGGER.error(
            "no validation-chosen source map in %s; run evaluate first",
            metrics,
        )
        raise SystemExit(1)

    frozen = freeze_module.build_freeze(
        config, sources, payload.get("provenance", {})
    )
    freeze_module.write_freeze(frozen, record)
    _LOGGER.info(
        "froze config %s; %s",
        frozen["config_hash"],
        ", ".join(f"{key}={value}" for key, value in sources.items()),
    )


def run_score_holdout(
    config: config_module.Config,
    models: pathlib.Path,
    metrics: pathlib.Path,
    record: pathlib.Path,
    seal: pathlib.Path,
) -> None:
    """Score the sealed holdout, once, against the frozen settings.

    This is the only function in the traditional track permitted to read
    holdout targets. It refuses to run twice, and it refuses to run at
    all if the configuration moved after the freeze.

    Args:
        config: Validated configuration.
        models: Directory the fitted heads were written to.
        metrics: The metrics record, read and rewritten in place.
        record: The freeze JSON written by :func:`run_freeze`.
        seal: Destination JSON recording that the holdout is now spent.

    Raises:
        SystemExit: If the freeze is absent, stale, or already spent, or
            if a model predicted no holdout row.
    """
    try:
        frozen = freeze_module.read_freeze(record)
        freeze_module.require_unsealed(seal)
        freeze_module.check_freeze_matches(frozen, config)
    except freeze_module.FreezeError as error:
        _LOGGER.error("%s", error)
        raise SystemExit(1) from error

    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)
        stored = storage.read_table(engine, storage.PREDICTIONS, config)

    del bookings
    scored = {
        train.CHAMPION_MODEL_NAME: (train.TRACK, train.CHAMPION_MODEL_NAME),
        train.MODEL_NAME: (train.TRACK, train.MODEL_NAME),
        baselines.MODEL_NAME: (baselines.TRACK, baselines.MODEL_NAME),
    }
    del models
    reports: dict[str, dict[str, Any]] = {}
    for name, (track, model) in scored.items():
        rows = _scored_rows(
            stored,
            frame,
            splits,
            track=track,
            model=model,
            split_name=HOLDOUT_SPLIT,
        )
        if rows.empty:
            _LOGGER.error(
                "%s/%s predicted no %s row", track, model, HOLDOUT_SPLIT
            )
            raise SystemExit(1)
        reports[name] = evaluate.evaluate_predictions(
            rows,
            config.evaluation.notification_match_ratio,
            config.evaluation.notification_support_minutes,
        )

    comparison = evaluate.compare_reports(
        reports[train.CHAMPION_MODEL_NAME], reports[baselines.MODEL_NAME]
    )
    catboost_comparison = evaluate.compare_reports(
        reports[train.MODEL_NAME], reports[baselines.MODEL_NAME]
    )
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    payload[HOLDOUT_SPLIT] = evaluate.json_ready(
        {
            "split": HOLDOUT_SPLIT,
            "holdout_scored": True,
            "scored_once": True,
            "frozen_config_hash": frozen["config_hash"],
            "head_sources": frozen["head_sources"],
            "rows": reports[train.CHAMPION_MODEL_NAME]["rows"],
            "champion": reports[train.CHAMPION_MODEL_NAME],
            "model": reports[train.MODEL_NAME],
            "baseline": reports[baselines.MODEL_NAME],
            "comparison": comparison,
            "catboost_comparison": catboost_comparison,
        }
    )
    train.write_metrics(payload, metrics)

    headline = {
        "selection_score": comparison["selection_score"]["model"],
        "baseline_selection_score": comparison["selection_score"]["baseline"],
        "catboost_selection_score": (
            catboost_comparison["selection_score"]["model"]
        ),
    }
    freeze_module.write_seal(
        freeze_module.build_seal(
            frozen, reports[train.CHAMPION_MODEL_NAME]["rows"], headline
        ),
        seal,
    )

    with tracking.run(
        config,
        name=f"{train.CHAMPION_MODEL_NAME}_{HOLDOUT_SPLIT}",
        track=train.TRACK,
    ) as logger:
        logger.log_params(
            {
                "split": HOLDOUT_SPLIT,
                "config_hash": frozen["config_hash"],
                "scored_once": True,
            }
        )
        logger.log_metrics(
            {
                **headline,
                "scored_rows": float(
                    reports[train.CHAMPION_MODEL_NAME]["rows"]
                ),
            }
        )

    _LOGGER.info(
        "%s scored ONCE on %d rows: selected heads %.4f against baseline "
        "%.4f — %s. CatBoost alone %.4f. Sealed at %s",
        HOLDOUT_SPLIT,
        reports[train.CHAMPION_MODEL_NAME]["rows"],
        headline["selection_score"],
        headline["baseline_selection_score"],
        comparison["selection_score"]["verdict"],
        headline["catboost_selection_score"],
        seal,
    )
