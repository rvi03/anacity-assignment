"""The predicted-versus-actual workbook, for both tracks.

Which split the workbook may render is derived from the seal rather
than hardcoded, so it cannot claim the holdout is sealed after it has
been scored, nor render holdout targets before it has.
"""

from __future__ import annotations

import json
import logging
import pathlib

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction import llm
from facility_prediction.cli.commands.analysis import _matching_slices
from facility_prediction.cli.commands.verification import review_split
from facility_prediction.cli.paths import (
    DEFAULT_COMPARISON_MANIFEST,
    DEFAULT_ERROR_ANALYSIS,
    DEFAULT_SEAL,
    LLM_REVIEW_REASON,
    REVIEW_SPLIT,
)
from facility_prediction.data import generate, samples, split, storage
from facility_prediction.evaluation import review
from facility_prediction.models import train

_LOGGER = logging.getLogger(__name__)


def run_review(
    config: config_module.Config,
    workbook: pathlib.Path,
    csv: pathlib.Path,
    metrics: pathlib.Path | None = None,
) -> None:
    """Render every track's predictions into the shared workbook.

    Leakage contract: rows are restricted to the split :func:`review_split`
    allows before any target is read. That is the validation split until
    the holdout has been scored once, and the holdout afterwards.

    Args:
        config: Validated configuration.
        workbook: Destination ``.xlsx``.
        csv: Destination CSV carrying the same rows.
        metrics: Where to record the rendered hash. None records it
            nowhere, which is what a test rendering to a scratch
            directory wants — writing it to the shipped record would
            stamp a smoke workbook's hash onto the real one.
    """
    rendered_split, reason = review_split(DEFAULT_SEAL)
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)
        predictions = storage.read_table(engine, storage.PREDICTIONS, config)

    rendered = splits.loc[
        splits[split.SPLIT_COLUMN] == rendered_split, split.SAMPLE_ID_COLUMN
    ]
    scored = frame.loc[frame[split.SAMPLE_ID_COLUMN].isin(set(rendered))]
    joined = predictions.merge(scored, on=split.SAMPLE_ID_COLUMN, how="inner")

    manifest = split.load_manifest(DEFAULT_COMPARISON_MANIFEST)
    sheet = review.build_predictions_sheet(
        joined,
        review.history_strings(scored, bookings, config),
        set(manifest["sample_ids"]),
        config,
    )
    coverage = review.Coverage(
        split=rendered_split, rows=len(sheet), reason=reason
    )
    # The slices are shown only when they describe the split being
    # rendered; a validation slice table under a holdout workbook would
    # read as holdout error analysis.
    slices = _matching_slices(DEFAULT_ERROR_ANALYSIS, rendered_split)
    digest = review.write_csv(sheet, csv)
    review.write_workbook(
        {
            review.PREDICTIONS_SHEET: sheet,
            review.SUMMARY_SHEET: review.build_summary_sheet(
                joined,
                coverage,
                {
                    "Seed": config.seed,
                    "Bookings digest": generate.bookings_digest(bookings),
                    "Samples digest": samples.samples_digest(frame),
                    "Review CSV sha256": digest,
                },
                review.track_states(predictions),
                config,
                slices,
            ),
            review.DEFINITIONS_SHEET: review.build_definitions_sheet(config),
        },
        workbook,
    )
    # Recorded beside the metrics so `verify` has something to compare
    # the workbook against rather than trusting that it was rebuilt.
    if metrics is not None and metrics.is_file():
        payload = json.loads(metrics.read_text(encoding="utf-8"))
        payload["review"] = {
            "split": rendered_split,
            "rows": len(sheet),
            "csv_sha256": digest,
        }
        train.write_metrics(payload, metrics)

    _LOGGER.info(
        "wrote %s and %s: %d rows over the %s split (sha256=%s)",
        workbook,
        csv,
        len(sheet),
        rendered_split,
        digest,
    )


def run_llm_review(
    config: config_module.Config,
    workbook: pathlib.Path,
    csv: pathlib.Path,
) -> None:
    """Render the shared workbook with this track's holdout rows added.

    Every track's development rows are rendered exactly as the shared
    renderer builds them, and this track's scored holdout rows are
    appended. No other track's rows are dropped or rewritten.

    Leakage contract: holdout rows are rendered only for tracks that
    have already taken their single scoring pass, which for this track
    is the pass that wrote its ``track=llm`` prediction rows.

    Args:
        config: Validated configuration.
        workbook: Destination ``.xlsx``.
        csv: Destination CSV carrying the same rows.
    """
    with storage.engine_scope(config) as engine:
        bookings = storage.read_table(engine, storage.BOOKINGS, config)
        frame = storage.read_table(engine, storage.SAMPLES, config)
        splits = storage.read_table(engine, storage.SPLITS, config)
        predictions = storage.read_table(engine, storage.PREDICTIONS, config)

    scored_ids = set(
        splits.loc[
            splits[split.SPLIT_COLUMN] == REVIEW_SPLIT, split.SAMPLE_ID_COLUMN
        ]
    )
    holdout_ids = set(
        splits.loc[
            splits[split.SPLIT_COLUMN] == split.TEST, split.SAMPLE_ID_COLUMN
        ]
    )
    development = predictions.merge(
        frame.loc[frame[split.SAMPLE_ID_COLUMN].isin(scored_ids)],
        on=split.SAMPLE_ID_COLUMN,
        how="inner",
    )
    llm_holdout = predictions.loc[predictions["track"] == llm.TRACK].merge(
        frame.loc[frame[split.SAMPLE_ID_COLUMN].isin(holdout_ids)],
        on=split.SAMPLE_ID_COLUMN,
        how="inner",
    )
    joined = pd.concat([development, llm_holdout], ignore_index=True)

    manifest = split.load_manifest(DEFAULT_COMPARISON_MANIFEST)
    sheet = review.build_predictions_sheet(
        joined,
        review.history_strings(
            frame.loc[
                frame[split.SAMPLE_ID_COLUMN].isin(scored_ids | holdout_ids)
            ],
            bookings,
            config,
        ),
        set(manifest["sample_ids"]),
        config,
    )
    coverage = review.Coverage(
        split=f"{REVIEW_SPLIT} + {split.TEST} ({llm.TRACK})",
        rows=len(sheet),
        reason=LLM_REVIEW_REASON,
    )
    digest = review.write_csv(sheet, csv)
    review.write_workbook(
        {
            review.PREDICTIONS_SHEET: sheet,
            review.SUMMARY_SHEET: review.build_summary_sheet(
                joined,
                coverage,
                {
                    "Seed": config.seed,
                    "Bookings digest": generate.bookings_digest(bookings),
                    "Samples digest": samples.samples_digest(frame),
                    "Review CSV sha256": digest,
                },
                review.track_states(predictions),
                config,
            ),
            review.DEFINITIONS_SHEET: review.build_definitions_sheet(config),
        },
        workbook,
    )
    _LOGGER.info(
        "wrote %s and %s: %d rows, of which %d are %s holdout rows (sha256=%s)",
        workbook,
        csv,
        len(sheet),
        len(llm_holdout),
        llm.TRACK,
        digest,
    )
