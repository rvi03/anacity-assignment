"""Builds the training and validation prompt datasets.

One JSONL row per sample: the system instruction, the rendered prompt,
and the answer it teaches. Summaries come from the shared feature table
rather than being recomputed, so a prompt and a tree model see the same
history.

    samples + splits + features ──> prompt rows ──> train.jsonl
                                        │           valid.jsonl
                                        └────────> prompt manifest

The holdout split is never written. Target-free holdout prompts are
built once the prompt, the model, and the decoding are frozen; until
then this module refuses to render a holdout row at all.

Leakage contract: a row's prompt is built from its resident's bookings
at or before the origin and from that sample's shared feature row. The
target is rendered separately and never appears in the prompt.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.data import split as split_module
from facility_prediction.evaluation import evaluate
from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import prompt_features, serialize
from facility_prediction.llm import settings as settings_module

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"
SPLIT_FILES = {
    split_module.TRAIN: TRAIN_FILE,
    split_module.VALIDATION: VALID_FILE,
}

SAMPLE_ID = "sample_id"
RESIDENT = "resident_id"
ORIGIN = "origin"

MINUTES_PER_HOUR = 60.0
HOURS_PER_DAY = 24.0

PERCENTILES = (50, 90, 99)

RUN_NAME = "prompt-dataset"


class DataError(Exception):
    """Raised when a prompt dataset cannot be built as specified."""


def _slug(name: str) -> str:
    """Returns the shared feature table's column form of a name.

    Args:
        name: A facility name as written in configuration.

    Returns:
        Lowercase text with runs of non-alphanumerics as underscores.
    """
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _counts(
    row: pd.Series, prefix: str, keys: tuple[str, ...], labels: tuple[str, ...]
) -> dict[str, int]:
    """Reads a count block out of a shared feature row.

    Args:
        row: One row of the shared feature table.
        prefix: Column prefix the block uses.
        keys: Column suffixes, in render order.
        labels: The label to render each suffix under.

    Returns:
        Label to count, zero counts dropped.

    Raises:
        DataError: If a column the block needs is absent.
    """
    block = {}
    for key, label in zip(keys, labels, strict=True):
        column = f"{prefix}{key}"
        if column not in row.index:
            msg = f"the shared feature table has no column {column!r}"
            raise DataError(msg)
        count = int(row[column])
        if count:
            block[label] = count
    return block


def build_summary(
    row: pd.Series,
    history: pd.DataFrame,
    config: config_module.Config,
    settings: settings_module.Settings,
) -> dict[str, Any]:
    """Assembles the history summary one prompt renders.

    Shared quantities are read from ``row``; only the values the shared
    table has no column for are computed from ``history``.

    Leakage contract: ``history`` is already bounded at the origin by
    the caller, and ``row`` is the sample's own feature row.

    Args:
        row: One row of the shared feature table.
        history: The resident's bookings at or before the origin.
        config: Validated shared configuration.
        settings: The LLM configuration.

    Returns:
        Summary name to value, in render order.

    Raises:
        DataError: If the shared feature table is missing a column.
    """
    facilities = config.facility_names
    facility_counts = _counts(
        row,
        "facility_count_",
        tuple(_slug(name) for name in facilities),
        facilities,
    )
    weekday_counts = _counts(
        row,
        "usage_weekday_count_",
        tuple(str(index) for index in range(len(serialize.WEEKDAY_NAMES))),
        tuple(name[:3] for name in serialize.WEEKDAY_NAMES),
    )
    change = prompt_features.summarise(
        history,
        facilities,
        settings.prompt.change.min_priors,
        settings.prompt.change.recent_depth,
        settings.prompt.change.threshold,
    )
    top = (
        max(facility_counts.items(), key=lambda pair: pair[1])[0]
        if facility_counts
        else None
    )
    return {
        "total_bookings": int(row["n_prior_bookings"]),
        "days_since_last_booking": float(row["days_since_previous_booking"]),
        "facility_counts": facility_counts,
        "usage_day_counts": weekday_counts,
        "most_frequent_facility": top,
        "preferred_usage_hour": change["preferred_usage_hour"],
        "last_facility": str(row["last_1_facility"]),
        "median_interbooking_hours": float(
            row["inter_booking_interval_median_days"] * HOURS_PER_DAY
        ),
        "median_lead_hours": float(
            row["lead_minutes_median"] / MINUTES_PER_HOUR
        ),
        "recent_behaviour_change": change[prompt_features.CHANGED],
        "change_history_insufficient": change[prompt_features.INSUFFICIENT],
    }


def build_rows(
    samples: pd.DataFrame,
    bookings: pd.DataFrame,
    features: pd.DataFrame,
    ladder: list[buckets_module.Bucket],
    config: config_module.Config,
    settings: settings_module.Settings,
) -> list[dict[str, Any]]:
    """Builds one prompt row per sample.

    Leakage contract: each row reads its resident's bookings at or
    before its own origin. The target is rendered into ``target`` only,
    never into ``prompt``.

    Args:
        samples: The samples to render, carrying their targets.
        bookings: The full booking table.
        features: The shared feature table.
        ladder: The frozen delay buckets.
        config: Validated shared configuration.
        settings: The LLM configuration.

    Returns:
        One row per sample, in the input's order.

    Raises:
        DataError: If a sample has no shared feature row.
    """
    indexed = features.set_index(SAMPLE_ID)
    labels = buckets_module.assign(
        samples[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]], ladder
    )
    rows = []
    for (_, sample), label in zip(samples.iterrows(), labels, strict=True):
        sample_id = str(sample[SAMPLE_ID])
        if sample_id not in indexed.index:
            msg = f"sample {sample_id} has no shared feature row"
            raise DataError(msg)
        history = prompt_features.past_bookings(
            bookings, str(sample[RESIDENT]), sample[ORIGIN]
        )
        context = serialize.PromptContext(
            origin=sample[ORIGIN],
            facilities=config.facility_names,
            ladder=ladder,
            summary=build_summary(
                indexed.loc[sample_id], history, config, settings
            ),
            events=prompt_features.recent_events(
                history, settings.prompt.recent_events
            ),
            decimals=settings.prompt.float_decimals,
        )
        rows.append(
            {
                SAMPLE_ID: sample_id,
                "system": serialize.SYSTEM_INSTRUCTION,
                "prompt": serialize.render_prompt(context),
                "target": serialize.render_target(
                    str(sample[evaluate.TARGET_COLUMNS[evaluate.FACILITY]]),
                    int(
                        sample[evaluate.TARGET_COLUMNS[evaluate.USAGE_WEEKDAY]]
                    ),
                    int(sample[evaluate.TARGET_COLUMNS[evaluate.USAGE_HOUR]]),
                    str(label),
                ),
            }
        )
    return rows


def check_sealed(rows: list[dict[str, Any]], splits: pd.DataFrame) -> None:
    """Refuses a dataset that carries a holdout row.

    Args:
        rows: The rows about to be written.
        splits: The frozen split labels.

    Raises:
        DataError: If any row belongs to the holdout split.
    """
    holdout = set(
        splits.loc[
            splits[split_module.SPLIT_COLUMN] == split_module.TEST,
            split_module.SAMPLE_ID_COLUMN,
        ]
    )
    inside = sorted({row[SAMPLE_ID] for row in rows} & holdout)
    if inside:
        msg = (
            f"{len(inside)} holdout row(s) reached a training file, first "
            f"{inside[0]!r}; the holdout stays sealed until scoring"
        )
        raise DataError(msg)


def deduplicate(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Drops rows whose prompt and target both repeat.

    A repeated pattern of behaviour is real signal, so only an exact
    prompt-and-target duplicate is removed.

    Args:
        rows: The rows to filter, in order.

    Returns:
        The kept rows and how many were dropped.
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    for row in rows:
        identity = (row["prompt"], row["target"])
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(row)
    return kept, len(rows) - len(kept)


def write_jsonl(rows: list[dict[str, Any]], path: pathlib.Path) -> str:
    """Writes prompt rows and returns the file's content hash.

    Args:
        rows: The rows to write, in order.
        path: Destination file; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def length_percentiles(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Returns prompt length percentiles, in characters.

    Characters, not tokens: no tokenizer is pinned yet, and reporting a
    token count without one would be a guess.

    Args:
        rows: The rows to measure.

    Returns:
        Percentile name to prompt length.
    """
    lengths = pd.Series([len(row["prompt"]) for row in rows])
    return {
        f"p{value}_characters": float(lengths.quantile(value / 100.0))
        for value in PERCENTILES
    }


def build_manifest(
    written: dict[str, dict[str, Any]],
    ladder: list[buckets_module.Bucket],
    config: config_module.Config,
    settings: settings_module.Settings,
    example_prompt: str,
) -> dict[str, Any]:
    """Assembles the record of what was built and how.

    Args:
        written: Split name to the counts and hash written for it.
        ladder: The frozen delay buckets.
        config: Validated shared configuration.
        settings: The LLM configuration.
        example_prompt: One rendered prompt, standing for the template.

    Returns:
        The manifest payload, ready to serialise.
    """
    schema = serialize.output_schema(
        config.facility_names, [bucket.label for bucket in ladder]
    )
    return {
        "prompt_version": settings.prompt.version,
        "prompt_hash": serialize.template_hash(
            settings.prompt.version, example_prompt
        ),
        "schema_hash": serialize.schema_hash(schema),
        "schema": schema,
        "system_instruction": serialize.SYSTEM_INSTRUCTION,
        "settings": {
            "recent_events": settings.prompt.recent_events,
            "float_decimals": settings.prompt.float_decimals,
            "change": settings.prompt.change.model_dump(),
        },
        "labels": [bucket.label for bucket in ladder],
        "splits": written,
        "community_history_features": "absent",
        "token_counts": "unavailable until a tokenizer is pinned",
    }
