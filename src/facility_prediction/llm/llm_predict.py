"""Runs a frozen set of prompts through the model and keeps every row.

One record per prompt per arm, written as it was produced: the four
fields when the answer parsed, and the stage and reason when it did
not. A row that failed is kept and counted, never dropped and never
replaced by a guess — a missing row would quietly improve the score.

    manifest ──> prompts ──> decode ──> record per row ──> JSONL
                                             │
                              valid fields  or  typed failure

The written file is the unit of replay: every number this branch
reports can be recomputed from it without loading a model again.

Leakage contract: the prompts are rendered target-free by the dataset
step, and this module never reads a target. Targets are joined to these
records only when they are scored.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import json
import pathlib
from typing import Any

import pandas as pd

from facility_prediction.evaluation import evaluate
from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import constrained_decode, serialize

SAMPLE_ID = "sample_id"
ARM = "arm"

ZERO_SHOT = "zero_shot"
PILOT_ADAPTER = "pilot_adapter"

BUCKET_COLUMN = "predicted_notification_bucket"
VALID_COLUMN = "valid"

RUN_NAME = "gate-pass"


class PredictionError(Exception):
    """Raised when a prediction pass cannot be run or replayed."""


def run_arm(
    prompts: Sequence[Mapping[str, Any]],
    arm: str,
    decode: Callable[[Mapping[str, Any]], constrained_decode.Outcome],
    identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Answers every prompt once and records what came back.

    Args:
        prompts: The rendered prompts, in manifest order.
        arm: Which pass these answers belong to.
        decode: Produces one outcome for one prompt.
        identity: What makes this pass reproducible.

    Returns:
        One record per prompt, in the same order.
    """
    records = []
    for prompt in prompts:
        outcome = decode(prompt)
        records.append(
            {
                SAMPLE_ID: str(prompt[SAMPLE_ID]),
                ARM: arm,
                "status": outcome.status,
                "fields": outcome.fields,
                "attempts": outcome.attempts,
                "failure_stage": outcome.failure_stage,
                "failure_reason": outcome.failure_reason,
                "semantic_invalid_reason": outcome.semantic_invalid_reason,
                "seconds": outcome.completion.seconds,
                "prompt_tokens": outcome.completion.prompt_tokens,
                "completion_tokens": outcome.completion.completion_tokens,
                "identity_hash": str(identity["identity_hash"]),
            }
        )
    return records


def write_records(
    records: Sequence[Mapping[str, Any]], path: pathlib.Path
) -> None:
    """Writes prediction records, one JSON object per line.

    Args:
        records: The records to write, in order.
        path: Destination file; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_records(path: pathlib.Path) -> list[dict[str, Any]]:
    """Reads prediction records back for replay.

    Args:
        path: The file to read.

    Returns:
        One mapping per line, in file order.

    Raises:
        PredictionError: If the file does not exist.
    """
    if not path.is_file():
        msg = f"cannot replay: {path} does not exist"
        raise PredictionError(msg)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def arm_records(
    records: Sequence[Mapping[str, Any]], arm: str
) -> list[dict[str, Any]]:
    """Returns one arm's records.

    Args:
        records: Every record read back.
        arm: The arm to keep.

    Returns:
        That arm's records, in file order.

    Raises:
        PredictionError: If the arm has no records.
    """
    kept = [dict(record) for record in records if record[ARM] == arm]
    if not kept:
        msg = f"no records for the {arm!r} pass"
        raise PredictionError(msg)
    return kept


def to_frame(
    records: Sequence[Mapping[str, Any]],
    ladder: Sequence[buckets_module.Bucket],
) -> pd.DataFrame:
    """Turns records into predicted columns the shared metrics read.

    A row that never parsed keeps its place with no prediction, so it
    stays in every denominator.

    Args:
        records: One arm's records, in manifest order.
        ladder: The frozen delay buckets, carrying representatives.

    Returns:
        One row per record, indexed by sample identifier, carrying the
        predicted columns, the predicted bucket, and whether the answer
        was usable.

    Raises:
        PredictionError: If an answer names a bucket the ladder has no
            representative for.
    """
    minutes = {
        bucket.label: bucket.representative
        for bucket in ladder
        if bucket.representative is not None
    }
    weekdays = {
        name: index for index, name in enumerate(serialize.WEEKDAY_NAMES)
    }

    rows = []
    for record in records:
        fields = record["fields"]
        row: dict[str, Any] = {
            SAMPLE_ID: str(record[SAMPLE_ID]),
            VALID_COLUMN: fields is not None,
            BUCKET_COLUMN: None,
            evaluate.PREDICTED_COLUMNS[evaluate.FACILITY]: None,
            evaluate.PREDICTED_COLUMNS[evaluate.USAGE_WEEKDAY]: None,
            evaluate.PREDICTED_COLUMNS[evaluate.USAGE_HOUR]: None,
            evaluate.PREDICTED_COLUMNS[evaluate.NOTIFICATION]: None,
        }
        if fields is not None:
            label = str(fields[serialize.BUCKET_FIELD])
            if label not in minutes:
                msg = f"the ladder has no representative for {label!r}"
                raise PredictionError(msg)
            row[BUCKET_COLUMN] = label
            row[evaluate.PREDICTED_COLUMNS[evaluate.FACILITY]] = str(
                fields[serialize.FACILITY_FIELD]
            )
            row[evaluate.PREDICTED_COLUMNS[evaluate.USAGE_WEEKDAY]] = weekdays[
                str(fields[serialize.USAGE_DAY_FIELD])
            ]
            row[evaluate.PREDICTED_COLUMNS[evaluate.USAGE_HOUR]] = int(
                fields[serialize.USAGE_HOUR_FIELD]
            )
            row[evaluate.PREDICTED_COLUMNS[evaluate.NOTIFICATION]] = minutes[
                label
            ]
        rows.append(row)
    return pd.DataFrame(rows).set_index(SAMPLE_ID)


def operational_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Summarises what the pass cost and how often it went wrong.

    Args:
        records: One arm's records.

    Returns:
        Measure name to value: row counts, validity and retry rates,
        latency, and token counts.
    """
    frame = pd.DataFrame(list(records))
    seconds = frame["seconds"]
    return {
        "rows": float(len(frame)),
        "valid": float((frame["status"] == constrained_decode.VALID).sum()),
        "failed": float((frame["status"] == constrained_decode.FAILED).sum()),
        "retried": float((frame["attempts"] > 1).sum()),
        "semantic_invalid": float(
            frame["semantic_invalid_reason"].notna().sum()
        ),
        "latency_p50_seconds": float(seconds.quantile(0.5)),
        "latency_p95_seconds": float(seconds.quantile(0.95)),
        "elapsed_seconds": float(seconds.sum()),
        "prompt_tokens_mean": float(frame["prompt_tokens"].mean()),
        "completion_tokens_mean": float(frame["completion_tokens"].mean()),
    }
