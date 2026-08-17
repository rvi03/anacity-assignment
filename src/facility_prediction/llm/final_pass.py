"""Scores the frozen adapter once, on the sealed holdout rows.

This is the only place the LLM branch reads a holdout target, and it
reads none of them until every answer is written down and hashed. The
ordering is enforced rather than described: :func:`score` refuses to
run against a set of answers whose file no longer matches the hash
frozen before the targets were opened.

    holdout ids ──> target-free prompts ──> answers ──> hash frozen
                                                             │
                                             targets joined <┘
                                                    │
                                            one set of metrics

Leakage contract: prompts are rendered from bookings at or before each
row's own origin and carry no target. Targets enter only after the
freeze, and no answer is changed by what a target turned out to be.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.data import split as split_module
from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import gate as gate_module
from facility_prediction.llm import (
    llm_data,
    llm_predict,
    prompt_features,
    serialize,
)
from facility_prediction.llm import settings as settings_module

GATE = "the prediction hash is frozen before any holdout target is read"
RUN_NAME = "final-pass"

TIER_B = "tier_b"

# The only comparator this branch may claim under the spine-first
# resequence: the shared frequency/recency baseline. Not CatBoost.
COMPARATOR = "frequency_recency"

EVIDENCE = "measured"

SAMPLE_ID = llm_data.SAMPLE_ID


class FinalPassError(Exception):
    """Raised when the single scored pass cannot be run as declared."""


@dataclasses.dataclass(frozen=True)
class Freeze:
    """The answers, sealed before any target was opened.

    Attributes:
        path: Where the answers were written.
        sha256: Hex digest of that file, taken before the join.
        rows: How many answers it holds.
    """

    path: pathlib.Path
    sha256: str
    rows: int

    def to_dict(self) -> dict[str, Any]:
        """Returns the freeze as a serialisable mapping.

        Returns:
            The file, its digest, and its row count.
        """
        return {
            "predictions_file": str(self.path),
            "predictions_sha256": self.sha256,
            "rows": self.rows,
        }


def prompt_rows(
    samples: pd.DataFrame,
    bookings: pd.DataFrame,
    features: pd.DataFrame,
    ladder: Sequence[buckets_module.Bucket],
    config: config_module.Config,
    settings: settings_module.Settings,
) -> list[dict[str, Any]]:
    """Renders one target-free prompt per holdout row.

    Leakage contract: each row reads its resident's bookings at or
    before its own origin. No target column is read here at all, which
    is what makes these prompts safe to build before the freeze.

    Args:
        samples: The holdout rows to render.
        bookings: The full booking table.
        features: The shared feature table.
        ladder: The frozen delay buckets.
        config: Validated shared configuration.
        settings: The LLM configuration.

    Returns:
        One prompt row per sample, in the input's order.

    Raises:
        FinalPassError: If a sample has no shared feature row.
    """
    indexed = features.set_index(SAMPLE_ID)
    rows = []
    for _, sample in samples.iterrows():
        sample_id = str(sample[SAMPLE_ID])
        if sample_id not in indexed.index:
            msg = f"sample {sample_id} has no shared feature row"
            raise FinalPassError(msg)
        history = prompt_features.past_bookings(
            bookings, str(sample[llm_data.RESIDENT]), sample[llm_data.ORIGIN]
        )
        context = serialize.PromptContext(
            origin=sample[llm_data.ORIGIN],
            facilities=config.facility_names,
            ladder=list(ladder),
            summary=llm_data.build_summary(
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
            }
        )
    return rows


def check_target_free(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuses a prompt row that carries a target of any kind.

    Args:
        rows: The rendered prompt rows.

    Raises:
        FinalPassError: If any row carries a target field.
    """
    carriers = sorted({str(row[SAMPLE_ID]) for row in rows if "target" in row})
    if carriers:
        msg = (
            f"{len(carriers)} holdout prompt row(s) carry a target, first "
            f"{carriers[0]!r}; the scored pass must be target-free"
        )
        raise FinalPassError(msg)


def check_manifest(
    rows: Sequence[Mapping[str, Any]],
    manifest_ids: Sequence[str],
    splits: pd.DataFrame,
) -> None:
    """Refuses a pass that scores rows the manifest did not seal.

    Args:
        rows: The rendered prompt rows.
        manifest_ids: The frozen comparison manifest's identifiers.
        splits: The frozen split labels.

    Raises:
        FinalPassError: If the rows are not exactly the manifest's, or
            any of them is not a holdout row.
    """
    scored = {str(row[SAMPLE_ID]) for row in rows}
    sealed = {str(sample_id) for sample_id in manifest_ids}
    if scored != sealed:
        msg = (
            f"the pass covers {len(scored)} rows and the frozen manifest "
            f"seals {len(sealed)}; they must be the same rows"
        )
        raise FinalPassError(msg)
    holdout = set(
        splits.loc[
            splits[split_module.SPLIT_COLUMN] == split_module.TEST,
            split_module.SAMPLE_ID_COLUMN,
        ].astype(str)
    )
    stray = sorted(scored - holdout)
    if stray:
        msg = (
            f"{len(stray)} scored row(s) are not holdout rows, first "
            f"{stray[0]!r}"
        )
        raise FinalPassError(msg)


def freeze_predictions(
    records: Sequence[Mapping[str, Any]], path: pathlib.Path
) -> Freeze:
    """Writes the answers and seals them before any target is read.

    Args:
        records: The answers, in manifest order.
        path: Destination file.

    Returns:
        The freeze: the file, its digest, and its row count.
    """
    llm_predict.write_records(records, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return Freeze(path=path, sha256=digest, rows=len(records))


def check_freeze(freeze: Freeze) -> None:
    """Refuses to score answers that changed after they were sealed.

    Args:
        freeze: The freeze taken before the targets were opened.

    Raises:
        FinalPassError: If the file no longer matches its digest.
    """
    if not freeze.path.is_file():
        msg = f"the frozen predictions at {freeze.path} are gone"
        raise FinalPassError(msg)
    digest = hashlib.sha256(freeze.path.read_bytes()).hexdigest()
    if digest != freeze.sha256:
        msg = (
            f"the predictions at {freeze.path} hash to {digest[:12]}, but "
            f"were frozen at {freeze.sha256[:12]}; they changed after the "
            "freeze and cannot be scored"
        )
        raise FinalPassError(msg)


def score(
    freeze: Freeze,
    predictions: pd.DataFrame,
    targets: pd.DataFrame,
    labels: pd.Series,
    config: config_module.Config,
    ceiling: float,
) -> dict[str, Any]:
    """Marks the frozen answers against the holdout targets.

    Leakage contract: this is the first and only read of a holdout
    target in this branch, and it happens only after ``freeze`` has
    been checked against the file on disk.

    Args:
        freeze: The freeze taken before the targets were opened.
        predictions: Predicted columns indexed by sample identifier.
        targets: Target columns indexed by sample identifier.
        labels: The real notification label per row.
        config: Validated shared configuration.
        ceiling: The representation ceiling the labels can reach.

    Returns:
        The pass's rates, its confusion, and its per-label support.

    Raises:
        FinalPassError: If the answers changed after the freeze.
    """
    check_freeze(freeze)
    marked = gate_module.arm_metrics(
        predictions, targets, labels, config, ceiling
    )
    return {
        "rates": marked["rates"],
        "confusion": marked["confusion"],
        "support": marked["support"],
    }


def build_report(
    freeze: Freeze,
    marked: Mapping[str, Any],
    operational: Mapping[str, float],
    context: Mapping[str, Any],
    comparator: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Assembles the single scored pass's report.

    Args:
        freeze: The freeze the scoring ran under.
        marked: What :func:`score` returned.
        operational: What the pass cost and how often it failed.
        context: The rows, split, manifest, and adapter it ran on.
        comparator: The shared baseline's rates on the same rows, or
            None when no comparator was scored.

    Returns:
        The report, ready to write.
    """
    report = {
        "evidence": EVIDENCE,
        "gate": GATE,
        "freeze": freeze.to_dict(),
        "context": dict(context),
        "rates": marked["rates"],
        "operational": dict(operational),
        "confusion": marked["confusion"],
        "support": marked["support"],
    }
    if comparator is not None:
        report["comparator"] = {
            "model": COMPARATOR,
            "rates": dict(comparator),
            "delta_overall": float(marked["rates"]["overall"])
            - float(comparator["overall"]),
        }
    return report


def run_params(report: Mapping[str, Any]) -> dict[str, str]:
    """Returns the run parameters for the scored pass.

    Args:
        report: The assembled report.

    Returns:
        Parameter name to value, as text.
    """
    params = {
        f"context_{key}": str(value) for key, value in report["context"].items()
    }
    params["predictions_sha256"] = str(report["freeze"]["predictions_sha256"])
    return params


def run_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Returns the run metrics for the scored pass.

    Args:
        report: The assembled report.

    Returns:
        Metric name to value.
    """
    metrics = {str(key): float(value) for key, value in report["rates"].items()}
    metrics.update(
        {
            f"operational_{key}": float(value)
            for key, value in report["operational"].items()
        }
    )
    return metrics
