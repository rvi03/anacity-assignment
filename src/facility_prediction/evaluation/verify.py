"""Does the committed result still describe what the code produces?

`metrics.json` is the claim. This module is what makes the claim
falsifiable: it recomputes the quantities from the store of record and
from the saved predictions, and compares. Anything that disagrees is
reported as a difference, not repaired.

What is compared, and why each one:

    input digests     the dataset the numbers were measured on. A
                      digest over rows in a declared order, not a file
                      hash, because Postgres never promised stable
                      page bytes.
    scored metrics    every headline number, recomputed from the
                      prediction rows rather than copied. If scoring
                      changed, this is where it shows.
    workbook          the reviewer's CSV, rehashed. A workbook whose
                      bytes moved while its rows did not is the failure
                      that shipped once already.
    tracking parity   the experiment tracker's headline metrics against
                      the committed ones. They are logged by different
                      code paths, so agreement is evidence rather than
                      tautology.

The `mlflow` database is excluded from every digest: run ids, timestamps
and trace ids are nondeterministic by construction, and dropping that
database must still leave every deliverable reproducible. Only its
headline metric *values* are compared, and only when it is reachable.

Leakage contract: reads stored rows and committed artifacts. Nothing is
fitted and no split is scored that was not already scored, so verifying
cannot spend the holdout — it re-reads a number that was already
written, and would refuse to produce a new one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import logging
import pathlib
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples
from facility_prediction.evaluation import evaluate
from facility_prediction.features import features

_LOGGER = logging.getLogger(__name__)

# How close a recomputed float must be to the committed one. Scoring is
# deterministic, so this is a float-representation tolerance and not a
# licence for the number to drift.
_TOLERANCE = 1e-9


class VerificationError(Exception):
    """Raised when verification cannot be attempted at all."""


def _difference(what: str, committed: Any, recomputed: Any) -> str:
    """Return a readable description of one disagreement.

    Args:
        what: The quantity that disagreed.
        committed: The value on record.
        recomputed: The value produced now.

    Returns:
        One line naming both values.
    """
    return f"{what}: committed {committed!r}, recomputed {recomputed!r}"


def check_digests(
    payload: Mapping[str, Any],
    bookings: pd.DataFrame,
    samples_table: pd.DataFrame,
    feature_table: pd.DataFrame,
) -> list[str]:
    """Recompute the input digests and compare them to the record.

    Args:
        payload: The committed metrics record.
        bookings: The booking table, read back from the store.
        samples_table: The sample table.
        feature_table: The feature table, rebuilt.

    Returns:
        One message per disagreement.
    """
    recomputed = {
        "bookings_digest": generate.bookings_digest(bookings),
        "samples_digest": samples.samples_digest(samples_table),
        "features_digest": features.features_digest(feature_table),
    }
    provenance = payload.get("provenance", {})
    return [
        _difference(name, provenance.get(name), value)
        for name, value in recomputed.items()
        if provenance.get(name) != value
    ]


def check_config_hash(
    payload: Mapping[str, Any], config: config_module.Config
) -> list[str]:
    """Compare the resolved configuration to the committed hash.

    Args:
        payload: The committed metrics record.
        config: The configuration this run resolved.

    Returns:
        One message if the hash moved, otherwise nothing.
    """
    current = config_module.config_hash(config)
    committed = payload.get("provenance", {}).get("config_hash")
    if committed == current:
        return []
    return [_difference("config_hash", committed, current)]


def _flatten(prefix: str, value: Any) -> dict[str, float]:
    """Return every numeric leaf of a nested mapping, dotted-key style.

    Args:
        prefix: Key path so far.
        value: The value to walk.

    Returns:
        Flat key to float.
    """
    if isinstance(value, Mapping):
        flat: dict[str, float] = {}
        for key, item in value.items():
            flat.update(
                _flatten(f"{prefix}.{key}" if prefix else str(key), item)
            )
        return flat
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {}
    return {prefix: float(value)}


def check_scored_metrics(
    payload: Mapping[str, Any],
    split_name: str,
    recomputed: Mapping[str, Any],
) -> list[str]:
    """Compare one split's committed report to a freshly scored one.

    Args:
        payload: The committed metrics record.
        split_name: Which split's block to compare.
        recomputed: The freshly computed block for that split.

    Returns:
        One message per numeric disagreement.
    """
    committed = payload.get(split_name)
    if committed is None:
        return [f"{split_name}: no committed block to compare against"]

    left = _flatten("", committed)
    right = _flatten("", recomputed)
    problems = []
    for key, value in right.items():
        if key not in left:
            continue
        if abs(left[key] - value) > _TOLERANCE:
            problems.append(
                _difference(f"{split_name}.{key}", left[key], value)
            )
    return problems


def check_workbook(csv: pathlib.Path, committed: str | None) -> list[str]:
    """Rehash the reviewer's CSV and compare it to the record.

    Args:
        csv: The review CSV.
        committed: The hash on record, or None when none was recorded.

    Returns:
        One message if the workbook moved or is absent.
    """
    if committed is None:
        return []
    if not csv.is_file():
        return [f"review CSV absent at {csv}"]
    recomputed = hashlib.sha256(csv.read_bytes()).hexdigest()
    if recomputed == committed:
        return []
    return [_difference("review_csv_sha256", committed, recomputed)]


def check_tracking_parity(
    logged: Mapping[str, float] | None,
    headline: Mapping[str, float],
) -> list[str]:
    """Compare the tracker's headline metrics to the committed ones.

    Args:
        logged: Metrics read back from the tracker, or None when it is
            unreachable.
        headline: The committed headline metrics.

    Returns:
        One message per disagreement; nothing when the tracker is
        unreachable, because an absent tracker is not a wrong number.
    """
    if logged is None:
        _LOGGER.info("tracking server unreachable; parity not checked")
        return []
    problems = []
    for name, value in headline.items():
        if name not in logged:
            problems.append(f"tracking: no logged {name} to compare")
        elif abs(logged[name] - value) > _TOLERANCE:
            problems.append(
                _difference(f"tracking.{name}", logged[name], value)
            )
    return problems


def rescore(
    config: config_module.Config,
    stored: pd.DataFrame,
    samples_table: pd.DataFrame,
    splits: pd.DataFrame,
    split_name: str,
    sources: Mapping[str, tuple[str, str]],
) -> dict[str, Any]:
    """Recompute one split's reports from the saved prediction rows.

    Leakage contract: reads targets of ``split_name`` only, and only for
    rows a model already predicted. It produces no new prediction and
    changes no model, so re-scoring a split that was already scored does
    not spend a second scoring of it.

    Args:
        config: Validated configuration.
        stored: Every stored prediction row.
        samples_table: The sample table, carrying targets.
        splits: The frozen split labels.
        split_name: The split to score.
        sources: Report name to the (track, model) that produced it.

    Returns:
        One report per source name.

    Raises:
        VerificationError: If a named source predicted no row.
    """
    wanted = set(splits.loc[splits["split"] == split_name, "sample_id"])
    reports: dict[str, Any] = {}
    for name, (track, model) in sources.items():
        rows = (
            samples_table.loc[samples_table["sample_id"].isin(wanted)]
            .merge(
                stored.loc[
                    (stored["track"] == track) & (stored["model"] == model)
                ],
                on="sample_id",
                how="inner",
            )
            .reset_index(drop=True)
        )
        if rows.empty:
            msg = f"{track}/{model} predicted no {split_name} row"
            raise VerificationError(msg)
        reports[name] = evaluate.evaluate_predictions(
            rows,
            config.evaluation.notification_match_ratio,
            config.evaluation.notification_support_minutes,
        )
    return reports


def report(problems: Sequence[str]) -> str:
    """Return a readable summary of a verification run.

    Args:
        problems: Every disagreement found.

    Returns:
        A one-line verdict followed by the differences, if any.
    """
    if not problems:
        return "verify: every committed value reproduced"
    lines = [f"verify: {len(problems)} value(s) disagree with the record"]
    lines += [f"  - {item}" for item in problems]
    return "\n".join(lines)


def load_metrics(path: pathlib.Path) -> dict[str, Any]:
    """Read the committed metrics record.

    Args:
        path: The metrics JSON.

    Returns:
        The committed payload.

    Raises:
        VerificationError: If nothing was committed there.
    """
    if not path.is_file():
        msg = f"no committed metrics at {path}; run the pipeline first"
        raise VerificationError(msg)
    return json.loads(path.read_text(encoding="utf-8"))
