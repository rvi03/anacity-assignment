"""Draws the fixed set of rows the pilot trains on.

The pilot exists to answer one question — can every notification label
be trained at all — so the draw guarantees a floor of rows per label
before it fills the rest uniformly. That makes it deliberately *not*
prevalence-representative: its per-label counts describe the draw, never
the population, and must never be read as one.

    training prompt rows ──> floor per label ──> seeded uniform fill
                                                       │
                                                 fixed pilot set

Leakage contract: reads rendered training prompts and the training
delays their labels come from. No validation or holdout row is drawn,
and no target is read for any row outside the training split.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction.evaluation import evaluate
from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import settings as settings_module

SAMPLE_ID = "sample_id"


class PilotError(Exception):
    """Raised when the pilot set cannot be drawn as declared."""


@dataclasses.dataclass(frozen=True)
class Selection:
    """The drawn rows and what the draw looked like.

    Attributes:
        rows: The drawn prompt rows, ordered by sample identifier.
        support: Label to rows drawn for it.
        available: Label to rows the draw could have taken.
        unrepresented: Labels with no row in the draw.
    """

    rows: list[dict[str, Any]]
    support: dict[str, int]
    available: dict[str, int]
    unrepresented: list[str]


def training_labels(
    samples: pd.DataFrame, ladder: Sequence[buckets_module.Bucket]
) -> dict[str, str]:
    """Returns the notification label of every training sample.

    Leakage contract: the caller passes training rows only; the delay
    read here is a training target and never leaves this split.

    Args:
        samples: Training samples carrying their notification delay.
        ladder: The frozen delay buckets.

    Returns:
        Sample identifier to label.
    """
    delays = samples[evaluate.TARGET_COLUMNS[evaluate.NOTIFICATION]]
    labels = buckets_module.assign(delays, list(ladder))
    return {
        str(sample_id): str(label)
        for sample_id, label in zip(samples[SAMPLE_ID], labels, strict=True)
    }


def _grouped(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> dict[str, list[int]]:
    """Groups row positions by label.

    Args:
        rows: The prompt rows to group, in file order.
        labels: Sample identifier to label.

    Returns:
        Label to the positions carrying it, ascending.

    Raises:
        PilotError: If a row has no label, which means it is not a
            training row.
    """
    grouped: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        sample_id = str(row[SAMPLE_ID])
        if sample_id not in labels:
            msg = f"prompt row {sample_id!r} has no training label"
            raise PilotError(msg)
        grouped.setdefault(labels[sample_id], []).append(position)
    return grouped


def select(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    train_rows: int,
    min_rows_per_label: int,
    seed: int,
) -> Selection:
    """Draws the pilot set: a floor per label, then a uniform fill.

    Args:
        rows: The rendered training prompt rows, in file order.
        labels: Sample identifier to label.
        train_rows: Rows the pilot trains on.
        min_rows_per_label: Rows drawn per label before the fill.
        seed: Seeds the draw.

    Returns:
        The drawn rows and the shape of the draw.

    Raises:
        PilotError: If the floor alone needs more rows than the pilot
            trains on, or fewer rows exist than it asks for.
    """
    if len(rows) < train_rows:
        msg = f"cannot draw {train_rows} pilot rows from {len(rows)} prompts"
        raise PilotError(msg)

    grouped = _grouped(rows, labels)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for label in sorted(grouped):
        pool = grouped[label]
        take = min(min_rows_per_label, len(pool))
        chosen.extend(
            int(position)
            for position in rng.choice(pool, size=take, replace=False)
        )

    if len(chosen) > train_rows:
        msg = (
            f"covering {len(grouped)} labels at {min_rows_per_label} rows each "
            f"needs {len(chosen)} rows, more than the {train_rows} the pilot "
            "trains on"
        )
        raise PilotError(msg)

    taken = set(chosen)
    remaining = [
        position for position in range(len(rows)) if position not in taken
    ]
    fill = rng.choice(remaining, size=train_rows - len(chosen), replace=False)
    chosen.extend(int(position) for position in fill)

    drawn = sorted(
        (dict(rows[position]) for position in chosen),
        key=lambda row: str(row[SAMPLE_ID]),
    )
    support = _support(drawn, labels)
    return Selection(
        rows=drawn,
        support=support,
        available={label: len(pool) for label, pool in sorted(grouped.items())},
        unrepresented=sorted(set(grouped) - set(support)),
    )


def _support(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> dict[str, int]:
    """Counts the drawn rows per label.

    Args:
        rows: The drawn rows.
        labels: Sample identifier to label.

    Returns:
        Label to rows drawn, ascending by label.
    """
    counts: dict[str, int] = {}
    for row in rows:
        label = labels[str(row[SAMPLE_ID])]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def draw_record(
    selection: Selection,
    pilot: settings_module.Pilot,
    ladder: Sequence[buckets_module.Bucket],
) -> dict[str, Any]:
    """Assembles the record of how the pilot set was drawn.

    The drawn identifiers are listed, so the training set can be rebuilt
    without rerunning the draw.

    Args:
        selection: The drawn rows and the shape of the draw.
        pilot: The pilot's settings.
        ladder: The frozen delay buckets.

    Returns:
        The draw's settings, its per-label counts, and its identifiers.
    """
    sample_ids = [str(row[SAMPLE_ID]) for row in selection.rows]
    joined = "\n".join(sample_ids).encode("utf-8")
    return {
        "train_rows": len(selection.rows),
        "epoch_equivalent": pilot.epoch_equivalent,
        "min_rows_per_label": pilot.min_rows_per_label,
        "sample_seed": pilot.sample_seed,
        "support": selection.support,
        "available": selection.available,
        "unrepresented": selection.unrepresented,
        "shares": shares(selection, ladder),
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(joined).hexdigest(),
    }


def shares(
    selection: Selection, ladder: Sequence[buckets_module.Bucket]
) -> dict[str, Any]:
    """Compares the draw's label proportions with the training split's.

    The pilot is drawn to a floor per label, so a deviation here is the
    design working rather than a fault. It is reported so no reader
    mistakes the draw for a prevalence estimate.

    Args:
        selection: The drawn rows and the shape of the draw.
        ladder: The frozen delay buckets, carrying training counts.

    Returns:
        Per-label shares and the largest absolute deviation between
        them.
    """
    drawn_total = sum(selection.support.values())
    train_total = sum(bucket.train_rows for bucket in ladder)
    per_label = {}
    largest = 0.0
    for bucket in ladder:
        drawn = selection.support.get(bucket.label, 0)
        pilot_share = drawn / drawn_total if drawn_total else 0.0
        train_share = bucket.train_rows / train_total if train_total else 0.0
        largest = max(largest, abs(pilot_share - train_share))
        per_label[bucket.label] = {
            "pilot_rows": drawn,
            "pilot_share": pilot_share,
            "train_rows": bucket.train_rows,
            "train_share": train_share,
        }
    return {
        "prevalence_representative": False,
        "per_label": per_label,
        "max_absolute_share_deviation": largest,
    }
