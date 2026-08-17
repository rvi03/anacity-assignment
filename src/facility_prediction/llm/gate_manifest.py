"""Freezes the rows both development passes are judged on.

Two hundred validation rows, drawn once and never redrawn. Both passes
answer the same rows in the same order, which is what makes them
comparable row by row rather than only in aggregate.

    validation rows ──> strata: facility x history length
                              │
                    one row each, then proportional fill
                              │
                        frozen manifest

Rare strata keep their single row rather than being rounded away, so a
facility nobody books is still represented in what the model is judged
on.

Leakage contract: reads validation-split rows only, and only their
target facility and prior-booking count, both of which are readable
before scoring. No holdout row is drawn.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction.evaluation import evaluate
from facility_prediction.llm import settings as settings_module

SAMPLE_ID = "sample_id"
PRIOR_BOOKINGS = "n_prior_bookings"

FACILITY_COLUMN = evaluate.TARGET_COLUMNS[evaluate.FACILITY]

STRATUM = "stratum"
HISTORY_BAND = "history_band"


class ManifestError(Exception):
    """Raised when the gate manifest cannot be drawn as declared."""


def history_band(count: int, bins: Sequence[int]) -> str:
    """Returns the history band a prior-booking count falls in.

    Args:
        count: Prior bookings the row carries.
        bins: Ascending band edges; each opens a band up to the next.

    Returns:
        The band's label, its own edges in it.
    """
    if count < bins[0]:
        return f"under_{bins[0]}"
    for lower, upper in itertools.pairwise(bins):
        if count < upper:
            return f"{lower}_to_{upper - 1}"
    return f"{bins[-1]}_plus"


def label_strata(rows: pd.DataFrame, bins: Sequence[int]) -> pd.DataFrame:
    """Labels each row with the stratum it is drawn from.

    Args:
        rows: Validation rows carrying their facility and history.
        bins: Ascending history band edges.

    Returns:
        The rows with their band and stratum labels added.

    Raises:
        ManifestError: If a column the draw needs is absent.
    """
    missing = [
        column
        for column in (SAMPLE_ID, FACILITY_COLUMN, PRIOR_BOOKINGS)
        if column not in rows.columns
    ]
    if missing:
        msg = f"the gate draw needs columns {missing}"
        raise ManifestError(msg)
    labelled = rows.copy()
    labelled[HISTORY_BAND] = [
        history_band(int(count), bins) for count in labelled[PRIOR_BOOKINGS]
    ]
    labelled[STRATUM] = (
        labelled[FACILITY_COLUMN].astype(str) + " | " + labelled[HISTORY_BAND]
    )
    return labelled


def allocate(sizes: Mapping[str, int], total: int) -> dict[str, int]:
    """Shares the draw across strata, keeping the rare ones.

    Every non-empty stratum receives one row before anything is shared
    proportionally, and the proportional remainder is settled largest
    first. A stratum never receives more rows than it holds.

    Args:
        sizes: Stratum to how many rows it holds.
        total: Rows the manifest draws in all.

    Returns:
        Stratum to how many rows it contributes.

    Raises:
        ManifestError: If the strata hold fewer rows than the manifest
            needs, or there are more strata than rows to cover them.
    """
    available = sum(sizes.values())
    if available < total:
        msg = f"cannot draw {total} gate rows from {available} rows"
        raise ManifestError(msg)
    if len(sizes) > total:
        msg = (
            f"{len(sizes)} strata cannot each keep a row in a manifest of "
            f"{total}"
        )
        raise ManifestError(msg)

    allocation = dict.fromkeys(sorted(sizes), 1)
    remaining = total - len(allocation)
    shares = [
        (
            (sizes[stratum] - 1) * remaining / (available - len(allocation)),
            stratum,
        )
        for stratum in sorted(sizes)
    ]
    for share, stratum in shares:
        take = min(int(share), sizes[stratum] - allocation[stratum])
        allocation[stratum] += take
        remaining -= take

    for _, stratum in sorted(shares, reverse=True):
        if remaining <= 0:
            break
        if allocation[stratum] < sizes[stratum]:
            allocation[stratum] += 1
            remaining -= 1
    return allocation


def draw(
    rows: pd.DataFrame,
    gate: settings_module.Gate,
    total: int,
) -> pd.DataFrame:
    """Draws the frozen manifest rows.

    Args:
        rows: Validation rows carrying their facility and history.
        gate: The gate's settings.
        total: Rows the manifest draws in all.

    Returns:
        The drawn rows, ordered by sample identifier.

    Raises:
        ManifestError: If the draw cannot be filled.
    """
    labelled = label_strata(rows, gate.history_bins)
    grouped = {
        str(stratum): frame
        for stratum, frame in labelled.groupby(STRATUM, sort=True)
    }
    allocation = allocate(
        {stratum: len(frame) for stratum, frame in grouped.items()}, total
    )

    rng = np.random.default_rng(gate.sample_seed)
    drawn = []
    for stratum in sorted(grouped):
        frame = grouped[stratum].sort_values(SAMPLE_ID)
        chosen = rng.choice(len(frame), size=allocation[stratum], replace=False)
        drawn.append(frame.iloc[sorted(int(index) for index in chosen)])
    return pd.concat(drawn).sort_values(SAMPLE_ID).reset_index(drop=True)


def build_manifest(
    drawn: pd.DataFrame,
    gate: settings_module.Gate,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assembles the record of what was drawn and how.

    Args:
        drawn: The drawn rows.
        gate: The gate's settings.
        provenance: What the draw was made from.

    Returns:
        The manifest payload, ready to serialise.
    """
    counts = drawn[STRATUM].value_counts().sort_index()
    return {
        "rows": len(drawn),
        "split": "validation",
        "settings": {
            "sample_seed": gate.sample_seed,
            "history_bins": list(gate.history_bins),
        },
        "provenance": dict(provenance),
        "strata": {str(name): int(value) for name, value in counts.items()},
        "singleton_strata": int((counts == 1).sum()),
        "sample_ids": [str(value) for value in drawn[SAMPLE_ID]],
    }


def check_sealed(manifest: Mapping[str, Any], holdout: set[str]) -> None:
    """Refuses a manifest that reached the sealed split.

    Args:
        manifest: The assembled manifest.
        holdout: Sample identifiers on the holdout split.

    Raises:
        ManifestError: If any drawn row belongs to the holdout.
    """
    inside = sorted(set(manifest["sample_ids"]) & holdout)
    if inside:
        msg = (
            f"{len(inside)} holdout row(s) reached the gate manifest, first "
            f"{inside[0]!r}; the holdout stays sealed until scoring"
        )
        raise ManifestError(msg)
