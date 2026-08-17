"""Draws and records the one full training run.

The pilot drew a floor of rows per label to prove every label was
trainable. This draw does the opposite: it is a plain seeded uniform
sample of the training prompts, so its label proportions are the
training split's own. A label the draw misses is recorded as
unrepresented and is never back-filled after the fact.

    training prompts ──> seeded uniform draw ──> sealed ladder step
                                                       │
                                            iterations, fixed in advance

The size is not chosen here. It was sealed by the timing-only ladder
decision before any quality number existed, and this module refuses to
train at any other size.

Leakage contract: reads rendered training prompts and the training
delays their labels come from. No validation or holdout target is read.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from typing import Any

from facility_prediction.llm import buckets as buckets_module
from facility_prediction.llm import ladder as ladder_module
from facility_prediction.llm import pilot_select
from facility_prediction.llm import settings as settings_module

GATE = "only adapter parameters train, and Outlines reloads the adapter"
RUN_NAME = "tier-b-adapter"

EVIDENCE = "measured"

SELECTED = "selected"

# A uniform draw takes no floor per label, unlike the pilot's.
_NO_FLOOR = 0


class TierBError(Exception):
    """Raised when the full run cannot be trained as it was sealed."""


def selected_step(decision: Mapping[str, Any]) -> dict[str, Any]:
    """Returns the ladder step this run is allowed to train.

    Args:
        decision: The sealed ladder decision.

    Returns:
        The selected step's rows, passes, and iteration count.

    Raises:
        TierBError: If the decision seals no step, or seals one that
            its own projection says does not fit the compute cap.
    """
    step = decision.get(SELECTED)
    if not step:
        msg = "the sealed ladder decision selects no step to train"
        raise TierBError(msg)
    if not step.get("fits"):
        msg = (
            f"ladder step {step['position']} is sealed but its projection "
            "does not fit the compute cap"
        )
        raise TierBError(msg)
    return dict(step)


def declared_iters(
    step: Mapping[str, Any], tuning: settings_module.Tuning
) -> int:
    """Returns the step's iteration count, having checked it holds.

    Args:
        step: The sealed ladder step.
        tuning: The training settings the iteration count assumes.

    Returns:
        The iteration count to train.

    Raises:
        TierBError: If the sealed count is not what the step's rows and
            passes imply under this batch size.
    """
    implied = ladder_module.iters_for(
        int(step["train_rows"]),
        int(step["epoch_equivalent"]),
        tuning.batch_size,
    )
    if implied != int(step["iters"]):
        msg = (
            f"the sealed step declares {step['iters']} iterations, but "
            f"{step['train_rows']} rows over {step['epoch_equivalent']} "
            f"pass(es) at batch size {tuning.batch_size} imply {implied}"
        )
        raise TierBError(msg)
    return implied


def select(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, str],
    step: Mapping[str, Any],
    seed: int,
) -> pilot_select.Selection:
    """Draws the full run's rows uniformly from the training prompts.

    The pilot's selector is reused with no per-label floor, so one
    definition of the draw exists rather than two.

    Args:
        rows: The rendered training prompt rows, in file order.
        labels: Sample identifier to notification label.
        step: The sealed ladder step.
        seed: Seeds the draw.

    Returns:
        The drawn rows and the shape of the draw.

    Raises:
        TierBError: If fewer training prompts exist than the step asks
            for.
    """
    try:
        return pilot_select.select(
            rows, labels, int(step["train_rows"]), _NO_FLOOR, seed
        )
    except pilot_select.PilotError as error:
        raise TierBError(str(error)) from error


def draw_record(
    selection: pilot_select.Selection,
    step: Mapping[str, Any],
    seed: int,
    ladder: Sequence[buckets_module.Bucket],
) -> dict[str, Any]:
    """Assembles the record of how the full run's rows were drawn.

    Args:
        selection: The drawn rows and the shape of the draw.
        step: The sealed ladder step.
        seed: The seed the draw used.
        ladder: The frozen delay buckets.

    Returns:
        The draw's settings, its per-label counts, and its identifiers.
    """
    sample_ids = [str(row[pilot_select.SAMPLE_ID]) for row in selection.rows]
    joined = "\n".join(sample_ids).encode("utf-8")
    return {
        "train_rows": len(selection.rows),
        "epoch_equivalent": int(step["epoch_equivalent"]),
        "min_rows_per_label": _NO_FLOOR,
        "sample_seed": seed,
        "ladder_position": int(step["position"]),
        "support": selection.support,
        "available": selection.available,
        "unrepresented": selection.unrepresented,
        "shares": pilot_select.shares(selection, ladder),
        "sample_ids": sample_ids,
        "sample_ids_sha256": hashlib.sha256(joined).hexdigest(),
    }


def check_reload(reloaded: Mapping[str, Any], saved: Mapping[str, Any]) -> None:
    """Refuses an adapter the scored runtime cannot load back.

    Args:
        reloaded: What the runtime read back off disk.
        saved: What the trainer reported writing.

    Raises:
        TierBError: If the reloaded adapter is not the saved one.
    """
    for key in ("parameters", "adapted_blocks", "sha256"):
        if reloaded.get(key) != saved.get(key):
            msg = (
                f"the reloaded adapter's {key} is {reloaded.get(key)}, but "
                f"the trainer saved {saved.get(key)}"
            )
            raise TierBError(msg)
