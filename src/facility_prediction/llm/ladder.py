"""Decides how large the full training run may be, on timing alone.

The pilot measures what one iteration costs on this machine. From that,
and from what one generated answer costs, the whole remaining bill can
be projected: the training itself, every answer still to be scored, and
an allowance for retries. The largest pre-declared size whose bill fits
the cap is the one that runs.

    pilot seconds/iteration ─┐
    seconds per scored row ──┼─> projected total ──> first size that fits
    compute already spent ───┘                          │
                                                        └─ none fits: stop

The order of the sizes is fixed before any of this is measured, and the
decision is sealed once written. Nothing here reads a loss, a metric, or
a prediction: a size chosen after seeing how well the model did is a
size chosen to flatter it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
import pathlib
from typing import Any

from facility_prediction.data import split as split_module
from facility_prediction.llm import settings as settings_module

GATE = "the cumulative compute projection fits the unattended cap"
RUN_NAME = "ladder-decision"
COMPUTE_TIMEOUT = "compute_timeout"

SECONDS_PER_HOUR = 3600.0

EVIDENCE = "measured"

# Terms whose presence in a decision would mean the size was chosen with
# knowledge of how well the model performed.
BLIND_TERMS = ("loss", "accuracy", "f1", "macro", "quality", "score")


class LadderError(Exception):
    """Raised when no pre-declared training size fits the cap."""


class DecisionError(Exception):
    """Raised when a decision is malformed or would change once sealed."""


@dataclasses.dataclass(frozen=True)
class Step:
    """One pre-declared training size.

    Attributes:
        position: Its place in the pre-declared order, counting from 1.
        train_rows: Training rows the step would draw.
        epoch_equivalent: Passes over those rows.
        iters: Minibatch iterations that implies.
        optimizer_updates: Optimizer updates that implies.
    """

    position: int
    train_rows: int
    epoch_equivalent: int
    iters: int
    optimizer_updates: int


@dataclasses.dataclass(frozen=True)
class Projection:
    """What one step would cost, end to end.

    Attributes:
        step: The size being projected.
        prior_seconds: Unattended compute already spent.
        training_seconds: What this step's training would take.
        generation_seconds: What every answer still to be scored would
            take, retries included.
    """

    step: Step
    prior_seconds: float
    training_seconds: float
    generation_seconds: float

    @property
    def total_seconds(self) -> float:
        """Returns the projected total spend in seconds.

        Returns:
            Prior spend plus training plus generation.
        """
        return (
            self.prior_seconds + self.training_seconds + self.generation_seconds
        )

    @property
    def total_hours(self) -> float:
        """Returns the projected total spend in hours.

        Returns:
            :attr:`total_seconds` expressed in hours.
        """
        return self.total_seconds / SECONDS_PER_HOUR


def iters_for(train_rows: int, epoch_equivalent: int, batch_size: int) -> int:
    """Returns the minibatch iterations a size implies.

    The trainer counts one iteration per minibatch, so the number of
    passes over the rows fixes the iteration count exactly.

    Args:
        train_rows: Training rows the step draws.
        epoch_equivalent: Passes over those rows.
        batch_size: Minibatches per iteration.

    Returns:
        The implied iteration count.
    """
    return math.ceil(train_rows / batch_size) * epoch_equivalent


def build_step(
    position: int,
    train_rows: int,
    epoch_equivalent: int,
    tuning: settings_module.Tuning,
) -> Step:
    """Builds one step and checks its arithmetic.

    Args:
        position: Its place in the pre-declared order, counting from 1.
        train_rows: Training rows the step draws.
        epoch_equivalent: Passes over those rows.
        tuning: The training controls the arithmetic assumes.

    Returns:
        The step, with its iteration and update counts.

    Raises:
        DecisionError: If the iteration count does not divide into whole
            optimizer updates, which would leave a partial update at the
            end of the run.
    """
    iters = iters_for(train_rows, epoch_equivalent, tuning.batch_size)
    if iters % tuning.grad_accumulation_steps:
        msg = (
            f"{iters} iterations do not divide into whole optimizer updates "
            f"at accumulation {tuning.grad_accumulation_steps}"
        )
        raise DecisionError(msg)
    return Step(
        position=position,
        train_rows=train_rows,
        epoch_equivalent=epoch_equivalent,
        iters=iters,
        optimizer_updates=iters // tuning.grad_accumulation_steps,
    )


def build_steps(settings: settings_module.Settings) -> list[Step]:
    """Builds every pre-declared step, in order.

    Args:
        settings: The LLM configuration.

    Returns:
        The steps, largest first.

    Raises:
        DecisionError: If a step's arithmetic does not hold, or the
            first step is not the declared full-training size.
    """
    steps = [
        build_step(
            position, entry.train_rows, entry.epoch_equivalent, settings.tuning
        )
        for position, entry in enumerate(settings.ladder, start=1)
    ]
    check_declared(steps[0], settings.tier_b, settings.tuning)
    return steps


def check_declared(
    first: Step,
    tier_b: settings_module.TierB,
    tuning: settings_module.Tuning,
) -> None:
    """Requires the first step to be the declared full-training size.

    Args:
        first: The first pre-declared step.
        tier_b: The declared full-training size.
        tuning: The training controls the arithmetic assumes.

    Raises:
        DecisionError: If the two disagree on rows, passes, or the
            iteration count they imply.
    """
    declared = iters_for(
        tier_b.train_rows, tier_b.epoch_equivalent, tuning.batch_size
    )
    if declared != tier_b.iters:
        msg = (
            f"the declared full training says {tier_b.iters} iterations, but "
            f"{tier_b.train_rows} rows over {tier_b.epoch_equivalent} passes "
            f"is {declared}"
        )
        raise DecisionError(msg)
    if (first.train_rows, first.epoch_equivalent) != (
        tier_b.train_rows,
        tier_b.epoch_equivalent,
    ):
        msg = (
            f"the first step is {first.train_rows} rows over "
            f"{first.epoch_equivalent} passes, but the declared full training "
            f"is {tier_b.train_rows} over {tier_b.epoch_equivalent}"
        )
        raise DecisionError(msg)


def scored_generations(
    compute: settings_module.Compute, comparison_rows: int
) -> int:
    """Returns how many scored answers the branch has left to generate.

    Args:
        compute: The projection's settings.
        comparison_rows: Rows on the shared holdout comparison manifest.

    Returns:
        Gate rows in every gate pass, plus the single scored pass over
        the comparison manifest.
    """
    return compute.gate_rows * compute.gate_passes + comparison_rows


def generation_seconds(
    scored: int, seconds_per_row: float, retry_reserve: float
) -> float:
    """Returns what the remaining scored answers would cost.

    Args:
        scored: Answers still to be generated.
        seconds_per_row: Measured cost of one answer.
        retry_reserve: Allowance for answers that must be generated
            twice.

    Returns:
        The projected generation seconds.
    """
    return scored * seconds_per_row * (1.0 + retry_reserve)


def project(
    steps: Sequence[Step],
    prior_seconds: float,
    seconds_per_iteration: float,
    generation: float,
) -> list[Projection]:
    """Projects the full bill for every step.

    Args:
        steps: The pre-declared steps, in order.
        prior_seconds: Unattended compute already spent.
        seconds_per_iteration: Measured cost of one training iteration.
        generation: Projected cost of every remaining scored answer.

    Returns:
        One projection per step, in the same order.
    """
    return [
        Projection(
            step=step,
            prior_seconds=prior_seconds,
            training_seconds=step.iters * seconds_per_iteration,
            generation_seconds=generation,
        )
        for step in steps
    ]


def decide(projections: Sequence[Projection], cap_seconds: float) -> Projection:
    """Returns the first step whose projected bill fits the cap.

    Args:
        projections: One projection per step, in the pre-declared order.
        cap_seconds: The unattended compute cap.

    Returns:
        The first projection that fits.

    Raises:
        LadderError: If none of them fits, which stops the branch.
    """
    for projection in projections:
        if projection.total_seconds <= cap_seconds:
            return projection
    smallest = projections[-1]
    msg = (
        f"the smallest pre-declared training of {smallest.step.train_rows} "
        f"rows over {smallest.step.epoch_equivalent} pass(es) projects "
        f"{smallest.total_hours:.2f}h against a cap of "
        f"{cap_seconds / SECONDS_PER_HOUR:.2f}h"
    )
    raise LadderError(msg)


def _step_payload(projection: Projection, cap_seconds: float) -> dict[str, Any]:
    """Renders one projection for the record.

    Args:
        projection: The projection to render.
        cap_seconds: The unattended compute cap.

    Returns:
        The projection's fields, ready to serialise.
    """
    return {
        "position": projection.step.position,
        "train_rows": projection.step.train_rows,
        "epoch_equivalent": projection.step.epoch_equivalent,
        "iters": projection.step.iters,
        "optimizer_updates": projection.step.optimizer_updates,
        "training_seconds": projection.training_seconds,
        "total_hours": projection.total_hours,
        "fits": projection.total_seconds <= cap_seconds,
    }


def build_decision(
    projections: Sequence[Projection],
    selected: Projection,
    settings: settings_module.Settings,
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Assembles the sealed decision.

    Args:
        projections: One projection per step, in the pre-declared order.
        selected: The step that was chosen.
        settings: The LLM configuration.
        inputs: The measured quantities the projection was built from.

    Returns:
        The decision payload, ready to serialise.
    """
    cap_seconds = settings.compute.cap_hours * SECONDS_PER_HOUR
    return {
        "evidence": EVIDENCE,
        "gate": GATE,
        "cap_hours": settings.compute.cap_hours,
        "batch_size": settings.tuning.batch_size,
        "grad_accumulation_steps": settings.tuning.grad_accumulation_steps,
        "inputs": dict(inputs),
        "steps": [
            _step_payload(projection, cap_seconds) for projection in projections
        ],
        "selected": _step_payload(selected, cap_seconds),
        "projected_total_hours": selected.total_hours,
        "generation_seconds": selected.generation_seconds,
        "prior_seconds": selected.prior_seconds,
    }


def check_blind(payload: Any, path: str = "") -> None:
    """Refuses a decision that carries anything about model quality.

    Field names are checked rather than values: a name is what a
    quantity means, and a hash or a path can hold any letters at all.

    Args:
        payload: The decision, or one of its members.
        path: Where this member sits, for the message.

    Raises:
        DecisionError: If a field names a quality measure, which would
            mean the size was not chosen blind.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            _check_blind_name(str(key), f"{path}.{key}")
            check_blind(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            check_blind(value, f"{path}[{index}]")


def _check_blind_name(name: str, path: str) -> None:
    """Refuses one field name that names a quality measure.

    Args:
        name: The field name to check.
        path: Where it sits, for the message.

    Raises:
        DecisionError: If it names a quality measure.
    """
    lowered = name.lower()
    for term in BLIND_TERMS:
        if term in lowered:
            msg = (
                f"the ladder decision carries {term!r} at "
                f"{path or 'the root'}, so it was not made on timing alone"
            )
            raise DecisionError(msg)


def write_once(payload: Mapping[str, Any], path: pathlib.Path) -> str:
    """Writes the decision, refusing to change one already sealed.

    Args:
        payload: The decision to write.
        path: Destination JSON.

    Returns:
        The hex SHA-256 of the bytes written or already present.

    Raises:
        DecisionError: If a different decision is already sealed there.
    """
    if path.is_file():
        sealed = json.loads(path.read_text(encoding="utf-8"))
        if sealed != json.loads(json.dumps(payload)):
            msg = (
                f"a different ladder decision is already sealed at {path}; "
                "it is walked once, before training, and never revisited"
            )
            raise DecisionError(msg)
    return split_module.write_json(dict(payload), path)


def run_params(payload: Mapping[str, Any]) -> dict[str, str]:
    """Returns the run parameters for this decision.

    Args:
        payload: The sealed decision.

    Returns:
        Parameter name to value, as text.
    """
    selected = payload["selected"]
    params = {f"selected_{key}": str(value) for key, value in selected.items()}
    params["cap_hours"] = str(payload["cap_hours"])
    params["batch_size"] = str(payload["batch_size"])
    params["grad_accumulation_steps"] = str(payload["grad_accumulation_steps"])
    params["evidence"] = str(payload["evidence"])
    return params


def run_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Returns the run metrics for this decision.

    Args:
        payload: The sealed decision.

    Returns:
        Metric name to value.
    """
    selected = payload["selected"]
    metrics = {
        "selected_position": float(selected["position"]),
        "selected_iters": float(selected["iters"]),
        "selected_train_rows": float(selected["train_rows"]),
        "training_seconds": float(selected["training_seconds"]),
        "generation_seconds": float(payload["generation_seconds"]),
        "prior_seconds": float(payload["prior_seconds"]),
        "projected_total_hours": float(payload["projected_total_hours"]),
    }
    metrics.update(
        {
            f"input_{key}": float(value)
            for key, value in payload["inputs"].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return metrics
