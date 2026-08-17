"""Loads and validates the LLM track's configuration file.

The shared pipeline's configuration is loaded by ``config.py`` and is
not restated here; ``configs/llm.yaml`` names it and adds only what the
LLM track needs.
"""

from __future__ import annotations

import itertools
import pathlib
from typing import ClassVar

import pydantic
import yaml

_COMMIT_LENGTH = 40
_HEX_DIGITS = frozenset("0123456789abcdef")


class SettingsError(Exception):
    """Raised when the LLM configuration cannot be read or validated."""


class _StrictModel(pydantic.BaseModel):
    """Base for every model here: frozen, and unknown keys are errors."""

    model_config: ClassVar[pydantic.ConfigDict] = pydantic.ConfigDict(
        frozen=True,
        extra="forbid",
    )


class NotificationBuckets(_StrictModel):
    """The delay ladder the notification output is generated over.

    Attributes:
        ratio: Multiplicative width of one finite bucket.
        floor_minutes: Upper bound of the single bucket below the
            ladder.
        ceiling_days: Lower bound of the single bucket above it.
        min_train_rows: Training rows a bucket must hold after merging.
        ceiling_gate: Lowest acceptable representation ceiling.
        representative_decimals: Decimal places a representative is
            rounded to before coverage is computed.
    """

    ratio: float = pydantic.Field(gt=1.0)
    floor_minutes: int = pydantic.Field(gt=0)
    ceiling_days: int = pydantic.Field(gt=0)
    min_train_rows: int = pydantic.Field(gt=0)
    ceiling_gate: float = pydantic.Field(gt=0.0, le=1.0)
    representative_decimals: int = pydantic.Field(ge=0)


class BehaviourChange(_StrictModel):
    """When a resident counts as having changed behaviour.

    Attributes:
        min_priors: Prior bookings needed before the flag is judged.
        recent_depth: How many of the newest bookings count as recent.
        threshold: Distance at or above which the flag is raised.
    """

    min_priors: int = pydantic.Field(gt=0)
    recent_depth: int = pydantic.Field(gt=0)
    threshold: float = pydantic.Field(gt=0.0, le=1.0)


class Prompt(_StrictModel):
    """How one prediction is rendered into text.

    Attributes:
        version: Template version; changes with the template text.
        recent_events: Prior bookings rendered in full.
        float_decimals: Decimal places every rendered float is fixed to.
        change: Behaviour-change flag settings.
    """

    version: int = pydantic.Field(gt=0)
    recent_events: int = pydantic.Field(gt=0)
    float_decimals: int = pydantic.Field(ge=0)
    change: BehaviourChange


class Model(_StrictModel):
    """The artifact every scored answer is generated from.

    Attributes:
        id: Repository the quantised artifact is fetched from.
        revision: Immutable commit the artifact is pinned at.
        upstream_id: The unquantised model it was converted from,
            recorded as provenance and never loaded.
        hashed_files: Files hashed into the runtime identity.
    """

    id: str = pydantic.Field(min_length=1)
    revision: str = pydantic.Field(min_length=1)
    upstream_id: str = pydantic.Field(min_length=1)
    hashed_files: tuple[str, ...] = pydantic.Field(min_length=1)

    @pydantic.field_validator("revision")
    @classmethod
    def _check_immutable(cls, value: str) -> str:
        """Requires a full commit hash rather than a moving reference.

        Args:
            value: The configured revision.

        Returns:
            The revision, unchanged.

        Raises:
            ValueError: If it is not 40 hexadecimal characters, which
                means a tag or a branch could move under a result.
        """
        if len(value) != _COMMIT_LENGTH or not all(
            character in _HEX_DIGITS for character in value
        ):
            msg = (
                "model revision must be a full 40-character commit hash, "
                f"got {value!r}"
            )
            raise ValueError(msg)
        return value


class Decoding(_StrictModel):
    """How one answer is generated and how the stack is checked.

    Attributes:
        max_tokens: Longest answer accepted.
        temperature: Sampling temperature; zero is greedy.
        retries: Deterministic retries after a completion fails to
            parse or validate.
        fixtures: Compatibility fixtures the stack must pass.
        repeat_fixtures: Fixtures decoded a second time.
        latency_fixtures: Fixtures whose timing is kept.
        retry_reserve_floor: Lowest retry allowance a projection uses.
        retry_confidence: Confidence level of the upper bound computed
            from observed fixture failures.
    """

    max_tokens: int = pydantic.Field(gt=0)
    temperature: float = pydantic.Field(ge=0.0)
    retries: int = pydantic.Field(ge=0)
    fixtures: int = pydantic.Field(gt=0)
    repeat_fixtures: int = pydantic.Field(ge=0)
    latency_fixtures: int = pydantic.Field(ge=0)
    retry_reserve_floor: float = pydantic.Field(ge=0.0, lt=1.0)
    retry_confidence: float = pydantic.Field(gt=0.0, lt=1.0)

    @pydantic.model_validator(mode="after")
    def _check_subsets(self) -> Decoding:
        """Requires every measured subset to fit inside the fixture set.

        Returns:
            The validated settings.

        Raises:
            ValueError: If a subset asks for more rows than exist.
        """
        for name, size in (
            ("repeat_fixtures", self.repeat_fixtures),
            ("latency_fixtures", self.latency_fixtures),
        ):
            if size > self.fixtures:
                msg = (
                    f"{name} is {size}, which is more than the "
                    f"{self.fixtures} fixtures it is drawn from"
                )
                raise ValueError(msg)
        return self


class LoraParameters(_StrictModel):
    """The adapter's own shape.

    Attributes:
        rank: Rank of the low-rank update.
        dropout: Dropout applied inside the adapter.
        scale: Scaling applied to the adapter's contribution.
    """

    rank: int = pydantic.Field(gt=0)
    dropout: float = pydantic.Field(ge=0.0, lt=1.0)
    scale: float = pydantic.Field(gt=0.0)


class Tuning(_StrictModel):
    """Training controls shared by the pilot and the full run.

    Attributes:
        fine_tune_type: Which parameters are adapted.
        optimizer: Optimizer name the pinned trainer resolves.
        training_seed: Seeds the trainer's batching and initialisation.
        num_layers: Transformer blocks that receive an adapter.
        learning_rate: Optimizer learning rate.
        mask_prompt: Whether loss covers answer tokens only.
        grad_checkpoint: Whether activations are recomputed.
        batch_size: Minibatches per iteration.
        grad_accumulation_steps: Iterations per optimizer update.
        val_batches: Validation batches per evaluation.
        steps_per_report: Iterations between training reports.
        steps_per_eval: Iterations between validations.
        max_seq_length: Longest sequence kept whole.
        lora_parameters: The adapter's own shape.
    """

    fine_tune_type: str = pydantic.Field(min_length=1)
    optimizer: str = pydantic.Field(min_length=1)
    training_seed: int
    num_layers: int = pydantic.Field(gt=0)
    learning_rate: float = pydantic.Field(gt=0.0)
    mask_prompt: bool
    grad_checkpoint: bool
    batch_size: int = pydantic.Field(gt=0)
    grad_accumulation_steps: int = pydantic.Field(gt=0)
    val_batches: int = pydantic.Field(gt=0)
    steps_per_report: int = pydantic.Field(gt=0)
    steps_per_eval: int = pydantic.Field(gt=0)
    max_seq_length: int = pydantic.Field(gt=0)
    lora_parameters: LoraParameters


class Pilot(_StrictModel):
    """The fixed measurement run.

    Attributes:
        train_rows: Training rows the pilot is drawn to.
        epoch_equivalent: Passes over those rows.
        iters: Minibatch iterations trained.
        min_rows_per_label: Rows drawn per notification label before
            the remaining slots are filled uniformly.
        sample_seed: Seeds the draw.
    """

    train_rows: int = pydantic.Field(gt=0)
    epoch_equivalent: int = pydantic.Field(gt=0)
    iters: int = pydantic.Field(gt=0)
    min_rows_per_label: int = pydantic.Field(gt=0)
    sample_seed: int


class TierB(_StrictModel):
    """The declared full-training size.

    Attributes:
        train_rows: Training rows drawn for the full run.
        epoch_equivalent: Passes over those rows.
        iters: Minibatch iterations the full run would train.
        sample_seed: Seeds the draw.
    """

    train_rows: int = pydantic.Field(gt=0)
    epoch_equivalent: int = pydantic.Field(gt=0)
    iters: int = pydantic.Field(gt=0)
    sample_seed: int


class LadderStep(_StrictModel):
    """One pre-declared training size the projection may fall back to.

    Attributes:
        train_rows: Training rows this step would draw.
        epoch_equivalent: Passes over those rows.
    """

    train_rows: int = pydantic.Field(gt=0)
    epoch_equivalent: int = pydantic.Field(gt=0)


class Compute(_StrictModel):
    """What the compute projection is measured against.

    Attributes:
        cap_hours: Unattended compute the whole repository may spend.
        shared_pipeline_seconds: Measured runtime of the shared
            pipeline.
        gate_rows: Validation rows in the gate manifest.
        gate_passes: Scored passes over those rows.
    """

    cap_hours: float = pydantic.Field(gt=0.0)
    shared_pipeline_seconds: float = pydantic.Field(ge=0.0)
    gate_rows: int = pydantic.Field(gt=0)
    gate_passes: int = pydantic.Field(gt=0)


class Gate(_StrictModel):
    """How the development gate is drawn and decided.

    Attributes:
        sample_seed: Seeds the manifest draw.
        history_bins: Prior-booking counts the draw is stratified by,
            each edge opening a band up to the next.
        bootstrap_resamples: Paired resamples behind the interval.
        bootstrap_confidence: Confidence that interval carries.
    """

    sample_seed: int
    history_bins: tuple[int, ...] = pydantic.Field(min_length=1)
    bootstrap_resamples: int = pydantic.Field(gt=0)
    bootstrap_confidence: float = pydantic.Field(gt=0.0, lt=1.0)

    @pydantic.field_validator("history_bins")
    @classmethod
    def _check_ascending(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Requires strictly ascending, non-negative edges.

        Args:
            value: The configured edges.

        Returns:
            The edges, unchanged.

        Raises:
            ValueError: If they are not ascending, or one is negative,
                which would leave a row in two bands or in none.
        """
        if value[0] < 0 or any(
            later <= earlier for earlier, later in itertools.pairwise(value)
        ):
            msg = f"history bins must ascend from zero or above, got {value}"
            raise ValueError(msg)
        return value


class Settings(_StrictModel):
    """The LLM track's configuration.

    Attributes:
        shared_config: Path to the shared pipeline configuration.
        notification_buckets: The delay ladder settings.
        prompt: How one prediction is rendered into text.
        model: The artifact answers are generated from.
        decoding: How one answer is generated and checked.
        tuning: Training controls shared by every fit.
        pilot: The fixed measurement run.
        tier_b: The declared full-training size.
        ladder: Training sizes the projection may fall back to.
        compute: What the projection is measured against.
        gate: How the development gate is drawn and decided.
    """

    shared_config: pathlib.Path
    notification_buckets: NotificationBuckets
    prompt: Prompt
    model: Model
    decoding: Decoding
    tuning: Tuning
    pilot: Pilot
    tier_b: TierB
    ladder: tuple[LadderStep, ...] = pydantic.Field(min_length=1)
    compute: Compute
    gate: Gate


def parse_settings(raw: object) -> Settings:
    """Validates a parsed YAML document.

    Args:
        raw: The document as loaded from YAML.

    Returns:
        The validated, frozen settings.

    Raises:
        SettingsError: If the document is not a mapping or fails
            validation.
    """
    if not isinstance(raw, dict):
        msg = f"llm config must be a mapping, got {type(raw).__name__}"
        raise SettingsError(msg)
    try:
        return Settings.model_validate(raw)
    except pydantic.ValidationError as error:
        msg = f"invalid llm config: {error}"
        raise SettingsError(msg) from error


def load_settings(path: pathlib.Path) -> Settings:
    """Loads and validates the LLM configuration file.

    Args:
        path: Path to ``configs/llm.yaml``.

    Returns:
        The validated, frozen settings.

    Raises:
        SettingsError: If the file is unreadable, is not valid YAML, or
            fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        msg = f"cannot read llm config {path}: {error}"
        raise SettingsError(msg) from error
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        msg = f"cannot parse llm config {path}: {error}"
        raise SettingsError(msg) from error
    return parse_settings(raw)
