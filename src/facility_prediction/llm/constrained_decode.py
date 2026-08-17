"""Generates one answer per prompt in a shape that cannot be wrong.

The model runs on the host GPU and its output is constrained to the
frozen schema while it is being produced, so a hallucinated facility,
a missing field, an extra field, or free prose is not something to
detect afterwards — it is unreachable.

    prompt ─> pinned 4-bit weights ─> schema-constrained decode ─┐
                                                                 │
    typed failure row <── second failure <── validate <───────────┘
                                                  │
                                            four fields

Shape is not meaning. A structurally perfect answer can still name an
hour the facility is shut, so every answer is checked against the
configured operating hours and the invalid ones are counted, never
rewritten.

The generation call is made at stream level rather than through the
model's own ``generate`` because that is where per-row prefill and
decode rates are observable; it passes the same formatted input and the
same logits processor that call would.

Leakage contract: this module reads a prompt and returns an answer. It
never sees a split, a target, or a booking table, so it has no origin
of its own to respect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import pathlib
import time
from typing import Any

from facility_prediction import config as config_module
from facility_prediction.llm import serialize
from facility_prediction.llm import settings as settings_module

GATE = "the pinned decoding stack passes every compatibility fixture"
RUN_NAME = "decode-check"

VALID = "valid"
FAILED = "failed"

GENERATION_STAGE = "generation"
PARSE_STAGE = "parse"
SCHEMA_STAGE = "schema"

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

# Probe strings used only to read the chat template's own shape back
# out of it. They never reach the model.
_PROBE = "probe"
_BODY = "@@ANSWER@@"

ADAPTER_FILE = "adapters.safetensors"

BYTES_PER_GIB = 1024**3
_PROBABILITY_TOLERANCE = 1e-12
_BISECTION_STEPS = 200


class DecodeError(Exception):
    """Raised when a completion cannot be turned into four fields."""

    def __init__(self, stage: str, reason: str) -> None:
        """Records which stage rejected the completion, and why.

        Args:
            stage: Where it was rejected.
            reason: What was wrong, in one line.
        """
        super().__init__(f"{stage}: {reason}")
        self.stage = stage
        self.reason = reason


class StackError(Exception):
    """Raised when the pinned runtime cannot be assembled."""


@dataclasses.dataclass(frozen=True)
class Completion:
    """One attempt's raw result and what it cost.

    Attributes:
        text: The text the model returned.
        prompt_tokens: Tokens in the formatted prompt.
        completion_tokens: Tokens the model produced.
        prefill_tokens_per_second: Rate the prompt was read at.
        decode_tokens_per_second: Rate the answer was written at.
        peak_memory_gib: Highest memory the runtime reached.
        seconds: Wall-clock time of the call.
        finish_reason: Why generation ended.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    peak_memory_gib: float
    seconds: float
    finish_reason: str


@dataclasses.dataclass(frozen=True)
class Outcome:
    """What one prompt produced, valid or not.

    Attributes:
        status: ``valid`` or ``failed``.
        fields: The four parsed fields, or None on a failure.
        completion: The last attempt's raw result.
        attempts: How many attempts were made.
        failure_stage: Which stage rejected the last attempt, or None.
        failure_reason: Why it was rejected, or None.
        semantic_invalid_reason: Why the answer is impossible in
            practice, or None. A valid row may still carry one.
    """

    status: str
    fields: dict[str, Any] | None
    completion: Completion
    attempts: int
    failure_stage: str | None
    failure_reason: str | None
    semantic_invalid_reason: str | None


@dataclasses.dataclass(frozen=True)
class Runtime:
    """The assembled generation path and what identifies it.

    Attributes:
        model: The Outlines model wrapping the quantised weights.
        processor: The logits processor holding the frozen schema.
        sampler: The token sampler; greedy at temperature zero.
        max_tokens: Longest answer accepted.
        assistant_prefix: Text the chat template puts in front of an
            assistant answer, supplied before the schema takes over.
        identity: Every field that makes one call reproducible.
    """

    model: Any
    processor: Any
    sampler: Any
    max_tokens: int
    assistant_prefix: str
    identity: dict[str, Any]


def parse_fields(text: str, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Turns one completion into the four contracted fields.

    Args:
        text: The raw completion.
        schema: The frozen output schema.

    Returns:
        The parsed fields, in contract order.

    Raises:
        DecodeError: If the text is not one JSON object, omits a
            required field, carries an extra field, or holds a value
            outside its enum.
    """
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise DecodeError(PARSE_STAGE, f"not JSON: {error}") from error
    if not isinstance(payload, dict):
        msg = f"not a JSON object but a {type(payload).__name__}"
        raise DecodeError(PARSE_STAGE, msg)

    properties = schema["properties"]
    missing = sorted(set(schema["required"]) - set(payload))
    if missing:
        raise DecodeError(SCHEMA_STAGE, f"missing field(s) {missing}")
    extra = sorted(set(payload) - set(properties))
    if extra:
        raise DecodeError(SCHEMA_STAGE, f"unexpected field(s) {extra}")
    for name, allowed in properties.items():
        if payload[name] not in allowed["enum"]:
            msg = f"{name}={payload[name]!r} is outside its allowed values"
            raise DecodeError(SCHEMA_STAGE, msg)
    return {name: payload[name] for name in schema["required"]}


def semantic_invalid_reason(
    fields: Mapping[str, Any], config: config_module.Config
) -> str | None:
    """Reports an answer that is well-formed but impossible.

    The schema cannot express that a facility is shut at three in the
    morning, so the hour is checked against the configured half-open
    operating interval here and the answer is counted, never corrected.

    Args:
        fields: The four parsed fields.
        config: Validated shared configuration.

    Returns:
        Why the answer is impossible, or None when it is not.
    """
    name = fields[serialize.FACILITY_FIELD]
    hour = fields[serialize.USAGE_HOUR_FIELD]
    for facility in config.facilities:
        if facility.name != name:
            continue
        if facility.open_hour <= hour < facility.close_hour:
            return None
        return (
            f"{name} is open {facility.open_hour:02d}:00-"
            f"{facility.close_hour:02d}:00 and the answer says "
            f"{hour:02d}:00"
        )
    return f"{name} is not in the facility catalog"


def decode_one(
    generate: Callable[[], Completion],
    schema: Mapping[str, Any],
    config: config_module.Config,
    retries: int,
    attempt_scope: Callable[[int], Any] | None = None,
) -> Outcome:
    """Generates one answer, retrying a rejected completion once.

    A completion that fails again is returned as a typed failure. It is
    never replaced by the true label, by a baseline, or by a repair.

    Args:
        generate: Produces one completion; called once per attempt.
        schema: The frozen output schema.
        config: Validated shared configuration.
        retries: Deterministic retries after the first attempt.
        attempt_scope: Opens a recording context for attempt ``index``,
            or None when the call is not being recorded.

    Returns:
        The outcome, valid or typed failure.
    """
    completion = None
    stage, reason = GENERATION_STAGE, "no attempt was made"
    for index in range(retries + 1):
        with _scope(attempt_scope, index) as attempt:
            completion = generate()
            try:
                fields = parse_fields(completion.text, schema)
            except DecodeError as error:
                stage, reason = error.stage, error.reason
                _record_failure(attempt, error, completion)
                continue
            _record_success(attempt, completion)
            return Outcome(
                status=VALID,
                fields=fields,
                completion=completion,
                attempts=index + 1,
                failure_stage=None,
                failure_reason=None,
                semantic_invalid_reason=semantic_invalid_reason(fields, config),
            )
    if completion is None:
        completion = Completion(
            text="",
            prompt_tokens=0,
            completion_tokens=0,
            prefill_tokens_per_second=0.0,
            decode_tokens_per_second=0.0,
            peak_memory_gib=0.0,
            seconds=0.0,
            finish_reason=GENERATION_STAGE,
        )
    return Outcome(
        status=FAILED,
        fields=None,
        completion=completion,
        attempts=retries + 1,
        failure_stage=stage,
        failure_reason=reason,
        semantic_invalid_reason=None,
    )


@contextlib.contextmanager
def _scope(
    attempt_scope: Callable[[int], Any] | None, index: int
) -> Iterator[Any]:
    """Opens the recording context for one attempt, if there is one.

    Args:
        attempt_scope: Opens a context for attempt ``index``, or None.
        index: Zero-based attempt number.

    Yields:
        The recorder, or None when the call is not being recorded.
    """
    if attempt_scope is None:
        yield None
        return
    with attempt_scope(index) as attempt:
        yield attempt


def _record_success(attempt: Any, completion: Completion) -> None:
    """Records a completion that parsed.

    Args:
        attempt: The recorder, or None.
        completion: What came back.
    """
    if attempt is not None:
        attempt.record_success(completion=completion.text, parse_outcome=VALID)


def _record_failure(
    attempt: Any, error: DecodeError, completion: Completion
) -> None:
    """Records a completion that was rejected.

    Args:
        attempt: The recorder, or None.
        error: What rejected it.
        completion: What came back.
    """
    if attempt is not None:
        attempt.record_failure(
            reason=f"{error.stage}: {error.reason}",
            completion=completion.text,
        )


def binomial_upper_bound(
    failures: int, trials: int, confidence: float
) -> float:
    """Returns the one-sided upper bound on an unseen failure rate.

    Zero failures in a hundred tries is not evidence of a zero rate,
    so the bound is the largest rate that would still leave the
    observed count plausible at this confidence.

    Args:
        failures: Failures observed.
        trials: Attempts made.
        confidence: Confidence level, strictly between 0 and 1.

    Returns:
        The upper bound, in ``[0, 1]``.

    Raises:
        ValueError: If ``trials`` is not positive, or ``failures`` lies
            outside it.
    """
    if trials <= 0:
        msg = f"cannot bound a rate over {trials} trials"
        raise ValueError(msg)
    if not 0 <= failures <= trials:
        msg = f"{failures} failures is outside {trials} trials"
        raise ValueError(msg)
    if failures == trials:
        return 1.0

    target = 1.0 - confidence
    low, high = 0.0, 1.0
    for _ in range(_BISECTION_STEPS):
        if high - low < _PROBABILITY_TOLERANCE:
            break
        middle = (low + high) / 2.0
        if _at_most(failures, trials, middle) > target:
            low = middle
        else:
            high = middle
    return high


def _at_most(successes: int, trials: int, rate: float) -> float:
    """Returns the chance of at most ``successes`` at this rate.

    Args:
        successes: The count to bound.
        trials: Attempts made.
        rate: The per-attempt rate.

    Returns:
        The cumulative binomial probability.
    """
    return sum(
        math.comb(trials, index)
        * rate**index
        * (1.0 - rate) ** (trials - index)
        for index in range(successes + 1)
    )


def retry_reserve(
    failures: int, trials: int, decoding: settings_module.Decoding
) -> float:
    """Returns the retry allowance a compute projection must carry.

    Args:
        failures: Fixture failures observed.
        trials: Fixtures run.
        decoding: The decoding settings.

    Returns:
        The larger of the configured floor and the observed bound.
    """
    bound = binomial_upper_bound(failures, trials, decoding.retry_confidence)
    return max(decoding.retry_reserve_floor, bound)


def file_hashes(
    directory: pathlib.Path, names: Sequence[str]
) -> dict[str, str]:
    """Hashes the acquired files that define the artifact.

    Args:
        directory: The downloaded snapshot.
        names: File names to hash, relative to it.

    Returns:
        File name to hex SHA-256.

    Raises:
        StackError: If a named file is absent from the snapshot.
    """
    hashes = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            msg = f"the pinned artifact has no {name}"
            raise StackError(msg)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def adapter_identity(adapter: pathlib.Path | None) -> str:
    """Returns what identifies the adapter in a run's identity.

    Args:
        adapter: Directory holding the adapter, or None for the base
            model on its own.

    Returns:
        The adapter file's hex SHA-256, or ``none``.

    Raises:
        StackError: If the directory holds no adapter file.
    """
    if adapter is None:
        return "none"
    path = adapter / ADAPTER_FILE
    if not path.is_file():
        msg = f"no adapter to load at {path}"
        raise StackError(msg)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_identity(
    settings: settings_module.Settings,
    hashes: Mapping[str, str],
    versions: Mapping[str, str],
    prompt_hash: str,
    schema_hash: str,
    adapter: str = "none",
    assistant_prefix: str = "",
) -> dict[str, Any]:
    """Assembles everything that makes one call reproducible.

    Two runs sharing this identity asked the same question of the same
    weights through the same runtime; two that differ are different
    experiments, however similar their numbers look.

    Args:
        settings: The LLM configuration.
        hashes: Acquired-file hashes from the snapshot.
        versions: Library name to installed version.
        prompt_hash: Hash of the frozen prompt template.
        schema_hash: Hash of the frozen output schema.
        adapter: What identifies the loaded adapter, or ``none``.
        assistant_prefix: Text supplied before the schema takes over.

    Returns:
        The identity payload, with its own hash under ``identity_hash``.
    """
    identity: dict[str, Any] = {
        "model_id": settings.model.id,
        "model_revision": settings.model.revision,
        "upstream_id": settings.model.upstream_id,
        "tokenizer_revision": settings.model.revision,
        "adapter": adapter,
        "assistant_prefix": assistant_prefix,
        "prompt_version": settings.prompt.version,
        "prompt_hash": prompt_hash,
        "schema_hash": schema_hash,
        "max_tokens": settings.decoding.max_tokens,
        "temperature": settings.decoding.temperature,
        "retries": settings.decoding.retries,
        "files": dict(hashes),
        "runtime": dict(versions),
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    identity["identity_hash"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return identity


def runtime_versions() -> dict[str, str]:
    """Returns the installed version of every library in the path.

    Returns:
        Library name to version.

    Raises:
        StackError: If a library in the pinned path is not
            installed, which means this host cannot generate.
    """
    versions = {}
    for name in ("outlines", "outlines-core", "mlx", "mlx-lm", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as error:
            msg = f"the pinned decoding path needs {name}, which is absent"
            raise StackError(msg) from error
    return versions


def snapshot(settings: settings_module.Settings) -> pathlib.Path:
    """Fetches the artifact at its pinned commit.

    Args:
        settings: The LLM configuration.

    Returns:
        The local snapshot directory.

    Raises:
        StackError: If the artifact cannot be fetched.
    """
    # Imported here, not at module scope: a host without the Metal
    # wheels must still be able to import this module.
    import huggingface_hub  # noqa: PLC0415

    try:
        directory = huggingface_hub.snapshot_download(
            settings.model.id, revision=settings.model.revision
        )
    except OSError as error:
        msg = f"cannot fetch {settings.model.id}: {error}"
        raise StackError(msg) from error
    return pathlib.Path(directory)


def load_runtime(
    settings: settings_module.Settings,
    schema: Mapping[str, Any],
    prompt_hash: str,
    schema_hash: str,
    adapter: pathlib.Path | None = None,
) -> Runtime:
    """Assembles the pinned generation path on this host.

    The adapter, when there is one, is loaded onto the same quantised
    weights that trained it and is hashed into the run's identity: two
    runs that differ by an adapter are different experiments.

    Args:
        settings: The LLM configuration.
        schema: The frozen output schema.
        prompt_hash: Hash of the frozen prompt template.
        schema_hash: Hash of the frozen output schema.
        adapter: Directory holding the adapter, or None for the base
            model on its own.

    Returns:
        The loaded runtime and its identity.

    Raises:
        StackError: If the path cannot be assembled on this host.
    """
    versions = runtime_versions()
    adapter_hash = adapter_identity(adapter)
    directory = snapshot(settings)
    try:
        # See snapshot(): the generation libraries are optional here.
        import mlx_lm  # noqa: PLC0415
        import mlx_lm.sample_utils  # noqa: PLC0415
        import outlines  # noqa: PLC0415
        import outlines.types  # noqa: PLC0415
    except ImportError as error:
        msg = f"the pinned decoding path does not import here: {error}"
        raise StackError(msg) from error

    loaded = mlx_lm.load(
        str(directory),
        adapter_path=None if adapter is None else str(adapter),
    )
    model = outlines.from_mlxlm(*loaded)
    generator = outlines.Generator(
        model, outlines.types.JsonSchema(dict(schema))
    )
    processor = getattr(generator, "logits_processor", None)
    if processor is None:
        msg = (
            "the runtime returned a generator that cannot constrain output, "
            "so the schema would be a request rather than a guarantee"
        )
        raise StackError(msg)

    prefix = derive_assistant_prefix(model)
    return Runtime(
        model=model,
        processor=processor,
        sampler=mlx_lm.sample_utils.make_sampler(
            temp=settings.decoding.temperature
        ),
        max_tokens=settings.decoding.max_tokens,
        assistant_prefix=prefix,
        identity=build_identity(
            settings,
            file_hashes(directory, settings.model.hashed_files),
            versions,
            prompt_hash,
            schema_hash,
            adapter_hash,
            prefix,
        ),
    )


def derive_assistant_prefix(model: Any) -> str:
    """Returns what the chat template puts before an answer's first byte.

    A model is trained on whole rendered turns, so whatever the template
    inserts between the generation prompt and the answer is part of what
    the model learned to produce. Constrained decoding starts at the
    answer's first byte, so that text has to be supplied rather than
    demanded from a decoder that is forbidden to emit it.

    It is read back out of the template rather than written down here,
    so a template change cannot leave a stale literal behind.

    Args:
        model: The loaded Outlines model.

    Returns:
        The text between the generation prompt and the answer, empty
        when the template inserts nothing.

    Raises:
        StackError: If the template does not render the generation
            prompt as a prefix of the full turn, which would mean the
            two cannot be compared at all.
    """
    tokenizer = model.mlx_tokenizer
    probe = [
        {"role": SYSTEM_ROLE, "content": _PROBE},
        {"role": USER_ROLE, "content": _PROBE},
    ]
    generation = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True
    )
    rendered = tokenizer.apply_chat_template(
        [*probe, {"role": ASSISTANT_ROLE, "content": _BODY}], tokenize=False
    )
    if not rendered.startswith(generation) or _BODY not in rendered:
        msg = (
            "the chat template does not render its generation prompt as a "
            "prefix of a full turn, so training and inference cannot be "
            "shown to agree"
        )
        raise StackError(msg)
    return rendered[len(generation) : rendered.index(_BODY)]


def generate_completion(
    runtime: Runtime, system: str, prompt: str
) -> Completion:
    """Generates one schema-constrained answer.

    Args:
        runtime: The loaded generation path.
        system: The system instruction.
        prompt: The rendered user prompt.

    Returns:
        What came back, with the call's token counts and rates.

    Raises:
        StackError: If generation produced no response at all.
    """
    # See snapshot(): the generation libraries are optional here.
    import mlx_lm  # noqa: PLC0415
    import outlines.inputs  # noqa: PLC0415

    chat = outlines.inputs.Chat(
        [
            {"role": SYSTEM_ROLE, "content": system},
            {"role": USER_ROLE, "content": prompt},
        ]
    )
    formatted = runtime.model.type_adapter.format_input(chat)
    if not isinstance(formatted, str):
        msg = f"the runtime formatted a prompt as {type(formatted).__name__}"
        raise StackError(msg)
    formatted += runtime.assistant_prefix
    runtime.processor.reset()

    pieces: list[str] = []
    last = None
    started = time.perf_counter()
    for response in mlx_lm.stream_generate(
        runtime.model.model,
        runtime.model.mlx_tokenizer,
        formatted,
        max_tokens=runtime.max_tokens,
        sampler=runtime.sampler,
        logits_processors=[runtime.processor],
    ):
        pieces.append(response.text)
        last = response
    seconds = time.perf_counter() - started

    if last is None:
        msg = "the runtime produced no response"
        raise StackError(msg)
    return Completion(
        text="".join(pieces),
        prompt_tokens=int(last.prompt_tokens),
        completion_tokens=int(last.generation_tokens),
        prefill_tokens_per_second=float(last.prompt_tps),
        decode_tokens_per_second=float(last.generation_tps),
        peak_memory_gib=float(last.peak_memory),
        seconds=seconds,
        finish_reason=str(last.finish_reason),
    )
