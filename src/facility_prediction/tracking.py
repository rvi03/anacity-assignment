"""The experiment record — a mirror of the pipeline, never its source.

Every run either track opens is written here with the tag that says who
opened it, so one experiment answers "which run produced this number"
for both model families at once:

    fit / score / stop ──> run   (tagged `track`, params, metrics,
                                  artifacts)
    prompt text        ──> registered version + alias
    generation call    ──> one trace, one nested span per attempt

Three properties hold, and the rest of the module exists to keep them:

- **A stopped branch still logs.** :func:`log_gate_stop` records the
  gate and the reason as a failed run. Silence is never the record of a
  branch that stopped.
- **A failed or retried call is a span with its reason**, not a gap in
  the trace.
- **The record is droppable.** Run ids, timestamps, and trace ids are
  nondeterministic, so the tracking database is absent from the
  comparison set :func:`verify_comparison_databases` returns, and every
  call here degrades to a no-op when tracking is disabled. Deleting the
  whole record leaves every deliverable reproducible.

The server address comes from the environment variable named in the
configuration, read from the same `.env` the service stack reads.

Leakage contract: this module observes no event data. It records what a
caller hands it and applies no as-of bound of its own.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
import dataclasses
import hashlib
import logging
import os
import pathlib
from typing import Any

import mlflow
import mlflow.entities
import mlflow.exceptions
import mlflow.genai

from facility_prediction import config as config_module
from facility_prediction.data import storage

_LOGGER = logging.getLogger(__name__)

TRACK_TAG = "track"
GATE_TAG = "gate"
GATE_REASON_TAG = "gate_reason"
OUTCOME_TAG = "outcome"
STOPPED_AT_GATE = "stopped_at_gate"

PROMPT_URI_SCHEME = "prompts:/"
MIRROR_DATABASE = "mlflow"

_FAILED_STATUS = "FAILED"
_GENERATION_SPAN = "LLM"
_ATTEMPT_SPAN = "CHAT_MODEL"


class TrackingError(Exception):
    """Raised when the tracking server cannot be reached or used."""


def verify_comparison_databases(
    config: config_module.Config,
) -> tuple[str, ...]:
    """Return the databases a reproduction check may compare.

    The store of record is in the set; the tracking mirror is not. Run
    ids, timestamps, and trace ids differ on every run.

    Args:
        config: Validated configuration.

    Returns:
        Database names, in comparison order.
    """
    return (config.storage.database,)


def tracking_uri(config: config_module.Config) -> str:
    """Return the tracking server URI named by the configuration.

    Args:
        config: Validated configuration.

    Returns:
        The server URI, for example ``http://localhost:5000``.

    Raises:
        TrackingError: If the named environment variable is unset.
    """
    storage.load_env_file()
    name = config.tracking.tracking_uri_env
    uri = os.environ.get(name)
    if not uri:
        msg = (
            f"{name} is unset; copy .env.example to .env or export it "
            "before recording a run"
        )
        raise TrackingError(msg)
    return uri


def configure(config: config_module.Config) -> str:
    """Point MLflow at the configured server and experiment.

    Args:
        config: Validated configuration.

    Returns:
        The experiment id runs will be written to.

    Raises:
        TrackingError: If the server is unreachable or refuses the
            experiment.
    """
    mlflow.set_tracking_uri(tracking_uri(config))
    try:
        experiment = mlflow.set_experiment(config.tracking.experiment)
    except mlflow.exceptions.MlflowException as error:
        msg = (
            f"cannot reach the tracking server at {tracking_uri(config)}: "
            f"{error}"
        )
        raise TrackingError(msg) from error
    return str(experiment.experiment_id)


def is_reachable(config: config_module.Config) -> bool:
    """Report whether the configured tracking server answers.

    Args:
        config: Validated configuration.

    Returns:
        True when the experiment could be resolved on the server.
    """
    try:
        configure(config)
    except TrackingError:
        return False
    return True


@dataclasses.dataclass(frozen=True)
class RunLogger:
    """Handle to one open run.

    Every method is a no-op when tracking is disabled, so no caller
    branches on whether a server is running.

    Attributes:
        run_id: The open run's id, or None when tracking is disabled.
    """

    run_id: str | None

    @property
    def recording(self) -> bool:
        """Return whether this handle writes to a server."""
        return self.run_id is not None

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record run inputs, rendered as strings.

        Args:
            params: Parameter name to value.
        """
        if not self.recording:
            return
        mlflow.log_params({key: str(value) for key, value in params.items()})

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Record run results.

        Args:
            metrics: Metric name to value.
        """
        if not self.recording:
            return
        mlflow.log_metrics(dict(metrics))

    def log_artifact(self, path: pathlib.Path) -> None:
        """Attach a produced file to the run.

        Args:
            path: File to upload.

        Raises:
            TrackingError: If the file does not exist.
        """
        if not self.recording:
            return
        if not path.is_file():
            msg = f"cannot log missing artifact {path}"
            raise TrackingError(msg)
        mlflow.log_artifact(str(path))

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Add tags to the open run.

        Args:
            tags: Tag name to value.
        """
        if not self.recording:
            return
        mlflow.set_tags(dict(tags))


@contextlib.contextmanager
def run(
    config: config_module.Config,
    *,
    name: str,
    track: str,
    tags: Mapping[str, str] | None = None,
) -> Iterator[RunLogger]:
    """Open one run, tagged with the track that produced it.

    Args:
        config: Validated configuration.
        name: Run name, as it appears in the experiment list.
        track: Which track opened it, for example ``baseline``.
        tags: Further tags to set on the run.

    Yields:
        A handle for logging into the open run.

    Raises:
        TrackingError: If tracking is enabled but the run cannot be
            opened.
    """
    if not config.tracking.enabled:
        _LOGGER.info("tracking disabled; run %r is not recorded", name)
        yield RunLogger(run_id=None)
        return

    configure(config)
    run_tags: dict[str, Any] = {TRACK_TAG: track}
    run_tags.update(tags or {})
    try:
        with mlflow.start_run(run_name=name, tags=run_tags) as active:
            yield RunLogger(run_id=active.info.run_id)
    except mlflow.exceptions.MlflowException as error:
        msg = f"cannot record run {name!r}: {error}"
        raise TrackingError(msg) from error


def log_gate_stop(
    config: config_module.Config,
    *,
    gate: str,
    reason: str,
    track: str,
) -> str | None:
    """Record a branch that stopped at a gate as a failed run.

    A stop is an outcome, so it gets a run carrying which gate stopped
    it and why.

    Args:
        config: Validated configuration.
        gate: The gate that was not met.
        reason: Why it was not met, in one line.
        track: Which track stopped.

    Returns:
        The run id, or None when tracking is disabled.

    Raises:
        TrackingError: If tracking is enabled but the run cannot be
            opened.
    """
    if not config.tracking.enabled:
        _LOGGER.warning(
            "tracking disabled; gate stop %r (%s) is not recorded", gate, reason
        )
        return None

    configure(config)
    tags = {
        TRACK_TAG: track,
        GATE_TAG: gate,
        GATE_REASON_TAG: reason,
        OUTCOME_TAG: STOPPED_AT_GATE,
    }
    try:
        active = mlflow.start_run(run_name=f"gate-stop-{gate}", tags=tags)
        run_id = str(active.info.run_id)
    except mlflow.exceptions.MlflowException as error:
        msg = f"cannot record the stop at gate {gate!r}: {error}"
        raise TrackingError(msg) from error
    mlflow.end_run(status=_FAILED_STATUS)
    _LOGGER.warning("stopped at gate %s: %s", gate, reason)
    return run_id


def prompt_hash(template: str) -> str:
    """Return the hash that identifies a prompt template.

    The committed manifest carries this hash, so an offline replay can
    prove it used the same text without reaching a server.

    Args:
        template: The prompt template, exactly as sent.

    Returns:
        Hexadecimal SHA-256 of the UTF-8 template.
    """
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class PromptRecord:
    """One registered prompt version, as the manifest records it.

    Attributes:
        name: Registered prompt name.
        version: Version number the registry assigned.
        alias: Alias pointing at this version, when one does.
        template_hash: Hash of the template text.
    """

    name: str
    version: int
    alias: str | None
    template_hash: str

    @property
    def uri(self) -> str:
        """Return the registry URI of this exact version."""
        return f"{PROMPT_URI_SCHEME}{self.name}/{self.version}"


def register_model(
    config: config_module.Config,
    *,
    name: str,
    artifact: pathlib.Path,
    run_id: str | None,
    tags: Mapping[str, str] | None = None,
) -> str | None:
    """Register one produced model file as a named version.

    Registration is what makes "the frozen model" a resolvable thing
    rather than a file someone remembers the path of. The artifact is
    logged into the run first, so the version points at bytes the run
    actually produced and not at whatever is on disk later.

    Args:
        config: Validated configuration.
        name: Registered model name.
        artifact: The model file to attach.
        run_id: The run the artifact belongs to.
        tags: Tags to set on the version, for example ``frozen``.

    Returns:
        The version string, or None when tracking is disabled or the
        run was never opened.

    Raises:
        TrackingError: If the artifact is missing or registration fails.
    """
    if not config.tracking.enabled or run_id is None:
        _LOGGER.info("tracking disabled; model %r is not registered", name)
        return None
    if not artifact.is_file():
        msg = f"cannot register missing artifact {artifact}"
        raise TrackingError(msg)

    configure(config)
    try:
        client = mlflow.MlflowClient()
        source = f"runs:/{run_id}/{artifact.name}"
        # Already registered is the normal case on a re-run; a second
        # run adds a version to the same name rather than failing.
        with contextlib.suppress(mlflow.exceptions.MlflowException):
            client.create_registered_model(name)
        version = client.create_model_version(
            name=name, source=source, run_id=run_id
        )
        for key, value in (tags or {}).items():
            client.set_model_version_tag(name, version.version, key, value)
    except mlflow.exceptions.MlflowException as error:
        msg = f"cannot register model {name!r}: {error}"
        raise TrackingError(msg) from error
    return str(version.version)


def registered_versions(
    config: config_module.Config, name: str
) -> tuple[str, ...]:
    """Return every version registered under one model name.

    Args:
        config: Validated configuration.
        name: Registered model name.

    Returns:
        Version strings, newest first; empty when tracking is disabled
        or nothing is registered under that name.
    """
    if not config.tracking.enabled:
        return ()
    configure(config)
    try:
        client = mlflow.MlflowClient()
        found = client.search_model_versions(f"name='{name}'")
    except mlflow.exceptions.MlflowException:
        return ()
    return tuple(str(item.version) for item in found)


def register_prompt(
    config: config_module.Config,
    *,
    name: str,
    template: str,
    commit_message: str | None = None,
) -> PromptRecord:
    """Register a prompt template as a new version.

    Args:
        config: Validated configuration.
        name: Registered prompt name.
        template: The template text.
        commit_message: What changed in this version.

    Returns:
        The registered version, with the hash the manifest records.

    Raises:
        TrackingError: If tracking is disabled or the registry refuses
            the version.
    """
    _require_enabled(config, "register a prompt")
    configure(config)
    try:
        version = mlflow.genai.register_prompt(
            name=name, template=template, commit_message=commit_message
        )
    except mlflow.exceptions.MlflowException as error:
        msg = f"cannot register prompt {name!r}: {error}"
        raise TrackingError(msg) from error
    return PromptRecord(
        name=name,
        version=int(version.version),
        alias=None,
        template_hash=prompt_hash(template),
    )


def set_prompt_alias(
    config: config_module.Config,
    *,
    name: str,
    alias: str,
    version: int,
) -> None:
    """Point an alias at a registered prompt version.

    Moving an alias is how a rollback happens: the live version changes
    without a code edit.

    Args:
        config: Validated configuration.
        name: Registered prompt name.
        alias: Alias to move, for example ``champion``.
        version: Version the alias should name.

    Raises:
        TrackingError: If tracking is disabled or the registry refuses
            the move.
    """
    _require_enabled(config, "move a prompt alias")
    configure(config)
    try:
        mlflow.genai.set_prompt_alias(name=name, alias=alias, version=version)
    except mlflow.exceptions.MlflowException as error:
        msg = f"cannot alias prompt {name!r} as {alias!r}: {error}"
        raise TrackingError(msg) from error


def resolve_prompt(
    config: config_module.Config, *, name: str, alias: str | None = None
) -> PromptRecord:
    """Resolve the live prompt version behind an alias.

    Args:
        config: Validated configuration.
        name: Registered prompt name.
        alias: Alias to resolve; defaults to the configured one.

    Returns:
        The version the alias names, with its template hash.

    Raises:
        TrackingError: If tracking is disabled or nothing resolves.
    """
    _require_enabled(config, "resolve a prompt")
    configure(config)
    resolved = alias or config.tracking.prompt_alias
    try:
        version = mlflow.genai.load_prompt(
            f"{PROMPT_URI_SCHEME}{name}@{resolved}"
        )
    except mlflow.exceptions.MlflowException as error:
        msg = f"cannot resolve prompt {name!r} at alias {resolved!r}: {error}"
        raise TrackingError(msg) from error
    return PromptRecord(
        name=name,
        version=int(version.version),
        alias=resolved,
        template_hash=prompt_hash(version.template),
    )


class Attempt:
    """One attempt at a generation call, as a span in its trace.

    An attempt records what came back and whether it parsed. A failure
    or a retry records why.
    """

    def __init__(self, span: mlflow.entities.LiveSpan | None) -> None:
        """Wrap the attempt's span.

        Args:
            span: The open span, or None when tracing is off.
        """
        self._span = span

    def record_success(self, *, completion: str, parse_outcome: str) -> None:
        """Record a completion that came back and how it parsed.

        Args:
            completion: The raw text the model returned.
            parse_outcome: How schema parsing ended, for example
                ``valid``.
        """
        if self._span is None:
            return
        self._span.set_outputs({"completion": completion})
        self._span.set_attribute("parse_outcome", parse_outcome)

    def record_failure(
        self, *, reason: str, completion: str | None = None
    ) -> None:
        """Record why this attempt failed or had to be retried.

        Args:
            reason: What went wrong, in one line.
            completion: The raw text that came back, when any did.
        """
        if self._span is None:
            return
        if completion is not None:
            self._span.set_outputs({"completion": completion})
        self._span.set_attribute("failure_reason", reason)
        self._span.set_status(mlflow.entities.SpanStatusCode.ERROR)


class GenerationTrace:
    """The root span of one generation call.

    Attributes:
        trace_id: The trace's id, or None when tracing is off.
    """

    def __init__(self, span: mlflow.entities.LiveSpan | None) -> None:
        """Wrap the call's root span.

        Args:
            span: The open root span, or None when tracing is off.
        """
        self._span = span
        self.trace_id: str | None = None if span is None else span.trace_id

    @contextlib.contextmanager
    def attempt(self, index: int, *, prompt: str) -> Iterator[Attempt]:
        """Open a nested span for one attempt at this call.

        Args:
            index: Zero-based attempt number; anything above zero is a
                retry.
            prompt: The prompt text as rendered for this attempt.

        Yields:
            The attempt, for recording what came back.
        """
        if self._span is None:
            yield Attempt(span=None)
            return
        with mlflow.start_span(
            name=f"attempt-{index}", span_type=_ATTEMPT_SPAN
        ) as span:
            span.set_inputs({"prompt": prompt})
            span.set_attribute("attempt", index)
            span.set_attribute("is_retry", index > 0)
            yield Attempt(span=span)


@contextlib.contextmanager
def trace_generation(
    config: config_module.Config,
    *,
    track: str,
    model: str,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[GenerationTrace]:
    """Open one trace for one generation call.

    The root span carries what identifies the call — the model, the
    adapter revision, the decoding settings — and each attempt is a
    nested span beneath it.

    Args:
        config: Validated configuration.
        track: Which track is generating.
        model: Model identifier, including its revision.
        attributes: Further call identity, such as the adapter revision
            and the decoding settings.

    Yields:
        The open trace, for opening attempt spans.
    """
    if not config.tracking.enabled:
        _LOGGER.info("tracking disabled; generation call is not traced")
        yield GenerationTrace(span=None)
        return

    configure(config)
    with mlflow.start_span(
        name="generation", span_type=_GENERATION_SPAN
    ) as span:
        span.set_attribute(TRACK_TAG, track)
        span.set_attribute("model", model)
        for key, value in (attributes or {}).items():
            span.set_attribute(key, value)
        yield GenerationTrace(span=span)


def flush_traces() -> None:
    """Wait for trace logging to reach the server.

    Traces are written asynchronously. Anything that reads a trace back
    calls this first.
    """
    mlflow.flush_trace_async_logging()


def _require_enabled(config: config_module.Config, action: str) -> None:
    """Raise unless tracking is enabled.

    Runs, traces, and metrics degrade to no-ops when tracking is off.
    The prompt registry cannot: a caller asking which version is live
    needs an answer or an error.

    Args:
        config: Validated configuration.
        action: What the caller was trying to do, for the message.

    Raises:
        TrackingError: If tracking is disabled.
    """
    if not config.tracking.enabled:
        msg = f"tracking is disabled; cannot {action}"
        raise TrackingError(msg)
