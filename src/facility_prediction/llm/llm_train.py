"""Trains one adapter over the pinned quantised base model.

The base weights stay frozen in their four-bit form; only a small set of
adapter parameters receives a gradient, and only the answer tokens
contribute to the loss. What comes out is an adapter file, its exact
parameter names, and the measurements the compute projection needs.

    pilot rows ──> chat records ──> pinned trainer ──> adapter
                                          │              │
                                    timing + memory   parameter names

    validated settings ──> flat trainer config ──> saved beside the run

Nothing is selected here. The saved adapter is the one at the declared
iteration count, because the checkpoint interval is set equal to it:
there is no intermediate checkpoint for a later step to prefer.

Leakage contract: reads rendered training prompts and their answers.
Validation prompts are read for the trainer's own loss monitoring only,
and no holdout row is read at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import pathlib
import subprocess
import sys
import time
import types
from typing import Any

import yaml

from facility_prediction.llm import ladder as ladder_module
from facility_prediction.llm import settings as settings_module

GATE = "the pilot adapter trains and saves at the declared iteration count"
RUN_NAME = "pilot-adapter"

EVIDENCE = "measured"

TRAIN_FILE = "train.jsonl"
VALID_FILE = "valid.jsonl"
CONFIG_FILE = "mlx_config.yaml"
ADAPTER_FILE = "adapters.safetensors"
TRAINING_LOG_FILE = "training.log"
TRAINING_STATUS_FILE = "training_status.json"

SAMPLE_ID = "sample_id"
MESSAGES = "messages"

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"

_HELP_COMMAND = ("-m", "mlx_lm", "lora", "--help")
_HELP_TIMEOUT_SECONDS = 120


class TrainingError(Exception):
    """Raised when the pinned trainer cannot produce an adapter here."""


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    """What one training run measured.

    Attributes:
        seconds: Wall-clock time of the run.
        iters: Minibatch iterations trained.
        train_reports: The trainer's training reports, in order.
        val_reports: The trainer's validation reports, in order.
    """

    seconds: float
    iters: int
    train_reports: list[dict[str, float]]
    val_reports: list[dict[str, float]]

    @property
    def seconds_per_iteration(self) -> float:
        """Returns the measured cost of one iteration.

        Returns:
            Wall-clock seconds divided by iterations trained.
        """
        return self.seconds / self.iters

    @property
    def peak_memory_gb(self) -> float:
        """Returns the highest memory the trainer reported.

        Returns:
            Peak memory in gigabytes, or zero if nothing was reported.
        """
        return max(
            (report["peak_memory"] for report in self.train_reports),
            default=0.0,
        )

    @property
    def trained_tokens(self) -> float:
        """Returns how many tokens the run trained on.

        Returns:
            The last report's cumulative token count, or zero.
        """
        if not self.train_reports:
            return 0.0
        return self.train_reports[-1]["trained_tokens"]


class _Recorder:
    """Collects the trainer's reports and writes readable progress updates."""

    def __init__(
        self,
        total_iterations: int,
        batch_size: int,
        adapter_directory: pathlib.Path,
    ) -> None:
        """Starts a progress log for one training run.

        Args:
            total_iterations: Number of minibatch iterations requested.
            batch_size: Rows in each minibatch.
            adapter_directory: Directory holding the adapter and progress files.
        """
        self._total_iterations = total_iterations
        self._batch_size = batch_size
        self._started = time.perf_counter()
        self._log_path = adapter_directory / TRAINING_LOG_FILE
        self._status_path = adapter_directory / TRAINING_STATUS_FILE
        adapter_directory.mkdir(parents=True, exist_ok=True)
        self.train_reports: list[dict[str, float]] = []
        self.val_reports: list[dict[str, float]] = []
        self._log_path.write_text("", encoding="utf-8")
        self._write_status({"iteration": 0.0})
        self._write_log(
            "started total_iterations=%d batch_size=%d",
            total_iterations,
            batch_size,
        )

    def _write_log(self, message: str, *args: object) -> None:
        """Appends one flushed, human-readable progress line.

        Args:
            message: Logging pattern for the line.
            *args: Values interpolated into the logging pattern.
        """
        elapsed = time.perf_counter() - self._started
        line = f"elapsed_seconds={elapsed:.1f} " + message % args
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()

    def _write_status(self, report: Mapping[str, float]) -> None:
        """Writes the latest exact progress state for polling.

        Args:
            report: Trainer report that includes the completed iteration.
        """
        iteration = int(report["iteration"])
        elapsed = time.perf_counter() - self._started
        remaining = self._total_iterations - iteration
        eta = elapsed * remaining / iteration if iteration else None
        status = {
            "iteration": iteration,
            "total_iterations": self._total_iterations,
            "row_passes_completed": iteration * self._batch_size,
            "percent_complete": iteration / self._total_iterations * 100,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": eta,
        }
        temporary = self._status_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self._status_path)

    def on_train_loss_report(self, train_info: dict[str, Any]) -> None:
        """Records one training report.

        Args:
            train_info: The trainer's report.
        """
        report = {key: float(value) for key, value in train_info.items()}
        self.train_reports.append(report)
        self._write_status(report)
        self._write_log(
            "progress iteration=%d/%d rows_processed=%d percent=%.1f "
            "eta_seconds=%.1f",
            int(report["iteration"]),
            self._total_iterations,
            int(report["iteration"]) * self._batch_size,
            report["iteration"] / self._total_iterations * 100,
            (time.perf_counter() - self._started)
            * (self._total_iterations - report["iteration"])
            / report["iteration"],
        )

    def on_val_loss_report(self, val_info: dict[str, Any]) -> None:
        """Records one validation report.

        Args:
            val_info: The trainer's report.
        """
        self.val_reports.append(
            {key: float(value) for key, value in val_info.items()}
        )

    def complete(self) -> None:
        """Records that the trainer returned normally."""
        self._write_log("completed")


def declared_iters(settings: settings_module.Settings) -> int:
    """Returns the pilot's iteration count, having checked it holds.

    Args:
        settings: The LLM configuration.

    Returns:
        The declared iteration count.

    Raises:
        TrainingError: If the declared count is not what the pilot's
            rows and passes imply, or does not divide into whole
            optimizer updates.
    """
    pilot, tuning = settings.pilot, settings.tuning
    implied = ladder_module.iters_for(
        pilot.train_rows, pilot.epoch_equivalent, tuning.batch_size
    )
    if implied != pilot.iters:
        msg = (
            f"the pilot declares {pilot.iters} iterations, but "
            f"{pilot.train_rows} rows over {pilot.epoch_equivalent} passes "
            f"is {implied}"
        )
        raise TrainingError(msg)
    if pilot.iters % tuning.grad_accumulation_steps:
        msg = (
            f"{pilot.iters} iterations do not divide into whole optimizer "
            f"updates at accumulation {tuning.grad_accumulation_steps}"
        )
        raise TrainingError(msg)
    return pilot.iters


def chat_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turns prompt rows into the trainer's conversation records.

    The system instruction, the prompt, and the answer are carried
    through unchanged, so the model is trained on the exact text it will
    be asked with.

    Args:
        rows: The prompt rows to convert, in order.

    Returns:
        One conversation record per row, in the same order.

    Raises:
        TrainingError: If a row carries no answer to train on.
    """
    records = []
    for row in rows:
        if not row.get("target"):
            msg = f"prompt row {row.get(SAMPLE_ID)!r} has no answer to train on"
            raise TrainingError(msg)
        records.append(
            {
                MESSAGES: [
                    {"role": SYSTEM_ROLE, "content": str(row["system"])},
                    {"role": USER_ROLE, "content": str(row["prompt"])},
                    {"role": ASSISTANT_ROLE, "content": str(row["target"])},
                ]
            }
        )
    return records


def write_jsonl(
    records: Sequence[Mapping[str, Any]], path: pathlib.Path
) -> str:
    """Writes conversation records and returns the file's content hash.

    Args:
        records: The records to write, in order.
        path: Destination file; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat_config(
    settings: settings_module.Settings,
    model_path: str,
    data_directory: pathlib.Path,
    adapter_directory: pathlib.Path,
    iters: int,
) -> dict[str, Any]:
    """Builds the pinned trainer's own configuration.

    The checkpoint interval is set equal to the iteration count, so the
    only adapter this run saves is the one at the declared end of it.

    Args:
        settings: The LLM configuration.
        model_path: Local path of the pinned quantised artifact.
        data_directory: Directory holding the trainer's JSONL files.
        adapter_directory: Where the adapter is written.
        iters: Minibatch iterations to train.

    Returns:
        The trainer's configuration, flat and ready to serialise.
    """
    tuning = settings.tuning
    return {
        "model": model_path,
        "train": True,
        "test": False,
        "data": str(data_directory),
        "adapter_path": str(adapter_directory),
        "fine_tune_type": tuning.fine_tune_type,
        "optimizer": tuning.optimizer,
        "seed": tuning.training_seed,
        "num_layers": tuning.num_layers,
        "batch_size": tuning.batch_size,
        "grad_accumulation_steps": tuning.grad_accumulation_steps,
        "iters": iters,
        "save_every": iters,
        "val_batches": tuning.val_batches,
        "learning_rate": tuning.learning_rate,
        "steps_per_report": tuning.steps_per_report,
        "steps_per_eval": tuning.steps_per_eval,
        "max_seq_length": tuning.max_seq_length,
        "grad_checkpoint": tuning.grad_checkpoint,
        "mask_prompt": tuning.mask_prompt,
        "lora_parameters": tuning.lora_parameters.model_dump(),
    }


def write_config(flat: Mapping[str, Any], path: pathlib.Path) -> str:
    """Writes the trainer's configuration beside its adapter.

    Args:
        flat: The trainer's configuration.
        path: Destination YAML; parent directories are created.

    Returns:
        The hex SHA-256 of the bytes written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(flat), handle, sort_keys=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_line(flat: Mapping[str, Any], config_path: pathlib.Path) -> str:
    """Returns the equivalent command line for the record.

    Args:
        flat: The trainer's configuration.
        config_path: Where that configuration was written.

    Returns:
        The command a reader can run to repeat this training.
    """
    parts = [
        "mlx_lm.lora",
        f"--config {config_path}",
        f"--iters {flat['iters']}",
        f"--batch-size {flat['batch_size']}",
        f"--grad-accumulation-steps {flat['grad_accumulation_steps']}",
        f"--num-layers {flat['num_layers']}",
    ]
    if flat["mask_prompt"]:
        parts.append("--mask-prompt")
    return " ".join(parts)


def capture_help() -> dict[str, str]:
    """Captures the pinned trainer's own documented controls.

    Returns:
        The command run, its output, and that output's hash.

    Raises:
        TrainingError: If the pinned trainer cannot be asked.
    """
    command = [sys.executable, *_HELP_COMMAND]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=True,
            text=True,
            timeout=_HELP_TIMEOUT_SECONDS,
        )
    except (subprocess.SubprocessError, OSError) as error:
        msg = f"cannot capture the pinned trainer's controls: {error}"
        raise TrainingError(msg) from error
    text = completed.stdout
    return {
        "command": " ".join(command),
        "text": text,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def train_adapter(flat: Mapping[str, Any]) -> TrainingResult:
    """Runs the pinned trainer and measures what it cost.

    The trainer is driven through its own training entry point rather
    than its command line, because the command line discards the
    callback the timing and memory measurements come from. The
    configuration passed is the one written beside the adapter.

    Args:
        flat: The trainer's configuration.

    Returns:
        What the run measured.

    Raises:
        TrainingError: If the pinned trainer is absent, or cannot train
            on this host.
    """
    # Imported here, not at module scope: a host without the Metal
    # wheels must still be able to import this module.
    try:
        import mlx_lm.lora  # noqa: PLC0415
        import mlx_lm.tuner.datasets  # noqa: PLC0415
        import mlx_lm.utils  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as error:
        msg = f"the pinned training path does not import here: {error}"
        raise TrainingError(msg) from error

    args = types.SimpleNamespace(**{**mlx_lm.lora.CONFIG_DEFAULTS, **flat})
    recorder = _Recorder(
        int(args.iters),
        int(args.batch_size),
        pathlib.Path(args.adapter_path),
    )
    started = time.perf_counter()
    try:
        # The trainer shuffles its batches through numpy's legacy global
        # generator, so seeding a fresh Generator would leave the batch
        # order unseeded.
        np.random.seed(args.seed)  # noqa: NPY002
        # The loaded triple and the trainer's callback protocol are
        # duck-typed here: this is the pinned release's own interface,
        # not one this package declares.
        loaded: Any = mlx_lm.utils.load(
            args.model, tokenizer_config={"trust_remote_code": True}
        )
        datasets: Any = mlx_lm.tuner.datasets.load_dataset(args, loaded[1])
        callback: Any = recorder
        mlx_lm.lora.train_model(
            args, loaded[0], datasets[0], datasets[1], callback
        )
    except (RuntimeError, ValueError, MemoryError, OSError) as error:
        msg = f"the pilot adapter did not train on this host: {error}"
        raise TrainingError(msg) from error
    recorder.complete()
    return TrainingResult(
        seconds=time.perf_counter() - started,
        iters=int(args.iters),
        train_reports=recorder.train_reports,
        val_reports=recorder.val_reports,
    )


def adapter_parameters(path: pathlib.Path) -> dict[str, Any]:
    """Reads back which parameters the saved adapter actually trained.

    The adapted set is read from the file rather than asserted from the
    settings, so the record describes what was trained rather than what
    was requested.

    Args:
        path: The saved adapter file.

    Returns:
        Its parameter count, adapted block count, projection names, and
        content hash.

    Raises:
        TrainingError: If the adapter is absent or unreadable.
    """
    if not path.is_file():
        msg = f"the trainer saved no adapter at {path}"
        raise TrainingError(msg)
    try:
        import mlx.core as mx  # noqa: PLC0415
    except ImportError as error:
        msg = f"cannot read the saved adapter here: {error}"
        raise TrainingError(msg) from error

    weights = mx.load(str(path))
    return describe_parameters(
        sorted(weights),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def describe_parameters(names: Sequence[str], digest: str) -> dict[str, Any]:
    """Summarises an adapter's parameter names.

    Args:
        names: The adapter's parameter names.
        digest: The adapter file's content hash.

    Returns:
        The parameter count, the adapted blocks, the projections those
        parameters sit on, and the file's hash.
    """
    blocks: set[str] = set()
    projections: set[str] = set()
    for name in names:
        for part in name.split("."):
            if part.isdigit():
                blocks.add(part)
            if part.endswith("_proj"):
                projections.add(part)
    return {
        "parameters": len(names),
        "adapted_blocks": len(blocks),
        "projections": sorted(projections),
        "names": list(names),
        "sha256": digest,
    }


def build_record(
    result: TrainingResult,
    flat: Mapping[str, Any],
    adapter: Mapping[str, Any],
    draw: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Assembles the record of what the pilot trained and what it cost.

    Timing and loss are kept in separate blocks because the compute
    projection reads the first and must never read the second.

    Args:
        result: What the run measured.
        flat: The trainer's configuration.
        adapter: What the saved adapter contains.
        draw: How the pilot rows were drawn.
        provenance: The model, the data files, and their hashes.

    Returns:
        The record payload, ready to serialise.
    """
    return {
        "evidence": EVIDENCE,
        "gate": GATE,
        "arithmetic": {
            "train_rows": draw["train_rows"],
            "epoch_equivalent": draw["epoch_equivalent"],
            "iters": flat["iters"],
            "batch_size": flat["batch_size"],
            "grad_accumulation_steps": flat["grad_accumulation_steps"],
            "optimizer_updates": flat["iters"]
            // flat["grad_accumulation_steps"],
        },
        "timing": {
            "seconds": result.seconds,
            "seconds_per_iteration": result.seconds_per_iteration,
            "trained_tokens": result.trained_tokens,
            "peak_memory_gb": result.peak_memory_gb,
        },
        "loss": {
            "initial_validation": (
                result.val_reports[0]["val_loss"]
                if result.val_reports
                else None
            ),
            "final_validation": (
                result.val_reports[-1]["val_loss"]
                if result.val_reports
                else None
            ),
            "validation_reports": result.val_reports,
            "training_reports": result.train_reports,
        },
        "adapter": dict(adapter),
        "draw": dict(draw),
        "config": dict(flat),
        "provenance": dict(provenance),
    }


def check_saved(record: Mapping[str, Any]) -> None:
    """Refuses a run that did not train what it declared.

    Args:
        record: The assembled record.

    Raises:
        TrainingError: If the adapter holds no parameters, or the run
            trained a different number of iterations than declared.
    """
    if not record["adapter"]["parameters"]:
        msg = "the saved adapter holds no parameters, so nothing was trained"
        raise TrainingError(msg)
    declared = record["arithmetic"]["iters"]
    reported = [
        report["iteration"]
        for report in record["loss"]["training_reports"]
        if report
    ]
    if reported and max(reported) != declared:
        msg = (
            f"the run reported {max(reported):.0f} iterations against a "
            f"declared {declared}"
        )
        raise TrainingError(msg)


def run_params(record: Mapping[str, Any]) -> dict[str, str]:
    """Returns the run parameters for this training run.

    Args:
        record: The assembled record.

    Returns:
        Parameter name to value, as text.
    """
    params = {
        f"config_{key}": str(value)
        for key, value in record["config"].items()
        if not isinstance(value, dict)
    }
    params.update(
        {
            f"lora_{key}": str(value)
            for key, value in record["config"]["lora_parameters"].items()
        }
    )
    params["evidence"] = str(record["evidence"])
    params["prevalence_representative"] = str(
        record["draw"]["shares"]["prevalence_representative"]
    )
    params["adapter_sha256"] = str(record["adapter"]["sha256"])
    return params


def run_metrics(record: Mapping[str, Any]) -> dict[str, float]:
    """Returns the run metrics for this training run.

    Args:
        record: The assembled record.

    Returns:
        Metric name to value.
    """
    metrics = {
        f"timing_{key}": float(value) for key, value in record["timing"].items()
    }
    metrics.update(
        {
            "adapter_parameters": float(record["adapter"]["parameters"]),
            "adapted_blocks": float(record["adapter"]["adapted_blocks"]),
            "train_rows": float(record["draw"]["train_rows"]),
            "labels_drawn": float(len(record["draw"]["support"])),
            "labels_unrepresented": float(len(record["draw"]["unrepresented"])),
            "max_absolute_share_deviation": float(
                record["draw"]["shares"]["max_absolute_share_deviation"]
            ),
        }
    )
    for name in ("initial_validation", "final_validation"):
        value = record["loss"][name]
        if value is not None:
            metrics[f"loss_{name}"] = float(value)
    return metrics
