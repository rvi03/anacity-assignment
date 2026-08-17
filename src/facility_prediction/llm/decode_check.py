"""Proves the pinned decoding path works before anything is scored.

A stack that has never generated is a plan, not a capability. This
module draws a seeded sample of held-out prompts, runs every one of
them through the real path, and records what came back: how many
parsed, how many needed the retry, how long each took, how fast the
model read and wrote, and how much memory it reached.

    validation prompts ─seeded draw─> fixtures ─> decode ─> record
                                          │                   │
                                          └── repeat a few ───┘
                                              byte for byte

Nothing here selects, tunes, or scores. The prompts are a sample of the
validation split, used only to exercise the runtime; the holdout is
untouched.

Leakage contract: reads rendered prompts and split labels only. No
target is read, and the answers are not compared with one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from facility_prediction.llm import constrained_decode
from facility_prediction.llm import settings as settings_module

GATE = constrained_decode.GATE
RUN_NAME = constrained_decode.RUN_NAME

SAMPLE_ID = "sample_id"
PERCENTILES = (50, 90, 99)

_EVIDENCE = "measured"


class FixtureError(Exception):
    """Raised when the pinned stack does not clear its fixtures."""


@dataclasses.dataclass(frozen=True)
class Fixture:
    """One prompt the runtime is checked against.

    Attributes:
        sample_id: Which sample it was rendered from.
        system: The system instruction.
        prompt: The rendered user prompt.
    """

    sample_id: str
    system: str
    prompt: str


@dataclasses.dataclass(frozen=True)
class FixtureResult:
    """What one fixture produced.

    Attributes:
        fixture: The prompt that was run.
        outcome: What came back, valid or typed failure.
        repeated_text: The second decode's raw text, or None when this
            fixture was not repeated.
    """

    fixture: Fixture
    outcome: constrained_decode.Outcome
    repeated_text: str | None

    @property
    def repeatable(self) -> bool | None:
        """Return whether the repeat matched, or None if not repeated."""
        if self.repeated_text is None:
            return None
        return self.repeated_text == self.outcome.completion.text


def read_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    """Reads a rendered prompt file.

    Args:
        path: The JSONL file to read.

    Returns:
        One mapping per line, in file order.

    Raises:
        FixtureError: If the file does not exist.
    """
    if not path.is_file():
        msg = f"cannot draw fixtures: {path} does not exist"
        raise FixtureError(msg)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_fixtures(
    rows: Sequence[Mapping[str, Any]], count: int, seed: int
) -> list[Fixture]:
    """Draws the fixture sample.

    The draw is seeded and the result is ordered by identifier, so the
    same configuration checks the same prompts on any host.

    Args:
        rows: The rendered prompt rows to draw from.
        count: How many to draw.
        seed: The configured seed.

    Returns:
        The drawn fixtures, ordered by sample identifier.

    Raises:
        FixtureError: If there are fewer rows than the draw needs.
    """
    if len(rows) < count:
        msg = f"cannot draw {count} fixtures from {len(rows)} rendered prompts"
        raise FixtureError(msg)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(rows), size=count, replace=False)
    drawn = [rows[int(index)] for index in chosen]
    drawn.sort(key=lambda row: str(row[SAMPLE_ID]))
    return [
        Fixture(
            sample_id=str(row[SAMPLE_ID]),
            system=str(row["system"]),
            prompt=str(row["prompt"]),
        )
        for row in drawn
    ]


def run_fixtures(
    fixtures: Sequence[Fixture],
    decode: Callable[[Fixture], constrained_decode.Outcome],
    repeats: int,
) -> list[FixtureResult]:
    """Runs every fixture, repeating the first few.

    A repeat is what turns "greedy" from a setting into evidence: the
    same prompt through the same weights must give the same bytes.

    Args:
        fixtures: The prompts to run, in order.
        decode: Produces one outcome for one fixture.
        repeats: How many of the leading fixtures to decode twice.

    Returns:
        One result per fixture, in the same order.
    """
    results = []
    for position, fixture in enumerate(fixtures):
        outcome = decode(fixture)
        repeated = (
            decode(fixture).completion.text if position < repeats else None
        )
        results.append(
            FixtureResult(
                fixture=fixture, outcome=outcome, repeated_text=repeated
            )
        )
    return results


def counts(results: Sequence[FixtureResult]) -> dict[str, int]:
    """Returns how the fixtures ended.

    Args:
        results: One result per fixture.

    Returns:
        Count name to value.
    """
    repeated = [
        result for result in results if result.repeated_text is not None
    ]
    return {
        "fixtures": len(results),
        "valid": sum(
            result.outcome.status == constrained_decode.VALID
            for result in results
        ),
        "failed": sum(
            result.outcome.status == constrained_decode.FAILED
            for result in results
        ),
        "retried": sum(result.outcome.attempts > 1 for result in results),
        "repeated": len(repeated),
        "repeated_identical": sum(bool(item.repeatable) for item in repeated),
        "semantic_invalid": sum(
            result.outcome.semantic_invalid_reason is not None
            for result in results
        ),
    }


def failures(results: Sequence[FixtureResult]) -> list[dict[str, str]]:
    """Returns one entry per fixture that never parsed.

    Args:
        results: One result per fixture.

    Returns:
        The typed failures, each naming its stage and reason.
    """
    return [
        {
            SAMPLE_ID: result.fixture.sample_id,
            "stage": str(result.outcome.failure_stage),
            "reason": str(result.outcome.failure_reason),
        }
        for result in results
        if result.outcome.status == constrained_decode.FAILED
    ]


def semantic_invalid(results: Sequence[FixtureResult]) -> list[dict[str, str]]:
    """Returns one entry per answer that is well-formed but impossible.

    Args:
        results: One result per fixture.

    Returns:
        The flagged answers, each with its reason. They are counted and
        reported, never corrected.
    """
    return [
        {
            SAMPLE_ID: result.fixture.sample_id,
            "reason": str(result.outcome.semantic_invalid_reason),
        }
        for result in results
        if result.outcome.semantic_invalid_reason is not None
    ]


def timing(results: Sequence[FixtureResult], sample: int) -> dict[str, float]:
    """Summarises how long the sampled fixtures took.

    Args:
        results: One result per fixture, in run order.
        sample: How many leading fixtures the summary covers.

    Returns:
        Measure name to value: latency percentiles in seconds, the
        prompt and answer token counts, and the read and write rates.

    Raises:
        FixtureError: If no fixture is in the sample.
    """
    completions = [result.outcome.completion for result in results[:sample]]
    if not completions:
        msg = "cannot summarise timing over an empty sample"
        raise FixtureError(msg)

    seconds = pd.Series([item.seconds for item in completions])
    summary = {
        f"latency_p{value}_seconds": float(seconds.quantile(value / 100.0))
        for value in PERCENTILES
    }
    summary["sampled_fixtures"] = float(len(completions))
    summary["latency_max_seconds"] = float(seconds.max())
    summary["latency_mean_seconds"] = float(seconds.mean())
    for name, values in (
        ("prompt_tokens", [item.prompt_tokens for item in completions]),
        ("completion_tokens", [item.completion_tokens for item in completions]),
        (
            "prefill_tokens_per_second",
            [item.prefill_tokens_per_second for item in completions],
        ),
        (
            "decode_tokens_per_second",
            [item.decode_tokens_per_second for item in completions],
        ),
    ):
        summary[f"{name}_mean"] = float(pd.Series(values).mean())
    return summary


def build_report(
    results: Sequence[FixtureResult],
    identity: Mapping[str, Any],
    decoding: settings_module.Decoding,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Assembles the record of what the stack did.

    Args:
        results: One result per fixture, in run order.
        identity: What makes these calls reproducible.
        decoding: The decoding settings.
        source: Where the fixtures came from.

    Returns:
        The report payload, ready to serialise.
    """
    tallies = counts(results)
    return {
        "evidence": _EVIDENCE,
        "gate": GATE,
        "identity": dict(identity),
        "source": dict(source),
        "counts": tallies,
        "timing": timing(results, decoding.latency_fixtures),
        "peak_memory_gib": max(
            result.outcome.completion.peak_memory_gib for result in results
        ),
        "elapsed_seconds": sum(
            result.outcome.completion.seconds for result in results
        ),
        "retry_reserve": constrained_decode.retry_reserve(
            tallies["failed"], tallies["fixtures"], decoding
        ),
        "semantic_invalid_rate": tallies["semantic_invalid"]
        / tallies["fixtures"],
        "failures": failures(results),
        "semantic_invalid": semantic_invalid(results),
    }


def check_gate(
    report: Mapping[str, Any], decoding: settings_module.Decoding
) -> None:
    """Refuses a stack that did not clear every fixture.

    Args:
        report: The assembled report.
        decoding: The decoding settings.

    Raises:
        FixtureError: On the first condition the stack did not meet.
    """
    tallies = report["counts"]
    if tallies["fixtures"] != decoding.fixtures:
        msg = (
            f"{tallies['fixtures']} fixtures ran, but "
            f"{decoding.fixtures} were configured"
        )
        raise FixtureError(msg)
    if tallies["failed"]:
        first = report["failures"][0]
        msg = (
            f"{tallies['failed']} of {tallies['fixtures']} fixtures never "
            f"parsed, first {first[SAMPLE_ID]} at {first['stage']}: "
            f"{first['reason']}"
        )
        raise FixtureError(msg)
    if tallies["repeated_identical"] != tallies["repeated"]:
        differed = tallies["repeated"] - tallies["repeated_identical"]
        msg = (
            f"{differed} of {tallies['repeated']} repeated fixtures decoded "
            "differently the second time, so this path is not deterministic"
        )
        raise FixtureError(msg)


def run_params(report: Mapping[str, Any]) -> dict[str, str]:
    """Returns the run parameters for this check.

    Args:
        report: The assembled report.

    Returns:
        Parameter name to value, as text.
    """
    identity = report["identity"]
    params = {
        key: str(value)
        for key, value in identity.items()
        if not isinstance(value, dict)
    }
    params.update(
        {
            f"runtime_{key}": str(value)
            for key, value in identity["runtime"].items()
        }
    )
    params["evidence"] = str(report["evidence"])
    return params


def run_metrics(report: Mapping[str, Any]) -> dict[str, float]:
    """Returns the run metrics for this check.

    Args:
        report: The assembled report.

    Returns:
        Metric name to value.
    """
    metrics = {key: float(value) for key, value in report["counts"].items()}
    metrics.update(report["timing"])
    metrics["peak_memory_gib"] = float(report["peak_memory_gib"])
    metrics["elapsed_seconds"] = float(report["elapsed_seconds"])
    metrics["retry_reserve"] = float(report["retry_reserve"])
    metrics["semantic_invalid_rate"] = float(report["semantic_invalid_rate"])
    return metrics
