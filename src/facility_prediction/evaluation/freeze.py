"""The freeze, and the seal that makes the holdout score mean something.

The holdout is scored **once per track, ever**. That rule is only worth
having if something enforces it, because the failure it prevents is not
malice — it is scoring, seeing a number, changing one setting, and
scoring again. Each of those runs looks reasonable on its own, and the
number at the end has quietly stopped being an estimate of anything.

Two artifacts do the enforcing:

    freeze record   what the configuration and the chosen sources were
                    at the moment they were locked. Written before any
                    holdout row is read.
    seal record     that the holdout has been scored, by which track,
                    against which freeze. Written after.

The order is the whole point:

    freeze  ──>  score once  ──>  seal
       │                            │
       └── config hash must match ──┘
           at scoring time, or the
           run is refused

A second scoring attempt finds the seal and stops. A scoring attempt
after the configuration moved finds a hash mismatch and stops. Neither
is a warning that can be scrolled past; both raise.

Leakage contract: nothing here reads a target, a feature, or a
prediction. It reads configuration and its own two records, so it can
be called before the holdout is touched without touching it.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import logging
import pathlib
from typing import Any

from facility_prediction import config as config_module

_LOGGER = logging.getLogger(__name__)

FREEZE_FILENAME = "freeze.json"
SEAL_FILENAME = "holdout_seal.json"

TRADITIONAL = "traditional"


class FreezeError(Exception):
    """Raised when a freeze is missing, stale, or already spent."""


def build_freeze(
    config: config_module.Config,
    head_sources: Mapping[str, str],
    digests: Mapping[str, str],
    track: str = TRADITIONAL,
) -> dict[str, Any]:
    """Assemble the record of what is being locked.

    Args:
        config: The configuration to freeze.
        head_sources: Which model supplies each output, chosen on
            validation and fixed from here.
        digests: The input digests the freeze is taken over.
        track: Which track is freezing.

    Returns:
        The freeze payload, ready to serialise.

    Raises:
        FreezeError: If no source map is supplied; a freeze that does
            not say which model answers is not a freeze.
    """
    if not head_sources:
        msg = "a freeze must name the source of every output"
        raise FreezeError(msg)
    return {
        "track": track,
        "config_hash": config_module.config_hash(config),
        "seed": config.seed,
        "head_sources": dict(head_sources),
        "digests": dict(digests),
    }


def write_freeze(payload: Mapping[str, Any], path: pathlib.Path) -> None:
    """Write the freeze record.

    Args:
        payload: The freeze payload.
        path: Destination JSON; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def read_freeze(path: pathlib.Path) -> dict[str, Any]:
    """Read the freeze record.

    Args:
        path: The freeze JSON.

    Returns:
        The freeze payload.

    Raises:
        FreezeError: If no freeze was written there.
    """
    if not path.is_file():
        msg = (
            f"no freeze at {path}; the configuration must be locked before "
            "the holdout is scored"
        )
        raise FreezeError(msg)
    return json.loads(path.read_text(encoding="utf-8"))


def check_freeze_matches(
    frozen: Mapping[str, Any], config: config_module.Config
) -> None:
    """Refuse a run whose configuration moved after the freeze.

    Args:
        frozen: The freeze payload.
        config: The configuration this run resolved.

    Raises:
        FreezeError: If the hashes differ.
    """
    current = config_module.config_hash(config)
    if frozen.get("config_hash") != current:
        msg = (
            "the configuration changed after the freeze: frozen at "
            f"{frozen.get('config_hash')}, this run resolves to {current}. "
            "The holdout may only be scored against the frozen settings"
        )
        raise FreezeError(msg)


def is_sealed(path: pathlib.Path) -> bool:
    """Return whether the holdout has already been scored.

    Args:
        path: The seal JSON.

    Returns:
        True when a seal exists.
    """
    return path.is_file()


def require_unsealed(path: pathlib.Path, track: str = TRADITIONAL) -> None:
    """Refuse a second scoring of the holdout.

    Args:
        path: The seal JSON.
        track: The track attempting to score.

    Raises:
        FreezeError: If the holdout was already scored by this track.
    """
    if is_sealed(path):
        existing = json.loads(path.read_text(encoding="utf-8"))
        msg = (
            f"the holdout was already scored by track "
            f"{existing.get('track')!r} against freeze "
            f"{existing.get('config_hash')}. It is scored once per track, "
            f"ever; {track!r} cannot score it again. Delete {path} only if "
            "you accept that every number measured from it becomes a "
            "best-of-N result and must be reported as one"
        )
        raise FreezeError(msg)


def build_seal(
    frozen: Mapping[str, Any],
    rows: int,
    headline: Mapping[str, float],
    track: str = TRADITIONAL,
) -> dict[str, Any]:
    """Assemble the record that the holdout has now been spent.

    Args:
        frozen: The freeze this scoring ran against.
        rows: How many holdout rows were scored.
        headline: The headline numbers the scoring produced.
        track: Which track scored.

    Returns:
        The seal payload, ready to serialise.
    """
    return {
        "track": track,
        "config_hash": frozen.get("config_hash"),
        "head_sources": dict(frozen.get("head_sources", {})),
        "rows_scored": rows,
        "headline": {key: float(value) for key, value in headline.items()},
    }


def write_seal(payload: Mapping[str, Any], path: pathlib.Path) -> None:
    """Write the seal record.

    Args:
        payload: The seal payload.
        path: Destination JSON; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
