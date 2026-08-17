r"""The notification output, framed around each resident's own cadence.

The scored rule is multiplicative: a submitted delay matches when the
actual delay lands inside a fixed ratio of it. Two consequences drive
everything here.

*The decision is mode-seeking, not mean-seeking.* The submission that
maximises the chance of a match is the centre of the window carrying the
most conditional mass, which is not the conditional mean or median. A
point regression fitted to either of those loses to a single constant on
this data, because inter-booking gaps are diffuse (log spread 1.64) and a
regression spends that spread moving predictions away from where the mass
actually is.

*The personalisation belongs in an offset, not in the model.* A resident's
own median gap is already a feature, and it carries most of what is
knowable::

    log delay  =  log(resident's median gap)  +  residual
                  \__________ offset _______/    \__ modelled __/

The offset is read straight off the feature row. The model only has to
shape the residual, whose spread is far smaller than the raw target's, and
the decode integrates the predicted residual density over the scored
window and submits its heaviest centre.

Leakage contract: the offset is a declared feature built from bookings at
or before the origin, so it is safe on any row. :func:`fit_calibration`
reads targets and must be handed training rows only; every other function
reads features and an already-fitted calibration.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import math
import pathlib
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from facility_prediction import config as config_module

OFFSET_FEATURE = "inter_booking_interval_median_days"
CALIBRATION_FILENAME = "notification_cadence.json"

_MINUTES_PER_DAY = 24.0 * 60.0
_GRID_PER_BUCKET = 3
# A probability block is one row per sample, one column per bucket.
_BLOCK_DIMENSIONS = 2


class CadenceError(Exception):
    """Raised when a calibration is unusable or does not match its inputs."""


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Everything the notification head needs beyond its estimator.

    Attributes:
        fallback_log_minutes: Offset used where a resident has no median
            gap yet, in natural-log minutes. Fitted on training rows.
        step: Width of one residual bucket, in natural-log units.
        span: Half-width of the modelled residual range; residuals
            outside it fall into the end buckets.
        marginal: The training residual distribution over those buckets,
            used to shrink a thin per-row density.
        blend: Weight on the model's own density against ``marginal``;
            one uses the model alone.
    """

    fallback_log_minutes: float
    step: float
    span: float
    marginal: tuple[float, ...]
    blend: float

    def edges(self) -> npt.NDArray[np.float64]:
        """Return the residual bucket edges, in natural-log units.

        Returns:
            Strictly increasing edges from ``-span`` to ``+span``.
        """
        count = len(self.marginal)
        return np.linspace(-self.span, self.span, count + 1)

    def to_dict(self) -> dict[str, Any]:
        """Return the calibration as plain JSON-ready data.

        Returns:
            A mapping with one entry per attribute.
        """
        return {
            "fallback_log_minutes": self.fallback_log_minutes,
            "step": self.step,
            "span": self.span,
            "marginal": list(self.marginal),
            "blend": self.blend,
        }


def from_dict(payload: Mapping[str, Any]) -> Calibration:
    """Rebuild a calibration from its stored form.

    Args:
        payload: A mapping as written by :meth:`Calibration.to_dict`.

    Returns:
        The calibration.

    Raises:
        CadenceError: If a required entry is absent or unusable.
    """
    try:
        calibration = Calibration(
            fallback_log_minutes=float(payload["fallback_log_minutes"]),
            step=float(payload["step"]),
            span=float(payload["span"]),
            marginal=tuple(float(value) for value in payload["marginal"]),
            blend=float(payload["blend"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        msg = f"unusable notification calibration: {error}"
        raise CadenceError(msg) from error
    if not calibration.marginal:
        msg = "a notification calibration must carry a residual marginal"
        raise CadenceError(msg)
    return calibration


def write_calibration(
    calibration: Calibration, directory: pathlib.Path
) -> None:
    """Write a calibration beside the saved heads.

    Args:
        calibration: The fitted calibration.
        directory: The model directory; created if absent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CALIBRATION_FILENAME
    with path.open("w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=2, sort_keys=False)
        handle.write("\n")


def read_calibration(directory: pathlib.Path) -> Calibration:
    """Read the calibration written beside the saved heads.

    Args:
        directory: The model directory.

    Returns:
        The stored calibration.

    Raises:
        CadenceError: If no calibration was written there.
    """
    path = directory / CALIBRATION_FILENAME
    if not path.is_file():
        msg = f"no notification calibration at {path}"
        raise CadenceError(msg)
    return from_dict(json.loads(path.read_text(encoding="utf-8")))


def offsets(
    table: pd.DataFrame, calibration: Calibration
) -> npt.NDArray[np.float64]:
    """Return each row's cadence offset, in natural-log minutes.

    Leakage contract: reads one declared feature column, whose history
    stops at the row's own origin.

    Args:
        table: A validated feature table.
        calibration: A fitted calibration, supplying the fallback.

    Returns:
        One offset per row; the fallback where the resident has no median
        gap yet.

    Raises:
        CadenceError: If the offset feature is absent.
    """
    if OFFSET_FEATURE not in table.columns:
        msg = f"the feature table carries no {OFFSET_FEATURE!r} to offset by"
        raise CadenceError(msg)
    days = table[OFFSET_FEATURE].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.log(days * _MINUTES_PER_DAY)
    return np.where(np.isfinite(value), value, calibration.fallback_log_minutes)


def window_mode(values: npt.ArrayLike, tolerance: float) -> float:
    """Return the centre of the window holding the most observations.

    Args:
        values: Observations on a log scale.
        tolerance: Half-width of the scored window, in the same units.

    Returns:
        The midpoint of the fullest window.

    Raises:
        CadenceError: If there is nothing to take a mode of.
    """
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    if not len(ordered):
        msg = "cannot take a window mode of an empty sample"
        raise CadenceError(msg)
    last = np.searchsorted(ordered, ordered + 2.0 * tolerance, side="right")
    start = int(np.argmax(last - np.arange(len(ordered))))
    stop = min(last[start] - 1, len(ordered) - 1)
    return float((ordered[start] + ordered[stop]) / 2.0)


def fit_calibration(
    table: pd.DataFrame,
    delays: pd.Series,
    config: config_module.Config,
) -> Calibration:
    """Fit the offset fallback and the residual marginal on training rows.

    Leakage contract: reads the targets of the rows it is given, which
    the caller must have restricted to the training split.

    Args:
        table: Feature rows for the training split.
        delays: Observed booking delays in minutes, in that row order.
        config: Validated configuration; supplies the residual geometry
            and the scored tolerance.

    Returns:
        A calibration ready to label, decode and store.

    Raises:
        CadenceError: If there is nothing to fit on, or a delay is not
            positive.
    """
    settings = config.catboost
    values = delays.to_numpy(dtype=np.float64)
    if not len(values):
        msg = "cannot calibrate the notification head without rows"
        raise CadenceError(msg)
    if (values <= 0.0).any():
        msg = "notification delays must be positive to take a log of"
        raise CadenceError(msg)

    tolerance = math.log(config.evaluation.notification_match_ratio)
    observed = np.log(values)
    fallback = window_mode(observed, tolerance)

    span = float(settings.notification_residual_span)
    step = float(settings.notification_residual_step)
    count = max(1, round(2.0 * span / step))
    seed = Calibration(
        fallback_log_minutes=fallback,
        step=step,
        span=span,
        marginal=(1.0,) * count,
        blend=float(settings.notification_residual_blend),
    )
    residual = observed - offsets(table, seed)
    counts = np.bincount(_bucket(residual, seed), minlength=count)
    marginal = counts.astype(np.float64) + 1.0
    marginal /= marginal.sum()
    return dataclasses.replace(seed, marginal=tuple(marginal.tolist()))


def _bucket(
    residual: npt.NDArray[np.float64], calibration: Calibration
) -> npt.NDArray[np.int64]:
    """Return the residual bucket of every value.

    Args:
        residual: Residuals in natural-log units.
        calibration: The calibration naming the bucket geometry.

    Returns:
        Bucket indices, clipped into the modelled range.
    """
    count = len(calibration.marginal)
    edges = calibration.edges()
    return np.clip(np.digitize(residual, edges) - 1, 0, count - 1)


def residual_labels(
    table: pd.DataFrame,
    delays: pd.Series,
    calibration: Calibration,
) -> pd.Series:
    """Return the residual bucket every row is trained to predict.

    Leakage contract: reads the target of the row it labels; for training
    rows only.

    Args:
        table: Feature rows.
        delays: Observed booking delays in minutes, in that row order.
        calibration: A fitted calibration.

    Returns:
        Integer bucket labels aligned to ``table``.

    Raises:
        CadenceError: If a delay is not positive.
    """
    values = delays.to_numpy(dtype=np.float64)
    if (values <= 0.0).any():
        msg = "notification delays must be positive to take a log of"
        raise CadenceError(msg)
    residual = np.log(values) - offsets(table, calibration)
    return pd.Series(
        _bucket(residual, calibration), index=table.index, dtype="int64"
    )


def decode(
    probabilities: npt.ArrayLike,
    labels: npt.ArrayLike,
    table: pd.DataFrame,
    calibration: Calibration,
    match_ratio: float,
) -> npt.NDArray[np.float64]:
    """Submit the delay whose scored window carries the most mass.

    The predicted residual density is shrunk toward the training marginal
    by ``calibration.blend``, integrated over every candidate window, and
    the heaviest window's centre is added back to the row's own offset.

    Args:
        probabilities: One row per sample, one column per emitted label.
        labels: The residual buckets those columns stand for.
        table: The feature rows being scored, for their offsets.
        calibration: A fitted calibration.
        match_ratio: The symmetric multiplicative tolerance that is
            scored; the window is that wide on each side.

    Returns:
        One positive submitted delay in minutes per row.

    Raises:
        CadenceError: If a label falls outside the calibrated range, or
            the probability block does not match the rows.
    """
    count = len(calibration.marginal)
    columns = np.asarray(labels, dtype=np.int64)
    if (columns < 0).any() or (columns >= count).any():
        msg = "the notification model emitted an unknown residual bucket"
        raise CadenceError(msg)
    block = np.asarray(probabilities, dtype=np.float64)
    if block.ndim != _BLOCK_DIMENSIONS or len(block) != len(table):
        msg = (
            f"a probability block of shape {block.shape} does not cover "
            f"{len(table)} rows"
        )
        raise CadenceError(msg)

    dense = np.zeros((len(block), count), dtype=np.float64)
    dense[:, columns] = block
    marginal = np.asarray(calibration.marginal, dtype=np.float64)
    blend = calibration.blend
    dense = blend * dense + (1.0 - blend) * marginal[None, :]

    edges = calibration.edges()
    grid = np.linspace(
        -calibration.span,
        calibration.span,
        max(2, count * _GRID_PER_BUCKET),
    )
    tolerance = math.log(match_ratio)
    overlap = (
        np.clip(
            np.minimum(grid[:, None] + tolerance, edges[None, 1:])
            - np.maximum(grid[:, None] - tolerance, edges[None, :-1]),
            0.0,
            None,
        )
        / (edges[1:] - edges[:-1])[None, :]
    )
    centre = grid[np.argmax(dense @ overlap.T, axis=1)]
    return np.exp(centre + offsets(table, calibration))
