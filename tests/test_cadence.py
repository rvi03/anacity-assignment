"""Tests the cadence framing of the notification head.

Covers the promise the framing rests on: the offset is read off a
declared feature and never from the target, the residual grid tiles the
modelled span exactly once, a calibration survives a round trip to disk,
and decoding submits the centre of the window carrying the most mass
rather than the centre of the likeliest bucket. Where the two disagree
is exactly where this framing earns its place, so that disagreement has
its own test.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.models import cadence

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

MINUTES_PER_DAY = 24.0 * 60.0


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


def _table(median_days) -> pd.DataFrame:
    return pd.DataFrame({cadence.OFFSET_FEATURE: list(median_days)})


def _calibration(count: int = 8, blend: float = 1.0) -> cadence.Calibration:
    return cadence.Calibration(
        fallback_log_minutes=math.log(2.0 * MINUTES_PER_DAY),
        step=0.5,
        span=2.0,
        marginal=(1.0 / count,) * count,
        blend=blend,
    )


# --- the residual grid -------------------------------------------------


def test_the_edges_tile_the_span_once():
    calibration = _calibration(count=8)

    edges = calibration.edges()

    assert len(edges) == len(calibration.marginal) + 1
    assert edges[0] == pytest.approx(-calibration.span)
    assert edges[-1] == pytest.approx(calibration.span)
    assert np.all(np.diff(edges) > 0)


def test_every_residual_lands_in_exactly_one_bucket():
    calibration = _calibration(count=8)
    table = _table([1.0] * 500)
    generator = np.random.default_rng(20260816)
    delays = pd.Series(
        np.exp(math.log(MINUTES_PER_DAY) + generator.normal(0.0, 1.5, size=500))
    )

    labels = cadence.residual_labels(table, delays, calibration)

    assert labels.between(0, len(calibration.marginal) - 1).all()
    assert labels.index.equals(table.index)


def test_a_residual_past_the_span_falls_into_an_end_bucket():
    calibration = _calibration(count=8)
    table = _table([1.0, 1.0])
    # One delay far below the offset, one far above it.
    delays = pd.Series([MINUTES_PER_DAY / 1000.0, MINUTES_PER_DAY * 1000.0])

    labels = cadence.residual_labels(table, delays, calibration)

    assert list(labels) == [0, len(calibration.marginal) - 1]


# --- the offset --------------------------------------------------------


def test_the_offset_is_the_resident_median_gap_in_log_minutes():
    calibration = _calibration()
    table = _table([1.0, 3.0])

    values = cadence.offsets(table, calibration)

    assert values[0] == pytest.approx(math.log(MINUTES_PER_DAY))
    assert values[1] == pytest.approx(math.log(3.0 * MINUTES_PER_DAY))


def test_a_resident_without_a_median_gap_takes_the_fitted_fallback():
    calibration = _calibration()
    table = _table([np.nan, 0.0])

    values = cadence.offsets(table, calibration)

    assert list(values) == [
        calibration.fallback_log_minutes,
        calibration.fallback_log_minutes,
    ]


def test_a_table_without_the_offset_feature_is_rejected():
    calibration = _calibration()

    with pytest.raises(cadence.CadenceError, match=r"no .* to offset by"):
        cadence.offsets(pd.DataFrame({"something_else": [1.0]}), calibration)


# --- the window mode ---------------------------------------------------


def test_the_window_mode_finds_the_densest_window_not_the_mean():
    # Nine observations packed inside one narrow window, one far outlier
    # heavy enough to drag a mean but not to own a window.
    packed = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    centre = cadence.window_mode([*packed, 40.0], tolerance=0.25)

    assert centre == pytest.approx(0.2, abs=0.15)
    assert centre < np.mean([*packed, 40.0])


def test_the_window_mode_of_an_empty_sample_is_rejected():
    with pytest.raises(cadence.CadenceError, match="empty sample"):
        cadence.window_mode([], tolerance=0.25)


# --- fitting -----------------------------------------------------------


def test_the_fitted_marginal_is_a_distribution(smoke_config):
    table = _table([1.0] * 200)
    generator = np.random.default_rng(20260816)
    delays = pd.Series(
        np.exp(math.log(MINUTES_PER_DAY) + generator.normal(0.0, 1.0, size=200))
    )

    calibration = cadence.fit_calibration(table, delays, smoke_config)

    assert sum(calibration.marginal) == pytest.approx(1.0)
    assert all(weight > 0.0 for weight in calibration.marginal)
    assert calibration.span == smoke_config.catboost.notification_residual_span


def test_fitting_is_deterministic(smoke_config):
    table = _table([1.0, 2.0, 3.0, 4.0])
    delays = pd.Series([600.0, 1200.0, 2400.0, 4800.0])

    first = cadence.fit_calibration(table, delays, smoke_config)
    second = cadence.fit_calibration(table, delays, smoke_config)

    assert first == second


def test_fitting_without_rows_is_rejected(smoke_config):
    with pytest.raises(cadence.CadenceError, match="without rows"):
        cadence.fit_calibration(
            _table([]), pd.Series(dtype="float64"), smoke_config
        )


def test_a_non_positive_delay_is_rejected(smoke_config):
    with pytest.raises(cadence.CadenceError, match="must be positive"):
        cadence.fit_calibration(
            _table([1.0, 1.0]), pd.Series([600.0, 0.0]), smoke_config
        )


def test_labelling_a_non_positive_delay_is_rejected():
    with pytest.raises(cadence.CadenceError, match="must be positive"):
        cadence.residual_labels(
            _table([1.0]), pd.Series([-1.0]), _calibration()
        )


# --- storage -----------------------------------------------------------


def test_a_calibration_round_trips_through_disk(tmp_path):
    calibration = _calibration(count=6, blend=0.7)

    cadence.write_calibration(calibration, tmp_path)

    assert cadence.read_calibration(tmp_path) == calibration


def test_reading_a_calibration_that_was_never_written_is_rejected(tmp_path):
    with pytest.raises(cadence.CadenceError, match="no notification"):
        cadence.read_calibration(tmp_path)


def test_a_calibration_without_a_marginal_is_rejected():
    payload = _calibration().to_dict()
    payload["marginal"] = []

    with pytest.raises(cadence.CadenceError, match="residual marginal"):
        cadence.from_dict(payload)


def test_a_calibration_missing_an_entry_is_rejected():
    payload = _calibration().to_dict()
    del payload["span"]

    with pytest.raises(cadence.CadenceError, match="unusable"):
        cadence.from_dict(payload)


# --- decoding ----------------------------------------------------------


def test_decoding_returns_one_positive_delay_per_row():
    calibration = _calibration(count=8)
    table = _table([1.0, 2.0, 3.0])
    labels = np.arange(8)
    block = np.full((3, 8), 1.0 / 8.0)

    submitted = cadence.decode(block, labels, table, calibration, 1.25)

    assert submitted.shape == (3,)
    assert np.all(submitted > 0.0)


def test_a_confident_row_decodes_near_its_own_offset():
    calibration = _calibration(count=8)
    table = _table([2.0])
    labels = np.arange(8)
    # All mass on the two buckets straddling a zero residual.
    block = np.zeros((1, 8))
    block[0, 3] = 0.5
    block[0, 4] = 0.5

    submitted = cadence.decode(block, labels, table, calibration, 1.25)

    assert submitted[0] == pytest.approx(2.0 * MINUTES_PER_DAY, rel=0.25)


def test_the_window_beats_the_likeliest_bucket_when_they_disagree():
    # One tall bucket at the far edge, and three neighbours that no
    # single bucket beats but one window covers. Argmax would take the
    # tall bucket; the window decode must take the cluster.
    calibration = _calibration(count=8)
    table = _table([1.0])
    labels = np.arange(8)
    block = np.zeros((1, 8))
    block[0, 0] = 0.31
    block[0, 4] = block[0, 5] = block[0, 6] = 0.23

    submitted = cadence.decode(block, labels, table, calibration, 4.0)
    residual = math.log(submitted[0]) - math.log(MINUTES_PER_DAY)

    assert residual > 0.0


def test_blending_toward_the_marginal_pulls_a_thin_row_back():
    # The marginal is concentrated low; the row's own density is a
    # single spike high. Trusting the row alone lands high, blending
    # most of the way to the marginal must land lower.
    count = 8
    marginal = np.full(count, 0.01)
    marginal[1] = 1.0 - 0.01 * (count - 1)
    table = _table([1.0])
    labels = np.arange(count)
    block = np.zeros((1, count))
    block[0, count - 1] = 1.0

    alone = cadence.decode(
        block,
        labels,
        table,
        cadence.Calibration(
            fallback_log_minutes=0.0,
            step=0.5,
            span=2.0,
            marginal=tuple(marginal.tolist()),
            blend=1.0,
        ),
        1.25,
    )
    blended = cadence.decode(
        block,
        labels,
        table,
        cadence.Calibration(
            fallback_log_minutes=0.0,
            step=0.5,
            span=2.0,
            marginal=tuple(marginal.tolist()),
            blend=0.05,
        ),
        1.25,
    )

    assert blended[0] < alone[0]


def test_an_unknown_residual_bucket_is_rejected():
    calibration = _calibration(count=4)
    table = _table([1.0])

    with pytest.raises(cadence.CadenceError, match="unknown residual bucket"):
        cadence.decode(np.ones((1, 1)), np.array([9]), table, calibration, 1.25)


def test_a_probability_block_that_misses_rows_is_rejected():
    calibration = _calibration(count=4)
    table = _table([1.0, 2.0])

    with pytest.raises(cadence.CadenceError, match="does not cover"):
        cadence.decode(np.ones((1, 4)), np.arange(4), table, calibration, 1.25)


# --- leakage -----------------------------------------------------------


def test_decoding_never_reads_a_target():
    # Identical features and identical predicted densities must decode
    # identically no matter what the actual delays were.
    calibration = _calibration(count=8)
    table = _table([1.0, 1.0])
    labels = np.arange(8)
    block = np.tile(np.full(8, 1.0 / 8.0), (2, 1))

    submitted = cadence.decode(block, labels, table, calibration, 1.25)

    assert submitted[0] == pytest.approx(submitted[1])
