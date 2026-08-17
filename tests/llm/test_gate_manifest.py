"""The gate manifest: what it draws, what it keeps, what it refuses.

The manifest decides which rows the whole branch is judged on, so the
tests are about coverage and sealing: every stratum represented, rare
ones kept, the same rows on a second draw, and nothing from the holdout.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from facility_prediction.llm import gate_manifest
from facility_prediction.llm import settings as settings_module

LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

FACILITIES = ("Gym", "Pool", "Yoga Room")
ROWS = 120
DRAW = 24


@pytest.fixture
def gate():
    return settings_module.load_settings(LLM_CONFIG).gate


@pytest.fixture
def rows():
    return pd.DataFrame(
        {
            "sample_id": [f"s{index:03d}" for index in range(ROWS)],
            "target_facility_id": [
                FACILITIES[index % len(FACILITIES)] for index in range(ROWS)
            ],
            "n_prior_bookings": [1 + index % 40 for index in range(ROWS)],
        }
    )


def test_history_band_covers_every_count(gate):
    bands = {
        gate_manifest.history_band(count, gate.history_bins)
        for count in range(0, 200)
    }

    assert "under_1" in bands
    assert "30_plus" in bands
    assert len(bands) == len(gate.history_bins) + 1


def test_every_row_lands_in_exactly_one_stratum(rows, gate):
    labelled = gate_manifest.label_strata(rows, gate.history_bins)

    assert len(labelled) == ROWS
    assert labelled[gate_manifest.STRATUM].notna().all()


def test_allocation_keeps_a_row_for_every_stratum():
    allocation = gate_manifest.allocate({"a": 100, "b": 1, "c": 5}, 20)

    assert allocation["b"] == 1
    assert sum(allocation.values()) == 20


def test_allocation_never_asks_a_stratum_for_rows_it_lacks():
    allocation = gate_manifest.allocate({"a": 3, "b": 3}, 6)

    assert allocation == {"a": 3, "b": 3}


def test_allocation_refuses_a_draw_larger_than_the_rows():
    with pytest.raises(gate_manifest.ManifestError, match="cannot draw"):
        gate_manifest.allocate({"a": 5}, 10)


def test_allocation_refuses_more_strata_than_rows():
    sizes = {f"s{index}": 2 for index in range(10)}

    with pytest.raises(gate_manifest.ManifestError, match="cannot each keep"):
        gate_manifest.allocate(sizes, 5)


def test_draw_is_the_declared_size(rows, gate):
    drawn = gate_manifest.draw(rows, gate, DRAW)

    assert len(drawn) == DRAW


def test_draw_repeats_exactly_under_the_same_seed(rows, gate):
    first = gate_manifest.draw(rows, gate, DRAW)
    second = gate_manifest.draw(rows, gate, DRAW)

    assert list(first["sample_id"]) == list(second["sample_id"])


def test_draw_covers_every_facility(rows, gate):
    drawn = gate_manifest.draw(rows, gate, DRAW)

    assert set(drawn["target_facility_id"]) == set(FACILITIES)


def test_draw_refuses_rows_without_the_columns_it_needs(rows, gate):
    with pytest.raises(gate_manifest.ManifestError, match="needs columns"):
        gate_manifest.draw(rows.drop(columns=["n_prior_bookings"]), gate, DRAW)


def test_manifest_records_its_strata_and_singletons(rows, gate):
    payload = gate_manifest.build_manifest(
        gate_manifest.draw(rows, gate, DRAW), gate, {"validation_rows": ROWS}
    )

    assert payload["rows"] == DRAW
    assert sum(payload["strata"].values()) == DRAW
    assert payload["split"] == "validation"


def test_manifest_refuses_a_holdout_row(rows, gate):
    payload = gate_manifest.build_manifest(
        gate_manifest.draw(rows, gate, DRAW), gate, {}
    )
    sealed = {payload["sample_ids"][0]}

    with pytest.raises(gate_manifest.ManifestError, match="holdout row"):
        gate_manifest.check_sealed(payload, sealed)


def test_manifest_accepts_a_draw_clear_of_the_holdout(rows, gate):
    payload = gate_manifest.build_manifest(
        gate_manifest.draw(rows, gate, DRAW), gate, {}
    )

    gate_manifest.check_sealed(payload, {"not-a-drawn-id"})
