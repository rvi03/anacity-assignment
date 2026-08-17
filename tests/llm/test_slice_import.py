"""What the LLM branch inherits, and what must stop it inheriting.

Every fixture here is tiny and built in memory: the point under test is
whether the import refuses a source that disagrees with the store, not
whether Postgres round-trips a timestamp.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import split
from facility_prediction.llm import slice_import
from facility_prediction.models import baselines

SMOKE_CONFIG = pathlib.Path("configs") / "smoke.yaml"
MODULE = pathlib.Path("src") / "facility_prediction" / "llm" / "slice_import.py"
ZONE = "Asia/Kolkata"

VALIDATION_IDS = ("s-1", "s-2")
HOLDOUT_IDS = ("s-3",)


@pytest.fixture(scope="module")
def config():
    return config_module.load_config(SMOKE_CONFIG)


def _stamp(text):
    return pd.Timestamp(text, tz=ZONE)


@pytest.fixture
def spine():
    bookings = pd.DataFrame(
        {
            "booking_id": ["b-1", "b-2", "b-3", "b-4"],
            "resident_id": ["r-1", "r-1", "r-2", "r-2"],
            "facility_id": ["Gym", "Pool", "Gym", "Gym"],
            "booking_timestamp": [
                _stamp("2025-01-01 08:00"),
                _stamp("2025-02-01 08:00"),
                _stamp("2025-03-01 08:00"),
                _stamp("2025-04-01 08:00"),
            ],
            "usage_timestamp": [
                _stamp("2025-01-02 18:00"),
                _stamp("2025-02-02 18:00"),
                _stamp("2025-03-02 18:00"),
                _stamp("2025-04-02 18:00"),
            ],
        }
    )
    samples = pd.DataFrame(
        {
            "sample_id": ["s-1", "s-2", "s-3"],
            "resident_id": ["r-1", "r-2", "r-2"],
            "origin": [
                _stamp("2025-01-02 18:00"),
                _stamp("2025-03-02 18:00"),
                _stamp("2025-04-02 18:00"),
            ],
            "origin_booking_id": ["b-1", "b-3", "b-4"],
            "n_prior_bookings": [1, 1, 2],
            "target_booking_id": ["b-2", "b-4", "b-4"],
            "target_booking_timestamp": [
                _stamp("2025-02-01 08:00"),
                _stamp("2025-04-01 08:00"),
                _stamp("2025-05-01 08:00"),
            ],
            "target_usage_timestamp": [
                _stamp("2025-02-02 18:00"),
                _stamp("2025-04-02 18:00"),
                _stamp("2025-05-02 18:00"),
            ],
            "target_facility_id": ["Pool", "Gym", "Gym"],
            "target_usage_weekday": [6, 2, 4],
            "target_usage_hour": [18, 18, 18],
            "notification_delay_minutes": [1920.0, 1920.0, 1920.0],
        }
    )
    splits = pd.DataFrame(
        {
            "sample_id": ["s-1", "s-2", "s-3"],
            "split": [split.VALIDATION, split.VALIDATION, split.TEST],
        }
    )
    predictions = pd.DataFrame(
        {
            "track": [baselines.TRACK] * 3,
            "model": [baselines.MODEL_NAME] * 3,
            "sample_id": ["s-1", "s-2", "s-3"],
            "predicted_facility_id": ["Gym", "Gym", "Pool"],
            "predicted_usage_weekday": [6, 3, 1],
            "predicted_usage_hour": [18, 9, 7],
            "predicted_delay_minutes": [1920.0, 60.0, 10.0],
            "predicted_transition_facility_id": ["Gym", "Gym", "Gym"],
        }
    )
    return slice_import.Spine(
        bookings=bookings,
        samples=samples,
        splits=splits,
        predictions=predictions,
    )


def _split_shape(spine):
    counts = spine.splits["split"].value_counts()
    return {
        name: {
            "rows": int(counts.get(name, 0)),
            "share": float(counts.get(name, 0)) / len(spine.splits),
            "first": "2025-01-01T00:00:00+05:30",
            "last": "2025-12-31T00:00:00+05:30",
        }
        for name in split.SPLIT_NAMES
    }


@pytest.fixture
def manifests(spine):
    live = slice_import.table_digests(spine)
    return slice_import.Manifests(
        generation={
            "provenance": {"bookings_digest": live["bookings_digest"]},
            "counts": {"bookings": len(spine.bookings)},
        },
        sample={
            "provenance": {
                "bookings_digest": live["bookings_digest"],
                "samples_digest": live["samples_digest"],
            },
            "counts": {
                "samples": len(spine.samples),
                "residents_excluded_no_prior_history": 0,
            },
        },
        split={
            "provenance": {"samples_digest": live["samples_digest"]},
            "settings": {
                "split_basis": "target_booking_timestamp",
                "comparison_rows": len(HOLDOUT_IDS),
            },
            "cutoffs": {
                "start": "2025-01-01T00:00:00+05:30",
                "train_cut": "2025-02-01T00:00:00+05:30",
                "val_cut": "2025-04-15T00:00:00+05:30",
                "end": "2025-12-31T00:00:00+05:30",
            },
            "splits": _split_shape(spine),
            "samples": len(spine.samples),
        },
        comparison={"sample_ids": list(HOLDOUT_IDS)},
    )


@pytest.fixture
def sources():
    return {"split_manifest_sha256": "0" * 64}


def _record(spine, manifests, sources, config):
    slice_import.check_agreement(spine, manifests)
    return slice_import.build_record(
        spine, manifests, sources, config, split.VALIDATION
    )


def test_the_record_reports_the_live_tables(spine, manifests, sources, config):
    record = _record(spine, manifests, sources, config)

    assert record["counts"]["bookings"] == len(spine.bookings)
    assert record["counts"]["samples"] == len(spine.samples)
    assert record["counts"]["comparison_rows"] == len(HOLDOUT_IDS)
    assert record["evidence"] == slice_import.EVIDENCE


def test_the_record_carries_a_source_hash_for_every_inherited_table(
    spine, manifests, sources, config
):
    record = _record(spine, manifests, sources, config)

    assert set(record["provenance"]) >= {
        "bookings_digest",
        "samples_digest",
        "splits_digest",
        "predictions_digest",
        "split_manifest_sha256",
    }


def test_a_manifest_digest_that_disagrees_stops_the_import(spine, manifests):
    stale = dataclasses.replace(
        manifests,
        generation={
            "provenance": {"bookings_digest": "0" * 64},
            "counts": {"bookings": len(spine.bookings)},
        },
    )

    with pytest.raises(slice_import.SliceImportError, match="booking digest"):
        slice_import.check_agreement(spine, stale)


def test_a_row_count_that_disagrees_stops_the_import(spine, manifests):
    shape = dict(manifests.split["splits"])
    shape[split.VALIDATION] = dict(shape[split.VALIDATION])
    shape[split.VALIDATION]["rows"] += 1
    drifted = dataclasses.replace(
        manifests, split={**manifests.split, "splits": shape}
    )

    with pytest.raises(slice_import.SliceImportError, match="row count"):
        slice_import.check_agreement(spine, drifted)


def test_a_missing_manifest_key_stops_the_import(spine, manifests):
    incomplete = dataclasses.replace(manifests, generation={"counts": {}})

    with pytest.raises(slice_import.SliceImportError, match="no provenance"):
        slice_import.check_agreement(spine, incomplete)


def test_a_missing_source_file_stops_the_import(tmp_path):
    with pytest.raises(slice_import.SliceImportError, match="does not exist"):
        slice_import.file_digest(tmp_path / "absent.json")


def test_a_comparison_row_outside_the_holdout_stops_the_import(
    spine, manifests
):
    widened = dataclasses.replace(
        manifests, comparison={"sample_ids": [VALIDATION_IDS[0]]}
    )

    with pytest.raises(slice_import.SliceImportError, match="outside"):
        slice_import.check_agreement(spine, widened)


def test_the_comparator_is_scored_on_the_named_split_only(spine, config):
    scores = slice_import.comparator_scores(spine, config, split.VALIDATION)

    assert scores["rows"] == len(VALIDATION_IDS)
    assert scores["split"] == split.VALIDATION


def test_a_holdout_target_never_reaches_the_comparator(spine, config):
    before = slice_import.comparator_scores(spine, config, split.VALIDATION)
    sealed = spine.samples.copy()
    holdout = sealed["sample_id"].isin(HOLDOUT_IDS)
    sealed.loc[holdout, "target_facility_id"] = "Tennis"
    sealed.loc[holdout, "notification_delay_minutes"] = 1.0

    after = slice_import.comparator_scores(
        slice_import.Spine(
            bookings=spine.bookings,
            samples=sealed,
            splits=spine.splits,
            predictions=spine.predictions,
        ),
        config,
        split.VALIDATION,
    )

    assert after == before


def test_a_split_with_no_scored_row_stops_the_import(spine, config):
    with pytest.raises(slice_import.SliceImportError, match="scored no"):
        slice_import.comparator_scores(spine, config, split.TRAIN)


def test_a_value_that_was_never_measured_stops_the_import():
    with pytest.raises(slice_import.SliceImportError, match="rows"):
        slice_import.check_measured({"counts": {"rows": float("nan")}})


def test_an_absent_value_stops_the_import():
    with pytest.raises(slice_import.SliceImportError, match="measures"):
        slice_import.check_measured({"counts": {"rows": None}})


def test_every_recorded_value_is_measured(spine, manifests, sources, config):
    record = _record(spine, manifests, sources, config)

    slice_import.check_measured(record)


def test_the_run_carries_finite_numbers(spine, manifests, sources, config):
    record = _record(spine, manifests, sources, config)

    metrics = slice_import.run_metrics(record)
    params = slice_import.run_params(record)

    assert metrics["count_bookings"] == float(len(spine.bookings))
    assert all(math.isfinite(value) for value in metrics.values())
    assert params["comparator_split"] == split.VALIDATION


def test_no_count_or_metric_is_written_into_the_module():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    }

    assert literals <= {0, 1}, (
        f"{MODULE.name} writes {sorted(literals)} into the code; an "
        "inherited quantity is read from the store, never typed here"
    )
