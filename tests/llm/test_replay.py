"""Replay: the committed answers must rebuild the committed result.

This is what ``make llm-reproduce`` asserts. Nothing here loads the
model, the adapter, or any generation library — if these pass, the
cache really is the reproducibility mechanism and a reviewer needs no
GPU and no API key to check the headline numbers.
"""

from __future__ import annotations

import hashlib
import pathlib

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import split as split_module
from facility_prediction.data import storage
from facility_prediction.evaluation import evaluate
from facility_prediction.llm import buckets, final_pass, gate, llm_predict

CONFIG = pathlib.Path("configs") / "default.yaml"
ARTIFACTS = pathlib.Path("artifacts")

REPORT = ARTIFACTS / "llm_metrics.json"
PREDICTIONS = ARTIFACTS / "llm_predictions.jsonl"
BUCKETS = ARTIFACTS / "llm_buckets.json"
MANIFEST = ARTIFACTS / "comparison_manifest.json"

TOLERANCE = 1e-12


@pytest.fixture(scope="module")
def report():
    if not REPORT.is_file():
        pytest.skip("the scored pass has not run yet")
    return split_module.load_manifest(REPORT)


@pytest.fixture(scope="module")
def records():
    if not PREDICTIONS.is_file():
        pytest.skip("the scored pass has not run yet")
    return llm_predict.read_records(PREDICTIONS)


@pytest.fixture(scope="module")
def frozen():
    return split_module.load_manifest(BUCKETS)


@pytest.fixture(scope="module")
def config():
    return config_module.load_config(CONFIG)


def test_the_committed_answers_are_the_frozen_ones(report, records):
    digest = hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest()
    assert digest == report["freeze"]["predictions_sha256"]
    assert len(records) == report["freeze"]["rows"]


def test_every_sealed_row_has_an_answer(records):
    sealed = {
        str(value)
        for value in split_module.load_manifest(MANIFEST)["sample_ids"]
    }
    answered = {str(record[llm_predict.SAMPLE_ID]) for record in records}
    assert answered == sealed


def test_no_answer_was_silently_dropped_or_substituted(records):
    for record in records:
        if record["status"] != "valid":
            assert record["fields"] is None
            assert record["failure_reason"]


def test_operational_numbers_recompute_from_the_cache(report, records):
    recomputed = llm_predict.operational_metrics(records)
    for key, value in report["operational"].items():
        assert recomputed[key] == pytest.approx(value, abs=TOLERANCE)


def test_metrics_recompute_from_the_cache_without_generating(
    report, records, frozen, config
):
    """The headline claim: saved answers reproduce every rate."""
    ladder = buckets.load_ladder(frozen)
    with storage.engine_scope(config) as engine:
        samples = storage.read_table(engine, storage.SAMPLES, config)

    identifiers = [str(record[llm_predict.SAMPLE_ID]) for record in records]
    chosen = samples.loc[
        samples[split_module.SAMPLE_ID_COLUMN]
        .astype(str)
        .isin(set(identifiers))
    ]
    indexed = chosen.set_index(
        chosen[split_module.SAMPLE_ID_COLUMN].astype(str)
    )
    targets = indexed[list(evaluate.TARGET_COLUMNS.values())].loc[identifiers]
    labels = gate.actual_labels(chosen, ladder).loc[identifiers]

    freeze = final_pass.Freeze(
        path=PREDICTIONS,
        sha256=str(report["freeze"]["predictions_sha256"]),
        rows=int(report["freeze"]["rows"]),
    )
    marked = final_pass.score(
        freeze,
        llm_predict.to_frame(records, ladder),
        targets,
        labels,
        config,
        float(frozen["representation_ceiling"]),
    )

    for key, value in report["rates"].items():
        assert marked["rates"][key] == pytest.approx(value, abs=TOLERANCE)


def test_the_workbook_carries_this_track_s_rows(report):
    csv = ARTIFACTS / "predictions_review.csv"
    if not csv.is_file():
        pytest.skip("the workbook has not been rendered yet")
    frame = pd.read_csv(csv)
    if "llm" not in set(frame["track"].astype(str)):
        pytest.skip("the workbook predates this track's scored pass")
    scored = frame.loc[frame["track"].astype(str) == "llm"]
    assert len(scored) == report["freeze"]["rows"]
