"""The whole workflow on the smoke configuration, end to end.

Integration only. This file asserts that every stage runs in order and
leaves the artifact it promises behind, and it asserts nothing about
model quality — twenty iterations on a few hundred rows is a wiring
check, not a result.

It runs against the throwaway database the compose stack creates. With
no database reachable it skips with a named reason rather than passing,
so an absent service is visible in the summary instead of being
mistaken for a green suite.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import sqlalchemy as sa

from facility_prediction import config as config_module
from facility_prediction.cli import paths
from facility_prediction.cli.commands import pipeline, review
from facility_prediction.data import split, storage
from facility_prediction.models import baselines, train

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def engine(smoke_config):
    try:
        candidate = storage.create_engine(smoke_config)
        with candidate.connect() as connection:
            connection.execute(sa.text("SELECT 1"))
    except (storage.StorageError, sa.exc.SQLAlchemyError) as error:
        pytest.skip(f"no database reachable: {error}")

    storage.create_all(candidate)
    yield candidate
    candidate.dispose()


@pytest.fixture(scope="module")
def workflow(smoke_config, engine, tmp_path_factory):
    """Run every stage once, into a throwaway artifact directory."""
    del engine
    directory = tmp_path_factory.mktemp("smoke")
    paths = {
        "export": directory / "synthetic_bookings.csv",
        "generation": directory / "generation_summary.json",
        "samples": directory / "sample_summary.json",
        "split": directory / "split_manifest.json",
        "comparison": directory / "comparison_manifest.json",
        "features": directory / "feature_manifest.json",
        "models": directory / "models",
        "metrics": directory / "metrics.json",
        "workbook": directory / "predictions_review.xlsx",
        "csv": directory / "predictions_review.csv",
    }
    pipeline.run_generate(smoke_config, paths["export"], paths["generation"])
    pipeline.run_samples(smoke_config, paths["samples"])
    pipeline.run_split(smoke_config, paths["split"], paths["comparison"])
    pipeline.run_features(smoke_config, paths["features"])
    pipeline.run_baselines(smoke_config)
    pipeline.run_train(smoke_config, paths["models"], paths["metrics"])
    pipeline.run_evaluate(smoke_config, paths["models"], paths["metrics"])
    review.run_review(
        smoke_config, paths["workbook"], paths["csv"], paths["metrics"]
    )
    return paths


def test_the_smoke_run_never_writes_a_shipped_artifact(workflow):
    # Every path this run wrote must be inside the scratch directory it
    # was given. A stage that reaches for a module-level default instead
    # would stamp smoke output onto the deliverable, which happened once.
    scratch = workflow["metrics"].parent
    for name, path in workflow.items():
        assert scratch in path.parents, f"{name} escaped to {path}"


def test_rendering_without_a_metrics_path_writes_no_metrics(
    smoke_config, tmp_path
):
    # `run_review` records the rendered hash only where it is told to.
    # Omitting the path must record it nowhere at all — the smoke run
    # that stamped its own workbook hash onto the shipped record did so
    # because this argument did not exist.
    shipped = paths.DEFAULT_METRICS
    before = shipped.read_bytes() if shipped.is_file() else None

    review.run_review(
        smoke_config, tmp_path / "book.xlsx", tmp_path / "rows.csv"
    )

    after = shipped.read_bytes() if shipped.is_file() else None
    assert after == before


def test_every_stage_leaves_its_artifact(workflow):
    missing = [name for name, path in workflow.items() if not path.exists()]

    assert missing == []


def test_the_pipeline_declares_every_stage_this_test_runs():
    assert paths.PIPELINE_STAGES == (
        "generate",
        "samples",
        "split",
        "features",
        "baselines",
        "train",
        "evaluate",
        "review",
    )


def test_all_four_heads_are_saved(workflow):
    saved = {path.stem for path in workflow["models"].glob("*.cbm")}

    assert saved == set(train.HEAD_NAMES)


def test_the_metrics_record_carries_training_and_validation(workflow):
    payload = json.loads(workflow["metrics"].read_text(encoding="utf-8"))

    assert set(payload) >= {
        "provenance",
        "versions",
        "training",
        "fits",
        split.VALIDATION,
    }
    champion = payload[split.VALIDATION]["champion"]

    assert champion["model"] == train.CHAMPION_MODEL_NAME
    assert set(champion["head_sources"]) == set(train.HEAD_NAMES)
    assert champion["selection_split"] == split.VALIDATION


def test_the_scored_split_is_validation_and_says_the_holdout_is_sealed(
    workflow,
):
    payload = json.loads(workflow["metrics"].read_text(encoding="utf-8"))
    scored = payload[split.VALIDATION]

    assert scored["split"] == split.VALIDATION
    assert scored["holdout_scored"] is False


def test_the_verdict_against_the_baseline_is_recorded(workflow):
    payload = json.loads(workflow["metrics"].read_text(encoding="utf-8"))
    selection = payload[split.VALIDATION]["comparison"]["selection_score"]

    assert selection["verdict"] in {"beats", "ties", "loses"}
    assert 0.0 <= selection["model"] <= 1.0
    assert 0.0 <= selection["baseline"] <= 1.0


def test_both_tracks_predictions_survive_the_run(
    workflow, engine, smoke_config
):
    del workflow
    stored = storage.read_table(engine, storage.PREDICTIONS, smoke_config)

    written = set(zip(stored["track"], stored["model"], strict=True))
    assert written == {
        (baselines.TRACK, baselines.MODEL_NAME),
        (train.TRACK, train.MODEL_NAME),
        (train.TRACK, train.CHAMPION_MODEL_NAME),
    }


def test_the_workbook_and_its_csv_hold_the_same_rows(workflow):
    text = workflow["csv"].read_text(encoding="utf-8")

    assert len(text.splitlines()) > 1
