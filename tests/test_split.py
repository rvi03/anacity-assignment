"""Tests the chronological split's contract.

Covers what the frozen split promises: boundaries placed on elapsed
target time rather than row counts, partitions that cannot overlap in
time, identical target timestamps that stay together, cutoffs that
reproduce from the manifest alone, and a comparison manifest drawn from
the seed without touching a label.
"""

from __future__ import annotations

import copy
import pathlib
import zoneinfo

import pandas as pd
import pytest
import yaml

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples, split

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def smoke_samples(smoke_config) -> pd.DataFrame:
    bookings = generate.generate_bookings(smoke_config)
    return samples.build_samples(bookings, smoke_config)


@pytest.fixture(scope="module")
def cutoffs(smoke_samples, smoke_config) -> split.Cutoffs:
    return split.compute_cutoffs(smoke_samples, smoke_config)


@pytest.fixture(scope="module")
def labelled(smoke_samples, cutoffs) -> pd.DataFrame:
    return split.assign_split(smoke_samples, cutoffs)


def config_with(**overrides) -> config_module.Config:
    """The smoke config with `split` keys replaced."""
    raw = copy.deepcopy(yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8")))
    raw["split"].update(overrides)
    return config_module.parse_config(raw)


def tiny_samples(target_times: list[str]) -> pd.DataFrame:
    """A minimal sample table carrying only what the splitter reads."""
    return pd.DataFrame(
        {
            "sample_id": [f"S{i + 1:07d}" for i in range(len(target_times))],
            "target_booking_timestamp": pd.to_datetime(
                target_times, format="ISO8601", utc=True
            ).tz_convert(KOLKATA),
        }
    )


# --- the boundary is a time, not a row position ------


def test_cutoffs_divide_the_elapsed_span_by_the_configured_fractions(
    cutoffs, smoke_config
):
    span = cutoffs.end - cutoffs.start

    assert (
        cutoffs.train_cut - cutoffs.start
        == span * smoke_config.split.train_frac
    )
    assert cutoffs.val_cut - cutoffs.start == span * (
        smoke_config.split.train_frac + smoke_config.split.val_frac
    )


def test_split_is_not_a_row_count_quantile(labelled, smoke_config):
    share = (labelled["split"] == split.TRAIN).mean()

    assert share != pytest.approx(smoke_config.split.train_frac, abs=1e-6)


def test_partitions_do_not_overlap_in_time(labelled):
    split.check_boundaries(labelled)

    bounds = labelled.groupby("split")["target_booking_timestamp"].agg(
        ["min", "max"]
    )

    assert bounds.loc[split.TRAIN, "max"] < bounds.loc[split.VALIDATION, "min"]
    assert bounds.loc[split.VALIDATION, "max"] < bounds.loc[split.TEST, "min"]


def test_identical_target_timestamps_stay_in_one_partition():
    tied = "2024-03-01T00:00:00+05:30"
    frame = tiny_samples(
        [
            "2024-01-01T00:00:00+05:30",
            tied,
            tied,
            tied,
            "2024-05-01T00:00:00+05:30",
        ]
    )
    config = config_with(train_frac=0.25, val_frac=0.25)

    labels = split.assign_split(frame, split.compute_cutoffs(frame, config))

    assert set(labels.loc[1:3, "split"]) == {labels.loc[1, "split"]}


def test_a_sample_landing_exactly_on_a_cut_joins_the_later_partition():
    # span is 20 days, so train_cut is day 10 and val_cut is day 15
    frame = tiny_samples(
        [
            "2024-01-01T00:00:00+05:30",
            "2024-01-11T00:00:00+05:30",
            "2024-01-21T00:00:00+05:30",
        ]
    )
    config = config_with(train_frac=0.5, val_frac=0.25)

    labels = split.assign_split(frame, split.compute_cutoffs(frame, config))

    assert list(labels["split"]) == [
        split.TRAIN,
        split.VALIDATION,
        split.TEST,
    ]


def test_every_sample_lands_in_exactly_one_partition(labelled, smoke_samples):
    counts = labelled["split"].value_counts()

    assert counts.sum() == len(smoke_samples)
    assert set(counts.index) <= set(split.SPLIT_NAMES)


def test_cutoffs_are_timezone_aware(cutoffs):
    for instant in (cutoffs.start, cutoffs.train_cut, cutoffs.val_cut):
        assert instant.tzinfo is not None
        assert str(instant.tz) == "Asia/Kolkata"


def test_overlapping_partitions_are_rejected(labelled):
    broken = labelled.copy()
    broken.loc[broken.index[0], "split"] = split.TEST

    with pytest.raises(ValueError, match="strictly before"):
        split.check_boundaries(broken)


# --- the manifest is the frozen record ------


def test_manifest_reproduces_the_cutoffs_on_its_own(
    labelled, cutoffs, smoke_config, tmp_path
):
    manifest = split.build_split_manifest(labelled, cutoffs, smoke_config, {})
    path = tmp_path / "split_manifest.json"
    split.write_json(manifest, path)

    reloaded = split.Cutoffs.from_dict(split.load_manifest(path)["cutoffs"])

    assert reloaded == cutoffs


def test_manifest_reports_realised_counts_and_ranges(
    labelled, cutoffs, smoke_config
):
    manifest = split.build_split_manifest(labelled, cutoffs, smoke_config, {})

    for name in split.SPLIT_NAMES:
        rows = labelled.loc[labelled["split"] == name]
        assert manifest["splits"][name]["rows"] == len(rows)
        assert (
            manifest["splits"][name]["first"]
            == rows["target_booking_timestamp"].min().isoformat()
        )
    assert manifest["samples"] == len(labelled)


def test_manifest_write_is_deterministic(
    labelled, cutoffs, smoke_config, tmp_path
):
    manifest = split.build_split_manifest(labelled, cutoffs, smoke_config, {})

    first = split.write_json(manifest, tmp_path / "first.json")
    second = split.write_json(manifest, tmp_path / "second.json")

    assert first == second


# --- the comparison manifest, drawn blind ------


def test_comparison_manifest_is_capped_and_holdout_only(labelled, smoke_config):
    drawn = split.draw_comparison_manifest(labelled, smoke_config)

    holdout = set(labelled.loc[labelled["split"] == split.TEST, "sample_id"])
    assert len(drawn) <= smoke_config.split.comparison_rows
    assert set(drawn) <= holdout


def test_comparison_manifest_is_identical_on_a_redraw(labelled, smoke_config):
    first = split.draw_comparison_manifest(labelled, smoke_config)

    second = split.draw_comparison_manifest(labelled, smoke_config)

    assert first == second


def test_a_different_seed_draws_a_different_manifest(labelled, smoke_config):
    raw = copy.deepcopy(yaml.safe_load(SMOKE_PATH.read_text(encoding="utf-8")))
    raw["seed"] = smoke_config.seed + 1
    other = config_module.parse_config(raw)

    drawn = split.draw_comparison_manifest(labelled, other)

    assert drawn != split.draw_comparison_manifest(labelled, smoke_config)


def test_a_small_holdout_is_used_whole(labelled):
    config = config_with(comparison_rows=100_000)

    drawn = split.draw_comparison_manifest(labelled, config)

    holdout = labelled.loc[labelled["split"] == split.TEST, "sample_id"]
    assert set(drawn) == set(holdout)


def test_comparison_manifest_ignores_the_labels(labelled, smoke_config):
    shuffled = labelled.copy()
    shuffled["target_facility_id"] = shuffled["target_facility_id"].to_numpy()[
        ::-1
    ]

    assert split.draw_comparison_manifest(
        shuffled, smoke_config
    ) == split.draw_comparison_manifest(labelled, smoke_config)


# --- the splitter refuses input it cannot read ------


def test_an_empty_sample_table_is_rejected(smoke_config):
    empty = tiny_samples([]).iloc[0:0]

    with pytest.raises(ValueError, match="empty"):
        split.compute_cutoffs(empty, smoke_config)


def test_a_naive_target_timestamp_is_rejected(smoke_samples, smoke_config):
    naive = smoke_samples.copy()
    naive["target_booking_timestamp"] = naive[
        "target_booking_timestamp"
    ].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timezone-aware"):
        split.compute_cutoffs(naive, smoke_config)


def test_a_non_zero_embargo_is_rejected(smoke_samples):
    config = config_with(embargo_days=3)

    with pytest.raises(ValueError, match="embargo_days"):
        split.compute_cutoffs(smoke_samples, config)
