"""The delay ladder, its merge pass, and its representation ceiling.

The ceiling is the number the branch is gated on, so most of what is
tested here is what must stop it: a gap in the partition, a bucket
wider than the match rule, a ladder that quietly loses long delays.
"""

from __future__ import annotations

import ast
import math
import pathlib

import numpy as np
import pandas as pd
import pytest

from facility_prediction.evaluation import evaluate
from facility_prediction.llm import buckets
from facility_prediction.llm import settings as settings_module

LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"
MODULE = pathlib.Path("src") / "facility_prediction" / "llm" / "buckets.py"
MATCH_RATIO = 1.25
SEED = 20260811


@pytest.fixture
def config():
    return settings_module.load_settings(LLM_CONFIG).notification_buckets


@pytest.fixture
def delays():
    generator = np.random.default_rng(SEED)
    drawn = generator.lognormal(mean=8.0, sigma=1.6, size=5000)
    return pd.Series(np.round(drawn, 1))


def test_the_shipped_config_fits_the_shared_match_rule(config):
    buckets.check_ratio(config, MATCH_RATIO)


def test_a_ratio_above_the_match_rule_squared_is_refused(config):
    widened = config.model_copy(update={"ratio": MATCH_RATIO**2 + 0.01})

    with pytest.raises(buckets.BucketError, match="exceeds the match ratio"):
        buckets.check_ratio(widened, MATCH_RATIO)


def test_the_ladder_partitions_every_non_negative_delay(config):
    ladder = buckets.build_ladder(config)

    buckets.check_partition(ladder)

    assert ladder[0].lower == buckets.TRANSFORM_OFFSET
    assert ladder[-1].upper is None


def test_every_finite_step_is_no_wider_than_configured(config):
    ladder = buckets.build_ladder(config)

    buckets.check_widths(ladder, config.ratio)

    widths = [b.width for b in ladder[1:-1]]
    assert max(widths) <= config.ratio


def test_a_gap_in_the_ladder_is_refused():
    gapped = [
        buckets.Bucket(label="a", lower=1.0, upper=10.0),
        buckets.Bucket(label="b", lower=11.0, upper=None),
    ]

    with pytest.raises(buckets.BucketError, match="ends at"):
        buckets.check_partition(gapped)


def test_a_bounded_top_bucket_is_refused():
    closed = [buckets.Bucket(label="a", lower=1.0, upper=10.0)]

    with pytest.raises(buckets.BucketError, match="top bucket is bounded"):
        buckets.check_partition(closed)


def test_every_random_delay_lands_in_exactly_one_bucket(config, delays):
    ladder = buckets.build_ladder(config)

    assigned = buckets.assign(delays, ladder)

    labels = {bucket.label for bucket in ladder}
    assert set(assigned) <= labels
    assert assigned.notna().all()
    assert len(assigned) == len(delays)


def test_a_delay_on_a_boundary_belongs_to_the_upper_bucket(config):
    ladder = buckets.build_ladder(config)
    edge = ladder[2].lower - buckets.TRANSFORM_OFFSET

    assigned = buckets.assign(pd.Series([edge]), ladder)

    assert assigned.iloc[0] == ladder[2].label


def test_a_negative_delay_is_refused(config):
    ladder = buckets.build_ladder(config)

    with pytest.raises(buckets.BucketError, match="negative delay"):
        buckets.assign(pd.Series([-1.0]), ladder)


def test_merging_leaves_no_bucket_below_the_threshold(config, delays):
    ladder = buckets.build_ladder(config)

    merged, log = buckets.merge_sparse(ladder, delays, config.min_train_rows)

    assert all(b.train_rows >= config.min_train_rows for b in merged)
    assert len(merged) == len(ladder) - len(log)
    buckets.check_partition(merged)


def test_merging_is_deterministic(config, delays):
    ladder = buckets.build_ladder(config)

    first, first_log = buckets.merge_sparse(
        ladder, delays, config.min_train_rows
    )
    second, second_log = buckets.merge_sparse(
        ladder, delays, config.min_train_rows
    )

    assert [b.label for b in first] == [b.label for b in second]
    assert first_log == second_log


def test_a_narrow_bucket_uses_its_geometric_midpoint(config, delays):
    ladder = buckets.resolve(
        buckets.build_ladder(config),
        delays,
        MATCH_RATIO,
        config.representative_decimals,
    )

    narrow = next(b for b in ladder if b.source == buckets.GEOMETRIC)
    expected = math.sqrt(narrow.lower * narrow.upper)
    assert narrow.representative == round(
        expected - buckets.TRANSFORM_OFFSET, config.representative_decimals
    )


def test_the_unbounded_ends_use_the_training_median(config, delays):
    ladder = buckets.resolve(
        buckets.build_ladder(config),
        delays,
        MATCH_RATIO,
        config.representative_decimals,
    )

    assert ladder[0].source == buckets.EMPIRICAL
    assert ladder[-1].source == buckets.EMPIRICAL


def test_a_geometric_midpoint_matches_both_edges_of_its_bucket(config):
    ladder = buckets.build_ladder(config)
    bucket = ladder[1]
    midpoint = math.sqrt(bucket.lower * bucket.upper)
    edges = pd.Series(
        [
            bucket.lower - buckets.TRANSFORM_OFFSET,
            bucket.upper - buckets.TRANSFORM_OFFSET,
        ]
    )

    matched = evaluate.notification_match(
        edges,
        pd.Series([midpoint - buckets.TRANSFORM_OFFSET] * 2),
        MATCH_RATIO,
    )

    assert matched.all()


def test_a_narrow_bucket_covers_all_of_its_training_rows(config, delays):
    ladder, _ = buckets.build(config, delays, MATCH_RATIO)
    _, per_bucket = buckets.representation_ceiling(ladder, delays, MATCH_RATIO)

    narrow = [b for b in ladder if b.source == buckets.GEOMETRIC]

    assert narrow
    assert all(per_bucket[b.label] == 1.0 for b in narrow)


def test_a_deliberately_widened_ladder_fails_the_gate(config, delays):
    coarse = config.model_copy(
        update={"ratio": MATCH_RATIO**2, "floor_minutes": 1440}
    )

    ladder, _ = buckets.build(coarse, delays, MATCH_RATIO)
    ceiling, _ = buckets.representation_ceiling(ladder, delays, MATCH_RATIO)

    with pytest.raises(buckets.BucketError, match="below the required"):
        buckets.check_ceiling(ceiling, config.ceiling_gate)


def test_the_ceiling_is_the_share_of_training_rows_covered(config, delays):
    ladder, _ = buckets.build(config, delays, MATCH_RATIO)

    ceiling, _ = buckets.representation_ceiling(ladder, delays, MATCH_RATIO)

    resolved = buckets.assign(delays, ladder).map(
        {b.label: b.representative for b in ladder}
    )
    matched = evaluate.notification_match(delays, resolved, MATCH_RATIO)
    assert ceiling == pytest.approx(matched.mean())


def test_an_empty_training_set_has_no_ceiling(config, delays):
    ladder, _ = buckets.build(config, delays, MATCH_RATIO)

    with pytest.raises(buckets.BucketError, match="without training delays"):
        buckets.representation_ceiling(
            ladder, pd.Series(dtype=float), MATCH_RATIO
        )


def test_the_manifest_records_every_label_and_its_merges(config, delays):
    ladder, log = buckets.build(config, delays, MATCH_RATIO)
    ceiling, per_bucket = buckets.representation_ceiling(
        ladder, delays, MATCH_RATIO
    )

    manifest = buckets.build_manifest(
        ladder, log, ceiling, per_bucket, config, MATCH_RATIO
    )

    assert manifest["labels"] == [b.label for b in ladder]
    assert len(manifest["buckets"]) == len(ladder)
    assert manifest["merges"] == log
    assert manifest["gate_passed"] == (ceiling >= config.ceiling_gate)


def test_no_label_or_edge_is_written_into_the_module():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    numbers = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    }

    # 0 and 1 are indices, 2 is the square in the match rule, 1440 is
    # minutes in a day. Anything else is an edge or a threshold, and
    # those come from configs/llm.yaml.
    assert numbers <= {0, 1, 2, buckets.MINUTES_PER_DAY}, (
        f"{MODULE.name} writes {sorted(numbers)} into the code; edges and "
        "thresholds come from configs/llm.yaml"
    )
