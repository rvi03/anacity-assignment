"""Every path the pipeline writes to, declared once.

Output is split by whether it can be regenerated, because the two kinds
have opposite handling and mixing them is what made the repository hard
to publish:

``artifacts/``
    Committed. The deliverables and the recorded numbers — metrics,
    manifests, the review workbook, plots, the four fitted heads, and
    the frozen predictions that ``llm-reproduce`` replays. Twelve
    megabytes, all of it reviewed as evidence.

``runs/``
    Ignored by git. The adapter weights, the rendered training data,
    and trainer logs. A quarter of a gigabyte, every byte of it rebuilt
    from the seed, so committing it would ship noise and prove nothing.

Constants live here rather than beside the commands that consume them
so that a path is stated once. A command that invents its own path
puts a file somewhere the freeze, the verifier, and the reviewer do not
look.

Leakage contract: this module names files. It reads nothing and
computes nothing, so it holds no origin of its own.
"""

from __future__ import annotations

import pathlib

from facility_prediction.data import profiles, split
from facility_prediction.evaluation import ablations, errors, review_html
from facility_prediction.evaluation import freeze as freeze_module

DEFAULT_CONFIG = pathlib.Path("configs") / "default.yaml"
DEFAULT_LLM_CONFIG = pathlib.Path("configs") / "llm.yaml"

_DATA = pathlib.Path("data")
_ARTIFACTS = pathlib.Path("artifacts")
_RUNS = pathlib.Path("runs")

# ---------------------------------------------------------------- data

DEFAULT_BOOKINGS_EXPORT = _DATA / "synthetic_bookings.csv"

# ----------------------------------------------- artifacts: committed

DEFAULT_SAMPLE_SUMMARY = _ARTIFACTS / "sample_summary.json"
DEFAULT_SPLIT_MANIFEST = _ARTIFACTS / "split_manifest.json"
DEFAULT_COMPARISON_MANIFEST = _ARTIFACTS / "comparison_manifest.json"
DEFAULT_FEATURE_MANIFEST = _ARTIFACTS / "feature_manifest.json"
DEFAULT_GENERATION_SUMMARY = _ARTIFACTS / "generation_summary.json"
DEFAULT_GENERATION_PROFILE = _ARTIFACTS / profiles.PROFILE_FILENAME
DEFAULT_ERROR_ANALYSIS = _ARTIFACTS / errors.ANALYSIS_FILENAME
DEFAULT_FREEZE = _ARTIFACTS / freeze_module.FREEZE_FILENAME
DEFAULT_SEAL = _ARTIFACTS / freeze_module.SEAL_FILENAME
DEFAULT_REGISTRY = _ARTIFACTS / "model_registry.json"
DEFAULT_ABLATIONS = _ARTIFACTS / ablations.REPORT_FILENAME
DEFAULT_METRICS = _ARTIFACTS / "metrics.json"
DEFAULT_WORKBOOK = _ARTIFACTS / "predictions_review.xlsx"
DEFAULT_REVIEW_CSV = _ARTIFACTS / "predictions_review.csv"
DEFAULT_REVIEW_PAGE = _ARTIFACTS / review_html.PAGE_FILENAME
DEFAULT_PROFILE_PLOTS = _ARTIFACTS / "plots"

# 2.7 MB for all four heads, and the brief asks for the model
# artifacts, so these are committed rather than treated as run state.
DEFAULT_MODELS = _ARTIFACTS / "models"

DEFAULT_SLICE_IMPORT = _ARTIFACTS / "llm_slice_import.json"
DEFAULT_BUCKET_MANIFEST = _ARTIFACTS / "llm_buckets.json"
DEFAULT_PROMPT_MANIFEST = _ARTIFACTS / "prompt_manifest.json"
DEFAULT_DECODE_CHECK = _ARTIFACTS / "llm_decode_check.json"
DEFAULT_PILOT_RECORD = _ARTIFACTS / "llm_pilot.json"
DEFAULT_LADDER_DECISION = _ARTIFACTS / "ladder_decision.json"
DEFAULT_GATE_MANIFEST = _ARTIFACTS / "llm_gate_manifest.json"
DEFAULT_GATE_PREDICTIONS = _ARTIFACTS / "llm_predictions_gate.jsonl"
DEFAULT_GATE_REPORT = _ARTIFACTS / "llm_gate.json"
DEFAULT_TIER_B_RECORD = _ARTIFACTS / "llm_tier_b.json"
DEFAULT_LLM_PREDICTIONS = _ARTIFACTS / "llm_predictions.jsonl"
DEFAULT_LLM_METRICS = _ARTIFACTS / "llm_metrics.json"

# ------------------------------------------- runs: regenerable, ignored

DEFAULT_LLM_DATA = _RUNS / "llm_data"
DEFAULT_PILOT_ADAPTER = _RUNS / "llm_adapter" / "pilot"
DEFAULT_TIER_B_ADAPTER = _RUNS / "llm_adapter" / "tier_b"

# Committed files hashed into the import record, keyed by the field
# name each hash is recorded under.
SLICE_SOURCES = {
    "generation_summary_sha256": DEFAULT_GENERATION_SUMMARY,
    "sample_summary_sha256": DEFAULT_SAMPLE_SUMMARY,
    "split_manifest_sha256": DEFAULT_SPLIT_MANIFEST,
    "comparison_manifest_sha256": DEFAULT_COMPARISON_MANIFEST,
    "review_csv_sha256": DEFAULT_REVIEW_CSV,
}

PIPELINE_STAGES = (
    "generate",
    "samples",
    "split",
    "features",
    "baselines",
    "train",
    "evaluate",
    "review",
)

# The sealed split. Read exactly once per track, by `score-holdout`
# and by nothing else.
HOLDOUT_SPLIT = split.TEST

# Model selection reads this split and no other. The holdout is scored
# once per track, and not from here.
EVALUATION_SPLIT = split.VALIDATION

# The development split: the one every track may render at any time,
# whether or not it has spent its single holdout scoring. Which split the
# traditional workbook renders is decided by `review_split`.
REVIEW_SPLIT = split.VALIDATION

SEALED_REASON = (
    "The holdout split is sealed until each track scores it once, so this "
    "workbook renders the validation split. It is a real end-to-end "
    "result, not the final holdout result, and must not be read as one."
)
SCORED_REASON = (
    "The holdout has been scored once, so this workbook renders it. Every "
    "row here was predicted before any holdout target was read, and the "
    "settings that produced it were frozen beforehand."
)
LLM_REVIEW_REASON = (
    "Rows from the development split for every track, plus this track's "
    "scored holdout rows. The LLM branch has taken its single holdout "
    "pass, so those rows are no longer sealed to it; another track's "
    "holdout rows appear only once that track has scored too."
)

# The name the LLM track's scored result is recorded under, in the
# metrics file and in every cross-track comparison.
LLM_MODEL_NAME = "tier_b_adapter"
