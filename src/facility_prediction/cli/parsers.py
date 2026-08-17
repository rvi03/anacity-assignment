"""The command line itself: one parser, assembled from every track.

Nothing else in the package owns an argument parser. Each subcommand
is declared here and dispatched in `main`, so the set of things a run
can be asked to do is readable in one place.
"""

from __future__ import annotations

import argparse
import pathlib

from facility_prediction.cli import paths
from facility_prediction.evaluation import ablations


def _add_llm_parsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    """Add this track's subcommands to the parser.

    They live in their own function because the entry point is a
    composition root: one track's subcommands are one unit of it.

    Args:
        sub: The subparser action every subcommand is added to.
        common: The parent parser carrying ``--config``.
    """
    slice_parser = sub.add_parser(
        "llm-import-slice",
        parents=[common],
        help="adopt the shared spine's measured facts",
    )
    slice_parser.add_argument(
        "--record", type=pathlib.Path, default=paths.DEFAULT_SLICE_IMPORT
    )

    bucket_parser = sub.add_parser(
        "llm-buckets",
        parents=[common],
        help="freeze the notification delay ladder",
    )
    bucket_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    bucket_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_BUCKET_MANIFEST
    )

    data_parser = sub.add_parser(
        "build-llm-data",
        parents=[common],
        help="render the train and validation prompt datasets",
    )
    data_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    data_parser.add_argument(
        "--buckets", type=pathlib.Path, default=paths.DEFAULT_BUCKET_MANIFEST
    )
    data_parser.add_argument(
        "--directory", type=pathlib.Path, default=paths.DEFAULT_LLM_DATA
    )
    data_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_PROMPT_MANIFEST
    )

    decode_parser = sub.add_parser(
        "llm-decode-check",
        parents=[common],
        help="run the pinned decoding stack against its fixtures",
    )
    decode_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    decode_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_PROMPT_MANIFEST
    )
    decode_parser.add_argument(
        "--directory", type=pathlib.Path, default=paths.DEFAULT_LLM_DATA
    )
    decode_parser.add_argument(
        "--record", type=pathlib.Path, default=paths.DEFAULT_DECODE_CHECK
    )

    pilot_parser = sub.add_parser(
        "llm-pilot",
        parents=[common],
        help="train the fixed pilot adapter and measure it",
    )
    pilot_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    pilot_parser.add_argument(
        "--buckets", type=pathlib.Path, default=paths.DEFAULT_BUCKET_MANIFEST
    )
    pilot_parser.add_argument(
        "--directory", type=pathlib.Path, default=paths.DEFAULT_LLM_DATA
    )
    pilot_parser.add_argument(
        "--adapter", type=pathlib.Path, default=paths.DEFAULT_PILOT_ADAPTER
    )
    pilot_parser.add_argument(
        "--record", type=pathlib.Path, default=paths.DEFAULT_PILOT_RECORD
    )

    ladder_parser = sub.add_parser(
        "llm-ladder",
        parents=[common],
        help="seal the full training size the compute cap allows",
    )
    ladder_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    ladder_parser.add_argument(
        "--pilot", type=pathlib.Path, default=paths.DEFAULT_PILOT_RECORD
    )
    ladder_parser.add_argument(
        "--decode-check", type=pathlib.Path, default=paths.DEFAULT_DECODE_CHECK
    )
    ladder_parser.add_argument(
        "--decision", type=pathlib.Path, default=paths.DEFAULT_LADDER_DECISION
    )

    gate_manifest_parser = sub.add_parser(
        "llm-gate-manifest",
        parents=[common],
        help="freeze the rows both development passes are judged on",
    )
    gate_manifest_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    gate_manifest_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_GATE_MANIFEST
    )

    gate_parser = sub.add_parser(
        "llm-gate",
        parents=[common],
        help="score zero-shot against the pilot adapter and decide",
    )
    gate_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    gate_parser.add_argument(
        "--buckets", type=pathlib.Path, default=paths.DEFAULT_BUCKET_MANIFEST
    )
    gate_parser.add_argument(
        "--prompts", type=pathlib.Path, default=paths.DEFAULT_PROMPT_MANIFEST
    )
    gate_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_GATE_MANIFEST
    )
    gate_parser.add_argument(
        "--directory", type=pathlib.Path, default=paths.DEFAULT_LLM_DATA
    )
    gate_parser.add_argument(
        "--pilot", type=pathlib.Path, default=paths.DEFAULT_PILOT_RECORD
    )
    gate_parser.add_argument(
        "--predictions",
        type=pathlib.Path,
        default=paths.DEFAULT_GATE_PREDICTIONS,
    )
    gate_parser.add_argument(
        "--report", type=pathlib.Path, default=paths.DEFAULT_GATE_REPORT
    )
    gate_parser.add_argument(
        "--from-cache",
        action="store_true",
        help="recompute from saved answers instead of generating",
    )


def _add_llm_final_parsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    """Add the full training run and the single scored pass.

    These two are separated from the development subcommands because
    they are the only ones that train at full size or open the holdout.

    Args:
        sub: The subparser action every subcommand is added to.
        common: The parent parser carrying ``--config``.
    """
    tier_b_parser = sub.add_parser(
        "llm-tier-b",
        parents=[common],
        help="train the one full adapter at the sealed ladder size",
    )
    tier_b_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    tier_b_parser.add_argument(
        "--buckets", type=pathlib.Path, default=paths.DEFAULT_BUCKET_MANIFEST
    )
    tier_b_parser.add_argument(
        "--prompts", type=pathlib.Path, default=paths.DEFAULT_PROMPT_MANIFEST
    )
    tier_b_parser.add_argument(
        "--decision", type=pathlib.Path, default=paths.DEFAULT_LADDER_DECISION
    )
    tier_b_parser.add_argument(
        "--directory", type=pathlib.Path, default=paths.DEFAULT_LLM_DATA
    )
    tier_b_parser.add_argument(
        "--adapter", type=pathlib.Path, default=paths.DEFAULT_TIER_B_ADAPTER
    )
    tier_b_parser.add_argument(
        "--record", type=pathlib.Path, default=paths.DEFAULT_TIER_B_RECORD
    )

    final_parser = sub.add_parser(
        "llm-final",
        parents=[common],
        help="score the frozen adapter once on the sealed holdout rows",
    )
    final_parser.add_argument(
        "--llm-config", type=pathlib.Path, default=paths.DEFAULT_LLM_CONFIG
    )
    final_parser.add_argument(
        "--buckets", type=pathlib.Path, default=paths.DEFAULT_BUCKET_MANIFEST
    )
    final_parser.add_argument(
        "--prompts", type=pathlib.Path, default=paths.DEFAULT_PROMPT_MANIFEST
    )
    final_parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=paths.DEFAULT_COMPARISON_MANIFEST,
    )
    final_parser.add_argument(
        "--adapter", type=pathlib.Path, default=paths.DEFAULT_TIER_B_ADAPTER
    )
    final_parser.add_argument(
        "--predictions",
        type=pathlib.Path,
        default=paths.DEFAULT_LLM_PREDICTIONS,
    )
    final_parser.add_argument(
        "--report", type=pathlib.Path, default=paths.DEFAULT_LLM_METRICS
    )
    final_parser.add_argument(
        "--from-cache",
        action="store_true",
        help="recompute from saved answers instead of generating",
    )

    review_parser = sub.add_parser(
        "llm-review",
        parents=[common],
        help="render the shared workbook with this track's holdout rows",
    )
    review_parser.add_argument(
        "--workbook", type=pathlib.Path, default=paths.DEFAULT_WORKBOOK
    )
    review_parser.add_argument(
        "--csv", type=pathlib.Path, default=paths.DEFAULT_REVIEW_CSV
    )
    review_parser.add_argument(
        "--page", type=pathlib.Path, default=paths.DEFAULT_REVIEW_PAGE
    )


def _add_freeze_parsers(
    sub: argparse._SubParsersAction[argparse.ArgumentParser],
    common: argparse.ArgumentParser,
) -> None:
    """Add the freeze, scoring, verification and registry commands.

    Kept out of :func:`build_parser` for the same reason the LLM
    parsers are: one function that declares every subcommand in the
    project grows without bound.

    Args:
        sub: The subparser action to register on.
        common: The shared parent parser.
    """
    register_parser = sub.add_parser(
        "register",
        parents=[common],
        help="register the frozen heads and query both tracks",
    )
    register_parser.add_argument(
        "--models", type=pathlib.Path, default=paths.DEFAULT_MODELS
    )
    register_parser.add_argument(
        "--freeze", type=pathlib.Path, default=paths.DEFAULT_FREEZE
    )
    register_parser.add_argument(
        "--registry", type=pathlib.Path, default=paths.DEFAULT_REGISTRY
    )

    tune_parser = sub.add_parser(
        "tune",
        parents=[common],
        help="post-freeze stretch variants, validation only",
    )
    tune_parser.add_argument(
        "--models", type=pathlib.Path, default=paths.DEFAULT_MODELS
    )
    tune_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )
    tune_parser.add_argument(
        "--report", type=pathlib.Path, default=paths.DEFAULT_ABLATIONS
    )
    tune_parser.add_argument(
        "--minutes", type=float, default=ablations.DEFAULT_MINUTES_BUDGET
    )

    verify_parser = sub.add_parser(
        "verify",
        parents=[common],
        help="recompute every committed value and compare",
    )
    verify_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )
    verify_parser.add_argument(
        "--csv", type=pathlib.Path, default=paths.DEFAULT_REVIEW_CSV
    )

    freeze_parser = sub.add_parser(
        "freeze",
        parents=[common],
        help="lock the config and the chosen sources before scoring",
    )
    freeze_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )
    freeze_parser.add_argument(
        "--freeze", type=pathlib.Path, default=paths.DEFAULT_FREEZE
    )

    holdout_parser = sub.add_parser(
        "score-holdout",
        parents=[common],
        help="score the sealed holdout ONCE against the freeze",
    )
    holdout_parser.add_argument(
        "--models", type=pathlib.Path, default=paths.DEFAULT_MODELS
    )
    holdout_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )
    holdout_parser.add_argument(
        "--freeze", type=pathlib.Path, default=paths.DEFAULT_FREEZE
    )
    holdout_parser.add_argument(
        "--seal", type=pathlib.Path, default=paths.DEFAULT_SEAL
    )


def build_parser() -> argparse.ArgumentParser:
    """Assemble the command-line interface.

    Returns:
        A parser with one subcommand per pipeline stage, plus
        ``pipeline`` for all of them in order. Every subcommand accepts
        ``--config``; there is one definition of that flag.
    """
    parser = argparse.ArgumentParser(
        prog="facility-prediction", description=__doc__
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config", type=pathlib.Path, default=paths.DEFAULT_CONFIG
    )

    sub = parser.add_subparsers(dest="command", required=True)

    generate_parser = sub.add_parser(
        "generate", parents=[common], help="build and store the dataset"
    )
    generate_parser.add_argument(
        "--export", type=pathlib.Path, default=paths.DEFAULT_BOOKINGS_EXPORT
    )
    generate_parser.add_argument(
        "--summary", type=pathlib.Path, default=paths.DEFAULT_GENERATION_SUMMARY
    )

    samples_parser = sub.add_parser(
        "samples", parents=[common], help="build rolling-origin samples"
    )
    samples_parser.add_argument(
        "--summary", type=pathlib.Path, default=paths.DEFAULT_SAMPLE_SUMMARY
    )

    split_parser = sub.add_parser(
        "split", parents=[common], help="freeze the chronological split"
    )
    split_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_SPLIT_MANIFEST
    )
    split_parser.add_argument(
        "--comparison",
        type=pathlib.Path,
        default=paths.DEFAULT_COMPARISON_MANIFEST,
    )

    features_parser = sub.add_parser(
        "features", parents=[common], help="build the feature table"
    )
    features_parser.add_argument(
        "--manifest", type=pathlib.Path, default=paths.DEFAULT_FEATURE_MANIFEST
    )

    sub.add_parser(
        "baselines", parents=[common], help="fit and score the baselines"
    )

    train_parser = sub.add_parser(
        "train", parents=[common], help="fit the four CatBoost heads"
    )
    train_parser.add_argument(
        "--models", type=pathlib.Path, default=paths.DEFAULT_MODELS
    )
    train_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )

    evaluate_parser = sub.add_parser(
        "evaluate",
        parents=[common],
        help="score the trained heads on validation, against the baseline",
    )
    evaluate_parser.add_argument(
        "--models", type=pathlib.Path, default=paths.DEFAULT_MODELS
    )
    evaluate_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )

    review_parser = sub.add_parser(
        "review", parents=[common], help="render the review workbook"
    )
    review_parser.add_argument(
        "--workbook", type=pathlib.Path, default=paths.DEFAULT_WORKBOOK
    )
    review_parser.add_argument(
        "--csv", type=pathlib.Path, default=paths.DEFAULT_REVIEW_CSV
    )
    review_parser.add_argument(
        "--metrics", type=pathlib.Path, default=paths.DEFAULT_METRICS
    )
    review_parser.add_argument(
        "--page", type=pathlib.Path, default=paths.DEFAULT_REVIEW_PAGE
    )
    profile_parser = sub.add_parser(
        "profile",
        parents=[common],
        help="generator rigour profile: quantiles, intervals, plots",
    )
    profile_parser.add_argument(
        "--profile", type=pathlib.Path, default=paths.DEFAULT_GENERATION_PROFILE
    )
    profile_parser.add_argument(
        "--plots", type=pathlib.Path, default=paths.DEFAULT_PROFILE_PLOTS
    )

    slices_parser = sub.add_parser(
        "slices",
        parents=[common],
        help="error analysis: slices, confusions, gain and SHAP",
    )
    slices_parser.add_argument(
        "--models", type=pathlib.Path, default=paths.DEFAULT_MODELS
    )
    slices_parser.add_argument(
        "--analysis", type=pathlib.Path, default=paths.DEFAULT_ERROR_ANALYSIS
    )
    slices_parser.add_argument(
        "--plots", type=pathlib.Path, default=paths.DEFAULT_PROFILE_PLOTS
    )

    _add_freeze_parsers(sub, common)

    sub.add_parser("pipeline", parents=[common], help="every stage, in order")

    _add_llm_parsers(sub, common)
    _add_llm_final_parsers(sub, common)
    return parser
