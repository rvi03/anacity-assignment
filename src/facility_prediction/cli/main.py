"""The entry point, and the stage order a full run performs.

    python -m facility_prediction.cli generate
    python -m facility_prediction.cli pipeline

This package is the composition root: it is the only place allowed to
import a single track's code, because assembling every track's
subcommands is the whole of its job. Asserted in tests/test_layering.py.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import logging

from facility_prediction import config as config_module
from facility_prediction.cli import paths
from facility_prediction.cli.commands import (
    analysis,
    llm_data,
    llm_scoring,
    llm_training,
    pipeline,
    review,
    verification,
)
from facility_prediction.cli.parsers import build_parser

_LOGGER = logging.getLogger(__name__)


def _dispatch(args: argparse.Namespace, config: config_module.Config) -> None:
    """Run the one stage the parsed arguments name.

    Args:
        args: Parsed command-line arguments.
        config: Validated configuration.
    """
    stages: dict[str, Callable[[], None]] = {
        "generate": lambda: pipeline.run_generate(
            config, args.export, args.summary
        ),
        "samples": lambda: pipeline.run_samples(config, args.summary),
        "split": lambda: pipeline.run_split(
            config, args.manifest, args.comparison
        ),
        "features": lambda: pipeline.run_features(config, args.manifest),
        "baselines": lambda: pipeline.run_baselines(config),
        "train": lambda: pipeline.run_train(config, args.models, args.metrics),
        "evaluate": lambda: pipeline.run_evaluate(
            config, args.models, args.metrics
        ),
        "review": lambda: review.run_review(
            config, args.workbook, args.csv, args.metrics
        ),
        "profile": lambda: analysis.run_profile(
            config, args.profile, args.plots
        ),
        "slices": lambda: analysis.run_slices(
            config, args.models, args.analysis, args.plots
        ),
        "freeze": lambda: verification.run_freeze(
            config, args.metrics, args.freeze
        ),
        "verify": lambda: verification.run_verify(
            config, args.metrics, args.csv
        ),
        "tune": lambda: analysis.run_tune(
            config, args.models, args.metrics, args.report, args.minutes
        ),
        "register": lambda: analysis.run_register(
            config, args.models, args.freeze, args.registry
        ),
        "score-holdout": lambda: verification.run_score_holdout(
            config, args.models, args.metrics, args.freeze, args.seal
        ),
        "llm-import-slice": lambda: llm_data.run_llm_import_slice(
            config, args.record
        ),
        "llm-buckets": lambda: llm_data.run_llm_buckets(
            config, args.llm_config, args.manifest
        ),
        "build-llm-data": lambda: llm_data.run_build_llm_data(
            config,
            args.llm_config,
            args.buckets,
            args.directory,
            args.manifest,
        ),
        "llm-decode-check": lambda: llm_data.run_llm_decode_check(
            config,
            args.llm_config,
            args.manifest,
            args.directory,
            args.record,
        ),
        "llm-pilot": lambda: llm_training.run_llm_pilot(
            config,
            args.llm_config,
            args.buckets,
            args.directory,
            args.adapter,
            args.record,
        ),
        "llm-ladder": lambda: llm_training.run_llm_ladder(
            config,
            args.llm_config,
            args.pilot,
            args.decode_check,
            args.decision,
        ),
        "llm-gate-manifest": lambda: llm_scoring.run_llm_gate_manifest(
            config, args.llm_config, args.manifest
        ),
        "llm-gate": lambda: llm_scoring.run_llm_gate(
            config,
            args.llm_config,
            llm_scoring._GatePaths(
                buckets=args.buckets,
                prompts=args.prompts,
                manifest=args.manifest,
                directory=args.directory,
                pilot=args.pilot,
                predictions=args.predictions,
                report=args.report,
            ),
            args.from_cache,
        ),
        "llm-review": lambda: review.run_llm_review(
            config, args.workbook, args.csv
        ),
        "llm-final": lambda: llm_scoring.run_llm_final(
            config,
            args.llm_config,
            llm_scoring._FinalPaths(
                buckets=args.buckets,
                prompts=args.prompts,
                manifest=args.manifest,
                adapter=args.adapter,
                predictions=args.predictions,
                report=args.report,
            ),
            args.from_cache,
        ),
        "llm-tier-b": lambda: llm_training.run_llm_tier_b(
            config,
            args.llm_config,
            args.buckets,
            args.prompts,
            args.decision,
            args.directory,
            args.adapter,
            args.record,
        ),
    }
    stages[args.command]()


def _run_pipeline(
    args: argparse.Namespace, config: config_module.Config
) -> None:
    """Run every stage in order, with this file's default paths.

    Args:
        args: Parsed command-line arguments; only the config path is
            read, because a whole-pipeline run writes the standard set
            of artifacts.
        config: Validated configuration.
    """
    del args
    pipeline.run_generate(
        config, paths.DEFAULT_BOOKINGS_EXPORT, paths.DEFAULT_GENERATION_SUMMARY
    )
    pipeline.run_samples(config, paths.DEFAULT_SAMPLE_SUMMARY)
    pipeline.run_split(
        config, paths.DEFAULT_SPLIT_MANIFEST, paths.DEFAULT_COMPARISON_MANIFEST
    )
    pipeline.run_features(config, paths.DEFAULT_FEATURE_MANIFEST)
    pipeline.run_baselines(config)
    pipeline.run_train(config, paths.DEFAULT_MODELS, paths.DEFAULT_METRICS)
    pipeline.run_evaluate(config, paths.DEFAULT_MODELS, paths.DEFAULT_METRICS)
    review.run_review(
        config,
        paths.DEFAULT_WORKBOOK,
        paths.DEFAULT_REVIEW_CSV,
        paths.DEFAULT_METRICS,
    )


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, load the configuration, and run the stage.

    Args:
        argv: Command-line arguments, defaulting to the process's own.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = config_module.load_config(args.config)

    if args.command == "pipeline":
        _run_pipeline(args, config)
    else:
        _dispatch(args, config)
