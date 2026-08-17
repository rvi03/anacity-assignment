"""Stretch variants, measured on validation, after the freeze.

The primary result is frozen and the holdout is spent. Everything here
is a question about what *else* might have worked, answered honestly and
too late to change what shipped. That ordering is deliberate: a variant
measured before the freeze could quietly become the shipped model
through nothing but repeated looking.

Three rules govern this module and each is enforced rather than
intended:

    validation only     no variant reads a holdout target. The split is
                        an argument and the caller passes validation.
    budget-bounded      the declared fit budget is a ceiling. The run
                        works the priority list top-down and stops at
                        the first item that would exceed it, so which
                        variants ran is deterministic rather than
                        whatever the clock allowed.
    primary untouched   nothing here writes a model file, a prediction
                        row, or `metrics.json`. It writes one report.

The priority order is the plan's, not this run's. Reordering it to put
a promising variant first would make the stopping point a choice about
results rather than a rule declared in advance.

Leakage contract: fits read the training rows the caller supplies and
score the validation rows it supplies. No holdout row is read, and no
artifact the submission depends on is written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import logging
import pathlib
import time
from typing import Any

import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.evaluation import evaluate
from facility_prediction.models import train

_LOGGER = logging.getLogger(__name__)

REPORT_FILENAME = "ablations.json"

# Wall-clock ceiling for a stretch run, in minutes.
#
# Deliberately NOT a key in `configs/default.yaml`. The configuration is
# frozen before the holdout is scored, and its hash is the freeze; a key
# added afterwards would invalidate that hash without being able to
# change a single shipped number. This governs one post-freeze
# diagnostic, it is recorded in the report it governs, and `cli tune`
# takes it as an argument for anyone who wants a different ceiling.
DEFAULT_MINUTES_BUDGET = 45.0


class AblationError(Exception):
    """Raised when a variant cannot be run as declared."""


@dataclasses.dataclass(frozen=True)
class Variant:
    """One declared alternative and what it costs to answer.

    Attributes:
        priority: Position in the plan's declared order. Lower runs
            first.
        name: Readable identifier, recorded in the report.
        question: What this variant is actually asking.
        fits: How many fits it costs against the budget.
        overrides: CatBoost settings that differ from the frozen ones.
        heads: Which heads to refit; the rest are reused frozen.
    """

    priority: int
    name: str
    question: str
    fits: int
    overrides: Mapping[str, Any]
    heads: tuple[str, ...]


# The plan's §9.6 order, with the items whose machinery exists. An item
# with no implementation is listed as skipped in the report rather than
# silently dropped — a budget that only counts what was convenient is
# not a budget.
DECLARED: tuple[Variant, ...] = (
    Variant(
        priority=1,
        name="notification_cadence_framing",
        question=(
            "Does bucketing the residual around a resident's own median "
            "booking gap beat bucketing the delay itself?"
        ),
        fits=1,
        overrides={"notification_framing": "cadence"},
        heads=(train.NOTIFICATION,),
    ),
    Variant(
        priority=2,
        name="notification_argmax_decode",
        question=(
            "Does taking the likeliest bucket beat integrating over the "
            "scored tolerance window?"
        ),
        fits=1,
        overrides={"notification_decode": "argmax"},
        heads=(train.NOTIFICATION,),
    ),
    Variant(
        priority=3,
        name="facility_ranking_framing",
        question=(
            "Does scoring one row per candidate facility beat scoring "
            "eight classes from one wide row?"
        ),
        fits=1,
        overrides={"facility_framing": "ranking"},
        heads=(train.FACILITY,),
    ),
    Variant(
        priority=4,
        name="deeper_trees",
        question="Does depth 8 beat depth 6 on the classifier heads?",
        fits=3,
        overrides={"depth": 8},
        heads=(train.FACILITY, train.USAGE_WEEKDAY, train.USAGE_HOUR),
    ),
    Variant(
        priority=5,
        name="slower_learning_rate",
        question=(
            "Does a slower rate under the same iteration ceiling beat "
            "the frozen rate?"
        ),
        fits=3,
        overrides={"learning_rate": 0.03},
        heads=(train.FACILITY, train.USAGE_WEEKDAY, train.USAGE_HOUR),
    ),
    Variant(
        priority=6,
        name="stronger_regularisation",
        question="Does more L2 on the leaves help the thin classes?",
        fits=3,
        overrides={"l2_leaf_reg": 20},
        heads=(train.FACILITY, train.USAGE_WEEKDAY, train.USAGE_HOUR),
    ),
)

# Declared in the plan but not implemented here. Named so the report
# says what was not asked, rather than implying the list was exhausted.
NOT_BUILT = {
    "class_weighting": "no weighting option exists on the heads",
    "chained_heads": "chaining is not implemented; L6 is untested",
    "resident_id_as_categorical": "the identifier policy excludes it",
    "hour_head_coarsened_to_bands": "no coarsened hour head exists",
    "ewma_halflife_60d": (
        "a feature-level variant; it would rebuild the feature table "
        "rather than refit a head"
    ),
}


def _with_overrides(
    config: config_module.Config, overrides: Mapping[str, Any]
) -> config_module.Config:
    """Return the configuration with one variant's settings applied.

    Args:
        config: The frozen configuration.
        overrides: CatBoost settings to change.

    Returns:
        A copy carrying those settings; the original is untouched.
    """
    return config.model_copy(
        update={"catboost": config.catboost.model_copy(update=dict(overrides))}
    )


def run_variant(
    variant: Variant,
    training_rows: pd.DataFrame,
    scoring_rows: pd.DataFrame,
    samples_table: pd.DataFrame,
    config: config_module.Config,
    frozen_heads: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit one variant's heads and score them on the supplied rows.

    Leakage contract: fits on ``training_rows`` and scores
    ``scoring_rows``. The caller supplies validation rows; no holdout
    row reaches either.

    Args:
        variant: The variant to run.
        training_rows: Feature rows to fit on.
        scoring_rows: Feature rows to score on.
        samples_table: The sample table, carrying targets.
        config: The frozen configuration.
        frozen_heads: The shipped estimators. Heads this variant does
            not refit are reused from here, because scoring reads all
            four outputs and only the refitted ones are being asked
            about.

    Returns:
        The variant's per-head match rates and its wall-clock cost.

    Raises:
        AblationError: If a head cannot be fitted as declared.
    """
    varied = _with_overrides(config, variant.overrides)
    fit_targets = train.align_targets(training_rows, samples_table)
    scored_targets = train.align_targets(scoring_rows, samples_table)

    started = time.monotonic()
    heads = dict(frozen_heads)
    calibration = None
    if train.notification_is_cadence(varied):
        calibration = train.fit_calibration(training_rows, fit_targets, varied)

    for name in variant.heads:
        try:
            design = train.build_design(
                name, training_rows, fit_targets, varied, calibration
            )
            # Each variant is fitted at the tree count of the head it
            # replaces. §9.6 asks for one-factor neighbours, and running
            # a fresh iteration search per variant would make every
            # comparison two factors at once — the setting under test
            # and a different iteration count — with neither
            # attributable. It is also what keeps a variant affordable:
            # searching to the ceiling on a 60-class residual grid costs
            # more than the whole rest of the list put together.
            iterations = int(frozen_heads[name].tree_count_)
            heads[name] = train.fit_head(
                name,
                design,
                varied,
                iterations=iterations,
                calibration=calibration,
            ).model
        except train.TrainingError as error:
            msg = f"variant {variant.name!r} could not fit {name!r}: {error}"
            raise AblationError(msg) from error

    predicted = train.predict(heads, scoring_rows, varied, calibration)
    joined = scored_targets.merge(predicted, on="sample_id", how="inner")

    matches: dict[str, float] = {}
    for name in variant.heads:
        column = train.PREDICTION_COLUMNS[name]
        if name == train.NOTIFICATION:
            clamped, _ = evaluate.clamp_delays(joined[column])
            matched = evaluate.notification_match(
                joined[train.TARGET_COLUMNS[name]],
                clamped,
                config.evaluation.notification_match_ratio,
            ).to_numpy()
        else:
            matched = (
                joined[train.TARGET_COLUMNS[name]].to_numpy()
                == joined[column].to_numpy()
            )
        matches[name] = round(float(pd.Series(matched).mean()), 4)

    return {
        "name": variant.name,
        "priority": variant.priority,
        "question": variant.question,
        "fits": variant.fits,
        "overrides": dict(variant.overrides),
        "heads": list(variant.heads),
        "matches": matches,
        "seconds": round(time.monotonic() - started, 2),
    }


def plan_run(
    budget: int, declared: Sequence[Variant] = DECLARED
) -> tuple[list[Variant], list[Variant], int]:
    """Work the priority list top-down and stop at the ceiling.

    Stopping at the *first* item that would exceed the budget, rather
    than skipping it for a cheaper one further down, is what makes the
    stopping point reproducible.

    Args:
        budget: Fits available.
        declared: The declared variants, in priority order.

    Returns:
        The variants that fit, those that did not, and the fits planned.
    """
    ordered = sorted(declared, key=lambda item: item.priority)
    chosen: list[Variant] = []
    spent = 0
    for index, variant in enumerate(ordered):
        if spent + variant.fits > budget:
            return chosen, list(ordered[index:]), spent
        chosen.append(variant)
        spent += variant.fits
    return chosen, [], spent


def would_exceed_time(
    elapsed_seconds: float,
    completed_fits: int,
    next_fits: int,
    minutes_budget: float,
) -> bool:
    """Return whether the next variant is projected past the time cap.

    The projection is the measured cost per fit so far, times the fits
    the next variant needs. It is a measurement, not an estimate made
    before the run: one variant here costs an order of magnitude more
    than another, so a fit count alone does not bound wall clock.

    Args:
        elapsed_seconds: Wall clock spent so far.
        completed_fits: Fits already spent.
        next_fits: Fits the next variant would cost.
        minutes_budget: The declared ceiling, in minutes.

    Returns:
        True when the next variant should not be started.

    Note:
        The first variant is never stopped, because there is nothing
        measured to project from. A single pathological variant can
        therefore overrun the ceiling once; the report records the
        elapsed time so that overrun is visible rather than implied.
    """
    if completed_fits <= 0:
        return False
    per_fit = elapsed_seconds / completed_fits
    projected = elapsed_seconds + per_fit * next_fits
    return projected > minutes_budget * 60.0


def build_report(
    completed: Sequence[Mapping[str, Any]],
    skipped: Sequence[Variant],
    budget: int,
    planned: int,
    baseline: Mapping[str, float],
    split_name: str,
    *,
    elapsed_seconds: float = 0.0,
    minutes_budget: float = 0.0,
    stopping_point: str = "",
) -> dict[str, Any]:
    """Assemble the record of what was asked and what was answered.

    Args:
        completed: Results from each variant that ran.
        skipped: Variants the budget did not reach.
        budget: The declared fit ceiling.
        planned: Fits the plan intended to spend.
        baseline: The frozen model's match rates, to compare against.
        split_name: Which split every number was measured on.
        elapsed_seconds: Wall clock the run spent.
        minutes_budget: The declared wall-clock ceiling.
        stopping_point: Why the run stopped, when time rather than the
            fit count ended it.

    Returns:
        The report payload, ready to serialise.
    """
    spent = sum(int(entry["fits"]) for entry in completed)
    improved = []
    for entry in completed:
        for head, value in entry["matches"].items():
            if head in baseline and value > baseline[head]:
                improved.append(
                    {
                        "variant": entry["name"],
                        "head": head,
                        "variant_match": value,
                        "frozen_match": baseline[head],
                        "delta": round(value - baseline[head], 4),
                    }
                )
    return {
        "split": split_name,
        "holdout_read": False,
        "primary_artifacts_changed": False,
        "budget": {
            "declared_fits": budget,
            "planned_fits": planned,
            "completed_fits": spent,
            "elapsed_seconds": round(elapsed_seconds, 1),
            "minutes_budget": minutes_budget,
            "stopping_point": stopping_point
            or (
                "every declared variant ran"
                if not skipped
                else f"stopped before {skipped[0].name!r}, which needs "
                f"{skipped[0].fits} more than the fit budget allows"
            ),
        },
        "frozen_reference": dict(baseline),
        "completed": list(completed),
        "skipped": [
            {
                "name": variant.name,
                "priority": variant.priority,
                "fits": variant.fits,
            }
            for variant in skipped
        ],
        "not_built": dict(NOT_BUILT),
        "improved_on_frozen": improved,
    }


def write_report(payload: Mapping[str, Any], path: pathlib.Path) -> str:
    """Write the ablation report and return its content hash.

    Args:
        payload: The report payload.
        path: Destination JSON; parent directories are created.

    Returns:
        The hex SHA-256 of the written bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
