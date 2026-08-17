"""Four CatBoost heads over one shared feature table.

Each output gets its own estimator, and every one of them reads the same
feature table — but not all of them read it in the same shape:

    feature table  ->  facility        ranking over 8 candidate rows,
                                       or MultiClass over 8 classes
                   ->  usage weekday   CatBoostClassifier, MultiClass
                   ->  usage hour      CatBoostClassifier, MultiClass
                   ->  notification    residual bucket around the
                                       resident's own booking cadence,
                                       or an absolute time bucket

Two of those framings are configurable because the obvious framing is
not the one that measures best.

The facility head can be framed as ranking (:mod:`ranking`), where each
sample becomes one row per candidate facility carrying that facility's
own history. One scoring function is then shared across the catalog
instead of eight one-vs-rest problems each rediscovering the same thing.

The notification head can be framed around cadence (:mod:`cadence`).
Its target is a booking gap, scored by a multiplicative window, and the
resident's own median gap already carries most of what is knowable — so
that median becomes an offset read off the feature row, and the model
predicts only the residual around it. Both framings decode by
integrating a predicted density over the scored window rather than by
minimising raw-minute error on a highly skewed interval, because the
window is what evaluation actually rewards.

Nothing outside the training split is chosen while training. Each head's
iteration count is either the configured constant or, when the inner
search is on, the count that scored best on the latest slice of the
*training* rows. The saved model is then refit from scratch on every
training row to exactly that count: ``use_best_model`` stays false and no
evaluation set is passed to the fit that is kept, so there is no
checkpoint to pick and no way for the validation split to influence it.

Bootstrap is pinned per head rather than left to an objective-sensitive
default: the regressor takes the configured regression setting, the
three classifiers take the configured classifier setting.

Leakage contract: fitting reads the rows the caller supplies, and the
caller is required to hand over training rows only — audited against
the frozen split before any fit begins. Prediction reads features and
never a target. No column here is derived from the row being predicted.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import catboost
import numpy as np
import numpy.typing as npt
import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.features import features as features_module
from facility_prediction.models import cadence as cadence_module
from facility_prediction.models import ranking as ranking_module

_LOGGER = logging.getLogger(__name__)

TRACK = "traditional"
MODEL_NAME = "catboost"
CHAMPION_MODEL_NAME = "selected_heads"

FACILITY = "facility"
USAGE_WEEKDAY = "usage_weekday"
USAGE_HOUR = "usage_hour"
NOTIFICATION = "notification"
HEAD_NAMES = (FACILITY, USAGE_WEEKDAY, USAGE_HOUR, NOTIFICATION)

# Each head's target column in the sample table, and the prediction
# column it fills. The sampler names the targets; they are not retyped.
TARGET_COLUMNS = {
    FACILITY: "target_facility_id",
    USAGE_WEEKDAY: "target_usage_weekday",
    USAGE_HOUR: "target_usage_hour",
    NOTIFICATION: "notification_delay_minutes",
}
PREDICTION_COLUMNS = {
    FACILITY: "predicted_facility_id",
    USAGE_WEEKDAY: "predicted_usage_weekday",
    USAGE_HOUR: "predicted_usage_hour",
    NOTIFICATION: "predicted_delay_minutes",
}

_PROBABILITY_HEADS = (FACILITY, USAGE_WEEKDAY, USAGE_HOUR)
_MULTICLASS_LOSS = "MultiClass"
_MODEL_SUFFIX = ".cbm"
_SAMPLE_ID = "sample_id"
_ORDER_COLUMN = "target_booking_timestamp"
_WINDOW_DECODE = "window"
_DECODE_GRID_PER_DECADE = 60
_RANKING_FRAMING = "ranking"
_CADENCE_FRAMING = "cadence"
_ACCURACY_METRIC = "Accuracy"
_RANKING_METRIC = "NDCG:top=1"


# What the shipped repository actually verifies. Declared here rather
# than inferred from which files happen to exist, so it is reviewable in
# one place and cannot quietly upgrade itself.
#
# The band is A because one of the four deeper checks has not been
# built. Naming the three that have is more useful than a bare label,
# and naming the one that has not is the point: absent evidence is
# reported as absent.
VERIFICATION_BAND = {
    "band": "A",
    "band_b_checks": {
        "future_perturbation_property_test": "shipped",
        "shuffled_label_negative_control": "shipped",
        "extended_distribution_profiles": "shipped",
        "semantic_xlsx_style_comparison": "not built",
    },
    "note": (
        "Band A, not B: three of the four deeper checks ship and the "
        "fourth does not. The workbook is compared by content hash, not "
        "by cell style."
    ),
}


class TrainingError(Exception):
    """Raised when a fit is refused or a saved model cannot be read."""


def notification_bucket_edges(config: config_module.Config) -> np.ndarray:
    """Return the fixed multiplicative notification-bucket edges.

    The finite bucket width is checked against the evaluation tolerance,
    so the geometric midpoint of any finite bucket is always a matching
    prediction for every delay inside that bucket.

    Args:
        config: Validated configuration with bucket settings and the
            evaluation match ratio.

    Returns:
        Strictly increasing minute values between the configured floor
        and ceiling.  Values below and above the returned range form
        the two open-ended buckets.

    Raises:
        TrainingError: If one finite bucket would be wider than the
            scored tolerance can cover.
    """
    settings = config.catboost
    if settings.notification_bucket_ratio > (
        config.evaluation.notification_match_ratio**2
    ):
        msg = "notification bucket ratio exceeds the scored tolerance"
        raise TrainingError(msg)
    ceiling = float(settings.notification_bucket_ceiling_days * 24 * 60)
    edge = float(settings.notification_bucket_floor_minutes)
    edges = [edge]
    while edge * settings.notification_bucket_ratio < ceiling:
        edge *= settings.notification_bucket_ratio
        edges.append(edge)
    if edges[-1] != ceiling:
        edges.append(ceiling)
    return np.asarray(edges, dtype=np.float64)


def notification_bucket_labels(
    delays: pd.Series, config: config_module.Config
) -> pd.Series:
    """Map non-negative booking delays to deterministic bucket labels.

    Args:
        delays: Observed delay in minutes.
        config: Validated configuration naming the bucket edges.

    Returns:
        Integer bucket labels aligned to ``delays``.

    Raises:
        TrainingError: If a delay is negative.
    """
    values = delays.to_numpy(dtype=np.float64)
    if (values < 0).any():
        msg = "notification delays must be non-negative"
        raise TrainingError(msg)
    labels = np.digitize(values, notification_bucket_edges(config), right=False)
    return pd.Series(labels, index=delays.index, dtype="int64")


def notification_bucket_representatives(
    labels: npt.ArrayLike, config: config_module.Config
) -> np.ndarray:
    """Resolve bucket labels to non-negative submitted delays.

    Args:
        labels: Integer labels emitted by the notification classifier.
        config: Validated configuration naming the fixed bucket scheme.

    Returns:
        One geometric bucket representative per label.

    Raises:
        TrainingError: If a model emits a label outside the scheme.
    """
    edges = notification_bucket_edges(config)
    values = np.asarray(labels, dtype=np.int64)
    maximum = len(edges)
    if (values < 0).any() or (values > maximum).any():
        msg = "notification model emitted an unknown bucket"
        raise TrainingError(msg)
    representatives = np.empty(len(values), dtype=np.float64)
    ratio_root = np.sqrt(config.catboost.notification_bucket_ratio)
    representatives[values == 0] = edges[0] / ratio_root
    representatives[values == maximum] = edges[-1] * ratio_root
    for label in range(1, maximum):
        representatives[values == label] = np.sqrt(
            edges[label - 1] * edges[label]
        )
    return representatives


def notification_window_predictions(
    probabilities: npt.ArrayLike,
    labels: npt.ArrayLike,
    config: config_module.Config,
) -> np.ndarray:
    """Submit the delay whose scored window carries the most mass.

    The notification rule scores a prediction as a match when the actual
    delay falls inside a fixed multiplicative window around it. The
    submission that maximises the chance of a match is therefore the
    centre of the heaviest such window under the predicted bucket
    distribution, which is not in general the most likely bucket: two
    adjacent buckets can share a window that no single bucket dominates.

    Mass is integrated in log space, spread uniformly inside each bucket.

    Args:
        probabilities: One row per sample, one column per emitted label.
        labels: The bucket labels those columns stand for.
        config: Validated configuration; names the bucket scheme and the
            scored tolerance.

    Returns:
        One non-negative submitted delay per row.

    Raises:
        TrainingError: If a label falls outside the bucket scheme.
    """
    edges = notification_bucket_edges(config)
    ratio = config.catboost.notification_bucket_ratio
    columns = np.asarray(labels, dtype=np.int64)
    if (columns < 0).any() or (columns > len(edges)).any():
        msg = "notification model emitted an unknown bucket"
        raise TrainingError(msg)

    lower = np.log(np.concatenate([[edges[0] / ratio], edges]))[columns]
    upper = np.log(np.concatenate([edges, [edges[-1] * ratio]]))[columns]
    span = np.log(edges[-1] * ratio) - np.log(edges[0] / ratio)
    grid = np.linspace(
        np.log(edges[0] / ratio),
        np.log(edges[-1] * ratio),
        max(2, int(_DECODE_GRID_PER_DECADE * span / np.log(10.0))),
    )

    tolerance = np.log(config.evaluation.notification_match_ratio)
    overlap = (
        np.clip(
            np.minimum(grid[:, None] + tolerance, upper[None, :])
            - np.maximum(grid[:, None] - tolerance, lower[None, :]),
            0.0,
            None,
        )
        / (upper - lower)[None, :]
    )
    mass = np.asarray(probabilities, dtype=np.float64) @ overlap.T
    return np.exp(grid[np.argmax(mass, axis=1)])


@dataclasses.dataclass(frozen=True)
class Head:
    """One trained estimator and what it was trained with.

    Attributes:
        name: Which output this head predicts.
        model: The fitted CatBoost estimator.
        params: The settings CatBoost actually resolved for this fit.
        rows: Samples the fit read; a ranked head reads one row per
            candidate, but it still learns from this many samples.
        fit_seconds: Wall-clock seconds the fit took.
        calibration: The notification calibration this head decodes
            with, on the notification head under the cadence framing and
            None everywhere else.
    """

    name: str
    model: catboost.CatBoost
    params: dict[str, Any]
    rows: int
    fit_seconds: float
    calibration: cadence_module.Calibration | None = None

    def summary(self) -> dict[str, Any]:
        """Return what a reviewer needs to reproduce this head.

        Returns:
            The resolved settings, the realised tree count, the row
            count, and the fit duration.
        """
        return {
            "params": self.params,
            "trees": int(self.model.tree_count_),
            "rows": self.rows,
            "fit_seconds": round(self.fit_seconds, 3),
        }


def head_seed(config: config_module.Config, name: str) -> int:
    """Return the seed one head fits with.

    Seeds are derived from the root seed by the head's declared position
    when configuration asks for it, so the whole run is reproducible
    from one number and no two heads share a stream.

    Args:
        config: Validated configuration.
        name: One of :data:`HEAD_NAMES`.

    Returns:
        The seed for that head.

    Raises:
        TrainingError: If ``name`` is not a declared head.
    """
    if name not in HEAD_NAMES:
        msg = f"unknown head {name!r}; known: {list(HEAD_NAMES)}"
        raise TrainingError(msg)
    if not config.catboost.random_seed_from_root:
        return config.seed
    return config.seed + HEAD_NAMES.index(name)


def facility_is_ranked(config: config_module.Config) -> bool:
    """Return whether the facility head scores candidates rather than classes.

    Args:
        config: Validated configuration.

    Returns:
        True when the facility framing is ranking.
    """
    return config.catboost.facility_framing == _RANKING_FRAMING


def notification_is_cadence(config: config_module.Config) -> bool:
    """Return whether the notification head predicts a cadence residual.

    Args:
        config: Validated configuration.

    Returns:
        True when the notification framing is cadence.
    """
    return config.catboost.notification_framing == _CADENCE_FRAMING


def head_params(
    config: config_module.Config, name: str, iterations: int | None = None
) -> dict[str, Any]:
    """Return the settings one head is constructed with.

    Args:
        config: Validated configuration.
        name: One of :data:`HEAD_NAMES`.
        iterations: Iteration count for this head, when the inner search
            has chosen one. Defaults to the configured constant.

    Returns:
        Keyword arguments for the head's CatBoost estimator, including
        its pinned bootstrap and its derived seed. A ranked facility head
        takes the configured ranking objective and no bootstrap, because
        the pairwise objective samples within a group instead.

    Raises:
        TrainingError: If ``name`` is not a declared head.
    """
    settings = config.catboost
    ranked = name == FACILITY and facility_is_ranked(config)
    bootstrap = (
        settings.bootstrap.notification
        if name == NOTIFICATION
        else settings.bootstrap.classifiers
    )
    params: dict[str, Any] = {
        "loss_function": (
            settings.facility_ranking_loss if ranked else _MULTICLASS_LOSS
        ),
        "iterations": (
            settings.iterations if iterations is None else int(iterations)
        ),
        "depth": settings.depth,
        "learning_rate": settings.learning_rate,
        "l2_leaf_reg": settings.l2_leaf_reg,
        "random_seed": head_seed(config, name),
        "task_type": settings.task_type,
        "thread_count": settings.thread_count,
        "use_best_model": settings.use_best_model,
        "allow_writing_files": settings.allow_writing_files,
        "verbose": False,
    }
    if ranked:
        return params
    params["bootstrap_type"] = bootstrap.type
    if bootstrap.subsample is not None:
        params["subsample"] = bootstrap.subsample
    return params


def _estimator(
    name: str, params: Mapping[str, Any], config: config_module.Config
) -> catboost.CatBoost:
    """Construct the estimator one head uses.

    Args:
        name: One of :data:`HEAD_NAMES`.
        params: Settings from :func:`head_params`.
        config: Validated configuration; names each head's framing.

    Returns:
        A ranker for a ranked facility head, and a classifier for every
        other output. No head is a raw-minute regression: the
        notification head predicts a time bucket.
    """
    if name == FACILITY and facility_is_ranked(config):
        return catboost.CatBoostRanker(**params)
    return catboost.CatBoostClassifier(**params)


def model_matrix(
    table: pd.DataFrame, config: config_module.Config
) -> pd.DataFrame:
    """Return the columns a model may read, in declared order.

    Identifiers are dropped here: they are kept in the feature table so
    a reviewer can trace a row, and they are never model inputs.

    Args:
        table: A validated feature table.
        config: Validated configuration.

    Returns:
        The feature columns only.

    Raises:
        TrainingError: If a declared feature column is absent, or a
            column that reaches the model names a target.
    """
    columns = list(features_module.feature_columns(config))
    missing = [name for name in columns if name not in table.columns]
    if missing:
        msg = f"the feature table is missing declared columns: {missing}"
        raise TrainingError(msg)
    try:
        features_module.check_denylist(columns)
    except ValueError as error:
        raise TrainingError(str(error)) from error
    return table[columns]


def candidate_matrix(
    table: pd.DataFrame, config: config_module.Config
) -> pd.DataFrame:
    """Return the long candidate table a ranked facility head reads.

    Args:
        table: A validated feature table.
        config: Validated configuration.

    Returns:
        One row per sample and facility, carrying the group key and every
        candidate model-input column.

    Raises:
        TrainingError: If the table cannot be reshaped into candidates.
    """
    try:
        return ranking_module.build_candidates(table, config)
    except ranking_module.RankingError as error:
        raise TrainingError(str(error)) from error


@dataclasses.dataclass(frozen=True)
class Design:
    """The rows, labels and grouping one head is fitted on.

    A head's framing decides its shape, so the shape travels with the
    head rather than being reconstructed at every call site.

    Attributes:
        matrix: Model-input columns, one row per training row of that
            framing — a sample for a classifier, a candidate for a
            ranker.
        labels: The value that framing learns to predict.
        categoricals: Text-valued columns CatBoost must be told about.
        groups: Group key per row for a ranking head, or None.
        width: Rows of ``matrix`` per sample; one unless ranked.
    """

    matrix: pd.DataFrame
    labels: pd.Series
    categoricals: tuple[str, ...]
    groups: npt.NDArray[np.int64] | None = None
    width: int = 1

    def pool(self, labelled: bool = True) -> catboost.Pool:
        """Return this design as a CatBoost pool.

        Args:
            labelled: Whether to attach the labels; predicting does not.

        Returns:
            A pool with the categorical columns and any grouping declared.
        """
        return catboost.Pool(
            data=self.matrix,
            label=self.labels.to_numpy() if labelled else None,
            cat_features=list(self.categoricals),
            group_id=self.groups,
        )

    def slice(self, start: int, stop: int) -> Design:
        """Return the samples in a positional half-open range.

        Args:
            start: First sample position to keep.
            stop: One past the last sample position to keep.

        Returns:
            A design over those samples only, with whole groups kept.
        """
        lower, upper = start * self.width, stop * self.width
        return dataclasses.replace(
            self,
            matrix=self.matrix.iloc[lower:upper],
            labels=self.labels.iloc[lower:upper],
            groups=(None if self.groups is None else self.groups[lower:upper]),
        )

    def samples(self) -> int:
        """Return how many samples this design covers.

        Returns:
            The sample count, which is the row count for every head that
            is not ranked.
        """
        return len(self.matrix) // self.width


def build_design(
    name: str,
    table: pd.DataFrame,
    targets: pd.DataFrame,
    config: config_module.Config,
    calibration: cadence_module.Calibration | None = None,
) -> Design:
    """Return the shape one head is fitted on.

    Leakage contract: reads the features of the supplied rows and the
    target of those same rows. The caller restricts both to one split.

    Args:
        name: One of :data:`HEAD_NAMES`.
        table: Feature rows, in the order they will be fitted in.
        targets: Target columns aligned to ``table``'s row order.
        config: Validated configuration.
        calibration: Required for a cadence notification head; supplies
            the offset and the residual geometry.

    Returns:
        The design for that head.

    Raises:
        TrainingError: If ``name`` is not a declared head, or a cadence
            notification head is asked for without a calibration.
    """
    if name not in HEAD_NAMES:
        msg = f"unknown head {name!r}; known: {list(HEAD_NAMES)}"
        raise TrainingError(msg)
    column = targets[TARGET_COLUMNS[name]]

    if name == FACILITY and facility_is_ranked(config):
        long = candidate_matrix(table, config)
        width = len(config.facility_names)
        return Design(
            matrix=long[list(ranking_module.candidate_columns(config))],
            labels=pd.Series(
                ranking_module.candidate_labels(long, column, config)
            ),
            categoricals=ranking_module.candidate_categoricals(config),
            groups=long[ranking_module.GROUP_COLUMN].to_numpy(dtype=np.int64),
            width=width,
        )

    matrix = model_matrix(table, config)
    categoricals = features_module.categorical_feature_names(config)
    if name != NOTIFICATION:
        return Design(matrix=matrix, labels=column, categoricals=categoricals)

    if notification_is_cadence(config):
        if calibration is None:
            msg = "a cadence notification head needs a fitted calibration"
            raise TrainingError(msg)
        labels = cadence_module.residual_labels(table, column, calibration)
    else:
        labels = notification_bucket_labels(column, config)
    return Design(matrix=matrix, labels=labels, categoricals=categoricals)


def check_no_early_stopping(head: Head, config: config_module.Config) -> None:
    """Reject a fit that did not run the iteration count it was given.

    The count is the configured constant, or the one the inner search
    resolved for this head; either way the kept fit must have run all of
    it, because a shorter tree count would mean a checkpoint was chosen.

    Args:
        head: The fitted head.
        config: Validated configuration.

    Raises:
        TrainingError: If the tree count differs from the head's own
            resolved iteration count.
    """
    del config
    resolved = int(head.params["iterations"])
    if int(head.model.tree_count_) != resolved:
        msg = (
            f"head {head.name!r} kept {head.model.tree_count_} trees but "
            f"{resolved} iterations were resolved for it; no head may stop "
            "early or select a checkpoint"
        )
        raise TrainingError(msg)


def search_iterations(
    name: str,
    design: Design,
    config: config_module.Config,
) -> int:
    """Choose one head's iteration count on an inner training fold.

    Leakage contract: the fold is cut from the rows the caller supplies,
    which are training rows only. The estimator fitted here is thrown
    away; only the iteration count survives, and the kept model is refit
    on every training row.

    Rows are assumed to arrive in the caller's chronological order, so
    the fold is the latest slice — the same shape as the real split. A
    ranked head is cut on sample boundaries, so no group is split.

    Args:
        name: One of :data:`HEAD_NAMES`.
        design: The head's design over the training rows, in order.
        config: Validated configuration.

    Returns:
        The iteration count to refit with, at least one and never above
        the configured ceiling.
    """
    settings = config.catboost
    samples = design.samples()
    held = round(samples * settings.inner_validation_frac)
    if held < 1 or samples - held < 1:
        return settings.iterations

    cut = samples - held
    ranked = name == FACILITY and facility_is_ranked(config)
    inner = _estimator(
        name,
        {
            **head_params(config, name),
            "eval_metric": _RANKING_METRIC if ranked else _ACCURACY_METRIC,
            "od_type": "Iter",
            "od_wait": settings.inner_od_wait,
        },
        config,
    )
    inner.fit(
        design.slice(0, cut).pool(),
        eval_set=design.slice(cut, samples).pool(),
        use_best_model=True,
    )
    chosen = max(
        1, min(int(inner.get_best_iteration()) + 1, settings.iterations)
    )
    _LOGGER.info(
        "head %s: inner fold of %d samples chose %d of %d iterations",
        name,
        held,
        chosen,
        settings.iterations,
    )
    return chosen


def fit_head(
    name: str,
    design: Design,
    config: config_module.Config,
    iterations: int | None = None,
    calibration: cadence_module.Calibration | None = None,
) -> Head:
    """Fit one head on the rows it is given.

    Leakage contract: reads exactly the rows in ``design``. The caller is
    responsible for handing over training rows only; no evaluation set is
    passed, so no other partition can reach the fit even indirectly.

    Args:
        name: One of :data:`HEAD_NAMES`.
        design: The head's design over the training rows.
        config: Validated configuration.
        iterations: Iteration count for this head; defaults to the
            configured constant.
        calibration: Carried on a cadence notification head so that
            decoding can find it later.

    Returns:
        The fitted :class:`Head`.

    Raises:
        TrainingError: If there is nothing to fit on, or the fit did not
            run the iteration count it was given.
    """
    if design.matrix.empty:
        msg = f"cannot fit head {name!r} without training rows"
        raise TrainingError(msg)

    params = head_params(config, name, iterations)
    model = _estimator(name, params, config)
    started = time.perf_counter()
    model.fit(design.pool())
    elapsed = time.perf_counter() - started

    head = Head(
        name=name,
        model=model,
        params=params,
        rows=design.samples(),
        fit_seconds=elapsed,
        calibration=calibration,
    )
    check_no_early_stopping(head, config)
    _LOGGER.info(
        "fitted head %s on %d samples in %.1fs (%d trees)",
        name,
        head.rows,
        head.fit_seconds,
        int(model.tree_count_),
    )
    return head


def fit_calibration(
    table: pd.DataFrame,
    targets: pd.DataFrame,
    config: config_module.Config,
) -> cadence_module.Calibration | None:
    """Fit the notification calibration, when the framing needs one.

    Leakage contract: reads the delays of the supplied rows, which the
    caller must have restricted to the training split.

    Args:
        table: Feature rows for the training split.
        targets: Target columns aligned to ``table``'s row order.
        config: Validated configuration.

    Returns:
        The fitted calibration, or None under the absolute framing.

    Raises:
        TrainingError: If the calibration cannot be fitted.
    """
    if not notification_is_cadence(config):
        return None
    try:
        return cadence_module.fit_calibration(
            table, targets[TARGET_COLUMNS[NOTIFICATION]], config
        )
    except cadence_module.CadenceError as error:
        raise TrainingError(str(error)) from error


def fit(
    table: pd.DataFrame,
    targets: pd.DataFrame,
    config: config_module.Config,
) -> dict[str, Head]:
    """Fit all four heads over one shared feature table.

    Leakage contract: reads the supplied rows only, which the caller
    must have restricted to the training split. Targets are read for
    those rows and for no others.

    Args:
        table: Feature rows for the training split.
        targets: Sample rows carrying ``sample_id`` and the four target
            columns, restricted to the same rows.
        config: Validated configuration.

    Returns:
        Head name to fitted :class:`Head`, in declared order. The
        notification head carries its calibration when it has one.

    Raises:
        TrainingError: If the two frames do not describe the same rows.
    """
    aligned = align_targets(table, targets)
    calibration = fit_calibration(table, aligned, config)
    designs = {
        name: build_design(name, table, aligned, config, calibration)
        for name in HEAD_NAMES
    }
    carried = {NOTIFICATION: calibration}

    if not config.catboost.iteration_search:
        return {
            name: fit_head(name, designs[name], config, None, carried.get(name))
            for name in HEAD_NAMES
        }

    order = chronological_order(table, targets)
    ordered = {
        name: build_design(
            name,
            table.iloc[order],
            aligned.iloc[order],
            config,
            calibration,
        )
        for name in HEAD_NAMES
    }
    return {
        name: fit_head(
            name,
            designs[name],
            config,
            search_iterations(name, ordered[name], config),
            carried.get(name),
        )
        for name in HEAD_NAMES
    }


def chronological_order(
    table: pd.DataFrame, targets: pd.DataFrame
) -> np.ndarray:
    """Return positions that put the feature rows in booking order.

    Args:
        table: A feature table carrying ``sample_id``.
        targets: Sample rows carrying ``sample_id`` and, when the split
            basis is available, ``target_booking_timestamp``.

    Returns:
        Positional indices into ``table``. The table's own order is kept
        when the sample rows carry no timestamp to sort on.
    """
    if _ORDER_COLUMN not in targets.columns:
        return np.arange(len(table))
    stamps = targets.set_index(_SAMPLE_ID)[_ORDER_COLUMN]
    return np.argsort(stamps.loc[table[_SAMPLE_ID]].to_numpy(), kind="stable")


def align_targets(table: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Put the target rows in the feature table's order.

    Args:
        table: A feature table carrying ``sample_id``.
        targets: Sample rows carrying ``sample_id`` and the targets.

    Returns:
        The target columns, one row per feature row, in that order.

    Raises:
        TrainingError: If a feature row has no target row, or a target
            column is absent.
    """
    missing = [
        column
        for column in TARGET_COLUMNS.values()
        if column not in targets.columns
    ]
    if missing:
        msg = f"the sample table is missing target columns: {missing}"
        raise TrainingError(msg)

    indexed = targets.set_index(_SAMPLE_ID)
    unknown = [
        str(name) for name in table[_SAMPLE_ID] if name not in indexed.index
    ]
    if unknown:
        msg = (
            f"{len(unknown)} feature rows have no target row, first: "
            f"{unknown[:5]}"
        )
        raise TrainingError(msg)
    return indexed.loc[table[_SAMPLE_ID], list(TARGET_COLUMNS.values())]


def _scoring_pools(
    table: pd.DataFrame, config: config_module.Config
) -> tuple[catboost.Pool, pd.DataFrame | None]:
    """Return the pools a prediction reads, one shape per framing.

    Args:
        table: Feature rows to score.
        config: Validated configuration.

    Returns:
        The wide pool every classifier head reads, and the long candidate
        table a ranked facility head reads, or None when it is not.

    Raises:
        TrainingError: If the table cannot be prepared.
    """
    wide = catboost.Pool(
        data=model_matrix(table, config),
        cat_features=list(features_module.categorical_feature_names(config)),
    )
    if not facility_is_ranked(config):
        return wide, None
    return wide, candidate_matrix(table, config)


def _facility_scores(
    head: catboost.CatBoost,
    long: pd.DataFrame,
    config: config_module.Config,
) -> npt.NDArray[np.float64]:
    """Score every candidate row of a ranked facility head.

    Args:
        head: The fitted or reloaded ranker.
        long: The long candidate table.
        config: Validated configuration.

    Returns:
        One score per candidate row, in that table's order.
    """
    pool = catboost.Pool(
        data=long[list(ranking_module.candidate_columns(config))],
        cat_features=list(ranking_module.candidate_categoricals(config)),
        group_id=long[ranking_module.GROUP_COLUMN].to_numpy(dtype=np.int64),
    )
    return np.asarray(head.predict(pool), dtype=np.float64)


def predict(
    heads: Mapping[str, catboost.CatBoost],
    table: pd.DataFrame,
    config: config_module.Config,
    calibration: cadence_module.Calibration | None = None,
) -> pd.DataFrame:
    """Predict all four outputs for every row of a feature table.

    Leakage contract: reads feature columns only. No target of a scored
    row takes part in its own prediction, and the notification offset is
    a declared feature whose history stops at that row's own origin.

    Args:
        heads: Fitted or reloaded estimators, keyed by head name; pass
            :func:`estimators` when holding fitted heads.
        table: Feature rows to score.
        config: Validated configuration.
        calibration: Required under the cadence notification framing;
            :func:`load_calibration` reads the stored one.

    Returns:
        One row per input row, carrying ``sample_id`` and the four
        prediction columns.

    Raises:
        TrainingError: If a declared head is missing, or a cadence
            notification head is scored without its calibration.
    """
    absent = [name for name in HEAD_NAMES if name not in heads]
    if absent:
        msg = f"no fitted head for {absent}"
        raise TrainingError(msg)
    pool, long = _scoring_pools(table, config)

    delays: npt.NDArray[np.float64] | None = None
    if notification_is_cadence(config):
        if calibration is None:
            msg = (
                "a cadence notification head cannot be scored without its "
                "calibration"
            )
            raise TrainingError(msg)
        delays = cadence_module.decode(
            heads[NOTIFICATION].predict_proba(pool),
            np.asarray(heads[NOTIFICATION].classes_, dtype=np.int64),
            table,
            calibration,
            config.evaluation.notification_match_ratio,
        )

    window = config.catboost.notification_decode == _WINDOW_DECODE
    predicted: dict[str, Any] = {_SAMPLE_ID: table[_SAMPLE_ID].to_numpy()}
    for name in HEAD_NAMES:
        if name == FACILITY and long is not None:
            predicted[PREDICTION_COLUMNS[name]] = (
                ranking_module.rank_candidates(
                    long, _facility_scores(heads[name], long, config), config
                ).to_numpy()
            )
            continue
        if name == NOTIFICATION and delays is not None:
            predicted[PREDICTION_COLUMNS[name]] = delays
            continue
        if name == NOTIFICATION and window:
            predicted[PREDICTION_COLUMNS[name]] = (
                notification_window_predictions(
                    heads[name].predict_proba(pool),
                    np.asarray(heads[name].classes_, dtype=np.int64),
                    config,
                )
            )
            continue
        raw = np.asarray(heads[name].predict(pool)).reshape(-1)
        predicted[PREDICTION_COLUMNS[name]] = (
            notification_bucket_representatives(raw.astype(int), config)
            if name == NOTIFICATION
            else raw
        )
    frame = pd.DataFrame(predicted)
    frame[PREDICTION_COLUMNS[FACILITY]] = frame[
        PREDICTION_COLUMNS[FACILITY]
    ].astype(str)
    for name in (USAGE_WEEKDAY, USAGE_HOUR):
        frame[PREDICTION_COLUMNS[name]] = (
            frame[PREDICTION_COLUMNS[name]].astype(float).astype(int)
        )
    frame[PREDICTION_COLUMNS[NOTIFICATION]] = frame[
        PREDICTION_COLUMNS[NOTIFICATION]
    ].astype(float)
    return frame


def predict_probabilities(
    heads: Mapping[str, catboost.CatBoost],
    table: pd.DataFrame,
    config: config_module.Config,
) -> dict[str, pd.DataFrame]:
    """Return per-class probabilities for the three classifier heads.

    Leakage contract: reads feature columns only, exactly as
    :func:`predict` does.

    Args:
        heads: Fitted or reloaded estimators, keyed by head name.
        table: Feature rows to score.
        config: Validated configuration.

    Returns:
        Head name to a frame with one column per class label, in the
        input's row order. A ranked facility head has no class
        probabilities of its own, so its candidate scores are softmaxed
        within each sample; that preserves the ordering exactly, so every
        top-k answer is the ranking's own.

    Raises:
        TrainingError: If a classifier head is missing.
    """
    absent = [name for name in _PROBABILITY_HEADS if name not in heads]
    if absent:
        msg = f"no fitted head for {absent}"
        raise TrainingError(msg)

    pool, long = _scoring_pools(table, config)
    sheets = {
        name: pd.DataFrame(
            heads[name].predict_proba(pool),
            columns=list(heads[name].classes_),
        )
        for name in _PROBABILITY_HEADS
        if not (name == FACILITY and long is not None)
    }
    if long is not None:
        sheets[FACILITY] = ranking_module.candidate_probabilities(
            long, _facility_scores(heads[FACILITY], long, config), config
        )
    return {name: sheets[name] for name in _PROBABILITY_HEADS}


def select_head_sources(
    reports: Mapping[str, Mapping[str, Any]], candidates: tuple[str, ...]
) -> dict[str, str]:
    """Selects the validation winner for each output independently.

    Leakage contract: reads only already-computed validation reports.
    It never reads a target, feature, or prediction from the holdout
    split. The returned mapping must be frozen before holdout scoring.

    Args:
        reports: Candidate name to its validation evaluation report.
        candidates: Candidate names in deterministic tie-break order.

    Returns:
        Head name to the candidate supplying its prediction.

    Raises:
        TrainingError: If a candidate report or component match rate is
            absent.
    """
    if not candidates:
        msg = "cannot select heads without at least one candidate"
        raise TrainingError(msg)

    chosen: dict[str, str] = {}
    for head in HEAD_NAMES:
        rates: list[float] = []
        for candidate in candidates:
            try:
                rates.append(float(reports[candidate]["matches"][head]))
            except KeyError as error:
                msg = f"candidate {candidate!r} has no match rate for {head!r}"
                raise TrainingError(msg) from error
        chosen[head] = candidates[int(np.argmax(rates))]
    return chosen


def compose_selected_predictions(
    candidates: Mapping[str, pd.DataFrame], sources: Mapping[str, str]
) -> pd.DataFrame:
    """Composes one prediction row from independently selected heads.

    Leakage contract: reads candidate predictions and their sample IDs
    only. It does not read targets, and the caller must use a source map
    frozen from validation before scoring any holdout row.

    Args:
        candidates: Candidate name to rows with every prediction column.
        sources: Head name to the candidate selected for that head.

    Returns:
        ``sample_id`` plus the four selected prediction columns.

    Raises:
        TrainingError: If candidates are missing, have unequal sample
            IDs, or a selected source does not provide a head.
    """
    if not candidates:
        msg = "cannot compose predictions without candidates"
        raise TrainingError(msg)

    reference_name, reference = next(iter(candidates.items()))
    required = ["sample_id", *PREDICTION_COLUMNS.values()]
    missing = [name for name in required if name not in reference]
    if missing:
        msg = f"candidate {reference_name!r} has no columns {missing}"
        raise TrainingError(msg)

    result = pd.DataFrame({"sample_id": reference["sample_id"].to_numpy()})
    reference_ids = reference["sample_id"].to_numpy()
    for head in HEAD_NAMES:
        if head not in sources:
            msg = f"no selected source for head {head!r}"
            raise TrainingError(msg)
        source = sources[head]
        if source not in candidates:
            msg = f"selected source {source!r} is not a candidate"
            raise TrainingError(msg)
        frame = candidates[source]
        column = PREDICTION_COLUMNS[head]
        if column not in frame:
            msg = f"candidate {source!r} has no column {column!r}"
            raise TrainingError(msg)
        if not np.array_equal(frame["sample_id"].to_numpy(), reference_ids):
            msg = "candidate sample IDs must match in the same order"
            raise TrainingError(msg)
        result[column] = frame[column].to_numpy()
    return result


def estimators(
    heads: Mapping[str, Head],
) -> dict[str, catboost.CatBoost]:
    """Unwrap fitted heads to the estimators alone.

    Args:
        heads: Fitted heads, as returned by :func:`fit`.

    Returns:
        Head name to its estimator.
    """
    return {name: head.model for name, head in heads.items()}


def model_path(directory: pathlib.Path, name: str) -> pathlib.Path:
    """Return where one head's binary lives.

    Args:
        directory: The model directory.
        name: One of :data:`HEAD_NAMES`.

    Returns:
        The path to that head's saved model.
    """
    return directory / f"{name}{_MODEL_SUFFIX}"


def save(heads: Mapping[str, Head], directory: pathlib.Path) -> None:
    """Write every fitted head to disk.

    A cadence notification head carries a calibration that its estimator
    file cannot hold — the offset fallback and the residual geometry —
    so that is written beside the binaries and reloaded with them.

    Args:
        heads: Fitted heads, as returned by :func:`fit`.
        directory: Destination directory; created if absent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for name, head in heads.items():
        head.model.save_model(str(model_path(directory, name)))
        if head.calibration is not None:
            cadence_module.write_calibration(head.calibration, directory)
    _LOGGER.info("saved %d heads to %s", len(heads), directory)


def load(
    directory: pathlib.Path, config: config_module.Config
) -> dict[str, catboost.CatBoost]:
    """Read every head back from disk.

    Args:
        directory: The directory :func:`save` wrote to.
        config: Validated configuration; supplies each head's settings so
            a reloaded estimator is constructed the way it was fitted.

    Returns:
        Head name to the loaded estimator.

    Raises:
        TrainingError: If any head's file is absent.
    """
    loaded: dict[str, catboost.CatBoost] = {}
    for name in HEAD_NAMES:
        path = model_path(directory, name)
        if not path.is_file():
            msg = f"no saved model for head {name!r} at {path}"
            raise TrainingError(msg)
        model = _estimator(name, head_params(config, name), config)
        model.load_model(str(path))
        loaded[name] = model
    return loaded


def load_calibration(
    directory: pathlib.Path, config: config_module.Config
) -> cadence_module.Calibration | None:
    """Read the notification calibration, when the framing has one.

    Args:
        directory: The directory :func:`save` wrote to.
        config: Validated configuration; names the notification framing.

    Returns:
        The stored calibration, or None under the absolute framing.

    Raises:
        TrainingError: If the framing needs a calibration and none was
            written.
    """
    if not notification_is_cadence(config):
        return None
    try:
        return cadence_module.read_calibration(directory)
    except cadence_module.CadenceError as error:
        raise TrainingError(str(error)) from error


def build_metrics(
    heads: Mapping[str, Head],
    config: config_module.Config,
    provenance: Mapping[str, Any],
    audited_rows: int,
) -> dict[str, Any]:
    """Assemble the record of how these models were trained.

    Args:
        heads: Fitted heads, as returned by :func:`fit`.
        config: Validated configuration.
        provenance: Seed, timezone, config hash, and input digests.
        audited_rows: Rows the fit-call audit cleared.

    Returns:
        The metrics payload, ready to serialise.
    """
    notification_head = heads.get(NOTIFICATION)
    calibration = (
        None if notification_head is None else notification_head.calibration
    )
    return {
        "provenance": dict(provenance),
        "versions": {"catboost": catboost.__version__},
        "training": {
            "model": MODEL_NAME,
            "track": TRACK,
            "iterations": config.catboost.iterations,
            "iteration_search": config.catboost.iteration_search,
            "inner_validation_frac": config.catboost.inner_validation_frac,
            "searched_iterations": {
                name: int(heads[name].params["iterations"])
                for name in HEAD_NAMES
                if name in heads
            },
            "notification_decode": config.catboost.notification_decode,
            "notification_framing": config.catboost.notification_framing,
            "facility_framing": config.catboost.facility_framing,
            "notification_calibration": (
                None if calibration is None else calibration.to_dict()
            ),
            "use_best_model": config.catboost.use_best_model,
            "early_stopping": False,
            "evaluation_set_passed": False,
            "task_type": config.catboost.task_type,
            "thread_count": config.catboost.thread_count,
            "seed": config.seed,
            "audited_fit_rows": audited_rows,
            "categorical_features": list(
                features_module.categorical_feature_names(config)
            ),
            "heads": {
                name: heads[name].summary()
                for name in HEAD_NAMES
                if name in heads
            },
        },
        "fits": {
            "primary": len(heads),
            "stretch_budget": config.catboost.stretch_fit_budget,
            "stretch_completed": 0,
            "stopping_point": "primary heads only; no stretch fit ran",
        },
        "verification": VERIFICATION_BAND,
    }


def write_metrics(payload: Mapping[str, Any], path: pathlib.Path) -> None:
    """Write the metrics record.

    Args:
        payload: The metrics payload.
        path: Destination JSON; parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def flat_params(heads: Mapping[str, Head]) -> dict[str, Any]:
    """Flatten every head's resolved settings for a run log.

    Args:
        heads: Fitted heads, as returned by :func:`fit`.

    Returns:
        ``head.setting`` to value, for every head and setting.
    """
    flat: dict[str, Any] = {}
    for name, head in heads.items():
        for key, value in head.params.items():
            flat[f"{name}.{key}"] = value
    return flat
