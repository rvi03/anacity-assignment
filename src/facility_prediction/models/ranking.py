"""The facility output, framed as ranking candidates instead of classes.

A multiclass head learns eight separate one-vs-rest problems over one wide
row, and every one of them has to rediscover the same thing on its own:
that a resident books the facility they have been booking. This module
reshapes each sample into one row per candidate facility and moves that
facility's own history onto the row::

    one sample, 8 wide columns per family   ->  8 rows, 1 matched column
    facility_share_gym, facility_share_pool     candidate_facility_share
    ...                                         (whichever the row is for)

One shared scoring function is then fitted across every candidate, so
evidence about the eight facilities pools instead of splitting eight ways.
The families are discovered from the declared feature columns rather than
listed here: the facility catalog stays the single source of truth.

Leakage contract: this module reshapes an already-validated feature table
and reads no target except in :func:`candidate_labels`, which the caller
uses for training rows only. Nothing here reads a row other than the one
it is reshaping.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

import numpy as np
import numpy.typing as npt
import pandas as pd

from facility_prediction import config as config_module
from facility_prediction.features import features as features_module

CANDIDATE_COLUMN = "candidate_facility"
GROUP_COLUMN = "candidate_group"
SAMPLE_ID = "sample_id"

_MATCHED_PREFIX = "candidate_"
_RANKED_FAMILIES = (
    "facility_share",
    "ewma_facility_share",
    "transition_share_from_last",
)
_PERSONAL_FAMILY = "facility_share"
_RECENT_EWMA_FAMILY = "ewma_facility_share"
_COMMUNITY_FAMILY = "community_facility_share"
_COUNT_FAMILY = "facility_count"


class RankingError(Exception):
    """Raised when the feature table cannot be reshaped into candidates."""


def _slug(name: str) -> str:
    """Return the column-safe form of a facility name.

    Args:
        name: A facility name as written in configuration.

    Returns:
        The same slug :mod:`features` builds its column names from.
    """
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def facility_families(config: config_module.Config) -> tuple[str, ...]:
    """Return the feature families that carry one column per facility.

    Discovered rather than declared: a family is any prefix for which the
    numeric feature list holds exactly one column per catalog facility.
    Adding a facility to configuration therefore cannot leave a family
    half-reshaped.

    Args:
        config: Validated configuration.

    Returns:
        Family prefixes in declaration order.

    Raises:
        RankingError: If a prefix covers some facilities but not all,
            which would make the candidate row ill-defined.
    """
    slugs = [_slug(name) for name in config.facility_names]
    seen: dict[str, set[str]] = {}
    order: list[str] = []
    for column in features_module.numeric_feature_names(config):
        for slug in slugs:
            suffix = f"_{slug}"
            if not column.endswith(suffix):
                continue
            prefix = column[: -len(suffix)]
            if prefix not in seen:
                seen[prefix] = set()
                order.append(prefix)
            seen[prefix].add(slug)
            break
    partial = [name for name in order if len(seen[name]) != len(slugs)]
    if partial:
        msg = (
            "these feature families do not cover every facility and so "
            f"cannot be reshaped into candidates: {sorted(partial)}"
        )
        raise RankingError(msg)
    return tuple(order)


def shared_columns(config: config_module.Config) -> tuple[str, ...]:
    """Return the feature columns that do not name a facility.

    Args:
        config: Validated configuration.

    Returns:
        Every declared feature column that is not part of a per-facility
        family, in declared order.
    """
    slugs = [_slug(name) for name in config.facility_names]
    families = facility_families(config)
    per_facility = {f"{family}_{slug}" for family in families for slug in slugs}
    return tuple(
        column
        for column in features_module.feature_columns(config)
        if column not in per_facility
    )


def matched_columns(config: config_module.Config) -> tuple[str, ...]:
    """Return the candidate-specific columns the long table gains.

    Args:
        config: Validated configuration.

    Returns:
        The matched family columns, the within-row ranks and gaps, the
        preference-over-popularity contrasts, and the recency flags.
    """
    families = facility_families(config)
    ranked = tuple(f for f in _RANKED_FAMILIES if f in families)
    return (
        *(f"{_MATCHED_PREFIX}{family}" for family in families),
        *(f"{_MATCHED_PREFIX}rank_{family}" for family in ranked),
        *(f"{_MATCHED_PREFIX}gap_to_top_{family}" for family in ranked),
        f"{_MATCHED_PREFIX}share_over_community",
        f"{_MATCHED_PREFIX}is_unseen",
        *(
            f"{_MATCHED_PREFIX}is_{column}"
            for column in features_module.categorical_feature_names(config)[1:]
        ),
    )


def candidate_columns(config: config_module.Config) -> tuple[str, ...]:
    """Return every model-input column of the long table, in order.

    Args:
        config: Validated configuration.

    Returns:
        The shared feature columns, the candidate's identity, and the
        candidate-specific columns.
    """
    return (
        *shared_columns(config),
        CANDIDATE_COLUMN,
        *matched_columns(config),
    )


def candidate_categoricals(config: config_module.Config) -> tuple[str, ...]:
    """Return the long table's text-valued columns.

    Args:
        config: Validated configuration.

    Returns:
        The shared categorical columns plus the candidate's own name.
    """
    shared = set(shared_columns(config))
    return (
        *(
            name
            for name in features_module.categorical_feature_names(config)
            if name in shared
        ),
        CANDIDATE_COLUMN,
    )


def _family_grid(
    table: pd.DataFrame, family: str, slugs: Sequence[str]
) -> npt.NDArray[np.float64]:
    """Return one family's per-facility values as a row-by-facility grid.

    Args:
        table: A validated feature table.
        family: The family prefix.
        slugs: Facility slugs in catalog order.

    Returns:
        A ``(rows, facilities)`` array of float values.

    Raises:
        RankingError: If a family column is absent from the table.
    """
    columns = [f"{family}_{slug}" for slug in slugs]
    missing = [name for name in columns if name not in table.columns]
    if missing:
        msg = f"the feature table is missing candidate columns: {missing}"
        raise RankingError(msg)
    return table[columns].to_numpy(dtype=np.float64)


def build_candidates(
    table: pd.DataFrame, config: config_module.Config
) -> pd.DataFrame:
    """Reshape a feature table into one row per sample and facility.

    Leakage contract: reads feature columns and ``sample_id`` only. Every
    value on a candidate row was already present on the sample row it
    came from, so the reshape cannot introduce information the wide table
    did not already carry.

    Args:
        table: A validated feature table.
        config: Validated configuration; supplies the facility catalog.

    Returns:
        ``len(table) * len(facilities)`` rows carrying ``sample_id``, the
        group key, the candidate's name, and every model-input column.
        Rows of one sample are contiguous and in catalog order.

    Raises:
        RankingError: If a declared feature column is absent.
    """
    facilities = list(config.facility_names)
    slugs = [_slug(name) for name in facilities]
    width = len(facilities)
    rows = len(table)

    shared = list(shared_columns(config))
    missing = [name for name in shared if name not in table.columns]
    if missing:
        msg = f"the feature table is missing declared columns: {missing}"
        raise RankingError(msg)
    if SAMPLE_ID not in table.columns:
        msg = "the feature table carries no sample_id to group candidates by"
        raise RankingError(msg)

    repeat = np.repeat(np.arange(rows), width)
    long = table[shared].reset_index(drop=True).iloc[repeat]
    long = long.reset_index(drop=True)
    long.insert(0, SAMPLE_ID, np.repeat(table[SAMPLE_ID].to_numpy(), width))
    long.insert(1, GROUP_COLUMN, repeat)
    long[CANDIDATE_COLUMN] = np.tile(np.asarray(facilities, dtype=object), rows)

    families = facility_families(config)
    grids = {family: _family_grid(table, family, slugs) for family in families}
    for family, grid in grids.items():
        long[f"{_MATCHED_PREFIX}{family}"] = grid.reshape(-1)

    for family in _RANKED_FAMILIES:
        if family not in grids:
            continue
        grid = grids[family]
        rank = (-grid).argsort(axis=1).argsort(axis=1).astype(np.float64)
        long[f"{_MATCHED_PREFIX}rank_{family}"] = rank.reshape(-1)
        top = grid.max(axis=1, keepdims=True)
        long[f"{_MATCHED_PREFIX}gap_to_top_{family}"] = (top - grid).reshape(-1)

    long[f"{_MATCHED_PREFIX}share_over_community"] = (
        grids[_PERSONAL_FAMILY] - grids[_COMMUNITY_FAMILY]
    ).reshape(-1)
    long[f"{_MATCHED_PREFIX}is_unseen"] = (
        grids[_COUNT_FAMILY].reshape(-1) <= 0.0
    ).astype(np.float64)

    candidate = long[CANDIDATE_COLUMN].to_numpy()
    for column in features_module.categorical_feature_names(config)[1:]:
        repeated = np.repeat(table[column].astype(object).to_numpy(), width)
        long[f"{_MATCHED_PREFIX}is_{column}"] = (repeated == candidate).astype(
            np.float64
        )

    return long[[SAMPLE_ID, GROUP_COLUMN, *candidate_columns(config)]]


def candidate_labels(
    long: pd.DataFrame, targets: pd.Series, config: config_module.Config
) -> npt.NDArray[np.int64]:
    """Return the relevance label of every candidate row.

    Leakage contract: reads the target of the sample each candidate row
    belongs to. The caller must restrict both frames to training rows.

    Args:
        long: A candidate table from :func:`build_candidates`.
        targets: The booked facility of every sample, in the wide table's
            row order.
        config: Validated configuration; supplies the catalog width.

    Returns:
        One label per candidate row: ``1`` on the facility that was
        booked, ``0`` elsewhere.

    Raises:
        RankingError: If the two frames do not describe the same samples.
    """
    width = len(config.facility_names)
    if len(long) != len(targets) * width:
        msg = (
            f"{len(long)} candidate rows do not cover {len(targets)} "
            f"samples at {width} facilities each"
        )
        raise RankingError(msg)
    booked = np.repeat(targets.astype(object).to_numpy(), width)
    return (booked == long[CANDIDATE_COLUMN].to_numpy()).astype(np.int64)


def _score_grid(
    long: pd.DataFrame,
    scores: npt.ArrayLike,
    config: config_module.Config,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.object_]]:
    """Fold flat candidate scores back into one row per sample.

    Args:
        long: A candidate table from :func:`build_candidates`.
        scores: One score per candidate row, in that table's order.
        config: Validated configuration.

    Returns:
        The score grid and the matching facility-name grid.

    Raises:
        RankingError: If the score count does not match the table.
    """
    width = len(config.facility_names)
    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(flat) != len(long):
        msg = f"{len(flat)} scores for {len(long)} candidate rows"
        raise RankingError(msg)
    names = long[CANDIDATE_COLUMN].to_numpy().reshape(-1, width)
    return flat.reshape(-1, width), names


def rank_candidates(
    long: pd.DataFrame,
    scores: npt.ArrayLike,
    config: config_module.Config,
) -> pd.Series:
    """Return the top-scoring facility for every sample.

    Args:
        long: A candidate table from :func:`build_candidates`.
        scores: One score per candidate row, in that table's order.
        config: Validated configuration.

    Returns:
        The chosen facility per sample, indexed positionally in the wide
        table's row order.

    Raises:
        RankingError: If the score count does not match the table.
    """
    grid, names = _score_grid(long, scores, config)
    chosen = names[np.arange(len(grid)), grid.argmax(axis=1)]
    return pd.Series(chosen, dtype=object)


def candidate_probabilities(
    long: pd.DataFrame,
    scores: npt.ArrayLike,
    config: config_module.Config,
) -> pd.DataFrame:
    """Return per-facility probabilities from candidate scores.

    A ranker emits an unbounded score, but the reports want a comparable
    number per facility so that top-k accuracy means the same thing it
    does for a classifier. Scores are softmaxed within each sample, which
    preserves the ordering exactly and so changes no top-k answer.

    Args:
        long: A candidate table from :func:`build_candidates`.
        scores: One score per candidate row, in that table's order.
        config: Validated configuration.

    Returns:
        One row per sample and one column per facility, in catalog order.

    Raises:
        RankingError: If the score count does not match the table.
    """
    grid, _ = _score_grid(long, scores, config)
    shifted = np.exp(grid - grid.max(axis=1, keepdims=True))
    return pd.DataFrame(
        shifted / shifted.sum(axis=1, keepdims=True),
        columns=list(config.facility_names),
    )
