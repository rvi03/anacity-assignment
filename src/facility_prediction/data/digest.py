"""Canonical row digest — the identity of a table, not of a file.

A file hash proves two *files* are the same. Once the data lives in a
database that makes no promise about page bytes, ordering, or physical
layout, a file hash can only be taken of an export, and then it is
answering a question about the export rather than about the data.

This module answers the question directly. A table is reduced to one
SHA-256 over its rows in a declared order, with a declared column order
and a fixed rendering of every value:

    timestamps   ISO 8601 in UTC, so a session timezone cannot move them
    floats       fixed decimal places, so 0.1 + 0.2 renders once
    integers     plain decimal, never 3.0
    missing      the literal token below, distinct from an empty string

The digest is therefore invariant to storage format, to insertion order,
and to whether the rows arrived from a CSV, a DataFrame, or Postgres.
Two stores agreeing on it agree on the data.

Leakage contract: this module reads whatever frame it is handed and has
no notion of an origin. It orders and renders; it never filters.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

MISSING_TOKEN = "\\N"
_FLOAT_PLACES = 6
_FIELD_SEPARATOR = "\x1f"
_ROW_SEPARATOR = "\x1e"


def _render(value: Any) -> str:
    """Render one cell into its canonical text form.

    Args:
        value: Any cell value taken from a DataFrame.

    Returns:
        The canonical rendering. Missing values become
        :data:`MISSING_TOKEN`; timestamps are converted to UTC before
        formatting, so an equal instant renders equally regardless of
        the timezone it was carried in.

    Raises:
        ValueError: If a timestamp is timezone-naive. A naive instant
            has no canonical rendering, and guessing one would make the
            digest depend on the reader's locale.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return MISSING_TOKEN
    if isinstance(value, pd.Timestamp | pd.Period):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            msg = f"cannot digest a timezone-naive timestamp: {stamp!r}"
            raise ValueError(msg)
        return stamp.tz_convert("UTC").isoformat()
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{_FLOAT_PLACES}f}"
    return str(value)


def canonical_digest(
    frame: pd.DataFrame,
    sort_by: tuple[str, ...],
    columns: tuple[str, ...] | None = None,
) -> str:
    """Return the canonical SHA-256 of a table's contents.

    Args:
        frame: The table to digest.
        sort_by: Columns defining the row order. They must make the
            order total, or the digest is not reproducible; a duplicate
            key is therefore rejected rather than tolerated.
        columns: Columns to include, in the order they contribute. When
            omitted, every column of ``frame`` is used in its current
            order.

    Returns:
        The hex SHA-256 over the rendered rows.

    Raises:
        ValueError: If a named column is absent, or ``sort_by`` does not
            order the rows uniquely.
    """
    selected = list(columns) if columns is not None else list(frame.columns)
    missing = [name for name in (*sort_by, *selected) if name not in frame]
    if missing:
        msg = f"cannot digest, columns absent: {missing}"
        raise ValueError(msg)

    if frame.duplicated(subset=list(sort_by)).any():
        msg = (
            f"sort_by {sort_by} does not order the rows uniquely; the "
            "digest would depend on arrival order"
        )
        raise ValueError(msg)

    ordered = frame.sort_values(list(sort_by), kind="mergesort")
    hasher = hashlib.sha256()
    hasher.update(_FIELD_SEPARATOR.join(selected).encode("utf-8"))
    hasher.update(_ROW_SEPARATOR.encode("utf-8"))
    for row in ordered[selected].itertuples(index=False, name=None):
        line = _FIELD_SEPARATOR.join(_render(value) for value in row)
        hasher.update(line.encode("utf-8"))
        hasher.update(_ROW_SEPARATOR.encode("utf-8"))
    return hasher.hexdigest()
