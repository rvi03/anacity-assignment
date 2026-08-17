"""The store of record — the only module that issues SQL.

Every table the pipeline produces lives in Postgres. Plain files are
exports of it, never the original, so there is one place where the data
is and one place that knows how to reach it.

    generate ──> bookings ──┐
    samples  ──> samples  ──┼──> read back as tz-aware DataFrames
    split    ──> splits   ──┤    by every later stage
    predict  ──> predictions┘

Two rules make this containable:

- **No SQL anywhere else.** Pipeline modules take and return DataFrames.
  `tests/test_layering.py` fails the build if another module imports
  SQLAlchemy or this module's engine.
- **Timestamps come back aware.** Postgres stores `timestamptz` as UTC.
  Reads convert into the configured zone, so no caller ever sees a naive
  or UTC-shifted instant and silently compares it against a local one.

Connection settings come from the configuration, except the password,
which comes from the environment. `.env` is read once if present, so the
service stack and the application take the same values from the same
file.

Leakage contract: this module moves rows; it never filters on time. A
caller asking for a table gets every row in it, and any as-of bound is
the caller's to apply.
"""

from __future__ import annotations

from collections.abc import Iterator
import contextlib
import logging
import os
import pathlib

import pandas as pd
import sqlalchemy as sa

from facility_prediction import config as config_module

_LOGGER = logging.getLogger(__name__)

BOOKINGS = "bookings"
SAMPLES = "samples"
SPLITS = "splits"
PREDICTIONS = "predictions"

_ENV_FILE = pathlib.Path(".env")
_TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    BOOKINGS: ("booking_timestamp", "usage_timestamp"),
    SAMPLES: ("origin", "target_booking_timestamp", "target_usage_timestamp"),
    SPLITS: (),
    PREDICTIONS: (),
}

metadata = sa.MetaData()

bookings_table = sa.Table(
    BOOKINGS,
    metadata,
    sa.Column("booking_id", sa.Text, primary_key=True),
    sa.Column("resident_id", sa.Text, nullable=False, index=True),
    sa.Column("facility_id", sa.Text, nullable=False),
    sa.Column(
        "booking_timestamp",
        sa.DateTime(timezone=True),
        nullable=False,
        index=True,
    ),
    sa.Column("usage_timestamp", sa.DateTime(timezone=True), nullable=False),
    sa.CheckConstraint(
        "booking_timestamp < usage_timestamp",
        name="ck_bookings_booked_before_used",
    ),
)

samples_table = sa.Table(
    SAMPLES,
    metadata,
    sa.Column("sample_id", sa.Text, primary_key=True),
    sa.Column("resident_id", sa.Text, nullable=False, index=True),
    sa.Column("origin", sa.DateTime(timezone=True), nullable=False),
    sa.Column("origin_booking_id", sa.Text, nullable=False),
    sa.Column("n_prior_bookings", sa.Integer, nullable=False),
    sa.Column("target_booking_id", sa.Text, nullable=False),
    sa.Column(
        "target_booking_timestamp",
        sa.DateTime(timezone=True),
        nullable=False,
        index=True,
    ),
    sa.Column(
        "target_usage_timestamp", sa.DateTime(timezone=True), nullable=False
    ),
    sa.Column("target_facility_id", sa.Text, nullable=False),
    sa.Column("target_usage_weekday", sa.Integer, nullable=False),
    sa.Column("target_usage_hour", sa.Integer, nullable=False),
    sa.Column("notification_delay_minutes", sa.Float, nullable=False),
    sa.CheckConstraint(
        "origin < target_booking_timestamp",
        name="ck_samples_origin_precedes_target",
    ),
)

splits_table = sa.Table(
    SPLITS,
    metadata,
    sa.Column("sample_id", sa.Text, primary_key=True),
    sa.Column("split", sa.Text, nullable=False, index=True),
)

predictions_table = sa.Table(
    PREDICTIONS,
    metadata,
    # `track` is the discriminator that lets both tracks write here
    # without either one overwriting the other.
    sa.Column("track", sa.Text, primary_key=True),
    sa.Column("model", sa.Text, primary_key=True),
    sa.Column("sample_id", sa.Text, primary_key=True),
    sa.Column("predicted_facility_id", sa.Text),
    sa.Column("predicted_usage_weekday", sa.Integer),
    sa.Column("predicted_usage_hour", sa.Integer),
    sa.Column("predicted_delay_minutes", sa.Float),
    sa.Column("predicted_transition_facility_id", sa.Text),
)


class StorageError(Exception):
    """Raised when the store cannot be reached or used as configured."""


def load_env_file(path: pathlib.Path | None = None) -> None:
    """Read ``.env`` into the environment without overriding it.

    The service stack reads this file natively; the application reads
    it here, so a port or credential is changed in one place. Variables
    already set win, so an explicit export overrides it.

    Args:
        path: The env file; defaults to ``.env`` in the working
            directory. A missing file is not an error.
    """
    source = _ENV_FILE if path is None else path
    if not source.is_file():
        return
    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def connection_url(
    config: config_module.Config, database: str | None = None
) -> sa.URL:
    """Build the connection URL for the application database.

    Args:
        config: Validated configuration; supplies every setting except
            the password.
        database: Database to connect to, overriding the configured one.
            Tests use this to reach the throwaway database.

    Returns:
        A SQLAlchemy URL. Rendering it hides the password.

    Raises:
        StorageError: If the password environment variable is unset.
    """
    load_env_file()
    settings = config.storage
    password = os.environ.get(settings.password_env)
    if not password:
        msg = (
            f"{settings.password_env} is unset; copy .env.example to .env "
            "or export it before connecting"
        )
        raise StorageError(msg)
    return sa.URL.create(
        drivername=settings.driver,
        username=settings.user,
        password=password,
        host=settings.host,
        port=settings.port,
        database=database or settings.database,
    )


def create_engine(
    config: config_module.Config, database: str | None = None
) -> sa.Engine:
    """Open a connection pool to the application database.

    Args:
        config: Validated configuration.
        database: Database to connect to, overriding the configured one.

    Returns:
        A SQLAlchemy engine. The caller disposes of it.
    """
    return sa.create_engine(connection_url(config, database), future=True)


@contextlib.contextmanager
def engine_scope(
    config: config_module.Config, database: str | None = None
) -> Iterator[sa.Engine]:
    """Hold an engine for the length of a block, then dispose of it.

    Keeps SQLAlchemy out of callers: the engine lifetime lives here
    with everything else that knows the database exists.

    Args:
        config: Validated configuration.
        database: Database to connect to, overriding the configured one.

    Yields:
        An open engine, disposed however the block exits.
    """
    engine = create_engine(config, database)
    try:
        yield engine
    finally:
        engine.dispose()


def create_all(engine: sa.Engine) -> None:
    """Create any missing table.

    Alembic owns schema evolution for a real database; this exists for
    a throwaway one.

    Args:
        engine: An open engine.
    """
    metadata.create_all(engine)


def write_table(engine: sa.Engine, name: str, frame: pd.DataFrame) -> int:
    """Replace a table's contents with ``frame``, in one transaction.

    Replacement rather than append: a stage owns its whole table, and
    the delete and the insert share a transaction, so a half-replaced
    table is never visible.

    Args:
        engine: An open engine.
        name: One of the module's table-name constants.
        frame: Rows to write; its columns must match the table's.

    Returns:
        The number of rows written.

    Raises:
        StorageError: If ``name`` is unknown or ``frame`` carries a
            column the table has no place for.
    """
    table = _table(name)
    known = {column.name for column in table.columns}
    unknown = [column for column in frame.columns if column not in known]
    if unknown:
        msg = f"{name} has no columns {unknown}"
        raise StorageError(msg)

    rows = frame.to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(sa.delete(table))
        if rows:
            connection.execute(sa.insert(table), rows)
    _LOGGER.info("wrote %d rows to %s", len(rows), name)
    return len(rows)


def write_predictions(
    engine: sa.Engine, frame: pd.DataFrame, *, track: str, model: str
) -> int:
    """Replace one model's prediction rows, leaving every other model's.

    The prediction table holds every track's output, so replacing the
    whole table would delete another track's rows. Only the rows this
    track and model wrote are removed, and the delete and the insert
    share a transaction.

    Args:
        engine: An open engine.
        frame: Prediction rows, carrying ``track`` and ``model``.
        track: The track whose rows are being replaced.
        model: The model whose rows are being replaced.

    Returns:
        The number of rows written.

    Raises:
        StorageError: If ``frame`` carries an unknown column, or a row
            names a different track or model than the one declared.
    """
    table = _table(PREDICTIONS)
    known = {column.name for column in table.columns}
    unknown = [column for column in frame.columns if column not in known]
    if unknown:
        msg = f"{PREDICTIONS} has no columns {unknown}"
        raise StorageError(msg)

    stray = frame.loc[(frame["track"] != track) | (frame["model"] != model)]
    if not stray.empty:
        msg = (
            f"{len(stray)} rows are not {track}/{model}; a write may only "
            "replace the rows it declares"
        )
        raise StorageError(msg)

    rows = frame.to_dict(orient="records")
    with engine.begin() as connection:
        connection.execute(
            sa.delete(table).where(
                table.c.track == track, table.c.model == model
            )
        )
        if rows:
            connection.execute(sa.insert(table), rows)
    _LOGGER.info(
        "wrote %d %s/%s rows to %s", len(rows), track, model, PREDICTIONS
    )
    return len(rows)


def read_table(
    engine: sa.Engine, name: str, config: config_module.Config
) -> pd.DataFrame:
    """Read a whole table back as a timezone-aware frame.

    Rows come back ordered by primary key. Without that, Postgres is
    free to return physical order, which changes whenever a table is
    rewritten — and any artifact rendered straight from the result, such
    as the review CSV, would then hash differently on a re-run while
    holding identical rows. Ordering here is what makes those artifacts
    byte-reproducible.

    Leakage contract: returns every row. Any as-of bound belongs to the
    caller, which is why this function takes no origin.

    Args:
        engine: An open engine.
        name: One of the module's table-name constants.
        config: Validated configuration; supplies the timezone that
            timestamps are converted into.

    Returns:
        The table's rows in primary-key order, with declared timestamp
        columns tz-aware in the configured zone and columns in the
        table's declared order.

    Raises:
        StorageError: If ``name`` is unknown.
    """
    table = _table(name)
    query = sa.select(table).order_by(*table.primary_key.columns)
    with engine.connect() as connection:
        frame = pd.read_sql(query, connection)
    for column in _TIMESTAMP_COLUMNS[name]:
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.tz_convert(
            config.tzinfo
        )
    return frame[[column.name for column in table.columns]]


def _table(name: str) -> sa.Table:
    """Look up a declared table by name.

    Args:
        name: Table name.

    Returns:
        The SQLAlchemy table.

    Raises:
        StorageError: If no such table is declared here.
    """
    if name not in metadata.tables:
        msg = f"unknown table {name!r}; known: {sorted(metadata.tables)}"
        raise StorageError(msg)
    return metadata.tables[name]
