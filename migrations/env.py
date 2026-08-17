"""Alembic environment.

The connection URL is not written in `alembic.ini`. It is built from the
same configuration and the same environment variable the application
uses, so a migration can never be applied to a different database than
the one the pipeline writes to.
"""

from __future__ import annotations

import pathlib

from alembic import context
import sqlalchemy as sa

from facility_prediction import config as config_module
from facility_prediction import storage

_CONFIG_PATH = pathlib.Path("configs") / "default.yaml"

config = context.config
target_metadata = storage.metadata


def _url() -> str:
    """Return the connection URL, password included, for Alembic's use."""
    settings = config_module.load_config(
        pathlib.Path(config.get_main_option("app_config") or _CONFIG_PATH)
    )
    database = config.get_main_option("app_database") or None
    return storage.connection_url(settings, database).render_as_string(
        hide_password=False
    )


def run_migrations_offline() -> None:
    """Emit SQL for the target database without connecting to it."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    engine = sa.create_engine(_url(), future=True)
    with engine.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
