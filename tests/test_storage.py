"""Storage round-trips, timezone handling, and digest stability.

These run against a real Postgres — the throwaway `facility_test`
database created by the compose stack. An in-memory substitute would
exercise none of what is actually being tested: the dialect, the
timestamptz round-trip, or the numeric types.

When no database is reachable the tests skip with a named marker rather
than passing, so an absent database is visible in the run summary
instead of being mistaken for a green suite.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
import sqlalchemy as sa

from facility_prediction import config as config_module
from facility_prediction.data import digest, generate, samples, storage

SMOKE_CONFIG = pathlib.Path("configs") / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke_config():
    return config_module.load_config(SMOKE_CONFIG)


@pytest.fixture(scope="module")
def engine(smoke_config):
    try:
        candidate = storage.create_engine(smoke_config)
        with candidate.connect() as connection:
            connection.execute(sa.text("SELECT 1"))
    except (storage.StorageError, sa.exc.SQLAlchemyError) as error:
        pytest.skip(f"no database reachable: {error}")

    storage.create_all(candidate)
    yield candidate
    candidate.dispose()


@pytest.fixture
def bookings(smoke_config):
    return generate.generate_bookings(smoke_config)


def test_bookings_round_trip_preserves_values(engine, smoke_config, bookings):
    storage.write_table(engine, storage.BOOKINGS, bookings)

    stored = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    assert len(stored) == len(bookings)
    assert list(stored.columns) == list(generate.BOOKING_COLUMNS)
    assert generate.bookings_digest(stored) == generate.bookings_digest(
        bookings
    )


def test_timestamps_come_back_aware_in_the_configured_zone(
    engine, smoke_config, bookings
):
    storage.write_table(engine, storage.BOOKINGS, bookings)

    stored = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    for column in ("booking_timestamp", "usage_timestamp"):
        assert isinstance(stored[column].dtype, pd.DatetimeTZDtype)
        assert str(stored[column].dt.tz) == smoke_config.timezone


def test_writing_twice_replaces_rather_than_appends(
    engine, smoke_config, bookings
):
    storage.write_table(engine, storage.BOOKINGS, bookings)
    storage.write_table(engine, storage.BOOKINGS, bookings)

    stored = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    assert len(stored) == len(bookings)


def test_digest_ignores_insertion_order(engine, smoke_config, bookings):
    storage.write_table(engine, storage.BOOKINGS, bookings)
    forwards = storage.read_table(engine, storage.BOOKINGS, smoke_config)
    storage.write_table(engine, storage.BOOKINGS, bookings.iloc[::-1])

    backwards = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    assert generate.bookings_digest(backwards) == generate.bookings_digest(
        forwards
    )


def test_rows_come_back_in_the_same_order_after_a_rewrite(
    engine, smoke_config, bookings
):
    storage.write_table(engine, storage.BOOKINGS, bookings)
    forwards = storage.read_table(engine, storage.BOOKINGS, smoke_config)
    storage.write_table(engine, storage.BOOKINGS, bookings.iloc[::-1])

    backwards = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    pd.testing.assert_frame_equal(backwards, forwards)


def test_rows_come_back_ordered_by_primary_key(engine, smoke_config, bookings):
    storage.write_table(engine, storage.BOOKINGS, bookings.iloc[::-1])

    stored = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    assert list(stored["booking_id"]) == sorted(stored["booking_id"])


def test_exported_file_carries_the_stored_digest(
    engine, smoke_config, bookings, tmp_path
):
    storage.write_table(engine, storage.BOOKINGS, bookings)
    stored = storage.read_table(engine, storage.BOOKINGS, smoke_config)

    path = tmp_path / "export.csv"
    generate.write_bookings(stored, path)
    reloaded = samples.load_bookings(path, smoke_config)

    assert generate.bookings_digest(reloaded) == generate.bookings_digest(
        stored
    )


def test_unknown_column_is_refused(engine, bookings):
    extra = bookings.assign(hidden_archetype="early_bird")

    with pytest.raises(storage.StorageError, match="hidden_archetype"):
        storage.write_table(engine, storage.BOOKINGS, extra)


def test_unknown_table_is_refused(engine):
    with pytest.raises(storage.StorageError, match="unknown table"):
        storage.write_table(engine, "not_a_table", pd.DataFrame())


def test_missing_password_names_the_variable(smoke_config, monkeypatch):
    monkeypatch.delenv(smoke_config.storage.password_env, raising=False)
    monkeypatch.setattr(storage, "_ENV_FILE", pathlib.Path("absent.env"))

    with pytest.raises(storage.StorageError, match="POSTGRES_PASSWORD"):
        storage.connection_url(smoke_config)


def test_naive_timestamp_has_no_canonical_rendering():
    frame = pd.DataFrame(
        {"id": ["a"], "when": [pd.Timestamp("2026-01-01T00:00:00")]}
    )

    with pytest.raises(ValueError, match="timezone-naive"):
        digest.canonical_digest(frame, sort_by=("id",))


def test_a_non_unique_sort_key_is_refused():
    frame = pd.DataFrame({"id": ["a", "a"], "value": [1, 2]})

    with pytest.raises(ValueError, match="uniquely"):
        digest.canonical_digest(frame, sort_by=("id",))


def prediction_rows(track: str, model: str, ids: list[str]) -> pd.DataFrame:
    """Minimal prediction rows for one track and model."""
    return pd.DataFrame(
        {
            "track": track,
            "model": model,
            "sample_id": ids,
            "predicted_facility_id": "Gym",
            "predicted_usage_weekday": 1,
            "predicted_usage_hour": 7,
            "predicted_delay_minutes": 120.0,
            "predicted_transition_facility_id": None,
        }
    )


def test_a_model_replaces_only_its_own_prediction_rows(engine, smoke_config):
    storage.write_predictions(
        engine,
        prediction_rows("baseline", "frequency_recency", ["S1", "S2"]),
        track="baseline",
        model="frequency_recency",
    )

    storage.write_predictions(
        engine,
        prediction_rows("traditional", "catboost", ["S1"]),
        track="traditional",
        model="catboost",
    )

    stored = storage.read_table(engine, storage.PREDICTIONS, smoke_config)
    baseline = stored.loc[
        (stored["track"] == "baseline")
        & (stored["model"] == "frequency_recency")
    ]
    catboost = stored.loc[
        (stored["track"] == "traditional") & (stored["model"] == "catboost")
    ]

    assert len(baseline) == 2
    assert len(catboost) == 1


def test_rewriting_one_model_leaves_the_other_untouched(engine, smoke_config):
    storage.write_predictions(
        engine,
        prediction_rows("baseline", "frequency_recency", ["S1", "S2"]),
        track="baseline",
        model="frequency_recency",
    )
    storage.write_predictions(
        engine,
        prediction_rows("traditional", "catboost", ["S1", "S2"]),
        track="traditional",
        model="catboost",
    )

    storage.write_predictions(
        engine,
        prediction_rows("traditional", "catboost", ["S3"]),
        track="traditional",
        model="catboost",
    )

    stored = storage.read_table(engine, storage.PREDICTIONS, smoke_config)
    assert sorted(stored.loc[stored["track"] == "baseline", "sample_id"]) == [
        "S1",
        "S2",
    ]
    catboost = stored.loc[
        (stored["track"] == "traditional") & (stored["model"] == "catboost")
    ]

    assert list(catboost["sample_id"]) == ["S3"]


def test_a_write_may_not_carry_another_models_rows(engine):
    mixed = pd.concat(
        [
            prediction_rows("traditional", "catboost", ["S1"]),
            prediction_rows("baseline", "frequency_recency", ["S2"]),
        ]
    )

    with pytest.raises(storage.StorageError, match="may only replace"):
        storage.write_predictions(
            engine, mixed, track="traditional", model="catboost"
        )
