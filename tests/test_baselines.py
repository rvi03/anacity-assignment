"""Tests the frequency and recency baselines.

Covers the five rules, the fallback chain a resident with no relevant
history falls back through, and the two properties that make the
comparator honest: nothing after the origin is read, and a rerun
predicts identically.
"""

from __future__ import annotations

import pathlib
import zoneinfo

import pandas as pd
import pytest

from facility_prediction import config as config_module
from facility_prediction.data import generate, samples, split
from facility_prediction.models import baselines

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"

KOLKATA = zoneinfo.ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="module")
def smoke_config() -> config_module.Config:
    return config_module.load_config(SMOKE_PATH)


@pytest.fixture(scope="module")
def smoke_bookings(smoke_config) -> pd.DataFrame:
    return generate.generate_bookings(smoke_config)


@pytest.fixture(scope="module")
def labelled(smoke_bookings, smoke_config) -> pd.DataFrame:
    frame = samples.build_samples(smoke_bookings, smoke_config)
    return split.assign_split(frame, split.compute_cutoffs(frame, smoke_config))


@pytest.fixture(scope="module")
def fallbacks(smoke_bookings, labelled) -> baselines.CommunityFallbacks:
    train = labelled.loc[labelled["split"] == split.TRAIN]
    return baselines.fit(smoke_bookings, train)


@pytest.fixture(scope="module")
def predictions(labelled, smoke_bookings, fallbacks) -> pd.DataFrame:
    return baselines.predict(labelled, smoke_bookings, fallbacks)


def bookings_frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """A tiny booking table from ISO strings carrying an offset."""
    frame = pd.DataFrame(rows, columns=list(generate.BOOKING_COLUMNS))
    for column in ("booking_timestamp", "usage_timestamp"):
        frame[column] = pd.to_datetime(
            frame[column], format="ISO8601", utc=True
        ).dt.tz_convert(KOLKATA)
    return frame


def one_resident() -> pd.DataFrame:
    """Four bookings: Gym three times on a Monday 7am, Pool once."""
    return bookings_frame(
        [
            (
                "B0000001",
                "R001",
                "Gym",
                "2024-01-01T06:00:00+05:30",
                "2024-01-01T07:00:00+05:30",
            ),
            (
                "B0000002",
                "R001",
                "Gym",
                "2024-01-08T06:00:00+05:30",
                "2024-01-08T07:00:00+05:30",
            ),
            (
                "B0000003",
                "R001",
                "Pool",
                "2024-01-15T06:00:00+05:30",
                "2024-01-16T18:00:00+05:30",
            ),
            (
                "B0000004",
                "R001",
                "Gym",
                "2024-01-22T06:00:00+05:30",
                "2024-01-22T07:00:00+05:30",
            ),
        ]
    )


# --- the five rules ------


def test_facility_is_the_residents_most_frequent_prior_facility(
    smoke_config, fallbacks
):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    assert predicted["predicted_facility_id"].iloc[-1] == "Gym"


def test_usage_day_and_hour_come_from_the_residents_own_history(
    smoke_config, fallbacks
):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    assert predicted["predicted_usage_weekday"].iloc[-1] == 0
    assert predicted["predicted_usage_hour"].iloc[-1] == 7


def test_delay_is_the_median_prior_booking_to_booking_interval(
    smoke_config, fallbacks
):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    assert predicted["predicted_delay_minutes"].iloc[-1] == pytest.approx(
        7 * 24 * 60
    )


def test_transition_follows_the_last_facility(smoke_config, fallbacks):
    # A fifth booking so the final sample's history ends on Gym, which
    # is the only way this resident's own Gym-> transitions decide the
    # answer instead of the community fallback.
    bookings = pd.concat(
        [
            one_resident(),
            bookings_frame(
                [
                    (
                        "B0000005",
                        "R001",
                        "Badminton",
                        "2024-01-29T06:00:00+05:30",
                        "2024-01-29T19:00:00+05:30",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    # history is Gym, Gym, Pool, Gym: the resident went Gym->Gym once
    # and Gym->Pool once, and the tie breaks towards the more recent
    assert predicted["predicted_transition_facility_id"].iloc[-1] == "Pool"


def test_transition_falls_back_when_the_last_facility_is_unrepeated(
    smoke_config, fallbacks
):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    # the final sample's history ends on Pool, which this resident has
    # never followed with anything, so the community fallback answers
    assert (
        predicted["predicted_transition_facility_id"].iloc[-1]
        == fallbacks.transition_facility_id["Pool"]
    )


def test_transition_falls_back_to_the_community_for_an_unseen_facility(
    smoke_config, fallbacks
):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    # the first sample has only one prior booking, so the resident has
    # no transition of their own yet
    assert (
        predicted["predicted_transition_facility_id"].iloc[0]
        == (fallbacks.transition_facility_id["Gym"])
    )


def test_the_delay_rule_does_not_subtract_a_lead_time(smoke_config, fallbacks):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    lead = smoke_config.application.notification_lead_minutes
    assert predicted["predicted_delay_minutes"].iloc[-1] != pytest.approx(
        7 * 24 * 60 - lead
    )


# --- the fallback chain ------


def test_a_resident_with_one_prior_booking_falls_back_for_the_delay(
    smoke_config, fallbacks
):
    bookings = one_resident()
    frame = samples.build_samples(bookings, smoke_config)

    predicted = baselines.predict(frame, bookings, fallbacks)

    assert predicted["predicted_delay_minutes"].iloc[0] == pytest.approx(
        fallbacks.delay_minutes
    )


def test_fallbacks_are_fitted_on_training_rows_only(smoke_bookings, labelled):
    train = labelled.loc[labelled["split"] == split.TRAIN]
    horizon = train["origin"].max()

    fitted = baselines.fit(smoke_bookings, train)
    later_only = baselines.fit(
        smoke_bookings.loc[smoke_bookings["booking_timestamp"] <= horizon],
        train,
    )

    assert fitted == later_only


def test_fitting_without_training_rows_is_rejected(smoke_bookings, labelled):
    with pytest.raises(ValueError, match="training samples"):
        baselines.fit(smoke_bookings, labelled.iloc[0:0])


def test_every_fallback_is_a_real_catalog_value(fallbacks, smoke_config):
    assert fallbacks.facility_id in smoke_config.facility_names
    assert 0 <= fallbacks.usage_weekday <= 6
    assert 0 <= fallbacks.usage_hour <= 23
    assert set(fallbacks.transition_facility_id) <= set(
        smoke_config.facility_names
    )


# --- what the comparator must never do ------


def test_no_prediction_reads_a_booking_after_the_origin(
    labelled, smoke_bookings, fallbacks
):
    subset = labelled.head(200)
    truncated = []
    for resident, origin in zip(
        subset["resident_id"], subset["origin"], strict=True
    ):
        rows = smoke_bookings.loc[
            (smoke_bookings["resident_id"] == resident)
            & (smoke_bookings["booking_timestamp"] <= origin)
        ]
        truncated.append(rows)
    past_only = pd.concat(truncated).drop_duplicates(subset="booking_id")

    predicted = baselines.predict(subset, past_only, fallbacks)

    pd.testing.assert_frame_equal(
        predicted, baselines.predict(subset, smoke_bookings, fallbacks)
    )


def test_changing_a_future_booking_cannot_change_a_prediction(
    labelled, smoke_bookings, fallbacks
):
    subset = labelled.head(200)
    horizon = subset["origin"].max()
    perturbed = smoke_bookings.copy()
    future = perturbed["booking_timestamp"] > horizon
    perturbed.loc[future, "facility_id"] = "Tennis"

    predicted = baselines.predict(subset, perturbed, fallbacks)

    pd.testing.assert_frame_equal(
        predicted, baselines.predict(subset, smoke_bookings, fallbacks)
    )


def test_predictions_are_identical_on_a_rerun(
    labelled, smoke_bookings, fallbacks, predictions
):
    again = baselines.predict(labelled, smoke_bookings, fallbacks)

    pd.testing.assert_frame_equal(again, predictions)


def test_every_sample_gets_all_four_outputs(predictions, labelled):
    assert tuple(predictions.columns) == baselines.PREDICTION_COLUMNS
    assert len(predictions) == len(labelled)
    assert not predictions.isna().to_numpy().any()


def test_predicted_values_stay_inside_their_domains(predictions, smoke_config):
    assert set(predictions["predicted_facility_id"]) <= set(
        smoke_config.facility_names
    )
    assert predictions["predicted_usage_weekday"].between(0, 6).all()
    assert predictions["predicted_usage_hour"].between(0, 23).all()
    assert (predictions["predicted_delay_minutes"] > 0).all()
