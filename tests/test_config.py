"""Unit tests for the typed configuration loader.

Covers the conventions the loader must enforce at load time: half-open
``hours``, zero-based ``available_from_month``, rejection of naive
datetimes, the facility catalog as the single categorical enum, the
generator mixture shares, and a lossless YAML round-trip.
"""

from __future__ import annotations

import copy
import datetime
import pathlib
from typing import Any
import zoneinfo

import pytest
import yaml

from facility_prediction import config as config_module

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "configs" / "default.yaml"
SMOKE_PATH = REPO_ROOT / "configs" / "smoke.yaml"


def raw_default() -> dict[str, Any]:
    """A fresh mutable copy of the parsed default config."""
    return copy.deepcopy(
        yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
    )


def test_default_config_loads():
    cfg = config_module.load_config(DEFAULT_PATH)
    assert cfg.seed == 20260811
    assert cfg.timezone == "Asia/Kolkata"
    assert cfg.community.residents == 500
    assert cfg.community.months == 24
    assert cfg.community.start_date == datetime.date(2024, 1, 1)
    assert len(cfg.facilities) == 8
    assert cfg.catboost.use_best_model is False
    assert cfg.catboost.iterations == 1500
    assert cfg.catboost.iteration_search is True
    assert cfg.catboost.notification_decode == "window"
    assert cfg.catboost.notification_framing == "absolute"
    assert cfg.catboost.facility_framing == "multiclass"
    assert cfg.catboost.bootstrap.notification.type == "MVS"
    assert cfg.catboost.bootstrap.notification.subsample is None
    assert cfg.catboost.bootstrap.classifiers.type == "Bernoulli"
    assert cfg.catboost.bootstrap.classifiers.subsample == 0.8


def test_smoke_config_loads_with_same_schema():
    cfg = config_module.load_config(SMOKE_PATH)
    assert cfg.community.residents < 500
    assert (
        cfg.facility_names
        == config_module.load_config(DEFAULT_PATH).facility_names
    )


def test_config_is_frozen():
    cfg = config_module.load_config(DEFAULT_PATH)
    # pydantic raises ValidationError on a frozen model
    with pytest.raises(Exception):  # noqa: B017
        cfg.seed = 1


def test_round_trips_without_loss(tmp_path):
    cfg = config_module.load_config(DEFAULT_PATH)
    out = tmp_path / "round_trip.yaml"
    config_module.dump_config(cfg, out)
    assert config_module.load_config(out) == cfg


# --- naive datetimes -------------------------------------


def test_rejects_naive_datetime_in_start_date():
    raw = raw_default()
    naive = datetime.datetime(2024, 1, 1, 0, 0)  # noqa: DTZ001
    raw["community"]["start_date"] = naive
    with pytest.raises(config_module.ConfigError, match="naive datetime"):
        config_module.parse_config(raw)


def test_rejects_naive_datetime_anywhere():
    raw = raw_default()
    naive = datetime.datetime(2024, 6, 1, 7, 30)  # noqa: DTZ001
    raw["facilities"][0]["name"] = naive
    with pytest.raises(config_module.ConfigError, match="naive datetime"):
        config_module.parse_config(raw)


def test_rejects_naive_datetime_parsed_from_yaml_text(tmp_path):
    text = DEFAULT_PATH.read_text(encoding="utf-8").replace(
        "start_date: 2024-01-01", "start_date: 2024-01-01T00:00:00"
    )
    path = tmp_path / "naive.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(config_module.ConfigError, match="naive datetime"):
        config_module.load_config(path)


def test_rejects_aware_datetime_for_start_date():
    raw = raw_default()
    raw["community"]["start_date"] = datetime.datetime(
        2024, 1, 1, tzinfo=zoneinfo.ZoneInfo("Asia/Kolkata")
    )
    with pytest.raises(config_module.ConfigError, match="calendar date"):
        config_module.parse_config(raw)


def test_start_instant_is_timezone_aware():
    cfg = config_module.load_config(DEFAULT_PATH)
    instant = cfg.start_instant
    assert instant.tzinfo is not None
    assert instant.utcoffset() == datetime.timedelta(hours=5, minutes=30)
    assert instant.date() == cfg.community.start_date


# --- half-open hours ---------------------------------------


@pytest.mark.parametrize("hours", [[22, 5], [7, 7], [6, 25], [-1, 10]])
def test_rejects_non_half_open_hours(hours):
    raw = raw_default()
    raw["facilities"][0]["hours"] = hours
    with pytest.raises(config_module.ConfigError, match="half-open"):
        config_module.parse_config(raw)


def test_hours_interval_is_half_open():
    gym = config_module.load_config(DEFAULT_PATH).facility("Gym")
    assert gym.open_hour == 5
    assert gym.close_hour == 22
    assert gym.is_open_at(5) is True
    assert gym.is_open_at(21) is True
    assert gym.is_open_at(22) is False
    assert gym.is_open_at(4) is False


# --- available_from_month ----------------------------------


def test_available_from_month_is_zero_based_index():
    cfg = config_module.load_config(DEFAULT_PATH)
    assert cfg.facility("Gym").available_from_month == 0
    assert cfg.facility("Yoga Room").available_from_month == 8


@pytest.mark.parametrize("month", [24, 25, 100])
def test_rejects_available_from_month_beyond_horizon(month):
    raw = raw_default()
    raw["facilities"][-1]["available_from_month"] = month
    with pytest.raises(config_module.ConfigError, match="beyond horizon"):
        config_module.parse_config(raw)


def test_rejects_negative_available_from_month():
    raw = raw_default()
    raw["facilities"][0]["available_from_month"] = -1
    with pytest.raises(config_module.ConfigError):
        config_module.parse_config(raw)


# --- popularity ------------------------------------------------------


@pytest.mark.parametrize("popularity", [0.0, -0.1, 1.5])
def test_rejects_out_of_range_popularity(popularity):
    raw = raw_default()
    raw["facilities"][0]["popularity"] = popularity
    with pytest.raises(config_module.ConfigError):
        config_module.parse_config(raw)


def test_rejects_popularity_not_summing_to_one():
    raw = raw_default()
    raw["facilities"][0]["popularity"] = 0.30
    with pytest.raises(config_module.ConfigError, match=r"sum to 1\.0"):
        config_module.parse_config(raw)


# --- catalog is the enum ---------------------------------------------


def test_facility_names_derive_from_the_config_list():
    cfg = config_module.load_config(DEFAULT_PATH)
    assert cfg.facility_names == (
        "Gym",
        "Pool",
        "Badminton",
        "Tennis",
        "Basketball",
        "Clubhouse",
        "Multipurpose Hall",
        "Yoga Room",
    )
    raw = raw_default()
    raw["facilities"] = raw["facilities"][:2]
    raw["facilities"][0]["popularity"] = 0.6
    raw["facilities"][1]["popularity"] = 0.4
    for archetype in raw["generator"]["archetypes"]:
        archetype["facilities"] = [
            name for name in archetype["facilities"] if name in {"Gym", "Pool"}
        ]
    assert config_module.parse_config(raw).facility_names == ("Gym", "Pool")


def test_rejects_archetype_naming_a_facility_outside_the_catalog():
    raw = raw_default()
    raw["generator"]["archetypes"][0]["facilities"] = ["Squash"]
    with pytest.raises(config_module.ConfigError, match="absent"):
        config_module.parse_config(raw)


def test_rejects_shares_that_do_not_sum_to_one():
    raw = raw_default()
    raw["generator"]["archetypes"][0]["share"] = 0.5
    with pytest.raises(config_module.ConfigError, match=r"sum to 1\.0"):
        config_module.parse_config(raw)


def test_rejects_duplicate_facility_names():
    raw = raw_default()
    raw["facilities"][1]["name"] = "Gym"
    with pytest.raises(config_module.ConfigError, match="unique"):
        config_module.parse_config(raw)


def test_unknown_facility_raises_key_error():
    cfg = config_module.load_config(DEFAULT_PATH)
    with pytest.raises(KeyError):
        cfg.facility("Squash")


# --- split, evaluation, timezone -------------------------------------


def test_split_test_fraction_is_the_residual():
    split = config_module.load_config(DEFAULT_PATH).split
    assert split.embargo_days == 0
    assert split.test_frac == pytest.approx(0.15)


def test_rejects_split_leaving_no_test_span():
    raw = raw_default()
    raw["split"]["val_frac"] = 0.30
    with pytest.raises(config_module.ConfigError, match="test span"):
        config_module.parse_config(raw)


@pytest.mark.parametrize(
    "minutes", [[60, 30, 180], [30, 30, 60], [0, 60], [-30, 60]]
)
def test_rejects_unordered_support_tolerances(minutes):
    raw = raw_default()
    raw["evaluation"]["notification_support_minutes"] = minutes
    with pytest.raises(config_module.ConfigError):
        config_module.parse_config(raw)


def test_rejects_non_multiplicative_match_ratio():
    raw = raw_default()
    raw["evaluation"]["notification_match_ratio"] = 1.0
    with pytest.raises(config_module.ConfigError):
        config_module.parse_config(raw)


def test_rejects_unknown_timezone():
    raw = raw_default()
    raw["timezone"] = "Mars/Olympus"
    with pytest.raises(config_module.ConfigError):
        config_module.parse_config(raw)


def test_tzinfo_resolves():
    cfg = config_module.load_config(DEFAULT_PATH)
    assert cfg.tzinfo == zoneinfo.ZoneInfo("Asia/Kolkata")


# --- loader hygiene --------------------------------------------------


def test_rejects_unknown_key():
    raw = raw_default()
    raw["mystery"] = 1
    with pytest.raises(config_module.ConfigError):
        config_module.parse_config(raw)


def test_rejects_non_mapping():
    with pytest.raises(config_module.ConfigError, match="must be a mapping"):
        config_module.parse_config([1, 2, 3])


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(config_module.ConfigError, match="cannot read"):
        config_module.load_config(tmp_path / "absent.yaml")
