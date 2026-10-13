from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.clean import RawFormatError
from demand.data.clean_weather import (
    WeatherGapError,
    WeatherSummary,
    clean_weather,
    fill_gaps,
    find_long_gap,
    local_hours,
    parse_weather,
)
from demand.schemas import HOUR_START, silver_weather_schema
from tests.unit.data.conftest import weather_payload, with_months

VARIABLES = ["temperature_2m", "precipitation"]
TIMEZONE = "America/New_York"

# A padded January: one UTC day before the month and one after it, as ingest stores it.
JANUARY_FIRST_UTC_HOUR = "2024-12-31T00:00"
JANUARY_PADDED_HOURS = 33 * 24
# 00:00 local time on 1 January is 05:00 UTC, the 30th hour of the padded response
# (position 29). Because every test value equals its position, the first silver value
# of an untouched January is 29.0.
JANUARY_FIRST_LOCAL_POSITION = 29


def _january(settings: Settings) -> Settings:
    return with_months(settings, "2025-01", "2025-01")


def _january_payload(settings: Settings, **gaps: list[int]) -> dict[str, object]:
    return weather_payload(
        JANUARY_FIRST_UTC_HOUR, JANUARY_PADDED_HOURS, settings.data.weather.variables, **gaps
    )


def _hourly_series(values: list[float | None]) -> pd.Series:
    index = pd.date_range("2025-01-01", periods=len(values), freq="h", tz=TIMEZONE)
    return pd.Series(values, index=index, dtype="float64")


# --- the hours of the period ------------------------------------------------------------


def test_local_hours_run_from_midnight_to_the_last_hour_of_the_last_day() -> None:
    hours = local_hours(date(2025, 1, 1), date(2025, 1, 2), TIMEZONE)

    assert len(hours) == 48
    assert hours[0] == pd.Timestamp("2025-01-01 00:00:00-05:00")
    assert hours[-1] == pd.Timestamp("2025-01-02 23:00:00-05:00")
    assert hours.name == HOUR_START


def test_day_the_clocks_go_forward_has_23_hours() -> None:
    hours = local_hours(date(2025, 3, 9), date(2025, 3, 9), TIMEZONE)

    assert len(hours) == 23
    assert pd.Timestamp("2025-03-09 01:00:00-05:00") in hours
    assert pd.Timestamp("2025-03-09 03:00:00-04:00") in hours


def test_day_the_clocks_go_back_has_25_hours() -> None:
    hours = local_hours(date(2025, 11, 2), date(2025, 11, 2), TIMEZONE)

    assert len(hours) == 25
    assert hours.is_unique


# --- reading the response ---------------------------------------------------------------


def test_winter_hour_is_five_hours_behind_utc() -> None:
    payload = weather_payload("2025-01-15T12:00", 2, VARIABLES)

    weather = parse_weather(payload, VARIABLES, TIMEZONE)

    assert list(weather.index) == [
        pd.Timestamp("2025-01-15 07:00:00-05:00"),
        pd.Timestamp("2025-01-15 08:00:00-05:00"),
    ]
    assert weather.index.name == HOUR_START


def test_summer_hour_is_four_hours_behind_utc() -> None:
    payload = weather_payload("2025-06-15T12:00", 1, VARIABLES)

    weather = parse_weather(payload, VARIABLES, TIMEZONE)

    assert list(weather.index) == [pd.Timestamp("2025-06-15 08:00:00-04:00")]


def test_values_become_floats_and_nulls_stay_missing() -> None:
    payload = weather_payload("2025-01-15T12:00", 3, VARIABLES, nulls=[1])

    weather = parse_weather(payload, VARIABLES, TIMEZONE)

    assert list(weather.columns) == VARIABLES
    assert weather["precipitation"].dtype == "float64"
    assert list(weather["precipitation"].isna()) == [False, True, False]


def test_variable_that_is_not_configured_is_ignored() -> None:
    payload = weather_payload("2025-01-15T12:00", 2, [*VARIABLES, "snowfall"])

    weather = parse_weather(payload, VARIABLES, TIMEZONE)

    assert list(weather.columns) == VARIABLES


BROKEN_PAYLOADS = {
    "not an object": ([1, 2, 3], "no 'hourly' section"),
    "no hourly section": ({"error": True}, "no 'hourly' section"),
    "local time instead of UTC": (
        weather_payload("2025-01-15T12:00", 2, VARIABLES) | {"utc_offset_seconds": -18000},
        "not in UTC",
    ),
    "variable missing": (
        weather_payload("2025-01-15T12:00", 2, ["temperature_2m"]),
        "no list of precipitation",
    ),
    "lists of different length": (
        {
            "hourly": {
                "time": ["2025-01-15T12:00"],
                "temperature_2m": [1.0, 2.0],
                "precipitation": [0.0],
            }
        },
        "differ in length",
    ),
    "unreadable time": (
        {"hourly": {"time": ["hour-0"], "temperature_2m": [1.0], "precipitation": [0.0]}},
        "times cannot be read",
    ),
    "same hour twice": (
        {
            "hourly": {
                "time": ["2025-01-15T12:00", "2025-01-15T12:00"],
                "temperature_2m": [1.0, 2.0],
                "precipitation": [0.0, 0.0],
            }
        },
        "same hour more than once",
    ),
}


@pytest.mark.parametrize(
    ("payload", "message"), BROKEN_PAYLOADS.values(), ids=BROKEN_PAYLOADS.keys()
)
def test_broken_response_is_refused(payload: object, message: str) -> None:
    with pytest.raises(RawFormatError, match=message):
        parse_weather(payload, VARIABLES, TIMEZONE)


# --- gaps -------------------------------------------------------------------------------


def test_no_long_gap_in_a_complete_series() -> None:
    assert find_long_gap(_hourly_series([1.0, 2.0, 3.0]), max_gap_hours=0) is None


def test_gap_at_the_limit_is_not_long() -> None:
    values = _hourly_series([1.0, None, None, None, 5.0])

    assert find_long_gap(values, max_gap_hours=3) is None


def test_first_gap_above_the_limit_is_found_with_its_start_and_length() -> None:
    values = _hourly_series([1.0, None, 3.0, None, None, None, None, 8.0])

    gap = find_long_gap(values, max_gap_hours=3)

    assert gap == (pd.Timestamp("2025-01-01 03:00:00-05:00"), 4)


def test_short_gap_is_filled_on_a_straight_line() -> None:
    weather = pd.DataFrame({"temperature_2m": _hourly_series([10.0, None, None, None, 18.0])})

    filled, interpolated = fill_gaps(weather, max_gap_hours=3)

    assert list(filled["temperature_2m"]) == [10.0, 12.0, 14.0, 16.0, 18.0]
    assert interpolated == {"temperature_2m": 3}
    assert weather["temperature_2m"].isna().sum() == 3, "the input must not be changed"


def test_long_gap_raises_and_names_the_variable_and_the_hour() -> None:
    weather = pd.DataFrame(
        {
            "temperature_2m": _hourly_series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            "precipitation": _hourly_series([0.0, None, None, None, None, 0.0]),
        }
    )

    with pytest.raises(WeatherGapError, match=r"'precipitation' has no value for 4 hours.*01:00"):
        fill_gaps(weather, max_gap_hours=3)


def test_any_gap_raises_when_the_limit_is_zero() -> None:
    weather = pd.DataFrame({"temperature_2m": _hourly_series([1.0, None, 3.0])})

    with pytest.raises(WeatherGapError, match="1 hours in a row"):
        fill_gaps(weather, max_gap_hours=0)


@pytest.mark.parametrize(
    "values", [[None, 2.0, 3.0], [1.0, 2.0, None]], ids=["first hour", "last hour"]
)
def test_gap_at_the_edge_raises_because_it_has_only_one_neighbour(
    values: list[float | None],
) -> None:
    weather = pd.DataFrame({"temperature_2m": _hourly_series(values)})

    with pytest.raises(WeatherGapError, match="edge of the period"):
        fill_gaps(weather, max_gap_hours=3)


# --- the whole weather cleaning ---------------------------------------------------------


def test_silver_weather_covers_exactly_the_local_period(settings: Settings) -> None:
    january = _january(settings)

    weather, summary = clean_weather(_january_payload(january), january)

    silver_weather_schema(january).validate(weather)
    assert len(weather) == 31 * 24
    assert weather[HOUR_START].iloc[0] == pd.Timestamp("2025-01-01 00:00:00-05:00")
    assert weather[HOUR_START].iloc[-1] == pd.Timestamp("2025-01-31 23:00:00-05:00")
    assert weather["temperature_2m"].iloc[0] == JANUARY_FIRST_LOCAL_POSITION
    assert summary == WeatherSummary(
        hours=31 * 24, interpolated=dict.fromkeys(january.data.weather.variables, 0)
    )


def test_silver_weather_has_one_row_per_real_hour_across_the_spring_change(
    settings: Settings,
) -> None:
    march = with_months(settings, "2025-03", "2025-03")
    payload = weather_payload("2025-02-28T00:00", 33 * 24, march.data.weather.variables)

    weather, _ = clean_weather(payload, march)

    silver_weather_schema(march).validate(weather)
    assert len(weather) == 31 * 24 - 1
    assert (weather[HOUR_START].diff().dropna() == pd.Timedelta(hours=1)).all()
    # Position in the response = hours since the first UTC hour, also across the change.
    assert (np.diff(weather["temperature_2m"]) == 1.0).all()


def test_null_values_inside_the_period_are_interpolated(settings: Settings) -> None:
    january = _january(settings)
    payload = _january_payload(january, nulls=[100, 101, 102])

    weather, summary = clean_weather(payload, january)

    silver_weather_schema(january).validate(weather)
    filled = weather["temperature_2m"].iloc[100 - JANUARY_FIRST_LOCAL_POSITION :][:3]
    assert list(filled) == [100.0, 101.0, 102.0]
    assert set(summary.interpolated.values()) == {3}


def test_hours_absent_from_the_response_are_interpolated(settings: Settings) -> None:
    january = _january(settings)
    payload = _january_payload(january, absent=[200, 201])

    weather, summary = clean_weather(payload, january)

    assert len(weather) == 31 * 24
    assert (np.diff(weather["wind_speed_10m"]) == 1.0).all()
    assert set(summary.interpolated.values()) == {2}


def test_gap_longer_than_the_configured_limit_stops_the_cleaning(settings: Settings) -> None:
    january = _january(settings)
    too_long = january.cleaning.max_weather_gap_hours + 1
    payload = _january_payload(january, absent=list(range(200, 200 + too_long)))

    with pytest.raises(WeatherGapError, match=f"{too_long} hours in a row"):
        clean_weather(payload, january)


def test_gap_in_the_padding_outside_the_period_does_not_matter(settings: Settings) -> None:
    january = _january(settings)
    payload = _january_payload(january, nulls=list(range(10)))

    _, summary = clean_weather(payload, january)

    assert set(summary.interpolated.values()) == {0}


def test_response_that_ends_before_the_period_does_stops_the_cleaning(
    settings: Settings,
) -> None:
    january = _january(settings)
    payload = weather_payload(JANUARY_FIRST_UTC_HOUR, 31 * 24, january.data.weather.variables)

    with pytest.raises(WeatherGapError):
        clean_weather(payload, january)
