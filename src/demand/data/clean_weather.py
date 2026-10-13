"""Turn the raw Open-Meteo response into the silver weather table.

Everything in this module is pure: the decoded JSON goes in, a dataframe comes out.
Reading and writing files is the job of ``silver.py``.

Decisions, and why:

* **From UTC to local time.** The raw file is in UTC (see ``weather.py``); the trips,
  and everything after silver, are in the project timezone. Converting an instant from
  UTC is always unambiguous, so daylight saving needs no special care here: the day the
  clocks go forward simply has 23 rows and the day they go back has 25.
* **Exactly the configured period.** The table runs from 00:00 local time on the first
  day to 23:00 local time on the last day. The raw file is a day longer on each side;
  that padding only exists so that the local period is covered, and it is cut off. The
  list of hours is built from the config, not from the file, so an hour the file lacks
  is noticed.
* **No missing hour.** An hour that the file lacks, or for which a variable is null, is
  a gap. A gap of at most ``cleaning.max_weather_gap_hours`` hours is filled with a
  straight line between its two neighbours. A longer gap, or a gap at the very start or
  end (which has only one neighbour), stops the stage: weather that is invented over
  many hours would mislead the model without anyone noticing.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.clean import RawFormatError
from demand.schemas import HOUR_START

RAW_TIME = "time"
RAW_HOURLY = "hourly"
RAW_UTC_OFFSET_SECONDS = "utc_offset_seconds"
_LAST_HOUR_OF_DAY = 23


class WeatherGapError(ValueError):
    """The weather has a gap that is too long, or one that cannot be interpolated."""


class WeatherSummary(BaseModel):
    """What the weather cleaning did, for the cleaning report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hours: int
    """Rows in the silver table: one per hour of the period."""
    interpolated: dict[str, int]
    """Variable name -> number of values that were filled by interpolation."""


def local_hours(start_date: date, end_date: date, timezone: str) -> pd.DatetimeIndex:
    """Every hour from 00:00 on ``start_date`` to 23:00 on ``end_date``, local time.

    The hours are real, consecutive hours: across a daylight-saving change the local
    clock time jumps, but two neighbours are always 60 minutes apart.
    """
    first = pd.Timestamp(start_date, tz=timezone)
    last = pd.Timestamp(end_date, tz=timezone).replace(hour=_LAST_HOUR_OF_DAY)
    return pd.date_range(first, last, freq="h", name=HOUR_START)


def parse_weather(payload: object, variables: list[str], timezone: str) -> pd.DataFrame:
    """Build a table from a decoded Open-Meteo response, indexed by local hour.

    Null values stay missing; nothing is filled or removed here.

    Raises:
        RawFormatError: If the response is not in UTC, lacks a variable, or its lists
            differ in length.
    """
    hourly = payload.get(RAW_HOURLY) if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or not isinstance(hourly, dict):
        raise RawFormatError(f"weather response has no '{RAW_HOURLY}' section")
    if payload.get(RAW_UTC_OFFSET_SECONDS, 0) != 0:
        raise RawFormatError("weather response is not in UTC")

    columns = [RAW_TIME, *variables]
    missing = [name for name in columns if not isinstance(hourly.get(name), list)]
    if missing:
        raise RawFormatError(f"weather response has no list of {', '.join(missing)}")
    lengths = {name: len(hourly[name]) for name in columns}
    if len(set(lengths.values())) != 1:
        raise RawFormatError(f"weather lists differ in length: {lengths}")

    try:
        utc_hours = pd.DatetimeIndex(hourly[RAW_TIME]).tz_localize("UTC")
    except (ValueError, TypeError) as error:
        raise RawFormatError(f"weather times cannot be read: {error}") from error
    if utc_hours.has_duplicates:
        raise RawFormatError("weather response contains the same hour more than once")

    values = {name: pd.Series(hourly[name], dtype="float64").to_numpy() for name in variables}
    index = utc_hours.tz_convert(timezone).rename(HOUR_START)
    return pd.DataFrame(values, index=index)


def find_long_gap(values: pd.Series, max_gap_hours: int) -> tuple[pd.Timestamp, int] | None:
    """First run of missing hours longer than ``max_gap_hours``: its start and length."""
    is_missing = values.isna()
    # Every present value starts a new group, so a run of missing hours shares one group.
    run_id = (~is_missing).cumsum()
    run_length = is_missing.groupby(run_id).transform("sum")
    too_long = is_missing & (run_length > max_gap_hours)
    if not too_long.any():
        return None
    position = int(too_long.to_numpy().argmax())
    return values.index[position], int(run_length.iloc[position])


def fill_gaps(weather: pd.DataFrame, max_gap_hours: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Interpolate short gaps linearly and refuse long ones.

    Args:
        weather: One row per hour, complete index, missing values where there is a gap.
        max_gap_hours: Longest run of missing hours that may be filled.

    Returns:
        The table without missing values, and the number of filled values per variable.

    Raises:
        WeatherGapError: If a variable has a longer gap, or a gap that touches the
            first or last hour.
    """
    filled = weather.copy()
    interpolated: dict[str, int] = {}
    for name in weather.columns:
        long_gap = find_long_gap(weather[name], max_gap_hours)
        if long_gap is not None:
            start, length = long_gap
            raise WeatherGapError(
                f"weather variable '{name}' has no value for {length} hours in a row from "
                f"{start}; at most {max_gap_hours} are interpolated "
                "(cleaning.max_weather_gap_hours). Run `python -m demand ingest` again or "
                "check the raw weather file."
            )
        # "inside" fills only between two known values and never extends past the ends.
        filled[name] = weather[name].interpolate(method="linear", limit_area="inside")
        still_missing = filled[name].isna()
        if still_missing.any():
            raise WeatherGapError(
                f"weather variable '{name}' has no value at {still_missing.idxmax()}, at "
                "the edge of the period, so there is nothing to interpolate from. The raw "
                "weather file does not cover the configured period; run "
                "`python -m demand ingest` again."
            )
        interpolated[name] = int(weather[name].isna().sum())
    return filled, interpolated


def clean_weather(payload: object, settings: Settings) -> tuple[pd.DataFrame, WeatherSummary]:
    """Build the silver weather table from a decoded Open-Meteo response.

    Returns:
        One row per local hour of the configured period, with the columns of
        ``silver_weather_schema``; and a summary for the cleaning report.

    Raises:
        RawFormatError: If the response cannot be read as hourly weather in UTC.
        WeatherGapError: If the weather has a gap that may not be interpolated.
    """
    data = settings.data
    timezone = settings.project.timezone
    weather = parse_weather(payload, data.weather.variables, timezone)
    # Hours outside the period are dropped; hours the file lacks appear as missing rows.
    weather = weather.reindex(local_hours(data.start_date, data.end_date, timezone))
    filled, interpolated = fill_gaps(weather, settings.cleaning.max_weather_gap_hours)
    silver = filled.reset_index()
    return silver, WeatherSummary(hours=len(silver), interpolated=interpolated)
