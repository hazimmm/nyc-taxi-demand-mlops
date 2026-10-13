"""Turn silver trips into hourly demand per zone: the first gold table.

Everything in this module is pure: dataframes go in, dataframes come out. Reading and
writing files is the job of ``gold.py``.

**What the table is.** One row per frozen zone and hour, with the number of pickups, the
weather of that hour and an outlier flag (``gold_demand_schema``). Only two silver
columns are used, the pickup time and the pickup zone. A trip whose dropoff time or
distance was not recorded is still a pickup and is counted like any other.

**A complete grid.** Counting trips only produces rows for hours that had a trip. An
hour without pickups would simply be absent, and every later step (a lag "24 hours
earlier", a rolling mean) would silently read the wrong hour. So the list of hours is
built from the config, not from the data (``local_hours``), every zone gets every hour,
and an hour without pickups is an explicit 0. The hours are real consecutive hours: the
day the clocks go forward has 23 rows per zone, the day they go back has 25.

**Which zones.** Only the zones in ``data.zones``. That list is chosen once with
``python -m demand select-zones`` from the *training period only* (``busiest_zones``)
and then frozen in the config. Choosing the busiest zones of the whole period would let
May and June, which the model must not know, decide what the model is trained on.

**The outlier flag** (``flag_outliers``). An hour is flagged, never removed, when its
pickups are far from what is normal for that zone at that kind of hour:

* Hours are compared with hours of the same *slot*: the same zone, the same hour of the
  day, and the same kind of day (Monday to Friday, or weekend). 08:00 on a Tuesday is
  compared with 08:00 on other working days, not with 03:00 or with a Sunday.
* For every slot, the mean and the standard deviation of the pickups are calculated
  **from the training period only**.
* An hour is an outlier when it is more than ``features.outlier_zscore`` standard
  deviations away from the mean of its slot, in either direction.

The same training-period numbers judge the replayed months, so nothing that happens
after ``data.train_end_date`` can change the flag of a training hour, and the yardstick
does not move while "production" is replayed.

Two things about this flag are easy to get wrong:

* It is calculated from the pickups of its own hour, which is exactly the value the
  model has to predict. It may be used to leave strange hours out of training or out of
  a score, but it must never be a model input. (It is not in ``feature_columns``.)
* Why not one slot per day of the week (168 slots per zone)? The training period has 17
  weeks, so most slots would hold 17 values, and among ``n`` values none can be further
  than ``(n - 1) / sqrt(n)`` standard deviations from their mean. For 17 values that is
  3.9, so with a threshold of 4 almost no training hour could ever be flagged. Working
  days and weekends give about 86 and 34 values per slot.

A slot whose training values are all the same has a standard deviation of zero, and a
slot with fewer than two training values has none. Nothing is flagged in such a slot,
because there is no spread to measure a distance with.
"""

from __future__ import annotations

import pandas as pd

from demand.config import Settings
from demand.data.clean_weather import local_hours
from demand.data.features import weekend_flag
from demand.errors import PipelineError
from demand.schemas import (
    HOUR_START,
    IS_OUTLIER,
    PICKUP_DATETIME,
    PICKUP_ZONE_ID,
    TARGET,
    ZONE_ID,
)


class TooFewZonesError(PipelineError, ValueError):
    """Fewer zones have pickups in the training period than ``data.n_zones`` asks for."""


class WeatherCoverageError(PipelineError, ValueError):
    """The silver weather does not have every hour of the configured period."""


def training_period_end(settings: Settings) -> pd.Timestamp:
    """The first moment *after* the training period, in the project timezone.

    Everything strictly before it belongs to the training period; everything from it on
    is replayed as "production".
    """
    return pd.Timestamp(settings.data.replay_start_date, tz=settings.project.timezone)


def pickup_hour(pickup: pd.Series) -> pd.Series:
    """The start of the hour each pickup falls into, in the timezone of ``pickup``.

    The rounding is done in UTC and converted back. Rounding local time directly fails
    on the night the clocks go back, when 01:00 exists twice; in UTC every hour exists
    exactly once. This is correct for a timezone whose offset from UTC is a whole number
    of hours, such as New York. (For one that is not, the gold schema refuses the table:
    it requires hours that start at minute 0 of local time.)
    """
    timezone = pickup.dt.tz
    hours: pd.Series = pickup.dt.tz_convert("UTC").dt.floor("h").dt.tz_convert(timezone)
    return hours


def count_pickups_per_zone(trips: pd.DataFrame, before: pd.Timestamp) -> pd.Series:
    """Pickups per zone among the trips that started before ``before``.

    Args:
        trips: Silver trips; only the pickup time and the pickup zone are used.
        before: Trips from this moment on are not counted (``training_period_end``).

    Returns:
        Zone ID -> number of pickups. A zone without pickups is absent.
    """
    in_period = trips[PICKUP_DATETIME] < before
    counts: pd.Series = trips.loc[in_period, PICKUP_ZONE_ID].value_counts()
    return counts.rename_axis(ZONE_ID).rename(TARGET)


def busiest_zones(pickups_per_zone: pd.Series, n_zones: int) -> dict[int, int]:
    """The ``n_zones`` zones with the most pickups, busiest first.

    Two zones with the same number of pickups are ordered by zone ID, lower first, so
    the result never depends on the order of the input.

    Args:
        pickups_per_zone: Zone ID -> pickups, counted on the training period only.
        n_zones: How many zones to return.

    Returns:
        Zone ID -> pickups, in order of rank.

    Raises:
        TooFewZonesError: If fewer than ``n_zones`` zones have any pickup.
    """
    ranked = (
        pickups_per_zone[pickups_per_zone > 0]
        .rename_axis(ZONE_ID)
        .reset_index(name=TARGET)
        .sort_values([TARGET, ZONE_ID], ascending=[False, True])
    )
    if len(ranked) < n_zones:
        raise TooFewZonesError(
            f"only {len(ranked)} zones have pickups in the training period, but "
            f"data.n_zones asks for {n_zones}. Lower data.n_zones in the config, or check "
            "that the silver trips cover the training period."
        )
    top = ranked.head(n_zones)
    return {
        int(zone): int(pickups) for zone, pickups in zip(top[ZONE_ID], top[TARGET], strict=True)
    }


def count_hourly_pickups(trips: pd.DataFrame, zones: list[int]) -> pd.Series:
    """Pickups per zone and hour, for the zones in ``zones`` only.

    Args:
        trips: Silver trips; only the pickup time and the pickup zone are used.
        zones: The frozen zone list. Trips that start anywhere else are ignored.

    Returns:
        Pickups indexed by zone and hour. An hour without pickups is absent here; it
        becomes a zero in ``to_complete_grid``.
    """
    wanted = trips[trips[PICKUP_ZONE_ID].isin(zones)]
    zone = wanted[PICKUP_ZONE_ID].rename(ZONE_ID)
    hour = pickup_hour(wanted[PICKUP_DATETIME]).rename(HOUR_START)
    counts: pd.Series = wanted.groupby([zone, hour]).size()
    return counts.rename(TARGET)


def add_counts(counts: list[pd.Series]) -> pd.Series:
    """Add up hourly counts that were made file by file."""
    total: pd.Series = pd.concat(counts).groupby(level=[ZONE_ID, HOUR_START]).sum()
    return total


def grid_index(zones: list[int], hours: pd.DatetimeIndex) -> pd.MultiIndex:
    """Every pair of a zone and an hour: the keys of a complete grid, zone by zone.

    The gold stage builds the demand table on this index, and the quality gate compares
    the stored tables with it, so both mean the same thing by "complete".
    """
    return pd.MultiIndex.from_product([sorted(zones), hours], names=[ZONE_ID, HOUR_START])


def to_complete_grid(counts: pd.Series, zones: list[int], hours: pd.DatetimeIndex) -> pd.DataFrame:
    """Put the counts on the full grid of zones and hours; an empty hour becomes 0.

    Args:
        counts: Pickups indexed by zone and hour (``count_hourly_pickups``).
        zones: Every zone of the grid.
        hours: Every hour of the grid (``local_hours``).

    Returns:
        One row per zone and hour, sorted by zone and then by hour. Counts for a zone or
        an hour that is not on the grid are left out.
    """
    grid = grid_index(zones, hours)
    return counts.reindex(grid, fill_value=0).astype("int64").rename(TARGET).reset_index()


def join_weather(demand: pd.DataFrame, weather: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Add the weather of each hour to the demand table.

    The weather is the same for every zone: it is one series for the whole city.

    Raises:
        WeatherCoverageError: If an hour of the demand table has no weather. Silver
            weather has every hour of the configured period, so this means the silver
            layer was built with a different period.
    """
    joined = demand.merge(weather[[HOUR_START, *variables]], on=HOUR_START, how="left")
    without_weather = joined[variables].isna().any(axis=1)
    if without_weather.any():
        first = joined.loc[without_weather, HOUR_START].iloc[0]
        raise WeatherCoverageError(
            f"silver weather has no value for {int(without_weather.sum())} rows of the "
            f"demand table, first at {first}. Run `python -m demand silver` again so "
            "that weather and trips cover the same period."
        )
    return joined.astype(dict.fromkeys(variables, "float64"))


def flag_outliers(demand: pd.DataFrame, training_end: pd.Timestamp, zscore: float) -> pd.Series:
    """Mark the hours whose pickups are unusual for their zone and kind of hour.

    See the module docstring for the definition and the reasons behind it.

    Args:
        demand: Hourly demand with zone, hour and pickups.
        training_end: First moment after the training period. The mean and standard
            deviation of every slot come from the hours before it and from no others.
        zscore: How many standard deviations from the mean make an hour an outlier.

    Returns:
        True for every outlier hour, with the index of ``demand``.
    """
    hours = demand[HOUR_START]
    slot = [demand[ZONE_ID], weekend_flag(hours), hours.dt.hour]
    # Replay hours become missing here, so they take no part in the mean or the
    # standard deviation. ``transform`` then gives every row the numbers of its slot.
    training_pickups = demand[TARGET].where(hours < training_end).groupby(slot)
    typical = training_pickups.transform("mean")
    spread = training_pickups.transform("std")
    distance = (demand[TARGET] - typical).abs()
    # A comparison with a missing spread is False, so such a slot flags nothing.
    is_outlier: pd.Series = (spread > 0) & (distance > zscore * spread)
    return is_outlier.rename(IS_OUTLIER)


def build_demand(counts: pd.Series, weather: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Build the gold demand table from hourly counts and silver weather.

    Args:
        counts: Pickups indexed by zone and hour, for the whole configured period.
        weather: Silver weather, one row per hour.
        settings: Project settings (zones, period, timezone, outlier threshold).

    Returns:
        A table with the columns of ``gold_demand_schema``, sorted by zone and hour.

    Raises:
        WeatherCoverageError: If the weather does not cover the period.
    """
    data = settings.data
    hours = local_hours(data.start_date, data.end_date, settings.project.timezone)
    demand = to_complete_grid(counts, data.zones, hours)
    demand[IS_OUTLIER] = flag_outliers(
        demand, training_period_end(settings), settings.features.outlier_zscore
    )
    return join_weather(demand, weather, data.weather.variables)
