"""The data the service forecasts from, and how one request becomes rows of model inputs.

**What a request asks for.** One local day ``D`` and one or more zones: "the pickups of
every hour of 14 May 2025 in zones 161 and 237". That is the product of the project
(``features.py``, "When a row can be built"): at 00:00 on the day before, a forecast for
every hour of the next day. The answer has one forecast per zone and local hour of
``D``: 24 on most days, 23 on the day the clocks go forward and 25 on the day they go
back. A request names a day and not single hours because every hour of a day is issued
at the same moment from the same data; a single hour would only be a slice of it.

**The client never sends features.** The service builds them itself, with
``build_features``, from two tables it reads once at startup:

* the *demand history*: zone, hour and pickups from the gold demand table
  (``data/gold/demand_hourly.parquet``);
* the *weather*: the silver weather table (``data/silver/weather/weather_hourly.parquet``),
  which stands in for a weather forecast. The weather of ``D`` is a forecast for ``D``,
  so it exists before ``D``, as in training (``backtest.py`` says what the stored values
  really are).

A client that sent its own feature values could send any numbers, and the model would
answer them without complaint; and every client would need its own copy of the feature
code, which is exactly the training/serving skew the one feature function exists to
prevent. Built here, the features of a served row are the features of a training row
for the same hour, and a test checks that they are equal to the last bit.

**The two files are a snapshot, not a feed.** The project has no live source of
pickups or weather: the TLC publishes each month about two months late. So the service
reads the files once, at startup, and never writes them. It answers for every day the
snapshot allows, including May and June 2025, the replay period: that is "production".
New data means a new snapshot, which means a restart (as a new model does; see
``model_loader.py``).

**The issue moment: what the service may know.** The forecast for ``D`` is issued at
00:00 on the day before, ``D - 1``. The service passes ``build_features`` only the hours
of demand that had *ended* by that moment, so a forecast cannot read the future even if
the feature function had a bug: the leakage rule is a property of the service, not only
of the features. (The features never read anything newer than ``t - horizon`` anyway;
``issue_moment`` is the earliest moment at which that is complete.) The 48 hours are
elapsed hours, so on the day the clocks go back, and the day after it, the moment is
01:00 on the day before instead of 00:00 (``backtest.py`` explains why). Neither day lies
in the data of this project; the tests cover them.

**Which days can be forecast.** Every day from ``first_day`` to ``last_day`` of the
snapshot, and no other:

* ``first_day`` is the first day with a full feature history: its first hour needs the
  pickups ``warm_up_hours`` (215) hours back. With demand from 1 January 2025, 00:00, that
  is 10 January.
* ``last_day`` is the last day whose weather is in the snapshot for every hour and whose
  issue moment the demand history reaches. With weather to 30 June 23:00 and demand to
  the same hour it is 30 June: the demand alone would allow 2 July.

A request for a day outside these is refused, with both days in the message. Nothing is
filled in: a missing lag set to 0 would be a forecast nobody could tell from a real one.

**What is checked at startup.** Both tables against their schemas
(``gold_demand_schema``, ``silver_weather_schema``), and that the demand covers every
configured zone on the same unbroken hourly timeline and the weather has no missing
hour. After that, a day inside the two limits always has all its inputs, so a request
can only be refused for what it asked, never for a hole found halfway.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import pandera.errors

from demand.config import Settings
from demand.data.clean_weather import local_hours
from demand.data.features import build_features, warm_up_hours
from demand.data.gold import gold_demand_path
from demand.data.silver import silver_weather_path
from demand.errors import PipelineError
from demand.schemas import (
    HOUR_START,
    TARGET,
    ZONE_ID,
    gold_demand_schema,
    silver_weather_schema,
)

ONE_HOUR = pd.Timedelta(hours=1)
REBUILD_ADVICE = "Rebuild the data with `dvc repro` and start the service again."


class SnapshotError(PipelineError):
    """The data files cannot serve forecasts: missing, invalid, incomplete or too short."""


class RequestRefusedError(PipelineError, ValueError):
    """A request asks for something this service cannot answer; the message says what can.

    ``field`` names the request parameter that has to change, ``value`` what it was.
    """

    def __init__(self, message: str, *, field: str, value: object, kind: str) -> None:
        super().__init__(message)
        self.field = field
        self.value = value
        self.kind = kind
        """A short machine-readable name of the reason, such as ``unknown_zone``."""


@dataclass(frozen=True)
class DataSnapshot:
    """The demand history and the weather the service reads, and the days they allow."""

    history: pd.DataFrame
    """Zone, hour and pickups: the gold demand table without its other columns."""
    weather: pd.DataFrame
    """Hour and one column per weather variable, one row per hour."""
    first_day: date
    last_day: date


def day_hours(day: date, settings: Settings) -> pd.DatetimeIndex:
    """Every local hour of ``day``: 24, or 23 and 25 on the days the clocks change."""
    return local_hours(day, day, settings.project.timezone)


def issue_moment(day: date, settings: Settings) -> pd.Timestamp:
    """The moment the forecast for ``day`` is issued, in the project timezone.

    It is 00:00 on the day before, unless a row of ``day`` reads an hour that is not
    complete by then. A row of hour ``t`` reads pickups up to the hour that starts at
    ``t - horizon``, complete one hour later (``features.py``); around the night the
    clocks go back that is 01:00 on the day before.
    """
    timezone = settings.project.timezone
    midnight_before = pd.Timestamp(day - timedelta(days=1), tz=timezone)
    last_hour = day_hours(day, settings)[-1]
    newest_read = last_hour - pd.Timedelta(hours=settings.features.horizon_hours)
    moment: pd.Timestamp = max(midnight_before, (newest_read + ONE_HOUR).tz_convert(timezone))
    return moment


def first_whole_day(moment: pd.Timestamp) -> date:
    """The first local day that starts at ``moment`` or later."""
    day: date = moment.date()
    return day if moment == moment.normalize() else day + timedelta(days=1)


def last_whole_day(end: pd.Timestamp) -> date:
    """The last local day that is over at ``end`` (the end of an hour) or earlier.

    A day is over at 00:00 on the next day, so this is always the day before the date
    of ``end``: at 00:00 the day before has just ended, and later on the same date the
    day of ``end`` has not ended yet.
    """
    day: date = end.date()
    return day - timedelta(days=1)


def _check_timeline(hours: pd.Series, what: str) -> pd.DatetimeIndex:
    """The distinct hours of ``hours``; refused unless they form one unbroken timeline."""
    distinct = pd.DatetimeIndex(hours.drop_duplicates()).sort_values()
    expected = pd.date_range(distinct[0], distinct[-1], freq="h")
    if len(distinct) != len(expected):
        raise SnapshotError(
            f"The {what} misses {len(expected) - len(distinct)} hours between "
            f"{distinct[0]} and {distinct[-1]}. {REBUILD_ADVICE}"
        )
    return distinct


def make_snapshot(demand: pd.DataFrame, weather: pd.DataFrame, settings: Settings) -> DataSnapshot:
    """Check the two tables and work out which days they can forecast.

    Args:
        demand: The gold demand table (``gold_demand_schema``).
        weather: The silver weather table (``silver_weather_schema``).
        settings: Project settings.

    Raises:
        SnapshotError: If a table does not pass its schema, the demand lacks a
            configured zone or an hour, the weather lacks an hour, or the two do not
            leave a single day to forecast.
    """
    try:
        demand = gold_demand_schema(settings).validate(demand)
        weather = silver_weather_schema(settings).validate(weather)
    except pandera.errors.SchemaError as error:
        raise SnapshotError(f"The data snapshot is not valid: {error}. {REBUILD_ADVICE}") from error
    missing_zones = sorted(set(settings.data.zones) - set(demand[ZONE_ID].unique().tolist()))
    if missing_zones:
        raise SnapshotError(
            f"The demand table has no rows for zone {', '.join(map(str, missing_zones))} of "
            f"data.zones. {REBUILD_ADVICE}"
        )
    demand_hours = _check_timeline(demand[HOUR_START], "demand table")
    if len(demand) != len(demand_hours) * len(settings.data.zones):
        raise SnapshotError(
            "The demand table does not have every hour for every zone. " + REBUILD_ADVICE
        )
    weather_hours = _check_timeline(weather[HOUR_START], "weather table")

    warm_up = pd.Timedelta(hours=warm_up_hours(settings))
    first_day = max(first_whole_day(demand_hours[0] + warm_up), first_whole_day(weather_hours[0]))
    last_day = last_whole_day(weather_hours[-1] + ONE_HOUR)
    known_until = demand_hours[-1] + ONE_HOUR
    while last_day >= first_day and issue_moment(last_day, settings) > known_until:
        last_day -= timedelta(days=1)
    if last_day < first_day:
        raise SnapshotError(
            f"The data snapshot leaves no day to forecast: demand from {demand_hours[0]} to "
            f"{demand_hours[-1]}, weather from {weather_hours[0]} to {weather_hours[-1]}. "
            f"A day needs {warm_up_hours(settings)} hours of demand before it. " + REBUILD_ADVICE
        )
    return DataSnapshot(
        history=demand[[ZONE_ID, HOUR_START, TARGET]].reset_index(drop=True),
        weather=weather.reset_index(drop=True),
        first_day=first_day,
        last_day=last_day,
    )


def load_snapshot(settings: Settings) -> DataSnapshot:
    """Read the gold demand and the silver weather from disk; see ``make_snapshot``.

    Raises:
        SnapshotError: If a file is missing, or ``make_snapshot`` refuses the tables.
    """
    paths = [gold_demand_path(settings.paths.gold), silver_weather_path(settings.paths.silver)]
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise SnapshotError(f"{' and '.join(missing)} not found. {REBUILD_ADVICE}")
    demand, weather = (pd.read_parquet(path) for path in paths)
    return make_snapshot(demand, weather, settings)


def check_request(
    snapshot: DataSnapshot, zones: Sequence[int], day: date, settings: Settings
) -> None:
    """Refuse a request for a zone or a day the service cannot forecast.

    Raises:
        RequestRefusedError: If a zone is not in ``data.zones``, or the day lies outside
            the days of the snapshot.
    """
    allowed = settings.data.zones
    unknown = [zone for zone in zones if zone not in allowed]
    if unknown:
        raise RequestRefusedError(
            f"Zone {', '.join(map(str, unknown))} is not forecast by this service. The "
            f"zones are {sorted(allowed)}.",
            field="zone",
            value=unknown,
            kind="unknown_zone",
        )
    if not snapshot.first_day <= day <= snapshot.last_day:
        raise RequestRefusedError(
            f"No forecast for {day.isoformat()}: this service forecasts the days "
            f"{snapshot.first_day.isoformat()} to {snapshot.last_day.isoformat()}. The first "
            f"is the first day with {warm_up_hours(settings)} hours of demand history before "
            "it; the last is the last day whose weather, and whose demand up to the moment "
            "the forecast is issued, are in the data snapshot.",
            field="date",
            value=day.isoformat(),
            kind="date_out_of_range",
        )


def day_features(
    snapshot: DataSnapshot, zones: Sequence[int], day: date, settings: Settings
) -> tuple[pd.Timestamp, pd.DataFrame]:
    """The moment the forecast for ``day`` is issued, and the model inputs of its rows.

    The rows are ordered by zone, in the order of ``zones``, then by hour. Only demand
    that had ended by the issue moment is passed to ``build_features``, and only as far
    back as the features read (``warm_up_hours``).

    Raises:
        RequestRefusedError: See ``check_request``.
    """
    check_request(snapshot, zones, day, settings)
    issued = issue_moment(day, settings)
    hours = day_hours(day, settings)
    history = snapshot.history
    oldest_read = hours[0] - pd.Timedelta(hours=warm_up_hours(settings))
    known = history[
        history[ZONE_ID].isin(zones)
        & (history[HOUR_START] >= oldest_read)
        & (history[HOUR_START] + ONE_HOUR <= issued)
    ]
    grid = pd.MultiIndex.from_product([list(zones), hours], names=[ZONE_ID, HOUR_START])
    targets = grid.to_frame(index=False).merge(snapshot.weather, on=HOUR_START, how="left")
    return issued, build_features(known, targets, settings)
