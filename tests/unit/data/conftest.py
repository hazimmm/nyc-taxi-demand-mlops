"""Helpers for the data tests: a fake web server and tiny, valid source files.

Nothing here touches the network. ``FakeServer`` plugs into ``httpx.MockTransport``, so
the code under test uses a real ``httpx.Client`` whose requests are answered in memory.
The silver tests build their raw trips and weather with ``raw_trip`` and
``weather_payload``; the gold tests build silver trips with ``silver_trips_frame`` and
hourly demand with ``demand_frame``; the quality-gate tests also use
``silver_weather_frame``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timedelta

import httpx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from demand.config import Settings
from demand.data.clean_weather import local_hours
from demand.data.ingest import RetryPolicy, trip_url
from demand.data.weather import expected_hours, request_period
from demand.schemas import (
    DROPOFF_DATETIME,
    DURATION_MINUTES,
    HOUR_START,
    IS_OUTLIER,
    PASSENGER_COUNT,
    PICKUP_DATETIME,
    PICKUP_ZONE_ID,
    TARGET,
    TOTAL_AMOUNT,
    TRIP_DISTANCE,
    ZONE_ID,
    silver_trips_schema,
    silver_weather_schema,
)

Reply = Callable[[httpx.Request], httpx.Response]

FAST_RETRY = RetryPolicy(max_attempts=3, backoff_base_seconds=1.0)


def ok(content: bytes) -> Reply:
    """Reply with HTTP 200 and ``content``."""
    return lambda request: httpx.Response(200, content=content)


def status(code: int) -> Reply:
    """Reply with an HTTP error status."""
    return lambda request: httpx.Response(code, content=b"error page")


def network_error() -> Reply:
    """Fail before any response arrives, like a dropped connection."""

    def reply(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    return reply


def broken_stream(first_chunk: bytes) -> Reply:
    """Send ``first_chunk``, then lose the connection in the middle of the body."""

    def body() -> Iterator[bytes]:
        yield first_chunk
        raise httpx.ReadError("connection lost")

    return lambda request: httpx.Response(200, content=body())


class FakeServer:
    """Answers requests from a table of replies and remembers every request.

    ``server.add(url, first, second)`` answers the first request to ``url`` with
    ``first`` and every later one with ``second``. The query string is ignored when
    matching, so one route covers a weather API whatever its parameters.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._replies: dict[str, list[Reply]] = {}

    def add(self, url: str, *replies: Reply) -> None:
        self._replies[url] = list(replies)

    def requests_to(self, url: str) -> list[httpx.Request]:
        return [request for request in self.requests if _without_query(request.url) == url]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        replies = self._replies.get(_without_query(request.url))
        if replies is None:
            return httpx.Response(404, content=b"no such route in the fake server")
        reply = replies.pop(0) if len(replies) > 1 else replies[0]
        return reply(request)


def _without_query(url: httpx.URL) -> str:
    return str(url.copy_with(query=None))


def parquet_bytes(n_rows: int) -> bytes:
    """A small but real Parquet file with ``n_rows`` rows."""
    table = pa.table({"PULocationID": list(range(n_rows))})
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


def weather_bytes(
    variables: Sequence[str], hours: int, *, null_in: str | None = None, marker: float = 1.0
) -> bytes:
    """An Open-Meteo style response; ``null_in`` puts one null into that variable."""
    hourly: dict[str, list[object]] = {"time": [f"hour-{index}" for index in range(hours)]}
    for name in variables:
        hourly[name] = [marker] * hours
    if null_in is not None:
        hourly[null_in][0] = None
    return json.dumps({"hourly": hourly}).encode("utf-8")


def complete_weather(settings: Settings, *, marker: float = 1.0) -> bytes:
    """A weather response that covers the whole configured period without gaps."""
    hours = expected_hours(*request_period(settings.data.start_date, settings.data.end_date))
    return weather_bytes(settings.data.weather.variables, hours, marker=marker)


# --- raw files for the silver tests -----------------------------------------------------

# The columns silver is built from, plus a few it ignores, with the types of a real TLC
# file (checked with ``pyarrow.parquet.read_schema`` on yellow_tripdata_2025-01.parquet).
RAW_TRIPS_SCHEMA = pa.schema(
    [
        ("VendorID", pa.int32()),
        ("tpep_pickup_datetime", pa.timestamp("us")),
        ("tpep_dropoff_datetime", pa.timestamp("us")),
        ("passenger_count", pa.int64()),
        ("trip_distance", pa.float64()),
        ("store_and_fwd_flag", pa.large_string()),
        ("PULocationID", pa.int32()),
        ("DOLocationID", pa.int32()),
        ("payment_type", pa.int64()),
        ("fare_amount", pa.float64()),
        ("total_amount", pa.float64()),
    ]
)


def raw_trip(pickup: str, dropoff: str | None = None, **changes: object) -> dict[str, object]:
    """One valid raw trip starting at ``pickup``; ``changes`` overwrite single values.

    Without ``dropoff`` the trip lasts 15 minutes.
    """
    pickup_time = datetime.fromisoformat(pickup)
    dropoff_time = (
        datetime.fromisoformat(dropoff) if dropoff else pickup_time + timedelta(minutes=15)
    )
    trip: dict[str, object] = {
        "VendorID": 2,
        "tpep_pickup_datetime": pickup_time,
        "tpep_dropoff_datetime": dropoff_time,
        "passenger_count": 1,
        "trip_distance": 2.5,
        "store_and_fwd_flag": "N",
        "PULocationID": 161,
        "DOLocationID": 237,
        "payment_type": 1,
        "fare_amount": 14.5,
        "total_amount": 20.0,
    }
    unknown = set(changes) - set(trip)
    assert not unknown, f"not a raw column: {unknown}"
    return trip | changes


def raw_trips_table(trips: Sequence[dict[str, object]]) -> pa.Table:
    """Raw trips as an Arrow table with the real column types."""
    return pa.Table.from_pylist(list(trips), schema=RAW_TRIPS_SCHEMA)


def raw_trips_frame(trips: Sequence[dict[str, object]]) -> pd.DataFrame:
    """Raw trips as pandas reads them from a TLC file."""
    frame: pd.DataFrame = raw_trips_table(trips).to_pandas()
    return frame


def weather_payload(
    first_utc_hour: str,
    hours: int,
    variables: Sequence[str],
    *,
    nulls: Sequence[int] = (),
    absent: Sequence[int] = (),
) -> dict[str, object]:
    """An Open-Meteo style response with real UTC timestamps.

    The value of every variable at position ``i`` is ``float(i)``, which makes an
    interpolated value easy to predict. ``nulls`` are positions whose values are null;
    ``absent`` are positions that are left out of the response altogether.
    """
    first = datetime.fromisoformat(first_utc_hour)
    kept = [index for index in range(hours) if index not in absent]
    hourly: dict[str, list[object]] = {
        "time": [(first + timedelta(hours=index)).strftime("%Y-%m-%dT%H:%M") for index in kept]
    }
    for name in variables:
        hourly[name] = [None if index in nulls else float(index) for index in kept]
    return {"utc_offset_seconds": 0, "timezone": "GMT", "hourly": hourly}


def with_months(settings: Settings, start_month: str, end_month: str) -> Settings:
    """The same settings for a shorter period, so tests need only a few tiny files."""
    data = settings.data.model_copy(update={"start_month": start_month, "end_month": end_month})
    return settings.model_copy(update={"data": data})


# --- silver and gold tables for the gold tests ------------------------------------------


def silver_trips_frame(
    settings: Settings, pickups: Sequence[tuple[str, int]], *, complete: bool = True
) -> pd.DataFrame:
    """Silver trips from ``(pickup time with UTC offset, pickup zone)`` pairs.

    With ``complete=False`` the dropoff time, duration and distance are missing, as for
    a trip whose source did not record them. The result is checked against the silver
    schema, so the gold tests cannot drift away from what silver really produces.
    """
    timezone = settings.project.timezone
    pickup = pd.DatetimeIndex(
        [pd.Timestamp(time).tz_convert(timezone) for time, _ in pickups],
        dtype=f"M8[us, {timezone}]",
    )
    dropoff = pickup + pd.Timedelta(minutes=15)
    trips = pd.DataFrame(
        {
            PICKUP_DATETIME: pickup,
            # ``pickup + NaT`` is a column of missing times with the type of the pickups.
            DROPOFF_DATETIME: dropoff if complete else pickup + pd.NaT,
            PICKUP_ZONE_ID: pd.Series([zone for _, zone in pickups], dtype="int64"),
            TRIP_DISTANCE: 2.5 if complete else np.nan,
            PASSENGER_COUNT: 1.0,
            TOTAL_AMOUNT: 20.0,
            DURATION_MINUTES: 15.0 if complete else np.nan,
        }
    )
    return silver_trips_schema(settings).validate(trips)


def silver_weather_frame(settings: Settings) -> pd.DataFrame:
    """Silver weather for every hour of the configured period.

    The value of every variable at position ``i`` is ``i / 10``. The result is checked
    against the silver schema.
    """
    data = settings.data
    hours = local_hours(data.start_date, data.end_date, settings.project.timezone)
    values = [position / 10 for position in range(len(hours))]
    weather = pd.DataFrame(
        {HOUR_START: hours, **dict.fromkeys(settings.data.weather.variables, values)}
    )
    return silver_weather_schema(settings).validate(weather)


def demand_frame(
    settings: Settings,
    first_hour: str,
    hours: int,
    zones: Sequence[int] | None = None,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    """A gold demand table on a complete grid, with random pickups and weather.

    ``first_hour`` is local time. The same ``seed`` always gives the same table.
    """
    zones = list(settings.data.zones if zones is None else zones)
    random = np.random.default_rng(seed)
    timeline = pd.date_range(
        pd.Timestamp(first_hour, tz=settings.project.timezone), periods=hours, freq="h"
    )
    grid = pd.MultiIndex.from_product([sorted(zones), timeline], names=[ZONE_ID, HOUR_START])
    demand = grid.to_frame(index=False)
    demand[TARGET] = random.integers(0, 300, len(demand))
    demand[IS_OUTLIER] = False
    weather = {name: random.normal(10, 5, hours) for name in settings.data.weather.variables}
    return demand.merge(pd.DataFrame({HOUR_START: timeline, **weather}), on=HOUR_START)


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def waits() -> list[float]:
    """Collects the seconds the code asked to sleep; pass ``waits.append`` as ``sleep``."""
    return []


@pytest.fixture
def full_server(settings: Settings) -> FakeServer:
    """A server that has every file the default settings ask for."""
    server = FakeServer()
    data = settings.data
    for index, (year, month) in enumerate(data.months, start=1):
        server.add(trip_url(data.tlc_url_template, year, month), ok(parquet_bytes(index)))
    server.add(data.zone_lookup_url, ok(b"LocationID,Borough,Zone\n1,EWR,Newark Airport\n"))
    server.add(data.weather.forecast_url, ok(complete_weather(settings)))
    server.add(data.weather.archive_url, ok(complete_weather(settings, marker=2.0)))
    return server
