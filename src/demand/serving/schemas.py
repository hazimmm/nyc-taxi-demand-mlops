"""The shapes of the API's requests and responses, checked by Pydantic.

FastAPI reads these to validate every request, to write every response and to build the
documentation at ``/docs``; the examples below are what ``/docs`` shows.

**Strict where it matters.** Pydantic converts generously by default: it would read the
day ``1747180800`` (seconds since 1970) or ``2025-05-14T00:00`` as 14 May, and the zone
``161.0`` as 161. A forecast for a day the client did not mean is worse than a refusal,
so a day must be written ``YYYY-MM-DD`` and a zone as a whole number, and an unknown
query parameter (``zones=`` instead of ``zone=``) is refused rather than ignored. Which
zones and days exist is checked by the service (``snapshot.check_request``), because it
depends on the config and the data, not on the shape of the request.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator

_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_WHOLE_NUMBER = re.compile(r"\d+")

EXAMPLE_DAY = "2025-05-14"
EXAMPLE_ZONE = 161


def _written_as_day(value: object) -> object:
    if isinstance(value, str) and _DAY.fullmatch(value):
        return value
    raise ValueError("must be a day written as YYYY-MM-DD, such as 2025-05-14")


def _written_as_whole_number(value: object) -> object:
    if isinstance(value, str) and _WHOLE_NUMBER.fullmatch(value):
        return value
    raise ValueError("must be a zone ID written as a whole number, such as 161")


Day = Annotated[dt.date, BeforeValidator(_written_as_day)]
ZoneId = Annotated[int, BeforeValidator(_written_as_whole_number)]


class _Response(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ForecastQuery(BaseModel):
    """The query parameters of ``GET /predict``."""

    model_config = ConfigDict(extra="forbid")

    date: Day = Field(
        description="The local day (New York time) to forecast, written YYYY-MM-DD.",
        examples=[EXAMPLE_DAY],
    )
    zone: list[ZoneId] = Field(
        min_length=1,
        description="A pickup zone ID; repeat the parameter for several zones "
        "(zone=161&zone=237). Each zone at most once.",
        examples=[[EXAMPLE_ZONE]],
    )

    @field_validator("zone")
    @classmethod
    def _each_zone_once(cls, zones: list[int]) -> list[int]:
        repeated = sorted({zone for zone in zones if zones.count(zone) > 1})
        if repeated:
            raise ValueError(f"names zone {', '.join(map(str, repeated))} more than once")
        return zones


class HourlyForecast(_Response):
    """The forecast pickups of one zone in one hour."""

    zone_id: int = Field(examples=[EXAMPLE_ZONE])
    hour_start: dt.datetime = Field(
        description="The start of the hour, local time with its UTC offset.",
        examples=["2025-05-14T08:00:00-04:00"],
    )
    forecast: float = Field(
        ge=0,
        description="Expected pickups in the hour; never negative, not rounded.",
        examples=[143.27],
    )


class ForecastResponse(_Response):
    """Every hourly forecast of one day for the requested zones."""

    date: dt.date = Field(examples=[EXAMPLE_DAY])
    registered_model: str = Field(examples=["taxi-demand"])
    model_version: int = Field(
        description="The registered version that computed these forecasts.", examples=[2]
    )
    issued_at: dt.datetime = Field(
        description="The moment the forecast is issued as of (normally 00:00 on the day "
        "before). Only demand known by then was read; this is simulated time, not the "
        "time of the request.",
        examples=["2025-05-13T00:00:00-04:00"],
    )
    hours: int = Field(
        description="Hours of the day per zone: 24, or 23 and 25 when the clocks change.",
        examples=[24],
    )
    forecasts: list[HourlyForecast] = Field(
        description="One per zone and hour: zones in the order asked, then hours in order."
    )


class HealthResponse(_Response):
    """The process is alive."""

    status: str = Field(examples=["ok"])


class ReadyResponse(_Response):
    """The model and the data snapshot are loaded."""

    status: str = Field(examples=["ready"])
    model_version: int = Field(examples=[2])


class ForecastDays(_Response):
    """The first and the last day the service forecasts."""

    first: dt.date = Field(examples=["2025-01-10"])
    last: dt.date = Field(examples=["2025-06-30"])


class ModelInfoResponse(_Response):
    """Which model is served, where it came from, and what it can be asked."""

    registered_model: str = Field(examples=["taxi-demand"])
    version: int = Field(examples=[2])
    alias: str | None = Field(
        description="The alias the version was found by at startup; null if pinned.",
        examples=["champion"],
    )
    model_uri: str = Field(examples=["models:/taxi-demand@champion"])
    run_id: str = Field(examples=["a8aa7b305856481b984bd97586d2f549"])
    backtest_mae: float | None = Field(
        description="Pooled MAE of the recipe on the four April backtest weeks, in pickups.",
        examples=[24.106025],
    )
    data_version: str | None = Field(examples=["7bacf1bf80cf4612d40552e74ef11cc5.dir"])
    git_commit: str | None = Field(examples=["5d1d9e4adc01ac34842c56dcab1e662e01dc27ac"])
    feature_columns: list[str] = Field(examples=[["zone_id", "lag_48h", "lag_168h"]])
    zones: list[int] = Field(examples=[[161, 237]])
    forecast_days: ForecastDays
