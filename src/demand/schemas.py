"""Data contracts between pipeline layers.

Every table that crosses a layer boundary has a column-name constant set and a Pandera
schema here. Producers validate against the schema before writing; consumers can rely
on it. Column names are defined once so training, serving and monitoring cannot drift
apart.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from demand.config import Settings

# Keys
ZONE_ID = "zone_id"
HOUR_START = "hour_start"

# Silver trips
PICKUP_DATETIME = "pickup_datetime"
DROPOFF_DATETIME = "dropoff_datetime"
PICKUP_ZONE_ID = "pickup_zone_id"
TRIP_DISTANCE = "trip_distance"
DURATION_MINUTES = "duration_minutes"
PASSENGER_COUNT = "passenger_count"
TOTAL_AMOUNT = "total_amount"

# Gold
TARGET = "pickups"
IS_OUTLIER = "is_outlier"

# Calendar features
HOUR = "hour"
DAY_OF_WEEK = "day_of_week"
MONTH = "month"
IS_WEEKEND = "is_weekend"
IS_HOLIDAY = "is_holiday"
CALENDAR_FEATURES = [HOUR, DAY_OF_WEEK, MONTH, IS_WEEKEND, IS_HOLIDAY]

# Predictions
FORECAST = "forecast"
ACTUAL = "actual"
MODEL_VERSION = "model_version"
GENERATED_AT = "generated_at"


def lag_column(hours: int) -> str:
    """Name of the demand-lag feature for ``hours`` hours back."""
    return f"lag_{hours}h"


def rolling_mean_column(hours: int) -> str:
    """Name of the rolling-mean feature over a ``hours``-hour window."""
    return f"roll_mean_{hours}h"


def rolling_std_column(hours: int) -> str:
    """Name of the rolling-standard-deviation feature over a ``hours``-hour window."""
    return f"roll_std_{hours}h"


def feature_columns(settings: Settings) -> list[str]:
    """Model input columns, in the exact order used for training and serving."""
    lags = [lag_column(lag) for lag in settings.features.lags]
    rolling = [
        column
        for window in settings.features.rolling_windows
        for column in (rolling_mean_column(window), rolling_std_column(window))
    ]
    return [ZONE_ID, *lags, *rolling, *CALENDAR_FEATURES, *settings.data.weather.variables]


def _timestamp_column(
    timezone: str, *, on_the_hour: bool = False, nullable: bool = False
) -> pa.Column:
    """Timezone-aware timestamp column, independent of the stored time resolution."""
    checks = [
        pa.Check(
            lambda series: (
                isinstance(series.dtype, pd.DatetimeTZDtype) and str(series.dtype.tz) == timezone
            ),
            error=f"must be timezone-aware in {timezone}",
        )
    ]
    if on_the_hour:
        checks.append(
            pa.Check(
                lambda series: (series.dt.minute == 0) & (series.dt.second == 0),
                error="must be aligned to the start of an hour",
            )
        )
    return pa.Column(checks=checks, nullable=nullable)


def _zone_column(settings: Settings, *, frozen_only: bool) -> pa.Column:
    """Zone ID column. ``frozen_only`` restricts it to the configured zone list.

    Raises:
        ValueError: If ``frozen_only`` is set but no zones are configured. Without this,
            the gold and prediction gates would silently accept any zone.
    """
    if not frozen_only:
        check = pa.Check.in_range(1, settings.cleaning.max_zone_id)
    elif settings.data.zones:
        check = pa.Check.isin(settings.data.zones)
    else:
        raise ValueError(
            "data.zones is empty: run `python -m demand select-zones` and paste the "
            "result into the config before building or validating the gold layer"
        )
    return pa.Column("int64", check, nullable=False)


def _weather_columns(settings: Settings) -> dict[str, pa.Column]:
    return {name: pa.Column("float64", nullable=False) for name in settings.data.weather.variables}


def silver_trips_schema(settings: Settings) -> pa.DataFrameSchema:
    """Cleaned trips: one row per valid trip."""
    cleaning = settings.cleaning
    timezone = settings.project.timezone
    return pa.DataFrameSchema(
        {
            PICKUP_DATETIME: _timestamp_column(timezone),
            # Dropoff time, duration and distance are missing where the source did not
            # record them (see data/clean.py). A recorded value must be in range.
            DROPOFF_DATETIME: _timestamp_column(timezone, nullable=True),
            PICKUP_ZONE_ID: _zone_column(settings, frozen_only=False),
            TRIP_DISTANCE: pa.Column(
                "float64",
                pa.Check.in_range(0, cleaning.max_distance_miles, include_min=False),
                nullable=True,
            ),
            DURATION_MINUTES: pa.Column(
                "float64",
                pa.Check.in_range(cleaning.min_duration_minutes, cleaning.max_duration_hours * 60),
                nullable=True,
            ),
            # Missing for a share of vendor-submitted trips that are otherwise valid,
            # so nulls are kept; only recorded values outside the range are removed.
            PASSENGER_COUNT: pa.Column(
                "float64",
                pa.Check.in_range(cleaning.min_passengers, cleaning.max_passengers),
                nullable=True,
            ),
            # A negative total is a refund or reversed charge, a second row for a trip
            # that is already there. The fare alone is not checked or kept: it is
            # negative on many real trips (see the notes in data/clean.py).
            TOTAL_AMOUNT: pa.Column("float64", pa.Check.ge(0)),
        },
        strict=True,
    )


def silver_weather_schema(settings: Settings) -> pa.DataFrameSchema:
    """Hourly weather: one row per hour, no gaps longer than the configured limit."""
    return pa.DataFrameSchema(
        {
            HOUR_START: _timestamp_column(settings.project.timezone, on_the_hour=True),
            **_weather_columns(settings),
        },
        strict=True,
        unique=[HOUR_START],
    )


def gold_demand_schema(settings: Settings) -> pa.DataFrameSchema:
    """Hourly pickups per zone on a complete grid, joined with weather."""
    return pa.DataFrameSchema(
        {
            ZONE_ID: _zone_column(settings, frozen_only=True),
            HOUR_START: _timestamp_column(settings.project.timezone, on_the_hour=True),
            TARGET: pa.Column("int64", pa.Check.ge(0)),
            IS_OUTLIER: pa.Column("bool"),
            **_weather_columns(settings),
        },
        strict=True,
        unique=[ZONE_ID, HOUR_START],
    )


def gold_features_schema(settings: Settings) -> pa.DataFrameSchema:
    """Model-ready table: keys, target and every feature, with no missing values."""
    features = settings.features
    lag_columns = {lag_column(lag): pa.Column("float64", pa.Check.ge(0)) for lag in features.lags}
    rolling_columns: dict[str, pa.Column] = {}
    for window in features.rolling_windows:
        rolling_columns[rolling_mean_column(window)] = pa.Column("float64", pa.Check.ge(0))
        rolling_columns[rolling_std_column(window)] = pa.Column("float64", pa.Check.ge(0))
    return pa.DataFrameSchema(
        {
            ZONE_ID: _zone_column(settings, frozen_only=True),
            HOUR_START: _timestamp_column(settings.project.timezone, on_the_hour=True),
            TARGET: pa.Column("int64", pa.Check.ge(0)),
            IS_OUTLIER: pa.Column("bool"),
            **lag_columns,
            **rolling_columns,
            HOUR: pa.Column("int64", pa.Check.in_range(0, 23)),
            DAY_OF_WEEK: pa.Column("int64", pa.Check.in_range(0, 6)),
            MONTH: pa.Column("int64", pa.Check.in_range(1, 12)),
            IS_WEEKEND: pa.Column("bool"),
            IS_HOLIDAY: pa.Column("bool"),
            **_weather_columns(settings),
        },
        strict=True,
        unique=[ZONE_ID, HOUR_START],
    )


def predictions_schema(settings: Settings) -> pa.DataFrameSchema:
    """Forecasts for whole days, issued a day before. ``actual`` is null until it is known."""
    timezone = settings.project.timezone
    return pa.DataFrameSchema(
        {
            ZONE_ID: _zone_column(settings, frozen_only=True),
            HOUR_START: _timestamp_column(timezone, on_the_hour=True),
            FORECAST: pa.Column("float64", pa.Check.ge(0)),
            ACTUAL: pa.Column("float64", pa.Check.ge(0), nullable=True),
            MODEL_VERSION: pa.Column(str),
            GENERATED_AT: _timestamp_column(timezone),
        },
        strict=True,
        unique=[ZONE_ID, HOUR_START],
    )
