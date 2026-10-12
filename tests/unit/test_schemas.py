from __future__ import annotations

import pandas as pd
import pandera.errors
import pytest

from demand import schemas
from demand.config import Settings

TZ = "America/New_York"


def _hours(n: int, start: str = "2025-03-08 22:00") -> pd.DatetimeIndex:
    """Hourly timestamps that cross the March daylight-saving change."""
    return pd.date_range(start, periods=n, freq="h", tz=TZ)


def _weather(settings: Settings, n: int) -> dict[str, list[float]]:
    return {name: [1.0] * n for name in settings.data.weather.variables}


def _silver_trips() -> pd.DataFrame:
    pickup = pd.Timestamp("2025-01-15 08:05", tz=TZ)
    return pd.DataFrame(
        {
            schemas.PICKUP_DATETIME: [pickup, pickup],
            schemas.DROPOFF_DATETIME: [pickup + pd.Timedelta(minutes=12)] * 2,
            schemas.PICKUP_ZONE_ID: [132, 161],
            schemas.TRIP_DISTANCE: [2.4, 0.9],
            schemas.DURATION_MINUTES: [12.0, 12.0],
            schemas.PASSENGER_COUNT: [1.0, None],
            schemas.FARE_AMOUNT: [14.5, 8.0],
        }
    )


def _gold_demand(settings: Settings) -> pd.DataFrame:
    hours = _hours(4)
    return pd.DataFrame(
        {
            schemas.ZONE_ID: [132] * 4,
            schemas.HOUR_START: hours,
            schemas.TARGET: [5, 0, 3, 9],
            schemas.IS_OUTLIER: [False] * 4,
            **_weather(settings, 4),
        }
    )


def _gold_features(settings: Settings) -> pd.DataFrame:
    frame = _gold_demand(settings)
    for column in schemas.feature_columns(settings):
        if column in frame.columns:
            continue
        frame[column] = 1.0
    hours = frame[schemas.HOUR_START]
    frame[schemas.HOUR] = hours.dt.hour.astype("int64")
    frame[schemas.DAY_OF_WEEK] = hours.dt.dayofweek.astype("int64")
    frame[schemas.MONTH] = hours.dt.month.astype("int64")
    frame[schemas.IS_WEEKEND] = hours.dt.dayofweek >= 5
    frame[schemas.IS_HOLIDAY] = False
    return frame


def test_feature_columns_order_is_stable(settings: Settings) -> None:
    assert schemas.feature_columns(settings) == [
        "zone_id",
        "lag_24h",
        "lag_48h",
        "lag_168h",
        "roll_mean_24h",
        "roll_std_24h",
        "roll_mean_168h",
        "roll_std_168h",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "is_holiday",
        "temperature_2m",
        "precipitation",
        "snowfall",
        "wind_speed_10m",
    ]


def test_features_schema_covers_every_feature_column(settings: Settings) -> None:
    schema_columns = set(schemas.gold_features_schema(settings).columns)

    assert set(schemas.feature_columns(settings)) <= schema_columns
    assert {schemas.TARGET, schemas.HOUR_START} <= schema_columns


def test_valid_silver_trips_pass(settings: Settings) -> None:
    schemas.silver_trips_schema(settings).validate(_silver_trips())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (schemas.PICKUP_ZONE_ID, 264),
        (schemas.TRIP_DISTANCE, 0.0),
        (schemas.TRIP_DISTANCE, 150.0),
        (schemas.DURATION_MINUTES, 0.5),
        (schemas.DURATION_MINUTES, 361.0),
        (schemas.PASSENGER_COUNT, 0.0),
        (schemas.PASSENGER_COUNT, 9.0),
        (schemas.FARE_AMOUNT, -3.0),
    ],
)
def test_silver_trips_reject_bad_values(settings: Settings, column: str, value: float) -> None:
    frame = _silver_trips()
    frame.loc[0, column] = value

    with pytest.raises(pandera.errors.SchemaError):
        schemas.silver_trips_schema(settings).validate(frame)


def test_silver_trips_reject_naive_timestamps(settings: Settings) -> None:
    frame = _silver_trips()
    frame[schemas.PICKUP_DATETIME] = frame[schemas.PICKUP_DATETIME].dt.tz_localize(None)

    with pytest.raises(pandera.errors.SchemaError):
        schemas.silver_trips_schema(settings).validate(frame)


def test_silver_trips_reject_unexpected_columns(settings: Settings) -> None:
    frame = _silver_trips().assign(tip_amount=1.0)

    # Schema-level failures (unlike failed column checks) raise the plural error.
    with pytest.raises(pandera.errors.SchemaErrors):
        schemas.silver_trips_schema(settings).validate(frame)


def test_valid_silver_weather_passes(settings: Settings) -> None:
    frame = pd.DataFrame({schemas.HOUR_START: _hours(3), **_weather(settings, 3)})

    schemas.silver_weather_schema(settings).validate(frame)


def test_silver_weather_rejects_missing_values(settings: Settings) -> None:
    frame = pd.DataFrame({schemas.HOUR_START: _hours(3), **_weather(settings, 3)})
    frame.loc[1, settings.data.weather.variables[0]] = None

    with pytest.raises(pandera.errors.SchemaError):
        schemas.silver_weather_schema(settings).validate(frame)


def test_valid_gold_demand_passes_across_daylight_saving(settings: Settings) -> None:
    schemas.gold_demand_schema(settings).validate(_gold_demand(settings))


def test_gold_demand_rejects_duplicate_keys(settings: Settings) -> None:
    frame = _gold_demand(settings)
    frame.loc[1, schemas.HOUR_START] = frame.loc[0, schemas.HOUR_START]

    with pytest.raises(pandera.errors.SchemaError):
        schemas.gold_demand_schema(settings).validate(frame)


def test_gold_demand_rejects_zone_outside_frozen_list(settings: Settings) -> None:
    frame = _gold_demand(settings)
    frame[schemas.ZONE_ID] = 1

    with pytest.raises(pandera.errors.SchemaError):
        schemas.gold_demand_schema(settings).validate(frame)


def test_gold_demand_rejects_timestamps_off_the_hour(settings: Settings) -> None:
    frame = _gold_demand(settings)
    frame[schemas.HOUR_START] = frame[schemas.HOUR_START] + pd.Timedelta(minutes=30)

    with pytest.raises(pandera.errors.SchemaError):
        schemas.gold_demand_schema(settings).validate(frame)


def test_gold_demand_rejects_negative_pickups(settings: Settings) -> None:
    frame = _gold_demand(settings)
    frame.loc[0, schemas.TARGET] = -1

    with pytest.raises(pandera.errors.SchemaError):
        schemas.gold_demand_schema(settings).validate(frame)


def test_valid_gold_features_pass(settings: Settings) -> None:
    schemas.gold_features_schema(settings).validate(_gold_features(settings))


def test_gold_features_reject_missing_lag(settings: Settings) -> None:
    frame = _gold_features(settings)
    frame.loc[0, schemas.lag_column(24)] = None

    with pytest.raises(pandera.errors.SchemaError):
        schemas.gold_features_schema(settings).validate(frame)


def test_predictions_allow_missing_actuals(settings: Settings) -> None:
    hours = _hours(2)
    frame = pd.DataFrame(
        {
            schemas.ZONE_ID: [132, 132],
            schemas.HOUR_START: hours,
            schemas.FORECAST: [4.2, 6.0],
            schemas.ACTUAL: [5.0, None],
            schemas.MODEL_VERSION: ["3", "3"],
            schemas.GENERATED_AT: [hours[0] - pd.Timedelta(hours=24)] * 2,
        }
    )

    schemas.predictions_schema(settings).validate(frame)
