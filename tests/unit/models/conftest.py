"""Helpers for the model tests: a tiny feature table and forecasters with known answers.

``feature_table`` builds a gold feature table whose pickups rise by one every hour, so
the error of a lag baseline is known without a calculator: a forecast that copies the
value of ``k`` hours ago is wrong by exactly ``k``. ``Lookup`` forecasts whatever a test
tells it to, and ``Spy`` records what the backtest hands to a forecaster.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import pandas as pd

from demand.config import Settings
from demand.data.aggregate import training_period_end
from demand.data.features import FloatArray, calendar_features
from demand.schemas import (
    HOUR_START,
    IS_OUTLIER,
    TARGET,
    ZONE_ID,
    feature_columns,
    gold_features_schema,
    lag_column,
)

# Pickups of the first hour of the first zone. Larger than the longest lag, so that
# "pickups minus 168" is never negative.
FIRST_PICKUPS = 200
PICKUPS_BETWEEN_ZONES = 1000
KEYS = [ZONE_ID, HOUR_START]


def with_backtest(
    settings: Settings, *, train_end_date: date, n_folds: int, fold_days: int
) -> Settings:
    """The same settings with another backtest layout and end of the training period."""
    return settings.model_copy(
        update={
            "data": settings.data.model_copy(update={"train_end_date": train_end_date}),
            "backtest": settings.backtest.model_copy(
                update={"n_folds": n_folds, "fold_days": fold_days}
            ),
        }
    )


def feature_table(
    settings: Settings, first_hour: str, hours: int, zones: Sequence[int] | None = None
) -> pd.DataFrame:
    """A gold feature table on a complete grid, with pickups that are easy to predict.

    ``first_hour`` is local time. In every zone the pickups rise by 1 per hour; the
    first zone starts at 200, the second at 1,200, and so on. A lag column of ``k``
    hours is the pickups minus ``k``, which is what a real lag would be on this table.
    The other model inputs are 0, except the calendar columns. The result is checked
    against the gold schema, so the tests cannot drift away from what gold produces.
    """
    zones = sorted(settings.data.zones if zones is None else zones)
    timeline = pd.date_range(
        pd.Timestamp(first_hour, tz=settings.project.timezone), periods=hours, freq="h"
    )
    grid = pd.MultiIndex.from_product([zones, timeline], names=KEYS)
    table = grid.to_frame(index=False)
    position = pd.Series(timeline.get_indexer(pd.DatetimeIndex(table[HOUR_START])))
    zone_number = table[ZONE_ID].map({zone: number for number, zone in enumerate(zones)})
    table[TARGET] = FIRST_PICKUPS + position + PICKUPS_BETWEEN_ZONES * zone_number
    table[IS_OUTLIER] = False
    calendar = calendar_features(table[HOUR_START])
    for column in feature_columns(settings):
        if column == ZONE_ID:
            continue
        if column in calendar:
            table[column] = calendar[column]
        elif column in _lag_hours(settings):
            table[column] = (table[TARGET] - _lag_hours(settings)[column]).astype("float64")
        else:
            table[column] = 0.0
    return gold_features_schema(settings).validate(table)


def _lag_hours(settings: Settings) -> dict[str, int]:
    return {lag_column(lag): lag for lag in settings.features.lags}


def training_rows(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The rows of ``table`` that lie in the training period."""
    return table[table[HOUR_START] < training_period_end(settings)].reset_index(drop=True)


class Lookup:
    """Forecasts the value a test wrote down for every zone and hour."""

    def __init__(self, table: pd.DataFrame, forecast: pd.Series | None = None) -> None:
        """With no ``forecast`` the pickups of ``table`` are forecast: a perfect forecaster."""
        values = table[TARGET] if forecast is None else forecast
        self.values = pd.Series(
            values.to_numpy(dtype="float64"), index=pd.MultiIndex.from_frame(table[KEYS])
        )

    def fit(self, train: pd.DataFrame) -> None:
        pass

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        wanted = pd.MultiIndex.from_frame(rows[KEYS])
        forecast: FloatArray = self.values.reindex(wanted).to_numpy(dtype="float64")
        return forecast


class Spy:
    """Forecasts 0 and remembers every table it was fitted on and asked to forecast."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.fitted_on: list[pd.DataFrame] = []
        self.asked_for: list[pd.DataFrame] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.calls.append("fit")
        self.fitted_on.append(train)

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        self.calls.append("predict")
        self.asked_for.append(rows)
        forecast: FloatArray = rows[ZONE_ID].to_numpy(dtype="float64") * 0
        return forecast
