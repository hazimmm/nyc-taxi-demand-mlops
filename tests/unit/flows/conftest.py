"""Helpers for the replay tests: a tiny snapshot, a replay period inside it, fake models.

The data are the API's test tables (``tests/unit/serving/conftest.py``): 25 February to
12 March 2025 for the test zones, so the snapshot forecasts 6 to 12 March. The replay
settings move ``data.train_end_date`` to 7 March: the "champion" exists from 00:00 on
8 March, so the replay forecasts 9 March (the day the clocks go forward, 23 hours) to 12
March, and its closing step stands at 00:00 on 13 March, the end of the demand history.
Nothing here reads ``data/``, ``mlflow.db`` or ``mlruns`` of the repository.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.features import FloatArray
from demand.flows.scoring import BacktestMae
from demand.schemas import feature_columns, lag_column
from demand.serving.model_loader import LoadedModel, Predictor
from demand.serving.snapshot import DataSnapshot, make_snapshot
from tests.unit.serving.conftest import model_info, tables

TRAIN_END = date(2025, 3, 7)
FIRST_REPLAY_DAY = date(2025, 3, 9)
LAST_DAY = date(2025, 3, 12)
REPLAY_DAYS = [date(2025, 3, 9), date(2025, 3, 10), date(2025, 3, 11), date(2025, 3, 12)]
BACKTEST_MAE = 20.0


class SumModel:
    """Forecasts the sum of every demand feature of a row, so that each one matters."""

    def __init__(self, settings: Settings) -> None:
        self.columns = [
            column for column in feature_columns(settings) if column.startswith(("lag_", "roll_"))
        ]
        self.calls = 0

    def predict(self, data: pd.DataFrame) -> FloatArray:
        self.calls += 1
        forecast: FloatArray = data[self.columns].sum(axis=1).to_numpy(dtype="float64")
        return forecast


class ShiftModel:
    """Forecasts ``lag_168h + shift``: the weekly baseline moved by a known amount."""

    def __init__(self, shift: float) -> None:
        self.shift = shift

    def predict(self, data: pd.DataFrame) -> FloatArray:
        forecast: FloatArray = data[lag_column(168)].to_numpy(dtype="float64") + self.shift
        return np.asarray(forecast, dtype="float64")


def replay(settings: Settings) -> Settings:
    """The test settings with the training period ending on ``TRAIN_END``."""
    return settings.model_copy(
        update={"data": settings.data.model_copy(update={"train_end_date": TRAIN_END})}
    )


def loaded(settings: Settings, predictor: Predictor, version: int = 7) -> LoadedModel:
    info = model_info(settings).model_copy(update={"version": version})
    return LoadedModel(predictor=predictor, info=info)


def fixed_backtest_mae(_: Settings) -> BacktestMae:
    return lambda version: BACKTEST_MAE


def snapshot_of(settings: Settings, demand: pd.DataFrame | None = None) -> DataSnapshot:
    tables_demand, weather = tables(settings, seed=5)
    return make_snapshot(tables_demand if demand is None else demand, weather, settings)


@pytest.fixture
def replay_settings(settings: Settings) -> Settings:
    return replay(settings)


@pytest.fixture
def snapshot(replay_settings: Settings) -> DataSnapshot:
    return snapshot_of(replay_settings)
