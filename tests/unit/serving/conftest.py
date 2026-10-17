"""Helpers for the API tests: tiny demand and weather tables, and a model with known answers.

Nothing here reads ``data/``, ``mlflow.db`` or ``mlruns`` of the repository. ``tables``
builds a gold demand table and a silver weather table for a few weeks, on a complete
grid of the test zones. ``EchoModel`` stands in for the MLflow model: its forecast of a
row is the row's ``lag_168h``, so a test knows every answer, and it remembers what it was
given. ``service`` puts the two together the way ``load_service`` would.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demand.config import Settings
from demand.data.clean_weather import local_hours
from demand.data.features import FloatArray
from demand.schemas import HOUR_START, feature_columns, lag_column
from demand.serving.app import Service, create_app
from demand.serving.model_loader import LoadedModel, ModelInfo, Predictor
from demand.serving.snapshot import make_snapshot
from tests.unit.data.conftest import demand_frame

# 25 February to 12 March 2025: the first day with a full history is 6 March, and the
# clocks go forward on 9 March, so that day has 23 hours.
FIRST_DATA_DAY = date(2025, 2, 25)
LAST_DATA_DAY = date(2025, 3, 12)
FIRST_DAY = date(2025, 3, 6)
CLOCKS_FORWARD = date(2025, 3, 9)
VERSION = 7
RUN_ID = "0123456789abcdef0123456789abcdef"


def tables(
    settings: Settings,
    first: date = FIRST_DATA_DAY,
    last: date = LAST_DATA_DAY,
    *,
    weather_last: date | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A gold demand table from ``first`` to ``last`` and silver weather to ``weather_last``.

    The weather starts on ``first`` as well. Every hour of every test zone is there.
    """
    timezone = settings.project.timezone
    hours = len(local_hours(first, last, timezone))
    demand = demand_frame(settings, f"{first.isoformat()} 00:00", hours, seed=seed)
    weather_hours = local_hours(first, weather_last or last, timezone)
    random = np.random.default_rng(seed + 1)
    weather = pd.DataFrame(
        {
            HOUR_START: weather_hours,
            **{
                name: random.normal(10, 5, len(weather_hours))
                for name in settings.data.weather.variables
            },
        }
    )
    variables = settings.data.weather.variables
    # The demand carries the same weather as the weather table, as in the gold layer.
    demand = demand.drop(columns=variables).merge(weather, on=HOUR_START, how="left")
    return demand, weather


class EchoModel:
    """Forecasts the ``lag_168h`` of every row, and remembers every table it was given."""

    def __init__(self) -> None:
        self.given: list[pd.DataFrame] = []

    def predict(self, data: pd.DataFrame) -> FloatArray:
        self.given.append(data)
        forecast: FloatArray = data[lag_column(168)].to_numpy(dtype="float64")
        return forecast


def model_info(settings: Settings) -> ModelInfo:
    return ModelInfo(
        registered_model="taxi-demand",
        version=VERSION,
        alias="champion",
        model_uri="models:/taxi-demand@champion",
        run_id=RUN_ID,
        backtest_mae=24.106025,
        data_version="7bac.dir",
        git_commit="5d1d9e4",
        feature_columns=feature_columns(settings),
    )


def service(
    settings: Settings,
    model: Predictor | None = None,
    *,
    first: date = FIRST_DATA_DAY,
    last: date = LAST_DATA_DAY,
    weather_last: date | None = None,
) -> Service:
    """A service from ``tables`` and ``model`` (by default a new ``EchoModel``)."""
    demand, weather = tables(settings, first, last, weather_last=weather_last)
    return Service(
        model=LoadedModel(predictor=model or EchoModel(), info=model_info(settings)),
        snapshot=make_snapshot(demand, weather, settings),
    )


@pytest.fixture
def echo() -> EchoModel:
    return EchoModel()


@pytest.fixture
def client(settings: Settings, echo: EchoModel) -> Iterator[TestClient]:
    """A started app with the tiny service; the startup has run."""
    loaded = service(settings, echo)
    with TestClient(create_app(settings, load=lambda _: loaded)) as started:
        yield started
