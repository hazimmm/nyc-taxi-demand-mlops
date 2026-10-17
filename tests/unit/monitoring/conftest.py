"""Small hand-made tables and a hand-made replay on disk for the monitoring tests.

The settings are the test settings (three zones, paths under ``tmp_path``) with the
config's training period (to 30 April 2025), ``window_days`` 7 and ``reference_days``
28. ``write_replay`` writes forecasts, inputs and scored files the way the replay does,
with values chosen by the test. Nothing here reads ``data/``, ``mlflow.db`` or
``mlruns`` of the repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from demand.config import Settings
from demand.data.silver import write_parquet
from demand.flows.clock import day_end, day_file, forecasts_dir, inputs_dir, scored_dir
from demand.monitoring.checks import DAY, drift_columns
from demand.monitoring.reference import ReferenceForecasts
from demand.schemas import (
    ACTUAL,
    FORECAST,
    GENERATED_AT,
    HOUR_START,
    IS_HOLIDAY,
    IS_OUTLIER,
    MODEL_VERSION,
    TARGET,
    ZONE_ID,
    feature_columns,
    lag_column,
)
from demand.serving.snapshot import day_hours, issue_moment

FIRST_DAY = date(2025, 5, 2)
LEVEL = 100.0
SPREAD = 10.0


def days_from(first: date, count: int) -> list[date]:
    return [first + timedelta(days=offset) for offset in range(count)]


def grid(settings: Settings, days: Sequence[date]) -> pd.DataFrame:
    """Every zone and hour of ``days``, with a ``day`` column."""
    parts = [
        pd.DataFrame({ZONE_ID: zone, HOUR_START: day_hours(day, settings), DAY: day})
        for day in days
        for zone in settings.data.zones
    ]
    return pd.concat(parts, ignore_index=True)


def normal_values(count: int, level: float = LEVEL) -> np.ndarray:
    """``count`` evenly spaced quantiles of a normal distribution around ``level``."""
    return level + SPREAD * scipy.stats.norm.ppf((np.arange(count) + 0.5) / count)


def numbers(
    rows: pd.DataFrame, settings: Settings, seed: int, shift: dict[str, float]
) -> pd.DataFrame:
    """``rows`` with every tested input spread around ``LEVEL``, moved by ``shift``.

    The values are the evenly spaced quantiles of a normal distribution, in an order
    shuffled with ``seed``: two tables made this way have the same distribution up to
    the number of rows, so an unshifted column shows no drift by chance.
    """
    rng = np.random.default_rng(seed)
    table = rows.copy()
    for column in drift_columns(settings):
        table[column] = rng.permutation(normal_values(len(table))) + shift.get(column, 0.0)
    table[IS_HOLIDAY] = False
    return table


def training_table(settings: Settings, seed: int = 1) -> pd.DataFrame:
    """Training rows of the last 35 days before the end of training, as the gold table has."""
    end = settings.data.train_end_date
    table = numbers(grid(settings, days_from(end - timedelta(days=34), 35)), settings, seed, {})
    table[TARGET] = 100
    table[IS_OUTLIER] = False
    others = [column for column in feature_columns(settings) if column not in table.columns]
    for column in others:
        table[column] = 0
    return table.drop(columns=DAY)


def write_replay(
    settings: Settings,
    days: Sequence[date],
    *,
    scored: Sequence[date] = (),
    version: str = "2",
    forecast: float = LEVEL,
    forecast_spread: float = 0.0,
    actual: float = LEVEL,
    baseline: float | None = None,
    shift: dict[str, float] | None = None,
    seed: int = 2,
) -> None:
    """Forecasts and inputs of ``days``, and scored files of ``scored``.

    The forecasts are ``forecast``, plus noise of standard deviation ``forecast_spread``;
    the inputs are drawn around ``LEVEL`` (moved by ``shift``); ``lag_168h``, the
    weekly baseline, is ``baseline`` if it is given.
    """
    for number, day in enumerate(days):
        rows = grid(settings, [day]).drop(columns=DAY)
        inputs = numbers(rows, settings, seed + number, shift or {})
        if baseline is not None:
            inputs[lag_column(168)] = baseline
        write_parquet(inputs, day_file(inputs_dir(settings), day))
        table = pd.DataFrame(
            {
                ZONE_ID: rows[ZONE_ID],
                HOUR_START: rows[HOUR_START],
                FORECAST: forecast + forecast_spread / SPREAD * (normal_values(len(rows), 0.0)),
                ACTUAL: np.nan,
                MODEL_VERSION: version,
                GENERATED_AT: issue_moment(day, settings),
            }
        )
        write_parquet(table, day_file(forecasts_dir(settings), day))
        if day in scored:
            write_parquet(table.assign(**{ACTUAL: actual}), day_file(scored_dir(settings), day))


def moment(day: date, settings: Settings) -> pd.Timestamp:
    """00:00 on ``day``."""
    return pd.Timestamp(day, tz=settings.project.timezone)


def over_at(days: Sequence[date], now: pd.Timestamp, settings: Settings) -> list[date]:
    return [day for day in days if day_end(day, settings) <= now]


class FixedReference:
    """A ``make_reference``: 2,016 forecasts drawn around ``level``, and a fixed MAE."""

    def __init__(self, level: float = LEVEL, mae: float = 20.0) -> None:
        self.level = level
        self.mae = mae
        self.calls = 0

    def __call__(self, table: pd.DataFrame, settings: Settings) -> ReferenceForecasts:
        self.calls += 1
        rows = pd.DataFrame({FORECAST: normal_values(2016, self.level)})
        return ReferenceForecasts(rows=rows, mae=self.mae)


def write_garbage(folder: Path, day: date) -> None:
    """A file named like a day of the replay that cannot be read as Parquet."""
    folder.mkdir(parents=True, exist_ok=True)
    day_file(folder, day).write_text("not parquet", encoding="utf-8")


@pytest.fixture
def training(settings: Settings) -> pd.DataFrame:
    return training_table(settings)
