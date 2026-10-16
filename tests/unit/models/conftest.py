"""Helpers for the model tests: a tiny feature table and forecasters with known answers.

``feature_table`` builds a gold feature table whose pickups rise by one every hour, so
the error of a lag baseline is known without a calculator: a forecast that copies the
value of ``k`` hours ago is wrong by exactly ``k``. ``Lookup`` forecasts whatever a test
tells it to, and ``Spy`` records what the backtest hands to a forecaster. ``ONE_TREE`` and
``FEW_TREES`` are LightGBM parameters for tests: a model small enough to check by hand,
and a quick one that still draws random rows and columns.

``Store`` is a temporary MLflow store in which a test registers versions with the
parameters, tags and scores it needs, for the promotion gate; ``RUN_PARAMS`` and the
fold scores are copied from the two versions of the repository's store.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.entities import Metric, Param, RunTag

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


# One tree with a learning rate of 1 and no minimum leaf size: the forecast of a row is
# the mean target of the training rows in its leaf, which can be checked by hand.
# (LightGBM stores a leaf value with seven digits, so compare with ``pytest.approx``.)
ONE_TREE = {
    "objective": "regression",
    "n_estimators": 1,
    "learning_rate": 1.0,
    "num_leaves": 4,
    "min_child_samples": 1,
    "verbosity": -1,
}

# The parameters of the config with few trees: random rows and columns for every tree,
# as in a real run, and fast enough for a unit test.
FEW_TREES = {
    "objective": "poisson",
    "learning_rate": 0.05,
    "n_estimators": 20,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbosity": -1,
}


def with_model_params(settings: Settings, params: dict[str, object]) -> Settings:
    """The same settings with other LightGBM parameters."""
    return settings.model_copy(
        update={"model": settings.model.model_copy(update={"params": dict(params)})}
    )


def with_levels(table: pd.DataFrame, levels: dict[int, float]) -> pd.DataFrame:
    """``table`` with the same pickups in every hour of a zone: ``levels[zone]``."""
    return table.assign(**{TARGET: table[ZONE_ID].map(levels)})


def with_mlflow_store(settings: Settings, folder: Path) -> Settings:
    """The same settings with an MLflow database in ``folder``, never the one of the repo."""
    uri = f"sqlite:///{(folder / 'mlflow.db').as_posix()}"
    return settings.model_copy(
        update={"mlflow": settings.mlflow.model_copy(update={"tracking_uri": uri})}
    )


# What the training run of version 2 in the repository's store recorded, copied from it:
# the parameters that define the backtest, the tags, and the MAE of every fold.
RUN_PARAMS = {
    "model.learning_rate": "0.104",
    "data.train_end_date": "2025-04-30",
    "data.zones": "[161, 237, 236, 132, 230, 186, 162, 142, 234, 170, 163, 239, 68, 138, 79, "
    "48, 249, 141, 164, 140]",
    "features.horizon_hours": "48",
    "features.lags": "[48, 168]",
    "features.rolling_windows": "[24, 168]",
    "features.outlier_zscore": "4.0",
    "feature_columns": "['zone_id', 'lag_48h', 'lag_168h', 'roll_mean_24h', 'roll_std_24h', "
    "'roll_mean_168h', 'roll_std_168h', 'hour', 'day_of_week', 'month', 'is_weekend', "
    "'is_holiday', 'temperature_2m', 'precipitation', 'snowfall', 'wind_speed_10m']",
    "backtest.n_folds": "4",
    "backtest.fold_days": "7",
}
RUN_TAGS = {
    "forecaster": "lightgbm",
    "baseline": "weekly_naive",
    "git_commit": "5d1d9e4adc01ac34842c56dcab1e662e01dc27ac",
    "git_dirty": "false",
    "data_version": "7bacf1bf80cf4612d40552e74ef11cc5.dir",
}
BASELINE_FOLDS = (27.614583, 26.572619, 27.91756, 28.697917)
"""The weekly baseline per fold; pooled 27.70067."""
UNTUNED_FOLDS = (24.345801, 22.66614, 25.722257, 26.985474)
"""Version 1 of the repository's store; pooled 24.929918."""
TUNED_FOLDS = (23.510474, 21.84912, 24.753347, 26.311159)
"""Version 2 of the repository's store; pooled 24.106025."""


def pooled(folds: Sequence[float]) -> float:
    """The MAE over all rows of folds of equal size: the mean of the fold MAEs."""
    return round(sum(folds) / len(folds), 6)


def mae_metrics(
    model: Sequence[float], baseline: Sequence[float] = BASELINE_FOLDS
) -> dict[str, float]:
    """The MAE metrics a training run records, for the model and the baseline."""
    metrics = {"lightgbm.mae": pooled(model), "weekly_naive.mae": pooled(baseline)}
    for name, folds in (("lightgbm", model), ("weekly_naive", baseline)):
        metrics |= {f"{name}.fold_{n}.mae": mae for n, mae in enumerate(folds, start=1)}
    return metrics


@dataclass(frozen=True)
class Store:
    """A temporary MLflow store, and a quick way to register versions in it.

    A version is registered without a stored model: the gate reads only the run of a
    version, and saving a real model would take seconds per version.
    """

    settings: Settings

    @property
    def client(self) -> mlflow.MlflowClient:
        uri = self.settings.mlflow.tracking_uri
        return mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)

    @property
    def name(self) -> str:
        return self.settings.mlflow.registered_model_name

    def register(
        self,
        model: Sequence[float] = TUNED_FOLDS,
        *,
        params: Mapping[str, str] | None = None,
        tags: Mapping[str, str] | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> int:
        """Record a finished run and register it as the next version; its number."""
        client = self.client
        experiment = client.get_experiment_by_name(self.settings.mlflow.experiment_name)
        assert experiment is not None
        run = client.create_run(experiment.experiment_id)
        client.log_batch(
            run.info.run_id,
            metrics=[
                Metric(key, value, 0, 0)
                for key, value in (mae_metrics(model) if metrics is None else metrics).items()
            ],
            params=[Param(key, value) for key, value in (params or RUN_PARAMS).items()],
            tags=[RunTag(key, value) for key, value in (tags or RUN_TAGS).items()],
        )
        client.set_terminated(run.info.run_id)
        if not client.search_registered_models(f"name='{self.name}'"):
            client.create_registered_model(self.name)
        version = client.create_model_version(
            self.name, source=run.info.artifact_uri, run_id=run.info.run_id
        )
        return int(version.version)

    def aliases(self) -> dict[str, int]:
        aliases = self.client.get_registered_model(self.name).aliases
        return {alias: int(version) for alias, version in aliases.items()}

    def version_aliases(self, version: int) -> list[str]:
        return list(self.client.get_model_version(self.name, str(version)).aliases)

    def tags(self, version: int) -> dict[str, str]:
        return dict(self.client.get_model_version(self.name, str(version)).tags)


@pytest.fixture(scope="session")
def empty_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An MLflow database with the experiment and nothing else, created once.

    Creating a database takes MLflow seconds; copying the file takes a moment, so every
    test gets a copy of this one (``store``).
    """
    folder = tmp_path_factory.mktemp("empty_mlflow_store")
    uri = f"sqlite:///{(folder / 'mlflow.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("taxi-demand", artifact_location=(folder / "mlruns").as_uri())
    return folder / "mlflow.db"


@pytest.fixture
def store(
    settings: Settings, tmp_path: Path, empty_store: Path, monkeypatch: pytest.MonkeyPatch
) -> Store:
    """A fresh MLflow store under ``tmp_path``, which is also the working directory."""
    shutil.copy(empty_store, tmp_path / "mlflow.db")
    monkeypatch.chdir(tmp_path)
    return Store(with_mlflow_store(settings, tmp_path))
