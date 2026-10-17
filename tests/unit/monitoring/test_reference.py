from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.features import FloatArray
from demand.models.backtest import TooLittleHistoryError, backtest, fold_windows
from demand.models.gbm import LIGHTGBM, lightgbm_forecaster
from demand.monitoring import reference
from demand.monitoring.reference import backtest_forecasts
from demand.schemas import FORECAST, HOUR_START, TARGET, lag_column
from tests.unit.monitoring.conftest import training_table


class LagForecaster:
    """Forecasts ``lag_168h`` and records the newest hour of every fit."""

    def __init__(self) -> None:
        self.fitted_until: list[pd.Timestamp] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.fitted_until.append(train[HOUR_START].max())

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        forecast: FloatArray = rows[lag_column(168)].to_numpy(dtype="float64")
        return forecast


def with_target(settings: Settings, seed: int = 4) -> pd.DataFrame:
    table = training_table(settings)
    table[TARGET] = np.random.default_rng(seed).integers(50, 150, len(table))
    return table


def test_each_fold_is_fitted_before_its_week_and_forecasts_it(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = LagForecaster()
    monkeypatch.setattr(reference, "lightgbm_forecaster", lambda _: fake)
    table = with_target(settings)

    result = backtest_forecasts(table, settings)

    folds = fold_windows(settings)
    assert fake.fitted_until == [pd.Timestamp(fold.last_train_hour) for fold in folds]
    tested = table[table[HOUR_START] >= folds[0].first_test_hour]
    assert len(result.rows) == len(tested) == 3 * 24 * 28
    assert list(result.rows.columns) == ["zone_id", HOUR_START, FORECAST]
    expected = np.abs(tested[TARGET] - tested[lag_column(168)]).mean()
    assert result.mae == pytest.approx(expected, abs=1e-6)


def test_the_forecasts_are_those_behind_the_backtest_mae(settings: Settings) -> None:
    fast = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={"params": settings.model.params | {"n_estimators": 20}}
            )
        }
    )
    table = with_target(fast)

    recorded = backtest(table, {LIGHTGBM: lightgbm_forecaster(fast)}, fast)

    assert backtest_forecasts(table, fast).mae == recorded.forecasters[LIGHTGBM].overall.mae


def test_a_fold_without_training_rows_is_an_error(settings: Settings) -> None:
    table = with_target(settings)
    late = table[table[HOUR_START] >= fold_windows(settings)[0].first_test_hour]

    with pytest.raises(TooLittleHistoryError, match="Fold 1"):
        backtest_forecasts(late, settings)
