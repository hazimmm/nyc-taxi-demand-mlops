"""The reference for prediction drift: the backtest's forecasts, computed again.

The backtest (``models/backtest.py``) fits the recipe of the model once per test week,
on the weeks before it, and forecasts the week: forecasts for hours the model never
learned from. It keeps their scores, not the forecasts. This module repeats exactly that
loop for the LightGBM recipe of the config (the folds of ``fold_windows``, the
forecaster of ``lightgbm_forecaster``, which is deterministic and seeded) and keeps the
forecasts. Their MAE is returned too, so the caller can compare it with the backtest MAE
the registry holds for the champion: only if the two agree are these the forecasts of
that champion's backtest (``checks.py``).

It fits four models, so it takes one to two minutes, on every run of the command. The
replay of monitoring day by day computes it once and passes it to every day
(``flows/monitoring_flow.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from demand.config import Settings
from demand.models.backtest import TooLittleHistoryError, fold_windows, score
from demand.models.gbm import lightgbm_forecaster
from demand.schemas import FORECAST, HOUR_START, TARGET, ZONE_ID


@dataclass(frozen=True)
class ReferenceForecasts:
    """Out-of-sample forecasts of the backtest's test weeks."""

    rows: pd.DataFrame
    """``zone_id``, ``hour_start`` and ``forecast`` of every test row of every fold."""
    mae: float
    """Their pooled MAE, rounded as the backtest rounds it."""


def backtest_forecasts(table: pd.DataFrame, settings: Settings) -> ReferenceForecasts:
    """Fit the recipe on each fold's training rows and forecast its test week.

    Args:
        table: The training rows of the gold feature table
            (``models.backtest.load_training_features``).
        settings: The folds, the features and the model parameters.

    Raises:
        TooLittleHistoryError: If a fold has no training rows or no test rows.
    """
    hours = table[HOUR_START]
    forecaster = lightgbm_forecaster(settings)
    parts = []
    for fold in fold_windows(settings):
        train = table[hours <= fold.last_train_hour]
        test = table[(hours >= fold.first_test_hour) & (hours <= fold.last_test_hour)]
        if train.empty or test.empty:
            raise TooLittleHistoryError(
                f"Fold {fold.number} of the backtest has {len(train)} training rows and "
                f"{len(test)} test rows, so its forecasts cannot be made again for the "
                "prediction-drift reference. Rebuild the data with `dvc repro`."
            )
        forecaster.fit(train)
        parts.append(
            pd.DataFrame(
                {
                    ZONE_ID: test[ZONE_ID].to_numpy(),
                    HOUR_START: test[HOUR_START].reset_index(drop=True),
                    TARGET: test[TARGET].to_numpy(dtype="float64"),
                    FORECAST: forecaster.predict(test),
                }
            )
        )
    rows = pd.concat(parts, ignore_index=True)
    mae = score(
        rows[TARGET].to_numpy(dtype="float64"), rows[FORECAST].to_numpy(dtype="float64")
    ).mae
    return ReferenceForecasts(rows=rows.drop(columns=TARGET), mae=mae)
