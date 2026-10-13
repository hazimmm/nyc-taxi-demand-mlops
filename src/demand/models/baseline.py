"""The forecaster interface, and the forecasts that need no model at all.

**Why a baseline.** "The model has a mean error of 15 pickups" means nothing alone. It
means something next to the error of the simplest forecast anyone could make. If a
gradient-boosted model with all its inputs cannot beat "the same as last week", it has
learned nothing worth deploying. The baseline is the number to beat, and it is measured
before any model exists so that nobody can choose it afterwards to flatter the model.

**The baselines.** Each copies a lag the feature table already holds, so which ones
exist depends on ``features.lags``, and through it on the horizon:

* ``weekly_naive``: the pickups of the same zone 168 hours earlier, the column
  ``lag_168h``. Taxi demand repeats by the hour of the day *and* by the day of the week,
  and this forecast gets both right. It is the main baseline, called the *seasonal
  naive* forecast in textbooks, with the week as the season. It must exist: without
  ``lag_168h`` the backtest refuses to run.
* ``daily_naive``: the pickups 24 hours earlier, ``lag_24h``. It is offered only when
  24 is among ``features.lags``, and the config allows that lag only for a horizon of
  24 hours or less. **With the horizon of this project, 48 hours, it does not exist:**
  "the same hour yesterday" has not happened yet when a forecast is made two days
  ahead, so the config refuses the lag and the weekly baseline is the only one. Where
  it does exist it uses newer information than the weekly baseline but copies a Sunday
  onto a Monday, which shows that newer is not always better: it is the weaker of the
  two on this data.

No baseline reads anything a real forecast could not have: every lag is a feature, and
a feature of hour ``t`` never looks closer than ``features.horizon_hours`` (see
``features.py``). A lag counts elapsed hours, so in the week after a clock change
``lag_168h`` points at the clock hour next to the one a person would expect.

**The interface.** ``Forecaster`` has two methods, ``fit`` on a table of past rows and
``predict`` on a table of rows to forecast. A baseline ignores ``fit``. A fitted model
(LightGBM, in ``gbm.py``) learns in ``fit``. The backtest only knows the two
methods, so it treats both kinds the same way and does not change when a model is added.
It is a ``Protocol``: a class satisfies it by having the two methods, without inheriting
from anything.
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from demand.config import Settings
from demand.data.features import FloatArray
from demand.errors import PipelineError
from demand.schemas import lag_column

HOURS_PER_DAY = 24
HOURS_PER_WEEK = 7 * HOURS_PER_DAY

WEEKLY_NAIVE = "weekly_naive"
DAILY_NAIVE = "daily_naive"


class Forecaster(Protocol):
    """Anything that can learn from past rows and then forecast pickups for other rows."""

    def fit(self, train: pd.DataFrame) -> None:
        """Learn from ``train`` and forget whatever an earlier call had learned.

        Args:
            train: Rows of the gold feature table (``gold_features_schema``): keys,
                target, outlier flag and the model inputs. Only ``feature_columns`` may
                be used as inputs. The outlier flag is there so that a model can leave
                strange hours out of its training; it is calculated from the target and
                must never be an input.
        """
        ...

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        """Forecast the pickups of every row of ``rows``, in the order of the rows.

        Args:
            rows: The hour and the model inputs (``feature_columns``, which include the
                zone) of the rows to forecast. The target and the outlier flag are not
                in the table, so they cannot be read by mistake.
        """
        ...


class LagBaseline:
    """Forecast for every hour the pickups of the same zone ``lag_hours`` earlier."""

    def __init__(self, lag_hours: int) -> None:
        self.column = lag_column(lag_hours)

    def fit(self, train: pd.DataFrame) -> None:
        """Nothing to learn: the forecast is a copy of a feature."""

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        """The lag column of ``rows``, unchanged."""
        forecast: FloatArray = rows[self.column].to_numpy(dtype="float64")
        return forecast


def baselines(settings: Settings) -> dict[str, Forecaster]:
    """The baseline forecasters by name, the main one first.

    The baselines copy lag features, so they exist only for lags the feature table has.
    The length of a week is a fact and is written here; whether the table has that lag
    is a matter of the config and is checked, not assumed.

    Raises:
        PipelineError: If 168 is not in ``features.lags``. Without the weekly baseline
            there is no number to beat, so the backtest must not run.
    """
    lags = settings.features.lags
    if HOURS_PER_WEEK not in lags:
        raise PipelineError(
            f"The weekly baseline needs the feature {lag_column(HOURS_PER_WEEK)}, but "
            f"features.lags is {lags}. Add {HOURS_PER_WEEK} to features.lags in the config "
            "and run `dvc repro` to rebuild the feature table."
        )
    forecasters: dict[str, Forecaster] = {WEEKLY_NAIVE: LagBaseline(HOURS_PER_WEEK)}
    # "Yesterday" exists only while the horizon allows a lag of 24 hours. With a longer
    # horizon the config refuses that lag, and this reference is left out.
    if HOURS_PER_DAY in lags:
        forecasters[DAILY_NAIVE] = LagBaseline(HOURS_PER_DAY)
    return forecasters
