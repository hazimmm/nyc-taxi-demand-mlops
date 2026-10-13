"""The model: gradient-boosted trees (LightGBM) behind the ``Forecaster`` interface.

``LightGbmForecaster`` has the two methods of ``Forecaster`` (``baseline.py``), so the
backtest scores it exactly as it scores the weekly baseline, and nothing in the
backtest had to change for it.

**What the model is.** Several hundred decision trees (600 in the config). Each tree asks questions
about the inputs of a row ("is ``lag_168h`` above 80?", "is the hour after 17?") and
ends in a number; every new tree is built to correct the error the trees before it
still make. The parameters (how many trees, how large, how fast they learn) are
``model.params`` in the config and are handed to LightGBM unchanged. This feature does
not search for better ones.

**Its inputs.** ``feature_columns``, in that order, and nothing else. ``fit`` receives
whole rows of the feature table and reads the target from them. The outlier flag is in
those rows too and is never an input: it is calculated from the pickups, so a model
that read it would be told part of the answer.

**Four choices, and the numbers behind them.** Each was measured on the four test weeks
of the backtest (3 to 30 April 2025), three times with the seeds 42, 43 and 44. Training
picks rows and columns at random for every tree (``subsample``, ``colsample_bytree``),
so the score moves a little with the seed, and a difference smaller than that movement
is noise. With the settings chosen here the MAE is 25.04 on average and moves by 0.20
between the three seeds. The weekly baseline has 27.70.

* *The zone is a category, not a number.* ``zone_id`` is a label: zone 162 is not
  "more" than zone 161. Read as a number, a tree can only ask "is the zone below 150?";
  told that it is a category, it can put any set of zones on one side. MAE 25.04
  against 25.96 as a number, lower in each of the four weeks (by 0.6 to 1.3). The
  column stays a whole number; LightGBM is only told how to read it.
* *Outlier hours stay in the training rows.* Leaving out the flagged hours (56 of the
  53,280 rows up to 30 April) gives 25.00 against 25.04, inside the noise, and better in
  two weeks and worse in two. They really happened, so the simpler choice is to keep
  them.
* *The objective stays ``poisson``,* as the config has it. It is made for counts: the
  model works with the logarithm of the pickups, so a forecast can never be negative
  and an error of 5 counts more at night than in the rush hour. Against plain
  regression (squared error) the MAE is the same within the noise (25.04 and 25.17),
  the sMAPE is clearly better (22.8 against 25.0 percent) and the RMSE a little worse
  (39.6 against 39.2). Plain regression also forecast negative pickups, which had to be
  cut off at 0.
* *Forecasts are not rounded.* The pickups are whole numbers, but a forecast is an
  expected value: "2.4 pickups" means 2 in some such hours and 3 in others. Rounding
  changes no metric by more than 0.03, far inside the noise, and throws that away: ten
  hours of 0.4 would add up to 0 and not to 4.

A fifth idea, the pickups of the same hour two weeks earlier as one more input
(``lag_336h``), is a matter of the config (``features.lags``) and not of this class.

**What the forecaster refuses.** A model that is asked something it was not trained for
answers anyway, and the answer looks like any other number. So ``predict`` stops, with
a message, when a column is missing, has another type than in training, or holds a
missing or infinite value, and when a row belongs to a zone that was not in the training
rows (for LightGBM an unknown category is simply "none of the zones I know"). Columns
are picked by name, so their order in the table does not matter and a column too many
is not read. Forecasts are never negative: the Poisson objective cannot produce one,
and for any other objective they are cut off at 0.

**Same input, same model.** Fitting twice on the same rows with the same config and seed
gives the same trees and the same forecasts, bit for bit, on this machine: the random
choices are drawn from ``project.seed``, and ``deterministic`` tells LightGBM to prefer
repeatable arithmetic over speed. This was also measured once by hand with 1, 4 and 12
threads, and the forecasts were identical; a test repeats it with 1 and 3. It is what
keeps the backtest report byte-identical from run to run. It is *not* promised across
versions of LightGBM, or across processors and operating systems: sums of decimal
numbers are added in another order there, the last digits differ, and now and then a
tree splits elsewhere. The scores then move in the same range as they do with another
seed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import lightgbm
import numpy as np
import pandas as pd

from demand.config import Settings
from demand.data.features import FloatArray
from demand.errors import PipelineError
from demand.schemas import TARGET, ZONE_ID, feature_columns

LIGHTGBM = "lightgbm"
"""The name of the model in the backtest report."""

SEED_PARAMS = ("random_state", "seed")
"""Names LightGBM accepts for the seed. The forecaster sets it from ``project.seed``."""

REPEATABLE = {"deterministic": True, "force_row_wise": True}
"""LightGBM's switches for results that do not depend on timing or on the thread count.

``force_row_wise`` belongs to ``deterministic``: without it LightGBM times two ways of
building its tables at the start of every fit and takes the faster one.
"""


class ModelInputError(PipelineError, ValueError):
    """The rows given to the model are not what it was trained on, or it is not fitted."""


class LightGbmForecaster:
    """A LightGBM model that forecasts pickups from the feature columns."""

    def __init__(self, columns: Sequence[str], params: Mapping[str, Any], seed: int) -> None:
        """Set up a forecaster that has not learned anything yet.

        Args:
            columns: The model inputs, in the order the model is given them. The zone
                must be among them.
            params: Parameters for LightGBM, as in ``model.params`` of the config.
            seed: Seed for the random choices of rows and columns during training.

        Raises:
            PipelineError: If ``params`` sets a seed, or ``columns`` lacks the zone.
        """
        seeds = [name for name in SEED_PARAMS if name in params]
        if seeds:
            raise PipelineError(
                f"model.params sets {', '.join(seeds)}, but the seed of the model is "
                f"project.seed ({seed}). Remove {', '.join(seeds)} from model.params in "
                "the config."
            )
        if ZONE_ID not in columns:
            raise PipelineError(
                f"The model inputs {list(columns)} lack {ZONE_ID}. The forecaster treats "
                "the zone as a category and checks it; pass feature_columns(settings)."
            )
        self.columns = list(columns)
        self.params = {**REPEATABLE, **params, "random_state": seed}
        self.model: lightgbm.LGBMRegressor | None = None
        self._dtypes: pd.Series | None = None
        self._zones: frozenset[int] = frozenset()

    def fit(self, train: pd.DataFrame) -> None:
        """Train a new model on ``train``; whatever an earlier call learned is dropped.

        Every row is used, the flagged outlier hours too (see the module docstring).

        Raises:
            ModelInputError: If ``train`` lacks the target or an input column, or an
                input holds a missing or infinite value.
        """
        if TARGET not in train.columns:
            raise ModelInputError(
                f"The training rows have no column {TARGET}. Pass rows of the gold "
                "feature table (demand.models.backtest.load_training_features)."
            )
        inputs = self._checked_inputs(train)
        model = lightgbm.LGBMRegressor(**self.params)
        model.fit(inputs, train[TARGET].to_numpy(dtype="float64"), categorical_feature=[ZONE_ID])
        self.model = model
        self._dtypes = inputs.dtypes
        self._zones = frozenset(inputs[ZONE_ID].unique().tolist())

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        """Forecast the pickups of every row of ``rows``: never negative, never rounded.

        Only the input columns are read, by name. A target or an outlier flag in
        ``rows`` is ignored.

        Raises:
            ModelInputError: If the forecaster is not fitted; if ``rows`` lacks an input
                column, or one has another type than in training or holds a missing or
                infinite value; or if a row belongs to a zone the model was not
                trained on.
        """
        if self.model is None or self._dtypes is None:
            raise ModelInputError(
                "The LightGBM forecaster has not been fitted. Call fit with rows of the "
                "gold feature table before predict."
            )
        inputs = self._checked_inputs(rows)
        different = [
            f"{column} is {inputs[column].dtype}, was {self._dtypes[column]}"
            for column in self.columns
            if inputs[column].dtype != self._dtypes[column]
        ]
        if different:
            raise ModelInputError(
                f"Model inputs with another type than in training: {'; '.join(different)}. "
                "Build the rows with demand.data.features.build_features, which is what "
                "the training rows were built with."
            )
        unknown = sorted(set(inputs[ZONE_ID].unique().tolist()) - self._zones)
        if unknown:
            raise ModelInputError(
                f"The model was not trained on zone {', '.join(map(str, unknown))}; it "
                f"knows {sorted(self._zones)}. Forecast only zones of data.zones, or add "
                "the zone to the config, rebuild with `dvc repro` and train again."
            )
        forecast: FloatArray = np.clip(np.asarray(self.model.predict(inputs), "float64"), 0, None)
        return forecast

    def _checked_inputs(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The input columns of ``rows`` in the order of the model, complete and finite."""
        missing = [column for column in self.columns if column not in rows.columns]
        if missing:
            raise ModelInputError(
                f"The rows have no column {', '.join(missing)}. The model needs "
                f"{self.columns}: build the rows with demand.data.features.build_features "
                "and the config the model was trained with."
            )
        inputs = rows[self.columns]
        finite = np.isfinite(inputs.to_numpy(dtype="float64")).all(axis=0)
        if not finite.all():
            holes = [
                column
                for column, complete in zip(self.columns, finite, strict=True)
                if not complete
            ]
            raise ModelInputError(
                f"Missing or infinite values in the model inputs {', '.join(holes)}. A lag "
                "or a rolling feature is empty when the demand history does not reach far "
                "enough back; such a row cannot be forecast."
            )
        return inputs


def lightgbm_forecaster(settings: Settings) -> LightGbmForecaster:
    """The forecaster the config describes: its inputs, its parameters and its seed.

    Raises:
        PipelineError: If ``model.params`` sets a seed.
    """
    return LightGbmForecaster(
        feature_columns(settings), settings.model.params, settings.project.seed
    )
