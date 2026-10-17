"""One day of forecasts from a model and a data snapshot: the code the API and the replay share.

The API (``app.py``) answers a request with it, and the batch forecast of the replay
(``demand.flows.forecast_flow``) writes its answer to disk. Both call ``predict_day``, so
a forecast written by the replay is the forecast the API gives for the same day, zones,
model and data: the rows come from ``day_features`` (``snapshot.py``), which passes the
feature function only the demand that had ended by the issue moment, and the model is
given exactly ``feature_columns(settings)`` of them.

The result keeps the rows the model was given. The API needs only the forecasts; the
replay also stores the rows, because the monitoring compares the inputs of production
with those of training, and because the weekly baseline (``lag_168h``) is one of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from demand.config import Settings
from demand.data.features import FloatArray
from demand.schemas import feature_columns
from demand.serving.model_loader import Predictor
from demand.serving.snapshot import DataSnapshot, day_features


@dataclass(frozen=True)
class DayForecast:
    """The forecasts of one day, the moment they were issued and the rows behind them."""

    issued: pd.Timestamp
    """The issue moment of the day (``issue_moment``), in the project timezone."""
    rows: pd.DataFrame
    """Zone, hour and the model inputs, as ``day_features`` built them: by zone, then hour."""
    values: FloatArray
    """One forecast per row of ``rows``, in the same order."""


def predict_day(
    predictor: Predictor,
    snapshot: DataSnapshot,
    zones: Sequence[int],
    day: date,
    settings: Settings,
) -> DayForecast:
    """The forecasts of every hour of ``day`` for ``zones``.

    Raises:
        RequestRefusedError: If a zone or the day cannot be forecast (``check_request``).
    """
    issued, rows = day_features(snapshot, zones, day, settings)
    values = np.asarray(predictor.predict(rows[feature_columns(settings)]), dtype="float64")
    return DayForecast(issued=issued, rows=rows, values=values)
