"""How far a forecast is from what happened: MAE, RMSE and sMAPE.

Everything in this module is pure: two arrays of the same length go in, a number comes
out. ``actual`` holds the pickups that were counted, ``forecast`` what was predicted for
the same zone and hour. One position is one zone and hour.

No single number describes a forecast, so three are reported. They answer different
questions, and each has a weakness the others cover:

* **MAE**, the mean absolute error, is the headline number of this project. It is in
  pickups, so it can be read aloud: "on average the forecast is off by 20 pickups per
  zone and hour". Every pickup of error costs the same. Its weakness: it is dominated by
  the busy zones and the busy hours, where an error of 20 is small, and says little
  about a quiet zone, where 20 is a disaster.
* **RMSE**, the root mean squared error, is also in pickups, but it squares the errors
  first, so one error of 100 counts as much as a hundred errors of 10. It is never
  smaller than the MAE. The *gap* between the two is the useful part: a large gap means
  the error is concentrated in a few very bad hours (a holiday, a storm) and not spread
  evenly. Its weakness is the same property: a handful of hours can decide it.
* **sMAPE**, the symmetric mean absolute percentage error, is a percentage, so a quiet
  zone and a busy zone weigh the same and zones can be compared. It has the most
  weaknesses, listed at ``smape``; it is reported as a second opinion and should never
  be the number a decision rests on.

A metric here is a plain mean over the rows it is given. Which rows those are (one fold,
one zone, all of them) is decided by the caller, ``backtest.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

FloatArray = npt.NDArray[np.float64]

SMAPE_MAX = 200.0
"""The sMAPE of a single hour when one of the two values is zero and the other is not."""


class Metrics(BaseModel):
    """The three error measures of one set of rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mae: float
    """Mean absolute error, in pickups per zone and hour."""
    rmse: float
    """Root mean squared error, in pickups per zone and hour."""
    smape: float
    """Symmetric mean absolute percentage error, from 0 to 200 percent."""


def _as_checked_arrays(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> tuple[FloatArray, ...]:
    """Both inputs as float arrays, after refusing what no metric can be calculated from.

    Raises:
        ValueError: If the two do not have the same length, are empty, or contain a
            value that is missing or infinite. A missing forecast is never skipped: a
            forecaster that returns one has failed, and the score must not hide it.
    """
    actual_values = np.asarray(actual, dtype="float64")
    forecast_values = np.asarray(forecast, dtype="float64")
    if actual_values.ndim != 1 or actual_values.shape != forecast_values.shape:
        raise ValueError(
            f"actual and forecast must be two lists of the same length, got shapes "
            f"{actual_values.shape} and {forecast_values.shape}"
        )
    if actual_values.size == 0:
        raise ValueError("a metric needs at least one row")
    if not (np.isfinite(actual_values).all() and np.isfinite(forecast_values).all()):
        raise ValueError("actual and forecast must not contain missing or infinite values")
    return actual_values, forecast_values


def mae(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> float:
    """Mean absolute error: the average size of the error, in pickups.

    Good at: being understood. Bad at: comparing a quiet zone with a busy one.
    """
    actual_values, forecast_values = _as_checked_arrays(actual, forecast)
    return float(np.abs(forecast_values - actual_values).mean())


def rmse(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> float:
    """Root mean squared error: like the MAE, but large errors count much more.

    Good at: showing that a forecast is sometimes badly wrong. Bad at: describing the
    usual hour, because a few extreme hours can decide the result.
    """
    actual_values, forecast_values = _as_checked_arrays(actual, forecast)
    return float(np.sqrt(np.square(forecast_values - actual_values).mean()))


def smape(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> float:
    """Symmetric mean absolute percentage error, in percent from 0 to 200.

    For one hour it is ``|forecast - actual| / ((|actual| + |forecast|) / 2)``: the
    error as a share of the average of the two values. The result is the mean over all
    hours, times 100.

    **Both values zero.** The formula gives 0 / 0. Such an hour counts as an error of 0
    percent and stays in the mean: the forecast said "no pickups", there were none, and
    that is a perfect forecast. Leaving these hours out would remove exactly the hours a
    forecaster got right, and the number of hours in the mean would depend on the
    forecast.

    **Weaknesses**, all of which matter for counts that are sometimes small:

    * *Zero against anything is the maximum.* If there was no pickup and the forecast
      said 1, the hour scores 200 percent, the same as a forecast of 1,000. With 0.7
      percent of the hours at zero pickups, these hours add a block to the mean that has
      nothing to do with the size of the errors.
    * *Small counts are loud.* 2 forecast against 1 counted is 67 percent; 201 against
      200 is 0.5 percent. Both are wrong by one pickup. The mean is therefore driven by
      the quiet night hours, which matter least to anyone who plans drivers.
    * *It is not symmetric*, in spite of its name. For an actual of 100, a forecast 50
      too low scores 67 percent and a forecast 50 too high scores 40 percent. A
      forecaster that leans high looks better than one that leans low.
    * *It is a mean of ratios*, not a ratio of totals, so it cannot be turned back into
      pickups, and its value for all zones is not the average of its values per zone
      weighted by demand.
    """
    actual_values, forecast_values = _as_checked_arrays(actual, forecast)
    error = np.abs(forecast_values - actual_values)
    scale = (np.abs(actual_values) + np.abs(forecast_values)) / 2
    # Where the scale is 0 both values are 0, so the error is 0 too: a perfect hour.
    share = np.divide(error, scale, out=np.zeros_like(error), where=scale > 0)
    return float(share.mean() * 100)


def evaluate(actual: npt.ArrayLike, forecast: npt.ArrayLike) -> Metrics:
    """All three metrics of one set of rows.

    Raises:
        ValueError: If the inputs differ in length, are empty, or contain a missing or
            infinite value.
    """
    return Metrics(
        mae=mae(actual, forecast), rmse=rmse(actual, forecast), smape=smape(actual, forecast)
    )
