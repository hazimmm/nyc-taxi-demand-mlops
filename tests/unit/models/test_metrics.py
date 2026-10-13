from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from demand.models.metrics import SMAPE_MAX, Metrics, evaluate, mae, rmse, smape

# Four hours, checked by hand. Errors: 2, 4, 0 and 4 pickups.
ACTUAL = [10, 20, 30, 0]
FORECAST = [12, 16, 30, 4]


def test_mae_is_the_mean_of_the_absolute_errors() -> None:
    # (2 + 4 + 0 + 4) / 4
    assert mae(ACTUAL, FORECAST) == 2.5


def test_rmse_is_the_root_of_the_mean_squared_error() -> None:
    # sqrt((4 + 16 + 0 + 16) / 4) = sqrt(9)
    assert rmse(ACTUAL, FORECAST) == 3.0


def test_smape_is_the_mean_error_as_a_share_of_the_average_of_both_values() -> None:
    # 2 / 11, 4 / 18, 0 / 30 and 4 / 2, averaged, in percent.
    expected = (2 / 11 + 4 / 18 + 0 + 2) / 4 * 100

    assert smape(ACTUAL, FORECAST) == pytest.approx(expected, abs=1e-12)
    assert smape(ACTUAL, FORECAST) == pytest.approx(60.10101, abs=1e-5)


def test_perfect_forecast_scores_zero_on_every_metric() -> None:
    assert evaluate(ACTUAL, ACTUAL) == Metrics(mae=0.0, rmse=0.0, smape=0.0)


def test_error_does_not_depend_on_its_direction_for_mae_and_rmse() -> None:
    assert mae([10, 10], [7, 13]) == 3.0
    assert rmse([10, 10], [7, 13]) == 3.0


def test_rmse_counts_one_large_error_more_than_many_small_ones() -> None:
    one_large = [10, 0, 0, 0]
    spread_out = [2.5, 2.5, 2.5, 2.5]
    nothing = [0, 0, 0, 0]

    assert mae(nothing, one_large) == mae(nothing, spread_out) == 2.5
    assert rmse(nothing, spread_out) == 2.5
    assert rmse(nothing, one_large) == 5.0


def test_smape_of_an_hour_where_both_values_are_zero_is_zero() -> None:
    assert smape([0], [0]) == 0.0


def test_hour_with_two_zeros_stays_in_the_mean_of_smape() -> None:
    # 0 percent for the first hour, 2 / 3 for the second: the mean is over two hours.
    assert smape([0, 2], [0, 1]) == pytest.approx(100 / 3)


@pytest.mark.parametrize(("actual", "forecast"), [(0, 1), (0, 1000), (7, 0)])
def test_smape_is_at_its_maximum_when_exactly_one_value_is_zero(
    actual: float, forecast: float
) -> None:
    assert smape([actual], [forecast]) == SMAPE_MAX == 200.0


def test_smape_punishes_the_same_error_harder_on_a_small_count() -> None:
    assert smape([1], [2]) == pytest.approx(66.667, abs=1e-3)
    assert smape([200], [201]) == pytest.approx(0.499, abs=1e-3)


def test_smape_is_not_symmetric_between_too_low_and_too_high() -> None:
    too_low = smape([100], [50])
    too_high = smape([100], [150])

    assert too_low == pytest.approx(66.667, abs=1e-3)
    assert too_high == pytest.approx(40.0)


def test_evaluate_returns_the_three_metrics_together() -> None:
    metrics = evaluate(ACTUAL, FORECAST)

    assert metrics.mae == mae(ACTUAL, FORECAST)
    assert metrics.rmse == rmse(ACTUAL, FORECAST)
    assert metrics.smape == smape(ACTUAL, FORECAST)


def test_metrics_accept_arrays_and_series_as_well_as_lists() -> None:
    from_arrays = evaluate(np.array(ACTUAL), np.array(FORECAST, dtype="float64"))
    from_series = evaluate(pd.Series(ACTUAL, index=[5, 6, 7, 8]), pd.Series(FORECAST))

    assert from_arrays == from_series == evaluate(ACTUAL, FORECAST)


def test_metrics_cannot_be_changed_after_they_are_calculated() -> None:
    metrics = evaluate(ACTUAL, FORECAST)

    with pytest.raises(ValidationError, match="frozen"):
        metrics.mae = 0.0


@pytest.mark.parametrize("metric", [mae, rmse, smape, evaluate])
def test_different_lengths_are_refused(metric: object) -> None:
    assert callable(metric)
    with pytest.raises(ValueError, match="same length"):
        metric([1, 2, 3], [1, 2])


def test_tables_instead_of_lists_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        mae([[1, 2], [3, 4]], [[1, 2], [3, 4]])


def test_no_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        evaluate([], [])


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_missing_or_infinite_forecast_is_refused_and_not_skipped(bad: float) -> None:
    with pytest.raises(ValueError, match="missing or infinite"):
        mae([1, 2], [1, bad])


def test_missing_actual_is_refused() -> None:
    with pytest.raises(ValueError, match="missing or infinite"):
        smape([1, math.nan], [1, 2])
