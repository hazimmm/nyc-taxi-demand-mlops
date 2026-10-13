from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from demand.config import Settings
from demand.errors import PipelineError
from demand.models.baseline import (
    DAILY_NAIVE,
    HOURS_PER_DAY,
    HOURS_PER_WEEK,
    WEEKLY_NAIVE,
    LagBaseline,
    baselines,
)
from demand.schemas import IS_OUTLIER, TARGET, feature_columns, lag_column
from tests.conftest import with_features
from tests.unit.models.conftest import feature_table


def test_a_week_has_168_hours() -> None:
    assert HOURS_PER_DAY == 24
    assert HOURS_PER_WEEK == 168


def test_lag_baseline_forecasts_the_lag_column_unchanged() -> None:
    rows = pd.DataFrame({lag_column(168): [5.0, 0.0, 12.0], lag_column(24): [1.0, 2.0, 3.0]})

    assert LagBaseline(168).predict(rows).tolist() == [5.0, 0.0, 12.0]
    assert LagBaseline(24).predict(rows).tolist() == [1.0, 2.0, 3.0]


def test_lag_baseline_does_not_need_the_target(settings: Settings) -> None:
    table = feature_table(settings, "2025-04-01 00:00", hours=3)
    inputs = table.drop(columns=[TARGET, IS_OUTLIER])

    forecast = LagBaseline(168).predict(inputs)

    assert forecast.tolist() == (table[TARGET] - 168).tolist()


def test_fitting_a_baseline_changes_nothing(settings: Settings) -> None:
    table = feature_table(settings, "2025-04-01 00:00", hours=3)
    baseline = LagBaseline(168)
    before = baseline.predict(table).tolist()

    baseline.fit(table.assign(**{lag_column(168): 0.0}))

    assert baseline.predict(table).tolist() == before


def test_baselines_are_the_weekly_one_first_and_then_the_daily_one(settings: Settings) -> None:
    # The daily baseline copies ``lag_24h``, which only a horizon of 24 hours or less allows.
    day_ahead = with_features(settings, horizon_hours=24, lags=[24, 168])

    forecasters = baselines(day_ahead)

    assert list(forecasters) == [WEEKLY_NAIVE, DAILY_NAIVE]
    rows = pd.DataFrame({lag_column(168): [7.0], lag_column(24): [3.0]})
    assert forecasters[WEEKLY_NAIVE].predict(rows).tolist() == [7.0]
    assert forecasters[DAILY_NAIVE].predict(rows).tolist() == [3.0]


def test_baselines_copy_columns_that_are_model_inputs(settings: Settings) -> None:
    day_ahead = with_features(settings, horizon_hours=24, lags=[24, 168])
    two_days_ahead = with_features(settings, horizon_hours=48, lags=[48, 168])

    assert list(baselines(day_ahead)) == [WEEKLY_NAIVE, DAILY_NAIVE]
    assert lag_column(HOURS_PER_WEEK) in feature_columns(day_ahead)
    assert lag_column(HOURS_PER_DAY) in feature_columns(day_ahead)
    assert list(baselines(two_days_ahead)) == [WEEKLY_NAIVE]
    assert lag_column(HOURS_PER_WEEK) in feature_columns(two_days_ahead)


def test_default_config_has_the_weekly_baseline_and_it_comes_first(settings: Settings) -> None:
    # Whatever horizon the config file sets, there must be a number to beat.
    assert next(iter(baselines(settings))) == WEEKLY_NAIVE


def test_missing_weekly_lag_stops_with_advice(settings: Settings) -> None:
    without_week = with_features(settings, horizon_hours=24, lags=[24, 48])

    with pytest.raises(PipelineError, match=r"lag_168h.*\[24, 48\].*dvc repro"):
        baselines(without_week)


def test_daily_baseline_is_left_out_when_the_horizon_does_not_allow_its_lag(
    settings: Settings,
) -> None:
    two_days_ahead = with_features(settings, horizon_hours=48, lags=[48, 168])

    assert list(baselines(two_days_ahead)) == [WEEKLY_NAIVE]


def test_settings_with_a_lag_of_24_hours_and_a_horizon_of_48_cannot_be_built(
    settings: Settings,
) -> None:
    # So no test and no run can ask for the daily baseline two days ahead.
    with pytest.raises(ValidationError, match=r"\[24\] are shorter than the 48 h"):
        with_features(settings, horizon_hours=48, lags=[24, 168])
