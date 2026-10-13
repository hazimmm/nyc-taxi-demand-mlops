from __future__ import annotations

import itertools
import json
import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.aggregate import training_period_end
from demand.data.features import FloatArray
from demand.data.gold import gold_features_path
from demand.data.silver import write_parquet
from demand.errors import PipelineError
from demand.models.backtest import (
    BacktestReport,
    MissingGoldFileError,
    TooLittleHistoryError,
    backtest,
    backtest_from_gold,
    backtest_report_path,
    fold_windows,
    load_training_features,
)
from demand.models.baseline import DAILY_NAIVE, WEEKLY_NAIVE, baselines
from demand.schemas import HOUR_START, IS_OUTLIER, TARGET, ZONE_ID, feature_columns, lag_column
from tests.conftest import TEST_ZONES, with_features
from tests.unit.models.conftest import (
    Lookup,
    Spy,
    feature_table,
    training_rows,
    with_backtest,
)

TIMEZONE = "America/New_York"
# The small backtest: two folds of two days at the end of April 2025. The table starts
# three days before the first test day and runs two days into May, the replayed period.
FIRST_HOUR = "2025-04-24 00:00"
TABLE_HOURS = 9 * 24
ZONES = len(TEST_ZONES)
TEST_HOURS_PER_FOLD = 2 * 24
TEST_ROWS = 2 * TEST_HOURS_PER_FOLD * ZONES


def _local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TIMEZONE)


@pytest.fixture
def small(settings: Settings) -> Settings:
    """Two folds of two days: 27-28 and 29-30 April 2025.

    The horizon and the lags are written here, not taken from the config file, because
    the numbers in these tests follow from them: a horizon of 48 hours and the lags 48
    and 168, so the weekly baseline is the only one.
    """
    layout = with_backtest(settings, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)
    return with_features(layout, horizon_hours=48, lags=[48, 168])


def _with_horizon(settings: Settings, horizon: int) -> Settings:
    """The same settings for another horizon, with the shortest lag it allows and the week."""
    return with_features(settings, horizon_hours=horizon, lags=[horizon, 168])


@pytest.fixture
def day_ahead(small: Settings) -> Settings:
    """The small backtest with a horizon of 24 hours, which allows the daily baseline."""
    return _with_horizon(small, 24)


@pytest.fixture
def day_ahead_gold_table(day_ahead: Settings) -> pd.DataFrame:
    return feature_table(day_ahead, FIRST_HOUR, TABLE_HOURS)


@pytest.fixture
def gold_table(small: Settings) -> pd.DataFrame:
    """The feature table as gold stores it: it goes on into the replayed period."""
    return feature_table(small, FIRST_HOUR, TABLE_HOURS)


@pytest.fixture
def table(small: Settings, gold_table: pd.DataFrame) -> pd.DataFrame:
    """The rows of the training period, which is what the backtest may be given."""
    return training_rows(gold_table, small)


def _fold_of(table: pd.DataFrame) -> pd.Series:
    """The fold of the small backtest that each test row belongs to; 0 for other rows."""
    hours = table[HOUR_START]
    fold = pd.Series(0, index=table.index)
    fold[(hours >= _local("2025-04-27 00:00")) & (hours < _local("2025-04-29 00:00"))] = 1
    fold[hours >= _local("2025-04-29 00:00")] = 2
    return fold


def _write_gold(settings: Settings, table: pd.DataFrame) -> Path:
    path = gold_features_path(settings.paths.gold)
    write_parquet(table, path)
    return path


# --- fold boundaries --------------------------------------------------------------------


def test_folds_are_the_last_four_weeks_of_the_training_period(settings: Settings) -> None:
    folds = fold_windows(settings)

    assert [fold.number for fold in folds] == [1, 2, 3, 4]
    assert [fold.first_test_hour for fold in folds] == [
        _local("2025-04-03 00:00"),
        _local("2025-04-10 00:00"),
        _local("2025-04-17 00:00"),
        _local("2025-04-24 00:00"),
    ]
    assert [fold.last_test_hour for fold in folds] == [
        _local("2025-04-09 23:00"),
        _local("2025-04-16 23:00"),
        _local("2025-04-23 23:00"),
        _local("2025-04-30 23:00"),
    ]


@pytest.mark.parametrize(
    ("horizon", "last_train_days"),
    [
        # The test weeks start on Thursday 3, 10, 17 and 24 April at 00:00.
        (24, ["2025-04-02", "2025-04-09", "2025-04-16", "2025-04-23"]),  # Wednesdays
        (48, ["2025-04-01", "2025-04-08", "2025-04-15", "2025-04-22"]),  # Tuesdays
    ],
)
def test_training_rows_end_one_horizon_before_the_test_week(
    settings: Settings, horizon: int, last_train_days: list[str]
) -> None:
    folds = fold_windows(_with_horizon(settings, horizon))

    assert [fold.last_train_hour for fold in folds] == [
        _local(f"{day} 00:00") for day in last_train_days
    ]
    for fold in folds:
        assert fold.first_test_hour - fold.last_train_hour == pd.Timedelta(hours=horizon)


def test_longer_horizon_moves_the_end_of_the_training_rows_further_back(
    settings: Settings,
) -> None:
    day_ahead_fold = fold_windows(_with_horizon(settings, 24))[0]
    two_days_ahead_fold = fold_windows(_with_horizon(settings, 48))[0]

    assert day_ahead_fold.first_test_hour == _local("2025-04-03 00:00")
    assert two_days_ahead_fold.first_test_hour == _local("2025-04-03 00:00")
    assert day_ahead_fold.last_train_hour == _local("2025-04-02 00:00")
    assert two_days_ahead_fold.last_train_hour == _local("2025-04-01 00:00")


@pytest.mark.parametrize(("horizon", "unused_hours"), [(1, 0), (24, 23), (48, 47)])
def test_hours_between_the_training_rows_and_the_test_week_are_one_fewer_than_the_horizon(
    settings: Settings, horizon: int, unused_hours: int
) -> None:
    for fold in fold_windows(_with_horizon(settings, horizon)):
        # The hours that start after the last training hour and before the first test hour.
        between = pd.date_range(
            fold.last_train_hour, fold.first_test_hour, freq="h", inclusive="neither"
        )
        assert len(between) == unused_hours


@pytest.mark.parametrize(("n_folds", "fold_days"), [(1, 1), (2, 2), (4, 7), (3, 10)])
def test_last_fold_ends_with_the_last_hour_of_the_training_period(
    settings: Settings, n_folds: int, fold_days: int
) -> None:
    layout = with_backtest(
        settings, train_end_date=date(2025, 4, 30), n_folds=n_folds, fold_days=fold_days
    )

    folds = fold_windows(layout)

    assert len(folds) == n_folds
    assert folds[-1].last_test_hour == _local("2025-04-30 23:00")
    assert all(fold.last_test_hour < training_period_end(layout) for fold in folds)


def test_test_weeks_follow_each_other_without_a_gap_or_an_overlap(settings: Settings) -> None:
    folds = fold_windows(settings)

    for earlier, later in itertools.pairwise(folds):
        assert later.first_test_hour - earlier.last_test_hour == pd.Timedelta(hours=1)


def test_week_in_which_the_clocks_go_forward_has_167_hours_and_ends_at_midnight(
    settings: Settings,
) -> None:
    # Two weeks up to Wednesday 12 March 2025; the clocks go forward on Sunday the 9th.
    march = with_backtest(settings, train_end_date=date(2025, 3, 12), n_folds=2, fold_days=7)

    normal_week, short_week = fold_windows(march)

    assert normal_week.first_test_hour == pd.Timestamp("2025-02-27 00:00:00-05:00")
    assert normal_week.last_test_hour == pd.Timestamp("2025-03-05 23:00:00-05:00")
    assert short_week.first_test_hour == pd.Timestamp("2025-03-06 00:00:00-05:00")
    assert short_week.last_test_hour == pd.Timestamp("2025-03-12 23:00:00-04:00")
    hours = [
        len(pd.date_range(fold.first_test_hour, fold.last_test_hour, freq="h"))
        for fold in (normal_week, short_week)
    ]
    assert hours == [168, 167]


def test_week_in_which_the_clocks_go_back_has_169_hours(settings: Settings) -> None:
    # One week up to Wednesday 5 November 2025; the clocks go back on Sunday the 2nd.
    november = with_backtest(settings, train_end_date=date(2025, 11, 5), n_folds=1, fold_days=7)

    (long_week,) = fold_windows(november)

    assert long_week.first_test_hour == pd.Timestamp("2025-10-30 00:00:00-04:00")
    assert long_week.last_test_hour == pd.Timestamp("2025-11-05 23:00:00-05:00")
    assert len(pd.date_range(long_week.first_test_hour, long_week.last_test_hour, freq="h")) == 169


@pytest.mark.parametrize(
    ("horizon", "last_train_hour"),
    [
        # 24 elapsed hours earlier is 23:00 on Saturday, not 00:00 on Sunday.
        (24, "2025-03-08 23:00:00-05:00"),
        # 48 elapsed hours earlier is 23:00 on Friday, not 00:00 on Saturday.
        (48, "2025-03-07 23:00:00-05:00"),
    ],
)
def test_gap_before_the_test_week_is_counted_in_real_hours_across_a_clock_change(
    settings: Settings, horizon: int, last_train_hour: str
) -> None:
    # The test week starts on Monday 10 March, the day after the 23-hour Sunday.
    march = with_backtest(settings, train_end_date=date(2025, 3, 16), n_folds=1, fold_days=7)

    (fold,) = fold_windows(_with_horizon(march, horizon))

    assert fold.first_test_hour == pd.Timestamp("2025-03-10 00:00:00-04:00")
    assert fold.last_train_hour == pd.Timestamp(last_train_hour)
    assert fold.first_test_hour - fold.last_train_hour == pd.Timedelta(hours=horizon)


# --- what a forecaster is given ---------------------------------------------------------


def test_training_rows_of_each_fold_are_strictly_before_its_test_rows(
    small: Settings, table: pd.DataFrame
) -> None:
    spy = Spy()

    backtest(table, {"spy": spy}, small)

    assert len(spy.fitted_on) == len(spy.asked_for) == 2
    for train, test in zip(spy.fitted_on, spy.asked_for, strict=True):
        assert train[HOUR_START].max() < test[HOUR_START].min()


@pytest.mark.parametrize(
    ("horizon", "last_train_hours"),
    [
        (24, ["2025-04-26 00:00", "2025-04-28 00:00"]),
        (48, ["2025-04-25 00:00", "2025-04-27 00:00"]),
    ],
)
def test_each_fold_trains_up_to_one_horizon_before_its_first_test_hour(
    small: Settings, horizon: int, last_train_hours: list[str]
) -> None:
    layout = _with_horizon(small, horizon)
    table = training_rows(feature_table(layout, FIRST_HOUR, TABLE_HOURS), layout)
    spy = Spy()

    backtest(table, {"spy": spy}, layout)

    assert [train[HOUR_START].max() for train in spy.fitted_on] == [
        _local(hour) for hour in last_train_hours
    ]
    assert [test[HOUR_START].min() for test in spy.asked_for] == [
        _local("2025-04-27 00:00"),
        _local("2025-04-29 00:00"),
    ]


@pytest.mark.parametrize(
    ("horizon", "train_hours"),
    [
        # 24 and 25 April and the first hour of the 26th; then two more days.
        (24, [49, 97]),
        # 24 April and the first hour of the 25th; then two more days.
        (48, [25, 73]),
    ],
)
def test_training_window_expands_from_the_first_row_of_the_table(
    small: Settings, horizon: int, train_hours: list[int]
) -> None:
    layout = _with_horizon(small, horizon)
    table = training_rows(feature_table(layout, FIRST_HOUR, TABLE_HOURS), layout)
    spy = Spy()

    backtest(table, {"spy": spy}, layout)

    assert [train[HOUR_START].min() for train in spy.fitted_on] == [_local(FIRST_HOUR)] * 2
    assert [len(train) for train in spy.fitted_on] == [hours * ZONES for hours in train_hours]


def test_each_fold_is_asked_for_every_zone_and_hour_of_its_test_days(
    small: Settings, table: pd.DataFrame
) -> None:
    spy = Spy()

    backtest(table, {"spy": spy}, small)

    for test, last_hour in zip(
        spy.asked_for, ["2025-04-28 23:00", "2025-04-30 23:00"], strict=True
    ):
        assert len(test) == TEST_HOURS_PER_FOLD * ZONES
        assert test[HOUR_START].max() == _local(last_hour)
        assert not test.duplicated(subset=[ZONE_ID, HOUR_START]).any()


def test_forecaster_is_fitted_again_before_every_fold(small: Settings, table: pd.DataFrame) -> None:
    spy = Spy()

    backtest(table, {"spy": spy}, small)

    assert spy.calls == ["fit", "predict", "fit", "predict"]


def test_predict_is_given_the_model_inputs_but_not_the_target_or_the_outlier_flag(
    small: Settings, table: pd.DataFrame
) -> None:
    spy = Spy()

    backtest(table, {"spy": spy}, small)

    for rows in spy.asked_for:
        assert list(rows.columns) == [HOUR_START, *feature_columns(small)]
        assert TARGET not in rows.columns
        assert IS_OUTLIER not in rows.columns


def test_fit_is_given_the_target(small: Settings, table: pd.DataFrame) -> None:
    spy = Spy()

    backtest(table, {"spy": spy}, small)

    assert all(TARGET in train.columns for train in spy.fitted_on)


def test_outlier_flag_is_not_a_model_input(settings: Settings) -> None:
    assert IS_OUTLIER not in feature_columns(settings)


# --- nothing after the training period --------------------------------------------------


def test_table_with_rows_after_the_training_period_is_refused(
    small: Settings, gold_table: pd.DataFrame
) -> None:
    spy = Spy()

    with pytest.raises(ValueError, match=r"144 rows after data.train_end_date \(2025-04-30\)"):
        backtest(gold_table, {"spy": spy}, small)

    assert spy.calls == []


def test_first_hour_after_the_training_period_is_already_refused(
    small: Settings, gold_table: pd.DataFrame
) -> None:
    one_hour_too_many = gold_table[gold_table[HOUR_START] <= _local("2025-05-01 00:00")]

    with pytest.raises(ValueError, match=r"given 3 rows after data\.train_end_date"):
        backtest(one_hour_too_many, {}, small)


def test_loading_leaves_out_every_row_after_the_training_period(
    small: Settings, gold_table: pd.DataFrame
) -> None:
    _write_gold(small, gold_table)

    loaded = load_training_features(small)

    assert gold_table[HOUR_START].max() == _local("2025-05-02 23:00")
    assert loaded[HOUR_START].max() == _local("2025-04-30 23:00")
    assert len(loaded) == 7 * 24 * ZONES


def test_no_forecaster_sees_a_row_after_the_training_period(
    small: Settings, gold_table: pd.DataFrame
) -> None:
    _write_gold(small, gold_table)
    spy = Spy()

    backtest_from_gold(small, {"spy": spy})

    seen = pd.concat([*spy.fitted_on, *spy.asked_for])
    assert len(spy.calls) == 4
    assert seen[HOUR_START].max() == _local("2025-04-30 23:00")


def test_rows_after_the_training_period_do_not_change_the_report(
    small: Settings, gold_table: pd.DataFrame, table: pd.DataFrame
) -> None:
    changed = gold_table.copy()
    in_replay = changed[HOUR_START] >= training_period_end(small)
    changed.loc[in_replay, [TARGET, lag_column(168)]] = 0
    _write_gold(small, changed)

    assert backtest_from_gold(small, baselines(small)) == backtest(table, baselines(small), small)


# --- scores -----------------------------------------------------------------------------


def test_perfect_forecaster_scores_zero_everywhere(small: Settings, table: pd.DataFrame) -> None:
    report = backtest(table, {"perfect": Lookup(table)}, small)

    scores = report.forecasters["perfect"]
    every_score = [scores.overall, *scores.per_fold.values(), *scores.per_zone.values()]
    assert [(s.mae, s.rmse, s.smape) for s in every_score] == [(0.0, 0.0, 0.0)] * 6
    assert scores.overall.rows == TEST_ROWS


def test_forecaster_that_is_always_5_too_high_scores_exactly_5(
    small: Settings, table: pd.DataFrame
) -> None:
    report = backtest(table, {"offset": Lookup(table, table[TARGET] + 5)}, small)

    scores = report.forecasters["offset"]
    every_score = [scores.overall, *scores.per_fold.values(), *scores.per_zone.values()]
    assert [(s.mae, s.rmse) for s in every_score] == [(5.0, 5.0)] * 6
    assert scores.overall.smape > 0


def test_baselines_are_wrong_by_exactly_their_lag_on_pickups_that_rise_by_one_an_hour(
    day_ahead: Settings, day_ahead_gold_table: pd.DataFrame
) -> None:
    table = training_rows(day_ahead_gold_table, day_ahead)

    report = backtest(table, baselines(day_ahead), day_ahead)

    assert list(report.forecasters) == [WEEKLY_NAIVE, DAILY_NAIVE]
    weekly, daily = report.forecasters[WEEKLY_NAIVE], report.forecasters[DAILY_NAIVE]
    assert (weekly.overall.mae, weekly.overall.rmse) == (168.0, 168.0)
    assert (daily.overall.mae, daily.overall.rmse) == (24.0, 24.0)
    assert [s.mae for s in weekly.per_fold.values()] == [168.0, 168.0]
    assert [s.mae for s in daily.per_zone.values()] == [24.0] * ZONES


def test_weekly_baseline_is_the_only_one_when_the_horizon_is_48_hours(
    small: Settings, table: pd.DataFrame
) -> None:
    report = backtest(table, baselines(small), small)

    assert list(report.forecasters) == [WEEKLY_NAIVE]
    weekly = report.forecasters[WEEKLY_NAIVE]
    assert (weekly.overall.mae, weekly.overall.rmse) == (168.0, 168.0)
    assert [s.mae for s in weekly.per_fold.values()] == [168.0, 168.0]
    assert [s.mae for s in weekly.per_zone.values()] == [168.0] * ZONES


def test_scores_are_given_per_fold(small: Settings, table: pd.DataFrame) -> None:
    # Right in the first fold, 10 too high in the second.
    forecast = table[TARGET] + 10 * (_fold_of(table) == 2)

    scores = backtest(table, {"f": Lookup(table, forecast)}, small).forecasters["f"]

    assert list(scores.per_fold) == [1, 2]
    assert [s.mae for s in scores.per_fold.values()] == [0.0, 10.0]


def test_overall_score_pools_the_rows_and_is_not_the_mean_of_the_folds(
    small: Settings, table: pd.DataFrame
) -> None:
    forecast = table[TARGET] + 10 * (_fold_of(table) == 2)

    scores = backtest(table, {"f": Lookup(table, forecast)}, small).forecasters["f"]

    assert scores.overall.mae == 5.0
    # sqrt((0 + 100) / 2), not (0 + 10) / 2.
    assert scores.overall.rmse == round(math.sqrt(50), 6) == 7.071068


def test_scores_are_given_per_zone_in_order_of_zone_id(
    small: Settings, table: pd.DataFrame
) -> None:
    too_high_by = table[ZONE_ID].map({132: 1, 161: 2, 237: 3})

    scores = backtest(table, {"f": Lookup(table, table[TARGET] + too_high_by)}, small).forecasters[
        "f"
    ]

    assert {zone: s.mae for zone, s in scores.per_zone.items()} == {132: 1.0, 161: 2.0, 237: 3.0}
    assert list(scores.per_zone) == [132, 161, 237]
    assert scores.overall.mae == 2.0


def test_scores_are_rounded_to_six_decimals(small: Settings, table: pd.DataFrame) -> None:
    # One row in three is 1 too high.
    forecast = table[TARGET] + (table[ZONE_ID] == 132)

    scores = backtest(table, {"f": Lookup(table, forecast)}, small).forecasters["f"]

    assert scores.overall.mae == 0.333333
    assert scores.overall.rmse == 0.57735


# --- row counts -------------------------------------------------------------------------


def test_row_counts_per_fold_and_per_zone_add_up_to_the_total(
    small: Settings, table: pd.DataFrame
) -> None:
    report = backtest(table, baselines(small), small)

    assert report.test_rows == TEST_ROWS
    assert sum(fold.test_rows for fold in report.folds) == TEST_ROWS
    for scores in report.forecasters.values():
        assert scores.overall.rows == TEST_ROWS
        assert [s.rows for s in scores.per_fold.values()] == [TEST_HOURS_PER_FOLD * ZONES] * 2
        assert [s.rows for s in scores.per_zone.values()] == [2 * TEST_HOURS_PER_FOLD] * ZONES


def test_report_states_the_boundaries_and_row_counts_of_every_fold(
    small: Settings, table: pd.DataFrame
) -> None:
    report = backtest(table, baselines(small), small)

    # ``small`` has a horizon of 48 hours: the training rows of a fold end at 00:00 two
    # days before its first test day, after 25 and 73 hours of the table.
    assert report.train_end_date == date(2025, 4, 30)
    assert report.horizon_hours == 48
    assert [fold.model_dump(mode="json") for fold in report.folds] == [
        {
            "number": 1,
            "last_train_hour": "2025-04-25T00:00:00-04:00",
            "first_test_hour": "2025-04-27T00:00:00-04:00",
            "last_test_hour": "2025-04-28T23:00:00-04:00",
            "train_rows": 25 * ZONES,
            "test_rows": 48 * ZONES,
            "outlier_rows": 0,
        },
        {
            "number": 2,
            "last_train_hour": "2025-04-27T00:00:00-04:00",
            "first_test_hour": "2025-04-29T00:00:00-04:00",
            "last_test_hour": "2025-04-30T23:00:00-04:00",
            "train_rows": 73 * ZONES,
            "test_rows": 48 * ZONES,
            "outlier_rows": 0,
        },
    ]


def test_fold_with_a_clock_change_has_one_hour_less_per_zone(settings: Settings) -> None:
    march = with_backtest(settings, train_end_date=date(2025, 3, 12), n_folds=2, fold_days=7)
    table = training_rows(feature_table(march, "2025-02-24 00:00", hours=18 * 24), march)

    report = backtest(table, baselines(march), march)

    assert [fold.test_rows for fold in report.folds] == [168 * ZONES, 167 * ZONES]
    assert report.test_rows == 335 * ZONES


def test_report_states_the_horizon_the_folds_were_cut_with(
    day_ahead: Settings, day_ahead_gold_table: pd.DataFrame
) -> None:
    table = training_rows(day_ahead_gold_table, day_ahead)

    report = backtest(table, baselines(day_ahead), day_ahead)

    assert report.horizon_hours == 24
    assert [fold.last_train_hour for fold in report.folds] == [
        _local("2025-04-26 00:00"),
        _local("2025-04-28 00:00"),
    ]
    assert [fold.train_rows for fold in report.folds] == [49 * ZONES, 97 * ZONES]


# --- outliers ---------------------------------------------------------------------------


def _with_outliers_that_are_100_too_high(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Flag the first three hours of the first test day in one zone; forecast them badly."""
    flagged = (
        (table[ZONE_ID] == 132)
        & (table[HOUR_START] >= _local("2025-04-27 00:00"))
        & (table[HOUR_START] <= _local("2025-04-27 02:00"))
    )
    return table.assign(**{IS_OUTLIER: flagged}), table[TARGET] + 100 * flagged


def test_outlier_hours_are_scored(small: Settings, table: pd.DataFrame) -> None:
    flagged, forecast = _with_outliers_that_are_100_too_high(table)

    report = backtest(flagged, {"f": Lookup(flagged, forecast)}, small)

    overall = report.forecasters["f"].overall
    assert overall.rows == TEST_ROWS
    assert overall.mae == round(3 * 100 / TEST_ROWS, 6)


def test_second_score_leaves_the_outlier_hours_out(small: Settings, table: pd.DataFrame) -> None:
    flagged, forecast = _with_outliers_that_are_100_too_high(table)

    report = backtest(flagged, {"f": Lookup(flagged, forecast)}, small)

    without = report.forecasters["f"].overall_without_outliers
    assert without is not None
    assert without.rows == TEST_ROWS - 3
    assert without.mae == 0.0


def test_outlier_rows_are_counted_per_fold_and_in_total(
    small: Settings, table: pd.DataFrame
) -> None:
    flagged, _ = _with_outliers_that_are_100_too_high(table)

    report = backtest(flagged, baselines(small), small)

    assert [fold.outlier_rows for fold in report.folds] == [3, 0]
    assert report.outlier_rows == 3


def test_outliers_outside_the_test_weeks_are_not_counted(
    small: Settings, table: pd.DataFrame
) -> None:
    flagged = table.assign(**{IS_OUTLIER: table[HOUR_START] < _local("2025-04-27 00:00")})

    report = backtest(flagged, baselines(small), small)

    assert report.outlier_rows == 0


def test_there_is_no_second_score_when_every_test_hour_is_an_outlier(
    small: Settings, table: pd.DataFrame
) -> None:
    all_flagged = table.assign(**{IS_OUTLIER: True})

    report = backtest(all_flagged, baselines(small), small)

    assert report.forecasters[WEEKLY_NAIVE].overall_without_outliers is None
    assert report.forecasters[WEEKLY_NAIVE].overall.rows == TEST_ROWS


# --- error paths ------------------------------------------------------------------------


def test_table_that_starts_with_the_first_test_week_is_too_short(
    small: Settings, table: pd.DataFrame
) -> None:
    no_history = table[table[HOUR_START] >= _local("2025-04-27 00:00")]

    with pytest.raises(TooLittleHistoryError, match=r"Fold 1 .* 0 training rows.*n_folds"):
        backtest(no_history, baselines(small), small)


def test_table_that_ends_before_a_test_week_is_refused(
    small: Settings, table: pd.DataFrame
) -> None:
    ends_early = table[table[HOUR_START] < _local("2025-04-29 00:00")]

    with pytest.raises(TooLittleHistoryError, match=r"Fold 2 .* 0 test rows"):
        backtest(ends_early, baselines(small), small)


def test_too_little_history_is_reported_in_one_line_by_the_command_line() -> None:
    assert issubclass(TooLittleHistoryError, PipelineError)
    assert issubclass(MissingGoldFileError, PipelineError)


class _OneForecastTooFew(Spy):
    def predict(self, rows: pd.DataFrame) -> FloatArray:
        return super().predict(rows)[1:]


class _OneMissingForecast(Spy):
    def predict(self, rows: pd.DataFrame) -> FloatArray:
        forecast = super().predict(rows)
        forecast[0] = np.nan
        return forecast


def test_forecaster_that_returns_too_few_forecasts_is_named(
    small: Settings, table: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match=r"'short' returned an array of shape \(143,\) for 144"):
        backtest(table, {"short": _OneForecastTooFew()}, small)


def test_forecaster_that_returns_a_missing_forecast_is_not_scored(
    small: Settings, table: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="missing or infinite"):
        backtest(table, {"gaps": _OneMissingForecast()}, small)


def test_missing_gold_file_says_to_run_dvc_repro(small: Settings) -> None:
    with pytest.raises(MissingGoldFileError, match=r"gold/features.parquet is missing.*dvc repro"):
        backtest_from_gold(small, baselines(small))

    assert not small.paths.reports_dir.exists()


def test_gold_file_built_with_other_lags_says_to_run_dvc_repro(
    small: Settings, gold_table: pd.DataFrame
) -> None:
    _write_gold(small, gold_table.drop(columns=[lag_column(168), lag_column(48)]))

    with pytest.raises(MissingGoldFileError, match=r"no column lag_48h, lag_168h.*dvc repro"):
        backtest_from_gold(small, baselines(small))

    assert not small.paths.reports_dir.exists()


# --- the report on disk -----------------------------------------------------------------


def test_report_path_is_in_the_reports_folder(tmp_path: Path) -> None:
    assert backtest_report_path(tmp_path) == tmp_path / "backtest_report.json"


def test_report_is_written_and_can_be_read_back(small: Settings, gold_table: pd.DataFrame) -> None:
    _write_gold(small, gold_table)

    report = backtest_from_gold(small, baselines(small))

    stored = backtest_report_path(small.paths.reports_dir).read_bytes()
    assert BacktestReport.model_validate_json(stored) == report
    assert stored.endswith(b"\n")


def test_report_has_no_time_stamp_and_no_run_time(
    small: Settings, gold_table: pd.DataFrame
) -> None:
    _write_gold(small, gold_table)

    backtest_from_gold(small, baselines(small))

    stored = json.loads(backtest_report_path(small.paths.reports_dir).read_bytes())
    assert list(stored) == [
        "train_end_date",
        "horizon_hours",
        "folds",
        "test_rows",
        "outlier_rows",
        "forecasters",
    ]
    assert list(stored["forecasters"][WEEKLY_NAIVE]) == [
        "overall",
        "overall_without_outliers",
        "per_fold",
        "per_zone",
    ]
    assert stored["forecasters"][WEEKLY_NAIVE]["overall"] == {
        "mae": 168.0,
        "rmse": 168.0,
        "smape": stored["forecasters"][WEEKLY_NAIVE]["overall"]["smape"],
        "rows": TEST_ROWS,
    }


def test_second_run_gives_an_identical_report(small: Settings, gold_table: pd.DataFrame) -> None:
    _write_gold(small, gold_table)
    report_path = backtest_report_path(small.paths.reports_dir)
    backtest_from_gold(small, baselines(small))
    first_run = report_path.read_bytes()

    backtest_from_gold(small, baselines(small))

    assert report_path.read_bytes() == first_run


def test_order_of_the_rows_in_the_file_does_not_change_the_report(
    small: Settings, gold_table: pd.DataFrame, table: pd.DataFrame
) -> None:
    shuffled = table.sample(frac=1, random_state=0)

    assert backtest(shuffled, baselines(small), small) == backtest(table, baselines(small), small)


def test_backtest_does_not_change_the_gold_file(small: Settings, gold_table: pd.DataFrame) -> None:
    path = _write_gold(small, gold_table)
    before = path.read_bytes()

    backtest_from_gold(small, baselines(small))

    assert path.read_bytes() == before
    assert [file.name for file in small.paths.gold.iterdir()] == ["features.parquet"]


def test_one_summary_line_is_logged_for_every_forecaster(
    day_ahead: Settings, day_ahead_gold_table: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    # The day-ahead settings have two baselines, so "every" means more than one line.
    caplog.set_level(logging.INFO)
    _write_gold(day_ahead, day_ahead_gold_table)

    backtest_from_gold(day_ahead, baselines(day_ahead))

    scored = [record for record in caplog.records if record.message == "forecaster scored"]
    assert [record.__dict__["forecaster"] for record in scored] == [WEEKLY_NAIVE, DAILY_NAIVE]
    weekly = scored[0].__dict__
    assert weekly["mae"] == 168.0
    assert weekly["rmse"] == 168.0
    assert weekly["rows"] == TEST_ROWS
    assert weekly["mae_without_outliers"] == 168.0
    assert weekly["mae_per_fold"] == {"1": 168.0, "2": 168.0}


def test_end_of_the_run_is_logged_with_the_report_path(
    day_ahead: Settings, day_ahead_gold_table: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    _write_gold(day_ahead, day_ahead_gold_table)

    backtest_from_gold(day_ahead, baselines(day_ahead))

    finished = next(record for record in caplog.records if record.message == "backtest finished")
    fields = finished.__dict__
    assert fields["report"] == str(backtest_report_path(day_ahead.paths.reports_dir))
    assert fields["folds"] == 2
    assert fields["test_rows"] == TEST_ROWS
    assert fields["outlier_rows"] == 0
    assert fields["forecasters"] == [WEEKLY_NAIVE, DAILY_NAIVE]


def test_summary_line_has_no_second_mae_when_every_hour_is_an_outlier(
    small: Settings, gold_table: pd.DataFrame, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    _write_gold(small, gold_table.assign(**{IS_OUTLIER: True}))

    backtest_from_gold(small, baselines(small))

    scored = next(record for record in caplog.records if record.message == "forecaster scored")
    assert scored.__dict__["mae_without_outliers"] is None
