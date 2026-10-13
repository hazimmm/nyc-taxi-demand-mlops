from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.features import (
    build_feature_table,
    build_features,
    calendar_features,
    holiday_flag,
    shift_later,
    warm_up_hours,
    weekend_flag,
    window_statistics,
)
from demand.schemas import (
    CALENDAR_FEATURES,
    DAY_OF_WEEK,
    HOUR,
    HOUR_START,
    IS_HOLIDAY,
    IS_OUTLIER,
    IS_WEEKEND,
    MONTH,
    TARGET,
    ZONE_ID,
    feature_columns,
    gold_features_schema,
    lag_column,
    rolling_mean_column,
    rolling_std_column,
)
from tests.conftest import TEST_ZONES
from tests.unit.data.conftest import demand_frame

TIMEZONE = "America/New_York"
HOURS_PER_WEEK = 168
# Three weeks from 1 March 2025: the clocks go forward in the second week (9 March).
FIRST_HOUR = "2025-03-01 00:00"
N_HOURS = 3 * HOURS_PER_WEEK


def _local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TIMEZONE)


def _hours(*texts: str) -> pd.Series:
    return pd.Series([_local(text) for text in texts])


def _small_settings(settings: Settings) -> Settings:
    """A horizon of 2 hours, lags of 2 and 4 hours and a 3-hour window: easy to check by hand."""
    features = settings.features.model_copy(
        update={"horizon_hours": 2, "lags": [2, 4], "rolling_windows": [3]}
    )
    return settings.model_copy(update={"features": features})


def _one_zone(settings: Settings, pickups: list[int]) -> pd.DataFrame:
    demand = demand_frame(settings, "2025-01-15 00:00", len(pickups), zones=[161])
    demand[TARGET] = pickups
    return demand


def _row(features: pd.DataFrame, zone: int, hour: pd.Timestamp) -> pd.Series:
    rows = features[(features[ZONE_ID] == zone) & (features[HOUR_START] == hour)]
    assert len(rows) == 1
    return rows.iloc[0]


def _changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> pd.Series:
    """True for every row in which at least one of ``columns`` differs."""
    old, new = before[columns], after[columns]
    same = (old == new) | (old.isna() & new.isna())
    changed: pd.Series = ~same.all(axis=1)
    return changed


@pytest.fixture
def demand(settings: Settings) -> pd.DataFrame:
    return demand_frame(settings, FIRST_HOUR, N_HOURS)


# --- shape of the result ----------------------------------------------------------------


def test_result_has_the_model_inputs_in_their_fixed_order(
    settings: Settings, demand: pd.DataFrame
) -> None:
    features = build_features(demand, demand, settings)

    inputs = feature_columns(settings)
    assert list(features.columns) == [ZONE_ID, HOUR_START, *inputs[1:]]
    assert list(features[inputs].columns) == inputs
    assert TARGET not in features.columns
    assert IS_OUTLIER not in features.columns


def test_the_target_and_the_outlier_flag_are_not_model_inputs(settings: Settings) -> None:
    assert TARGET not in feature_columns(settings)
    assert IS_OUTLIER not in feature_columns(settings)


def test_result_has_one_row_per_target_in_the_order_of_the_targets(
    settings: Settings, demand: pd.DataFrame
) -> None:
    shuffled = demand.sample(frac=1.0, random_state=1)

    in_order = build_features(demand, demand, settings)
    out_of_order = build_features(demand, shuffled, settings)

    assert list(out_of_order.index) == list(shuffled.index)
    pd.testing.assert_frame_equal(out_of_order.sort_index(), in_order)


def test_order_of_the_history_rows_does_not_matter(
    settings: Settings, demand: pd.DataFrame
) -> None:
    shuffled = demand.sample(frac=1.0, random_state=2)

    pd.testing.assert_frame_equal(
        build_features(shuffled, demand, settings), build_features(demand, demand, settings)
    )


# --- lags and rolling statistics, by hand -----------------------------------------------


def test_lags_and_window_statistics_match_hand_calculated_values(settings: Settings) -> None:
    small = _small_settings(settings)
    #  hour:               0  1  2  3   4   5   6   7
    demand = _one_zone(small, [1, 2, 4, 8, 16, 32, 64, 128])

    features = build_features(demand, demand, small)

    # Target hour 6: lag 2 is hour 4, lag 4 is hour 2. The window ends 2 hours back, at
    # hour 4, and holds hours 2, 3 and 4: the values 4, 8 and 16.
    hour_6 = features.iloc[6]
    assert hour_6[lag_column(2)] == 16
    assert hour_6[lag_column(4)] == 4
    assert hour_6[rolling_mean_column(3)] == pytest.approx(28 / 3)
    assert hour_6[rolling_std_column(3)] == pytest.approx(np.sqrt(112 / 3))  # sample std
    # Target hour 7: everything moves one hour on.
    hour_7 = features.iloc[7]
    assert hour_7[lag_column(2)] == 32
    assert hour_7[lag_column(4)] == 8
    assert hour_7[rolling_mean_column(3)] == pytest.approx(56 / 3)


def test_features_are_missing_where_the_history_does_not_reach(settings: Settings) -> None:
    small = _small_settings(settings)
    demand = _one_zone(small, [1, 2, 4, 8, 16, 32, 64, 128])

    features = build_features(demand, demand, small)

    assert features[lag_column(2)].isna().tolist() == [True] * 2 + [False] * 6
    assert features[lag_column(4)].isna().tolist() == [True] * 4 + [False] * 4
    # The first complete window holds hours 0 to 2 and ends 2 hours before hour 4.
    assert features[rolling_mean_column(3)].isna().tolist() == [True] * 4 + [False] * 4
    assert features[rolling_std_column(3)].isna().tolist() == [True] * 4 + [False] * 4
    assert features.iloc[4][rolling_mean_column(3)] == pytest.approx(7 / 3)


def test_hour_missing_from_the_history_is_not_read_as_zero(settings: Settings) -> None:
    small = _small_settings(settings)
    demand = _one_zone(small, [1, 2, 4, 8, 16, 32, 64, 128])
    history = demand.drop(index=3)  # hour 3 was never observed

    features = build_features(history, demand, small)

    assert np.isnan(features.iloc[5][lag_column(2)])
    assert np.isnan(features.iloc[7][lag_column(4)])
    # Hour 3 is in the windows of the target hours 5, 6 and 7.
    assert features[rolling_mean_column(3)].isna().tolist() == [True] * 4 + [False] + [True] * 3
    assert features.iloc[4][lag_column(2)] == 4


def test_constant_demand_has_a_standard_deviation_of_exactly_zero(settings: Settings) -> None:
    small = _small_settings(settings)
    demand = _one_zone(small, [7] * 8)

    features = build_features(demand, demand, small)

    assert features[rolling_std_column(3)].dropna().tolist() == [0.0] * 4
    assert features[rolling_mean_column(3)].dropna().tolist() == [7.0] * 4


def test_default_lags_read_the_pickups_that_many_hours_earlier(
    settings: Settings, demand: pd.DataFrame
) -> None:
    features = build_features(demand, demand, settings)
    zone = demand[demand[ZONE_ID] == 161].reset_index(drop=True)
    zone_features = features[features[ZONE_ID] == 161].reset_index(drop=True)

    for lag in settings.features.lags:
        expected = zone[TARGET].shift(lag)
        pd.testing.assert_series_equal(zone_features[lag_column(lag)], expected, check_names=False)


def test_default_windows_end_one_horizon_before_the_target_hour(
    settings: Settings, demand: pd.DataFrame
) -> None:
    horizon = settings.features.horizon_hours
    features = build_features(demand, demand, settings)
    zone = demand[demand[ZONE_ID] == 132].reset_index(drop=True)
    zone_features = features[features[ZONE_ID] == 132].reset_index(drop=True)
    target = 400

    for window in settings.features.rolling_windows:
        values = zone[TARGET].iloc[target - horizon - window + 1 : target - horizon + 1]
        assert len(values) == window
        row = zone_features.iloc[target]
        assert row[rolling_mean_column(window)] == pytest.approx(values.mean())
        assert row[rolling_std_column(window)] == pytest.approx(values.std(ddof=1))


def test_shift_later_moves_values_down_and_leaves_the_start_empty() -> None:
    values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    shifted = shift_later(values, 2)

    assert np.isnan(shifted[:2]).all()
    assert shifted[2].tolist() == [1.0, 10.0]
    assert np.isnan(shift_later(values, 3)).all()
    assert np.isnan(shift_later(values, 10)).all()


def test_window_statistics_are_missing_when_the_table_is_shorter_than_the_window() -> None:
    values = np.array([[1.0], [2.0]])

    mean, std = window_statistics(values, 3)

    assert np.isnan(mean).all()
    assert np.isnan(std).all()


# --- daylight saving --------------------------------------------------------------------


def test_a_lag_counts_elapsed_hours_across_a_clock_change(
    settings: Settings, demand: pd.DataFrame
) -> None:
    features = build_features(demand, demand, settings)
    # 08:00 on the Sunday the clocks went forward. 24 real hours earlier the clock
    # showed 07:00 on Saturday, because the night in between lost an hour.
    target = _local("2025-03-09 08:00")
    assert target - pd.Timedelta(hours=24) == _local("2025-03-08 07:00")

    row = _row(features, 161, target)

    assert row[lag_column(24)] == _row(demand, 161, _local("2025-03-08 07:00"))[TARGET]
    assert row[HOUR] == 8


def test_features_do_not_break_on_a_25_hour_day(settings: Settings) -> None:
    demand = demand_frame(settings, "2025-10-20 00:00", N_HOURS)
    assert (demand[HOUR_START].dt.date == pd.Timestamp("2025-11-02").date()).sum() == 3 * 25

    features = build_features(demand, demand, settings)

    # The second 01:00 of the night, already in standard time. 24 real hours earlier
    # the clock showed 02:00 on Saturday, because the night in between gained an hour.
    target = pd.Timestamp("2025-11-02 01:00:00-05:00").tz_convert(TIMEZONE)
    earlier = pd.Timestamp("2025-11-01 02:00:00-04:00").tz_convert(TIMEZONE)
    assert target - pd.Timedelta(hours=24) == earlier
    assert _row(features, 132, target)[lag_column(24)] == _row(demand, 132, earlier)[TARGET]
    assert _row(features, 132, target)[HOUR] == 1


# --- calendar ---------------------------------------------------------------------------


def test_calendar_features_are_read_from_the_local_clock() -> None:
    hours = _hours("2025-01-15 08:00", "2025-03-08 23:00", "2025-03-09 03:00", "2025-06-30 00:00")

    calendar = calendar_features(hours)

    assert list(calendar.columns) == CALENDAR_FEATURES
    assert calendar[HOUR].tolist() == [8, 23, 3, 0]
    assert calendar[DAY_OF_WEEK].tolist() == [2, 5, 6, 0]  # Wednesday, Saturday, Sunday, Monday
    assert calendar[MONTH].tolist() == [1, 3, 3, 6]
    assert calendar[IS_WEEKEND].tolist() == [False, True, True, False]
    assert calendar[IS_HOLIDAY].tolist() == [False] * 4
    assert calendar.dtypes.astype(str).tolist() == ["int64", "int64", "int64", "bool", "bool"]


def test_weekend_is_saturday_and_sunday() -> None:
    week = pd.Series(pd.date_range(_local("2025-01-13 12:00"), periods=7, freq="D"))  # Monday

    assert weekend_flag(week).tolist() == [False] * 5 + [True] * 2


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        ("2025-01-01", True),  # New Year's Day
        ("2025-01-20", True),  # Martin Luther King Jr. Day
        ("2025-05-26", True),  # Memorial Day
        ("2025-06-19", True),  # Juneteenth
        ("2025-07-04", True),  # Independence Day
        ("2026-07-03", True),  # Friday, the day off for Saturday 4 July 2026
        ("2025-02-12", False),  # Lincoln's Birthday: a New York state holiday only
        ("2025-01-02", False),
        ("2025-03-17", False),
    ],
)
def test_us_public_holidays_are_flagged(day: str, expected: bool) -> None:
    assert holiday_flag(_hours(f"{day} 12:00")).tolist() == [expected]


def test_a_holiday_lasts_from_local_midnight_to_local_midnight() -> None:
    hours = _hours("2024-12-31 23:00", "2025-01-01 00:00", "2025-01-01 23:00", "2025-01-02 00:00")

    # In UTC the first of these is already 1 January and the third is 2 January.
    assert holiday_flag(hours).tolist() == [False, True, True, False]


def test_calendar_features_of_a_row_describe_its_target_hour(
    settings: Settings, demand: pd.DataFrame
) -> None:
    features = build_features(demand, demand, settings)

    row = _row(features, 237, _local("2025-03-15 21:00"))

    assert row[HOUR] == 21
    assert row[DAY_OF_WEEK] == 5
    assert row[MONTH] == 3
    assert row[IS_WEEKEND]
    assert not row[IS_HOLIDAY]


def test_hours_given_in_another_timezone_are_read_in_the_project_timezone(
    settings: Settings, demand: pd.DataFrame
) -> None:
    in_utc = demand.assign(**{HOUR_START: demand[HOUR_START].dt.tz_convert("UTC")})

    pd.testing.assert_frame_equal(
        build_features(in_utc, in_utc, settings), build_features(demand, demand, settings)
    )


# --- weather ----------------------------------------------------------------------------


def test_weather_of_the_target_hour_is_passed_through_unchanged(
    settings: Settings, demand: pd.DataFrame
) -> None:
    variables = settings.data.weather.variables

    features = build_features(demand, demand, settings)

    pd.testing.assert_frame_equal(features[variables], demand[variables])
    assert (features[variables].dtypes == "float64").all()


# --- no leakage -------------------------------------------------------------------------

# Changed hours: an ordinary one, the last one before the clocks go forward (hour 193
# is 01:00 on 9 March) and one after the change.
CHANGED_HOURS = [30, 193, 260]


@pytest.mark.parametrize("zone", TEST_ZONES)
@pytest.mark.parametrize("position", CHANGED_HOURS)
def test_no_feature_reads_anything_newer_than_the_horizon(
    settings: Settings, demand: pd.DataFrame, zone: int, position: int
) -> None:
    """The most important test of the gold layer.

    Change the pickups of one zone at one hour ``t`` and build the features again.
    Nothing may change in a row whose target hour is before ``t + horizon``, because
    that row is predicted before the pickups of ``t`` are known. Nothing may change in
    another zone at all.
    """
    horizon = pd.Timedelta(hours=settings.features.horizon_hours)
    inputs = feature_columns(settings)
    changed_hour = demand[HOUR_START].unique()[position]
    is_changed_cell = (demand[ZONE_ID] == zone) & (demand[HOUR_START] == changed_hour)
    assert is_changed_cell.sum() == 1
    before = build_features(demand, demand, settings)

    changed_demand = demand.copy()
    changed_demand.loc[is_changed_cell, TARGET] += 1000
    after = build_features(changed_demand, changed_demand, settings)

    changed = _changed_rows(before, after, inputs)
    too_early = before[HOUR_START] < changed_hour + horizon
    other_zone = before[ZONE_ID] != zone
    assert not (changed & too_early).any()
    assert not (changed & other_zone).any()
    # The test would also pass if nothing ever changed, so check that the change is
    # seen at all, and first at exactly ``t + horizon``.
    assert changed.any()
    assert before.loc[changed, HOUR_START].min() == changed_hour + horizon
    # Every lag sees it at its own distance, and nowhere else.
    for lag in settings.features.lags:
        lag_changed = _changed_rows(before, after, [lag_column(lag)])
        expected_hour = changed_hour + pd.Timedelta(hours=lag)
        assert before.loc[lag_changed, HOUR_START].tolist() == [expected_hour]
    # Every window sees it for as many hours as the window is long, starting one
    # horizon later. (Fewer where the table ends first.)
    for window in settings.features.rolling_windows:
        window_changed = _changed_rows(before, after, [rolling_mean_column(window)])
        seen_at = before.loc[window_changed, HOUR_START]
        assert seen_at.min() >= changed_hour + horizon
        assert seen_at.max() < changed_hour + horizon + pd.Timedelta(hours=window)


def test_features_of_a_day_do_not_depend_on_any_later_pickups(
    settings: Settings, demand: pd.DataFrame
) -> None:
    inputs = feature_columns(settings)
    cut = _local("2025-03-16 00:00")
    before = build_features(demand, demand, settings)

    scrambled = demand.copy()
    is_later = scrambled[HOUR_START] >= cut
    scrambled.loc[is_later, TARGET] = scrambled.loc[is_later, TARGET] * 7 + 13
    after = build_features(scrambled, scrambled, settings)

    horizon = pd.Timedelta(hours=settings.features.horizon_hours)
    known_in_time = before[HOUR_START] < cut + horizon
    pd.testing.assert_frame_equal(
        after.loc[known_in_time, inputs], before.loc[known_in_time, inputs]
    )
    assert _changed_rows(before, after, inputs)[~known_in_time].all()


# --- serving: future hours without a target ---------------------------------------------


def test_features_for_tomorrow_need_no_pickups_for_tomorrow(
    settings: Settings, demand: pd.DataFrame
) -> None:
    """The serving use: demand known so far, plus tomorrow's hours and weather forecast."""
    variables = settings.data.weather.variables
    midnight = _local("2025-03-20 00:00")
    is_tomorrow = (demand[HOUR_START] >= midnight) & (
        demand[HOUR_START] < midnight + pd.Timedelta(days=1)
    )
    history = demand.loc[demand[HOUR_START] < midnight, [ZONE_ID, HOUR_START, TARGET]]
    targets = demand.loc[is_tomorrow, [ZONE_ID, HOUR_START, *variables]]
    assert TARGET not in targets.columns

    served = build_features(history, targets, settings)

    assert len(served) == 3 * 24
    assert served.notna().all().all()
    # The same rows as training builds them, to the last bit.
    trained = build_features(demand, demand, settings)
    pd.testing.assert_frame_equal(served, trained[is_tomorrow], check_exact=True)


def test_a_short_history_gives_the_same_features_as_a_long_one(
    settings: Settings, demand: pd.DataFrame
) -> None:
    midnight = _local("2025-03-20 00:00")
    targets = demand[demand[HOUR_START].between(midnight, midnight + pd.Timedelta(hours=23))]
    long_history = demand[demand[HOUR_START] < midnight]
    just_enough = long_history[
        long_history[HOUR_START] >= midnight - pd.Timedelta(hours=warm_up_hours(settings))
    ]

    from_long = build_features(long_history, targets, settings)
    from_short = build_features(just_enough, targets, settings)

    assert from_short.notna().all().all()
    pd.testing.assert_frame_equal(from_short, from_long, check_exact=True)


def test_target_beyond_the_known_history_has_missing_features(
    settings: Settings, demand: pd.DataFrame
) -> None:
    midnight = _local("2025-03-20 00:00")
    history = demand[demand[HOUR_START] < midnight]
    # 24 hours after midnight is the first hour whose 24 h lag has not been observed.
    targets = demand[demand[HOUR_START] == midnight + pd.Timedelta(hours=24)]

    features = build_features(history, targets, settings)

    assert features[lag_column(24)].isna().all()
    assert features[rolling_mean_column(24)].isna().all()
    assert features[lag_column(48)].notna().all()


def test_target_before_the_history_or_in_an_unknown_zone_has_missing_features(
    settings: Settings, demand: pd.DataFrame
) -> None:
    history = demand[(demand[HOUR_START] >= _local("2025-03-10 00:00")) & (demand[ZONE_ID] != 237)]

    features = build_features(history, demand, settings)

    before_history = features[HOUR_START] < _local("2025-03-10 00:00")
    unknown_zone = features[ZONE_ID] == 237
    history_columns = [lag_column(lag) for lag in settings.features.lags]
    assert features.loc[before_history | unknown_zone, history_columns].isna().all().all()
    assert features.loc[~before_history & ~unknown_zone, lag_column(24)].notna().any()
    assert features[CALENDAR_FEATURES].notna().all().all()


# --- refusing bad input -----------------------------------------------------------------


def test_history_without_pickups_is_refused(settings: Settings, demand: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=f"history has no column {TARGET}"):
        build_features(demand.drop(columns=TARGET), demand, settings)


def test_targets_without_weather_are_refused(settings: Settings, demand: pd.DataFrame) -> None:
    variable = settings.data.weather.variables[0]

    with pytest.raises(ValueError, match=f"targets has no column {variable}"):
        build_features(demand, demand.drop(columns=variable), settings)


def test_empty_tables_are_refused(settings: Settings, demand: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="history has no rows"):
        build_features(demand.head(0), demand, settings)
    with pytest.raises(ValueError, match="targets has no rows"):
        build_features(demand, demand.head(0), settings)


def test_two_rows_for_the_same_zone_and_hour_are_refused(
    settings: Settings, demand: pd.DataFrame
) -> None:
    doubled = pd.concat([demand, demand.head(1)], ignore_index=True)

    with pytest.raises(ValueError, match="history has more than one row"):
        build_features(doubled, demand, settings)
    with pytest.raises(ValueError, match="targets has more than one row"):
        build_features(demand, doubled, settings)


# --- the training table -----------------------------------------------------------------


def test_warm_up_is_the_longest_look_back_of_any_feature(settings: Settings) -> None:
    assert warm_up_hours(settings) == 24 + 168 - 1
    assert warm_up_hours(_small_settings(settings)) == 4


def test_feature_table_follows_the_gold_schema(settings: Settings, demand: pd.DataFrame) -> None:
    table, _ = build_feature_table(demand, settings)

    schema = gold_features_schema(settings)
    schema.validate(table)
    assert list(table.columns) == list(schema.columns)
    assert table.notna().all().all()


def test_feature_table_drops_the_warm_up_hours_of_every_zone(
    settings: Settings, demand: pd.DataFrame
) -> None:
    warm_up = warm_up_hours(settings)

    table, rows_dropped = build_feature_table(demand, settings)

    assert rows_dropped == 3 * warm_up
    assert len(table) == 3 * (N_HOURS - warm_up)
    first_kept = _local(FIRST_HOUR) + pd.Timedelta(hours=warm_up)
    assert table.groupby(ZONE_ID)[HOUR_START].min().tolist() == [first_kept] * 3
    assert list(table.index) == list(range(len(table)))


def test_feature_table_keeps_the_target_and_flag_of_each_hour(
    settings: Settings, demand: pd.DataFrame
) -> None:
    demand[IS_OUTLIER] = demand[TARGET] > 250

    table, _ = build_feature_table(demand, settings)

    keys = [ZONE_ID, HOUR_START]
    merged = table.merge(demand, on=keys, suffixes=("", "_demand"), validate="one_to_one")
    assert len(merged) == len(table)
    assert (merged[TARGET] == merged[f"{TARGET}_demand"]).all()
    assert (merged[IS_OUTLIER] == merged[f"{IS_OUTLIER}_demand"]).all()
    assert merged[IS_OUTLIER].any()


def test_feature_table_rows_equal_the_rows_served_for_the_same_hours(
    settings: Settings, demand: pd.DataFrame
) -> None:
    """Training and serving cannot drift apart: both come from ``build_features``."""
    table, _ = build_feature_table(demand, settings)
    inputs = feature_columns(settings)
    day = _local("2025-03-18 00:00")
    last_hour = day + pd.Timedelta(hours=23)
    targets = demand[demand[HOUR_START].between(day, last_hour)]

    served = build_features(demand[demand[HOUR_START] < day], targets, settings)

    trained = table[table[HOUR_START].between(day, last_hour)]
    pd.testing.assert_frame_equal(
        served[inputs].reset_index(drop=True),
        trained[inputs].reset_index(drop=True),
        check_exact=True,
    )
