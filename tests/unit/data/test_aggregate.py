from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from demand.config import Settings
from demand.data.aggregate import (
    TooFewZonesError,
    WeatherCoverageError,
    add_counts,
    build_demand,
    busiest_zones,
    count_hourly_pickups,
    count_pickups_per_zone,
    flag_outliers,
    join_weather,
    pickup_hour,
    to_complete_grid,
    training_period_end,
)
from demand.data.clean_weather import local_hours
from demand.schemas import (
    DROPOFF_DATETIME,
    HOUR_START,
    IS_OUTLIER,
    PICKUP_DATETIME,
    PICKUP_ZONE_ID,
    TARGET,
    TRIP_DISTANCE,
    ZONE_ID,
    gold_demand_schema,
)
from tests.conftest import TEST_ZONES
from tests.unit.data.conftest import demand_frame, silver_trips_frame

TIMEZONE = "America/New_York"
SPRING_FORWARD = date(2025, 3, 9)  # 02:00 does not exist: 23 hours
FALL_BACK = date(2025, 11, 2)  # 01:00 exists twice: 25 hours


def _hour(text: str) -> pd.Timestamp:
    return pd.Timestamp(text).tz_convert(TIMEZONE)


def _counts(settings: Settings, pickups: list[tuple[str, int]]) -> dict[tuple[int, str], int]:
    """Hourly counts as a plain dictionary, with hours written as text."""
    counts = count_hourly_pickups(silver_trips_frame(settings, pickups), TEST_ZONES)
    return {(zone, hour.isoformat()): int(count) for (zone, hour), count in counts.items()}


def _weather(settings: Settings, hours: pd.DatetimeIndex) -> pd.DataFrame:
    variables = settings.data.weather.variables
    values = {name: [float(position) for position in range(len(hours))] for name in variables}
    return pd.DataFrame({HOUR_START: hours, **values})


# --- the training period ----------------------------------------------------------------


def test_training_period_ends_at_local_midnight_after_the_last_training_day(
    settings: Settings,
) -> None:
    assert training_period_end(settings) == pd.Timestamp("2025-05-01 00:00:00-04:00")


# --- the hour of a pickup ---------------------------------------------------------------


def test_pickup_hour_is_the_start_of_the_local_hour(settings: Settings) -> None:
    trips = silver_trips_frame(
        settings, [("2025-01-15 08:00:00-05:00", 161), ("2025-01-15 08:59:59-05:00", 161)]
    )

    hours = pickup_hour(trips[PICKUP_DATETIME])

    assert list(hours) == [_hour("2025-01-15 08:00:00-05:00")] * 2
    assert str(hours.dt.tz) == TIMEZONE


def test_the_two_one_o_clock_hours_of_the_fall_back_night_stay_apart(
    settings: Settings,
) -> None:
    trips = silver_trips_frame(
        settings, [("2025-11-02 01:30:00-04:00", 161), ("2025-11-02 01:30:00-05:00", 161)]
    )

    hours = pickup_hour(trips[PICKUP_DATETIME])

    assert list(hours) == [_hour("2025-11-02 01:00:00-04:00"), _hour("2025-11-02 01:00:00-05:00")]
    assert hours.iloc[1] - hours.iloc[0] == pd.Timedelta(hours=1)


# --- counting ---------------------------------------------------------------------------


def test_pickups_are_counted_per_zone_and_hour(settings: Settings) -> None:
    counts = _counts(
        settings,
        [
            ("2025-01-15 08:05:00-05:00", 161),
            ("2025-01-15 08:55:00-05:00", 161),
            ("2025-01-15 09:00:00-05:00", 161),
            ("2025-01-15 08:30:00-05:00", 132),
        ],
    )

    assert counts == {
        (132, "2025-01-15T08:00:00-05:00"): 1,
        (161, "2025-01-15T08:00:00-05:00"): 2,
        (161, "2025-01-15T09:00:00-05:00"): 1,
    }


def test_trips_from_zones_outside_the_frozen_list_are_ignored(settings: Settings) -> None:
    counts = _counts(
        settings, [("2025-01-15 08:05:00-05:00", 161), ("2025-01-15 08:10:00-05:00", 50)]
    )

    assert counts == {(161, "2025-01-15T08:00:00-05:00"): 1}


def test_trips_without_dropoff_time_or_distance_are_still_counted(settings: Settings) -> None:
    trips = silver_trips_frame(
        settings,
        [("2025-01-15 08:05:00-05:00", 161), ("2025-01-15 08:10:00-05:00", 161)],
        complete=False,
    )
    assert trips[DROPOFF_DATETIME].isna().all()
    assert trips[TRIP_DISTANCE].isna().all()

    counts = count_hourly_pickups(trips, TEST_ZONES)

    assert list(counts) == [2]


def test_only_the_pickup_columns_are_needed(settings: Settings) -> None:
    trips = silver_trips_frame(settings, [("2025-01-15 08:05:00-05:00", 161)])

    counts = count_hourly_pickups(trips[[PICKUP_DATETIME, PICKUP_ZONE_ID]], TEST_ZONES)

    assert list(counts) == [1]


def test_counts_of_several_files_are_added(settings: Settings) -> None:
    first = count_hourly_pickups(
        silver_trips_frame(settings, [("2025-01-31 23:10:00-05:00", 161)]), TEST_ZONES
    )
    second = count_hourly_pickups(
        silver_trips_frame(
            settings, [("2025-01-31 23:50:00-05:00", 161), ("2025-02-01 00:10:00-05:00", 161)]
        ),
        TEST_ZONES,
    )

    total = add_counts([first, second])

    assert list(total) == [2, 1]
    assert total.index.names == [ZONE_ID, HOUR_START]


# --- the complete grid ------------------------------------------------------------------


def test_hours_without_pickups_are_explicit_zeros(settings: Settings) -> None:
    trips = silver_trips_frame(
        settings, [("2025-01-15 08:05:00-05:00", 161), ("2025-01-15 08:10:00-05:00", 161)]
    )
    hours = local_hours(date(2025, 1, 15), date(2025, 1, 15), TIMEZONE)

    grid = to_complete_grid(count_hourly_pickups(trips, TEST_ZONES), TEST_ZONES, hours)

    assert len(grid) == 3 * 24
    assert list(grid.columns) == [ZONE_ID, HOUR_START, TARGET]
    assert grid[TARGET].dtype == "int64"
    assert int(grid[TARGET].sum()) == 2
    busy = grid[(grid[ZONE_ID] == 161) & (grid[HOUR_START] == _hour("2025-01-15 08:00:00-05:00"))]
    assert list(busy[TARGET]) == [2]
    assert (grid.loc[grid[ZONE_ID] != 161, TARGET] == 0).all()


def test_grid_is_sorted_by_zone_and_then_by_hour(settings: Settings) -> None:
    hours = local_hours(date(2025, 1, 15), date(2025, 1, 16), TIMEZONE)
    no_counts = count_hourly_pickups(silver_trips_frame(settings, []), TEST_ZONES)

    grid = to_complete_grid(no_counts, [237, 132, 161], hours)

    assert list(grid[ZONE_ID].unique()) == [132, 161, 237]
    for _, zone_rows in grid.groupby(ZONE_ID):
        assert zone_rows[HOUR_START].is_monotonic_increasing
        assert len(zone_rows) == 48


def test_the_day_the_clocks_go_forward_has_23_hours(settings: Settings) -> None:
    trips = silver_trips_frame(
        settings, [("2025-03-09 01:30:00-05:00", 161), ("2025-03-09 03:30:00-04:00", 161)]
    )
    hours = local_hours(SPRING_FORWARD, SPRING_FORWARD, TIMEZONE)

    grid = to_complete_grid(count_hourly_pickups(trips, TEST_ZONES), [161], hours)

    assert len(grid) == 23
    assert list(grid[HOUR_START].dt.hour[:4]) == [0, 1, 3, 4]
    assert list(grid[TARGET][:4]) == [0, 1, 1, 0]
    assert (grid[HOUR_START].diff().dropna() == pd.Timedelta(hours=1)).all()


def test_the_day_the_clocks_go_back_has_25_hours(settings: Settings) -> None:
    trips = silver_trips_frame(
        settings,
        [
            ("2025-11-02 01:10:00-04:00", 161),
            ("2025-11-02 01:20:00-05:00", 161),
            ("2025-11-02 01:40:00-05:00", 161),
        ],
    )
    hours = local_hours(FALL_BACK, FALL_BACK, TIMEZONE)

    grid = to_complete_grid(count_hourly_pickups(trips, TEST_ZONES), [161], hours)

    assert len(grid) == 25
    assert list(grid[HOUR_START].dt.hour[:4]) == [0, 1, 1, 2]
    assert list(grid[TARGET][:4]) == [0, 1, 2, 0]
    assert not grid.duplicated(subset=[ZONE_ID, HOUR_START]).any()


def test_counts_outside_the_grid_are_left_out(settings: Settings) -> None:
    trips = silver_trips_frame(
        settings, [("2025-01-15 08:05:00-05:00", 161), ("2025-01-16 08:05:00-05:00", 161)]
    )
    hours = local_hours(date(2025, 1, 15), date(2025, 1, 15), TIMEZONE)

    grid = to_complete_grid(count_hourly_pickups(trips, TEST_ZONES), TEST_ZONES, hours)

    assert int(grid[TARGET].sum()) == 1


# --- zone selection ---------------------------------------------------------------------


def test_zone_counts_use_only_trips_before_the_end_of_the_training_period(
    settings: Settings,
) -> None:
    trips = silver_trips_frame(
        settings,
        [
            ("2025-04-30 23:59:59-04:00", 161),
            ("2025-05-01 00:00:00-04:00", 50),
            ("2025-05-01 00:00:01-04:00", 50),
            ("2025-06-01 12:00:00-04:00", 50),
        ],
    )

    counts = count_pickups_per_zone(trips, training_period_end(settings))

    assert counts.to_dict() == {161: 1}


def test_zone_busy_only_after_the_training_period_is_not_selected(settings: Settings) -> None:
    training_trips = [("2025-02-10 08:00:00-05:00", 161)] * 3
    training_trips += [("2025-02-10 08:00:00-05:00", 132)] * 2
    replay_trips = [("2025-05-10 08:00:00-04:00", 50)] * 40
    trips = silver_trips_frame(settings, training_trips + replay_trips)

    selected = busiest_zones(count_pickups_per_zone(trips, training_period_end(settings)), 2)

    assert selected == {161: 3, 132: 2}


def test_busiest_zones_come_first() -> None:
    counts = pd.Series({7: 10, 3: 30, 9: 20, 1: 5})

    assert list(busiest_zones(counts, 3).items()) == [(3, 30), (9, 20), (7, 10)]


@pytest.mark.parametrize("order", [[5, 2, 8, 4], [8, 4, 2, 5], [4, 8, 5, 2]])
def test_ties_are_broken_by_the_lower_zone_id(order: list[int]) -> None:
    pickups = {5: 10, 2: 10, 8: 10, 4: 99}
    counts = pd.Series({zone: pickups[zone] for zone in order})

    assert list(busiest_zones(counts, 3)) == [4, 2, 5]


def test_selection_fails_when_too_few_zones_have_pickups() -> None:
    counts = pd.Series({7: 10, 3: 0})

    with pytest.raises(TooFewZonesError, match=r"only 1 zones.*asks for 2"):
        busiest_zones(counts, 2)


# --- weather ----------------------------------------------------------------------------


def test_weather_of_the_hour_is_added_to_every_zone(settings: Settings) -> None:
    variables = settings.data.weather.variables
    hours = local_hours(date(2025, 1, 15), date(2025, 1, 15), TIMEZONE)
    no_counts = count_hourly_pickups(silver_trips_frame(settings, []), TEST_ZONES)
    grid = to_complete_grid(no_counts, TEST_ZONES, hours)

    joined = join_weather(grid, _weather(settings, hours), variables)

    assert list(joined.columns) == [ZONE_ID, HOUR_START, TARGET, *variables]
    assert len(joined) == len(grid)
    for _, zone_rows in joined.groupby(ZONE_ID):
        assert list(zone_rows[variables[0]]) == [float(position) for position in range(24)]
    assert (joined[variables].dtypes == "float64").all()


def test_missing_weather_hour_stops_the_join(settings: Settings) -> None:
    hours = local_hours(date(2025, 1, 15), date(2025, 1, 15), TIMEZONE)
    no_counts = count_hourly_pickups(silver_trips_frame(settings, []), TEST_ZONES)
    grid = to_complete_grid(no_counts, TEST_ZONES, hours)
    weather = _weather(settings, hours).drop(index=5)

    with pytest.raises(WeatherCoverageError, match=r"3 rows.*python -m demand silver"):
        join_weather(grid, weather, settings.data.weather.variables)


# --- outliers ---------------------------------------------------------------------------

FIRST_MONDAY = "2025-01-06 00:00"
TRAINING_WEEKS = 8
TRAINING_END = pd.Timestamp("2025-03-03 00:00", tz=TIMEZONE)  # eight weeks after FIRST_MONDAY
ZSCORE = 3.0


def _regular_demand(settings: Settings, weeks: int) -> pd.DataFrame:
    """One zone with a dull pattern: about 100 pickups on working days, 10 at weekends."""
    demand = demand_frame(settings, FIRST_MONDAY, weeks * 7 * 24, zones=[161])
    hours = demand[HOUR_START]
    wobble = hours.dt.dayofyear % 2  # so that every slot has a spread above zero
    demand[TARGET] = (100 - 90 * (hours.dt.dayofweek >= 5)) + wobble
    return demand


def _flagged_hours(demand: pd.DataFrame, flags: pd.Series) -> list[str]:
    return [hour.strftime("%Y-%m-%d %H:%M") for hour in demand.loc[flags, HOUR_START]]


def _set_pickups(demand: pd.DataFrame, hour: str, pickups: int) -> None:
    demand.loc[demand[HOUR_START] == pd.Timestamp(hour, tz=TIMEZONE), TARGET] = pickups


def test_regular_demand_has_no_outliers(settings: Settings) -> None:
    demand = _regular_demand(settings, TRAINING_WEEKS + 2)

    flags = flag_outliers(demand, TRAINING_END, ZSCORE)

    assert flags.dtype == "bool"
    assert flags.name == IS_OUTLIER
    assert not flags.any()


def test_hour_far_above_or_below_its_slot_is_flagged(settings: Settings) -> None:
    demand = _regular_demand(settings, TRAINING_WEEKS + 2)
    _set_pickups(demand, "2025-01-15 08:00", 400)  # a Wednesday in the training period
    _set_pickups(demand, "2025-03-05 17:00", 0)  # a Wednesday in the replay period

    flags = flag_outliers(demand, TRAINING_END, ZSCORE)

    assert _flagged_hours(demand, flags) == ["2025-01-15 08:00", "2025-03-05 17:00"]


def test_working_days_and_weekends_are_judged_separately(settings: Settings) -> None:
    demand = _regular_demand(settings, TRAINING_WEEKS + 2)
    # 100 pickups are normal at 08:00 on a Friday and far too many on a Saturday.
    _set_pickups(demand, "2025-03-07 08:00", 100)
    _set_pickups(demand, "2025-03-08 08:00", 100)

    flags = flag_outliers(demand, TRAINING_END, ZSCORE)

    assert _flagged_hours(demand, flags) == ["2025-03-08 08:00"]


def test_replay_data_cannot_change_the_flag_of_a_training_hour(settings: Settings) -> None:
    demand = _regular_demand(settings, TRAINING_WEEKS + 2)
    _set_pickups(demand, "2025-01-15 08:00", 400)
    is_training = demand[HOUR_START] < TRAINING_END
    before = flag_outliers(demand, TRAINING_END, ZSCORE)

    changed = demand.copy()
    changed.loc[~is_training, TARGET] = changed.loc[~is_training, TARGET] * 50 + 1000
    after = flag_outliers(changed, TRAINING_END, ZSCORE)

    assert before[is_training].sum() == 1
    pd.testing.assert_series_equal(after[is_training], before[is_training])
    # The replay hours themselves are measured with the training yardstick.
    assert after[~is_training].all()


def test_training_statistics_are_the_same_with_or_without_replay_rows(
    settings: Settings,
) -> None:
    demand = _regular_demand(settings, TRAINING_WEEKS + 2)
    _set_pickups(demand, "2025-01-15 08:00", 400)
    training_only = demand[demand[HOUR_START] < TRAINING_END]

    with_replay = flag_outliers(demand, TRAINING_END, ZSCORE)
    without_replay = flag_outliers(training_only, TRAINING_END, ZSCORE)

    pd.testing.assert_series_equal(with_replay.loc[training_only.index], without_replay)


def test_slot_without_spread_flags_nothing(settings: Settings) -> None:
    demand = _regular_demand(settings, TRAINING_WEEKS + 2)
    demand[TARGET] = 50
    _set_pickups(demand, "2025-03-05 17:00", 5000)

    assert not flag_outliers(demand, TRAINING_END, ZSCORE).any()


def test_nothing_is_flagged_when_there_is_no_training_data(settings: Settings) -> None:
    demand = _regular_demand(settings, 2)
    _set_pickups(demand, "2025-01-15 08:00", 400)
    before_the_data = pd.Timestamp("2025-01-01 00:00", tz=TIMEZONE)

    assert not flag_outliers(demand, before_the_data, ZSCORE).any()


def test_zones_are_judged_separately(settings: Settings) -> None:
    quiet = _regular_demand(settings, TRAINING_WEEKS + 2)
    busy = quiet.assign(**{ZONE_ID: 237, TARGET: quiet[TARGET] * 10})
    demand = pd.concat([quiet, busy], ignore_index=True)
    # 1000 pickups are normal for the busy zone and far too many for the quiet one.
    is_quiet_zone = demand[ZONE_ID] == 161
    at_hour = demand[HOUR_START] == pd.Timestamp("2025-03-05 17:00", tz=TIMEZONE)
    demand.loc[at_hour, TARGET] = 1000

    flags = flag_outliers(demand, TRAINING_END, ZSCORE)

    assert list(demand.loc[flags, ZONE_ID]) == [161]
    assert flags[is_quiet_zone & at_hour].all()


# --- the whole demand table -------------------------------------------------------------


def test_demand_table_follows_the_gold_schema(settings: Settings) -> None:
    data = settings.data
    hours = local_hours(data.start_date, data.end_date, TIMEZONE)
    trips = silver_trips_frame(
        settings,
        [
            ("2025-01-15 08:05:00-05:00", 161),
            ("2025-03-09 03:30:00-04:00", 132),
            ("2025-06-30 23:59:00-04:00", 237),
            ("2025-06-30 23:59:00-04:00", 50),
        ],
    )

    demand = build_demand(
        count_hourly_pickups(trips, TEST_ZONES), _weather(settings, hours), settings
    )

    schema = gold_demand_schema(settings)
    schema.validate(demand)
    assert list(demand.columns) == list(schema.columns)
    assert len(demand) == 3 * len(hours)
    assert len(hours) == 181 * 24 - 1
    assert int(demand[TARGET].sum()) == 3
