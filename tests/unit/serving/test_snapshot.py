"""The data snapshot: which days it allows, the issue moment, and the rows of a request."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from demand.config import Settings
from demand.data.gold import gold_demand_path
from demand.data.silver import silver_weather_path, write_parquet
from demand.schemas import HOUR_START, TARGET, ZONE_ID, feature_columns
from demand.serving.snapshot import (
    RequestRefusedError,
    SnapshotError,
    day_features,
    day_hours,
    first_whole_day,
    issue_moment,
    last_whole_day,
    load_snapshot,
    make_snapshot,
)
from tests.conftest import TEST_ZONES
from tests.unit.serving.conftest import (
    CLOCKS_FORWARD,
    FIRST_DATA_DAY,
    FIRST_DAY,
    LAST_DATA_DAY,
    tables,
)

TIMEZONE = "America/New_York"
CLOCKS_BACK = date(2025, 11, 2)


def at(text: str) -> pd.Timestamp:
    """A local time; with a UTC offset in ``text`` where the clock time exists twice."""
    moment = pd.Timestamp(text)
    return moment.tz_convert(TIMEZONE) if moment.tz else moment.tz_localize(TIMEZONE)


# --- the issue moment and the hours of a day --------------------------------------------


@pytest.mark.parametrize(
    ("day", "hours"), [(date(2025, 3, 8), 24), (CLOCKS_FORWARD, 23), (CLOCKS_BACK, 25)]
)
def test_a_day_has_its_local_hours(settings: Settings, day: date, hours: int) -> None:
    assert len(day_hours(day, settings)) == hours


@pytest.mark.parametrize(
    ("day", "moment"),
    [
        (date(2025, 5, 14), "2025-05-13 00:00"),
        # The clocks go forward: 00:00 on the day before is an hour later than needed.
        (CLOCKS_FORWARD, "2025-03-08 00:00"),
        (date(2025, 3, 10), "2025-03-09 00:00"),
        # The clocks go back: the last hour reads an hour that is complete at 01:00.
        (CLOCKS_BACK, "2025-11-01 01:00"),
        # 01:00 exists twice that night; this is the first one, still summer time.
        (date(2025, 11, 3), "2025-11-02 01:00-04:00"),
    ],
)
def test_the_forecast_is_issued_at_midnight_the_day_before_or_when_its_data_is_complete(
    settings: Settings, day: date, moment: str
) -> None:
    assert issue_moment(day, settings) == at(moment)


def test_whole_days_start_at_or_end_by_a_moment() -> None:
    assert first_whole_day(at("2025-03-06 00:00")) == date(2025, 3, 6)
    assert first_whole_day(at("2025-03-05 23:00")) == date(2025, 3, 6)
    assert last_whole_day(at("2025-03-13 00:00")) == date(2025, 3, 12)
    assert last_whole_day(at("2025-03-13 05:00")) == date(2025, 3, 12)


# --- which days a snapshot allows -------------------------------------------------------


def test_the_first_day_is_the_first_with_a_full_history(settings: Settings) -> None:
    snapshot = make_snapshot(*tables(settings), settings)

    # 25 February 00:00 plus 215 hours is 5 March 23:00, so 6 March is the first whole day.
    assert snapshot.first_day == FIRST_DAY
    assert snapshot.last_day == LAST_DATA_DAY


def test_the_last_day_is_limited_by_the_weather(settings: Settings) -> None:
    demand, weather = tables(settings, weather_last=LAST_DATA_DAY)
    shorter = weather[weather[HOUR_START] < at("2025-03-11 00:00")]

    assert make_snapshot(demand, shorter, settings).last_day == date(2025, 3, 10)


def test_the_last_day_is_limited_by_the_demand_known_at_its_issue_moment(
    settings: Settings,
) -> None:
    # Demand to 10 March 23:00 is known at 11 March 00:00, the issue moment of 12 March.
    demand, weather = tables(settings, last=date(2025, 3, 10), weather_last=date(2025, 3, 16))

    assert make_snapshot(demand, weather, settings).last_day == date(2025, 3, 12)


def test_the_day_the_clocks_go_back_needs_the_demand_of_an_extra_hour(settings: Settings) -> None:
    # 2 November is issued at 01:00 on 1 November: demand to 31 October 23:00 is not enough.
    demand, weather = tables(
        settings, date(2025, 10, 20), date(2025, 10, 31), weather_last=date(2025, 11, 5)
    )
    assert make_snapshot(demand, weather, settings).last_day == date(2025, 11, 1)

    one_more_hour, _ = tables(settings, date(2025, 10, 20), date(2025, 11, 1))
    known_at_one = one_more_hour[one_more_hour[HOUR_START] <= at("2025-11-01 00:00")]
    assert make_snapshot(known_at_one, weather, settings).last_day == CLOCKS_BACK


def test_weather_that_starts_late_moves_the_first_day(settings: Settings) -> None:
    demand, weather = tables(settings)
    late = weather[weather[HOUR_START] >= at("2025-03-07 12:00")]

    assert make_snapshot(demand, late, settings).first_day == date(2025, 3, 8)


def test_a_snapshot_without_a_single_day_is_refused(settings: Settings) -> None:
    demand, weather = tables(settings, FIRST_DATA_DAY, date(2025, 3, 3))

    with pytest.raises(SnapshotError, match=r"leaves no day to forecast.*215 hours"):
        make_snapshot(demand, weather, settings)


# --- what is checked at startup ---------------------------------------------------------


def test_a_table_that_breaks_its_schema_is_refused(settings: Settings) -> None:
    demand, weather = tables(settings)
    demand.loc[0, TARGET] = -1

    with pytest.raises(SnapshotError, match=r"not valid.*dvc repro"):
        make_snapshot(demand, weather, settings)


def test_a_configured_zone_without_demand_is_refused(settings: Settings) -> None:
    demand, weather = tables(settings)

    with pytest.raises(SnapshotError, match="no rows for zone 237"):
        make_snapshot(demand[demand[ZONE_ID] != 237], weather, settings)


def test_a_zone_that_misses_an_hour_is_refused(settings: Settings) -> None:
    demand, weather = tables(settings)
    hole = (demand[ZONE_ID] == 161) & (demand[HOUR_START] == at("2025-03-01 12:00"))

    with pytest.raises(SnapshotError, match="does not have every hour for every zone"):
        make_snapshot(demand[~hole], weather, settings)


def test_an_hour_missing_in_every_zone_is_refused(settings: Settings) -> None:
    demand, weather = tables(settings)
    hole = demand[HOUR_START] == at("2025-03-01 12:00")

    with pytest.raises(SnapshotError, match="demand table misses 1 hours"):
        make_snapshot(demand[~hole], weather, settings)


def test_weather_with_a_missing_hour_is_refused(settings: Settings) -> None:
    demand, weather = tables(settings)
    hole = weather[HOUR_START] == at("2025-03-10 12:00")

    with pytest.raises(SnapshotError, match="weather table misses 1 hours"):
        make_snapshot(demand, weather[~hole], settings)


def test_the_snapshot_is_read_from_the_gold_and_silver_files(settings: Settings) -> None:
    demand, weather = tables(settings)
    write_parquet(demand, gold_demand_path(settings.paths.gold))
    write_parquet(weather, silver_weather_path(settings.paths.silver))

    snapshot = load_snapshot(settings)

    assert (snapshot.first_day, snapshot.last_day) == (FIRST_DAY, LAST_DATA_DAY)
    assert list(snapshot.history.columns) == [ZONE_ID, HOUR_START, TARGET]


def test_missing_files_are_named(settings: Settings) -> None:
    with pytest.raises(SnapshotError, match=r"demand_hourly\.parquet and .*weather_hourly"):
        load_snapshot(settings)


def test_the_snapshot_reads_nothing_but_its_two_files(settings: Settings, tmp_path: Path) -> None:
    demand, weather = tables(settings)
    write_parquet(demand, gold_demand_path(settings.paths.gold))
    write_parquet(weather, silver_weather_path(settings.paths.silver))
    files_before = sorted(path for path in tmp_path.rglob("*") if path.is_file())

    load_snapshot(settings)

    assert sorted(path for path in tmp_path.rglob("*") if path.is_file()) == files_before


# --- the rows of a request --------------------------------------------------------------


def test_the_rows_are_every_hour_of_the_day_for_each_zone_in_the_order_asked(
    settings: Settings,
) -> None:
    snapshot = make_snapshot(*tables(settings), settings)

    issued, rows = day_features(snapshot, [237, 132], date(2025, 3, 11), settings)

    assert issued == at("2025-03-10 00:00")
    assert rows[ZONE_ID].tolist() == [237] * 24 + [132] * 24
    assert rows[HOUR_START].tolist() == list(day_hours(date(2025, 3, 11), settings)) * 2
    assert not rows[feature_columns(settings)].isna().any().any()


def test_the_day_the_clocks_go_forward_has_23_rows_per_zone(settings: Settings) -> None:
    snapshot = make_snapshot(*tables(settings), settings)

    issued, rows = day_features(snapshot, TEST_ZONES, CLOCKS_FORWARD, settings)

    assert issued == at("2025-03-08 00:00")
    assert len(rows) == 23 * len(TEST_ZONES)


def test_the_day_the_clocks_go_back_has_25_rows_per_zone(settings: Settings) -> None:
    demand, weather = tables(settings, date(2025, 10, 20), date(2025, 11, 3))
    snapshot = make_snapshot(demand, weather, settings)

    issued, rows = day_features(snapshot, [161], CLOCKS_BACK, settings)

    assert issued == at("2025-11-01 01:00")
    assert len(rows) == 25
    assert not rows[feature_columns(settings)].isna().any().any()


def test_no_demand_after_the_issue_moment_is_read(settings: Settings) -> None:
    demand, weather = tables(settings)
    day = date(2025, 3, 11)
    unknown_then = demand[HOUR_START] >= at("2025-03-10 00:00")
    changed = demand.assign(**{TARGET: demand[TARGET].where(~unknown_then, 999)})

    _, rows = day_features(make_snapshot(demand, weather, settings), TEST_ZONES, day, settings)
    _, changed_rows = day_features(
        make_snapshot(changed, weather, settings), TEST_ZONES, day, settings
    )

    pd.testing.assert_frame_equal(rows, changed_rows, check_exact=True)


def test_the_weather_of_the_day_is_the_weather_of_each_hour(settings: Settings) -> None:
    demand, weather = tables(settings)
    snapshot = make_snapshot(demand, weather, settings)

    _, rows = day_features(snapshot, [161], date(2025, 3, 11), settings)

    expected = weather.set_index(HOUR_START).loc[rows[HOUR_START]]
    variables = settings.data.weather.variables
    assert rows[variables].to_numpy().tolist() == expected[variables].to_numpy().tolist()


@pytest.mark.parametrize("zones", [[999], [161, 0]])
def test_a_zone_outside_the_config_is_refused_with_the_zones_there_are(
    settings: Settings, zones: list[int]
) -> None:
    snapshot = make_snapshot(*tables(settings), settings)

    with pytest.raises(RequestRefusedError, match=r"is not forecast.*\[132, 161, 237\]") as error:
        day_features(snapshot, zones, date(2025, 3, 11), settings)

    assert (error.value.field, error.value.kind) == ("zone", "unknown_zone")
    assert error.value.value == [zone for zone in zones if zone not in TEST_ZONES]


@pytest.mark.parametrize("day", [date(2025, 3, 5), date(2025, 3, 13)])
def test_a_day_outside_the_snapshot_is_refused_with_the_days_there_are(
    settings: Settings, day: date
) -> None:
    snapshot = make_snapshot(*tables(settings), settings)

    with pytest.raises(RequestRefusedError, match="the days 2025-03-06 to 2025-03-12") as error:
        day_features(snapshot, [161], day, settings)

    assert (error.value.field, error.value.kind) == ("date", "date_out_of_range")
    assert error.value.value == day.isoformat()
