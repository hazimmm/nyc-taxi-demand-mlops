from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from demand.config import Settings
from demand.data.clean import (
    DISTANCE_OUT_OF_RANGE,
    DROPOFF_BEFORE_PICKUP,
    DUPLICATE,
    DURATION_OUT_OF_RANGE,
    MISSING_REQUIRED_VALUE,
    NEGATIVE_TOTAL_AMOUNT,
    NONEXISTENT_LOCAL_TIME,
    PASSENGER_COUNT_OUT_OF_RANGE,
    PICKUP_OUTSIDE_MONTH,
    RULE_ORDER,
    UNKNOWN_PICKUP_ZONE,
    CleaningCounts,
    RawFormatError,
    clean_trips,
    count_removed,
    localize,
    month_bounds,
    total_counts,
)
from demand.schemas import (
    DROPOFF_DATETIME,
    DURATION_MINUTES,
    PASSENGER_COUNT,
    PICKUP_DATETIME,
    PICKUP_ZONE_ID,
    TOTAL_AMOUNT,
    TRIP_DISTANCE,
    silver_trips_schema,
)
from tests.unit.data.conftest import raw_trip, raw_trips_frame

GOOD_PICKUP = "2025-01-15 08:00:00"
TIMEZONE = "America/New_York"


def _clean(
    settings: Settings, trips: list[dict[str, object]], *, year: int = 2025, month: int = 1
) -> tuple[pd.DataFrame, CleaningCounts]:
    silver, counts = clean_trips(raw_trips_frame(trips), year, month, settings)
    silver_trips_schema(settings).validate(silver)
    return silver, counts


def _only(rule: str | None) -> dict[str, int]:
    """Expected ``removed`` when one row was removed by ``rule`` (or none at all)."""
    return {name: int(name == rule) for name in RULE_ORDER}


def _local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TIMEZONE)


# --- a valid trip -----------------------------------------------------------------------


def test_valid_trip_is_kept_with_silver_names_types_and_timezone(settings: Settings) -> None:
    silver, counts = _clean(settings, [raw_trip(GOOD_PICKUP)])

    assert list(silver.columns) == [
        PICKUP_DATETIME,
        DROPOFF_DATETIME,
        PICKUP_ZONE_ID,
        TRIP_DISTANCE,
        PASSENGER_COUNT,
        TOTAL_AMOUNT,
        DURATION_MINUTES,
    ]
    trip = silver.iloc[0]
    assert trip[PICKUP_DATETIME] == _local("2025-01-15 08:00:00")
    assert trip[DROPOFF_DATETIME] == _local("2025-01-15 08:15:00")
    assert trip[PICKUP_ZONE_ID] == 161
    assert trip[DURATION_MINUTES] == 15.0
    assert silver[PICKUP_ZONE_ID].dtype == "int64"
    assert silver[PASSENGER_COUNT].dtype == "float64"
    assert counts == CleaningCounts(rows_in=1, removed=_only(None), rows_out=1)


def test_trips_keep_their_original_order(settings: Settings) -> None:
    later = raw_trip("2025-01-20 10:00:00")
    earlier = raw_trip("2025-01-03 10:00:00")

    silver, _ = _clean(settings, [later, earlier])

    assert list(silver[PICKUP_DATETIME].dt.day) == [20, 3]


def test_no_rows_in_gives_an_empty_table_with_the_silver_schema(settings: Settings) -> None:
    silver, counts = _clean(settings, [])

    assert silver.empty
    assert counts == CleaningCounts(rows_in=0, removed=_only(None), rows_out=0)


# --- one test case per way a row can break a rule ---------------------------------------

BAD_TRIPS = {
    "no pickup time": (MISSING_REQUIRED_VALUE, raw_trip(GOOD_PICKUP, tpep_pickup_datetime=None)),
    "no pickup zone": (MISSING_REQUIRED_VALUE, raw_trip(GOOD_PICKUP, PULocationID=None)),
    "no dropoff time": (MISSING_REQUIRED_VALUE, raw_trip(GOOD_PICKUP, tpep_dropoff_datetime=None)),
    "no distance": (MISSING_REQUIRED_VALUE, raw_trip(GOOD_PICKUP, trip_distance=None)),
    "no total amount": (MISSING_REQUIRED_VALUE, raw_trip(GOOD_PICKUP, total_amount=None)),
    "evening before the month": (PICKUP_OUTSIDE_MONTH, raw_trip("2024-12-31 23:50:00")),
    "first second of the next month": (PICKUP_OUTSIDE_MONTH, raw_trip("2025-02-01 00:00:00")),
    "stray year": (PICKUP_OUTSIDE_MONTH, raw_trip("2002-12-31 22:59:00")),
    "dropoff before pickup": (
        DROPOFF_BEFORE_PICKUP,
        raw_trip("2025-01-15 08:00:00", "2025-01-15 07:45:00"),
    ),
    "shorter than a minute": (
        DURATION_OUT_OF_RANGE,
        raw_trip("2025-01-15 08:00:00", "2025-01-15 08:00:30"),
    ),
    "longer than six hours": (
        DURATION_OUT_OF_RANGE,
        raw_trip("2025-01-15 08:00:00", "2025-01-15 14:00:01"),
    ),
    "no time passed and no distance": (
        DURATION_OUT_OF_RANGE,
        raw_trip(GOOD_PICKUP, GOOD_PICKUP, trip_distance=0.0),
    ),
    "zero distance inside one zone": (
        DISTANCE_OUT_OF_RANGE,
        raw_trip(GOOD_PICKUP, trip_distance=0.0, PULocationID=161, DOLocationID=161),
    ),
    "negative distance": (DISTANCE_OUT_OF_RANGE, raw_trip(GOOD_PICKUP, trip_distance=-1.2)),
    "too far": (DISTANCE_OUT_OF_RANGE, raw_trip(GOOD_PICKUP, trip_distance=100.1)),
    "negative total amount": (NEGATIVE_TOTAL_AMOUNT, raw_trip(GOOD_PICKUP, total_amount=-0.01)),
    "no passengers": (PASSENGER_COUNT_OUT_OF_RANGE, raw_trip(GOOD_PICKUP, passenger_count=0)),
    "too many passengers": (
        PASSENGER_COUNT_OUT_OF_RANGE,
        raw_trip(GOOD_PICKUP, passenger_count=9),
    ),
    "zone unknown": (UNKNOWN_PICKUP_ZONE, raw_trip(GOOD_PICKUP, PULocationID=264)),
    "zone outside NYC": (UNKNOWN_PICKUP_ZONE, raw_trip(GOOD_PICKUP, PULocationID=265)),
    "zone zero": (UNKNOWN_PICKUP_ZONE, raw_trip(GOOD_PICKUP, PULocationID=0)),
}


@pytest.mark.parametrize(("rule", "bad_trip"), BAD_TRIPS.values(), ids=BAD_TRIPS.keys())
def test_bad_trip_is_removed_and_counted_under_its_rule(
    settings: Settings, rule: str, bad_trip: dict[str, object]
) -> None:
    good_trip = raw_trip("2025-01-20 12:00:00", trip_distance=7.7)

    silver, counts = _clean(settings, [bad_trip, good_trip])

    assert list(silver[TRIP_DISTANCE]) == [7.7]
    assert counts == CleaningCounts(rows_in=2, removed=_only(rule), rows_out=1)


EDGE_TRIPS = {
    "first second of the month": raw_trip("2025-01-01 00:00:00"),
    "last second of the month": raw_trip("2025-01-31 23:59:59"),
    "exactly one minute": raw_trip("2025-01-15 08:00:00", "2025-01-15 08:01:00"),
    "exactly six hours": raw_trip("2025-01-15 08:00:00", "2025-01-15 14:00:00"),
    "exactly 100 miles": raw_trip(GOOD_PICKUP, trip_distance=100.0),
    "free ride": raw_trip(GOOD_PICKUP, total_amount=0.0),
    "negative fare with a positive total": raw_trip(
        GOOD_PICKUP, fare_amount=-4.75, total_amount=8.0, payment_type=0
    ),
    "one passenger": raw_trip(GOOD_PICKUP, passenger_count=1),
    "eight passengers": raw_trip(GOOD_PICKUP, passenger_count=8),
    "passenger count not recorded": raw_trip(GOOD_PICKUP, passenger_count=None),
    "lowest zone": raw_trip(GOOD_PICKUP, PULocationID=1),
    "highest zone": raw_trip(GOOD_PICKUP, PULocationID=263),
}


@pytest.mark.parametrize("edge_trip", EDGE_TRIPS.values(), ids=EDGE_TRIPS.keys())
def test_trip_on_the_edge_of_a_rule_is_kept(
    settings: Settings, edge_trip: dict[str, object]
) -> None:
    _, counts = _clean(settings, [edge_trip])

    assert counts == CleaningCounts(rows_in=1, removed=_only(None), rows_out=1)


def test_missing_passenger_count_stays_missing(settings: Settings) -> None:
    silver, _ = _clean(settings, [raw_trip(GOOD_PICKUP, passenger_count=None)])

    assert silver[PASSENGER_COUNT].isna().all()


# --- duplicates -------------------------------------------------------------------------


def test_exact_copies_are_removed_and_the_first_one_stays(settings: Settings) -> None:
    trip = raw_trip(GOOD_PICKUP)
    other = raw_trip("2025-01-20 12:00:00")

    silver, counts = _clean(settings, [trip, other, trip, trip])

    assert len(silver) == 2
    assert counts == CleaningCounts(
        rows_in=4, removed=_only(DUPLICATE) | {DUPLICATE: 2}, rows_out=2
    )


def test_rows_that_differ_in_a_column_silver_drops_are_not_duplicates(
    settings: Settings,
) -> None:
    paid_by_card = raw_trip(GOOD_PICKUP, payment_type=1)
    paid_in_cash = raw_trip(GOOD_PICKUP, payment_type=2)

    silver, counts = _clean(settings, [paid_by_card, paid_in_cash])

    assert len(silver) == 2
    assert counts.removed[DUPLICATE] == 0


def test_copies_with_a_missing_value_are_duplicates_too(settings: Settings) -> None:
    trip = raw_trip(GOOD_PICKUP, passenger_count=None)

    _, counts = _clean(settings, [trip, trip])

    assert counts.removed[DUPLICATE] == 1
    assert counts.rows_out == 1


# --- counting and rule order ------------------------------------------------------------


def test_row_that_breaks_several_rules_is_counted_under_the_first(settings: Settings) -> None:
    backwards_negative_unknown = raw_trip(
        "2025-01-15 08:00:00", "2025-01-15 07:00:00", total_amount=-5.0, PULocationID=264
    )

    _, counts = _clean(settings, [backwards_negative_unknown])

    assert counts.removed == _only(DROPOFF_BEFORE_PICKUP)


def test_missing_value_takes_precedence_over_the_month_rule(settings: Settings) -> None:
    stray_without_total = raw_trip("2019-06-01 08:00:00", total_amount=None)

    _, counts = _clean(settings, [stray_without_total])

    assert counts.removed == _only(MISSING_REQUIRED_VALUE)


def test_copy_of_a_bad_row_is_a_duplicate_and_the_original_is_counted_under_its_rule(
    settings: Settings,
) -> None:
    negative = raw_trip(GOOD_PICKUP, total_amount=-5.0)

    _, counts = _clean(settings, [negative, negative])

    assert counts.removed == _only(None) | {DUPLICATE: 1, NEGATIVE_TOTAL_AMOUNT: 1}
    assert counts.rows_out == 0


def test_counts_add_up_for_a_mixed_month(settings: Settings) -> None:
    good = raw_trip(GOOD_PICKUP)
    trips = [good, good, *[bad_trip for _, bad_trip in BAD_TRIPS.values()]]

    silver, counts = _clean(settings, trips)

    assert counts.rows_in == len(trips)
    assert counts.rows_out == len(silver) == 1
    assert counts.rows_in == counts.rows_out + sum(counts.removed.values())
    assert list(counts.removed) == list(RULE_ORDER)


def test_count_removed_follows_the_order_of_the_rules_it_is_given() -> None:
    rejected = {
        "first": pd.Series([True, False, True, False]),
        "second": pd.Series([True, True, False, False]),
    }

    removed, is_kept = count_removed(rejected)

    assert removed == {"first": 2, "second": 1}
    assert list(is_kept) == [False, False, False, True]


def test_counts_that_do_not_add_up_are_refused() -> None:
    with pytest.raises(ValidationError, match="do not add up"):
        CleaningCounts(rows_in=10, removed={DUPLICATE: 1}, rows_out=8)


def test_total_counts_adds_up_months() -> None:
    january = CleaningCounts(rows_in=10, removed=_only(DUPLICATE), rows_out=9)
    february = CleaningCounts(rows_in=5, removed=_only(NEGATIVE_TOTAL_AMOUNT), rows_out=4)

    total = total_counts([january, february])

    assert total.rows_in == 15
    assert total.rows_out == 13
    assert total.removed == _only(None) | {DUPLICATE: 1, NEGATIVE_TOTAL_AMOUNT: 1}
    assert total.rows_removed == 2
    assert total.percent_removed == 13.333


def test_percent_removed_of_an_empty_month_is_zero() -> None:
    assert CleaningCounts(rows_in=0, removed={}, rows_out=0).percent_removed == 0.0


# --- thresholds and input format --------------------------------------------------------


def test_thresholds_come_from_the_settings(settings: Settings) -> None:
    strict = settings.model_copy(
        update={"cleaning": settings.cleaning.model_copy(update={"max_distance_miles": 2.0})}
    )
    trip = raw_trip(GOOD_PICKUP, trip_distance=2.5)

    _, default_counts = clean_trips(raw_trips_frame([trip]), 2025, 1, settings)
    _, strict_counts = clean_trips(raw_trips_frame([trip]), 2025, 1, strict)

    assert default_counts.rows_out == 1
    assert strict_counts.removed == _only(DISTANCE_OUT_OF_RANGE)


# --- values the source did not record ---------------------------------------------------


def test_trip_without_a_recorded_dropoff_time_is_kept_with_the_time_missing(
    settings: Settings,
) -> None:
    same_time = raw_trip(GOOD_PICKUP, GOOD_PICKUP, trip_distance=1.5)

    silver, counts = _clean(settings, [raw_trip(GOOD_PICKUP), same_time])

    assert counts.removed == _only(None)
    assert silver[DROPOFF_DATETIME].isna().tolist() == [False, True]
    assert silver[DURATION_MINUTES].isna().tolist() == [False, True]
    assert silver[PICKUP_DATETIME].iloc[1] == _local(GOOD_PICKUP)
    assert silver[TRIP_DISTANCE].iloc[1] == 1.5
    assert counts.not_recorded == {DROPOFF_DATETIME: 1, TRIP_DISTANCE: 0}


def test_trip_without_a_recorded_distance_is_kept_with_the_distance_missing(
    settings: Settings,
) -> None:
    changed_zone = raw_trip(GOOD_PICKUP, trip_distance=0.0, PULocationID=48, DOLocationID=107)

    silver, counts = _clean(settings, [raw_trip(GOOD_PICKUP), changed_zone])

    assert counts.removed == _only(None)
    assert silver[TRIP_DISTANCE].isna().tolist() == [False, True]
    assert silver[DURATION_MINUTES].iloc[1] == 15.0
    assert counts.not_recorded == {DROPOFF_DATETIME: 0, TRIP_DISTANCE: 1}


def test_unrecorded_value_does_not_protect_a_row_from_the_other_rules(
    settings: Settings,
) -> None:
    refund = raw_trip(GOOD_PICKUP, GOOD_PICKUP, trip_distance=1.5, total_amount=-9.0)
    too_long = raw_trip("2025-01-15 08:00:00", "2025-01-15 15:00:00", trip_distance=0.0)

    silver, counts = _clean(settings, [refund, too_long])

    assert silver.empty
    assert counts.removed == _only(None) | {NEGATIVE_TOTAL_AMOUNT: 1, DURATION_OUT_OF_RANGE: 1}
    assert counts.not_recorded == {DROPOFF_DATETIME: 0, TRIP_DISTANCE: 0}


def test_total_adds_up_the_unrecorded_values_of_the_months() -> None:
    january = CleaningCounts(
        rows_in=3,
        removed=_only(None),
        rows_out=3,
        not_recorded={DROPOFF_DATETIME: 1, TRIP_DISTANCE: 2},
    )
    february = CleaningCounts(
        rows_in=4,
        removed=_only(None),
        rows_out=4,
        not_recorded={DROPOFF_DATETIME: 3, TRIP_DISTANCE: 0},
    )

    total = total_counts([january, february])

    assert total.not_recorded == {DROPOFF_DATETIME: 4, TRIP_DISTANCE: 2}


def test_raw_file_without_the_dropoff_zone_is_refused(settings: Settings) -> None:
    raw = raw_trips_frame([raw_trip(GOOD_PICKUP)]).drop(columns=["DOLocationID"])

    with pytest.raises(RawFormatError, match="DOLocationID"):
        clean_trips(raw, 2025, 1, settings)


def test_raw_file_without_a_needed_column_is_refused(settings: Settings) -> None:
    raw = raw_trips_frame([raw_trip(GOOD_PICKUP)]).drop(columns=["total_amount", "PULocationID"])

    with pytest.raises(RawFormatError, match="PULocationID, total_amount"):
        clean_trips(raw, 2025, 1, settings)


def test_month_bounds_handle_december() -> None:
    assert month_bounds(2025, 12) == (pd.Timestamp("2025-12-01"), pd.Timestamp("2026-01-01"))


# --- daylight saving --------------------------------------------------------------------
# In New York the clocks went forward on 2025-03-09 at 02:00 (to 03:00) and go back on
# 2025-11-02 at 02:00 (to 01:00).


def test_time_the_clocks_skipped_becomes_missing() -> None:
    local = pd.Series(pd.to_datetime(["2025-03-09 01:59:59", "2025-03-09 02:30:00"]))

    localized = localize(local, TIMEZONE)

    assert localized[0] == pd.Timestamp("2025-03-09 01:59:59-05:00")
    assert pd.isna(localized[1])


def test_time_that_happens_twice_is_read_as_the_first_occurrence() -> None:
    local = pd.Series(
        pd.to_datetime(["2025-11-02 00:30:00", "2025-11-02 01:30:00", "2025-11-02 02:30:00"])
    )

    localized = localize(local, TIMEZONE)

    assert list(localized) == [
        pd.Timestamp("2025-11-02 00:30:00-04:00"),
        pd.Timestamp("2025-11-02 01:30:00-04:00"),
        pd.Timestamp("2025-11-02 02:30:00-05:00"),
    ]


def test_pickup_in_the_skipped_hour_is_removed_and_its_neighbours_are_kept(
    settings: Settings,
) -> None:
    before = raw_trip("2025-03-09 01:40:00")
    skipped = raw_trip("2025-03-09 02:30:00", "2025-03-09 03:10:00")
    after = raw_trip("2025-03-09 03:00:00")

    silver, counts = _clean(settings, [before, skipped, after], month=3)

    assert list(silver[PICKUP_DATETIME]) == [
        pd.Timestamp("2025-03-09 01:40:00-05:00"),
        pd.Timestamp("2025-03-09 03:00:00-04:00"),
    ]
    assert counts == CleaningCounts(rows_in=3, removed=_only(NONEXISTENT_LOCAL_TIME), rows_out=2)


def test_dropoff_in_the_skipped_hour_is_removed(settings: Settings) -> None:
    trip = raw_trip("2025-03-09 01:50:00", "2025-03-09 02:05:00")

    _, counts = _clean(settings, [trip], month=3)

    assert counts.removed == _only(NONEXISTENT_LOCAL_TIME)


def test_duration_across_the_spring_change_is_real_elapsed_time(settings: Settings) -> None:
    # The wall clock shows 80 minutes, but only 20 minutes passed.
    trip = raw_trip("2025-03-09 01:50:00", "2025-03-09 03:10:00")

    silver, _ = _clean(settings, [trip], month=3)

    assert list(silver[DURATION_MINUTES]) == [20.0]


def test_pickups_in_the_repeated_hour_are_kept_with_their_neighbours(
    settings: Settings,
) -> None:
    before = raw_trip("2025-11-02 00:30:00")
    repeated = raw_trip("2025-11-02 01:30:00")
    after = raw_trip("2025-11-02 02:30:00")

    silver, counts = _clean(settings, [before, repeated, after], month=11)

    assert list(silver[PICKUP_DATETIME]) == [
        pd.Timestamp("2025-11-02 00:30:00-04:00"),
        pd.Timestamp("2025-11-02 01:30:00-04:00"),
        pd.Timestamp("2025-11-02 02:30:00-05:00"),
    ]
    assert counts == CleaningCounts(rows_in=3, removed=_only(None), rows_out=3)
