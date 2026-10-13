"""Cleaning rules that turn one month of raw TLC trips into silver trips.

Everything in this module is pure: a dataframe goes in, a dataframe and its counts come
out. Reading and writing files is the job of ``silver.py``.

**Why pandas.** One month is at most about 4.6 million rows, which fits in memory, and
each rule becomes one readable line that marks the rows it rejects. pandas also has the
one feature this stage cannot do without: ``tz_localize`` lets the code say exactly what
happens to a local time that does not exist or exists twice (see ``localize``).

**The rules and their order.** Every rule is evaluated on every row, then each removed
row is counted under the *first* rule in ``RULE_ORDER`` that rejects it. A row is
therefore counted once, and ``rows_in == rows_out + sum(removed)`` always holds
(``CleaningCounts`` refuses to exist otherwise). The order goes from "this is not a
usable record at all" to "the record is fine but its values are impossible":

1. ``duplicate``                    identical to an earlier row in *every* raw column
2. ``missing_required_value``       no pickup time, pickup zone, dropoff time, distance
                                    or total amount (a missing passenger count is
                                    allowed)
3. ``pickup_outside_month``         pickup is not in the month the file is named after
4. ``nonexistent_local_time``       pickup or dropoff is a time the clocks skipped
5. ``dropoff_before_pickup``
6. ``duration_out_of_range``        shorter than ``min_duration_minutes`` or longer
                                    than ``max_duration_hours``
7. ``distance_out_of_range``        zero, negative or above ``max_distance_miles``
8. ``negative_total_amount``        the charge was taken back (see below)
9. ``passenger_count_out_of_range`` recorded, but not ``min_passengers..max_passengers``
10. ``unknown_pickup_zone``         zone ID below 1 or above ``max_zone_id``

All thresholds come from ``settings.cleaning``.

**Why the total amount and not the fare.** A refund or a disputed charge appears in
the raw file as a second row for the same trip, with the fare and the total amount both
negative. That row is not a pickup, and counting it would count the trip twice. But
about 880,000 rows of the first half of 2025 have a negative ``fare_amount`` and a
*positive* ``total_amount``: money was charged, distance and duration are ordinary,
and there is no second row for the same trip. They are real pickups with a fare field
that cannot be trusted, and their share doubles from January to June, so removing them
would add a falling trend to the demand that does not exist. The rule therefore looks
at the total amount, and silver does not keep the fare.

**Rows of a neighbouring month.** A raw file contains a few trips that started just
before or after its month (the January file has pickups from the evening of
31 December). They are removed by rule 3, not moved to the other month's file, so that
every silver file is built from exactly one raw file. The cost is a handful of real
trips per month, visible in the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from demand.config import Settings
from demand.schemas import (
    DROPOFF_DATETIME,
    DURATION_MINUTES,
    PASSENGER_COUNT,
    PICKUP_DATETIME,
    PICKUP_ZONE_ID,
    TOTAL_AMOUNT,
    TRIP_DISTANCE,
)

# Column names of the raw TLC files. Only this module reads them.
RAW_PICKUP_DATETIME = "tpep_pickup_datetime"
RAW_DROPOFF_DATETIME = "tpep_dropoff_datetime"
RAW_PICKUP_ZONE_ID = "PULocationID"
RAW_TRIP_DISTANCE = "trip_distance"
RAW_PASSENGER_COUNT = "passenger_count"
RAW_TOTAL_AMOUNT = "total_amount"

# Raw column -> silver column, for the columns that silver keeps.
RAW_TO_SILVER = {
    RAW_PICKUP_DATETIME: PICKUP_DATETIME,
    RAW_DROPOFF_DATETIME: DROPOFF_DATETIME,
    RAW_PICKUP_ZONE_ID: PICKUP_ZONE_ID,
    RAW_TRIP_DISTANCE: TRIP_DISTANCE,
    RAW_PASSENGER_COUNT: PASSENGER_COUNT,
    RAW_TOTAL_AMOUNT: TOTAL_AMOUNT,
}

# Rule names, as they appear in the cleaning report.
DUPLICATE = "duplicate"
MISSING_REQUIRED_VALUE = "missing_required_value"
PICKUP_OUTSIDE_MONTH = "pickup_outside_month"
NONEXISTENT_LOCAL_TIME = "nonexistent_local_time"
DROPOFF_BEFORE_PICKUP = "dropoff_before_pickup"
DURATION_OUT_OF_RANGE = "duration_out_of_range"
DISTANCE_OUT_OF_RANGE = "distance_out_of_range"
NEGATIVE_TOTAL_AMOUNT = "negative_total_amount"
PASSENGER_COUNT_OUT_OF_RANGE = "passenger_count_out_of_range"
UNKNOWN_PICKUP_ZONE = "unknown_pickup_zone"

RULE_ORDER = (
    DUPLICATE,
    MISSING_REQUIRED_VALUE,
    PICKUP_OUTSIDE_MONTH,
    NONEXISTENT_LOCAL_TIME,
    DROPOFF_BEFORE_PICKUP,
    DURATION_OUT_OF_RANGE,
    DISTANCE_OUT_OF_RANGE,
    NEGATIVE_TOTAL_AMOUNT,
    PASSENGER_COUNT_OUT_OF_RANGE,
    UNKNOWN_PICKUP_ZONE,
)

_SECONDS_PER_MINUTE = 60
_MINUTES_PER_HOUR = 60
_FIRST_ZONE_ID = 1


class RawFormatError(ValueError):
    """A raw file does not have the structure this stage expects."""


class CleaningCounts(BaseModel):
    """Rows in, rows removed per rule and rows out, for one month or for the total."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows_in: int = Field(ge=0)
    removed: dict[str, int]
    """Rule name -> rows removed by it, in ``RULE_ORDER``."""
    rows_out: int = Field(ge=0)

    @model_validator(mode="after")
    def _check_counts_add_up(self) -> CleaningCounts:
        if self.rows_in != self.rows_out + self.rows_removed:
            raise ValueError(
                f"cleaning counts do not add up: {self.rows_in} rows in, "
                f"{self.rows_removed} removed, {self.rows_out} out"
            )
        return self

    @property
    def rows_removed(self) -> int:
        """Rows removed by all rules together."""
        return sum(self.removed.values())

    @property
    def percent_removed(self) -> float:
        """Share of the incoming rows that was removed, in percent."""
        return round(100 * self.rows_removed / self.rows_in, 3) if self.rows_in else 0.0


def month_bounds(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First moment of a month and first moment of the next one, without a timezone."""
    start = pd.Timestamp(year=year, month=month, day=1)
    return start, start + pd.offsets.MonthBegin(1)


def find_exact_duplicates(raw: pd.DataFrame) -> pd.Series:
    """Mark every row that repeats an earlier row in all columns; the first one stays.

    Comparing all twenty raw columns of millions of rows is slow and needs a lot of
    memory. A row can only be a full copy of another if it also equals it in the few
    columns silver keeps, so those are compared first, and the full comparison is run
    only on the rows that passed (a few hundred per month).
    """
    candidates = raw.duplicated(subset=list(RAW_TO_SILVER), keep=False)
    is_duplicate = pd.Series(False, index=raw.index)
    is_duplicate[candidates] = raw[candidates].duplicated(keep="first")
    return is_duplicate


def localize(local_time: pd.Series, timezone: str) -> pd.Series:
    """Attach ``timezone`` to timestamps that were recorded as local wall-clock time.

    Twice a year the wall clock is not a faithful record of time:

    * **Spring forward.** The clocks jump from 02:00 to 03:00, so 02:30 on that night
      never happened. A trip with such a time was stamped by a wrong clock and there is
      no way to tell what the right time was. The result is ``NaT`` (missing), and the
      ``nonexistent_local_time`` rule removes the row.
    * **Autumn, fall back.** The clocks go from 02:00 back to 01:00, so 01:30 happens
      twice and the raw value does not say which one is meant. These are real trips,
      and removing them would empty an hour of real demand, so they are kept and read
      as the *first* 01:30 (daylight time). The price: on that one night the first
      01:00 hour holds the pickups of both, the second holds none, and a trip that
      spans the change can get a duration that is wrong by an hour (and then fail a
      duration rule it should have passed).
    """
    is_first_occurrence = np.ones(len(local_time), dtype=bool)
    localized: pd.Series = local_time.dt.tz_localize(
        timezone, nonexistent="NaT", ambiguous=is_first_occurrence
    )
    return localized


def duration_minutes(pickup: pd.Series, dropoff: pd.Series) -> pd.Series:
    """Minutes from pickup to dropoff, measured in real elapsed time."""
    minutes: pd.Series = (dropoff - pickup).dt.total_seconds() / _SECONDS_PER_MINUTE
    return minutes


def to_silver_columns(raw: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Keep the silver columns, rename them, localise the times and add the duration.

    No row is removed here. A timestamp that cannot be localised becomes ``NaT``.
    """
    trips = raw[list(RAW_TO_SILVER)].rename(columns=RAW_TO_SILVER)
    trips[PICKUP_DATETIME] = localize(trips[PICKUP_DATETIME], timezone)
    trips[DROPOFF_DATETIME] = localize(trips[DROPOFF_DATETIME], timezone)
    trips[DURATION_MINUTES] = duration_minutes(trips[PICKUP_DATETIME], trips[DROPOFF_DATETIME])
    return trips


def find_rejected_rows(
    raw: pd.DataFrame, trips: pd.DataFrame, year: int, month: int, settings: Settings
) -> dict[str, pd.Series]:
    """For every rule, mark the rows it rejects. Keys are in ``RULE_ORDER``.

    Each rule looks at all rows, also at rows an earlier rule has already rejected;
    ``count_removed`` then decides which rule a row is counted under.

    Args:
        raw: The raw trips of one file.
        trips: The same rows after ``to_silver_columns``.
        year: Year the file is named after.
        month: Month the file is named after.
        settings: Project settings; the thresholds come from ``settings.cleaning``.
    """
    cleaning = settings.cleaning
    month_start, next_month_start = month_bounds(year, month)
    local_pickup = raw[RAW_PICKUP_DATETIME]
    local_dropoff = raw[RAW_DROPOFF_DATETIME]
    pickup = trips[PICKUP_DATETIME]
    dropoff = trips[DROPOFF_DATETIME]
    duration = trips[DURATION_MINUTES]
    distance = trips[TRIP_DISTANCE]
    passengers = trips[PASSENGER_COUNT]
    zone = trips[PICKUP_ZONE_ID]

    required = [RAW_PICKUP_DATETIME, RAW_PICKUP_ZONE_ID, RAW_DROPOFF_DATETIME]
    required += [RAW_TRIP_DISTANCE, RAW_TOTAL_AMOUNT]
    return {
        DUPLICATE: find_exact_duplicates(raw),
        MISSING_REQUIRED_VALUE: raw[required].isna().any(axis=1),
        # Compared as local wall-clock time: the month of a file is a local month.
        PICKUP_OUTSIDE_MONTH: (local_pickup < month_start) | (local_pickup >= next_month_start),
        # The raw value is there, but ``localize`` could not turn it into a real time.
        NONEXISTENT_LOCAL_TIME: (local_pickup.notna() & pickup.isna())
        | (local_dropoff.notna() & dropoff.isna()),
        DROPOFF_BEFORE_PICKUP: dropoff < pickup,
        DURATION_OUT_OF_RANGE: (duration < cleaning.min_duration_minutes)
        | (duration > cleaning.max_duration_hours * _MINUTES_PER_HOUR),
        DISTANCE_OUT_OF_RANGE: (distance <= 0) | (distance > cleaning.max_distance_miles),
        NEGATIVE_TOTAL_AMOUNT: trips[TOTAL_AMOUNT] < 0,
        # A missing count compares as False on both sides, so it is not rejected.
        PASSENGER_COUNT_OUT_OF_RANGE: (passengers < cleaning.min_passengers)
        | (passengers > cleaning.max_passengers),
        UNKNOWN_PICKUP_ZONE: (zone < _FIRST_ZONE_ID) | (zone > cleaning.max_zone_id),
    }


def count_removed(rejected: dict[str, pd.Series]) -> tuple[dict[str, int], pd.Series]:
    """Count each rejected row under the first rule that rejects it.

    Args:
        rejected: Rule name -> rows rejected by that rule, in the order of precedence.

    Returns:
        Rows removed per rule, and a mask of the rows that no rule rejects.
    """
    removed: dict[str, int] = {}
    first_mask = next(iter(rejected.values()))
    already_removed = pd.Series(False, index=first_mask.index)
    for rule, is_rejected in rejected.items():
        removed[rule] = int((is_rejected & ~already_removed).sum())
        already_removed = already_removed | is_rejected
    return removed, ~already_removed


def clean_trips(
    raw: pd.DataFrame, year: int, month: int, settings: Settings
) -> tuple[pd.DataFrame, CleaningCounts]:
    """Clean the raw trips of one month.

    Args:
        raw: Content of one raw TLC file, all columns, raw column names.
        year: Year the file is named after.
        month: Month the file is named after.
        settings: Project settings (thresholds and timezone).

    Returns:
        The trips that passed every rule, with the columns and types of
        ``silver_trips_schema``, in their original order; and the counts for the report.

    Raises:
        RawFormatError: If ``raw`` lacks a column that silver is built from.
    """
    missing_columns = [column for column in RAW_TO_SILVER if column not in raw.columns]
    if missing_columns:
        raise RawFormatError(f"raw trips have no column {', '.join(missing_columns)}")

    trips = to_silver_columns(raw, settings.project.timezone)
    rejected = find_rejected_rows(raw, trips, year, month, settings)
    removed, is_kept = count_removed(rejected)

    silver = trips[is_kept].reset_index(drop=True)
    silver = silver.astype(
        {
            PICKUP_ZONE_ID: "int64",
            TRIP_DISTANCE: "float64",
            PASSENGER_COUNT: "float64",
            TOTAL_AMOUNT: "float64",
        }
    )
    counts = CleaningCounts(rows_in=len(raw), removed=removed, rows_out=len(silver))
    return silver, counts


def total_counts(counts: list[CleaningCounts]) -> CleaningCounts:
    """Add up the counts of several months."""
    return CleaningCounts(
        rows_in=sum(month.rows_in for month in counts),
        removed={rule: sum(month.removed[rule] for month in counts) for rule in RULE_ORDER},
        rows_out=sum(month.rows_out for month in counts),
    )
