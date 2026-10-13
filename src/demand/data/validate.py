"""Quality gates: check the files of a layer before the next stage reads them.

A gate answers one question: *is this layer fit to build on?* It reads the files from
disk, because the files are what the next stage will read. The stage that wrote them
checked its tables in memory, but a file can be replaced, damaged or left over from an
older config afterwards, and the gate is the place where that is noticed.

Layout of the reports (``settings.paths.reports_dir``)::

    reports/
      silver_validation_report.json    result of ``python -m demand validate silver``
      gold_validation_report.json      result of ``python -m demand validate gold``

A report lists every check with its name, whether it passed, and a short detail. It is
written whether the gate passes or fails. It contains no time stamp, so the same files
and the same config always give the same report, byte for byte.

**Two kinds of checks.** Every table is first checked against its Pandera schema from
``schemas.py``: column names, types, value ranges, missing values, unique keys. The
schema is run in *lazy* mode, which collects every failure instead of stopping at the
first, so one run shows everything that is wrong with a table. Then come the checks a
schema cannot express, because they need the config, a second column, or a second file:

Silver, for the trips of every configured month:

* ``pickups_in_month``: every pickup lies in the month the file is named after.
* ``duration_matches_times``: the stored duration is the time from pickup to dropoff,
  and it is missing exactly where the dropoff time is missing.
* ``counts_match_report``: the file has the number of rows, and of values that were not
  recorded, that the cleaning report states for this month.

Silver, weather and cleaning report:

* ``weather.covers_period``: one row for every hour of the configured period, no other.
* ``weather.hours_match_report``: the report states the same number of hours.
* ``cleaning_report.months``: the report describes exactly the configured months.
* ``cleaning_report.total``: its total is the sum of its months.

Gold:

* ``demand.complete_grid``: one row for every frozen zone and every hour of the period.
* ``demand.weather_same_for_all_zones``: an hour has one weather, whatever the zone.
* ``features.columns``: the keys, the target, the outlier flag and exactly the model
  inputs of ``feature_columns``.
* ``features.rows``: the grid of the demand table without the warm-up hours.
* ``features.matches_demand``: target, outlier flag and weather are those of the demand
  table for the same zone and hour.
* ``features.lags_match_demand``: every lag is the pickup count of the same zone that
  many real hours earlier. This is the check that would notice a lag that looks at the
  wrong hour, which is the most damaging mistake a forecasting table can contain.

**A check that could not run has not passed.** The checks of the second kind rely on
what the schema promises (the columns exist and have the right types). If a file is
missing or its table fails the schema, they are not run and are listed as failed with
the detail ``not checked``. The first failed check of a table is the one to read.

**Memory.** The silver trips are 23 million rows. They are read and checked one month
at a time, like the silver stage wrote them, so the gate needs memory for one month
(about 0.8 GB at its peak) and not for six.

A gate changes no data. It writes its report, and when a check failed it raises
``QualityGateError``, which the command line turns into one line and exit code 1.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd
import pandera.errors
import pandera.pandas as pa
import pyarrow
from pydantic import BaseModel, ConfigDict, ValidationError

from demand.config import Settings
from demand.data.aggregate import grid_index
from demand.data.clean import NOT_RECORDED_COLUMNS, CleaningCounts, duration_minutes, month_bounds
from demand.data.clean_weather import local_hours
from demand.data.features import warm_up_hours
from demand.data.gold import check_zones_selected, gold_demand_path, gold_features_path
from demand.data.silver import (
    CleaningReport,
    cleaning_report_path,
    month_key,
    report_to_json,
    silver_trips_path,
    silver_weather_path,
    write_text,
)
from demand.errors import QualityGateError
from demand.logs import get_logger
from demand.schemas import (
    DROPOFF_DATETIME,
    DURATION_MINUTES,
    HOUR_START,
    IS_OUTLIER,
    PICKUP_DATETIME,
    TARGET,
    ZONE_ID,
    feature_columns,
    gold_demand_schema,
    gold_features_schema,
    lag_column,
    silver_trips_schema,
    silver_weather_schema,
)

log = get_logger(__name__)

SILVER = "silver"
GOLD = "gold"
VALIDATION_REPORT_SUFFIX = "_validation_report.json"

NOT_CHECKED = "not checked: the file is missing or the table did not pass its schema"
_MAX_EXAMPLES = 3
_MAX_NAMES_IN_MESSAGE = 3

KEYS = [ZONE_ID, HOUR_START]


class CheckResult(BaseModel):
    """The outcome of one check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Table and check, e.g. ``trips_2025-01.schema``."""
    passed: bool
    detail: str
    """What was found: for a failed check, how many rows and a few example values."""


class GateReport(BaseModel):
    """The result of one quality gate; saved as JSON next to the other reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    layer: str
    passed: bool
    """True only if every check passed."""
    checks: list[CheckResult]


# A function that looks at one thing and says what is wrong with it; ``None`` means
# nothing is.
type Finder[Subject] = Callable[[Subject], str | None]


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def validation_report_path(reports_dir: Path, layer: str) -> Path:
    """Where the report of one gate is stored."""
    return reports_dir / f"{layer}{VALIDATION_REPORT_SUFFIX}"


def to_result(name: str, problem: str | None, passed_detail: str = "ok") -> CheckResult:
    """Turn what a finder returned into a check result."""
    if problem is None:
        return CheckResult(name=name, passed=True, detail=passed_detail)
    return CheckResult(name=name, passed=False, detail=problem)


def run_checks[Subject](
    subject: Subject | None, checks: dict[str, Finder[Subject]]
) -> list[CheckResult]:
    """Run every check on ``subject``, or list them all as not checked if there is none."""
    if subject is None:
        return [to_result(name, NOT_CHECKED) for name in checks]
    return [to_result(name, find(subject)) for name, find in checks.items()]


def examples(values: Iterable[object]) -> str:
    """The first few different values as text, so that a detail stays short.

    A key of several parts, such as a zone and an hour, is written as ``161 / 2025-...``.
    """
    texts = [
        " / ".join(str(part) for part in value) if isinstance(value, tuple) else str(value)
        for value in values
    ]
    return ", ".join(list(dict.fromkeys(texts))[:_MAX_EXAMPLES])


def counted(number: int, noun: str) -> str:
    """A number with its noun, e.g. ``1 row`` or ``3 rows``."""
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def find_schema_problems(table: pd.DataFrame, schema: pa.DataFrameSchema) -> str | None:
    """Check a table against its schema and sum up everything that fails.

    ``lazy=True`` makes Pandera run every check and report all failures together. The
    failures are grouped by column and check, each with its count and a few of the
    failing values.
    """
    try:
        schema.validate(table, lazy=True)
    except pandera.errors.SchemaErrors as error:
        cases = error.failure_cases
        # A failure of the table as a whole (a missing column) has no column name.
        where = cases["column"].fillna("table").astype(str) + ": " + cases["check"].astype(str)
        parts = [
            f"{place} ({len(failed)} failing, e.g. {examples(failed)})"
            for place, failed in cases["failure_case"].groupby(where, sort=False)
        ]
        return "; ".join(parts)
    return None


def find_key_differences(expected: pd.Index, actual: pd.Index, noun: str) -> str | None:
    """Compare the keys a table has with the keys it must have.

    ``noun`` says what one key is, for the detail: ``hour``, ``row``, ``column``.

    A key that appears twice is not looked for here: every schema already demands unique
    keys, and these checks only run on a table that passed its schema.
    """
    missing = expected.difference(actual)
    unexpected = actual.difference(expected)
    parts = []
    if len(missing):
        parts.append(f"{counted(len(missing), noun)} missing, e.g. {examples(missing)}")
    if len(unexpected):
        parts.append(f"{counted(len(unexpected), noun)} not expected, e.g. {examples(unexpected)}")
    return "; ".join(parts) or None


def find_pickups_outside_month(
    trips: pd.DataFrame, year: int, month: int, timezone: str
) -> str | None:
    """Pickups that are not in the month their file is named after."""
    start, end = (bound.tz_localize(timezone) for bound in month_bounds(year, month))
    pickup = trips[PICKUP_DATETIME]
    outside = pickup[(pickup < start) | (pickup >= end)]
    if outside.empty:
        return None
    return (
        f"{counted(len(outside), 'pickup')} outside {month_key(year, month)}, "
        f"e.g. {examples(outside)}"
    )


def find_wrong_durations(trips: pd.DataFrame) -> str | None:
    """Rows whose duration is not the time from pickup to dropoff.

    The duration is calculated again with the function the silver stage used, so the two
    numbers must be equal to the last digit. Where the dropoff time is missing both are
    missing, and that counts as equal.
    """
    stored = trips[DURATION_MINUTES]
    expected = duration_minutes(trips[PICKUP_DATETIME], trips[DROPOFF_DATETIME])
    wrong = ~((stored == expected) | (stored.isna() & expected.isna()))
    if not wrong.any():
        return None
    return (
        f"{counted(int(wrong.sum()), 'row')} with a duration that is not dropoff minus pickup, "
        f"e.g. at row {examples(trips.index[wrong])}"
    )


def find_count_differences(trips: pd.DataFrame, counts: CleaningCounts | None) -> str | None:
    """Compare a silver trip file with the counts the cleaning report gives for it."""
    if counts is None:
        return "the cleaning report has no readable counts for this month"
    in_file = {"rows_out": len(trips)}
    in_report = {"rows_out": counts.rows_out}
    for column in NOT_RECORDED_COLUMNS:
        in_file[f"not_recorded.{column}"] = int(trips[column].isna().sum())
        in_report[f"not_recorded.{column}"] = counts.not_recorded.get(column, 0)
    different = [
        f"{name}: the file has {in_file[name]}, the report says {in_report[name]}"
        for name in in_file
        if in_file[name] != in_report[name]
    ]
    return "; ".join(different) or None


def find_weather_hours_difference(
    weather: pd.DataFrame, report: CleaningReport | None
) -> str | None:
    """Compare the rows of the silver weather with the hours the cleaning report states."""
    if report is None:
        return "the cleaning report cannot be read"
    if len(weather) == report.weather.hours:
        return None
    return f"the file has {len(weather)} rows, the report says {report.weather.hours} hours"


def find_month_differences(report: CleaningReport, settings: Settings) -> str | None:
    """Compare the months in the cleaning report with the configured months."""
    configured = pd.Index([month_key(year, month) for year, month in settings.data.months])
    return find_key_differences(configured, pd.Index(list(report.months)), "month")


def find_total_differences(report: CleaningReport) -> str | None:
    """Compare the total of the cleaning report with the sum of its months."""
    months = report.months.values()
    sums = {
        "rows_in": sum(month.rows_in for month in months),
        "rows_out": sum(month.rows_out for month in months),
    }
    stated = {"rows_in": report.total.rows_in, "rows_out": report.total.rows_out}
    different = [
        f"{name}: the months add up to {sums[name]}, the total says {stated[name]}"
        for name in sums
        if sums[name] != stated[name]
    ]
    return "; ".join(different) or None


def find_weather_differences_between_zones(
    demand: pd.DataFrame, variables: list[str]
) -> str | None:
    """Hours for which two zones have different weather. There is one weather per hour."""
    values_per_hour = demand.groupby(HOUR_START)[variables].nunique()
    differs = values_per_hour.gt(1).any(axis=1)
    if not differs.any():
        return None
    return (
        f"{counted(int(differs.sum()), 'hour')} with different weather in different zones, "
        f"e.g. {examples(values_per_hour.index[differs])}"
    )


def find_column_differences(features: pd.DataFrame, settings: Settings) -> str | None:
    """Compare the columns of the feature table with what the model will ask for."""
    # The zone is a key and a model input; ``unique`` keeps it once.
    expected = pd.Index([*KEYS, TARGET, IS_OUTLIER, *feature_columns(settings)]).unique()
    return find_key_differences(expected, features.columns, "column")


def find_differences_from_demand(
    features: pd.DataFrame, demand: pd.DataFrame, columns: list[str]
) -> str | None:
    """Rows of the feature table whose ``columns`` differ from the demand table.

    A row whose zone and hour the demand table does not have counts as different.
    """
    both = features[[*KEYS, *columns]].merge(
        demand[[*KEYS, *columns]], on=KEYS, how="left", suffixes=("", "_demand")
    )
    differs = pd.Series(False, index=both.index)
    for column in columns:
        differs = differs | (both[column] != both[f"{column}_demand"])
    if not differs.any():
        return None
    keys = both.loc[differs, KEYS].itertuples(index=False, name=None)
    return (
        f"{counted(int(differs.sum()), 'row')} different from the demand table in "
        f"{', '.join(columns)}, e.g. {examples(keys)}"
    )


def find_wrong_lags(features: pd.DataFrame, demand: pd.DataFrame, lags: list[int]) -> str | None:
    """Lag values that are not the pickups of the same zone that many hours earlier.

    Subtracting a ``Timedelta`` from a timezone-aware time moves by real, elapsed hours,
    also across a clock change. That is how ``features.py`` defines a lag.
    """
    pickups = demand.set_index(KEYS)[TARGET]
    parts = []
    for lag in lags:
        earlier = pd.MultiIndex.from_arrays(
            [features[ZONE_ID], features[HOUR_START] - pd.Timedelta(hours=lag)]
        )
        # An earlier hour the demand table lacks gives NaN, which equals nothing.
        expected = pickups.reindex(earlier).to_numpy(dtype="float64")
        wrong = features[lag_column(lag)].to_numpy() != expected
        if wrong.any():
            keys = features.loc[wrong, KEYS].itertuples(index=False, name=None)
            parts.append(
                f"{lag_column(lag)}: {counted(int(wrong.sum()), 'row')} not equal to the "
                f"pickups {lag} h earlier, e.g. {examples(keys)}"
            )
    return "; ".join(parts) or None


def failure_message(report: GateReport, report_path: Path) -> str:
    """One line for the terminal: which gate, how many checks, which ones, and where to look.

    Only the checks that found something are named. The ones that could not run are a
    consequence of those, so they are counted and not named.
    """
    not_passed = [check for check in report.checks if not check.passed]
    failed = [check.name for check in not_passed if check.detail != NOT_CHECKED]
    not_checked = len(not_passed) - len(failed)
    names = ", ".join(failed[:_MAX_NAMES_IN_MESSAGE])
    if len(failed) > _MAX_NAMES_IN_MESSAGE:
        names += f" and {len(failed) - _MAX_NAMES_IN_MESSAGE} more"
    could_not_run = f", and {not_checked} more could not be checked" if not_checked else ""
    return (
        f"{report.layer} quality gate failed: {len(failed)} of {len(report.checks)} checks "
        f"failed ({names}){could_not_run}. The details are in {report_path.as_posix()}. "
        f"Fix the cause, then run `python -m demand {report.layer}` and the gate again."
    )


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def load_table(
    name: str, path: Path, schema: pa.DataFrameSchema, stage: str
) -> tuple[pd.DataFrame | None, list[CheckResult]]:
    """Read one Parquet file and check it against its schema.

    Args:
        name: Name of the table in the report, e.g. ``trips_2025-01``.
        path: The file to read.
        schema: The Pandera schema of the table.
        stage: The command that writes the file, for the advice in the detail.

    Returns:
        The table, or ``None`` if the file is missing, cannot be read or fails the
        schema; and the results of the two checks ``<name>.file`` and ``<name>.schema``.
    """
    file_check, schema_check = f"{name}.file", f"{name}.schema"
    advice = f"Run `python -m demand {stage}` to write it again."
    if not path.is_file():
        problem = f"{path.as_posix()} is missing. {advice}"
        return None, [to_result(file_check, problem), to_result(schema_check, NOT_CHECKED)]
    try:
        table = pd.read_parquet(path)
    except (pyarrow.ArrowInvalid, OSError) as error:
        problem = f"{path.as_posix()} cannot be read as Parquet ({error}). {advice}"
        return None, [to_result(file_check, problem), to_result(schema_check, NOT_CHECKED)]
    # pyarrow holds on to the memory it needed while reading. Handing it back lowers the
    # peak of the silver gate from about 1.1 GB to about 0.8 GB (measured on six months).
    pyarrow.default_memory_pool().release_unused()

    schema_problem = find_schema_problems(table, schema)
    results = [
        to_result(file_check, None, f"{len(table)} rows"),
        to_result(schema_check, schema_problem),
    ]
    return (table if schema_problem is None else None), results


def load_cleaning_report(reports_dir: Path) -> tuple[CleaningReport | None, CheckResult]:
    """Read the cleaning report; a missing or unreadable one is a failed check."""
    name = "cleaning_report.file"
    path = cleaning_report_path(reports_dir)
    advice = "Run `python -m demand silver` to write it again."
    if not path.is_file():
        return None, to_result(name, f"{path.as_posix()} is missing. {advice}")
    try:
        report = CleaningReport.model_validate_json(path.read_bytes())
    except ValidationError as error:
        problem = (
            f"{path.as_posix()} is not a cleaning report "
            f"({error.error_count()} errors, first: {error.errors()[0]['msg']}). {advice}"
        )
        return None, to_result(name, problem)
    return report, to_result(name, None)


def check_silver_trips(
    settings: Settings, year: int, month: int, counts: CleaningCounts | None
) -> list[CheckResult]:
    """Run every check of the silver trips of one month.

    Args:
        settings: Project settings.
        year: Year of the file.
        month: Month of the file.
        counts: What the cleaning report says about this month, or ``None`` if the
            report cannot be read or does not have the month.
    """
    name = f"trips_{month_key(year, month)}"
    path = silver_trips_path(settings.paths.silver, year, month)
    trips, results = load_table(name, path, silver_trips_schema(settings), SILVER)
    results += run_checks(
        trips,
        {
            f"{name}.pickups_in_month": lambda trips: find_pickups_outside_month(
                trips, year, month, settings.project.timezone
            ),
            f"{name}.duration_matches_times": find_wrong_durations,
            f"{name}.counts_match_report": lambda trips: find_count_differences(trips, counts),
        },
    )
    return results


def check_silver(settings: Settings) -> list[CheckResult]:
    """Run every check of the silver layer: the report, the weather, then month by month."""
    data = settings.data
    timezone = settings.project.timezone
    silver_dir = settings.paths.silver

    report, report_file = load_cleaning_report(settings.paths.reports_dir)
    results = [report_file]
    results += run_checks(
        report,
        {
            "cleaning_report.months": lambda report: find_month_differences(report, settings),
            "cleaning_report.total": find_total_differences,
        },
    )

    weather, weather_results = load_table(
        "weather", silver_weather_path(silver_dir), silver_weather_schema(settings), SILVER
    )
    hours = local_hours(data.start_date, data.end_date, timezone)
    results += weather_results
    results += run_checks(
        weather,
        {
            "weather.covers_period": lambda weather: find_key_differences(
                hours, pd.Index(weather[HOUR_START]), "hour"
            ),
            "weather.hours_match_report": lambda weather: find_weather_hours_difference(
                weather, report
            ),
        },
    )

    for year, month in data.months:
        # One call per month: its trips are in memory only until the call returns.
        counts = report.months.get(month_key(year, month)) if report else None
        results += check_silver_trips(settings, year, month, counts)
    return results


def check_gold(settings: Settings) -> list[CheckResult]:
    """Run every check of the gold layer: the demand table, then the feature table.

    Raises:
        ZonesNotSelectedError: If ``data.zones`` is empty. Without a zone list there is
            no grid to compare the tables with.
    """
    check_zones_selected(settings)
    data = settings.data
    gold_dir = settings.paths.gold
    variables = data.weather.variables
    hours = local_hours(data.start_date, data.end_date, settings.project.timezone)

    demand, results = load_table(
        "demand", gold_demand_path(gold_dir), gold_demand_schema(settings), GOLD
    )
    results += run_checks(
        demand,
        {
            "demand.complete_grid": lambda demand: find_key_differences(
                grid_index(data.zones, hours),
                pd.MultiIndex.from_frame(demand[KEYS]),
                "row",
            ),
            "demand.weather_same_for_all_zones": lambda demand: (
                find_weather_differences_between_zones(demand, variables)
            ),
        },
    )

    features, features_results = load_table(
        "features", gold_features_path(gold_dir), gold_features_schema(settings), GOLD
    )
    results += features_results
    # The first hours of every zone have no complete history and are not in the table.
    hours_with_history = hours[warm_up_hours(settings) :]
    results += run_checks(
        features,
        {
            "features.columns": lambda features: find_column_differences(features, settings),
            "features.rows": lambda features: find_key_differences(
                grid_index(data.zones, hours_with_history),
                pd.MultiIndex.from_frame(features[KEYS]),
                "row",
            ),
        },
    )
    # These two compare the tables with each other, so they need both.
    both = (features, demand) if features is not None and demand is not None else None
    results += run_checks(
        both,
        {
            "features.matches_demand": lambda tables: find_differences_from_demand(
                tables[0], tables[1], [TARGET, IS_OUTLIER, *variables]
            ),
            "features.lags_match_demand": lambda tables: find_wrong_lags(
                tables[0], tables[1], settings.features.lags
            ),
        },
    )
    return results


GATES: dict[str, Callable[[Settings], list[CheckResult]]] = {
    SILVER: check_silver,
    GOLD: check_gold,
}


def validate_layer(settings: Settings, layer: str) -> GateReport:
    """Run the quality gate of one layer and write its report.

    Safe to run again: nothing but the report is written, and it comes out identical
    for the same files and config.

    Args:
        settings: Project settings; the folders, the period and the zones come from here.
        layer: ``"silver"`` or ``"gold"``.

    Returns:
        The report, if every check passed.

    Raises:
        ValueError: If ``layer`` has no gate.
        ZonesNotSelectedError: For the gold layer, if ``data.zones`` is empty.
        QualityGateError: If a check failed. The report is written before this is
            raised.
    """
    if layer not in GATES:
        raise ValueError(f"no quality gate for layer '{layer}'; choose from {sorted(GATES)}")
    started = time.monotonic()
    checks = GATES[layer](settings)
    report = GateReport(layer=layer, passed=all(check.passed for check in checks), checks=checks)
    report_path = validation_report_path(settings.paths.reports_dir, layer)
    write_text(report_to_json(report), report_path)

    if not report.passed:
        raise QualityGateError(failure_message(report, report_path))
    log.info(
        "quality gate passed",
        extra={
            "layer": layer,
            "checks": len(checks),
            "report": str(report_path),
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return report
