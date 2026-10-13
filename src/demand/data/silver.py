"""Build the silver layer: cleaned trips and weather, checked against their schemas.

Layout of the silver folder (``settings.paths.silver``) and of the report::

    silver/
      trips/trips_2025-01.parquet       one file per configured month
      weather/weather_hourly.parquet    one row per local hour of the period
    reports/
      silver_cleaning_report.json       rows in, removed per rule, rows out

The path helpers below (``silver_trips_path``, ``silver_weather_path``,
``cleaning_report_path``) are the only place where this layout is written down; later
stages import them instead of building paths themselves.

This module only moves data between disk and the pure cleaning functions:

* ``clean.py`` holds the trip rules, their order and the daylight-saving decisions.
* ``clean_weather.py`` holds the weather conversion and the gap handling.

How the stage runs:

1. Every raw file the config asks for must exist, otherwise nothing is started.
2. Trips are cleaned one month at a time, so only one month is in memory.
3. Each table is validated with its Pandera schema *before* it is written. A table that
   fails stops the stage and is not written.
4. Each file is written under a temporary name and then renamed, so a file under its
   final name is always complete.

The stage has no state of its own: the same raw files and the same config always give
the same silver files and the same report, byte for byte.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.clean import RULE_ORDER, CleaningCounts, clean_trips, total_counts
from demand.data.clean_weather import WeatherSummary, clean_weather
from demand.data.ingest import manifest_key, trips_path
from demand.data.manifest import temp_path_for
from demand.data.weather import weather_path
from demand.logs import get_logger
from demand.schemas import silver_trips_schema, silver_weather_schema

log = get_logger(__name__)

SILVER_TRIPS_DIR = "trips"
SILVER_WEATHER_DIR = "weather"
SILVER_WEATHER_FILE_NAME = "weather_hourly.parquet"
CLEANING_REPORT_NAME = "silver_cleaning_report.json"


class MissingRawFileError(FileNotFoundError):
    """A raw file that the config asks for is not on disk."""


class CleaningReport(BaseModel):
    """What the silver stage removed and why; saved as JSON next to other reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_order: list[str]
    """The trip rules in the order they take precedence when counting."""
    months: dict[str, CleaningCounts]
    """Counts per trip file, keyed by month, e.g. ``"2025-01"``."""
    total: CleaningCounts
    weather: WeatherSummary


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def month_key(year: int, month: int) -> str:
    """A month as text, e.g. ``2025-01``; used in file names and in the report."""
    return f"{year}-{month:02d}"


def silver_trips_path(silver_dir: Path, year: int, month: int) -> Path:
    """Where the cleaned trips of one month are stored."""
    return silver_dir / SILVER_TRIPS_DIR / f"trips_{month_key(year, month)}.parquet"


def silver_weather_path(silver_dir: Path) -> Path:
    """Where the cleaned hourly weather is stored."""
    return silver_dir / SILVER_WEATHER_DIR / SILVER_WEATHER_FILE_NAME


def cleaning_report_path(reports_dir: Path) -> Path:
    """Where the cleaning report is stored."""
    return reports_dir / CLEANING_REPORT_NAME


def required_raw_files(settings: Settings) -> list[Path]:
    """Every raw file this stage reads: one trip file per month, and the weather."""
    data = settings.data
    raw_dir = settings.paths.raw
    trip_files = [
        trips_path(raw_dir, data.tlc_url_template, year, month) for year, month in data.months
    ]
    return [*trip_files, weather_path(raw_dir)]


def report_to_json(report: CleaningReport) -> str:
    """Serialise the report. Rules keep their order of precedence, so keys are not sorted."""
    return json.dumps(report.model_dump(mode="json"), indent=2) + "\n"


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def check_raw_files_exist(settings: Settings) -> None:
    """Stop before any work is done if a raw file is missing.

    Raises:
        MissingRawFileError: Naming every missing file.
    """
    raw_dir = settings.paths.raw
    missing = [path for path in required_raw_files(settings) if not path.is_file()]
    if missing:
        names = ", ".join(manifest_key(raw_dir, path) for path in missing)
        raise MissingRawFileError(
            f"Raw files missing in {raw_dir}: {names}. Run `python -m demand ingest` first."
        )


def read_raw_trips(path: Path) -> pd.DataFrame:
    """Read one raw trip file with all its columns.

    Silver keeps only a few columns, but the duplicate rule compares whole rows, so
    everything is read.
    """
    raw = pd.read_parquet(path)
    # pyarrow holds on to the memory it needed while reading. Handing it back to the
    # operating system lowers the peak of the stage by several hundred megabytes.
    pa.default_memory_pool().release_unused()
    return raw


def write_parquet(table: pd.DataFrame, path: Path) -> None:
    """Write a table as Parquet atomically: to a temporary file first, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_path_for(path)
    try:
        table.to_parquet(temp_path, index=False)
        temp_path.replace(path)
    finally:
        # After a successful rename there is nothing left to delete.
        temp_path.unlink(missing_ok=True)


def save_cleaning_report(reports_dir: Path, report: CleaningReport) -> Path:
    """Write the report atomically and return its path."""
    path = cleaning_report_path(reports_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_path_for(path)
    try:
        temp_path.write_bytes(report_to_json(report).encode("utf-8"))
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
    return path


def build_silver_trips(settings: Settings) -> dict[str, CleaningCounts]:
    """Clean, validate and write the trips of every configured month.

    Returns:
        The cleaning counts per month, keyed like ``"2025-01"``.

    Raises:
        pandera.errors.SchemaError: If cleaned trips break ``silver_trips_schema``.
    """
    data = settings.data
    schema = silver_trips_schema(settings)
    counts_per_month: dict[str, CleaningCounts] = {}
    for year, month in data.months:
        started = time.monotonic()
        raw_path = trips_path(settings.paths.raw, data.tlc_url_template, year, month)
        silver_path = silver_trips_path(settings.paths.silver, year, month)

        raw = read_raw_trips(raw_path)
        trips, counts = clean_trips(raw, year, month, settings)
        # Free the raw month before validating; it is several times larger than silver.
        del raw
        schema.validate(trips)
        write_parquet(trips, silver_path)

        counts_per_month[month_key(year, month)] = counts
        log.info(
            "trips cleaned",
            extra={
                "file": silver_path.name,
                "rows_in": counts.rows_in,
                "rows_out": counts.rows_out,
                "percent_removed": counts.percent_removed,
                "removed": counts.removed,
                "seconds": round(time.monotonic() - started, 2),
            },
        )
    return counts_per_month


def build_silver_weather(settings: Settings) -> WeatherSummary:
    """Clean, validate and write the hourly weather.

    Raises:
        RawFormatError: If the raw file is not a readable Open-Meteo response.
        WeatherGapError: If the weather has a gap that may not be interpolated.
        pandera.errors.SchemaError: If the table breaks ``silver_weather_schema``.
    """
    raw_path = weather_path(settings.paths.raw)
    silver_path = silver_weather_path(settings.paths.silver)

    payload = json.loads(raw_path.read_bytes())
    weather, summary = clean_weather(payload, settings)
    silver_weather_schema(settings).validate(weather)
    write_parquet(weather, silver_path)

    log.info(
        "weather cleaned",
        extra={
            "file": silver_path.name,
            "hours": summary.hours,
            "interpolated": summary.interpolated,
        },
    )
    return summary


def build_silver(settings: Settings) -> CleaningReport:
    """Build the whole silver layer and write the cleaning report.

    Safe to run again: every output is rebuilt from the raw layer and replaced.

    Raises:
        MissingRawFileError: If a raw file is missing; nothing is written in that case.
    """
    started = time.monotonic()
    check_raw_files_exist(settings)

    # Weather first: it takes a moment, so a problem with it shows up before the
    # minutes spent on trips.
    weather = build_silver_weather(settings)
    months = build_silver_trips(settings)
    total = total_counts(list(months.values()))
    report = CleaningReport(
        rule_order=list(RULE_ORDER), months=months, total=total, weather=weather
    )
    report_path = save_cleaning_report(settings.paths.reports_dir, report)

    log.info(
        "silver finished",
        extra={
            "silver_dir": str(settings.paths.silver),
            "report": str(report_path),
            "rows_in": total.rows_in,
            "rows_out": total.rows_out,
            "percent_removed": total.percent_removed,
            "removed": total.removed,
            "weather_hours": weather.hours,
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return report
