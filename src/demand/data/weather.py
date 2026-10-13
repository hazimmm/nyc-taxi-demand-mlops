"""Hourly weather from Open-Meteo, stored in the raw layer as the JSON it returned.

File: ``raw/weather/open_meteo_hourly.json``. The response has this shape::

    {"hourly": {"time": ["2025-01-01T00:00", ...],
                "temperature_2m": [8.7, ...],
                "precipitation": [0.0, ...], ...},
     "hourly_units": {...}, "latitude": ..., "longitude": ..., ...}

Decisions, and why:

* **Forecast first, observations as fallback.** The model predicts tomorrow, so the
  weather it may use is what a forecaster knew, not what was measured afterwards. The
  *historical forecast* API (``forecast_url``) is therefore tried first. If it fails or
  its answer is incomplete, the *archive* API (``archive_url``, observed weather) is
  used, and the manifest records ``weather_source`` so this limitation stays visible.
* **Times in UTC** (``timezone=GMT``). Local time repeats an hour when daylight saving
  ends, which makes timestamps ambiguous. Converting to New York time is the job of the
  silver layer.
* **One day of padding on each side of the period.** The request is made in UTC days,
  but the trips are in New York time, four or five hours behind. Without padding, the
  last evening of the period (local time) would have no weather.
* **One request for the whole period**, not one per month. Six months of four hourly
  variables are about 170 KB, so size is no reason to split. One request also means
  one source for the whole period: per-month requests could end up with forecast data
  for some months and observed data for others, which is harder to reason about.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode

import httpx

from demand.config import Settings, WeatherConfig
from demand.data.ingest import (
    DEFAULT_RETRY,
    DownloadError,
    IngestedFile,
    RetryPolicy,
    Sleep,
    download_file,
    find_reason_to_download,
    manifest_key,
    record_download,
    skipped,
)
from demand.data.manifest import Manifest, WeatherSource
from demand.logs import get_logger

log = get_logger(__name__)

WEATHER_DIR = "weather"
WEATHER_FILE_NAME = "open_meteo_hourly.json"
REQUEST_TIMEZONE = "GMT"
PADDING_DAYS = 1
_HOURS_PER_DAY = 24


class UnusableWeatherError(DownloadError):
    """The weather API answered, but the answer does not cover the request."""


# --------------------------------------------------------------------------------------
# Pure helpers: no network, no disk
# --------------------------------------------------------------------------------------


def weather_path(raw_dir: Path) -> Path:
    """Where the weather response is stored."""
    return raw_dir / WEATHER_DIR / WEATHER_FILE_NAME


def request_period(start_date: date, end_date: date) -> tuple[date, date]:
    """First and last UTC day to request: the period plus a day of padding each side."""
    padding = timedelta(days=PADDING_DAYS)
    return start_date - padding, end_date + padding


def expected_hours(first_day: date, last_day: date) -> int:
    """Number of hourly values a complete response contains (both days included)."""
    return ((last_day - first_day).days + 1) * _HOURS_PER_DAY


def build_weather_url(
    base_url: str, weather: WeatherConfig, first_day: date, last_day: date
) -> str:
    """Full Open-Meteo URL for hourly values of the configured variables, in UTC."""
    params = {
        "latitude": weather.latitude,
        "longitude": weather.longitude,
        "start_date": first_day.isoformat(),
        "end_date": last_day.isoformat(),
        "hourly": ",".join(weather.variables),
        "timezone": REQUEST_TIMEZONE,
    }
    return f"{base_url}?{urlencode(params, safe=',')}"


def find_weather_problems(payload: object, variables: Sequence[str], hours: int) -> list[str]:
    """List what is wrong with a decoded Open-Meteo response; empty means usable.

    A usable response has ``hours`` timestamps and, for every variable, ``hours``
    values of which none is null.
    """
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    if not isinstance(hourly, dict):
        return ["response has no 'hourly' section"]

    problems: list[str] = []
    for name in ("time", *variables):
        values = hourly.get(name)
        if not isinstance(values, list):
            problems.append(f"'{name}' is missing")
            continue
        if len(values) != hours:
            problems.append(f"'{name}' has {len(values)} values, expected {hours}")
        null_count = sum(value is None for value in values)
        if null_count:
            problems.append(f"'{name}' has {null_count} null values")
    return problems


# --------------------------------------------------------------------------------------
# Disk and network
# --------------------------------------------------------------------------------------


def check_weather_file(path: Path, variables: Sequence[str], hours: int) -> None:
    """Reject a downloaded weather response that is not complete.

    Raises:
        UnusableWeatherError: If the file is not JSON or ``find_weather_problems``
            reports anything.
    """
    try:
        payload = json.loads(path.read_bytes())
    except ValueError as error:
        raise UnusableWeatherError(f"weather response is not valid JSON: {error}") from error
    problems = find_weather_problems(payload, variables, hours)
    if problems:
        raise UnusableWeatherError("weather response is incomplete: " + "; ".join(problems))


def ingest_weather(
    client: httpx.Client,
    settings: Settings,
    manifest: Manifest,
    *,
    retry: RetryPolicy = DEFAULT_RETRY,
    sleep: Sleep = time.sleep,
) -> IngestedFile:
    """Make sure the weather file is present; prefer forecasts, fall back to observations.

    Raises:
        DownloadError: If neither API returns a complete response.
    """
    raw_dir = settings.paths.raw
    weather = settings.data.weather
    path = weather_path(raw_dir)
    first_day, last_day = request_period(settings.data.start_date, settings.data.end_date)
    hours = expected_hours(first_day, last_day)
    urls: dict[WeatherSource, str] = {
        "forecast": build_weather_url(weather.forecast_url, weather, first_day, last_day),
        "archive": build_weather_url(weather.archive_url, weather, first_day, last_day),
    }

    reason = find_reason_to_download(raw_dir, manifest, path, set(urls.values()))
    if reason is None:
        return skipped(raw_dir, manifest, path)
    log.info("downloading", extra={"file": manifest_key(raw_dir, path), "reason": reason})

    def check(temp_path: Path) -> None:
        check_weather_file(temp_path, weather.variables, hours)

    source: WeatherSource = "forecast"
    try:
        download = download_file(client, urls[source], path, retry=retry, sleep=sleep, check=check)
    except DownloadError as error:
        log.warning(
            "weather forecast unavailable, falling back to observed weather",
            extra={"url": urls[source], "error": str(error)},
        )
        source = "archive"
        download = download_file(client, urls[source], path, retry=retry, sleep=sleep, check=check)
    return record_download(raw_dir, manifest, path, urls[source], download, weather_source=source)
