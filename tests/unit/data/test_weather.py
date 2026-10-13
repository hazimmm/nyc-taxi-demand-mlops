from __future__ import annotations

import json
from datetime import date

import httpx
import pytest

from demand.config import Settings
from demand.data.ingest import DownloadError, RejectedRequestError
from demand.data.manifest import Manifest, load_manifest
from demand.data.weather import (
    build_weather_url,
    expected_hours,
    find_weather_problems,
    ingest_weather,
    request_period,
    weather_path,
)
from tests.unit.data.conftest import (
    FAST_RETRY,
    FakeServer,
    complete_weather,
    ok,
    status,
    weather_bytes,
)

VARIABLES = ["temperature_2m", "precipitation"]


def _payload(**changes: object) -> object:
    return json.loads(weather_bytes(VARIABLES, hours=48)) | changes


# --- pure helpers ---------------------------------------------------------------------


def test_request_period_adds_a_day_on_each_side() -> None:
    assert request_period(date(2025, 1, 1), date(2025, 6, 30)) == (
        date(2024, 12, 31),
        date(2025, 7, 1),
    )


def test_expected_hours_counts_both_end_days() -> None:
    assert expected_hours(date(2025, 1, 1), date(2025, 1, 2)) == 48


def test_weather_url_is_built_from_the_settings(settings: Settings) -> None:
    weather = settings.data.weather

    url = httpx.URL(
        build_weather_url(weather.forecast_url, weather, date(2025, 1, 1), date(2025, 6, 30))
    )

    assert str(url.copy_with(query=None)) == weather.forecast_url
    assert dict(url.params) == {
        "latitude": "40.7128",
        "longitude": "-74.006",
        "start_date": "2025-01-01",
        "end_date": "2025-06-30",
        "hourly": "temperature_2m,precipitation,snowfall,wind_speed_10m",
        "timezone": "GMT",
    }


def test_complete_response_has_no_problems() -> None:
    assert find_weather_problems(_payload(), VARIABLES, hours=48) == []


def test_null_values_are_reported() -> None:
    payload = json.loads(weather_bytes(VARIABLES, hours=48, null_in="precipitation"))

    assert find_weather_problems(payload, VARIABLES, hours=48) == [
        "'precipitation' has 1 null values"
    ]


def test_missing_variable_is_reported() -> None:
    problems = find_weather_problems(_payload(), [*VARIABLES, "snowfall"], hours=48)

    assert problems == ["'snowfall' is missing"]


def test_too_few_hours_are_reported() -> None:
    problems = find_weather_problems(_payload(), VARIABLES, hours=72)

    assert "'time' has 48 values, expected 72" in problems


def test_response_without_hourly_section_is_reported() -> None:
    problems = find_weather_problems({"error": True, "reason": "bad request"}, VARIABLES, hours=48)

    assert problems == ["response has no 'hourly' section"]


# --- ingest_weather -------------------------------------------------------------------


def test_weather_uses_the_forecast_api_when_it_works(
    settings: Settings, full_server: FakeServer
) -> None:
    weather = settings.data.weather
    manifest = Manifest()

    ingest_weather(full_server.client(), settings, manifest)

    assert len(full_server.requests_to(weather.forecast_url)) == 1
    assert full_server.requests_to(weather.archive_url) == []
    path = weather_path(settings.paths.raw)
    assert path.read_bytes() == complete_weather(settings)
    entry = load_manifest(settings.paths.raw).files["weather/open_meteo_hourly.json"]
    assert entry.weather_source == "forecast"
    assert entry.url.startswith(weather.forecast_url)
    assert entry.row_count is None


def test_weather_request_covers_the_padded_period_in_utc(
    settings: Settings, full_server: FakeServer
) -> None:
    ingest_weather(full_server.client(), settings, Manifest())

    params = full_server.requests[0].url.params
    assert params["start_date"] == "2024-12-31"
    assert params["end_date"] == "2025-07-01"
    assert params["timezone"] == "GMT"
    assert params["hourly"] == ",".join(settings.data.weather.variables)
    assert params["latitude"] == str(settings.data.weather.latitude)


def test_weather_falls_back_to_the_archive_when_the_forecast_api_errors(
    settings: Settings, full_server: FakeServer, waits: list[float]
) -> None:
    weather = settings.data.weather
    full_server.add(weather.forecast_url, status(500))
    manifest = Manifest()

    ingest_weather(full_server.client(), settings, manifest, retry=FAST_RETRY, sleep=waits.append)

    assert len(full_server.requests_to(weather.forecast_url)) == FAST_RETRY.max_attempts
    assert len(full_server.requests_to(weather.archive_url)) == 1
    entry = manifest.files["weather/open_meteo_hourly.json"]
    assert entry.weather_source == "archive"
    assert entry.url.startswith(weather.archive_url)


def test_weather_falls_back_to_the_archive_when_the_forecast_has_nulls(
    settings: Settings, full_server: FakeServer
) -> None:
    weather = settings.data.weather
    hours = expected_hours(*request_period(settings.data.start_date, settings.data.end_date))
    with_null = weather_bytes(weather.variables, hours, null_in="snowfall")
    full_server.add(weather.forecast_url, ok(with_null))
    manifest = Manifest()

    ingest_weather(full_server.client(), settings, manifest)

    assert manifest.files["weather/open_meteo_hourly.json"].weather_source == "archive"
    stored = weather_path(settings.paths.raw).read_bytes()
    assert stored == complete_weather(settings, marker=2.0)


def test_weather_fails_when_both_apis_fail_and_leaves_no_file(
    settings: Settings, full_server: FakeServer, waits: list[float]
) -> None:
    weather = settings.data.weather
    full_server.add(weather.forecast_url, status(500))
    full_server.add(weather.archive_url, ok(b"not json"))
    manifest = Manifest()

    with pytest.raises(DownloadError, match="not valid JSON"):
        ingest_weather(
            full_server.client(), settings, manifest, retry=FAST_RETRY, sleep=waits.append
        )

    assert list(weather_path(settings.paths.raw).parent.iterdir()) == []
    assert manifest.files == {}


def test_refused_forecast_request_stops_without_a_fallback(
    settings: Settings, full_server: FakeServer
) -> None:
    weather = settings.data.weather
    full_server.add(weather.forecast_url, status(400))
    manifest = Manifest()

    with pytest.raises(RejectedRequestError, match=r"HTTP 400 .* data\.weather"):
        ingest_weather(full_server.client(), settings, manifest)

    assert full_server.requests_to(weather.archive_url) == []
    assert manifest.files == {}


def test_second_weather_ingest_makes_no_request(
    settings: Settings, full_server: FakeServer
) -> None:
    ingest_weather(full_server.client(), settings, load_manifest(settings.paths.raw))
    full_server.requests.clear()

    result = ingest_weather(full_server.client(), settings, load_manifest(settings.paths.raw))

    assert full_server.requests == []
    assert not result.downloaded


def test_observed_weather_is_replaced_once_the_forecast_api_works_again(
    settings: Settings, full_server: FakeServer, waits: list[float]
) -> None:
    weather = settings.data.weather
    forecast_reply = ok(complete_weather(settings))
    full_server.add(weather.forecast_url, status(500))
    ingest_weather(
        full_server.client(),
        settings,
        load_manifest(settings.paths.raw),
        retry=FAST_RETRY,
        sleep=waits.append,
    )
    full_server.add(weather.forecast_url, forecast_reply)
    full_server.requests.clear()

    result = ingest_weather(full_server.client(), settings, load_manifest(settings.paths.raw))

    assert result.downloaded
    assert full_server.requests_to(weather.archive_url) == []
    assert weather_path(settings.paths.raw).read_bytes() == complete_weather(settings)
    entry = load_manifest(settings.paths.raw).files["weather/open_meteo_hourly.json"]
    assert entry.weather_source == "forecast"


def test_observed_weather_is_kept_while_the_forecast_api_stays_down(
    settings: Settings,
    full_server: FakeServer,
    waits: list[float],
    caplog: pytest.LogCaptureFixture,
) -> None:
    weather = settings.data.weather
    full_server.add(weather.forecast_url, status(500))
    client = full_server.client()
    ingest_weather(
        client, settings, load_manifest(settings.paths.raw), retry=FAST_RETRY, sleep=waits.append
    )
    full_server.requests.clear()

    result = ingest_weather(
        client, settings, load_manifest(settings.paths.raw), retry=FAST_RETRY, sleep=waits.append
    )

    assert not result.downloaded
    assert full_server.requests_to(weather.archive_url) == []
    assert len(full_server.requests_to(weather.forecast_url)) == FAST_RETRY.max_attempts
    assert weather_path(settings.paths.raw).read_bytes() == complete_weather(settings, marker=2.0)
    assert result.entry.weather_source == "archive"
    assert "weather forecast still unavailable, keeping observed weather" in caplog.messages
