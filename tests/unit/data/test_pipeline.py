from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from demand.config import Settings
from demand.data import pipeline
from demand.data.ingest import DownloadError, trip_url
from demand.data.manifest import load_manifest
from demand.data.pipeline import run_ingest
from tests.unit.data.conftest import FAST_RETRY, FakeServer, status


def _relative_files(folder: Path) -> list[str]:
    return sorted(
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    )


EXPECTED_FILES = [
    "manifest.json",
    "trips/yellow_tripdata_2025-01.parquet",
    "trips/yellow_tripdata_2025-02.parquet",
    "trips/yellow_tripdata_2025-03.parquet",
    "trips/yellow_tripdata_2025-04.parquet",
    "trips/yellow_tripdata_2025-05.parquet",
    "trips/yellow_tripdata_2025-06.parquet",
    "weather/open_meteo_hourly.json",
    "zones/taxi_zone_lookup.csv",
]


def test_ingest_creates_the_documented_raw_layout(
    settings: Settings, full_server: FakeServer
) -> None:
    run_ingest(settings, client=full_server.client())

    assert _relative_files(settings.paths.raw) == EXPECTED_FILES


def test_manifest_lists_every_raw_file_with_its_row_count(
    settings: Settings, full_server: FakeServer
) -> None:
    run_ingest(settings, client=full_server.client())

    files = load_manifest(settings.paths.raw).files
    assert sorted(files) == [name for name in EXPECTED_FILES if name != "manifest.json"]
    assert files["trips/yellow_tripdata_2025-01.parquet"].row_count == 1
    assert files["trips/yellow_tripdata_2025-06.parquet"].row_count == 6


def test_second_run_makes_no_http_request(settings: Settings, full_server: FakeServer) -> None:
    run_ingest(settings, client=full_server.client())
    manifest_before = load_manifest(settings.paths.raw)
    full_server.requests.clear()

    run_ingest(settings, client=full_server.client())

    assert full_server.requests == []
    assert load_manifest(settings.paths.raw) == manifest_before


def test_only_the_damaged_file_is_downloaded_again(
    settings: Settings, full_server: FakeServer
) -> None:
    run_ingest(settings, client=full_server.client())
    damaged = settings.paths.raw / "trips" / "yellow_tripdata_2025-03.parquet"
    damaged.write_bytes(b"damaged")
    full_server.requests.clear()

    run_ingest(settings, client=full_server.client())

    march_url = trip_url(settings.data.tlc_url_template, 2025, 3)
    assert [str(request.url) for request in full_server.requests] == [march_url]


def test_missing_month_stops_the_run_but_keeps_finished_files(
    settings: Settings, full_server: FakeServer, waits: list[float]
) -> None:
    march_url = trip_url(settings.data.tlc_url_template, 2025, 3)
    full_server.add(march_url, status(404))

    with pytest.raises(DownloadError, match="yellow_tripdata_2025-03"):
        run_ingest(settings, client=full_server.client(), retry=FAST_RETRY, sleep=waits.append)

    assert _relative_files(settings.paths.raw) == [
        "manifest.json",
        "trips/yellow_tripdata_2025-01.parquet",
        "trips/yellow_tripdata_2025-02.parquet",
    ]


def test_each_file_is_logged_as_downloaded_then_skipped(
    settings: Settings, full_server: FakeServer, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    run_ingest(settings, client=full_server.client())
    first_run = [record.message for record in caplog.records]
    caplog.clear()
    run_ingest(settings, client=full_server.client())
    second_run = [record.message for record in caplog.records]

    assert first_run.count("downloaded") == 8
    assert second_run.count("skipped") == 8
    assert "downloaded" not in second_run


def test_default_client_is_created_and_closed_when_none_is_passed(
    settings: Settings, full_server: FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    offline_client = full_server.client()
    options_used: dict[str, object] = {}

    def fake_client(**options: object) -> httpx.Client:
        options_used.update(options)
        return offline_client

    # Replace the real client class, so that this test cannot reach the network.
    monkeypatch.setattr(pipeline.httpx, "Client", fake_client)

    run_ingest(settings)

    timeout = options_used["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == settings.data.download.connect_timeout_seconds
    assert timeout.read == settings.data.download.read_timeout_seconds
    assert offline_client.is_closed
    assert len(full_server.requests) == 8


def test_retry_policy_comes_from_the_config(settings: Settings) -> None:
    policy = pipeline.retry_policy(settings)

    assert policy.max_attempts == settings.data.download.max_attempts
    assert policy.backoff_base_seconds == settings.data.download.backoff_base_seconds


def test_gold_stage_hands_over_to_build_gold(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Settings] = []
    monkeypatch.setattr(pipeline, "build_gold", calls.append)

    pipeline.run_gold(settings)

    assert calls == [settings]


def test_select_zones_stage_prints_the_list_for_the_config(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(pipeline, "select_zones", lambda settings: {161: 30, 132: 20})

    pipeline.run_select_zones(settings)

    comment, zones_line = capsys.readouterr().out.splitlines()
    assert comment.strip().startswith("#")
    assert "2025-01-01 to 2025-04-30" in comment
    assert zones_line == "  zones: [161, 132]"


def test_validate_says_it_is_not_built_yet(settings: Settings) -> None:
    with pytest.raises(NotImplementedError, match="phase 1"):
        pipeline.run_validate(settings, "silver")
