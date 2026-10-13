from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from demand.data.ingest import (
    TRIP_NOT_PUBLISHED_HINT,
    DownloadError,
    RejectedRequestError,
    RetryPolicy,
    backoff_seconds,
    download_file,
    file_name_from_url,
    ingest_file,
    is_retryable_status,
    parquet_row_count,
    trip_url,
    trips_path,
    zones_path,
)
from demand.data.manifest import Manifest, load_manifest
from tests.unit.data.conftest import (
    FAST_RETRY,
    FakeServer,
    broken_stream,
    network_error,
    ok,
    parquet_bytes,
    status,
)

TEMPLATE = "https://example.test/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
URL = "https://example.test/trip-data/yellow_tripdata_2025-01.parquet"
CSV_URL = "https://example.test/misc/taxi_zone_lookup.csv"


def _files_in(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir())


# --- pure helpers ---------------------------------------------------------------------


def test_trip_url_pads_the_month_to_two_digits() -> None:
    assert trip_url(TEMPLATE, 2025, 1) == URL


def test_file_name_is_the_last_part_of_the_url() -> None:
    assert file_name_from_url(URL) == "yellow_tripdata_2025-01.parquet"


def test_trip_files_live_in_the_trips_folder(tmp_path: Path) -> None:
    expected = tmp_path / "trips" / "yellow_tripdata_2025-01.parquet"

    assert trips_path(tmp_path, TEMPLATE, 2025, 1) == expected


def test_zone_lookup_lives_in_the_zones_folder(tmp_path: Path) -> None:
    assert zones_path(tmp_path, CSV_URL) == tmp_path / "zones" / "taxi_zone_lookup.csv"


@pytest.mark.parametrize("code", [429, 500, 502, 503])
def test_server_errors_and_rate_limits_are_retryable(code: int) -> None:
    assert is_retryable_status(code)


@pytest.mark.parametrize("code", [400, 403, 404])
def test_client_errors_are_not_retryable(code: int) -> None:
    assert not is_retryable_status(code)


def test_backoff_doubles_with_every_attempt() -> None:
    policy = RetryPolicy(max_attempts=4, backoff_base_seconds=0.5)

    assert [backoff_seconds(attempt, policy) for attempt in (1, 2, 3)] == [0.5, 1.0, 2.0]


# --- download_file --------------------------------------------------------------------


def test_download_writes_the_bytes_exactly_as_received(tmp_path: Path, server: FakeServer) -> None:
    content = b"LocationID,Zone\n1,Newark Airport\n"
    server.add(CSV_URL, ok(content))
    destination = tmp_path / "zones" / "lookup.csv"

    download = download_file(server.client(), CSV_URL, destination)

    assert destination.read_bytes() == content
    assert download.size_bytes == len(content)
    assert download.sha256 == hashlib.sha256(content).hexdigest()
    assert _files_in(destination.parent) == ["lookup.csv"]


def test_download_retries_after_a_server_error_and_then_succeeds(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(CSV_URL, status(503), ok(b"data"))
    destination = tmp_path / "lookup.csv"

    download_file(server.client(), CSV_URL, destination, retry=FAST_RETRY, sleep=waits.append)

    assert destination.read_bytes() == b"data"
    assert len(server.requests) == 2
    assert waits == [1.0]


def test_download_retries_after_a_network_error(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(CSV_URL, network_error(), ok(b"data"))
    destination = tmp_path / "lookup.csv"

    download_file(server.client(), CSV_URL, destination, retry=FAST_RETRY, sleep=waits.append)

    assert destination.read_bytes() == b"data"
    assert len(server.requests) == 2


def test_download_retries_after_a_rate_limit(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(CSV_URL, status(429), ok(b"data"))

    download_file(
        server.client(), CSV_URL, tmp_path / "lookup.csv", retry=FAST_RETRY, sleep=waits.append
    )

    assert len(server.requests) == 2


def test_download_gives_up_after_the_configured_number_of_attempts(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(CSV_URL, status(500))
    destination = tmp_path / "lookup.csv"

    with pytest.raises(DownloadError, match="Gave up after 3 attempts"):
        download_file(server.client(), CSV_URL, destination, retry=FAST_RETRY, sleep=waits.append)

    assert len(server.requests) == 3
    assert waits == [1.0, 2.0]
    assert _files_in(tmp_path) == []


def test_404_fails_at_once_with_the_url_in_the_message(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(URL, status(404))

    with pytest.raises(DownloadError, match=r"HTTP 404 from .*yellow_tripdata_2025-01\.parquet"):
        download_file(
            server.client(), URL, tmp_path / "trips.parquet", retry=FAST_RETRY, sleep=waits.append
        )

    assert len(server.requests) == 1
    assert waits == []


def test_404_leaves_no_file_and_no_temporary_file(tmp_path: Path, server: FakeServer) -> None:
    server.add(URL, status(404))

    with pytest.raises(DownloadError):
        download_file(server.client(), URL, tmp_path / "trips.parquet")

    assert _files_in(tmp_path) == []


def test_failure_in_the_middle_of_the_stream_leaves_no_file(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(CSV_URL, broken_stream(b"first half"))

    with pytest.raises(DownloadError):
        download_file(
            server.client(), CSV_URL, tmp_path / "lookup.csv", retry=FAST_RETRY, sleep=waits.append
        )

    assert _files_in(tmp_path) == []


def test_failure_in_the_middle_of_the_stream_keeps_the_previous_file(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    destination = tmp_path / "lookup.csv"
    destination.write_bytes(b"old but complete")
    server.add(CSV_URL, broken_stream(b"first half"))

    with pytest.raises(DownloadError):
        download_file(server.client(), CSV_URL, destination, retry=FAST_RETRY, sleep=waits.append)

    assert destination.read_bytes() == b"old but complete"
    assert _files_in(tmp_path) == ["lookup.csv"]


def test_stream_that_breaks_once_is_downloaded_again(
    tmp_path: Path, server: FakeServer, waits: list[float]
) -> None:
    server.add(CSV_URL, broken_stream(b"first half"), ok(b"whole file"))
    destination = tmp_path / "lookup.csv"

    download_file(server.client(), CSV_URL, destination, retry=FAST_RETRY, sleep=waits.append)

    assert destination.read_bytes() == b"whole file"


def test_rejected_content_is_deleted(tmp_path: Path, server: FakeServer) -> None:
    server.add(CSV_URL, ok(b"data"))

    def reject(path: Path) -> None:
        raise DownloadError("content rejected")

    with pytest.raises(DownloadError, match="content rejected"):
        download_file(server.client(), CSV_URL, tmp_path / "lookup.csv", check=reject)

    assert _files_in(tmp_path) == []


# --- Parquet row count ----------------------------------------------------------------


def test_row_count_is_read_from_the_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "trips.parquet"
    path.write_bytes(parquet_bytes(7))

    assert parquet_row_count(path) == 7


def test_row_count_of_a_non_parquet_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "trips.parquet"
    path.write_bytes(b"<html>not parquet</html>")

    with pytest.raises(DownloadError, match="Not a valid Parquet file"):
        parquet_row_count(path)


# --- ingest_file: download + manifest -------------------------------------------------


def test_ingest_writes_the_file_and_its_manifest_entry(tmp_path: Path, server: FakeServer) -> None:
    content = parquet_bytes(5)
    server.add(URL, ok(content))
    path = trips_path(tmp_path, TEMPLATE, 2025, 1)
    manifest = Manifest()

    result = ingest_file(server.client(), tmp_path, manifest, path, URL)

    assert result.downloaded
    assert path.read_bytes() == content
    entry = load_manifest(tmp_path).files["trips/yellow_tripdata_2025-01.parquet"]
    assert entry.url == URL
    assert entry.size_bytes == len(content)
    assert entry.sha256 == hashlib.sha256(content).hexdigest()
    assert entry.row_count == 5
    assert datetime.now(UTC) - entry.downloaded_at < timedelta(minutes=1)


def test_csv_file_has_no_row_count_in_the_manifest(tmp_path: Path, server: FakeServer) -> None:
    server.add(CSV_URL, ok(b"LocationID\n1\n"))
    manifest = Manifest()

    ingest_file(server.client(), tmp_path, manifest, zones_path(tmp_path, CSV_URL), CSV_URL)

    assert manifest.files["zones/taxi_zone_lookup.csv"].row_count is None


def test_second_ingest_makes_no_request(tmp_path: Path, server: FakeServer) -> None:
    server.add(URL, ok(parquet_bytes(5)))
    path = trips_path(tmp_path, TEMPLATE, 2025, 1)
    ingest_file(server.client(), tmp_path, load_manifest(tmp_path), path, URL)
    server.requests.clear()

    result = ingest_file(server.client(), tmp_path, load_manifest(tmp_path), path, URL)

    assert server.requests == []
    assert not result.downloaded


def test_corrupted_file_is_downloaded_again(tmp_path: Path, server: FakeServer) -> None:
    content = parquet_bytes(5)
    server.add(URL, ok(content))
    path = trips_path(tmp_path, TEMPLATE, 2025, 1)
    ingest_file(server.client(), tmp_path, load_manifest(tmp_path), path, URL)
    path.write_bytes(b"corrupted")
    server.requests.clear()

    result = ingest_file(server.client(), tmp_path, load_manifest(tmp_path), path, URL)

    assert len(server.requests) == 1
    assert result.downloaded
    assert path.read_bytes() == content


def test_missing_file_is_downloaded_again(tmp_path: Path, server: FakeServer) -> None:
    server.add(URL, ok(parquet_bytes(5)))
    path = trips_path(tmp_path, TEMPLATE, 2025, 1)
    ingest_file(server.client(), tmp_path, load_manifest(tmp_path), path, URL)
    path.unlink()
    server.requests.clear()

    ingest_file(server.client(), tmp_path, load_manifest(tmp_path), path, URL)

    assert len(server.requests) == 1
    assert path.is_file()


def test_error_page_served_as_parquet_never_reaches_the_raw_folder(
    tmp_path: Path, server: FakeServer
) -> None:
    server.add(URL, ok(b"<html>temporarily unavailable</html>"))
    path = trips_path(tmp_path, TEMPLATE, 2025, 1)
    manifest = Manifest()

    with pytest.raises(DownloadError, match="Not a valid Parquet file"):
        ingest_file(server.client(), tmp_path, manifest, path, URL)

    assert _files_in(path.parent) == []
    assert manifest.files == {}


def test_refused_request_adds_the_hint_of_the_source(tmp_path: Path, server: FakeServer) -> None:
    server.add(URL, status(403))
    path = trips_path(tmp_path, TEMPLATE, 2025, 1)

    with pytest.raises(RejectedRequestError, match=r"HTTP 403 .* not published yet"):
        ingest_file(
            server.client(), tmp_path, Manifest(), path, URL, rejected_hint=TRIP_NOT_PUBLISHED_HINT
        )


def test_refused_request_without_a_hint_keeps_the_plain_message(
    tmp_path: Path, server: FakeServer
) -> None:
    server.add(CSV_URL, status(404))

    with pytest.raises(
        RejectedRequestError, match=r"check the configured URL and its parameters\.$"
    ):
        ingest_file(
            server.client(), tmp_path, Manifest(), tmp_path / "zones" / "lookup.csv", CSV_URL
        )
