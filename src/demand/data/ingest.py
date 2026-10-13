"""Download source files into the raw layer, exactly as received.

Layout of the raw folder (``settings.paths.raw``)::

    raw/
      manifest.json                           record of every file (see manifest.py)
      trips/yellow_tripdata_2025-01.parquet   one TLC file per configured month
      zones/taxi_zone_lookup.csv              zone ID -> borough and zone name
      weather/open_meteo_hourly.json          Open-Meteo response (see weather.py)

The path helpers below (``trips_path``, ``zones_path``) are the only place where this
layout is written down; later stages import them instead of building paths themselves.

How a download works:

1. The response is streamed into ``<name>.part`` in the same folder.
2. An optional check inspects the temporary file (for example: is it valid Parquet?).
3. The temporary file is renamed to the final name. A rename inside one folder is
   atomic, so a file under its final name is always complete.

Transient failures (network errors, HTTP 5xx, HTTP 429) are retried with exponential
backoff. Other HTTP errors, such as 404, fail at once: waiting does not create a file
that is not there.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from demand.data.manifest import (
    Manifest,
    ManifestEntry,
    WeatherSource,
    download_reason,
    save_manifest,
    sha256_of_file,
    temp_path_for,
)
from demand.logs import get_logger

log = get_logger(__name__)

TRIPS_DIR = "trips"
ZONES_DIR = "zones"
PARQUET_SUFFIX = ".parquet"
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_TOO_MANY_REQUESTS = 429

Sleep = Callable[[float], None]
FileCheck = Callable[[Path], None]


class DownloadError(Exception):
    """A file could not be downloaded. The message names the URL."""


class TransientDownloadError(DownloadError):
    """A failure that may go away on its own, so the download is worth repeating."""


@dataclass(frozen=True)
class RetryPolicy:
    """How often a failed download is repeated and how long to wait in between."""

    max_attempts: int = 4
    backoff_base_seconds: float = 1.0


DEFAULT_RETRY = RetryPolicy()


@dataclass(frozen=True)
class Download:
    """Facts about one finished download."""

    size_bytes: int
    sha256: str
    seconds: float


@dataclass(frozen=True)
class IngestedFile:
    """Outcome for one raw file: its manifest entry and whether it was downloaded now."""

    name: str
    entry: ManifestEntry
    downloaded: bool
    seconds: float = 0.0


# --------------------------------------------------------------------------------------
# Pure helpers: no network, no disk
# --------------------------------------------------------------------------------------


def trip_url(url_template: str, year: int, month: int) -> str:
    """URL of the TLC trip file of one month."""
    return url_template.format(year=year, month=month)


def file_name_from_url(url: str) -> str:
    """Last path segment of a URL, e.g. ``yellow_tripdata_2025-01.parquet``."""
    return httpx.URL(url).path.rsplit("/", maxsplit=1)[-1]


def trips_path(raw_dir: Path, url_template: str, year: int, month: int) -> Path:
    """Where the trip file of one month is stored. The name is the one TLC uses."""
    return raw_dir / TRIPS_DIR / file_name_from_url(trip_url(url_template, year, month))


def zones_path(raw_dir: Path, zone_lookup_url: str) -> Path:
    """Where the zone lookup table is stored."""
    return raw_dir / ZONES_DIR / file_name_from_url(zone_lookup_url)


def manifest_key(raw_dir: Path, path: Path) -> str:
    """Manifest key of a raw file: its path relative to the raw folder, with ``/``."""
    return path.relative_to(raw_dir).as_posix()


def is_retryable_status(status_code: int) -> bool:
    """Whether an HTTP status means "try again later" (server error or rate limit)."""
    return status_code >= 500 or status_code == _TOO_MANY_REQUESTS


def backoff_seconds(attempt: int, policy: RetryPolicy) -> float:
    """Wait after failed attempt number ``attempt`` (1-based): base, 2x base, 4x base..."""
    return float(policy.backoff_base_seconds * 2 ** (attempt - 1))


# --------------------------------------------------------------------------------------
# Disk and network
# --------------------------------------------------------------------------------------


def parquet_row_count(path: Path) -> int:
    """Number of rows in a Parquet file, read from its footer without loading the data.

    Raises:
        DownloadError: If the file is not valid Parquet.
    """
    try:
        return int(pq.read_metadata(path).num_rows)
    except (pa.ArrowInvalid, OSError) as error:
        raise DownloadError(f"Not a valid Parquet file: {path.name} ({error})") from error


def _download_once(
    client: httpx.Client, url: str, destination: Path, check: FileCheck | None
) -> Download:
    """Make one attempt: stream to a temporary file, check it, rename it."""
    temp_path = temp_path_for(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size_bytes = 0
    started = time.monotonic()
    try:
        with client.stream("GET", url) as response:
            status = response.status_code
            if is_retryable_status(status):
                raise TransientDownloadError(f"HTTP {status} from {url}")
            if not response.is_success:
                raise DownloadError(
                    f"HTTP {status} from {url}. The file does not exist there (the TLC "
                    "server answers 403 for a month that is not published) or the "
                    "request is wrong; check the configured months and URLs."
                )
            with temp_path.open("wb") as file:
                for chunk in response.iter_bytes(_DOWNLOAD_CHUNK_BYTES):
                    file.write(chunk)
                    digest.update(chunk)
                    size_bytes += len(chunk)
        if check is not None:
            check(temp_path)
        temp_path.replace(destination)
    except httpx.TransportError as error:
        raise TransientDownloadError(f"{type(error).__name__} for {url}: {error}") from error
    finally:
        # After a successful rename there is nothing left to delete.
        temp_path.unlink(missing_ok=True)
    return Download(size_bytes, digest.hexdigest(), time.monotonic() - started)


def download_file(
    client: httpx.Client,
    url: str,
    destination: Path,
    *,
    retry: RetryPolicy = DEFAULT_RETRY,
    sleep: Sleep = time.sleep,
    check: FileCheck | None = None,
) -> Download:
    """Download ``url`` to ``destination`` atomically, retrying transient failures.

    Args:
        client: HTTP client; tests pass one built on ``httpx.MockTransport``.
        url: Full URL, including any query string.
        destination: Final path of the file.
        retry: Number of attempts and backoff.
        sleep: Function used to wait between attempts; tests pass a fake.
        check: Called with the temporary file before it is renamed. It raises
            ``DownloadError`` to reject the content, which is then deleted.

    Raises:
        DownloadError: On a permanent failure, a rejected file, or when every attempt
            failed. ``destination`` is left untouched in all of these cases.
    """
    for attempt in range(1, retry.max_attempts + 1):
        try:
            return _download_once(client, url, destination, check)
        except TransientDownloadError as error:
            if attempt == retry.max_attempts:
                raise DownloadError(
                    f"Gave up after {attempt} attempts. Last error: {error}"
                ) from error
            wait_seconds = backoff_seconds(attempt, retry)
            log.warning(
                "retried",
                extra={
                    "file": destination.name,
                    "url": url,
                    "attempt": attempt,
                    "wait_seconds": wait_seconds,
                    "error": str(error),
                },
            )
            sleep(wait_seconds)
    raise ValueError("retry.max_attempts must be at least 1")


def find_reason_to_download(
    raw_dir: Path, manifest: Manifest, path: Path, accepted_urls: set[str]
) -> str | None:
    """Compare the file on disk with the manifest; ``None`` means it is up to date."""
    actual_sha256 = sha256_of_file(path) if path.is_file() else None
    entry = manifest.files.get(manifest_key(raw_dir, path))
    return download_reason(entry, actual_sha256, accepted_urls)


def skipped(raw_dir: Path, manifest: Manifest, path: Path) -> IngestedFile:
    """Log that a file is already valid and return its existing manifest entry."""
    key = manifest_key(raw_dir, path)
    entry = manifest.files[key]
    log.info("skipped", extra={"file": key, "size_bytes": entry.size_bytes})
    return IngestedFile(name=key, entry=entry, downloaded=False)


def record_download(
    raw_dir: Path,
    manifest: Manifest,
    path: Path,
    url: str,
    download: Download,
    *,
    weather_source: WeatherSource | None = None,
) -> IngestedFile:
    """Add a finished download to the manifest, save the manifest and log the file.

    The manifest is saved after every file, so an interrupted run keeps what it has
    already finished.
    """
    key = manifest_key(raw_dir, path)
    entry = ManifestEntry(
        url=url,
        downloaded_at=datetime.now(UTC),
        size_bytes=download.size_bytes,
        sha256=download.sha256,
        row_count=parquet_row_count(path) if path.suffix == PARQUET_SUFFIX else None,
        weather_source=weather_source,
    )
    manifest.files[key] = entry
    save_manifest(raw_dir, manifest)
    log.info(
        "downloaded",
        extra={
            "file": key,
            "size_bytes": entry.size_bytes,
            "seconds": round(download.seconds, 2),
            "row_count": entry.row_count,
            "weather_source": entry.weather_source,
        },
    )
    return IngestedFile(name=key, entry=entry, downloaded=True, seconds=download.seconds)


def ingest_file(
    client: httpx.Client,
    raw_dir: Path,
    manifest: Manifest,
    path: Path,
    url: str,
    *,
    retry: RetryPolicy = DEFAULT_RETRY,
    sleep: Sleep = time.sleep,
) -> IngestedFile:
    """Make sure one single-source file is present and matches the manifest.

    Nothing is downloaded if the file is already valid. A Parquet file is opened before
    it is accepted, so an error page saved as ``.parquet`` never reaches the raw layer.
    """
    reason = find_reason_to_download(raw_dir, manifest, path, {url})
    if reason is None:
        return skipped(raw_dir, manifest, path)
    log.info("downloading", extra={"file": manifest_key(raw_dir, path), "reason": reason})

    def check_parquet(temp_path: Path) -> None:
        parquet_row_count(temp_path)

    check = check_parquet if path.suffix == PARQUET_SUFFIX else None
    download = download_file(client, url, path, retry=retry, sleep=sleep, check=check)
    return record_download(raw_dir, manifest, path, url, download)
