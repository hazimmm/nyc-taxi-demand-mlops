"""Stage functions of the data pipeline, called by ``python -m demand <command>``.

Each function takes the ``Settings`` object first and returns ``None``.
"""

from __future__ import annotations

import time

import httpx

from demand.config import Settings
from demand.data.gold import build_gold, select_zones, zones_config_text
from demand.data.ingest import (
    TRIP_NOT_PUBLISHED_HINT,
    IngestedFile,
    RetryPolicy,
    Sleep,
    ingest_file,
    trip_url,
    trips_path,
    zones_path,
)
from demand.data.manifest import load_manifest
from demand.data.silver import build_silver
from demand.data.validate import validate_layer
from demand.data.weather import ingest_weather
from demand.logs import get_logger

log = get_logger(__name__)


def http_timeout(settings: Settings) -> httpx.Timeout:
    """Timeouts from the config.

    They limit one network operation, not a whole download: a large file may take
    minutes in total, but no single read may stall for longer than the read timeout.
    """
    download = settings.data.download
    return httpx.Timeout(download.read_timeout_seconds, connect=download.connect_timeout_seconds)


def retry_policy(settings: Settings) -> RetryPolicy:
    """Retry policy from the config."""
    download = settings.data.download
    return RetryPolicy(download.max_attempts, download.backoff_base_seconds)


def run_ingest(
    settings: Settings,
    *,
    client: httpx.Client | None = None,
    retry: RetryPolicy | None = None,
    sleep: Sleep = time.sleep,
) -> None:
    """Download trips, the zone lookup and weather into the raw layer.

    Safe to run again: a file that is present and matches its checksum in the manifest
    is not downloaded a second time.

    Args:
        settings: Project settings; the folder, URLs and period come from here.
        client: HTTP client to use. Left out in normal use; tests pass a client built
            on ``httpx.MockTransport`` so that nothing touches the network.
        retry: Number of attempts and backoff for transient failures. Taken from
            ``settings.data.download`` when left out.
        sleep: Function used to wait between attempts.

    Raises:
        DownloadError: If a file cannot be downloaded. Files finished before the
            failure stay in place and are skipped by the next run.
    """
    retry = retry or retry_policy(settings)
    if client is None:
        timeout = http_timeout(settings)
        with httpx.Client(timeout=timeout, follow_redirects=True) as default_client:
            run_ingest(settings, client=default_client, retry=retry, sleep=sleep)
        return

    started = time.monotonic()
    data = settings.data
    raw_dir = settings.paths.raw
    manifest = load_manifest(raw_dir)

    results: list[IngestedFile] = []
    for year, month in data.months:
        url = trip_url(data.tlc_url_template, year, month)
        path = trips_path(raw_dir, data.tlc_url_template, year, month)
        results.append(
            ingest_file(
                client,
                raw_dir,
                manifest,
                path,
                url,
                retry=retry,
                sleep=sleep,
                rejected_hint=TRIP_NOT_PUBLISHED_HINT,
            )
        )
    results.append(
        ingest_file(
            client,
            raw_dir,
            manifest,
            zones_path(raw_dir, data.zone_lookup_url),
            data.zone_lookup_url,
            retry=retry,
            sleep=sleep,
        )
    )
    results.append(ingest_weather(client, settings, manifest, retry=retry, sleep=sleep))

    downloaded = [result for result in results if result.downloaded]
    log.info(
        "ingest finished",
        extra={
            "raw_dir": str(raw_dir),
            "files_downloaded": len(downloaded),
            "files_skipped": len(results) - len(downloaded),
            "bytes_downloaded": sum(result.entry.size_bytes for result in downloaded),
            "seconds": round(time.monotonic() - started, 2),
        },
    )


def run_select_zones(settings: Settings) -> None:
    """Print the busiest pickup zones of the training period, ready for the config.

    The zones are counted on the silver trips up to ``data.train_end_date`` and on
    nothing later. The config file is not changed: the printed line has to be pasted
    under ``data:`` by hand. The work itself is in ``gold.py``.

    Raises:
        MissingSilverFileError: If a silver trip file of the training period is
            missing; run the silver stage first.
        TooFewZonesError: If fewer than ``data.n_zones`` zones have pickups.
    """
    selected = select_zones(settings)
    data = settings.data
    # The result goes to standard output on purpose: logs go to standard error, so
    # the lines below can be copied or redirected without any log text in between.
    print(
        f"  # The {len(selected)} busiest pickup zones from {data.start_date} to "
        f"{data.train_end_date} (training period only), busiest first."
    )
    print(zones_config_text(selected))


def run_silver(settings: Settings) -> None:
    """Clean the raw layer into the silver layer and write the cleaning report.

    Safe to run again: the silver files and the report are rebuilt from the raw layer
    and come out identical. The work itself is in ``silver.py``.

    Raises:
        MissingRawFileError: If a raw file is missing; run the ingest stage first.
        pandera.errors.SchemaError: If a cleaned table breaks its silver schema.
    """
    build_silver(settings)


def run_gold(settings: Settings) -> None:
    """Aggregate the silver layer into hourly demand and build the feature table.

    Safe to run again: both gold files are rebuilt from the silver layer and come out
    identical. The work itself is in ``gold.py``.

    Raises:
        ZonesNotSelectedError: If ``data.zones`` is empty; run ``select-zones`` first.
        MissingSilverFileError: If a silver file is missing; run the silver stage first.
        pandera.errors.SchemaError: If a table breaks its gold schema.
    """
    build_gold(settings)


def run_validate(settings: Settings, layer: str) -> None:
    """Check the files of one layer and write the validation report.

    Safe to run again: no data is changed, and the report comes out identical. The work
    itself is in ``validate.py``.

    Args:
        settings: Project settings.
        layer: ``"silver"`` or ``"gold"``.

    Raises:
        QualityGateError: If a check failed; the report says which and why.
        ZonesNotSelectedError: For the gold layer, if ``data.zones`` is empty.
    """
    validate_layer(settings, layer)
