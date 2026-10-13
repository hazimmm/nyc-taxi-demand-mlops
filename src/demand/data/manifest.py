"""The raw-layer manifest: a record of every downloaded file.

The manifest is one JSON file, ``manifest.json``, in the raw folder. For each file it
stores where the file came from, when it was downloaded, its size and its SHA-256
checksum. Ingestion compares the files on disk with this record to decide what, if
anything, has to be downloaded again.

Example::

    {
      "files": {
        "trips/yellow_tripdata_2025-01.parquet": {
          "downloaded_at": "2026-10-13T08:15:02.123456Z",
          "row_count": 3475226,
          "sha256": "9f2c...",
          "size_bytes": 59158238,
          "url": "https://.../yellow_tripdata_2025-01.parquet"
        }
      }
    }

Keys are file paths relative to the raw folder, always written with forward slashes so
the manifest is identical on Windows and Linux.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MANIFEST_NAME = "manifest.json"
TEMP_SUFFIX = ".part"
_HASH_CHUNK_BYTES = 1024 * 1024

WeatherSource = Literal["forecast", "archive"]


class ManifestEntry(BaseModel):
    """What is known about one downloaded file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    """Full source URL, including the query string."""
    downloaded_at: datetime
    """When the download finished, in UTC."""
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    row_count: int | None = Field(default=None, ge=0)
    """Number of rows, for Parquet files only."""
    weather_source: WeatherSource | None = None
    """For the weather file only: which Open-Meteo API produced it."""


class Manifest(BaseModel):
    """Every file in the raw layer, keyed by its path relative to the raw folder."""

    model_config = ConfigDict(extra="forbid")

    files: dict[str, ManifestEntry] = Field(default_factory=dict)


def temp_path_for(path: Path) -> Path:
    """Path of the temporary file that is written before ``path`` is replaced."""
    return path.with_name(path.name + TEMP_SUFFIX)


def manifest_path(raw_dir: Path) -> Path:
    """Location of the manifest inside the raw folder."""
    return raw_dir / MANIFEST_NAME


def sha256_of_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file, reading it in chunks to save memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def download_reason(
    entry: ManifestEntry | None,
    actual_sha256: str | None,
    accepted_urls: Collection[str],
) -> str | None:
    """Decide whether a file must be downloaded, and say why.

    Args:
        entry: The manifest entry of the file, or ``None`` if it has none.
        actual_sha256: Checksum of the file on disk, or ``None`` if the file is absent.
        accepted_urls: The URLs the file may come from with the current settings.

    Returns:
        A short reason if the file must be downloaded, or ``None`` if the file on disk
        is exactly what the manifest describes.
    """
    if actual_sha256 is None:
        return "file is missing"
    if entry is None:
        return "file is not in the manifest"
    if entry.url not in accepted_urls:
        return "source URL changed"
    if entry.sha256 != actual_sha256:
        return "checksum differs from the manifest"
    return None


def manifest_to_json(manifest: Manifest) -> str:
    """Serialise the manifest with sorted keys, so two versions diff line by line."""
    content = manifest.model_dump(mode="json", exclude_none=True)
    return json.dumps(content, indent=2, sort_keys=True) + "\n"


def load_manifest(raw_dir: Path) -> Manifest:
    """Read the manifest of ``raw_dir``; an absent manifest is an empty one.

    Raises:
        ValueError: If the manifest file exists but cannot be understood.
    """
    path = manifest_path(raw_dir)
    if not path.is_file():
        return Manifest()
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise ValueError(f"Manifest is not valid, fix or delete it: {path}\n{error}") from error


def save_manifest(raw_dir: Path, manifest: Manifest) -> None:
    """Write the manifest atomically: to a temporary file first, then rename."""
    path = manifest_path(raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = temp_path_for(path)
    temp_path.write_bytes(manifest_to_json(manifest).encode("utf-8"))
    temp_path.replace(path)
