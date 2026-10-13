from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from demand.data.manifest import (
    Manifest,
    ManifestEntry,
    download_reason,
    load_manifest,
    manifest_path,
    manifest_to_json,
    save_manifest,
    sha256_of_file,
)

URL = "https://example.test/file.csv"
SHA_A = "a" * 64
SHA_B = "b" * 64


def _entry(**changes: object) -> ManifestEntry:
    values: dict[str, object] = {
        "url": URL,
        "downloaded_at": datetime(2025, 7, 1, 12, 0, tzinfo=UTC),
        "size_bytes": 10,
        "sha256": SHA_A,
    }
    return ManifestEntry.model_validate({**values, **changes})


def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "file.bin"
    path.write_bytes(b"taxi" * 1000)

    assert sha256_of_file(path) == hashlib.sha256(b"taxi" * 1000).hexdigest()


def test_no_download_when_file_matches_the_manifest() -> None:
    assert download_reason(_entry(), SHA_A, {URL}) is None


def test_download_when_file_is_missing() -> None:
    assert download_reason(_entry(), None, {URL}) == "file is missing"


def test_download_when_file_is_not_in_the_manifest() -> None:
    assert download_reason(None, SHA_A, {URL}) == "file is not in the manifest"


def test_download_when_checksum_differs() -> None:
    assert download_reason(_entry(), SHA_B, {URL}) == "checksum differs from the manifest"


def test_download_when_the_configured_url_changed() -> None:
    assert download_reason(_entry(), SHA_A, {"https://example.test/other.csv"}) == (
        "source URL changed"
    )


def test_manifest_survives_a_save_and_load(tmp_path: Path) -> None:
    manifest = Manifest(files={"trips/a.parquet": _entry(row_count=3)})

    save_manifest(tmp_path, manifest)

    assert load_manifest(tmp_path) == manifest


def test_missing_manifest_loads_as_empty(tmp_path: Path) -> None:
    assert load_manifest(tmp_path) == Manifest()


def test_manifest_json_has_sorted_keys_and_utc_time() -> None:
    manifest = Manifest(files={"zones/z.csv": _entry(), "trips/a.parquet": _entry(row_count=3)})

    text = manifest_to_json(manifest)

    content = json.loads(text)
    assert list(content["files"]) == ["trips/a.parquet", "zones/z.csv"]
    assert list(content["files"]["trips/a.parquet"]) == [
        "downloaded_at",
        "row_count",
        "sha256",
        "size_bytes",
        "url",
    ]
    assert content["files"]["zones/z.csv"]["downloaded_at"] == "2025-07-01T12:00:00Z"
    assert text.endswith("}\n")


def test_saving_the_manifest_leaves_no_temporary_file(tmp_path: Path) -> None:
    save_manifest(tmp_path, Manifest(files={"zones/z.csv": _entry()}))

    assert [path.name for path in tmp_path.iterdir()] == ["manifest.json"]


def test_broken_manifest_raises_an_error_naming_the_file(tmp_path: Path) -> None:
    manifest_path(tmp_path).write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match=r"manifest\.json"):
        load_manifest(tmp_path)


def test_manifest_entry_rejects_a_malformed_checksum() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _entry(sha256="not-a-checksum")
