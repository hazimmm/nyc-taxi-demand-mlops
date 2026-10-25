from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "audit_dependencies", ROOT / "tools" / "audit_dependencies.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_tool()


def _ignore_file(tmp_path: Path, body: str) -> Path:
    path = tmp_path / ".trivyignore.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_an_entry_applies_until_the_day_it_expires(tmp_path: Path) -> None:
    path = _ignore_file(
        tmp_path,
        "vulnerabilities:\n"
        "  - id: CVE-1\n    expired_at: 2026-11-04\n"
        "  - id: CVE-2\n    expired_at: 2026-12-01\n",
    )

    assert audit.accepted_ids(path, dt.date(2026, 11, 3)) == ["CVE-1", "CVE-2"]
    assert audit.accepted_ids(path, dt.date(2026, 11, 4)) == ["CVE-2"]
    assert audit.accepted_ids(path, dt.date(2026, 12, 1)) == []


def test_an_entry_without_an_expiry_day_is_refused(tmp_path: Path) -> None:
    path = _ignore_file(tmp_path, "vulnerabilities:\n  - id: CVE-1\n")

    with pytest.raises(ValueError, match="CVE-1"):
        audit.accepted_ids(path, dt.date(2026, 11, 1))


def test_the_command_audits_the_pinned_lock_without_the_accepted_ids() -> None:
    command = audit.pip_audit_command(Path("requirements.lock"), ["CVE-1", "GHSA-x"])

    assert command[1:5] == ["-m", "pip_audit", "--requirement", "requirements.lock"]
    assert "--no-deps" in command and "--disable-pip" in command
    assert command[-4:] == ["--ignore-vuln", "CVE-1", "--ignore-vuln", "GHSA-x"]


def test_every_entry_of_the_project_file_has_a_reason_and_an_expiry_day() -> None:
    import yaml

    entries = yaml.safe_load((ROOT / ".trivyignore.yaml").read_text(encoding="utf-8"))
    for entry in entries["vulnerabilities"]:
        assert entry["statement"].strip(), entry["id"]
        assert isinstance(entry["expired_at"], dt.date), entry["id"]
