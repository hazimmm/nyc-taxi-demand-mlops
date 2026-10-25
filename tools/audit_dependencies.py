r"""Check the lock files for known vulnerabilities, with pip-audit.

Run from the project folder::

    .venv\Scripts\python.exe tools\audit_dependencies.py

It audits ``requirements.lock`` and the two lock files of the images, and exits with
pip-audit's code: 0 when nothing is found, 1 when something is.

Findings that the project accepts for a limited time are listed in
``.trivyignore.yaml``, the file Trivy reads when it scans the images, so that both
scanners accept the same list. Each entry has an ``id`` and an ``expired_at`` day. An
entry is passed to pip-audit as ``--ignore-vuln`` only *before* that day, which is also
how Trivy reads it: on the day an entry expires, the finding comes back and the check
fails. pip-audit matches the id against a finding's own id and its aliases, so a CVE
or GHSA id works for both tools.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / ".trivyignore.yaml"
LOCK_FILES = (
    ROOT / "requirements.lock",
    ROOT / "docker" / "requirements-api.lock",
    ROOT / "docker" / "requirements-pipeline.lock",
)


def accepted_ids(ignore_file: Path, today: dt.date) -> list[str]:
    """The ids of ``ignore_file`` whose ``expired_at`` day has not been reached."""
    entries = (yaml.safe_load(ignore_file.read_text(encoding="utf-8")) or {}).get(
        "vulnerabilities", []
    )
    ids = []
    for entry in entries:
        expires = entry.get("expired_at")
        if expires is None:
            raise ValueError(f"{entry['id']} in {ignore_file.name} has no expired_at day")
        if isinstance(expires, str):
            expires = dt.date.fromisoformat(expires)
        if today < expires:
            ids.append(entry["id"])
    return ids


def pip_audit_command(lock_file: Path, ids: list[str]) -> list[str]:
    """The pip-audit call that audits ``lock_file`` without the accepted ``ids``."""
    # --no-deps and --disable-pip: the lock is complete and pinned, so pip-audit checks
    # exactly the versions in it instead of resolving them again.
    command = [sys.executable, "-m", "pip_audit", "--requirement", str(lock_file)]
    command += ["--no-deps", "--disable-pip", "--progress-spinner", "off"]
    for vuln_id in ids:
        command += ["--ignore-vuln", vuln_id]
    return command


def main(argv: list[str] | None = None) -> int:
    """Audit each lock file; the worst exit code of pip-audit."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("lock_files", nargs="*", type=Path, default=list(LOCK_FILES))
    args = parser.parse_args(argv)

    ids = accepted_ids(IGNORE_FILE, dt.datetime.now(dt.UTC).date())
    worst = 0
    for lock_file in args.lock_files:
        print(f"== {lock_file.relative_to(ROOT) if lock_file.is_absolute() else lock_file}")
        sys.stdout.flush()
        worst = max(
            worst, subprocess.run(pip_audit_command(lock_file, ids), check=False).returncode
        )
    return worst


if __name__ == "__main__":
    sys.exit(main())
