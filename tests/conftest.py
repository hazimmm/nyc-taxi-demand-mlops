"""Fixtures shared by every test. Tests never touch the network or the real data folders."""

from __future__ import annotations

from pathlib import Path

import pytest

from demand.config import Settings, load_settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "configs" / "config.yaml"

TEST_ZONES = [132, 161, 237]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Project settings with a small frozen zone list and all paths under ``tmp_path``."""
    base = load_settings(CONFIG_FILE)
    return base.model_copy(
        update={
            "paths": base.paths.model_copy(
                update={"data_dir": tmp_path / "data", "reports_dir": tmp_path / "reports"}
            ),
            "data": base.data.model_copy(update={"zones": TEST_ZONES}),
        }
    )
