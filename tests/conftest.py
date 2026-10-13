"""Fixtures shared by every test. Tests never touch the network or the real data folders."""

from __future__ import annotations

from pathlib import Path

import pytest

from demand.config import FeatureConfig, Settings, load_settings

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


def with_features(
    settings: Settings,
    *,
    horizon_hours: int,
    lags: list[int],
    rolling_windows: list[int] | None = None,
) -> Settings:
    """The same settings with another horizon and other lags (and windows, if given).

    A test whose result depends on the horizon or on which lags exist names them here,
    so it keeps testing the same thing when ``configs/config.yaml`` changes. The new
    values go through the checks of the config: a lag shorter than the horizon raises
    ``pydantic.ValidationError`` here, as it would when the file is loaded.
    """
    values = settings.features.model_dump() | {"horizon_hours": horizon_hours, "lags": lags}
    if rolling_windows is not None:
        values["rolling_windows"] = rolling_windows
    return settings.model_copy(update={"features": FeatureConfig.model_validate(values)})
