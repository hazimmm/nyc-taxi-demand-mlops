from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from demand.config import CONFIG_FILE_ENV, Settings, get_settings, load_settings
from tests.conftest import CONFIG_FILE


def _write_config(tmp_path: Path, **section_overrides: dict[str, object]) -> Path:
    content = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    for section, values in section_overrides.items():
        content[section].update(values)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_default_config_loads() -> None:
    settings = load_settings(CONFIG_FILE)

    assert settings.project.timezone == "America/New_York"
    assert settings.features.horizon_hours == 24


def test_months_cover_the_configured_period(settings: Settings) -> None:
    assert settings.data.months == [(2025, m) for m in range(1, 7)]
    assert settings.data.start_date == date(2025, 1, 1)
    assert settings.data.end_date == date(2025, 6, 30)
    assert settings.data.replay_start_date == date(2025, 5, 1)


def test_months_roll_over_the_year_end(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        data={"start_month": "2024-11", "end_month": "2025-02", "train_end_date": "2024-12-31"},
    )

    data = load_settings(path).data

    assert data.months == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]
    assert data.end_date == date(2025, 2, 28)


def test_layer_paths_hang_off_the_data_dir(settings: Settings) -> None:
    paths = settings.paths

    assert paths.raw == paths.data_dir / "raw"
    assert paths.silver == paths.data_dir / "silver"
    assert paths.gold == paths.data_dir / "gold"
    assert paths.predictions == paths.data_dir / "predictions"


def test_environment_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEMAND_MLFLOW__TRACKING_URI", "http://mlflow:5000")

    settings = load_settings(CONFIG_FILE)

    assert settings.mlflow.tracking_uri == "http://mlflow:5000"
    assert settings.mlflow.experiment_name == "taxi-demand"


def test_config_file_env_var_is_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_config(tmp_path, project={"seed": 7})
    monkeypatch.setenv(CONFIG_FILE_ENV, str(path))
    get_settings.cache_clear()

    try:
        assert get_settings().project.seed == 7
    finally:
        get_settings.cache_clear()


def test_lag_shorter_than_horizon_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, features={"lags": [1, 24]})

    with pytest.raises(ValidationError, match="leak"):
        load_settings(path)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_month": "2025-07"}, "start_month"),
        ({"train_end_date": "2025-07-15"}, "train_end_date"),
        ({"zones": [132, 132]}, "duplicates"),
        ({"start_month": "2025-13"}, "start_month"),
    ],
)
def test_invalid_data_section_is_rejected(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    path = _write_config(tmp_path, data=overrides)

    with pytest.raises(ValidationError, match=message):
        load_settings(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, project={"sead": 1})

    with pytest.raises(ValidationError):
        load_settings(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "absent.yaml")


def test_non_mapping_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        load_settings(path)
