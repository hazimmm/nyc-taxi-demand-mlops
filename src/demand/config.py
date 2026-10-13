"""Typed application settings.

Values come from a YAML file (``configs/config.yaml`` by default, or the file named by
``DEMAND_CONFIG_FILE``) and can be overridden by environment variables prefixed with
``DEMAND_``, using a double underscore between nested keys. The same variables can be
kept in a ``.env`` file in the working directory.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

CONFIG_FILE_ENV = "DEMAND_CONFIG_FILE"
DEFAULT_CONFIG_FILE = Path("configs/config.yaml")
DOTENV_FILE = Path(".env")

Month = Annotated[str, Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]


class _Section(BaseModel):
    """Base class for config sections: immutable and strict about unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ProjectConfig(_Section):
    """Project-wide constants."""

    name: str
    seed: int
    timezone: str


class PathsConfig(_Section):
    """Filesystem locations, relative to the working directory unless absolute."""

    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")

    @property
    def raw(self) -> Path:
        """Raw (bronze) layer: files exactly as downloaded."""
        return self.data_dir / "raw"

    @property
    def silver(self) -> Path:
        """Silver layer: cleaned, typed, de-duplicated tables."""
        return self.data_dir / "silver"

    @property
    def gold(self) -> Path:
        """Gold layer: hourly demand and the model-ready feature table."""
        return self.data_dir / "gold"

    @property
    def predictions(self) -> Path:
        """Day-ahead forecasts written by the batch forecast flow."""
        return self.data_dir / "predictions"


class WeatherConfig(_Section):
    """Open-Meteo request parameters."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    forecast_url: str
    archive_url: str
    variables: list[str] = Field(min_length=1)


class DownloadConfig(_Section):
    """Retry and timeout limits for downloads into the raw layer."""

    max_attempts: int = Field(default=4, gt=0)
    backoff_base_seconds: float = Field(default=1.0, ge=0)
    connect_timeout_seconds: float = Field(default=10.0, gt=0)
    read_timeout_seconds: float = Field(default=60.0, gt=0)


class DataConfig(_Section):
    """What to ingest and where the training period ends."""

    tlc_url_template: str
    zone_lookup_url: str
    start_month: Month
    end_month: Month
    train_end_date: date
    download: DownloadConfig = DownloadConfig()
    n_zones: int = Field(gt=0)
    zones: list[int] = Field(default_factory=list)
    weather: WeatherConfig

    @model_validator(mode="after")
    def _check_period(self) -> DataConfig:
        if self.start_month > self.end_month:
            raise ValueError("data.start_month must not be after data.end_month")
        if not self.start_date <= self.train_end_date < self.end_date:
            raise ValueError("data.train_end_date must fall inside the ingested period")
        if self.zones and len(set(self.zones)) != len(self.zones):
            raise ValueError("data.zones must not contain duplicates")
        return self

    @property
    def months(self) -> list[tuple[int, int]]:
        """Every ``(year, month)`` from ``start_month`` to ``end_month`` inclusive."""
        year, month = (int(part) for part in self.start_month.split("-"))
        end = tuple(int(part) for part in self.end_month.split("-"))
        result: list[tuple[int, int]] = []
        while (year, month) <= end:
            result.append((year, month))
            year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        return result

    @property
    def start_date(self) -> date:
        """First day of the ingested period."""
        year, month = self.months[0]
        return date(year, month, 1)

    @property
    def end_date(self) -> date:
        """Last day of the ingested period."""
        year, month = self.months[-1]
        next_month = date(year + 1, 1, 1) if month == 12 else date(year, month + 1, 1)
        return next_month - timedelta(days=1)

    @property
    def replay_start_date(self) -> date:
        """First day replayed as "production" by the simulated clock."""
        return self.train_end_date + timedelta(days=1)


class CleaningConfig(_Section):
    """Thresholds for the trip-level cleaning rules that build the silver layer."""

    min_duration_minutes: float = Field(ge=0)
    max_duration_hours: float = Field(gt=0)
    max_distance_miles: float = Field(gt=0)
    min_passengers: int = Field(ge=0)
    max_passengers: int = Field(gt=0)
    max_zone_id: int = Field(gt=0)
    max_weather_gap_hours: int = Field(ge=0)


class FeatureConfig(_Section):
    """Feature definitions. Nothing may look closer to the target than the horizon."""

    horizon_hours: int = Field(gt=0)
    lags: list[int] = Field(min_length=1)
    # A standard deviation needs at least two values.
    rolling_windows: list[Annotated[int, Field(ge=2)]] = Field(min_length=1)
    outlier_zscore: float = Field(gt=0)

    @model_validator(mode="after")
    def _check_no_leakage(self) -> FeatureConfig:
        too_recent = [lag for lag in self.lags if lag < self.horizon_hours]
        if too_recent:
            raise ValueError(
                f"features.lags {too_recent} are shorter than the {self.horizon_hours} h "
                "forecast horizon and would leak information unavailable at prediction time"
            )
        return self


class ModelConfig(_Section):
    """LightGBM parameters and hyper-parameter search budget."""

    params: dict[str, Any]
    tuning_trials: int = Field(ge=0)
    tuning_timeout_seconds: int = Field(gt=0)


class BacktestConfig(_Section):
    """Rolling-origin backtest layout."""

    n_folds: int = Field(gt=0)
    fold_days: int = Field(gt=0)


class MlflowConfig(_Section):
    """Experiment tracking and model registry."""

    tracking_uri: str
    experiment_name: str
    registered_model_name: str
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"


class ServingConfig(_Section):
    """API settings. ``model_uri`` is a registry alias or a local model directory."""

    model_uri: str
    host: str = "0.0.0.0"
    port: int = Field(default=8000, gt=0, lt=65536)


class MonitoringConfig(_Section):
    """Drift and accuracy thresholds and retraining safety limits."""

    reference_days: int = Field(gt=0)
    window_days: int = Field(gt=0)
    drift_share_threshold: float = Field(gt=0, le=1)
    mae_degradation_ratio: float = Field(gt=1)
    retrain_cooldown_days: int = Field(ge=0)


class Settings(BaseSettings):
    """Complete, validated application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="DEMAND_",
        env_nested_delimiter="__",
        frozen=True,
        extra="forbid",
    )

    project: ProjectConfig
    paths: PathsConfig = PathsConfig()
    data: DataConfig
    cleaning: CleaningConfig
    features: FeatureConfig
    model: ModelConfig
    backtest: BacktestConfig
    mlflow: MlflowConfig
    serving: ServingConfig
    monitoring: MonitoringConfig

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Let environment variables override the YAML values passed to ``__init__``."""
        return env_settings, init_settings


def load_settings(config_file: Path | None = None) -> Settings:
    """Load settings from ``config_file``, ``DEMAND_CONFIG_FILE`` or the default path.

    A ``.env`` file in the working directory is read first. Variables that are already
    set in the real environment win over it, so the order of precedence is: environment,
    ``.env``, YAML.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file does not contain a YAML mapping.
    """
    load_dotenv(DOTENV_FILE, override=False)
    path = config_file or Path(os.environ.get(CONFIG_FILE_ENV, DEFAULT_CONFIG_FILE))
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path.resolve()}")
    return Settings(**content)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, loaded once."""
    return load_settings()
