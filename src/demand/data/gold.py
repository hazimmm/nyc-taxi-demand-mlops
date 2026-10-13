"""Build the gold layer: hourly demand per zone and the model-ready feature table.

Layout of the gold folder (``settings.paths.gold``)::

    gold/
      demand_hourly.parquet    one row per frozen zone and hour, pickups + weather + flag
      features.parquet         the same hours with every model input and the target

The path helpers below (``gold_demand_path``, ``gold_features_path``) are the only place
where this layout is written down; later stages import them.

This module only moves data between disk and the pure functions:

* ``aggregate.py`` counts pickups, completes the grid, joins weather, flags outliers and
  ranks zones.
* ``features.py`` holds the one feature function shared by training and serving.

Two commands are served from here:

* ``select-zones`` (``select_zones``) counts the pickups per zone in the training period
  and returns the busiest ``data.n_zones``. It writes nothing and does not touch the
  config: the list is printed, and a person pastes it under ``data.zones``. Freezing the
  list by hand is deliberate. It is a decision that every later stage depends on, so it
  belongs in a reviewed config file and not in a file a pipeline run can overwrite.
* ``gold`` (``build_gold``) builds both tables.

How the gold stage runs:

1. ``data.zones`` must not be empty and every silver file must exist, otherwise nothing
   is started.
2. Trips are counted one month at a time, and only the pickup time and pickup zone are
   read, so the 23 million trips never sit in memory together.
3. Both tables are built and validated with their Pandera schemas *before* either is
   written, so the two files on disk always belong to the same run.
4. Each file is written under a temporary name and then renamed.

The stage has no state of its own: the same silver files and the same config always
give the same gold files, byte for byte.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.aggregate import (
    add_counts,
    build_demand,
    busiest_zones,
    count_hourly_pickups,
    count_pickups_per_zone,
    training_period_end,
)
from demand.data.features import build_feature_table
from demand.data.silver import silver_trips_path, silver_weather_path, write_parquet
from demand.logs import get_logger
from demand.schemas import (
    IS_OUTLIER,
    PICKUP_DATETIME,
    PICKUP_ZONE_ID,
    TARGET,
    ZONE_ID,
    gold_demand_schema,
    gold_features_schema,
)

log = get_logger(__name__)

GOLD_DEMAND_FILE_NAME = "demand_hourly.parquet"
GOLD_FEATURES_FILE_NAME = "features.parquet"

# The only silver trip columns the gold layer needs.
PICKUP_COLUMNS = [PICKUP_DATETIME, PICKUP_ZONE_ID]


class MissingSilverFileError(FileNotFoundError):
    """A silver file that this stage reads is not on disk."""


class ZonesNotSelectedError(ValueError):
    """``data.zones`` is empty, so there is nothing to build the gold layer for."""


class GoldSummary(BaseModel):
    """What the gold stage produced; logged at the end of a run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zones: int
    hours: int
    """Hours of the configured period: rows per zone in the demand table."""
    demand_rows: int
    pickups: int
    """Pickups in the demand table."""
    pickups_outside_grid: int
    """Pickups in the frozen zones whose hour is not in the configured period."""
    zero_pickup_share: float
    """Share of the demand rows with no pickup at all."""
    outlier_hours: int
    feature_rows: int
    rows_dropped_for_history: int
    """Demand rows that are not in the feature table: too early to have full history."""


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def gold_demand_path(gold_dir: Path) -> Path:
    """Where the hourly demand table is stored."""
    return gold_dir / GOLD_DEMAND_FILE_NAME


def gold_features_path(gold_dir: Path) -> Path:
    """Where the model-ready feature table is stored."""
    return gold_dir / GOLD_FEATURES_FILE_NAME


def training_months(settings: Settings) -> list[tuple[int, int]]:
    """The configured months that contain at least one day of the training period."""
    train_end = settings.data.train_end_date
    return [
        (year, month)
        for year, month in settings.data.months
        if (year, month) <= (train_end.year, train_end.month)
    ]


def zones_config_text(pickups_per_zone: dict[int, int]) -> str:
    """The selected zones as a line to paste under ``data:`` in the config file.

    The same list, without the ``zones:`` key, is what ``DEMAND_DATA__ZONES`` accepts.
    """
    zone_ids = ", ".join(str(zone) for zone in pickups_per_zone)
    return f"  zones: [{zone_ids}]"


def summarise(
    demand: pd.DataFrame, features: pd.DataFrame, pickups_counted: int, rows_dropped: int
) -> GoldSummary:
    """Collect the numbers that are logged at the end of a gold run."""
    zones = int(demand[ZONE_ID].nunique())
    pickups = int(demand[TARGET].sum())
    return GoldSummary(
        zones=zones,
        hours=len(demand) // zones,
        demand_rows=len(demand),
        pickups=pickups,
        pickups_outside_grid=pickups_counted - pickups,
        zero_pickup_share=round(float((demand[TARGET] == 0).mean()), 4),
        outlier_hours=int(demand[IS_OUTLIER].sum()),
        feature_rows=len(features),
        rows_dropped_for_history=rows_dropped,
    )


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def check_silver_files_exist(settings: Settings, files: list[Path]) -> None:
    """Stop before any work is done if a silver file is missing.

    Raises:
        MissingSilverFileError: Naming every missing file.
    """
    silver_dir = settings.paths.silver
    missing = [path for path in files if not path.is_file()]
    if missing:
        names = ", ".join(path.relative_to(silver_dir).as_posix() for path in missing)
        raise MissingSilverFileError(
            f"Silver files missing in {silver_dir}: {names}. Run `python -m demand silver` first."
        )


def read_pickups(path: Path) -> pd.DataFrame:
    """Read the pickup time and pickup zone of one silver trip file, and nothing else."""
    return pd.read_parquet(path, columns=PICKUP_COLUMNS)


def select_zones(settings: Settings) -> dict[int, int]:
    """Find the busiest pickup zones of the training period.

    Only the files of months that reach into the training period are read, and inside
    them only the trips that started before the training period ended. Nothing is
    written.

    Returns:
        Zone ID -> pickups in the training period, for ``data.n_zones`` zones, busiest
        first.

    Raises:
        MissingSilverFileError: If a silver trip file of the training period is missing.
        TooFewZonesError: If fewer than ``data.n_zones`` zones have pickups.
    """
    started = time.monotonic()
    data = settings.data
    silver_dir = settings.paths.silver
    files = [
        silver_trips_path(silver_dir, year, month) for year, month in training_months(settings)
    ]
    check_silver_files_exist(settings, files)

    training_end = training_period_end(settings)
    per_file = [count_pickups_per_zone(read_pickups(path), training_end) for path in files]
    pickups_per_zone = pd.concat(per_file).groupby(level=0).sum()
    selected = busiest_zones(pickups_per_zone, data.n_zones)

    log.info(
        "zones selected",
        extra={
            "training_period": f"{data.start_date} to {data.train_end_date}",
            "zones": list(selected),
            "pickups_per_zone": {str(zone): pickups for zone, pickups in selected.items()},
            "zones_with_pickups": len(pickups_per_zone),
            "share_of_all_pickups": round(sum(selected.values()) / int(pickups_per_zone.sum()), 4),
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return selected


def count_silver_pickups(settings: Settings) -> pd.Series:
    """Hourly pickups of the frozen zones over every configured month, file by file."""
    data = settings.data
    counts = []
    for year, month in data.months:
        path = silver_trips_path(settings.paths.silver, year, month)
        counts.append(count_hourly_pickups(read_pickups(path), data.zones))
    return add_counts(counts)


def build_gold(settings: Settings) -> GoldSummary:
    """Build, validate and write the demand table and the feature table.

    Safe to run again: both files are rebuilt from the silver layer and replaced.

    Raises:
        ZonesNotSelectedError: If ``data.zones`` is empty.
        MissingSilverFileError: If a silver file is missing. Nothing is written.
        WeatherCoverageError: If the silver weather does not cover the period.
        pandera.errors.SchemaError: If a table breaks its gold schema. Nothing is
            written.
    """
    started = time.monotonic()
    data = settings.data
    silver_dir = settings.paths.silver
    gold_dir = settings.paths.gold
    if not data.zones:
        raise ZonesNotSelectedError(
            "data.zones is empty. Run `python -m demand select-zones` and paste the "
            "result into configs/config.yaml before building the gold layer."
        )
    trip_files = [silver_trips_path(silver_dir, year, month) for year, month in data.months]
    check_silver_files_exist(settings, [*trip_files, silver_weather_path(silver_dir)])

    counts = count_silver_pickups(settings)
    weather = pd.read_parquet(silver_weather_path(silver_dir))
    demand = build_demand(counts, weather, settings)
    features, rows_dropped = build_feature_table(demand, settings)

    gold_demand_schema(settings).validate(demand)
    gold_features_schema(settings).validate(features)
    write_parquet(demand, gold_demand_path(gold_dir))
    write_parquet(features, gold_features_path(gold_dir))

    summary = summarise(demand, features, int(counts.sum()), rows_dropped)
    log.info(
        "gold finished",
        extra={
            "gold_dir": str(gold_dir),
            **summary.model_dump(),
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return summary
