from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from demand.config import Settings
from demand.data import gold
from demand.data.aggregate import TooFewZonesError, WeatherCoverageError
from demand.data.clean_weather import local_hours
from demand.data.features import warm_up_hours
from demand.data.gold import (
    MissingSilverFileError,
    ZonesNotSelectedError,
    gold_demand_path,
    gold_features_path,
    select_zones,
    training_months,
    zones_config_text,
)
from demand.data.pipeline import run_gold, run_select_zones
from demand.data.silver import silver_trips_path, silver_weather_path, write_parquet
from demand.schemas import (
    HOUR_START,
    IS_OUTLIER,
    TARGET,
    ZONE_ID,
    feature_columns,
    gold_demand_schema,
    gold_features_schema,
    lag_column,
)
from tests.conftest import with_features
from tests.unit.data.conftest import silver_trips_frame

# February and March 2025: two silver files, and March contains the 23-hour day.
# The training period is February; March is "production".
PERIOD_HOURS = (28 + 31) * 24 - 1
FEBRUARY = [
    ("2025-02-10 08:05:00-05:00", 161),
    ("2025-02-10 08:40:00-05:00", 161),
    ("2025-02-10 09:00:00-05:00", 161),
    ("2025-02-10 08:10:00-05:00", 132),
    ("2025-02-11 18:00:00-05:00", 132),
    ("2025-02-12 07:30:00-05:00", 237),
    ("2025-02-12 07:45:00-05:00", 50),  # not a frozen zone
]
MARCH = [
    ("2025-03-09 01:30:00-05:00", 161),
    ("2025-03-09 03:30:00-04:00", 161),
    ("2025-03-11 08:15:00-04:00", 161),  # the pickup the lag test reads back
    ("2025-03-12 08:20:00-04:00", 237),
    *[("2025-03-20 12:00:00-04:00", 50)] * 30,  # busy, but only after the training period
]
PICKUPS_IN_FROZEN_ZONES = 10


@pytest.fixture
def two_months(settings: Settings) -> Settings:
    """Settings for February and March 2025 with tiny silver files already on disk."""
    data = settings.data.model_copy(
        update={
            "start_month": "2025-02",
            "end_month": "2025-03",
            "train_end_date": date(2025, 2, 28),
            "n_zones": 2,
        }
    )
    settings = settings.model_copy(update={"data": data})
    silver_dir = settings.paths.silver
    write_parquet(silver_trips_frame(settings, FEBRUARY), silver_trips_path(silver_dir, 2025, 2))
    # March trips have no dropoff time and no distance: gold must not need them.
    write_parquet(
        silver_trips_frame(settings, MARCH, complete=False),
        silver_trips_path(silver_dir, 2025, 3),
    )
    _write_silver_weather(settings)
    return settings


def _write_silver_weather(settings: Settings, *, hours_to_drop: int = 0) -> None:
    data = settings.data
    hours = local_hours(data.start_date, data.end_date, settings.project.timezone)
    values = {
        name: [position / 10 for position in range(len(hours))] for name in _variables(settings)
    }
    weather = pd.DataFrame({HOUR_START: hours, **values})
    kept = weather.iloc[: len(weather) - hours_to_drop]
    write_parquet(kept, silver_weather_path(settings.paths.silver))


def _variables(settings: Settings) -> list[str]:
    return settings.data.weather.variables


def _without_zones(settings: Settings) -> Settings:
    return settings.model_copy(update={"data": settings.data.model_copy(update={"zones": []})})


def _files_with_content(folder: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _demand(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(gold_demand_path(settings.paths.gold))


def _features(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(gold_features_path(settings.paths.gold))


def _pickups(table: pd.DataFrame, zone: int, hour: str) -> int:
    rows = table[(table[ZONE_ID] == zone) & (table[HOUR_START] == pd.Timestamp(hour))]
    assert len(rows) == 1
    return int(rows[TARGET].iloc[0])


# --- layout -----------------------------------------------------------------------------


def test_path_helpers_describe_the_gold_layout(tmp_path: Path) -> None:
    assert gold_demand_path(tmp_path) == tmp_path / "demand_hourly.parquet"
    assert gold_features_path(tmp_path) == tmp_path / "features.parquet"


def test_gold_creates_the_documented_layout(two_months: Settings) -> None:
    run_gold(two_months)

    assert sorted(_files_with_content(two_months.paths.gold)) == [
        "demand_hourly.parquet",
        "features.parquet",
    ]


# --- the demand table -------------------------------------------------------------------


def test_stored_demand_follows_the_gold_schema_on_a_complete_grid(two_months: Settings) -> None:
    run_gold(two_months)

    demand = _demand(two_months)
    schema = gold_demand_schema(two_months)
    schema.validate(demand)
    assert list(demand.columns) == list(schema.columns)
    assert len(demand) == 3 * PERIOD_HOURS
    assert demand.groupby(ZONE_ID).size().to_dict() == dict.fromkeys([132, 161, 237], PERIOD_HOURS)
    assert demand[HOUR_START].min() == pd.Timestamp("2025-02-01 00:00:00-05:00")
    assert demand[HOUR_START].max() == pd.Timestamp("2025-03-31 23:00:00-04:00")
    assert demand.notna().all().all()


def test_stored_demand_counts_the_pickups_of_the_frozen_zones(two_months: Settings) -> None:
    run_gold(two_months)

    demand = _demand(two_months)
    assert int(demand[TARGET].sum()) == PICKUPS_IN_FROZEN_ZONES
    assert _pickups(demand, 161, "2025-02-10 08:00:00-05:00") == 2
    assert _pickups(demand, 161, "2025-02-10 09:00:00-05:00") == 1
    assert _pickups(demand, 132, "2025-02-10 08:00:00-05:00") == 1
    assert _pickups(demand, 237, "2025-02-10 08:00:00-05:00") == 0
    # Trips without a dropoff time or a distance, on both sides of the skipped hour.
    assert _pickups(demand, 161, "2025-03-09 01:00:00-05:00") == 1
    assert _pickups(demand, 161, "2025-03-09 03:00:00-04:00") == 1


def test_the_day_the_clocks_change_has_23_rows_per_zone(two_months: Settings) -> None:
    run_gold(two_months)

    demand = _demand(two_months)
    on_the_day = demand[demand[HOUR_START].dt.date == date(2025, 3, 9)]
    assert on_the_day.groupby(ZONE_ID).size().tolist() == [23, 23, 23]


def test_stored_demand_has_the_weather_of_each_hour(two_months: Settings) -> None:
    run_gold(two_months)

    demand = _demand(two_months)
    for _, zone_rows in demand.groupby(ZONE_ID):
        for name in _variables(two_months):
            assert zone_rows[name].tolist() == [position / 10 for position in range(PERIOD_HOURS)]


# --- the feature table ------------------------------------------------------------------


def test_stored_features_follow_the_gold_schema(two_months: Settings) -> None:
    run_gold(two_months)

    features = _features(two_months)
    schema = gold_features_schema(two_months)
    schema.validate(features)
    assert list(features.columns) == list(schema.columns)
    assert set(feature_columns(two_months)) < set(features.columns)
    assert features.notna().all().all()


def test_stored_features_lack_only_the_warm_up_hours(two_months: Settings) -> None:
    warm_up = warm_up_hours(two_months)

    run_gold(two_months)

    features = _features(two_months)
    assert len(features) == 3 * (PERIOD_HOURS - warm_up)
    first_hour = pd.Timestamp("2025-02-01 00:00:00-05:00") + pd.Timedelta(hours=warm_up)
    assert features[HOUR_START].min() == first_hour


@pytest.mark.parametrize(
    ("lag", "later_hour"),
    [
        # Zone 161 has one pickup in the hour of Tuesday 11 March 08:00 and none in the
        # two days after it.
        (24, "2025-03-12 08:00:00-04:00"),
        (48, "2025-03-13 08:00:00-04:00"),
    ],
)
def test_stored_lag_is_the_pickup_count_of_that_many_hours_earlier(
    two_months: Settings, lag: int, later_hour: str
) -> None:
    with_lag = with_features(two_months, horizon_hours=lag, lags=[lag, 168])

    run_gold(with_lag)

    features = _features(with_lag)
    later = features[
        (features[ZONE_ID] == 161) & (features[HOUR_START] == pd.Timestamp(later_hour))
    ]
    assert later[lag_column(lag)].tolist() == [1.0]
    assert later[TARGET].tolist() == [0]


# --- logging ----------------------------------------------------------------------------


def test_summary_is_logged(two_months: Settings, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    warm_up = warm_up_hours(two_months)

    run_gold(two_months)

    finished = next(record for record in caplog.records if record.message == "gold finished")
    fields = finished.__dict__
    assert fields["zones"] == 3
    assert fields["hours"] == PERIOD_HOURS
    assert fields["demand_rows"] == 3 * PERIOD_HOURS
    assert fields["pickups"] == PICKUPS_IN_FROZEN_ZONES
    assert fields["pickups_outside_grid"] == 0
    assert fields["zero_pickup_share"] == round(1 - 9 / (3 * PERIOD_HOURS), 4)
    assert fields["outlier_hours"] == int(_demand(two_months)[IS_OUTLIER].sum())
    assert fields["feature_rows"] == 3 * (PERIOD_HOURS - warm_up)
    assert fields["rows_dropped_for_history"] == 3 * warm_up


# --- running again and failing ----------------------------------------------------------


def test_second_run_gives_identical_files(two_months: Settings) -> None:
    run_gold(two_months)
    first_run = _files_with_content(two_months.paths.gold)

    run_gold(two_months)

    assert _files_with_content(two_months.paths.gold) == first_run


def test_gold_never_changes_the_silver_layer(two_months: Settings) -> None:
    before = _files_with_content(two_months.paths.silver)

    run_gold(two_months)

    assert _files_with_content(two_months.paths.silver) == before


def test_missing_silver_files_are_named_and_nothing_is_written(two_months: Settings) -> None:
    silver_trips_path(two_months.paths.silver, 2025, 3).unlink()
    silver_weather_path(two_months.paths.silver).unlink()

    with pytest.raises(MissingSilverFileError) as error:
        run_gold(two_months)

    message = str(error.value)
    assert "trips/trips_2025-03.parquet" in message
    assert "weather/weather_hourly.parquet" in message
    assert "trips_2025-02" not in message
    assert "python -m demand silver" in message
    assert not two_months.paths.gold.exists()


def test_empty_zone_list_stops_the_stage_with_a_clear_message(two_months: Settings) -> None:
    with pytest.raises(ZonesNotSelectedError, match="python -m demand select-zones"):
        run_gold(_without_zones(two_months))

    assert not two_months.paths.gold.exists()


def test_weather_that_does_not_cover_the_period_stops_the_stage(two_months: Settings) -> None:
    _write_silver_weather(two_months, hours_to_drop=5)

    with pytest.raises(WeatherCoverageError, match="15 rows"):
        run_gold(two_months)

    assert not two_months.paths.gold.exists()


def test_table_that_breaks_its_schema_stops_the_stage_and_nothing_is_written(
    two_months: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_build_feature_table = gold.build_feature_table
    broken_column = lag_column(two_months.features.lags[0])

    def build_feature_table_with_a_bug(*args: object) -> object:
        table, rows_dropped = real_build_feature_table(*args)  # type: ignore[arg-type]
        table[broken_column] = -1.0
        return table, rows_dropped

    monkeypatch.setattr(gold, "build_feature_table", build_feature_table_with_a_bug)

    with pytest.raises(pandera.errors.SchemaError, match=broken_column):
        run_gold(two_months)

    assert not two_months.paths.gold.exists()


# --- select-zones -----------------------------------------------------------------------


def test_training_months_stop_at_the_month_of_the_training_end(
    settings: Settings, two_months: Settings
) -> None:
    assert training_months(settings) == [(2025, 1), (2025, 2), (2025, 3), (2025, 4)]
    assert training_months(two_months) == [(2025, 2)]


def test_zones_are_selected_from_the_training_period_only(two_months: Settings) -> None:
    # Zone 50 has 30 pickups in March and 1 in February; 161 has 3 and 132 has 2.
    assert select_zones(two_months) == {161: 3, 132: 2}


def test_zone_selection_does_not_need_the_frozen_list_or_the_replay_files(
    two_months: Settings,
) -> None:
    silver_trips_path(two_months.paths.silver, 2025, 3).unlink()
    silver_weather_path(two_months.paths.silver).unlink()

    assert list(select_zones(_without_zones(two_months))) == [161, 132]


def test_trips_after_the_training_end_inside_a_training_month_are_not_counted(
    two_months: Settings,
) -> None:
    data = two_months.data.model_copy(update={"train_end_date": date(2025, 2, 10)})
    shorter_training = two_months.model_copy(update={"data": data})

    # 132 has one pickup on 10 February and one on the 11th; 237 and 50 start on the 12th.
    with pytest.raises(TooFewZonesError):
        select_zones(
            shorter_training.model_copy(update={"data": data.model_copy(update={"n_zones": 3})})
        )
    assert select_zones(shorter_training) == {161: 3, 132: 1}


def test_zone_selection_writes_nothing(two_months: Settings) -> None:
    before = _files_with_content(two_months.paths.data_dir)

    run_select_zones(two_months)

    assert _files_with_content(two_months.paths.data_dir) == before


def test_select_zones_prints_a_line_for_the_config_and_logs_the_counts(
    two_months: Settings, capsys: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    run_select_zones(two_months)

    output = capsys.readouterr().out.splitlines()
    assert output[-1] == "  zones: [161, 132]"
    assert "2025-02-01 to 2025-02-28" in output[0]
    selected = next(record for record in caplog.records if record.message == "zones selected")
    assert selected.__dict__["zones"] == [161, 132]
    assert selected.__dict__["pickups_per_zone"] == {"161": 3, "132": 2}
    assert selected.__dict__["zones_with_pickups"] == 4
    assert selected.__dict__["share_of_all_pickups"] == round(5 / 7, 4)


def test_zone_selection_names_missing_silver_files(two_months: Settings) -> None:
    silver_trips_path(two_months.paths.silver, 2025, 2).unlink()

    with pytest.raises(MissingSilverFileError, match=r"trips_2025-02.*python -m demand silver"):
        run_select_zones(two_months)


def test_config_text_lists_the_zones_in_order_of_rank() -> None:
    assert zones_config_text({161: 30, 132: 20, 237: 10}) == "  zones: [161, 132, 237]"
