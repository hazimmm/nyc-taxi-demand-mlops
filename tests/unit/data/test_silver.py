from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pandera.errors
import pyarrow.parquet as pq
import pytest

from demand.config import Settings
from demand.data import silver
from demand.data.clean import (
    NEGATIVE_TOTAL_AMOUNT,
    NONEXISTENT_LOCAL_TIME,
    NOT_RECORDED_COLUMNS,
    PICKUP_OUTSIDE_MONTH,
    RULE_ORDER,
    UNKNOWN_PICKUP_ZONE,
    RawFormatError,
)
from demand.data.clean_weather import WeatherGapError
from demand.data.ingest import trips_path
from demand.data.pipeline import run_silver
from demand.data.silver import (
    MissingRawFileError,
    cleaning_report_path,
    silver_trips_path,
    silver_weather_path,
)
from demand.data.weather import weather_path
from demand.schemas import (
    HOUR_START,
    PICKUP_DATETIME,
    TOTAL_AMOUNT,
    silver_trips_schema,
    silver_weather_schema,
)
from tests.unit.data.conftest import raw_trip, raw_trips_table, weather_payload, with_months

# February and March 2025: two files, and March contains the night the clocks go forward.
FEBRUARY = [
    raw_trip("2025-02-10 08:00:00"),
    raw_trip("2025-02-10 09:00:00", passenger_count=None),
    raw_trip("2025-01-31 23:55:00"),  # belongs to January
    raw_trip("2025-02-11 08:00:00", total_amount=-3.0),
]
MARCH = [
    raw_trip("2025-03-09 01:30:00"),
    raw_trip("2025-03-09 02:30:00"),  # the clocks skipped this time
    raw_trip("2025-03-09 03:30:00"),
    raw_trip("2025-03-20 18:00:00", PULocationID=265),
    raw_trip("2025-03-21 18:00:00"),
]
PERIOD_HOURS = (28 + 31) * 24 - 1


@pytest.fixture
def two_months(settings: Settings) -> Settings:
    """Settings for February and March 2025 with tiny raw files already on disk."""
    settings = with_months(settings, "2025-02", "2025-03")
    _write_raw_trips(settings, 2025, 2, FEBRUARY)
    _write_raw_trips(settings, 2025, 3, MARCH)
    _write_raw_weather(settings, weather_payload("2025-01-31T00:00", 61 * 24, _variables(settings)))
    return settings


def _variables(settings: Settings) -> list[str]:
    return settings.data.weather.variables


def _write_raw_trips(
    settings: Settings, year: int, month: int, trips: list[dict[str, object]]
) -> None:
    path = trips_path(settings.paths.raw, settings.data.tlc_url_template, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(raw_trips_table(trips), path)


def _write_raw_weather(settings: Settings, payload: dict[str, object]) -> None:
    path = weather_path(settings.paths.raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _files_with_content(*folders: Path) -> dict[str, bytes]:
    return {
        path.relative_to(folder).as_posix(): path.read_bytes()
        for folder in folders
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _report(settings: Settings) -> dict[str, object]:
    content = cleaning_report_path(settings.paths.reports_dir).read_text(encoding="utf-8")
    report: dict[str, object] = json.loads(content)
    return report


def _removed(**counts: int) -> dict[str, int]:
    return dict.fromkeys(RULE_ORDER, 0) | counts


NOTHING_UNRECORDED = dict.fromkeys(NOT_RECORDED_COLUMNS, 0)


# --- layout -----------------------------------------------------------------------------


def test_path_helpers_describe_the_silver_layout(tmp_path: Path) -> None:
    assert silver_trips_path(tmp_path, 2025, 1) == tmp_path / "trips" / "trips_2025-01.parquet"
    assert silver_weather_path(tmp_path) == tmp_path / "weather" / "weather_hourly.parquet"
    assert cleaning_report_path(tmp_path) == tmp_path / "silver_cleaning_report.json"


def test_silver_creates_the_documented_layout(two_months: Settings) -> None:
    run_silver(two_months)

    assert sorted(_files_with_content(two_months.paths.silver)) == [
        "trips/trips_2025-02.parquet",
        "trips/trips_2025-03.parquet",
        "weather/weather_hourly.parquet",
    ]
    assert sorted(_files_with_content(two_months.paths.reports_dir)) == [
        "silver_cleaning_report.json"
    ]


# --- content ----------------------------------------------------------------------------


def test_stored_trips_follow_the_silver_schema(two_months: Settings) -> None:
    run_silver(two_months)

    schema = silver_trips_schema(two_months)
    february = pd.read_parquet(silver_trips_path(two_months.paths.silver, 2025, 2))
    march = pd.read_parquet(silver_trips_path(two_months.paths.silver, 2025, 3))
    schema.validate(february)
    schema.validate(march)
    assert len(february) == 2
    assert (february[TOTAL_AMOUNT] >= 0).all()
    assert list(march[PICKUP_DATETIME]) == [
        pd.Timestamp("2025-03-09 01:30:00-05:00"),
        pd.Timestamp("2025-03-09 03:30:00-04:00"),
        pd.Timestamp("2025-03-21 18:00:00-04:00"),
    ]


def test_stored_weather_follows_the_silver_schema(two_months: Settings) -> None:
    run_silver(two_months)

    weather = pd.read_parquet(silver_weather_path(two_months.paths.silver))
    silver_weather_schema(two_months).validate(weather)
    assert len(weather) == PERIOD_HOURS
    assert weather[HOUR_START].iloc[0] == pd.Timestamp("2025-02-01 00:00:00-05:00")
    assert weather[HOUR_START].iloc[-1] == pd.Timestamp("2025-03-31 23:00:00-04:00")


def test_report_counts_every_month_and_the_total(two_months: Settings) -> None:
    run_silver(two_months)

    assert _report(two_months) == {
        "rule_order": list(RULE_ORDER),
        "months": {
            "2025-02": {
                "rows_in": 4,
                "removed": _removed(**{PICKUP_OUTSIDE_MONTH: 1, NEGATIVE_TOTAL_AMOUNT: 1}),
                "rows_out": 2,
                "not_recorded": NOTHING_UNRECORDED,
            },
            "2025-03": {
                "rows_in": 5,
                "removed": _removed(**{NONEXISTENT_LOCAL_TIME: 1, UNKNOWN_PICKUP_ZONE: 1}),
                "rows_out": 3,
                "not_recorded": NOTHING_UNRECORDED,
            },
        },
        "total": {
            "rows_in": 9,
            "removed": _removed(
                **{
                    PICKUP_OUTSIDE_MONTH: 1,
                    NONEXISTENT_LOCAL_TIME: 1,
                    NEGATIVE_TOTAL_AMOUNT: 1,
                    UNKNOWN_PICKUP_ZONE: 1,
                }
            ),
            "rows_out": 5,
            "not_recorded": NOTHING_UNRECORDED,
        },
        "weather": {
            "hours": PERIOD_HOURS,
            "interpolated": dict.fromkeys(_variables(two_months), 0),
        },
    }


def test_report_lists_the_rules_in_their_order_of_precedence(two_months: Settings) -> None:
    run_silver(two_months)

    total = _report(two_months)["total"]
    assert isinstance(total, dict)
    assert list(total["removed"]) == list(RULE_ORDER)


def test_interpolated_weather_is_reported(two_months: Settings) -> None:
    payload = weather_payload("2025-01-31T00:00", 61 * 24, _variables(two_months), nulls=[500])
    _write_raw_weather(two_months, payload)

    run_silver(two_months)

    assert _report(two_months)["weather"] == {
        "hours": PERIOD_HOURS,
        "interpolated": dict.fromkeys(_variables(two_months), 1),
    }


def test_totals_are_logged(two_months: Settings, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    run_silver(two_months)

    assert caplog.messages.count("trips cleaned") == 2
    assert caplog.messages.count("weather cleaned") == 1
    finished = next(record for record in caplog.records if record.message == "silver finished")
    assert finished.__dict__["rows_in"] == 9
    assert finished.__dict__["rows_out"] == 5
    assert finished.__dict__["removed"][NEGATIVE_TOTAL_AMOUNT] == 1
    assert finished.__dict__["weather_hours"] == PERIOD_HOURS


# --- running again and failing ----------------------------------------------------------


def test_second_run_gives_identical_files(two_months: Settings) -> None:
    folders = (two_months.paths.silver, two_months.paths.reports_dir)
    run_silver(two_months)
    first_run = _files_with_content(*folders)

    run_silver(two_months)

    assert _files_with_content(*folders) == first_run


def test_silver_never_changes_the_raw_layer(two_months: Settings) -> None:
    before = _files_with_content(two_months.paths.raw)

    run_silver(two_months)

    assert _files_with_content(two_months.paths.raw) == before


def test_missing_raw_files_are_named_and_nothing_is_written(two_months: Settings) -> None:
    trips_path(two_months.paths.raw, two_months.data.tlc_url_template, 2025, 3).unlink()
    weather_path(two_months.paths.raw).unlink()

    with pytest.raises(MissingRawFileError) as error:
        run_silver(two_months)

    message = str(error.value)
    assert "trips/yellow_tripdata_2025-03.parquet" in message
    assert "weather/open_meteo_hourly.json" in message
    assert "yellow_tripdata_2025-02" not in message
    assert "python -m demand ingest" in message
    assert not two_months.paths.silver.exists()
    assert not two_months.paths.reports_dir.exists()


def test_trips_that_break_the_schema_stop_the_stage_and_are_not_written(
    two_months: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_clean_trips = silver.clean_trips

    def clean_trips_with_a_bug(*args: object) -> object:
        trips, counts = real_clean_trips(*args)  # type: ignore[arg-type]
        trips[TOTAL_AMOUNT] = -1.0
        return trips, counts

    monkeypatch.setattr(silver, "clean_trips", clean_trips_with_a_bug)

    with pytest.raises(pandera.errors.SchemaError, match=TOTAL_AMOUNT):
        run_silver(two_months)

    assert not (two_months.paths.silver / "trips").exists()
    assert not cleaning_report_path(two_months.paths.reports_dir).exists()


def test_long_weather_gap_stops_the_stage_before_anything_is_written(
    two_months: Settings,
) -> None:
    payload = weather_payload(
        "2025-01-31T00:00", 61 * 24, _variables(two_months), absent=list(range(500, 510))
    )
    _write_raw_weather(two_months, payload)

    with pytest.raises(WeatherGapError, match="10 hours in a row"):
        run_silver(two_months)

    assert _files_with_content(two_months.paths.silver) == {}


def test_raw_trips_without_a_needed_column_name_the_file_and_what_to_do(
    two_months: Settings,
) -> None:
    path = trips_path(two_months.paths.raw, two_months.data.tlc_url_template, 2025, 3)
    pq.write_table(raw_trips_table(MARCH).drop_columns(["total_amount"]), path)

    with pytest.raises(RawFormatError) as error:
        run_silver(two_months)

    message = str(error.value)
    assert message.startswith("yellow_tripdata_2025-03.parquet: raw trips have no column")
    assert "total_amount" in message
    assert "python -m demand ingest" in message


def test_raw_weather_that_is_not_json_names_the_file_and_what_to_do(
    two_months: Settings,
) -> None:
    weather_path(two_months.paths.raw).write_bytes(b"<html>not the weather</html>")

    with pytest.raises(RawFormatError) as error:
        run_silver(two_months)

    message = str(error.value)
    assert message.startswith("open_meteo_hourly.json: not valid JSON")
    assert "python -m demand ingest" in message
    assert not two_months.paths.silver.exists()


def test_raw_weather_with_another_structure_names_the_file_and_what_to_do(
    two_months: Settings,
) -> None:
    _write_raw_weather(two_months, {"daily": {}})

    with pytest.raises(RawFormatError) as error:
        run_silver(two_months)

    message = str(error.value)
    assert message.startswith("open_meteo_hourly.json: weather response has no 'hourly' section")
    assert "python -m demand ingest" in message


def test_failed_write_leaves_no_temporary_file_and_keeps_the_old_one(
    two_months: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_silver(two_months)
    before = _files_with_content(two_months.paths.silver)

    def failing_write(self: pd.DataFrame, path: Path, **options: object) -> None:
        Path(path).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", failing_write)

    with pytest.raises(OSError, match="disk full"):
        run_silver(two_months)

    assert _files_with_content(two_months.paths.silver) == before
