from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data import validate
from demand.data.clean import RULE_ORDER, CleaningCounts, total_counts
from demand.data.clean_weather import WeatherSummary
from demand.data.features import build_feature_table, warm_up_hours
from demand.data.gold import ZonesNotSelectedError, gold_demand_path, gold_features_path
from demand.data.silver import (
    CleaningReport,
    cleaning_report_path,
    report_to_json,
    silver_trips_path,
    silver_weather_path,
    write_parquet,
    write_text,
)
from demand.data.validate import (
    NOT_CHECKED,
    CheckResult,
    GateReport,
    examples,
    failure_message,
    validate_layer,
    validation_report_path,
)
from demand.errors import QualityGateError
from demand.schemas import (
    DROPOFF_DATETIME,
    DURATION_MINUTES,
    HOUR_START,
    IS_OUTLIER,
    PICKUP_ZONE_ID,
    TARGET,
    TOTAL_AMOUNT,
    TRIP_DISTANCE,
    ZONE_ID,
    lag_column,
)
from tests.unit.data.conftest import demand_frame, silver_trips_frame, silver_weather_frame

# February and March 2025: two silver trip files, and March contains the 23-hour day.
PERIOD_HOURS = (28 + 31) * 24 - 1
FEBRUARY = [
    ("2025-02-01 00:00:00-05:00", 161),  # the first moment of the month
    ("2025-02-10 08:05:00-05:00", 161),
    ("2025-02-28 23:59:59-05:00", 132),  # the last second of the month
]
MARCH = [
    ("2025-03-09 01:30:00-05:00", 161),
    ("2025-03-09 03:30:00-04:00", 237),
    ("2025-03-31 23:00:00-04:00", 50),
]
FIRST_HOUR = pd.Timestamp("2025-02-01 00:00:00-05:00")

SILVER_CHECKS = [
    "cleaning_report.file",
    "cleaning_report.months",
    "cleaning_report.total",
    "weather.file",
    "weather.schema",
    "weather.covers_period",
    "weather.hours_match_report",
    "trips_2025-02.file",
    "trips_2025-02.schema",
    "trips_2025-02.pickups_in_month",
    "trips_2025-02.duration_matches_times",
    "trips_2025-02.counts_match_report",
    "trips_2025-03.file",
    "trips_2025-03.schema",
    "trips_2025-03.pickups_in_month",
    "trips_2025-03.duration_matches_times",
    "trips_2025-03.counts_match_report",
]
GOLD_CHECKS = [
    "demand.file",
    "demand.schema",
    "demand.complete_grid",
    "demand.weather_same_for_all_zones",
    "features.file",
    "features.schema",
    "features.columns",
    "features.rows",
    "features.matches_demand",
    "features.lags_match_demand",
]


# --- a good layer on disk, and helpers to damage one file of it ---------------------------


@pytest.fixture
def two_months(settings: Settings) -> Settings:
    """Settings for February and March 2025; nothing is on disk yet."""
    data = settings.data.model_copy(
        update={
            "start_month": "2025-02",
            "end_month": "2025-03",
            "train_end_date": date(2025, 2, 28),
        }
    )
    return settings.model_copy(update={"data": data})


def _counts(rows_out: int, *, not_recorded: int = 0) -> CleaningCounts:
    return CleaningCounts(
        rows_in=rows_out,
        removed=dict.fromkeys(RULE_ORDER, 0),
        rows_out=rows_out,
        not_recorded={DROPOFF_DATETIME: not_recorded, TRIP_DISTANCE: not_recorded},
    )


def _good_report() -> CleaningReport:
    # The March trips have no dropoff time and no distance (``complete=False`` below).
    months = {"2025-02": _counts(len(FEBRUARY)), "2025-03": _counts(len(MARCH), not_recorded=3)}
    return CleaningReport(
        rule_order=list(RULE_ORDER),
        months=months,
        total=total_counts(list(months.values())),
        weather=WeatherSummary(hours=PERIOD_HOURS, interpolated={}),
    )


def _write_report(settings: Settings, report: CleaningReport) -> None:
    write_text(report_to_json(report), cleaning_report_path(settings.paths.reports_dir))


def _write_trips(settings: Settings, month: int, trips: pd.DataFrame) -> None:
    write_parquet(trips, silver_trips_path(settings.paths.silver, 2025, month))


def _write_weather(settings: Settings, weather: pd.DataFrame) -> None:
    write_parquet(weather, silver_weather_path(settings.paths.silver))


@pytest.fixture
def good_silver(two_months: Settings) -> Settings:
    """A silver layer that passes its gate: two trip files, weather and the report."""
    _write_trips(two_months, 2, silver_trips_frame(two_months, FEBRUARY))
    _write_trips(two_months, 3, silver_trips_frame(two_months, MARCH, complete=False))
    _write_weather(two_months, silver_weather_frame(two_months))
    _write_report(two_months, _good_report())
    return two_months


def _good_demand(settings: Settings) -> pd.DataFrame:
    return demand_frame(settings, "2025-02-01", PERIOD_HOURS)


def _write_gold(settings: Settings, demand: pd.DataFrame, features: pd.DataFrame) -> None:
    write_parquet(demand, gold_demand_path(settings.paths.gold))
    write_parquet(features, gold_features_path(settings.paths.gold))


@pytest.fixture
def good_gold(two_months: Settings) -> Settings:
    """A gold layer that passes its gate; the features come from the real function."""
    demand = _good_demand(two_months)
    features, _ = build_feature_table(demand, two_months)
    _write_gold(two_months, demand, features)
    return two_months


def _demand(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(gold_demand_path(settings.paths.gold))


def _features(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(gold_features_path(settings.paths.gold))


def _stored_report(settings: Settings, layer: str) -> GateReport:
    path = validation_report_path(settings.paths.reports_dir, layer)
    return GateReport.model_validate_json(path.read_bytes())


def _failures(settings: Settings, layer: str) -> dict[str, str]:
    """Run a gate that must fail; return the failed checks and their details."""
    with pytest.raises(QualityGateError):
        validate_layer(settings, layer)
    report = _stored_report(settings, layer)
    assert not report.passed
    return {check.name: check.detail for check in report.checks if not check.passed}


def _not_checked(*names: str) -> dict[str, str]:
    return dict.fromkeys(names, NOT_CHECKED)


# --- a layer that passes -------------------------------------------------------------------


def test_report_path_describes_the_layout(tmp_path: Path) -> None:
    assert validation_report_path(tmp_path, "silver") == tmp_path / "silver_validation_report.json"
    assert validation_report_path(tmp_path, "gold") == tmp_path / "gold_validation_report.json"


def test_good_silver_layer_passes_every_check(good_silver: Settings) -> None:
    report = validate_layer(good_silver, "silver")

    assert report.layer == "silver"
    assert report.passed
    assert [check.name for check in report.checks] == SILVER_CHECKS
    assert all(check.passed for check in report.checks)


def test_good_gold_layer_passes_every_check(good_gold: Settings) -> None:
    report = validate_layer(good_gold, "gold")

    assert report.layer == "gold"
    assert report.passed
    assert [check.name for check in report.checks] == GOLD_CHECKS
    assert all(check.passed for check in report.checks)


def test_report_on_disk_is_the_returned_report(good_silver: Settings) -> None:
    report = validate_layer(good_silver, "silver")

    assert _stored_report(good_silver, "silver") == report
    by_name = {check.name: check.detail for check in report.checks}
    assert by_name["trips_2025-02.file"] == "3 rows"
    assert by_name["weather.file"] == f"{PERIOD_HOURS} rows"
    assert by_name["weather.covers_period"] == "ok"


def test_second_run_gives_an_identical_report(good_gold: Settings) -> None:
    path = validation_report_path(good_gold.paths.reports_dir, "gold")
    validate_layer(good_gold, "gold")
    first_run = path.read_bytes()

    validate_layer(good_gold, "gold")

    assert path.read_bytes() == first_run


def test_gate_changes_no_data(good_silver: Settings) -> None:
    folder = good_silver.paths.data_dir
    before = {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()}

    validate_layer(good_silver, "silver")

    assert {path: path.read_bytes() for path in folder.rglob("*") if path.is_file()} == before


def test_passing_gate_logs_one_summary_line(
    good_gold: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    validate_layer(good_gold, "gold")

    (passed,) = [record for record in caplog.records if record.name == validate.log.name]
    assert passed.message == "quality gate passed"
    assert passed.__dict__["layer"] == "gold"
    assert passed.__dict__["checks"] == len(GOLD_CHECKS)
    assert passed.__dict__["report"].endswith("gold_validation_report.json")


# --- silver: files that are missing or unreadable ------------------------------------------


def test_missing_trip_file_is_a_failed_check_not_a_crash(good_silver: Settings) -> None:
    path = silver_trips_path(good_silver.paths.silver, 2025, 3)
    path.unlink()

    failures = _failures(good_silver, "silver")

    assert failures.pop("trips_2025-03.file") == (
        f"{path.as_posix()} is missing. Run `python -m demand silver` to write it again."
    )
    assert failures == _not_checked(
        "trips_2025-03.schema",
        "trips_2025-03.pickups_in_month",
        "trips_2025-03.duration_matches_times",
        "trips_2025-03.counts_match_report",
    )


def test_file_that_is_not_parquet_is_a_failed_check(good_silver: Settings) -> None:
    silver_weather_path(good_silver.paths.silver).write_bytes(b"this is not a Parquet file")

    failures = _failures(good_silver, "silver")

    assert "cannot be read as Parquet" in failures.pop("weather.file")
    assert failures == _not_checked(
        "weather.schema", "weather.covers_period", "weather.hours_match_report"
    )


def test_empty_layer_fails_every_check(two_months: Settings) -> None:
    failures = _failures(two_months, "silver")

    assert sorted(failures) == sorted(SILVER_CHECKS)


# --- silver: trips -------------------------------------------------------------------------


def test_trips_that_break_the_schema_report_every_failure_at_once(good_silver: Settings) -> None:
    trips = silver_trips_frame(good_silver, FEBRUARY)
    trips[PICKUP_ZONE_ID] = [264, 300, 132]
    trips.loc[2, TOTAL_AMOUNT] = -1.0
    _write_trips(good_silver, 2, trips)

    failures = _failures(good_silver, "silver")

    # Lazy validation: the zone failures did not hide the failure of the amount.
    # (Pandera lists all failing values in one column, so the zones show as 264.0.)
    assert failures.pop("trips_2025-02.schema") == (
        "pickup_zone_id: in_range(1, 263) (2 failing, e.g. 264.0, 300.0); "
        "total_amount: greater_than_or_equal_to(0) (1 failing, e.g. -1.0)"
    )
    assert failures == _not_checked(
        "trips_2025-02.pickups_in_month",
        "trips_2025-02.duration_matches_times",
        "trips_2025-02.counts_match_report",
    )


def test_trips_without_a_schema_column_fail_the_schema(good_silver: Settings) -> None:
    trips = silver_trips_frame(good_silver, FEBRUARY).drop(columns=[TOTAL_AMOUNT])
    _write_trips(good_silver, 2, trips)

    failures = _failures(good_silver, "silver")

    assert failures["trips_2025-02.schema"] == (
        "table: column_in_dataframe (1 failing, e.g. total_amount)"
    )


def test_pickup_of_another_month_fails_the_month_check(good_silver: Settings) -> None:
    # The first moment of March does not belong to February.
    pickups = [*FEBRUARY[:2], ("2025-03-01 00:00:00-05:00", 161)]
    _write_trips(good_silver, 2, silver_trips_frame(good_silver, pickups))

    failures = _failures(good_silver, "silver")

    assert failures == {
        "trips_2025-02.pickups_in_month": (
            "1 pickup outside 2025-02, e.g. 2025-03-01 00:00:00-05:00"
        )
    }


def test_pickup_before_the_month_fails_the_month_check(good_silver: Settings) -> None:
    pickups = [("2025-02-28 23:59:59-05:00", 161), *MARCH[:2]]
    _write_trips(good_silver, 3, silver_trips_frame(good_silver, pickups, complete=False))

    failures = _failures(good_silver, "silver")

    assert list(failures) == ["trips_2025-03.pickups_in_month"]


def test_duration_that_disagrees_with_the_times_fails(good_silver: Settings) -> None:
    trips = silver_trips_frame(good_silver, FEBRUARY)
    trips.loc[1, DURATION_MINUTES] = 16.0
    _write_trips(good_silver, 2, trips)

    failures = _failures(good_silver, "silver")

    assert failures == {
        "trips_2025-02.duration_matches_times": (
            "1 row with a duration that is not dropoff minus pickup, e.g. at row 1"
        )
    }


def test_duration_without_a_dropoff_time_fails(good_silver: Settings) -> None:
    trips = silver_trips_frame(good_silver, MARCH, complete=False)
    trips[DURATION_MINUTES] = 15.0
    _write_trips(good_silver, 3, trips)

    failures = _failures(good_silver, "silver")

    assert list(failures) == ["trips_2025-03.duration_matches_times"]
    assert failures["trips_2025-03.duration_matches_times"].startswith("3 rows with a duration")


def test_row_count_that_differs_from_the_report_fails(good_silver: Settings) -> None:
    _write_trips(good_silver, 2, silver_trips_frame(good_silver, FEBRUARY[:2]))

    failures = _failures(good_silver, "silver")

    assert failures == {
        "trips_2025-02.counts_match_report": "rows_out: the file has 2, the report says 3"
    }


def test_not_recorded_count_that_differs_from_the_report_fails(good_silver: Settings) -> None:
    # The report says that no March trip has a dropoff time or a distance.
    _write_trips(good_silver, 3, silver_trips_frame(good_silver, MARCH))

    failures = _failures(good_silver, "silver")

    assert failures == {
        "trips_2025-03.counts_match_report": (
            "not_recorded.dropoff_datetime: the file has 0, the report says 3; "
            "not_recorded.trip_distance: the file has 0, the report says 3"
        )
    }


# --- silver: weather -----------------------------------------------------------------------


def test_weather_with_missing_hours_fails_the_coverage_check(good_silver: Settings) -> None:
    weather = silver_weather_frame(good_silver).iloc[:-2]
    _write_weather(good_silver, weather)

    failures = _failures(good_silver, "silver")

    assert failures == {
        "weather.covers_period": (
            "2 hours missing, e.g. 2025-03-31 22:00:00-04:00, 2025-03-31 23:00:00-04:00"
        ),
        "weather.hours_match_report": (
            f"the file has {PERIOD_HOURS - 2} rows, the report says {PERIOD_HOURS} hours"
        ),
    }


def test_weather_with_an_hour_outside_the_period_fails_the_coverage_check(
    good_silver: Settings,
) -> None:
    weather = silver_weather_frame(good_silver)
    weather.loc[0, HOUR_START] = pd.Timestamp("2025-04-01 00:00:00-04:00")
    _write_weather(good_silver, weather)

    failures = _failures(good_silver, "silver")

    assert failures == {
        "weather.covers_period": (
            "1 hour missing, e.g. 2025-02-01 00:00:00-05:00; "
            "1 hour not expected, e.g. 2025-04-01 00:00:00-04:00"
        )
    }


def test_weather_with_a_repeated_hour_fails_the_schema(good_silver: Settings) -> None:
    weather = silver_weather_frame(good_silver)
    weather.loc[1, HOUR_START] = weather.loc[0, HOUR_START]
    _write_weather(good_silver, weather)

    failures = _failures(good_silver, "silver")

    assert failures.pop("weather.schema") == (
        "hour_start: multiple_fields_uniqueness (2 failing, e.g. 2025-02-01 00:00:00-05:00)"
    )
    assert failures == _not_checked("weather.covers_period", "weather.hours_match_report")


def test_weather_hours_that_differ_from_the_report_fail(good_silver: Settings) -> None:
    report = _good_report()
    _write_report(
        good_silver,
        report.model_copy(update={"weather": WeatherSummary(hours=10, interpolated={})}),
    )

    failures = _failures(good_silver, "silver")

    assert failures == {
        "weather.hours_match_report": f"the file has {PERIOD_HOURS} rows, the report says 10 hours"
    }


# --- silver: the cleaning report -----------------------------------------------------------


def test_missing_cleaning_report_is_a_failed_check_not_a_crash(good_silver: Settings) -> None:
    path = cleaning_report_path(good_silver.paths.reports_dir)
    path.unlink()

    failures = _failures(good_silver, "silver")

    assert failures == {
        "cleaning_report.file": (
            f"{path.as_posix()} is missing. Run `python -m demand silver` to write it again."
        ),
        "cleaning_report.months": NOT_CHECKED,
        "cleaning_report.total": NOT_CHECKED,
        "weather.hours_match_report": "the cleaning report cannot be read",
        "trips_2025-02.counts_match_report": (
            "the cleaning report has no readable counts for this month"
        ),
        "trips_2025-03.counts_match_report": (
            "the cleaning report has no readable counts for this month"
        ),
    }


def test_file_that_is_not_a_cleaning_report_is_a_failed_check(good_silver: Settings) -> None:
    cleaning_report_path(good_silver.paths.reports_dir).write_bytes(b'{"rule_order": []}')

    failures = _failures(good_silver, "silver")

    assert (
        "is not a cleaning report (3 errors, first: Field required)"
        in (failures["cleaning_report.file"])
    )
    assert failures["cleaning_report.months"] == NOT_CHECKED


def test_report_for_other_months_fails_the_months_check(good_silver: Settings) -> None:
    report = _good_report()
    months = {"2025-02": report.months["2025-02"], "2025-04": report.months["2025-03"]}
    _write_report(good_silver, report.model_copy(update={"months": months}))

    failures = _failures(good_silver, "silver")

    assert failures == {
        "cleaning_report.months": (
            "1 month missing, e.g. 2025-03; 1 month not expected, e.g. 2025-04"
        ),
        "trips_2025-03.counts_match_report": (
            "the cleaning report has no readable counts for this month"
        ),
    }


def test_total_that_is_not_the_sum_of_the_months_fails(good_silver: Settings) -> None:
    _write_report(good_silver, _good_report().model_copy(update={"total": _counts(100)}))

    failures = _failures(good_silver, "silver")

    assert failures == {
        "cleaning_report.total": (
            "rows_in: the months add up to 6, the total says 100; "
            "rows_out: the months add up to 6, the total says 100"
        )
    }


# --- gold: the demand table ----------------------------------------------------------------


def test_missing_gold_file_is_a_failed_check_not_a_crash(good_gold: Settings) -> None:
    path = gold_features_path(good_gold.paths.gold)
    path.unlink()

    failures = _failures(good_gold, "gold")

    assert failures.pop("features.file") == (
        f"{path.as_posix()} is missing. Run `python -m demand gold` to write it again."
    )
    assert failures == _not_checked(
        "features.schema",
        "features.columns",
        "features.rows",
        "features.matches_demand",
        "features.lags_match_demand",
    )


def test_demand_that_breaks_the_schema_stops_the_checks_that_need_it(good_gold: Settings) -> None:
    demand = _demand(good_gold)
    demand.loc[0, TARGET] = -1
    _write_gold(good_gold, demand, _features(good_gold))

    failures = _failures(good_gold, "gold")

    assert failures.pop("demand.schema") == (
        "pickups: greater_than_or_equal_to(0) (1 failing, e.g. -1)"
    )
    # The feature table itself is fine, so its own checks ran and passed.
    assert failures == _not_checked(
        "demand.complete_grid",
        "demand.weather_same_for_all_zones",
        "features.matches_demand",
        "features.lags_match_demand",
    )


def test_demand_without_one_hour_of_a_zone_fails_the_grid_check(good_gold: Settings) -> None:
    # The first hour of a zone is in no feature row, so only the grid notices.
    demand = _demand(good_gold).iloc[1:]
    _write_gold(good_gold, demand, _features(good_gold))

    failures = _failures(good_gold, "gold")

    assert failures == {
        "demand.complete_grid": "1 row missing, e.g. 132 / 2025-02-01 00:00:00-05:00"
    }


def test_demand_with_an_hour_outside_the_period_fails_the_grid_check(good_gold: Settings) -> None:
    demand = _demand(good_gold)
    after_the_period = demand.iloc[[-1]].assign(
        **{HOUR_START: pd.Timestamp("2025-04-01 00:00:00-04:00")}
    )
    _write_gold(
        good_gold, pd.concat([demand, after_the_period], ignore_index=True), _features(good_gold)
    )

    failures = _failures(good_gold, "gold")

    assert failures == {
        "demand.complete_grid": "1 row not expected, e.g. 237 / 2025-04-01 00:00:00-04:00"
    }


def test_weather_that_differs_between_zones_fails(good_gold: Settings) -> None:
    variable = good_gold.data.weather.variables[0]
    demand = _demand(good_gold)
    demand.loc[0, variable] += 1.0
    _write_gold(good_gold, demand, _features(good_gold))

    failures = _failures(good_gold, "gold")

    assert failures == {
        "demand.weather_same_for_all_zones": (
            "1 hour with different weather in different zones, e.g. 2025-02-01 00:00:00-05:00"
        )
    }


# --- gold: the feature table ---------------------------------------------------------------


def test_missing_value_in_the_features_fails_the_schema(good_gold: Settings) -> None:
    features = _features(good_gold)
    features.loc[5, lag_column(24)] = np.nan
    _write_gold(good_gold, _demand(good_gold), features)

    failures = _failures(good_gold, "gold")

    assert failures.pop("features.schema") == "lag_24h: not_nullable (1 failing, e.g. nan)"
    assert failures == _not_checked(
        "features.columns",
        "features.rows",
        "features.matches_demand",
        "features.lags_match_demand",
    )


def test_features_with_a_column_the_model_does_not_ask_for_fail_the_schema(
    good_gold: Settings,
) -> None:
    features = _features(good_gold).assign(extra_input=1.0)
    _write_gold(good_gold, _demand(good_gold), features)

    failures = _failures(good_gold, "gold")

    assert failures["features.schema"] == "table: column_in_schema (1 failing, e.g. extra_input)"


def test_model_input_that_the_table_lacks_fails_the_columns_check(
    good_gold: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The schema and ``feature_columns`` are built from the same config, so they can
    # only disagree if one of them is changed in code. That is what this simulates.
    real_feature_columns = validate.feature_columns
    monkeypatch.setattr(
        validate, "feature_columns", lambda settings: [*real_feature_columns(settings), "new_input"]
    )

    failures = _failures(good_gold, "gold")

    assert failures == {"features.columns": "1 column missing, e.g. new_input"}


def test_features_without_a_row_fail_the_rows_check(good_gold: Settings) -> None:
    features = _features(good_gold).iloc[1:]
    _write_gold(good_gold, _demand(good_gold), features)

    failures = _failures(good_gold, "gold")

    first_hour_with_history = FIRST_HOUR + pd.Timedelta(hours=warm_up_hours(good_gold))
    assert failures == {"features.rows": f"1 row missing, e.g. 132 / {first_hour_with_history}"}


def test_features_with_a_warm_up_hour_fail_the_rows_check(good_gold: Settings) -> None:
    features = _features(good_gold)
    warm_up_row = features.iloc[[0]].assign(**{HOUR_START: FIRST_HOUR})
    _write_gold(
        good_gold, _demand(good_gold), pd.concat([features, warm_up_row], ignore_index=True)
    )

    failures = _failures(good_gold, "gold")

    assert failures["features.rows"] == f"1 row not expected, e.g. 132 / {FIRST_HOUR}"


def test_target_that_differs_from_the_demand_table_fails(good_gold: Settings) -> None:
    features = _features(good_gold)
    features.loc[0, TARGET] += 1
    features.loc[1, IS_OUTLIER] = True
    _write_gold(good_gold, _demand(good_gold), features)

    failures = _failures(good_gold, "gold")

    assert list(failures) == ["features.matches_demand"]
    assert failures["features.matches_demand"].startswith(
        "2 rows different from the demand table in pickups, is_outlier, temperature_2m"
    )


def test_lag_that_is_not_the_earlier_pickup_count_fails(good_gold: Settings) -> None:
    features = _features(good_gold)
    # Every 24 h lag looks one hour too far back: 25 hours instead of 24.
    features[lag_column(24)] = features.groupby(ZONE_ID)[lag_column(24)].shift(1).bfill()
    _write_gold(good_gold, _demand(good_gold), features)

    failures = _failures(good_gold, "gold")

    assert list(failures) == ["features.lags_match_demand"]
    assert failures["features.lags_match_demand"].startswith("lag_24h: ")
    assert "not equal to the pickups 24 h earlier" in failures["features.lags_match_demand"]
    assert "lag_48h" not in failures["features.lags_match_demand"]


def test_lags_are_checked_in_real_hours_across_the_clock_change(good_gold: Settings) -> None:
    # 08:00 on the Sunday the clocks went forward is 24 real hours after 07:00 on Saturday.
    features, demand = _features(good_gold), _demand(good_gold)
    sunday = features[
        (features[ZONE_ID] == 161)
        & (features[HOUR_START] == pd.Timestamp("2025-03-09 08:00:00-04:00"))
    ]
    saturday = demand[
        (demand[ZONE_ID] == 161) & (demand[HOUR_START] == pd.Timestamp("2025-03-08 07:00:00-05:00"))
    ]
    assert sunday[lag_column(24)].tolist() == saturday[TARGET].astype(float).tolist()

    assert validate_layer(good_gold, "gold").passed


# --- what stops a gate before it starts ----------------------------------------------------


def test_gold_gate_needs_the_frozen_zones(good_gold: Settings) -> None:
    data = good_gold.data.model_copy(update={"zones": []})

    with pytest.raises(ZonesNotSelectedError, match="python -m demand select-zones"):
        validate_layer(good_gold.model_copy(update={"data": data}), "gold")


def test_unknown_layer_is_refused(settings: Settings) -> None:
    with pytest.raises(ValueError, match="no quality gate for layer 'bronze'"):
        validate_layer(settings, "bronze")


# --- the error message ---------------------------------------------------------------------


def test_failed_gate_names_the_layer_the_checks_and_the_report(good_silver: Settings) -> None:
    _write_trips(good_silver, 2, silver_trips_frame(good_silver, FEBRUARY[:2]))
    report_path = validation_report_path(good_silver.paths.reports_dir, "silver")

    with pytest.raises(QualityGateError) as error:
        validate_layer(good_silver, "silver")

    assert str(error.value) == (
        "silver quality gate failed: 1 of 17 checks failed "
        "(trips_2025-02.counts_match_report). "
        f"The details are in {report_path.as_posix()}. "
        "Fix the cause, then run `python -m demand silver` and the gate again."
    )
    assert "\n" not in str(error.value)


def _result(name: str, detail: str = "ok") -> CheckResult:
    return CheckResult(name=name, passed=detail == "ok", detail=detail)


def test_message_names_the_first_three_failed_checks_and_counts_the_rest() -> None:
    checks = [_result("a"), *[_result(name, "wrong") for name in "bcdef"]]
    report = GateReport(layer="gold", passed=False, checks=checks)

    message = failure_message(report, Path("reports/gold_validation_report.json"))

    assert "5 of 6 checks failed (b, c, d and 2 more). The details" in message
    assert "reports/gold_validation_report.json" in message


def test_message_names_the_causes_and_only_counts_checks_that_could_not_run() -> None:
    checks = [
        _result("demand.file", "the file is missing"),
        _result("demand.schema", NOT_CHECKED),
        _result("demand.complete_grid", NOT_CHECKED),
        _result("features.schema", "lag_24h: not_nullable"),
    ]
    report = GateReport(layer="gold", passed=False, checks=checks)

    message = failure_message(report, Path("report.json"))

    assert (
        "2 of 4 checks failed (demand.file, features.schema), and 2 more could not be checked."
        in message
    )


# --- small helpers -------------------------------------------------------------------------


def test_examples_show_the_first_three_different_values() -> None:
    assert examples([264, 264, 300, 1, 2]) == "264, 300, 1"
    assert examples([(161, "08:00"), (132, "09:00")]) == "161 / 08:00, 132 / 09:00"
    assert examples([]) == ""
