from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount

from demand.config import Settings
from demand.monitoring.checks import (
    ACCURACY,
    BASELINE,
    DAY,
    DRIFT_METHOD,
    DRIFT_THRESHOLD,
    INPUT_DRIFT,
    PREDICTION_DRIFT,
    UNTESTED,
    ColumnDrift,
    MonitoringReport,
    accuracy_check,
    column_drifts,
    drift_columns,
    drift_share,
    drift_snapshot,
    input_drift_check,
    monitoring_report,
    prediction_drift_check,
)
from demand.schemas import (
    ACTUAL,
    FORECAST,
    IS_HOLIDAY,
    MODEL_VERSION,
    feature_columns,
    lag_column,
    rolling_mean_column,
)
from tests.unit.monitoring.conftest import (
    LEVEL,
    SPREAD,
    days_from,
    grid,
    normal_values,
    numbers,
    training_table,
)

WINDOW_DAYS = days_from(date(2025, 5, 2), 7)


def window(settings: Settings, shift: dict[str, float], seed: int = 7) -> pd.DataFrame:
    return numbers(grid(settings, WINDOW_DAYS), settings, seed, shift)


def reference(settings: Settings) -> pd.DataFrame:
    table = training_table(settings)
    return table.assign(**{DAY: table["hour_start"].dt.date}).iloc[-3 * 24 * 28 :]


def shifted(columns: list[str], spreads: float = 2.0) -> dict[str, float]:
    return dict.fromkeys(columns, spreads * SPREAD)


# --------------------------------------------------------------------------------------
# Which columns
# --------------------------------------------------------------------------------------


def test_drift_columns_are_the_model_inputs_minus_the_calendar_and_zone(settings: Settings) -> None:
    tested = drift_columns(settings)
    assert len(feature_columns(settings)) == 16
    assert len(tested) == 10
    assert set(tested) | set(UNTESTED) == set(feature_columns(settings))
    assert not set(tested) & set(UNTESTED)


# --------------------------------------------------------------------------------------
# Drift of one column
# --------------------------------------------------------------------------------------


def test_a_shifted_column_drifts_and_an_unshifted_one_does_not(settings: Settings) -> None:
    rng = np.random.default_rng(42)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"a": rng.normal(1, 1, 500), "b": rng.normal(0, 1, 500)})

    drifts = column_drifts(drift_snapshot(cur, ref, ["a", "b"]))

    assert [drift.column for drift in drifts] == ["a", "b"]
    assert drifts[0].drifted
    assert drifts[0].distance == pytest.approx(1.0, abs=0.15)
    assert not drifts[1].drifted
    assert drifts[1].distance < DRIFT_THRESHOLD


def test_the_distance_is_the_wasserstein_distance_divided_by_the_reference_spread() -> None:
    ref = pd.DataFrame({"a": [0.0, 2.0] * 50})
    cur = pd.DataFrame({"a": [1.0, 3.0] * 50})  # every value moved by 1; reference std is 1

    (drift,) = column_drifts(drift_snapshot(cur, ref, ["a"]))

    assert drift.distance == pytest.approx(1.0)


def test_a_constant_column_in_both_tables_has_not_drifted() -> None:
    ref = pd.DataFrame({"snow": np.zeros(300)})
    cur = pd.DataFrame({"snow": np.zeros(100)})

    (drift,) = column_drifts(drift_snapshot(cur, ref, ["snow"]))

    assert drift == ColumnDrift(column="snow", distance=0.0, drifted=False)


def test_the_threshold_itself_counts_as_drift() -> None:
    ref = pd.DataFrame({"a": [0.0, 2.0] * 50})
    cur = pd.DataFrame({"a": [0.1, 2.1] * 50})  # moved by exactly 0.1 standard deviation

    (drift,) = column_drifts(drift_snapshot(cur, ref, ["a"]))

    assert drift.distance == pytest.approx(DRIFT_THRESHOLD)
    assert drift.drifted


def test_drift_share_is_drifted_over_tested() -> None:
    drifts = [ColumnDrift(column=str(n), distance=0.0, drifted=n < 3) for n in range(10)]

    assert drift_share(drifts) == 0.3


def test_the_share_agrees_with_evidently(settings: Settings) -> None:
    columns = drift_columns(settings)
    cur = window(settings, shifted(columns[:4]))
    ref = reference(settings)
    definition = DataDefinition(numerical_columns=columns)
    evidently = Report(
        [DriftedColumnsCount(columns=columns, method=DRIFT_METHOD, threshold=DRIFT_THRESHOLD)]
    ).run(
        Dataset.from_pandas(cur[columns], data_definition=definition),
        Dataset.from_pandas(ref[columns], data_definition=definition),
    )

    share = drift_share(column_drifts(drift_snapshot(cur, ref, columns)))

    assert share == pytest.approx(evidently.dict()["metrics"][0]["value"]["share"])
    assert share == 0.4


# --------------------------------------------------------------------------------------
# Check 1: input drift
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("drifting", "drifted"), [(0, False), (2, False), (3, True), (5, True)])
def test_inputs_have_drifted_from_the_share_threshold_on(
    settings: Settings, drifting: int, drifted: bool
) -> None:
    columns = drift_columns(settings)

    check, snapshot = input_drift_check(
        window(settings, shifted(columns[:drifting])), reference(settings), settings, None
    )

    assert check.evaluated
    assert check.drifted_columns == drifting
    assert check.drift_share == drifting / 10
    assert check.share_threshold == settings.monitoring.drift_share_threshold == 0.3
    assert check.drifted is drifted
    assert [drift.drifted for drift in check.columns] == [n < drifting for n in range(10)]
    assert snapshot is not None


def test_input_drift_reports_its_window_reference_and_holidays(settings: Settings) -> None:
    rows = window(settings, {})
    rows.loc[rows[DAY] == date(2025, 5, 5), IS_HOLIDAY] = True

    check, _ = input_drift_check(rows, reference(settings), settings, None)

    assert (check.window_first_day, check.window_last_day) == (date(2025, 5, 2), date(2025, 5, 8))
    assert check.window_rows == 3 * 24 * 7
    assert (check.reference_first_day, check.reference_last_day) == (
        date(2025, 4, 3),
        date(2025, 4, 30),
    )
    assert check.reference_rows == 3 * 24 * 28
    assert check.holidays_in_window == [date(2025, 5, 5)]
    assert check.untested == UNTESTED


def test_input_drift_without_a_window_is_not_evaluated(settings: Settings) -> None:
    check, snapshot = input_drift_check(None, reference(settings), settings, "too few days")

    assert not check.evaluated
    assert check.note == "too few days"
    assert check.drift_share is None
    assert check.drifted is None
    assert snapshot is None


# --------------------------------------------------------------------------------------
# Check 2: prediction drift
# --------------------------------------------------------------------------------------


def forecasts(level: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            FORECAST: normal_values(504, level),
            MODEL_VERSION: "2",
            DAY: np.repeat(WINDOW_DAYS, 72),
        }
    )


def backtest() -> pd.DataFrame:
    return pd.DataFrame({FORECAST: normal_values(2016)})


def test_forecasts_like_the_backtest_have_not_drifted() -> None:
    check = prediction_drift_check(forecasts(LEVEL), backtest(), 20.0, 20.0, None)

    assert check.evaluated
    assert check.model_version == "2"
    assert check.distance is not None
    assert check.distance < DRIFT_THRESHOLD
    assert check.drifted is False
    assert check.reference_rows == 2016


def test_shifted_forecasts_have_drifted() -> None:
    check = prediction_drift_check(forecasts(LEVEL + SPREAD), backtest(), 20.0, 20.0, None)

    assert check.drifted is True
    assert check.distance == pytest.approx(1.0, abs=0.01)


def test_a_reference_whose_mae_differs_from_the_registry_is_not_used() -> None:
    check = prediction_drift_check(forecasts(LEVEL + SPREAD), backtest(), 20.5, 20.0, None)

    assert not check.evaluated
    assert check.drifted is None
    assert check.distance is None
    assert check.note is not None
    assert "20.5" in check.note
    assert "20.0" in check.note


def test_without_a_recorded_backtest_mae_prediction_drift_is_not_evaluated() -> None:
    check = prediction_drift_check(forecasts(LEVEL), None, None, None, None)

    assert not check.evaluated
    assert check.note is not None
    assert "no backtest MAE" in check.note


def test_without_a_window_prediction_drift_keeps_the_reason() -> None:
    check = prediction_drift_check(None, None, None, None, "3 forecast days issued")

    assert not check.evaluated
    assert check.note == "3 forecast days issued"
    assert check.model_version is None
    assert check.window_rows == 0


# --------------------------------------------------------------------------------------
# Check 3: accuracy
# --------------------------------------------------------------------------------------


def scored(forecast: list[float], actual: list[float], baseline: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            DAY: [date(2025, 5, 2), date(2025, 5, 2), date(2025, 5, 3), date(2025, 5, 3)],
            FORECAST: forecast,
            ACTUAL: actual,
            BASELINE: baseline,
            MODEL_VERSION: "2",
        }
    )


def test_accuracy_by_hand(settings: Settings) -> None:
    # Errors 10, 30, 20, 40: MAE 25. Baseline errors 20, 20, 40, 40: MAE 30.
    rows = scored([110, 70, 120, 60], [100, 100, 100, 100], [120, 80, 140, 140])

    check = accuracy_check(rows, 20.0, settings, None)

    assert check.evaluated
    assert check.live_mae == 25.0
    assert check.baseline_mae == 30.0
    assert check.live_to_backtest_mae == 1.25
    assert check.alert  # 1.25 is the threshold, and the threshold itself alerts
    assert (check.first_day, check.last_day, check.days, check.rows) == (
        date(2025, 5, 2),
        date(2025, 5, 3),
        2,
        4,
    )


def test_accuracy_alerts_from_the_ratio_threshold_on(settings: Settings) -> None:
    rows = scored([110, 90, 110, 90], [100, 100, 100, 100], [100, 100, 100, 100])  # MAE 10

    check = accuracy_check(rows, 8.0, settings, None)

    assert check.live_to_backtest_mae == 1.25
    assert check.alert
    assert not accuracy_check(rows, 8.1, settings, None).alert


def test_accuracy_without_rows_is_not_evaluated(settings: Settings) -> None:
    check = accuracy_check(None, 20.0, settings, "2 days over")

    assert not check.evaluated
    assert check.note == "2 days over"
    assert check.live_mae is None
    assert not check.alert


def test_accuracy_without_a_backtest_mae_is_not_evaluated(settings: Settings) -> None:
    rows = scored([110, 90, 110, 90], [100, 100, 100, 100], [100, 100, 100, 100])

    check = accuracy_check(rows, None, settings, None)

    assert not check.evaluated
    assert check.note is not None
    assert "no backtest MAE" in check.note
    assert check.days == 2


# --------------------------------------------------------------------------------------
# The report: alerts and warnings
# --------------------------------------------------------------------------------------


def checks(settings: Settings, *, drift: bool, accuracy_ratio: float) -> tuple:
    """Both drift checks drifted (or not), and accuracy at ``accuracy_ratio``."""
    shift = shifted([lag_column(48), rolling_mean_column(24), drift_columns(settings)[-1]])
    inputs, _ = input_drift_check(
        window(settings, shift if drift else {}), reference(settings), settings, None
    )
    level = LEVEL + SPREAD if drift else LEVEL
    predictions = prediction_drift_check(forecasts(level), backtest(), 20.0, 20.0, None)
    error = 20.0 * accuracy_ratio  # backtest MAE 20, every forecast off by ``error``
    accuracy = accuracy_check(
        scored([100 + error, 100 - error] * 2, [100] * 4, [100] * 4), 20.0, settings, None
    )
    return inputs, predictions, accuracy


def report_of(settings: Settings, *, drift: bool, accuracy_ratio: float) -> MonitoringReport:
    return monitoring_report(
        pd.Timestamp("2025-05-10", tz=settings.project.timezone),
        settings,
        *checks(settings, drift=drift, accuracy_ratio=accuracy_ratio),
    )


def test_drift_is_a_warning_and_accuracy_an_alert(settings: Settings) -> None:
    report = report_of(settings, drift=True, accuracy_ratio=1.5)

    assert report.alert
    assert [alert.check for alert in report.alerts] == [ACCURACY]
    assert "1.5 times the backtest MAE 20.0" in report.alerts[0].reason
    assert report.warning
    assert [warning.check for warning in report.warnings] == [INPUT_DRIFT, PREDICTION_DRIFT]
    assert "3 of 10 model inputs drifted" in report.warnings[0].reason
    assert "lag_48h" in report.warnings[0].reason
    assert "version 2" in report.warnings[1].reason
    assert report.window_days == 7
    assert report.reference_days == 28


def test_drift_alone_never_raises_an_alert(settings: Settings) -> None:
    report = report_of(settings, drift=True, accuracy_ratio=1.0)

    assert report.input_drift.drifted
    assert report.prediction_drift.drifted
    assert not report.alert
    assert report.alerts == []
    assert report.warning


def test_an_accuracy_alert_without_drift_has_no_warning(settings: Settings) -> None:
    report = report_of(settings, drift=False, accuracy_ratio=1.25)

    assert report.alert
    assert not report.warning
    assert report.warnings == []


def test_nothing_when_no_check_was_evaluated(settings: Settings) -> None:
    inputs, _ = input_drift_check(None, reference(settings), settings, "none")
    predictions = prediction_drift_check(None, None, None, None, "none")
    accuracy = accuracy_check(None, None, settings, "none")

    report = monitoring_report(
        pd.Timestamp("2025-05-10", tz=settings.project.timezone),
        settings,
        inputs,
        predictions,
        accuracy,
    )

    assert (report.alert, report.warning) == (False, False)
    assert report.alerts == report.warnings == []
