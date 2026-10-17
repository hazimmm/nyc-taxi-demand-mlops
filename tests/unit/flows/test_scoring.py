"""Actuals and live scores, on tables small enough to check every number by hand."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.flows.clock import Step
from demand.flows.scoring import (
    BASELINE,
    DAY,
    ReplayReport,
    ScoringError,
    build_report,
    scored_table,
    with_actuals,
)
from demand.schemas import (
    ACTUAL,
    FORECAST,
    GENERATED_AT,
    HOUR_START,
    MODEL_VERSION,
    TARGET,
    ZONE_ID,
)

TZ = "America/New_York"
NOW = Step(now=pd.Timestamp("2025-03-19 00:00", tz=TZ), day=None)


def hand_table() -> pd.DataFrame:
    """Five scored rows on three days, two model versions and two calendar weeks."""
    rows = [
        # day, version, actual, forecast, baseline
        (date(2025, 3, 10), "7", 10.0, 12.0, 7.0),
        (date(2025, 3, 10), "7", 0.0, 0.0, 2.0),
        (date(2025, 3, 11), "8", 20.0, 16.0, 20.0),
        (date(2025, 3, 11), "8", 4.0, 4.0, 8.0),
        (date(2025, 3, 17), "8", 5.0, 5.0, 6.0),
    ]
    return pd.DataFrame(
        {
            ZONE_ID: [161, 237, 161, 237, 161],
            HOUR_START: [pd.Timestamp(row[0], tz=TZ) for row in rows],
            DAY: [row[0] for row in rows],
            MODEL_VERSION: [row[1] for row in rows],
            ACTUAL: [row[2] for row in rows],
            FORECAST: [row[3] for row in rows],
            BASELINE: [row[4] for row in rows],
        }
    )


def report(settings: Settings, backtest: dict[str, float | None] | None = None) -> ReplayReport:
    known = backtest or {"7": 20.0, "8": None}
    return build_report(hand_table(), NOW, settings, known.get)


def test_every_day_is_scored_by_hand(settings: Settings) -> None:
    days = {day.day: day for day in report(settings).days}

    first = days[date(2025, 3, 10)]
    # Errors 2 and 0; the second hour is 0 against 0, a perfect hour for sMAPE.
    assert first.model.mae == pytest.approx(1.0)
    assert first.model.rmse == pytest.approx(np.sqrt(2), abs=1e-6)
    assert first.model.smape == pytest.approx(100 * (2 / 11) / 2, abs=1e-6)
    # Errors 3 and 2; 2 against 0 is the sMAPE maximum of 200 percent.
    assert first.baseline.mae == pytest.approx(2.5)
    assert first.baseline.rmse == pytest.approx(np.sqrt(6.5), abs=1e-6)
    assert first.baseline.smape == pytest.approx(100 * (3 / 8.5 + 2) / 2, abs=1e-6)
    assert first.model_version == "7"
    assert first.model.rows == 2

    second = days[date(2025, 3, 11)]
    assert second.model.mae == second.baseline.mae == pytest.approx(2.0)
    assert second.model.smape == pytest.approx(100 * (4 / 18) / 2, abs=1e-6)
    assert second.baseline.smape == pytest.approx(100 * (4 / 6) / 2, abs=1e-6)


def test_the_pooled_scores_are_over_all_rows_not_a_mean_of_days(settings: Settings) -> None:
    result = report(settings)

    assert result.model is not None and result.baseline_score is not None
    assert result.model.mae == pytest.approx(6 / 5)
    assert result.model.rmse == pytest.approx(2.0)
    assert result.model.smape == pytest.approx(100 * (2 / 11 + 4 / 18) / 5, abs=1e-6)
    assert result.baseline_score.mae == pytest.approx(2.0)
    assert result.baseline_score.rmse == pytest.approx(np.sqrt(6), abs=1e-6)
    assert result.model.rows == result.baseline_score.rows == 5
    assert (result.days_scored, result.first_scored_day, result.last_scored_day) == (
        3,
        date(2025, 3, 10),
        date(2025, 3, 17),
    )


def test_weeks_run_from_monday_to_sunday(settings: Settings) -> None:
    weeks = report(settings).weeks

    assert [(week.week_start, week.days) for week in weeks] == [
        (date(2025, 3, 10), 2),
        (date(2025, 3, 17), 1),
    ]
    assert weeks[0].model.mae == pytest.approx(6 / 4)
    assert weeks[0].baseline.mae == pytest.approx(9 / 4)
    assert weeks[1].model.mae == 0.0


def test_each_version_is_set_against_its_own_backtest_mae(settings: Settings) -> None:
    versions = {version.model_version: version for version in report(settings).versions}

    assert versions["7"].live.mae == pytest.approx(1.0)
    assert versions["7"].backtest_mae == 20.0
    assert versions["7"].live_to_backtest_mae == pytest.approx(0.05)
    # Version 8 has no recorded backtest MAE: no ratio rather than a made-up one.
    assert versions["8"].live.mae == pytest.approx(4 / 3, abs=1e-6)
    assert versions["8"].days == 2
    assert versions["8"].live_to_backtest_mae is None


def test_a_tie_is_not_a_win_over_the_baseline(settings: Settings) -> None:
    # 10 March: 1 against 2.5; 11 March: 2 against 2 (a tie); 17 March: 0 against 1.
    assert report(settings).days_model_beat_baseline == 2


def test_the_report_says_which_days_were_left_out_at_the_start(settings: Settings) -> None:
    result = report(settings)

    assert result.first_replay_day == date(2025, 5, 2)
    assert "Not forecast: 2025-05-01." in result.replay_start_note
    assert result.clock == NOW.now.to_pydatetime()


def test_before_any_day_is_over_there_are_no_scores(settings: Settings) -> None:
    result = build_report(scored_table(settings), NOW, settings, lambda version: None)

    assert result.model is None and result.baseline_score is None
    assert result.days == result.weeks == result.versions == []
    assert result.days_forecast == 0 and result.last_forecast_day is None
    assert result.first_scored_day is None


# --- with_actuals ------------------------------------------------------------------------


def forecast_rows(settings: Settings, forecast: float = 3.0) -> pd.DataFrame:
    hours = pd.date_range("2025-03-10 00:00", periods=2, freq="h", tz=TZ)
    return pd.DataFrame(
        {
            ZONE_ID: [161, 161],
            HOUR_START: hours,
            FORECAST: [forecast, 4.5],
            ACTUAL: [np.nan, np.nan],
            MODEL_VERSION: ["7", "7"],
            GENERATED_AT: pd.Timestamp("2025-03-09 00:00", tz=TZ),
        }
    )


def history_rows(hours: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ZONE_ID: [161] * hours,
            HOUR_START: pd.date_range("2025-03-10 00:00", periods=hours, freq="h", tz=TZ),
            TARGET: np.arange(hours, dtype="int64") + 5,
        }
    )


def test_actuals_are_added_and_nothing_else_changes(settings: Settings) -> None:
    forecast = forecast_rows(settings)

    scored = with_actuals(forecast, history_rows(), settings)

    assert scored[ACTUAL].tolist() == [5.0, 6.0]
    issued = [column for column in forecast.columns if column != ACTUAL]
    pd.testing.assert_frame_equal(scored[issued], forecast[issued])
    assert forecast[ACTUAL].isna().all()


def test_a_forecast_hour_without_counted_pickups_cannot_be_scored(settings: Settings) -> None:
    with pytest.raises(ScoringError, match="no pickups for 1 zone-hours"):
        with_actuals(forecast_rows(settings), history_rows(hours=1), settings)


def test_a_scored_forecast_that_breaks_the_schema_is_refused(settings: Settings) -> None:
    with pytest.raises(ScoringError, match="does not pass predictions_schema"):
        with_actuals(forecast_rows(settings, forecast=-1.0), history_rows(), settings)
