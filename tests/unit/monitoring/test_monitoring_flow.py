from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.silver import write_parquet
from demand.errors import PipelineError
from demand.flows import monitoring_flow
from demand.flows.clock import ReplayStateError, day_file, forecasts_dir, inputs_dir, scored_dir
from demand.flows.monitoring_flow import (
    MonitoringError,
    evaluate,
    evaluation_moment,
    input_drift_html_path,
    issued_days,
    monitoring_report_path,
    over_days,
    reference_rows,
    run_monitoring,
    window_inputs,
)
from demand.flows.scoring import BacktestMae
from demand.monitoring.checks import ACCURACY, MonitoringReport
from demand.schemas import lag_column
from tests.unit.monitoring.conftest import (
    FIRST_DAY,
    FixedReference,
    days_from,
    moment,
    normal_values,
    write_garbage,
    write_replay,
)

REPLAY = days_from(FIRST_DAY, 12)  # 2 to 13 May
SCORED = REPLAY[:-1]  # 2 to 12 May: the days over at the clock, 00:00 on 13 May
CLOCK = date(2025, 5, 13)


def recorded(mae: float | None = 20.0) -> BacktestMae:
    return lambda version: mae


@pytest.fixture
def replay(settings: Settings, training: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """A replay of 2 to 13 May on disk; the clock stands at 00:00 on 13 May."""
    write_replay(settings, REPLAY, scored=SCORED, forecast_spread=10.0, actual=110.0, baseline=95.0)
    monkeypatch.setattr(
        monitoring_flow, "load_snapshot", lambda _: SimpleNamespace(last_day=date(2025, 6, 30))
    )
    monkeypatch.setattr(monitoring_flow, "load_training_features", lambda _: training)
    return settings


def run(
    settings: Settings,
    as_of: date | None = None,
    reference: FixedReference | None = None,
    mae: float | None = 20.0,
    **options: Any,
) -> MonitoringReport:
    return run_monitoring(
        settings,
        as_of,
        backtest_mae=lambda _: recorded(mae),
        make_reference=reference or FixedReference(),
        **options,
    )


# --------------------------------------------------------------------------------------
# "Now", and what existed then
# --------------------------------------------------------------------------------------


def test_now_is_the_clock_unless_a_day_is_given(settings: Settings) -> None:
    clock = moment(CLOCK, settings)

    assert evaluation_moment(settings, None, clock) == clock
    assert evaluation_moment(settings, date(2025, 5, 9), clock) == moment(
        date(2025, 5, 9), settings
    )


def test_a_day_after_the_clock_is_refused(settings: Settings) -> None:
    with pytest.raises(MonitoringError, match="nothing after it exists yet"):
        evaluation_moment(settings, date(2025, 5, 14), moment(CLOCK, settings))


def test_a_day_before_the_replay_is_refused(settings: Settings) -> None:
    with pytest.raises(MonitoringError, match="it watches the replay"):
        evaluation_moment(settings, date(2025, 4, 30), moment(CLOCK, settings))


def test_forecasts_exist_from_their_issue_moment_and_scores_from_the_end_of_the_day(
    replay: Settings,
) -> None:
    now = moment(date(2025, 5, 9), replay)

    assert issued_days(replay, now)[-1] == date(2025, 5, 10)
    assert over_days(replay, now)[-1] == date(2025, 5, 8)


def test_as_of_reads_no_file_of_a_later_day(replay: Settings, training: pd.DataFrame) -> None:
    now = moment(date(2025, 5, 10), replay)
    before, _ = evaluate(replay, now, training, recorded(), FixedReference())
    for day in REPLAY:
        if day > date(2025, 5, 11):  # issued after 00:00 on 10 May
            write_garbage(forecasts_dir(replay), day)
            write_garbage(inputs_dir(replay), day)
        if day > date(2025, 5, 9):  # over after 00:00 on 10 May
            write_garbage(scored_dir(replay), day)

    after, _ = evaluate(replay, now, training, recorded(), FixedReference())

    assert after == before
    assert after.input_drift.window_last_day == date(2025, 5, 11)
    assert after.accuracy.last_day == date(2025, 5, 9)


# --------------------------------------------------------------------------------------
# The windows
# --------------------------------------------------------------------------------------


def test_the_windows_at_the_clock(replay: Settings) -> None:
    report = run(replay)

    assert report.now == moment(CLOCK, replay).to_pydatetime()
    drift = report.input_drift
    assert (drift.window_first_day, drift.window_last_day) == (date(2025, 5, 7), date(2025, 5, 13))
    assert drift.window_rows == 3 * 24 * 7
    assert report.prediction_drift.window_last_day == date(2025, 5, 13)
    accuracy = report.accuracy
    assert (accuracy.first_day, accuracy.last_day, accuracy.days) == (
        date(2025, 5, 6),
        date(2025, 5, 12),
        7,
    )


def test_accuracy_of_the_window_by_hand(replay: Settings) -> None:
    accuracy = run(replay).accuracy

    # Every day holds the same 72 forecasts (3 zones x 24 hours) against 110 pickups;
    # the baseline is 95 everywhere.
    live = np.abs(normal_values(72) - 110.0).mean()
    assert accuracy.live_mae == pytest.approx(live, abs=1e-6)
    assert accuracy.baseline_mae == 15.0
    assert accuracy.backtest_mae == 20.0
    assert accuracy.live_to_backtest_mae == round(live / 20.0, 4)


def test_too_few_days_are_not_evaluated(replay: Settings) -> None:
    report = run(replay, date(2025, 5, 5))

    assert not report.input_drift.evaluated
    assert report.input_drift.note == "5 forecast days issued by now; the check needs 7."
    assert not report.prediction_drift.evaluated
    assert not report.accuracy.evaluated
    assert report.accuracy.note == "3 days over and scored by now; the check needs 7."
    assert not report.alert


def test_a_new_model_version_is_judged_only_on_a_full_window_of_its_own(
    settings: Settings, training: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_replay(settings, REPLAY[:9], scored=REPLAY[:9], version="2")
    write_replay(settings, REPLAY[9:], scored=REPLAY[9:11], version="3")
    reference = FixedReference()

    report, _ = evaluate(settings, moment(CLOCK, settings), training, recorded(), reference)

    assert report.input_drift.evaluated
    assert not report.prediction_drift.evaluated
    assert report.prediction_drift.note is not None
    assert "Only 3 of the last 7 days are of model version 3" in report.prediction_drift.note
    assert not report.accuracy.evaluated
    assert report.accuracy.note is not None
    assert "Only 2 of the last 7 days are of model version 3" in report.accuracy.note
    assert reference.calls == 0


def test_the_reference_is_made_only_when_it_can_be_used(
    replay: Settings, training: pd.DataFrame
) -> None:
    reference = FixedReference()

    report, _ = evaluate(replay, moment(CLOCK, replay), training, recorded(None), reference)

    assert reference.calls == 0
    assert not report.prediction_drift.evaluated
    assert not report.accuracy.evaluated


# --------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------


def test_the_report_is_written_and_the_same_every_time(replay: Settings) -> None:
    report = run(replay)
    first = monitoring_report_path(replay).read_bytes()

    run(replay)

    assert monitoring_report_path(replay).read_bytes() == first
    assert MonitoringReport.model_validate_json(first) == report
    assert input_drift_html_path(replay).is_file()


def test_the_html_page_is_optional(replay: Settings) -> None:
    run(replay, html=False)

    assert monitoring_report_path(replay).is_file()
    assert not input_drift_html_path(replay).exists()


def test_the_numbers_are_pushed_when_asked(
    replay: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    pushed: list[tuple[MonitoringReport, str]] = []
    monkeypatch.setattr(monitoring_flow, "push", lambda report, url: pushed.append((report, url)))

    report = run(replay, push_to="http://localhost:9091")

    assert pushed == [(report, "http://localhost:9091")]


def test_an_accuracy_alert_is_in_the_report(replay: Settings) -> None:
    report = run(replay, mae=5.0)

    assert report.accuracy.alert
    assert ACCURACY in [alert.check for alert in report.alerts]
    assert report.alert


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


def test_a_broken_replay_is_a_pipeline_error(replay: Settings) -> None:
    day_file(forecasts_dir(replay), date(2025, 5, 5)).unlink()

    with pytest.raises(ReplayStateError):
        run(replay)


def test_inputs_without_a_tested_column_are_refused(replay: Settings) -> None:
    path = day_file(inputs_dir(replay), date(2025, 5, 10))
    write_parquet(pd.read_parquet(path).drop(columns=lag_column(48)), path)

    with pytest.raises(MonitoringError, match="hold no lag_48h"):
        window_inputs(replay, [date(2025, 5, 10)])


def test_training_rows_too_short_for_the_reference_are_refused(
    settings: Settings, training: pd.DataFrame
) -> None:
    late = training[training["hour_start"].dt.date >= date(2025, 4, 10)]

    with pytest.raises(MonitoringError, match=r"Lower monitoring\.reference_days") as caught:
        reference_rows(late, settings)
    assert isinstance(caught.value, PipelineError)


def test_the_reference_is_the_last_reference_days_of_training(
    settings: Settings, training: pd.DataFrame
) -> None:
    rows = reference_rows(training, settings)

    assert rows["day"].min() == date(2025, 4, 3)
    assert rows["day"].max() == date(2025, 4, 30)
    assert len(rows) == 3 * 24 * 28
