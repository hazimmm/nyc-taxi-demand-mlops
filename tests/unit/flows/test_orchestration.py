"""The Prefect wrapper of the daily step, with fake days and a temporary Prefect home.

Most tests call ``run_days`` with a fake daily step: plain Python, no Prefect. One test
runs the real flow, with ``run_day``'s three parts replaced by fakes. Without a Prefect
server, Prefect starts a temporary one for it (about 20 seconds, once per test session);
its database goes to ``tmp_path``, never to ``~/.prefect``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from prefect.cache_policies import NO_CACHE
from prefect.client.schemas.objects import ConcurrencyLimitStrategy
from prefect.logging.handlers import APILogHandler
from prefect.server.api.server import SubprocessASGIServer
from prefect.settings import (
    PREFECT_API_URL,
    PREFECT_HOME,
    PREFECT_SERVER_DATABASE_CONNECTION_URL,
    temporary_settings,
)

from demand.config import Settings
from demand.flows import orchestration, retrain_flow
from demand.flows.clock import Step
from demand.flows.orchestration import (
    OnceReference,
    OrchestrationError,
    daily_step,
    day_task,
    push_task,
    run_days,
    serve_daily_step,
    summary,
)
from demand.flows.retrain_flow import DayDone, Outcome, RetrainReport
from demand.monitoring.checks import (
    AccuracyCheck,
    MonitoringReport,
    input_drift_check,
    monitoring_report,
    prediction_drift_check,
)
from demand.monitoring.metrics import push
from demand.monitoring.reference import ReferenceForecasts
from demand.serving.snapshot import DataSnapshot
from tests.unit.monitoring.conftest import grid

FIRST_DAY = date(2025, 5, 2)
LAST_DAY = date(2025, 5, 4)
"""The last day the fake data can forecast: three days, then the closing step."""
GATEWAY = "http://pushgateway:9091"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def moment(day: date) -> pd.Timestamp:
    return pd.Timestamp(day, tz="America/New_York")


def monitoring(settings: Settings, now: pd.Timestamp) -> MonitoringReport:
    """A report with nothing evaluated: no alert, no warning."""
    inputs, _ = input_drift_check(None, grid(settings, [date(2025, 4, 30)]), settings, "none")
    accuracy = AccuracyCheck(
        evaluated=False,
        note="none",
        model_version=None,
        first_day=None,
        last_day=None,
        days=0,
        rows=0,
        live_mae=None,
        baseline_mae=None,
        backtest_mae=None,
        live_to_backtest_mae=None,
        ratio_threshold=settings.monitoring.mae_degradation_ratio,
        alert=False,
    )
    predictions = prediction_drift_check(None, None, None, None, "none")
    return monitoring_report(now, settings, inputs, predictions, accuracy)


def retrain(now: pd.Timestamp, outcome: Outcome = Outcome.NO_ALERT) -> RetrainReport:
    return RetrainReport(
        now=now.to_pydatetime(),
        outcome=outcome,
        reason="test",
        champion_version=1,
        live_to_backtest_mae=None,
        last_retrain=None,
        cutoff_day=None,
        challenger_version=None,
        scores=None,
        decision=None,
    )


class FakeDays:
    """A daily step that walks the clock from ``FIRST_DAY`` and asks for the reference."""

    def __init__(self, settings: Settings, table: pd.DataFrame) -> None:
        self.settings = settings
        self.table = table
        self.calls: list[tuple[DataSnapshot, Any]] = []

    def __call__(
        self, settings: Settings, snapshot: DataSnapshot, *, make_reference: Any
    ) -> DayDone:
        self.calls.append((snapshot, make_reference))
        make_reference(self.table, settings)
        day = FIRST_DAY + timedelta(days=len(self.calls) - 1)
        now = moment(day - timedelta(days=1))
        forecast = day if day <= LAST_DAY else None
        return DayDone(
            step=Step(now=now, day=forecast),
            replay=None,
            monitoring=monitoring(settings, now),
            retrain=retrain(now),
        )


class Counter:
    """A ``MakeReference`` that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, table: pd.DataFrame, settings: Settings) -> ReferenceForecasts:
        self.calls += 1
        return ReferenceForecasts(rows=table, mae=float(self.calls))


SNAPSHOT = object()


@pytest.fixture
def fake_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestration, "load_snapshot", lambda _: SNAPSHOT)


def table() -> pd.DataFrame:
    return pd.DataFrame({"a": [1.0, 2.0]})


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_snapshot")
def test_one_day_by_default_on_the_snapshot_loaded_once(settings: Settings) -> None:
    days = FakeDays(settings, table())

    done = run_days(settings, day=days, make_reference=Counter())

    assert len(done) == 1
    assert [snapshot for snapshot, _ in days.calls] == [SNAPSHOT]
    assert done[0].step.day == FIRST_DAY


@pytest.mark.usefixtures("fake_snapshot")
def test_n_days_run_the_daily_step_n_times(settings: Settings) -> None:
    days = FakeDays(settings, table())

    done = run_days(settings, 2, day=days, make_reference=Counter())

    assert [d.step.day for d in done] == [FIRST_DAY, FIRST_DAY + timedelta(days=1)]
    assert len(days.calls) == 2


@pytest.mark.usefixtures("fake_snapshot")
def test_the_loop_stops_after_the_closing_step(settings: Settings) -> None:
    days = FakeDays(settings, table())

    done = run_days(settings, 10, day=days, make_reference=Counter())

    assert [d.step.day for d in done] == [*pd.date_range(FIRST_DAY, LAST_DAY).date, None]
    assert len(days.calls) == 4


@pytest.mark.usefixtures("fake_snapshot")
def test_the_reference_is_computed_once_for_every_day_of_a_run(settings: Settings) -> None:
    days, counter = FakeDays(settings, table()), Counter()

    run_days(settings, 3, day=days, make_reference=counter)

    assert counter.calls == 1
    references = {id(reference) for _, reference in days.calls}
    assert len(references) == 1


@pytest.mark.usefixtures("fake_snapshot")
def test_each_run_computes_its_own_reference(settings: Settings) -> None:
    counter = Counter()

    run_days(settings, 2, day=FakeDays(settings, table()), make_reference=counter)
    run_days(settings, 2, day=FakeDays(settings, table()), make_reference=counter)

    assert counter.calls == 2


@pytest.mark.usefixtures("fake_snapshot")
def test_the_monitoring_report_of_every_day_goes_to_the_push_target(
    settings: Settings,
) -> None:
    pushed: list[tuple[MonitoringReport, str]] = []

    done = run_days(
        settings,
        2,
        GATEWAY,
        day=FakeDays(settings, table()),
        push_report=lambda report, gateway: pushed.append((report, gateway)),
        make_reference=Counter(),
    )

    assert pushed == [(d.monitoring, GATEWAY) for d in done]


@pytest.mark.usefixtures("fake_snapshot")
def test_without_a_push_target_nothing_is_pushed(settings: Settings) -> None:
    pushed: list[object] = []

    run_days(
        settings,
        2,
        day=FakeDays(settings, table()),
        push_report=lambda *values: pushed.append(values),
        make_reference=Counter(),
    )

    assert pushed == []


def test_zero_days_are_refused(settings: Settings) -> None:
    with pytest.raises(OrchestrationError, match="days must be 1 or more, not 0"):
        run_days(settings, 0)


def test_a_day_in_one_line(settings: Settings) -> None:
    now = moment(date(2025, 5, 1))
    done = DayDone(
        step=Step(now=now, day=FIRST_DAY),
        replay=None,
        monitoring=monitoring(settings, now),
        retrain=retrain(now),
    )

    assert summary(done) == (
        "day done at 2025-05-01T00:00:00-04:00: forecast 2025-05-02; alerts none; "
        "warnings none; retrain no_alert"
    )
    closing = DayDone(
        step=Step(now=now, day=None),
        replay=None,
        monitoring=done.monitoring,
        retrain=retrain(now, Outcome.END_OF_DATA),
    )
    assert "forecast nothing (no day left)" in summary(closing)
    assert summary(closing).endswith("retrain end_of_data")


# --------------------------------------------------------------------------------------
# The reference cache
# --------------------------------------------------------------------------------------


def test_the_reference_is_kept_for_the_same_rows_and_settings(settings: Settings) -> None:
    counter = Counter()
    once = OnceReference(counter)

    first = once(table(), settings)
    again = once(table(), settings.model_copy())

    assert again is first
    assert (counter.calls, once.computed) == (1, 1)


def test_other_rows_or_other_settings_compute_it_again(settings: Settings) -> None:
    counter = Counter()
    once = OnceReference(counter)
    other_settings = settings.model_copy(
        update={"monitoring": settings.monitoring.model_copy(update={"window_days": 3})}
    )

    once(table(), settings)
    once(table().assign(a=[1.0, 3.0]), settings)
    once(table().assign(a=[1.0, 3.0]), other_settings)

    assert (counter.calls, once.computed) == (3, 3)


# --------------------------------------------------------------------------------------
# The Prefect tasks and the flow
# --------------------------------------------------------------------------------------


def test_the_tasks_are_the_daily_step_and_the_push_and_never_cached() -> None:
    assert day_task.fn is retrain_flow.run_day
    assert push_task.fn is push
    assert day_task.cache_policy is NO_CACHE
    assert push_task.cache_policy is NO_CACHE
    assert (day_task.retries, push_task.retries) == (0, 2)
    assert push_task.retry_delay_seconds == 10


@pytest.fixture
def prefect_home(tmp_path: Path) -> Iterator[Path]:
    """A Prefect home and database under ``tmp_path``, and no server address.

    Prefect then starts a temporary server for the flow run. It is stopped here, once
    the logs still waiting to be sent have reached it: left to Prefect, it is stopped when
    Python exits, after pytest has closed the output Prefect writes to.
    """
    home = tmp_path / "prefect"
    database = f"sqlite+aiosqlite:///{(home / 'prefect.db').as_posix()}"
    with temporary_settings(
        updates={PREFECT_HOME: home, PREFECT_SERVER_DATABASE_CONNECTION_URL: database},
        restore_defaults={PREFECT_API_URL},
    ):
        yield home
        APILogHandler().flush()
        SubprocessASGIServer().stop()


def test_the_flow_runs_each_day_as_a_task_without_a_prefect_server(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, prefect_home: Path
) -> None:
    calls: list[str] = []
    steps = iter([Step(moment(date(2025, 5, 1)), FIRST_DAY), Step(moment(date(2025, 7, 1)), None)])

    def fake_step(*_: object, **__: object) -> tuple[Step, None]:
        calls.append("forecast")
        return next(steps), None

    def fake_monitoring(settings_: Settings, **values: Any) -> MonitoringReport:
        calls.append("monitor")
        return monitoring(settings_, moment(values["as_of"]))

    monkeypatch.setattr(orchestration, "get_settings", lambda: settings)
    monkeypatch.setattr(orchestration, "load_snapshot", lambda _: SNAPSHOT)
    monkeypatch.setattr(retrain_flow, "run_step", fake_step)
    monkeypatch.setattr(retrain_flow, "run_monitoring", fake_monitoring)
    monkeypatch.setattr(
        retrain_flow, "retrain_if_needed", lambda _, __, now, ___, **____: retrain(now)
    )

    state = daily_step(days=5, return_state=True)

    assert state.is_completed()
    assert (prefect_home / "prefect.db").is_file()
    done = state.result()
    assert [d.step.day for d in done] == [FIRST_DAY, None]
    assert calls == ["forecast", "monitor", "forecast", "monitor"]


# --------------------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------------------


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(daily_step, "serve", lambda **values: calls.append(values))
    return calls


def test_the_demo_schedule_serves_a_run_every_few_minutes(
    settings: Settings, served: list[dict[str, Any]]
) -> None:
    serve_daily_step(settings, every_minutes=2, push_to=GATEWAY)

    (call,) = served
    assert call["name"] == settings.project.name
    assert call["interval"] == timedelta(minutes=2)
    assert call["cron"] is None
    assert call["parameters"] == {"days": 1, "push_to": GATEWAY}
    assert call["limit"] == 1
    assert call["global_limit"].limit == 1
    assert call["global_limit"].collision_strategy == ConcurrencyLimitStrategy.CANCEL_NEW


def test_the_production_schedule_is_a_cron(
    settings: Settings, served: list[dict[str, Any]]
) -> None:
    serve_daily_step(settings, cron="0 1 * * *", days=2)

    (call,) = served
    assert (call["interval"], call["cron"]) == (None, "0 1 * * *")
    assert call["parameters"] == {"days": 2, "push_to": None}


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({}, "exactly one schedule"),
        ({"every_minutes": 2, "cron": "0 1 * * *"}, "exactly one schedule"),
        ({"every_minutes": 0}, "every_minutes must be 1 or more, not 0"),
        ({"every_minutes": 2, "days": 0}, "days must be 1 or more, not 0"),
    ],
)
def test_a_schedule_that_makes_no_sense_is_refused(
    settings: Settings, served: list[dict[str, Any]], values: dict[str, Any], message: str
) -> None:
    with pytest.raises(OrchestrationError, match=message):
        serve_daily_step(settings, **values)
    assert served == []
