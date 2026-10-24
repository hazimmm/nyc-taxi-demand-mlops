"""Prefect runs the daily step: forecast, monitor, retrain if due, on a schedule.

The daily step is ``retrain_flow.run_day``; this module adds nothing to what a day does.
It adds three things around it:

1. **A loop over simulated days** (``run_days``): load the data once, run ``run_day`` for
   N days (one by default), stop early at the end of the data, and send each day's
   monitoring numbers to a Pushgateway if one is given.
2. **A Prefect flow** (``daily_step``) that runs that loop with each day as a Prefect
   task, so the Prefect UI shows every day as a task run of its own (its state, its
   duration, its logs), and retries the push to the Pushgateway.
3. **A schedule** (``serve_daily_step``): a long-running process that asks the Prefect
   server for due runs of the flow and runs them.

**What a schedule means here.** The data end on 30 June 2025, so "today" is the
simulated clock of the replay (``clock.py``), never the wall clock. A scheduled run moves
that clock by ``days`` days, whatever the date on the wall. So:

* ``--every-minutes 1`` (the compose stack): a *demo*. May and June pass in about an
  hour, one simulated day a minute. It shows the mechanism (Prefect starts the run, the
  run does the day, the numbers reach Grafana) and nothing about timing in production.
* ``--cron "0 1 * * *"``: what *production* would look like, one run at 01:00 every
  night. On this data it would take sixty real days to replay May and June, so it is
  written down to show the shape of the real thing, not to be run.

A run that finds no day left to forecast does the closing step, which changes nothing
(``clock.py``), so a schedule left running after 30 June is harmless.

**Two runs never overlap.** Two runs on the same files would both forecast the same day,
and could train two challengers at once (``retrain_flow.py`` has no lock against that).
So the deployment allows one run at a time, and a run that comes due while another is
still going is cancelled, not queued: the clock is the folder of forecasts, so the next
run simply does the next day, and nothing is lost by skipping a tick.

**The prediction-drift reference is computed once per run.** Monitoring compares the
forecasts with the backtest's forecasts, which take 15 to 30 seconds to compute again
(``monitoring/reference.py``). They depend only on the training rows and the settings,
which a run does not change, so ``OnceReference`` keeps them for the rest of the run. A
new champion mid-run (after a promotion) does not change them either: monitoring keeps
the reference of the first training period, and the check compares its MAE with the
backtest MAE the registry holds for the version that issued the forecasts, which is
looked up afresh every day. If the training rows or the settings did change, the
reference is computed again rather than reused. A run of one day computes it at most
once anyway, so the cache pays off when a run does several days.

**Retries, and where there are none.** The push to the Pushgateway is retried twice, ten
seconds apart: it is a network call that may meet the gateway restarting, and repeating
it is harmless (each push replaces all the numbers). The day itself is not retried by
Prefect. A step is safe to repeat (``forecast_flow.py``), but most of its failures are
not passing ones (a missing file, a refused retrain, a model that cannot be loaded
because no version is champion). What the next scheduled run does depends on where the
day failed, because the clock is the folder of forecasts. A failure before the forecast
file is written (the model cannot be loaded, a file is missing) leaves the clock where it
was, so the next run does the same day again. A failure after it (monitoring, or a
retrain that is refused) does not: the forecast exists, so the next run does the next
day, and the failed monitoring or retrain of that day is not done again. Its day is in
the Prefect UI as a failed run, which is where a person sees it.

**Without a Prefect server.** ``run_days`` is plain Python and needs no Prefect at all.
Calling ``daily_step()`` without a server works too: Prefect then starts a temporary
server of its own for the run (its "ephemeral mode", about 20 seconds on this machine),
which keeps its database under ``PREFECT_HOME`` (``~/.prefect`` unless set).

``python -m demand schedule --every-minutes 1`` serves the flow; see ``serve_daily_step``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import timedelta
from typing import Protocol

import pandas as pd
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy

from demand.config import Settings, get_settings
from demand.errors import PipelineError
from demand.flows.monitoring_flow import MakeReference
from demand.flows.retrain_flow import DayDone, run_day
from demand.logs import get_logger
from demand.monitoring.checks import MonitoringReport
from demand.monitoring.metrics import push
from demand.monitoring.reference import ReferenceForecasts, backtest_forecasts
from demand.serving.snapshot import DataSnapshot, load_snapshot

log = get_logger(__name__)

FLOW_NAME = "daily-step"
PUSH_RETRIES = 2
PUSH_RETRY_DELAY_SECONDS = 10


class DayStep(Protocol):
    """``retrain_flow.run_day``, or the Prefect task that runs it."""

    def __call__(
        self, settings: Settings, snapshot: DataSnapshot, *, make_reference: MakeReference
    ) -> DayDone:
        """One day of the clock, with this prediction-drift reference."""
        ...


class OrchestrationError(PipelineError):
    """The daily step was asked for something it cannot do; the message says what."""


PushReport = Callable[[MonitoringReport, str], None]
"""``monitoring.metrics.push``, or the Prefect task that runs it."""


class OnceReference:
    """A ``MakeReference`` that computes the reference once and then reuses it.

    It is computed again only if it is asked for with other training rows or other
    settings than the first time (see the module docstring).
    """

    def __init__(self, make: MakeReference = backtest_forecasts) -> None:
        self._make = make
        self._kept: tuple[pd.DataFrame, Settings, ReferenceForecasts] | None = None
        self.computed = 0
        """How many times the reference was really computed; for the log and the tests."""

    def __call__(self, table: pd.DataFrame, settings: Settings) -> ReferenceForecasts:
        """The reference for these training rows and settings, computed if not kept."""
        if self._kept is not None:
            kept_table, kept_settings, reference = self._kept
            if settings == kept_settings and table.equals(kept_table):
                return reference
        reference = self._make(table, settings)
        self._kept = (table, settings, reference)
        self.computed += 1
        return reference


def summary(done: DayDone) -> str:
    """One line on what a day did, for the log (and so for the Prefect UI)."""
    forecast = "nothing (no day left)" if done.step.day is None else done.step.day.isoformat()
    alerts = ", ".join(alert.check for alert in done.monitoring.alerts) or "none"
    warnings = ", ".join(warning.check for warning in done.monitoring.warnings) or "none"
    return (
        f"day done at {done.step.now.isoformat()}: forecast {forecast}; alerts {alerts}; "
        f"warnings {warnings}; retrain {done.retrain.outcome.value}"
    )


def run_days(
    settings: Settings,
    days: int = 1,
    push_to: str | None = None,
    *,
    day: DayStep = run_day,
    push_report: PushReport = push,
    make_reference: MakeReference = backtest_forecasts,
) -> list[DayDone]:
    """Run the daily step for ``days`` simulated days; stop early at the end of the data.

    Args:
        settings: Project settings.
        days: Steps of the simulated clock to run.
        push_to: Address of a Pushgateway to send each day's monitoring numbers to;
            ``None`` sends nothing.
        day: The daily step; ``daily_step`` passes the Prefect task, tests their own.
        push_report: Sends a report to the Pushgateway; the same.
        make_reference: Computes the prediction-drift reference, at most once here.

    Raises:
        OrchestrationError: If ``days`` is not positive.
        PipelineError: See ``retrain_flow.run_day``; ``PushError`` if the Pushgateway
            does not answer (the day is done and its reports are written by then).
    """
    if days < 1:
        raise OrchestrationError(f"days must be 1 or more, not {days}.")
    snapshot = load_snapshot(settings)
    reference = OnceReference(make_reference)
    done: list[DayDone] = []
    for _ in range(days):
        today = day(settings, snapshot, make_reference=reference)
        done.append(today)
        log.info(summary(today), extra={"reference_computed": reference.computed})
        if push_to is not None:
            push_report(today.monitoring, push_to)
        if today.step.day is None:
            log.info("replay over: the data has no day left to forecast")
            break
    return done


# The two steps as Prefect tasks. NO_CACHE: by default Prefect would hash the inputs of a
# task (the settings and the whole snapshot) to look for a cached result, which a day
# must never reuse: the same inputs on another day of the clock do another day's work.
day_task = task(name="day", cache_policy=NO_CACHE)(run_day)
push_task = task(
    name="push to the pushgateway",
    retries=PUSH_RETRIES,
    retry_delay_seconds=PUSH_RETRY_DELAY_SECONDS,
    cache_policy=NO_CACHE,
)(push)


@flow(name=FLOW_NAME)
def daily_step(days: int = 1, push_to: str | None = None) -> list[DayDone]:
    """The daily step for ``days`` simulated days, each day a Prefect task run.

    Args:
        days: Steps of the simulated clock to run; stops early at the end of the data.
        push_to: Address of a Pushgateway (``http://pushgateway:9091`` in the stack) to
            send each day's monitoring numbers to; ``None`` sends nothing.
    """
    # Prefect hands the project's log lines to its own handlers (PREFECT_LOGGING_EXTRA_LOGGERS)
    # but leaves their level at the root's WARNING, so the INFO line of each day would be
    # dropped. Setting the level here makes them reach the Prefect UI and the console.
    logging.getLogger("demand").setLevel(logging.INFO)
    return run_days(get_settings(), days, push_to, day=day_task, push_report=push_task)


def serve_daily_step(
    settings: Settings,
    every_minutes: int | None = None,
    cron: str | None = None,
    days: int = 1,
    push_to: str | None = None,
) -> None:
    """Serve ``daily_step`` on a schedule until the process is stopped.

    The deployment is registered with the Prefect server of ``PREFECT_API_URL`` under the
    project's name, and this process runs its due runs, one at a time. Give exactly one
    of ``every_minutes`` (the demo) and ``cron`` (production); the module docstring says
    what each means.

    Raises:
        OrchestrationError: If not exactly one schedule is given, or ``every_minutes`` or
            ``days`` is not positive.
    """
    if (every_minutes is None) == (cron is None):
        raise OrchestrationError("Give exactly one schedule: --every-minutes or --cron.")
    if every_minutes is not None and every_minutes < 1:
        raise OrchestrationError(f"every_minutes must be 1 or more, not {every_minutes}.")
    if days < 1:
        raise OrchestrationError(f"days must be 1 or more, not {days}.")
    daily_step.serve(
        name=settings.project.name,
        interval=None if every_minutes is None else timedelta(minutes=every_minutes),
        cron=cron,
        parameters={"days": days, "push_to": push_to},
        # One run at a time, here and for the deployment; a run that comes due while one
        # is going is cancelled (the module docstring says why).
        limit=1,
        global_limit=ConcurrencyLimitConfig(
            limit=1, collision_strategy=ConcurrencyLimitStrategy.CANCEL_NEW
        ),
    )
