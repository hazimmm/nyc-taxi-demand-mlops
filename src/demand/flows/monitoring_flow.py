"""The monitoring command: the three checks of ``monitoring/checks.py`` at one moment.

``python -m demand monitor`` evaluates them at the simulated "now" of the replay, the
moment the clock stands at (``clock.next_step``; after the last day, 00:00 on 1 July),
and writes::

    reports/monitoring_report.json         the checks, numbers, thresholds, alerts, warnings
    reports/monitoring_input_drift.html    Evidently's page of the input drift (optional)

and, when a Pushgateway address is given, sends the numbers to it (``monitoring/
metrics.py``). The JSON report holds no wall-clock time and is rounded, so the same
files give the same report byte for byte. The HTML page is Evidently's and is not: it
is for looking at, not for comparing.

**As of an earlier moment.** ``run_monitoring(settings, as_of=day)`` evaluates the
checks at 00:00 on ``day`` instead, for any day from the start of the replay up to the
clock. That is how to see *when* an alert would have fired during May and June, and
what the retraining feature will use to replay its decisions day by day. A moment after
the clock is refused: the replay has not got there, and nothing may be read that the
clock has not yet made.

**Only what existed at "now" is read.** Of the files of the replay
(``clock.py`` lists the folders):

* a forecast and its inputs exist from the forecast's issue moment, 00:00 on the day
  before the day it is for. At 00:00 on 1 June the newest is the forecast for 2 June;
* a scored day exists from the end of the day (00:00 on the next day). At 00:00 on
  1 June the newest is 31 May.

Files of later days are not opened at all: the days are chosen from the file names, and
only the chosen files are read. The training rows (``models/backtest.py``) end with the
training period, before any "now" monitoring accepts. The backtest MAE of a version was
recorded when the version was trained, before the replay.

**The windows.** Input and prediction drift: the last ``window_days`` forecast days
issued by "now" (at 00:00 on 1 July, 24 to 30 June). Accuracy: the last ``window_days``
days that are over (the same days at the end of the replay; earlier, they lag the drift
window by two days, the time a forecast waits for its day to end).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet

from demand.config import Settings
from demand.data.aggregate import training_period_end
from demand.data.silver import report_to_json, write_text
from demand.errors import PipelineError
from demand.flows.clock import (
    day_end,
    day_file,
    forecasts_dir,
    inputs_dir,
    next_step,
    scored_dir,
    stored_days,
)
from demand.flows.forecast_flow import registry_backtest_mae
from demand.flows.scoring import BacktestMae
from demand.logs import get_logger
from demand.models.backtest import load_training_features
from demand.models.baseline import HOURS_PER_WEEK
from demand.monitoring.checks import (
    BASELINE,
    DAY,
    MonitoringReport,
    accuracy_check,
    drift_columns,
    input_drift_check,
    monitoring_report,
    prediction_drift_check,
)
from demand.monitoring.metrics import push
from demand.monitoring.reference import ReferenceForecasts, backtest_forecasts
from demand.schemas import (
    HOUR_START,
    IS_HOLIDAY,
    MODEL_VERSION,
    ZONE_ID,
    lag_column,
)
from demand.serving.snapshot import issue_moment, load_snapshot

log = get_logger(__name__)

MONITORING_REPORT_NAME = "monitoring_report.json"
INPUT_DRIFT_HTML_NAME = "monitoring_input_drift.html"

MakeReference = Callable[[pd.DataFrame, Settings], ReferenceForecasts]
"""``monitoring.reference.backtest_forecasts``; tests, and the day-by-day replay, pass their own."""


class MonitoringError(PipelineError):
    """Monitoring cannot be evaluated as asked; the message says why."""


def monitoring_report_path(settings: Settings) -> Path:
    """Where the monitoring report is stored."""
    return settings.paths.reports_dir / MONITORING_REPORT_NAME


def input_drift_html_path(settings: Settings) -> Path:
    """Where Evidently's page of the input drift is stored."""
    return settings.paths.reports_dir / INPUT_DRIFT_HTML_NAME


def evaluation_moment(settings: Settings, as_of: date | None, clock: pd.Timestamp) -> pd.Timestamp:
    """The "now" of the evaluation: ``clock``, or 00:00 on ``as_of``.

    Raises:
        MonitoringError: If ``as_of`` is after the clock, or before the end of training.
    """
    if as_of is None:
        return clock
    now = pd.Timestamp(as_of, tz=settings.project.timezone)
    if now > clock:
        raise MonitoringError(
            f"Monitoring cannot be evaluated as of {as_of.isoformat()}: the replay clock "
            f"stands at {clock.isoformat()}, and nothing after it exists yet. Run the "
            "forecast step further first (`python -m demand forecast`)."
        )
    start = training_period_end(settings)
    if now < start:
        raise MonitoringError(
            f"Monitoring cannot be evaluated as of {as_of.isoformat()}: it watches the "
            f"replay, which starts at {start.isoformat()}, the end of the training period."
        )
    return now


def issued_days(settings: Settings, now: pd.Timestamp) -> list[date]:
    """The forecast days whose forecast had been issued by ``now``, oldest first."""
    days = stored_days(forecasts_dir(settings))
    return [day for day in days if issue_moment(day, settings) <= now]


def over_days(settings: Settings, now: pd.Timestamp) -> list[date]:
    """The scored days that were over by ``now``, oldest first."""
    return [day for day in stored_days(scored_dir(settings)) if day_end(day, settings) <= now]


def read_days(folder: Path, days: list[date], columns: list[str] | None = None) -> pd.DataFrame:
    """The files of ``days`` in ``folder``, one table, with a ``day`` column."""
    tables = [
        pd.read_parquet(day_file(folder, day), columns=columns).assign(**{DAY: day}) for day in days
    ]
    return pd.concat(tables, ignore_index=True)


def last_days(days: list[date], settings: Settings, what: str) -> tuple[list[date], str | None]:
    """The last ``window_days`` of ``days``, or none and why."""
    wanted = settings.monitoring.window_days
    if len(days) < wanted:
        return [], f"{len(days)} {what} by now; the check needs {wanted}."
    return days[-wanted:], None


def newest_version(
    table: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame | None, str | None]:
    """The rows of the model version of the newest day, if it has every day of the window."""
    version = table.loc[table[DAY] == table[DAY].max(), MODEL_VERSION].iloc[0]
    rows = table[table[MODEL_VERSION] == version]
    days, wanted = rows[DAY].nunique(), settings.monitoring.window_days
    if days < wanted:
        return None, (
            f"Only {days} of the last {wanted} days are of model version {version}, the "
            "newest; the check needs all of them to be."
        )
    return rows.reset_index(drop=True), None


def reference_rows(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The last ``reference_days`` of the training rows, with a ``day`` column.

    Raises:
        MonitoringError: If the training rows do not reach back that far.
    """
    first = settings.data.train_end_date - timedelta(days=settings.monitoring.reference_days - 1)
    days = table[HOUR_START].dt.date
    if days.min() > first:
        raise MonitoringError(
            f"The training rows start on {days.min().isoformat()}, after the first day of "
            f"the drift reference ({first.isoformat()}). Lower monitoring.reference_days."
        )
    chosen = days >= first
    rows: pd.DataFrame = table[chosen].assign(**{DAY: days[chosen]})
    return rows.reset_index(drop=True)


def window_inputs(settings: Settings, days: list[date]) -> pd.DataFrame:
    """The model inputs of ``days`` (the tested columns and the holiday flag).

    Raises:
        MonitoringError: If an inputs file lacks one of those columns.
    """
    columns = [*drift_columns(settings), IS_HOLIDAY]
    folder = inputs_dir(settings)
    for day in days:
        names = pyarrow.parquet.read_schema(day_file(folder, day)).names
        missing = [column for column in columns if column not in names]
        if missing:
            raise MonitoringError(
                f"The inputs of {day.isoformat()} hold no {', '.join(missing)}: they were "
                "written with another config. Start the replay again with this one."
            )
    return read_days(folder, days, columns)


def scored_rows(settings: Settings, days: list[date]) -> pd.DataFrame:
    """The scored rows of ``days``, with the weekly baseline from the inputs."""
    scored = read_days(scored_dir(settings), days)
    baseline = read_days(
        inputs_dir(settings), days, [ZONE_ID, HOUR_START, lag_column(HOURS_PER_WEEK)]
    )
    baseline = baseline.rename(columns={lag_column(HOURS_PER_WEEK): BASELINE}).drop(columns=DAY)
    return scored.merge(baseline, on=[ZONE_ID, HOUR_START], how="left", validate="one_to_one")


def evaluate(
    settings: Settings,
    now: pd.Timestamp,
    training: pd.DataFrame,
    backtest_mae: BacktestMae,
    make_reference: MakeReference,
) -> tuple[MonitoringReport, Any]:
    """The three checks at ``now``; with Evidently's input drift result (or ``None``).

    Args:
        settings: Paths, windows and thresholds.
        now: The simulated moment; nothing newer is read.
        training: The training rows of the gold feature table.
        backtest_mae: The backtest MAE of a model version, from the registry.
        make_reference: Computes the prediction-drift reference; called only when the
            prediction window is complete and its version has a backtest MAE.
    """
    drift_days, drift_note = last_days(issued_days(settings, now), settings, "forecast days issued")
    inputs = window_inputs(settings, drift_days) if drift_days else None
    inputs_check, snapshot = input_drift_check(
        inputs, reference_rows(training, settings), settings, drift_note
    )

    forecasts, reference, recorded = None, None, None
    forecast_note = drift_note
    if drift_days:
        forecasts, forecast_note = newest_version(
            read_days(forecasts_dir(settings), drift_days), settings
        )
    if forecasts is not None:
        recorded = backtest_mae(str(forecasts[MODEL_VERSION].iloc[0]))
        if recorded is not None:
            reference = make_reference(training, settings)
    predictions_check = prediction_drift_check(
        forecasts,
        None if reference is None else reference.rows,
        None if reference is None else reference.mae,
        recorded,
        forecast_note,
    )

    score_days, score_note = last_days(over_days(settings, now), settings, "days over and scored")
    scored = None
    if score_days:
        scored, score_note = newest_version(scored_rows(settings, score_days), settings)
    version_mae = None if scored is None else backtest_mae(str(scored[MODEL_VERSION].iloc[0]))
    accuracy = accuracy_check(scored, version_mae, settings, score_note)

    report = monitoring_report(now, settings, inputs_check, predictions_check, accuracy)
    return report, snapshot


def run_monitoring(
    settings: Settings,
    as_of: date | None = None,
    push_to: str | None = None,
    *,
    html: bool = True,
    backtest_mae: Callable[[Settings], BacktestMae] = registry_backtest_mae,
    make_reference: MakeReference = backtest_forecasts,
) -> MonitoringReport:
    """Evaluate the checks at the clock (or as of 00:00 on ``as_of``) and write the report.

    Args:
        settings: Project settings.
        as_of: Evaluate at 00:00 on this day instead of at the clock.
        push_to: Address of a Pushgateway (``http://localhost:9091``) to send the
            numbers to; ``None`` sends nothing.
        html: Also write Evidently's page of the input drift.
        backtest_mae: The registry; tests pass their own.
        make_reference: The prediction-drift reference; tests pass their own.

    Raises:
        PipelineError: If the replay files are not one unbroken run, ``as_of`` is out of
            range, the gold table is missing, an inputs file lacks a column, or the
            Pushgateway does not answer (the report is written first).
    """
    started = time.monotonic()
    clock = next_step(settings, load_snapshot(settings)).now
    now = evaluation_moment(settings, as_of, clock)
    report, snapshot = evaluate(
        settings, now, load_training_features(settings), backtest_mae(settings), make_reference
    )
    write_text(report_to_json(report), monitoring_report_path(settings))
    if html and snapshot is not None:
        snapshot.save_html(str(input_drift_html_path(settings)))
    if push_to is not None:
        push(report, push_to)
    log.info(
        "monitoring evaluated",
        extra={
            "now": now.isoformat(),
            "alerts": [alert.check for alert in report.alerts],
            "warnings": [warning.check for warning in report.warnings],
            "drift_share": report.input_drift.drift_share,
            "prediction_drift_distance": report.prediction_drift.distance,
            "live_to_backtest_mae": report.accuracy.live_to_backtest_mae,
            "seconds": round(time.monotonic() - started, 1),
        },
    )
    return report
