"""The batch forecast of the replay: one step of the simulated clock at a time.

``python -m demand forecast`` runs one step; ``--days N`` runs N steps in a row. A step:

1. **Moves the clock** (``clock.py``): the next day to forecast is the day after the
   newest forecast on disk, and "now" is its issue moment, 00:00 on the day before.
2. **Issues the forecast for that day.** It loads the champion, resolving the alias to a
   version once (``model_loader.py``), and forecasts every hour of the day for every
   configured zone with ``predict_day``: the code the API answers with, reading only the
   demand that had ended by "now". The rows are checked against ``predictions_schema``
   and written once, never again.
3. **Scores every day that is now over** (``scoring.py``): fills in the pickups that
   were counted, in a table of its own, and writes the report.

Layout (``paths.predictions``, under the git-ignored ``data/``)::

    data/predictions/
      forecasts/2025-05-02.parquet   what was issued (predictions_schema, actual empty)
      inputs/2025-05-02.parquet      the rows the model was given: zone, hour, features
      scored/2025-05-02.parquet      the same forecasts once the day is over, actual filled
    reports/replay_report.json       live scores so far: pooled, per week, per day

One file per day, because a day is the unit of everything here: it is issued at one
moment by one model version, scored at one moment, and a step writes one of each. A file
of a day is written whole or not at all (a temporary file first, then renamed).

**``generated_at`` is the simulated issue moment, not the wall clock.** It says when the
forecast counts as made, which is what every later question about it needs: was the
model in use then, what could it have known, how old was the forecast when the hour
came. The wall clock of the replay would say only that all of May and June were
"generated" within a few minutes of one afternoon, and would make two runs of the same
step write different files. The time a step really ran is in the log.

**A step is safe to repeat.** A forecast already on disk is never computed or written
again. The inputs are written before the forecast, so a step interrupted between the two
is simply done again (the forecast file is what moves the clock), and gives the same
inputs. A scored day is not scored again. The report is rebuilt from the files every
time. Once the data has no day left to forecast, every step is the same closing step and
writes the same files.

**Left out on purpose.** Prefect: these are plain functions, which a Prefect flow will
call in the orchestration feature (a ``@flow`` wrapper here would start a temporary
Prefect server at every run and add nothing a loop does not). Alerts and drift: the
monitoring feature reads the files written here.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pandera.errors

from demand.config import Settings
from demand.data.silver import report_to_json, write_parquet, write_text
from demand.errors import PipelineError
from demand.flows.clock import (
    Step,
    day_file,
    forecasts_dir,
    inputs_dir,
    next_step,
    scored_dir,
)
from demand.flows.scoring import (
    BacktestMae,
    ReplayReport,
    build_report,
    replay_report_path,
    scorable_days,
    scored_table,
    with_actuals,
)
from demand.logs import get_logger
from demand.models.registry import Registry
from demand.schemas import (
    ACTUAL,
    FORECAST,
    GENERATED_AT,
    HOUR_START,
    MODEL_VERSION,
    ZONE_ID,
    predictions_schema,
)
from demand.serving.forecasting import DayForecast, predict_day
from demand.serving.model_loader import BACKTEST_MAE_METRIC, LoadedModel, load_model
from demand.serving.snapshot import ONE_HOUR, DataSnapshot, load_snapshot

log = get_logger(__name__)

LoadModel = Callable[[Settings], LoadedModel]


class ForecastError(PipelineError):
    """A forecast cannot be issued as it is; nothing was written for its day."""


def forecast_table(predicted: DayForecast, model_version: str, settings: Settings) -> pd.DataFrame:
    """The rows of a forecast as ``predictions_schema`` describes them, ``actual`` empty.

    Raises:
        ForecastError: If the rows do not pass the schema (a negative or missing forecast,
            a zone that is not configured).
    """
    rows = predicted.rows
    table = pd.DataFrame(
        {
            ZONE_ID: rows[ZONE_ID].to_numpy(),
            HOUR_START: rows[HOUR_START].reset_index(drop=True),
            FORECAST: predicted.values,
            ACTUAL: np.full(len(rows), np.nan),
            MODEL_VERSION: model_version,
            GENERATED_AT: predicted.issued,
        }
    )
    try:
        validated: pd.DataFrame = predictions_schema(settings).validate(table)
    except pandera.errors.SchemaError as error:
        raise ForecastError(
            f"The forecast issued at {predicted.issued} does not pass predictions_schema, so "
            f"it was not written: {error}. Check the model that is champion "
            "(`python -m demand serve` and /model-info show it)."
        ) from error
    return validated


def issue_forecast(
    day: date, model: LoadedModel, snapshot: DataSnapshot, settings: Settings
) -> bool:
    """Forecast ``day`` at its issue moment and write it; ``False`` if it is on disk already.

    Raises:
        ForecastError: See ``forecast_table``.
        RequestRefusedError: If the snapshot cannot forecast the day.
    """
    target = day_file(forecasts_dir(settings), day)
    if target.exists():
        return False
    predicted = predict_day(model.predictor, snapshot, settings.data.zones, day, settings)
    table = forecast_table(predicted, str(model.info.version), settings)
    write_parquet(predicted.rows.reset_index(drop=True), day_file(inputs_dir(settings), day))
    write_parquet(table, target)
    log.info(
        "forecast issued",
        extra={
            "day": day.isoformat(),
            "issued_at": predicted.issued.isoformat(),
            "model_version": model.info.version,
            "rows": len(table),
        },
    )
    return True


def score_days(step: Step, snapshot: DataSnapshot, settings: Settings) -> list[str]:
    """Score every forecast day that is over at ``step.now``; the days scored, as text.

    Only the pickups of hours that ended by ``step.now`` are read.

    Raises:
        ScoringError: If a day cannot be scored (``with_actuals``).
    """
    history = snapshot.history
    counted = history[history[HOUR_START] + ONE_HOUR <= step.now]
    done = []
    for day in scorable_days(settings, step.now):
        forecast = pd.read_parquet(day_file(forecasts_dir(settings), day))
        write_parquet(
            with_actuals(forecast, counted, settings), day_file(scored_dir(settings), day)
        )
        done.append(day.isoformat())
    return done


def registry_backtest_mae(settings: Settings) -> BacktestMae:
    """Look up the backtest MAE that training recorded for a version, in the registry."""
    registry = Registry(settings)
    return lambda version: registry.version(int(version)).metrics.get(BACKTEST_MAE_METRIC)


def run_step(
    settings: Settings,
    snapshot: DataSnapshot,
    *,
    load: LoadModel = load_model,
    backtest_mae: Callable[[Settings], BacktestMae] = registry_backtest_mae,
) -> tuple[Step, ReplayReport]:
    """One step of the clock: forecast the next day, score what is over, write the report.

    Raises:
        PipelineError: If the replay files are not one unbroken run, the model cannot be
            loaded, a forecast does not pass its schema, or a day cannot be scored.
    """
    step = next_step(settings, snapshot)
    if step.day is not None:
        issue_forecast(step.day, load(settings), snapshot, settings)
    scored = score_days(step, snapshot, settings)
    report = build_report(scored_table(settings), step, settings, backtest_mae(settings))
    write_text(report_to_json(report), replay_report_path(settings))
    log.info(
        "replay step done",
        extra={
            "now": step.now.isoformat(),
            "forecast_day": None if step.day is None else step.day.isoformat(),
            "scored": scored,
            "days_scored": report.days_scored,
            "live_mae": None if report.model is None else report.model.mae,
            "baseline_mae": None if report.baseline_score is None else report.baseline_score.mae,
        },
    )
    return step, report


def run_forecast(
    settings: Settings,
    days: int = 1,
    *,
    load: LoadModel = load_model,
    backtest_mae: Callable[[Settings], BacktestMae] = registry_backtest_mae,
) -> None:
    """Run ``days`` steps of the replay; stop early at the end of the data.

    ``load`` and ``backtest_mae`` are the model and the registry; tests pass their own.

    Raises:
        ForecastError: If ``days`` is not positive.
        PipelineError: See ``run_step``.
    """
    if days < 1:
        raise ForecastError(f"--days must be 1 or more, not {days}.")
    snapshot = load_snapshot(settings)
    for _ in range(days):
        step, report = run_step(settings, snapshot, load=load, backtest_mae=backtest_mae)
        if step.day is None:
            log.info(
                "replay over: the data has no day left to forecast",
                extra={"last_forecast_day": str(report.last_forecast_day)},
            )
            break
