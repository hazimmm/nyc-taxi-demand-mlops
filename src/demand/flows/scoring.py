"""Live accuracy: the pickups that were counted, next to the forecasts that were issued.

**When a day is scored.** When the simulated clock has passed its end (00:00 on the next
day): only then are all its pickups counted. Before that its actuals are not read, even
though the data snapshot already holds them, exactly as no forecast reads a pickup newer
than its issue moment.

**Where the actuals go: a table of their own, ``data/predictions/scored/``.** A forecast
file in ``forecasts/`` is written once, at its issue moment, and never opened for writing
again; its ``actual`` stays empty. When the day is over, its rows are copied into
``scored/`` with ``actual`` filled in; every other column is copied as it was issued.
Filling ``actual`` into the forecast file itself would mean rewriting the record of what
was issued, after the fact, every day; one bug in that rewrite could change a forecast
and leave no trace of the original.
With two tables the record of what was said cannot change, and the scored table is a
plain function of it and of the counted pickups. Both follow ``predictions_schema``, so
the monitoring reads either with the same code: ``forecasts/`` for what the model said
(prediction drift, also on days not yet over), ``scored/`` for how right it was.

**The scores.** MAE, RMSE and sMAPE (``models/metrics.py``) of every scored day, of every
calendar week (Monday to Sunday) and of all scored rows pooled, for the model and for
the weekly baseline on the same rows. The baseline is ``lag_168h`` as the replay's own
inputs hold it (``data/predictions/inputs/``): the pickups of the same zone a week
before, which was known at the issue moment, as it was in the backtest
(``models/baseline.py``). The pooled MAE of each model version is divided by the backtest
MAE of that version, the number the monitoring compares with
``monitoring.mae_degradation_ratio``; this module reports the ratio and raises no alert.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pandera.errors
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.errors import PipelineError
from demand.flows.clock import (
    Step,
    day_end,
    day_file,
    first_replay_day,
    forecasts_dir,
    inputs_dir,
    scored_dir,
    stored_days,
)
from demand.models.backtest import Score, score
from demand.models.baseline import HOURS_PER_WEEK, WEEKLY_NAIVE
from demand.schemas import (
    ACTUAL,
    FORECAST,
    HOUR_START,
    MODEL_VERSION,
    TARGET,
    ZONE_ID,
    lag_column,
    predictions_schema,
)

REPLAY_REPORT_NAME = "replay_report.json"
BASELINE = "baseline"
DAY = "day"
_RATIO_DECIMALS = 4

BacktestMae = Callable[[str], float | None]
"""The backtest MAE recorded for a model version, or ``None`` if none was recorded."""


class ScoringError(PipelineError):
    """A day that is over cannot be scored; the message says why."""


class DayScore(BaseModel):
    """The scores of one day that is over."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    model_version: str
    model: Score
    baseline: Score


class WeekScore(BaseModel):
    """The scores of the scored days of one calendar week, Monday to Sunday."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    week_start: date
    """The Monday of the week."""
    days: int
    model: Score
    baseline: Score


class VersionScore(BaseModel):
    """How one model version did live, next to how its recipe did in the backtest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_version: str
    days: int
    live: Score
    backtest_mae: float | None
    live_to_backtest_mae: float | None
    """Live MAE / backtest MAE; ``None`` if no backtest MAE was recorded."""


class ReplayReport(BaseModel):
    """Where the replay stands, and how the forecasts have done so far."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    clock: datetime
    """The simulated moment of the last step."""
    first_replay_day: date
    replay_start_note: str
    days_forecast: int
    last_forecast_day: date | None
    days_scored: int
    first_scored_day: date | None
    last_scored_day: date | None
    baseline: str
    model: Score | None
    """All scored rows pooled; ``None`` before the first day is scored."""
    baseline_score: Score | None
    days_model_beat_baseline: int
    """Scored days on which the model's MAE was lower than the baseline's."""
    versions: list[VersionScore]
    weeks: list[WeekScore]
    days: list[DayScore]


def replay_report_path(settings: Settings) -> Path:
    """Where the replay report is stored."""
    return settings.paths.reports_dir / REPLAY_REPORT_NAME


def with_actuals(forecast: pd.DataFrame, history: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The issued forecast of a day that is over, with ``actual`` filled from ``history``.

    Raises:
        ScoringError: If the history lacks a zone and hour of the forecast, or the result
            does not pass ``predictions_schema`` (a forecast file changed by hand).
    """
    counted = history[[ZONE_ID, HOUR_START, TARGET]].astype(
        {HOUR_START: forecast[HOUR_START].dtype}
    )
    merged = forecast.drop(columns=ACTUAL).merge(
        counted, on=[ZONE_ID, HOUR_START], how="left", validate="one_to_one"
    )
    missing = int(merged[TARGET].isna().sum())
    if missing:
        raise ScoringError(
            f"The demand history has no pickups for {missing} zone-hours of a forecast "
            f"starting {forecast[HOUR_START].min()}. The forecast is kept; rebuild the "
            "data with `dvc repro` and run the step again."
        )
    merged[ACTUAL] = merged.pop(TARGET).astype("float64")
    scored = merged[list(forecast.columns)]
    try:
        validated: pd.DataFrame = predictions_schema(settings).validate(scored)
    except pandera.errors.SchemaError as error:
        raise ScoringError(
            f"A scored forecast does not pass predictions_schema: {error}"
        ) from error
    return validated


def scorable_days(settings: Settings, now: pd.Timestamp) -> list[date]:
    """The forecast days that are over at ``now`` and have not been scored yet."""
    scored = set(stored_days(scored_dir(settings)))
    return [
        day
        for day in stored_days(forecasts_dir(settings))
        if day_end(day, settings) <= now and day not in scored
    ]


def scored_table(settings: Settings) -> pd.DataFrame:
    """Every scored row, with its day and the weekly baseline of the same zone and hour.

    Raises:
        ScoringError: If the inputs of a scored day do not hold ``lag_168h``.
    """
    baseline_column = lag_column(HOURS_PER_WEEK)
    tables = []
    for day in stored_days(scored_dir(settings)):
        scored = pd.read_parquet(day_file(scored_dir(settings), day))
        inputs = pd.read_parquet(day_file(inputs_dir(settings), day))
        if baseline_column not in inputs.columns:
            raise ScoringError(
                f"The inputs of {day.isoformat()} hold no {baseline_column}, so the weekly "
                "baseline cannot be scored. features.lags must include 168, as the "
                "backtest requires."
            )
        baseline = inputs[[ZONE_ID, HOUR_START, baseline_column]].rename(
            columns={baseline_column: BASELINE}
        )
        table = scored.merge(baseline, on=[ZONE_ID, HOUR_START], how="left", validate="one_to_one")
        table[DAY] = day
        tables.append(table)
    if not tables:
        return pd.DataFrame(
            columns=[ZONE_ID, HOUR_START, FORECAST, ACTUAL, MODEL_VERSION, BASELINE, DAY]
        )
    return pd.concat(tables, ignore_index=True)


def _scores(rows: pd.DataFrame) -> tuple[Score, Score]:
    actual = rows[ACTUAL].to_numpy(dtype="float64")
    return (
        score(actual, rows[FORECAST].to_numpy(dtype="float64")),
        score(actual, rows[BASELINE].to_numpy(dtype="float64")),
    )


def build_report(
    table: pd.DataFrame, step: Step, settings: Settings, backtest_mae: BacktestMae
) -> ReplayReport:
    """The replay report from ``scored_table`` and the step the clock is at."""
    forecast_days = stored_days(forecasts_dir(settings))
    days = []
    for day, rows in table.groupby(DAY, sort=True):
        model, baseline = _scores(rows)
        versions = sorted(rows[MODEL_VERSION].unique().tolist())
        days.append(
            DayScore(day=day, model_version=",".join(versions), model=model, baseline=baseline)
        )
    weeks = []
    week_starts = table[DAY].map(lambda day: day - timedelta(days=day.weekday()))
    for week_start, rows in table.groupby(week_starts, sort=True):
        model, baseline = _scores(rows)
        weeks.append(
            WeekScore(
                week_start=week_start, days=rows[DAY].nunique(), model=model, baseline=baseline
            )
        )
    versions = []
    for version, rows in table.groupby(MODEL_VERSION, sort=True):
        live, _ = _scores(rows)
        recorded = backtest_mae(str(version))
        ratio = None if recorded is None else round(live.mae / recorded, _RATIO_DECIMALS)
        versions.append(
            VersionScore(
                model_version=str(version),
                days=rows[DAY].nunique(),
                live=live,
                backtest_mae=recorded,
                live_to_backtest_mae=ratio,
            )
        )
    pooled = _scores(table) if len(table) else (None, None)
    first = first_replay_day(settings)
    skipped = pd.date_range(settings.data.replay_start_date, first, inclusive="left")
    note = (
        f"The replay starts on {first.isoformat()}, not on "
        f"{settings.data.replay_start_date.isoformat()}. Not forecast: "
        f"{', '.join(day.date().isoformat() for day in skipped)}. The forecast for such a day "
        "was due before the end of the training data, when the model did not exist yet."
    )
    return ReplayReport(
        clock=step.now.to_pydatetime(),
        first_replay_day=first,
        replay_start_note=note,
        days_forecast=len(forecast_days),
        last_forecast_day=forecast_days[-1] if forecast_days else None,
        days_scored=len(days),
        first_scored_day=days[0].day if days else None,
        last_scored_day=days[-1].day if days else None,
        baseline=f"{WEEKLY_NAIVE} ({lag_column(HOURS_PER_WEEK)})",
        model=pooled[0],
        baseline_score=pooled[1],
        days_model_beat_baseline=sum(day.model.mae < day.baseline.mae for day in days),
        versions=versions,
        weeks=weeks,
        days=days,
    )
