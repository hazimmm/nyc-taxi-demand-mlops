"""The backtest: score forecasters on past weeks as if they had been used then.

Layout of the report (``settings.paths.reports_dir``)::

    reports/
      backtest_report.json    folds, row counts and the scores of every forecaster

The report contains no time stamp, and every score is rounded to six decimals, so the
same feature table and the same config always give the same file, byte for byte.

**What a fold is.** The test weeks are the last ``backtest.n_folds`` blocks of
``backtest.fold_days`` days of the training period, the last block ending on
``data.train_end_date``. With 4 folds of 7 days and a training period that ends on
30 April 2025 these are 3-9, 10-16, 17-23 and 24-30 April. For every fold each
forecaster is fitted on the rows *before* the test week, from the first row of the table
(an expanding window), and then forecasts every zone and hour of the week.

**Days are local days.** A test week runs from 00:00 local time on its first day to
23:00 on its last. The boundaries are built from calendar dates and then placed in
``project.timezone``; they are never found by adding 168 hours. A week in which the
clocks go forward therefore has 167 rows per zone and one in which they go back has
169, and no fold starts at 01:00 or 23:00.

**Nothing after the training period is read.** May and June are replayed later as
"production", and a backtest that had seen them would spoil that test. Two things make
sure of it. ``load_training_features`` asks the Parquet reader for the rows before the
end of the training period only, so later rows never reach memory. And ``backtest``
refuses a table that contains such a row: it does not quietly cut it, because a caller
that passes one has a bug worth knowing about.

**The horizon, and what the forecast is for.** ``h`` below is the forecast horizon,
``features.horizon_hours``: 48 hours in the config of this project, and every date and
number in this section is worked out for that value. A row for target hour ``t`` reads
no pickups newer than the hour that starts at ``t - h`` (``features.py``). That hour is
complete at ``t - h + 1``, so the forecast for ``t`` can be issued from that moment on,
``h - 1`` = 47 hours before ``t`` starts.

For a whole local day the last hour decides. Take Thursday 3 April 2025. Its last hour,
23:00, reads the hour that starts 48 hours earlier, Tuesday 1 April 23:00, which is
complete at Wednesday 2 April 00:00. So **the forecast for all 24 hours of a day can be
issued at once at 00:00 on the day before it**: the first hour of the day then starts
24 hours later and its last hour 47 hours later. That is the product: at the first
moment of today, the whole of tomorrow. (With ``h = 24`` the same sum gives 00:00 on
the day itself: a forecast for the day that is just beginning.) Two details:

* The features of an hour depend on that hour only, not on the moment the forecast is
  issued. Thursday 00:00 reads the pickups of Tuesday 00:00 and nothing newer, although
  23 more hours are complete by Wednesday 00:00. Issuing the forecast later on
  Wednesday therefore changes nothing; it uses none of Wednesday's pickups.
* The 48 hours are elapsed hours. For the day the clocks go forward and the day after
  it (9 and 10 March 2025) the moment is an hour earlier, so 00:00 on the day before
  still works. For the day the clocks go back and the day after it, it is 01:00 on the
  day before instead of 00:00. No such day lies in the data of this project (January
  to June 2025).

**Leakage at the fold boundary.** Leakage through the *features* is impossible, whatever
table a row is built from: see above. What is left is the *target* of the training
rows. A model learns from the pickups of every training hour, so the question is
whether it could really have known them when it made its first forecast.

Take the first fold. Its test week starts at 00:00 on Thursday 3 April. The newest input
of that hour is the pickups of the hour that started 48 hours earlier, Tuesday 1 April
00:00, which is complete at Tuesday 01:00. So the features allow this forecast to be
issued on Tuesday at 01:00, 47 hours before the hour starts. A model fitted on every row
before Thursday 00:00 would have learned from the pickups of Tuesday 01:00 to Wednesday
23:00, which did not exist yet at that moment. That is a leak of 47 hours, at the start
of every fold, in the target and not in a feature.

The rule that closes it is the rule the features already follow: **a model may learn
from no target that its own features would not be allowed to read.** For the first test
hour that is everything up to ``h`` hours before it. So the training rows of a fold end
``horizon_hours`` real hours before its first test hour, and the ``h - 1`` hourly rows
per zone in between (47 here) are used for neither. Later test hours could have had a
newer model; they get the same, older one, which makes the test slightly harder than
needed, never easier. The first fold on a timeline (each time is the start of an hour)::

    Tue 1 April 00:00   last training hour; it is complete at 01:00
    Tue 1 April 01:00   first of the 47 hours that are used for neither
    Wed 2 April 00:00   the forecast for all of Thursday can be issued
    Wed 2 April 23:00   last of the 47 hours that are used for neither
    Thu 3 April 00:00   first test hour, 48 hours after the last training hour
    Wed 9 April 23:00   last test hour, 215 hours after the last training hour

What a fold simulates, exactly: the model is retrained **once a week**, at the earliest
moment the first forecast of the week can be issued (01:00 two days before the week
starts, here Tuesday 1 April), on every hour that is complete by then. It is then
**frozen for the week**. Every hour of the week is forecast from its own features,
which read pickups up to 48 hours before that hour and nothing newer. At the last test
hour the model is one hour short of nine days old; the newest pickups it reads are
always exactly 48 hours older than the hour it forecasts.

The gap is strict for the product, and never loose. The forecast for the first test
day is issued at 00:00 on the day before (Wednesday 2 April), and a model fitted at
that moment could have learned from every hour up to Tuesday 23:00. The fold stops at
Tuesday 00:00, so it gives up 23 hours of training data out of the 1,945 (first fold)
to 2,449 (last fold) hours per zone it has, which changes no conclusion. The forecasts
themselves are the same in both readings, because the features of an hour do not
depend on when the forecast is issued.

What the backtest does *not* simulate: it assumes the pickups of an hour are known the
moment the hour ends, and that the weather column is what a weather forecast issued at
the moment of the demand forecast, up to 47 hours before the hour, would have said. And
the folds lie inside the period the zone list and the outlier statistics were chosen
from, which is the same for every forecaster.

**Outlier hours are scored.** ``is_outlier`` marks hours whose pickups were unusual for
their zone, hour and kind of day. They really happened and a deployed forecast faces
them, so they count in every score. One extra number, ``overall_without_outliers``,
shows how much of the error they carry. It is a diagnosis, not a result: the flag is
calculated from the pickups themselves, so nobody can know in advance which hours it
will leave out.

**Pooled scores.** ``overall`` is one mean over all test rows of all folds, not the
average of the fold scores. With folds of equal size the two agree for the MAE, but not
for the RMSE, and not when a week has a clock change.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.aggregate import training_period_end
from demand.data.features import FloatArray
from demand.data.gold import gold_features_path
from demand.data.silver import report_to_json, write_text
from demand.errors import PipelineError
from demand.logs import get_logger
from demand.models.baseline import Forecaster
from demand.models.metrics import Metrics, evaluate
from demand.schemas import HOUR_START, IS_OUTLIER, TARGET, ZONE_ID, feature_columns

log = get_logger(__name__)

BACKTEST_REPORT_NAME = "backtest_report.json"
FOLD = "fold"
_SCORE_DECIMALS = 6


class MissingGoldFileError(PipelineError, FileNotFoundError):
    """The gold feature table is not on disk, or was built with another config."""


class TooLittleHistoryError(PipelineError, ValueError):
    """The feature table does not reach far enough back for the configured folds."""


class Fold(BaseModel):
    """The boundaries of one fold. Each is the start of an hour, and each is included."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int
    """1 is the oldest test week."""
    last_train_hour: datetime
    """The newest hour the forecasters learn from: one horizon before the test week."""
    first_test_hour: datetime
    """00:00 local time on the first day of the test week."""
    last_test_hour: datetime
    """23:00 local time on the last day of the test week."""


class FoldSummary(Fold):
    """A fold with the number of rows that were used in it."""

    train_rows: int
    test_rows: int
    outlier_rows: int
    """Test rows that are flagged as outliers. They are scored like any other row."""


class Score(Metrics):
    """The metrics of one set of test rows, with the number of rows behind them."""

    rows: int


class ForecasterScores(BaseModel):
    """Every score of one forecaster."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: Score
    """All test rows of all folds as one set. This is the number to quote."""
    overall_without_outliers: Score | None
    """The same without the rows flagged as outliers; ``None`` if every row is flagged.

    A diagnosis only: the flag is calculated from the pickups that are being forecast.
    """
    per_fold: dict[int, Score]
    per_zone: dict[int, Score]


class BacktestReport(BaseModel):
    """The result of a backtest; saved as JSON next to the other reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_end_date: date
    horizon_hours: int
    folds: list[FoldSummary]
    test_rows: int
    outlier_rows: int
    forecasters: dict[str, ForecasterScores]
    """Scores by forecaster name, in the order the forecasters were given."""


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def backtest_report_path(reports_dir: Path) -> Path:
    """Where the backtest report is stored."""
    return reports_dir / BACKTEST_REPORT_NAME


def fold_windows(settings: Settings) -> list[Fold]:
    """The folds of the configured backtest, oldest first, from the config alone.

    The last test week ends with the training period, so no fold can reach into the
    replayed months. See the module docstring for the local days and for the gap between
    the training rows and the test week.
    """
    config = settings.backtest
    timezone = settings.project.timezone
    # Subtracting a Timedelta from a timezone-aware pandas time moves by real, elapsed
    # hours, which is how the horizon is defined.
    horizon = pd.Timedelta(hours=settings.features.horizon_hours)
    folds = []
    for number in range(1, config.n_folds + 1):
        days_before_end = (config.n_folds - number + 1) * config.fold_days
        first_day = settings.data.replay_start_date - timedelta(days=days_before_end)
        test_start = pd.Timestamp(first_day, tz=timezone)
        # Midnight after the last test day. On the calendar, so a week with a clock
        # change still ends at midnight.
        test_end = pd.Timestamp(first_day + timedelta(days=config.fold_days), tz=timezone)
        folds.append(
            Fold(
                number=number,
                last_train_hour=test_start - horizon,
                first_test_hour=test_start,
                last_test_hour=test_end - pd.Timedelta(hours=1),
            )
        )
    return folds


def score(actual: FloatArray, forecast: FloatArray) -> Score:
    """The rounded metrics of one set of rows."""
    metrics = evaluate(actual, forecast)
    return Score(
        rows=len(actual),
        mae=round(metrics.mae, _SCORE_DECIMALS),
        rmse=round(metrics.rmse, _SCORE_DECIMALS),
        smape=round(metrics.smape, _SCORE_DECIMALS),
    )


def score_by(keys: pd.Series, actual: FloatArray, forecast: FloatArray) -> dict[int, Score]:
    """One score for every value of ``keys`` (a fold number or a zone), smallest first."""
    scores = {}
    for key in sorted(keys.unique().tolist()):
        chosen = (keys == key).to_numpy()
        scores[int(key)] = score(actual[chosen], forecast[chosen])
    return scores


def score_forecaster(rows: pd.DataFrame, forecast: FloatArray) -> ForecasterScores:
    """Every score of one forecaster.

    Args:
        rows: All test rows of all folds: zone, fold number, pickups and outlier flag.
        forecast: The forecast for every row of ``rows``, in the same order.
    """
    actual = rows[TARGET].to_numpy(dtype="float64")
    usual = ~rows[IS_OUTLIER].to_numpy()
    return ForecasterScores(
        overall=score(actual, forecast),
        overall_without_outliers=score(actual[usual], forecast[usual]) if usual.any() else None,
        per_fold=score_by(rows[FOLD], actual, forecast),
        per_zone=score_by(rows[ZONE_ID], actual, forecast),
    )


def backtest(
    table: pd.DataFrame, forecasters: Mapping[str, Forecaster], settings: Settings
) -> BacktestReport:
    """Fit and score every forecaster on every fold.

    Args:
        table: Rows of the gold feature table, none of them after the training period.
        forecasters: The forecasters to compare, by name. Each is fitted again for every
            fold, so one object per name is enough.
        settings: Project settings (folds, horizon, training period, timezone).

    Raises:
        ValueError: If ``table`` has a row after the training period, or a forecaster
            does not return one finite forecast per row.
        TooLittleHistoryError: If a fold has no training rows or no test rows.
    """
    hours = table[HOUR_START]
    after_training = int((hours >= training_period_end(settings)).sum())
    if after_training:
        raise ValueError(
            f"the backtest was given {after_training} rows after data.train_end_date "
            f"({settings.data.train_end_date}); those are reserved for the replay"
        )
    # The zone is a model input, so it is in ``feature_columns``. The target and the
    # outlier flag are deliberately not handed to ``predict``.
    inputs = [HOUR_START, *feature_columns(settings)]

    summaries: list[FoldSummary] = []
    scored_rows: list[pd.DataFrame] = []
    forecasts: dict[str, list[FloatArray]] = {name: [] for name in forecasters}
    for fold in fold_windows(settings):
        train = table[hours <= fold.last_train_hour]
        test = table[(hours >= fold.first_test_hour) & (hours <= fold.last_test_hour)]
        if train.empty or test.empty:
            raise TooLittleHistoryError(
                f"Fold {fold.number} of the backtest has {len(train)} training rows and "
                f"{len(test)} test rows. Its test week starts at {fold.first_test_hour} "
                f"and the feature table covers {hours.min()} to {hours.max()}. Lower "
                "backtest.n_folds or backtest.fold_days, or start the data earlier."
            )
        for name, forecaster in forecasters.items():
            forecaster.fit(train)
            forecast = np.asarray(forecaster.predict(test[inputs]), dtype="float64")
            if forecast.shape != (len(test),):
                raise ValueError(
                    f"forecaster '{name}' returned an array of shape {forecast.shape} for "
                    f"{len(test)} rows; it must return one forecast per row"
                )
            forecasts[name].append(forecast)
        scored_rows.append(test[[ZONE_ID, TARGET, IS_OUTLIER]].assign(**{FOLD: fold.number}))
        summaries.append(
            FoldSummary(
                **fold.model_dump(),
                train_rows=len(train),
                test_rows=len(test),
                outlier_rows=int(test[IS_OUTLIER].sum()),
            )
        )

    rows = pd.concat(scored_rows, ignore_index=True)
    return BacktestReport(
        train_end_date=settings.data.train_end_date,
        horizon_hours=settings.features.horizon_hours,
        folds=summaries,
        test_rows=len(rows),
        outlier_rows=int(rows[IS_OUTLIER].sum()),
        forecasters={
            name: score_forecaster(rows, np.concatenate(parts)) for name, parts in forecasts.items()
        },
    )


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def load_training_features(settings: Settings) -> pd.DataFrame:
    """Read the rows of the gold feature table that lie in the training period.

    The filter is applied by the Parquet reader, so the rows of the replayed months are
    not loaded at all.

    Raises:
        MissingGoldFileError: If the file is missing, or lacks a column the config asks
            for (it was built with other lags or weather variables).
    """
    path = gold_features_path(settings.paths.gold)
    advice = "Run `dvc repro` to build the gold layer."
    if not path.is_file():
        raise MissingGoldFileError(f"{path.as_posix()} is missing. {advice}")
    columns = [HOUR_START, TARGET, IS_OUTLIER, *feature_columns(settings)]
    in_file = pyarrow.parquet.read_schema(path).names
    missing = [column for column in columns if column not in in_file]
    if missing:
        raise MissingGoldFileError(
            f"{path.as_posix()} has no column {', '.join(missing)}: it was built with a "
            f"different config. {advice}"
        )
    return pd.read_parquet(
        path, columns=columns, filters=[(HOUR_START, "<", training_period_end(settings))]
    )


def backtest_from_gold(settings: Settings, forecasters: Mapping[str, Forecaster]) -> BacktestReport:
    """Backtest ``forecasters`` on the gold feature table and write the report.

    Safe to run again: nothing but the report is written, and it comes out identical for
    the same feature table and config.

    Raises:
        MissingGoldFileError: If the gold feature table is missing or out of date.
        TooLittleHistoryError: If the table is too short for the configured folds.
    """
    started = time.monotonic()
    report = backtest(load_training_features(settings), forecasters, settings)
    report_path = backtest_report_path(settings.paths.reports_dir)
    write_text(report_to_json(report), report_path)

    for name, scores in report.forecasters.items():
        without_outliers = scores.overall_without_outliers
        log.info(
            "forecaster scored",
            extra={
                "forecaster": name,
                **scores.overall.model_dump(),
                "mae_without_outliers": without_outliers.mae if without_outliers else None,
                "mae_per_fold": {str(fold): s.mae for fold, s in scores.per_fold.items()},
            },
        )
    log.info(
        "backtest finished",
        extra={
            "report": str(report_path),
            "folds": len(report.folds),
            "test_rows": report.test_rows,
            "outlier_rows": report.outlier_rows,
            "forecasters": list(report.forecasters),
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return report
