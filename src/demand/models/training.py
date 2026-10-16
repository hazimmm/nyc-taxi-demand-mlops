"""Training: score the forecasters, fit the model that will be used, and record the run.

Layout of the report (``settings.paths.reports_dir``)::

    reports/
      backtest_report.json    written first, by the backtest (``backtest.py``)
      training_report.json    the MLflow run, the model version, the data version and
                              the headline scores of this training run
      shap_importance.json    how much every input moves the forecasts of the final
      shap_summary.png        model, as numbers and as a picture (``explain.py``)

One training run does four things, in this order:

1. **The backtest**, for the baseline and for the model, exactly as
   ``python -m demand backtest`` does it. Its scores say how good the *recipe* is
   (these inputs, these parameters), measured on four weeks the model had not seen.
2. **The final fit.** The recipe is fitted once more, on more rows than any fold had.
   This is the model that is stored and later served.
3. **The explanation.** For a fixed sample of the rows of the final fit, the
   contribution of every input to the forecast is calculated (SHAP, ``explain.py``).
4. **The record.** Parameters, scores, the model, the explanation and the facts that
   make the run reproducible go to MLflow (``tracking.py``), and a short report is
   written to disk.

**The scores belong to the recipe, not to the stored model.** The final model has no
score of its own: every April week it could be tested on is now part of its training
rows. The backtest scores are the honest estimate of how such a model performs on
weeks it has not seen, and the final model has slightly more data than the models that
earned them. Its first real test is the replay of May and June.

**The rows of the final fit.** Every row of the feature table inside the training
period: from the first row (9 January 2025 at 23:00, after the warm-up of the features)
to 30 April at 23:00, with no gap at the end. Nothing after ``data.train_end_date`` is
read, as in the backtest.

In a fold of the backtest the training rows stop 48 hours before the test week
(``backtest.py``), and the final fit must follow the same rule: *a model may learn
from no pickups that did not exist yet when it made its forecast.* It does, with the
moment worked out. The pickups of 30 April 23:00 are complete at 00:00 on 1 May, so
that is the earliest moment this model can exist. A forecast for a whole day is issued
at 00:00 on the day before it. So:

* the forecast for **2 May** is issued at 00:00 on 1 May, and this model exists then.
  From 2 May on, every forecast it makes is one it could really have made. There is no
  gap because nothing has to be kept apart: the days it forecasts are not in the table.
* the forecast for **1 May** was due at 00:00 on 30 April. This model did not exist
  then: it has learned from the pickups of 30 April. A forecast it makes for 1 May
  still reads no pickups newer than 48 hours before each hour (the features see to
  that), but it comes from a model that is a day younger than it could have been.
  Whoever scores the replay must either start at 2 May or say so.

The backtest is stricter than this by 23 hours (its docstring explains why that does
not matter), so the backtest scores are not flattered by the difference.

**What makes a run reproducible.** Three facts are recorded with every run:

* the *git commit* of the code, and whether files were changed or added since that
  commit. A run from changed files cannot be reproduced from the commit alone, and the
  record says so (``git_dirty``). To get a clean record, commit the code first and
  train after.
* the *data version*: the checksum DVC wrote into ``dvc.lock`` for the gold folder when
  it built it. It is read, not calculated, so it is right when the gold layer was built
  by ``dvc repro``. After ``python -m demand gold`` by hand the lock describes the
  older files and the tag is stale; training through ``dvc repro`` cannot go wrong in
  this way.
* the *config*: the parameters of the model and the keys that define the features and
  the backtest.

A fact that cannot be found (no git, no ``dvc.lock``) is recorded as ``unknown`` and a
warning is logged. The run is not refused: the tests and a first try-out have neither.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.aggregate import training_period_end
from demand.data.silver import cleaning_report_path, report_to_json, write_text
from demand.errors import PipelineError
from demand.logs import get_logger
from demand.models.backtest import (
    BacktestReport,
    Score,
    backtest_from_gold,
    backtest_report_path,
    load_training_features,
)
from demand.models.baseline import WEEKLY_NAIVE, Forecaster
from demand.models.explain import ShapReport, explain_to_reports, shap_plot_path, shap_report_path
from demand.models.gbm import LIGHTGBM, LightGbmForecaster
from demand.models.tracking import log_training_run
from demand.schemas import HOUR_START, feature_columns

log = get_logger(__name__)

TRAINING_REPORT_NAME = "training_report.json"
DVC_LOCK_FILE = Path("dvc.lock")
UNKNOWN = "unknown"
INPUT_EXAMPLE_ROWS = 5
_PERCENT_DECIMALS = 3


class GitState(BaseModel):
    """The commit the code was run from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    commit: str
    """The full hash of the commit, or ``unknown``."""
    dirty: bool | None
    """True if files were changed or added since the commit; ``None`` if unknown."""


class TrainingReport(BaseModel):
    """The result of a training run; saved as JSON next to the other reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    """The MLflow run that holds the parameters, the scores and the model."""
    registered_model: str
    model_version: int
    data_version: str
    git_commit: str
    git_dirty: bool | None
    train_rows: int
    """Rows of the final fit."""
    first_train_hour: datetime
    last_train_hour: datetime
    model: Score
    """Backtest score of the model over all test rows."""
    baseline: Score
    """Backtest score of the weekly baseline over the same rows."""
    mae_improvement_percent: float
    """How much lower the MAE of the model is than that of the baseline."""


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def training_report_path(reports_dir: Path) -> Path:
    """Where the training report is stored."""
    return reports_dir / TRAINING_REPORT_NAME


def improvement_percent(model_mae: float, baseline_mae: float) -> float:
    """By how many percent the MAE of the model is below that of the baseline.

    Positive means the model is better: 25 against 100 is 75 percent. Negative means
    the model is worse than copying last week.

    Raises:
        PipelineError: If the MAE of the baseline is 0. Then the baseline is perfect on
            the test weeks and no percentage can be calculated; on real data this
            means the test rows are wrong.
    """
    if baseline_mae == 0:
        raise PipelineError(
            "The weekly baseline has an MAE of 0 on the test weeks, so the model cannot "
            "be compared with it. Check the feature table: the target and lag_168h are "
            "identical on every test row."
        )
    return round((baseline_mae - model_mae) / baseline_mae * 100, _PERCENT_DECIMALS)


def run_params(settings: Settings, train_rows: int) -> dict[str, object]:
    """The settings that define a training run, as MLflow parameters."""
    features = settings.features
    return {
        **{f"model.{name}": value for name, value in settings.model.params.items()},
        "model.seed": settings.project.seed,
        "features.horizon_hours": features.horizon_hours,
        "features.lags": features.lags,
        "features.rolling_windows": features.rolling_windows,
        "features.outlier_zscore": features.outlier_zscore,
        "feature_columns": feature_columns(settings),
        "data.train_end_date": settings.data.train_end_date.isoformat(),
        "data.zones": settings.data.zones,
        "backtest.n_folds": settings.backtest.n_folds,
        "backtest.fold_days": settings.backtest.fold_days,
        "train_rows": train_rows,
    }


def run_metrics(report: BacktestReport) -> dict[str, float]:
    """The backtest scores of the model and of the baseline, as MLflow metrics.

    For each of the two: ``<name>.mae``, ``.rmse`` and ``.smape`` over all test rows,
    and ``<name>.fold_<n>.mae`` and so on for every fold. Then
    ``improvement.mae_percent``, and the same for every fold. The scores per zone are
    not repeated here; they are in the backtest report, which is stored with the run.
    """
    metrics: dict[str, float] = {}
    for name in (LIGHTGBM, WEEKLY_NAIVE):
        scores = report.forecasters[name]
        by_group = {name: scores.overall} | {
            f"{name}.fold_{fold}": score for fold, score in scores.per_fold.items()
        }
        for group, score in by_group.items():
            metrics |= {f"{group}.{metric}": value for metric, value in _errors(score).items()}
    model, baseline = report.forecasters[LIGHTGBM], report.forecasters[WEEKLY_NAIVE]
    metrics["improvement.mae_percent"] = improvement_percent(
        model.overall.mae, baseline.overall.mae
    )
    for fold, score in model.per_fold.items():
        metrics[f"improvement.fold_{fold}.mae_percent"] = improvement_percent(
            score.mae, baseline.per_fold[fold].mae
        )
    return metrics


def _errors(score: Score) -> dict[str, float]:
    return {"mae": score.mae, "rmse": score.rmse, "smape": score.smape}


def shap_metrics(report: ShapReport) -> dict[str, float]:
    """The mean absolute SHAP value of every input, as MLflow metrics.

    One metric per input, ``shap.mean_abs.<input>``, so that two training runs can be
    compared in MLflow: an input whose value jumps between two runs has changed its
    role in the model.
    """
    return {f"shap.mean_abs.{name}": value for name, value in report.mean_abs_shap.items()}


def final_training_rows(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The rows the final model is fitted on: all of ``table``, once it is checked.

    See the module docstring for why every row of the training period is used.

    Raises:
        ValueError: If ``table`` is empty or has a row after the training period. Such
            a row is not cut away quietly: a caller that passes one has a bug.
    """
    after_training = int((table[HOUR_START] >= training_period_end(settings)).sum())
    if after_training:
        raise ValueError(
            f"the final fit was given {after_training} rows after data.train_end_date "
            f"({settings.data.train_end_date}); those are reserved for the replay"
        )
    if table.empty:
        raise ValueError("the final fit was given a table without rows")
    return table


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def git_state(folder: Path) -> GitState:
    """The commit that is checked out in ``folder``, and whether files differ from it.

    Files that git does not track yet count as a difference (a new module changes what
    the code does); files that git ignores do not. Without git, or outside a
    repository, the commit is ``unknown``.
    """
    answers = []
    for command in (["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"]):
        try:
            done = subprocess.run(command, cwd=folder, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            log.warning("git state unknown", extra={"folder": str(folder), "command": command})
            return GitState(commit=UNKNOWN, dirty=None)
        answers.append(done.stdout.strip())
    commit, changes = answers
    return GitState(commit=commit, dirty=bool(changes))


def data_version(lock_file: Path, gold_dir: Path) -> str:
    """The checksum DVC recorded for the gold folder, read from ``dvc.lock``.

    ``dvc.lock`` lists every stage with the checksums of its outputs. The gold folder
    is an output of the stage that builds it, and its checksum (ending in ``.dir``,
    because it stands for a folder) identifies both gold files at once. With it, the
    exact data of a run can be found in the DVC cache again.

    Returns:
        The checksum, or ``unknown`` if there is no lock file or no stage has the gold
        folder among its outputs. See the module docstring for when it can be stale.
    """
    if lock_file.is_file():
        stages = (yaml.safe_load(lock_file.read_text(encoding="utf-8")) or {}).get("stages", {})
        for stage in stages.values():
            for output in stage.get("outs", []):
                # Paths in dvc.lock are relative to the folder the lock file is in.
                same_folder = (lock_file.parent / output["path"]).resolve() == gold_dir.resolve()
                if same_folder and "md5" in output:
                    return str(output["md5"])
    log.warning(
        "data version unknown: run the pipeline with `dvc repro` to record it",
        extra={"lock_file": str(lock_file), "gold_dir": gold_dir.as_posix()},
    )
    return UNKNOWN


def train_from_gold(settings: Settings, candidates: Mapping[str, Forecaster]) -> TrainingReport:
    """Backtest the forecasters, fit the final model, record the run, write the report.

    Nothing is written into the data folder. Running it again records a new MLflow run
    and a new model version; the scores, the model and the SHAP values come out the
    same for the same feature table, config and machine (see ``gbm.py``).

    Args:
        settings: Project settings.
        candidates: The forecasters to backtest, by name (``pipeline.forecasters``).
            The one named ``lightgbm`` is fitted on all training rows, explained and
            stored; the one named ``weekly_naive`` is the baseline it is compared
            with. Only a ``LightGbmForecaster`` can be explained: for any other object
            under that name the explanation is left out, with a warning.

    Raises:
        MissingGoldFileError: If the gold feature table is missing or out of date.
        TooLittleHistoryError: If the table is too short for the configured folds.
        PipelineError: If the baseline is perfect on the test weeks, or the SHAP values
            do not add up to the output of the model.
    """
    started = time.monotonic()
    backtest_report = backtest_from_gold(settings, candidates)
    metrics = run_metrics(backtest_report)

    rows = final_training_rows(load_training_features(settings), settings)
    model = candidates[LIGHTGBM]
    fit_started = time.monotonic()
    model.fit(rows)
    fit_seconds = round(time.monotonic() - fit_started, 2)

    git = git_state(Path.cwd())
    version = data_version(DVC_LOCK_FILE, settings.paths.gold)
    reports_dir = settings.paths.reports_dir
    artifacts = [backtest_report_path(reports_dir)]
    cleaning_report = cleaning_report_path(reports_dir)
    if cleaning_report.is_file():
        artifacts.append(cleaning_report)
    else:
        log.warning("no cleaning report to store", extra={"missing": str(cleaning_report)})
    if isinstance(model, LightGbmForecaster):
        explanation = explain_to_reports(model, rows, reports_dir, settings.project.seed)
        metrics |= shap_metrics(explanation)
        artifacts += [shap_report_path(reports_dir), shap_plot_path(reports_dir)]
    else:
        log.warning("no explanation to store", extra={"forecaster": type(model).__name__})
    logged = log_training_run(
        settings,
        forecaster=model,
        input_example=rows[feature_columns(settings)].head(INPUT_EXAMPLE_ROWS),
        params=run_params(settings, train_rows=len(rows)),
        metrics=metrics | {"final_fit_seconds": fit_seconds},
        tags={
            "forecaster": LIGHTGBM,
            "baseline": WEEKLY_NAIVE,
            "git_commit": git.commit,
            "git_dirty": UNKNOWN if git.dirty is None else str(git.dirty).lower(),
            "data_version": version,
        },
        artifacts=artifacts,
    )

    report = TrainingReport(
        run_id=logged.run_id,
        registered_model=settings.mlflow.registered_model_name,
        model_version=logged.model_version,
        data_version=version,
        git_commit=git.commit,
        git_dirty=git.dirty,
        train_rows=len(rows),
        first_train_hour=rows[HOUR_START].min(),
        last_train_hour=rows[HOUR_START].max(),
        model=backtest_report.forecasters[LIGHTGBM].overall,
        baseline=backtest_report.forecasters[WEEKLY_NAIVE].overall,
        mae_improvement_percent=metrics["improvement.mae_percent"],
    )
    report_path = training_report_path(reports_dir)
    write_text(report_to_json(report), report_path)
    log.info(
        "training finished",
        extra={
            "report": str(report_path),
            **report.model_dump(mode="json", exclude={"model", "baseline"}),
            "model_mae": report.model.mae,
            "baseline_mae": report.baseline.mae,
            "final_fit_seconds": fit_seconds,
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return report
