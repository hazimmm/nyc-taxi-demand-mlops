"""Stage functions of the modelling pipeline, called by ``python -m demand <command>``.

Each function takes the ``Settings`` object first and returns ``None``.
"""

from __future__ import annotations

from demand.config import Settings
from demand.models.backtest import backtest_from_gold
from demand.models.baseline import Forecaster, baselines
from demand.models.gbm import LIGHTGBM, lightgbm_forecaster


def forecasters(settings: Settings) -> dict[str, Forecaster]:
    """Every forecaster that is compared: the baselines first, then the model.

    Raises:
        PipelineError: If the weekly baseline cannot be built (see ``baselines``), or
            ``model.params`` sets a seed (see ``gbm.py``).
    """
    return {**baselines(settings), LIGHTGBM: lightgbm_forecaster(settings)}


def run_backtest(settings: Settings) -> None:
    """Score the baseline and the model on the last weeks of the training period.

    Safe to run again: only ``reports/backtest_report.json`` is written, and it comes
    out identical. The work itself is in ``backtest.py``. Nothing is recorded in MLflow
    and no model is kept: this is the quick look, ``run_training`` is the run that
    counts.

    Raises:
        PipelineError: If the gold feature table is missing or out of date (run
            `dvc repro`), if ``features.lags`` lacks the lag the weekly baseline
            copies, if ``model.params`` sets a seed, or if the table is too short for
            the configured folds.
    """
    backtest_from_gold(settings, forecasters(settings))


def run_training(settings: Settings) -> None:
    """Backtest the forecasters, fit the final model and record the run in MLflow.

    Writes ``reports/backtest_report.json`` (the same file ``run_backtest`` writes) and
    ``reports/training_report.json``, and adds one run and one model version to the
    MLflow store named by ``mlflow.tracking_uri``. Running it again adds another run
    and another version; nothing is overwritten there. The work is in ``training.py``.

    Raises:
        PipelineError: As for ``run_backtest``.
    """
    # Imported here and not at the top: ``training`` loads MLflow, which takes about a
    # second and is of no use to the backtest command.
    from demand.models.training import train_from_gold

    train_from_gold(settings, forecasters(settings))
