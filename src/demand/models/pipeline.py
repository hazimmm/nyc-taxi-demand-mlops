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

    Writes ``reports/backtest_report.json`` (the same file ``run_backtest`` writes),
    ``reports/training_report.json``, and the explanation of the final model
    (``reports/shap_importance.json`` and ``reports/shap_summary.png``), and adds one
    run and one model version to the MLflow store named by ``mlflow.tracking_uri``.
    Running it again adds another run and another version; nothing is overwritten
    there. The work is in ``training.py``.

    Raises:
        PipelineError: As for ``run_backtest``.
    """
    # Imported here and not at the top: ``training`` loads MLflow, SHAP and matplotlib,
    # which takes several seconds and is of no use to the backtest command.
    from demand.models.training import train_from_gold

    train_from_gold(settings, forecasters(settings))


def run_tuning(settings: Settings) -> None:
    """Search for better LightGBM parameters and print them, ready for the config.

    The search scores the parameters of the config first and then at most
    ``model.tuning_trials`` other sets, for at most ``model.tuning_timeout_seconds``,
    on the folds of the backtest. Only ``reports/tuning_report.json`` is written. The
    config file is not changed and no model is trained or recorded: the printed block
    has to be pasted under ``model:`` by hand, after reading the numbers above it. The
    work, and what those numbers do and do not prove, is in ``tuning.py``.

    Raises:
        PipelineError: If the gold feature table is missing or out of date (run
            `dvc repro`), if ``model.params`` sets a seed or names a searched parameter
            by another name, or if the table is too short for the configured folds.
    """
    # Imported here and not at the top: only this command needs Optuna.
    from demand.models.tuning import tune_from_gold, tuning_config_text

    report = tune_from_gold(settings)
    # The result goes to standard output on purpose: logs go to standard error, so the
    # lines below can be copied or redirected without any log text in between.
    print(tuning_config_text(report))
