"""Stage functions of the modelling pipeline, called by ``python -m demand <command>``.

Each function takes the ``Settings`` object first and returns ``None``.
"""

from __future__ import annotations

from demand.config import Settings
from demand.models.backtest import backtest_from_gold
from demand.models.baseline import baselines


def run_backtest(settings: Settings) -> None:
    """Score the baseline forecasts on the last weeks of the training period.

    Safe to run again: only ``reports/backtest_report.json`` is written, and it comes
    out identical. The work itself is in ``backtest.py``; the forecasters come from
    ``baseline.py``. A model is compared with the baselines by adding it to the
    forecasters that are passed on here.

    Raises:
        PipelineError: If the gold feature table is missing or out of date (run
            `dvc repro`), if ``features.lags`` lacks the lag the weekly baseline
            copies, or if the table is too short for the configured folds.
    """
    backtest_from_gold(settings, baselines(settings))


def run_training(settings: Settings) -> None:
    """Train and register a challenger model. Not built yet.

    Raises:
        NotImplementedError: Always. The command line reports it in one line.
    """
    raise NotImplementedError("demand.models.pipeline.run_training (training arrives later)")
