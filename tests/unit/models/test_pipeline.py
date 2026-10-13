from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pytest
from typer.testing import CliRunner

from demand.cli import app
from demand.config import CONFIG_FILE_ENV, Settings, get_settings
from demand.data.gold import gold_features_path
from demand.data.silver import write_parquet
from demand.errors import PipelineError
from demand.models import training
from demand.models.backtest import BacktestReport, backtest_report_path
from demand.models.baseline import DAILY_NAIVE, WEEKLY_NAIVE
from demand.models.gbm import LIGHTGBM, LightGbmForecaster
from demand.models.pipeline import forecasters, run_backtest, run_training
from demand.models.tracking import LoggedRun
from demand.models.training import TrainingReport, training_report_path
from tests.conftest import CONFIG_FILE, with_features
from tests.unit.models.conftest import (
    FEW_TREES,
    feature_table,
    with_backtest,
    with_mlflow_store,
    with_model_params,
)


@pytest.fixture
def small(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Two folds of two days at the end of April 2025, with a gold feature table on disk.

    The horizon and the lags are written here, not taken from the config file: 48 hours
    and the lags 48 and 168, so the weekly baseline is the only one. The model has 20
    trees instead of 600. The working directory and the MLflow database are under
    ``tmp_path``, so nothing can reach the repository.
    """
    small = with_backtest(settings, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)
    small = with_features(small, horizon_hours=48, lags=[48, 168])
    small = with_mlflow_store(with_model_params(small, FEW_TREES), tmp_path)
    _write_gold(small)
    monkeypatch.chdir(tmp_path)
    return small


def _write_gold(settings: Settings) -> None:
    table = feature_table(settings, "2025-04-24 00:00", hours=9 * 24)
    write_parquet(table, gold_features_path(settings.paths.gold))


def _report(settings: Settings) -> BacktestReport:
    stored = backtest_report_path(settings.paths.reports_dir).read_bytes()
    return BacktestReport.model_validate_json(stored)


def test_forecasters_are_the_baseline_first_and_then_the_model(small: Settings) -> None:
    compared = forecasters(small)

    assert list(compared) == [WEEKLY_NAIVE, LIGHTGBM]
    assert isinstance(compared[LIGHTGBM], LightGbmForecaster)


def test_backtest_stage_scores_the_baseline_and_the_model_and_writes_the_report(
    small: Settings,
) -> None:
    run_backtest(small)

    report = _report(small)
    assert report.horizon_hours == 48
    assert list(report.forecasters) == [WEEKLY_NAIVE, LIGHTGBM]
    assert report.forecasters[WEEKLY_NAIVE].overall.mae == 168.0
    assert report.forecasters[LIGHTGBM].overall.rows == 2 * 2 * 24 * 3
    assert [fold.number for fold in report.folds] == [1, 2]


def test_backtest_stage_scores_both_baselines_when_the_horizon_allows_the_daily_one(
    small: Settings,
) -> None:
    day_ahead = with_features(small, horizon_hours=24, lags=[24, 168])
    _write_gold(day_ahead)

    run_backtest(day_ahead)

    report = _report(day_ahead)
    assert report.horizon_hours == 24
    assert list(report.forecasters) == [WEEKLY_NAIVE, DAILY_NAIVE, LIGHTGBM]
    assert report.forecasters[WEEKLY_NAIVE].overall.mae == 168.0
    assert report.forecasters[DAILY_NAIVE].overall.mae == 24.0
    assert [fold.number for fold in report.folds] == [1, 2]


def test_backtest_stage_writes_nothing_but_the_report(small: Settings) -> None:
    run_backtest(small)

    reports = [file.name for file in small.paths.reports_dir.iterdir()]
    assert reports == ["backtest_report.json"]


def test_backtest_stage_stops_without_the_gold_layer(settings: Settings) -> None:
    with pytest.raises(PipelineError, match="dvc repro"):
        run_backtest(settings)

    assert not settings.paths.reports_dir.exists()


def test_backtest_stage_stops_when_the_weekly_lag_is_not_configured(small: Settings) -> None:
    without_week = with_features(small, horizon_hours=24, lags=[24, 48])

    with pytest.raises(PipelineError, match="lag_168h"):
        run_backtest(without_week)

    assert not small.paths.reports_dir.exists()


@pytest.mark.parametrize("stage", [run_backtest, run_training])
def test_both_stages_stop_before_any_work_when_the_config_sets_a_seed_for_the_model(
    small: Settings, tmp_path: Path, stage: Callable[[Settings], None]
) -> None:
    with_seed = with_model_params(small, FEW_TREES | {"seed": 7})

    with pytest.raises(PipelineError, match=r"Remove seed from model\.params"):
        stage(with_seed)

    assert not small.paths.reports_dir.exists()
    assert not (tmp_path / "mlflow.db").exists()


def test_training_stage_writes_both_reports_and_records_one_run(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    # MLflow itself is replaced here; test_tracking.py records a run in a real store.
    recorded = []

    def record(settings: Settings, **values: object) -> LoggedRun:
        recorded.append(values)
        return LoggedRun(run_id="run-1", model_version=3)

    monkeypatch.setattr(training, "log_training_run", record)

    run_training(small)

    assert len(recorded) == 1
    assert list(_report(small).forecasters) == [WEEKLY_NAIVE, LIGHTGBM]
    stored = training_report_path(small.paths.reports_dir).read_bytes()
    report = TrainingReport.model_validate_json(stored)
    assert (report.run_id, report.model_version) == ("run-1", 3)
    assert report.baseline.mae == 168.0
    assert report.train_rows == 7 * 24 * 3


def test_training_stage_stops_without_the_gold_layer(settings: Settings) -> None:
    with pytest.raises(PipelineError, match="dvc repro"):
        run_training(settings)

    assert not settings.paths.reports_dir.exists()


def test_train_command_reports_a_missing_gold_layer_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real config file, with the data and report folders moved away from the repository.
    monkeypatch.setenv(CONFIG_FILE_ENV, str(CONFIG_FILE))
    monkeypatch.setenv("DEMAND_PATHS__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("DEMAND_PATHS__REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["train"])

    get_settings.cache_clear()
    assert result.exit_code == 1
    assert "Error: " in result.output
    assert "features.parquet is missing. Run `dvc repro`" in result.output
    assert "Traceback" not in result.output
    assert list(tmp_path.iterdir()) == []


def test_backtest_command_does_not_load_mlflow() -> None:
    # In a fresh interpreter, because other tests of this run have loaded MLflow already.
    code = (
        "import sys, demand.models.pipeline, demand.models.backtest, demand.models.gbm; "
        "sys.exit('mlflow' in sys.modules)"
    )

    done = subprocess.run([sys.executable, "-c", code], check=False)

    assert done.returncode == 0
