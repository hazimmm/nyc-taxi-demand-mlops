from __future__ import annotations

from datetime import date

import pytest
from typer.testing import CliRunner

from demand.cli import app
from demand.config import CONFIG_FILE_ENV, Settings, get_settings
from demand.data.gold import gold_features_path
from demand.data.silver import write_parquet
from demand.errors import PipelineError
from demand.models.backtest import BacktestReport, backtest_report_path
from demand.models.baseline import DAILY_NAIVE, WEEKLY_NAIVE
from demand.models.pipeline import run_backtest, run_training
from tests.conftest import CONFIG_FILE, with_features
from tests.unit.models.conftest import feature_table, with_backtest


@pytest.fixture
def small(settings: Settings) -> Settings:
    """Two folds of two days at the end of April 2025, with a gold feature table on disk.

    The horizon and the lags are written here, not taken from the config file: 48 hours
    and the lags 48 and 168, so the weekly baseline is the only one.
    """
    small = with_backtest(settings, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)
    small = with_features(small, horizon_hours=48, lags=[48, 168])
    _write_gold(small)
    return small


def _write_gold(settings: Settings) -> None:
    table = feature_table(settings, "2025-04-24 00:00", hours=9 * 24)
    write_parquet(table, gold_features_path(settings.paths.gold))


def _report(settings: Settings) -> BacktestReport:
    stored = backtest_report_path(settings.paths.reports_dir).read_bytes()
    return BacktestReport.model_validate_json(stored)


def test_backtest_stage_scores_the_weekly_baseline_and_writes_the_report(small: Settings) -> None:
    run_backtest(small)

    report = _report(small)
    assert report.horizon_hours == 48
    assert list(report.forecasters) == [WEEKLY_NAIVE]
    assert report.forecasters[WEEKLY_NAIVE].overall.mae == 168.0
    assert [fold.number for fold in report.folds] == [1, 2]


def test_backtest_stage_scores_both_baselines_when_the_horizon_allows_the_daily_one(
    small: Settings,
) -> None:
    day_ahead = with_features(small, horizon_hours=24, lags=[24, 168])
    _write_gold(day_ahead)

    run_backtest(day_ahead)

    report = _report(day_ahead)
    assert report.horizon_hours == 24
    assert list(report.forecasters) == [WEEKLY_NAIVE, DAILY_NAIVE]
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


def test_training_stage_is_a_placeholder(settings: Settings) -> None:
    with pytest.raises(NotImplementedError, match="run_training"):
        run_training(settings)


def test_train_command_says_not_implemented_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(CONFIG_FILE))
    get_settings.cache_clear()

    result = CliRunner().invoke(app, ["train"])

    get_settings.cache_clear()
    assert result.exit_code == 1
    assert "Not implemented yet: demand.models.pipeline.run_training" in result.output
    assert "Traceback" not in result.output
