from __future__ import annotations

import json
import logging
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from demand.config import Settings
from demand.data.gold import gold_features_path
from demand.data.silver import cleaning_report_path, write_parquet, write_text
from demand.errors import PipelineError
from demand.models import training
from demand.models.backtest import (
    BacktestReport,
    ForecasterScores,
    Score,
    backtest_report_path,
)
from demand.models.baseline import WEEKLY_NAIVE, Forecaster, baselines
from demand.models.gbm import LIGHTGBM, LightGbmForecaster
from demand.models.pipeline import forecasters
from demand.models.tracking import LoggedRun
from demand.models.training import (
    UNKNOWN,
    GitState,
    TrainingReport,
    data_version,
    final_training_rows,
    git_state,
    improvement_percent,
    run_metrics,
    run_params,
    train_from_gold,
    training_report_path,
)
from demand.schemas import HOUR_START, ZONE_ID, feature_columns
from tests.conftest import with_features
from tests.unit.models.conftest import (
    FEW_TREES,
    Lookup,
    feature_table,
    training_rows,
    with_backtest,
    with_mlflow_store,
    with_model_params,
)

TIMEZONE = "America/New_York"
# The gold table of these tests: three zones, 24 April to 2 May 2025. The training
# period ends on 30 April, so 7 days of it can be learned from: 7 * 24 * 3 = 504 rows.
FIRST_HOUR = "2025-04-24 00:00"
TABLE_HOURS = 9 * 24
TRAINING_ROWS = 7 * 24 * 3
GOLD_MD5 = "7bacf1bf80cf4612d40552e74ef11cc5.dir"


def _local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz=TIMEZONE)


@pytest.fixture
def small(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Two folds of two days, a quick model, a gold table on disk, and a working directory
    that is neither a git repository nor has a ``dvc.lock``.

    The MLflow database is under ``tmp_path``. Most tests replace the function that
    writes to it; no test may reach the database of the repository.
    """
    small = with_backtest(settings, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)
    small = with_features(small, horizon_hours=48, lags=[48, 168])
    small = with_mlflow_store(with_model_params(small, FEW_TREES), tmp_path)
    write_parquet(_gold_table(small), gold_features_path(small.paths.gold))
    monkeypatch.chdir(tmp_path)
    return small


def _gold_table(settings: Settings) -> pd.DataFrame:
    return feature_table(settings, FIRST_HOUR, TABLE_HOURS)


class FakeMlflow:
    """Stands in for ``tracking.log_training_run`` and remembers what it was given."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, settings: Settings, **values: Any) -> LoggedRun:
        self.calls.append(values)
        return LoggedRun(run_id="run-1", model_version=7)


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch) -> FakeMlflow:
    fake = FakeMlflow()
    monkeypatch.setattr(training, "log_training_run", fake)
    return fake


class PerfectSpy(Lookup):
    """Forecasts the pickups of the gold table exactly and remembers what it was fitted on."""

    def __init__(self, table: pd.DataFrame) -> None:
        super().__init__(table)
        self.fitted_on: list[pd.DataFrame] = []

    def fit(self, train: pd.DataFrame) -> None:
        self.fitted_on.append(train)


@pytest.fixture
def perfect(small: Settings) -> PerfectSpy:
    """A forecaster that is always right, to stand in for the LightGBM model."""
    return PerfectSpy(_gold_table(small))


def _with_model(settings: Settings, model: Forecaster) -> dict[str, Forecaster]:
    """The forecasters of a training run, with ``model`` in the place of LightGBM."""
    return {**baselines(settings), LIGHTGBM: model}


def _files(folder: Path) -> list[str]:
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*"))


# --- improvement over the baseline --------------------------------------------------------


@pytest.mark.parametrize(
    ("model_mae", "baseline_mae", "expected"),
    [
        (25.0, 100.0, 75.0),  # a quarter of the error is left
        (100.0, 100.0, 0.0),  # no better than copying last week
        (110.0, 100.0, -10.0),  # worse than the baseline
        (2.0, 3.0, 33.333),  # rounded to three decimals
    ],
)
def test_improvement_is_the_share_of_the_baseline_error_that_is_gone(
    model_mae: float, baseline_mae: float, expected: float
) -> None:
    assert improvement_percent(model_mae, baseline_mae) == expected


def test_improvement_over_a_perfect_baseline_cannot_be_calculated() -> None:
    with pytest.raises(PipelineError, match=r"MAE of 0 .*Check the feature table"):
        improvement_percent(5.0, 0.0)


# --- what is recorded: parameters and metrics ---------------------------------------------


def test_run_parameters_are_the_model_parameters_and_the_keys_that_define_the_run(
    small: Settings,
) -> None:
    assert run_params(small, train_rows=504) == {
        "model.objective": "poisson",
        "model.learning_rate": 0.05,
        "model.n_estimators": 20,
        "model.num_leaves": 63,
        "model.min_child_samples": 20,
        "model.subsample": 0.8,
        "model.subsample_freq": 1,
        "model.colsample_bytree": 0.8,
        "model.verbosity": -1,
        "model.seed": 42,
        "features.horizon_hours": 48,
        "features.lags": [48, 168],
        "features.rolling_windows": [24, 168],
        "features.outlier_zscore": 4.0,
        "feature_columns": feature_columns(small),
        "data.train_end_date": "2025-04-30",
        "data.zones": [132, 161, 237],
        "backtest.n_folds": 2,
        "backtest.fold_days": 2,
        "train_rows": 504,
    }


def _score(mae: float) -> Score:
    return Score(rows=10, mae=mae, rmse=mae + 1, smape=mae + 2)


def _scores(overall: float, fold_1: float, fold_2: float) -> ForecasterScores:
    return ForecasterScores(
        overall=_score(overall),
        overall_without_outliers=None,
        per_fold={1: _score(fold_1), 2: _score(fold_2)},
        per_zone={132: _score(overall)},
    )


def test_run_metrics_are_the_scores_of_model_and_baseline_and_the_improvement() -> None:
    report = BacktestReport(
        train_end_date=date(2025, 4, 30),
        horizon_hours=48,
        folds=[],
        test_rows=20,
        outlier_rows=0,
        forecasters={
            WEEKLY_NAIVE: _scores(overall=40.0, fold_1=50.0, fold_2=30.0),
            LIGHTGBM: _scores(overall=30.0, fold_1=25.0, fold_2=33.0),
        },
    )

    assert run_metrics(report) == {
        "lightgbm.mae": 30.0,
        "lightgbm.rmse": 31.0,
        "lightgbm.smape": 32.0,
        "lightgbm.fold_1.mae": 25.0,
        "lightgbm.fold_1.rmse": 26.0,
        "lightgbm.fold_1.smape": 27.0,
        "lightgbm.fold_2.mae": 33.0,
        "lightgbm.fold_2.rmse": 34.0,
        "lightgbm.fold_2.smape": 35.0,
        "weekly_naive.mae": 40.0,
        "weekly_naive.rmse": 41.0,
        "weekly_naive.smape": 42.0,
        "weekly_naive.fold_1.mae": 50.0,
        "weekly_naive.fold_1.rmse": 51.0,
        "weekly_naive.fold_1.smape": 52.0,
        "weekly_naive.fold_2.mae": 30.0,
        "weekly_naive.fold_2.rmse": 31.0,
        "weekly_naive.fold_2.smape": 32.0,
        "improvement.mae_percent": 25.0,  # 30 against 40
        "improvement.fold_1.mae_percent": 50.0,  # 25 against 50
        "improvement.fold_2.mae_percent": -10.0,  # 33 against 30: worse in this fold
    }


# --- the rows of the final fit -------------------------------------------------------------


def test_final_fit_uses_every_row_of_the_training_period(small: Settings) -> None:
    table = training_rows(_gold_table(small), small)

    rows = final_training_rows(table, small)

    assert len(rows) == TRAINING_ROWS
    assert rows[HOUR_START].min() == _local("2025-04-24 00:00")
    # No gap at the end: the last hour of the training period is learned from.
    assert rows[HOUR_START].max() == _local("2025-04-30 23:00")
    assert rows.equals(table)


def test_final_fit_refuses_a_row_after_the_training_period(small: Settings) -> None:
    gold = _gold_table(small)  # goes on to 2 May
    one_hour_too_many = gold[gold[HOUR_START] <= _local("2025-05-01 00:00")]

    with pytest.raises(ValueError, match=r"3 rows after data.train_end_date \(2025-04-30\)"):
        final_training_rows(one_hour_too_many, small)


def test_final_fit_refuses_an_empty_table(small: Settings) -> None:
    with pytest.raises(ValueError, match="without rows"):
        final_training_rows(_gold_table(small).iloc[:0], small)


# --- git ---------------------------------------------------------------------------------


def _git(folder: Path, *arguments: str) -> None:
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    signing = ["-c", "commit.gpgsign=false"]
    subprocess.run(
        ["git", *identity, *signing, *arguments], cwd=folder, check=True, capture_output=True
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    """A git repository with one commit: a tracked file and a rule that ignores ``*.db``."""
    folder = tmp_path / "repository"
    folder.mkdir()
    (folder / "code.py").write_text("x = 1\n", encoding="utf-8")
    (folder / ".gitignore").write_text("*.db\n", encoding="utf-8")
    _git(folder, "init", "--quiet")
    _git(folder, "add", ".")
    _git(folder, "commit", "--quiet", "-m", "first")
    return folder


def test_git_state_of_an_unchanged_checkout_is_its_commit_and_clean(repository: Path) -> None:
    state = git_state(repository)

    assert len(state.commit) == 40
    assert set(state.commit) <= set("0123456789abcdef")
    assert state.dirty is False


def test_a_changed_tracked_file_makes_the_checkout_dirty(repository: Path) -> None:
    (repository / "code.py").write_text("x = 2\n", encoding="utf-8")

    assert git_state(repository).dirty is True


def test_a_new_untracked_file_makes_the_checkout_dirty(repository: Path) -> None:
    (repository / "new_module.py").write_text("y = 1\n", encoding="utf-8")

    assert git_state(repository).dirty is True


def test_an_ignored_file_does_not_make_the_checkout_dirty(repository: Path) -> None:
    (repository / "mlflow.db").write_text("runs", encoding="utf-8")

    assert git_state(repository).dirty is False


def test_git_state_outside_a_repository_is_unknown(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        state = git_state(tmp_path)

    assert state == GitState(commit=UNKNOWN, dirty=None)
    assert "git state unknown" in caplog.text


# --- the data version ----------------------------------------------------------------------


def _write_lock(folder: Path, outs_path: str) -> Path:
    lock = folder / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\n"
        "stages:\n"
        "  build_silver:\n"
        "    cmd: python -m demand silver\n"
        "    outs:\n"
        "    - path: data/silver\n"
        "      md5: 1111.dir\n"
        "  build_gold:\n"
        "    cmd: python -m demand gold\n"
        "    deps:\n"
        "    - path: data/silver\n"
        "      md5: 1111.dir\n"
        "    outs:\n"
        f"    - path: {outs_path}\n"
        "      hash: md5\n"
        f"      md5: {GOLD_MD5}\n"
        "  validate_gold:\n"
        "    cmd: python -m demand validate gold\n",
        encoding="utf-8",
    )
    return lock


def test_data_version_is_the_checksum_dvc_recorded_for_the_gold_folder(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, "data/gold")

    assert data_version(lock, tmp_path / "data" / "gold") == GOLD_MD5


def test_data_version_is_unknown_without_a_lock_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        version = data_version(tmp_path / "dvc.lock", tmp_path / "data" / "gold")

    assert version == UNKNOWN
    assert "dvc repro" in caplog.text


def test_data_version_is_unknown_when_no_stage_writes_the_gold_folder(tmp_path: Path) -> None:
    lock = _write_lock(tmp_path, "data/somewhere_else")

    assert data_version(lock, tmp_path / "data" / "gold") == UNKNOWN


def test_data_version_of_an_empty_lock_file_is_unknown(tmp_path: Path) -> None:
    lock = tmp_path / "dvc.lock"
    lock.write_text("", encoding="utf-8")

    assert data_version(lock, tmp_path / "data" / "gold") == UNKNOWN


# --- a whole training run, with MLflow replaced --------------------------------------------


def test_training_scores_fits_on_all_training_rows_and_writes_both_reports(
    small: Settings, perfect: PerfectSpy, fake_mlflow: FakeMlflow
) -> None:
    report = train_from_gold(small, _with_model(small, perfect))

    # Two fits for the two folds, then the final one on every row up to 30 April 23:00:
    # the rows of the test weeks and the 47 hours before them are in it.
    assert [len(train) for train in perfect.fitted_on] == [3 * 25, 3 * 73, TRAINING_ROWS]
    final = perfect.fitted_on[-1]
    assert final[HOUR_START].min() == _local("2025-04-24 00:00")
    assert final[HOUR_START].max() == _local("2025-04-30 23:00")
    assert sorted(final[ZONE_ID].unique()) == [132, 161, 237]

    assert report == TrainingReport(
        run_id="run-1",
        registered_model="taxi-demand",
        model_version=7,
        data_version=UNKNOWN,
        git_commit=UNKNOWN,
        git_dirty=None,
        train_rows=TRAINING_ROWS,
        first_train_hour=_local("2025-04-24 00:00"),
        last_train_hour=_local("2025-04-30 23:00"),
        model=Score(rows=288, mae=0.0, rmse=0.0, smape=0.0),
        # Pickups rise by one per hour, so last week's value is 168 too low.
        baseline=Score(rows=288, mae=168.0, rmse=168.0, smape=report.baseline.smape),
        mae_improvement_percent=100.0,
    )
    stored = training_report_path(small.paths.reports_dir).read_bytes()
    assert TrainingReport.model_validate_json(stored) == report
    assert _files(small.paths.reports_dir) == ["backtest_report.json", "training_report.json"]


def test_training_hands_mlflow_the_model_the_scores_and_where_the_run_came_from(
    small: Settings, perfect: PerfectSpy, fake_mlflow: FakeMlflow
) -> None:
    train_from_gold(small, _with_model(small, perfect))

    (call,) = fake_mlflow.calls
    assert call["forecaster"] is perfect
    assert list(call["input_example"].columns) == feature_columns(small)
    assert len(call["input_example"]) == 5
    assert call["params"] == run_params(small, train_rows=TRAINING_ROWS)
    metrics = dict(call["metrics"])
    assert metrics.pop("final_fit_seconds") >= 0
    stored = backtest_report_path(small.paths.reports_dir).read_bytes()
    assert metrics == run_metrics(BacktestReport.model_validate_json(stored))
    assert metrics["lightgbm.mae"] == 0.0
    assert metrics["weekly_naive.mae"] == 168.0
    assert metrics["improvement.mae_percent"] == 100.0
    assert call["tags"] == {
        "forecaster": "lightgbm",
        "baseline": "weekly_naive",
        "git_commit": "unknown",
        "git_dirty": "unknown",
        "data_version": "unknown",
    }
    # There is no cleaning report in this folder, so only the backtest report is stored.
    assert call["artifacts"] == [backtest_report_path(small.paths.reports_dir)]


def test_training_records_the_data_version_and_the_cleaning_report_when_they_exist(
    small: Settings, tmp_path: Path, perfect: PerfectSpy, fake_mlflow: FakeMlflow
) -> None:
    # The working directory is tmp_path, and the gold folder is tmp_path/data/gold.
    _write_lock(tmp_path, "data/gold")
    cleaning_report = cleaning_report_path(small.paths.reports_dir)
    write_text("{}\n", cleaning_report)

    report = train_from_gold(small, _with_model(small, perfect))

    (call,) = fake_mlflow.calls
    assert report.data_version == GOLD_MD5
    assert call["tags"]["data_version"] == GOLD_MD5
    assert call["artifacts"] == [backtest_report_path(small.paths.reports_dir), cleaning_report]


def test_training_records_the_commit_and_that_files_were_changed(
    small: Settings,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    perfect: PerfectSpy,
    fake_mlflow: FakeMlflow,
) -> None:
    (repository / "code.py").write_text("x = 2\n", encoding="utf-8")
    monkeypatch.chdir(repository)

    report = train_from_gold(small, _with_model(small, perfect))

    (call,) = fake_mlflow.calls
    assert call["tags"]["git_commit"] == report.git_commit == git_state(repository).commit
    assert call["tags"]["git_dirty"] == "true"
    assert report.git_dirty is True


def test_training_writes_nothing_into_the_data_folder(
    small: Settings, perfect: PerfectSpy, fake_mlflow: FakeMlflow
) -> None:
    before = _files(small.paths.data_dir)
    gold_bytes = gold_features_path(small.paths.gold).read_bytes()

    train_from_gold(small, _with_model(small, perfect))

    assert _files(small.paths.data_dir) == before == ["gold", "gold/features.parquet"]
    assert gold_features_path(small.paths.gold).read_bytes() == gold_bytes


def test_training_with_the_real_model_stores_a_fitted_forecaster(
    small: Settings, fake_mlflow: FakeMlflow
) -> None:
    report = train_from_gold(small, forecasters(small))

    (call,) = fake_mlflow.calls
    forecaster = call["forecaster"]
    assert isinstance(forecaster, LightGbmForecaster)
    forecast = forecaster.predict(call["input_example"])
    assert forecast.shape == (5,)
    assert (forecast > 0).all()
    assert report.model.rows == report.baseline.rows == 288
    assert report.train_rows == TRAINING_ROWS


def test_training_twice_gives_the_same_scores_and_the_same_backtest_report(
    small: Settings, fake_mlflow: FakeMlflow
) -> None:
    first = train_from_gold(small, forecasters(small))
    first_bytes = backtest_report_path(small.paths.reports_dir).read_bytes()

    second = train_from_gold(small, forecasters(small))

    assert second == first
    assert backtest_report_path(small.paths.reports_dir).read_bytes() == first_bytes
    first_forecaster, second_forecaster = (call["forecaster"] for call in fake_mlflow.calls)
    example = fake_mlflow.calls[0]["input_example"]
    assert first_forecaster.predict(example).tolist() == second_forecaster.predict(example).tolist()


# --- error paths -----------------------------------------------------------------------------


def test_training_stops_without_the_gold_layer_and_records_nothing(
    small: Settings, fake_mlflow: FakeMlflow
) -> None:
    gold_features_path(small.paths.gold).unlink()

    with pytest.raises(PipelineError, match=r"features.parquet is missing. Run `dvc repro`"):
        train_from_gold(small, forecasters(small))

    assert fake_mlflow.calls == []
    assert not small.paths.reports_dir.exists()


def test_training_report_lists_its_fields_in_order_with_hours_in_local_time(
    small: Settings, perfect: PerfectSpy, fake_mlflow: FakeMlflow
) -> None:
    train_from_gold(small, _with_model(small, perfect))

    stored = json.loads(training_report_path(small.paths.reports_dir).read_text(encoding="utf-8"))
    assert list(stored) == [
        "run_id",
        "registered_model",
        "model_version",
        "data_version",
        "git_commit",
        "git_dirty",
        "train_rows",
        "first_train_hour",
        "last_train_hour",
        "model",
        "baseline",
        "mae_improvement_percent",
    ]
    assert stored["first_train_hour"] == "2025-04-24T00:00:00-04:00"
    assert stored["last_train_hour"] == "2025-04-30T23:00:00-04:00"
