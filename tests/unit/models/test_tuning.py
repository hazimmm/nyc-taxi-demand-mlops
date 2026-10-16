"""The search for better parameters, on tiny tables with tiny models.

Two kinds of test. Some run the real search: two folds of two days, 20 trees and two or
three trials. The others replace the backtest by ``Scripted``, which hands out scores a
test wrote down, so that what the search does with the scores can be checked by hand.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
import pytest
import yaml

from demand.config import Settings
from demand.data.gold import gold_features_path
from demand.data.silver import write_parquet
from demand.errors import PipelineError
from demand.models import tuning
from demand.models.backtest import ForecasterScores, Score, backtest_report_path
from demand.models.gbm import LIGHTGBM, SEED_PARAMS
from demand.models.pipeline import run_backtest, run_tuning
from demand.models.tuning import (
    ALIASES,
    CAUTION,
    SEARCH_SPACE,
    SeedCheck,
    SeedScores,
    TuningReport,
    check_searchable,
    compare_seeds,
    rounded,
    score_params,
    search,
    seed_scores,
    suggest,
    tune_from_gold,
    tuning_config_text,
    tuning_report_path,
)
from tests.conftest import CONFIG_FILE, with_features
from tests.unit.models.conftest import (
    FEW_TREES,
    feature_table,
    training_rows,
    with_backtest,
    with_model_params,
)

SEARCHED = ["learning_rate", "num_leaves", "min_child_samples", "subsample", "colsample_bytree"]
# What the search must never choose: the seed, and the two decisions of gbm.py.
FORBIDDEN = {*SEED_PARAMS, "objective", "n_estimators", "categorical_feature"}


def with_tuning(settings: Settings, *, trials: int, timeout_seconds: int = 900) -> Settings:
    """The same settings with other limits for the search."""
    limits = {"tuning_trials": trials, "tuning_timeout_seconds": timeout_seconds}
    return settings.model_copy(update={"model": settings.model.model_copy(update=limits)})


def with_seed(settings: Settings, seed: int) -> Settings:
    return settings.model_copy(
        update={"project": settings.project.model_copy(update={"seed": seed})}
    )


@pytest.fixture
def small(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Two folds of two days at the end of April 2025, 20 trees, three trials, and a gold
    feature table on disk. The working directory is under ``tmp_path``.
    """
    small = with_backtest(settings, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)
    small = with_features(small, horizon_hours=48, lags=[48, 168])
    small = with_tuning(with_model_params(small, FEW_TREES), trials=3)
    write_parquet(_gold_table(small), gold_features_path(small.paths.gold))
    monkeypatch.chdir(tmp_path)
    return small


@pytest.fixture
def beatable(small: Settings) -> Settings:
    """``small`` with parameters a search is sure to beat.

    No leaf may hold fewer than 150 rows and the first fold has 75 training rows, so no
    tree can split and the forecast is one number for every row. Any suggestion that
    allows a split is better.
    """
    return with_model_params(small, FEW_TREES | {"min_child_samples": 150})


def _gold_table(settings: Settings) -> pd.DataFrame:
    return feature_table(settings, "2025-04-24 00:00", hours=9 * 24)


def _table(settings: Settings) -> pd.DataFrame:
    return training_rows(_gold_table(settings), settings)


def _scores(fold_1: float, fold_2: float) -> ForecasterScores:
    """Scores of two folds of equal size, so the overall MAE is the mean of the two."""

    def score(mae: float) -> Score:
        return Score(rows=10, mae=mae, rmse=mae + 1, smape=mae + 2)

    overall = (fold_1 + fold_2) / 2
    return ForecasterScores(
        overall=score(overall).model_copy(update={"rows": 20}),
        overall_without_outliers=None,
        per_fold={1: score(fold_1), 2: score(fold_2)},
        per_zone={132: score(overall)},
    )


class Scripted:
    """Stands in for ``score_params``: the n-th call gets the n-th pair of fold MAEs."""

    def __init__(self, fold_maes: list[tuple[float, float]]) -> None:
        self.fold_maes = fold_maes
        self.calls: list[tuple[dict[str, Any], int]] = []

    def __call__(
        self, table: pd.DataFrame, settings: Settings, params: Mapping[str, Any], seed: int
    ) -> ForecasterScores:
        self.calls.append((dict(params), seed))
        return _scores(*self.fold_maes[len(self.calls) - 1])


def _scripted(monkeypatch: pytest.MonkeyPatch, fold_maes: list[tuple[float, float]]) -> Scripted:
    fake = Scripted(fold_maes)
    monkeypatch.setattr(tuning, "score_params", fake)
    return fake


class Clock:
    """A clock that is ``seconds_per_call`` further every time it is read."""

    def __init__(self, seconds_per_call: float) -> None:
        self.seconds_per_call = seconds_per_call
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.seconds_per_call
        return self.now


# --- the search space ------------------------------------------------------------------------


def test_search_space_is_the_five_parameters_and_holds_nothing_forbidden() -> None:
    assert list(SEARCH_SPACE) == SEARCHED
    assert not FORBIDDEN & set(SEARCH_SPACE)
    assert not FORBIDDEN & {alias for aliases in ALIASES.values() for alias in aliases}
    assert set(ALIASES) == set(SEARCH_SPACE)


def test_values_of_the_config_file_lie_inside_the_search_space(settings: Settings) -> None:
    for name, values in SEARCH_SPACE.items():
        assert values.low < settings.model.params[name] < values.high


def test_every_suggestion_stays_inside_its_bounds_and_has_the_right_type() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))

    suggestions = [suggest(study.ask()) for _ in range(200)]

    for params in suggestions:
        assert list(params) == SEARCHED
        for name, value in params.items():
            assert SEARCH_SPACE[name].low <= value <= SEARCH_SPACE[name].high
            assert isinstance(value, int if SEARCH_SPACE[name].whole else float)
            assert not isinstance(value, bool)
    # Both ends of a range can be reached, and the search does not sit in one corner.
    leaves = [params["num_leaves"] for params in suggestions]
    assert min(leaves) < 12 and max(leaves) > 100
    shares = {params["subsample"] for params in suggestions}
    assert shares == {0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}


def test_decimal_suggestions_have_three_significant_digits() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))

    rates = [suggest(study.ask())["learning_rate"] for _ in range(50)]

    assert all(rate == float(f"{rate:.3g}") for rate in rates)
    assert len(set(rates)) > 40


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.03728193, 0.0373), (0.1996, 0.2), (0.6000000000000001, 0.6), (0.85, 0.85), (0.01, 0.01)],
)
def test_rounded_keeps_three_significant_digits(value: float, expected: float) -> None:
    assert rounded(value) == expected


# --- trial 0 and the best trial ----------------------------------------------------------------


def test_trial_0_is_the_config_and_scores_what_the_backtest_scores(small: Settings) -> None:
    report = search(_table(small), small)
    run_backtest(small)

    stored = json.loads(backtest_report_path(small.paths.reports_dir).read_text(encoding="utf-8"))
    backtest_mae = stored["forecasters"][LIGHTGBM]["overall"]["mae"]
    first = report.trials[0]
    assert first.number == 0
    assert first.params == {name: FEW_TREES[name] for name in SEARCHED}
    assert first.mae == report.current_mae == backtest_mae
    assert list(first.mae_per_fold) == [1, 2]


def test_every_trial_keeps_the_parameters_that_are_not_searched(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _scripted(monkeypatch, [(10, 20), (9, 19), (8, 18), (7, 17), *[(1, 1)] * 4])

    report = search(_table(small), small)

    fixed = {"objective": "poisson", "n_estimators": 20, "subsample_freq": 1, "verbosity": -1}
    assert report.fixed_params == fixed
    assert fake.calls[0] == (FEW_TREES, 42)
    for params, seed in fake.calls[:4]:
        assert seed == 42
        assert list(params) == list(FEW_TREES)
        assert {name: params[name] for name in fixed} == fixed
        assert not set(SEED_PARAMS) & set(params)
    assert [trial.params for trial in report.trials[1:]] == [
        {name: params[name] for name in SEARCHED} for params, _ in fake.calls[1:4]
    ]


def test_best_trial_has_the_lowest_mae_and_its_complete_parameters(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _scripted(monkeypatch, [(10, 20), (8, 18), (9, 25), (9, 19), *[(1, 1)] * 4])

    report = search(_table(small), small)

    assert [trial.mae for trial in report.trials] == [15.0, 13.0, 17.0, 14.0]
    assert (report.best_trial, report.best_mae, report.current_mae) == (1, 13.0, 15.0)
    assert report.mae_gain == 2.0
    assert report.best_params == fake.calls[1][0]
    assert list(report.best_params) == list(FEW_TREES)


def test_best_is_the_config_when_no_trial_beats_it(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _scripted(monkeypatch, [(10, 20), (11, 21), (10, 21), (30, 30)])

    report = search(_table(small), small)

    assert (report.best_trial, report.best_mae, report.mae_gain) == (0, 15.0, 0.0)
    assert report.best_params == FEW_TREES
    # Nothing to compare, so no further backtests with other seeds.
    assert report.seed_check is None
    assert len(fake.calls) == 4


def test_a_trial_that_only_ties_with_the_config_does_not_replace_it(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted(monkeypatch, [(10, 20), (20, 10), (15, 15), (16, 16)])

    report = search(_table(small), small)

    assert [trial.mae for trial in report.trials] == [15.0, 15.0, 15.0, 16.0]
    assert report.best_trial == 0


@pytest.mark.parametrize("config", ["small", "beatable"])
def test_best_of_a_real_search_is_never_worse_than_the_config(
    config: str, request: pytest.FixtureRequest
) -> None:
    settings: Settings = request.getfixturevalue(config)

    report = search(_table(settings), settings)

    # On this tiny table the parameters of ``small`` win and those of ``beatable`` lose.
    assert (report.best_trial == 0) == (config == "small")
    assert report.best_mae <= report.current_mae
    assert report.best_mae == min(trial.mae for trial in report.trials)
    assert report.mae_gain == round(report.current_mae - report.best_mae, 6)
    assert len({json.dumps(trial.params) for trial in report.trials}) == 4


# --- the seed check ----------------------------------------------------------------------------


def test_seed_check_scores_the_config_and_the_best_trial_with_two_more_seeds(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _scripted(
        monkeypatch,
        [
            # The search, with seed 42: the config, then three suggestions.
            (10, 20),
            (8, 18),
            (9, 25),
            (9, 19),
            # The config with the seeds 43 and 44, then the best trial with both.
            (11, 21),
            (12, 19),
            (9, 19),
            (7, 17),
        ],
    )

    check = search(_table(small), small).seed_check

    assert [seed for _, seed in fake.calls[4:]] == [43, 44, 43, 44]
    assert [params for params, _ in fake.calls[4:]] == [FEW_TREES] * 2 + [fake.calls[1][0]] * 2
    assert check == SeedCheck(
        current=SeedScores(
            mae_by_seed={42: 15.0, 43: 16.0, 44: 15.5},
            mae_mean=15.5,
            mae_range=1.0,
            mae_per_fold_mean={1: 11.0, 2: 20.0},
        ),
        best=SeedScores(
            mae_by_seed={42: 13.0, 43: 14.0, 44: 12.0},
            mae_mean=13.0,
            mae_range=2.0,
            mae_per_fold_mean={1: 8.0, 2: 18.0},
        ),
        mae_gain=2.5,
        gain_per_fold={1: 3.0, 2: 2.0},
        folds_better=2,
    )


def test_seed_check_counts_only_the_folds_in_which_the_best_is_better() -> None:
    current = seed_scores({42: _scores(10, 20), 43: _scores(12, 20)})
    best = seed_scores({42: _scores(5, 20), 43: _scores(5, 22)})

    check = compare_seeds(current, best)

    # Fold 1: 11 against 5. Fold 2: 20 against 21, so the "best" is worse there.
    assert check.gain_per_fold == {1: 6.0, 2: -1.0}
    assert check.folds_better == 1
    assert check.mae_gain == 2.5


def test_seed_check_of_a_real_search_starts_from_the_scores_of_the_search(
    beatable: Settings,
) -> None:
    small = beatable
    report = search(_table(small), small)

    check = report.seed_check
    assert check is not None
    assert list(check.current.mae_by_seed) == list(check.best.mae_by_seed) == [42, 43, 44]
    assert check.current.mae_by_seed[42] == report.current_mae
    assert check.best.mae_by_seed[42] == report.best_mae
    other_seed = score_params(_table(small), small, report.best_params, 44)
    assert check.best.mae_by_seed[44] == other_seed.overall.mae


# --- reproducibility ---------------------------------------------------------------------------


def test_same_seed_gives_the_same_trials_and_the_same_report_file(small: Settings) -> None:
    first = tune_from_gold(small)
    first_bytes = tuning_report_path(small.paths.reports_dir).read_bytes()

    second = tune_from_gold(small)

    assert second == first
    assert tuning_report_path(small.paths.reports_dir).read_bytes() == first_bytes


def test_another_seed_gives_other_trials(small: Settings) -> None:
    first = search(_table(small), small)

    second = search(_table(small), with_seed(small, 7))

    assert second.seed == 7
    assert second.trials[0].params == first.trials[0].params
    assert [trial.params for trial in second.trials[1:]] != [
        trial.params for trial in first.trials[1:]
    ]


# --- the two limits ----------------------------------------------------------------------------


def test_search_stops_at_the_trial_limit_and_says_so(small: Settings) -> None:
    report = search(_table(small), with_tuning(small, trials=2))

    assert (report.stopped_by, report.suggested_trials, report.trial_limit) == ("trials", 2, 2)
    assert [trial.number for trial in report.trials] == [0, 1, 2]


def test_search_with_no_trials_scores_the_config_only(small: Settings) -> None:
    report = search(_table(small), with_tuning(small, trials=0))

    assert (report.stopped_by, report.suggested_trials, report.best_trial) == ("trials", 0, 0)
    assert [trial.number for trial in report.trials] == [0]


def test_search_stops_when_the_time_is_up_and_says_so(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _scripted(monkeypatch, [(10, 20)] * 20)
    # The clock is read once at the start (10 s) and once before every suggestion:
    # 10 and 20 seconds after the start are inside the limit of 25, 30 is not.
    limited = with_tuning(small, trials=10, timeout_seconds=25)

    report = search(_table(small), limited, clock=Clock(seconds_per_call=10))

    assert (report.stopped_by, report.suggested_trials) == ("timeout", 2)
    assert (report.trial_limit, report.timeout_seconds) == (10, 25)
    assert [trial.number for trial in report.trials] == [0, 1, 2]
    assert len(fake.calls) == 3


def test_trial_0_is_scored_even_when_the_time_is_up_at_once(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted(monkeypatch, [(10, 20)])

    report = search(_table(small), with_tuning(small, trials=5, timeout_seconds=1), clock=Clock(60))

    assert (report.stopped_by, report.suggested_trials, report.best_trial) == ("timeout", 0, 0)
    assert report.current_mae == report.best_mae == 15.0


def test_trials_before_a_timeout_are_the_first_trials_of_the_full_search(small: Settings) -> None:
    full = search(_table(small), small)

    cut = search(_table(small), with_tuning(small, trials=3, timeout_seconds=25), clock=Clock(10))

    assert cut.stopped_by == "timeout"
    assert cut.trials == full.trials[:3]


# --- the report and the printed block ------------------------------------------------------------


def test_report_holds_the_limits_the_space_every_trial_and_the_caution(small: Settings) -> None:
    report = tune_from_gold(small)

    stored = json.loads(tuning_report_path(small.paths.reports_dir).read_text(encoding="utf-8"))
    assert TuningReport.model_validate(stored) == report
    assert list(stored) == [
        "train_end_date",
        "horizon_hours",
        "folds",
        "test_rows",
        "seed",
        "search_space",
        "fixed_params",
        "trial_limit",
        "timeout_seconds",
        "stopped_by",
        "suggested_trials",
        "current_mae",
        "best_trial",
        "best_mae",
        "mae_gain",
        "best_params",
        "seed_check",
        "caution",
        "trials",
    ]
    assert stored["train_end_date"] == "2025-04-30"
    assert (stored["horizon_hours"], stored["folds"], stored["seed"]) == (48, 2, 42)
    assert stored["test_rows"] == 2 * 2 * 24 * 3
    assert stored["search_space"]["learning_rate"] == {
        "low": 0.01,
        "high": 0.2,
        "log": True,
        "whole": False,
        "step": None,
    }
    assert stored["caution"] == CAUTION
    assert "too good" in CAUTION
    assert len(stored["trials"]) == 4
    assert list(stored["trials"][1]) == ["number", "params", "mae", "rmse", "smape", "mae_per_fold"]
    assert stored["best_params"] == report.best_params


def test_tuning_writes_nothing_but_its_report(small: Settings, tmp_path: Path) -> None:
    gold_bytes = gold_features_path(small.paths.gold).read_bytes()

    tune_from_gold(small)

    assert [file.name for file in small.paths.reports_dir.iterdir()] == ["tuning_report.json"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["data", "reports"]
    assert gold_features_path(small.paths.gold).read_bytes() == gold_bytes


def _pasted(text: str) -> dict[str, Any]:
    """The parameters a config would hold after ``text`` was pasted under ``model:``."""
    params: dict[str, Any] = yaml.safe_load("model:\n" + text)["model"]["params"]
    return params


def test_printed_block_parses_back_to_the_best_parameters(
    beatable: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    small = beatable
    config_before = CONFIG_FILE.read_bytes()

    run_tuning(small)

    printed = capsys.readouterr().out
    stored = tuning_report_path(small.paths.reports_dir).read_bytes()
    report = TuningReport.model_validate_json(stored)
    assert report.best_trial != 0
    assert _pasted(printed) == report.best_params
    assert list(_pasted(printed)) == list(FEW_TREES)
    assert CONFIG_FILE.read_bytes() == config_before
    # Pasted into the config, the parameters score exactly what the report says.
    pasted = with_model_params(small, _pasted(printed))
    assert score_params(_table(small), pasted, pasted.model.params, 42).overall.mae == (
        report.best_mae
    )


def test_printed_block_shows_the_scores_the_seed_check_and_the_caution(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted(
        monkeypatch,
        [(10, 20), (8, 18), (9, 25), (9, 19), (11, 21), (12, 19), (9, 19), (7, 17)],
    )

    text = tuning_config_text(search(_table(small), small))

    lines = text.splitlines()
    comments = [line for line in lines if line.startswith("  # ")]
    assert lines[: len(comments)] == comments
    assert lines[len(comments)] == "  params:"
    assert all(line.startswith("    ") for line in lines[len(comments) + 1 :])
    assert comments == [
        "  # Best of 3 trials: backtest MAE 13.00, against 15.00 for the config (seed 42).",
        "  # Mean of 3 seeds: 13.00 against 15.50, a gain of 2.50, better in 2 of 2 folds.",
        "  # The seed alone moves the score by 2.00 (best) and 1.00 (config).",
        "  # Paste only if the gain is clearly larger than that, in every fold.",
        "  # The test weeks chose these values: the score is too good by an unknown amount.",
    ]


def test_printed_block_says_to_keep_the_config_when_nothing_beat_it(
    small: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scripted(monkeypatch, [(10, 20), (11, 21), (10, 21), (30, 30)])

    text = tuning_config_text(search(_table(small), small))

    assert text.splitlines()[0] == (
        "  # None of 3 trials beat the parameters of the config (backtest MAE 15.00). Keep them:"
    )
    assert "seeds" not in text
    assert _pasted(text) == FEW_TREES


# --- error paths -------------------------------------------------------------------------------


def test_tuning_stops_without_the_gold_layer(settings: Settings) -> None:
    with pytest.raises(PipelineError, match="dvc repro"):
        run_tuning(settings)

    assert not settings.paths.reports_dir.exists()


def test_tuning_stops_before_any_work_when_the_config_sets_a_seed(small: Settings) -> None:
    with_seed_param = with_model_params(small, FEW_TREES | {"seed": 7})

    with pytest.raises(PipelineError, match=r"Remove seed from model\.params"):
        run_tuning(with_seed_param)

    assert not small.paths.reports_dir.exists()


@pytest.mark.parametrize(
    ("alias", "name"),
    [
        ("eta", "learning_rate"),
        ("max_leaves", "num_leaves"),
        ("min_data_in_leaf", "min_child_samples"),
        ("bagging_fraction", "subsample"),
        ("feature_fraction", "colsample_bytree"),
    ],
)
def test_tuning_refuses_a_searched_parameter_under_another_name(
    small: Settings, monkeypatch: pytest.MonkeyPatch, alias: str, name: str
) -> None:
    fake = _scripted(monkeypatch, [])
    renamed = {key: value for key, value in FEW_TREES.items() if key != name} | {alias: 1}

    with pytest.raises(PipelineError, match=rf"Rename {alias} to {name} in the config"):
        search(_table(small), with_model_params(small, renamed))

    assert fake.calls == []


@pytest.mark.parametrize("frequency", [None, 0])
def test_tuning_refuses_a_config_in_which_subsample_has_no_effect(frequency: int | None) -> None:
    params = {key: value for key, value in FEW_TREES.items() if key != "subsample_freq"}
    if frequency is not None:
        params["subsample_freq"] = frequency

    with pytest.raises(PipelineError, match=r"Set subsample_freq: 1 under model\.params"):
        check_searchable(params)


def test_parameters_of_the_config_file_can_be_searched(settings: Settings) -> None:
    check_searchable(settings.model.params)


def test_tuning_refuses_a_row_after_the_training_period(small: Settings) -> None:
    with pytest.raises(ValueError, match=r"rows after data.train_end_date \(2025-04-30\)"):
        search(_gold_table(small), small)


# --- what is loaded ------------------------------------------------------------------------------


def test_tuning_loads_optuna_but_neither_mlflow_nor_the_plotting_libraries() -> None:
    # In a fresh interpreter, because other tests of this run have loaded them already.
    code = (
        "import sys, demand.models.tuning; "
        "loaded = [m for m in ('mlflow', 'shap', 'matplotlib') if m in sys.modules]; "
        "sys.exit(bool(loaded) or 'optuna' not in sys.modules)"
    )

    done = subprocess.run([sys.executable, "-c", code], check=False)

    assert done.returncode == 0
