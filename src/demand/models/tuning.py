"""Hyper-parameter tuning: search for better LightGBM parameters, show them, change nothing.

Layout of the report (``settings.paths.reports_dir``)::

    reports/
      tuning_report.json    the search space, every trial with its scores, the best
                            parameters, and what stopped the search

**What tuning is.** ``model.params`` in the config says how the trees are built: how
large a tree may be, how much each tree may correct, how many rows a tree sees. Nobody
knows the best values in advance; they are found by trying. One try, a *trial*, is one
set of values scored with the backtest. `Optuna <https://optuna.org>`_ chooses the
values: the first ten trials are drawn at random, and after that it draws more often
near the values that scored well (its *TPE* sampler).

**Compute, show, freeze by hand.** ``python -m demand tune`` works like
``select-zones``: it calculates, writes a report, and prints a block to paste under
``model:`` in the config. It does not change the config and it does not train the final
model. ``train`` keeps reading ``model.params`` and nothing else. Three reasons: the
config stays the one place that says what the model is; a person reads the evidence
before the model changes; and ``dvc repro`` stays quick and gives the same result every
time, which a search with a time limit cannot promise. So ``tune`` is not a DVC stage.

**The search space: five parameters.** Thirty trials cannot explore many dimensions,
so only the parameters that decide how closely the trees follow the training rows are
searched:

===================== =========== ====== ==============================================
parameter             range       scale  what it does
===================== =========== ====== ==============================================
``learning_rate``     0.01 to 0.2 log    how much of its correction each tree applies
``num_leaves``        8 to 128    log    how many groups (leaves) a tree may split the
                                         rows into
``min_child_samples`` 5 to 200    log    the fewest training rows a leaf may hold
``subsample``         0.5 to 1.0  linear the share of rows each tree is built from
``colsample_bytree``  0.4 to 1.0  linear the share of inputs each tree may use
===================== =========== ====== ==============================================

*Log scale* means that every doubling is equally likely to be tried: between 0.01 and
0.02 as often as between 0.1 and 0.2. It is used where the ratio matters and not the
difference (8 leaves against 16 is a large change, 120 against 128 none). The two
shares are steps of 0.05 on a plain scale. The values in the config lie inside every
range, with room on both sides.

What is *not* searched, and why:

* ``objective`` stays ``poisson`` and the zone stays a category. Both were decided with
  measurements in ``gbm.py``; searching them again here would let the search undo a
  decision without anyone reading the numbers.
* The seed. A seed is not a property of a good model. A search over seeds only finds
  the seed that is lucky on the four test weeks.
* ``n_estimators``, the number of trees. It does the same job as the learning rate from
  the other side: half the rate with twice the trees gives nearly the same model. A
  search over both spends its trials on pairs that are equally good. So the number of
  trees stays what the config says, the learning rate is searched, and a trial takes
  about as long as a backtest does today.
* ``subsample_freq`` stays in the config; without it ``subsample`` does nothing, so the
  search refuses a config that does not set it.
* The regularisation terms (``reg_alpha``, ``reg_lambda``), ``max_depth``, ``max_bin``
  and the settings for categories. They overlap with the five above, and every extra
  dimension makes thirty trials thinner.

Every decimal value is rounded to three significant digits *before* it is scored, so
the block that is printed holds exactly the values that were measured.

**What a trial optimises.** The MAE of the backtest over all test rows of all folds,
the headline number of the project (``backtest.py``), with the seed of the project. A
trial is simply ``backtest`` with other parameters, so the rules of the backtest hold
in every trial: the training rows of a fold end one horizon before its test week, and
nothing after ``data.train_end_date`` is read.

* *No early stopping.* LightGBM can stop adding trees when the score on a second set of
  rows stops improving. Inside a fold the only such rows are the test week, and then
  the test week would choose the number of trees and score the result: an exam marked
  with its own answers. A piece of the training rows could be set aside for it
  instead, but that is one more moving part for a parameter the search does not need.
* *No pruning.* Optuna can break off a trial that looks bad after the first folds. A
  trial here takes five to ten seconds and the whole search about four minutes;
  pruning would save perhaps one of them, and a pruned trial has no score that can be
  compared with the others. Every trial is finished.
* *One number, but four are shown.* A set of parameters can win the average by being
  much better in one week and a little worse in three. The search does not try to
  prevent that; the report shows it. Every trial is stored with its MAE per fold, and
  the best trial is compared with the current parameters fold by fold.

**Trial 0 is the config.** Before Optuna suggests anything, the parameters that are in
the config now are scored. The best trial is the one with the lowest MAE, and on a tie
the earlier one, so the result can never be worse than what the project already has,
and if nothing beats trial 0 the report says so. Optuna is not told about trial 0: the
config may hold values outside the ranges above.

**Is the gain real? The seed check.** Training draws rows and columns at random, so the
same parameters score a little differently with another seed (by about 0.2 MAE with the
parameters of the config). A gain smaller than that is luck. When a trial beats trial 0,
both sets of parameters are therefore scored again with the two seeds after
``project.seed``, and the report holds the mean and the spread over the three seeds and
the mean per fold. The rule of this project for accepting a change: *the gain is clearly
larger than the spread between seeds, and it holds in every fold.* The search does not
apply the rule; it shows the numbers, and a person decides.

**The scores of the best trial are too good, by an unknown amount.** This is the most
important thing to understand about tuning. The four test weeks were not seen by the
*model* of a fold, but they were seen by the *search*: it tried thirty sets of
parameters and kept the one that did best on exactly these weeks. Some of that lead is
skill and some is luck that will not repeat. The same weeks had already chosen the
handling of the zone and the objective. So ``best_mae`` is the score of the winner of a
contest, not an estimate of how the winner will do on new weeks.

How much of the gain is luck was measured once, by hand, without touching May or
June. The config of that day had a learning rate of 0.05, 63 leaves and at least 20 rows
per leaf, and every number below is the mean of the seeds 42 to 44:

* On the four April weeks the search lowered the MAE from 25.04 to 24.14, a gain of
  0.90. The seed alone moves both scores by 0.19, and the gain is there in each of the
  four weeks (0.66 to 1.28). That is the number the search reports about itself.
* A second search was run on the four weeks *before* them (6 March to 2 April) and
  never saw April. Its parameters scored 24.40 on the April weeks: a gain of 0.64, again
  in each of the four weeks (0.40 to 0.89). About 70 percent of the gain was still
  there on weeks that search had not seen. It is a strict test: the earlier weeks have
  four weeks less history to learn from, and the clocks change in the first of them.
* A third search saw only the first two April weeks. On the last two its parameters
  gained 1.41 and 0.87.
* The other way round, the parameters found on April lowered the MAE on the March
  weeks from 27.19 to 25.94.

The three searches agreed on the direction (larger trees, 107 to 125 leaves, with at
least 61 to 175 rows per leaf) but not on the exact values, and their April scores lie
between 23.94 and 24.40: the search finds a good region, not a best point. The agreement
is weaker evidence than it looks: all three used the same sampler seed, so their first
proposals were the same parameter sets, and the March winner is the thirteenth proposal
of the April search too. The out-of-sample test is still fair, because the March scores
alone chose the March winner. So the gain was real that day, several times the spread
between seeds, and its honest size is "about 0.6 to 0.9", not the 0.90 of the first
line. This check is not part of ``tune``: it doubles the run time and needs a second set
of weeks with enough history before them. The seed check is, because it is cheap and no
gain should be accepted without it.

The clean exam is the replay of May and June in a later phase: weeks that no decision
in this project has looked at. Nothing in this module reads them.

**What makes two searches identical.** The sampler is seeded with ``project.seed``,
trials run one after another, and a fit is repeatable (``gbm.py``). So when the search
ends because ``model.tuning_trials`` trials are done, a second run on the same machine
gives the same trials, the same scores and the same report, byte for byte. When it ends
because ``model.tuning_timeout_seconds`` have passed, the trials that were finished are
the same but *how many* were finished depends on the speed of the machine, and with it
possibly the best trial. The report says which of the two stopped the search
(``stopped_by``). For that reason it holds no durations and no time stamp. The time
limit is checked before a trial starts: a trial that has begun is finished, trial 0 is
always scored, and the seed check at the end is not counted.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Literal

import optuna
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.silver import report_to_json, write_text
from demand.errors import PipelineError
from demand.logs import get_logger
from demand.models.backtest import ForecasterScores, backtest, load_training_features
from demand.models.gbm import LIGHTGBM, LightGbmForecaster
from demand.schemas import feature_columns

log = get_logger(__name__)

TUNING_REPORT_NAME = "tuning_report.json"
SEED_CHECK_SEEDS = 3
"""How many seeds the current and the best parameters are compared with."""
_SIGNIFICANT_DIGITS = 3
_SCORE_DECIMALS = 6

CAUTION = (
    "The test weeks chose these parameters, so the scores of the best trial are too "
    "good by an unknown amount. They are not an estimate for new weeks; the replay of "
    "the months after data.train_end_date is."
)

Clock = Callable[[], float]
ParamValue = int | float


class SearchRange(BaseModel):
    """The values one parameter may take in the search. Both ends are included."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    low: float
    high: float
    log: bool = False
    """True: every doubling is equally likely. False: every value is."""
    whole: bool = False
    """True for a parameter that is a whole number."""
    step: float | None = None
    """The distance between two values that can be tried; ``None`` for any value."""


SEARCH_SPACE: dict[str, SearchRange] = {
    "learning_rate": SearchRange(low=0.01, high=0.2, log=True),
    "num_leaves": SearchRange(low=8, high=128, log=True, whole=True),
    "min_child_samples": SearchRange(low=5, high=200, log=True, whole=True),
    "subsample": SearchRange(low=0.5, high=1.0, step=0.05),
    "colsample_bytree": SearchRange(low=0.4, high=1.0, step=0.05),
}
"""The parameters that are searched. See the module docstring for the reasons."""

ALIASES: dict[str, tuple[str, ...]] = {
    "learning_rate": ("eta", "shrinkage_rate"),
    "num_leaves": ("num_leaf", "max_leaves", "max_leaf", "max_leaf_nodes"),
    "min_child_samples": ("min_data_in_leaf", "min_data_per_leaf", "min_data"),
    "subsample": ("bagging_fraction", "sub_row", "bagging"),
    "colsample_bytree": ("feature_fraction", "sub_feature"),
}
"""Other names LightGBM accepts for the searched parameters.

A config that uses one of them would hand LightGBM the same parameter twice in every
trial, once under each name, and LightGBM would pick one without saying which.
"""

SUBSAMPLE_FREQ = "subsample_freq"


class TrialResult(BaseModel):
    """One set of parameters and its backtest scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int
    """0 is the config as it is; 1 and up were suggested by the search."""
    params: dict[str, ParamValue]
    """The searched parameters only. The others are ``fixed_params`` of the report."""
    mae: float
    """Over all test rows of all folds. This is what the search minimises."""
    rmse: float
    smape: float
    mae_per_fold: dict[int, float]


class SeedScores(BaseModel):
    """The backtest MAE of one set of parameters with several seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mae_by_seed: dict[int, float]
    mae_mean: float
    mae_range: float
    """The highest minus the lowest: how far the seed alone moves the score."""
    mae_per_fold_mean: dict[int, float]


class SeedCheck(BaseModel):
    """The current and the best parameters, each scored with the same seeds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    current: SeedScores
    best: SeedScores
    mae_gain: float
    """Mean MAE of the current parameters minus that of the best. Positive: better."""
    gain_per_fold: dict[int, float]
    folds_better: int
    """In how many folds the best parameters have the lower mean MAE."""


class TuningReport(BaseModel):
    """The result of a search; saved as JSON next to the other reports."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    train_end_date: date
    horizon_hours: int
    folds: int
    test_rows: int
    seed: int
    search_space: dict[str, SearchRange]
    fixed_params: dict[str, Any]
    """The parameters of the config that are not searched; the same in every trial."""
    trial_limit: int
    """``model.tuning_trials``: suggested trials, not counting trial 0."""
    timeout_seconds: int
    stopped_by: Literal["trials", "timeout"]
    """``trials``: the limit was reached. ``timeout``: the time was up first."""
    suggested_trials: int
    """Trials that were suggested and scored. Smaller than the limit after a timeout."""
    current_mae: float
    """The MAE of trial 0, the parameters of the config."""
    best_trial: int
    """0 means that no suggested trial beat the config."""
    best_mae: float
    mae_gain: float
    """``current_mae`` minus ``best_mae``, with the seed of the project only."""
    best_params: dict[str, Any]
    """Complete ``model.params`` of the best trial, ready for the config."""
    seed_check: SeedCheck | None
    """``None`` when the best trial is trial 0: there is nothing to compare."""
    caution: str
    trials: list[TrialResult]


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def tuning_report_path(reports_dir: Path) -> Path:
    """Where the tuning report is stored."""
    return reports_dir / TUNING_REPORT_NAME


def check_searchable(params: Mapping[str, Any]) -> None:
    """Refuse ``model.params`` that would make the search measure something else.

    Raises:
        PipelineError: If a searched parameter is set under another of its LightGBM
            names, or ``subsample_freq`` is not positive (then no rows are sampled and
            ``subsample`` has no effect).
    """
    for name, aliases in ALIASES.items():
        used = [alias for alias in aliases if alias in params]
        if used:
            raise PipelineError(
                f"model.params sets {', '.join(used)}, another name for {name}, which the "
                f"search sets itself. Rename {', '.join(used)} to {name} in the config."
            )
    if params.get(SUBSAMPLE_FREQ, 0) <= 0:
        raise PipelineError(
            f"model.params has no positive {SUBSAMPLE_FREQ}, so LightGBM uses every row "
            f"for every tree and the searched parameter subsample would do nothing. Set "
            f"{SUBSAMPLE_FREQ}: 1 under model.params in the config."
        )


def rounded(value: float) -> float:
    """``value`` with three significant digits: 0.037281 becomes 0.0373."""
    return float(f"{value:.{_SIGNIFICANT_DIGITS}g}")


def suggest(trial: optuna.Trial) -> dict[str, ParamValue]:
    """Let Optuna choose one value for every parameter of ``SEARCH_SPACE``.

    Decimal values are rounded here, before they are scored (see the module docstring).
    """
    params: dict[str, ParamValue] = {}
    for name, values in SEARCH_SPACE.items():
        if values.whole:
            params[name] = trial.suggest_int(
                name, round(values.low), round(values.high), log=values.log
            )
        else:
            suggested = trial.suggest_float(
                name, values.low, values.high, log=values.log, step=values.step
            )
            params[name] = rounded(suggested)
    return params


def score_params(
    table: pd.DataFrame, settings: Settings, params: Mapping[str, Any], seed: int
) -> ForecasterScores:
    """The backtest scores of a LightGBM model with ``params`` and ``seed``.

    This is the backtest of ``backtest.py``, unchanged, for one forecaster.

    Raises:
        PipelineError: If ``params`` sets a seed (see ``gbm.py``).
        TooLittleHistoryError: If ``table`` is too short for the configured folds.
    """
    forecaster = LightGbmForecaster(feature_columns(settings), params, seed)
    return backtest(table, {LIGHTGBM: forecaster}, settings).forecasters[LIGHTGBM]


def _trial_result(number: int, params: Mapping[str, Any], scores: ForecasterScores) -> TrialResult:
    return TrialResult(
        number=number,
        params={name: params[name] for name in SEARCH_SPACE if name in params},
        mae=scores.overall.mae,
        rmse=scores.overall.rmse,
        smape=scores.overall.smape,
        mae_per_fold={fold: score.mae for fold, score in scores.per_fold.items()},
    )


def seed_scores(scores_by_seed: Mapping[int, ForecasterScores]) -> SeedScores:
    """The mean and the spread of the MAE over several seeds, overall and per fold."""
    maes = {seed: scores.overall.mae for seed, scores in scores_by_seed.items()}
    folds = next(iter(scores_by_seed.values())).per_fold
    return SeedScores(
        mae_by_seed=maes,
        mae_mean=_mean(list(maes.values())),
        mae_range=round(max(maes.values()) - min(maes.values()), _SCORE_DECIMALS),
        mae_per_fold_mean={
            fold: _mean([scores.per_fold[fold].mae for scores in scores_by_seed.values()])
            for fold in folds
        },
    )


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), _SCORE_DECIMALS)


def compare_seeds(current: SeedScores, best: SeedScores) -> SeedCheck:
    """Put the scores of the current and of the best parameters side by side."""
    gain_per_fold = {
        fold: round(mae - best.mae_per_fold_mean[fold], _SCORE_DECIMALS)
        for fold, mae in current.mae_per_fold_mean.items()
    }
    return SeedCheck(
        current=current,
        best=best,
        mae_gain=round(current.mae_mean - best.mae_mean, _SCORE_DECIMALS),
        gain_per_fold=gain_per_fold,
        folds_better=sum(gain > 0 for gain in gain_per_fold.values()),
    )


def search(
    table: pd.DataFrame, settings: Settings, *, clock: Clock = time.monotonic
) -> TuningReport:
    """Score the config, then let Optuna suggest and score better parameters.

    Args:
        table: Rows of the gold feature table, none of them after the training period.
        settings: Project settings (``model``: the parameters, the trial limit and the
            time limit; ``project.seed``; and everything the backtest reads).
        clock: Returns the seconds that have passed; replaced in tests.

    Raises:
        PipelineError: If ``model.params`` cannot be searched (``check_searchable``) or
            sets a seed.
        TooLittleHistoryError: If ``table`` is too short for the configured folds.
        ValueError: If ``table`` has a row after the training period.
    """
    config = settings.model
    seed = settings.project.seed
    check_searchable(config.params)
    started = clock()

    # Trial 0: the config as it is. Scored first, and never left out.
    all_scores = [score_params(table, settings, config.params, seed)]
    all_params: list[dict[str, Any]] = [dict(config.params)]
    _log_trial(0, all_params[0], all_scores[0])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    stopped_by: Literal["trials", "timeout"] = "trials"
    while len(all_scores) <= config.tuning_trials:
        if clock() - started >= config.tuning_timeout_seconds:
            stopped_by = "timeout"
            break
        trial = study.ask()
        params = {**config.params, **suggest(trial)}
        scores = score_params(table, settings, params, seed)
        study.tell(trial, scores.overall.mae)
        _log_trial(len(all_scores), params, scores)
        all_scores.append(scores)
        all_params.append(params)

    trials = [
        _trial_result(number, params, scores)
        for number, (params, scores) in enumerate(zip(all_params, all_scores, strict=True))
    ]
    # The lowest MAE wins and, on a tie, the earlier trial: trial 0 is only replaced by
    # a trial that is really better.
    best = min(trials, key=lambda trial: (trial.mae, trial.number))
    seed_check = None
    if best.number != 0:

        def with_more_seeds(number: int) -> SeedScores:
            # The seed of the project was scored in the search already.
            scores = {seed: all_scores[number]}
            for other in range(seed + 1, seed + SEED_CHECK_SEEDS):
                scores[other] = score_params(table, settings, all_params[number], other)
            return seed_scores(scores)

        seed_check = compare_seeds(current=with_more_seeds(0), best=with_more_seeds(best.number))
    return TuningReport(
        train_end_date=settings.data.train_end_date,
        horizon_hours=settings.features.horizon_hours,
        folds=settings.backtest.n_folds,
        test_rows=all_scores[0].overall.rows,
        seed=seed,
        search_space=SEARCH_SPACE,
        fixed_params={
            name: value for name, value in config.params.items() if name not in SEARCH_SPACE
        },
        trial_limit=config.tuning_trials,
        timeout_seconds=config.tuning_timeout_seconds,
        stopped_by=stopped_by,
        suggested_trials=len(trials) - 1,
        current_mae=trials[0].mae,
        best_trial=best.number,
        best_mae=best.mae,
        mae_gain=round(trials[0].mae - best.mae, _SCORE_DECIMALS),
        best_params=all_params[best.number],
        seed_check=seed_check,
        caution=CAUTION,
        trials=trials,
    )


def _log_trial(number: int, params: Mapping[str, Any], scores: ForecasterScores) -> None:
    log.info(
        "trial scored",
        extra={
            "trial": number,
            "mae": scores.overall.mae,
            "mae_per_fold": {str(fold): score.mae for fold, score in scores.per_fold.items()},
            **{name: params[name] for name in SEARCH_SPACE if name in params},
        },
    )


def tuning_config_text(report: TuningReport) -> str:
    """The result of a search as lines to paste under ``model:`` in the config file.

    The comment lines say what the numbers are and what they are not; the ``params:``
    block replaces the one in the config. When no trial beat the config, the block is
    the one the config already has. For example::

        # Best of 30 trials: backtest MAE 24.11, against 24.93 for the config (seed 42).
        # Mean of 3 seeds: 24.14 against 25.04, a gain of 0.90, better in 4 of 4 folds.
        # The seed alone moves the score by 0.19 (best) and 0.19 (config).
        # Paste only if the gain is clearly larger than that, in every fold.
        # The test weeks chose these values: the score is too good by an unknown amount.
        params:
          objective: poisson
          learning_rate: 0.104
          ...
    """
    check = report.seed_check
    if check is None:
        lines = [
            f"None of {report.suggested_trials} trials beat the parameters of the config "
            f"(backtest MAE {report.current_mae:.2f}). Keep them:"
        ]
    else:
        lines = [
            f"Best of {report.suggested_trials} trials: backtest MAE {report.best_mae:.2f}, "
            f"against {report.current_mae:.2f} for the config (seed {report.seed}).",
            f"Mean of {len(check.best.mae_by_seed)} seeds: {check.best.mae_mean:.2f} against "
            f"{check.current.mae_mean:.2f}, a gain of {check.mae_gain:.2f}, better in "
            f"{check.folds_better} of {len(check.gain_per_fold)} folds.",
            f"The seed alone moves the score by {check.best.mae_range:.2f} (best) and "
            f"{check.current.mae_range:.2f} (config).",
            "Paste only if the gain is clearly larger than that, in every fold.",
            "The test weeks chose these values: the score is too good by an unknown amount.",
        ]
    block = yaml.safe_dump({"params": report.best_params}, sort_keys=False, indent=2)
    indented = [f"  {line}" for line in block.splitlines()]
    return "\n".join([*(f"  # {line}" for line in lines), *indented])


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def tune_from_gold(settings: Settings) -> TuningReport:
    """Search on the gold feature table and write the report.

    Nothing but the report is written: not the config, not a model, nothing in MLflow.

    Raises:
        MissingGoldFileError: If the gold feature table is missing or out of date.
        TooLittleHistoryError: If the table is too short for the configured folds.
        PipelineError: If ``model.params`` cannot be searched or sets a seed.
    """
    started = time.monotonic()
    report = search(load_training_features(settings), settings)
    report_path = tuning_report_path(settings.paths.reports_dir)
    write_text(report_to_json(report), report_path)
    log.info(
        "tuning finished",
        extra={
            "report": str(report_path),
            "stopped_by": report.stopped_by,
            "suggested_trials": report.suggested_trials,
            "current_mae": report.current_mae,
            "best_trial": report.best_trial,
            "best_mae": report.best_mae,
            "seconds": round(time.monotonic() - started, 2),
        },
    )
    return report
