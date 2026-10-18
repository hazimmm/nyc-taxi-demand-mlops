"""Retraining: a new model from newer data, served only if it beats the champion's forecasts.

When the champion's live accuracy falls, the recipe is trained again on the data up to
"now", and the new model replaces the champion only if it beats what the champion really
forecast in the last weeks.

``python -m demand retrain`` decides at the issue moment of the newest forecast file
(``clock.py``). That is the moment of the last forecast step, not the clock's next step:
after the closing step the clock stands at 1 July 00:00, but the newest forecast (for 30
June) was issued at 00:00 on 29 June. Monitoring is evaluated as of that moment first, and
its report overwrites ``reports/monitoring_report.json``; run ``python -m demand monitor``
afterwards to have the report at the clock again. ``run_day`` is the whole daily step:
forecast and score (``forecast_flow.run_step``), monitor as of the same moment
(``monitoring_flow.run_monitoring``), retrain if needed. The orchestrator calls it once a
day; the replay with retraining is that function in a loop. Layout::

    reports/retrain_report.json         the last evaluation: what was decided and why
    reports/retrain/2025-05-30/         the training reports of the retrain of that day
    reports/promotion_report.json       the last decision of the gate (``promotion.py``)

**When a retrain is considered.** At a step whose "now" is 00:00 on day X, a challenger
is trained only if all of these hold, checked in this order:

1. *A day is left to forecast.* A model trained after the last forecast would serve
   nothing (``end_of_data``).
2. *Monitoring as of X has an accuracy alert, about the champion* (``no_alert``). Only
   ``accuracy.alert`` is read. Input and prediction drift are warnings and never trigger
   anything (``monitoring/checks.py`` says why). An alert about a version that is no
   longer the champion is not a reason to replace the one that is.
3. *The cooldown is over* (``cooldown``): the last retrain was at least
   ``monitoring.retrain_cooldown_days`` calendar days before X. The last retrain is read
   from the registry, the tag ``retrain.now`` that every retrain writes onto the version
   it registers, promoted or not; there is no state file to disagree with it. Without
   the cooldown an alert that lasts five days would train five challengers, each judged
   on nearly the same weeks.
4. *The champion issued every day of the comparison weeks* (``waiting``; below). This
   also stands in for "the champion's start of service": a champion is never judged
   before it has served the four weeks it is judged on.

**What the challenger learns from: the cutoff.** The challenger is the same recipe
(``model.params``, the features, the frozen zone list: no tuning, no new zones) with
``data.train_end_date`` moved to the cutoff day, X - 1, in a copy of the settings
(``challenger_settings``). The rule is the one of ``models/training.py``: a model may
learn from no pickups that did not exist yet when it makes its first forecast. For the
retrain of 30 May, on a timeline (each time is the start of an hour)::

    Fri 30 May 00:00   now: the step issues the forecast for 31 May (with the champion),
                       scores 29 May, monitoring runs, then the challenger is trained
    Thu 29 May 23:00   the challenger's last training hour, complete at now
    Sat 31 May 00:00   the next step: the forecast for 1 June is issued, by the new
                       champion if the challenger was promoted
    Sun  1 Jun 23:00   its last hour, which reads pickups up to 30 May 23:00 (48 h)

Every training target ended by now, before the first forecast the challenger can issue
(00:00 on 31 May), so nothing it learned from lies after a moment it forecasts from.
The challenger could even have issued the forecast for 31 May at 00:00 on 30 May, as the
first champion issued 2 May at 00:00 on 1 May; it does not, because the step's forecast
is due at "now" and is not held back while a model trains. The gold rows are read up to
the cutoff only (``load_training_features`` with the challenger's settings); they are the
rows the replay forecast from, because the features of an hour do not depend on when they
are built. Two things stay as they were chosen on the first training period: the zone
list (phase 1) and the outlier statistics of gold. The outlier flag is not a model input;
in a backtest it only splits a diagnostic score.

**The comparison the gate sees.** A backtest of the same recipe on the champion's folds
would be a tie: the same code on the same rows gives the same model. The question a
retrain asks is another one: *would a model that kept learning from the newest weeks
have done better than the frozen champion did?* So:

* the champion is scored by the forecasts it **really issued** on the four weeks before
  the cutoff (``data/predictions/scored/``; with the config, 2 to 29 May for the retrain
  of 30 May). They are as out-of-sample as a forecast can be: each was issued at 00:00
  on the day before, from pickups that had ended by then, by a model fitted on April.
* the challenger's recipe is scored by a backtest of the same four weeks, as folds of the
  challenger's settings (``fold_windows``: the last ``backtest.n_folds`` weeks of its
  training period, an expanding window, each fold trained up to 48 hours before its
  first test hour). The rows are computed by ``monitoring.reference.backtest_forecasts``,
  the backtest's own loop that keeps the forecasts, and they must give exactly the scores
  the training run of the challenger records; that is checked.
* the weekly baseline, ``lag_168h``, is scored on the same rows.

All three are scored on exactly the same zone-hours, and that is checked, not assumed
(``comparison_table``): the same keys, the same pickups in the replay and in gold, the same
``lag_168h`` in the replay's inputs and in gold. Then the rule of ``promotion.py``,
unchanged: the pooled MAE must be at least 0.2 lower than both the champion's and the
baseline's, and no fold may be higher than either of them.

**Fair, and where it leans.** Both sides use only what existed at their issue moment: the
champion's forecasts were really issued before their day, and every fold model of the
backtest learned from no pickup newer than 48 hours before its first test hour. What
differs is the age of the model. The champion was frozen on 30 April, so in these weeks
it is 1 to 29 days old; a fold model of the backtest is 2 to 9 days old, because the
backtest refits every week. That difference is exactly what a retrain offers, so it is
what should be measured. Two ways the comparison leans towards the challenger, which the
0.2 margin only partly offsets:

* after a promotion the new model is frozen too and ages, while the backtest refitted
  every week. The comparison measures a fresh model; the deployed one is fresh only at
  first.
* a retrain is triggered *because* the champion did badly in the last week, and that
  week is one of the four it is judged on. Part of a bad week is bad luck, which the
  challenger does not share on the same rows, so the champion's live MAE there is likely
  higher than its next weeks will be.

**A known weakness of the rule: the first fold against the champion.** When the
comparison weeks start the day the champion started serving, the first fold is nearly a
tie by construction. For the retrain of 30 May, the backtest's model of that fold learned
from the rows up to 30 April 00:00 and the champion from the rows up to 30 April 23:00:
the same recipe on the same data, 23 hours apart. Their difference in that week is close
to noise, and "no fold higher" has no margin, so that fold can reject a challenger
whatever the other three weeks show. In the replay with retraining the challenger was
higher in the first fold both times: by 0.17 on 30 May (the near-tie case above) and by
0.38 on 25 June (there the fold's model had learned from about 25 days more than the champion,
so that week is not a tie). This was noticed while the replay of May and June was being
run with retraining, so the rule was **not** changed: a rule loosened on the evidence of
the data it then judges is the same mistake as tuning on the test set. Whether "no fold
higher" should apply against the champion's live forecasts is for the owner to decide,
on new evidence, before a new replay.

**Waiting for the comparison weeks.** The comparison needs the champion's forecast of
every day of the four weeks. The first champion issued its first forecast for 2 May, so
the first retrain can be judged as of 30 May (folds 2 to 29 May); an alert as of 27 to 29
May waits. After a promotion the same holds for the new champion: four weeks of its own
forecasts first. Judging it on fewer days, or on days another version forecast, would
not be the comparison above.

**Registration and the decision.** The challenger's final model is fitted on every hour
up to the cutoff, logged and registered by the training code (``training.train_from_gold``
with the challenger's settings; its reports go to ``reports/retrain/<day>/``, so the files
DVC tracks for ``train`` are never touched). The version gets the tags ``retrain.now``,
``retrain.cutoff``, ``retrain.trigger`` (the alert, with the monitoring numbers) and
``retrain.comparison``. It is judged once, at once: ``record_decision`` writes the
outcome (``promotion.decision``, with the reason and every number) before the aliases
move. A promoted version is ``champion``, and the next step's forecast loads it
(``model_loader`` resolves the alias at every step). A rejected one is ``challenger``. A
version must be reproducible to be judged (a clean git commit, a known data version), as
for ``promote``; this is checked before anything is trained.

**Rollback stays by hand** (``python -m demand set-champion``). An automatic rollback
would have to show that the new champion is worse than the old one *on the same days*,
and the old one forecast none of them: only one model serves. Comparing their live MAEs
on different weeks (late May against June) measures the weeks, not the models. Doing it
honestly needs shadow forecasts of the old champion every day, which are not built. What
does happen automatically: a new champion that falls behind raises the accuracy alert
like any other, and after its four weeks it faces a retrained challenger.

**Left out on purpose.** Tuning in a retrain; a sliding training window (every retrain
learns from all the data up to the cutoff, as the first training did); new zones; moving
the drift reference of monitoring to the new training period (``monitoring/`` keeps the
last weeks of the first training period); any lock against two retrains at once.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from demand.config import DataConfig, Settings
from demand.data.silver import report_to_json, write_text
from demand.errors import PipelineError
from demand.flows.clock import Step, forecasts_dir, scored_dir, stored_days
from demand.flows.forecast_flow import LoadModel, registry_backtest_mae, run_step
from demand.flows.monitoring_flow import (
    MakeReference,
    over_days,
    read_days,
    run_monitoring,
    scored_rows,
)
from demand.flows.scoring import BacktestMae, ReplayReport
from demand.logs import get_logger
from demand.models.backtest import fold_windows, load_training_features, score, score_by
from demand.models.baseline import HOURS_PER_WEEK, WEEKLY_NAIVE
from demand.models.promotion import (
    MIN_MAE_GAIN,
    UNKNOWN,
    Decision,
    Scores,
    candidate,
    check_reproducible,
    compare,
    decision_of,
    record_decision,
)
from demand.models.registry import Registry
from demand.monitoring.checks import BASELINE, DAY, MonitoringReport
from demand.monitoring.reference import ReferenceForecasts, backtest_forecasts
from demand.schemas import ACTUAL, FORECAST, HOUR_START, MODEL_VERSION, TARGET, ZONE_ID, lag_column
from demand.serving.model_loader import load_model
from demand.serving.snapshot import DataSnapshot, issue_moment, last_whole_day, load_snapshot

log = get_logger(__name__)

RETRAIN_REPORT_NAME = "retrain_report.json"
RETRAIN_DIR = "retrain"
NOW_TAG = "retrain.now"
"""The tag every retrain writes onto the version it registers: the moment it ran."""
CHAMPION = "champion"
CHALLENGER = "challenger"
FOLD = "fold"
KEYS = [ZONE_ID, HOUR_START]
WEEKLY = lag_column(HOURS_PER_WEEK)
"""The weekly baseline's column: the pickups of the same zone and hour a week before."""

Train = Callable[[Settings], int]
"""Train the recipe with these settings, record and register it; the new version."""
FoldForecasts = Callable[[pd.DataFrame, Settings], ReferenceForecasts]
"""``monitoring.reference.backtest_forecasts``; tests pass their own."""


class RetrainError(PipelineError):
    """A retrain cannot be evaluated or carried out as things are; the message says why."""


class ComparisonError(RetrainError):
    """The champion, the challenger and the baseline were not scored on the same rows."""


class Outcome(StrEnum):
    """What an evaluation of the retrain came to."""

    END_OF_DATA = "end_of_data"
    NO_ALERT = "no_alert"
    COOLDOWN = "cooldown"
    WAITING = "waiting"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class ComparisonScores(BaseModel):
    """The three forecasters on the same zone-hours: the weeks before the cutoff."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first_day: date
    last_day: date
    rows: int
    champion: Scores
    """The champion's live forecasts, as issued."""
    challenger: Scores
    """The retrained recipe, backtested on the same weeks."""
    baseline: Scores


class RetrainReport(BaseModel):
    """One evaluation of the retrain: at which moment, what came of it, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now: datetime
    outcome: Outcome
    reason: str
    champion_version: int
    """The champion when the evaluation started."""
    live_to_backtest_mae: float | None
    """The accuracy check of monitoring at ``now``; ``None`` if it was not evaluated."""
    last_retrain: datetime | None
    cutoff_day: date | None
    """The last day the challenger learned from; ``None`` if none was trained."""
    challenger_version: int | None
    scores: ComparisonScores | None
    decision: Decision | None


@dataclass(frozen=True)
class DayDone:
    """What one daily step did: the forecast, the monitoring and the retrain."""

    step: Step
    replay: ReplayReport
    monitoring: MonitoringReport
    retrain: RetrainReport


# --------------------------------------------------------------------------------------
# Pure: the trigger, the cutoff, the comparison
# --------------------------------------------------------------------------------------


def retrain_report_path(settings: Settings) -> Path:
    """Where the report of the last evaluation is stored."""
    return settings.paths.reports_dir / RETRAIN_REPORT_NAME


def cutoff_day(now: pd.Timestamp) -> date:
    """The last day the challenger learns from: the last whole day that is over by ``now``."""
    return last_whole_day(now)


def challenger_settings(settings: Settings, now: pd.Timestamp) -> Settings:
    """The settings of a challenger trained at ``now``.

    The same as ``settings`` except two keys: ``data.train_end_date`` is the cutoff day,
    and the reports go to ``reports/retrain/<day of now>/``. The new end of training goes
    through the checks of the config.
    """
    data = DataConfig.model_validate(
        settings.data.model_dump() | {"train_end_date": cutoff_day(now)}
    )
    reports = settings.paths.reports_dir / RETRAIN_DIR / now.date().isoformat()
    paths = settings.paths.model_copy(update={"reports_dir": reports})
    return settings.model_copy(update={"data": data, "paths": paths})


def trigger(
    monitoring: MonitoringReport,
    champion: int,
    now: pd.Timestamp,
    last_retrain: pd.Timestamp | None,
    settings: Settings,
) -> tuple[Outcome, str] | None:
    """Why no retrain is due at ``now``; ``None`` if one is.

    Only the accuracy check is read: drift never triggers a retrain.
    """
    accuracy = monitoring.accuracy
    if not accuracy.alert:
        why = accuracy.note or (
            f"live MAE is {accuracy.live_to_backtest_mae} times the backtest MAE, below "
            f"{accuracy.ratio_threshold}"
        )
        return Outcome.NO_ALERT, f"No accuracy alert: {why}"
    if accuracy.model_version != str(champion):
        return Outcome.NO_ALERT, (
            f"The accuracy alert is about version {accuracy.model_version}, which is not "
            f"the champion (version {champion})."
        )
    if last_retrain is not None:
        days = (now.date() - last_retrain.tz_convert(now.tz).date()).days
        cooldown = settings.monitoring.retrain_cooldown_days
        if last_retrain >= now or days < cooldown:
            return Outcome.COOLDOWN, (
                f"The last retrain was at {last_retrain.isoformat()}, {days} days ago; "
                f"monitoring.retrain_cooldown_days is {cooldown}."
            )
    return None


def comparison_days(challenger: Settings) -> list[date]:
    """The days of the challenger's folds: the weeks the three are compared on."""
    folds = fold_windows(challenger)
    first, last = folds[0].first_test_hour.date(), folds[-1].last_test_hour.date()
    return [first + timedelta(days=offset) for offset in range((last - first).days + 1)]


def comparison_table(
    training: pd.DataFrame,
    champion: pd.DataFrame,
    challenger: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """One row per zone-hour of the comparison weeks: the three forecasts and the pickups.

    Args:
        training: The challenger's training rows of gold (pickups and ``lag_168h``).
        champion: The champion's scored rows of those weeks (``scored_rows``).
        challenger: The challenger's backtest forecasts (``backtest_forecasts``).
        settings: The challenger's settings, for the folds.

    Raises:
        ComparisonError: If the three do not cover the same zone-hours, or the replay's
            pickups or ``lag_168h`` differ from those of gold.
    """
    folds = fold_windows(settings)
    hours = training[HOUR_START]
    window = training[(hours >= folds[0].first_test_hour) & (hours <= folds[-1].last_test_hour)]
    hour_type = training[HOUR_START].dtype
    parts = {
        "gold": window[[*KEYS, TARGET, WEEKLY]],
        "the champion's scored forecasts": champion[[*KEYS, ACTUAL, BASELINE]].assign(
            **{CHAMPION: champion[FORECAST]}
        ),
        "the challenger's backtest": challenger[KEYS].assign(**{CHALLENGER: challenger[FORECAST]}),
    }
    indexed = {
        name: part.astype({HOUR_START: hour_type}).set_index(KEYS).sort_index()
        for name, part in parts.items()
    }
    reference = indexed["gold"].index
    for name, part in indexed.items():
        if part.index.has_duplicates or not part.index.equals(reference):
            raise ComparisonError(
                f"{name.capitalize()} and gold do not cover the same zone-hours of "
                f"{folds[0].first_test_hour.date()} to {folds[-1].last_test_hour.date()}: "
                f"{len(part)} rows against {len(reference)}, "
                f"{len(part.index.difference(reference))} of them not in gold. Nothing was "
                "trained. Rebuild the data with `dvc repro` and start the replay again."
            )
    table = pd.concat(list(indexed.values()), axis=1).reset_index()
    for replay, gold in ((ACTUAL, TARGET), (BASELINE, WEEKLY)):
        if not np.array_equal(table[replay].to_numpy("float64"), table[gold].to_numpy("float64")):
            raise ComparisonError(
                f"The replay's {replay} differs from gold's {gold} on the comparison weeks: "
                "the data changed after the replay was run. Nothing was trained. Start the "
                "replay again on this data."
            )
    table[FOLD] = 0
    for fold in folds:
        inside = (table[HOUR_START] >= fold.first_test_hour) & (
            table[HOUR_START] <= fold.last_test_hour
        )
        table.loc[inside, FOLD] = fold.number
    return table


def scores_of(table: pd.DataFrame, column: str) -> Scores:
    """The pooled MAE of a forecast column of ``comparison_table``, and its MAE per fold."""
    actual = table[TARGET].to_numpy("float64")
    forecast = table[column].to_numpy("float64")
    by_fold = score_by(table[FOLD], actual, forecast)
    return Scores(
        mae=score(actual, forecast).mae, fold_mae={fold: s.mae for fold, s in by_fold.items()}
    )


def comparison_scores(table: pd.DataFrame) -> ComparisonScores:
    """The three scores of ``comparison_table``."""
    return ComparisonScores(
        first_day=table[HOUR_START].min().date(),
        last_day=table[HOUR_START].max().date(),
        rows=len(table),
        champion=scores_of(table, CHAMPION),
        challenger=scores_of(table, CHALLENGER),
        baseline=scores_of(table, WEEKLY),
    )


def judge(version: int, champion: int, scores: ComparisonScores) -> Decision:
    """The rule of ``promotion.py`` on the comparison; see the module docstring."""
    against_baseline = compare(
        scores.challenger, scores.baseline, f"the baseline ({WEEKLY_NAIVE})", MIN_MAE_GAIN
    )
    against_champion = compare(
        scores.challenger,
        scores.champion,
        f"the live forecasts of the champion (version {champion})",
        MIN_MAE_GAIN,
    )
    return decision_of(version, champion, against_baseline, against_champion, MIN_MAE_GAIN)


# --------------------------------------------------------------------------------------
# Registry and disk
# --------------------------------------------------------------------------------------


def last_retrain(registry: Registry) -> pd.Timestamp | None:
    """The moment of the newest retrain, from the ``retrain.now`` tags of the versions."""
    moments = []
    for version in registry.versions():
        tags = registry.version_tags(version)
        if NOW_TAG in tags:
            moments.append(pd.Timestamp(tags[NOW_TAG]))
    return max(moments, default=None)


def champion_gap(
    settings: Settings, now: pd.Timestamp, days: list[date], champion: int
) -> str | None:
    """Why the champion's live forecasts do not cover ``days`` yet; ``None`` if they do."""
    over = set(over_days(settings, now))
    missing = [day for day in days if day not in over]
    if missing:
        return (
            f"The comparison needs the champion's scored forecasts of {days[0]} to "
            f"{days[-1]}; {len(missing)} of these {len(days)} days have none by now (the "
            f"first is {missing[0]})."
        )
    versions = read_days(scored_dir(settings), days, [MODEL_VERSION])
    others = sorted(set(versions.loc[versions[MODEL_VERSION] != str(champion), DAY]))
    if others:
        return (
            f"The comparison needs forecasts of the champion (version {champion}) for every "
            f"day of {days[0]} to {days[-1]}; {len(others)} of them were issued by another "
            f"version (the last is {others[-1]})."
        )
    return None


def check_clean_checkout(settings: Settings) -> None:
    """Refuse to train a challenger that the gate could not promote.

    Raises:
        RetrainError: If the working directory is not a clean git checkout, or the data
            version is unknown (see ``promotion.check_reproducible``).
    """
    # Imported here: ``training`` loads SHAP and matplotlib, needed only to train.
    from demand.models.training import DVC_LOCK_FILE, data_version, git_state

    git = git_state(Path.cwd())
    problems = []
    if git.commit == UNKNOWN or git.dirty is not False:
        problems.append("the working directory is not a clean git checkout")
    if data_version(DVC_LOCK_FILE, settings.paths.gold) == UNKNOWN:
        problems.append("the data version is unknown")
    if problems:
        raise RetrainError(
            f"A retrain is due, but {' and '.join(problems)}, so the gate could not promote "
            "the new model. Nothing was trained. Commit the code, build the data with "
            "`dvc repro`, and run the step again."
        )


def train_challenger(settings: Settings) -> int:
    """Train, record and register the recipe with ``settings``; the new version."""
    from demand.models.pipeline import forecasters
    from demand.models.training import train_from_gold

    return train_from_gold(settings, forecasters(settings)).model_version


def retrain_if_needed(
    settings: Settings,
    snapshot: DataSnapshot,
    now: pd.Timestamp,
    monitoring: MonitoringReport,
    *,
    train: Train = train_challenger,
    fold_forecasts: FoldForecasts = backtest_forecasts,
) -> RetrainReport:
    """Decide whether to retrain at ``now``, and if so train, judge and record a challenger.

    Args:
        settings: Project settings.
        snapshot: The data, for the last day that can be forecast.
        now: The moment of the step; nothing newer is read.
        monitoring: The monitoring report as of ``now``.
        train: Trains and registers the challenger; tests pass their own.
        fold_forecasts: The challenger's backtest forecasts; tests pass their own.

    Raises:
        RetrainError: If no model is champion, a retrain is due but the checkout or the
            data version would make it unpromotable, or the three forecasters were not
            scored on the same rows (``ComparisonError``).
        PipelineError: If the gold table is missing or training fails.
    """
    started = time.monotonic()
    registry = Registry(settings)
    champion = registry.aliases().get(settings.mlflow.champion_alias)
    if champion is None:
        raise RetrainError(
            "No version is champion, so there is nothing to retrain against. Train and "
            "promote one first: `dvc repro`, then `python -m demand promote`."
        )
    last = last_retrain(registry)
    known = {
        "now": now.to_pydatetime(),
        "champion_version": champion,
        "live_to_backtest_mae": monitoring.accuracy.live_to_backtest_mae,
        "last_retrain": None if last is None else last.to_pydatetime(),
    }
    forecast = stored_days(forecasts_dir(settings))
    found = (
        (
            Outcome.END_OF_DATA,
            f"No day is left to forecast after {snapshot.last_day}, so a new model would "
            "serve nothing.",
        )
        if not forecast or forecast[-1] >= snapshot.last_day
        else trigger(monitoring, champion, now, last, settings)
    )
    if found is None:
        candidate_settings = challenger_settings(settings, now)
        days = comparison_days(candidate_settings)
        gap = champion_gap(settings, now, days, champion)
        found = None if gap is None else (Outcome.WAITING, gap)
    if found is not None:
        report = RetrainReport(
            **known,
            outcome=found[0],
            reason=found[1],
            cutoff_day=None,
            challenger_version=None,
            scores=None,
            decision=None,
        )
        return _finish(settings, report, started)

    check_clean_checkout(settings)
    training = load_training_features(candidate_settings)
    challenger = fold_forecasts(training, candidate_settings)
    table = comparison_table(
        training, scored_rows(settings, days), challenger.rows, candidate_settings
    )
    scores = comparison_scores(table)
    version = train(candidate_settings)
    cutoff = candidate_settings.data.train_end_date
    registry.tag(
        version,
        {
            NOW_TAG: now.isoformat(),
            "retrain.cutoff": cutoff.isoformat(),
            "retrain.trigger": monitoring.alerts[0].reason,
            "retrain.comparison": (
                f"{scores.rows} zone-hours of {scores.first_day} to {scores.last_day}: MAE "
                f"{scores.challenger.mae} for the retrained recipe (backtest), "
                f"{scores.champion.mae} for the champion's live forecasts (version "
                f"{champion}), {scores.baseline.mae} for the baseline"
            ),
        },
    )
    recorded = candidate(registry.version(version))
    check_reproducible(recorded)
    if (recorded.model, recorded.baseline) != (scores.challenger, scores.baseline):
        raise ComparisonError(
            f"Version {version} recorded a backtest MAE of {recorded.model.mae} (baseline "
            f"{recorded.baseline.mae}), but the comparison scored {scores.challenger.mae} "
            f"({scores.baseline.mae}) on the same weeks: they are not the same forecasts. "
            f"Version {version} was registered but not judged, and is not served."
        )
    decision = judge(version, champion, scores)
    record_decision(settings, registry, decision)
    report = RetrainReport(
        **known,
        outcome=Outcome.PROMOTED if decision.promoted else Outcome.REJECTED,
        reason=decision.reason,
        cutoff_day=cutoff,
        challenger_version=version,
        scores=scores,
        decision=decision,
    )
    return _finish(settings, report, started)


def _finish(settings: Settings, report: RetrainReport, started: float) -> RetrainReport:
    write_text(report_to_json(report), retrain_report_path(settings))
    log.info(
        "retrain evaluated",
        extra={
            "now": report.now.isoformat(),
            "outcome": report.outcome.value,
            "reason": report.reason,
            "champion": report.champion_version,
            "challenger": report.challenger_version,
            "seconds": round(time.monotonic() - started, 1),
        },
    )
    return report


def run_retrain(
    settings: Settings,
    *,
    backtest_mae: Callable[[Settings], BacktestMae] = registry_backtest_mae,
    make_reference: MakeReference = backtest_forecasts,
    train: Train = train_challenger,
    fold_forecasts: FoldForecasts = backtest_forecasts,
) -> RetrainReport:
    """Stage function of ``python -m demand retrain``: evaluate at the last step's moment.

    The moment is the issue moment of the newest forecast; monitoring is evaluated as of
    it first, and its report overwrites ``reports/monitoring_report.json`` (see the module
    docstring). The other arguments are for tests.

    Raises:
        RetrainError: If no forecast has been issued yet; see also ``retrain_if_needed``.
    """
    days = stored_days(forecasts_dir(settings))
    if not days:
        raise RetrainError(
            "No forecast has been issued yet, so there is no step to retrain at. Run "
            "`python -m demand forecast` first."
        )
    now = issue_moment(days[-1], settings)
    monitoring = run_monitoring(
        settings,
        as_of=now.date(),
        html=False,
        backtest_mae=backtest_mae,
        make_reference=make_reference,
    )
    return retrain_if_needed(
        settings,
        load_snapshot(settings),
        now,
        monitoring,
        train=train,
        fold_forecasts=fold_forecasts,
    )


def run_day(
    settings: Settings,
    snapshot: DataSnapshot,
    *,
    load: LoadModel = load_model,
    backtest_mae: Callable[[Settings], BacktestMae] = registry_backtest_mae,
    make_reference: MakeReference = backtest_forecasts,
    train: Train = train_challenger,
    fold_forecasts: FoldForecasts = backtest_forecasts,
) -> DayDone:
    """One daily step: forecast and score, monitor as of the same moment, retrain if due.

    The Prefect flow calls this once a day. The other arguments are for tests and the
    replay (which passes a ``make_reference`` it computes once).

    Raises:
        PipelineError: See ``forecast_flow.run_step``, ``monitoring_flow.run_monitoring``
            and ``retrain_if_needed``.
    """
    step, replay = run_step(settings, snapshot, load=load, backtest_mae=backtest_mae)
    monitoring = run_monitoring(
        settings,
        as_of=step.now.date(),
        html=False,
        backtest_mae=backtest_mae,
        make_reference=make_reference,
    )
    retrain = retrain_if_needed(
        settings, snapshot, step.now, monitoring, train=train, fold_forecasts=fold_forecasts
    )
    return DayDone(step=step, replay=replay, monitoring=monitoring, retrain=retrain)
