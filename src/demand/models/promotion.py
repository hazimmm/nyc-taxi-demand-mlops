"""The promotion gate: decide whether a newly trained model replaces the one that is served.

Layout of the report (``settings.paths.reports_dir``)::

    reports/
      promotion_report.json    the last decision of the gate: the versions, every number
                               that was compared, and the reason

Every ``python -m demand train`` registers a new version of the model in MLflow. Which
version is *served* is said by the alias ``champion`` (``serving.model_uri`` is
``models:/taxi-demand@champion``). This module decides when that alias moves. The rule,
in one sentence:

    **The newest version becomes champion only if its backtest MAE is at least 0.2
    lower than that of the weekly baseline and than that of the current champion, and
    not higher than either of them in any fold; the scores of the champion count only if
    they were measured on the same folds of the same data; and a version is judged
    once, never again.**

The decision itself, ``decide``, is a pure function: numbers and labels go in, a
``Decision`` comes out, and nothing is read or written. Reading the versions and moving
the aliases is done by ``promote`` through ``registry.py``.

**When the gate runs: its own command, after training.** ``python -m demand promote``
looks at the newest version and decides. It is not part of ``train``, for three reasons.

* ``train`` is a DVC stage. DVC tracks files, and the aliases live in the MLflow
  database, which DVC does not see. A gate inside ``train`` would make ``dvc repro``
  change the served model as a side effect, and ``dvc repro --force train``, which is
  the documented way to record a model again, would do it too.
* Training answers "how good is this recipe?" and the gate answers "should it be
  served?". The second question deserves a moment in which a person can read the first
  answer (the backtest report, the SHAP values) before acting on it.
* The retraining flow (``flows/retrain_flow.py``) runs both in a row, with scores it
  measures itself (below, "The retraining flow").

**What "better" means.** Four choices, each with its reason:

* *The MAE pooled over all test rows of all folds is the number that decides.* It is the
  headline of the project (``metrics.py``). The RMSE and the sMAPE are shown in the
  reports but do not decide: the sMAPE should never carry a decision (``metrics.py``
  lists why), and the RMSE is decided by a handful of hours. Every extra metric that
  must improve is one more way for a real gain to be rejected by chance.
* *It must be at least 0.2 lower* (``MIN_MAE_GAIN``). The same recipe trained with
  another seed scores up to 0.19 higher or lower on these four weeks (the seed check of
  ``tuning.py``: 0.194539 before tuning and 0.192245 after, as the range over the seeds
  42, 43 and 44). A gain smaller than that can be the seed alone. A margin of 0.2 lets
  real gains through (tuning gained 0.82 with seed 42, and about 0.64 on weeks the
  search had not seen) and stops a model that is merely different. A tie, a gain of
  0, never promotes: training the same recipe again gives exactly the same scores.
  The same margin is used against the baseline, because the challenger's own score
  carries the noise, whatever it is compared with.
* *It must not be higher in any fold.* A recipe can win the pooled number by being much
  better in one week and worse in the other three. The tuning rule of this project asks
  for a gain in every fold, and so does the gate. Only the direction counts in a single
  fold, with no margin: a fold has a quarter of the rows, so its seed noise is larger
  than 0.2 (it was not measured), and a margin in each of four folds would reject real
  gains. A fold of exactly equal MAE is not "higher".
* *The margin is a constant here, not a key of the config.* It is not a preference but a
  measured fact about this backtest, the size of the seed noise. It must change when that
  measurement changes (other folds, other parameters), and then together with the new
  measurement, in a commit that shows it; a key in the config could be lowered to let a
  model through without anyone measuring anything.

Even with both checks, a model that is no better can pass by luck now and then: two
recipes measured with one seed each can differ by the noise of both. The gate makes that
rare, not impossible. The clean exam of any version is the replay of May and June.

**Comparable scores.** A backtest score belongs to the rows it was measured on. The
champion's stored scores can only be compared with the challenger's if both runs scored
the same rows with the same values, the same way. The gate checks this from what every
training run records (``training.py``): the parameters in ``SETUP_PARAMS`` (end of the
training period, zones, horizon, lags, windows, outlier threshold, model inputs, number
and length of the folds) and the tags in ``SETUP_TAGS`` (the data version and the
baseline) must be the same for both. A value of ``unknown`` never counts as the same,
not even as the same as another ``unknown``. Then a second check: the baseline does not
learn anything, so on the same rows it must score exactly the same in every fold. If it
does not, the data changed without a new data version (``dvc.lock`` is stale after
``python -m demand gold`` by hand), and the gate stops instead of trusting the tags.

When the two differ, the gate **refuses**: it raises ``IncomparableRunsError``, says
which keys differ, and changes nothing. It does not re-score the champion on the
challenger's folds. That would mean training the champion's recipe again inside this
command, with its parameters read back from text, its feature config possibly no longer
buildable, and a backtest of minutes. The retraining flow, which has new folds every
time, measures both sides itself (below). A person who wants to replace a champion
across a change of setup compares the two by hand and uses ``set-champion`` (below),
which records that it was done by hand.

**The champion must be reproducible.** A version is promoted only if its run was
recorded from a known git commit with no changed files, and with a known data version.
Otherwise nobody could rebuild the model that is served, so the gate refuses with a
``PipelineError`` that says what to do: commit, train again with ``dvc repro``, promote.

**No champion yet.** The first time, there is nothing to compare with except the
baseline. Then the newest version must beat the baseline by the same rule, and the
decision records that there was no champion before it.

**The aliases after a decision.** ``champion`` points at the served version.
``challenger`` points at the newest version the gate looked at and did *not* promote,
so a person can find the model that lost. When a version is promoted it becomes
``champion`` and the ``challenger`` alias is removed, so the two never point at the
same version. Only the newest version is judged; an older version that never went
through the gate keeps no alias and no decision.

**A version is judged once.** If the newest version already carries a decision (the tag
``promotion.decision``: promoted, rejected or manual), ``promote`` decides nothing and
says so in the log. Without this rule a rollback would not last. Suppose version 2 is
promoted, turns out badly in production, and a person sets version 1 back by hand. The
next routine ``promote`` (by a person, or by the retraining flow) would judge version 2
again, against version 1, on the same backtest numbers, and serve it again: the reason
the person had would be overruled by numbers that never knew it. The same holds for a
rejected version: when the champion changes, it must not slip in against a weaker one.
So a judgement stands. A version comes back only through ``set-champion``, by a person
who gives a reason, or a new version is trained and judged. A version the gate refused
(not reproducible, or not comparable) carries no decision and is judged when ``promote``
runs again.

**The record.** Every decision is written onto the version it is about, as tags whose
names start with ``promotion.``: the outcome, the reason (with every number that was
compared and the margin), the time, and the champion before it. Tags stay with the version
for as long as the store exists, so "why is version 2 the champion" is answered by the
tags of version 2 (``promotion.decision = promoted``) and "why not version 3" by those of
version 3. The full decision, with the MAE of every fold, is also written to
``reports/promotion_report.json``. That file holds only the last decision and, like the
other reports, is not in git; the tags are the record that lasts. The tags are written
before the aliases move, so an alias never moves without a record. If the command stops
between the two (two quick writes to the same database, so this is rare), the version is
recorded as promoted but not served, and running ``promote`` again does not finish the
job, because the version has been judged; ``set-champion`` does, and records it.

**By hand: set-champion.** A team needs a way to undo a promotion (the model turns out
worse on the replay) or to promote across a change of setup the gate cannot compare.
``set_champion`` moves the ``champion`` alias to any existing version, without any check,
and requires a reason. It writes the same tags, with ``promotion.decision = manual``, so
the record shows that the rule was not applied and why. It is the only way back for a
version the gate has judged, and it replaces that version's earlier record: the tags of a
version hold its latest decision, not its history.

**The retraining flow.** A retrain learns from newer data than the champion, so its
backtest folds are not the champion's and ``check_comparable`` would always refuse. The
flow therefore measures both sides on the same rows itself: the champion by the
forecasts it really issued in the last four weeks, the retrained recipe by a backtest of
the same four weeks. It applies the rule with ``compare`` and ``decision_of`` and records
the outcome with ``record_decision``, as ``promote`` does. The rule is the same in every
part: the margin on the pooled MAE and no fold higher, against both references.

**What the gate does not do.** It does not re-score anything, does not look at May or
June, does not lock the store against two people promoting at once, does not judge
versions other than the newest, and does not judge a version twice.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.silver import report_to_json, write_text
from demand.errors import PipelineError
from demand.logs import get_logger
from demand.models.registry import Registry, StoredVersion

log = get_logger(__name__)

PROMOTION_REPORT_NAME = "promotion_report.json"

MIN_MAE_GAIN = 0.2
"""How much lower the pooled MAE must be, in pickups. See the module docstring."""

SETUP_PARAMS = (
    "data.train_end_date",
    "data.zones",
    "features.horizon_hours",
    "features.lags",
    "features.rolling_windows",
    "features.outlier_zscore",
    "feature_columns",
    "backtest.n_folds",
    "backtest.fold_days",
)
"""Run parameters that decide which rows a backtest scores, and how."""

SETUP_TAGS = ("data_version", "baseline")
"""Run tags that decide the same: the exact data, and what the model is compared with."""

UNKNOWN = "unknown"
"""What ``training.py`` records for a fact it could not find (``training.UNKNOWN``).

Written here and not imported: importing ``training`` loads SHAP and matplotlib, which
take seconds and are of no use to this command. A test checks that the two agree.
"""

DECISION_TAG = "promotion.decision"
"""The tag that records an outcome on a version; a version that has it is not judged again."""

NO_CHAMPION = "none"
_DECIMALS = 6


class IncomparableRunsError(PipelineError, ValueError):
    """The champion and the challenger were not scored on the same backtest."""


class Scores(BaseModel):
    """The MAE of one forecaster in a backtest: over all test rows, and per fold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mae: float
    fold_mae: dict[int, float]


class Candidate(BaseModel):
    """What the gate knows about one version: where it came from and how it scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    git_commit: str
    git_dirty: str
    """``false``, ``true`` or ``unknown``, as the training run recorded it."""
    setup: dict[str, str]
    """The values of ``SETUP_PARAMS`` and ``SETUP_TAGS`` of its run."""
    model: Scores
    baseline: Scores
    """The baseline, scored in the same run on the same rows."""


class Comparison(BaseModel):
    """The challenger against one reference: the baseline or the champion."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    against: str
    reference_mae: float
    challenger_mae: float
    mae_gain: float
    """How much lower the challenger's MAE is; negative if it is higher."""
    worse_folds: list[int]
    """The folds in which the challenger's MAE is higher than the reference's."""
    beaten: bool


class Decision(BaseModel):
    """What the gate decided about one version, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    challenger_version: int
    champion_version: int | None
    """The champion before the decision; ``None`` if there was none."""
    promoted: bool
    reason: str
    min_mae_gain: float
    against_baseline: Comparison
    against_champion: Comparison | None


class PromotionReport(Decision):
    """A decision as it is saved next to the other reports."""

    registered_model: str
    decided_at: datetime


# --------------------------------------------------------------------------------------
# Pure: the decision
# --------------------------------------------------------------------------------------


def promotion_report_path(reports_dir: Path) -> Path:
    """Where the report of the last decision is stored."""
    return reports_dir / PROMOTION_REPORT_NAME


def candidate(stored: StoredVersion) -> Candidate:
    """The facts and scores of a registered version, read from what its run recorded.

    Raises:
        PipelineError: If the run lacks a parameter, tag or score the gate needs; it was
            then not recorded by ``python -m demand train`` of this project.
    """
    params, tags, metrics = stored.params, stored.tags, stored.metrics
    needed_tags = ("forecaster", "git_commit", "git_dirty", *SETUP_TAGS)
    _require(
        stored.version,
        [key for key in SETUP_PARAMS if key not in params]
        + [key for key in needed_tags if key not in tags],
    )
    folds = range(1, int(params["backtest.n_folds"]) + 1)
    names = (tags["forecaster"], tags["baseline"])
    keys = [
        f"{name}.{group}mae" for name in names for group in ["", *(f"fold_{n}." for n in folds)]
    ]
    _require(stored.version, [key for key in keys if key not in metrics])

    def scores(name: str) -> Scores:
        return Scores(
            mae=metrics[f"{name}.mae"],
            fold_mae={n: metrics[f"{name}.fold_{n}.mae"] for n in folds},
        )

    return Candidate(
        version=stored.version,
        git_commit=tags["git_commit"],
        git_dirty=tags["git_dirty"],
        setup={key: params[key] for key in SETUP_PARAMS} | {key: tags[key] for key in SETUP_TAGS},
        model=scores(names[0]),
        baseline=scores(names[1]),
    )


def _require(version: int, missing: list[str]) -> None:
    if missing:
        raise PipelineError(
            f"The run of version {version} has no {', '.join(missing)}. The gate reads what "
            "`python -m demand train` records; a run recorded in another way, or by an "
            "older version of train, cannot be judged. Train again with `dvc repro`."
        )


def check_reproducible(challenger: Candidate) -> None:
    """Refuse a version that could not be rebuilt from its git commit and data version.

    Raises:
        PipelineError: If the commit or the data version is unknown, or files differed
            from the commit when the run was recorded.
    """
    problems = []
    if challenger.git_commit == UNKNOWN:
        problems.append("its git commit is unknown")
    elif challenger.git_dirty != "false":
        problems.append(
            f"files differed from commit {challenger.git_commit[:7]} when it was trained "
            f"(git_dirty is {challenger.git_dirty})"
        )
    if challenger.setup["data_version"] == UNKNOWN:
        problems.append("its data version is unknown")
    if problems:
        raise PipelineError(
            f"Version {challenger.version} cannot be promoted: {'; '.join(problems)}. A "
            "champion must be reproducible from its commit and its data version. Commit "
            "the code, train again with `dvc repro`, and run `python -m demand promote`."
        )


def check_comparable(champion: Candidate, challenger: Candidate) -> None:
    """Refuse to compare two versions that were not scored on the same backtest.

    Raises:
        IncomparableRunsError: If a key of the setup differs or is unknown, or the
            baseline scored differently on the two runs.
    """
    differences = [
        f"{key} is {champion.setup[key]} for the champion and {challenger.setup[key]} for "
        f"version {challenger.version}"
        for key in (*SETUP_PARAMS, *SETUP_TAGS)
        if champion.setup[key] != challenger.setup[key] or champion.setup[key] == UNKNOWN
    ]
    if differences:
        raise IncomparableRunsError(
            f"Version {challenger.version} and the champion, version {champion.version}, "
            f"were not scored on the same backtest: {'; '.join(differences)}. Nothing was "
            "changed. The gate cannot compare them (re-scoring the champion is not built). "
            "Compare the two by hand, and if the new version should be served, run "
            f"`python -m demand set-champion {challenger.version} --reason TEXT`."
        )
    # Exact equality on purpose: the baseline learns nothing, and its scores are rounded
    # the same way every time, so on the same rows they are the same numbers.
    if champion.baseline != challenger.baseline:
        raise IncomparableRunsError(
            f"Version {challenger.version} and the champion, version {champion.version}, "
            "have the same setup, but the baseline scored differently on them (MAE "
            f"{champion.baseline.mae} and {challenger.baseline.mae}), so the rows were not "
            "the same. The data changed without a new data version, which happens after "
            "`python -m demand gold` by hand: dvc.lock then names the old data. Nothing "
            "was changed. Rebuild with `dvc repro` and run `python -m demand promote`."
        )


def compare(challenger: Scores, reference: Scores, against: str, min_mae_gain: float) -> Comparison:
    """Whether ``challenger`` beats ``reference``: MAE lower by the margin, no fold higher."""
    gain = round(reference.mae - challenger.mae, _DECIMALS)
    worse = [fold for fold, mae in challenger.fold_mae.items() if mae > reference.fold_mae[fold]]
    return Comparison(
        against=against,
        reference_mae=reference.mae,
        challenger_mae=challenger.mae,
        mae_gain=gain,
        worse_folds=worse,
        beaten=gain >= min_mae_gain and not worse,
    )


def _describe(comparison: Comparison, min_mae_gain: float) -> str:
    worse = comparison.worse_folds
    folds = (
        f"higher in fold{'s' if len(worse) > 1 else ''} {', '.join(map(str, worse))}"
        if worse
        else "no fold higher"
    )
    return (
        f"against {comparison.against}, MAE {comparison.challenger_mae} against "
        f"{comparison.reference_mae}, a gain of {comparison.mae_gain} (at least "
        f"{min_mae_gain} needed), {folds}"
    )


def decide(
    challenger: Candidate, champion: Candidate | None, min_mae_gain: float = MIN_MAE_GAIN
) -> Decision:
    """Apply the rule of the module docstring to the newest version and the champion.

    Args:
        challenger: The version to judge.
        champion: The version that is served now, or ``None`` if none is.
        min_mae_gain: How much lower the pooled MAE must be.

    Raises:
        PipelineError: If the challenger is not reproducible.
        IncomparableRunsError: If the two were not scored on the same backtest.
    """
    check_reproducible(challenger)
    if champion is not None:
        check_comparable(champion, challenger)
    against_baseline = compare(
        challenger.model,
        challenger.baseline,
        f"the baseline ({challenger.setup['baseline']})",
        min_mae_gain,
    )
    against_champion = (
        None
        if champion is None
        else compare(
            challenger.model,
            champion.model,
            f"the champion (version {champion.version})",
            min_mae_gain,
        )
    )
    return decision_of(
        challenger.version,
        None if champion is None else champion.version,
        against_baseline,
        against_champion,
        min_mae_gain,
    )


def decision_of(
    challenger_version: int,
    champion_version: int | None,
    against_baseline: Comparison,
    against_champion: Comparison | None,
    min_mae_gain: float = MIN_MAE_GAIN,
) -> Decision:
    """The decision from the comparisons: promoted only if every reference is beaten.

    ``decide`` calls it with the backtest scores of two runs, the retraining flow with
    scores it measured itself. ``against_champion`` is ``None`` when there is no champion.
    """
    comparisons = (
        [against_baseline] if against_champion is None else [against_baseline, against_champion]
    )
    promoted = all(comparison.beaten for comparison in comparisons)
    reason = (
        f"{'Promoted' if promoted else 'Rejected'}: "
        + "; ".join(_describe(comparison, min_mae_gain) for comparison in comparisons)
        + "."
    )
    if against_champion is None:
        reason += " There was no champion, so only the baseline had to be beaten."
    return Decision(
        challenger_version=challenger_version,
        champion_version=champion_version,
        promoted=promoted,
        reason=reason,
        min_mae_gain=min_mae_gain,
        against_baseline=against_baseline,
        against_champion=against_champion,
    )


def decision_tags(
    outcome: str, reason: str, champion_before: int | None, decided_at: datetime
) -> dict[str, str]:
    """The tags that record a decision on the version it is about.

    Args:
        outcome: ``promoted``, ``rejected`` or ``manual``.
        reason: Why; for the gate, every number that was compared and the margin.
        champion_before: The champion at the moment of the decision, if there was one.
        decided_at: When, in UTC.
    """
    return {
        DECISION_TAG: outcome,
        "promotion.reason": reason,
        "promotion.decided_at": decided_at.isoformat(),
        "promotion.champion_before": NO_CHAMPION
        if champion_before is None
        else str(champion_before),
    }


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------


def promote(settings: Settings) -> Decision | None:
    """Judge the newest version, record the decision, and move the aliases.

    Returns:
        The decision, or ``None`` if there is nothing to decide: the newest version
        already carries a decision (of the gate or by hand), or is already the champion.
        A rejected version is a normal outcome, not an error.

    Raises:
        PipelineError: If no model is registered, a run lacks what the gate reads, or
            the newest version is not reproducible.
        IncomparableRunsError: If it was not scored on the same backtest as the champion.
    """
    config = settings.mlflow
    registry = Registry(settings)
    newest = registry.newest()
    champion_version = registry.aliases().get(config.champion_alias)
    earlier = registry.version_tags(newest.version).get(DECISION_TAG)
    if earlier is not None:
        log.info(
            "nothing to decide: the newest version was judged before, and a version is judged once",
            extra={
                "version": newest.version,
                "earlier_decision": earlier,
                "champion": champion_version,
            },
        )
        return None
    if champion_version == newest.version:
        # Possible only if the alias was set outside this module, without a record.
        log.info(
            "nothing to decide: the newest version is already the champion",
            extra={"version": newest.version},
        )
        return None
    champion = None if champion_version is None else candidate(registry.version(champion_version))
    decision = decide(candidate(newest), champion)
    record_decision(settings, registry, decision)
    return decision


def record_decision(settings: Settings, registry: Registry, decision: Decision) -> PromotionReport:
    """Write a decision onto its version, move the aliases, and write the report.

    The tags are written before the aliases move (see "The record" above). Used by
    ``promote`` and by the retraining flow.
    """
    config = settings.mlflow
    version = decision.challenger_version
    decided_at = datetime.now(UTC)
    outcome = "promoted" if decision.promoted else "rejected"
    registry.tag(
        version, decision_tags(outcome, decision.reason, decision.champion_version, decided_at)
    )
    if decision.promoted:
        registry.set_alias(config.champion_alias, version)
        registry.remove_alias(config.challenger_alias)
    else:
        registry.set_alias(config.challenger_alias, version)

    report = PromotionReport(
        **decision.model_dump(), registered_model=registry.name, decided_at=decided_at
    )
    report_path = promotion_report_path(settings.paths.reports_dir)
    write_text(report_to_json(report), report_path)
    log.info(
        "promotion decided",
        extra={
            "version": version,
            "outcome": outcome,
            "champion_before": decision.champion_version,
            "reason": decision.reason,
            "report": str(report_path),
        },
    )
    return report


def run_promotion(settings: Settings) -> None:
    """Stage function of ``python -m demand promote``; see ``promote``."""
    promote(settings)


def set_champion(settings: Settings, version: int, reason: str) -> None:
    """Make ``version`` the champion by hand, without the rule, and record why.

    For a rollback, or a change of setup the gate cannot compare. The version gets the
    tags of a decision with ``promotion.decision = manual``. If it was the challenger, it
    stops being one. Nothing happens if it is already the champion.

    Raises:
        PipelineError: If ``reason`` is empty, no model is registered, or there is no
            such version.
    """
    reason = reason.strip()
    if not reason:
        raise PipelineError(
            "Say why the champion is set by hand: --reason TEXT. The reason is recorded on "
            "the version, so that it can be read later why the rule was not applied."
        )
    config = settings.mlflow
    registry = Registry(settings)
    registry.version(version)
    aliases = registry.aliases()
    before = aliases.get(config.champion_alias)
    if before == version:
        log.info("nothing to do: the version is already the champion", extra={"version": version})
        return
    tags = decision_tags("manual", f"Set by hand: {reason}", before, datetime.now(UTC))
    registry.tag(version, tags)
    registry.set_alias(config.champion_alias, version)
    if aliases.get(config.challenger_alias) == version:
        registry.remove_alias(config.challenger_alias)
    log.info(
        "champion set by hand",
        extra={"version": version, "champion_before": before, "reason": reason},
    )
