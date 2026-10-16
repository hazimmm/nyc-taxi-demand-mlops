"""The promotion gate: the rule on hand-made numbers, then the aliases in a real store.

The first half needs no MLflow: ``decide`` is given two ``Candidate`` objects. The
second half registers versions in a temporary store (``Store`` in ``conftest.py``) and
checks what ``promote`` and ``set_champion`` leave behind. The fold scores of the real
versions 1 and 2 are used where a test needs realistic numbers.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from demand.errors import PipelineError
from demand.models import training
from demand.models.promotion import (
    MIN_MAE_GAIN,
    UNKNOWN,
    Candidate,
    IncomparableRunsError,
    PromotionReport,
    Scores,
    candidate,
    compare,
    decide,
    decision_tags,
    promote,
    promotion_report_path,
    run_promotion,
    set_champion,
)
from demand.models.registry import StoredVersion
from tests.unit.models.conftest import (
    BASELINE_FOLDS,
    RUN_PARAMS,
    RUN_TAGS,
    TUNED_FOLDS,
    UNTUNED_FOLDS,
    Store,
    mae_metrics,
    pooled,
)


def _scores(folds: Sequence[float]) -> Scores:
    return Scores(mae=pooled(folds), fold_mae=dict(enumerate(folds, start=1)))


def _candidate(
    version: int,
    model: Sequence[float],
    *,
    baseline_folds: Sequence[float] = BASELINE_FOLDS,
    git_commit: str = "5d1d9e4",
    git_dirty: str = "false",
    **setup: str,
) -> Candidate:
    """A version with the setup of the real runs, changed where ``setup`` says."""
    real = candidate(
        StoredVersion(
            version=version,
            run_id="run",
            params=RUN_PARAMS,
            tags=RUN_TAGS,
            metrics=mae_metrics(model),
        )
    )
    return real.model_copy(
        update={
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "setup": real.setup | setup,
            "baseline": _scores(baseline_folds),
        }
    )


def _minus(folds: Sequence[float], amount: float) -> tuple[float, ...]:
    return tuple(round(mae - amount, 6) for mae in folds)


# --------------------------------------------------------------------------------------
# Reading a run
# --------------------------------------------------------------------------------------


def test_a_candidate_is_read_from_what_the_training_run_recorded() -> None:
    stored = StoredVersion(
        version=2,
        run_id="run",
        params=RUN_PARAMS,
        tags=RUN_TAGS | {"mlflow.user": "someone"},
        metrics=mae_metrics(TUNED_FOLDS) | {"lightgbm.rmse": 38.2},
    )

    found = candidate(stored)

    assert found.version == 2
    assert found.git_commit == RUN_TAGS["git_commit"]
    assert found.git_dirty == "false"
    assert found.model == Scores(
        mae=24.106025, fold_mae={1: 23.510474, 2: 21.84912, 3: 24.753347, 4: 26.311159}
    )
    assert found.baseline.mae == 27.70067
    assert found.setup["data_version"] == "7bacf1bf80cf4612d40552e74ef11cc5.dir"
    assert found.setup["features.lags"] == "[48, 168]"
    # The parameters of the model are not part of the setup: they are what may change.
    assert "model.learning_rate" not in found.setup


def test_a_run_without_a_parameter_or_tag_of_the_setup_cannot_be_judged() -> None:
    params = {key: value for key, value in RUN_PARAMS.items() if key != "features.lags"}
    tags = {key: value for key, value in RUN_TAGS.items() if key != "data_version"}
    stored = StoredVersion(
        version=3, run_id="run", params=params, tags=tags, metrics=mae_metrics(TUNED_FOLDS)
    )

    with pytest.raises(PipelineError, match=r"version 3 has no features\.lags, data_version"):
        candidate(stored)


def test_a_run_without_the_score_of_a_fold_cannot_be_judged() -> None:
    metrics = mae_metrics(TUNED_FOLDS)
    del metrics["weekly_naive.fold_4.mae"]
    stored = StoredVersion(
        version=3, run_id="run", params=RUN_PARAMS, tags=RUN_TAGS, metrics=metrics
    )

    with pytest.raises(PipelineError, match=r"no weekly_naive\.fold_4\.mae\. .*dvc repro"):
        candidate(stored)


def test_unknown_means_what_training_records() -> None:
    assert UNKNOWN == training.UNKNOWN


# --------------------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------------------


def test_the_first_version_is_promoted_when_it_beats_the_baseline() -> None:
    decision = decide(_candidate(2, TUNED_FOLDS), None)

    assert decision.promoted
    assert decision.champion_version is None
    assert decision.against_champion is None
    assert decision.against_baseline.mae_gain == 3.594645
    assert decision.reason == (
        "Promoted: against the baseline (weekly_naive), MAE 24.106025 against 27.70067, a "
        "gain of 3.594645 (at least 0.2 needed), no fold higher. There was no champion, so "
        "only the baseline had to be beaten."
    )


def test_the_first_version_is_rejected_when_it_does_not_beat_the_baseline() -> None:
    decision = decide(_candidate(1, _minus(BASELINE_FOLDS, 0.1)), None)

    assert not decision.promoted
    assert decision.against_baseline.mae_gain == 0.1
    assert decision.reason.startswith("Rejected:")


def test_a_version_that_beats_the_champion_and_the_baseline_is_promoted() -> None:
    decision = decide(_candidate(2, TUNED_FOLDS), _candidate(1, UNTUNED_FOLDS))

    assert decision.promoted
    assert decision.champion_version == 1
    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain == 0.823893
    assert decision.against_champion.worse_folds == []
    assert "against the champion (version 1), MAE 24.106025 against 24.929918" in (decision.reason)


def test_a_version_that_beats_only_the_baseline_is_rejected() -> None:
    decision = decide(_candidate(3, UNTUNED_FOLDS), _candidate(2, TUNED_FOLDS))

    assert not decision.promoted
    assert decision.against_baseline.beaten
    assert decision.against_champion is not None
    assert not decision.against_champion.beaten
    assert decision.against_champion.mae_gain == -0.823893
    assert decision.against_champion.worse_folds == [1, 2, 3, 4]
    assert "higher in folds 1, 2, 3, 4." in decision.reason


def test_a_version_that_beats_only_the_champion_is_rejected() -> None:
    # A champion set by hand need not beat the baseline; the challenger still must.
    weak_champion = _candidate(1, _minus(BASELINE_FOLDS, -1.0))

    decision = decide(_candidate(2, _minus(BASELINE_FOLDS, 0.1)), weak_champion)

    assert decision.against_champion is not None
    assert decision.against_champion.beaten
    assert not decision.against_baseline.beaten
    assert not decision.promoted


def test_a_tie_with_the_champion_is_rejected() -> None:
    decision = decide(_candidate(3, TUNED_FOLDS), _candidate(2, TUNED_FOLDS))

    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain == 0
    assert decision.against_champion.worse_folds == []
    assert not decision.promoted


def test_a_gain_inside_the_margin_is_rejected() -> None:
    decision = decide(_candidate(3, _minus(TUNED_FOLDS, 0.19)), _candidate(2, TUNED_FOLDS))

    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain == 0.19
    assert decision.against_champion.worse_folds == []
    assert not decision.promoted
    assert "a gain of 0.19 (at least 0.2 needed)" in decision.reason


def test_a_gain_of_exactly_the_margin_is_enough() -> None:
    decision = decide(_candidate(3, _minus(TUNED_FOLDS, MIN_MAE_GAIN)), _candidate(2, TUNED_FOLDS))

    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain == MIN_MAE_GAIN
    assert decision.promoted


def test_a_large_gain_with_one_fold_higher_is_rejected() -> None:
    folds = list(_minus(TUNED_FOLDS, 1.0))
    folds[2] = TUNED_FOLDS[2] + 0.01

    decision = decide(_candidate(3, folds), _candidate(2, TUNED_FOLDS))

    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain > 0.7
    assert decision.against_champion.worse_folds == [3]
    assert not decision.promoted
    assert decision.reason.endswith("higher in fold 3.")


def test_a_fold_with_the_same_score_is_not_higher() -> None:
    folds = list(_minus(TUNED_FOLDS, 1.0))
    folds[0] = TUNED_FOLDS[0]

    decision = decide(_candidate(3, folds), _candidate(2, TUNED_FOLDS))

    assert decision.promoted


def test_the_margin_can_be_given() -> None:
    decision = decide(_candidate(2, TUNED_FOLDS), _candidate(1, UNTUNED_FOLDS), min_mae_gain=1.0)

    assert decision.min_mae_gain == 1.0
    assert not decision.promoted


def test_compare_counts_a_higher_mae_as_a_negative_gain() -> None:
    comparison = compare(_scores([10.0, 12.0]), _scores([11.0, 10.5]), "x", 0.2)

    assert comparison.mae_gain == -0.25
    assert comparison.worse_folds == [2]
    assert not comparison.beaten


# --------------------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("data_version", "0000.dir"),
        ("data.train_end_date", "2025-05-31"),
        ("features.lags", "[48, 168, 336]"),
        ("backtest.n_folds", "5"),
        ("baseline", "daily_naive"),
    ],
)
def test_versions_scored_on_another_backtest_are_not_compared(key: str, value: str) -> None:
    champion = _candidate(1, UNTUNED_FOLDS)
    challenger = _candidate(2, TUNED_FOLDS, **{key: value})

    with pytest.raises(IncomparableRunsError, match=f"{key} is .* for version 2.*set-champion 2"):
        decide(challenger, champion)


def test_an_unknown_data_version_of_the_champion_is_never_the_same() -> None:
    champion = _candidate(1, UNTUNED_FOLDS, data_version=UNKNOWN)

    with pytest.raises(IncomparableRunsError, match="data_version is unknown for the champion"):
        decide(_candidate(2, TUNED_FOLDS), champion)


def test_a_baseline_that_scored_differently_shows_the_rows_were_not_the_same() -> None:
    champion = _candidate(1, UNTUNED_FOLDS, baseline_folds=_minus(BASELINE_FOLDS, 0.5))

    with pytest.raises(IncomparableRunsError, match=r"baseline scored differently.*dvc repro"):
        decide(_candidate(2, TUNED_FOLDS), champion)


@pytest.mark.parametrize(
    ("facts", "problem"),
    [
        ({"git_dirty": "true"}, "files differed from commit 5d1d9e4 when it was trained"),
        ({"git_dirty": UNKNOWN}, r"\(git_dirty is unknown\)"),
        ({"git_commit": UNKNOWN}, "its git commit is unknown"),
        ({"data_version": UNKNOWN}, "its data version is unknown"),
    ],
)
def test_a_version_that_cannot_be_rebuilt_is_not_promoted(
    facts: dict[str, str], problem: str
) -> None:
    challenger = _candidate(2, TUNED_FOLDS, **facts)

    with pytest.raises(PipelineError, match=f"Version 2 cannot be promoted: .*{problem}"):
        decide(challenger, None)


def test_decision_tags_record_the_outcome_the_reason_the_time_and_the_champion_before() -> None:
    moment = datetime(2026, 10, 17, 8, 30, tzinfo=UTC)

    assert decision_tags("rejected", "Rejected: worse.", 2, moment) == {
        "promotion.decision": "rejected",
        "promotion.reason": "Rejected: worse.",
        "promotion.decided_at": "2026-10-17T08:30:00+00:00",
        "promotion.champion_before": "2",
    }
    assert decision_tags("promoted", "", None, moment)["promotion.champion_before"] == "none"


# --------------------------------------------------------------------------------------
# In a store
# --------------------------------------------------------------------------------------


def _report(store: Store) -> PromotionReport:
    path = promotion_report_path(store.settings.paths.reports_dir)
    return PromotionReport.model_validate_json(path.read_text(encoding="utf-8"))


def test_the_first_promotion_sets_the_champion_records_it_and_writes_the_report(
    store: Store,
) -> None:
    store.register(UNTUNED_FOLDS)

    decision = promote(store.settings)

    assert decision is not None
    assert decision.promoted
    assert store.aliases() == {"champion": 1}
    tags = store.tags(1)
    assert tags["promotion.decision"] == "promoted"
    assert tags["promotion.champion_before"] == "none"
    assert tags["promotion.reason"] == decision.reason
    assert datetime.fromisoformat(tags["promotion.decided_at"]).tzinfo is not None
    report = _report(store)
    assert report.registered_model == "taxi-demand"
    assert report.challenger_version == 1
    assert report.against_baseline.challenger_mae == 24.929918
    assert report.decided_at.isoformat() == tags["promotion.decided_at"]


def test_a_better_version_takes_over_and_the_old_record_stays(store: Store) -> None:
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    first_record = store.tags(1)
    store.register(TUNED_FOLDS)

    decision = promote(store.settings)

    assert decision is not None
    assert decision.promoted
    assert store.aliases() == {"champion": 2}
    assert store.version_aliases(1) == []
    assert store.tags(1) == first_record
    assert store.tags(2)["promotion.champion_before"] == "1"
    assert "a gain of 0.823893" in store.tags(2)["promotion.reason"]


def test_a_rejected_version_becomes_challenger_and_the_champion_stays(store: Store) -> None:
    store.register(TUNED_FOLDS)
    promote(store.settings)
    store.register(_minus(TUNED_FOLDS, 0.1))

    decision = promote(store.settings)

    assert decision is not None
    assert not decision.promoted
    assert store.aliases() == {"champion": 1, "challenger": 2}
    assert store.version_aliases(2) == ["challenger"]
    assert store.tags(2)["promotion.decision"] == "rejected"
    assert store.tags(2)["promotion.champion_before"] == "1"
    assert not _report(store).promoted


def test_the_challenger_alias_moves_to_the_newest_rejected_version(store: Store) -> None:
    store.register(TUNED_FOLDS)
    promote(store.settings)
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(TUNED_FOLDS)

    promote(store.settings)

    assert store.aliases() == {"champion": 1, "challenger": 3}
    assert store.version_aliases(2) == []
    assert store.tags(2)["promotion.decision"] == "rejected"
    assert store.tags(3)["promotion.decision"] == "rejected"


def test_a_promoted_version_is_no_longer_challenger_and_the_old_challenger_loses_it(
    store: Store,
) -> None:
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(TUNED_FOLDS)

    promote(store.settings)

    assert store.aliases() == {"champion": 3}


def test_a_promoted_version_is_not_judged_again(store: Store) -> None:
    store.register(TUNED_FOLDS)
    promote(store.settings)
    record = store.tags(1)
    report = promotion_report_path(store.settings.paths.reports_dir)
    report.unlink()

    assert promote(store.settings) is None

    assert store.aliases() == {"champion": 1}
    assert store.tags(1) == record
    assert not report.exists()


def test_a_rollback_by_hand_is_not_undone_by_the_next_promote(
    store: Store, caplog: pytest.LogCaptureFixture
) -> None:
    store.register(UNTUNED_FOLDS)
    store.register(TUNED_FOLDS)
    promote(store.settings)
    set_champion(store.settings, 1, "trying a rollback")
    records = {version: store.tags(version) for version in (1, 2)}

    with caplog.at_level(logging.INFO):
        assert promote(store.settings) is None

    assert store.aliases() == {"champion": 1}
    assert {version: store.tags(version) for version in (1, 2)} == records
    (line,) = [r for r in caplog.records if r.getMessage().startswith("nothing to decide")]
    assert "judged once" in line.getMessage()
    assert (line.__dict__["version"], line.__dict__["champion"]) == (2, 1)
    assert line.__dict__["earlier_decision"] == "promoted"


def test_a_rolled_back_version_returns_only_by_hand(store: Store) -> None:
    store.register(UNTUNED_FOLDS)
    store.register(TUNED_FOLDS)
    promote(store.settings)
    set_champion(store.settings, 1, "trying a rollback")
    promote(store.settings)

    set_champion(store.settings, 2, "the rollback is over")

    assert store.aliases() == {"champion": 2}
    assert store.tags(2)["promotion.decision"] == "manual"
    assert store.tags(2)["promotion.champion_before"] == "1"


def test_a_rejected_version_is_not_judged_again_against_a_weaker_champion(
    store: Store,
) -> None:
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(TUNED_FOLDS)
    promote(store.settings)
    store.register(_minus(TUNED_FOLDS, 0.1))
    promote(store.settings)
    record = store.tags(3)
    # Against version 1 it would win by 0.92: it must not slip in after a rollback.
    set_champion(store.settings, 1, "version 2 is worse on the replay")

    assert promote(store.settings) is None

    assert store.aliases() == {"champion": 1, "challenger": 3}
    assert store.tags(3) == record


def test_a_version_set_as_champion_by_hand_is_not_judged_later(store: Store) -> None:
    store.register(UNTUNED_FOLDS)
    store.register(_minus(BASELINE_FOLDS, 0.1))
    set_champion(store.settings, 2, "a try-out")
    set_champion(store.settings, 1, "the try-out is over")

    assert promote(store.settings) is None

    assert store.aliases() == {"champion": 1}
    assert store.tags(2)["promotion.decision"] == "manual"


def test_a_champion_set_outside_the_gate_is_not_judged(store: Store) -> None:
    store.register(TUNED_FOLDS)
    store.client.set_registered_model_alias("taxi-demand", "champion", "1")

    assert promote(store.settings) is None

    assert store.tags(1) == {}


def test_a_refused_version_carries_no_decision_and_is_judged_later(store: Store) -> None:
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(TUNED_FOLDS, tags=RUN_TAGS | {"data_version": "0000.dir"})
    with pytest.raises(IncomparableRunsError):
        promote(store.settings)
    set_champion(store.settings, 2, "compared by hand on the new data")
    store.register(_minus(TUNED_FOLDS, 0.3), tags=RUN_TAGS | {"data_version": "0000.dir"})

    decision = promote(store.settings)

    assert decision is not None
    assert decision.promoted
    assert store.aliases() == {"champion": 3}


def test_an_incomparable_version_changes_nothing(store: Store) -> None:
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(TUNED_FOLDS, tags=RUN_TAGS | {"data_version": "0000.dir"})

    with pytest.raises(IncomparableRunsError, match="data_version is 7bac"):
        promote(store.settings)

    assert store.aliases() == {"champion": 1}
    assert store.tags(2) == {}


def test_a_version_that_cannot_be_rebuilt_changes_nothing(store: Store) -> None:
    store.register(TUNED_FOLDS, tags=RUN_TAGS | {"git_dirty": "true"})

    with pytest.raises(PipelineError, match="Commit the code, train again with `dvc repro`"):
        promote(store.settings)

    assert store.aliases() == {}
    assert store.tags(1) == {}
    assert not promotion_report_path(store.settings.paths.reports_dir).exists()


def test_the_stage_function_promotes(store: Store) -> None:
    store.register(TUNED_FOLDS)

    run_promotion(store.settings)

    assert store.aliases() == {"champion": 1}


def test_the_report_is_json_with_the_decision_first(store: Store) -> None:
    store.register(TUNED_FOLDS)
    promote(store.settings)

    path = promotion_report_path(store.settings.paths.reports_dir)
    stored = json.loads(path.read_text(encoding="utf-8"))

    assert list(stored) == [
        "challenger_version",
        "champion_version",
        "promoted",
        "reason",
        "min_mae_gain",
        "against_baseline",
        "against_champion",
        "registered_model",
        "decided_at",
    ]


# --------------------------------------------------------------------------------------
# By hand
# --------------------------------------------------------------------------------------


def test_a_rollback_by_hand_moves_the_champion_and_records_why(store: Store) -> None:
    store.register(UNTUNED_FOLDS)
    promote(store.settings)
    store.register(TUNED_FOLDS)
    promote(store.settings)
    first_record = store.tags(1)

    set_champion(store.settings, 1, "  version 2 is worse on the May replay  ")

    assert store.aliases() == {"champion": 1}
    tags = store.tags(1)
    assert tags["promotion.decision"] == "manual"
    assert tags["promotion.reason"] == "Set by hand: version 2 is worse on the May replay"
    assert tags["promotion.champion_before"] == "2"
    assert tags["promotion.decided_at"] != first_record["promotion.decided_at"]
    assert store.tags(2)["promotion.decision"] == "promoted"


def test_a_challenger_set_as_champion_by_hand_is_no_longer_challenger(store: Store) -> None:
    store.register(TUNED_FOLDS)
    promote(store.settings)
    store.register(UNTUNED_FOLDS)
    promote(store.settings)

    set_champion(store.settings, 2, "chosen by hand")

    assert store.aliases() == {"champion": 2}


def test_a_champion_set_by_hand_need_not_be_reproducible(store: Store) -> None:
    store.register(TUNED_FOLDS, tags=RUN_TAGS | {"git_dirty": "true"})

    set_champion(store.settings, 1, "a try-out")

    assert store.aliases() == {"champion": 1}


def test_a_champion_is_not_set_by_hand_without_a_reason(store: Store) -> None:
    store.register(TUNED_FOLDS)

    with pytest.raises(PipelineError, match="Say why the champion is set by hand: --reason"):
        set_champion(store.settings, 1, "   ")

    assert store.aliases() == {}


def test_a_version_that_does_not_exist_cannot_be_set_by_hand(store: Store) -> None:
    store.register(TUNED_FOLDS)

    with pytest.raises(PipelineError, match=r"has no version 5\. Its versions are \[1\]"):
        set_champion(store.settings, 5, "rollback")

    assert store.aliases() == {}


def test_setting_the_champion_it_already_is_changes_nothing(store: Store) -> None:
    store.register(TUNED_FOLDS)
    promote(store.settings)
    record = store.tags(1)

    set_champion(store.settings, 1, "again")

    assert store.tags(1) == record
