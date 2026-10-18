"""The retrain flow on hand-made replay files, a hand-made gold table and a temporary store.

The settings are the test settings (three zones, paths under ``tmp_path``, training to 30
April) with a backtest of two folds of two days, so the comparison weeks are four days.
The retrain is evaluated at 00:00 on 10 May: the cutoff is 9 May and the folds are 6-7
and 8-9 May. The champion's forecasts are ``LIVE``, every pickup is ``ACTUAL`` and the
weekly baseline is ``WEEKLY``, so every MAE is a difference of two of these numbers. The
challenger's backtest and its training are fakes; nothing reads ``data/``, ``mlflow.db``
or ``mlruns`` of the repository.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import pandas as pd
import pytest

from demand.config import Settings
from demand.data.aggregate import training_period_end
from demand.data.gold import gold_features_path
from demand.data.silver import write_parquet
from demand.flows import retrain_flow
from demand.flows.clock import Step, day_file, scored_dir
from demand.flows.retrain_flow import (
    CHALLENGER,
    CHAMPION,
    FOLD,
    NOW_TAG,
    WEEKLY,
    ComparisonError,
    Outcome,
    RetrainError,
    challenger_settings,
    comparison_days,
    comparison_scores,
    comparison_table,
    cutoff_day,
    judge,
    last_retrain,
    retrain_if_needed,
    retrain_report_path,
    run_day,
    run_retrain,
    scores_of,
    train_challenger,
    trigger,
)
from demand.models import training
from demand.models.backtest import fold_windows, load_training_features
from demand.models.promotion import DECISION_TAG, Scores
from demand.models.registry import Registry
from demand.models.training import GitState
from demand.monitoring.checks import (
    BASELINE,
    DAY,
    AccuracyCheck,
    MonitoringReport,
    input_drift_check,
    monitoring_report,
    prediction_drift_check,
)
from demand.monitoring.reference import ReferenceForecasts
from demand.schemas import (
    ACTUAL,
    FORECAST,
    HOUR_START,
    IS_OUTLIER,
    MODEL_VERSION,
    TARGET,
    ZONE_ID,
    feature_columns,
)
from demand.serving.snapshot import DataSnapshot
from tests.unit.models.conftest import RUN_PARAMS, Store, mae_metrics, with_backtest
from tests.unit.monitoring.conftest import days_from, grid, moment, write_replay

ACTUAL_PICKUPS = 100.0
LIVE = 110.0
"""The champion's forecast of every hour: an MAE of 10."""
WEEKLY_VALUE = 112.0
"""``lag_168h`` of every hour: an MAE of 12 for the baseline."""
NOW_DAY = date(2025, 5, 10)
REPLAY = days_from(date(2025, 5, 2), 10)
"""Forecast days 2 to 11 May; the forecast for 11 May is issued at 00:00 on 10 May."""
SCORED = days_from(date(2025, 5, 2), 8)
"""2 to 9 May: the days over by 00:00 on 10 May."""
LAST_DAY = date(2025, 6, 30)
PARAMS = RUN_PARAMS | {
    "data.train_end_date": "2025-05-09",
    "backtest.n_folds": "2",
    "backtest.fold_days": "2",
}


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def two_folds(settings: Settings) -> Settings:
    return with_backtest(settings, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)


def now_at(day: date, settings: Settings) -> pd.Timestamp:
    return moment(day, settings)


def gold_table(settings: Settings, first: date, last: date) -> pd.DataFrame:
    """A gold feature table of ``first`` to ``last``: the pickups and the baseline known."""
    table = grid(settings, days_from(first, (last - first).days + 1)).drop(columns=DAY)
    for column in feature_columns(settings):
        if column not in table.columns:
            table[column] = 0.0
    table[TARGET] = ACTUAL_PICKUPS
    table[WEEKLY] = WEEKLY_VALUE
    table[IS_OUTLIER] = False
    return table


def write_gold(settings: Settings) -> pd.DataFrame:
    """Gold to 11 May, two days past the cutoff: rows the challenger must not read."""
    table = gold_table(settings, date(2025, 4, 20), date(2025, 5, 11))
    write_parquet(table, gold_features_path(settings.paths.gold))
    return table


def write_days(settings: Settings, version: str = "1", days: list[date] = SCORED) -> None:
    write_replay(
        settings,
        REPLAY,
        scored=days,
        version=version,
        forecast=LIVE,
        actual=ACTUAL_PICKUPS,
        baseline=WEEKLY_VALUE,
    )


def accuracy(*, alert: bool, version: str = "1", note: str | None = None) -> AccuracyCheck:
    return AccuracyCheck(
        evaluated=note is None,
        note=note,
        model_version=version,
        first_day=date(2025, 5, 3),
        last_day=date(2025, 5, 9),
        days=7,
        rows=504,
        live_mae=30.0 if alert else 20.0,
        baseline_mae=32.0,
        backtest_mae=20.0,
        live_to_backtest_mae=1.5 if alert else 1.0,
        ratio_threshold=1.25,
        alert=alert,
    )


def report(
    settings: Settings, *, alert: bool, drift: bool = False, version: str = "1"
) -> MonitoringReport:
    """A monitoring report as of 10 May; with ``drift``, both drift checks drifted."""
    inputs, _ = input_drift_check(None, grid(settings, [date(2025, 4, 30)]), settings, "none")
    predictions = prediction_drift_check(None, None, None, None, "none")
    if drift:
        inputs = inputs.model_copy(update={"drifted": True, "drift_share": 0.9})
        predictions = predictions.model_copy(
            update={"drifted": True, "distance": 0.5, "model_version": version}
        )
    return monitoring_report(
        now_at(NOW_DAY, settings),
        settings,
        inputs,
        predictions,
        accuracy(alert=alert, version=version),
    )


def snapshot() -> DataSnapshot:
    return DataSnapshot(
        history=pd.DataFrame(),
        weather=pd.DataFrame(),
        first_day=date(2025, 1, 10),
        last_day=LAST_DAY,
    )


class Folds:
    """A fake backtest: forecasts ``ACTUAL + error`` in each fold, and remembers its input."""

    def __init__(self, errors: Mapping[int, float]) -> None:
        self.errors = errors
        self.given: list[tuple[pd.DataFrame, Settings]] = []

    def __call__(self, table: pd.DataFrame, settings: Settings) -> ReferenceForecasts:
        self.given.append((table, settings))
        parts = []
        for fold in fold_windows(settings):
            hours = table[HOUR_START]
            rows = table[(hours >= fold.first_test_hour) & (hours <= fold.last_test_hour)]
            parts.append(
                rows[[ZONE_ID, HOUR_START]].assign(**{FORECAST: 100 + self.errors[fold.number]})
            )
        return ReferenceForecasts(rows=pd.concat(parts, ignore_index=True), mae=0.0)


class Train:
    """A fake training: registers a version that recorded the given backtest scores."""

    def __init__(self, store: Store, folds: tuple[float, ...], baseline: float = 12.0) -> None:
        self.store = store
        self.folds = folds
        self.baseline = baseline
        self.given: list[Settings] = []

    def __call__(self, settings: Settings) -> int:
        self.given.append(settings)
        metrics = mae_metrics(self.folds, (self.baseline,) * len(self.folds))
        return self.store.register(metrics=metrics, params=PARAMS)


@pytest.fixture(scope="session")
def blank_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("retrain_mlflow_store")
    uri = f"sqlite:///{(folder / 'mlflow.db').as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.create_experiment("taxi-demand", artifact_location=(folder / "mlruns").as_uri())
    return folder / "mlflow.db"


@pytest.fixture
def store(
    settings: Settings, tmp_path: Path, blank_store: Path, monkeypatch: pytest.MonkeyPatch
) -> Store:
    """A store whose version 1 is champion, in ``tmp_path``, the working directory."""
    shutil.copy(blank_store, tmp_path / "mlflow.db")
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow_config = settings.mlflow.model_copy(update={"tracking_uri": uri})
    created = Store(two_folds(settings).model_copy(update={"mlflow": mlflow_config}))
    created.register()
    created.client.set_registered_model_alias(created.name, "champion", "1")
    return created


@pytest.fixture
def clean_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The working directory counts as a clean commit with a known data version."""
    monkeypatch.setattr(training, "git_state", lambda _: GitState(commit="abc", dirty=False))
    monkeypatch.setattr(training, "data_version", lambda *_: "md5.dir")


@pytest.fixture
def replay(store: Store) -> Settings:
    """The replay files and the gold table of the module docstring."""
    write_days(store.settings)
    write_gold(store.settings)
    return store.settings


def evaluate(
    settings: Settings,
    monitoring: MonitoringReport,
    train: Train,
    folds: Folds,
) -> retrain_flow.RetrainReport:
    return retrain_if_needed(
        settings,
        snapshot(),
        now_at(NOW_DAY, settings),
        monitoring,
        train=train,
        fold_forecasts=folds,
    )


# --------------------------------------------------------------------------------------
# The cutoff
# --------------------------------------------------------------------------------------


def test_the_challenger_learns_up_to_the_last_day_that_is_over(settings: Settings) -> None:
    now = now_at(date(2025, 5, 30), settings)

    challenger = challenger_settings(settings, now)

    assert cutoff_day(now) == date(2025, 5, 29)
    assert challenger.data.train_end_date == date(2025, 5, 29)
    assert training_period_end(challenger) == now
    assert challenger.paths.reports_dir == settings.paths.reports_dir / "retrain" / "2025-05-30"
    unchanged = ("project", "features", "model", "backtest", "mlflow", "monitoring")
    for section in unchanged:
        assert getattr(challenger, section) == getattr(settings, section)
    assert challenger.data.zones == settings.data.zones
    assert challenger.paths.data_dir == settings.paths.data_dir


def test_the_comparison_weeks_are_the_four_weeks_before_the_cutoff(settings: Settings) -> None:
    challenger = challenger_settings(settings, now_at(date(2025, 5, 30), settings))

    days = comparison_days(challenger)

    assert days == days_from(date(2025, 5, 2), 28)


def test_the_challenger_reads_no_row_after_the_cutoff_and_the_rows_of_gold_before_it(
    replay: Settings, store: Store, clean_checkout: None
) -> None:
    folds, train = Folds({1: 9.5, 2: 5.0}), Train(store, (9.5, 5.0))

    evaluate(replay, report(replay, alert=True), train, folds)

    ((given, given_settings),) = folds.given
    now = now_at(NOW_DAY, replay)
    assert given[HOUR_START].max() < now
    gold = pd.read_parquet(gold_features_path(replay.paths.gold))
    expected = gold[gold[HOUR_START] < now].reset_index(drop=True)
    pd.testing.assert_frame_equal(given.reset_index(drop=True), expected[list(given.columns)])
    assert given_settings.data.train_end_date == date(2025, 5, 9)
    (trained,) = train.given
    assert trained == given_settings
    assert load_training_features(trained)[HOUR_START].max() < now


# --------------------------------------------------------------------------------------
# The trigger
# --------------------------------------------------------------------------------------


def test_drift_alone_never_triggers_a_retrain(settings: Settings) -> None:
    monitoring = report(settings, alert=False, drift=True)

    found = trigger(monitoring, 1, now_at(NOW_DAY, settings), None, settings)

    assert monitoring.warning
    assert found is not None
    assert found[0] is Outcome.NO_ALERT


def test_an_accuracy_alert_about_the_champion_triggers_a_retrain(settings: Settings) -> None:
    found = trigger(report(settings, alert=True), 1, now_at(NOW_DAY, settings), None, settings)

    assert found is None


def test_an_alert_about_a_version_that_is_not_the_champion_does_not_trigger(
    settings: Settings,
) -> None:
    found = trigger(report(settings, alert=True), 3, now_at(NOW_DAY, settings), None, settings)

    assert found is not None
    assert found[0] is Outcome.NO_ALERT
    assert "version 1, which is not the champion (version 3)" in found[1]


def test_an_accuracy_check_that_was_not_evaluated_does_not_trigger(settings: Settings) -> None:
    monitoring = report(settings, alert=False).model_copy(
        update={"accuracy": accuracy(alert=False, note="Only 3 days are over.")}
    )

    found = trigger(monitoring, 1, now_at(NOW_DAY, settings), None, settings)

    assert found == (Outcome.NO_ALERT, "No accuracy alert: Only 3 days are over.")


@pytest.mark.parametrize(("days_ago", "blocked"), [(1, True), (6, True), (7, False), (20, False)])
def test_the_cooldown_counts_calendar_days_since_the_last_retrain(
    settings: Settings, days_ago: int, blocked: bool
) -> None:
    now = now_at(NOW_DAY, settings)
    last = now - pd.Timedelta(days=days_ago)

    found = trigger(report(settings, alert=True), 1, now, last, settings)

    assert (found is not None and found[0] is Outcome.COOLDOWN) is blocked


def test_a_retrain_is_never_repeated_at_the_same_moment(settings: Settings) -> None:
    no_cooldown = settings.model_copy(
        update={"monitoring": settings.monitoring.model_copy(update={"retrain_cooldown_days": 0})}
    )
    now = now_at(NOW_DAY, settings)

    found = trigger(report(settings, alert=True), 1, now, now, no_cooldown)

    assert found is not None
    assert found[0] is Outcome.COOLDOWN


# --------------------------------------------------------------------------------------
# The comparison
# --------------------------------------------------------------------------------------


def comparison_inputs(store: Store) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Settings]:
    settings = store.settings
    write_days(settings)
    challenger = challenger_settings(settings, now_at(NOW_DAY, settings))
    training_rows = gold_table(settings, date(2025, 4, 20), date(2025, 5, 9))
    champion = retrain_flow.scored_rows(settings, comparison_days(challenger))
    backtest = Folds({1: 9.5, 2: 5.0})(training_rows, challenger).rows
    return training_rows, champion, backtest, challenger


def test_the_three_are_scored_on_the_same_rows_by_fold(store: Store) -> None:
    training_rows, champion, backtest, challenger = comparison_inputs(store)

    table = comparison_table(training_rows, champion, backtest, challenger)
    scores = comparison_scores(table)

    assert len(table) == 4 * 24 * 3
    assert table.groupby(FOLD).size().to_dict() == {1: 144, 2: 144}
    assert (scores.first_day, scores.last_day) == (date(2025, 5, 6), date(2025, 5, 9))
    assert scores.champion == Scores(mae=10.0, fold_mae={1: 10.0, 2: 10.0})
    assert scores.challenger == Scores(mae=7.25, fold_mae={1: 9.5, 2: 5.0})
    assert scores.baseline == Scores(mae=12.0, fold_mae={1: 12.0, 2: 12.0})
    assert scores_of(table, CHAMPION) == scores.champion
    assert set(table.columns) >= {CHAMPION, CHALLENGER, WEEKLY, TARGET, ACTUAL, BASELINE}


def test_a_zone_hour_missing_from_one_side_stops_the_comparison(store: Store) -> None:
    training_rows, champion, backtest, challenger = comparison_inputs(store)

    with pytest.raises(ComparisonError, match="The challenger's backtest and gold do not cover"):
        comparison_table(training_rows, champion, backtest.iloc[1:], challenger)
    with pytest.raises(ComparisonError, match="The champion's scored forecasts and gold"):
        comparison_table(
            training_rows, pd.concat([champion, champion.iloc[:1]]), backtest, challenger
        )


def test_pickups_that_differ_from_gold_stop_the_comparison(store: Store) -> None:
    training_rows, champion, backtest, challenger = comparison_inputs(store)

    with pytest.raises(ComparisonError, match="actual differs from gold's pickups"):
        comparison_table(training_rows, champion.assign(**{ACTUAL: 99.0}), backtest, challenger)
    with pytest.raises(ComparisonError, match="baseline differs from gold's lag_168h"):
        comparison_table(training_rows, champion.assign(**{BASELINE: 1.0}), backtest, challenger)


def scores(champion: float, challenger: tuple[float, float], baseline: float = 12.0) -> Any:
    def of(folds: tuple[float, float]) -> Scores:
        return Scores(mae=round(sum(folds) / 2, 6), fold_mae={1: folds[0], 2: folds[1]})

    return retrain_flow.ComparisonScores(
        first_day=date(2025, 5, 6),
        last_day=date(2025, 5, 9),
        rows=288,
        champion=of((champion, champion)),
        challenger=of(challenger),
        baseline=of((baseline, baseline)),
    )


def test_against_the_champion_a_higher_fold_rejects_whatever_the_pooled_gain() -> None:
    decision = judge(2, 1, scores(10.0, (10.5, 2.0)))

    assert not decision.promoted
    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain == pytest.approx(3.75)
    assert decision.against_champion.worse_folds == [1]
    assert "the live forecasts of the champion (version 1)" in decision.reason
    assert "higher in fold 1" in decision.reason


def test_against_the_baseline_a_higher_fold_still_rejects() -> None:
    decision = judge(2, 1, scores(30.0, (12.5, 2.0)))

    assert not decision.promoted
    assert decision.against_baseline.worse_folds == [1]


def test_a_gain_on_the_champion_inside_the_margin_is_rejected() -> None:
    decision = judge(2, 1, scores(10.0, (9.9, 9.9)))

    assert not decision.promoted
    assert decision.against_champion is not None
    assert decision.against_champion.mae_gain == pytest.approx(0.1)


# --------------------------------------------------------------------------------------
# Registry and disk
# --------------------------------------------------------------------------------------


def test_the_last_retrain_is_read_from_the_tags_of_the_versions(store: Store) -> None:
    registry = Registry(store.settings)
    assert last_retrain(registry) is None

    for moment_text in ("2025-05-30T00:00:00-04:00", "2025-06-08T00:00:00-04:00"):
        version = store.register()
        registry.tag(version, {NOW_TAG: moment_text})
    store.register()

    assert last_retrain(Registry(store.settings)) == pd.Timestamp("2025-06-08T00:00:00-04:00")


def test_a_retrain_that_beats_both_promotes_and_records_everything(
    replay: Settings, store: Store, clean_checkout: None
) -> None:
    folds, train = Folds({1: 9.5, 2: 3.0}), Train(store, (9.5, 3.0))

    done = evaluate(replay, report(replay, alert=True), train, folds)

    assert done.outcome is Outcome.PROMOTED
    assert done.challenger_version == 2
    assert done.cutoff_day == date(2025, 5, 9)
    assert done.scores is not None
    assert (done.scores.champion.mae, done.scores.challenger.mae) == (10.0, 6.25)
    assert store.aliases() == {"champion": 2}
    tags = store.tags(2)
    assert tags[DECISION_TAG] == "promoted"
    assert tags["promotion.champion_before"] == "1"
    assert tags[NOW_TAG] == "2025-05-10T00:00:00-04:00"
    assert tags["retrain.cutoff"] == "2025-05-09"
    assert "1.5 times the backtest MAE 20.0" in tags["retrain.trigger"]
    assert "288 zone-hours of 2025-05-06 to 2025-05-09" in tags["retrain.comparison"]
    assert "6.25 for the retrained recipe" in tags["retrain.comparison"]
    stored = json.loads(retrain_report_path(replay).read_text(encoding="utf-8"))
    assert stored["outcome"] == "promoted"
    assert stored["decision"]["against_champion"]["worse_folds"] == []


def test_a_retrain_that_does_not_beat_the_champion_is_challenger(
    replay: Settings, store: Store, clean_checkout: None
) -> None:
    done = evaluate(
        replay, report(replay, alert=True), Train(store, (9.9, 9.9)), Folds({1: 9.9, 2: 9.9})
    )

    assert done.outcome is Outcome.REJECTED
    assert store.aliases() == {"champion": 1, "challenger": 2}
    assert store.tags(2)[DECISION_TAG] == "rejected"


def test_the_decision_is_recorded_once_and_the_cooldown_comes_from_the_registry(
    replay: Settings, store: Store, clean_checkout: None
) -> None:
    train = Train(store, (9.9, 9.9))
    evaluate(replay, report(replay, alert=True), train, Folds({1: 9.9, 2: 9.9}))
    decided = store.tags(2)

    again = evaluate(replay, report(replay, alert=True), train, Folds({1: 9.9, 2: 9.9}))

    assert again.outcome is Outcome.COOLDOWN
    assert again.last_retrain == pd.Timestamp("2025-05-10T00:00:00-04:00").to_pydatetime()
    assert len(train.given) == 1
    assert Registry(store.settings).versions() == [1, 2]
    assert store.tags(2) == decided


def test_without_an_alert_nothing_is_trained_and_the_report_says_why(
    replay: Settings, store: Store
) -> None:
    train = Train(store, (9.9, 9.9))

    done = evaluate(replay, report(replay, alert=False, drift=True), train, Folds({}))

    assert done.outcome is Outcome.NO_ALERT
    assert train.given == []
    assert json.loads(retrain_report_path(replay).read_text(encoding="utf-8"))["outcome"] == (
        "no_alert"
    )


def test_no_retrain_once_no_day_is_left_to_forecast(store: Store) -> None:
    write_days(store.settings)
    ends = DataSnapshot(
        history=pd.DataFrame(), weather=pd.DataFrame(), first_day=REPLAY[0], last_day=REPLAY[-1]
    )

    done = retrain_if_needed(
        store.settings,
        ends,
        now_at(NOW_DAY, store.settings),
        report(store.settings, alert=True),
        train=Train(store, ()),
        fold_forecasts=Folds({}),
    )

    assert done.outcome is Outcome.END_OF_DATA


def test_the_retrain_waits_until_the_champion_forecast_every_comparison_day(
    store: Store,
) -> None:
    write_days(store.settings, days=SCORED[:-1])

    done = evaluate(store.settings, report(store.settings, alert=True), Train(store, ()), Folds({}))

    assert done.outcome is Outcome.WAITING
    assert "1 of these 4 days have none by now (the first is 2025-05-09)" in done.reason


def test_the_retrain_waits_while_a_comparison_day_was_forecast_by_another_version(
    store: Store,
) -> None:
    write_days(store.settings)
    path = day_file(scored_dir(store.settings), date(2025, 5, 7))
    write_parquet(pd.read_parquet(path).assign(**{MODEL_VERSION: "9"}), path)

    done = evaluate(store.settings, report(store.settings, alert=True), Train(store, ()), Folds({}))

    assert done.outcome is Outcome.WAITING
    assert "1 of them were issued by another version (the last is 2025-05-07)" in done.reason


def test_a_checkout_the_gate_could_not_promote_from_trains_nothing(
    replay: Settings, store: Store
) -> None:
    train = Train(store, (9.9, 9.9))

    with pytest.raises(RetrainError, match="not a clean git checkout"):
        evaluate(replay, report(replay, alert=True), train, Folds({1: 9.9, 2: 9.9}))
    assert train.given == []


def test_a_run_that_recorded_other_scores_is_not_judged(
    replay: Settings, store: Store, clean_checkout: None
) -> None:
    with pytest.raises(ComparisonError, match="Version 2 was registered but not judged"):
        evaluate(
            replay, report(replay, alert=True), Train(store, (8.0, 8.0)), Folds({1: 9.9, 2: 9.9})
        )
    assert DECISION_TAG not in store.tags(2)
    assert store.aliases() == {"champion": 1}


def test_without_a_champion_there_is_nothing_to_retrain_against(store: Store) -> None:
    store.client.delete_registered_model_alias(store.name, "champion")

    with pytest.raises(RetrainError, match="No version is champion"):
        evaluate(store.settings, report(store.settings, alert=True), Train(store, ()), Folds({}))


# --------------------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------------------


def test_training_a_challenger_is_the_training_of_the_project(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[Settings, list[str]]] = []

    def fake(given: Settings, candidates: Mapping[str, object]) -> SimpleNamespace:
        calls.append((given, list(candidates)))
        return SimpleNamespace(model_version=5)

    monkeypatch.setattr(training, "train_from_gold", fake)

    assert train_challenger(settings) == 5
    assert calls == [(settings, ["weekly_naive", "lightgbm"])]


def test_the_command_refuses_before_the_first_forecast(settings: Settings) -> None:
    with pytest.raises(RetrainError, match="No forecast has been issued yet"):
        run_retrain(settings)


def test_the_command_monitors_and_decides_at_the_newest_issue_moment(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_days(settings)
    calls: list[Any] = []
    monitoring = report(settings, alert=False)
    monkeypatch.setattr(retrain_flow, "load_snapshot", lambda _: "snapshot")
    monkeypatch.setattr(
        retrain_flow,
        "run_monitoring",
        lambda given, **values: calls.append(("monitor", values["as_of"])) or monitoring,
    )
    monkeypatch.setattr(
        retrain_flow,
        "retrain_if_needed",
        lambda given, snap, now, report_, **_: calls.append(("retrain", snap, now, report_)),
    )

    run_retrain(settings)

    assert calls == [
        ("monitor", NOW_DAY),
        ("retrain", "snapshot", now_at(NOW_DAY, settings), monitoring),
    ]


def test_the_daily_step_forecasts_then_monitors_then_retrains_at_one_moment(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = now_at(NOW_DAY, settings)
    calls: list[Any] = []
    monitoring = report(settings, alert=False)
    monkeypatch.setattr(
        retrain_flow,
        "run_step",
        lambda *_, **__: calls.append("forecast") or (Step(now=now, day=REPLAY[-1]), "replay"),
    )
    monkeypatch.setattr(
        retrain_flow,
        "run_monitoring",
        lambda _, **values: calls.append(("monitor", values["as_of"])) or monitoring,
    )
    monkeypatch.setattr(
        retrain_flow,
        "retrain_if_needed",
        lambda _, snap, at, report_, **__: calls.append(("retrain", at)) or "retrain",
    )

    done = run_day(settings, snapshot())

    assert calls == ["forecast", ("monitor", NOW_DAY), ("retrain", now)]
    assert (done.replay, done.monitoring, done.retrain) == ("replay", monitoring, "retrain")
