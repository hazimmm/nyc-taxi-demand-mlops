"""The batch forecast of the replay, step by step, on the tiny snapshot and fake models."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from demand.config import Settings
from demand.data.gold import gold_demand_path
from demand.data.silver import silver_weather_path, write_parquet
from demand.flows import forecast_flow
from demand.flows.clock import day_file, forecasts_dir, inputs_dir, scored_dir, stored_days
from demand.flows.forecast_flow import (
    ForecastError,
    issue_forecast,
    registry_backtest_mae,
    run_forecast,
    run_step,
)
from demand.flows.scoring import ScoringError, replay_report_path
from demand.models.registry import StoredVersion
from demand.schemas import (
    ACTUAL,
    FORECAST,
    GENERATED_AT,
    HOUR_START,
    MODEL_VERSION,
    TARGET,
    ZONE_ID,
    feature_columns,
    predictions_schema,
)
from demand.serving.app import Service, forecast_day
from demand.serving.model_loader import LoadedModel, Predictor
from demand.serving.snapshot import DataSnapshot, day_hours, issue_moment
from tests.conftest import TEST_ZONES
from tests.unit.flows.conftest import (
    BACKTEST_MAE,
    FIRST_REPLAY_DAY,
    LAST_DAY,
    REPLAY_DAYS,
    ShiftModel,
    SumModel,
    fixed_backtest_mae,
    loaded,
    snapshot_of,
)
from tests.unit.serving.conftest import tables


def step(
    settings: Settings, snapshot: DataSnapshot, predictor: Predictor | None = None
) -> date | None:
    """Run one step with ``predictor`` (a new ``SumModel`` by default); the day forecast."""
    model = loaded(settings, predictor or SumModel(settings))
    done, _ = run_step(settings, snapshot, load=lambda _: model, backtest_mae=fixed_backtest_mae)
    return done.day


def files(settings: Settings) -> dict[str, bytes]:
    """Every file the replay wrote, by path, with its bytes."""
    found = {
        path.relative_to(settings.paths.predictions).as_posix(): path.read_bytes()
        for path in settings.paths.predictions.rglob("*")
        if path.is_file()
    }
    found["report"] = replay_report_path(settings).read_bytes()
    return found


def read(folder: Path, day: date) -> pd.DataFrame:
    return pd.read_parquet(day_file(folder, day))


def test_the_whole_replay_forecasts_every_day_once_and_scores_them_all(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    days = [step(replay_settings, snapshot) for _ in range(5)]

    assert days == [*REPLAY_DAYS, None]
    for folder in (forecasts_dir, inputs_dir, scored_dir):
        assert stored_days(folder(replay_settings)) == REPLAY_DAYS
    report = json.loads(replay_report_path(replay_settings).read_text(encoding="utf-8"))
    assert report["days_scored"] == 4
    assert report["clock"] == "2025-03-13T00:00:00-04:00"
    assert report["versions"][0]["model_version"] == "7"
    assert report["versions"][0]["backtest_mae"] == BACKTEST_MAE


def test_a_step_after_the_end_writes_the_same_files_again(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    for _ in range(5):
        step(replay_settings, snapshot)
    before = files(replay_settings)

    assert step(replay_settings, snapshot) is None

    assert files(replay_settings) == before


def test_a_forecast_follows_the_schema_and_is_dated_at_its_issue_moment(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step(replay_settings, snapshot)

    forecast = read(forecasts_dir(replay_settings), FIRST_REPLAY_DAY)
    predictions_schema(replay_settings).validate(forecast)
    # 9 March has 23 hours (the clocks go forward); every configured zone is forecast.
    assert len(forecast) == 23 * len(TEST_ZONES)
    assert set(forecast[ZONE_ID]) == set(TEST_ZONES)
    assert forecast[ACTUAL].isna().all()
    assert (forecast[MODEL_VERSION] == "7").all()
    assert (forecast[GENERATED_AT] == issue_moment(FIRST_REPLAY_DAY, replay_settings)).all()


def test_the_replay_forecasts_what_the_api_answers(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    model = SumModel(replay_settings)
    step(replay_settings, snapshot, model)

    answer = forecast_day(
        Service(model=loaded(replay_settings, model), snapshot=snapshot),
        replay_settings.data.zones,
        FIRST_REPLAY_DAY,
        replay_settings,
    )

    forecast = read(forecasts_dir(replay_settings), FIRST_REPLAY_DAY)
    assert forecast[FORECAST].tolist() == [hour.forecast for hour in answer.forecasts]
    assert forecast[HOUR_START].tolist() == [hour.hour_start for hour in answer.forecasts]


def test_the_inputs_are_the_rows_the_model_was_given(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step(replay_settings, snapshot)

    inputs = read(inputs_dir(replay_settings), FIRST_REPLAY_DAY)

    assert list(inputs.columns) == [ZONE_ID, HOUR_START, *feature_columns(replay_settings)[1:]]
    assert (
        inputs[HOUR_START].tolist()
        == read(forecasts_dir(replay_settings), FIRST_REPLAY_DAY)[HOUR_START].tolist()
    )


# --- no leakage ---------------------------------------------------------------------------


def changed(settings: Settings, chosen: Callable[[pd.Series], pd.Series]) -> DataSnapshot:
    """The snapshot with 1000 more pickups in the hours that ``chosen`` picks."""
    demand, _ = tables(settings, seed=5)
    rows = chosen(demand[HOUR_START])
    demand.loc[rows, TARGET] = demand.loc[rows, TARGET] + 1000
    return snapshot_of(settings, demand)


def elsewhere(settings: Settings, folder: Path) -> Settings:
    """The same settings with their data in another folder, for a second replay."""
    return settings.model_copy(
        update={"paths": settings.paths.model_copy(update={"data_dir": folder})}
    )


def test_pickups_after_the_issue_moment_do_not_change_a_forecast(
    replay_settings: Settings, snapshot: DataSnapshot, tmp_path: Path
) -> None:
    step(replay_settings, snapshot)
    original = read(forecasts_dir(replay_settings), FIRST_REPLAY_DAY)
    moment = issue_moment(FIRST_REPLAY_DAY, replay_settings)
    other = elsewhere(replay_settings, tmp_path / "other")

    step(other, changed(other, lambda hours: hours >= moment))

    pd.testing.assert_frame_equal(read(forecasts_dir(other), FIRST_REPLAY_DAY), original)


def test_the_newest_hour_a_forecast_reads_does_change_it(
    replay_settings: Settings, snapshot: DataSnapshot, tmp_path: Path
) -> None:
    # The control of the test above. The last hour of 9 March reads the pickups 48 hours
    # before it: 22:00 on 7 March, because the clocks go forward in between.
    step(replay_settings, snapshot)
    original = read(forecasts_dir(replay_settings), FIRST_REPLAY_DAY)
    newest = day_hours(FIRST_REPLAY_DAY, replay_settings)[-1] - pd.Timedelta(hours=48)
    assert newest == pd.Timestamp("2025-03-07 22:00", tz="America/New_York")
    other = elsewhere(replay_settings, tmp_path / "other")

    step(other, changed(other, lambda hours: hours == newest))

    assert not read(forecasts_dir(other), FIRST_REPLAY_DAY)[FORECAST].equals(original[FORECAST])


# --- actuals --------------------------------------------------------------------------------


def test_a_day_is_scored_only_once_it_is_over(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step(replay_settings, snapshot)  # 9 March, issued at 00:00 on 8 March
    step(replay_settings, snapshot)  # 10 March, issued at 00:00 on 9 March
    assert stored_days(scored_dir(replay_settings)) == []

    step(replay_settings, snapshot)  # 11 March, issued at 00:00 on 10 March: 9 March is over

    assert stored_days(scored_dir(replay_settings)) == [FIRST_REPLAY_DAY]


def test_scoring_fills_the_actual_and_never_touches_the_forecast(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step(replay_settings, snapshot)
    issued = day_file(forecasts_dir(replay_settings), FIRST_REPLAY_DAY).read_bytes()

    for _ in range(4):
        step(replay_settings, snapshot)

    assert day_file(forecasts_dir(replay_settings), FIRST_REPLAY_DAY).read_bytes() == issued
    forecast = read(forecasts_dir(replay_settings), FIRST_REPLAY_DAY)
    scored = read(scored_dir(replay_settings), FIRST_REPLAY_DAY)
    pd.testing.assert_frame_equal(scored.drop(columns=ACTUAL), forecast.drop(columns=ACTUAL))
    counted = snapshot.history.set_index([ZONE_ID, HOUR_START])[TARGET]
    keys = pd.MultiIndex.from_frame(scored[[ZONE_ID, HOUR_START]])
    assert scored[ACTUAL].tolist() == counted.loc[keys].astype("float64").tolist()


def test_the_baseline_is_scored_on_the_same_rows_from_the_stored_inputs(
    replay_settings: Settings, snapshot: DataSnapshot, tmp_path: Path
) -> None:
    # A model that forecasts lag_168h itself scores exactly as the baseline, day by day.
    for _ in range(5):
        step(replay_settings, snapshot, ShiftModel(0.0))
    report = json.loads(replay_report_path(replay_settings).read_text(encoding="utf-8"))
    assert report["model"] == report["baseline_score"]
    assert all(day["model"] == day["baseline"] for day in report["days"])
    assert report["days_model_beat_baseline"] == 0

    # Another model leaves the baseline as it was: it comes from the inputs, not the model.
    other = elsewhere(replay_settings, tmp_path / "other")
    for _ in range(5):
        step(other, snapshot, ShiftModel(3.0))
    shifted = json.loads(replay_report_path(other).read_text(encoding="utf-8"))
    assert shifted["baseline_score"] == report["baseline_score"]
    assert shifted["model"] != report["model"]


def test_an_issued_forecast_is_never_computed_again(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step(replay_settings, snapshot)
    before = files(replay_settings)
    model = SumModel(replay_settings)

    written = issue_forecast(
        FIRST_REPLAY_DAY, loaded(replay_settings, model), snapshot, replay_settings
    )

    assert written is False
    assert model.calls == 0
    assert files(replay_settings) == before


def test_a_step_cut_off_before_its_forecast_is_written_is_done_again_the_same(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step(replay_settings, snapshot)
    written = files(replay_settings)
    # The inputs are written first; a crash before the forecast leaves only them.
    day_file(forecasts_dir(replay_settings), FIRST_REPLAY_DAY).unlink()

    assert step(replay_settings, snapshot) == FIRST_REPLAY_DAY

    assert files(replay_settings) == written


def test_a_forecast_that_breaks_the_schema_is_not_written(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    with pytest.raises(ForecastError, match="does not pass predictions_schema"):
        step(replay_settings, snapshot, ShiftModel(-10_000.0))

    assert stored_days(forecasts_dir(replay_settings)) == []
    assert stored_days(inputs_dir(replay_settings)) == []


def test_inputs_without_the_weekly_lag_cannot_be_scored(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    for _ in range(3):
        step(replay_settings, snapshot)
    path = day_file(inputs_dir(replay_settings), FIRST_REPLAY_DAY)
    write_parquet(pd.read_parquet(path).drop(columns="lag_168h"), path)

    with pytest.raises(ScoringError, match="hold no lag_168h"):
        step(replay_settings, snapshot)


# --- run_forecast ----------------------------------------------------------------------------


def write_snapshot(settings: Settings) -> None:
    demand, weather = tables(settings, seed=5)
    write_parquet(demand, gold_demand_path(settings.paths.gold))
    write_parquet(weather, silver_weather_path(settings.paths.silver))


def test_run_forecast_runs_the_steps_asked_for_and_loads_the_model_at_each(
    replay_settings: Settings,
) -> None:
    write_snapshot(replay_settings)
    loads: list[Settings] = []

    def load(settings: Settings) -> LoadedModel:
        loads.append(settings)
        return loaded(settings, SumModel(settings))

    run_forecast(replay_settings, 2, load=load, backtest_mae=fixed_backtest_mae)

    assert stored_days(forecasts_dir(replay_settings)) == REPLAY_DAYS[:2]
    assert len(loads) == 2


def test_run_forecast_stops_at_the_end_of_the_data(replay_settings: Settings) -> None:
    write_snapshot(replay_settings)

    run_forecast(
        replay_settings,
        100,
        load=lambda settings: loaded(settings, SumModel(settings)),
        backtest_mae=fixed_backtest_mae,
    )

    assert stored_days(forecasts_dir(replay_settings))[-1] == LAST_DAY
    assert stored_days(scored_dir(replay_settings)) == REPLAY_DAYS


def test_run_forecast_refuses_less_than_one_step(replay_settings: Settings) -> None:
    with pytest.raises(ForecastError, match="--days must be 1 or more, not 0"):
        run_forecast(replay_settings, 0)


def test_the_backtest_mae_of_a_version_is_read_from_the_registry(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeRegistry:
        def __init__(self, given: Settings) -> None:
            assert given is settings

        def version(self, number: int) -> StoredVersion:
            metrics = {"lightgbm.mae": 24.1} if number == 2 else {}
            return StoredVersion(version=number, run_id="r", params={}, tags={}, metrics=metrics)

    monkeypatch.setattr(forecast_flow, "Registry", FakeRegistry)

    lookup = registry_backtest_mae(settings)

    assert lookup("2") == 24.1
    assert lookup("3") is None
