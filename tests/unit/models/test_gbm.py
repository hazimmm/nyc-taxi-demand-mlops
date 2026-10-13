from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from demand.config import Settings
from demand.errors import PipelineError
from demand.models.backtest import backtest
from demand.models.gbm import (
    LIGHTGBM,
    LightGbmForecaster,
    ModelInputError,
    lightgbm_forecaster,
)
from demand.schemas import (
    DAY_OF_WEEK,
    HOUR,
    HOUR_START,
    IS_HOLIDAY,
    IS_OUTLIER,
    TARGET,
    ZONE_ID,
    feature_columns,
    lag_column,
)
from tests.conftest import with_features
from tests.unit.models.conftest import (
    FEW_TREES,
    ONE_TREE,
    feature_table,
    training_rows,
    with_backtest,
    with_levels,
    with_model_params,
)

# Two days of three zones (132, 161 and 237): 144 rows.
FIRST_HOUR = "2025-04-01 00:00"
HOURS = 48
# LightGBM stores the value of a leaf with seven digits.
LEAF = 1e-5


@pytest.fixture
def one_tree(settings: Settings) -> Settings:
    """Settings whose model is a single tree that can be checked by hand."""
    return with_model_params(settings, ONE_TREE)


@pytest.fixture
def few_trees(settings: Settings) -> Settings:
    """Settings whose model is the one of the config, with 20 trees instead of 600."""
    return with_model_params(settings, FEW_TREES)


@pytest.fixture
def table(settings: Settings) -> pd.DataFrame:
    return feature_table(settings, FIRST_HOUR, HOURS)


def _one_row_per_zone(table: pd.DataFrame) -> pd.DataFrame:
    """The first hour of the zones 132, 161 and 237, in that order."""
    return table.groupby(ZONE_ID).head(1)


def _fitted(settings: Settings, train: pd.DataFrame) -> LightGbmForecaster:
    forecaster = lightgbm_forecaster(settings)
    forecaster.fit(train)
    return forecaster


# --- what it learns ---------------------------------------------------------------------


def test_one_tree_forecasts_the_mean_pickups_of_each_zone(
    one_tree: Settings, table: pd.DataFrame
) -> None:
    train = with_levels(table, {132: 10, 161: 30, 237: 80})

    forecaster = _fitted(one_tree, train)

    assert forecaster.predict(_one_row_per_zone(train)) == pytest.approx([10, 30, 80], abs=LEAF)


def test_forecasts_are_not_rounded(one_tree: Settings, table: pd.DataFrame) -> None:
    # Zone 132 has 10 pickups in the even hours and 11 in the odd ones: 10.5 on average.
    train = with_levels(table, {132: 10, 161: 30, 237: 80})
    odd_hours_of_132 = (train[ZONE_ID] == 132) & (train[HOUR] % 2 == 1)
    train.loc[odd_hours_of_132, TARGET] = 11
    # With the hour as an input a tree could tell the two apart; without it, it cannot.
    columns = [column for column in feature_columns(one_tree) if column != HOUR]
    forecaster = LightGbmForecaster(columns, ONE_TREE | {"num_leaves": 3}, seed=42)

    forecaster.fit(train)

    assert forecaster.predict(_one_row_per_zone(train)) == pytest.approx([10.5, 30, 80], abs=LEAF)


def test_zone_is_read_as_a_category_not_as_a_number(
    one_tree: Settings, table: pd.DataFrame
) -> None:
    # One question, two leaves. The middle zone differs from the other two. "Is the
    # zone 161?" separates them; no question of the kind "is the zone below ...?" can.
    two_leaves = with_model_params(one_tree, ONE_TREE | {"num_leaves": 2})
    train = with_levels(table, {132: 10, 161: 30, 237: 10})

    forecaster = _fitted(two_leaves, train)

    assert forecaster.predict(_one_row_per_zone(train)) == pytest.approx([10, 30, 10], abs=LEAF)


def test_fit_starts_again_and_keeps_nothing_of_the_earlier_fit(
    one_tree: Settings, table: pd.DataFrame
) -> None:
    first = with_levels(table, {132: 10, 161: 30, 237: 80})
    second = with_levels(table, {132: 500, 161: 600, 237: 700})
    rows = _one_row_per_zone(table)
    forecaster = _fitted(one_tree, first)

    forecaster.fit(second)

    assert forecaster.predict(rows) == pytest.approx([500, 600, 700], abs=LEAF)
    assert forecaster.predict(rows).tolist() == _fitted(one_tree, second).predict(rows).tolist()


def test_backtest_scores_the_model_like_any_other_forecaster(one_tree: Settings) -> None:
    # Two folds of two days. Every zone has the same pickups in every hour, so one tree
    # forecasts them exactly and every error is 0.
    small = with_backtest(one_tree, train_end_date=date(2025, 4, 30), n_folds=2, fold_days=2)
    small = with_features(small, horizon_hours=48, lags=[48, 168])
    gold = feature_table(small, "2025-04-24 00:00", hours=7 * 24)
    train = with_levels(training_rows(gold, small), {132: 10, 161: 30, 237: 80})

    report = backtest(train, {LIGHTGBM: lightgbm_forecaster(small)}, small)

    assert report.forecasters[LIGHTGBM].overall.rows == 2 * 2 * 24 * 3
    assert report.forecasters[LIGHTGBM].overall.mae == pytest.approx(0, abs=LEAF)


# --- what it reads ----------------------------------------------------------------------


def test_model_inputs_are_the_feature_columns_in_their_order(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    forecaster = _fitted(few_trees, table)

    assert forecaster.model is not None
    assert forecaster.model.feature_name_ == feature_columns(few_trees)
    assert IS_OUTLIER not in forecaster.model.feature_name_
    assert TARGET not in forecaster.model.feature_name_
    assert HOUR_START not in forecaster.model.feature_name_


def test_outlier_flag_changes_nothing(few_trees: Settings, table: pd.DataFrame) -> None:
    # It is not an input, and flagged hours are not left out of the training rows.
    all_flagged = table.assign(**{IS_OUTLIER: True})

    usual = _fitted(few_trees, table).predict(table)
    flagged = _fitted(few_trees, all_flagged).predict(all_flagged)

    assert flagged.tolist() == usual.tolist()


def test_predict_needs_neither_the_target_nor_the_outlier_flag(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    forecaster = _fitted(few_trees, table)
    inputs_only = table[feature_columns(few_trees)]

    forecast = forecaster.predict(inputs_only)

    assert forecast.shape == (len(table),)
    assert forecast.dtype == "float64"
    assert forecast.tolist() == forecaster.predict(table.assign(**{TARGET: 0})).tolist()


def test_columns_are_read_by_name_so_their_order_and_extra_columns_do_not_matter(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    forecaster = _fitted(few_trees, table)
    reversed_columns = table[list(reversed(table.columns))].assign(extra=1.0)

    assert forecaster.predict(reversed_columns).tolist() == forecaster.predict(table).tolist()


def test_forecaster_reads_columns_parameters_and_seed_from_the_config(
    few_trees: Settings,
) -> None:
    forecaster = lightgbm_forecaster(few_trees)

    assert forecaster.columns == feature_columns(few_trees)
    assert forecaster.params == {
        "deterministic": True,
        "force_row_wise": True,
        **FEW_TREES,
        "random_state": 42,
    }
    assert few_trees.project.seed == 42


# --- what it promises about a forecast --------------------------------------------------


def test_poisson_forecasts_are_positive_and_finite_even_for_strange_inputs(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    forecaster = _fitted(few_trees, table)
    strange = table.assign(**{lag_column(48): 1e12, lag_column(168): 0.0, "temperature_2m": -1e6})

    forecast = forecaster.predict(strange)

    assert np.isfinite(forecast).all()
    assert (forecast > 0).all()


def test_a_negative_forecast_is_cut_off_at_zero(one_tree: Settings, table: pd.DataFrame) -> None:
    # Pickups are never negative, but plain regression repeats whatever it is shown.
    # A zone with -10 "pickups" makes the tree answer -10, and the forecaster 0.
    train = with_levels(table, {132: -10, 161: 30, 237: 80})

    forecaster = _fitted(one_tree, train)

    assert forecaster.model is not None
    raw = forecaster.model.predict(_one_row_per_zone(train)[forecaster.columns])
    assert raw == pytest.approx([-10, 30, 80], abs=LEAF)
    assert forecaster.predict(_one_row_per_zone(train)) == pytest.approx([0, 30, 80], abs=LEAF)
    assert forecaster.predict(train).min() == 0.0


# --- the same input gives the same model ------------------------------------------------


def test_same_rows_config_and_seed_give_the_same_model_and_forecasts(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    first = _fitted(few_trees, table)
    second = _fitted(few_trees, table)

    assert first.model is not None
    assert second.model is not None
    assert first.model.booster_.model_to_string() == second.model.booster_.model_to_string()
    assert first.predict(table).tolist() == second.predict(table).tolist()


def test_another_seed_gives_another_model(few_trees: Settings, table: pd.DataFrame) -> None:
    # Without this the test above would also pass for a model that draws nothing at random.
    other_seed = few_trees.model_copy(
        update={"project": few_trees.project.model_copy(update={"seed": 43})}
    )

    first = _fitted(few_trees, table).predict(table)
    second = _fitted(other_seed, table).predict(table)

    assert first.tolist() != second.tolist()


def test_number_of_threads_does_not_change_the_forecasts(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    one_thread = with_model_params(few_trees, FEW_TREES | {"n_jobs": 1})
    three_threads = with_model_params(few_trees, FEW_TREES | {"n_jobs": 3})

    first = _fitted(one_thread, table).predict(table)
    second = _fitted(three_threads, table).predict(table)

    assert first.tolist() == second.tolist()


# --- what it refuses --------------------------------------------------------------------


def test_predict_before_fit_is_refused(few_trees: Settings, table: pd.DataFrame) -> None:
    forecaster = lightgbm_forecaster(few_trees)

    with pytest.raises(ModelInputError, match=r"has not been fitted. Call fit"):
        forecaster.predict(table)


def test_training_rows_without_the_target_are_refused(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    forecaster = lightgbm_forecaster(few_trees)

    with pytest.raises(ModelInputError, match=r"no column pickups..*load_training_features"):
        forecaster.fit(table.drop(columns=TARGET))


@pytest.mark.parametrize("method", ["fit", "predict"])
def test_a_missing_input_column_is_refused(
    few_trees: Settings, table: pd.DataFrame, method: str
) -> None:
    forecaster = _fitted(few_trees, table)
    incomplete = table.drop(columns=[lag_column(168), DAY_OF_WEEK])

    with pytest.raises(ModelInputError, match=r"no column lag_168h, day_of_week\..*build_features"):
        getattr(forecaster, method)(incomplete)


@pytest.mark.parametrize("method", ["fit", "predict"])
@pytest.mark.parametrize("hole", [np.nan, np.inf, -np.inf])
def test_a_missing_or_infinite_input_is_refused(
    few_trees: Settings, table: pd.DataFrame, method: str, hole: float
) -> None:
    forecaster = _fitted(few_trees, table)
    with_hole = table.copy()
    with_hole.loc[5, lag_column(48)] = hole

    with pytest.raises(ModelInputError, match=r"Missing or infinite values.*lag_48h..*history"):
        getattr(forecaster, method)(with_hole)


def test_an_input_of_another_type_than_in_training_is_refused(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    forecaster = _fitted(few_trees, table)
    other_types = table.assign(
        **{ZONE_ID: table[ZONE_ID].astype("float64"), IS_HOLIDAY: table[IS_HOLIDAY].astype("int64")}
    )

    with pytest.raises(
        ModelInputError,
        match=r"zone_id is float64, was int64; is_holiday is int64, was bool\..*build_features",
    ):
        forecaster.predict(other_types)


def test_a_zone_the_model_was_not_trained_on_is_refused(
    few_trees: Settings, table: pd.DataFrame
) -> None:
    # LightGBM itself would answer: an unknown category is "none of the zones I know".
    forecaster = _fitted(few_trees, table[table[ZONE_ID] != 237])

    with pytest.raises(
        ModelInputError, match=r"not trained on zone 237; it knows \[132, 161\]\..*data\.zones"
    ):
        forecaster.predict(table)


@pytest.mark.parametrize("name", ["random_state", "seed"])
def test_a_seed_in_the_model_parameters_is_refused(few_trees: Settings, name: str) -> None:
    with_seed = with_model_params(few_trees, FEW_TREES | {name: 7})

    with pytest.raises(PipelineError, match=rf"sets {name}.*project\.seed \(42\)\. Remove {name}"):
        lightgbm_forecaster(with_seed)


def test_inputs_without_the_zone_are_refused() -> None:
    with pytest.raises(PipelineError, match=r"lack zone_id..*feature_columns"):
        LightGbmForecaster([lag_column(168), HOUR], FEW_TREES, seed=42)


def test_model_input_errors_are_reported_by_the_command_line_in_one_line() -> None:
    assert issubclass(ModelInputError, PipelineError)
