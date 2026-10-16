"""The SHAP explanation, on a small model that is fitted once for the whole module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import shap

from demand.config import Settings, load_settings
from demand.errors import PipelineError
from demand.models import explain as explain_module
from demand.models.explain import (
    SHAP_SAMPLE_ROWS,
    Explanation,
    ShapReport,
    explain,
    explain_to_reports,
    sample_rows,
    shap_plot_path,
    shap_report,
    shap_report_path,
    summary_plot,
)
from demand.models.gbm import LightGbmForecaster, ModelInputError, lightgbm_forecaster
from demand.schemas import HOUR, ZONE_ID, feature_columns, lag_column
from tests.conftest import CONFIG_FILE, TEST_ZONES
from tests.unit.models.conftest import FEW_TREES, ONE_TREE, feature_table, with_model_params

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def small() -> Settings:
    base = load_settings(CONFIG_FILE)
    settings = base.model_copy(update={"data": base.data.model_copy(update={"zones": TEST_ZONES})})
    return with_model_params(settings, FEW_TREES)


@pytest.fixture(scope="module")
def table(small: Settings) -> pd.DataFrame:
    """Three zones and nine days: 648 rows."""
    return feature_table(small, "2025-04-01 00:00", hours=9 * 24)


@pytest.fixture(scope="module")
def forecaster(small: Settings, table: pd.DataFrame) -> LightGbmForecaster:
    fitted = lightgbm_forecaster(small)
    fitted.fit(table)
    return fitted


@pytest.fixture(scope="module")
def explanation(forecaster: LightGbmForecaster, table: pd.DataFrame) -> Explanation:
    return explain(forecaster, table)


# --- the sample of rows ----------------------------------------------------------------------


def test_sample_is_the_same_for_the_same_seed_and_keeps_the_order_of_the_table(
    table: pd.DataFrame,
) -> None:
    first = sample_rows(table, 100, seed=42)

    assert len(first) == 100
    assert first.index.is_monotonic_increasing
    assert first.index.is_unique
    assert first.equals(sample_rows(table, 100, seed=42))
    assert not first.index.equals(sample_rows(table, 100, seed=43).index)
    assert first.equals(table.loc[first.index])


def test_a_table_smaller_than_the_sample_is_explained_whole(table: pd.DataFrame) -> None:
    assert sample_rows(table, len(table), seed=42) is table
    assert sample_rows(table, len(table) + 1, seed=42) is table


def test_the_sample_of_a_training_run_is_a_few_thousand_rows() -> None:
    assert SHAP_SAMPLE_ROWS == 5000


# --- the SHAP values ---------------------------------------------------------------------------


def test_one_contribution_per_row_and_input(
    explanation: Explanation, table: pd.DataFrame, small: Settings
) -> None:
    columns = feature_columns(small)

    assert explanation.values.shape == (len(table), len(columns))
    assert list(explanation.inputs.columns) == columns
    assert np.isfinite(explanation.values).all()


def test_base_value_and_contributions_add_up_to_the_log_of_the_forecast(
    explanation: Explanation, forecaster: LightGbmForecaster, table: pd.DataFrame
) -> None:
    output = explanation.base_value + explanation.values.sum(axis=1)

    # The objective is poisson: the output of the trees is the logarithm of the pickups.
    np.testing.assert_allclose(np.exp(output), forecaster.predict(table), rtol=1e-9)


def test_shap_library_and_lightgbm_itself_give_the_same_contributions(
    explanation: Explanation, forecaster: LightGbmForecaster, table: pd.DataFrame
) -> None:
    assert forecaster.model is not None
    own = forecaster.model.predict(table[forecaster.columns], pred_contrib=True)

    # LightGBM returns one more column: the base value, the same in every row.
    np.testing.assert_allclose(explanation.values, own[:, :-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(own[:, -1], explanation.base_value, rtol=0, atol=1e-12)


def test_an_input_the_trees_never_use_contributes_nothing(
    explanation: Explanation, small: Settings
) -> None:
    columns = feature_columns(small)

    # Snow is 0 in every row of the test table, so no tree can split on it.
    assert not explanation.values[:, columns.index("snowfall")].any()
    assert explanation.values[:, columns.index(lag_column(168))].any()


def test_contributions_of_one_tree_can_be_checked_by_hand(small: Settings) -> None:
    # Two zones with the same pickups in every hour: 100 and 300. One tree with one
    # split on the zone forecasts both exactly. The base value is the mean, 200, and
    # the zone, read as a label, moves it by -100 and by +100. No other input can.
    two_zones = small.model_copy(update={"data": small.data.model_copy(update={"zones": [1, 2]})})
    rows = feature_table(two_zones, "2025-04-01 00:00", hours=24)
    rows = rows.assign(pickups=rows[ZONE_ID].map({1: 100, 2: 300}))
    columns = [ZONE_ID, HOUR]
    one_tree = LightGbmForecaster(columns, ONE_TREE | {"num_leaves": 2}, seed=42)
    one_tree.fit(rows)

    explained = explain(one_tree, rows)

    assert explained.base_value == pytest.approx(200.0)
    zone = explained.values[:, 0]
    assert zone[(rows[ZONE_ID] == 1).to_numpy()].tolist() == pytest.approx([-100.0] * 24)
    assert zone[(rows[ZONE_ID] == 2).to_numpy()].tolist() == pytest.approx([100.0] * 24)
    assert not explained.values[:, 1].any()


def test_same_model_and_rows_give_the_same_contributions(
    explanation: Explanation, forecaster: LightGbmForecaster, table: pd.DataFrame
) -> None:
    again = explain(forecaster, table)

    assert again.base_value == explanation.base_value
    assert again.values.tolist() == explanation.values.tolist()


def test_only_the_inputs_are_read_whatever_else_the_rows_hold(
    explanation: Explanation, forecaster: LightGbmForecaster, table: pd.DataFrame
) -> None:
    shuffled = table[list(reversed(table.columns))]

    assert explain(forecaster, shuffled).values.tolist() == explanation.values.tolist()


# --- the report ------------------------------------------------------------------------------


def _explanation(values: list[list[float]], columns: list[str]) -> Explanation:
    inputs = pd.DataFrame(np.zeros((len(values), len(columns))), columns=columns)
    return Explanation(inputs=inputs, values=np.array(values, dtype="float64"), base_value=4.5)


def test_report_is_the_mean_size_of_every_contribution_the_largest_first() -> None:
    explained = _explanation([[1.0, -3.0, 0.0], [-2.0, 5.0, 0.0]], ["a", "b", "c"])

    report = shap_report(explained, seed=42, objective="poisson")

    assert report == ShapReport(
        rows=2,
        seed=42,
        objective="poisson",
        base_value=4.5,
        # The direction does not count: -3 and 5 are 4 on average.
        mean_abs_shap={"b": 4.0, "a": 1.5, "c": 0.0},
        share_percent={"b": 72.73, "a": 27.27, "c": 0.0},
    )
    assert list(report.mean_abs_shap) == ["b", "a", "c"]


def test_inputs_of_equal_size_are_listed_by_name() -> None:
    explained = _explanation([[1.0, 1.0, 2.0]], ["b", "a", "c"])

    assert list(shap_report(explained, 42, "poisson").mean_abs_shap) == ["c", "a", "b"]


def test_report_of_a_model_that_uses_no_input_has_no_shares() -> None:
    explained = _explanation([[0.0, 0.0]], ["a", "b"])

    assert shap_report(explained, 42, "poisson").share_percent == {"a": 0.0, "b": 0.0}


def test_report_of_a_real_model_ranks_the_inputs_it_uses_first(
    explanation: Explanation, small: Settings
) -> None:
    report = shap_report(explanation, seed=42, objective="poisson")

    assert sorted(report.mean_abs_shap) == sorted(feature_columns(small))
    sizes = list(report.mean_abs_shap.values())
    assert sizes == sorted(sizes, reverse=True)
    assert sizes[0] > 0
    assert report.mean_abs_shap["snowfall"] == 0.0
    assert sum(report.share_percent.values()) == pytest.approx(100, abs=0.1)
    assert report.rows == 648


# --- the plot --------------------------------------------------------------------------------


def test_plot_is_a_png_drawn_in_memory_and_leaves_no_figure_open(
    explanation: Explanation, recwarn: pytest.WarningsRecorder
) -> None:
    image = summary_plot(explanation, seed=42, objective="poisson")

    assert [str(warning.message) for warning in recwarn] == []

    assert image.startswith(PNG_SIGNATURE)
    assert plt.get_backend().lower() == "agg"
    assert plt.get_fignums() == []
    # The version of matplotlib is not written into the file.
    assert b"Software" not in image


def test_plot_of_the_same_explanation_shows_the_same_dots(explanation: Explanation) -> None:
    first = summary_plot(explanation, seed=42, objective="poisson")

    # Same process, same matplotlib: only then are the bytes equal (see explain.py).
    assert summary_plot(explanation, seed=42, objective="poisson") == first
    # The seed decides how the dots are spread out, and nothing else does.
    assert summary_plot(explanation, seed=43, objective="poisson") != first


def test_plot_colours_every_input_by_its_value_but_not_the_zone(
    explanation: Explanation, monkeypatch: pytest.MonkeyPatch
) -> None:
    drawn: dict[str, Any] = {}

    def record(values: Any, **options: Any) -> None:
        drawn.update(options, values=values)

    monkeypatch.setattr(shap, "summary_plot", record)

    summary_plot(explanation, seed=42, objective="poisson")

    colours = drawn["features"]
    assert list(colours.columns) == list(explanation.inputs.columns)
    # A missing value is drawn grey: the zone is a label, and a label is not high or low.
    assert colours[ZONE_ID].isna().all()
    others = colours.drop(columns=ZONE_ID)
    assert others.notna().all().all()
    assert others[HOUR].tolist() == explanation.inputs[HOUR].tolist()
    assert drawn["values"] is explanation.values
    assert drawn["show"] is False
    assert drawn["max_display"] == len(explanation.inputs.columns)
    # The SHAP values themselves are untouched: the zone keeps its contributions.
    assert not explanation.inputs[ZONE_ID].isna().any()


@pytest.mark.parametrize(
    ("objective", "label"),
    [
        ("poisson", "SHAP value: added to the logarithm of the forecast (+0.69 doubles it)"),
        ("regression", "SHAP value: pickups added to the forecast"),
    ],
)
def test_plot_says_which_scale_the_values_are_on(
    explanation: Explanation, monkeypatch: pytest.MonkeyPatch, objective: str, label: str
) -> None:
    labels = []
    # Drawing is left out: only the label under the plot is of interest here.
    monkeypatch.setattr(shap, "summary_plot", lambda values, **options: None)
    monkeypatch.setattr(plt, "xlabel", labels.append)

    summary_plot(explanation, seed=42, objective=objective)

    assert labels == [label]


# --- on disk ---------------------------------------------------------------------------------


def test_explanation_is_written_as_a_report_and_a_picture(
    forecaster: LightGbmForecaster, table: pd.DataFrame, tmp_path: Path
) -> None:
    report = explain_to_reports(forecaster, table, tmp_path, seed=42)

    assert sorted(file.name for file in tmp_path.iterdir()) == [
        "shap_importance.json",
        "shap_summary.png",
    ]
    stored = json.loads(shap_report_path(tmp_path).read_text(encoding="utf-8"))
    assert ShapReport.model_validate(stored) == report
    assert list(stored) == [
        "rows",
        "seed",
        "objective",
        "base_value",
        "mean_abs_shap",
        "share_percent",
    ]
    assert (stored["rows"], stored["seed"], stored["objective"]) == (648, 42, "poisson")
    assert shap_plot_path(tmp_path).read_bytes().startswith(PNG_SIGNATURE)


def test_report_file_is_identical_when_the_same_model_is_explained_again(
    forecaster: LightGbmForecaster, table: pd.DataFrame, tmp_path: Path
) -> None:
    explain_to_reports(forecaster, table, tmp_path / "first", seed=42)
    explain_to_reports(forecaster, table, tmp_path / "second", seed=42)

    first = shap_report_path(tmp_path / "first").read_bytes()
    assert shap_report_path(tmp_path / "second").read_bytes() == first


def test_only_a_sample_of_a_large_table_is_explained(
    forecaster: LightGbmForecaster,
    table: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(explain_module, "SHAP_SAMPLE_ROWS", 50)

    report = explain_to_reports(forecaster, table, tmp_path, seed=42)

    assert report.rows == 50


# --- error paths -----------------------------------------------------------------------------


def test_a_forecaster_that_is_not_fitted_cannot_be_explained(
    small: Settings, table: pd.DataFrame
) -> None:
    with pytest.raises(ModelInputError, match=r"Call fit .* before explain"):
        explain(lightgbm_forecaster(small), table)


def test_explanation_stops_when_the_contributions_do_not_add_up(
    forecaster: LightGbmForecaster, table: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongExplainer:
        """Claims that no input contributes anything."""

        expected_value = 0.0

        def __init__(self, model: object) -> None:
            pass

        def shap_values(self, inputs: pd.DataFrame) -> np.ndarray:
            return np.zeros(inputs.shape)

    monkeypatch.setattr(shap, "TreeExplainer", WrongExplainer)

    with pytest.raises(PipelineError, match=r"do not add up .* pip install -r requirements\.lock"):
        explain(forecaster, table)
