"""The MLflow logging, tested against a real MLflow store in a temporary folder.

Creating a store and saving a model takes MLflow several seconds, so one run is recorded
once for the whole module (``recorded``) and every test reads from it. Nothing here
touches ``mlflow.db`` or ``mlruns`` of the repository: the database and the working
directory are a temporary folder while the run is recorded.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel

from demand.config import Settings, load_settings
from demand.models.gbm import LightGbmForecaster, lightgbm_forecaster
from demand.models.tracking import MODEL_ARTIFACT_NAME, ForecasterModel, LoggedRun, log_training_run
from demand.schemas import HOUR, IS_WEEKEND, ZONE_ID, feature_columns, lag_column
from tests.conftest import CONFIG_FILE, REPO_ROOT, TEST_ZONES
from tests.unit.models.conftest import (
    FEW_TREES,
    feature_table,
    with_mlflow_store,
    with_model_params,
)

PARAMS = {"model.n_estimators": 20, "features.lags": [48, 168], "data.train_end_date": "2025-04-30"}
METRICS = {"lightgbm.mae": 24.5, "weekly_naive.mae": 27.7, "improvement.mae_percent": 11.552}
TAGS = {"git_commit": "0123abc", "git_dirty": "false", "data_version": "7bac.dir"}


@dataclass(frozen=True)
class Recorded:
    """One training run in a temporary MLflow store, and what went into it."""

    settings: Settings
    folder: Path
    forecaster: LightGbmForecaster
    table: pd.DataFrame
    logged: LoggedRun

    @property
    def client(self) -> mlflow.MlflowClient:
        return mlflow.MlflowClient(tracking_uri=self.settings.mlflow.tracking_uri)

    @property
    def model_uri(self) -> str:
        name = self.settings.mlflow.registered_model_name
        return f"models:/{name}/{self.logged.model_version}"

    def load(self) -> PyFuncModel:
        return mlflow.pyfunc.load_model(self.model_uri)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Recorded]:
    folder = tmp_path_factory.mktemp("mlflow_store")
    base = load_settings(CONFIG_FILE)
    settings = base.model_copy(update={"data": base.data.model_copy(update={"zones": TEST_ZONES})})
    settings = with_mlflow_store(with_model_params(settings, FEW_TREES), folder)
    table = feature_table(settings, "2025-04-01 00:00", hours=48)
    forecaster = lightgbm_forecaster(settings)
    forecaster.fit(table)
    reports = [folder / "backtest_report.json", folder / "silver_cleaning_report.json"]
    for number, report in enumerate(reports):
        report.write_text(json.dumps({"report": number}), encoding="utf-8")

    uri_before = mlflow.get_tracking_uri()
    # MLflow keeps the files of a run in ``mlruns`` of the working directory.
    with contextlib.chdir(folder):
        logged = log_training_run(
            settings,
            forecaster=forecaster,
            input_example=table[feature_columns(settings)].head(5),
            params=PARAMS,
            metrics=METRICS,
            tags=TAGS,
            artifacts=reports,
        )
    yield Recorded(settings, folder, forecaster, table, logged)
    mlflow.set_tracking_uri(uri_before)


def test_run_is_recorded_as_finished_in_the_configured_experiment(recorded: Recorded) -> None:
    run = recorded.client.get_run(recorded.logged.run_id)
    experiment = recorded.client.get_experiment(run.info.experiment_id)

    assert run.info.status == "FINISHED"
    assert experiment.name == "taxi-demand"


def test_parameters_are_recorded(recorded: Recorded) -> None:
    run = recorded.client.get_run(recorded.logged.run_id)

    # MLflow stores every parameter as text.
    assert run.data.params == {
        "model.n_estimators": "20",
        "features.lags": "[48, 168]",
        "data.train_end_date": "2025-04-30",
    }


def test_metrics_are_recorded(recorded: Recorded) -> None:
    run = recorded.client.get_run(recorded.logged.run_id)

    assert run.data.metrics == METRICS


def test_tags_are_recorded(recorded: Recorded) -> None:
    run = recorded.client.get_run(recorded.logged.run_id)

    ours = {name: value for name, value in run.data.tags.items() if not name.startswith("mlflow.")}
    assert ours == TAGS


def test_report_files_are_stored_with_the_run(recorded: Recorded) -> None:
    stored = {artifact.path for artifact in recorded.client.list_artifacts(recorded.logged.run_id)}

    assert stored == {"backtest_report.json", "silver_cleaning_report.json"}
    local = mlflow.artifacts.download_artifacts(
        run_id=recorded.logged.run_id,
        artifact_path="backtest_report.json",
        tracking_uri=recorded.settings.mlflow.tracking_uri,
    )
    assert json.loads(Path(local).read_text(encoding="utf-8")) == {"report": 0}


def test_model_is_registered_as_version_1_and_no_alias_is_set(recorded: Recorded) -> None:
    (version,) = recorded.client.search_model_versions("name='taxi-demand'")

    assert recorded.logged.model_version == 1
    assert int(version.version) == 1
    assert version.run_id == recorded.logged.run_id
    assert version.aliases == []
    assert recorded.client.get_registered_model("taxi-demand").aliases == {}


def test_model_belongs_to_the_run_under_its_name(recorded: Recorded) -> None:
    run = recorded.client.get_run(recorded.logged.run_id)

    (output,) = run.outputs.model_outputs
    assert recorded.client.get_logged_model(output.model_id).name == MODEL_ARTIFACT_NAME


def test_loaded_model_forecasts_exactly_what_the_forecaster_forecasts(recorded: Recorded) -> None:
    inputs = recorded.table[feature_columns(recorded.settings)]

    loaded = recorded.load()

    assert list(loaded.metadata.flavors) == ["python_function"]
    assert loaded.predict(inputs).tolist() == recorded.forecaster.predict(inputs).tolist()


def test_stored_object_is_the_forecaster_with_its_checks(recorded: Recorded) -> None:
    stored = recorded.load().unwrap_python_model()

    assert isinstance(stored, ForecasterModel)
    assert isinstance(stored.forecaster, LightGbmForecaster)
    assert stored.forecaster.columns == feature_columns(recorded.settings)


def test_signature_names_the_feature_columns_in_order_with_their_types(recorded: Recorded) -> None:
    signature = recorded.load().metadata.signature

    assert signature.inputs.input_names() == feature_columns(recorded.settings)
    types = {column.name: column.type.name for column in signature.inputs.inputs}
    assert types[ZONE_ID] == "long"
    assert types[HOUR] == "long"
    assert types[lag_column(168)] == "double"
    assert types["temperature_2m"] == "double"
    assert types[IS_WEEKEND] == "boolean"
    assert set(types.values()) == {"long", "double", "boolean"}
    assert [str(output.type) for output in signature.outputs.inputs] == ["float64"]


def test_input_example_is_stored_with_the_model(recorded: Recorded) -> None:
    example = recorded.load().input_example

    expected = recorded.table[feature_columns(recorded.settings)].head(5)
    pd.testing.assert_frame_equal(example, expected, check_dtype=False)


def test_loaded_model_puts_shuffled_columns_back_in_order(recorded: Recorded) -> None:
    columns = feature_columns(recorded.settings)
    inputs = recorded.table[columns].head(10)

    forecast = recorded.load().predict(inputs[list(reversed(columns))])

    assert forecast.tolist() == recorded.forecaster.predict(inputs).tolist()


def test_loaded_model_refuses_a_missing_column(recorded: Recorded) -> None:
    inputs = recorded.table[feature_columns(recorded.settings)].drop(columns=lag_column(168))

    with pytest.raises(MlflowException, match="lag_168h"):
        recorded.load().predict(inputs)


def test_loaded_model_refuses_a_whole_number_column_that_arrives_as_a_float(
    recorded: Recorded,
) -> None:
    inputs = recorded.table[feature_columns(recorded.settings)]
    float_zone = inputs.assign(**{ZONE_ID: inputs[ZONE_ID].astype("float64")})

    with pytest.raises(MlflowException, match="zone_id"):
        recorded.load().predict(float_zone)


def test_nothing_was_written_into_the_repository(recorded: Recorded) -> None:
    assert (recorded.folder / "mlflow.db").is_file()
    assert (recorded.folder / "mlruns").is_dir()
    experiment = recorded.client.get_experiment_by_name("taxi-demand")
    assert experiment is not None
    assert REPO_ROOT.as_posix().lower() not in experiment.artifact_location.lower()
