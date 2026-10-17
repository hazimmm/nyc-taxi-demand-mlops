"""Loading the served model, against a real MLflow store in a temporary folder.

Recording a model takes MLflow seconds, so one tiny LightGBM model is trained, recorded
and given the alias ``champion`` once for the module (``store``); every test loads it
from there. Nothing here touches ``mlflow.db`` or ``mlruns`` of the repository.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd
import pytest

from demand.config import Settings, load_settings
from demand.errors import PipelineError
from demand.models.gbm import LightGbmForecaster, lightgbm_forecaster
from demand.models.tracking import log_training_run
from demand.schemas import feature_columns
from demand.serving.model_loader import (
    ModelLoadError,
    check_store_exists,
    load_model,
    parse_model_uri,
)
from tests.conftest import CONFIG_FILE, TEST_ZONES, with_features
from tests.unit.models.conftest import (
    FEW_TREES,
    feature_table,
    with_mlflow_store,
    with_model_params,
)

METRICS = {"lightgbm.mae": 24.5, "weekly_naive.mae": 27.7}
TAGS = {"git_commit": "0123abc", "git_dirty": "false", "data_version": "7bac.dir"}


@dataclass(frozen=True)
class Store:
    """A temporary store with version 1 of ``taxi-demand``, alias ``champion`` on it."""

    settings: Settings
    folder: Path
    forecaster: LightGbmForecaster
    table: pd.DataFrame
    run_id: str

    def serving(self, model_uri: str, settings: Settings | None = None) -> Settings:
        """The settings of the store with another ``serving.model_uri``."""
        base = settings or self.settings
        return base.model_copy(
            update={"serving": base.serving.model_copy(update={"model_uri": model_uri})}
        )

    def files(self) -> dict[Path, int]:
        return {path: path.stat().st_mtime_ns for path in self.folder.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Store]:
    folder = tmp_path_factory.mktemp("serving_store")
    base = load_settings(CONFIG_FILE)
    settings = base.model_copy(update={"data": base.data.model_copy(update={"zones": TEST_ZONES})})
    settings = with_mlflow_store(with_model_params(settings, FEW_TREES), folder)
    table = feature_table(settings, "2025-04-01 00:00", hours=48)
    forecaster = lightgbm_forecaster(settings)
    forecaster.fit(table)

    uri_before = mlflow.get_tracking_uri()
    # MLflow keeps the files of a run in ``mlruns`` of the working directory.
    with contextlib.chdir(folder):
        logged = log_training_run(
            settings,
            forecaster=forecaster,
            input_example=table[feature_columns(settings)].head(5),
            params={},
            metrics=METRICS,
            tags=TAGS,
            artifacts=[],
        )
    uri = settings.mlflow.tracking_uri
    client = mlflow.MlflowClient(tracking_uri=uri, registry_uri=uri)
    client.set_registered_model_alias("taxi-demand", "champion", str(logged.model_version))
    yield Store(settings, folder, forecaster, table, logged.run_id)
    mlflow.set_tracking_uri(uri_before)


# --- the address ------------------------------------------------------------------------


def test_an_alias_address_names_the_model_and_the_alias() -> None:
    assert parse_model_uri("models:/taxi-demand@champion") == ("taxi-demand", "champion")


def test_a_version_address_names_the_model_and_the_version() -> None:
    assert parse_model_uri("models:/taxi-demand/3") == ("taxi-demand", 3)


@pytest.mark.parametrize(
    "uri",
    [
        "runs:/0123/model",
        "mlruns/1/models/m-1/artifacts",
        "models:/taxi-demand",
        "models:/taxi-demand/latest",
        "models:/taxi-demand@champion/1",
    ],
)
def test_other_addresses_are_refused_because_they_carry_no_version(uri: str) -> None:
    with pytest.raises(ModelLoadError, match=r"loads a registered model only.*models:/<name>@"):
        parse_model_uri(uri)


def test_a_missing_sqlite_store_is_refused_and_not_created(tmp_path: Path) -> None:
    database = tmp_path / "mlflow.db"

    with pytest.raises(ModelLoadError, match=r"does not exist.*DEMAND_MLFLOW__TRACKING_URI"):
        check_store_exists(f"sqlite:///{database.as_posix()}")

    assert not database.exists()


def test_an_existing_or_remote_store_passes(store: Store) -> None:
    check_store_exists(store.settings.mlflow.tracking_uri)
    check_store_exists("http://mlflow:5000")


# --- loading ----------------------------------------------------------------------------


def test_the_alias_is_resolved_to_its_version_once_and_reported(store: Store) -> None:
    loaded = load_model(store.serving("models:/taxi-demand@champion"))

    info = loaded.info
    assert (info.registered_model, info.version, info.alias) == ("taxi-demand", 1, "champion")
    assert info.model_uri == "models:/taxi-demand@champion"
    assert info.run_id == store.run_id
    assert info.backtest_mae == 24.5
    assert (info.data_version, info.git_commit) == ("7bac.dir", "0123abc")
    assert info.feature_columns == feature_columns(store.settings)


def test_the_loaded_model_forecasts_what_the_forecaster_forecasts(store: Store) -> None:
    inputs = store.table[feature_columns(store.settings)]

    loaded = load_model(store.serving("models:/taxi-demand@champion"))

    assert loaded.predictor.predict(inputs).tolist() == store.forecaster.predict(inputs).tolist()


def test_a_pinned_version_is_loaded_without_an_alias(store: Store) -> None:
    info = load_model(store.serving("models:/taxi-demand/1")).info

    assert (info.version, info.alias) == (1, None)


def test_loading_writes_nothing_into_the_store(store: Store) -> None:
    before = store.files()

    load_model(store.serving("models:/taxi-demand@champion"))

    assert store.files() == before
    assert not list(store.folder.rglob("registered_model_meta"))


def test_an_alias_that_is_not_set_is_refused_with_what_is_set(store: Store) -> None:
    with pytest.raises(
        ModelLoadError,
        match=r"'challenger' is not set on 'taxi-demand' \(aliases set: \{'champion': 1\}; "
        r"versions: \[1\]\).*demand promote",
    ):
        load_model(store.serving("models:/taxi-demand@challenger"))


def test_a_version_that_does_not_exist_is_refused(store: Store) -> None:
    with pytest.raises(PipelineError, match="has no version 5"):
        load_model(store.serving("models:/taxi-demand/5"))


def test_a_model_of_another_name_is_refused(store: Store) -> None:
    with pytest.raises(ModelLoadError, match=r"names the model 'other'.*'taxi-demand'"):
        load_model(store.serving("models:/other@champion"))


def test_a_model_with_other_inputs_than_the_config_builds_is_refused(store: Store) -> None:
    other_lags = with_features(store.settings, horizon_hours=48, lags=[48, 336])

    with pytest.raises(ModelLoadError, match=r"takes the inputs .*lag_168h.*builds .*lag_336h"):
        load_model(store.serving("models:/taxi-demand@champion", other_lags))


def test_a_missing_store_is_refused_before_mlflow_is_asked(store: Store, tmp_path: Path) -> None:
    settings = with_mlflow_store(store.serving("models:/taxi-demand@champion"), tmp_path)

    with pytest.raises(ModelLoadError, match="does not exist"):
        load_model(settings)

    assert not (tmp_path / "mlflow.db").exists()
