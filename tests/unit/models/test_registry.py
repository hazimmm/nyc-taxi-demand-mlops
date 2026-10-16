"""The registry edge, tested against a temporary MLflow store (``Store`` in ``conftest.py``)."""

from __future__ import annotations

import mlflow
import pytest
from mlflow.exceptions import MlflowException

from demand.errors import PipelineError
from demand.models.registry import Registry
from tests.unit.models.conftest import RUN_PARAMS, RUN_TAGS, TUNED_FOLDS, Store, mae_metrics


def test_a_store_without_the_registered_model_is_refused_with_what_to_do(store: Store) -> None:
    with pytest.raises(
        PipelineError, match=r"No model is registered under 'taxi-demand'.*dvc repro"
    ):
        Registry(store.settings)


def test_a_registered_model_without_a_version_has_no_newest(store: Store) -> None:
    store.client.create_registered_model("taxi-demand")

    with pytest.raises(PipelineError, match=r"'taxi-demand' has no version\. Train one first"):
        Registry(store.settings).newest()


def test_the_newest_version_comes_with_what_its_run_recorded(store: Store) -> None:
    store.register()
    store.register()

    registry = Registry(store.settings)
    newest = registry.newest()

    assert registry.versions() == [1, 2]
    assert newest.version == 2
    assert newest.params == RUN_PARAMS
    assert {key: newest.tags[key] for key in RUN_TAGS} == RUN_TAGS
    assert newest.metrics == mae_metrics(TUNED_FOLDS)
    assert newest.run_id == store.client.get_model_version("taxi-demand", "2").run_id


def test_a_missing_version_is_refused_with_the_versions_there_are(store: Store) -> None:
    store.register()

    with pytest.raises(PipelineError, match=r"has no version 3\. Its versions are \[1\]"):
        Registry(store.settings).version(3)


def test_a_version_without_a_training_run_is_refused(store: Store) -> None:
    store.register()
    store.client.create_model_version("taxi-demand", source="file:///somewhere/model")

    with pytest.raises(PipelineError, match="Version 2 of 'taxi-demand' was registered without"):
        Registry(store.settings).newest()


def test_aliases_are_set_moved_and_removed(store: Store) -> None:
    store.register()
    store.register()
    registry = Registry(store.settings)
    assert registry.aliases() == {}

    registry.set_alias("champion", 1)
    registry.set_alias("champion", 2)
    registry.set_alias("challenger", 1)
    assert registry.aliases() == {"champion": 2, "challenger": 1}

    registry.remove_alias("challenger")
    registry.remove_alias("challenger")
    assert registry.aliases() == {"champion": 2}


def test_tags_are_written_onto_the_version(store: Store) -> None:
    store.register()
    registry = Registry(store.settings)

    registry.tag(1, {"promotion.decision": "promoted", "promotion.reason": "first"})
    registry.tag(1, {"promotion.reason": "again"})

    assert store.tags(1) == {"promotion.decision": "promoted", "promotion.reason": "again"}


def _fail(*args: object, **kwargs: object) -> None:
    raise MlflowException("the database is locked", error_code="INTERNAL_ERROR")


def test_another_mlflow_error_when_opening_is_not_hidden(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mlflow.MlflowClient, "get_registered_model", _fail)

    with pytest.raises(MlflowException, match="locked"):
        Registry(store.settings)


def test_another_mlflow_error_when_reading_a_version_is_not_hidden(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.register()
    registry = Registry(store.settings)
    monkeypatch.setattr(mlflow.MlflowClient, "get_model_version", _fail)

    with pytest.raises(MlflowException, match="locked"):
        registry.version(1)


def test_the_tags_of_a_version_are_not_those_of_its_run(store: Store) -> None:
    store.register()
    registry = Registry(store.settings)
    assert registry.version_tags(1) == {}

    registry.tag(1, {"promotion.decision": "rejected"})

    assert registry.version_tags(1) == {"promotion.decision": "rejected"}
