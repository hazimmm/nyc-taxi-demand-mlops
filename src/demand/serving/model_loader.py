"""Load the served model once, at startup, and know exactly which version it is.

**From an alias to a version, once.** ``serving.model_uri`` is
``models:/taxi-demand@champion``: "whichever version the alias points at". The alias can
move at any time (``python -m demand promote``). If the service asked MLflow for the alias
again later, or reported the alias instead of a number, a response could name another
version than the one that computed it. So the alias is resolved to a version number once,
before the model is loaded; that version is loaded by its number
(``models:/taxi-demand/2``), and that number is what every response, ``/model-info`` and
``/metrics`` report, for as long as the process lives. ``models:/<name>/<version>`` is
accepted as well, to pin a version by hand. Other kinds of address (a run, a folder) are
refused: they carry no version number, and this service reports one with every forecast.

**A new champion is picked up by a restart.** The service does not watch the alias. A
service that swapped its model while running would need a lock around every forecast,
and two replicas behind one address could serve two versions for minutes without anyone
deciding it. A restart is one clear moment: in Kubernetes ``kubectl rollout restart``
replaces the pods one by one, each new pod resolves the alias when it starts, and the
old pods serve until the new ones are ready. The same restart picks up a new data
snapshot.

**Nothing is written into the store, and only the configured store is asked.** Loading a
``models:/`` address has two problems in MLflow 3.17. It writes a small file,
``registered_model_meta``, into the folder it loads from, which with a store on this
machine is the model's own folder in ``mlruns`` (and fails on a read-only filesystem).
And it looks the version up in the registry set for the whole process
(``MLFLOW_TRACKING_URI``, or ``sqlite:///mlflow.db`` in the working folder), not in
``mlflow.tracking_uri`` of the config: a service pointed at a server could report one
version and load another from a local file. So the address of the version's files is
asked of the configured registry (``get_model_version_download_uri``), those files are
copied into a temporary folder, the model is loaded from there, and the folder is
deleted. The store is only read, and the loaded model needs no file afterwards.

**What startup checks, and why a failure stops it.** The store must exist (a SQLite file
that is not there would otherwise be created, empty, by the first query), the alias must
be set, and the model's signature must name exactly ``feature_columns(settings)``, in
that order: the service builds its features from the config, and a model trained with
other lags would be given columns it never saw. Any failure raises, the process exits,
and the message says what to do. See ``app.py`` for why stopping beats starting
without a model.
"""

from __future__ import annotations

import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlflow
import pandas as pd
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.errors import PipelineError
from demand.models.gbm import LIGHTGBM
from demand.models.registry import Registry
from demand.schemas import feature_columns

BACKTEST_MAE_METRIC = f"{LIGHTGBM}.mae"
"""The pooled backtest MAE of the model, as ``training.run_metrics`` records it."""
DATA_VERSION_TAG = "data_version"
GIT_COMMIT_TAG = "git_commit"
"""Run tags written by ``training.train_from_gold``."""

_SQLITE_PREFIX = "sqlite:///"
_REGISTRY_URI = re.compile(r"^models:/(?P<name>[^/@]+)(?:@(?P<alias>[^/@]+)|/(?P<version>\d+))$")


class ModelLoadError(PipelineError):
    """The configured model cannot be served; the message says what to do."""


class Predictor(Protocol):
    """Anything with the ``predict`` of an MLflow python-function model."""

    def predict(self, data: pd.DataFrame) -> Any:
        """One forecast per row of ``data``."""


class ModelInfo(BaseModel):
    """Which model is served, and what its training run recorded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    registered_model: str
    version: int
    alias: str | None
    """The alias the version was found by at startup, or ``None`` if it was pinned."""
    model_uri: str
    """``serving.model_uri`` as configured."""
    run_id: str
    backtest_mae: float | None
    """The pooled backtest MAE of the recipe (``training.py``); ``None`` if not recorded."""
    data_version: str | None
    git_commit: str | None
    feature_columns: list[str]
    """The input columns of the model's signature, in order."""


@dataclass(frozen=True)
class LoadedModel:
    """The model in memory, and what it is."""

    predictor: Predictor
    info: ModelInfo


def parse_model_uri(uri: str) -> tuple[str, str | int]:
    """The registered name of a ``models:/`` address, and its alias or its version.

    Returns:
        The name, and the alias (text) of ``models:/<name>@<alias>`` or the version
        (a number) of ``models:/<name>/<version>``.

    Raises:
        ModelLoadError: If ``uri`` has neither form.
    """
    found = _REGISTRY_URI.match(uri)
    if found is None:
        raise ModelLoadError(
            f"serving.model_uri is {uri!r}. The service loads a registered model only, as "
            "models:/<name>@<alias> (such as models:/taxi-demand@champion) or "
            "models:/<name>/<version>, so that it can report the version it serves."
        )
    version = found["version"]
    return found["name"], found["alias"] if version is None else int(version)


def check_store_exists(tracking_uri: str) -> None:
    """Refuse a SQLite store whose file does not exist, before MLflow creates it.

    Raises:
        ModelLoadError: If ``tracking_uri`` names a SQLite file that is not there.
    """
    if tracking_uri.startswith(_SQLITE_PREFIX):
        path = Path(tracking_uri.removeprefix(_SQLITE_PREFIX))
        if not path.is_file():
            raise ModelLoadError(
                f"The MLflow store {tracking_uri} does not exist ({path.resolve()}). Start "
                "the service in the project folder, or set DEMAND_MLFLOW__TRACKING_URI to "
                "the store that holds the model."
            )


def resolve_alias(registry: Registry, alias: str) -> int:
    """The version number ``alias`` points at now.

    Raises:
        ModelLoadError: If the alias is not set.
    """
    aliases = registry.aliases()
    if alias not in aliases:
        raise ModelLoadError(
            f"The alias '{alias}' is not set on '{registry.name}' (aliases set: "
            f"{aliases or 'none'}; versions: {registry.versions()}). Promote a version "
            "first: `python -m demand promote`, or `python -m demand set-champion`."
        )
    return aliases[alias]


def load_model(settings: Settings) -> LoadedModel:
    """Resolve ``serving.model_uri`` to a version, load it and check its inputs.

    Raises:
        ModelLoadError: If the address is not a registered model of this project, the
            store does not exist, the alias is not set, or the model's inputs are not
            ``feature_columns(settings)``.
        PipelineError: If the model or the version is not registered (``Registry``).
    """
    uri = settings.serving.model_uri
    name, target = parse_model_uri(uri)
    expected = settings.mlflow.registered_model_name
    if name != expected:
        raise ModelLoadError(
            f"serving.model_uri names the model '{name}', but the model of this project is "
            f"registered as '{expected}' (mlflow.registered_model_name)."
        )
    tracking_uri = settings.mlflow.tracking_uri
    check_store_exists(tracking_uri)
    registry = Registry(settings)
    alias = target if isinstance(target, str) else None
    version = resolve_alias(registry, target) if isinstance(target, str) else target
    stored = registry.version(version)

    # Where the files of the version are, asked of the configured registry (see the module
    # docstring for why a models:/ address is not used here).
    source = registry.client.get_model_version_download_uri(name, str(version))
    with tempfile.TemporaryDirectory(prefix="demand-model-") as folder:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=source, dst_path=folder, tracking_uri=tracking_uri
        )
        predictor = mlflow.pyfunc.load_model(local)

    signature = predictor.metadata.signature
    columns = [] if signature is None else list(signature.inputs.input_names())
    if columns != feature_columns(settings):
        raise ModelLoadError(
            f"Version {version} of '{name}' takes the inputs {columns}, but the config "
            f"builds {feature_columns(settings)}. Serve a model trained with this config: "
            "train and promote one, or point serving.model_uri at a matching version."
        )
    metric = stored.metrics.get(BACKTEST_MAE_METRIC)
    info = ModelInfo(
        registered_model=name,
        version=version,
        alias=alias,
        model_uri=uri,
        run_id=stored.run_id,
        backtest_mae=metric,
        data_version=stored.tags.get(DATA_VERSION_TAG),
        git_commit=stored.tags.get(GIT_COMMIT_TAG),
        feature_columns=columns,
    )
    return LoadedModel(predictor=predictor, info=info)
