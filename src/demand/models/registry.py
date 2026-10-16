"""The model registry: read what training registered, move the aliases, record why.

``tracking.py`` writes a training run and registers its model as a new version. This
module works on what is already there. It reads a version together with the parameters,
tags and scores of the run that produced it, it moves the two aliases (``champion`` and
``challenger``), and it writes tags onto a version. It never trains, and it never
changes a run. What the promotion gate decides is worked out in ``promotion.py`` from
plain numbers; this module only carries the numbers in and the decision out, so the
rule can be tested without MLflow.

**Why it is not part of ``tracking.py``.** ``tracking.py`` is a dependency of the DVC
stage ``train`` (``dvc.yaml``), and DVC trains again when one byte of it changes. The
gate does not change how a model is trained or recorded, so it lives here, and adding it
did not make the recorded model versions stale.

**Aliases and tags.** An *alias* is a name that points at one version of a registered
model, and it can be moved: ``models:/taxi-demand@champion`` loads whichever version
the alias points at now. ``serving.model_uri`` is such an address, so moving the alias
changes the model that is served, and nothing else has to change. A *tag* on a version
is a label that stays with that version (``promotion.decision = promoted``). The
aliases say what is true now; the tags say what was decided about each version and why.
"""

from __future__ import annotations

from collections.abc import Mapping

import mlflow
from mlflow.exceptions import MlflowException
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.errors import PipelineError

_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"


class StoredVersion(BaseModel):
    """One registered version and what its training run recorded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: int
    run_id: str
    params: dict[str, str]
    """MLflow stores every parameter as text: ``"[48, 168]"``, not a list."""
    tags: dict[str, str]
    metrics: dict[str, float]


class Registry:
    """The registered model named in ``mlflow.registered_model_name``, in its store."""

    def __init__(self, settings: Settings) -> None:
        """Open the registered model.

        Raises:
            PipelineError: If no model is registered under that name in that store.
        """
        config = settings.mlflow
        self.name = config.registered_model_name
        self.client = mlflow.MlflowClient(
            tracking_uri=config.tracking_uri, registry_uri=config.tracking_uri
        )
        try:
            self.client.get_registered_model(self.name)
        except MlflowException as error:
            if error.error_code != _NOT_FOUND:
                raise
            raise PipelineError(
                f"No model is registered under '{self.name}' in {config.tracking_uri}. "
                "Train one first: `dvc repro` (or `python -m demand train`). Check also "
                "that the command runs in the project folder, or that "
                "DEMAND_MLFLOW__TRACKING_URI names the store you mean."
            ) from error

    def versions(self) -> list[int]:
        """The numbers of all registered versions, oldest first."""
        found = self.client.search_model_versions(f"name='{self.name}'")
        return sorted(int(version.version) for version in found)

    def newest(self) -> StoredVersion:
        """The version that was registered last.

        Raises:
            PipelineError: If the registered model has no version.
        """
        numbers = self.versions()
        if not numbers:
            raise PipelineError(
                f"The registered model '{self.name}' has no version. Train one first: "
                "`dvc repro` (or `python -m demand train`)."
            )
        return self.version(numbers[-1])

    def version(self, number: int) -> StoredVersion:
        """One version, with the parameters, tags and scores of its run.

        Raises:
            PipelineError: If there is no such version, or it was registered without a
                training run (by hand, from a file).
        """
        try:
            found = self.client.get_model_version(self.name, str(number))
        except MlflowException as error:
            if error.error_code != _NOT_FOUND:
                raise
            raise PipelineError(
                f"The registered model '{self.name}' has no version {number}. Its versions "
                f"are {self.versions()}."
            ) from error
        if not found.run_id:
            raise PipelineError(
                f"Version {number} of '{self.name}' was registered without a training run, "
                "so nothing says how it was trained or how it scored. Register models "
                "with `python -m demand train` only."
            )
        run = self.client.get_run(found.run_id)
        return StoredVersion(
            version=number,
            run_id=found.run_id,
            params=dict(run.data.params),
            tags=dict(run.data.tags),
            metrics=dict(run.data.metrics),
        )

    def version_tags(self, number: int) -> dict[str, str]:
        """The tags written onto a version (not those of its run)."""
        return dict(self.client.get_model_version(self.name, str(number)).tags)

    def aliases(self) -> dict[str, int]:
        """Every alias that is set, with the version it points at."""
        aliases = self.client.get_registered_model(self.name).aliases
        return {alias: int(version) for alias, version in aliases.items()}

    def set_alias(self, alias: str, version: int) -> None:
        """Point ``alias`` at ``version``; a version it pointed at before loses it."""
        self.client.set_registered_model_alias(self.name, alias, str(version))

    def remove_alias(self, alias: str) -> None:
        """Remove ``alias``. Nothing happens if it is not set."""
        self.client.delete_registered_model_alias(self.name, alias)

    def tag(self, version: int, tags: Mapping[str, str]) -> None:
        """Write ``tags`` onto ``version``, replacing tags of the same name."""
        for key, value in tags.items():
            self.client.set_model_version_tag(self.name, str(version), key, value)
