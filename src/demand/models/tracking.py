"""The one module that talks to MLflow: it records a training run and its model.

The forecaster, the backtest and the training code do not import ``mlflow``. They hand
this module plain values (a fitted forecaster, dictionaries of parameters, metrics and
tags, a list of files) and get back the id of the run and the version of the model. So
MLflow can be replaced, or left out of a test, without touching the modelling code.

**Where it is written.** ``mlflow.tracking_uri`` says where runs are recorded:
``sqlite:///mlflow.db`` in the config of this project, one database file in the working
directory. Files (the model, the reports) are not stored in the database. With a
database on this machine and no MLflow server, MLflow puts them in the folder
``mlruns`` of the working directory. Both are ignored by git. The database stores the
*absolute* path of that folder when the experiment is created, so the project folder
cannot be moved afterwards without losing the link between a run and its files.

**What a run holds.** MLflow has four kinds of record, and each value goes where it can
be used later:

* a *parameter* is a setting that was chosen before the run (the learning rate, the
  lags). Two runs are compared by their parameters.
* a *metric* is a number the run produced (the backtest MAE). Runs are sorted and
  charted by their metrics.
* a *tag* is a label that says where the run came from (the git commit, the version of
  the data). Without the tags a good number cannot be reproduced.
* an *artifact* is a file (the model, the backtest report).

**The model is stored as the forecaster itself.** MLflow can store a LightGBM model
directly, but the forecaster of this project does more than LightGBM does: it puts the
columns in the right order, refuses rows it must not forecast, and never returns a
negative number (``gbm.py``). A service that loaded the bare LightGBM model would have
to repeat those steps, and a step that is written twice is sooner or later different in
the two places. So the object that is stored is the fitted forecaster, wrapped in
``ForecasterModel``: the model that is served is the object the backtest scored. This
is MLflow's "python function" flavour; it is loaded with ``mlflow.pyfunc.load_model``.
The price: the file is a Python pickle, so loading it needs this package (and the
versions of LightGBM and pandas named in the ``requirements.txt`` MLflow stores next to
it), and a pickle must only be loaded from a store that is trusted. MLflow prints a
warning about exactly that on every run.

**The signature.** The model is stored with a *signature*: the names and types of its
input columns, in the order of ``feature_columns``, and the type of its output. When
the model is loaded through ``mlflow.pyfunc``, MLflow checks every table against it
before the forecaster sees it: columns are put in the stored order, a missing column is
refused, and so is a column of the wrong type (a whole-number column that arrives as a
float, for example). A column the signature does not name is dropped, with a warning.
An *input example*, the first rows of the training table, is stored next to the model
and shows what a valid request looks like.

**The registry.** The model is also registered under ``mlflow.registered_model_name``,
which gives it a version number: 1 for the first training run, 2 for the second. The
number is the handle by which a later step can say "serve this one". No alias
(``champion``, ``challenger``) is set here: which version is served is decided by the
promotion rule, which is not built yet.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.data.features import FloatArray
from demand.models.baseline import Forecaster

MODEL_ARTIFACT_NAME = "model"
"""The name of the stored model inside its run."""


class LoggedRun(BaseModel):
    """What MLflow answered: where the run and its model can be found again."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    model_version: int
    """The version the registry gave the model; ``models:/<name>/<version>`` loads it."""


class ForecasterModel(PythonModel):
    """A fitted forecaster in the form MLflow stores and loads."""

    def __init__(self, forecaster: Forecaster) -> None:
        self.forecaster = forecaster

    def predict(
        self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> FloatArray:
        """The forecast of the wrapped forecaster for every row of ``model_input``.

        ``context`` and ``params`` belong to MLflow's interface and are not used.
        """
        return self.forecaster.predict(model_input)


def log_training_run(
    settings: Settings,
    *,
    forecaster: Forecaster,
    input_example: pd.DataFrame,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    artifacts: Sequence[Path],
) -> LoggedRun:
    """Record one training run in MLflow and register its model.

    Args:
        settings: Project settings (``mlflow``: where to record, and under which names).
        forecaster: The fitted forecaster to store.
        input_example: A few rows of model inputs, exactly as the forecaster takes them.
            They are stored with the model, and the signature is read from them.
        params: The settings that define the run, by name.
        metrics: The numbers the run produced, by name.
        tags: Where the run came from, by name.
        artifacts: Files to store with the run. Every one must exist.

    Returns:
        The id of the run and the version the registry gave the model.
    """
    config = settings.mlflow
    mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)
    with mlflow.start_run() as run:
        mlflow.set_tags(dict(tags))
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        for path in artifacts:
            mlflow.log_artifact(str(path))
        logged = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT_NAME,
            python_model=ForecasterModel(forecaster),
            signature=infer_signature(input_example, forecaster.predict(input_example)),
            input_example=input_example,
            registered_model_name=config.registered_model_name,
        )
    return LoggedRun(run_id=run.info.run_id, model_version=int(logged.registered_model_version))
