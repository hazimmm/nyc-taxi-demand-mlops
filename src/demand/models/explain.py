"""Explaining the final model: which inputs move its forecasts, and by how much (SHAP).

Layout of the files (``settings.paths.reports_dir``)::

    reports/
      shap_importance.json    the mean size of every input's contribution
      shap_summary.png        one dot per explained row and input

Both are written by a training run, after the final fit, and stored with the MLflow
run. This module and ``tracking.py`` are the edge of the modelling code: only here are
``shap`` and ``matplotlib`` imported, and only ``training.py`` imports this module, so
the backtest and the tuning never load them.

**What a SHAP value is.** A forecast of the model is a sum. It starts from a *base
value*, the output the model gives on average over its training rows, and every input
of the row then adds or takes away its *contribution*: "``lag_168h`` is high, plus
0.9", "it is 4 o'clock at night, minus 1.2". Those contributions are the SHAP values of
the row. They are worked out from the trees themselves, exactly, and they always add
up: base value plus all contributions of a row is the output of the model for that row.
``explain`` checks that for every row it explains and stops if it does not hold.

**The scale is the logarithm of the pickups.** With the ``poisson`` objective the trees
do not add up to pickups but to the *logarithm* of pickups, and the forecast is ``exp``
of the sum. SHAP values live on that scale, so they are read as factors, not as
pickups: a contribution of +0.69 doubles the forecast (``exp(0.69) = 2``), -0.69 halves
it, +0.1 is about 10 percent more. The same +0.69 is 5 more pickups in a quiet hour and
200 more in a busy one. ``exp`` of the base value alone is a typical forecast (it is
lower than the mean of the pickups: the mean of logarithms gives less weight to the
busiest hours). With another objective the scale is the pickups themselves.

**The zone is a label.** ``zone_id`` is a category for the model (``gbm.py``), and its
SHAP value is correct as it is: "this is zone 132" moves the forecast by so much. What
has no meaning is the *size of the number* 132. The summary plot colours every dot by
the value of the input, from low (blue) to high (red); for the zone that would be a
colour for the label, so its dots are drawn grey.

**The mean absolute SHAP value** of an input is the average size of its contribution
over the explained rows, whatever the direction. It answers "how much does this input
move a typical forecast?". It is not the *gain* that LightGBM reports for an input,
which answers "how much did splits on this input reduce the training error?". The two
usually rank the inputs alike but they are different questions: the gain counts an
error in a busy hour more than one at night, and a contribution on the log scale
counts every row the same.

**What SHAP does not say.** It explains the model, not the city. A large contribution
of ``lag_168h`` means the model leans on last week, not that last week causes this
week. And inputs that say nearly the same thing (the two lags, the rolling means)
share the credit in a way that depends on which of them the trees happened to split on.

**A sample of rows, always the same one.** The exact calculation takes long for all
53,280 training rows: about 80 seconds with 600 trees of 63 leaves and about 160 with
the tuned 110 leaves, against about 8 and 15 seconds for 5,000 rows (measured on one
machine; larger trees take longer). On 5,000 rows the mean absolute values of the five
strongest inputs were within about 1 percent of those on all rows, and a plot with
53,280 dots per input shows nothing a plot with 5,000 does not. The sample is drawn from
the rows of the final fit with ``project.seed`` and kept in the order of the table, so
the same model and the same table always give the same numbers. The rows are training
rows: the question is how the model works, not how well, and for that no row has to be
held back.

**What is promised to be identical.** ``shap_importance.json`` is, byte for byte, for
the same model on the same machine: its numbers are rounded to six decimals. The
picture shows the same dots (their random vertical spread is drawn from the seed), but
its bytes are not promised: another version of matplotlib or of a font draws the same
picture with other bytes.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from pydantic import BaseModel, ConfigDict

from demand.data.features import FloatArray
from demand.data.silver import report_to_json, write_text
from demand.errors import PipelineError
from demand.logs import get_logger
from demand.models.gbm import LightGbmForecaster, ModelInputError
from demand.schemas import ZONE_ID

log = get_logger(__name__)

SHAP_REPORT_NAME = "shap_importance.json"
SHAP_PLOT_NAME = "shap_summary.png"
SHAP_SAMPLE_ROWS = 5000
"""How many training rows are explained. See the module docstring for the reason."""
ADDITIVITY_TOLERANCE = 1e-6
"""The largest gap allowed between the sum of a row's contributions and its output."""
LOG_OBJECTIVES = frozenset({"poisson", "gamma", "tweedie"})
"""Objectives for which the trees add up to the logarithm of the forecast."""
_DECIMALS = 6
_PLOT_DPI = 150


class ShapReport(BaseModel):
    """How much every input moves the forecasts; saved as JSON and stored with the run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rows: int
    """Rows that were explained."""
    seed: int
    """The seed the rows were drawn with."""
    objective: str
    """The objective of the model. ``poisson``: all values are on the log scale."""
    base_value: float
    """The output of the model before any input has contributed."""
    mean_abs_shap: dict[str, float]
    """The mean size of the contribution of every input, the largest first."""
    share_percent: dict[str, float]
    """The same as a share of the sum over all inputs."""


@dataclass(frozen=True)
class Explanation:
    """The SHAP values of some rows."""

    inputs: pd.DataFrame
    """The model inputs of the explained rows."""
    values: FloatArray
    """One row per explained row, one column per input, in the order of ``inputs``."""
    base_value: float


# --------------------------------------------------------------------------------------
# Pure helpers: no disk
# --------------------------------------------------------------------------------------


def shap_report_path(reports_dir: Path) -> Path:
    """Where the mean absolute SHAP values are stored."""
    return reports_dir / SHAP_REPORT_NAME


def shap_plot_path(reports_dir: Path) -> Path:
    """Where the summary plot is stored."""
    return reports_dir / SHAP_PLOT_NAME


def sample_rows(table: pd.DataFrame, rows: int, seed: int) -> pd.DataFrame:
    """At most ``rows`` rows of ``table``, drawn with ``seed``, in the order of the table."""
    if len(table) <= rows:
        return table
    return table.sample(n=rows, random_state=seed).sort_index()


def explain(forecaster: LightGbmForecaster, rows: pd.DataFrame) -> Explanation:
    """The contribution of every input to the output of the model, for every row.

    Raises:
        ModelInputError: If the forecaster is not fitted.
        PipelineError: If the contributions of a row and the base value do not add up
            to the output of the model for that row.
    """
    model = forecaster.model
    if model is None:
        raise ModelInputError(
            "The LightGBM forecaster has not been fitted, so there is nothing to explain. "
            "Call fit with rows of the gold feature table before explain."
        )
    inputs = rows[forecaster.columns]
    explainer = shap.TreeExplainer(model)
    values: FloatArray = np.asarray(explainer.shap_values(inputs), dtype="float64")
    base_value = float(np.ravel(explainer.expected_value)[0])

    output = np.asarray(model.predict(inputs, raw_score=True), dtype="float64")
    gap = float(np.abs(values.sum(axis=1) + base_value - output).max())
    if gap > ADDITIVITY_TOLERANCE:
        raise PipelineError(
            f"The SHAP values do not add up to the output of the model: the largest gap "
            f"is {gap:.3g}, allowed is {ADDITIVITY_TOLERANCE}. The installed versions of "
            "shap and lightgbm do not work together. Install the versions of "
            "requirements.lock (`python -m pip install -r requirements.lock`) and train again."
        )
    return Explanation(inputs=inputs, values=values, base_value=base_value)


def shap_report(explanation: Explanation, seed: int, objective: str) -> ShapReport:
    """The mean absolute SHAP value of every input, the largest first."""
    means = np.abs(explanation.values).mean(axis=0)
    total = float(means.sum())
    by_input = dict(zip(explanation.inputs.columns, means.tolist(), strict=True))
    # Sorted by size and then by name, so that two inputs of equal size keep their order.
    ordered = sorted(by_input.items(), key=lambda item: (-item[1], item[0]))
    return ShapReport(
        rows=len(explanation.inputs),
        seed=seed,
        objective=objective,
        base_value=round(explanation.base_value, _DECIMALS),
        mean_abs_shap={name: round(mean, _DECIMALS) for name, mean in ordered},
        share_percent={
            name: round(mean / total * 100, 2) if total else 0.0 for name, mean in ordered
        },
    )


def summary_plot(explanation: Explanation, seed: int, objective: str) -> bytes:
    """The SHAP summary plot of ``explanation`` as a PNG image.

    One line per input, the strongest at the top; one dot per explained row. A dot to
    the right of 0 is a row whose forecast this input pushed up. Its colour is the
    value of the input in that row; the dots of the zone are grey (see the module
    docstring). The label under the plot says which scale the values are on, which
    depends on ``objective``. No window is opened: the picture is drawn in memory.
    """
    # "Agg" draws into memory. It has to be chosen before the first figure exists.
    plt.switch_backend("Agg")
    colours = explanation.inputs.astype("float64")
    # A missing value is what the plot draws in grey.
    colours[ZONE_ID] = np.nan
    with warnings.catch_warnings():
        # The plot looks for the lowest and the highest value of every input to choose
        # the colours, and numpy warns that the zone has none. That is intended here.
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        shap.summary_plot(
            explanation.values,
            features=colours,
            max_display=len(colours.columns),
            show=False,
            # The dots are spread out vertically at random.
            rng=np.random.default_rng(seed),
        )
    figure = plt.gcf()
    if objective in LOG_OBJECTIVES:
        plt.xlabel("SHAP value: added to the logarithm of the forecast (+0.69 doubles it)")
    else:
        plt.xlabel("SHAP value: pickups added to the forecast")
    plt.title(f"{len(colours)} training rows. {ZONE_ID} is a label and has no colour.")
    image = io.BytesIO()
    # No "Software" entry: it holds the version of matplotlib and nothing else.
    figure.savefig(
        image, format="png", dpi=_PLOT_DPI, bbox_inches="tight", metadata={"Software": None}
    )
    plt.close(figure)
    return image.getvalue()


# --------------------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------------------


def explain_to_reports(
    forecaster: LightGbmForecaster, table: pd.DataFrame, reports_dir: Path, seed: int
) -> ShapReport:
    """Explain a sample of ``table`` and write the report and the plot.

    Args:
        forecaster: The fitted model.
        table: The rows the model was fitted on.
        reports_dir: Where ``shap_importance.json`` and ``shap_summary.png`` go.
        seed: Seed for the sample of rows and for the spread of the dots.

    Raises:
        ModelInputError: If the forecaster is not fitted.
        PipelineError: If the contributions do not add up to the output of the model.
    """
    explanation = explain(forecaster, sample_rows(table, SHAP_SAMPLE_ROWS, seed))
    objective = str(forecaster.params.get("objective", "regression"))
    report = shap_report(explanation, seed, objective)
    write_text(report_to_json(report), shap_report_path(reports_dir))
    shap_plot_path(reports_dir).write_bytes(summary_plot(explanation, seed, objective))
    log.info(
        "model explained",
        extra={
            "rows": report.rows,
            "base_value": report.base_value,
            "strongest": dict(list(report.mean_abs_shap.items())[:5]),
        },
    )
    return report
