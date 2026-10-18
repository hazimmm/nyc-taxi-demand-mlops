"""Command-line entry point: ``python -m demand <command>``.

Each command resolves the settings, configures logging and hands over to one stage
function. The stage functions are the contract between this module and the rest of the
package: every one takes the ``Settings`` object as its first argument and returns
``None``.
"""

from __future__ import annotations

import importlib
import importlib.util
from datetime import datetime
from enum import StrEnum
from typing import Annotated

import typer

from demand.config import get_settings
from demand.errors import PipelineError
from demand.logs import configure_logging

app = typer.Typer(
    help="Hourly taxi demand forecasts for NYC zones: every hour of the next day, a day before.",
    no_args_is_help=True,
    add_completion=False,
)


class Layer(StrEnum):
    """Data layers that have a quality gate."""

    SILVER = "silver"
    GOLD = "gold"


def _not_implemented(module: str) -> typer.Exit:
    typer.secho(f"Not implemented yet: {module}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _run_stage(module: str, function: str, *args: object) -> None:
    """Import ``module`` lazily and call ``function(settings, *args)``."""
    if importlib.util.find_spec(module) is None:
        raise _not_implemented(module)
    configure_logging()
    try:
        getattr(importlib.import_module(module), function)(get_settings(), *args)
    except NotImplementedError as error:
        # The module exists but this stage is still a placeholder.
        raise _not_implemented(str(error) or f"{module}.{function}") from error
    except PipelineError as error:
        # An expected failure: the message is enough, a traceback would only hide it.
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


@app.command()
def config() -> None:
    """Print the resolved configuration as JSON."""
    typer.echo(get_settings().model_dump_json(indent=2))


@app.command()
def ingest() -> None:
    """Download trip and weather files into the raw layer."""
    _run_stage("demand.data.pipeline", "run_ingest")


@app.command("select-zones")
def select_zones() -> None:
    """Print the busiest pickup zones of the training period, to freeze in the config."""
    _run_stage("demand.data.pipeline", "run_select_zones")


@app.command()
def silver() -> None:
    """Clean the raw layer into the silver layer."""
    _run_stage("demand.data.pipeline", "run_silver")


@app.command()
def gold() -> None:
    """Aggregate the silver layer into hourly demand and build the feature table."""
    _run_stage("demand.data.pipeline", "run_gold")


@app.command()
def validate(layer: Annotated[Layer, typer.Argument(help="Layer to check.")]) -> None:
    """Run the quality gate of a data layer; exits non-zero if it fails."""
    _run_stage("demand.data.pipeline", "run_validate", layer.value)


@app.command()
def backtest() -> None:
    """Score the forecasters on past weeks they have not seen and write a report."""
    _run_stage("demand.models.pipeline", "run_backtest")


@app.command()
def train() -> None:
    """Backtest the forecasters, fit the final model and record it in MLflow."""
    _run_stage("demand.models.pipeline", "run_training")


@app.command()
def tune() -> None:
    """Search for better model parameters on the backtest folds and print them for the config."""
    _run_stage("demand.models.pipeline", "run_tuning")


@app.command()
def promote() -> None:
    """Make the newest model version champion if it beats the champion and the baseline."""
    _run_stage("demand.models.promotion", "run_promotion")


@app.command("set-champion")
def set_champion(
    version: Annotated[int, typer.Argument(help="The model version to serve.")],
    reason: Annotated[str, typer.Option(help="Why the gate is bypassed; recorded on the version.")],
) -> None:
    """Make a model version champion by hand, without the gate (to roll back)."""
    _run_stage("demand.models.promotion", "set_champion", version, reason)


@app.command()
def forecast(
    days: Annotated[
        int,
        typer.Option(
            min=1,
            help="Steps of the simulated clock to run; stops early at the end of the data.",
        ),
    ] = 1,
) -> None:
    """Forecast the next day of the replay and score the days that are over."""
    _run_stage("demand.flows.forecast_flow", "run_forecast", days)


@app.command()
def monitor(
    as_of: Annotated[
        datetime | None,
        typer.Option(
            "--as-of",
            formats=["%Y-%m-%d"],
            help="Evaluate at 00:00 on this day (YYYY-MM-DD) instead of at the replay clock.",
        ),
    ] = None,
    push_to: Annotated[
        str | None,
        typer.Option(help="Pushgateway to send the numbers to, e.g. http://localhost:9091."),
    ] = None,
) -> None:
    """Check input drift, prediction drift and live accuracy."""
    _run_stage(
        "demand.flows.monitoring_flow",
        "run_monitoring",
        None if as_of is None else as_of.date(),
        push_to,
    )


@app.command()
def retrain() -> None:
    """Retrain if live accuracy has alerted, and promote the new model if it passes the gate.

    Monitoring is evaluated first, as of the newest forecast's issue moment, and its report
    is rewritten.
    """
    _run_stage("demand.flows.retrain_flow", "run_retrain")


@app.command()
def serve() -> None:
    """Start the prediction API."""
    module = "demand.serving.app"
    if importlib.util.find_spec(module) is None:
        raise _not_implemented(module)
    import uvicorn

    configure_logging()
    settings = get_settings()
    uvicorn.run(f"{module}:app", host=settings.serving.host, port=settings.serving.port)
