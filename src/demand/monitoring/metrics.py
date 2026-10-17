"""The numbers of the monitoring report, for Prometheus, through a Pushgateway.

**Why a Pushgateway.** Prometheus *pulls*: it asks a running program for its page of
numbers. Monitoring is not a running program; it is a batch command that runs once a
day, writes its report and ends, and there is nothing left to ask. The Pushgateway is
Prometheus's own answer for this case, a small server that keeps the last numbers a
batch job sent it and shows them to Prometheus like any other target
(``monitoring/prometheus/prometheus.yml``). The two other ways were heavier for the same
result: a ``/metrics`` exporter of our own would be one more long-running program of our
code to build and keep up, only to read one file; node-exporter's text files come with
an exporter of machine statistics that this project does not need.

**What is sent.** ``registry_of`` turns a report into these gauges (no labels unless
said):

* ``demand_monitoring_now_timestamp_seconds``: the simulated moment of the evaluation,
  as Unix time. Prometheus stamps every number with the *wall-clock* time it collected
  it; this gauge says which simulated day the numbers belong to.
* ``demand_monitoring_evaluated{check}``: 1 or 0 for ``input_drift``,
  ``prediction_drift`` and ``accuracy``.
* ``demand_monitoring_alert{check="accuracy"}``: 1 when accuracy raised an alert, the
  only check that can (``checks.py`` says why).
* ``demand_monitoring_warning{check}``: 1 when ``input_drift`` or ``prediction_drift``
  found drift, a warning that nothing acts on.
* ``demand_monitoring_input_drift_share`` and ``..._share_threshold``;
  ``demand_monitoring_input_drift_distance{column}`` for each tested input.
* ``demand_monitoring_prediction_drift_distance`` and
  ``demand_monitoring_drift_distance_threshold``.
* ``demand_monitoring_live_mae``, ``..._baseline_mae``, ``..._backtest_mae``,
  ``demand_monitoring_live_to_backtest_mae`` and ``..._live_to_backtest_mae_threshold``.

A check that was not evaluated sends its ``evaluated`` and its ``alert`` or ``warning``
(all 0) and none of its numbers: an empty panel says "no number", where a 0 would claim
one. Every push replaces all the numbers of the job (HTTP PUT), so nothing of an older
report is left.
"""

from __future__ import annotations

from collections.abc import Callable

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from demand.errors import PipelineError
from demand.monitoring.checks import ACCURACY, INPUT_DRIFT, PREDICTION_DRIFT, MonitoringReport

JOB = "demand_monitoring"
"""The ``job`` label the numbers carry in Prometheus."""
PREFIX = "demand_monitoring_"
_TIMEOUT_SECONDS = 10

Push = Callable[..., None]
"""``prometheus_client.push_to_gateway``; tests pass their own."""


class PushError(PipelineError):
    """The Pushgateway could not be reached; the report on disk is written all the same."""


def _gauge(
    registry: CollectorRegistry, name: str, text: str, labels: tuple[str, ...] = ()
) -> Gauge:
    return Gauge(PREFIX + name, text, labels, registry=registry)


def _set(registry: CollectorRegistry, name: str, text: str, value: float | None) -> None:
    """A gauge without labels, left out when there is no value."""
    if value is not None:
        _gauge(registry, name, text).set(value)


def registry_of(report: MonitoringReport) -> CollectorRegistry:
    """The gauges of one report, in a registry of their own."""
    registry = CollectorRegistry()
    _set(
        registry,
        "now_timestamp_seconds",
        "Simulated moment of the evaluation.",
        report.now.timestamp(),
    )
    evaluated = _gauge(registry, "evaluated", "1 if the check was evaluated.", ("check",))
    alert = _gauge(registry, "alert", "1 if the check raised an alert.", ("check",))
    warning = _gauge(registry, "warning", "1 if the check found drift.", ("check",))
    inputs, predictions, accuracy = report.input_drift, report.prediction_drift, report.accuracy
    for check, result in ((INPUT_DRIFT, inputs), (PREDICTION_DRIFT, predictions)):
        evaluated.labels(check=check).set(int(result.evaluated))
        warning.labels(check=check).set(int(bool(result.drifted)))
    evaluated.labels(check=ACCURACY).set(int(accuracy.evaluated))
    alert.labels(check=ACCURACY).set(int(accuracy.alert))

    _set(registry, "input_drift_share", "Share of drifted model inputs.", inputs.drift_share)
    _set(
        registry,
        "input_drift_share_threshold",
        "Share of drifted inputs that raises a warning.",
        inputs.share_threshold,
    )
    if inputs.columns:
        distance = _gauge(
            registry, "input_drift_distance", "Normed Wasserstein distance per input.", ("column",)
        )
        for drift in inputs.columns:
            distance.labels(column=drift.column).set(drift.distance)
    _set(
        registry,
        "prediction_drift_distance",
        "Normed Wasserstein distance of the forecasts.",
        predictions.distance,
    )
    _set(
        registry,
        "drift_distance_threshold",
        "Distance at which a column has drifted.",
        predictions.threshold,
    )
    _set(registry, "live_mae", "Live MAE of the window.", accuracy.live_mae)
    _set(registry, "baseline_mae", "Weekly baseline MAE on the same rows.", accuracy.baseline_mae)
    _set(registry, "backtest_mae", "Backtest MAE of the model version.", accuracy.backtest_mae)
    _set(
        registry,
        "live_to_backtest_mae",
        "Live MAE / backtest MAE.",
        accuracy.live_to_backtest_mae,
    )
    _set(
        registry,
        "live_to_backtest_mae_threshold",
        "Live MAE / backtest MAE that raises an alert.",
        accuracy.ratio_threshold,
    )
    return registry


def push(report: MonitoringReport, gateway: str, send: Push = push_to_gateway) -> None:
    """Send the gauges of ``report`` to the Pushgateway at ``gateway``, replacing the last.

    Raises:
        PushError: If the Pushgateway does not answer, or answers with an error.
    """
    try:
        send(gateway, job=JOB, registry=registry_of(report), timeout=_TIMEOUT_SECONDS)
    except OSError as error:
        raise PushError(
            f"Could not send the monitoring numbers to the Pushgateway at {gateway}: "
            f"{error}. The report on disk is written. Start it (`docker compose up -d "
            "pushgateway`) and run the command again."
        ) from error
