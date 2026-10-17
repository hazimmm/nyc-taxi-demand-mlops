from __future__ import annotations

from datetime import date
from typing import Any
from urllib.error import URLError

import pandas as pd
import pytest
from prometheus_client import generate_latest

from demand.config import Settings
from demand.errors import PipelineError
from demand.monitoring.checks import (
    BASELINE,
    DAY,
    MonitoringReport,
    accuracy_check,
    input_drift_check,
    monitoring_report,
    prediction_drift_check,
)
from demand.monitoring.metrics import JOB, PushError, push, registry_of
from demand.schemas import ACTUAL, FORECAST, MODEL_VERSION
from tests.unit.monitoring.conftest import days_from, grid, numbers, training_table

NOW = "2025-05-10"


def reference(settings: Settings) -> pd.DataFrame:
    table = training_table(settings)
    return table.assign(**{DAY: table["hour_start"].dt.date}).iloc[-3 * 24 * 28 :]


def report_with_accuracy(settings: Settings) -> MonitoringReport:
    """Input drift evaluated, prediction drift not, accuracy evaluated with ratio 1.5."""
    window = numbers(grid(settings, days_from(date(2025, 5, 2), 7)), settings, 7, {})
    inputs, _ = input_drift_check(window, reference(settings), settings, None)
    rows = pd.DataFrame(
        {
            DAY: [date(2025, 5, 2)] * 2,
            FORECAST: [130.0, 70.0],
            ACTUAL: [100.0, 100.0],
            BASELINE: [100.0, 120.0],
            MODEL_VERSION: "2",
        }
    )
    return monitoring_report(
        pd.Timestamp(NOW, tz=settings.project.timezone),
        settings,
        inputs,
        prediction_drift_check(None, None, None, None, "too few days"),
        accuracy_check(rows, 20.0, settings, None),
    )


def lines(report: MonitoringReport) -> list[str]:
    text: bytes = generate_latest(registry_of(report))
    return [line for line in text.decode().splitlines() if not line.startswith("#")]


def test_the_numbers_of_a_report_in_the_text_format_of_prometheus(settings: Settings) -> None:
    found = lines(report_with_accuracy(settings))

    now = pd.Timestamp(NOW, tz=settings.project.timezone).timestamp()
    (stamp,) = [line for line in found if line.startswith("demand_monitoring_now_timestamp")]
    assert float(stamp.split(" ")[1]) == now
    assert 'demand_monitoring_evaluated{check="input_drift"} 1.0' in found
    assert 'demand_monitoring_evaluated{check="prediction_drift"} 0.0' in found
    assert 'demand_monitoring_evaluated{check="accuracy"} 1.0' in found
    assert 'demand_monitoring_warning{check="input_drift"} 0.0' in found
    assert 'demand_monitoring_warning{check="prediction_drift"} 0.0' in found
    assert 'demand_monitoring_alert{check="accuracy"} 1.0' in found
    assert not any(
        line.startswith('demand_monitoring_alert{check="input_drift"}') for line in found
    )
    assert not any(line.startswith('demand_monitoring_warning{check="accuracy"}') for line in found)
    assert "demand_monitoring_input_drift_share 0.0" in found
    assert "demand_monitoring_input_drift_share_threshold 0.3" in found
    (lag,) = [
        line
        for line in found
        if line.startswith('demand_monitoring_input_drift_distance{column="lag_48h"}')
    ]
    assert 0.0 <= float(lag.split(" ")[1]) < 0.1
    assert "demand_monitoring_live_mae 30.0" in found
    assert "demand_monitoring_baseline_mae 10.0" in found
    assert "demand_monitoring_backtest_mae 20.0" in found
    assert "demand_monitoring_live_to_backtest_mae 1.5" in found
    assert "demand_monitoring_live_to_backtest_mae_threshold 1.25" in found
    assert "demand_monitoring_drift_distance_threshold 0.1" in found


def test_a_check_that_was_not_evaluated_sends_no_numbers(settings: Settings) -> None:
    found = lines(report_with_accuracy(settings))

    assert not any(line.startswith("demand_monitoring_prediction_drift_distance") for line in found)


def test_a_report_without_any_evaluated_check_sends_flags_only(settings: Settings) -> None:
    inputs, _ = input_drift_check(None, reference(settings), settings, "none")
    report = monitoring_report(
        pd.Timestamp(NOW, tz=settings.project.timezone),
        settings,
        inputs,
        prediction_drift_check(None, None, None, None, "none"),
        accuracy_check(None, None, settings, "none"),
    )

    names = {line.split("{")[0].split(" ")[0] for line in lines(report)}

    assert names == {
        "demand_monitoring_now_timestamp_seconds",
        "demand_monitoring_evaluated",
        "demand_monitoring_alert",
        "demand_monitoring_warning",
        "demand_monitoring_input_drift_share_threshold",
        "demand_monitoring_drift_distance_threshold",
        "demand_monitoring_live_to_backtest_mae_threshold",
    }


def test_push_sends_the_registry_under_the_job(settings: Settings) -> None:
    sent: dict[str, Any] = {}

    def send(gateway: str, **kwargs: Any) -> None:
        sent.update(kwargs, gateway=gateway)

    push(report_with_accuracy(settings), "http://localhost:9091", send)

    assert sent["gateway"] == "http://localhost:9091"
    assert sent["job"] == JOB
    assert b"demand_monitoring_live_mae 30.0" in generate_latest(sent["registry"])


def test_a_pushgateway_that_does_not_answer_is_a_pipeline_error(settings: Settings) -> None:
    def send(gateway: str, **kwargs: Any) -> None:
        raise URLError("connection refused")

    with pytest.raises(PushError, match="docker compose up -d pushgateway") as caught:
        push(report_with_accuracy(settings), "http://localhost:9", send)
    assert isinstance(caught.value, PipelineError)
