from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from datetime import date

import pytest
from typer.testing import CliRunner

from demand.cli import _run_stage, app
from demand.config import CONFIG_FILE_ENV, Settings, get_settings
from demand.errors import QualityGateError
from tests.conftest import CONFIG_FILE

runner = CliRunner()


@pytest.fixture(autouse=True)
def _default_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv(CONFIG_FILE_ENV, str(CONFIG_FILE))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _fake_module(monkeypatch: pytest.MonkeyPatch, name: str, **functions: object) -> None:
    module = types.ModuleType(name)
    module.__spec__ = types.SimpleNamespace(name=name)  # type: ignore[assignment]
    for function_name, function in functions.items():
        setattr(module, function_name, function)
    monkeypatch.setitem(sys.modules, name, module)


def test_config_command_prints_resolved_settings() -> None:
    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["project"]["name"] == "nyc-taxi-demand"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage" in result.output


def test_stage_receives_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(monkeypatch, "demand.data.fake_stage", run=lambda *args: calls.append(args))

    _run_stage("demand.data.fake_stage", "run", "silver")

    assert len(calls) == 1
    assert isinstance(calls[0][0], Settings)
    assert calls[0][1] == "silver"


def test_validate_passes_the_layer_to_the_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(monkeypatch, "demand.data.pipeline", run_validate=lambda *args: calls.append(args))

    result = runner.invoke(app, ["validate", "gold"])

    assert result.exit_code == 0
    assert calls[0][1] == "gold"


def test_set_champion_passes_the_version_and_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(
        monkeypatch, "demand.models.promotion", set_champion=lambda *args: calls.append(args)
    )

    result = runner.invoke(app, ["set-champion", "1", "--reason", "version 2 failed in May"])

    assert result.exit_code == 0, result.output
    assert calls[0][1:] == (1, "version 2 failed in May")


def test_forecast_runs_one_step_unless_told_otherwise(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(
        monkeypatch, "demand.flows.forecast_flow", run_forecast=lambda *args: calls.append(args)
    )

    runner.invoke(app, ["forecast"])
    runner.invoke(app, ["forecast", "--days", "61"])

    assert [call[1:] for call in calls] == [(1,), (61,)]


def test_monitor_passes_the_day_and_the_pushgateway(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(
        monkeypatch, "demand.flows.monitoring_flow", run_monitoring=lambda *args: calls.append(args)
    )

    runner.invoke(app, ["monitor"])
    runner.invoke(app, ["monitor", "--as-of", "2025-05-29", "--push-to", "http://localhost:9091"])

    assert [call[1:] for call in calls] == [
        (None, None),
        (date(2025, 5, 29), "http://localhost:9091"),
    ]


def test_monitor_refuses_a_day_in_another_format() -> None:
    result = runner.invoke(app, ["monitor", "--as-of", "29-05-2025"])

    assert result.exit_code == 2


def test_schedule_passes_the_schedule_on(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(
        monkeypatch, "demand.flows.orchestration", serve_daily_step=lambda *args: calls.append(args)
    )

    runner.invoke(app, ["schedule", "--every-minutes", "2", "--push-to", "http://gw:9091"])
    runner.invoke(app, ["schedule", "--cron", "0 1 * * *", "--days", "3"])

    assert [call[1:] for call in calls] == [
        (2, None, 1, "http://gw:9091"),
        (None, "0 1 * * *", 3, None),
    ]


def test_forecast_refuses_zero_steps() -> None:
    result = runner.invoke(app, ["forecast", "--days", "0"])

    assert result.exit_code == 2


def test_set_champion_needs_a_reason() -> None:
    result = runner.invoke(app, ["set-champion", "1"])

    assert result.exit_code == 2


def test_validate_rejects_unknown_layer() -> None:
    result = runner.invoke(app, ["validate", "bronze"])

    assert result.exit_code == 2


@pytest.mark.parametrize(
    ("command", "module", "function"),
    [
        ("ingest", "demand.data.pipeline", "run_ingest"),
        ("select-zones", "demand.data.pipeline", "run_select_zones"),
        ("silver", "demand.data.pipeline", "run_silver"),
        ("gold", "demand.data.pipeline", "run_gold"),
        ("backtest", "demand.models.pipeline", "run_backtest"),
        ("train", "demand.models.pipeline", "run_training"),
        ("tune", "demand.models.pipeline", "run_tuning"),
        ("promote", "demand.models.promotion", "run_promotion"),
        ("forecast", "demand.flows.forecast_flow", "run_forecast"),
        ("monitor", "demand.flows.monitoring_flow", "run_monitoring"),
        ("retrain", "demand.flows.retrain_flow", "run_retrain"),
    ],
)
def test_each_command_calls_its_stage(
    monkeypatch: pytest.MonkeyPatch, command: str, module: str, function: str
) -> None:
    calls: list[tuple[object, ...]] = []
    _fake_module(monkeypatch, module, **{function: lambda *args: calls.append(args)})

    result = runner.invoke(app, [command])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1


def test_unimplemented_stage_exits_with_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("demand.cli.importlib.util.find_spec", lambda name: None)

    result = runner.invoke(app, ["ingest"])

    assert result.exit_code == 1
    assert "Not implemented yet" in result.output


def test_placeholder_stage_exits_with_an_error_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def placeholder(*args: object) -> None:
        raise NotImplementedError("validate arrives later")

    _fake_module(monkeypatch, "demand.data.pipeline", run_validate=placeholder)

    result = runner.invoke(app, ["validate", "gold"])

    assert result.exit_code == 1
    assert "Not implemented yet: validate arrives later" in result.output


def test_expected_failure_prints_one_line_instead_of_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_gate(*args: object) -> None:
        raise QualityGateError("silver did not pass its quality gate: 2 checks failed")

    _fake_module(monkeypatch, "demand.data.pipeline", run_validate=failing_gate)

    result = runner.invoke(app, ["validate", "silver"])

    assert result.exit_code == 1
    assert "Error: silver did not pass its quality gate: 2 checks failed" in result.output
    assert "Traceback" not in result.output


def test_unexpected_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def bug(*args: object) -> None:
        raise KeyError("a bug, not an expected failure")

    _fake_module(monkeypatch, "demand.data.pipeline", run_gold=bug)

    result = runner.invoke(app, ["gold"])

    assert isinstance(result.exception, KeyError)


def test_serve_exits_with_an_error_until_the_api_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("demand.cli.importlib.util.find_spec", lambda name: None)

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 1


def test_serve_starts_uvicorn_with_configured_host_and_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: dict[str, object] = {}
    monkeypatch.setattr("demand.cli.importlib.util.find_spec", lambda name: object())
    _fake_module(
        monkeypatch,
        "uvicorn",
        run=lambda target, host, port: started.update(target=target, host=host, port=port),
    )

    result = runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert started == {"target": "demand.serving.app:app", "host": "0.0.0.0", "port": 8000}
