from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator

import pytest
from typer.testing import CliRunner

from demand.cli import _run_stage, app
from demand.config import CONFIG_FILE_ENV, Settings, get_settings
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
        ("train", "demand.models.pipeline", "run_training"),
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
