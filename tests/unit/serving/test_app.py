"""The API, through FastAPI's test client, with a tiny data snapshot and ``EchoModel``.

``client`` (``conftest.py``) is an app whose startup has run. A test that needs the app
before its startup builds one without entering ``TestClient`` as a context manager: then
FastAPI does not run ``lifespan`` and nothing is loaded.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from demand.config import Settings
from demand.errors import PipelineError
from demand.schemas import feature_columns, lag_column
from demand.serving import app as app_module
from demand.serving.app import Service, create_app, load_service
from demand.serving.snapshot import day_features
from tests.conftest import TEST_ZONES
from tests.unit.serving.conftest import (
    CLOCKS_FORWARD,
    RUN_ID,
    VERSION,
    EchoModel,
    service,
)

DAY = "2025-03-11"


def metric_value(client: TestClient, line_start: str) -> float:
    """The value of the first line of ``/metrics`` that starts with ``line_start``."""
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(line_start):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def requests_counted(client: TestClient, endpoint: str, status: int) -> float:
    return metric_value(
        client, f'demand_api_requests_total{{endpoint="{endpoint}",status="{status}"}}'
    )


# --- before the startup ------------------------------------------------------------------


def test_health_answers_before_anything_is_loaded(settings: Settings) -> None:
    client = TestClient(create_app(settings, load=lambda _: pytest.fail("loaded")))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/ready", f"/predict?date={DAY}&zone=161", "/model-info"])
def test_endpoints_that_need_the_model_answer_503_before_it_is_loaded(
    settings: Settings, path: str
) -> None:
    client = TestClient(create_app(settings, load=lambda _: pytest.fail("loaded")))

    response = client.get(path)

    assert response.status_code == 503
    assert "not loaded yet" in response.json()["detail"]


def test_a_failed_startup_stops_the_app_with_its_message(settings: Settings) -> None:
    def broken(_: Settings) -> Service:
        raise PipelineError("The alias 'champion' is not set")

    # Entering the client runs the startup, which raises.
    with (
        pytest.raises(PipelineError, match="alias 'champion' is not set"),
        TestClient(create_app(settings, load=broken)),
    ):
        pass


def test_the_model_is_loaded_once_at_startup(settings: Settings) -> None:
    calls: list[Settings] = []
    loaded = service(settings)

    def load(given: Settings) -> Service:
        calls.append(given)
        return loaded

    with TestClient(create_app(settings, load=load)) as client:
        for _ in range(3):
            client.get(f"/predict?date={DAY}&zone=161")

    assert calls == [settings]


# --- health, ready, model-info ------------------------------------------------------------


def test_ready_reports_the_loaded_version(client: TestClient) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": VERSION}


def test_model_info_reports_the_model_its_run_and_the_days_served(
    client: TestClient, settings: Settings
) -> None:
    info = client.get("/model-info").json()

    assert info == {
        "registered_model": "taxi-demand",
        "version": VERSION,
        "alias": "champion",
        "model_uri": "models:/taxi-demand@champion",
        "run_id": RUN_ID,
        "backtest_mae": 24.106025,
        "data_version": "7bac.dir",
        "git_commit": "5d1d9e4",
        "feature_columns": feature_columns(settings),
        "zones": TEST_ZONES,
        "forecast_days": {"first": "2025-03-06", "last": "2025-03-12"},
    }


# --- predict ---------------------------------------------------------------------------


def test_a_forecast_has_every_hour_of_the_day_for_each_zone(
    client: TestClient, settings: Settings, echo: EchoModel
) -> None:
    response = client.get(f"/predict?date={DAY}&zone=237&zone=132")

    assert response.status_code == 200
    body = response.json()
    assert body["date"] == DAY
    assert body["registered_model"] == "taxi-demand"
    assert body["model_version"] == VERSION
    assert body["issued_at"] == "2025-03-10T00:00:00-04:00"
    assert body["hours"] == 24
    forecasts = body["forecasts"]
    assert [row["zone_id"] for row in forecasts] == [237] * 24 + [132] * 24
    assert forecasts[0]["hour_start"] == "2025-03-11T00:00:00-04:00"
    assert forecasts[23]["hour_start"] == "2025-03-11T23:00:00-04:00"


def test_the_forecasts_are_the_models_answers_to_the_rows_of_the_service(
    client: TestClient, settings: Settings, echo: EchoModel
) -> None:
    response = client.get(f"/predict?date={DAY}&zone=161")

    given = echo.given[-1]
    assert list(given.columns) == feature_columns(settings)
    expected = given[lag_column(168)].tolist()
    assert [row["forecast"] for row in response.json()["forecasts"]] == expected


def test_the_model_is_given_the_rows_day_features_builds(settings: Settings) -> None:
    echo = EchoModel()
    loaded = service(settings, echo)
    with TestClient(create_app(settings, load=lambda _: loaded)) as client:
        client.get(f"/predict?date={DAY}&zone=161")

    _, rows = day_features(loaded.snapshot, [161], date(2025, 3, 11), settings)
    (given,) = echo.given
    assert given.equals(rows[feature_columns(settings)])


def test_the_day_the_clocks_go_forward_has_23_forecasts_per_zone(client: TestClient) -> None:
    body = client.get(f"/predict?date={CLOCKS_FORWARD.isoformat()}&zone=161&zone=237").json()

    assert body["hours"] == 23
    assert len(body["forecasts"]) == 46
    starts = [row["hour_start"] for row in body["forecasts"][:4]]
    assert starts == [
        "2025-03-09T00:00:00-05:00",
        "2025-03-09T01:00:00-05:00",
        "2025-03-09T03:00:00-04:00",
        "2025-03-09T04:00:00-04:00",
    ]


@pytest.mark.parametrize(
    ("query", "field", "kind"),
    [
        ("date=2025-03-11", "zone", "missing"),
        ("zone=161", "date", "missing"),
        ("date=2025-3-11&zone=161", "date", "value_error"),
        ("date=1741651200&zone=161", "date", "value_error"),
        ("date=2025-03-11T00:00&zone=161", "date", "value_error"),
        ("date=2025-02-30&zone=161", "date", "date_from_datetime_parsing"),
        ("date=2025-03-11&zone=161.0", "zone", "value_error"),
        ("date=2025-03-11&zone=abc", "zone", "value_error"),
        ("date=2025-03-11&zone=161&zone=161", "zone", "value_error"),
        ("date=2025-03-11&zone=161&zones=237", "zones", "extra_forbidden"),
        ("date=2025-03-11&zone=999", "zone", "unknown_zone"),
        ("date=2025-03-05&zone=161", "date", "date_out_of_range"),
        ("date=2025-03-13&zone=161", "date", "date_out_of_range"),
    ],
)
def test_a_request_that_cannot_be_answered_is_a_422_that_names_the_parameter(
    client: TestClient, query: str, field: str, kind: str
) -> None:
    response = client.get(f"/predict?{query}")

    assert response.status_code == 422
    (error,) = response.json()["detail"]
    assert error["loc"][:2] == ["query", field]
    assert error["type"] == kind


def test_a_refusal_says_what_can_be_asked(client: TestClient) -> None:
    zone = client.get(f"/predict?date={DAY}&zone=999").json()["detail"][0]
    day = client.get("/predict?date=2025-06-01&zone=161").json()["detail"][0]

    assert "The zones are [132, 161, 237]" in zone["msg"]
    assert zone["input"] == [999]
    assert "the days 2025-03-06 to 2025-03-12" in day["msg"]
    assert day["input"] == "2025-06-01"


def test_a_refused_request_does_not_reach_the_model(client: TestClient, echo: EchoModel) -> None:
    calls = len(echo.given)

    client.get(f"/predict?date={DAY}&zone=999")
    client.get("/predict?date=2025-06-01&zone=161")

    assert len(echo.given) == calls


def test_the_documentation_shows_the_parameters_with_examples(client: TestClient) -> None:
    parameters = client.get("/openapi.json").json()["paths"]["/predict"]["get"]["parameters"]

    assert {parameter["name"]: parameter["required"] for parameter in parameters} == {
        "date": True,
        "zone": True,
    }
    assert client.get("/docs").status_code == 200


# --- metrics ----------------------------------------------------------------------------


def test_metrics_count_requests_by_route_and_status(client: TestClient) -> None:
    before = requests_counted(client, "/predict", 200)

    client.get(f"/predict?date={DAY}&zone=161")
    client.get(f"/predict?date={DAY}&zone=161&zone=237")
    client.get(f"/predict?date={DAY}&zone=999")
    client.get("/no-such-page")
    client.get("/docs")

    assert requests_counted(client, "/predict", 200) == before + 2
    assert requests_counted(client, "/docs", 200) == 1
    assert requests_counted(client, "/predict", 422) == 1
    assert requests_counted(client, "unmatched", 404) == 1
    assert (
        metric_value(client, 'demand_api_request_duration_seconds_count{endpoint="/predict"}') >= 3
    )


def test_metrics_count_the_forecasts_returned(client: TestClient) -> None:
    client.get(f"/predict?date={DAY}&zone=161&zone=237")
    client.get(f"/predict?date={CLOCKS_FORWARD.isoformat()}&zone=161")

    assert metric_value(client, "demand_api_forecasts_count") == 48 + 23


def test_metrics_show_the_served_model_version(client: TestClient) -> None:
    line = (
        f'demand_model_info{{alias="champion",registered_model="taxi-demand",'
        f'run_id="{RUN_ID}",version="{VERSION}"}}'
    )

    assert metric_value(client, line) == 1.0


def test_metrics_are_in_the_prometheus_text_format(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "# TYPE demand_api_requests_total counter" in response.text


def test_two_apps_count_separately(settings: Settings) -> None:
    loaded = service(settings)
    with TestClient(create_app(settings, load=lambda _: loaded)) as first:
        first.get("/health")
        with TestClient(create_app(settings, load=lambda _: loaded)) as second:
            assert requests_counted(second, "/health", 200) == 0
            assert requests_counted(first, "/health", 200) == 1


def test_a_failing_model_is_a_500_and_is_counted(settings: Settings) -> None:
    class Broken:
        def predict(self, data: object) -> object:
            raise RuntimeError("a bug")

    loaded = service(settings, Broken())
    app = create_app(settings, load=lambda _: loaded)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(f"/predict?date={DAY}&zone=161")

        assert response.status_code == 500
        assert requests_counted(client, "/predict", 500) == 1


# --- load_service and the module-level app ------------------------------------------------


def test_load_service_loads_both_and_forecasts_the_first_day_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    echo = EchoModel()
    built = service(settings, echo)
    monkeypatch.setattr(app_module, "load_model", lambda _: built.model)
    monkeypatch.setattr(app_module, "load_snapshot", lambda _: built.snapshot)

    loaded = load_service(settings)

    assert loaded == built
    (checked,) = echo.given
    assert len(checked) == 24 * len(TEST_ZONES)


def test_load_service_fails_when_the_model_refuses_the_data(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Refusing:
        def predict(self, data: object) -> object:
            raise PipelineError("The model was not trained on zone 132")

    built = service(settings, Refusing())
    monkeypatch.setattr(app_module, "load_model", lambda _: built.model)
    monkeypatch.setattr(app_module, "load_snapshot", lambda _: built.snapshot)

    with pytest.raises(PipelineError, match="not trained on zone 132"):
        load_service(settings)


def test_the_module_app_is_built_from_the_settings_without_loading_anything() -> None:
    client = TestClient(app_module.app)

    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 503
