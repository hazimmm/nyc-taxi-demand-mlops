"""The prediction API: hourly pickup forecasts for a whole day, over HTTP.

Start it with ``python -m demand serve`` (host and port from ``serving`` in the config);
``/docs`` then shows every endpoint with examples.

=================  ==========================================================  =========
Endpoint           Answers                                                     Needs
=================  ==========================================================  =========
``/health``        200 while the process runs                                  nothing
``/ready``         200 once the model and the data are loaded, 503 before      both
``/predict``       the forecasts of one day for one or more zones              both
``/model-info``    the served version, its run, its scores, the days served    both
``/metrics``       request, latency and forecast counters for Prometheus       nothing
=================  ==========================================================  =========

**A forecast is a GET.** ``GET /predict?date=2025-05-14&zone=161&zone=237``. Asking for a
forecast changes nothing on the server, and the same question gets the same answer from
the same model and data: that is what GET means in HTTP, and it lets a browser, ``curl``
or a probe ask without a request body. The question is two small values, so a JSON body
(POST) would add nothing; POST is the usual choice when a client sends the model its
features, which this service does not let a client do (``snapshot.py`` says why).

**Status codes.**

* 200: the forecasts.
* 422: the request cannot be answered as asked, and the message says what can be: a day
  not written ``YYYY-MM-DD``, a zone that is not a whole number, a parameter that does
  not exist, a zone twice, a zone the service does not forecast, or a day outside the
  days the data allows. A day outside the data is 422 and not 404 on purpose: the
  allowed days are a published range (``/model-info``), the same kind of limit as the
  list of zones, and the client fixes it the same way, by asking for another value.
  A 404 says "this address does not exist", and a client, a proxy or a smoke test would
  read it as a wrong URL. All 422 answers have FastAPI's shape, ``{"detail": [{"loc":
  ..., "msg": ..., "type": ...}]}``, so a client handles them in one place.
* 503: the model and the data are not loaded (``/ready``, ``/predict``,
  ``/model-info``).
* 500: a bug. It is not caught: the traceback goes to the log.

**Loading happens once, before the first request, and a failure stops the process.**
FastAPI runs ``lifespan`` at startup: it loads the model (``model_loader.py``) and the data
(``snapshot.py``) and forecasts the first day once for every zone, which proves that the
model, the config and the data fit together before anyone asks. If anything fails, the
process exits with the message. The alternative, a process that starts without a model
and answers 503 on ``/ready`` until it has one, needs a loop that keeps retrying inside
the process, and without it the process would wait for ever: Kubernetes restarts a
container whose *liveness* probe fails, not one that is merely not ready, and ``/health``
would keep saying it is alive. A process that exits is restarted by Kubernetes with a
growing pause (``CrashLoopBackOff``), its message is in ``kubectl logs``, and it comes up
by itself once the store or the alias is back. uvicorn opens the port only after
``lifespan`` has finished, so in practice ``/ready`` answers 200 from its first
request; until then the probe cannot connect, which counts as not ready too. The 503 is
for an app used without its startup, as in the tests.

**Stateless.** Nothing is written at request time: no file, no database, no cache. Every
replica holds the same model and the same read-only data and can answer any request.
The forecasts and the actual pickups are written by the batch forecast of the replay
(``demand.flows.forecast_flow``), not here; it computes them with the same function,
``predict_day`` (``forecasting.py``). Logs
go to stderr and counters to ``/metrics``.

**Left out on purpose:** authentication, CORS, rate limits, caching of answers,
forecasts for single hours, and hot reloading of the model (a restart picks up a new
champion; ``model_loader.py`` explains why).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.routing import Match, Route

from demand.config import Settings, get_settings
from demand.logs import get_logger
from demand.schemas import HOUR_START, ZONE_ID
from demand.serving.forecasting import predict_day
from demand.serving.metrics import UNMATCHED, ApiMetrics
from demand.serving.model_loader import LoadedModel, load_model
from demand.serving.schemas import (
    ForecastDays,
    ForecastQuery,
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    ModelInfoResponse,
    ReadyResponse,
)
from demand.serving.snapshot import DataSnapshot, RequestRefusedError, load_snapshot

log = get_logger(__name__)

NOT_READY = "The model and the data snapshot are not loaded yet; try again shortly."


@dataclass(frozen=True)
class Service:
    """Everything a forecast needs: the model and the data it reads."""

    model: LoadedModel
    snapshot: DataSnapshot


def forecast_day(
    service: Service, zones: Sequence[int], day: date, settings: Settings
) -> ForecastResponse:
    """The forecasts of every hour of ``day`` for ``zones``, as the API answers them.

    They come from ``predict_day``, the function the batch forecast of the replay calls.

    Raises:
        RequestRefusedError: If a zone or the day cannot be forecast (``check_request``).
    """
    predicted = predict_day(service.model.predictor, service.snapshot, zones, day, settings)
    rows = predicted.rows
    info = service.model.info
    return ForecastResponse(
        date=day,
        registered_model=info.registered_model,
        model_version=info.version,
        issued_at=predicted.issued.to_pydatetime(),
        hours=len(rows) // len(zones),
        forecasts=[
            HourlyForecast(zone_id=zone, hour_start=hour.to_pydatetime(), forecast=value)
            for zone, hour, value in zip(
                rows[ZONE_ID].tolist(), rows[HOUR_START], predicted.values.tolist(), strict=True
            )
        ],
    )


def load_service(settings: Settings) -> Service:
    """Load the model and the data, and forecast the first day once as a check.

    Raises:
        PipelineError: If the model or the data cannot be loaded (see their modules).
        ModelInputError: If the model refuses the rows built from the data.
    """
    service = Service(model=load_model(settings), snapshot=load_snapshot(settings))
    forecast_day(service, settings.data.zones, service.snapshot.first_day, settings)
    return service


def create_app(settings: Settings, load: Callable[[Settings], Service] = load_service) -> FastAPI:
    """Build the API. ``load`` is called once at startup; tests pass their own."""
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        started = time.monotonic()
        service = load(settings)
        app.state.service = service
        metrics.record_model(service.model.info)
        log.info(
            "service ready",
            extra={
                **service.model.info.model_dump(include={"registered_model", "version", "alias"}),
                "first_day": service.snapshot.first_day.isoformat(),
                "last_day": service.snapshot.last_day.isoformat(),
                "seconds": round(time.monotonic() - started, 2),
            },
        )
        yield

    app = FastAPI(
        title="NYC taxi demand forecasts",
        description="Hourly pickups per zone for a whole day, issued at 00:00 the day before.",
        lifespan=lifespan,
    )
    app.state.service = None

    def ready_service() -> Service:
        service: Service | None = app.state.service
        if service is None:
            raise HTTPException(status_code=503, detail=NOT_READY)
        return service

    @app.middleware("http")
    async def count_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        status = 500  # what the client gets if the endpoint raises
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            metrics.record_request(route_of(request), status, time.perf_counter() - started)

    def route_of(request: Request) -> str:
        # The route template, never the raw path, so that the label stays small. FastAPI's
        # own pages (/docs, /openapi.json, /redoc) do not record themselves in the scope,
        # so they are found by matching.
        for route in app.routes:
            if isinstance(route, Route) and route.matches(request.scope)[0] == Match.FULL:
                return route.path
        return UNMATCHED

    not_ready: dict[int | str, dict[str, Any]] = {
        503: {"description": "The model and the data are not loaded yet."}
    }
    refused: dict[int | str, dict[str, Any]] = {
        422: {"description": "The request cannot be answered; see the message."}
    }

    @app.get("/health", summary="Is the process alive?")
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/ready", summary="Are the model and the data loaded?", responses=not_ready)
    def ready() -> ReadyResponse:
        return ReadyResponse(status="ready", model_version=ready_service().model.info.version)

    @app.get(
        "/predict",
        summary="Forecast every hour of one day",
        responses={**refused, **not_ready},
    )
    def predict(query: Annotated[ForecastQuery, Query()]) -> ForecastResponse:
        """Hourly pickups of one local day for the zones asked, from the served model."""
        service = ready_service()
        try:
            answer = forecast_day(service, query.zone, query.date, settings)
        except RequestRefusedError as error:
            raise RequestValidationError(
                [
                    {
                        "type": error.kind,
                        "loc": ("query", error.field),
                        "msg": str(error),
                        "input": error.value,
                    }
                ]
            ) from error
        metrics.record_forecasts(forecast.forecast for forecast in answer.forecasts)
        return answer

    @app.get("/model-info", summary="Which model is served?", responses=not_ready)
    def model_info() -> ModelInfoResponse:
        service = ready_service()
        snapshot = service.snapshot
        return ModelInfoResponse(
            **service.model.info.model_dump(),
            zones=settings.data.zones,
            forecast_days=ForecastDays(first=snapshot.first_day, last=snapshot.last_day),
        )

    @app.get("/metrics", summary="Counters for Prometheus", response_class=Response)
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app(get_settings())
"""The app uvicorn serves (``python -m demand serve``), configured from the settings."""
