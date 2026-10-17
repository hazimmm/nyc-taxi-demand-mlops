"""What the service counts, for Prometheus: ``GET /metrics`` returns it as text.

Prometheus asks every replica for this page every few seconds and keeps the numbers as
time series; Grafana draws them. Four series (the library adds ``_created`` lines of
its own):

* ``demand_api_requests_total{endpoint, status}``: requests answered, by route and HTTP
  status. The error rate is the share with a status of 400 or more.
* ``demand_api_request_duration_seconds{endpoint}``: a histogram of how long requests
  took, from which Prometheus works out percentiles (the 95th, the 99th).
* ``demand_api_forecasts``: a histogram of the forecast values the service returned.
  Its ``_count`` is the number of hourly forecasts (480 for 20 zones of a 24-hour day),
  its buckets show how the forecasts are spread, so a shift is visible on a dashboard.
* ``demand_model_info{registered_model, version, alias, run_id}``: always 1. It carries
  the served version in its labels, so a dashboard can show which version each replica
  serves; it appears once the model is loaded.

**Labels are kept few.** Every distinct combination of label values is a time series of
its own, and Prometheus slows down with too many. So ``endpoint`` is the route as it is
declared (``/predict``), never the URL a client sent: ``/predict?date=...`` would make a
series per day and zone, and a client calling made-up paths a series per path (those are
all counted as ``unmatched``). There is no label for the zone or the day, and none for
the HTTP method, which a client can set to any word. The status code has a few dozen
possible values at most; the model labels have one value per process.

Every app gets its own ``CollectorRegistry`` instead of the global one of
``prometheus_client``, so that two apps in one process (as in the tests) do not count
into the same series. The numbers belong to one process: with several replicas,
Prometheus asks each of them and adds them up.
"""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from demand.serving.model_loader import ModelInfo

UNMATCHED = "unmatched"
"""The ``endpoint`` of a request whose path is no route of the app."""

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
"""Seconds. A forecast for one zone takes some milliseconds; all zones, some tens."""
FORECAST_BUCKETS = (1, 5, 10, 25, 50, 100, 200, 400, 800, 1600)
"""Pickups per zone and hour. The busiest zones see several hundred in the rush hour."""


class ApiMetrics:
    """The counters of one app, in a registry of their own."""

    def __init__(self) -> None:
        """Create the series; they start at 0."""
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "demand_api_requests",
            "Requests answered, by route and HTTP status.",
            ["endpoint", "status"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "demand_api_request_duration_seconds",
            "Time to answer a request, by route.",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.forecasts = Histogram(
            "demand_api_forecasts",
            "Hourly forecasts returned, by forecast pickups.",
            buckets=FORECAST_BUCKETS,
            registry=self.registry,
        )
        self.model = Gauge(
            "demand_model_info",
            "The served model version; always 1.",
            ["registered_model", "version", "alias", "run_id"],
            registry=self.registry,
        )

    def record_request(self, endpoint: str, status: int, seconds: float) -> None:
        """Count one answered request and how long it took."""
        self.requests.labels(endpoint=endpoint, status=str(status)).inc()
        self.latency.labels(endpoint=endpoint).observe(seconds)

    def record_forecasts(self, values: Iterable[float]) -> None:
        """Count the forecasts of one response."""
        for value in values:
            self.forecasts.observe(value)

    def record_model(self, info: ModelInfo) -> None:
        """Show the loaded model version."""
        self.model.labels(
            registered_model=info.registered_model,
            version=str(info.version),
            alias=info.alias or "",
            run_id=info.run_id,
        ).set(1)

    def render(self) -> bytes:
        """All series in the text format Prometheus reads."""
        text: bytes = generate_latest(self.registry)
        return text
