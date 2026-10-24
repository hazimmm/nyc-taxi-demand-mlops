# Chapter 31 – Starting, proving and watching the service

Version 2 is the champion, and the service of Chapter 30 is running. A person makes version 1 champion with `set-champion`. A minute later a planner asks for a forecast. Which model answers, and how can the planner tell? This chapter is about everything around the requests: how the service starts, how it was shown to build exactly the features of training, and what it tells the systems that watch it.

## What you will learn

- What happens at startup, and why the alias is turned into a version number once.
- Why the service refuses to start without its model, and what that means in Kubernetes.
- How the model is loaded without writing into the MLflow store.
- How training/serving skew was ruled out, on a small table and on real data.
- What `/health`, `/ready`, `/model-info` and `/metrics` answer, and why the metrics have few labels.
- What the service leaves out on purpose, and its limits.

## Startup: everything once, before the first request

FastAPI runs a function named `lifespan` once, when the application starts. In [`src/demand/serving/app.py`](../../src/demand/serving/app.py) it calls `load_service`, which does three things: it loads the model ([`model_loader.py`](../../src/demand/serving/model_loader.py)), loads the data snapshot of Chapter 30, and forecasts the first day, 10 January, once for every zone. That forecast is thrown away. It proves that the model, the config and the data fit together before anyone asks.

uvicorn opens the port only after `lifespan` has finished. In the log of a real start, the line `Application startup complete.` comes before `Uvicorn running on http://0.0.0.0:8000`. Between the two is the line `service ready`, which reported `"seconds": 2.33` for the whole load on the machine of this book.

## From an alias to a number, once

The config says `serving.model_uri: "models:/taxi-demand@champion"`. The alias can move at any time (Chapter 29). If the service asked MLflow for the alias at every request, or reported the alias instead of a number, a response could name another version than the one that computed it.

So the loader asks once, at startup, which version the alias points at: 2. It loads `models:/taxi-demand/2` by its number, and reports that number in every forecast, in `/ready`, in `/model-info` and in `/metrics`, for as long as the process lives.

This answers the opening question. After `set-champion 1`, the running service keeps answering with version 2, and its answers say `"model_version": 2`. That is true: version 2 computed them. Version 1 is served after a restart.

The service does not watch the alias. The docstring of `model_loader.py` gives the reasons. A service that swapped its model while running would need a lock around every forecast. And two copies of the service behind one address could serve two versions for minutes without anyone deciding it. A restart is one clear moment. In Kubernetes, `kubectl rollout restart` replaces the copies one by one; each new copy reads the alias when it starts, and the old ones answer until the new ones are ready. A new data snapshot needs the same restart.

Startup checks three more things. The SQLite file of the store must exist, because MLflow would otherwise create an empty one. The alias must be set. And the model's signature (Chapter 26) must name exactly the 16 input columns that the config builds, in order: a model trained with other lags would be given columns it never saw.

## A service that refuses to start

If any check fails, the process stops. Here the service is pointed at the alias `challenger`, which is not set in this store:

```text
demand.serving.model_loader.ModelLoadError: The alias 'challenger' is not set on 'taxi-demand' (aliases set: {'champion': 2}; versions: [1, 2]). Promote a version first: `python -m demand promote`, or `python -m demand set-champion`.

ERROR:    Application startup failed. Exiting.
```

The **exit code** (Chapter 9) is 3, uvicorn's code for a failed startup. The message comes at the end of a traceback, not as the one line of Chapter 20, because uvicorn prints the error itself.

The alternative would be a service that starts without a model and answers 503 on `/ready` until it has one. The docstring of `app.py` explains why that is worse. It needs a loop inside the process that keeps retrying, and without that loop the process waits for ever. Kubernetes, the system of Phase 6 that runs the containers, asks two different questions with automatic requests named probes. A **liveness** probe asks "is the process alive?", and Kubernetes restarts a container that fails it. A **readiness** probe asks "can it answer now?", and Kubernetes sends requests only to containers that pass it, but restarts none. A service with no model would be alive, never ready, and never restarted. A process that exits is restarted, with a growing pause between attempts, its message is in the log, and it comes up by itself once the alias is back. Chapter 33 shows this with Docker Compose, and Chapter 40 with Kubernetes.

In practice a client never sees the 503 of Chapter 30. Until `lifespan` has finished the port is closed, and a probe that cannot connect counts as not ready too. The 503 is for an application used without its startup, as in the tests.

## Reading the store without writing to it

Chapter 26 showed that loading a model through the registry, with a `models:/` address, makes MLflow write a small file, `registered_model_meta`, into the model's folder in `mlruns`. A service that did this would change the store at every start, and it would fail where the store is read-only, as it can be in a container. So the loader never loads a `models:/` address. It asks the configured registry where the files of the version are, copies them into a temporary folder, and loads the model from there:

```python
    source = registry.client.get_model_version_download_uri(name, str(version))
    with tempfile.TemporaryDirectory(prefix="demand-model-") as folder:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=source, dst_path=folder, tracking_uri=tracking_uri
        )
        predictor = mlflow.pyfunc.load_model(local)
```

The folder is deleted when the block ends. The model stays in memory and needs no file afterwards. The progress bar `Downloading artifacts` in the startup log is this copy of seven files. The test `test_loading_writes_nothing_into_the_store` compares the files of a store before and after a load.

The first line has a second reason, which the containers of Chapter 33 found: with a `models:/` address, MLflow 3.17 looks the version up in the registry set for the whole process, which need not be the registry of the config. That chapter tells the story and the test that now guards it.

## Proving that training and serving agree

Chapter 30 claimed that a served row equals the training row of the same hour. Training builds its rows with `build_feature_table`, which passes the whole demand table to `build_features` as history and as targets. The service passes only the demand known at the issue moment, and the weather from the silver table. Both paths must give the same numbers.

**A unit test.** [`tests/unit/serving/test_skew.py`](../../tests/unit/serving/test_skew.py) builds a small demand table and weather table, from 25 February to 12 March 2025 for three zones. It builds the training rows and the served rows of four days, and compares them with `check_exact=True`, so every value and every type must be equal. The days are 6 March (the first day with full history), 9 March (the clocks go forward), 10 March (the day after) and 12 March (the last day).

**A check on the real data.** The test cannot read the real data or the real model. So for this phase the service's rows were compared by hand with `data/gold/features.parquet`, for six days and all 20 zones, and the forecasts of the service were compared with the forecasts of the champion on the stored rows.

| Day | `issued_at` | Rows | Hours per zone |
|---|---|---|---|
| Tuesday 1 April | 31 March, 00:00 | 480 | 24 |
| Thursday 3 April | 2 April, 00:00 | 480 | 24 |
| Tuesday 15 April | 14 April, 00:00 | 480 | 24 |
| Wednesday 30 April | 29 April, 00:00 | 480 | 24 |
| Sunday 9 March | 8 March, 00:00 | 460 | 23 |
| Monday 10 March | 9 March, 00:00 | 480 | 24 |

*Figure 31.1 – The real-data check: 2,860 rows. Every feature was equal to the stored one, bit for bit, and so was every forecast.*

3 April is the first test day of the backtest (Chapter 23), 30 April the last day of the training period, and 9 and 10 March are the clock change and the day after, where lags cross the missing hour. No day after 30 April was compared, because this project does not analyse May and June. Chapter 4 promised one feature function for training and serving, with a check that both get identical output. This is that check for the service.

## Four endpoints that describe the service

| Endpoint | Needs the model | Answer of the real service |
|---|---|---|
| `/health` | no | `{"status":"ok"}`: the process runs |
| `/ready` | yes | `{"status":"ready","model_version":2}`: model and data loaded |
| `/model-info` | yes | version 2, alias `champion`, run `a8aa7b30…`, `backtest_mae` 24.106025, days 2025-01-10 to 2025-06-30 |
| `/metrics` | no | counters as text, below |

*Figure 31.2 – The endpoints besides `/predict`.*

`/health` is for a liveness probe and `/ready` for a readiness probe. `/model-info` answers "which model is this?" with the record of Chapter 26: the run id, the data version, the git commit `5d1d9e4…`, the input columns, the zones and the days served.

## Metrics for Prometheus

**Prometheus**, which runs next to the service in Chapter 33, is a database for measurements over time. It asks every copy of the service for `/metrics` every few seconds and keeps the numbers; Grafana draws them. The page is plain text: a line `# HELP` and a line `# TYPE` for each series, then one line per value.

```text
# TYPE demand_api_requests_total counter
demand_api_requests_total{endpoint="/predict",status="200"} 1.0
demand_api_requests_total{endpoint="/predict",status="422"} 1.0
```

[`src/demand/serving/metrics.py`](../../src/demand/serving/metrics.py) defines four series.

| Series | Kind | What it shows |
|---|---|---|
| `demand_api_requests_total` | counter | Requests answered, by `endpoint` and `status`. The error rate is the share with a status of 400 or more. |
| `demand_api_request_duration_seconds` | histogram | Time to answer, by `endpoint`. Prometheus calculates percentiles from it. |
| `demand_api_forecasts` | histogram | The forecast values returned. A shift in them shows on a dashboard. |
| `demand_model_info` | gauge, always 1 | The served version, in its labels. |

*Figure 31.3 – The four series.*

A counter only goes up. A gauge is a value that can go up and down. A histogram counts values in ranges, written as lines `_bucket{le="..."}`, and adds a `_count` and a `_sum`. The library also writes lines that end in `_created`, the moment a series started; they can be ignored.

**Few labels, on purpose.** A **label** is a name and a value that split a series, such as `status="422"`. Every distinct combination of label values is a time series of its own, and the number of combinations is the **cardinality** of a series. Prometheus slows down when it is large. So `endpoint` is the route as it is declared, `/predict`, never the address a client sent. A label with the day and the zone would make, for requests of one zone each, 172 days times 20 zones: 3,440 series for every status code. A path that is no route is counted as `unmatched`, so a client that calls invented paths adds no series. There is no label for the HTTP method, which a client can set to any word.

The numbers belong to one process: with several copies, Prometheus asks each and adds them up.

## Left out on purpose, and the limits

The docstring of `app.py` lists what is left out on purpose: authentication, CORS (the rules that let a web page from another address call the service from a browser), rate limits, caching of answers, forecasts for single hours, and swapping the model while running. The service is **stateless**: it writes nothing at request time, so every copy can answer any request.

The honest limits, measured on the machine of this book:

- **Speed.** A request for one or two zones took about 35 ms, and one for all 20 zones about 53 ms. Of the 35 ms, building the features took about 25 ms, mostly pandas, and the model about 7 ms. The first request after a start was slower, between about 120 and 200 ms.
- **One process.** uvicorn runs one worker process. More load needs more copies, which is the job of Phase 6. No load test was run here; Chapter 40 runs a small one on Kubernetes.
- **Stopping.** Whether requests in progress are finished when the process is told to stop (a graceful shutdown) is not tested.
- **Two small gaps.** A failed start prints a traceback, not one line. And FastAPI's own pages `/docs` and `/openapi.json` are counted as `unmatched`.

## Try it yourself

Start the service again, as in Chapter 30, so that the counters start at 0. In the second window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/ready
(Invoke-RestMethod http://127.0.0.1:8000/model-info) | Select-Object version, alias, run_id, backtest_mae
$r = Invoke-RestMethod "http://127.0.0.1:8000/predict?date=2025-05-14&zone=161&zone=237"
curl.exe -s -o NUL "http://127.0.0.1:8000/predict?date=2025-05-14&zone=999"
curl.exe -s http://127.0.0.1:8000/metrics | Select-String "^demand_api_requests_total|^demand_api_forecasts_count|^demand_model_info"
```

You should see a small table with `ready` and `2`, then one with `2`, `champion`, `a8aa7b305856481b984bd97586d2f549` and `24.106025`, and then:

```text
demand_api_requests_total{endpoint="/ready",status="200"} 1.0
demand_api_requests_total{endpoint="/model-info",status="200"} 1.0
demand_api_requests_total{endpoint="/predict",status="200"} 1.0
demand_api_requests_total{endpoint="/predict",status="422"} 1.0
demand_api_forecasts_count 48.0
demand_model_info{alias="champion",registered_model="taxi-demand",run_id="a8aa7b305856481b984bd97586d2f549",version="2"} 1.0
```

The forecast of 10 January at startup is not counted: only answers are. Stop the service with Ctrl+C. Then start it with an alias that is not set.

```powershell
$env:DEMAND_SERVING__MODEL_URI = "models:/taxi-demand@challenger"
.venv\Scripts\python.exe -m demand serve
$LASTEXITCODE
Remove-Item Env:DEMAND_SERVING__MODEL_URI
```

You should see a traceback that ends with the message shown earlier in this chapter, then `ERROR:    Application startup failed. Exiting.`, and then `3`.

Count the files MLflow writes when a model is loaded through the registry.

```powershell
(Get-ChildItem -Recurse mlruns -Filter registered_model_meta).Count
```

On the machine of this book it is `0` before and after every start of the service. If you ran the last load of Chapter 26, you may see `1`: that file was written then, and starting the service does not change the number.

Compare the served rows of 9 March with the stored training rows.

```powershell
.venv\Scripts\python.exe -c "from datetime import date; import pandas as pd; from demand.config import get_settings; from demand.schemas import feature_columns; from demand.serving.snapshot import load_snapshot, day_features; s = get_settings(); day = date(2025, 3, 9); issued, served = day_features(load_snapshot(s), s.data.zones, day, s); f = pd.read_parquet('data/gold/features.parquet'); keys = ['zone_id', 'hour_start']; cols = keys + feature_columns(s)[1:]; served = served.astype({'hour_start': f.hour_start.dtype}); stored = served[keys].merge(f, on=keys, how='left'); print(issued, len(served), len(served) // 20); print(served[cols].reset_index(drop=True).equals(stored[cols]))"
```

You should see `2025-03-08 00:00:00-05:00 460 23` and then `True`. The command makes the hour column of the served rows the same type as the stored one, as the test does: they are keys, not model inputs.

Run all tests of the service.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/serving -q
```

You should see `92 passed, 3 warnings`. The warnings come from FastAPI's test client and from MLflow.

## Key takeaways

- At startup the service loads the model and the data once and forecasts one day as a check. The port opens only afterwards.
- The alias is resolved to a version number once. That number is in every answer, so it always names the model that computed it. A new champion needs a restart.
- Without its model the service does not start: exit code 3 and a message. Kubernetes restarts a container that exits, but not one that is only not ready.
- The configured registry is asked where the model's files are, and they are copied to a temporary folder before the model is loaded, so the store is only read.
- A unit test and a check on 2,860 real rows showed that served features equal training features, bit for bit.
- Metrics have labels with few values: the route and the status, never the day or the zone.

## Exercises

**1.** A colleague adds the lag 336 to the config and restarts the service, without training a new model. What happens, and what would happen without the signature check?

<details>
<summary>Answer</summary>

The service does not start. The loader finds that version 2 takes 16 inputs and the config builds 17. The message lists both and says to serve a model trained with this config, and the process exits with code 3. Without the check, the service would build 17 columns, and MLflow would drop the one the signature does not name, with only a warning (Chapter 26). The forecasts would come from a model that does not match the config, and nothing would fail.

</details>

**2.** A colleague wants a label `zone` on `demand_api_requests_total`, "to see which zones are asked for most". What do you answer?

<details>
<summary>Answer</summary>

A request may name several zones, so the label would hold combinations of zones, and their number has no practical limit. Even with one zone per request it multiplies every series by 20. The question is better answered from the log: uvicorn writes one line for every request, with its full address and so its zones. A label must have a small, fixed set of values.

</details>

**3. Interview question.** "What is the difference between a liveness probe and a readiness probe, and how did your service use them?"

<details>
<summary>Points of a good answer</summary>

- Liveness: is the process alive? A failure makes Kubernetes restart the container. Readiness: can it answer now? A failure only stops traffic to it.
- `/health` answers liveness and `/ready` readiness; `/ready` names the model version.
- Everything is loaded before the port opens, and a failed load stops the process, so a broken service is restarted instead of waiting alive and unready.
- A new model or new data arrives with a rolling restart, so every answer names exactly one version.

</details>

---

[Previous: Chapter 30 – A forecast over HTTP: the prediction service](30-a-forecast-over-http.md) | [Contents](README.md) | [Next: Chapter 32 – Images: the service with everything it needs](32-images.md)
