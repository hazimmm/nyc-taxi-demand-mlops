# NYC Taxi Demand Forecasting

Day-ahead forecast of hourly taxi pickups per NYC zone, wrapped in an end-to-end MLOps stack built only from free, open-source tools.

Work in progress. See [docs/architecture.md](docs/architecture.md) for the system design and [docs/learning](docs/learning/README.md) for a textbook-style guide to the stack and the ideas behind it.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
pre-commit install
python -m demand --help
```

`pre-commit install` makes git run ruff and mypy before every commit. The hooks use the tools from the venv, so activate it in the shell you commit from.

## Checks

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest --cov
```

`--cov` measures coverage and fails the run below 80%. Leave it out when running a single test file.

## Data pipeline

```powershell
dvc repro
```

Runs six stages in order: download the raw files, clean them into silver, check silver, build gold, check gold, and train: score the forecasters on four past weeks (the backtest), fit the final model and record it in MLflow. A stage is skipped when nothing it depends on has changed, so a second run does nothing. If a check fails, the pipeline stops before the next stage and the details are in `reports/`.

Activate the venv first, because the stages call `python`. The first run downloads about 390 MB from public servers. The data is not stored in git and no DVC remote is needed: the pipeline rebuilds everything from the sources.

The pipeline does not notice when a source publishes a corrected version of a file it already has: ingest compares each file on disk with its recorded checksum and never asks the server again. To fetch a file again, delete it from `data/raw` and run `dvc repro --force ingest_raw`.

## Model

```powershell
python -m demand promote
```

`dvc repro` trains and registers a new model version in MLflow but never changes which version is served. `promote` makes the newest version champion only if its backtest MAE beats both the current champion and the weekly baseline by at least 0.2, with no test week worse. Each version is judged once, so a later `promote` never undoes a rollback (rule and reasons in `src/demand/models/promotion.py`). `python -m demand set-champion VERSION --reason TEXT` sets the champion by hand, for a rollback or to bring a judged version back, and records why. What the current model is for and where it is weak: [docs/model-card.md](docs/model-card.md).

## Prediction service

```powershell
python -m demand serve
```

Starts the API on port 8000 with the champion model. `GET /predict?date=2025-05-14&zone=161` returns the 24 hourly forecasts of that day, built from the data known at 00:00 on the day before; `/docs` shows every endpoint. `/health`, `/ready`, `/model-info` and `/metrics` (Prometheus) describe the running service. A new champion is picked up by restarting it.

## Replay and monitoring

```powershell
python -m demand forecast --days 61         # replay May and June 2025 day by day
python -m demand monitor                     # drift and live accuracy at the replay clock
```

`forecast` moves a simulated clock: each step issues the next day's forecast at 00:00 on the day before, with the same code as the API, and scores the days that are over against the real pickups and the weekly baseline (`data/predictions/`, `reports/replay_report.json`). `monitor` checks input drift, prediction drift and live accuracy over the last 7 days (`reports/monitoring_report.json`). Only accuracy raises an alert (live MAE at least 1.25 times the backtest MAE); drift is reported as a warning. `--as-of YYYY-MM-DD` evaluates an earlier day of the replay; `--push-to http://localhost:9091` sends the numbers to the Pushgateway of the Docker stack. Each run takes under a minute. `python -m demand retrain` retrains when the accuracy alert is on and the last retrain is at least `monitoring.retrain_cooldown_days` days old: the new model learns from every hour before the clock, is scored on the last 4 weeks against the champion's real forecasts and the baseline on the same hours, and is promoted only if it passes the same gate as `promote`. Rollback stays manual (`set-champion`). `retrain` evaluates monitoring first, as of the issue moment of the newest forecast, and rewrites `reports/monitoring_report.json`; run `monitor` again for the report at the clock. Results of the replay: [docs/model-card.md](docs/model-card.md).

## Run with Docker

```powershell
dvc repro                                   # build the data on the host first
New-Item -ItemType Directory -Force data/predictions, data/predictions-stack   # replay mount points
docker compose up -d --wait mlflow          # an empty MLflow server of the stack's own
docker compose run --rm pipeline train      # train and register a model in it
docker compose run --rm pipeline promote    # make it champion if it passes the gate
docker compose up -d --wait                 # API, Prometheus, Pushgateway, Grafana
```

API at http://localhost:8000/docs, MLflow at http://localhost:5000, Prometheus at http://localhost:9090, the Pushgateway at http://localhost:9091, Grafana at http://localhost:3000 (dashboards open without a login; admin login from `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in `.env`, `admin`/`admin` if unset, for a laptop only). The stack keeps its own model store in Docker volumes and never reads `mlflow.db` or `mlruns/`; it reads `data/` read-only. Commit before `train`: the gate refuses a model trained from changed files. Until a champion exists the API exits with code 3 and `docker compose ps` shows it restarting. `docker compose down` stops everything and keeps the volumes; `docker compose down -v` deletes them.

## Scheduled with Prefect

```powershell
docker compose up -d --wait                 # the stack, now with a Prefect server
docker compose --profile schedule up -d scheduler   # one simulated day a minute
```

The scheduler serves the daily step (forecast, monitor, retrain if due) as a Prefect deployment and pushes each day's monitoring numbers to the Pushgateway. Runs are listed at http://localhost:4200. One simulated day a minute is a demo of the mechanism: the replay covers May and June 2025 in about an hour. In production the same command would run once a night: `python -m demand schedule --cron "0 1 * * *"`. The stack writes its own replay into `data/predictions-stack` with its own champion; `docker compose --profile schedule down` stops it.

## Run it on Kubernetes (kind)

```powershell
docker compose down                         # the cluster needs port 5000 and the memory
docker compose build api pipeline
kind create cluster --config k8s/kind-cluster.yaml
kind load docker-image nyc-taxi-demand/api:local nyc-taxi-demand/pipeline:local --name nyc-taxi-demand
kubectl apply -k k8s/overlays/kind/metrics-server
kubectl apply -k k8s/overlays/kind
```

The same system on a local Kubernetes cluster ([kind](https://kind.sigs.k8s.io/) v0.33.0): MLflow, the API on two to four pods scaled by CPU, and the daily step as a CronJob (`python -m demand step`, one simulated day per run, without Prefect). The data is copied into a volume, a model is trained on the host against the cluster's MLflow and passes the same gate, and the API is reached with `kubectl port-forward`. Every step, what it prints and why each choice was made: [k8s/README.md](k8s/README.md). Retraining inside the cluster is refused by the gate, because the images carry no git commit; Prometheus and Grafana stay in compose. `kind delete cluster --name nyc-taxi-demand` removes everything.

## Learning book as a PDF

```powershell
python tools\build_book.py
```

Builds `docs/learning/nyc-taxi-demand-mlops-textbook.pdf` from the Markdown chapters. It needs Chrome or Edge installed, and downloads the Mermaid diagram library once on the first run. The PDF is not committed; the Markdown is the source.
