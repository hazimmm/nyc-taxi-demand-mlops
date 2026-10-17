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

## Learning book as a PDF

```powershell
python tools\build_book.py
```

Builds `docs/learning/nyc-taxi-demand-mlops-textbook.pdf` from the Markdown chapters. It needs Chrome or Edge installed, and downloads the Mermaid diagram library once on the first run. The PDF is not committed; the Markdown is the source.
