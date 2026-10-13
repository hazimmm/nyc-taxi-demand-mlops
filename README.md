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

Runs five stages in order: download the raw files, clean them into silver, check silver, build gold, check gold. A stage is skipped when nothing it depends on has changed, so a second run does nothing. If a check fails, the pipeline stops before the next stage and the details are in `reports/`.

Activate the venv first, because the stages call `python`. The first run downloads about 390 MB from public servers. The data is not stored in git and no DVC remote is needed: the pipeline rebuilds everything from the sources.

The pipeline does not notice when a source publishes a corrected version of a file it already has: ingest compares each file on disk with its recorded checksum and never asks the server again. To fetch a file again, delete it from `data/raw` and run `dvc repro --force ingest_raw`.

## Learning book as a PDF

```powershell
python tools\build_book.py
```

Builds `docs/learning/nyc-taxi-demand-mlops-textbook.pdf` from the Markdown chapters. It needs Chrome or Edge installed, and downloads the Mermaid diagram library once on the first run. The PDF is not committed; the Markdown is the source.
