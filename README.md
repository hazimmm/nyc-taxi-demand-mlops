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

## Learning book as a PDF

```powershell
python toolsuild_book.py
```

Builds `docs/learning/nyc-taxi-demand-mlops-textbook.pdf` from the Markdown chapters. It needs Chrome or Edge installed, and downloads the Mermaid diagram library once on the first run. The PDF is not committed; the Markdown is the source.
