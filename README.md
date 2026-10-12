# NYC Taxi Demand Forecasting

Day-ahead forecast of hourly taxi pickups per NYC zone, wrapped in an end-to-end MLOps stack built only from free, open-source tools.

Work in progress. See [docs/architecture.md](docs/architecture.md) for the system design.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.lock
.venv\Scripts\python -m pip install -e . --no-deps
.venv\Scripts\python -m demand --help
```
