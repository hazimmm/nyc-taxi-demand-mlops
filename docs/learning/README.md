# Building a Forecasting System: a Study Book for This Repository

This book explains this project from the beginning, one idea at a time. It is written for someone who knows Python, pandas and basic modelling from courses or Kaggle, and who has not yet worked on a production codebase. Every chapter uses the real code in this repository as its example.

## How to read it

Read the chapters in order. Each one builds on the one before. A chapter takes about 10 to 20 minutes and always has the same shape: a question to start from, the explanation, a worked example from the code, commands to run yourself, the key points, and exercises with hidden answers.

Keep a PowerShell window open in the repository root and run the commands. They all use `.venv\Scripts\python.exe`, so they work whether or not the virtual environment is activated. If `.venv` does not exist yet, follow the setup steps in the [project README](../../README.md) first.

The book grows with the project. New chapters are added when new code arrives. The table at the bottom of this page shows what changed with each push.

## Contents

### Part I – The problem

| Chapter | What it covers | Time |
|---|---|---|
| [1. What we are building](01-what-we-are-building.md) | The forecast, who needs it, and the path from raw files to a served number | 8 min |
| [2. Forecasting without cheating](02-forecasting-without-cheating.md) | The forecast horizon, lag features, and three ways a model can see the future by accident | 12 min |
| [3. Testing a forecast honestly](03-testing-a-forecast-honestly.md) | Time-based splits, backtests, and the simulated clock | 12 min |
| [4. Data in layers, features in one place](04-data-in-layers.md) | Raw, silver and gold data, and why features must have one definition | 10 min |

### Part II – Foundations of a production codebase

| Chapter | What it covers | Time |
|---|---|---|
| [5. Project layout and packaging](05-project-layout-and-packaging.md) | The `src/` layout, installing your own code, and why `python -m demand` works | 10 min |
| [6. Environments, dependencies and the lock file](06-environments-and-dependencies.md) | Virtual environments, `pyproject.toml`, optional groups, pinned versions | 12 min |
| [7. Configuration](07-configuration.md) | One typed settings object, YAML, environment variables, and a config that refuses to leak | 15 min |
| [8. Data contracts](08-data-contracts.md) | Pandera schemas as gates between data layers | 13 min |
| [9. The command line and logging](09-command-line-and-logging.md) | One entry point for every environment, lazy imports, exit codes, JSON logs | 13 min |
| [10. Testing](10-testing.md) | Fixtures, temporary folders, monkeypatch, parametrised tests, coverage | 14 min |
| [11. Code quality tools](11-code-quality-tools.md) | ruff, mypy in strict mode, and pre-commit | 12 min |
| [12. Working with git](12-working-with-git.md) | Line endings, what stays out of the repository, commit messages, the `dev` branch | 10 min |
| [13. A code review of our own foundation](13-a-code-review-of-our-own-foundation.md) | Five real problems found by reading this code, and what each one teaches | 13 min |

### Part III – The data pipeline

This part is being written. More chapters follow when aggregation, features and data versioning are built.

| Chapter | What it covers | Time |
|---|---|---|
| [14. Getting the data: downloads you can trust](14-getting-the-data.md) | The raw layer, half files and error pages, atomic rename, retries with backoff, testing without a network | 18 min |
| [15. Knowing what you have: the manifest and idempotency](15-knowing-what-you-have.md) | Checksums, the manifest, skip or download, the weather source, UTC and padding, a first look at the raw data | 20 min |
| [16. Cleaning the trips: rules you can count](16-cleaning-the-trips.md) | One row is one valid trip, ten rules in a fixed order, counts that must add up, the cleaning report, and a rule from the plan that the data proved wrong | 14 min |
| [17. Clocks, weather and the silver stage](17-clocks-weather-and-the-silver-stage.md) | Local time and daylight saving, elapsed time, weather from UTC to local hours, gaps, and a stage that validates before it writes | 14 min |

### Later parts

These parts have no chapters yet. They are written when their code is added.

| Part | Topic | Arrives with |
|---|---|---|
| IV | Modelling: baseline, LightGBM, backtesting, tuning, MLflow | Phase 2 |
| V | Serving: the FastAPI prediction service | Phase 3 |
| VI | Containers: Docker and Docker Compose | Phase 4 |
| VII | Monitoring and retraining: Prefect, Evidently, Prometheus, Grafana | Phase 5 |
| VIII | Kubernetes with kind and Kustomize | Phase 6 |
| IX | CI/CD, the public demo and documentation | Phase 7 |

### Appendices

| Appendix | What it is |
|---|---|
| [A. Stack map](appendix-a-stack-map.md) | Every tool in the planned stack, what it is for, and whether it is in the repository yet |
| [B. Glossary](appendix-b-glossary.md) | Every technical term used in the book, in alphabetical order |
| [C. Further reading](appendix-c-further-reading.md) | Official documentation for the tools in use today |

## What's new

| Push | What changed in the project | Chapters added or changed |
|---|---|---|
| 1 – Project foundation (Phase 0) | Packaging, settings, data contracts, CLI skeleton, logging, tests, quality tools | 1 to 12, appendices A to C |
| 2 – Foundation fixes found in review | `.env` file is read; the gold zone gate fails closed; coverage is opt-in; pre-commit is installed | 13 added; 7, 8, 10 and 11 describe the fixed behaviour |
| 3 – Ingestion into the raw layer (Phase 1) | `python -m demand ingest` downloads trips, zones and weather safely and records them in a manifest; new `data.download` settings | 14 and 15 added; 1, 2, 4, 5, 9, 10 and 11 updated; appendices A to C extended |
| 4 – Cleaning into the silver layer (Phase 1) | `python -m demand silver` cleans trips and weather, validates them against the silver contracts and writes a cleaning report; the silver column `fare_amount` became `total_amount` | 16 and 17 added; 1, 2, 4, 8, 9, 10, 11, 14 and 15 updated; appendices A to C extended |
