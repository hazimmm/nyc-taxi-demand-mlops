# Appendix A – Stack map

This appendix is a reference. It lists every tool in the planned stack, what it is, the problem it solves in this project, and whether it is in the repository yet. The chapters explain the tools; this page is for looking things up.

The **Status** column has three values.

- **In use** – code in the repository uses the tool today.
- **Declared** – the tool is listed as a dependency in [`pyproject.toml`](../../pyproject.toml) and pinned in [`requirements.lock`](../../requirements.lock), and no code uses it yet.
- **Planned** – not in the repository yet.

## Foundation (Phase 0)

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| Python 3.12 | The programming language, fixed to one minor version | The same behaviour on the laptop, in containers and in automatic checks | In use | 6 |
| venv | Python's built-in tool for virtual environments | Keeps this project's libraries separate from all others | In use | 6 |
| hatchling | A build backend | Makes the code installable, so `import demand` works from anywhere | In use | 5 |
| uv | A fast Python packaging tool, used here to create the lock file | One lock file valid on Windows and Linux | In use | 6 |
| pydantic, pydantic-settings | Libraries that check data against typed classes | Settings that are checked when the program starts | In use | 7 |
| PyYAML | Reads YAML files | A config file that people can read | In use | 7 |
| python-dotenv | Reads a `.env` file into environment variables | Local values and secrets in one file that is not in git | In use | 7 |
| Pandera | Schema checks for pandas tables | Data contracts between the data layers | In use (the silver and gold stages validate every table before writing it; the `validate` command checks the files of a layer with the same schemas, in lazy mode) | 8, 17, 18, 20 |
| pandas | Library for tables of data | The cleaning rules, time zones and daylight saving, counting pickups on a complete grid, and the table type that the schemas check | In use | 8, 16, 17, 18 |
| NumPy | Library for fast calculations on arrays of numbers | The lags and rolling windows of the feature function; the three metrics | In use | 19, 22 |
| Typer | Library for building command-line programs | One entry point on every operating system | In use | 9 |
| pytest, pytest-cov | Test runner and coverage measurement | Rules of the project that run after every change | In use | 10 |
| ruff | Linter and formatter | One style; common mistakes found automatically | In use | 11 |
| mypy | Static type checker | Type mistakes found without running the code | In use | 11 |
| pre-commit | Runs checks before each git commit | Problems are stopped before they enter the history | In use (each clone runs `pre-commit install` once) | 11 |

## Data pipeline (Phase 1)

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| NYC TLC trip records | Public monthly files of taxi trips | The demand data | In use (downloaded by the ingest stage, cleaned by the silver stage, counted per zone and hour by the gold stage) | 14, 15, 16, 18 |
| Open-Meteo | Free weather service, no key needed | Weather features | In use (downloaded by the ingest stage, cleaned by the silver stage, joined to the demand table by the gold stage) | 15, 17, 18 |
| httpx | Library for web requests | Downloads with time limits and retries; `MockTransport` for tests without a network | In use | 14 |
| pyarrow | Reader and writer for Parquet files | The file format of the raw, silver and gold tables; the footer check before a file is accepted; the filter that keeps the rows after the training period out of the backtest | In use | 14, 17, 18, 23 |
| DuckDB | A database engine that runs inside the Python process | Reads and counts large files without loading everything into memory | Declared (used in this book to look at the raw data and to check a cleaning rule) | 15, 16 |
| holidays | Library of public holidays | The `is_holiday` feature: the federal holidays of the United States | In use | 19 |
| DVC | Data Version Control: versions data files and pipeline steps next to git | Runs the six stages in order, skips a stage when nothing it depends on changed, and stops when a quality gate fails; its checksum of the gold folder is the data version of a training run | In use (pipeline and local cache; no remote, so the data itself is not shared) | 21, 26 |

## Modelling (Phase 2)

The baseline, the metrics and the backtest of Chapters 22 and 23 need no new tool. They use pandas, NumPy and pyarrow from the tables above. The model, its record, its tuning and its explanation use the tools of this table, except scikit-learn.

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| LightGBM | A library for gradient-boosted decision trees | The forecasting model: 600 trees with the Poisson objective and the zone as a category, behind the `Forecaster` interface | In use (parameters from the config, tuned with Optuna and pasted by hand) | 24, 25, 27 |
| MLflow | Records of training runs, and a model registry | Every training run is recorded with its parameters, scores, git commit and data version; the model is stored with a signature and registered as a version of `taxi-demand` | In use (a local SQLite file and the folder `mlruns`; no alias is set, so there is no champion yet) | 26 |
| scikit-learn | General machine-learning library | LightGBM's class `LGBMRegressor` follows its interface and needs it installed | Declared (no code of the project imports it) | – |
| Optuna | Library that searches for good model parameters | `python -m demand tune`: a seeded search over five LightGBM hyper-parameters, at most 30 trials or 900 seconds, that prints a block for the config | In use (only `tuning.py` imports it) | 27 |
| SHAP | Library that explains model predictions | Splits each forecast of the final model into contributions per input; a training run stores the mean sizes and a summary plot | In use (only `explain.py` imports it) | 28 |
| matplotlib | Library for drawing charts | Draws the SHAP summary plot into a PNG file, in memory and without a window | In use (only `explain.py` imports it) | 28 |

## Serving (Phase 3)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| FastAPI, uvicorn | A web framework and the server that runs it | The prediction service | Declared |
| prometheus-client | Library that publishes numbers from a running program | Request counts and response times | Declared |

## Containers (Phase 4)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| Docker | Packs a program and its dependencies into an image | The same runtime everywhere | Planned ([`.dockerignore`](../../.dockerignore) is present) |
| Docker Compose | Starts several containers together | The development stack | Planned |

## Monitoring and retraining (Phase 5)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| Prefect | A workflow scheduler | Runs the pipeline steps on a schedule | Declared |
| Evidently | Library for data and model monitoring | Drift and accuracy reports | Declared |
| Prometheus | A database for measurements over time | Stores service and monitoring numbers | Planned |
| Grafana | Dashboards | Shows the measurements | Planned |

## Kubernetes (Phase 6)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| Kubernetes with kind | A system that runs and scales containers; kind runs it locally inside Docker | A local "production" target | Planned |
| Kustomize | Tool for variants of Kubernetes configuration | One base configuration and one variant for kind | Planned |

## CI/CD and demo (Phase 7)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| GitHub Actions | Service that runs automatic checks and delivery | Lint, types, tests and security checks on every push | Planned |
| GHCR | GitHub's storage for container images | Stores the built images | Planned |
| pip-audit, Trivy | Scanners for known security problems in libraries and images | Security checks | pip-audit declared, Trivy planned |
| Streamlit on Hugging Face Spaces | A simple web-app library and a free host | The public demo | Planned |

---

[Previous: Chapter 28 – Explaining a forecast: SHAP](28-explaining-the-model-with-shap.md) | [Contents](README.md) | [Next: Appendix B – Glossary](appendix-b-glossary.md)
