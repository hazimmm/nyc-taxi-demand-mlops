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
| Pandera | Schema checks for pandas tables | Data contracts between the data layers | In use (schemas defined; the gates run from Phase 1) | 8 |
| pandas | Library for tables of data | Table work, and the table type that the schemas check | In use | 8 |
| Typer | Library for building command-line programs | One entry point on every operating system | In use | 9 |
| pytest, pytest-cov | Test runner and coverage measurement | Rules of the project that run after every change | In use | 10 |
| ruff | Linter and formatter | One style; common mistakes found automatically | In use | 11 |
| mypy | Static type checker | Type mistakes found without running the code | In use | 11 |
| pre-commit | Runs checks before each git commit | Problems are stopped before they enter the history | In use (each clone runs `pre-commit install` once) | 11 |

## Data pipeline (Phase 1)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| NYC TLC trip records | Public monthly files of taxi trips | The demand data | Planned (addresses are in the config) |
| Open-Meteo | Free weather service, no key needed | Weather features | Planned (addresses are in the config) |
| DuckDB | A database engine that runs inside the Python process | Reads and counts large files without loading everything into memory | Declared |
| pyarrow | Reader and writer for Parquet files | The file format of every data layer | Declared |
| httpx | Library for web requests | Downloads with time limits and retries | Declared |
| holidays | Library of public holidays | The `is_holiday` feature | Declared |
| DVC | Data Version Control: versions data files and pipeline steps next to git | A pipeline that can be repeated; data versions without data in git | Declared |

## Modelling (Phase 2)

| Tool | What it is | Problem it solves here | Status |
|---|---|---|---|
| scikit-learn | General machine-learning library | Metrics and helpers | Declared |
| LightGBM | A model built from many decision trees | The main forecasting model | Declared |
| Optuna | Library that searches for good model parameters | Tuning within a fixed budget | Declared |
| SHAP | Library that explains model predictions | Shows which features drive the forecasts | Declared |
| MLflow | Records of training runs, and a model registry | Every run can be traced; holds the champion and challenger models | Declared |

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

[Previous: Chapter 13 – A code review of our own foundation](13-a-code-review-of-our-own-foundation.md) | [Contents](README.md) | [Next: Appendix B – Glossary](appendix-b-glossary.md)
