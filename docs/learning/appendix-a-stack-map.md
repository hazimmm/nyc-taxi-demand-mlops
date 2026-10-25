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
| pydantic, pydantic-settings | Libraries that check data against typed classes | Settings that are checked when the program starts; the shapes of the requests and responses of the API | In use | 7, 30 |
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
| httpx | Library for web requests | Downloads with time limits and retries; `MockTransport` for tests without a network; FastAPI's test client sends its requests with it | In use | 14, 30 |
| pyarrow | Reader and writer for Parquet files | The file format of the raw, silver and gold tables; the footer check before a file is accepted; the filter that keeps the rows after the training period out of the backtest | In use | 14, 17, 18, 23 |
| DuckDB | A database engine that runs inside the Python process | Reads and counts large files without loading everything into memory | Declared (used in this book to look at the raw data and to check a cleaning rule) | 15, 16 |
| holidays | Library of public holidays | The `is_holiday` feature: the federal holidays of the United States | In use | 19 |
| DVC | Data Version Control: versions data files and pipeline steps next to git | Runs the six stages in order, skips a stage when nothing it depends on changed, and stops when a quality gate fails; its checksum of the gold folder is the data version of a training run | In use (pipeline and local cache; no remote, so the data itself is not shared) | 21, 26 |

## Modelling (Phase 2)

The baseline, the metrics and the backtest of Chapters 22 and 23 need no new tool. They use pandas, NumPy and pyarrow from the tables above. The model, its record, its tuning and its explanation use the tools of this table, except scikit-learn. The promotion gate of Chapter 29 needs no new tool: it reads and moves aliases in MLflow.

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| LightGBM | A library for gradient-boosted decision trees | The forecasting model: 600 trees with the Poisson objective and the zone as a category, behind the `Forecaster` interface | In use (parameters from the config, tuned with Optuna and pasted by hand) | 24, 25, 27 |
| MLflow | Records of training runs, and a model registry | Every training run is recorded with its parameters, scores, git commit and data version; the model is stored with a signature and registered as a version of `taxi-demand`; the alias `champion` names the version to serve, and the promotion gate moves it; the service resolves the alias once and loads that version from a temporary copy | In use (on the laptop a local SQLite file and the folder `mlruns`; in Docker Compose an MLflow server with its store in two named volumes; on Kubernetes the same server with its store on a PersistentVolumeClaim; both servers run with `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`; only `tracking.py`, `registry.py` and `serving/model_loader.py` import it) | 26, 29, 31, 33, 39 |
| scikit-learn | General machine-learning library | LightGBM's class `LGBMRegressor` follows its interface and needs it installed | Declared (no code of the project imports it) | – |
| Optuna | Library that searches for good model parameters | `python -m demand tune`: a seeded search over five LightGBM hyper-parameters, at most 30 trials or 900 seconds, that prints a block for the config | In use (only `tuning.py` imports it) | 27 |
| SHAP | Library that explains model predictions | Splits each forecast of the final model into contributions per input; a training run stores the mean sizes and a summary plot | In use (only `explain.py` imports it) | 28 |
| matplotlib | Library for drawing charts | Draws the SHAP summary plot into a PNG file, in memory and without a window | In use (only `explain.py` imports it) | 28 |

## Serving (Phase 3)

The service also needs the libraries of the modelling table: MLflow loads the model, and the model is a LightGBM forecaster.

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| FastAPI | A web framework, built on the library Starlette: maps paths to Python functions, checks requests with Pydantic, writes JSON and the page `/docs` | The prediction service: five endpoints in `serving/app.py`, and a test client that calls them without a network | In use | 30, 31 |
| uvicorn | A web server: receives HTTP requests on a port and hands them to the application through the ASGI interface | `python -m demand serve` runs the service with one worker process on port 8000 | In use | 30, 31 |
| prometheus-client | Library that publishes numbers from a running program in the Prometheus text format | `/metrics`: requests by route and status, response times, the forecasts returned, and the served version | In use (a Prometheus server collects it in Docker Compose) | 31, 33 |

## Containers (Phase 4)

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| Docker | Builds images that hold a program with Python, its libraries and the files of the operating system, and runs them as containers | The same runtime on every machine: two images, one for the service (`docker/api.Dockerfile`) and one for the training commands and the MLflow server (`docker/pipeline.Dockerfile`), each from a base pinned by digest and with its own lock file | In use (Docker Desktop with the Linux engine; DVC stays on the host) | 32 |
| python:3.12.15-slim-trixie | The official Python image on Debian 13, without compilers and manuals | The base of both images, 179 MB | In use (pinned by tag and digest) | 32 |
| Docker Compose | Starts several containers together from one file, with their volumes, mounts, ports and network | `docker-compose.yml`: MLflow, the API, Prometheus and Grafana, since Phase 5 the Pushgateway and the Prefect server, the training commands behind the profile `pipeline` and the scheduler behind the profile `schedule`; health checks decide the order of start | In use | 33, 36, 38 |
| Prometheus server | A database for measurements over time that asks its targets for their numbers | Collects `/metrics` of the API and the monitoring numbers held by the Pushgateway every 15 seconds; its settings are in `monitoring/prometheus/prometheus.yml` | In use (in Docker Compose, image pinned by digest; no alert rules) | 33, 36 |
| Grafana | A web application for dashboards | The dashboards "Prediction API", with four panels, and "Model monitoring", with nine, set up from files in `monitoring/grafana/`; dashboards are visible without a login | In use (in Docker Compose, image pinned by digest) | 33, 36 |

## Monitoring and retraining (Phase 5)

The replay of Chapter 34 needs no new tool, and neither does the command `forecast` that runs it by hand. Its simulated clock and its scores are plain Python with pandas, pyarrow and Pandera; it loads the champion with MLflow and forecasts with the code of the service. Its files are in `data/predictions/` and its report is `reports/replay_report.json`. The accuracy check of monitoring needs no new tool either; the two drift checks use Evidently. Retraining (Chapter 37) needs no new tool: it trains with LightGBM, records and registers with MLflow, and writes `reports/retrain_report.json`.

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| Prefect | An orchestrator: runs flows on a schedule, one at a time, records every run with its logs, and retries tasks | `python -m demand schedule` serves the flow `daily-step` of `flows/orchestration.py`, which calls `run_day` of `flows/retrain_flow.py` for each simulated day as the task `day`, and pushes the monitoring numbers as a task with 2 retries; one run a minute in the stack (a demo), a nightly cron in production | In use (version 3.8.8; only `flows/orchestration.py` imports it; the client runs from the pipeline image; without a server Prefect starts a temporary one; on Kubernetes a CronJob runs the same step without Prefect) | 38, 39 |
| prefecthq/prefect:3.8.8-python3.12 | Prefect's official image | The Prefect server of the stack, service `prefect` on port 4200: it keeps the deployment, the schedule and the record of every run, in a SQLite database in the volume `prefect-data`. Chosen over the pipeline image because the server could not schedule with SQLAlchemy 2.1.4 of `requirements.lock`; this image has SQLAlchemy 2.0.45 | In use (in Docker Compose, pinned by tag and digest; runs as root, as published; 12 days old when pinned, an exception to the two-week rule) | 38 |
| Evidently | Library for data and model monitoring | The input-drift and prediction-drift checks: the normed Wasserstein distance per column with a fixed threshold of 0.1, and the HTML page `reports/monitoring_input_drift.html` | In use (version 0.7.23; only `monitoring/checks.py` imports it) | 35 |
| prometheus-client | See the Serving table | Builds the gauges of a monitoring report and sends them to the Pushgateway with one HTTP PUT | In use (only `monitoring/metrics.py`, `serving/metrics.py` and `serving/app.py` import it) | 36 |
| Prometheus Pushgateway | A small server that keeps the last numbers a batch job pushed, for Prometheus to scrape | `python -m demand monitor --push-to http://localhost:9091` sends the numbers there, because a command that has ended cannot be scraped | In use (in Docker Compose, image `prom/pushgateway:v1.11.3` pinned by digest; numbers kept in memory only) | 36 |
| Prometheus and Grafana | See the Containers table | Store and show the numbers of drift and accuracy | In use: the dashboard "Model monitoring" | 36 |

## Kubernetes (Phase 6)

The cluster runs the same two images as Docker Compose. The daily step needs no new library: the CronJob runs the new command `python -m demand step`, which calls `run_days` without Prefect. Prometheus, the Pushgateway and Grafana stay in Docker Compose.

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| Kubernetes | A system that runs containers on a group of machines and keeps them in the state its objects describe | The files of `k8s/base/`: the API as a Deployment of 2 to 4 pods behind a Service, with probes and measured resources; MLflow as one pod with its store on a PersistentVolumeClaim; the data on a volume of its own; settings as a ConfigMap and a Secret; the daily step as a CronJob; a namespace with the "restricted" Pod Security Standard | In use (version 1.36.4 on kind; one node) | 39, 40 |
| kubectl | The command-line tool that talks to the API server of a cluster | Applies the manifests (`kubectl apply -k`), copies the data in, starts Jobs by hand, opens port-forwards, and shows pods, logs and the HPA | In use (version 1.36.1, the one that comes with Docker Desktop) | 39, 40 |
| kind | "Kubernetes in Docker": runs each node of a cluster as a Docker container | A local cluster of one node from `k8s/kind-cluster.yaml`; `kind load docker-image` copies the project's images into the node | In use (v0.33.0, checksum checked; node image `kindest/node:v1.36.4` pinned by digest) | 39, 40 |
| Kustomize | A tool that builds manifests from a base and overlays, built into `kubectl` | One base for every cluster and one overlay for kind, which sets the image tags, never pulls images, and suspends the CronJob; it names the ConfigMap and the Secret by the hash of their content | In use (version 5.8.1, inside `kubectl`; no Helm) | 39 |
| metrics-server | A cluster add-on that measures the CPU and memory of every pod | The CPU readings that the HorizontalPodAutoscaler of the API scales by, and `kubectl top` | In use (v0.9.0, its official manifest vendored in `k8s/overlays/kind/metrics-server/`, image pinned by digest; `--kubelet-insecure-tls` for kind only) | 39, 40 |
| kubeconform | A validator that checks manifests against the official schemas of Kubernetes | Checks the 21 objects of `k8s/` without a cluster | In use by hand (image `ghcr.io/yannh/kubeconform:v0.8.0` pinned by digest; not yet in CI) | 40 |

## CI/CD and demo (Phase 7)

The checks of CI need no new library for the code: they run ruff, mypy and pytest from `requirements.lock`, in the image `python:3.12.15-slim-trixie` of the Containers table.

| Tool | What it is | Problem it solves here | Status | Chapter |
|---|---|---|---|---|
| GitHub Actions | GitHub's service that runs workflows of jobs on fresh runners | `.github/workflows/ci.yml`: lint, types and tests, the dependency audit, and both images built and scanned, on every push to `dev` and `main` and on every pull request | In use (three jobs on `ubuntu-24.04` runners; actions pinned by full commit hash; the token can only read; the first run on GitHub was still to come when Chapter 41 was written) | 41 |
| pip-audit | A scanner of the Python Packaging Authority that checks a list of Python packages against known vulnerabilities | `tools/audit_dependencies.py` audits `requirements.lock` and the two image locks, without the findings that `.trivyignore.yaml` accepts on that day | In use (version 2.10.1, in the `dev` group and in CI) | 41 |
| Trivy | A scanner for known vulnerabilities in images: the packages of the operating system and the libraries installed in it | Scans the api and pipeline images in CI for HIGH and CRITICAL findings that have a fix; reads `.trivyignore.yaml` with its expiry days | In use in CI (version 0.75.0, official image `aquasec/trivy` pinned by digest, not through an action) | 41 |
| actionlint | A linter for GitHub Actions workflows; it also runs shellcheck on every `run:` step | Found no problem in `ci.yml` | In use by hand (version 1.7.12; not in CI) | 41 |
| GHCR | GitHub's storage for container images | Stores the built images | Planned (CI builds the images but does not push them) | – |
| Streamlit on Hugging Face Spaces | A simple web-app library and a free host | The public demo | Planned | – |

---

[Previous: Chapter 41 – Every push checked: continuous integration and known vulnerabilities](41-every-push-checked.md) | [Contents](README.md) | [Next: Appendix B – Glossary](appendix-b-glossary.md)
