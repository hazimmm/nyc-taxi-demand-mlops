# Learning guide

A study guide for this repository, written for a junior who knows Python, pandas and basic modelling, and wants to understand how a production-style ML project is built.

**Contents**

1. [How to use this guide](#1-how-to-use-this-guide)
2. [The project in one page](#2-the-project-in-one-page)
3. [Stack map](#3-stack-map)
4. [Core ideas to understand first](#4-core-ideas-to-understand-first)
5. [Lessons by push](#5-lessons-by-push)
   - [Push 1 – Project foundation (Phase 0)](#push-1--project-foundation-phase-0)
6. [Glossary](#6-glossary)
7. [Where to learn more](#7-where-to-learn-more)

---

## 1. How to use this guide

- Read sections 2 to 4 once. They explain what the system does and the forecasting ideas that every later decision depends on.
- Section 5 has one lesson per push to the repository, oldest first. Each lesson uses the same five parts: *What was added*, *Concepts*, *Try it yourself*, *Check your understanding*, *Interview angle*.
- This guide grows. Before every push, a new lesson is added to section 5, the *Status* column of the stack map is updated, and new terms go into the glossary.
- Keep the code open while you read. Every concept names the file and the function or class where you can see it. Run the commands; do not only read them.

---

## 2. The project in one page

**What it predicts.** For each of 20 busy New York City taxi zones, the number of yellow-taxi pickups in each hour of the next day. This is called a *day-ahead hourly forecast*. "Day-ahead" means the forecast for an hour is made at least 24 hours before that hour starts.

**For whom.** Think of a fleet operator who decides today how many drivers to place in each zone tomorrow. They need the numbers one day early. A forecast that is only available one hour before is useless for planning shifts. This is why the horizon is 24 hours, and the horizon controls almost every design decision in the project.

**The path from raw data to a served forecast** (the full diagram is in [architecture.md](architecture.md)):

1. **Ingest.** Download monthly trip files from the NYC Taxi and Limousine Commission (TLC) and hourly weather from Open-Meteo. Store them unchanged. This is the *raw* layer.
2. **Clean.** Remove duplicate and impossible trips (for example, a trip that ends before it starts). The result is the *silver* layer: one row is one valid trip.
3. **Aggregate and build features.** Count pickups per zone per hour, join weather, and compute the model inputs (for example, "pickups in this zone at the same hour yesterday"). This is the *gold* layer.
4. **Check.** At the exit of silver and of gold, a *data contract* checks the table. If the check fails, the pipeline stops.
5. **Train.** Fit a simple baseline and a LightGBM model, and compare them on past periods.
6. **Register.** Store the model in a model registry. A new model becomes the production model (the *champion*) only if it beats the current champion and the baseline.
7. **Serve.** An API returns forecasts on request. A batch job writes the full day-ahead forecast every simulated day.
8. **Monitor.** Compare new data and new errors with the training period. If they move too far, retrain, with safety rules.

**What exists today.** Only the foundation: project packaging, configuration, data contracts, the command-line skeleton, logging, tests and code-quality tools. No data is downloaded and no model is trained yet. The foundation comes first because every later part reads the same settings and the same column definitions.

---

## 3. Stack map

One row per tool in the planned stack. *Phase* is where the tool's code arrives. Status has three values:

- **In use** – code in the repository uses it today.
- **Declared** – it is listed as a dependency in [`pyproject.toml`](../pyproject.toml) and pinned in [`requirements.lock`](../requirements.lock), but no committed code uses it yet.
- **Planned** – not in the repository yet.

| Tool | What it is | Problem it solves here | Phase | Status |
|---|---|---|---|---|
| Python 3.12 | The programming language, fixed to one minor version | Same behaviour on the laptop, in containers and in CI | 0 | In use |
| venv | Python's built-in virtual environment tool | Keeps this project's packages separate from every other project | 0 | In use |
| hatchling | A build backend: turns the source folder into an installable package | Makes `import demand` and `python -m demand` work from anywhere | 0 | In use |
| uv | A fast Python package tool; used here only to compile the lock file | One lock file that is valid on Windows and Linux | 0 | In use |
| pydantic + pydantic-settings | Libraries that validate data against typed classes | Configuration that is checked when the program starts, not when it crashes later | 0 | In use |
| PyYAML | Reads YAML files | Human-readable config file | 0 | In use |
| Pandera | Schema validation for pandas dataframes | Data contracts between pipeline layers | 0 | In use (schemas defined; the gates run in Phase 1) |
| pandas | Dataframe library | Table transformations and the type the schemas validate | 0 | In use |
| Typer | Library for building command-line programs from typed functions | One entry point (`python -m demand ...`) on every operating system | 0 | In use |
| pytest + pytest-cov | Test runner and coverage measurement | Proves each rule works; fails the run if coverage drops below 80% | 0 | In use |
| ruff | Linter and formatter | One style and many common bugs caught automatically | 0 | In use |
| mypy | Static type checker | Finds type errors without running the code | 0 | In use |
| pre-commit | Runs checks before each git commit | Bad code is stopped before it enters history | 0 | In use (configured) |
| NYC TLC trip records | Public monthly Parquet files of taxi trips | The demand data | 1 | Planned (URLs are in the config) |
| Open-Meteo | Free weather API, no key needed | Weather features | 1 | Planned (URLs are in the config) |
| DuckDB | In-process SQL engine for analytics | Reads and aggregates large Parquet files without loading them all into memory | 1 | Declared |
| pyarrow | Parquet reader and writer | File format of every data layer | 1 | Declared |
| httpx | HTTP client | Downloads with timeouts and retries | 1 | Declared |
| holidays | Library of public-holiday calendars | The `is_holiday` feature | 1 | Declared |
| DVC | Data Version Control: tracks data files and pipeline stages next to git | Reproducible pipeline; data is versioned without putting it in git | 1 | Declared |
| scikit-learn | General machine-learning library | Metrics and utilities | 2 | Declared |
| LightGBM | Gradient-boosted decision trees | The main forecasting model | 2 | Declared |
| Optuna | Hyper-parameter search library | Tunes LightGBM inside a fixed budget | 2 | Declared |
| SHAP | Model explanation library | Shows which features drive the forecasts | 2 | Declared |
| MLflow | Experiment tracking and model registry | Records every training run; holds the `champion` and `challenger` models | 2 | Declared |
| FastAPI + uvicorn | Web framework and the server that runs it | The prediction API | 3 | Declared |
| prometheus-client | Library that exposes numeric metrics from a Python process | Request counts and latency for the API | 3 | Declared |
| Docker | Packages a program with its dependencies into an image | The same runtime everywhere | 4 | Planned ([`.dockerignore`](../.dockerignore) is already present) |
| Docker Compose | Starts several containers together | The development stack: API, MLflow, Prefect, Prometheus, Grafana | 4 | Planned |
| Prefect | Workflow orchestrator | Schedules the pipeline stages in "production" | 5 | Declared |
| Evidently | Data and model monitoring library | Drift and accuracy reports | 5 | Declared |
| Prometheus | Time-series database for metrics | Stores API and monitoring metrics | 5 | Planned |
| Grafana | Dashboards | Shows the metrics | 5 | Planned |
| Kubernetes (kind) | Container orchestrator; kind runs a cluster inside Docker | A local "production" target | 6 | Planned |
| Kustomize | Tool for Kubernetes configuration variants | One base configuration plus a kind-specific overlay | 6 | Planned |
| GitHub Actions | CI/CD service | Runs lint, types, tests and security scans on every push | 7 | Planned |
| GHCR | GitHub Container Registry | Stores the built images | 7 | Planned |
| pip-audit, Trivy | Security scanners for Python packages and container images | Known-vulnerability checks in CI | 7 | pip-audit declared, Trivy planned |
| Streamlit on Hugging Face Spaces | Simple web-app library and a free host | Public demo | 7 | Planned |

Tools that are not *In use* have only this row for now. Their detailed lessons are added in section 5 when their code lands.

---

## 4. Core ideas to understand first

These six ideas are specific to forecasting and to ML systems. Courses and Kaggle competitions often hide them, because the dataset arrives already split and nobody has to serve the model.

### 4.1 Forecast horizon, and why "lag 1 hour" is impossible here

The **forecast horizon** is the time between the moment a forecast is made and the moment it describes. Here it is 24 hours (`features.horizon_hours: 24` in [`configs/config.yaml`](../configs/config.yaml)).

A **lag feature** is a past value of the target used as a model input. "Lag 24 h" for Tuesday 18:00 is the number of pickups on Monday 18:00.

**Taxi example.** You must forecast pickups in zone 132 for Tuesday 18:00. The forecast has to exist by Monday 18:00 at the latest. A feature "pickups one hour ago" means pickups on Tuesday 17:00. On Monday that number does not exist. In a notebook you can compute it easily with `df["pickups"].shift(1)`, because the historical table contains every hour. The model will then look very accurate, and it can never be used.

**The rule in this project:** every feature for hour *t* uses only information from *t* − 24 h or earlier. The configured lags are 24, 48 and 168 hours (same hour yesterday, two days ago, last week). Rolling statistics (for example the mean of the last 24 hours) are planned to be shifted back by 24 hours for the same reason. The rule is partly enforced in code already: see "config that refuses to leak" in Push 1.

### 4.2 Data leakage

**Data leakage** means the model learns from information it will not have when it is really used. The result is always the same: the test score is better than the real performance. Leakage does not raise an error, so you must design against it. The plan found three cases in this project.

| Case | How it leaks | Decision in this project |
|---|---|---|
| **Lag** | A lag shorter than the horizon uses pickups that do not exist yet at prediction time (4.1). | Only lags ≥ 24 h. The settings object refuses to load shorter lags. |
| **Weather** | Using the *observed* weather of Tuesday 18:00 as a feature. On Monday you only have a weather *forecast* for Tuesday, and the forecast is sometimes wrong. A model trained on perfect weather trusts weather too much. | Use Open-Meteo's archived *forecast* data (`data.weather.forecast_url`). If it is missing for a period, fall back to observed weather (`archive_url`) and record this as a known limitation. |
| **Zone selection** | Choosing the "top 20 zones" from all six months uses the test months. The choice itself is information from the future. | Zones are chosen from the training period only (up to `data.train_end_date`), then written into `data.zones` and never recomputed. |

The zone case is the easiest to miss, because no feature is involved. Any decision that looks at the data (which zones, which outlier threshold, which scaling) can leak if it looks at the test period.

### 4.3 Time-based splits versus random splits

A **random split** puts rows into train and test sets by chance. For time series this is wrong. With a random split, the model trains on Tuesday 17:00 and Tuesday 19:00 and is tested on Tuesday 18:00. It can interpolate between its neighbours. In real use, everything after "now" is unknown.

A **time-based split** puts everything before a date into training and everything after it into testing. In this project the date is `data.train_end_date: 2025-04-30`. January to April 2025 are for training; May and June are never seen during training.

One split gives one number, and one number can be lucky. So Phase 2 uses a **rolling-origin backtest**: train up to a date, test on the next 7 days, move the date forward, repeat (`backtest.n_folds: 4`, `backtest.fold_days: 7`). The average over the folds is a more reliable estimate.

### 4.4 Training/serving skew

**Training/serving skew** means the features are computed one way when the model is trained and a slightly different way when it is used. Example: the training code computes `is_weekend` as Saturday and Sunday; the API code, written weeks later by someone else, uses a library where the week starts on Sunday and marks Sunday and Monday. The model receives inputs it was not trained on. Nothing crashes. The forecasts are just worse.

**Decision:** one feature function, imported by training, by the batch forecast job and by the API, plus a test that checks all three give the same output. The alternative was to write the features in SQL for training and again in Python for the API. It lost because two implementations always drift apart.

A first part of this defence is already in the repository: `feature_columns(settings)` in [`src/demand/schemas.py`](../src/demand/schemas.py) is the single definition of which columns the model reads, and in which order.

### 4.5 The simulated clock, and why it is honest

TLC publishes each month of trip data about two months late. So this project cannot have a live data feed. A dashboard that claimed "real-time monitoring" would be fake.

**Decision:** a **simulated clock**. The model is trained on January to April 2025. Then May and June are replayed one day at a time as if they were happening now: each simulated day, the batch job writes tomorrow's forecast; later, the actual pickups for that day "arrive" and the error is computed. In the code, the first replayed day is `DataConfig.replay_start_date` in [`src/demand/config.py`](../src/demand/config.py): the day after `train_end_date`.

Why this is honest:

- The model never sees the replay months during training, so the monitoring numbers are real out-of-sample results.
- The delay of the actual values is also simulated. In real systems the true answer (the **ground truth**) always arrives late, so accuracy cannot be checked immediately. This is why the plan has two monitors: **drift** (has the input data changed? available at once) and **accuracy** (available when the actuals arrive).
- The README will say openly that the clock is simulated. Stating a limitation is better than hiding it; interviewers ask about it.

### 4.6 Medallion layers: raw, silver, gold

The **medallion architecture** stores data in layers of increasing quality.

| Layer | Folder | Contents | Rule |
|---|---|---|---|
| Raw (also called bronze) | `data/raw/` | Files exactly as downloaded, plus a manifest (source URL, download time, row count, checksum) | Never modified |
| Silver | `data/silver/` | Cleaned, typed, de-duplicated trips and weather | One row is one valid trip, or one weather hour |
| Gold | `data/gold/` | Hourly pickups per zone, then the model-ready feature table | What training, serving and monitoring read |

**Each layer is built only from the layer before it.** Gold never reads raw files. Three reasons:

1. **Every table has one origin.** If a gold number looks wrong, you check silver; if silver is right, the bug is in the gold step. If gold also read raw files, some rows would skip the cleaning rules, and you could not tell which.
2. **Rebuilds are cheap and safe.** Raw is immutable, so if a cleaning rule changes you rebuild silver and gold without downloading again, and you can always compare the old and new result.
3. **Quality gates have a clear place.** A contract check stands at the exit of silver and of gold. Because nothing goes around a layer, nothing goes around its gate.

The alternative is one long script from download to feature table. It is shorter to write and loses on all three points. The folder locations are already defined as properties (`raw`, `silver`, `gold`, `predictions`) of `PathsConfig` in [`src/demand/config.py`](../src/demand/config.py).

---

## 5. Lessons by push

> **Template for every push.** Copy these five headings for each new lesson, newest last:
> **What was added** · **Concepts** (for each: *What it is* / *Why here* / *Where* / *Without it*) · **Try it yourself** · **Check your understanding** · **Interview angle**

---

## Push 1 – Project foundation (Phase 0)

Commit `72b97b4`, message `chore: project foundation (phase 0)`. No pipeline code yet. This push creates the rules that all later code must follow.

### What was added

| File | Purpose |
|---|---|
| [`pyproject.toml`](../pyproject.toml) | Package definition, dependencies, and the settings of ruff, mypy, pytest and coverage |
| [`requirements.lock`](../requirements.lock) | Exact version of every package, direct and indirect (281 packages) |
| [`configs/config.yaml`](../configs/config.yaml) | All configuration values: dates, zones, thresholds, model parameters |
| [`src/demand/config.py`](../src/demand/config.py) | Typed, validated settings loaded from the YAML file and environment variables |
| [`src/demand/schemas.py`](../src/demand/schemas.py) | Column-name constants and Pandera schemas for silver, gold and predictions |
| [`src/demand/cli.py`](../src/demand/cli.py) | The Typer command-line program |
| [`src/demand/__main__.py`](../src/demand/__main__.py) | Makes `python -m demand` work |
| [`src/demand/logs.py`](../src/demand/logs.py) | JSON logging |
| `src/demand/{data,models,serving,monitoring,flows}/__init__.py` | Empty sub-packages; one per later phase |
| [`tests/conftest.py`](../tests/conftest.py), `tests/unit/test_*.py` | Shared test fixture and 56 unit tests |
| [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) | Checks that run before a commit |
| [`.gitattributes`](../.gitattributes), [`.gitignore`](../.gitignore), [`.dockerignore`](../.dockerignore), [`.env.example`](../.env.example) | Line endings; files kept out of git; files kept out of Docker images; documented environment variables |
| [`docs/architecture.md`](architecture.md), [`README.md`](../README.md), `LICENSE` | Design diagram, setup steps, MIT licence |

### Concepts

#### 1. Virtual environments

- **What it is.** A folder (here `.venv/`) with its own Python interpreter and its own installed packages. Packages installed there are invisible to other projects, and the other way round.
- **Why here.** This project pins 281 package versions. Another project on the same machine may need different versions of the same packages. Without isolation, installing one breaks the other.
- **Where.** `.venv/` is created by `python -m venv .venv` (see [`README.md`](../README.md)). It is listed in `.gitignore`, because it is large, specific to your operating system, and can be rebuilt from the lock file. All commands in this guide call `.venv\Scripts\python.exe` directly, so they work whether the environment is "activated" or not.
- **Without it.** "It works on my machine" bugs: the code runs for you because of a package you installed months ago for something else.

#### 2. The `src/` layout and an installable package

- **What it is.** A **package** is a folder of Python modules that can be imported by name. In the **src layout**, the package lives in `src/demand/`, not in the repository root. To import it, you must *install* it. An **editable install** (`pip install -e .`) does not copy the files; it tells Python "the package `demand` is in this folder", so your edits take effect immediately.
- **Why here.** The same code must run in four places: your shell, pytest, a Docker image and CI. If it only works when you start Python from the repository root, it will fail in the other three. Installing the package gives one import path everywhere. The alternative, a flat layout with `demand/` in the root, lost because Python then imports the folder by accident when you run from the root. The tests would pass even if the packaging were broken.
- **Where.** In `pyproject.toml`:

  ```toml
  [project.scripts]
  demand = "demand.cli:app"

  [tool.hatch.build.targets.wheel]
  packages = ["src/demand"]
  ```

  The second block tells the build backend (hatchling) where the package is. The first creates a command `demand` (on Windows, `.venv\Scripts\demand.exe`) that calls the `app` object in `demand/cli.py`.

  **Why `python -m demand` works.** `python -m <name>` means "find the importable package `<name>` and run its `__main__.py`". The package is importable because of the editable install. The whole of [`src/demand/__main__.py`](../src/demand/__main__.py) is:

  ```python
  """Allow ``python -m demand``."""

  from demand.cli import app

  if __name__ == "__main__":
      app()
  ```

- **Without it.** Imports that depend on the current folder, `sys.path` hacks at the top of scripts, and a Docker image that fails with `ModuleNotFoundError` on the first run.

#### 3. `pyproject.toml`, optional dependency groups, and the lock file

- **What it is.** `pyproject.toml` is the standard file that describes a Python project. It has three jobs here: package metadata (`[project]`), dependencies, and tool settings (`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`, `[tool.coverage.*]`). One file replaces `setup.py`, `requirements.txt`, `pytest.ini`, `mypy.ini` and `.coveragerc`.

  **Dependencies** are declared as ranges, for example `"pandas>=2.2"`. A range says what the code is compatible with.

  **Optional dependency groups** (also called *extras*) are named sets of additional packages:

  | Group | Contents | Who needs it |
  |---|---|---|
  | (core) | duckdb, pandas, pandera, pydantic, typer, ... | everything |
  | `ml` | lightgbm, mlflow, optuna, shap, scikit-learn, matplotlib | training |
  | `serving` | fastapi, uvicorn, prometheus-client | the API |
  | `monitoring` | evidently, prometheus-client | monitoring |
  | `orchestration` | prefect | scheduled flows |
  | `versioning` | dvc | the data pipeline |
  | `dev` | ruff, mypy, pytest, pytest-cov, pre-commit, pip-audit, uv, type stubs | developers and CI |
  | `all` | every group above | a full local install |

  A **lock file** records the exact version of every package, including the packages that your packages depend on (*transitive* dependencies). The first lines of `requirements.lock` show how it was made:

  ```text
  # This file was autogenerated by uv via the following command:
  #    uv pip compile pyproject.toml --all-extras --universal --python-version 3.12 -o requirements.lock
  aiohappyeyeballs==2.7.1
      # via aiohttp
  ```

  `--universal` produces one file that is valid on both Windows and Linux. The `# via` comments say which package required each line.

- **Why here.** Groups keep the images small: the API image (Phase 4) does not need Optuna or SHAP, and the training image does not need FastAPI. The lock file gives reproducibility: `pandas>=2.2` installs a different version next year, and the model results could change without any change in the code. Ranges and a lock file together are better than either alone. Pins only in `pyproject.toml` make upgrades painful and say nothing about transitive packages; ranges only are not reproducible.
- **Where.** `[project.dependencies]` and `[project.optional-dependencies]` in `pyproject.toml`; `requirements.lock`. The install order in the README matters:

  ```powershell
  .venv\Scripts\python -m pip install -r requirements.lock
  .venv\Scripts\python -m pip install -e . --no-deps
  ```

  The first line installs the pinned versions. The second installs this project in editable mode, and `--no-deps` stops pip from choosing versions again.
- **Without it.** A colleague, or CI, installs newer versions than you have. A test fails for them and passes for you, and nobody can say which of 281 packages changed.

`requires-python = ">=3.12,<3.13"` fixes the Python minor version for the same reason.

#### 4. Typed settings with pydantic-settings

- **What it is.** One object, `Settings`, holds every configurable value. Each value has a declared type and, where useful, a constraint. The object is built once at start-up. If anything is wrong, the program stops at once with a message that names the bad field.
- **Why here.** The coding standard is "no hard-coded paths, dates or thresholds". Dates such as the end of the training period are used by ingestion, features, training and monitoring. If each module had its own copy, they would disagree. The alternative, reading the YAML into a plain `dict`, lost because a dict accepts anything: a typing mistake in a key gives `KeyError` (or worse, a silent default) hours into a run.
- **Where.** [`src/demand/config.py`](../src/demand/config.py). Four details are worth studying.

  **(a) YAML first, environment variables on top.** `load_settings()` reads the YAML file and passes its content to `Settings(...)`. The method `settings_customise_sources` then sets the priority:

  ```python
          """Let environment variables override the YAML values passed to ``__init__``."""
          return env_settings, init_settings
  ```

  Earlier in the tuple means higher priority, so an environment variable wins over the YAML file. The variable name is `DEMAND_` + section + `__` + key, for example `DEMAND_MLFLOW__TRACKING_URI`. Why both sources: the YAML file is committed and holds the defaults; environment variables change a value for one environment without editing a file. The MLflow address is a file path on a laptop, `http://mlflow:5000` inside Docker Compose, and something else in Kubernetes. The code is the same in all three. A different config file can be chosen with `DEMAND_CONFIG_FILE`.

  Note: the committed code reads real environment variables only. It does not read a `.env` file by itself (the `dotenv_settings` source is not returned). [`.env.example`](../.env.example) documents the variable names.

  **(b) Frozen models.** Every section inherits from `_Section`:

  ```python
  class _Section(BaseModel):
      """Base class for config sections: immutable and strict about unknown keys."""

      model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())
  ```

  `frozen=True` makes the object immutable: `settings.features.horizon_hours = 1` raises an error. Settings are shared by the whole program. If one function could change them, the result of another function would depend on call order, which is very hard to debug. When a test needs different values, it creates a changed *copy* with `model_copy(update=...)` (see `tests/conftest.py`).

  **(c) `extra="forbid"`.** An unknown key is an error. The test `test_unknown_key_is_rejected` writes `sead` instead of `seed` and expects a `ValidationError`. With the default behaviour (ignore unknown keys), the misspelled key would be dropped silently and you would believe you had changed the seed.

  **(d) Config that refuses to leak.** `FeatureConfig` has a validator, a function that pydantic runs after the fields are loaded:

  ```python
      @model_validator(mode="after")
      def _check_no_leakage(self) -> FeatureConfig:
          too_recent = [lag for lag in self.lags if lag < self.horizon_hours]
          if too_recent:
              raise ValueError(
                  f"features.lags {too_recent} are shorter than the {self.horizon_hours} h "
                  "forecast horizon and would leak information unavailable at prediction time"
              )
          return self
  ```

  This turns the rule from 4.1 into something the program enforces. If someone adds `1` to `features.lags` to "improve" the score, no command will start. A rule in a README is forgotten; a rule in a validator cannot be. Limit of this check: it covers `lags` only. The 24-hour shift of the rolling statistics must be enforced by the feature code and its leakage test in Phase 1.

  `DataConfig._check_period` does the same for dates: `train_end_date` must be inside the ingested period, and `zones` must not contain duplicates.

  **(e) Loaded once.** `get_settings()` is wrapped in `@lru_cache(maxsize=1)`, so the file is read once per process and every caller gets the same object.

- **Without it.** Magic numbers spread over many files; a typo that silently uses a default; a model trained with one training end date and monitored with another.

#### 5. Data contracts with Pandera

- **What it is.** A **data contract** is a precise, checkable description of a table: which columns, which types, which value ranges, which keys are unique. The step that writes the table (the *producer*) validates it before writing. The step that reads it (the *consumer*) can rely on it. Pandera expresses the contract as a `DataFrameSchema` and checks a pandas dataframe against it.
- **Why here.** In ML, bad data does not crash a program. A negative fare or a duplicated hour flows into the model and produces a slightly worse forecast, and nobody notices. A contract converts silent data problems into loud failures at the layer boundary. The alternative, `assert` lines spread through the pipeline, lost because nobody can read the full contract in one place and each stage checks different things.
- **Where.** [`src/demand/schemas.py`](../src/demand/schemas.py): `silver_trips_schema`, `silver_weather_schema`, `gold_demand_schema`, `gold_features_schema`, `predictions_schema`.

  **Schemas are built from settings.** Each schema is a function that takes `Settings`:

  ```python
              TRIP_DISTANCE: pa.Column(
                  "float64",
                  pa.Check.in_range(0, cleaning.max_distance_miles, include_min=False),
              ),
  ```

  The maximum distance comes from `cleaning.max_distance_miles` in the config. The cleaning code (Phase 1) will read the same value. So the cleaning rule and the gate that checks it cannot disagree. If the schema had its own number `100`, someone would change one and forget the other.

  **Column names are constants.** The file starts with lines such as `ZONE_ID = "zone_id"` and `TARGET = "pickups"`. All other code must write `schemas.TARGET`, never the string `"pickups"`. A typo in a constant name is caught by ruff and mypy before the code runs. A typo in a string creates a new empty column at run time, or a `KeyError` deep in a job. It also means training, serving and monitoring use the same names, which is one more defence against skew. Names that depend on the config are built by functions (`lag_column(24)` returns `"lag_24h"`), and `feature_columns(settings)` returns the model inputs in a fixed order.

  **`strict=True`.** Every schema ends with `strict=True`: a column that is not in the schema is an error. The test `test_silver_trips_reject_unexpected_columns` adds `tip_amount` and expects a failure. Reasons: an extra column may be a leak (for example a column computed from the target); and a model trained on a table with 18 columns will not behave the same when served 17. The table is exactly the contract, no more and no less.

  **Other checks to notice.** `unique=[ZONE_ID, HOUR_START]` in the gold schemas: one row per zone and hour, so a bad join that duplicates rows is caught. `_timestamp_column` requires timestamps to be timezone-aware in `America/New_York`; the tests use hours that cross the March daylight-saving change on purpose. `ACTUAL` in `predictions_schema` is `nullable=True`, because the ground truth arrives after the forecast.

  **One thing to know.** `_zone_column(settings, frozen_only=True)` restricts zones to the frozen list *only if the list is not empty*. In the committed config, `data.zones` is `[]`, so until the zones are selected and frozen (Phase 1), the gold schemas accept any zone ID from 1 to 263.

- **Without it.** A change in the source data (a renamed column, a new unit, a duplicate file) reaches the model. You find out weeks later from a drop in accuracy, with no clue where it started.

#### 6. The Typer CLI as the single entry point

- **What it is.** A **CLI** (command-line interface) is a program you control with typed commands. Typer builds one from normal Python functions: the function name becomes the command, the docstring becomes the help text, and the type hints become argument validation.
- **Why here.** Every way of running the project (your shell, DVC, Prefect, Docker, Kubernetes CronJobs, CI) must run the same code. The usual tool for this is a `Makefile`, but `make` is not installed on Windows. The CLI is plain Python, so `python -m demand <command>` is identical everywhere. Each command does three things only: get the settings, configure logging, call one stage function. No logic lives in the CLI.
- **Where.** [`src/demand/cli.py`](../src/demand/cli.py).

  **Lazy imports.** The stage modules are not imported at the top of the file. They are imported when a command runs:

  ```python
  def _run_stage(module: str, function: str, *args: object) -> None:
      """Import ``module`` lazily and call ``function(settings, *args)``."""
      if importlib.util.find_spec(module) is None:
          raise _not_implemented(module)
      configure_logging()
      getattr(importlib.import_module(module), function)(get_settings(), *args)
  ```

  Three reasons:

  1. **Optional groups.** If `cli.py` had `import mlflow` at the top, every command, including `--help`, would fail in an environment without the `ml` group, such as the API image. With lazy imports, a command needs only its own dependencies. The `serve` command follows the same idea: `import uvicorn` is inside the function.
  2. **Speed.** Importing LightGBM, MLflow and Prefect takes seconds. `python -m demand --help` should not pay for that.
  3. **Building in phases.** The full command list exists today, although the stages do not. `find_spec` checks whether a module exists without importing it. If it does not, the command prints `Not implemented yet: <module>` and exits with code 1. The commands `ingest`, `select-zones`, `silver`, `gold` and `validate` arrive in Phase 1; `train` in Phase 2; `serve` in Phase 3; `forecast`, `monitor` and `retrain` in Phase 5.

  **The contract for stage functions** is stated in the module docstring: each takes the `Settings` object as its first argument and returns `None`.

  **Exit codes.** A program reports success with exit code 0 and failure with any other number. DVC, CI and Kubernetes read only this number. That is why `validate` is documented as "exits non-zero if it fails": a failed quality gate must stop the pipeline. Typer checks arguments itself: `validate` accepts only the values of the `Layer` enum (`silver`, `gold`) and exits with code 2 for anything else.

- **Without it.** A folder of scripts, each with its own argument parsing and its own way of loading config; and commands in the README that work on Linux only.

#### 7. Structured JSON logging versus `print`

- **What it is.** **Structured logging** writes each log event as data with named fields, here one JSON object per line, not as a free-text sentence.
- **Why here.** In Phase 4 and later, the code runs in containers, and logs from several replicas are collected in one place. Nobody reads them line by line; they are searched and filtered by a machine. "Show all events where `stage` is `silver` and `level` is `ERROR`" is easy with JSON and fragile with free text. `print` has no level, no timestamp and no logger name, and it cannot be turned off for one library.
- **Where.** [`src/demand/logs.py`](../src/demand/logs.py): `JsonFormatter`, `configure_logging()`, `get_logger()`.

  ```python
          payload: dict[str, Any] = {
              "time": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
              "level": record.levelname,
              "logger": record.name,
              "message": record.getMessage(),
          }
  ```

  Details: times are in UTC, so logs from different machines can be ordered. Anything passed in `extra={...}` becomes its own field, so you write `log.info("cleaned trips", extra={"rows_out": 1200})` and can later filter on `rows_out`. Exceptions are included as an `exception` field. Logs go to **stderr**, not stdout, so the real output of a command stays clean: `python -m demand config` prints pure JSON that another program can read. `configure_logging()` replaces the handlers of the root logger and does not add to them, so calling it twice does not print each line twice (test: `test_configure_logging_is_idempotent`).
- **Without it.** When a nightly job fails, you have a wall of text without timestamps, and you cannot say which stage or zone produced which line.

#### 8. The testing approach

- **What it is.** Automated tests are small functions that run the code and `assert` what must be true. `pytest` finds and runs them. The tests are in `tests/unit/` and each file mirrors one source module.
- **Why here.** In this project the tests are how design rules are kept. "Lags shorter than the horizon are rejected" is a test (`test_lag_shorter_than_horizon_is_rejected`). "The column order of the model inputs does not change" is a test (`test_feature_columns_order_is_stable`). If a later change breaks a rule, the test names the rule.
- **Where and how.**

  **Fixtures.** A **fixture** is a function that prepares something a test needs. A test asks for it by naming it as a parameter. [`tests/conftest.py`](../tests/conftest.py) defines `settings`, available to every test:

  ```python
  @pytest.fixture
  def settings(tmp_path: Path) -> Settings:
      """Project settings with a small frozen zone list and all paths under ``tmp_path``."""
      base = load_settings(CONFIG_FILE)
  ```

  It loads the real config, then returns a copy with three test zones and with `data_dir` and `reports_dir` moved under `tmp_path`.

  **`tmp_path`** is a built-in pytest fixture: a new empty folder for each test. Any test that writes files writes there, so tests cannot damage your real `data/` folder and cannot affect each other.

  **`monkeypatch`** is a built-in fixture that changes something for one test and restores it afterwards. `tests/unit/test_config.py` uses `monkeypatch.setenv("DEMAND_MLFLOW__TRACKING_URI", ...)` to test the environment override without changing your real environment. `tests/unit/test_cli.py` uses `monkeypatch.setitem(sys.modules, name, module)` to insert a fake stage module, so the CLI can be tested before any stage exists.

  **Parametrised tests** run one test function with several inputs. Each input is reported as its own test:

  ```python
  @pytest.mark.parametrize(
      ("column", "value"),
      [
          (schemas.PICKUP_ZONE_ID, 264),
          (schemas.TRIP_DISTANCE, 0.0),
          (schemas.TRIP_DISTANCE, 150.0),
  ```

  The full list in `test_silver_trips_reject_bad_values` has eight bad values. Pattern: build one valid table, break one value, expect the schema to reject it. Without parametrisation this is eight nearly identical functions.

  **Coverage gate.** **Coverage** is the share of code lines (and here also branches: `branch = true`) that the tests execute. `pyproject.toml` sets `fail_under = 80`, and `addopts` adds `--cov=demand` to every pytest run. If coverage is below 80%, pytest exits with an error even when all tests pass. Coverage does not prove the tests are good; it only shows which code no test touches. It stops new code from arriving without tests. One practical result: running a single test file covers little of the code and fails the gate. Add `--no-cov` for such partial runs (see *Try it yourself*).

  **No network.** The first line of `conftest.py` says: "Tests never touch the network or the real data folders." A test that downloads a TLC file is slow, fails when the site is down, and gives different results when the file changes. Then a red test no longer means "the code is broken", and people stop trusting the tests. The plan's answer is a small synthetic dataset (Phase 1) and fake modules.

  **Other settings.** `--strict-markers` makes a misspelled marker an error. The `integration` marker is declared for end-to-end tests that arrive later.

- **Without it.** Every change is checked by hand, or not at all. Rules such as "no leakage" exist only in the memory of the author.

#### 9. ruff, mypy strict and pre-commit

- **What they are.**
  - **ruff** is a **linter** (finds likely mistakes and style problems by reading the code) and a **formatter** (rewrites the layout in one fixed style).
  - **mypy** is a **static type checker**: it reads the type hints and reports contradictions, without running the program. `strict = true` means every function must be annotated and no untyped call is allowed.
  - **pre-commit** runs a list of commands each time you run `git commit`, and stops the commit if one fails.
- **Why here.** Formatting discussions waste review time; one formatter ends them. Types are documentation that is checked: `def load_settings(config_file: Path | None = None) -> Settings` tells you how to call the function, and mypy proves the callers do it correctly. Many bugs in data code are type bugs (`None` where a dataframe was expected, a `str` path where a `Path` was expected), and mypy finds them in seconds.
- **Where.** `pyproject.toml`, sections `[tool.ruff]` and `[tool.mypy]`. Some of the selected ruff rule groups:

  | Code | Checks | Why it matters here |
  |---|---|---|
  | `E`, `F`, `W` | Basic errors and unused names | Standard |
  | `I` | Import order | Smaller diffs, fewer merge conflicts |
  | `B` | Common bug patterns | For example mutable default arguments |
  | `UP` | Old syntax | Keeps the code on modern Python 3.12 style |
  | `PTH` | Use of `os.path` in place of `pathlib` | Paths that work on Windows and Linux |
  | `D` | Docstrings (Google style) | Every public function is documented; turned off for `tests/**` |

  mypy checks `src` only (`files = ["src"]`), uses the pydantic plugin so it understands the settings classes, and has an `overrides` block with `ignore_missing_imports = true` for libraries that ship no type information (LightGBM, MLflow and others).

  [`.pre-commit-config.yaml`](../.pre-commit-config.yaml) defines three hooks: `ruff format`, `ruff check --fix`, `mypy`. They use `repo: local` and `language: system`, which means they run the tools already installed in the venv. The usual alternative lets pre-commit download its own copy of each tool; it lost because the hook could then use a different ruff version than CI, and the two would disagree. Two consequences of the choice made here: the hooks call `python`, so the venv must be active in the shell where you commit; and the hooks run only after a one-time `pre-commit install` in your clone, which writes a small script to `.git/hooks/pre-commit`. `Test-Path .git\hooks\pre-commit` tells you whether that was done.
- **Without it.** Style drifts from file to file, and errors that a machine can find in one second are found by a reviewer, or in production.

#### 10. `.gitattributes` and `.gitignore`

- **Line endings.** Windows ends a text line with two characters (CRLF); Linux and macOS use one (LF). This project is written on Windows and runs in Linux containers and Linux CI. Mixed endings cause two problems: git shows whole files as changed when only the endings differ, and a shell script with CRLF fails on Linux with a confusing error. [`.gitattributes`](../.gitattributes) solves it:

  ```gitattributes
  * text=auto eol=lf
  *.parquet binary
  *.png binary
  *.ps1 text eol=crlf
  ```

  Line 1: all text files use LF, in the repository and in your working folder. Lines 2 and 3: never touch the bytes of binary files (changing "line endings" inside a Parquet file would corrupt it). Line 4: PowerShell scripts keep CRLF, the Windows convention.

- **What `.gitignore` keeps out, and why.**

  | Pattern | Why it is not in git |
  |---|---|
  | `.venv/`, `__pycache__/`, `*.egg-info/`, `build/`, `dist/` | Generated; rebuilt from the lock file and the source |
  | `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/` | Tool caches and reports |
  | `/data/`, `*.parquet`, `/reports/` | Data is large and changes often. Git stores every version of every file for ever, so the repository would become very large and slow. Data will be versioned by DVC. |
  | `/mlruns/`, `/mlartifacts/`, `mlflow.db` | Experiment records and models belong to MLflow |
  | `.env`, `.env.*` | May contain passwords. A secret committed once stays in the history even after you delete the file. |
  | `.idea/`, `.vscode/`, `Thumbs.db`, `.DS_Store` | Personal editor and operating-system files |

  Two lines start with `!`, which means "do not ignore": `!tests/fixtures/**/*.parquet` (small test data must be in git so tests run without downloads) and `!.env.example` (the template that documents which variables exist, with no real values).

- **Without it.** A 3 GB repository, a leaked password, and pull requests where every line shows as changed.

#### 11. Conventional commits and the dev-branch workflow

- **Conventional commits** is a format for commit messages: `type: short description`. The first commit is `chore: project foundation (phase 0)`. Common types: `feat` (new behaviour), `fix` (bug fix), `docs`, `test`, `refactor` (same behaviour, better structure), `chore` (setup and maintenance), `ci`. The history becomes easy to scan (`git log --oneline`), and tools can build a changelog from it.
- **Workflow in this project.** Work happens on the `dev` branch. Each push contains one finished feature, with its tests and its lesson in this guide, and is pushed only after the owner approves it. Later, the `main` branch receives reviewed work and triggers delivery (Phase 7).
- **Why.** One feature per push keeps each change small enough to review and to revert alone. If a push breaks something, you know which feature did it. A push that mixes a data-cleaning change, a refactor and a new dependency cannot be reviewed well and cannot be partly undone.
- **Without it.** Commits named "update" and "fix stuff", and a history that explains nothing when you look for the change that broke the model.

### Try it yourself

Run these in Windows PowerShell from the repository root. If `.venv` does not exist yet, follow the *Setup* section of the [README](../README.md) first. All commands below were run on commit `72b97b4`.

**1. See the commands.**

```powershell
.venv\Scripts\python.exe -m demand --help
```

Expected: a usage line, then a table of commands: `config`, `ingest`, `select-zones`, `silver`, `gold`, `validate`, `train`, `forecast`, `monitor`, `retrain`, `serve`. Only `config` does real work at this commit; the stage commands arrive in later phases (Phase 1 for `ingest`, `select-zones`, `silver`, `gold` and `validate`).

**2. Print the resolved configuration.**

```powershell
.venv\Scripts\python.exe -m demand config
```

Expected: the whole configuration as JSON, starting with `"project": { "name": "nyc-taxi-demand", "seed": 42, "timezone": "America/New_York" }`.

**3. Override one value with an environment variable.**

```powershell
$env:DEMAND_MLFLOW__TRACKING_URI = "http://localhost:5000"
.venv\Scripts\python.exe -m demand config | Select-String tracking_uri
Remove-Item Env:DEMAND_MLFLOW__TRACKING_URI
```

Expected: `"tracking_uri": "http://localhost:5000",` in place of the YAML value `sqlite:///mlflow.db`. The last line removes the variable again.

**4. Try to configure a leaking lag.** A list value is written as JSON.

```powershell
$env:DEMAND_FEATURES__LAGS = "[1, 24]"
.venv\Scripts\python.exe -m demand config
Remove-Item Env:DEMAND_FEATURES__LAGS
```

Expected: a `ValidationError` traceback that ends with `Value error, features.lags [1] are shorter than the 24 h forecast horizon and would leak information unavailable at prediction time`. No configuration is printed.

**5. Try to change a frozen setting.**

```powershell
.venv\Scripts\python.exe -c "from demand.config import get_settings; s = get_settings(); s.features.horizon_hours = 1"
```

Expected: a `ValidationError` with `Instance is frozen`.

**6. See the model's input columns.**

```powershell
.venv\Scripts\python.exe -c "from demand.config import get_settings; from demand.schemas import feature_columns; print(feature_columns(get_settings()))"
```

Expected: a list of 17 names: `zone_id`, three lags (`lag_24h`, `lag_48h`, `lag_168h`), four rolling columns, five calendar columns, four weather columns.

**7. Write one structured log line.**

```powershell
.venv\Scripts\python.exe -c "from demand.logs import configure_logging, get_logger; configure_logging(); get_logger('demo').info('cleaned trips', extra={'stage': 'silver', 'rows_out': 1200})"
```

Expected: one JSON line with the fields `time`, `level`, `logger`, `message`, `stage` and `rows_out`.

**8. Give the CLI a wrong argument.**

```powershell
.venv\Scripts\python.exe -m demand validate bronze
$LASTEXITCODE
```

Expected: `Invalid value for 'layer': 'bronze' is not one of 'silver', 'gold'.` and then `2`. The stage code is never reached; Typer rejects the argument first.

**9. Run one test file.**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_config.py -q --no-cov
```

Expected: `14 passed`. Now run it again without `--no-cov`: the 14 tests still pass, but the run ends with `FAIL Required test coverage of 80.0% not reached. Total coverage: 50.31%`, because one file alone does not execute the other modules. This is the coverage gate working.

**10. Run one test by name.**

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/test_schemas.py -q --no-cov -k "unexpected_columns or naive"
```

Expected: `2 passed, 21 deselected`. Open the two tests and read what they break in the table.

**11. Run the whole suite with coverage.**

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Expected at Push 1: `56 passed`, a coverage table, and `Required test coverage of 80.0% reached. Total coverage: 99.69%`. Later pushes add tests, so the numbers will grow.

**12. Lint, check the format, check the types.**

```powershell
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m ruff format --check .
.venv\Scripts\python.exe -m mypy
```

Expected at Push 1: `All checks passed!`, then `21 files already formatted`, then `Success: no issues found in 11 source files`.

**13. Confirm the editable install and the line endings.**

```powershell
.venv\Scripts\python.exe -m pip show demand
git ls-files --eol pyproject.toml
```

Expected: the first output contains `Editable project location:` followed by the path of your clone. The second shows `i/lf    w/lf    attr/text=auto eol=lf`: LF in the repository (`i`), LF in the working folder (`w`).

### Check your understanding

1. `python -m demand config` works from any folder once the venv is set up, but `import demand` fails in a fresh venv where you have not run `pip install -e .`. Why?
2. `pyproject.toml` says `pandas>=2.2`. Why is that not enough for reproducibility, and which file fixes it?
3. You set `DEMAND_PROJECT__SEED=7` and the YAML file says `seed: 42`. Which value does the program use, and which lines of code decide it?
4. Someone adds `lags: [12, 24, 168]` to `config.yaml` because it lowers the validation error. What happens when they run any command, and why is that the right behaviour?
5. A new upstream file contains an extra column `congestion_surcharge`, and the cleaning code passes it through to silver by mistake. Which setting catches this, and why is rejecting it better than ignoring it?
6. Why does `cli.py` not have `import demand.models.pipeline` at the top of the file?

<details>
<summary>Answers</summary>

1. The code is in `src/demand/`, and `src/` is not on Python's import path by default. The editable install registers that folder in the venv. After that, `demand` is importable from anywhere, and `python -m demand` runs `src/demand/__main__.py`. This is the purpose of the src layout: nothing works by accident.
2. `>=2.2` is a range. It installs a different version at a different date, and it says nothing about the packages pandas itself depends on. `requirements.lock` pins the exact version of all 281 packages, direct and transitive.
3. `7`. `Settings.settings_customise_sources` returns `env_settings, init_settings`; the earlier source has priority, and the YAML values enter through `init_settings` (they are passed to `Settings(**content)` in `load_settings`).
4. Every command fails at start-up with a `ValidationError` from `FeatureConfig._check_no_leakage`, because 12 is less than `horizon_hours` (24). It is right because pickups from 12 hours before the target hour do not exist when a day-ahead forecast is made. The lower error would be a leak, not an improvement.
5. `strict=True` on `silver_trips_schema`. The quality gate fails and the pipeline stops. Ignoring the column would let a table that differs from the contract flow into gold; later code could start to depend on it, and an extra column can carry information the model must not have.
6. Three reasons: commands would fail in environments that do not install the `ml` group (for example the API image); `--help` would become slow because heavy libraries would load; and the module does not exist yet at this commit, so the import itself would crash the whole CLI. `_run_stage` imports the module only when its command runs, and prints "Not implemented yet" if it is missing.

</details>

### Interview angle

**"How do you prevent data leakage in a forecasting project?"**
- Start from the horizon: say when the forecast is made and what is known at that moment.
- Name concrete cases: lags shorter than the horizon, observed weather used as if it were a forecast, selections (top zones, thresholds) computed on the full dataset.
- Say that you enforce it, not only that you know it: a config validator that rejects short lags, a time-based split, a test that no feature for hour *t* uses data newer than *t* − 24 h (the test is planned for Phase 1; the validator exists now).
- Mention the symptom: a model that is much better offline than in production.

**"How do you manage configuration across development and production?"**
- One typed settings object, validated at start-up; fail fast with a clear message.
- Defaults in a committed YAML file; differences between environments through environment variables (`DEMAND_SECTION__KEY`); secrets never in git, only documented in `.env.example`.
- Immutable (`frozen`) and strict about unknown keys (`extra="forbid"`), and why each matters.
- A good extra point: validators can hold domain rules, not only types (the lag and horizon check).

**"What is a data contract, and what happens when it is broken?"**
- A checkable schema for a table at a boundary: columns, types, ranges, uniqueness, timezone.
- The producer validates before writing; if validation fails, the pipeline stops with a non-zero exit code. Stopping is better than training on bad data.
- Design points from this repository: schemas built from the same settings as the cleaning rules, column names as constants, `strict=True`.
- Be honest about limits: a schema checks shape and ranges, not meaning. A column of believable but wrong values passes. That is what drift monitoring is for (Phase 5).

---

<!-- Add the next lesson here: "## Push 2 – <feature> (Phase <n>)", using the five template headings. -->

## 6. Glossary

| Term | Meaning |
|---|---|
| Actuals | The real observed values (here: real pickups), known only after the hour has passed. |
| Backtest (rolling-origin) | Testing a forecasting model on several past periods: train up to a date, test on the next days, move the date forward, repeat. |
| Baseline | A very simple model used as the minimum to beat. Here: seasonal-naive, "same hour last week". |
| Branch (git) | A named line of commits. Work here happens on `dev`. |
| Branch coverage | Coverage that also checks that both outcomes of each `if` were executed. |
| Build backend | The tool that turns source code into an installable package. Here: hatchling. |
| Champion / challenger | The model currently used in production / a new model that must beat it to replace it. |
| CI/CD | Continuous integration (automatic checks on every push) and continuous delivery (automatic build and deployment). |
| CLI | Command-line interface: a program controlled by typed commands. |
| Container | A running instance of an image: a program packaged together with its dependencies. |
| Conventional commits | A commit-message format: `type: description`, for example `feat:`, `fix:`, `chore:`. |
| Coverage | The share of code that the tests execute. A gate (`fail_under = 80`) fails the test run below a limit. |
| CRLF / LF | The two line-ending conventions: Windows (two characters) and Linux/macOS (one character). |
| Data contract | A precise, checkable description of a table that producer and consumer both rely on. |
| Data leakage | Training or evaluating with information that will not be available at prediction time. |
| Day-ahead forecast | A forecast made at least 24 hours before the hour it describes. |
| Dependency | A package that your code needs. A *transitive* dependency is one that your dependencies need. |
| Drift | A change over time in the distribution of the input data or of the predictions, compared with the training period. |
| Editable install | `pip install -e .`: the installed package points to your source folder, so edits take effect without reinstalling. |
| Entry point | The place where a program starts. Here: `python -m demand`, which runs `demand/__main__.py`. |
| Environment variable | A named value set outside the program, by the shell, Docker or Kubernetes, and read by the program. |
| Exit code | The number a program returns when it ends. 0 means success; anything else means failure. |
| Extra | See *optional dependency group*. |
| Fixture (pytest) | A function that prepares something a test needs; a test requests it by parameter name. |
| Forecast horizon | The time between making a forecast and the moment it describes. Here: 24 hours. |
| Formatter | A tool that rewrites code layout in one fixed style. |
| Frozen model | A pydantic object that cannot be changed after creation. |
| Gold layer | The model-ready data: hourly pickups per zone and the feature table. |
| Ground truth | The correct answer for a prediction. In forecasting it arrives late. |
| Hook (git) | A script that git runs at a certain moment, for example before a commit. |
| Idempotent | Safe to run more than once: the second run gives the same result as the first. |
| Immutable | Cannot be changed after it is created. |
| Lag feature | A past value of the target used as an input, for example pickups 24 hours earlier. |
| Lazy import | Importing a module only when it is needed, not at the top of the file. |
| Linter | A tool that finds likely mistakes and style problems by reading the code without running it. |
| Lock file | A file with the exact version of every package, direct and transitive. |
| Medallion architecture | Data stored in layers of increasing quality: raw (bronze), silver, gold. |
| Model registry | A store of model versions with labels (aliases) such as `champion`. |
| monkeypatch | A pytest fixture that changes an attribute, dictionary entry or environment variable for one test and then restores it. |
| Optional dependency group | A named set of extra packages, installed only where needed, for example `demand[ml]`. |
| Orchestration | Scheduling and running the pipeline steps in the right order, with retries. |
| Package | A folder of Python modules that can be imported by name. |
| Parametrised test | One test function that runs with several inputs; each input is reported separately. |
| Parquet | A column-oriented, compressed file format for tables. |
| Pin | To fix a dependency to one exact version (`==`). |
| Quality gate | A check that stops the pipeline when data does not meet its contract. |
| Raw layer (bronze) | Data exactly as downloaded, never modified. |
| Rolling statistic | A statistic over a moving window, for example the mean of the last 24 hours. |
| Schema | The formal definition of a table's columns, types and constraints. |
| Silver layer | Cleaned, typed, de-duplicated data. |
| Simulated clock | Replaying held-out historical data day by day as if it were live. |
| src layout | Keeping the package in `src/<name>/`, so it must be installed to be imported. |
| Stage | One step of the pipeline, for example ingest, silver, gold or train. |
| Stateless | A service that keeps no data between requests, so any number of copies can run side by side. |
| Static type checker | A tool that checks type hints without running the program. Here: mypy. |
| stderr / stdout | The two output streams of a program: stderr for logs and errors, stdout for results. |
| `strict=True` (Pandera) | Columns that are not in the schema are an error. |
| Structured logging | Log events written as data with named fields (here JSON), not as free text. |
| Target | The value the model predicts. Here: the column `pickups`. |
| Time-based split | Train on data before a date and test on data after it. |
| Timezone-aware timestamp | A timestamp that carries its timezone, so daylight-saving changes are unambiguous. |
| `tmp_path` | A pytest fixture that gives each test a new empty folder. |
| Training/serving skew | A difference between how features are computed for training and for prediction. |
| Validator (pydantic) | A function that checks a rule when an object is created and raises an error if it is broken. |
| Virtual environment | An isolated folder with its own Python interpreter and installed packages. |
| YAML | A human-readable text format for configuration. |

---

## 7. Where to learn more

Official documentation for the tools that are in use in the repository today. More are added when their code arrives.

- Python Packaging User Guide (for `pyproject.toml`, src layout and editable installs) – <https://packaging.python.org>
- Python documentation, `venv` and `logging` modules – <https://docs.python.org>
- Hatch documentation (hatchling build backend) – <https://hatch.pypa.io>
- uv documentation (see `uv pip compile`) – <https://docs.astral.sh/uv/>
- Pydantic documentation, including the pydantic-settings section – <https://docs.pydantic.dev>
- Pandera documentation – <https://pandera.readthedocs.io>
- pandas documentation (see time zones and time series) – <https://pandas.pydata.org>
- Typer documentation – <https://typer.tiangolo.com>
- pytest documentation (see fixtures, `monkeypatch`, `tmp_path`, parametrize) – <https://docs.pytest.org>
- pytest-cov and Coverage.py documentation
- Ruff documentation – <https://docs.astral.sh/ruff/>
- mypy documentation – <https://mypy.readthedocs.io>
- pre-commit documentation – <https://pre-commit.com>
- Git documentation (see `gitattributes` and `gitignore`) – <https://git-scm.com>
- Conventional Commits specification – <https://www.conventionalcommits.org>
