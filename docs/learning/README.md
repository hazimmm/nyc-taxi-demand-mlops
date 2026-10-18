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
| [2. Forecasting without cheating](02-forecasting-without-cheating.md) | Lag features, the moment a forecast for a whole day can be issued, the horizon of 48 hours, and three ways a model can see the future by accident | 15 min |
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

This part follows the data pipeline of Phase 1: from the download to the gold layer, then the quality gates between the layers, and the tool that runs all of it in order.

| Chapter | What it covers | Time |
|---|---|---|
| [14. Getting the data: downloads you can trust](14-getting-the-data.md) | The raw layer, half files and error pages, atomic rename, retries with backoff, testing without a network | 18 min |
| [15. Knowing what you have: the manifest and idempotency](15-knowing-what-you-have.md) | Checksums, the manifest, skip or download, the weather source, UTC and padding, a first look at the raw data | 20 min |
| [16. Cleaning the trips: rules you can count](16-cleaning-the-trips.md) | One row is one valid trip, ten rules in a fixed order, counts that must add up, the cleaning report, and a rule from the plan that the data proved wrong | 14 min |
| [17. Clocks, weather and the silver stage](17-clocks-weather-and-the-silver-stage.md) | Local time and daylight saving, elapsed time, weather from UTC to local hours, gaps, and a stage that validates before it writes | 14 min |
| [18. From trips to a demand table](18-from-trips-to-a-demand-table.md) | Zones chosen from the training period and frozen, the complete grid and its zeros, the weather join, and an outlier flag that is never a model input | 15 min |
| [19. Features without leakage](19-features-without-leakage.md) | Lags and rolling windows shifted by the horizon, the moment the rows of a whole day can be built, elapsed hours, warm-up rows, one feature function for training and serving, and the leakage test | 15 min |
| [20. Quality gates: checking the files on disk](20-quality-gates.md) | A check before reading, lazy validation, checks a schema cannot express, a check that could not run, and one-line messages for expected failures | 14 min |
| [21. A pipeline that knows what changed](21-a-pipeline-that-knows-what-changed.md) | DVC stages with `deps`, `params` and `outs`, `dvc.lock`, a gate between two stages, the raw layer that persists, the cache, what DVC does not see, and a date without quotes | 14 min |

### Part IV – Modelling

This part follows the modelling of Phase 2. It starts with the forecast that needs no model, and with the test that every later model must pass. Then it builds the model, shows how its settings were decided, records the trained model, tunes it, explains its forecasts, and decides which version is served. It ends with a look back at what Part IV proved and what it did not.

| Chapter | What it covers | Time |
|---|---|---|
| [22. A number to beat: the baseline and three metrics](22-a-number-to-beat.md) | Why a baseline comes before any model, "the same hour one week ago", one interface for baselines and models, MAE, RMSE and sMAPE calculated by hand, and the weaknesses of sMAPE in the real data | 14 min |
| [23. The backtest: four weeks the forecaster has not seen](23-the-backtest.md) | Four test weeks with an expanding training window, two protections for May and June, the gap before each test week, the result per week, zone and day, and what a backtest does not prove | 15 min |
| [24. A model behind the same two methods: LightGBM as a forecaster](24-a-model-behind-the-same-two-methods.md) | Gradient-boosted trees in plain words, the zone as a category, an objective for counts, what the model reads and refuses, the result per week, zone and hour, and what feature importance does not say | 14 min |
| [25. Deciding by measurement: four choices and the noise of a seed](25-deciding-by-measurement.md) | What a seed fixes and what was tested about it, the noise of a score, the rule for keeping a change, four real choices, and a lag of two weeks that helped on average and was not added | 15 min |
| [26. The model that is kept: the final fit and MLflow](26-the-final-fit-and-mlflow.md) | The fit on all training rows and why it needs no gap, experiment tracking, what a run records, the model signature, the model registry, and four limits of the record | 15 min |
| [27. Tuning: a search, and how much of its score to believe](27-tuning-and-how-much-to-believe.md) | Hyper-parameters, Optuna and its seeded sampler, a search that only proposes, five searched settings, the seed check, the gain on weeks the search did not see, and the two-week lag closed | 15 min |
| [28. Explaining a forecast: SHAP](28-explaining-the-model-with-shap.md) | Contributions that add up to a forecast, factors on the log scale, Good Friday input by input, what a training run stores, SHAP share against gain, and what SHAP does not tell | 14 min |
| [29. Which model is served: the promotion gate and the model card](29-which-model-is-served.md) | Aliases, why `dvc repro` never changes the served model, a margin of 0.2 taken from the seed noise, scores that can be compared, a rollback that a routine promotion undid, the record of a decision, the model card, and a look back at Part IV | 14 min |

### Part V – Serving

This part follows the prediction service of Phase 3. It starts from zero, with what a web service is, and builds up to the real service: one request and its answer, the data it forecasts from, and how it refuses what it cannot answer. Then it shows how the service starts, how it was proved to build the same features as training, and what it reports to the systems that watch it.

| Chapter | What it covers | Time |
|---|---|---|
| [30. A forecast over HTTP: the prediction service](30-a-forecast-over-http.md) | Requests and responses, methods, paths, query parameters and status codes, FastAPI and uvicorn, `/docs`, a whole day with GET, features built inside the service, the issue moment, the data snapshot and the days it allows, strict validation and 422 | 15 min |
| [31. Starting, proving and watching the service](31-starting-proving-and-watching-the-service.md) | Loading once at startup, an alias resolved to a version once, a service that refuses to start, liveness and readiness, a store that is only read, the skew check on 2,860 real rows, `/health`, `/ready`, `/model-info`, metrics for Prometheus and their labels, and the limits | 14 min |

### Part VI – Containers

This part follows the containers of Phase 4. It starts from zero, with what an image and a container are, and builds the two images of the project: one for the service, and one for the training commands and the MLflow server. Then it starts the whole stack with one file, with its own model store, and with Prometheus and Grafana to watch the service. On the way, the containers find a bug that the tests had missed.

| Chapter | What it covers | Time |
|---|---|---|
| [32. Images: the service with everything it needs](32-images.md) | Images, containers, volumes and networks, the Dockerfile and its layers, a base pinned by tag and digest, two stages, one lock per image, a user by number, a health check against `/ready`, `.dockerignore` as an allow-list, the pipeline image, and why DVC stays on the host | 14 min |
| [33. The whole stack with Docker Compose](33-the-whole-stack-with-docker-compose.md) | A store that names Windows paths, an MLflow server of the stack's own, the same model trained again, a bug found by a second environment, five services started in order, `Restarting (3)`, volumes, mounts and ports, a read-only container, Prometheus and Grafana from files, the commands from zero, and the limits | 14 min |

### Part VII – Monitoring and retraining

This part follows Phase 5. It starts with the replay: a simulated clock plays May and June 2025 one day at a time, forecasts each day at 00:00 on the day before with the code of the service, and scores every day when it is over. The replay is the clean exam of the model. Monitoring is built on its files: three checks every day, and the lesson that a check which fires on almost every day may not trigger anything. Retraining comes last: a new model trained after an accuracy alert, compared with the champion's real forecasts, and a rule that was not changed during the test.

| Chapter | What it covers | Time |
|---|---|---|
| [34. The clean exam: replaying May and June](34-the-replay-of-may-and-june.md) | Why a replay, a clock that is a folder, one step from forecast to score, forecasts written once, the issue moment as `generated_at`, why 1 May is not forecast, the closing step, the exam result per week and per day, a 7-day window against the alert ratio, and why nothing was changed after the result | 15 min |
| [35. Watching a model when the truth arrives late](35-watching-a-model-when-the-truth-arrives-late.md) | Input drift, prediction drift and accuracy, the window and the reference, the ten tested inputs and the six left out, the Wasserstein distance by hand, a test fixed instead of chosen, the backtest's forecasts as an honest reference, two windows two days apart, `--as-of`, and the report at 1 July | 14 min |
| [36. Alerts that mean something, and the numbers in Grafana](36-alerts-that-mean-something.md) | The three checks on every day of the replay, a drift test that fires in 10 of 11 quiet weeks, the false-alarm rate, one alert and two warnings with no threshold changed, the Pushgateway, gauges with the simulated moment, the dashboard "Model monitoring", and the limits | 14 min |
| [37. Retraining, and a rule that was not changed](37-retraining-and-a-rule-that-was-not-changed.md) | Why retrain and only on the accuracy alert, four checks in order, a cooldown read from the registry, waiting for 28 days of the champion's forecasts, the cutoff, why a backtest of both would tie, the champion's real forecasts against a backtest of the recipe on the same 13,440 zone-hours, where the comparison leans, a rule loosened during the test and restored, two retrains rejected, rollback by hand and shadow forecasts | 14 min |

### Later parts

These parts have no chapters yet. They are written when their code is added.

| Part | Topic | Arrives with |
|---|---|---|
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
| 5 – Aggregation and features: the gold layer (Phase 1) | `python -m demand select-zones` finds the 20 busiest zones of the training period, and the list is frozen in the config; `python -m demand gold` builds the hourly demand table and the feature table with one feature function for training and serving; the config refuses a rolling window shorter than 2 hours | 18 and 19 added; 1 to 5, 7 to 11, 13, 14, 16 and 17 updated; appendices A to C extended |
| 6 – Quality gates and the DVC pipeline (Phase 1) | `python -m demand validate silver` and `validate gold` check the files of a layer and write a validation report; `dvc repro` runs the five stages in order and skips a stage when nothing it depends on changed; an expected failure is printed as one line that says what to do next, not as a traceback; `train_end_date` is quoted in the config | 20 and 21 added; 1, 3, 4, 5, 8 to 11, 14, 17, 18 and 19 updated; appendices A to C extended |
| 7 – A horizon of 48 hours, the baseline and the backtest (Phase 2) | The forecast horizon changed from 24 to 48 hours, so that the forecast for a whole day can be issued at 00:00 on the day before; the lag of 24 hours is gone, and the feature table has 82,560 rows and 19 columns; `python -m demand backtest` scores the baseline "the same hour one week ago" on four test weeks with three metrics and writes a report; `dvc repro` runs it as a sixth stage | 22 and 23 added; 2 and 19 rewritten around the new horizon; 1, 3, 4, 5, 7, 9, 10, 11, 13, 14, 18, 20 and 21 updated; appendices A to C extended |
| 8 – The LightGBM model, training and MLflow tracking (Phase 2) | `python -m demand train` scores the baseline and a LightGBM model on the four test weeks, fits the final model on all training rows, and records the run in MLflow with a registered model version; the model has an MAE of 24.93 against 27.70 for the baseline; `dvc repro` runs `train` as its sixth stage, in place of `backtest`; `python -m demand backtest` now scores both forecasters and keeps nothing | 24, 25 and 26 added; 1, 5, 9, 10, 11, 21, 22 and 23 updated; appendices A to C extended |
| 9 – Tuning and SHAP explanations (Phase 2) | `python -m demand tune` searches five LightGBM hyper-parameters with Optuna on the four test weeks, writes a report and prints a block for the config, and changes nothing itself; the block was pasted, and the model now has an MAE of 24.11 against 27.70 for the baseline (12.98% better; about 0.6 of the in-sample gain of 0.90 held on weeks the search did not see); `python -m demand train` now also explains the final model with SHAP and stores `shap_importance.json`, `shap_summary.png` and 16 metrics with the MLflow run; the lag of two weeks was tested again and not added | 27 and 28 added; 1, 9, 10, 11, 21, 23, 24, 25 and 26 updated; appendices A to C extended |
| 10 – The promotion gate and the model card (Phase 2) | `python -m demand promote` makes the newest model version champion only if its backtest MAE is at least 0.2 lower than that of the weekly baseline and of the current champion, with no fold higher, and only when both were scored on the same backtest; a version is judged once, so a rollback by hand is not undone; `python -m demand set-champion VERSION --reason TEXT` sets the champion by hand and records why; every decision is written as tags on the version and to `reports/promotion_report.json`; `docs/model-card.md` describes version 2; Part IV is complete | 29 added; 1, 9, 10, 11, 26 and 28 updated; appendices A to C extended |
| 11 – The prediction service (Phase 3) | `python -m demand serve` starts a FastAPI service on port 8000. `GET /predict?date=...&zone=...` returns the forecast of every hour of one day for the zones asked, built inside the service with the training feature function from the demand known at the issue moment, and names the model version that computed it; the service forecasts the days 2025-01-10 to 2025-06-30 of its data snapshot and refuses anything else with a 422 that says what can be asked; `/health`, `/ready`, `/model-info` and `/metrics` describe it; the alias `champion` is resolved to a version once at startup, and without it the service does not start; the model is loaded without writing into `mlruns`; `serve` now sets up the JSON logging of Chapter 9, so the events of the service, such as `service ready`, appear as JSON lines | 30 and 31 added; 1, 4, 5, 6, 9, 10, 11, 19, 26 and 29 updated; appendices A to C extended |
| 12 – Containers (Phase 4) | Two Docker images, one for the service and one for the training commands and the MLflow server, each from a Python base pinned by digest, in two stages, with its own lock file compiled from `requirements.lock`, and run as user 10001; `.dockerignore` became an allow-list; `docker-compose.yml` starts MLflow with its own store in named volumes, the API, Prometheus and Grafana with a provisioned dashboard, and runs `train` and `promote` behind a profile; the data is still built on the host with `dvc repro`; the loader now asks the configured registry where the model's files are, because MLflow looked a `models:/` version up in the registry of the whole process | 32 and 33 added; 1, 5, 6, 9, 10, 30 and 31 updated; appendices A to C extended |
| 13 – The simulated clock and the replay (Phase 5) | `python -m demand forecast` runs one step of a simulated clock: it forecasts the next day of May or June 2025 at 00:00 on the day before with the champion, writes the forecast once to `data/predictions/forecasts/` and the model inputs to `inputs/`, scores every day that is over into `scored/`, and rebuilds `reports/replay_report.json`; `--days N` runs N steps; the clock is the folder of forecasts; the replay starts on 2 May and ends with a closing step; the API and the replay share one function, `predict_day`; the replay of version 2 gave an MAE of 27.80 against 31.83 for the baseline and 24.11 in the backtest; `docker-compose.yml` mounts `data/predictions-stack` for the replay of the stack; the model card has a section on the replay | 34 added; Part VII started; 1, 3, 5, 9, 10, 11, 20, 23, 25, 26, 27, 29, 30 and 33 updated; appendices A to C extended |
| 14 – Monitoring (Phase 5) | `python -m demand monitor` evaluates three checks at the replay clock, or with `--as-of DAY` at 00:00 on an earlier day, reading only the files that existed then: input drift of 10 model inputs against 3 to 30 April and prediction drift against the backtest's forecasts computed again, both with Evidently's normed Wasserstein distance, and the live MAE of the last 7 days that are over against the backtest MAE; it writes `reports/monitoring_report.json` and an Evidently page; after the replay showed that input drift fires on almost every day, only accuracy raises an alert and the drift checks raise warnings, with every threshold unchanged; `--push-to` sends the numbers to a new Pushgateway service, which Prometheus scrapes, and a second Grafana dashboard "Model monitoring" shows them | 35 and 36 added; Part VII extended; 1, 3, 5, 9, 10, 11, 19, 29, 33 and 34 updated; appendices A to C extended |
| 15 – Gated retraining (Phase 5) | `python -m demand retrain` evaluates at the issue moment of the newest forecast whether to retrain, and writes `reports/retrain_report.json`; a challenger is trained only after an accuracy alert about the current champion, with a day left to forecast, at least `monitoring.retrain_cooldown_days` (7) days after the last retrain, read from the tag `retrain.now` in the registry, and once the champion has issued all 28 days of the comparison weeks; it learns from every hour up to the day before, is scored by a backtest of the last 4 weeks against the champion's real forecasts and the weekly baseline on the same 13,440 zone-hours, and is promoted only by the rule of `promote`; `run_day` runs forecast, monitoring and retraining as one daily step; `promotion.py` gained `decision_of` and `record_decision`, with no change of behaviour; in a simulation of the replay with retraining both retrains, on 30 May and 25 June, were rejected, and version 2 served every day; rollback stays by hand | 37 added; Part VII extended; 1, 9, 10, 11, 29, 34, 35 and 36 updated; appendices A to C extended |
