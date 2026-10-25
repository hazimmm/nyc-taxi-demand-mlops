# Appendix B – Glossary

Every technical term used in the book, in alphabetical order. The number in the last column is the chapter where the term is explained.

| Term | Meaning | Ch. |
|---|---|---|
| Accepted risk / time-boxed exception | A known problem that a team decides, in writing, to live with, with the reason. When the decision has a fixed end, it is a time-boxed exception. Here every entry of `.trivyignore.yaml` has a statement and an `expired_at` day; on that day the finding counts again and CI fails. | 41 |
| Action (GitHub Actions) | A reusable step that someone published in a repository, used with `uses:`, such as `actions/checkout`, which copies the repository onto the runner. Here every action is pinned by its full commit hash. | 41 |
| Actuals | The real observed values, here the real pickups, known only after the hour has passed. | 3 |
| Alert / warning (monitoring) | An alert means "act on this"; a warning means "look at this". Here only the accuracy check raises an alert, which the retraining flow reads; the two drift checks raise warnings, which nothing acts on. | 35, 36, 37 |
| Alias (model registry) | A name that points at one version of a registered model and can be moved to another. `models:/taxi-demand@champion` loads whichever version the alias `champion` points at now. The service resolves it to a version number once, at startup. | 26, 29, 31 |
| Alias (vulnerability id) | Another id of the same vulnerability in another list. `CVE-2026-114145` and `GHSA-83vg-56qc-22m7` are one problem in gitpython. pip-audit matches an ignored id against the aliases too. | 41 |
| Allow-list | A list of what is let in, where everything else is left out. `.dockerignore` is one: its first line, `*`, leaves out everything, and the next lines let back in what the images are built from. | 32 |
| API server (Kubernetes) | The part of the control plane that every request goes to. `kubectl` talks to it, and it refuses objects that break the rules, such as a pod that breaks the namespace's Pod Security Standard. | 39 |
| Artifact | A file that is stored with a training run, such as a report or the model. | 26 |
| As of (monitoring) | The moment at which monitoring is evaluated: 00:00 on the day given to `--as-of`, or the position of the replay clock. Nothing that did not exist at that moment is read. | 35 |
| ASGI | The standard interface between a Python web server and a web application. uvicorn and FastAPI meet at it, so either can be replaced alone. | 30 |
| Atomic | Said of an operation that happens completely or not at all, with no state in between. Renaming a file inside one folder is atomic. | 14 |
| Automated test | A small function that runs a piece of code and checks the result with `assert`. | 10 |
| Backtest | Testing a forecasting model on the past as if it had been used then. Here: four test weeks in April 2025, scored by `python -m demand backtest`. | 3, 23 |
| Base / overlay (Kustomize) | The base holds the manifests that are the same on every cluster (`k8s/base/`). An overlay takes the base and changes only what differs on one cluster (`k8s/overlays/kind/`: the image tags, never pulling images, a suspended CronJob). | 39 |
| Base image | The image that a Dockerfile starts from, named in its first `FROM`. Here the official Python image `python:3.12.15-slim-trixie`, pinned by digest. | 32 |
| Base value (SHAP) | The output a model gives on average over its training rows, before any input of a row has contributed. Here about 4.49 on the log scale, which is about 89 pickups. | 28 |
| Base64 | A way to write any bytes as plain letters and digits. It is an encoding, not encryption: anyone can decode it. A Kubernetes Secret is stored this way. | 39 |
| Baseline | A very simple method that sets the minimum a model must beat. Here: "the same hour one week ago", a copy of the feature `lag_168h`, named `weekly_naive`. | 1, 22 |
| Batch job | A command that runs once, writes its result and ends, such as `python -m demand monitor`. Prometheus cannot scrape it, so it pushes its numbers to a Pushgateway. | 36 |
| Bias (of forecasts) | The mean of forecast minus actual. Negative means the forecasts were too low on average. In the replay it is −8.41 pickups per zone and hour. | 34 |
| Binary file | A file that is not text. Git must not change any byte of it. | 12 |
| Bind mount | A folder or file of your machine made visible inside a container. Here `data/gold` and `data/silver/weather` for the API, and the whole checkout for the pipeline, all read-only. | 32, 33 |
| Branch (git) | A named line of commits. Work happens on `dev`. | 12 |
| Branch coverage | Coverage that also requires both outcomes of every `if` to be executed. | 10 |
| Build backend | The tool that turns a source folder into something installable. Here: hatchling. | 5 |
| Build context | The folder whose files the steps of a Dockerfile may copy, filtered by `.dockerignore`. Here the project folder, given as the last `.` of `docker build`. | 32 |
| Cache (DVC) | The folder `.dvc/cache`, in which DVC keeps a copy of every output of a stage. | 21 |
| Cache policy (Prefect) | The rule by which Prefect decides whether a task may reuse an earlier result. Both tasks of this project use `NO_CACHE`, because the same inputs on another day of the clock must do another day's work. | 38 |
| Capability (Linux) | One part of the power of root, such as opening ports below 1024. The API container drops all of them (`cap_drop: ["ALL"]`). | 33 |
| Cardinality (of a metric) | The number of distinct combinations of label values of a series. Each combination is a time series of its own, so a label must have few values: here the route and the status, never a day or a zone. | 31 |
| Categorical feature | An input whose values are labels from a fixed list, not amounts. Here: `zone_id`. A tree can put any group of labels on one side of a question. | 24 |
| Certificate authority (CA) | An organisation that signs the certificates by which a website proves who it is. A root CA is one that software trusts directly; certifi is Python's list of them. | 41 |
| Champion / challenger | The model in use / a new model that must beat it to replace it. As aliases: `champion` points at the served version, and `challenger` at the newest version the promotion gate judged and did not promote. In retraining, the challenger is the recipe trained again up to the cutoff; if it is rejected, it gets the alias `challenger`. | 1, 29, 37 |
| Clean exam | The score on a hold-out that no decision has used. Here the replay of May and June 2025: MAE 27.80, against 24.11 in the April backtest that chose the settings. | 23, 27, 34 |
| Cleaning report | The record of a cleaning run: rows in, rows removed by each rule, rows out. Here: `reports/silver_cleaning_report.json`. | 16 |
| Cleaning rule | A condition that marks a row as not valid, with a name and a fixed place in the order of the rules. | 16 |
| Checksum | A short code calculated from the content of a file. If the content changes, the code changes. | 4, 15 |
| Chunk | One piece of a file that is read or received in pieces. | 14 |
| CI | Continuous integration: every change is checked automatically on a clean machine as soon as it is pushed. Here the workflow `.github/workflows/ci.yml` on GitHub Actions runs lint, types, tests, a dependency audit and an image scan on every push and pull request. Building the images with their commit written into them is still to come. | 13, 40, 41 |
| CLI | Command-line interface: a program controlled by typed commands. | 9 |
| Client / server | The program that sends a request / the program that waits for requests and answers them. | 30 |
| Closing step | The last step of the replay. It moves the simulated clock to 00:00 on 1 July 2025, the end of the demand history, forecasts nothing, and scores 29 and 30 June. Every later step repeats it and changes nothing. | 34 |
| Cluster | A group of machines that Kubernetes treats as one. Here a kind cluster with one node. | 39 |
| ClusterIP | The type of a Kubernetes Service that is reachable only inside the cluster. Both Services of this project have it; from the laptop they are reached with a port-forward. | 39 |
| Code review | Reading a change before it is accepted, to find mistakes and to check that code, tests and documents agree. | 13 |
| Commit hash pinning | Naming an action by the full commit hash of one exact version of its code instead of by a tag. A tag can be moved to other code; a hash cannot. Compare *digest* for images. | 41 |
| Comparison weeks (retraining) | The 28 days before the cutoff, on which the champion's real forecasts, a backtest of the retrained recipe and the weekly baseline are scored on the same 13,440 zone-hours. | 37 |
| Complete (of an hourly count) | Said of the count of an hour once the hour has ended. The pickups of 18:00 are complete at 19:00. | 2 |
| Complete grid | A table with one row for every combination of zone and hour, whether something happened or not. An hour without pickups is a row with 0. | 18 |
| Concurrency limit | The largest number of runs that may be in progress at once. The deployment `daily-step` allows one, with the strategy `CANCEL_NEW`: a run that becomes due while another is in progress is cancelled, not queued. | 38 |
| ConfigMap | A Kubernetes object of names and values that are not secret, given to pods as environment variables. Here `demand-settings`, three `DEMAND_*` variables; Kustomize adds a hash of the content to its name (`demand-settings-htgkd2d88h`), so a changed value rolls the pods. | 39 |
| Configuration | The values that can change without the logic changing: dates, paths, limits, addresses. | 7 |
| Constant | A name for a value that never changes, written in capitals. | 8 |
| Constraint (lock file) | A file given to `uv pip compile` with `-c`, whose versions the new lock must use. The lock of each image is made with `-c requirements.lock`, so it holds the same versions as the main lock. | 32 |
| Container | A program started from an image. It sees the files of the image as its own filesystem and has its own network, and sees nothing of the machine unless it is given access. Many containers can run from one image. | 32 |
| Control plane | The part of a Kubernetes cluster that decides: it keeps the objects, chooses a node for each pod, and runs the controllers. | 39 |
| Controller (Kubernetes) | A program of the control plane that compares what exists with an object's desired state and acts until they agree, for example by starting a pod when one is missing. | 39 |
| Conventional commits | A format for commit messages: `type: description`. | 12 |
| Cooldown | A period after an action in which the action is not repeated. Here a retrain waits at least `monitoring.retrain_cooldown_days` (7) calendar days after the last one, read from the tag `retrain.now` in the registry. | 37 |
| CORS | Cross-origin resource sharing: rules that let a web page from another address call a service from a browser. Left out of this service on purpose. | 31 |
| Counter, gauge, histogram | Three kinds of Prometheus series: a number that only goes up; a value that can go up and down; counts of values in ranges, with their number and their sum. Every number of the monitoring report is sent as a gauge. | 31, 36 |
| Coverage | The share of the code that the tests execute. | 10 |
| Coverage gate | A rule that fails the test run when coverage is under a limit. Here 80%, applied with `--cov`. | 10 |
| Crafted (file or input) | Made on purpose to trigger a bug, for example an image file built to crash a decoder. | 41 |
| CrashLoopBackOff | The state Kubernetes shows for a container that exits again and again, restarted with growing pauses: 10 seconds, then twice as long each time, up to five minutes. Docker Compose shows the same situation as `Restarting (3)`. The API pods were in it on kind until a model existed. | 33, 39, 40 |
| CRLF / LF | The two ways to mark the end of a line: Windows (two characters) and Linux or macOS (one). | 12 |
| Cron schedule | A schedule written as five fields: minute, hour, day of month, month, day of week, named after the Unix tool cron. `"0 1 * * *"` means 01:00 every day. | 38 |
| CronJob (Kubernetes) | A Kubernetes object that creates a Job on a cron schedule. `daily-step` runs `demand step` at 01:00 New York time; `concurrencyPolicy: Forbid` skips a run while the last one is still running. On kind it is **suspended**: the schedule starts nothing, and runs are started by hand with `kubectl create job --from=cronjob/daily-step`. | 38, 39, 40 |
| Cutoff (retraining) | The last day a retrained model may learn from: the day before the moment of the retrain, so every training target ended before its first forecast. | 37 |
| CVE | "Common Vulnerabilities and Exposures": the most common list of published vulnerabilities, with ids such as `CVE-2026-53995`. | 41 |
| Data contract | A precise description of a table that can be checked automatically and that producer and consumer both rely on. | 8 |
| Data leakage | Training or testing with information that is not available when the model is really used. | 2, 19 |
| Daylight saving time | The summer clock, one hour ahead of the winter clock. The change skips one local hour in spring and repeats one in autumn. | 17 |
| Day-ahead forecast | A forecast for all hours of a day, issued on the day before. Here it can be issued at 00:00 on the day before. | 1, 2 |
| Decision tree | A model that asks yes-or-no questions about the inputs of a row and ends in a number. The place where it ends is a leaf. | 24 |
| Decorator | A marker above a function, starting with `@`, that changes how the function is used. | 9 |
| Dependency | A library that the code needs in order to run. | 6 |
| Dependency injection | Giving a function the things it depends on, such as a network client, as arguments, so that a test can pass different ones. | 14 |
| Deployment (Kubernetes) | A Kubernetes object that keeps a number of identical pods running from one template, and replaces them when the template changes. Here `api` (rolling update, replicas set by the HPA) and `mlflow` (one replica, `Recreate`). Not the same as a Prefect deployment. | 39 |
| Deployment (Prefect) | A flow registered with the Prefect server, with its parameters and schedule, so that the server can say when a run is due. Here `daily-step/nyc-taxi-demand`, registered by `python -m demand schedule`. | 38 |
| `deps`, `params`, `outs` | The three lists of a stage in `dvc.yaml`: the files the stage reads, the config keys it uses, and what it writes. `deps` is short for dependencies, here in the sense of files, not libraries. | 21 |
| Desired state | What should exist, described as objects. Kubernetes keeps working until the cluster matches it, so a dead container is replaced without anyone noticing. | 39 |
| Deterministic | Said of a process that gives the same output for the same input, every time. | 25 |
| Digest (image) | The SHA-256 checksum of one exact image, written after `@`. Unlike a tag, it cannot be moved to another image. | 32 |
| Docker | A tool that builds images and runs them as containers. On Windows, Docker Desktop runs the containers in a small Linux virtual machine. | 32 |
| Docker Compose | A tool that starts several containers together from one file, here `docker-compose.yml`, with their volumes, mounts, ports and network. | 33 |
| Dockerfile | A text file with one instruction per step, from which an image is built. Here `docker/api.Dockerfile` and `docker/pipeline.Dockerfile`. | 32 |
| Docstring | A text at the start of a file, class or function that documents it. | 5 |
| Drift | A change in the data compared with the training period. It says that the world changed, not that the model became worse. See *input drift* and *prediction drift*. | 3, 35 |
| DVC | Data Version Control: a tool next to git that runs a pipeline of stages and repeats only the stages for which something changed. It can also store versions of data files. | 21 |
| `dvc.lock` | The record that DVC writes after a run: a checksum of every file a stage read and wrote, and the value of every config key it uses. Stored in git. | 21 |
| `dvc.yaml` | The file that describes the stages of the pipeline for DVC. | 21 |
| Editable install | An install that points to the source folder, so edits are active without installing again. | 5 |
| Elapsed time | Real time that passed between two moments, as opposed to the difference between two clock readings. The two differ on the days the clocks change. | 17, 19 |
| emptyDir | A Kubernetes volume that is an empty folder, created with its pod and deleted with it. Here `/tmp` of every pod, because the root filesystem is read-only. | 39 |
| Endpoint | One path of a web service with its method, such as GET `/predict`. This service has five. | 30 |
| Entry point | The place where a program starts. Here: `python -m demand`. | 9 |
| `.env` file | A text file with lines of the form `NAME=value`, read into environment variables when the program starts. Not stored in git. | 7 |
| Environment variable | A named value outside the program, set by the shell or the platform and read at start-up. | 7 |
| Ephemeral mode (Prefect) | What Prefect does when a flow runs without a server address: it starts a temporary server for the run, about 17 to 20 seconds here, with a database under `PREFECT_HOME`. Prefect waits 20 seconds for it by default; the tests allow 120, because a busy machine with few CPUs needed more. | 38, 41 |
| Exit code | The number a program returns when it ends. Zero is success. uvicorn returns 3 when the startup of the service fails. | 9, 31 |
| Expanding window | A way to cut the training rows of a backtest: every fold trains from the first row of the table, so later folds have more training rows. | 23 |
| Expected failure | A failure that is not a mistake in the code, such as a missing input file or a failed quality gate. It is printed as one line that says what to do next. Here: every `PipelineError`. | 20 |
| Experiment tracking | Recording every training run with its settings, scores and origin, so that runs can be found and compared later. Here: MLflow. | 26 |
| Exponential backoff | Waiting twice as long after every failed attempt before trying again. | 14 |
| Extra | See *optional dependency group*. | 6 |
| Fail closed / fail open | What a check does when it cannot do its job. Fail closed: stop with an error. Fail open: let everything pass. | 8 |
| False alarm | A check that fires when nothing that it should report has happened. | 36 |
| False-alarm rate | The share of quiet periods on which a check fires. The input-drift test fired in 10 of 11 weeks of the training period, so its rate is near 100% and it may not trigger an action. | 36 |
| Feature | An input column of a model. | 1, 19 |
| Feature importance (gain) | For each input, the share of the improvement during training that came from tree questions about that input. It shows what the trees used, not a direction or a cause. | 24 |
| Final fit | Fitting the chosen recipe once more on all training rows, after the backtest. The result is the model that is stored. | 26 |
| Fit | To let a forecaster learn from past rows. A baseline has the method and learns nothing. | 22 |
| Fix version | The first release of a library in which a known vulnerability is fixed. Trivy's `--ignore-unfixed` leaves out findings that have none yet. | 41 |
| Fixture | A function that prepares something for tests. A test asks for it by parameter name. | 10 |
| Flaky test | A test that passes and fails on the same code. It is a bug that has not been found yet. The fix is to make it fail every time, find the cause and remove it, not to run it again. | 41 |
| Flow (Prefect) | A Python function, marked with `@flow`, that Prefect runs and records as one run with its state and logs. Here `daily_step`. | 38 |
| Fold | One repetition of a backtest, with its own training and test period. | 3, 23 |
| Footer (Parquet) | A small block at the end of a Parquet file that describes its columns, types and number of rows. | 14 |
| Forecast | A prediction about a future time, made from past data. | 1 |
| Forecast horizon | How far ahead of the newest known pickups a forecast must reach. Here 48 hours: no feature of an hour reads pickups newer than 48 hours before it. | 1, 2 |
| Forecaster | Anything that has the two methods `fit` and `predict`: a baseline or a model. The backtest scores all forecasters with the same code. | 22 |
| Formatter | A tool that rewrites the layout of code in one fixed style. | 11 |
| Frozen | Said of an object that cannot be changed after it is created. | 7 |
| Gap (backtest) | The hours between the last training hour of a fold and its first test hour, used for neither. Here 47 hours per zone, so that no model learns from pickups that did not exist when its first forecast was issued. | 23 |
| GHSA | A GitHub Security Advisory: GitHub's list of vulnerabilities, with ids such as `GHSA-6q48-6c7w-9gw2`. | 41 |
| Git hook | A script that git runs automatically at a certain moment, for example before a commit. | 11 |
| Gold layer | Data in the form its users need: one row per zone and hour, and the table of model inputs. | 4, 18 |
| Graceful shutdown | Stopping a server so that the requests in progress are finished first. Not tested for this service. | 31 |
| Gradient boosting | Building many decision trees in a row, where each new tree is trained on what the trees before it still get wrong. The forecast is the sum of all trees. | 24 |
| Ground truth | The real value that a prediction tried to match. | 3 |
| Hard-coded | Said of a value written directly in the code. | 7 |
| Hold-out | A period of data kept away from training and from every decision, so that its score is unbiased. Here May and June 2025. | 34 |
| Health check | A command that Docker runs inside a container at intervals, which marks it healthy or unhealthy. The API image asks `/ready`. Compose can start a service only when another is healthy (`condition: service_healthy`). | 32, 33 |
| HorizontalPodAutoscaler (HPA) | A Kubernetes object that changes the number of replicas of a Deployment from a measurement. The API's HPA keeps 2 to 4 pods at an average of 70 % of their CPU request: average = CPU used by all pods / CPU requested by all pods. It reads the CPU from metrics-server. | 39, 40 |
| HTTP | The protocol of the web: the agreed format of requests and responses between a client and a server. | 30 |
| HTTP method | The word at the start of a request that says what the client wants. GET asks to read and changes nothing; POST sends data to be processed or stored. | 30 |
| Hyper-parameter | A setting fixed before training that decides how the model is trained, such as the number of leaves of a tree. Training cannot learn it; tuning searches for it. In the config: `model.params`. | 27 |
| Idempotent | Said of an operation that gives the same result when it is run again. | 15, 17 |
| Image | A package of files and settings from which containers are started: a small Linux filesystem, Python, the libraries, the code and the start command. Built once and never changed. | 32 |
| Immutable | Cannot be modified after it is written or created. | 4 |
| Imputation | Filling a missing value with a calculated one, for example the average. | 16 |
| Import path | The list of folders where Python searches for code to import. | 5 |
| In-sample / out-of-sample (score) | A score measured on the data that chose the settings / on data that took no part in the choice. An in-sample score is too good. | 27 |
| Ingest | To bring data from outside sources into the project. The first stage of the pipeline. | 14 |
| Ingress | A Kubernetes object that brings requests from outside the cluster to a Service. This project has none; one person uses a port-forward. | 39 |
| Input drift | A change in the distribution of the model inputs compared with the reference. Here 10 of the 16 inputs are tested, and the inputs count as drifted when the share of drifted columns is 0.3 or more. | 35 |
| Input example | A few rows of valid model inputs, stored next to the model. Here: the first five training rows. | 26 |
| Interval schedule | A schedule that makes a run due every fixed period, such as one minute: `--every-minutes 1`, the demo of the stack. | 38 |
| Issue moment | The moment a forecast is made, which fixes what it may know. Here 00:00 on the day before the forecast day, or 01:00 around the night the clocks go back. The service reports it as `issued_at`; the replay stores it as `generated_at`. | 2, 30, 34 |
| Job (GitHub Actions) | A part of a workflow that runs on its own runner, as a list of steps. Jobs run at the same time unless one waits for another. `ci.yml` has three: `checks`, `audit` and `images`. | 41 |
| Job (Kubernetes) | A Kubernetes object that runs a pod until it finishes and records whether it succeeded (`Complete` or `Failed`). `backoffLimit: 0` means a failed Job is not retried. A CronJob creates one per run. | 39, 40 |
| JSON | A standard text format for structured data. The service answers in it. | 9, 30 |
| Junction (Windows) | A Windows link from one folder to another, made with `mklink /J`. The clean worktree used for training had its `data` folder as a junction to the project's data. | 40 |
| kind | "Kubernetes in Docker": a tool that runs each node of a Kubernetes cluster as a Docker container. Here v0.33.0, one node, Kubernetes 1.36.4. | 39, 40 |
| kubeconform | A tool that checks Kubernetes manifests against the official schemas, without a cluster. It found the 21 objects of `k8s/` valid. | 40 |
| kubectl | The command-line tool that talks to the API server of a Kubernetes cluster. It also contains Kustomize. | 39 |
| Kustomize | A tool, built into `kubectl`, that builds the final manifests from a base and an overlay. `kubectl kustomize` prints them; `kubectl apply -k` applies them. | 39 |
| Label (metric) | A name and a value that split a Prometheus series, such as `status="422"`. | 31 |
| Label / selector (Kubernetes) | A label is a key and a value on an object, such as `app.kubernetes.io/name: api`. A selector picks every object with a label; Deployments and Services find their pods this way. | 39 |
| Lag feature | A past value of the target used as an input. | 2, 19 |
| Layer (image) | The files that one step of a Dockerfile added or changed. An image is a stack of layers. | 32 |
| Layer cache | Docker's reuse of a step whose instruction and copied files have not changed since the last build. The reason the lock is installed before the code is copied. | 32 |
| Lazy import | Importing a module at the moment it is needed, not at the top of the file. | 9 |
| Lazy validation | Checking a table against a schema in a way that runs every rule and reports all failures together, where the normal way stops at the first. | 20 |
| Lead time | The time from the issue moment of a forecast to the start of the hour it forecasts. Here 24 hours for 00:00 on the forecast day and 47 hours for 23:00. | 34 |
| Leakage test | A test that changes the target at one hour, builds the features again, and checks that no row earlier than that hour plus the horizon changed. | 19 |
| Lifespan | The function that FastAPI runs once when the application starts. Here it loads the model and the data, before uvicorn opens the port. | 31 |
| LightGBM | A library for gradient boosting. Here: the forecasting model, wrapped in the class `LightGbmForecaster`. | 24 |
| Linear interpolation | Filling missing values on a straight line between the known value before the gap and the known value after it. | 17 |
| Linter | A tool that reads code and reports likely mistakes and bad habits. | 11 |
| Live MAE | The MAE of the forecasts that were issued in use, scored when their days are over. Here the MAE of the replay, 27.80. Divided by the backtest MAE of the same version it gives `live_to_backtest_mae`, 1.15. | 34 |
| Liveness / readiness | Two questions that Kubernetes asks a container with automatic requests called probes: is the process alive (if not, it is restarted), and can it answer now (if not, it gets no requests but is not restarted). Here `/health` and `/ready`. See also Probe. | 31, 39 |
| Localize | To attach a timezone to a timestamp that was recorded without one. The clock time stays the same. | 17 |
| Lock file | A list of the exact version of every installed library. | 6 |
| Log scale (of SHAP values) | With the Poisson objective the trees add up to the natural logarithm of the forecast. A contribution is then a factor: +0.69 doubles the forecast, −0.69 halves it. | 28 |
| Log scale (search space) | A range in which every doubling is equally likely to be tried: values from 8 to 16 as often as from 64 to 128. | 27 |
| MAE | Mean absolute error: the mean size of the error, here in pickups per zone and hour. The main metric of this project. | 22 |
| Manifest | The record of every raw file: source address, download time, size, checksum, row count. | 4, 15 |
| Manifest (Kubernetes) | A YAML file that describes Kubernetes objects. Not the same as the manifest of the raw layer. The manifests of this project are in `k8s/`. | 39 |
| Margin (promotion) | The smallest gain in pooled MAE that lets a new version replace the champion. Here 0.2, the measured range of the seeds on the backtest folds, written as the constant `MIN_MAE_GAIN` in the code. | 29 |
| Mask | A column of true and false values, one per row, that marks the rows for which a condition holds. | 16 |
| Matrix (GitHub Actions) | A list of values for which the same job runs once each. Here `image: [api, pipeline]` builds and scans both images with one job description. | 41 |
| MD5 | A method for calculating a checksum of 32 characters. DVC uses it to see whether a file changed. | 21 |
| Mean absolute SHAP value | The mean size of an input's contributions over many rows, whatever their direction. Here stored as `shap.mean_abs.<input>`. | 28 |
| Medallion architecture | Storing data in layers of increasing quality: raw (bronze), silver, gold. | 4 |
| Metric | A rule that turns the errors of many forecasts into one number. | 22 |
| metrics-server | A cluster add-on that measures the CPU and memory of every pod, for the HPA and `kubectl top`. kind does not ship it; here v0.9.0 is vendored. | 39, 40 |
| Missing value | A value that is not there: it was never recorded, or it was lost. Stored as an empty cell, not as a number. | 15, 16 |
| MLflow | A tool for experiment tracking with a model registry. Here it writes to `mlflow.db` and `mlruns`. | 26 |
| Model card | A short document, written by hand, that says what one model version is for, where it must not be used, how it was evaluated and where it is weak. Here: `docs/model-card.md`. | 29 |
| Model registry | A place that keeps every version of a model with a label. Here: the name `taxi-demand` in MLflow. | 1, 26 |
| Model signature | The names and types of the input columns of a stored model, in order, and the type of its output. MLflow checks every table against it when the model is loaded. The service checks at startup that it names exactly the columns the config builds. | 26, 31 |
| Model version | The number the registry gives a model each time one is registered under the same name: 1, 2, 3. | 26 |
| monkeypatch | A pytest fixture that changes something for one test and restores it afterwards. | 10 |
| MSYS path conversion | Git Bash rewrites arguments that look like Unix paths into Windows paths when it starts a Windows program. It broke `kubectl cp ...:/tmp/...`; `MSYS_NO_PATHCONV=1` turns it off. | 40 |
| Multi-stage build | A Dockerfile with more than one stage, where a later stage copies only what it needs from an earlier one. Here the venv crosses from `builder` to `runtime`, and the install tools stay behind. | 32 |
| Naive timestamp | A date and time without a timezone. | 17 |
| Named volume | A volume that Docker creates and keeps under a name. The stack has six, with the prefix `nyc-taxi-demand_`; `docker compose down -v` deletes them. | 32, 33 |
| Namespace (Kubernetes) | A named group of objects inside a cluster. Everything of this project is in `nyc-taxi-demand`. | 39 |
| NaT | "Not a time": the missing value for timestamps in pandas. | 17 |
| Network (containers) | A private connection between containers, on which one reaches another by its service name, such as `http://mlflow:5000`. Compose creates one for the stack. | 32, 33 |
| NetworkPolicy | A Kubernetes rule of which pods may talk to which. This project has none, so every pod of the namespace can call every other one. | 39 |
| Node | One machine of a Kubernetes cluster, real or virtual. On kind, a node is a Docker container. | 39 |
| Noise (of a score) | Movement of a score that comes from chance, such as another seed, and not from a difference in the model. Here about 0.2 pickups of MAE. | 25 |
| Non-root user | A user without the privileges of root. Both images run as user 10001, given by number so that Kubernetes can check `runAsNonRoot`. | 32 |
| Objective | The measure of error that training makes small. Here: `poisson`, which is made for counts. | 24 |
| OOMKilled | "Out of memory, killed": the reason Kubernetes shows when a container used more memory than its limit and was stopped. | 39 |
| OpenAPI | A standard format that describes the endpoints, parameters and answers of a web service. FastAPI writes it at `/openapi.json` and shows it at `/docs`. | 30 |
| Optional dependency group | A named set of additional libraries, installed only where needed. | 6 |
| Optuna | A library that searches for hyper-parameters: it proposes values, sees the score, and proposes better ones. Used by `python -m demand tune`. | 27 |
| Orchestrator | A program that starts the steps of a pipeline on a schedule, runs one at a time where needed, records every run with its logs, and retries what is safe to retry. Here Prefect, which runs the daily step `run_day` as the flow `daily-step`. | 37, 38 |
| Outlier flag | A true/false column that marks an hour whose pickups are far from normal for its slot. The hour is kept. Here: `is_outlier`. | 18 |
| Over-fitting to the validation set | Settings chosen on a validation set fit it better than they fit new data, so its score for them is too good. Here about 30% of the in-sample tuning gain did not hold on weeks the search had not seen. | 27 |
| p-value | The probability of a difference at least as large as the one seen, if both samples came from the same distribution. It assumes independent rows, which hours of a time series are not, so this project's drift test uses no p-value. | 35 |
| Package | A folder of Python files that can be imported by its name. | 5 |
| Parameter (of a model) | A value that training finds in the data, such as the threshold of a tree question. Compare *hyper-parameter*. | 27 |
| Parametrised test | One test function that runs once for each entry in a list of inputs. | 10 |
| Parquet | A compact file format for tables. | 12 |
| Path (of a request) | The part of an address that says which part of a service should answer, such as `/predict`. | 30 |
| Path traversal | A name such as `../../secret` that leads a program out of the folder it should stay in. | 41 |
| Persist (DVC) | The option `persist: true` of an output. It tells DVC not to delete the output before it runs the stage. | 21 |
| PersistentVolumeClaim (PVC) | A request for storage that outlives every pod. The cluster gives it a PersistentVolume that fits. Here `demand-data` for the data and the replay, and `mlflow-store` for MLflow. | 39 |
| Pickle | Python's own format for saving objects to a file. It can only be loaded where the classes are installed, and only from a trusted source. | 26 |
| Pin | To fix a library to one exact version. | 6 |
| Placeholder | A function that exists by name and only raises `NotImplementedError` until the real one is written. | 9 |
| Placeholder value | A value such as zero, or a copy of another field, that a source writes where it has no measurement. | 16 |
| Pod | The smallest thing Kubernetes runs: one or more containers that share a network address and volumes. A pod is never repaired; a new pod replaces it. | 39 |
| Pod Security Standard | One of three levels of rules for pods in Kubernetes. The level "restricted", set by a label on the namespace, makes the API server refuse any pod that could run as root, keep Linux capabilities, gain privileges, or run without a seccomp profile. | 39 |
| Pooled MAE | The MAE over all test rows of all folds together. With folds of equal size, as here, it equals the mean of the fold MAEs. In the replay, the MAE over all 28,800 scored rows of the 60 days. | 29, 34 |
| Port | A number that tells a computer which program a network request is for. The service listens on 8000. | 30 |
| Port mapping | A rule that makes a port of a container reachable from the machine, such as `127.0.0.1:8000:8000`. Bound to `127.0.0.1`, it is not reachable from the network. | 33 |
| Port-forward | `kubectl port-forward` opens a tunnel from a port of your machine into the cluster. Through a Service it sends every request to one pod; it does not spread them. | 39, 40 |
| Prediction drift | A change in the distribution of the forecasts compared with forecasts of the same recipe on hours it had not learned from: here the backtest's fold forecasts, computed again. | 35 |
| Probe | An automatic request that Kubernetes sends to a container at fixed intervals. The startup probe asks whether it has started, and holds back the other two; the liveness probe whether it is alive (if not, it is restarted); the readiness probe whether it can answer (if not, it gets no requests). | 39 |
| Producer / consumer | The step that writes a table / the step that reads it. | 8 |
| Profile (Compose) | A label that keeps a service out of `docker compose up`. The service `pipeline` has the profile `pipeline`, because it runs a command and ends; the `scheduler` has `schedule`, because it writes a new day of forecasts every minute. | 33, 38 |
| Prometheus text format | The plain-text format of `/metrics`: lines `# HELP` and `# TYPE` for each series, then one line per value with its labels. | 31 |
| Promotion gate | The rule that decides whether the newest model version becomes champion, run as `python -m demand promote`. It is not part of `dvc repro`, so rebuilding never changes the served model. | 29 |
| Provisioning (Grafana) | Setting up Grafana from files at every start: here the Prometheus data source and the dashboards "Prediction API" and "Model monitoring". | 33, 36 |
| Pruning (of a trial) | Stopping a trial early when its first results look bad. Not used here. | 27 |
| Pull / push (metrics) | Prometheus pulls: it asks a running program for its numbers. A batch job pushes: it sends its numbers to a Pushgateway before it ends. | 36 |
| Pull request | A request to merge one branch into another, which others can review. CI runs on every pull request. | 41 |
| Pure function | A function whose result depends only on its arguments and that reads and writes nothing outside itself. | 16 |
| Push | To send commits to the shared copy of the repository. | 12 |
| Pushgateway | A small server of the Prometheus project that keeps the last numbers a batch job pushed, and lets Prometheus scrape them. Here the service `pushgateway` on port 9091; it keeps the numbers in memory only. | 36 |
| PYSEC | The list of vulnerabilities of the Python Packaging Authority, for packages on PyPI, with ids such as `PYSEC-2026-4320`. | 41 |
| Quality gate | A check that can stop a pipeline. A stage runs one on its table before it writes it. The command `validate` runs one on the files of a layer before the next stage reads them. | 8, 20 |
| Query parameter | A pair of a name and a value in the address of a request, after `?` and separated by `&`, such as `date=2025-05-14`. A name may appear more than once. | 30 |
| Race (race condition) | A bug that appears only when two things happen in a certain order or at the same moment. | 41 |
| Random split | Putting rows into training and test sets by chance. Wrong for data ordered in time. | 3 |
| Raw layer | Data exactly as downloaded, never modified. Also called bronze. | 4 |
| Read-only filesystem | A container whose files cannot be written (`read_only: true`). Writing into the API container fails with `Read-only file system`. | 33 |
| ReadWriteOnce / ReadWriteMany | Access modes of a volume. ReadWriteOnce: one node may mount it, with any number of pods on that node. ReadWriteMany: many nodes may. This project's volumes are ReadWriteOnce, which works only on one node. | 39 |
| Reference (drift) | The data that a window is compared with. For input drift, the inputs of the last 28 training days, 3 to 30 April (13,440 rows); for prediction drift, the backtest's out-of-sample forecasts. | 35 |
| Remote (DVC) | A shared storage place to which the DVC cache can be sent, so that other people can fetch the same data. This project has none. | 21 |
| Replay | Playing historical days that the model has never seen one at a time, each at the moment the system would have handled it. Here 2 May to 30 June 2025, with `python -m demand forecast`. | 3, 34 |
| Replica | One of the identical pods that a Deployment keeps running. | 39 |
| Request / limit (resources) | The request is the CPU and memory that the cluster reserves for a container; the limit is the most it may use. CPU is counted in thousandths: `500m` is half a CPU. Over its CPU limit a container is slowed down; over its memory limit it is killed (`OOMKilled`). | 39 |
| Request / response | The question a client sends to a server / the answer, with a status code and a body. | 30 |
| Restart policy | What happens when a container ends. In Docker Compose, `unless-stopped` starts it again, with growing pauses, unless a person stopped it. In a Kubernetes Job, `restartPolicy: Never` does not start a failed container again in the same pod. | 33, 39 |
| Retraining | Training the same recipe again on newer data, with a later end of training and nothing tuned. Here `python -m demand retrain`, only after an accuracy alert about the champion; the new model is served only if it passes the promotion rule. | 37 |
| Reversal | A row with negative amounts that cancels the charge of a trip, for example after a refund. It is not a second trip. | 16 |
| RMSE | Root mean squared error: like the MAE, with large errors counted much more. It is never smaller than the MAE. | 22 |
| Rollback | Making an earlier model version champion again, by hand, because the newer one behaved badly. Here: `python -m demand set-champion VERSION --reason TEXT`. It stays by hand after retraining, because an honest automatic rollback would need shadow forecasts. | 29, 37 |
| Rolling statistic | A summary, such as the mean, over a moving window of time. Here every window ends one horizon before the target hour. | 2, 19 |
| Rolling update / Recreate | Two ways a Deployment replaces its pods. A rolling update starts new pods before it removes old ones, so the service keeps answering (the API). Recreate stops the old pod first (MLflow, because two servers must not open one SQLite file). | 39 |
| Rolling window (of a score) | A score over the last few days, moved forward one day at a time. With 7 days, the ratio of live to backtest MAE in the replay peaked at 1.41 for the window that ends on 28 May. The accuracy check of monitoring computes it. | 34, 35 |
| Rolling-origin backtest | A backtest repeated with a training end date that moves forward. | 3 |
| Run (MLflow) | One recorded execution of the training, with an id. It holds parameters (settings chosen before), metrics (numbers produced), tags (where it came from) and artifacts (files). | 26 |
| Runbook | A list of commands to run in order, with what each should print. Here "Running it" in `k8s/README.md`. | 40 |
| Runner (GitHub Actions) | The virtual machine on which one job runs. GitHub starts a fresh one for each job and deletes it afterwards, so it has only what the repository describes. | 41 |
| Sampler | The part of Optuna that proposes the next values. Here TPE, seeded with `project.seed`: ten random trials, then more trials near the values that scored well. | 27 |
| Schedule | A rule that says when runs of a flow are due: an interval or a cron line. Here each run moves the simulated clock by one day, whatever the date on the wall. | 38 |
| Schema | The formal definition of a table: columns, types and rules. | 8 |
| Scrape | One request of Prometheus for the `/metrics` page of a target. Here every 15 seconds, of the API and of the Pushgateway. | 33, 36 |
| Search space | The ranges in which a search may choose each hyper-parameter. | 27 |
| Seasonal naive forecast | The forecast "the same as one season ago". Here the season is a week. See *baseline*. | 22 |
| Seccomp | A Linux filter of the requests that a process may make to the kernel. Every pod here uses `RuntimeDefault`, the standard filter of the container runtime. | 39 |
| Secret (Kubernetes) | Like a ConfigMap, for passwords and keys, stored base64-encoded. Here `demand-secrets` exists and is empty, because nothing in the cluster needs a secret yet. | 39 |
| Seed | The number from which a computer starts its sequence of random-looking numbers. The same seed gives the same random choices. Here: `project.seed`. | 25 |
| Seed check | Scoring the current and the best hyper-parameters again with two more seeds, to see whether a gain is larger than the noise. | 27 |
| Service (Compose) | One kind of container in `docker-compose.yml`, with its image and settings: `mlflow`, `api`, `prometheus`, `pushgateway`, `grafana`, `prefect`, and `pipeline` and `scheduler` behind profiles. | 33, 36, 38 |
| Service (Kubernetes) | A fixed name and address in front of the pods that match a selector, which spreads requests over those that are ready. Here `api` (`http://api:8000`) and `mlflow` (`http://mlflow:5000`). | 39 |
| Severity (vulnerability) | A rough measure of how bad the worst case of a vulnerability is: LOW, MEDIUM, HIGH or CRITICAL. The image scan of CI fails on HIGH and CRITICAL. | 41 |
| SHA-256 | A method for calculating a checksum of 64 characters. | 15 |
| Shadow forecast | A forecast of a model that is not served, issued and scored next to the served one and shown to nobody. It would let two models be compared on the same days; this project does not build it, so rollback stays by hand. | 37 |
| SHAP value | The contribution of one input to one forecast. The base value plus the SHAP values of all inputs of a row is exactly the output of the model. | 28 |
| Share of drifted columns | The number of tested columns that drifted, divided by the number tested. At 1 July it is 0.5: 5 of 10. | 35 |
| shellcheck | A linter for shell scripts. actionlint runs it on every `run:` step of a workflow. | 41 |
| Silver layer | Cleaned data: valid rows, fixed types, no duplicates. One row is one valid trip, or one hour of weather. | 4, 16 |
| Simulated clock | A "now" that the program sets itself, without the wall clock, to replay unseen historical data one day at a time as if it were happening. Here its position is the newest file in `data/predictions/forecasts/`. | 3, 34 |
| Slot | The group of hours that an hour is compared with when outliers are flagged: the same zone, the same hour of the day and the same kind of day (working day or weekend). | 18 |
| sMAPE | Symmetric mean absolute percentage error: for each hour the error divided by the average of the actual and the forecast, from 0 to 200 percent. Hours with few or no pickups decide it. | 22 |
| Snapshot (data) | A copy of data as it was at one moment, read once and never written. The service reads the gold demand and the silver weather as a snapshot at startup, in place of live feeds the project does not have. | 30 |
| src layout | Keeping the package inside a folder named `src/`, so that it must be installed to be imported. | 5 |
| Stabilisation window | The time an HPA waits before it removes pods, so that a short quiet moment does not shrink the number of pods. Five minutes by default. | 40 |
| Stage | One step of the pipeline, such as ingest, silver, gold or train. | 9, 21 |
| Standard deviation | A number that says how far the values of a group typically lie from their mean. | 18 |
| Stateless | Said of a service that writes nothing at request time, so every copy of it can answer any request. | 31 |
| Static analysis | Finding problems by reading code without running it. | 11 |
| Static type checker | A tool that reads type hints and reports contradictions. Here: mypy. | 11 |
| Status code | The number in the answer of a web server that says how the request went, for example 200, 404 or 503. From 400 to 499 the client must change the request; from 500 the server has the problem. 422 means the request was understood but cannot be answered as asked. | 14, 30 |
| stdout / stderr | The two output channels of a program: one for results, one for messages about the run. | 9 |
| Step (GitHub Actions) | One item of a job: a shell command (`run:`) or an action (`uses:`). The steps of a job run in order, and the job stops at the first that fails. | 41 |
| Streaming | Receiving or reading a file piece by piece, so that it is never in memory as a whole. | 14 |
| Strict schema | A schema that refuses columns it does not list (`strict=True`). | 8 |
| Structured logging | Writing each log event as a record with named fields. | 9 |
| Supply-chain attack | An attack on the tools and libraries a project uses, instead of on the project itself, such as a bad release on PyPI or an action whose tag was moved to bad code. Pinning by hash and the two-week rule are defences. | 41 |
| Tag (image) | The readable part of an image name after `:`, such as `3.12.15-slim-trixie` or `local`. A tag can be moved to another image; compare *digest*. | 32 |
| Tag (model version) | A label that stays with one model version, such as `promotion.decision = promoted`. Compare *alias*, which can move. | 29 |
| Target | The thing a model predicts. Here: the number of pickups. | 2 |
| Task (Prefect) | A function called inside a flow that Prefect records as a run of its own, with its own state, time, logs and retries. Here `day` (no retries) and `push to the pushgateway` (2 retries). | 38 |
| Test double | A simple object that stands in for a real one during a test. | 10 |
| Test set | The data that judges a model and takes part in no decision. A choice tuned on it makes its score too good. Here the replay of May and June plays this role. | 37 |
| Time-based split | Training on data before a date and testing on data after it. | 3 |
| Time-boxed exception | See *accepted risk*. | 41 |
| Timeout | The longest time a program waits for one operation before it treats it as failed. | 14 |
| Timezone-aware timestamp | A timestamp that carries its timezone, so that it names exactly one moment. | 17 |
| `tmp_path` | A pytest fixture that gives each test a new empty folder. | 10 |
| tmpfs | A folder kept in memory, not on disk. The API container's only writable folder is a tmpfs at `/tmp`. | 33 |
| Token (GitHub Actions) | A temporary password that a job gets to talk to GitHub. `permissions: contents: read` lets it only read the repository, and `persist-credentials: false` keeps it out of the checked-out folder. | 41 |
| Traceback | The report of files and line numbers that Python prints when an error is not handled. It is made for finding a mistake in the code. | 20 |
| Training report | The short record of a training run on disk: the MLflow run, the model version, the data version and the headline scores. Here: `reports/training_report.json`. | 26 |
| Training/serving skew | A difference between how features are calculated for training and for prediction. | 4, 19, 31 |
| Transient error | A failure that may go away on its own, such as a short network problem. Worth a retry. | 14 |
| Transitive dependency | A library that is installed because one of your dependencies needs it. | 6 |
| Trial | One set of hyper-parameters scored with the backtest. Trial 0 is the config as it is. | 27 |
| Trigger (retraining) | The condition that starts a retrain: an accuracy alert about the current champion, with a day left to forecast, the cooldown over, and the champion's forecasts for all comparison weeks. Drift never triggers it. | 37 |
| Tuning | Searching for better hyper-parameters by trying them. | 27 |
| Type hint | A note in the code that says which type a value has. | 11 |
| Typed settings object | An object in which every setting has a declared name and type, built once at start-up. | 7 |
| UTC | The world standard time, without summer time. | 9 |
| Validation report | The result of one quality gate: every check with its name, whether it passed, and a detail. Here: `reports/silver_validation_report.json` and `reports/gold_validation_report.json`. | 20 |
| Validator | A function that checks a rule when an object is created and raises an error if the rule is broken. | 7 |
| Vendoring | Downloading a file from another project once and committing it unchanged, with its checksum, instead of fetching it at install time. The manifest of metrics-server is vendored. | 39 |
| Virtual environment | A folder with its own Python and its own installed libraries, separate from all others. | 6 |
| Volume | Storage that lives outside every container, so that data outlives it. See *named volume* and *bind mount*. | 32 |
| Vulnerability | A mistake in software that an attacker can use. A known one is published with an id (CVE, GHSA, PYSEC), the versions it affects and, when one exists, a fix version. | 41 |
| Vulnerability database | A collection of published vulnerabilities with the versions they affect. OSV.dev joins many of them; pip-audit asks PyPI's service, and Trivy downloads its own database. | 41 |
| Warm-up rows | The first rows of every zone, for which the history is too short to fill all lags and windows. Here 215 hours per zone. They are dropped, not filled. | 19 |
| Wasserstein distance (normed) | The average distance the values of one distribution must travel to become the other, divided by the standard deviation of the reference. A column has drifted at 0.1 or more, a tenth of a standard deviation. | 35 |
| Web framework | A library in which a web application is written. It maps paths to functions and turns their results into responses. Here: FastAPI. | 30 |
| Web service | A program that starts, prepares itself and then waits for requests over the network, and answers each one. | 30 |
| Window (monitoring) | The last days a check looks at: 7 here. For drift, the last 7 forecast days issued; for accuracy, the last 7 days that are over, which end two days earlier. | 35 |
| Work pool / worker (Prefect) | The way to run deployments on infrastructure chosen per run: the server holds the work, and a worker process starts each run, for example in a new container. Not used here; `serve()` runs the flow in one long-running process. | 38 |
| Workflow (GitHub Actions) | A YAML file in `.github/workflows/` that says when it runs (`on:`) and which jobs it runs. Here `ci.yml`. | 41 |
| Worktree (git) | A second working folder of the same repository, at a commit you choose: `git worktree add --detach <folder> HEAD`. The cluster's first model was trained in one, so that it came from a clean commit. | 40 |
| Write-once | Said of a record that is never changed after it is written. Here a forecast file in `data/predictions/forecasts/`; the actual values go into a separate table, `scored/`. | 34 |
| YAML | A text format for configuration that is easy for people to read. | 7 |
| Z-score | The distance of a value from the mean of its group, counted in standard deviations. | 18 |

---

[Previous: Appendix A – Stack map](appendix-a-stack-map.md) | [Contents](README.md) | [Next: Appendix C – Further reading](appendix-c-further-reading.md)
