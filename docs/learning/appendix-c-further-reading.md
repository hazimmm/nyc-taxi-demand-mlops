# Appendix C – Further reading

Official documentation for the tools that the repository uses today, a few books, and one paper. Entries for later tools are added when their chapters are written. A web address is given for the main page of each project, and for a few pages that a chapter depends on; use the search of each site to find the topics named here.

## Python and packaging

- **Python Packaging User Guide** – `pyproject.toml`, the src layout, editable installs. <https://packaging.python.org>
- **Python documentation** – the modules `venv`, `logging`, `pathlib` and `importlib`, and the tutorial chapter on errors and exceptions, which explains tracebacks and error classes that inherit from each other. <https://docs.python.org>
- **Hatch documentation** – the hatchling build backend. <https://hatch.pypa.io>
- **uv documentation** – the command `uv pip compile`. <https://docs.astral.sh/uv/>

## Configuration and data

- **Pydantic documentation** – models, validators, and the section on settings management. <https://docs.pydantic.dev>
- **python-dotenv documentation** – the format of `.env` files.
- **Pandera documentation** – schemas, checks, strict mode, and lazy validation. <https://pandera.readthedocs.io>
- **PyYAML documentation** – which values YAML reads as a date, a number or true and false when they have no quotes. <https://pyyaml.org>
- **pandas documentation** – time series, time zones (`tz_localize` and `tz_convert`, with their options for times that do not exist or exist twice), and `interpolate`. For the gold layer: `reindex`, `merge`, `groupby` with `transform`, and the windowing operations (`shift`, `rolling`). <https://pandas.pydata.org>
- **NumPy documentation** – `sliding_window_view`, which the feature function uses to calculate every rolling window from its own values. <https://numpy.org>
- **holidays documentation** – the list of countries and what "observed" holidays are. <https://holidays.readthedocs.io>

## Getting data

- **HTTPX documentation** – clients, timeouts, streaming responses, and mock transports for tests. <https://www.python-httpx.org>
- **Apache Arrow documentation** – the Python library pyarrow and the Parquet format, and the `filters` argument that reads only the rows that match a condition. <https://arrow.apache.org>
- **DuckDB documentation** – reading Parquet files with SQL. <https://duckdb.org>
- **Open-Meteo documentation** – the historical forecast API and the historical weather (archive) API. <https://open-meteo.com>
- **NYC Taxi and Limousine Commission, TLC Trip Record Data** – the page that lists the monthly files and the data dictionary of the columns. Read the dictionary before you write a cleaning rule: it defines `fare_amount`, `total_amount` and the codes of `payment_type`.
- **IANA Time Zone Database** – the source of names such as `America/New_York` and of the dates on which clocks change. <https://www.iana.org/time-zones>

## Running the pipeline

- **DVC documentation** – pipelines (`dvc.yaml`, `dvc.lock`, `dvc repro`, `dvc status`), the `params`, `persist` and `metrics` fields, the cache, and remotes. <https://dvc.org/doc>

## Forecasting and its evaluation

- **Forecasting: Principles and Practice** by Rob Hyndman and George Athanasopoulos – a free online textbook. Read the sections on simple forecasting methods (the seasonal naive method is the baseline of Chapter 22), on the accuracy of point forecasts (MAE, RMSE and the problems of percentage errors), and on time series cross-validation, which is the backtest of Chapter 23. <https://otexts.com/fpp3/>
- **The Elements of Statistical Learning** by Trevor Hastie, Robert Tibshirani and Jerome Friedman – a textbook that its authors offer as a free PDF. The chapter on boosting and additive trees explains the method behind Chapter 24 in full.

## The model and its record

- **LightGBM documentation** – the list of parameters (`objective`, `num_leaves`, `learning_rate`, `deterministic`), the page on categorical features, and the Python class `LGBMRegressor`. <https://lightgbm.readthedocs.io>
- **MLflow documentation** – tracking (runs, parameters, metrics, tags, artifacts), models (the `pyfunc` flavour, signatures and input examples), and the model registry (versions and aliases). <https://mlflow.org/docs/latest/>
- **MLflow Model Registry** – the page on registered models, versions, aliases such as `champion`, and tags on versions, which the promotion gate of Chapter 29 uses. <https://mlflow.org/docs/latest/ml/model-registry/>

## Promoting and describing a model

- **Model Cards for Model Reporting** by Margaret Mitchell, Simone Wu, Andrew Zaldivar and others – the paper that proposed model cards, presented at the conference FAT* in 2019. Read its list of sections (intended use, factors, metrics, evaluation data, caveats) and compare it with `docs/model-card.md`. <https://arxiv.org/abs/1810.03993>

## Tuning and explaining the model

- **Optuna documentation** – the tutorial on the ask-and-tell interface (`study.ask` and `study.tell`, which `tuning.py` uses), the page on samplers with the TPE sampler and its seed, and the page on pruning. <https://optuna.org>
- **SHAP documentation** – `TreeExplainer`, the meaning of `expected_value` (the base value), and the summary plot. <https://shap.readthedocs.io>
- **Interpretable Machine Learning** by Christoph Molnar – a free online book. Read the chapter on Shapley values and SHAP, and its sections on the limits of feature importance. <https://christophm.github.io/interpretable-ml-book/>
- **matplotlib documentation** – backends (the `Agg` backend draws without a window) and `savefig`. <https://matplotlib.org>

## Serving

- **FastAPI documentation** – the tutorial, in particular the pages "Query Parameters", "Query Parameter Models" (the form `ForecastQuery` uses), "Handling Errors" and "Testing"; and in the advanced guide, "Lifespan Events". <https://fastapi.tiangolo.com/tutorial/>
- **Uvicorn documentation** – the pages on settings, deployment, and the lifespan of an application. <https://uvicorn.dev>
- **MDN Web Docs on HTTP** – an overview of HTTP, request methods (GET and POST), and the list of response status codes, including 404, 422 and 503. <https://developer.mozilla.org/en-US/docs/Web/HTTP>
- **Prometheus documentation** – the pages "Metric types" (counter, gauge, histogram), "Metric and label naming", and "Instrumentation", whose section "Do not overuse labels" explains the cost of cardinality. <https://prometheus.io/docs/concepts/metric_types/>, <https://prometheus.io/docs/practices/naming/> and <https://prometheus.io/docs/practices/instrumentation/>
- **Prometheus Python client** – the pages on Counter, Gauge, Histogram, labels and registries. <https://prometheus.github.io/client_python/>
- **Kubernetes documentation** – the page "Configure Liveness, Readiness and Startup Probes", used in Chapter 39; and the page "Configure a Security Context for a Pod or Container", which describes `runAsNonRoot` (Chapter 32). <https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/>

## Containers

- **Docker overview** – what images, containers and registries are, and how the parts of Docker fit together. <https://docs.docker.com/get-started/docker-overview/>
- **Dockerfile reference** – every instruction used in `docker/`: `FROM`, `ARG`, `RUN` with `--mount=type=cache`, `COPY --from`, `ENV`, `USER`, `HEALTHCHECK` and the exec form of `CMD`. <https://docs.docker.com/reference/dockerfile/>
- **Multi-stage builds** – the two stages of Chapter 32. <https://docs.docker.com/build/building/multi-stage/>
- **Building best practices** – pinning base images by digest, the order of steps for the layer cache, `apt-get` with `--no-install-recommends`, and a user without privileges. <https://docs.docker.com/build/building/best-practices/>
- **Build context** – what `docker build` sends, and the section on `.dockerignore` files. <https://docs.docker.com/build/concepts/context/>
- **Volumes** – named volumes, and how a new volume is filled from the image. <https://docs.docker.com/engine/storage/volumes/>
- **Compose file reference** – services, `depends_on`, `healthcheck`, `restart`, `profiles`, `read_only`, `tmpfs`, `cap_drop` and the long form of `volumes`. <https://docs.docker.com/reference/compose-file/>
- **Control startup and shutdown order in Compose** – `depends_on` with `condition: service_healthy`. <https://docs.docker.com/compose/how-tos/startup-order/>
- **Prometheus configuration** – `global`, `scrape_interval` and `scrape_configs`. <https://prometheus.io/docs/prometheus/latest/configuration/configuration/>
- **Provision Grafana** – data sources and dashboards set up from files. <https://grafana.com/docs/grafana/latest/administration/provisioning/>

## The replay and live scores

- **Designing Machine Learning Systems** by Chip Huyen (O'Reilly, 2022) – the chapter "Data Distribution Shifts and Monitoring" explains why a model gets worse after it is deployed, why accuracy needs actual values that arrive late, and what can be watched before they arrive. It is the background of Chapter 34 and of the monitoring that follows.
- **Forecasting: Principles and Practice** (see above) – the section on residual diagnostics: residuals whose mean is not zero mean that the forecasts are biased, as the replay's are.
- **pandas documentation** – `read_parquet`, whose path may be a folder: Chapter 34 reads all the files of `data/predictions/scored/` as one table this way.

## Monitoring

- **Evidently documentation** – the quickstart for data drift, and the page "Customize Data Drift", which lists every drift method with its default threshold and the rule by which Evidently chooses a method by itself (Chapter 35). <https://docs.evidentlyai.com> and <https://docs.evidentlyai.com/metrics/customize_data_drift>
- **SciPy documentation** – `scipy.stats.wasserstein_distance`, the function that Evidently divides by the standard deviation of the reference. <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html>
- **When to use the Pushgateway** – a page of the Prometheus documentation. It says that the usual valid use of a Pushgateway is the outcome of a batch job, and names its costs: one more component that can fail, and series that stay until they are deleted (Chapter 36). <https://prometheus.io/docs/practices/pushing/>
- **Pushgateway README** – how to push, why a PUT replaces every metric of its group, and the flag `--persistence.file`, which this project does not use, so the numbers live in memory only. <https://github.com/prometheus/pushgateway>
- **Designing Machine Learning Systems** (see above) – the same chapter on distribution shifts discusses how to detect shifts and why many detected shifts are harmless, which is the subject of Chapter 36.

## Retraining

- **Designing Machine Learning Systems** (see above) – the chapter "Continual Learning and Test in Production" discusses how often to retrain, why a retrained model must be evaluated against the one in production before it replaces it, and shadow deployment, the idea behind the shadow forecasts of Chapter 37.

## Scheduling

- **Prefect documentation, "Flows"** – what a flow is, how a flow calls tasks, and the section "Why both flows and tasks?" (Chapter 38). <https://docs.prefect.io/v3/concepts/flows>
- **Prefect documentation, "Tasks"** – what a task run records, and the options for retries and caching, which link to their own guides. <https://docs.prefect.io/v3/concepts/tasks>
- **Prefect documentation, "Deployments"** – the difference between `serve()`, one long-running process for flows on a fixed machine, and work pools with workers, which start each run on infrastructure chosen per run. <https://docs.prefect.io/v3/concepts/deployments>
- **Prefect documentation, "Schedule flow runs"** – the cron and interval schedules. <https://docs.prefect.io/v3/concepts/schedules>
- **Kubernetes documentation, "CronJob"** – used in Chapter 39: a schedule for containers, and `concurrencyPolicy`, the Kubernetes form of the rule "never two runs at once". <https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/>

## Kubernetes

- **Kubernetes documentation, "Learn Kubernetes Basics"** – a short tutorial: a cluster, a Deployment, a Service, scaling and a rolling update, in that order. Read it before Chapter 39. <https://kubernetes.io/docs/tutorials/kubernetes-basics/>
- **Kubernetes documentation, "Kubernetes Components"** – the control plane, the API server, the controllers and the nodes. <https://kubernetes.io/docs/concepts/overview/components/>
- **Kubernetes documentation, "Deployments" and "Service"** – rolling updates with `maxSurge` and `maxUnavailable`, the strategy `Recreate`, selectors, and the Service type `ClusterIP`. <https://kubernetes.io/docs/concepts/workloads/controllers/deployment/> and <https://kubernetes.io/docs/concepts/services-networking/service/>
- **Kubernetes documentation, "Resource Management for Pods and Containers"** – requests and limits, CPU in thousandths, and what happens to a container over its memory limit. <https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/>
- **Kubernetes documentation, "Horizontal Pod Autoscaling"** – the rule by which the HPA computes the number of replicas, its tolerance, how it treats pods without measurements, and the stabilisation window (Chapter 40). <https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/>
- **Kubernetes documentation, "Persistent Volumes"** – claims, volumes and the access modes ReadWriteOnce and ReadWriteMany. <https://kubernetes.io/docs/concepts/storage/persistent-volumes/>
- **Kubernetes documentation, "ConfigMaps" and "Secrets"** – settings as environment variables, and why a Secret is only encoded, not encrypted. <https://kubernetes.io/docs/concepts/configuration/configmap/> and <https://kubernetes.io/docs/concepts/configuration/secret/>
- **Kubernetes documentation, "Jobs"** – `backoffLimit`, `activeDeadlineSeconds`, `ttlSecondsAfterFinished`, and the restart policy of a Job's pod. <https://kubernetes.io/docs/concepts/workloads/controllers/job/>
- **Kubernetes documentation, "Pod Security Standards"** – the three levels, and every rule of the level "restricted". <https://kubernetes.io/docs/concepts/security/pod-security-standards/>
- **Kubernetes documentation, "Use Port Forwarding to Access Applications in a Cluster"** – what `kubectl port-forward` does. <https://kubernetes.io/docs/tasks/access-application-cluster/port-forward-access-application-cluster/>
- **Kubernetes documentation, "Declarative Management of Kubernetes Objects Using Kustomize"** – bases, overlays, patches, and the ConfigMap and Secret generators with their hashed names. <https://kubernetes.io/docs/tasks/manage-kubernetes-objects/kustomization/>
- **Kustomize reference** – every field of a `kustomization.yaml`, such as `images`, `patches`, `labels` and `configMapGenerator`. <https://kubectl.docs.kubernetes.io/references/kustomize/>
- **kind documentation, "Quick Start"** – installing kind, creating and deleting a cluster from a config file, pinning the node image, and `kind load docker-image`. <https://kind.sigs.k8s.io/docs/user/quick-start/>
- **metrics-server README** – what it measures, how to install it, and the flag `--kubelet-insecure-tls`, with its warning. <https://github.com/kubernetes-sigs/metrics-server>
- **kubeconform README** – checking manifests against the schemas of one Kubernetes version, without a cluster. <https://github.com/yannh/kubeconform>
- **Git documentation, `git worktree`** – a second working folder of the same repository, used in Chapter 40 to train from a clean commit. <https://git-scm.com/docs/git-worktree>
- **Kubernetes: Up and Running** by Brendan Burns, Joe Beda, Kelsey Hightower and Lachlan Evenson (O'Reilly, third edition, 2022) – a book that explains the same objects at more length, with the reasons behind them.

## Continuous integration and known vulnerabilities

- **GitHub Actions documentation** – the pages on workflows, jobs, runners and the matrix strategy, the reference of the workflow syntax (`on`, `permissions`, `concurrency`, `container`, `timeout-minutes`), and the page on the `GITHUB_TOKEN` and its permissions (Chapter 41). <https://docs.github.com/en/actions>
- **GitHub Actions documentation, "Security hardening for GitHub Actions"** (in newer versions of the site, "Secure use reference") – why a third-party action should be pinned by its full commit hash, the least permissions for the token, and how untrusted input can reach a `run:` step. Search for its title on the site above.
- **actions/checkout README** – the option `persist-credentials`. <https://github.com/actions/checkout>
- **actionlint README** – what it checks, and how it runs shellcheck on `run:` steps. <https://github.com/rhysd/actionlint>
- **pip-audit README** – auditing a requirements file, `--no-deps`, `--disable-pip`, `--ignore-vuln`, and the vulnerability services it can ask. <https://github.com/pypa/pip-audit>
- **Trivy documentation** – scanning an image, the options `--severity` and `--ignore-unfixed`, and the page on filtering, which describes the YAML ignore file with `statement` and `expired_at`. <https://trivy.dev>
- **OSV.dev** – a database that joins vulnerability records from many sources, with a search by package name or id; useful to read a CVE, GHSA or PYSEC record and its aliases. <https://osv.dev>
- **PyPA Advisory Database** – the source of the PYSEC ids. <https://github.com/pypa/advisory-database>
- **GitHub Advisory Database** – the GHSA records. <https://github.com/advisories>

## Command line, tests and code quality

- **Typer documentation** – commands, arguments, testing a CLI. <https://typer.tiangolo.com>
- **pytest documentation** – fixtures, `tmp_path`, `monkeypatch`, parametrised tests. <https://docs.pytest.org>
- **pytest-cov and Coverage.py documentation** – measuring coverage, branch coverage, `fail_under`.
- **Ruff documentation** – the list of rules and the formatter. <https://docs.astral.sh/ruff/>
- **mypy documentation** – type hints and strict mode. <https://mypy.readthedocs.io>
- **pre-commit documentation** – hooks and local hooks. <https://pre-commit.com>

## Git

- **Git documentation** – the pages on `gitattributes` and `gitignore`. <https://git-scm.com>
- **Conventional Commits** – the specification of the message format. <https://www.conventionalcommits.org>

---

[Previous: Appendix B – Glossary](appendix-b-glossary.md) | [Contents](README.md)
