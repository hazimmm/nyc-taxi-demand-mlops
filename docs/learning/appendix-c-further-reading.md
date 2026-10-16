# Appendix C – Further reading

Official documentation for the tools that the repository uses today, and one textbook on forecasting. Entries for later tools are added when their chapters are written. A web address is given only for the main page of each project; use the search of each site to find the topics named here.

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

## Tuning and explaining the model

- **Optuna documentation** – the tutorial on the ask-and-tell interface (`study.ask` and `study.tell`, which `tuning.py` uses), the page on samplers with the TPE sampler and its seed, and the page on pruning. <https://optuna.org>
- **SHAP documentation** – `TreeExplainer`, the meaning of `expected_value` (the base value), and the summary plot. <https://shap.readthedocs.io>
- **Interpretable Machine Learning** by Christoph Molnar – a free online book. Read the chapter on Shapley values and SHAP, and its sections on the limits of feature importance. <https://christophm.github.io/interpretable-ml-book/>
- **matplotlib documentation** – backends (the `Agg` backend draws without a window) and `savefig`. <https://matplotlib.org>

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
