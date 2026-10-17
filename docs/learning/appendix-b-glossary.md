# Appendix B – Glossary

Every technical term used in the book, in alphabetical order. The number in the last column is the chapter where the term is explained.

| Term | Meaning | Ch. |
|---|---|---|
| Actuals | The real observed values, here the real pickups, known only after the hour has passed. | 3 |
| Alias (model registry) | A name that points at one version of a registered model and can be moved to another. `models:/taxi-demand@champion` loads whichever version the alias `champion` points at now. The service resolves it to a version number once, at startup. | 26, 29, 31 |
| Artifact | A file that is stored with a training run, such as a report or the model. | 26 |
| ASGI | The standard interface between a Python web server and a web application. uvicorn and FastAPI meet at it, so either can be replaced alone. | 30 |
| Atomic | Said of an operation that happens completely or not at all, with no state in between. Renaming a file inside one folder is atomic. | 14 |
| Automated test | A small function that runs a piece of code and checks the result with `assert`. | 10 |
| Backtest | Testing a forecasting model on the past as if it had been used then. Here: four test weeks in April 2025, scored by `python -m demand backtest`. | 3, 23 |
| Base value (SHAP) | The output a model gives on average over its training rows, before any input of a row has contributed. Here about 4.49 on the log scale, which is about 89 pickups. | 28 |
| Baseline | A very simple method that sets the minimum a model must beat. Here: "the same hour one week ago", a copy of the feature `lag_168h`, named `weekly_naive`. | 1, 22 |
| Binary file | A file that is not text. Git must not change any byte of it. | 12 |
| Branch (git) | A named line of commits. Work happens on `dev`. | 12 |
| Branch coverage | Coverage that also requires both outcomes of every `if` to be executed. | 10 |
| Build backend | The tool that turns a source folder into something installable. Here: hatchling. | 5 |
| Cache (DVC) | The folder `.dvc/cache`, in which DVC keeps a copy of every output of a stage. | 21 |
| Cardinality (of a metric) | The number of distinct combinations of label values of a series. Each combination is a time series of its own, so a label must have few values: here the route and the status, never a day or a zone. | 31 |
| Categorical feature | An input whose values are labels from a fixed list, not amounts. Here: `zone_id`. A tree can put any group of labels on one side of a question. | 24 |
| Champion / challenger | The model in use / a new model that must beat it to replace it. As aliases: `champion` points at the served version, and `challenger` at the newest version the promotion gate judged and did not promote. | 1, 29 |
| Cleaning report | The record of a cleaning run: rows in, rows removed by each rule, rows out. Here: `reports/silver_cleaning_report.json`. | 16 |
| Cleaning rule | A condition that marks a row as not valid, with a name and a fixed place in the order of the rules. | 16 |
| Checksum | A short code calculated from the content of a file. If the content changes, the code changes. | 4, 15 |
| Chunk | One piece of a file that is read or received in pieces. | 14 |
| CI | Continuous integration: automatic checks that run on every push. Planned for Phase 7. | 13 |
| CLI | Command-line interface: a program controlled by typed commands. | 9 |
| Client / server | The program that sends a request / the program that waits for requests and answers them. | 30 |
| Code review | Reading a change before it is accepted, to find mistakes and to check that code, tests and documents agree. | 13 |
| Complete (of an hourly count) | Said of the count of an hour once the hour has ended. The pickups of 18:00 are complete at 19:00. | 2 |
| Complete grid | A table with one row for every combination of zone and hour, whether something happened or not. An hour without pickups is a row with 0. | 18 |
| Configuration | The values that can change without the logic changing: dates, paths, limits, addresses. | 7 |
| Constant | A name for a value that never changes, written in capitals. | 8 |
| Conventional commits | A format for commit messages: `type: description`. | 12 |
| CORS | Cross-origin resource sharing: rules that let a web page from another address call a service from a browser. Left out of this service on purpose. | 31 |
| Counter, gauge, histogram | Three kinds of Prometheus series: a number that only goes up; a value that can go up and down; counts of values in ranges, with their number and their sum. | 31 |
| Coverage | The share of the code that the tests execute. | 10 |
| Coverage gate | A rule that fails the test run when coverage is under a limit. Here 80%, applied with `--cov`. | 10 |
| CRLF / LF | The two ways to mark the end of a line: Windows (two characters) and Linux or macOS (one). | 12 |
| Data contract | A precise description of a table that can be checked automatically and that producer and consumer both rely on. | 8 |
| Data leakage | Training or testing with information that is not available when the model is really used. | 2, 19 |
| Daylight saving time | The summer clock, one hour ahead of the winter clock. The change skips one local hour in spring and repeats one in autumn. | 17 |
| Day-ahead forecast | A forecast for all hours of a day, issued on the day before. Here it can be issued at 00:00 on the day before. | 1, 2 |
| Decision tree | A model that asks yes-or-no questions about the inputs of a row and ends in a number. The place where it ends is a leaf. | 24 |
| Decorator | A marker above a function, starting with `@`, that changes how the function is used. | 9 |
| Dependency | A library that the code needs in order to run. | 6 |
| Dependency injection | Giving a function the things it depends on, such as a network client, as arguments, so that a test can pass different ones. | 14 |
| `deps`, `params`, `outs` | The three lists of a stage in `dvc.yaml`: the files the stage reads, the config keys it uses, and what it writes. `deps` is short for dependencies, here in the sense of files, not libraries. | 21 |
| Deterministic | Said of a process that gives the same output for the same input, every time. | 25 |
| Docstring | A text at the start of a file, class or function that documents it. | 5 |
| Drift | A change in the data compared with the training period. | 3 |
| DVC | Data Version Control: a tool next to git that runs a pipeline of stages and repeats only the stages for which something changed. It can also store versions of data files. | 21 |
| `dvc.lock` | The record that DVC writes after a run: a checksum of every file a stage read and wrote, and the value of every config key it uses. Stored in git. | 21 |
| `dvc.yaml` | The file that describes the stages of the pipeline for DVC. | 21 |
| Editable install | An install that points to the source folder, so edits are active without installing again. | 5 |
| Elapsed time | Real time that passed between two moments, as opposed to the difference between two clock readings. The two differ on the days the clocks change. | 17, 19 |
| Endpoint | One path of a web service with its method, such as GET `/predict`. This service has five. | 30 |
| Entry point | The place where a program starts. Here: `python -m demand`. | 9 |
| `.env` file | A text file with lines of the form `NAME=value`, read into environment variables when the program starts. Not stored in git. | 7 |
| Environment variable | A named value outside the program, set by the shell or the platform and read at start-up. | 7 |
| Exit code | The number a program returns when it ends. Zero is success. uvicorn returns 3 when the startup of the service fails. | 9, 31 |
| Expanding window | A way to cut the training rows of a backtest: every fold trains from the first row of the table, so later folds have more training rows. | 23 |
| Expected failure | A failure that is not a mistake in the code, such as a missing input file or a failed quality gate. It is printed as one line that says what to do next. Here: every `PipelineError`. | 20 |
| Experiment tracking | Recording every training run with its settings, scores and origin, so that runs can be found and compared later. Here: MLflow. | 26 |
| Exponential backoff | Waiting twice as long after every failed attempt before trying again. | 14 |
| Extra | See *optional dependency group*. | 6 |
| Fail closed / fail open | What a check does when it cannot do its job. Fail closed: stop with an error. Fail open: let everything pass. | 8 |
| Feature | An input column of a model. | 1, 19 |
| Feature importance (gain) | For each input, the share of the improvement during training that came from tree questions about that input. It shows what the trees used, not a direction or a cause. | 24 |
| Final fit | Fitting the chosen recipe once more on all training rows, after the backtest. The result is the model that is stored. | 26 |
| Fit | To let a forecaster learn from past rows. A baseline has the method and learns nothing. | 22 |
| Fixture | A function that prepares something for tests. A test asks for it by parameter name. | 10 |
| Fold | One repetition of a backtest, with its own training and test period. | 3, 23 |
| Footer (Parquet) | A small block at the end of a Parquet file that describes its columns, types and number of rows. | 14 |
| Forecast | A prediction about a future time, made from past data. | 1 |
| Forecast horizon | How far ahead of the newest known pickups a forecast must reach. Here 48 hours: no feature of an hour reads pickups newer than 48 hours before it. | 1, 2 |
| Forecaster | Anything that has the two methods `fit` and `predict`: a baseline or a model. The backtest scores all forecasters with the same code. | 22 |
| Formatter | A tool that rewrites the layout of code in one fixed style. | 11 |
| Frozen | Said of an object that cannot be changed after it is created. | 7 |
| Gap (backtest) | The hours between the last training hour of a fold and its first test hour, used for neither. Here 47 hours per zone, so that no model learns from pickups that did not exist when its first forecast was issued. | 23 |
| Git hook | A script that git runs automatically at a certain moment, for example before a commit. | 11 |
| Gold layer | Data in the form its users need: one row per zone and hour, and the table of model inputs. | 4, 18 |
| Graceful shutdown | Stopping a server so that the requests in progress are finished first. Not tested for this service. | 31 |
| Gradient boosting | Building many decision trees in a row, where each new tree is trained on what the trees before it still get wrong. The forecast is the sum of all trees. | 24 |
| Ground truth | The real value that a prediction tried to match. | 3 |
| Hard-coded | Said of a value written directly in the code. | 7 |
| HTTP | The protocol of the web: the agreed format of requests and responses between a client and a server. | 30 |
| HTTP method | The word at the start of a request that says what the client wants. GET asks to read and changes nothing; POST sends data to be processed or stored. | 30 |
| Hyper-parameter | A setting fixed before training that decides how the model is trained, such as the number of leaves of a tree. Training cannot learn it; tuning searches for it. In the config: `model.params`. | 27 |
| Idempotent | Said of an operation that gives the same result when it is run again. | 15, 17 |
| Immutable | Cannot be modified after it is written or created. | 4 |
| Imputation | Filling a missing value with a calculated one, for example the average. | 16 |
| Import path | The list of folders where Python searches for code to import. | 5 |
| In-sample / out-of-sample (score) | A score measured on the data that chose the settings / on data that took no part in the choice. An in-sample score is too good. | 27 |
| Ingest | To bring data from outside sources into the project. The first stage of the pipeline. | 14 |
| Input example | A few rows of valid model inputs, stored next to the model. Here: the first five training rows. | 26 |
| Issue moment | The moment a forecast is made, which fixes what it may know. Here 00:00 on the day before the forecast day, or 01:00 around the night the clocks go back. The service reports it as `issued_at`. | 2, 30 |
| JSON | A standard text format for structured data. The service answers in it. | 9, 30 |
| Label (metric) | A name and a value that split a Prometheus series, such as `status="422"`. | 31 |
| Lag feature | A past value of the target used as an input. | 2, 19 |
| Lazy import | Importing a module at the moment it is needed, not at the top of the file. | 9 |
| Lazy validation | Checking a table against a schema in a way that runs every rule and reports all failures together, where the normal way stops at the first. | 20 |
| Leakage test | A test that changes the target at one hour, builds the features again, and checks that no row earlier than that hour plus the horizon changed. | 19 |
| Lifespan | The function that FastAPI runs once when the application starts. Here it loads the model and the data, before uvicorn opens the port. | 31 |
| LightGBM | A library for gradient boosting. Here: the forecasting model, wrapped in the class `LightGbmForecaster`. | 24 |
| Linear interpolation | Filling missing values on a straight line between the known value before the gap and the known value after it. | 17 |
| Linter | A tool that reads code and reports likely mistakes and bad habits. | 11 |
| Liveness / readiness | Two questions that Kubernetes asks a container with automatic requests called probes: is the process alive (if not, it is restarted), and can it answer now (if not, it gets no requests but is not restarted). Here `/health` and `/ready`. | 31 |
| Localize | To attach a timezone to a timestamp that was recorded without one. The clock time stays the same. | 17 |
| Lock file | A list of the exact version of every installed library. | 6 |
| Log scale (of SHAP values) | With the Poisson objective the trees add up to the natural logarithm of the forecast. A contribution is then a factor: +0.69 doubles the forecast, −0.69 halves it. | 28 |
| Log scale (search space) | A range in which every doubling is equally likely to be tried: values from 8 to 16 as often as from 64 to 128. | 27 |
| MAE | Mean absolute error: the mean size of the error, here in pickups per zone and hour. The main metric of this project. | 22 |
| Manifest | The record of every raw file: source address, download time, size, checksum, row count. | 4, 15 |
| Margin (promotion) | The smallest gain in pooled MAE that lets a new version replace the champion. Here 0.2, the measured range of the seeds on the backtest folds, written as the constant `MIN_MAE_GAIN` in the code. | 29 |
| Mask | A column of true and false values, one per row, that marks the rows for which a condition holds. | 16 |
| MD5 | A method for calculating a checksum of 32 characters. DVC uses it to see whether a file changed. | 21 |
| Mean absolute SHAP value | The mean size of an input's contributions over many rows, whatever their direction. Here stored as `shap.mean_abs.<input>`. | 28 |
| Medallion architecture | Storing data in layers of increasing quality: raw (bronze), silver, gold. | 4 |
| Metric | A rule that turns the errors of many forecasts into one number. | 22 |
| Missing value | A value that is not there: it was never recorded, or it was lost. Stored as an empty cell, not as a number. | 15, 16 |
| MLflow | A tool for experiment tracking with a model registry. Here it writes to `mlflow.db` and `mlruns`. | 26 |
| Model card | A short document, written by hand, that says what one model version is for, where it must not be used, how it was evaluated and where it is weak. Here: `docs/model-card.md`. | 29 |
| Model registry | A place that keeps every version of a model with a label. Here: the name `taxi-demand` in MLflow. | 1, 26 |
| Model signature | The names and types of the input columns of a stored model, in order, and the type of its output. MLflow checks every table against it when the model is loaded. The service checks at startup that it names exactly the columns the config builds. | 26, 31 |
| Model version | The number the registry gives a model each time one is registered under the same name: 1, 2, 3. | 26 |
| monkeypatch | A pytest fixture that changes something for one test and restores it afterwards. | 10 |
| Naive timestamp | A date and time without a timezone. | 17 |
| NaT | "Not a time": the missing value for timestamps in pandas. | 17 |
| Noise (of a score) | Movement of a score that comes from chance, such as another seed, and not from a difference in the model. Here about 0.2 pickups of MAE. | 25 |
| Objective | The measure of error that training makes small. Here: `poisson`, which is made for counts. | 24 |
| OpenAPI | A standard format that describes the endpoints, parameters and answers of a web service. FastAPI writes it at `/openapi.json` and shows it at `/docs`. | 30 |
| Optional dependency group | A named set of additional libraries, installed only where needed. | 6 |
| Optuna | A library that searches for hyper-parameters: it proposes values, sees the score, and proposes better ones. Used by `python -m demand tune`. | 27 |
| Outlier flag | A true/false column that marks an hour whose pickups are far from normal for its slot. The hour is kept. Here: `is_outlier`. | 18 |
| Over-fitting to the validation set | Settings chosen on a validation set fit it better than they fit new data, so its score for them is too good. Here about 30% of the in-sample tuning gain did not hold on weeks the search had not seen. | 27 |
| Package | A folder of Python files that can be imported by its name. | 5 |
| Parameter (of a model) | A value that training finds in the data, such as the threshold of a tree question. Compare *hyper-parameter*. | 27 |
| Parametrised test | One test function that runs once for each entry in a list of inputs. | 10 |
| Parquet | A compact file format for tables. | 12 |
| Path (of a request) | The part of an address that says which part of a service should answer, such as `/predict`. | 30 |
| Persist (DVC) | The option `persist: true` of an output. It tells DVC not to delete the output before it runs the stage. | 21 |
| Pickle | Python's own format for saving objects to a file. It can only be loaded where the classes are installed, and only from a trusted source. | 26 |
| Pin | To fix a library to one exact version. | 6 |
| Placeholder | A function that exists by name and only raises `NotImplementedError` until the real one is written. | 9 |
| Placeholder value | A value such as zero, or a copy of another field, that a source writes where it has no measurement. | 16 |
| Pooled MAE | The MAE over all test rows of all folds together. With folds of equal size, as here, it equals the mean of the fold MAEs. | 29 |
| Port | A number that tells a computer which program a network request is for. The service listens on 8000. | 30 |
| Producer / consumer | The step that writes a table / the step that reads it. | 8 |
| Prometheus text format | The plain-text format of `/metrics`: lines `# HELP` and `# TYPE` for each series, then one line per value with its labels. | 31 |
| Promotion gate | The rule that decides whether the newest model version becomes champion, run as `python -m demand promote`. It is not part of `dvc repro`, so rebuilding never changes the served model. | 29 |
| Pruning (of a trial) | Stopping a trial early when its first results look bad. Not used here. | 27 |
| Pure function | A function whose result depends only on its arguments and that reads and writes nothing outside itself. | 16 |
| Push | To send commits to the shared copy of the repository. | 12 |
| Quality gate | A check that can stop a pipeline. A stage runs one on its table before it writes it. The command `validate` runs one on the files of a layer before the next stage reads them. | 8, 20 |
| Query parameter | A pair of a name and a value in the address of a request, after `?` and separated by `&`, such as `date=2025-05-14`. A name may appear more than once. | 30 |
| Random split | Putting rows into training and test sets by chance. Wrong for data ordered in time. | 3 |
| Raw layer | Data exactly as downloaded, never modified. Also called bronze. | 4 |
| Remote (DVC) | A shared storage place to which the DVC cache can be sent, so that other people can fetch the same data. This project has none. | 21 |
| Request / response | The question a client sends to a server / the answer, with a status code and a body. | 30 |
| Reversal | A row with negative amounts that cancels the charge of a trip, for example after a refund. It is not a second trip. | 16 |
| RMSE | Root mean squared error: like the MAE, with large errors counted much more. It is never smaller than the MAE. | 22 |
| Rollback | Making an earlier model version champion again, by hand, because the newer one behaved badly. Here: `python -m demand set-champion VERSION --reason TEXT`. | 29 |
| Rolling statistic | A summary, such as the mean, over a moving window of time. Here every window ends one horizon before the target hour. | 2, 19 |
| Rolling-origin backtest | A backtest repeated with a training end date that moves forward. | 3 |
| Run (MLflow) | One recorded execution of the training, with an id. It holds parameters (settings chosen before), metrics (numbers produced), tags (where it came from) and artifacts (files). | 26 |
| Sampler | The part of Optuna that proposes the next values. Here TPE, seeded with `project.seed`: ten random trials, then more trials near the values that scored well. | 27 |
| Schema | The formal definition of a table: columns, types and rules. | 8 |
| Search space | The ranges in which a search may choose each hyper-parameter. | 27 |
| Seasonal naive forecast | The forecast "the same as one season ago". Here the season is a week. See *baseline*. | 22 |
| Seed | The number from which a computer starts its sequence of random-looking numbers. The same seed gives the same random choices. Here: `project.seed`. | 25 |
| Seed check | Scoring the current and the best hyper-parameters again with two more seeds, to see whether a gain is larger than the noise. | 27 |
| SHA-256 | A method for calculating a checksum of 64 characters. | 15 |
| SHAP value | The contribution of one input to one forecast. The base value plus the SHAP values of all inputs of a row is exactly the output of the model. | 28 |
| Silver layer | Cleaned data: valid rows, fixed types, no duplicates. One row is one valid trip, or one hour of weather. | 4, 16 |
| Simulated clock | Replaying unseen historical data one day at a time, as if it were happening now. | 3 |
| Slot | The group of hours that an hour is compared with when outliers are flagged: the same zone, the same hour of the day and the same kind of day (working day or weekend). | 18 |
| sMAPE | Symmetric mean absolute percentage error: for each hour the error divided by the average of the actual and the forecast, from 0 to 200 percent. Hours with few or no pickups decide it. | 22 |
| Snapshot (data) | A copy of data as it was at one moment, read once and never written. The service reads the gold demand and the silver weather as a snapshot at startup, in place of live feeds the project does not have. | 30 |
| src layout | Keeping the package inside a folder named `src/`, so that it must be installed to be imported. | 5 |
| Stage | One step of the pipeline, such as ingest, silver, gold or train. | 9, 21 |
| Standard deviation | A number that says how far the values of a group typically lie from their mean. | 18 |
| Stateless | Said of a service that writes nothing at request time, so every copy of it can answer any request. | 31 |
| Static analysis | Finding problems by reading code without running it. | 11 |
| Static type checker | A tool that reads type hints and reports contradictions. Here: mypy. | 11 |
| Status code | The number in the answer of a web server that says how the request went, for example 200, 404 or 503. From 400 to 499 the client must change the request; from 500 the server has the problem. 422 means the request was understood but cannot be answered as asked. | 14, 30 |
| stdout / stderr | The two output channels of a program: one for results, one for messages about the run. | 9 |
| Streaming | Receiving or reading a file piece by piece, so that it is never in memory as a whole. | 14 |
| Strict schema | A schema that refuses columns it does not list (`strict=True`). | 8 |
| Structured logging | Writing each log event as a record with named fields. | 9 |
| Tag (model version) | A label that stays with one model version, such as `promotion.decision = promoted`. Compare *alias*, which can move. | 29 |
| Target | The thing a model predicts. Here: the number of pickups. | 2 |
| Test double | A simple object that stands in for a real one during a test. | 10 |
| Time-based split | Training on data before a date and testing on data after it. | 3 |
| Timeout | The longest time a program waits for one operation before it treats it as failed. | 14 |
| Timezone-aware timestamp | A timestamp that carries its timezone, so that it names exactly one moment. | 17 |
| `tmp_path` | A pytest fixture that gives each test a new empty folder. | 10 |
| Traceback | The report of files and line numbers that Python prints when an error is not handled. It is made for finding a mistake in the code. | 20 |
| Training report | The short record of a training run on disk: the MLflow run, the model version, the data version and the headline scores. Here: `reports/training_report.json`. | 26 |
| Training/serving skew | A difference between how features are calculated for training and for prediction. | 4, 19, 31 |
| Transient error | A failure that may go away on its own, such as a short network problem. Worth a retry. | 14 |
| Transitive dependency | A library that is installed because one of your dependencies needs it. | 6 |
| Trial | One set of hyper-parameters scored with the backtest. Trial 0 is the config as it is. | 27 |
| Tuning | Searching for better hyper-parameters by trying them. | 27 |
| Type hint | A note in the code that says which type a value has. | 11 |
| Typed settings object | An object in which every setting has a declared name and type, built once at start-up. | 7 |
| UTC | The world standard time, without summer time. | 9 |
| Validation report | The result of one quality gate: every check with its name, whether it passed, and a detail. Here: `reports/silver_validation_report.json` and `reports/gold_validation_report.json`. | 20 |
| Validator | A function that checks a rule when an object is created and raises an error if the rule is broken. | 7 |
| Virtual environment | A folder with its own Python and its own installed libraries, separate from all others. | 6 |
| Warm-up rows | The first rows of every zone, for which the history is too short to fill all lags and windows. Here 215 hours per zone. They are dropped, not filled. | 19 |
| Web framework | A library in which a web application is written. It maps paths to functions and turns their results into responses. Here: FastAPI. | 30 |
| Web service | A program that starts, prepares itself and then waits for requests over the network, and answers each one. | 30 |
| YAML | A text format for configuration that is easy for people to read. | 7 |
| Z-score | The distance of a value from the mean of its group, counted in standard deviations. | 18 |

---

[Previous: Appendix A – Stack map](appendix-a-stack-map.md) | [Contents](README.md) | [Next: Appendix C – Further reading](appendix-c-further-reading.md)
