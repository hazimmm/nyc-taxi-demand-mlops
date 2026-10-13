# Appendix B – Glossary

Every technical term used in the book, in alphabetical order. The number in the last column is the chapter where the term is explained.

| Term | Meaning | Ch. |
|---|---|---|
| Actuals | The real observed values, here the real pickups, known only after the hour has passed. | 3 |
| Atomic | Said of an operation that happens completely or not at all, with no state in between. Renaming a file inside one folder is atomic. | 14 |
| Automated test | A small function that runs a piece of code and checks the result with `assert`. | 10 |
| Backtest | Testing a forecasting model on the past as if it had been used then. | 3 |
| Baseline | A very simple method that sets the minimum a model must beat. Here: "the same hour last week". | 1 |
| Binary file | A file that is not text. Git must not change any byte of it. | 12 |
| Branch (git) | A named line of commits. Work happens on `dev`. | 12 |
| Branch coverage | Coverage that also requires both outcomes of every `if` to be executed. | 10 |
| Build backend | The tool that turns a source folder into something installable. Here: hatchling. | 5 |
| Champion / challenger | The model in use / a new model that must beat it to replace it. | 1 |
| Cleaning report | The record of a cleaning run: rows in, rows removed by each rule, rows out. Here: `reports/silver_cleaning_report.json`. | 16 |
| Cleaning rule | A condition that marks a row as not valid, with a name and a fixed place in the order of the rules. | 16 |
| Checksum | A short code calculated from the content of a file. If the content changes, the code changes. | 4, 15 |
| Chunk | One piece of a file that is read or received in pieces. | 14 |
| CI | Continuous integration: automatic checks that run on every push. Planned for Phase 7. | 13 |
| CLI | Command-line interface: a program controlled by typed commands. | 9 |
| Code review | Reading a change before it is accepted, to find mistakes and to check that code, tests and documents agree. | 13 |
| Configuration | The values that can change without the logic changing: dates, paths, limits, addresses. | 7 |
| Constant | A name for a value that never changes, written in capitals. | 8 |
| Conventional commits | A format for commit messages: `type: description`. | 12 |
| Coverage | The share of the code that the tests execute. | 10 |
| Coverage gate | A rule that fails the test run when coverage is under a limit. Here 80%, applied with `--cov`. | 10 |
| CRLF / LF | The two ways to mark the end of a line: Windows (two characters) and Linux or macOS (one). | 12 |
| Data contract | A precise description of a table that can be checked automatically and that producer and consumer both rely on. | 8 |
| Data leakage | Training or testing with information that is not available when the model is really used. | 2 |
| Daylight saving time | The summer clock, one hour ahead of the winter clock. The change skips one local hour in spring and repeats one in autumn. | 17 |
| Day-ahead forecast | A forecast made at least 24 hours before the hour it describes. | 1 |
| Decorator | A marker above a function, starting with `@`, that changes how the function is used. | 9 |
| Dependency | A library that the code needs in order to run. | 6 |
| Dependency injection | Giving a function the things it depends on, such as a network client, as arguments, so that a test can pass different ones. | 14 |
| Docstring | A text at the start of a file, class or function that documents it. | 5 |
| Drift | A change in the data compared with the training period. | 3 |
| Editable install | An install that points to the source folder, so edits are active without installing again. | 5 |
| Entry point | The place where a program starts. Here: `python -m demand`. | 9 |
| `.env` file | A text file with lines of the form `NAME=value`, read into environment variables when the program starts. Not stored in git. | 7 |
| Environment variable | A named value outside the program, set by the shell or the platform and read at start-up. | 7 |
| Exit code | The number a program returns when it ends. Zero is success. | 9 |
| Exponential backoff | Waiting twice as long after every failed attempt before trying again. | 14 |
| Extra | See *optional dependency group*. | 6 |
| Fail closed / fail open | What a check does when it cannot do its job. Fail closed: stop with an error. Fail open: let everything pass. | 8 |
| Feature | An input column of a model. | 1 |
| Fixture | A function that prepares something for tests. A test asks for it by parameter name. | 10 |
| Fold | One repetition of a backtest, with its own training and test period. | 3 |
| Footer (Parquet) | A small block at the end of a Parquet file that describes its columns, types and number of rows. | 14 |
| Forecast | A prediction about a future time, made from past data. | 1 |
| Forecast horizon | The time between making a forecast and the moment it describes. Here 24 hours. | 1 |
| Formatter | A tool that rewrites the layout of code in one fixed style. | 11 |
| Frozen | Said of an object that cannot be changed after it is created. | 7 |
| Git hook | A script that git runs automatically at a certain moment, for example before a commit. | 11 |
| Gold layer | Data in the form its users need: one row per zone and hour, and the table of model inputs. | 4 |
| Ground truth | The real value that a prediction tried to match. | 3 |
| Hard-coded | Said of a value written directly in the code. | 7 |
| Idempotent | Said of an operation that gives the same result when it is run again. | 15, 17 |
| Immutable | Cannot be modified after it is written or created. | 4 |
| Imputation | Filling a missing value with a calculated one, for example the average. | 16 |
| Import path | The list of folders where Python searches for code to import. | 5 |
| Ingest | To bring data from outside sources into the project. The first stage of the pipeline. | 14 |
| JSON | A standard text format for structured data. | 9 |
| Lag feature | A past value of the target used as an input. | 2 |
| Lazy import | Importing a module at the moment it is needed, not at the top of the file. | 9 |
| Linear interpolation | Filling missing values on a straight line between the known value before the gap and the known value after it. | 17 |
| Linter | A tool that reads code and reports likely mistakes and bad habits. | 11 |
| Localize | To attach a timezone to a timestamp that was recorded without one. The clock time stays the same. | 17 |
| Lock file | A list of the exact version of every installed library. | 6 |
| Manifest | The record of every raw file: source address, download time, size, checksum, row count. | 4, 15 |
| Mask | A column of true and false values, one per row, that marks the rows for which a condition holds. | 16 |
| Medallion architecture | Storing data in layers of increasing quality: raw (bronze), silver, gold. | 4 |
| Missing value | A value that is not there: it was never recorded, or it was lost. Stored as an empty cell, not as a number. | 15, 16 |
| Model registry | A place that keeps every version of a model with a label. | 1 |
| monkeypatch | A pytest fixture that changes something for one test and restores it afterwards. | 10 |
| Naive timestamp | A date and time without a timezone. | 17 |
| NaT | "Not a time": the missing value for timestamps in pandas. | 17 |
| Optional dependency group | A named set of additional libraries, installed only where needed. | 6 |
| Package | A folder of Python files that can be imported by its name. | 5 |
| Parametrised test | One test function that runs once for each entry in a list of inputs. | 10 |
| Parquet | A compact file format for tables. | 12 |
| Pin | To fix a library to one exact version. | 6 |
| Placeholder | A function that exists by name and only raises `NotImplementedError` until the real one is written. | 9 |
| Placeholder value | A value such as zero, or a copy of another field, that a source writes where it has no measurement. | 16 |
| Producer / consumer | The step that writes a table / the step that reads it. | 8 |
| Pure function | A function whose result depends only on its arguments and that reads and writes nothing outside itself. | 16 |
| Push | To send commits to the shared copy of the repository. | 12 |
| Quality gate | A check that can stop a pipeline. | 8 |
| Random split | Putting rows into training and test sets by chance. Wrong for data ordered in time. | 3 |
| Raw layer | Data exactly as downloaded, never modified. Also called bronze. | 4 |
| Reversal | A row with negative amounts that cancels the charge of a trip, for example after a refund. It is not a second trip. | 16 |
| Rolling statistic | A summary, such as the mean, over a moving window of time. | 2 |
| Rolling-origin backtest | A backtest repeated with a training end date that moves forward. | 3 |
| Schema | The formal definition of a table: columns, types and rules. | 8 |
| SHA-256 | A method for calculating a checksum of 64 characters. | 15 |
| Silver layer | Cleaned data: valid rows, fixed types, no duplicates. One row is one valid trip, or one hour of weather. | 4, 16 |
| Simulated clock | Replaying unseen historical data one day at a time, as if it were happening now. | 3 |
| src layout | Keeping the package inside a folder named `src/`, so that it must be installed to be imported. | 5 |
| Stage | One step of the pipeline, such as ingest, silver, gold or train. | 9 |
| Static analysis | Finding problems by reading code without running it. | 11 |
| Static type checker | A tool that reads type hints and reports contradictions. Here: mypy. | 11 |
| Status code | The number in the answer of a web server that says how the request went, for example 200, 404 or 503. | 14 |
| stdout / stderr | The two output channels of a program: one for results, one for messages about the run. | 9 |
| Streaming | Receiving or reading a file piece by piece, so that it is never in memory as a whole. | 14 |
| Strict schema | A schema that refuses columns it does not list (`strict=True`). | 8 |
| Structured logging | Writing each log event as a record with named fields. | 9 |
| Target | The thing a model predicts. Here: the number of pickups. | 2 |
| Test double | A simple object that stands in for a real one during a test. | 10 |
| Time-based split | Training on data before a date and testing on data after it. | 3 |
| Timeout | The longest time a program waits for one operation before it treats it as failed. | 14 |
| Timezone-aware timestamp | A timestamp that carries its timezone, so that it names exactly one moment. | 17 |
| `tmp_path` | A pytest fixture that gives each test a new empty folder. | 10 |
| Training/serving skew | A difference between how features are calculated for training and for prediction. | 4 |
| Transient error | A failure that may go away on its own, such as a short network problem. Worth a retry. | 14 |
| Transitive dependency | A library that is installed because one of your dependencies needs it. | 6 |
| Type hint | A note in the code that says which type a value has. | 11 |
| Typed settings object | An object in which every setting has a declared name and type, built once at start-up. | 7 |
| UTC | The world standard time, without summer time. | 9 |
| Validator | A function that checks a rule when an object is created and raises an error if the rule is broken. | 7 |
| Virtual environment | A folder with its own Python and its own installed libraries, separate from all others. | 6 |
| YAML | A text format for configuration that is easy for people to read. | 7 |

---

[Previous: Appendix A – Stack map](appendix-a-stack-map.md) | [Contents](README.md) | [Next: Appendix C – Further reading](appendix-c-further-reading.md)
