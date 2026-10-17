# Chapter 20 – Quality gates: checking the files on disk

The file `data/silver/trips/trips_2025-01.parquet` holds 3,341,755 trips. The silver stage checked this table against its contract before it wrote the file. Suppose that next week somebody lowers `max_distance_miles` in the config from 100 to 80 and runs `python -m demand gold`. The file still holds 31 trips longer than 80 miles. The gold stage reads the file as it is, and nothing complains, because the only check of this table happened in the past, under the old limit. This chapter builds the check that runs today, on the file, before the next stage reads it.

## What you will learn

- Why a table that was checked before writing must be checked again before reading.
- What the schema checks, and what lazy validation is.
- Which checks a schema cannot express, with three real examples.
- Why a check that could not run is reported as failed.
- Why the gate does not calculate the whole layer a second time.
- How the program separates an expected failure from a bug.

## A check before writing is not a check before reading

Chapter 8 gave the rule for the producer: check the table against its contract, then write it. The silver and gold stages do this (Chapters 17 and 18). That check looks at a table in memory, at one moment, with the config of that moment.

The next stage reads a file, and it reads it later. Between the two moments several things can happen. The config changes, as in the opening. A file is copied in from another machine. A file is damaged or deleted. The silver stage stops at the fourth month, and three new files lie next to an old cleaning report (Chapter 17 named this limit).

Chapter 8 defined a **quality gate** as a check that can stop a pipeline. The checks inside the stages are gates of this kind. This chapter adds one more per layer, as a command of its own: `python -m demand validate silver` and `python -m demand validate gold`. From here on, "the gate" means this command. A gate reads the files from disk, because the files are what the next stage will read. It changes no data. It writes a report, and if one check fails, the command ends with exit code 1.

| | Silver gate | Gold gate |
|---|---|---|
| Files it reads | 6 trip files, the weather file, the cleaning report | the demand table and the feature table |
| Rows | 23,197,032 trips, one month at a time | 86,860 and 82,560 |
| Checks | 37 | 10 |
| Time | about 4 seconds | under 1 second |

*Figure 20.1 – The two gates, measured on the real data.*

All of the code is in one file, [`src/demand/data/validate.py`](../../src/demand/data/validate.py). It has two kinds of checks.

## The first kind: the schema, with every failure at once

Every table is first checked against its Pandera schema from Chapter 8. The schema is built from the settings of today, so the 31 long trips of the opening would fail the rule `in_range(0, 80)`.

The stages call `schema.validate(table)`. In that form Pandera stops at the first broken rule. This is enough for a stage, which only has to refuse to write. A gate has a different reader: a person who must decide what to repair. That person needs the complete list.

```python
    try:
        schema.validate(table, lazy=True)
    except pandera.errors.SchemaErrors as error:
        cases = error.failure_cases
```

This is the start of `find_schema_problems`. **Lazy validation** means that the schema runs every rule and reports all failures together. `error.failure_cases` is a table with one row per failing value. The lines that follow group it by column and rule, and build one short text.

A test shows the result. It writes a February file in which two trips have an invalid zone and one has a negative amount. The detail of the failed check is:

```text
pickup_zone_id: in_range(1, 263) (2 failing, e.g. 264.0, 300.0); total_amount: greater_than_or_equal_to(0) (1 failing, e.g. -1.0)
```

Both problems are visible after one run. Without `lazy=True`, the amount would appear only after the zones were repaired.

## The second kind: what a schema cannot say

A schema judges one table, and it judges the values that are there. Some rules need more: the config, a second file, or a second table. Here is one example of each.

**A rule that needs the config.** The demand table must be a complete grid (Chapter 18): 20 zones times 4,343 hours. The schema can say that no pair of zone and hour appears twice. It cannot say that a pair is absent, because an absent row has no value to judge. The check `demand.complete_grid` builds the expected pairs from the config, with the same function `grid_index` that the gold stage uses, and compares them with the pairs in the file.

**A rule that needs a second file.** The cleaning report of Chapter 16 says that January has 3,341,755 rows after cleaning. The check `trips_2025-01.counts_match_report` counts the rows of the file and compares. A different number means that the file and the report come from different runs.

**A rule that needs a second table.** Every lag in the feature table must be the pickups of the same zone, that many real hours earlier. The docstring calls a wrong lag "the most damaging mistake a forecasting table can contain": the table looks normal and the model learns from the wrong hour.

```python
    pickups = demand.set_index(KEYS)[TARGET]
    parts = []
    for lag in lags:
        earlier = pd.MultiIndex.from_arrays(
            [features[ZONE_ID], features[HOUR_START] - pd.Timedelta(hours=lag)]
        )
        # An earlier hour the demand table lacks gives NaN, which equals nothing.
        expected = pickups.reindex(earlier).to_numpy(dtype="float64")
        wrong = features[lag_column(lag)].to_numpy() != expected
```

This is the centre of `find_wrong_lags`. The first line turns the demand table into a lookup: zone and hour go in, pickups come out. For each lag, `earlier` holds the zone of every feature row and its hour minus the lag. Subtracting a `Timedelta` from a timezone-aware time moves by elapsed hours, also across a clock change. `reindex` looks up all these pairs at once. The last line compares what the feature table stores with what was found.

Notice that this is not the code that built the lags. The feature function moves values down by a number of rows (Chapter 19). The check looks each value up by its time. Two different methods must give the same number.

## A check that could not run has not passed

The checks of the second kind rely on the schema. `find_wrong_lags` assumes that the column `lag_48h` exists and holds numbers. If the file is missing, or the table failed its schema, these checks cannot run. What should the report say about them?

```python
    if subject is None:
        return [to_result(name, NOT_CHECKED) for name in checks]
    return [to_result(name, find(subject)) for name, find in checks.items()]
```

This is the body of `run_checks`. `subject` is the table, or `None` when it could not be loaded. In that case every check is listed as failed, with the detail `not checked: the file is missing or the table did not pass its schema`.

This is "fail closed" from Chapter 8, applied to a report. Leaving the checks out would make the list shorter and every remaining line green. A reader would have to know which lines are absent. The one-line message of a failed gate still separates cause from consequence: it names the checks that found something, and only counts those that could not run. You will see this below.

The report is a JSON file in `reports/`. It lists every check with its name, `passed`, and a detail. It contains no time, so the same files and the same config always give the same bytes. Chapter 21 needs this property.

## What the gate does not check

Two checks were considered for the gold gate and left out.

The first was to count the pickups again from the 23 million silver trips and compare them with the demand table. That is the gold stage itself. The gate would take as long as the stage. If it called the same counting function, it would repeat any mistake in it. If it had its own counting code, the project would have two versions of one rule to keep equal.

The second was to calculate the rolling statistics again. The only honest way is to call `build_features`, and a function that is compared with its own output always agrees.

The rule behind both decisions: a check is worth having when it is much cheaper and much simpler than the thing it checks. The lag check is one lookup, and you can see that it is right by reading it. A second copy of the feature code would be neither.

So the gate does not prove that a rolling mean is correct. The schema only says that it is there and not negative. The correctness of the feature function is the job of the tests in Chapter 19.

## Expected failures and bugs

Until now, a missing silver file ended the program with a **traceback**: the long report of files and line numbers that Python prints when an error is not handled. A traceback is made for a programmer who must find a mistake in the code. A missing file is not a mistake in the code. The person at the terminal needs two things: what happened, and what to do next.

[`src/demand/errors.py`](../../src/demand/errors.py)

```python
class PipelineError(Exception):
    """An expected failure of a pipeline stage. The message says what to do next."""
```

Every failure that a person can act on now inherits from this class: a missing file, an empty zone list, a failed download, a failed gate. `_run_stage` in `cli.py` got one more branch:

```python
    except PipelineError as error:
        # An expected failure: the message is enough, a traceback would only hide it.
        typer.secho(f"Error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
```

It prints the message as one line on stderr and ends with exit code 1. Any other exception, such as a `KeyError`, is a bug and keeps its traceback. Hiding it would remove the information needed to repair it.

One line is only enough if the line is complete. So every message of this kind ends with the next step:

```text
data.zones is empty. Run `python -m demand select-zones` and paste the result into configs/config.yaml before building or checking the gold layer.
```

The older error classes kept their first parent. `MissingSilverFileError` now inherits from `PipelineError` and from `FileNotFoundError`, so code that catches the built-in class still works.

## Try it yourself

Run both gates on the real data. They only read it.

```powershell
.venv\Scripts\python.exe -m demand validate silver
.venv\Scripts\python.exe -m demand validate gold
```

After about four seconds you should see two lines like these:

```text
{"time": "2026-10-13T14:56:25.628339+00:00", "level": "INFO", "logger": "demand.data.validate", "message": "quality gate passed", "layer": "silver", "checks": 37, "report": "reports\\silver_validation_report.json", "seconds": 4.02}
{"time": "2026-10-13T14:56:27.148133+00:00", "level": "INFO", "logger": "demand.data.validate", "message": "quality gate passed", "layer": "gold", "checks": 10, "report": "reports\\gold_validation_report.json", "seconds": 0.2}
```

Now see a gate fail. Never damage the real `data` folder for this. Copy the gold layer to a temporary folder, and point the program at the copy with two environment variables.

```powershell
$demo = "$env:TEMP\gate-demo"
New-Item -ItemType Directory -Force "$demo\data" | Out-Null
Copy-Item -Recurse data\gold "$demo\data\gold"
$env:DEMAND_PATHS__DATA_DIR = "$demo\data"
$env:DEMAND_PATHS__REPORTS_DIR = "$demo\reports"
.venv\Scripts\python.exe -m demand validate gold
```

You should see `quality gate passed` again, this time with a report path inside your temporary folder. Add 1 to one lag in the copy: the `lag_48h` of the first row, which Figure 19.1 showed as 72.

```powershell
.venv\Scripts\python.exe -c "import os, pandas as pd; p = os.environ['DEMAND_PATHS__DATA_DIR'] + '/gold/features.parquet'; f = pd.read_parquet(p); f.loc[0, 'lag_48h'] += 1; f.to_parquet(p, index=False)"
.venv\Scripts\python.exe -m demand validate gold
$LASTEXITCODE
```

You should see one line and then `1`. The path in the middle is your own temporary folder and is shortened here:

```text
Error: gold quality gate failed: 1 of 10 checks failed (features.lags_match_demand). The details are in C:/.../gate-demo/reports/gold_validation_report.json. Fix the cause, then run `python -m demand gold` and the gate again.
```

Print the failed checks from the report.

```powershell
.venv\Scripts\python.exe -c "import json, os; r = json.load(open(os.environ['DEMAND_PATHS__REPORTS_DIR'] + '/gold_validation_report.json')); print(r['passed']); [print(c['name'], '|', c['detail']) for c in r['checks'] if not c['passed']]"
```

You should see:

```text
False
features.lags_match_demand | lag_48h: 1 row not equal to the pickups 48 h earlier, e.g. 48 / 2025-01-09 23:00:00-05:00
```

One value among 82,560 rows was changed by 1, and the report names the zone and the hour. Now delete the demand table of the copy and run the gate again.

```powershell
Remove-Item "$demo\data\gold\demand_hourly.parquet"
.venv\Scripts\python.exe -m demand validate gold
```

The line now begins with:

```text
Error: gold quality gate failed: 1 of 10 checks failed (demand.file), and 5 more could not be checked.
```

The lag check is one of the five. The wrong lag is still in the file, but the gate cannot see it without the demand table, and it does not report that check as passed. Remove the variables and the copy.

```powershell
Remove-Item Env:DEMAND_PATHS__DATA_DIR
Remove-Item Env:DEMAND_PATHS__REPORTS_DIR
Remove-Item -Recurse $demo
```

See an expected failure of a stage. The variable points the program at a folder that does not exist, so nothing can be written.

```powershell
$env:DEMAND_PATHS__DATA_DIR = "no-such-folder"
.venv\Scripts\python.exe -m demand gold
$LASTEXITCODE
Remove-Item Env:DEMAND_PATHS__DATA_DIR
```

You should see one line and then `1`:

```text
Error: Silver files missing in no-such-folder\silver: trips/trips_2025-01.parquet, trips/trips_2025-02.parquet, trips/trips_2025-03.parquet, trips/trips_2025-04.parquet, trips/trips_2025-05.parquet, trips/trips_2025-06.parquet, weather/weather_hourly.parquet. Run `python -m demand silver` first.
```

Run the tests of the gates, of the error classes, and of the two kinds of failure on the command line.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/data/test_validate.py -q
.venv\Scripts\python.exe -m pytest tests/unit/data/test_errors.py -q
.venv\Scripts\python.exe -m pytest tests/unit/test_cli.py -q -k "expected_failure or unexpected_failure"
```

You should see `47 passed`, then `14 passed`, then `2 passed, 24 deselected`.

## Key takeaways

- A stage checks a table in memory before writing. A gate checks the files on disk before the next stage reads them, with the config of today.
- Lazy validation reports every broken rule of a schema in one run.
- A schema cannot see a missing row, a second file or a second table. Those rules need their own checks.
- A check that could not run is reported as failed, with the reason.
- A check must be much cheaper and simpler than what it checks. The gate does not build the layer a second time.
- An expected failure gets one line that says what to do next. A bug keeps its traceback.

## Exercises

**1.** After a schema failure in the demand table, the report lists `features.lags_match_demand` as failed with the detail "not checked". A colleague wants to remove such lines from the report, "because they are noise". What do you answer?

<details>
<summary>Answer</summary>

Without the lines, the report shows fewer checks and no sign that some were skipped. A person who reads it, or a program that counts the passed checks, would have to know the full list by heart to notice. With the lines, the report always has the same 10 entries, and every entry has an honest state. The noise is already handled in the right place: the one-line message names only the checks that found something and counts the others.

</details>

**2.** Someone proposes a new gold check: "Calculate `roll_mean_24h` again with `build_features` and compare it with the file." What would this check prove, and what would it not prove?

<details>
<summary>Answer</summary>

It would prove that the file still holds what `build_features` produces today, so it would notice a file that was edited by hand or built by an older version of the code. It would not prove that the rolling mean is correct: if the function has a mistake, the file and the new calculation have the same mistake and agree. The correctness of the function is proven by the tests of Chapter 19, which use small tables with known answers. The check would also make the gate about as slow as the stage.

</details>

**3. Interview question.** "Your pipeline validates every table before it writes it. Why would you validate again between the stages?"

<details>
<summary>Points of a good answer</summary>

- The first check sees a table in memory. The next stage reads a file, later. Files are replaced, damaged, deleted, or left over from an older config.
- The gate builds its schema from the current config, so it also notices data that was correct under old settings.
- It adds rules a schema cannot express: completeness against the config, agreement between a file and its report, agreement between two tables.
- It reports all failures at once, lists checks that could not run as failed, and ends with a non-zero exit code so that the program that runs the pipeline stops.
- It stays much cheaper than the stage, and does not repeat the stage's own logic.

</details>

---

[Previous: Chapter 19 – Features without leakage](19-features-without-leakage.md) | [Contents](README.md) | [Next: Chapter 21 – A pipeline that knows what changed](21-a-pipeline-that-knows-what-changed.md)
