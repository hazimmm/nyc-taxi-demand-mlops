# Chapter 22 – A number to beat: the baseline and three metrics

On Tuesday 8 April 2025 at 18:00, Midtown Center (zone 161) had 739 pickups. One week earlier, at the same hour, it had 694. Suppose 694 had been used as the forecast. It needs no model, no training and no weather, and it is wrong by 45 pickups. Is 45 a large error? And if a model is wrong by 40, is the model worth building? This chapter gives the two things needed to answer: a simple forecast to compare with, and three ways to turn many errors into one number.

## What you will learn

- What a baseline is, and why it is measured before any model exists.
- Why the baseline of this project is "the same hour one week ago", and what happens to "the same hour yesterday".
- How a forecaster is described in the code, so that a baseline and a model are scored in the same way.
- What MAE, RMSE and sMAPE are, with one example calculated by hand.
- What each metric is good at and bad at, and how the weaknesses of sMAPE show in the real data.
- How to read an MAE next to the mean demand.

## Why the baseline comes first

A **baseline** is a very simple forecasting method that sets the minimum a model must beat (Chapter 1). There are three reasons to build it before the model.

An error has no meaning alone. "The model is wrong by 26 pickups on average" is good if the simple method is wrong by 40, and worthless if the simple method is wrong by 25.

The order protects the comparison. A baseline that is chosen after the model exists can be chosen to make the model look good. A baseline that is measured first cannot.

A baseline also finds mistakes. A model with many inputs that cannot beat a copy of last week usually has a mistake in its data, not a weakness in its method.

## The baseline of this project

Taxi demand repeats by the hour of the day and by the day of the week. The forecast "the same zone, the same hour, one week ago" gets both right. In this project it is called `weekly_naive`. Textbooks call it the seasonal naive forecast.

It needs no new calculation. The value 168 hours earlier is already a column of the feature table: `lag_168h` (Chapter 19). So the baseline is a copy of one feature, and it inherits the rule of that feature: it reads nothing newer than the horizon allows.

[`src/demand/models/baseline.py`](../../src/demand/models/baseline.py)

```python
class LagBaseline:
    """Forecast for every hour the pickups of the same zone ``lag_hours`` earlier."""

    def __init__(self, lag_hours: int) -> None:
        self.column = lag_column(lag_hours)

    def fit(self, train: pd.DataFrame) -> None:
        """Nothing to learn: the forecast is a copy of a feature."""

    def predict(self, rows: pd.DataFrame) -> FloatArray:
        """The lag column of ``rows``, unchanged."""
        forecast: FloatArray = rows[self.column].to_numpy(dtype="float64")
        return forecast
```

The class has two methods. `fit` receives past rows and does nothing with them. `predict` receives the rows to forecast and returns one column of them. To **fit** a forecaster means to let it learn from past rows. This one learns nothing, and it still has the method.

The reason is in the same file. A **forecaster** is defined there as anything that has these two methods, `fit` and `predict`. The code that scores forecasts (Chapter 23) knows only the two methods. A LightGBM model will learn in `fit` and calculate in `predict`. It will be scored by the same code, on the same rows, without any change to that code.

## Why not a more recent hour?

A week is a long time. A copy of a more recent hour looks like a better idea. The table shows the mean error of three copies on the same hours, the four test weeks of Chapter 23.

| The forecast is the pickups of the same zone | Allowed with a horizon of 48 hours | Mean error in pickups |
|---|---|---|
| 24 hours earlier | no | 41.32 |
| 48 hours earlier (`lag_48h`) | yes | 60.30 |
| 168 hours earlier (`lag_168h`) | yes | 27.70 |

*Figure 22.1 – Three copies. Only the last one is the same day of the week.*

Newer information does not give a better forecast here. Both recent copies are much worse than the copy of last week. Two days before a Tuesday is a Sunday, and two days before a Monday is a Saturday. One day before a Monday is a Sunday. A copy of a recent hour often copies the wrong kind of day. The copy of last week never does.

The first row is "the same hour yesterday". The code calls it `daily_naive` and offers it only when 24 is among the configured lags. With the horizon of 48 hours the config refuses that lag (Chapter 2), so this baseline does not exist in the real run. The tests build it with settings of their own, which have a horizon of 24 hours.

## From many errors to one number

Four test weeks of 20 zones are 13,440 hours, so there are 13,440 errors. A **metric** is a rule that turns them into one number. No single rule is fair to every hour, so the project reports three. All of them are in [`src/demand/models/metrics.py`](../../src/demand/models/metrics.py).

Take four real hours: LaGuardia Airport (zone 138) on Friday 4 April, from midnight. "Actual" is the number of pickups that happened. "Forecast" is the weekly baseline.

| Hour | Actual | Forecast | Absolute error | Squared error | Error as a share |
|---|---|---|---|---|---|
| 00:00 | 135 | 69 | 66 | 4,356 | 66 / 102 = 64.7% |
| 01:00 | 7 | 0 | 7 | 49 | 7 / 3.5 = 200% |
| 02:00 | 0 | 2 | 2 | 4 | 2 / 1 = 200% |
| 03:00 | 0 | 0 | 0 | 0 | 0% |

*Figure 22.2 – Four hours and the three ways to count their errors.*

**MAE**, the mean absolute error, is the mean of the fourth column: (66 + 7 + 2 + 0) / 4 = 18.75 pickups. It can be said in one sentence: "on average the forecast is off by 19 pickups per zone and hour". Every pickup of error counts the same. Its weakness is that busy hours decide it. The hour at 00:00 gives 66 of the 75 pickups of error here, and the three quiet hours hardly matter.

**RMSE**, the root mean squared error, takes the mean of the fifth column and then the square root: √(4,409 / 4) = 33.20 pickups. Squaring makes large errors count much more, so the RMSE is never smaller than the MAE. The distance between the two is the useful part. A large distance means that the error sits in a few very bad hours and is not spread evenly. The weakness is the same property: a few hours can decide the whole number.

**sMAPE**, the symmetric mean absolute percentage error, is the mean of the last column: (64.7 + 200 + 200 + 0) / 4 = 116.2%. For one hour it divides the absolute error by the average of the actual and the forecast. The result is a percentage between 0 and 200, so a quiet zone and a busy zone weigh the same.

## The weaknesses of sMAPE

The last column of Figure 22.2 shows three of them.

- **Zero against anything is the maximum.** At 02:00 the forecast was 2 and nothing happened: 200%. A forecast of 1,000 would also score 200%.
- **Small counts are loud.** At 01:00 the forecast missed 7 pickups and scores 200%. At 00:00 it missed 66 and scores 65%.
- **Zero against zero is not defined.** At 03:00 the formula gives 0 / 0. The code counts such an hour as 0% and keeps it in the mean. The forecast said "no pickups" and there were none, which is a perfect forecast.

```python
    error = np.abs(forecast_values - actual_values)
    scale = (np.abs(actual_values) + np.abs(forecast_values)) / 2
    # Where the scale is 0 both values are 0, so the error is 0 too: a perfect hour.
    share = np.divide(error, scale, out=np.zeros_like(error), where=scale > 0)
    return float(share.mean() * 100)
```

`np.divide` with `where=scale > 0` divides only where the scale is not zero. Everywhere else it leaves the value of `out`, which is 0.

A fourth weakness is in the name. sMAPE is not symmetric. For an actual of 100, a forecast of 50 scores 66.7% and a forecast of 150 scores 40%. Both are wrong by 50 pickups. A forecaster that is too high looks better than one that is too low.

These weaknesses are visible in the real result. Over the four test weeks, the weekly baseline has a sMAPE between 17.8% and 30.5% in 19 of the 20 zones. In LaGuardia it has 44.1%. The forecast is not twice as bad there. LaGuardia has 65 test hours without any pickup and 147 with fewer than ten, all of them between 01:00 and 08:00. In 58 of its 672 test hours the sMAPE is exactly 200%.

So the sMAPE is reported as a second opinion, and no decision of this project rests on it. The MAE is the main number.

## An MAE next to the mean demand

An MAE in pickups is hard to judge without the size of the demand. A reading aid helps: divide the MAE by the mean number of pickups of the same rows.

For LaGuardia the MAE is 31.7 and the mean demand is 144.2 pickups per hour, which gives 22.0%. The other zones lie between 13.2% and 23.3%. By this measure LaGuardia is at the weak end and inside the range. The sMAPE put it far outside.

The difference is the order of the two steps. sMAPE divides every hour and then takes the mean, so an hour with one pickup weighs as much as an hour with 700. The reading aid takes the two means first and divides once, so every pickup weighs the same.

For all 20 zones together the weekly baseline has an MAE of 27.70 against a mean of 156.4 pickups per zone and hour: 17.7%. This percentage is not in the report. It is calculated here to make the MAE easier to read.

## A forecast that is missing

One decision in `metrics.py` is easy to overlook. A metric refuses its input if a forecast is missing or infinite. It does not skip the row. A forecaster that returns no number for an hour has failed for that hour, and a mean over the remaining hours would hide it.

## Try it yourself

Calculate the three metrics of Figure 22.2.

```powershell
.venv\Scripts\python.exe -c "from demand.models.metrics import evaluate; print(evaluate([135, 7, 0, 0], [69, 0, 2, 0]))"
```

You should see `mae=18.75 rmse=33.20015060206806 smape=116.17647058823528`.

See that sMAPE is not symmetric, and that a forecast of 1 and a forecast of 1,000 score the same when nothing happened.

```powershell
.venv\Scripts\python.exe -c "from demand.models.metrics import smape; print(round(smape([100], [50]), 2), round(smape([100], [150]), 2), smape([0], [1]), smape([0], [1000]))"
```

You should see `66.67 40.0 200.0 200.0`.

Ask which baselines exist with the real config, and then with a horizon of 24 hours.

```powershell
.venv\Scripts\python.exe -c "from demand.config import get_settings; from demand.models.baseline import baselines; print(list(baselines(get_settings())))"
$env:DEMAND_FEATURES__HORIZON_HOURS = "24"
$env:DEMAND_FEATURES__LAGS = "[24, 168]"
.venv\Scripts\python.exe -c "from demand.config import get_settings; from demand.models.baseline import baselines; print(list(baselines(get_settings())))"
Remove-Item Env:DEMAND_FEATURES__HORIZON_HOURS
Remove-Item Env:DEMAND_FEATURES__LAGS
```

You should see `['weekly_naive']` and then `['weekly_naive', 'daily_naive']`. The last two lines remove the variables again.

Calculate the three numbers of Figure 22.1 from the demand table. The command reads only the hours before 1 May, and it needs the gold layer of Chapter 18.

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; tz = 'America/New_York'; end = pd.Timestamp('2025-05-01', tz=tz); d = pd.read_parquet('data/gold/demand_hourly.parquet', filters=[('hour_start', '<', end)]); p = d.set_index(['zone_id', 'hour_start']).pickups; test = d[d.hour_start >= pd.Timestamp('2025-04-03', tz=tz)]; [print(k, round((p.reindex(pd.MultiIndex.from_arrays([test.zone_id, test.hour_start - pd.Timedelta(hours=k)])).to_numpy() - test.pickups.to_numpy()).__abs__().mean(), 2)) for k in (24, 48, 168)]"
```

You should see:

```text
24 41.32
48 60.3
168 27.7
```

Read the scores of three zones from the report of Chapter 23. If the file `reports\backtest_report.json` does not exist, run `.venv\Scripts\python.exe -m demand backtest` first.

```powershell
.venv\Scripts\python.exe -c "import json; z = json.load(open('reports/backtest_report.json'))['forecasters']['weekly_naive']['per_zone']; [print(k, z[k]['mae'], z[k]['smape']) for k in ('138', '161', '237')]"
```

You should see:

```text
138 31.705357 44.053747
161 31.97619 21.450703
237 38.691964 25.293538
```

LaGuardia (138) and Midtown Center (161) have almost the same MAE, and the sMAPE of LaGuardia is twice as high.

Run the tests of the metrics and the baselines.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/models/test_metrics.py tests/unit/models/test_baseline.py -q
```

You should see `35 passed`.

## Key takeaways

- A baseline gives an error its meaning. It is measured before the model, so that it cannot be chosen to favour the model.
- The baseline here is the feature `lag_168h`: the same zone and hour one week ago. Nothing is fitted.
- A copy of a more recent hour is worse, because it is often a different kind of day.
- A baseline and a model have the same two methods, `fit` and `predict`, and are scored by the same code.
- MAE is in pickups and easy to say. Busy hours decide it.
- RMSE counts large errors more. Its distance from the MAE shows whether the error sits in a few bad hours.
- sMAPE is a percentage. Hours with few or no pickups decide it, and it favours forecasts that are too high.
- An MAE divided by the mean demand is a simple aid for reading: 27.70 / 156.4 = 17.7%.

## Exercises

**1.** Forecaster A is wrong by 10 pickups in each of 100 hours. Forecaster B is exact in 99 hours and wrong by 1,000 in one. Calculate the MAE and the RMSE of both. Which one would a planner prefer?

<details>
<summary>Answer</summary>

Both have an MAE of 10: A has 100 × 10 / 100, and B has 1,000 / 100. The RMSE of A is 10, because every squared error is 100. The RMSE of B is √(1,000,000 / 100) = 100. The MAE cannot tell them apart. The RMSE can, and the distance between RMSE and MAE says that the error of B sits in one hour. Which is better depends on the user. A planner who can absorb small errors and is hurt badly by one very wrong hour prefers A. This is why both numbers are reported.

</details>

**2.** The file `baseline.py` contains the name `daily_naive`, and the report has no scores for it. Explain why, starting from the config.

<details>
<summary>Answer</summary>

`daily_naive` copies the column `lag_24h`. That column exists only if 24 is in `features.lags`. The validator of Chapter 7 refuses a lag that is shorter than the horizon, and the horizon is 48 hours, so the config cannot contain 24. The function `baselines` adds the daily baseline only when 24 is among the lags, so with the real config it returns the weekly baseline alone. The code is still needed for a config with a horizon of 24 hours, and the tests run it with such settings.

</details>

**3.** A colleague wants to report only the sMAPE, "because a percentage is easier for managers than pickups". What do you answer, and what do you offer instead?

<details>
<summary>Answer</summary>

The sMAPE of this data is decided by the quiet night hours. An error of one pickup at 03:00 can score 200%, and an error of 60 pickups at 18:00 can score 10%. A planner cares about the second and not about the first. The sMAPE also favours a forecast that is too high. Offer the MAE as the main number, and next to it the MAE divided by the mean demand, which is also a percentage: 17.7% for the baseline. Keep the sMAPE in the report as a second opinion.

</details>

**4. Interview question.** "Your model has an MAE of 26. How do you know whether that is good?"

<details>
<summary>Points of a good answer</summary>

- Compare it with a baseline that needs no model. For hourly demand that is the same hour one week earlier.
- Measure the baseline first, on the same rows and with the same code as the model.
- Put the MAE next to the mean demand, so that the size of the error is clear.
- Report more than one metric, and say what each one hides: MAE hides rare large errors, RMSE is decided by them, and percentage errors are decided by small counts.
- Look at the parts and not only at the total: per week, per zone, per hour of the day.

</details>

---

[Previous: Chapter 21 – A pipeline that knows what changed](21-a-pipeline-that-knows-what-changed.md) | [Contents](README.md) | [Next: Chapter 23 – The backtest: four weeks the forecaster has not seen](23-the-backtest.md)
