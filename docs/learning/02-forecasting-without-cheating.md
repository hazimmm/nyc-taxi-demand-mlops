# Chapter 2 – Forecasting without cheating

Your model scores very well in the notebook. You deploy it on Monday, and by Tuesday the forecasts are clearly worse than the notebook promised. Nothing crashed and the code is the same. What happened? In most cases the model was using, during training, information that does not exist at the moment a real forecast is made. This chapter shows how that happens and how this project prevents it.

## What you will learn

- What a lag feature is and why it is the most useful input in demand forecasting.
- At what moment the forecast for a whole day can be issued, and why.
- Why this project changed its horizon from 24 to 48 hours, and what that costs.
- What data leakage is, and why it gives no error message.
- Three leaks that the plan for this project found, and the decision made for each.

## The most useful feature: the past

Taxi demand repeats. Tuesday at 18:00 looks like last Tuesday at 18:00. So the best predictor of demand is earlier demand.

A **lag feature** is a past value of the thing you predict, used as an input. The thing you predict is called the **target**; here it is the number of pickups. "Lag 168 hours" for Tuesday 18:00 is the number of pickups in the same zone on the Tuesday before at 18:00, because a week has 168 hours. "Lag 48 hours" is the same hour two days earlier.

In pandas a lag is one line: `df["pickups"].shift(48)` on an hourly table moves every value 48 rows down. This is so easy that it is dangerous, as the next section shows.

## Why "one hour ago" is impossible

Suppose you want the best possible model. You try `shift(1)`: the pickups one hour earlier. The error drops a lot. Demand at 18:00 is very close to demand at 17:00, so the model learns to copy it.

Now use the model for real. The planner needs the forecast for Tuesday during Monday (Chapter 1). The feature for Tuesday 18:00 needs the pickups of Tuesday 17:00. On Monday, Tuesday 17:00 has not happened.

In the notebook the value existed, because the historical table contains every hour. In real use it does not exist. So a lag of one hour is out. The next question is harder: which lag is short enough to be useful and long enough to exist?

## When is a number known?

The first answer of this project was 24 hours. The owner of the project reasoned: "To forecast Tuesday 18:00 we use the pickups of Monday 18:00." This is true, and the first version of the config said `horizon_hours: 24` and had a lag of 24 hours. Two details were missing.

The first detail is about one row. In the demand table, the row "Monday 18:00" holds the number of trips that started between 18:00 and 19:00. The number does not exist at 18:00. It is **complete** at 19:00, when the hour has ended.

The second detail is about the user. A planner does not want one hour. A planner wants the whole of tomorrow at once: 24 numbers per zone, from 00:00 to 23:00.

Put the two together. The table below uses real counts of Midtown Center (zone 161) for Tuesday 4 March 2025.

| Hour of Tuesday to forecast | "Same hour yesterday" is the count of | Pickups | That count is complete at |
|---|---|---|---|
| 00:00 | Monday 00:00 to 01:00 | 18 | Monday 01:00 |
| 17:00 | Monday 17:00 to 18:00 | 563 | Monday 18:00 |
| 18:00 | Monday 18:00 to 19:00 | 680 | Monday 19:00 |
| 23:00 | Monday 23:00 to 24:00 | 91 | Tuesday 00:00 |

*Figure 2.1 – A lag of 24 hours. Each hour of Tuesday waits for one hour of Monday to end.*

Read the last column from top to bottom. On Monday at 18:00 the forecast for Tuesday can be filled up to the row 17:00, and no further. The row 18:00 can be filled at 19:00. The last row, 23:00, can be filled only when Monday is over. So the whole of Tuesday can be issued at Tuesday 00:00 at the earliest, when Tuesday has already begun. With a lag of 24 hours the system forecasts the day that is starting, not the next day.

## Two options, and the one that was chosen

There were two honest ways out.

**Option A** was to keep 24 hours and to describe the product as it is: a forecast that grows hour by hour, where each hour of tomorrow becomes available 23 hours before it starts.

**Option B** was to use 48 hours: no feature may read pickups newer than two days before its target hour. The same table then looks like this.

| Hour of Tuesday to forecast | Newest count it may use | Pickups | That count is complete at |
|---|---|---|---|
| 00:00 | Sunday 00:00 to 01:00 | 148 | Sunday 01:00 |
| 18:00 | Sunday 18:00 to 19:00 | 278 | Sunday 19:00 |
| 23:00 | Sunday 23:00 to 24:00 | 39 | Monday 00:00 |

*Figure 2.2 – A horizon of 48 hours. The last row decides when the whole day can be issued.*

The last row is complete at Monday 00:00. At that moment every row of the table can be filled. So the forecast for the whole of Tuesday can be issued at 00:00 on Monday. Its first hour starts 24 hours after that moment, and its last hour starts 47 hours after it. The planner has all of Monday to use it.

The owner chose option B. The general rule is: **the forecast for all 24 hours of a day can be issued at 00:00 on the day before**.

The choice has a price, and the table shows it. Tuesday 18:00 had 706 pickups. Monday 18:00 had 680, which is close. Sunday 18:00 had 278, because a Sunday evening in Midtown is quiet. Pickups from two days ago are older information than pickups from yesterday, so the model becomes somewhat less accurate. The owner accepted this: a forecast that arrives in time and is a little less exact is worth more than an exact one that arrives when the day has started. The same hour one week earlier, which had 768 pickups, is not affected by the choice. Chapter 22 measures what each of these copies is worth.

> [!IMPORTANT]
> Every feature for hour *t* may use only pickups from the hour that starts at *t* − 48 hours, or earlier.

The number 48 is the **forecast horizon** of this project: how far ahead of the newest known pickups a forecast must reach. You can read it in [`configs/config.yaml`](../../configs/config.yaml):

```yaml
features:
  # No feature of hour t reads pickups newer than t - 48 h. With 48 the forecast for a
  # whole day can be issued at 00:00 on the day before it. With 24 it could only be
  # issued at 00:00 on the day itself. No lag may be shorter than the horizon, so a
  # lag of 24 ("the same hour yesterday") is not allowed.
  horizon_hours: 48
  lags: [48, 168]
  rolling_windows: [24, 168]
  outlier_zscore: 4.0
```

The configured lags are 48 and 168 hours. The line `rolling_windows` describes **rolling statistics**: a summary, such as the mean, over a moving window of hours. "The mean of the last 24 hours" has the same problem as a short lag, because its newest hours are too recent. So the feature code computes these windows so that they end 48 hours before the target hour. Chapter 19 shows the code and the test that guards it.

## Data leakage

The general name for this problem is **data leakage**: the model is trained or tested with information that will not be available when it is really used.

Leakage has one property that makes it worse than a normal bug: it produces no error. The program runs. The score is good, in fact better than it should be. A model with a leak always looks better in testing than it performs in use. So a surprisingly good result is a reason to search for a leak, not a reason to celebrate.

The first version of this project is an example. No program complained about the lag of 24 hours, and every single row respected the 24-hour rule. But the product was described as "the whole of tomorrow, one day early", and for that product the pickups of yesterday evening are information that does not exist yet. The mistake was found by asking at what moment each number is known, not by a failing test.

> [!WARNING]
> If a simple change makes the error drop by a large amount, first ask: "Would I have this information at prediction time?" Most large and easy improvements in forecasting are leaks.

The lag case is the best-known leak. The plan for this project found two more, and both are easy to miss.

## A second leak: weather

Rain increases taxi demand, so weather is a useful feature. A weather service gives two kinds of data for any past hour. **Observed weather** is what really happened. A **weather forecast** is what was predicted before it happened.

For training, observed weather looks like the natural choice: it is the truth. But think about real use. On Monday you forecast Tuesday 18:00. You do not know the real rain of Tuesday 18:00. You only have the weather forecast for it, and weather forecasts are sometimes wrong.

A model trained on observed weather learns that the weather column is always correct, so it trusts it strongly. In real use it receives forecasts with errors, trusts them just as strongly, and makes mistakes that it never made in testing.

The decision: train on archived weather *forecasts*, which is what the model will receive in real use. The config file has two addresses for this reason.

```yaml
  weather:
    latitude: 40.7128
    longitude: -74.0060
    forecast_url: "https://historical-forecast-api.open-meteo.com/v1/forecast"
    archive_url: "https://archive-api.open-meteo.com/v1/archive"
```

The first address returns past forecasts. The second returns observed weather and is only a fallback. If the forecast service fails or its answer is incomplete, the download uses observed weather and records that fact next to the file. A limitation that is written down is acceptable. One that is hidden is a leak. Chapter 15 shows the code.

One limitation of this kind remains. The project has not checked how long before each hour the archived forecast was made. A forecast made six hours ahead is better than one made 47 hours ahead, and the model will get the second kind. Chapter 23 lists this among the things the test of the model does not prove.

## A third leak: choosing the zones

The project forecasts only the 20 busiest zones. Which are they? The obvious method is to count the pickups per zone over all six months and take the top 20.

This also leaks, and no feature is involved. The six months include May and June, which are kept for testing. If a zone became busy only in June, it enters the list because of data the model is not allowed to see. The choice itself carries information from the future into the project.

The decision: the zones are chosen from the training months only, then written into the config file as a fixed list. The list is calculated again only if the end of the training period changes.

```yaml
  n_zones: 20
  # Frozen list of pickup zone IDs, chosen from the training period only.
  # Output of `python -m demand select-zones`, busiest first. Run it again and paste the
  # result here only when train_end_date changes.
  zones: [161, 237, 236, 132, 230, 186, 162, 142, 234, 170, 163, 239, 68, 138, 79, 48, 249, 141, 164, 140]
```

Chapter 18 shows the command, and the one zone that the full six months would have chosen differently.

The lesson is wider than zones. Any decision made by looking at data can leak: which rows count as outliers, how to fill gaps, which columns to keep. If the decision looked at the test period, the test is no longer clean.

## Rules that the program enforces

Knowing these rules is not enough. Six months from now someone will try a lag of 24 hours, see a better score, and keep it. That person may be you.

So this project puts the most important rule into code. When the program starts, it compares every lag with the horizon and refuses to run if a lag is too short. Chapter 7 walks through that code. You can already see it work.

## Try it yourself

Read the lags that the program uses.

```powershell
(.venv\Scripts\python.exe -m demand config | ConvertFrom-Json).features.lags
```

You should see `48` and `168` on two lines. This proves the values come from the config file, not from a number written inside the code.

Now try to bring back "the same hour yesterday". The line below sets an **environment variable**, a named value that the shell gives to the program. Chapter 7 explains how this replaces the value from the file.

```powershell
$env:DEMAND_FEATURES__LAGS = "[24, 168]"
.venv\Scripts\python.exe -m demand config
Remove-Item Env:DEMAND_FEATURES__LAGS
```

You should see a long error report. Near the end it says:

```text
Value error, features.lags [24] are shorter than the 48 h forecast horizon
and would leak information unavailable at prediction time
```

No configuration is printed. This proves that a leaking lag cannot be configured: the program stops before any work starts. The last line removes the variable so that later commands behave normally.

Check the numbers of Figure 2.1 and Figure 2.2. This command needs the gold layer of Chapter 18.

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; d = pd.read_parquet('data/gold/demand_hourly.parquet'); p = d[d.zone_id == 161].set_index('hour_start').pickups; t = pd.Timestamp('2025-03-04 18:00', tz='America/New_York'); h = pd.Timedelta(hours=1); print([int(p[t - k * h]) for k in (0, 24, 48, 168)])"
```

You should see `[706, 680, 278, 768]`: Tuesday 18:00, then the same hour one day, two days and one week earlier.

## Key takeaways

- A lag feature is a past value of the target. It is powerful and easy to compute wrongly.
- The count of an hour is complete only when the hour has ended, and a planner needs a whole day at once. The last hour of the day decides when the forecast can be issued.
- With a horizon of 48 hours, the forecast for a whole day can be issued at 00:00 on the day before. With 24 hours it could be issued only when the day begins.
- The longer horizon costs accuracy, because the newest pickups a feature may read are two days old.
- Leakage gives no error. Its only sign is a result that is too good.
- Observed weather is a leak when the real system will only have a weather forecast.
- Decisions made from data, such as which zones to keep, must use the training period only.
- A rule that the program checks is safer than a rule that people must remember.

## Exercises

**1.** A colleague suggests the feature "average pickups in this zone over the whole dataset". It uses no lag. Does it leak?

<details>
<summary>Answer</summary>

Yes. "The whole dataset" includes the test months, so the average contains future information for every training row. The clean version is the average over the training period only, computed once and then reused. This is the same problem as the zone selection.

</details>

**2.** The horizon is 48 hours. At what moment can the forecast for the whole of Saturday be issued? A second customer needs the forecast one day earlier than that. Which horizon do they need?

<details>
<summary>Answer</summary>

The last hour of Saturday, 23:00, may use the count of Thursday 23:00 to 24:00, which is complete at Friday 00:00. So the whole of Saturday can be issued at 00:00 on Friday. Its first hour starts 24 hours later and its last hour 47 hours later. The second customer needs it at 00:00 on Thursday. At that moment the newest complete count is Wednesday 23:00 to 24:00, which is 72 hours before Saturday 23:00. They need a horizon of 72 hours, and the lag of 48 hours would then be refused.

</details>

**3.** Imagine that weather forecasts were always perfect. Would using observed weather for training still be a leak?

<details>
<summary>Answer</summary>

No. Leakage means a difference between what the model sees in training and what it will have in use. If forecasts were perfect, the two would be the same numbers. The leak exists because forecasts contain errors and observations do not. This shows that leakage is not about the type of data but about its availability at prediction time.

</details>

**4. Interview question.** "How do you prevent data leakage in a forecasting project?"

<details>
<summary>Points of a good answer</summary>

- Start from the horizon: say when the forecast is issued, for which hours, and what is known at that moment. Remember that the count of an hour is known only when the hour has ended.
- Give concrete cases: lags shorter than the horizon; observed weather used where only a forecast would exist; selections such as "top zones" calculated on all the data.
- Say that you enforce the rule and do not only know it: a check at start-up that rejects short lags, a split by time, and a test that no feature for hour *t* uses data newer than *t* − 48 hours.
- Name the warning sign: a model that is much better offline than in real use.

</details>

---

[Previous: Chapter 1 – What we are building](01-what-we-are-building.md) | [Contents](README.md) | [Next: Chapter 3 – Testing a forecast honestly](03-testing-a-forecast-honestly.md)
