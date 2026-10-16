# Chapter 24 – A model behind the same two methods: LightGBM as a forecaster

Chapter 22 opened with one hour: Tuesday 8 April 2025 at 18:00 in Midtown Center (zone 161). There were 739 pickups, the weekly baseline said 694, and its error was 45. The LightGBM model of this chapter says 670 for that hour. Its error is 70, larger than the baseline's. Over all 13,440 test hours the same model has a mean error of 24.11 pickups, and the baseline has 27.70. Both statements are true. This chapter builds the model, and then reads its result with enough care to explain both.

The numbers of this chapter are those of the current config, whose values were tuned in Chapter 27. The first version of the model, before tuning, had an MAE of 24.93.

## What you will learn

- How LightGBM is wrapped so that the backtest scores it exactly like the baseline.
- What gradient-boosted trees do, in plain words.
- Why the zone is a category, why the objective is made for counts, and why a forecast is never negative and never rounded.
- Which columns the model reads, and what it refuses.
- How to read the real result: per week, per zone, per hour, and on one unusual day.
- What feature importance says, and what it does not say.

## The same two methods

Chapter 22 defined a forecaster as anything that has the two methods `fit` and `predict`. The model is a class with these two methods.

[`src/demand/models/gbm.py`](../../src/demand/models/gbm.py)

```python
        inputs = self._checked_inputs(train)
        model = lightgbm.LGBMRegressor(**self.params)
        model.fit(inputs, train[TARGET].to_numpy(dtype="float64"), categorical_feature=[ZONE_ID])
```

These are the three lines of `fit` that do the work. The first picks the input columns out of the training rows and checks them. The second creates a LightGBM model with the parameters of the config. The third trains it on the inputs and on the target, the column `pickups`.

One more line was needed, in [`src/demand/models/pipeline.py`](../../src/demand/models/pipeline.py):

```python
    return {**baselines(settings), LIGHTGBM: lightgbm_forecaster(settings)}
```

This is the list of forecasters that are compared: the baseline first, then the model under the name `lightgbm`. The function `backtest` of Chapter 23 did not change by one character. It calls `fit` and `predict` on every forecaster in the list, on the same folds and the same rows.

## What gradient-boosted trees do

A **decision tree** asks yes-or-no questions about the inputs of a row and ends in a number. The place where it ends is called a **leaf**. The first question of the first tree of the real model is: "is `lag_168h` at most 155.5?" On both sides the next question is about `lag_168h` again, with the limits 67.5 and 324.5. After a few questions the row is in one of 110 leaves, and each leaf holds a number.

One tree gives a rough forecast. **Gradient boosting** builds many trees in a row, and every new tree is trained on what the trees before it still get wrong. The final forecast is the sum of the answers of all trees. The config asks for 600 trees (`n_estimators`) with at most 110 leaves each (`num_leaves`). Each tree corrects only a part of the remaining error (`learning_rate: 0.104`), because many small steps give a better model than a few large ones. Chapter 27 shows how these values were found.

The table shows the mean error of the real model on its own training rows when only its first trees are used.

| Trees used | MAE on the training rows |
|---|---|
| 1 | 95.31 |
| 10 | 64.50 |
| 100 | 20.92 |
| 600 | 13.35 |

*Figure 24.1 – Every tree corrects a little. The model is fitted on all 53,280 training rows.*

The last number is 13.35, and the test weeks gave 24.11. A model is always better on rows it has learned from. This is why no score in this book is measured on training rows.

To make the trees differ from each other, every tree is built from a random 85% of the rows and a random 80% of the columns (`subsample` and `colsample_bytree`). Chapter 25 shows what this randomness does to a score.

## The zone is a label

`zone_id` is a whole number, but 162 is not "more" than 161. A tree that reads the zone as a number can only ask "is the zone below 150?". The argument `categorical_feature=[ZONE_ID]` tells LightGBM that the column is a **categorical feature**: a label from a fixed list. The tree can then put any group of zones on one side. One real question of the model is: "is the zone 79 or 249?" Those are East Village and West Village, two zones that are far apart in the numbering. Chapter 25 measures how much this is worth.

## A model for counts

The **objective** is the measure of error that training tries to make small. The config sets `objective: poisson`, which is made for counts. With it the model works with the logarithm of the pickups, and two things follow. A forecast can never be negative. And an error of 5 pickups counts more in an hour with 10 pickups than in an hour with 500.

The last line of `predict` adds a second protection:

```python
        forecast: FloatArray = np.clip(np.asarray(self.model.predict(inputs), "float64"), 0, None)
```

`np.clip(..., 0, None)` replaces every value below 0 by 0. With the Poisson objective it changes nothing. It is there for the day someone sets another objective: plain regression can forecast −3 pickups.

A forecast is not rounded. Pickups are whole numbers, but a forecast is an expected value: "2.4 pickups" means 2 in some such hours and 3 in others. Ten hours with a forecast of 0.4 add up to 4 pickups. Rounded, they add up to 0. On the test weeks rounding changes the MAE from 24.106 to 24.105, so nothing is gained by it.

## What the model reads, and what it refuses

The model reads the 16 columns of `feature_columns` (Chapter 19) and no others. Two columns of the feature table must never be inputs: the target `pickups`, and `is_outlier`, which is calculated from the pickups (Chapter 18). They are kept out twice. The backtest passes `test[inputs]` to `predict`, a table without the two columns. And `predict` itself takes its columns by name, `rows[self.columns]`, so a target that someone leaves in the table is not read. A test proves it: it changes the target of every row to 0, and the forecasts stay the same.

A model that is asked something it was not trained for still answers, and the answer looks like any other number. So `predict` stops with a message in five cases: the forecaster was never fitted, a column is missing, a column has another type than in training, a value is missing or infinite, or a row belongs to a zone that was not in the training rows.

The last case needs a reason. For LightGBM an unknown category is "none of the zones I know". A row of zone 999 would go down every tree on the "no" side of every zone question and come out with some number. Nobody could see that it is a guess. A refusal with the list of the 20 known zones can be seen.

## The result

`python -m demand backtest` now scores both forecasters. On the 13,440 test rows the model has an MAE of 24.11, an RMSE of 38.24 and a sMAPE of 21.88%. The baseline has 27.70, 44.51 and 24.64%. The MAE is 13.0% lower. Against the mean demand of 156.4 pickups it is 15.4%, where the baseline had 17.7% (Chapter 22).

Now the hours from the opening.

| Hour on 8 April, zone 161 | Actual | Baseline | Model | Error of the baseline | Error of the model |
|---|---|---|---|---|---|
| 16:00 | 541 | 514 | 494.8 | 27 | 46.2 |
| 17:00 | 674 | 651 | 637.4 | 23 | 36.6 |
| 18:00 | 739 | 694 | 669.5 | 45 | 69.5 |
| 19:00 | 591 | 539 | 594.4 | 52 | 3.4 |
| 20:00 | 735 | 606 | 671.2 | 129 | 63.8 |

*Figure 24.2 – Five real hours, forecast by the model of fold 1. The model is better in two and worse in three.*

A lower MAE does not mean "better in every hour". Over all test rows the model is closer to the actual value in 57% of the hours, and the baseline is closer in 43%. The model wins on average. So the parts must be read, as in Chapter 23.

**Per week.**

| Fold | Test week | MAE of the baseline | MAE of the model | The model is better by |
|---|---|---|---|---|
| 1 | 3 to 9 April | 27.61 | 23.51 | 14.9% |
| 2 | 10 to 16 April | 26.57 | 21.85 | 17.8% |
| 3 | 17 to 23 April | 27.92 | 24.75 | 11.3% |
| 4 | 24 to 30 April | 28.70 | 26.31 | 8.3% |

*Figure 24.3 – The four folds, from `reports/backtest_report.json`.*

The model is better in every week. The gain is smaller in folds 3 and 4, the Easter week and the week that copies it.

**Per zone.** The model is better in 19 of the 20 zones, by 4% to 23%. In West Village (zone 249) it is worse: 22.34 against 22.04.

**Per hour of the day.** From 02:00 to 07:00 the gain is small: at 03:00 the model is worse than the baseline by 2.5%, and in the other five of these hours it is better by less than 5%. In the remaining 18 hours it is better by 8.3% to 16.6%. On the 2,072 test hours with fewer than 20 pickups the model has an MAE of 4.09 and the baseline 4.27. On the busier hours it has 27.75 against 31.97. Before tuning, the model was worse than the baseline in the quiet hours.

**One unusual day.** On Good Friday, 18 April, the city had 141.5 pickups per zone and hour. The baseline said 180.6 and the model said 169.1. The MAE was 47.5 for the baseline and 37.0 for the model. The model is better, and it is still wrong by a large amount in the same direction. No column tells it that the day is unusual (Chapter 23).

All of these parts come from one run. Chapter 25 shows that a run with other random choices moves the total by about 0.2 pickups, and a part moves more than the total. Read the parts as a description, not as a ranking.

## Which inputs the model uses

**Feature importance by gain** answers one question: how much of the improvement during training came from questions about each column? Every question of a tree lowers the error on the training rows by some amount. LightGBM adds these amounts per column.

| Input | Share of the gain |
|---|---|
| `lag_168h` | 75.3% |
| `hour` | 10.6% |
| `lag_48h` | 4.6% |
| `zone_id` | 2.8% |
| `day_of_week` | 2.4% |
| the four weather columns together | 1.5% |
| `is_holiday` | 0.1% |

*Figure 24.4 – The largest shares, for the model fitted on all training rows with the tuned values. Before tuning, `lag_168h` had 76.3%.*

Three quarters of the gain come from the pickups of one week ago. In effect the model is a weekly copy with corrections. This fits the result: it is better than the baseline by about an eighth, and it is wrong on Good Friday in the same direction.

The table has limits. Gain is counted on training rows, so it shows what the trees used, not what helps on new weeks. It gives no direction: it does not say whether rain raises or lowers a forecast. And inputs that carry similar information share the credit: `roll_mean_168h` has 0.9%, partly because `lag_168h` was asked first. A method that explains single forecasts, SHAP, is the subject of Chapter 28.

## Try it yourself

These commands need the gold layer of Chapter 18. Run the backtest. It fits four models and takes about seven seconds.

```powershell
.venv\Scripts\python.exe -m demand backtest
```

You should see three lines. The second is like this:

```text
{"time": "2026-10-14T00:07:59.625347+00:00", "level": "INFO", "logger": "demand.models.backtest", "message": "forecaster scored", "forecaster": "lightgbm", "mae": 24.106025, "rmse": 38.244101, "smape": 21.882594, "rows": 13440, "mae_without_outliers": 24.092702, "mae_per_fold": {"1": 23.510474, "2": 21.84912, "3": 24.753347, "4": 26.311159}}
```

Fit the model of fold 1 by hand and forecast the five hours of Figure 24.2. Then ask it for a zone it does not know.

```powershell
.venv\Scripts\python.exe -c "from demand.config import get_settings; from demand.models.backtest import load_training_features; from demand.models.gbm import lightgbm_forecaster; s = get_settings(); t = load_training_features(s); f = lightgbm_forecaster(s); f.fit(t[t.hour_start <= '2025-04-01 00:00']); rows = t[(t.zone_id == 161) & (t.hour_start >= '2025-04-08 16:00') & (t.hour_start <= '2025-04-08 20:00')]; print(rows.assign(model=f.predict(rows).round(1))[['hour_start', 'pickups', 'lag_168h', 'model']].to_string(index=False)); f.predict(rows.assign(zone_id=999))"
```

You should see:

```text
               hour_start  pickups  lag_168h  model
2025-04-08 16:00:00-04:00      541     514.0  494.8
2025-04-08 17:00:00-04:00      674     651.0  637.4
2025-04-08 18:00:00-04:00      739     694.0  669.5
2025-04-08 19:00:00-04:00      591     539.0  594.4
2025-04-08 20:00:00-04:00      735     606.0  671.2
```

and then a traceback that ends with:

```text
demand.models.gbm.ModelInputError: The model was not trained on zone 999; it knows [48, 68, 79, 132, 138, 140, 141, 142, 161, 162, 163, 164, 170, 186, 230, 234, 236, 237, 239, 249]. Forecast only zones of data.zones, or add the zone to the config, rebuild with `dvc repro` and train again.
```

Calculate the gain shares of Figure 24.4.

```powershell
.venv\Scripts\python.exe -c "import pandas as pd; from demand.config import get_settings; from demand.models.backtest import load_training_features; from demand.models.gbm import lightgbm_forecaster; s = get_settings(); f = lightgbm_forecaster(s); f.fit(load_training_features(s)); b = f.model.booster_; g = pd.Series(b.feature_importance(importance_type='gain'), index=b.feature_name()); print((g / g.sum() * 100).round(1).sort_values(ascending=False).head(6).to_string())"
```

You should see:

```text
lag_168h          75.3
hour              10.6
lag_48h            4.6
zone_id            2.8
day_of_week        2.4
roll_mean_168h     0.9
```

Run the tests of the forecaster.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/models/test_gbm.py -q
```

You should see `31 passed`.

## Key takeaways

- The model is a class with `fit` and `predict`. One line added it to the list of forecasters, and the backtest did not change.
- Gradient boosting builds many small trees, and each one corrects part of what the trees before it get wrong.
- The zone is a label, so LightGBM is told to read it as a category.
- The Poisson objective is made for counts. A forecast is never negative and never rounded.
- The model reads the 16 feature columns by name. The target and the outlier flag cannot reach it.
- A row the model was not trained for is refused with a message, because a silent guess looks like a forecast.
- With the tuned values the model has an MAE of 24.11 against 27.70: 13.0% better, better in every fold and in 19 of 20 zones, with little gain in the quiet hours of the early morning.
- Gain importance shows which columns the trees used on training rows. Here 75% is `lag_168h`.

## Exercises

**1.** In Figure 24.2 the model is worse than the baseline at 16:00, 17:00 and 18:00. A colleague concludes: "The model is bad in the evening rush hour of Midtown." What is wrong with this conclusion, and how would you test the claim?

<details>
<summary>Answer</summary>

Three hours of one day in one zone are not evidence for a pattern. The model is closer than the baseline in only 57% of all test hours, so a few losses in a row happen often by chance. To test the claim, take all test rows of zone 161 between 17:00 and 19:00 in the four weeks, calculate the MAE of both forecasters, and repeat it with other seeds (Chapter 25). With seed 42 these 84 rows give 47.06 for the model and 58.76 for the baseline, so the claim does not hold. Over all hours, zone 161 has 27.66 against 31.98.

</details>

**2.** Someone removes the check for unknown zones: "LightGBM can handle unknown categories itself, the check only causes errors." What do you answer?

<details>
<summary>Answer</summary>

LightGBM does return a number for an unknown zone. It treats the row as "none of the zones I know", so the row takes the "no" side of every zone question. The number is not a forecast for that zone, and nothing in the output marks it as different. A planner would use it like any other number. The check turns a silent wrong answer into a visible error with the list of known zones and the steps to add a zone. An error that is seen can be fixed.

</details>

**3. Interview question.** "Your feature importance shows one feature with 75%. What do you conclude, and what do you not conclude?"

<details>
<summary>Points of a good answer</summary>

- Say which importance it is: gain is the share of the training improvement that came from questions about that column.
- Conclude that the model leans on this input. Here it is the same hour one week ago, so the model is close to the weekly baseline and shares its weaknesses, for example on an unusual day.
- Do not conclude a direction or a size of the effect on a single forecast.
- Do not conclude that inputs with a small share are useless: inputs with similar information share the credit, and gain is counted on training rows.
- To decide whether an input is worth keeping, remove it and compare backtest scores, with more than one seed.

</details>

---

[Previous: Chapter 23 – The backtest: four weeks the forecaster has not seen](23-the-backtest.md) | [Contents](README.md) | [Next: Chapter 25 – Deciding by measurement: four choices and the noise of a seed](25-deciding-by-measurement.md)
