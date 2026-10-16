# Chapter 28 – Explaining a forecast: SHAP

On Friday 18 April 2025 at 08:00, Upper East Side North (zone 236) had 177 pickups. A model that was fitted on the hours up to 15 April said 388. Chapter 24 could say which inputs the model uses in general: 75% of the gain belongs to `lag_168h`. It could not say why this one forecast was more than twice too high. This chapter adds the method that can, and uses it on this hour.

## What you will learn

- What a SHAP value is, and why this model's values are read as factors.
- What the model saw on Good Friday.
- What a training run now calculates and stores, and how to read the plot.
- Why the SHAP share of an input differs from its share of the gain.
- Four things SHAP does not tell you.

## A forecast as a sum

Gain importance (Chapter 24) is one number per input for the whole model, with no direction and nothing about a single row. **SHAP** is a method that splits one forecast into parts. It starts from the **base value**: the output the model gives on average over its training rows. Then every input of the row adds or takes away an amount. This amount is the **SHAP value** of the input for this row. This book also calls it the contribution of the input. For trees the values are calculated exactly, and they have one fixed property:

> [!IMPORTANT]
> The base value plus the contributions of all inputs of a row is exactly the output of the model for that row.

## The scale is the logarithm

With the Poisson objective (Chapter 24) the trees do not add up to pickups. They add up to the **natural logarithm** of the forecast, written `log` in Python. The forecast is `exp` of the sum, and `exp` is the function that reverses the logarithm.

On this scale, adding means multiplying: `exp(a + b)` is `exp(a)` times `exp(b)`. So a contribution is read as a factor. A contribution of +0.69 doubles the forecast, because `exp(0.69)` is 1.99. A contribution of −0.69 halves it. A contribution of +0.10 makes it about 10% larger. The same +0.69 is 5 more pickups in an hour that would have had 5, and 200 more in an hour that would have had 200.

## Good Friday, input by input

The table explains the forecast from the opening. The model is the one of fold 3 of the backtest: fitted on the 45,620 rows up to Tuesday 15 April at 00:00, so Good Friday is a day it has not seen.

| Input | Value in this row | Contribution | As a factor |
|---|---|---|---|
| Base value | – | 4.484 | 88.6 pickups |
| `lag_168h` | 348 | +0.840 | × 2.32 |
| `lag_48h` | 414 | +0.200 | × 1.22 |
| `hour` | 8 | +0.182 | × 1.20 |
| `zone_id` | 236 | +0.117 | × 1.12 |
| `day_of_week` | 4 (Friday) | +0.067 | × 1.07 |
| `is_holiday` | false | +0.003 | × 1.00 |
| The other ten inputs together | – | +0.067 | × 1.07 |
| **Sum** | | **5.960** | **387.7 pickups** |

*Figure 28.1 – One forecast as a sum. `exp(5.960)` is 387.7, the forecast of the model. The factors in the last column, multiplied with 88.6, give the same number up to rounding. The model before tuning said 366.*

Read the table from the top. Before it knows anything about the row, the model says 88.6 pickups. The same hour one week ago had 348 pickups, and this alone multiplies the forecast by 2.32. Two days ago there were 414: times 1.22. It is 08:00, in a busy zone, on a Friday: three more factors above 1.

Every input says "an ordinary Friday morning". None of them could say anything else. Good Friday is not a federal holiday, so `is_holiday` is false, and its contribution is a factor of 1.003. One week earlier, on Friday 11 April at 08:00, the row of the same zone looked much the same: `lag_168h` was 340, and the hour, the weekday and the holiday flag were equal. For that hour, one of its training rows, the model says 369, and there were 348 pickups. To the model the two Fridays are nearly the same row.

This closes a thread from Chapter 23. The baseline had its worst day on Good Friday because it copies the Friday before. The model fails for the same reason: the factor 2.32 is that copy, and no column announces the day. A richer holiday calendar could help, for example a column that also marks Good Friday. It is not built here. The data from January to April holds one Good Friday, so such a column could not be tested.

The other direction: on Wednesday 23 April at 04:00, the same zone had 1 pickup one week earlier, and `lag_168h` multiplies the base value by 0.10. The lag of two days, the hour and the zone give 0.67, 0.68 and 0.80. The forecast is 3.5, and there were 6 pickups.

## What a training run does now

`python -m demand train` has a new third step, between the final fit and the record (Figure 26.1). The work is in [`src/demand/models/explain.py`](../../src/demand/models/explain.py).

```python
    explanation = explain(forecaster, sample_rows(table, SHAP_SAMPLE_ROWS, seed))
```

**A sample of 5,000 rows.** The run explains 5,000 of the 53,280 training rows, drawn with `project.seed`, so the same model and table give the same numbers. The reason is time: for the tuned model, all rows took 161 seconds on the machine of this book and the sample 15, and the five strongest inputs differed by less than 1%. (The modeller measured 82 against 8 seconds for the smaller trees before tuning.) They are training rows, because the question is how the model works, not how good it is.

**Two files.** `reports/shap_importance.json` holds the base value and, for each of the 16 inputs, the **mean absolute SHAP value**: the mean size of its contribution over the 5,000 rows, whatever the direction. `reports/shap_summary.png` is a picture of all 80,000 contributions.

**The record.** Both files are stored with the MLflow run, with 16 metrics named `shap.mean_abs.<input>` (Chapter 26). An input whose value jumps between two runs has changed its role in the model.

**Two protections.** Only `explain.py` imports `shap` and `matplotlib`, so the backtest and the tuning never load them. And `explain` checks the property of the callout above for every row:

```python
    gap = float(np.abs(values.sum(axis=1) + base_value - output).max())
    if gap > ADDITIVITY_TOLERANCE:
        raise PipelineError(
```

`ADDITIVITY_TOLERANCE` is 0.000001. If the contributions of one row do not add up to the output of the model, the training run stops: the installed versions of `shap` and LightGBM do not fit together, and a wrong explanation would be stored with the model. The JSON file is the same, byte for byte, for the same model on the same machine. The picture is not promised to be: another version of matplotlib can draw the same dots with other bytes.

## Reading the summary plot

The picture is not in this book, because `reports/` is not stored in git (Chapter 12). A training run writes it to `reports\shap_summary.png`. For the tuned model it shows this.

There are 16 horizontal lines, one per input, the strongest at the top, with one dot per explained row. The position of a dot is the contribution of the input in that row: right of 0 it pushed the forecast up, left of 0 down. Its colour is the value of the input, from blue for low to red for high.

The top line is `lag_168h`. Its dots reach from about −2.5 to +1.5, blue on the left and red on the right: few pickups one week ago push the forecast down, to as little as one twelfth, and many push it up, to as much as four times. The line of `hour` reaches from −0.9 to +0.4, with the early hours on the left.

The dots of `zone_id` are grey. A zone is a label (Chapter 24), so "high" and "low" mean nothing for it, and the code gives it no colour. Its contributions are still correct: they reach from −1.0 to +0.9.

The line of `is_holiday` is a thin column of blue dots at 0, with a few red dots between −0.3 and +0.5. Remember this line for the limits below.

## Two measures of importance

The share of an input in the sum of all mean absolute SHAP values can be put next to its share of the gain.

| Input | SHAP share | Share of the gain |
|---|---|---|
| `lag_168h` | 49.4% | 75.3% |
| `hour` | 17.8% | 10.6% |
| `lag_48h` | 10.0% | 4.6% |
| `zone_id` | 6.9% | 2.8% |
| `day_of_week` | 5.4% | 2.4% |
| the four weather columns | 4.1% | 1.5% |
| the four rolling columns | 3.9% | 2.5% |
| `month` | 1.7% | 0.2% |
| `is_weekend` | 0.4% | 0.1% |
| `is_holiday` | 0.4% | 0.1% |

*Figure 28.2 – Both measures for the final model with the tuned values of Chapter 27. The gain was calculated again for this model. The SHAP shares come from a training run in a scratch folder with the same config and data.*

The order is almost the same, and the sizes are not, because the questions differ. Gain asks how much the questions about an input lowered the error on the training rows, and an error in a busy hour is a large error. A SHAP value on the log scale asks how far an input moves a forecast as a factor, and every row counts the same: 04:00 weighs as much as 18:00. This is probably why `hour` and `zone_id`, which separate quiet rows from busy ones, have the larger SHAP share.

Both agree on the main point of Chapter 24: the model leans on last week more than on anything else. Before tuning, the modeller measured a SHAP share of 57.2% for `lag_168h`, and the gain share was 76.3%. The tuned model leans on it less.

## What SHAP does not tell you

**It explains the model, not the city.** The factor 2.32 says that the model leans on last week. It does not say that last week causes this week. In the plot, rainy hours have contributions of `precipitation` right of 0. That describes what the trees learned from four months, not what rain does to taxi demand.

**Inputs with similar information share the credit.** The two lags and the four rolling columns all describe recent demand. How the credit is divided among them depends on which column the trees happened to ask about. A small share for `roll_mean_168h` does not show that the column is useless.

**A mean hides rare effects.** `is_holiday` has a share of 0.4%. On the 98 holiday rows of the sample its mean contribution has a size of 0.164, a factor of about 1.18 up or 0.85 down. On the other 4,902 rows it is 0.003. The input is strong on the few days it is set, and those days are 2 of the 111 days: Martin Luther King Jr. Day and Washington's Birthday. The model learned the flag from two days.

**A sample is a sample.** The five strongest inputs were stable against all rows. A rare input can move more.

## Try it yourself

The first command needs a training run with the code of this chapter; `dvc repro` makes one.

Read the report.

```powershell
.venv\Scripts\python.exe -c "import json; r = json.load(open('reports/shap_importance.json')); print(r['rows'], r['seed'], r['objective'], r['base_value']); [print(k, v, r['share_percent'][k]) for k, v in list(r['mean_abs_shap'].items())[:5]]"
```

You should see:

```text
5000 42 poisson 4.489909
lag_168h 0.658183 49.35
hour 0.236751 17.75
lag_48h 0.133252 9.99
zone_id 0.091885 6.89
day_of_week 0.072514 5.44
```

The base value of the final model is 4.49, and `exp(4.49)` is 89 pickups, against a mean of 153 pickups in the training rows: a mean of logarithms gives less weight to the busiest hours.

Open the picture with `Invoke-Item reports\shap_summary.png`.

Explain the forecast of Figure 28.1. The command fits the model of fold 3 and takes a few seconds.

```powershell
.venv\Scripts\python.exe -c "import numpy as np, pandas as pd; from demand.config import get_settings; from demand.models.backtest import load_training_features; from demand.models.explain import explain; from demand.models.gbm import lightgbm_forecaster; s = get_settings(); t = load_training_features(s); f = lightgbm_forecaster(s); f.fit(t[t.hour_start <= '2025-04-15 00:00']); row = t[(t.zone_id == 236) & (t.hour_start == '2025-04-18 08:00')]; e = explain(f, row); c = pd.Series(e.values[0], index=e.inputs.columns).sort_values(key=abs, ascending=False); print(round(e.base_value, 3), round(float(np.exp(e.base_value)), 1)); print(pd.DataFrame({'shap': c.round(3), 'factor': np.exp(c).round(2)}).head(5).to_string()); print(round(float(c.sum()), 3), round(float(e.base_value + c.sum()), 3), f.predict(row).round(1), int(row.pickups.iloc[0]))"
```

You should see:

```text
4.484 88.6
              shap  factor
lag_168h     0.840    2.32
lag_48h      0.200    1.22
hour         0.182    1.20
zone_id      0.117    1.12
day_of_week  0.067    1.07
1.476 5.96 [387.7] 177
```

The last line holds the sum of all 16 contributions, the sum with the base value, the forecast and the actual pickups.

Measure the holiday flag on the rows where it is set. The command fits the final model and explains the sample, in about half a minute.

```powershell
.venv\Scripts\python.exe -c "import numpy as np; from demand.config import get_settings; from demand.models.backtest import load_training_features; from demand.models.explain import SHAP_SAMPLE_ROWS, explain, sample_rows; from demand.models.gbm import lightgbm_forecaster; s = get_settings(); t = load_training_features(s); f = lightgbm_forecaster(s); f.fit(t); rows = sample_rows(t, SHAP_SAMPLE_ROWS, s.project.seed); e = explain(f, rows); h = np.abs(e.values[:, list(e.inputs.columns).index('is_holiday')]); on = rows.is_holiday.to_numpy(); print(int(on.sum()), round(float(h[on].mean()), 3), round(float(h[~on].mean()), 3), round(float(h.mean()), 6))"
```

You should see `98 0.164 0.003 0.005665`. The last number is the value of `is_holiday` in the report.

Check what the other commands load.

```powershell
.venv\Scripts\python.exe -c "import sys; import demand.models.pipeline, demand.models.tuning; print([m in sys.modules for m in ('optuna', 'shap', 'matplotlib', 'mlflow')])"
```

You should see `[True, False, False, False]`.

Run the tests of the explanation.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/models/test_explain.py -q
```

You should see `24 passed`. One test builds a model with one tree and two zones, where the contributions can be checked by hand: a base value of 200, and −100 and +100 for the zone.

## Key takeaways

- A SHAP value is the contribution of one input to one forecast. The base value plus all contributions of a row is exactly the output of the model.
- With the Poisson objective the output is the logarithm of the forecast. A contribution is read as a factor: +0.69 doubles, −0.69 halves.
- A training run explains a fixed sample of 5,000 training rows, writes `shap_importance.json` and `shap_summary.png`, and stores both with 16 metrics in MLflow. It stops if the contributions do not add up.
- The SHAP share and the gain share rank the inputs alike and differ in size, because gain weighs busy hours more and the log scale weighs every row the same.
- On Good Friday every input described an ordinary Friday. The model failed for the same reason as the baseline.
- SHAP explains the model and not the world, divides credit among similar inputs, and hides rare effects in a mean.

## Exercises

**1.** A row has a base value of 4.49 and three contributions: +0.69, −0.69 and +0.10. All other contributions are 0. What is the forecast?

<details>
<summary>Answer</summary>

The contributions add up to +0.10, so the sum is 4.59 and the forecast is `exp(4.59)`, about 98.5 pickups. In factors: 89.1 times 2, times one half, times 1.105. The doubling and the halving cancel each other. This is the reason for adding on the log scale first and applying `exp` once at the end.

</details>

**2.** A colleague reads Figure 28.2: "`is_holiday` has 0.4%. Remove it, the model does not need it." What do you answer?

<details>
<summary>Answer</summary>

The 0.4% is a mean over 5,000 rows, and 4,902 of them are not holidays. On the 98 holiday rows the flag moves the forecast more, on average, than `lag_48h` does in a typical row (0.164 against 0.133). So the model does use it where it applies. Whether it uses it well is a different question, and neither number answers it: the model saw two holidays, and no test row of the backtest is a holiday (Chapter 23). The honest position is that the flag is cheap, plausible and untested. The decision to remove an input is made with a backtest and several seeds (Chapter 25), not with an importance table.

</details>

**3. Interview question.** "How do you explain a single prediction of a tree model to a stakeholder?"

<details>
<summary>Points of a good answer</summary>

- With SHAP values: the prediction is a base value plus one contribution per input, and the parts add up exactly.
- Say which scale the contributions are on. With a log link they are factors, and they should be shown as factors.
- Show a concrete row, with the values of the inputs next to their contributions.
- Separate the two uses: one row explains a forecast, and the mean absolute value over many rows describes the model.
- Name the limits: no causal claim, shared credit between correlated inputs, means that hide rare effects, and an explanation of a wrong forecast is still an explanation of the model.
- Check the tool: assert that the contributions add up to the output of the model.

</details>

---

[Previous: Chapter 27 – Tuning: a search, and how much of its score to believe](27-tuning-and-how-much-to-believe.md) | [Contents](README.md) | [Next: Appendix A – Stack map](appendix-a-stack-map.md)
