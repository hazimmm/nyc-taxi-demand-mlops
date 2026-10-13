# Chapter 25 – Deciding by measurement: four choices and the noise of a seed

The backtest report says that the model has an MAE of 24.93. Change one number in the config, `seed: 42` to `seed: 43`, and nothing else. The MAE is 25.05. With 44 it is 25.12. The data, the code and the parameters are the same in all three runs. Now a colleague proposes a change to the model and shows that it lowers the MAE from 25.04 to 25.00. Is that an improvement? This chapter gives the rule that answers the question, and applies it to four real choices. One of them was a narrow decision.

## What you will learn

- What a seed is, and what makes two runs of this model identical.
- What was tested about this, and what was not.
- How large the noise of a score is, and why one run is not a measurement.
- The rule that decided four choices, and the numbers behind each of them.
- Why an input that helps on average was not added yet.
- What the backtest scores are worth after they were used to choose.

## The same input gives the same model

Chapter 24 said that every tree is built from a random 80% of the rows and of the columns. A computer cannot choose by chance. It calculates a long sequence of numbers that look random, starting from one number, the **seed**. The same seed gives the same sequence, so the same rows and columns are picked.

The forecaster takes its seed from `project.seed` in the config. It refuses a seed inside `model.params`, so that there is one place for it. It also sets two switches of LightGBM:

[`src/demand/models/gbm.py`](../../src/demand/models/gbm.py)

```python
REPEATABLE = {"deterministic": True, "force_row_wise": True}
```

`deterministic` tells LightGBM to prefer repeatable arithmetic over speed. `force_row_wise` is needed with it: without it LightGBM measures at the start of every fit which of two ways to build its tables is faster, and the answer depends on how busy the machine is.

A process is **deterministic** when the same input always gives the same output. Be exact about how far this was shown.

| Claim | Status |
|---|---|
| Two fits on the same rows, config and seed give the same trees and forecasts | Tested: a test compares the two models as text |
| The number of threads does not change the forecasts | Tested with 1 and 3 threads; the docstring of `gbm.py` also reports a measurement with 1, 4 and 12 |
| The backtest report has the same bytes after every run on this machine | Checked for this chapter: a new run gave a file identical to the stored one |
| The same holds on another processor, another operating system or another version of LightGBM | Not tested, and not promised |

*Figure 25.1 – What "the same model" means here.*

The last row has a reason. Another machine can add decimal numbers in another order. The last digits then differ, and now and then a tree asks a different question.

Determinism has a practical use. DVC compares the report by checksum (Chapter 21), and a report that differed in every run would look like a change every time.

## The noise of a score

Determinism makes one run repeatable. It does not make that run special. Every other seed gives an equally good, slightly different model.

| Seed | MAE on the four test weeks |
|---|---|
| 42 | 24.93 |
| 43 | 25.05 |
| 44 | 25.12 |

*Figure 25.2 – Three runs of the same model. The mean is 25.04 and the range is 0.20.*

The differences between these runs are **noise**: movement of a score that comes from chance and not from a difference in the model. The range of three seeds, 0.20 pickups, is a rough measure of it. Three seeds are few, so the true noise may be larger.

This answers the opening. A change from 25.04 to 25.00 is five times smaller than the noise. It cannot be told apart from another seed.

The rule that was applied to every choice in this chapter has two parts:

> [!IMPORTANT]
> Keep a change only if its gain is clearly above the seed noise, and if it holds across the folds.

The second part guards against a gain that comes from one week only.

## Four choices

The modeller of this project measured each choice on the four test weeks of Chapter 23, three times, with the seeds 42, 43 and 44. For this chapter every line of the table was run again, in a scratch script outside the repository that reads no row after 30 April, and gave the same numbers.

| Setting | MAE, mean of 3 seeds | Range of the seeds | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|---|
| Weekly baseline | 27.70 | – | 27.61 | 26.57 | 27.92 | 28.70 |
| Zone as a number | 25.96 | 0.27 | 25.20 | 23.71 | 26.60 | 28.35 |
| **Zone as a category (chosen)** | 25.04 | 0.20 | 24.37 | 22.80 | 25.96 | 27.02 |
| Chosen, without the outlier rows in training | 25.00 | 0.07 | 24.45 | 22.98 | 25.72 | 26.85 |
| Chosen, with the plain regression objective | 25.17 | 0.12 | 24.47 | 23.33 | 26.25 | 26.63 |
| Chosen, on the shorter table of the next line | 25.16 | 0.28 | 24.56 | 22.92 | 25.98 | 27.18 |
| Chosen, with `lag_336h` added | 24.52 | 0.09 | 23.85 | 23.60 | 24.46 | 26.16 |

*Figure 25.3 – Every fold column is the mean of three seeds. The last two lines use 50,860 rows and not 53,280; the reason follows below.*

**The zone as a category: kept.** The gain is 0.93 pickups, more than three times the noise. It appears in every fold, between 0.64 and 1.33. Both parts of the rule hold.

**Outlier rows stay in training: kept as it was.** Chapter 18 flagged unusual hours and kept them. Leaving them out of the training rows gives 25.00 against 25.04. The difference is 0.04, far inside the noise, and two folds are better and two are worse. There are only 56 such rows among 53,280, and they really happened. So nothing changes.

**The Poisson objective: kept.** Plain regression (squared error) gives an MAE of 25.17 against 25.04. That is a tie: the difference is smaller than the noise, and fold 4 goes the other way. Two other facts decide. The sMAPE is 25.0% with plain regression and 22.8% with Poisson, a difference far above the movement between seeds. The sMAPE is the metric that quiet hours decide (Chapter 22). And plain regression produced negative forecasts: 69 in the three runs together, which had to be cut off at 0.

## The fourth choice: the same hour two weeks ago

The owner of this project proposed one more input, `lag_336h`: the pickups of the same zone at the same hour two weeks earlier. The reasoning was this. The pickups of two days ago come from a different kind of day: for a Monday they are from a Saturday. The same day of the week is the better guide. So a second week of the same weekday should help.

The reasoning is correct, and the model agrees with it: `lag_168h` carries 76% of the gain and `lag_48h` less than 6% (Figure 24.4). The open question is whether a second week adds enough to what the first week already gives.

**The measurement.** A longer lag needs a longer history. The warm-up of Chapter 19 grows from 215 to 336 hours, so 121 more hours of every zone cannot become training rows: 2,420 rows. A fair comparison uses the same rows on both sides, which is why Figure 25.3 has the line "on the shorter table". On those rows the MAE falls from 25.16 to 24.52. The gain is 0.65, more than twice the noise. The first part of the rule holds.

The second part does not. Folds 1, 3 and 4 are better by 0.71, 1.52 and 1.02. Fold 2 is worse by 0.67.

**A wider look.** One worse week in four may be chance. So the same comparison was run over eight test weeks instead of four.

| Test week | Without | With `lag_336h` | Difference |
|---|---|---|---|
| 6 to 12 March | 30.80 | 28.81 | −1.99 |
| 13 to 19 March | 28.93 | 32.30 | +3.37 |
| 20 to 26 March | 24.88 | 24.39 | −0.50 |
| 27 March to 2 April | 25.11 | 24.68 | −0.43 |
| 3 to 9 April | 24.56 | 23.85 | −0.71 |
| 10 to 16 April | 22.92 | 23.60 | +0.67 |
| 17 to 23 April | 25.98 | 24.46 | −1.52 |
| 24 to 30 April | 27.18 | 26.16 | −1.02 |

*Figure 25.4 – Eight test weeks, mean of three seeds. Negative means the two-week lag helps. The difference is calculated before rounding.*

The lag helps in six of eight weeks. The mean gain is 0.27, less than half of what the four April weeks showed. And one week is worse by 3.37, which is larger than any gain.

That week, 13 to 19 March, comes directly after the clock change of 9 March. Lags count elapsed hours (Chapter 19). For two weeks after the change, 336 elapsed hours before 18:00 is 17:00 on the old clock, so the lag reads the wrong hour of the day. This may be the cause. It is a guess of the modeller. Nobody measured it, and the week before, which is affected on four of its days, was better with the lag.

**The costs.** 2,420 training rows are lost. And the serving code of a later phase would need two weeks of history for every zone before it can forecast, where it now needs nine days.

**The decision: not now.** The lag is not added in this feature. It will be measured again after the model parameters are tuned, because a tuned model may use it differently.

This decision was narrow, and it should be read as narrow. The average gain is above the noise. A reasonable person could add the lag and accept that some weeks become worse. The rule asks for a gain that holds across the folds, and this gain does not. What matters most is that the decision is written down with its numbers, so it can be opened again.

## One run is not a measurement

This project learned the lesson once already. An early trial, made before this feature, reported that LightGBM was "about 10% better than the baseline". That number came from a single run, at a time when the horizon was still 24 hours. Under the horizon of 48 hours, three seeds gave about 6% with the zone as a number (the 25.96 of Chapter 23) and about 10% with the zone as a category. The first "10%" and the last "10%" look the same and describe two different models on two different problems. A single number does not show how far it could move. Three runs with their range do.

## What the scores are worth now

Chapter 23 warned that folds stop being a clean test once they are used to choose. That has now happened four times, on the same four weeks. The score of the chosen settings is therefore a little too good, by an amount nobody knows.

Two honest statements remain. With seed 42 the model is 10.0% better than the baseline. As the mean of three seeds it is 9.6% better, and that is the fairer number. The clean test is the replay of May and June, which no decision has touched.

## Try it yourself

Run the backtest with three seeds. The command writes nothing and takes about half a minute.

```powershell
.venv\Scripts\python.exe -c "from demand.config import get_settings; from demand.models.backtest import backtest, load_training_features; from demand.models.gbm import LightGbmForecaster; from demand.schemas import feature_columns; s = get_settings(); t = load_training_features(s); [print(seed, backtest(t, {'m': LightGbmForecaster(feature_columns(s), s.model.params, seed)}, s).forecasters['m'].overall.mae) for seed in (42, 43, 44)]"
```

You should see:

```text
42 24.929918
43 25.05439
44 25.124457
```

These digits were produced on one Windows machine with LightGBM 4.7.0. On another machine the last digits may differ (Figure 25.1). The size of the differences between the seeds should be similar.

Compare the two objectives with seed 42. The two numbers are the MAE and the sMAPE.

```powershell
.venv\Scripts\python.exe -c "from demand.config import get_settings; from demand.models.backtest import backtest, load_training_features; from demand.models.gbm import LightGbmForecaster; from demand.schemas import feature_columns; s = get_settings(); t = load_training_features(s); [print(o, r.mae, r.smape) for o in ('poisson', 'regression') for r in [backtest(t, {'m': LightGbmForecaster(feature_columns(s), {**s.model.params, 'objective': o}, 42)}, s).forecasters['m'].overall]]"
```

You should see:

```text
poisson 24.929918 22.709391
regression 25.148985 24.980576
```

Run the three tests about repeatable results.

```powershell
.venv\Scripts\python.exe -m pytest tests/unit/models/test_gbm.py -q -k "same_rows or another_seed or threads"
```

You should see `3 passed, 28 deselected`. One of the three checks that another seed gives another model. Without it, the first test would also pass for a model that picks nothing at random.

Give the forecaster a seed inside its parameters: `.venv\Scripts\python.exe -c "from demand.models.gbm import LightGbmForecaster; LightGbmForecaster(['zone_id'], {'seed': 1}, 42)"`. The traceback ends with `model.params sets seed, but the seed of the model is project.seed (42). Remove seed from model.params in the config.`

## Key takeaways

- The seed fixes the random choices of training. With the same rows, config and seed, the model is the same on this machine. This is tested. Across machines and library versions it is not promised.
- Another seed gives another, equally good model. Here the MAE moves by about 0.2 pickups between three seeds. That movement is noise.
- A change is kept only if its gain is clearly above the noise and holds across the folds.
- The zone as a category passed both parts. Dropping outlier rows and changing the objective did not change the MAE beyond the noise, so the simpler or safer setting stayed.
- `lag_336h` helps on average and is worse in some weeks. It was not added, and it will be measured again after tuning. The decision was narrow.
- One run is not a measurement. Report the mean of several seeds and their range.
- The folds were used for four choices, so the scores are slightly too good. May and June are the clean test.

## Exercises

**1.** A change lowers the mean MAE of three seeds from 25.04 to 24.96. The range of the seeds is 0.20. All four folds are better, each by 0.05 to 0.10. Do you keep the change?

<details>
<summary>Answer</summary>

By the rule, no: a gain of 0.08 is well inside the noise of 0.20, so the first part fails. The four folds that agree are a reason to look closer, not a proof, because the four folds share most of their training rows and the same seeds, so they are not four independent measurements. The next step is to run more seeds, for example ten. If the gain stays near 0.08 while the mean becomes steadier, it is real and small. Then the question is whether a small gain is worth what the change costs in code.

</details>

**2.** Use Figure 25.4. A colleague says: "The lag helps in six of eight weeks, so add it." Another says: "It makes one week worse by 3.37, so never add it." What does each of them leave out?

<details>
<summary>Answer</summary>

The first leaves out the size of the losses and the costs. The mean gain over eight weeks is 0.27, close to the seed noise, and the lag costs 2,420 training rows and five more days of history at serving time. The second treats one week as a law. That week follows a clock change, and there is an untested explanation for it. If the explanation is right, the problem could perhaps be repaired. Both leave out that the measurement can be repeated after tuning. "Not now, and here are the numbers" keeps the question open.

</details>

**3. Interview question.** "You tried a new feature and the validation error went down. How do you decide whether to keep it?"

<details>
<summary>Points of a good answer</summary>

- Measure the noise first: train the unchanged model with several seeds and look at the range of the score.
- Compare means of several seeds, on the same rows for both variants. A new lag shortens the table, so the old variant must be run on the shorter table too.
- Look at every fold, not only at the total. A gain from one week is weak evidence.
- Count the costs: lost training rows, more history needed when serving, more code.
- Fix the rule before looking at the numbers, and write the decision down with the numbers, also when the answer is "not now".
- Remember that every choice made on the validation folds makes their score a little too good. Keep a period that no choice has touched.

</details>

---

[Previous: Chapter 24 – A model behind the same two methods: LightGBM as a forecaster](24-a-model-behind-the-same-two-methods.md) | [Contents](README.md) | [Next: Chapter 26 – The model that is kept: the final fit and MLflow](26-the-final-fit-and-mlflow.md)
