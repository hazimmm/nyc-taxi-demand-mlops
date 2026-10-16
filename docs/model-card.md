# Model card: taxi-demand, version 2

This page describes version 2 of the registered model `taxi-demand`. It is the version the promotion gate would make champion. It was **written by hand** on 17 October 2026 from the files named below. Every number carries a mark that says where it comes from, so a reviewer can check it. When another version becomes champion, this page has to be written again; nothing updates it automatically.

| Mark | Source |
|---|---|
| [T] | `reports/training_report.json` of the run that produced version 2 |
| [B] | `reports/backtest_report.json` of the same run |
| [M] | the metrics and parameters MLflow stored with that run (`a8aa7b30…`) |
| [U] | `reports/tuning_report.json` (the search that chose the parameters) |
| [S] | `reports/shap_importance.json` |
| [Q] | a scratch query: the backtest re-run in memory with the same config, which reproduces the MAE of [T] exactly (24.106025), with each row's forecast kept. It read only rows up to 30 April 2025. |
| [D] | measured by hand earlier and written down in the docstrings of `src/demand/models/gbm.py`, `tuning.py` and `training.py`. Not re-run for this page. |
| [C] | `configs/config.yaml` at commit `5d1d9e4` |

## What it does

For each of 20 NYC taxi zones it forecasts how many yellow-taxi pickups each hour of a day will have. The forecast for the whole day is issued at 00:00 on the day before. No input is newer than 48 hours before the hour it forecasts [C]. The output is an expected count, not a whole number: 2.4 means "2 in some such hours, 3 in others".

- **Model:** LightGBM, 600 trees, Poisson objective, learning rate 0.104, 110 leaves, at least 161 rows per leaf [C][M]. The zone is treated as a category, and forecasts are never negative [D].
- **Registered as** `taxi-demand` version 2. Run `a8aa7b305856481b984bd97586d2f549`, from git commit `5d1d9e4` with no changed files, on data version `7bacf1bf80cf4612d40552e74ef11cc5.dir` [T].

## Intended use

- To plan supply ahead of time in the 20 zones it was trained on, for example how many cabs to have near Midtown or JFK tomorrow evening. Use it as a planning number, read next to the error figures below.
- As the subject of a portfolio MLOps project: its training, tracking, promotion and monitoring are the point, as much as the forecast.

## Out of scope

- **Other zones, other fleets.** It knows 20 zones: 18 in Manhattan, plus JFK and LaGuardia airports (zone table in `data/raw/zones`). It refuses any other zone [D]. It has seen yellow taxis only, not green taxis or ride-hailing apps.
- **Other horizons.** It does not give same-day updates, and nothing beyond tomorrow.
- **Individual decisions.** It does not judge, pay or rank drivers, and it sets no fares or surge prices. It has no information about any person.
- **Unusual days.** Events, strikes, storms outside the weather inputs, and holidays the calendar feature does not know (see weaknesses).
- **After June 2025 without monitoring.** Demand changes, and the model has not been checked on any day after 30 April 2025.

## Data

- **Source:** NYC TLC yellow-taxi trip records, cleaned and counted per zone and hour, plus hourly weather from Open-Meteo [C].
- **Training rows:** 53,280 = 20 zones × 2,664 hours. They run from 9 January 2025 23:00, the first hour with a full history for every feature, to 30 April 2025 23:00 [T]. May and June 2025 are held back for the replay and were not read.
- **Zones:** the 20 busiest of the training period, frozen in the config [C].
- **Inputs (16):** zone; pickups 48 and 168 hours earlier; mean and spread of the pickups over 24 and 168 hours, ending 48 hours back; hour, day of week, month, weekend, US federal holiday; temperature, precipitation, snowfall, wind speed [M].
- **What it leans on:** pickups a week earlier carry 49.4 % of the mean absolute SHAP value, the hour 17.8 %, pickups two days earlier 10.0 %, the zone 6.9 %, and the four weather inputs together 4.1 % [S].

## How it was evaluated

- **Backtest:** four test weeks, 3–9, 10–16, 17–23 and 24–30 April 2025, with 3,360 rows each and 13,440 in total [B]. For each week a model is fitted from scratch on every row up to 48 hours before the week starts (an expanding window with a gap), so it never learns from pickups it could not have known [B][D].
- **Baseline:** "same hour last week" (`lag_168h`), scored on the same rows [B].
- **Scores, pooled over all 13,440 rows** [T]:

| | MAE | RMSE | sMAPE |
|---|---|---|---|
| version 2 | **24.11** | 38.24 | 21.9 % |
| weekly baseline | 27.70 | 44.51 | 24.6 % |

That is 13.0 % lower MAE than the baseline [T]. Version 2 is lower in every week [M]:

| Week | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| version 2 MAE | 23.51 | 21.85 | 24.75 | 26.31 |
| baseline MAE | 27.61 | 26.57 | 27.92 | 28.70 |

The parameters before tuning (version 1) scored 24.93 [M]. With three seeds the tuned parameters average 24.14, and the seed alone moves the score by 0.19 [U].

### What these scores do and do not prove

- **They are optimistic.** The same four April weeks chose the zone handling and the objective [D], and then the 30-trial parameter search kept the set that did best on them [U]. A winner picked on a set of weeks scores better on those weeks than it will elsewhere.
- **About 0.6 of the 0.9 tuning gain survived on unseen weeks.** A separate search on March weeks gained 0.64 when scored on April, against 0.90 for the search on April itself [D]. So the real gain from tuning is closer to 0.6 MAE than to the full 0.9.
- **The clean exam has not been taken.** The day-by-day replay of May and June 2025 is weeks no decision has looked at. It is not run yet. Until it is, the backtest is the best estimate available, and an optimistic one.

## Known weaknesses

- **Holidays outside the federal calendar.** `is_holiday` marks US federal holidays only [D]. The training rows contain two of them, 20 January and 17 February [Q], and the test weeks none. On Good Friday, 18 April, the MAE was 36.96 against 24.11 overall, with forecasts 27.6 pickups too high on average. On Saturday 19 April it was 32.86, worse than the baseline's 30.68 [Q]. Easter, Passover, school breaks and events are not inputs.
- **Busy weekends.** The last weekend of April (25–27 April) had MAEs of 29.8 to 36.3. Forecasts were 22 to 31 pickups too low on average [Q]. The data does not say why.
- **Zone 249 (West Village).** This is the one zone where the model is worse than the baseline: MAE 22.34 against 22.04 [B]. It is worse in three of the four weeks [Q].
- **LaGuardia (zone 138), in percentage terms.** sMAPE 48.2 % against the baseline's 44.1 %, the worst of all zones [B].
- **Quiet hours.** Hours with 0–4 pickups are 3.7 % of the test rows. There the MAE is 3.71, worse than the baseline's 3.10, and the sMAPE is 91 % [Q]. 91 test hours had no pickup at all [Q]. In busy zones the absolute error is larger: 41.9 pickups in hours with 200 or more [Q].
- **The weather assumption is unverified.** The backtest assumes the weather inputs are what a forecast issued up to 47 hours ahead would have said. The stored values come from Open-Meteo's *historical forecast* API (`data/raw/manifest.json`: `weather_source: forecast`). How far ahead those archived values were forecast was not checked. If they are closer to the event than 47 hours, the backtest gives the weather inputs too much credit. Their share of the model is small, 4.1 % [S], but the bias has not been measured.
- **1 May.** The final model learned from the pickups up to 30 April 23:00. So it exists only from 00:00 on 1 May, a day after the forecast for 1 May was due [D]. A replay score that includes 1 May has to say so, or start on 2 May.
- **Reproducibility is per machine.** Retraining gives the same model bit for bit on the same machine and with the same versions (LightGBM 4.7.0, Python 3.12.7). It is not promised elsewhere [D].

## Ethical and operational notes

- **No personal data.** The model sees counts of pickups per zone and hour, never a trip, a driver or a passenger.
- **A dispatcher should not:**
  - use it to send cabs away from a zone because its forecast is low. Fewer cabs there mean fewer pickups and a lower forecast next week, a loop that quietly under-serves an area.
  - trust it on holidays, event days or the airports' quiet hours without a person's judgement (see the weaknesses).
  - read a forecast as a promise. The typical error is 24 pickups per zone and hour, and larger in busy zones.
- **Coverage is uneven by design.** 18 of the 20 zones are in Manhattan, so other boroughs are not served at all.

## How to reproduce it

From the project folder, with the virtual environment active:

```
git checkout 5d1d9e4
dvc checkout     # the data of that commit: gold version 7bacf1bf80cf4612d40552e74ef11cc5.dir
dvc repro        # rebuilds what is out of date and runs `train`: one new model version
```

On the same machine and the same library versions, the new version has the numbers above exactly, in `reports/training_report.json` [D]. To score without recording anything in MLflow, run `python -m demand backtest` instead of `dvc repro`. The promotion gate is not in that commit. Run back on the branch, `python -m demand promote` would reject the copy against version 2 as a tie (gain 0).
