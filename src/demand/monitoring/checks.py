"""What monitoring checks, against what, and which results are alerts or warnings.

Monitoring answers one question every day: *is the champion still working as it did when
it was chosen?* It looks at three things, each with a number, a threshold from
``monitoring:`` in the config, and a yes/no. Only the accuracy check raises an *alert*;
the two drift checks raise *warnings* (see "Alerts and warnings" at the end). It reads
only what the replay wrote (``flows/forecast_flow.py``) and the gold feature table, and
only what existed at its "now" (``flows/monitoring_flow.py``).

**1. Input drift: do the model inputs still look like the ones it learned from?**

* *The window* is the model inputs of the last ``window_days`` forecast days that were
  issued by "now" (``data/predictions/inputs/``): exactly the rows the model was given,
  20 zones x 24 hours x 7 days = 3,360 rows. They are known the moment a forecast is
  issued, so this check needs no actuals: its window reaches the day after tomorrow,
  two days further than the accuracy check can see.
* *The reference* is the same columns in the last ``reference_days`` of the training
  period (``data/gold/features.parquet``; with the config, 3 to 30 April, 13,440 rows):
  what the model saw last before it was frozen.
* *Which columns.* Of the 16 model inputs, 10 are tested: the demand lags, the rolling
  means and standard deviations, and the weather. The six others are left out because
  a test on them can only ever give one answer, whatever happens in the world
  (``UNTESTED`` says it per column). ``zone_id``, ``hour``, ``day_of_week`` and
  ``is_weekend`` have the same share of rows in every whole week, window and reference
  alike: 20 zones, 24 hours, each weekday once. ``month`` differs in *every* window (May
  is not April). ``is_holiday`` is a fact of the calendar, known a year ahead; the report
  lists the holidays of the window instead (the reference, 3 to 30 April, has none; the
  whole training period has two). Counting columns that cannot drift would lower the
  share of drifted columns, and one that always drifts would raise it, both for no
  reason.
* *What drift in a lag means.* ``lag_48h`` is the pickups of the same zone two days
  earlier, and the rolling features summarise the pickups of the last day or week. Their
  distribution over a week of rows is the distribution of recent demand. If it moves,
  demand itself has moved (a summer slowdown, a holiday week), and the model now sees
  inputs in a range it saw less of. That is not an error by itself: a model that learned
  "next Tuesday is like last Tuesday" can follow a slow change. Weather moves with the
  seasons in the same way: June is warmer than April every year. So input drift is a
  warning that the model is outside the conditions it was tested in, not a measure of
  how wrong it is; the accuracy check below is that measure.
* *The test: Wasserstein distance, fixed, threshold 0.1, for every column.* Evidently
  would choose a test by itself, from the size of the reference and the number of
  distinct values: for a reference of 1,000 rows or fewer the Kolmogorov-Smirnov test
  (chi-square or Z for five distinct values or fewer), above it the Wasserstein distance
  (Jensen-Shannon for five or fewer, which ``snowfall`` has in April to June: four).
  Left to choose, the same code would run another test in a unit test (small tables)
  than on the real data, and another on snowfall than on rain, so the choice is fixed.
  And a test of significance is the wrong tool for this data anyway: the hours of a
  zone are not independent draws (one busy evening fills several rows), so its p-value
  takes 3,360 rows for 3,360 independent observations and calls the smallest difference
  "significant". The Wasserstein distance measures *how far* the values moved instead:
  the average distance a value has to travel to turn one distribution into the other,
  divided by the standard deviation of the reference (Evidently's "normed" version). 0.1
  means a tenth of a standard deviation, Evidently's default for this test. It does not
  grow with the number of rows. A column has drifted when the distance is 0.1 or more.
* *The share and the verdict.* The share of drifted columns is drifted / tested, here
  out of 10. The inputs count as drifted (``drifted``, a warning) when the share is
  ``drift_share_threshold`` (0.3) or more, that is three columns or more of the ten.
  The comparison is "or more" because it is Evidently's rule for its own share
  (``DriftedColumnsCount``; a unit test checks that the two shares agree).

**2. Prediction drift: do the forecasts still look like they did in the backtest?**

The distribution of the window's forecasts, against forecasts of the same recipe that
were *not* fitted on the hours they forecast. The champion's own forecasts for April
cannot be the reference: it was fitted on every hour of April, so on April it repeats
what it learned (``flows/scoring.py``), and its forecasts there look better behaved than
any forecast it makes live.

The out-of-sample forecasts of the backtest are the honest reference: the four test
weeks before the end of training (3 to 30 April), each forecast by a model fitted only on
the weeks before it. The backtest stores their scores, not the forecasts themselves, so
``reference.py`` computes them again: the same folds (``models/backtest.py``), the same
recipe (``models/gbm.py``, deterministic, seeded). Whether they really are the forecasts
behind the champion's backtest MAE is checked, not assumed: their MAE must equal the
backtest MAE recorded for the champion in the registry, to the six decimals stored. If
the config, the data or the code no longer match the champion, the MAE differs, and this
check is not evaluated (its ``note`` says why) instead of comparing against forecasts of
some other model. The four fits take one to two minutes, on every run of the command
(nothing caches them).

The test is the same as for the inputs: Wasserstein distance on ``forecast``,
threshold 0.1. One column, so no share: the forecasts count as drifted (``drifted``, a
warning) when the distance is 0.1 or more. It uses the days of the window forecast by
the newest model version only, and needs ``window_days`` of them; it is the same
reference period as the input drift with this config (``backtest.n_folds`` x
``backtest.fold_days`` = 28 days = ``reference_days``), by choice of the config, not by
a rule of the code.

**3. Accuracy: is the champion as accurate as its backtest promised?**

The pooled MAE of the last ``window_days`` days that are over (``data/predictions/
scored/``), divided by the backtest MAE that training recorded for the model version
(the registry, as the replay report does). The check raises an alert when the ratio is
``mae_degradation_ratio`` (1.25) or more: live forecasts 25 % worse than the backtest.
The weekly baseline (``lag_168h``) is scored on the same rows, for context: if it got
worse by as much, the week was hard to forecast for anyone, and a retrained model may
not do better. Only days of the model version of the newest scored day are used, and the
check needs ``window_days`` of them: right after a new champion, its accuracy is not
judged on a few days, nor on days of the model before it.

**Alerts and warnings.** An *alert* means "act on this": it is what the retraining
feature reads (``alerts`` of the report, and ``alert``). Only the accuracy check raises
one. A *warning* means "look at this": the two drift checks raise warnings (``warnings``
and ``warning``), which nothing acts on. Each is an entry ``{check, reason}``: which check,
and a sentence with the number and the threshold. Each check keeps its own flag:
``drifted`` for the drift checks, ``alert`` for accuracy. A check that could not be
evaluated (too few days, no matching reference) raises nothing and says why in ``note``.
No message is sent (no email, no chat); the flags also go to Prometheus (``metrics.py``).

Why drift only warns, decided after the replay of May and June had been run:

* *Input drift carries no information as a trigger.* The same test, run on weeks inside
  the training period (each against the 28 days before it), crosses the 0.3 share in 10
  of 11 weeks, mostly through the weather: one week of weather is never like the four
  before it. In the replay it crossed on every day it could be evaluated.
* *Prediction drift crossed once*, at 0.100032 against a threshold of 0.1 (as of 21
  May), and never again: that looks like noise, not a signal.
* *The accuracy alert separated exactly the two bad periods* of the replay: the weeks
  ending 26 to 30 May (up to 1.41 times the backtest MAE) and 24 to 25 June.
* *No threshold is tuned after the replay was seen.* Changing 0.1, 0.3 or 1.25 now, to
  fit May and June, would make the replay a test the thresholds were chosen to pass. So
  they stay as they are, and what changed is only what the drift checks are allowed to
  do.

Drift is still computed, reported and drawn in Grafana: it explains an accuracy alert
after the fact (the heat of late June, warmer than anything in the training data, shows
as ``temperature_2m`` drifting), even though it never triggers anything.

**What monitoring does not decide.** Whether to retrain, when, and on what: that is the
retraining feature, which reads this report. Monitoring changes no model, no alias, no
threshold and no file of the replay.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from pydantic import BaseModel, ConfigDict

from demand.config import Settings
from demand.models.backtest import score
from demand.schemas import (
    ACTUAL,
    DAY_OF_WEEK,
    FORECAST,
    HOUR,
    IS_HOLIDAY,
    IS_WEEKEND,
    MODEL_VERSION,
    MONTH,
    ZONE_ID,
    feature_columns,
)

DRIFT_METHOD = "wasserstein"
"""Evidently's name for the normed Wasserstein distance."""
DRIFT_THRESHOLD = 0.1
"""A column has drifted at this distance or more: a tenth of a reference standard deviation."""
INPUT_DRIFT = "input_drift"
PREDICTION_DRIFT = "prediction_drift"
ACCURACY = "accuracy"
DAY = "day"
BASELINE = "baseline"
_DECIMALS = 6
_RATIO_DECIMALS = 4

_SAME_IN_EVERY_WEEK = "the same share of rows in every whole week, so it cannot drift"
UNTESTED = {
    ZONE_ID: _SAME_IN_EVERY_WEEK,
    HOUR: _SAME_IN_EVERY_WEEK,
    DAY_OF_WEEK: _SAME_IN_EVERY_WEEK,
    IS_WEEKEND: _SAME_IN_EVERY_WEEK,
    MONTH: "differs from the reference in every window: a fact of the calendar",
    IS_HOLIDAY: "a fact of the calendar, known in advance; see holidays_in_window",
}
"""The model inputs that are not drift-tested, and why."""


class ColumnDrift(BaseModel):
    """The drift of one column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    column: str
    distance: float
    """Normed Wasserstein distance between the window and the reference."""
    drifted: bool


class InputDriftCheck(BaseModel):
    """Check 1: the model inputs of the window against the end of the training period."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated: bool
    note: str | None
    """Why the check was not evaluated; ``None`` when it was."""
    window_first_day: date | None
    window_last_day: date | None
    window_rows: int
    reference_first_day: date
    reference_last_day: date
    reference_rows: int
    method: str
    column_threshold: float
    columns: list[ColumnDrift]
    untested: dict[str, str]
    drifted_columns: int | None
    drift_share: float | None
    share_threshold: float
    holidays_in_window: list[date]
    drifted: bool | None
    """Share of drifted columns at the threshold or more; ``None`` if not evaluated."""


class PredictionDriftCheck(BaseModel):
    """Check 2: the forecasts of the window against the backtest's forecasts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated: bool
    note: str | None
    model_version: str | None
    window_first_day: date | None
    window_last_day: date | None
    window_rows: int
    reference: str
    reference_rows: int | None
    reference_mae: float | None
    """MAE of the recomputed backtest forecasts."""
    recorded_backtest_mae: float | None
    """The backtest MAE in the registry; the reference is used only if the two agree."""
    method: str
    threshold: float
    distance: float | None
    drifted: bool | None
    """Distance at the threshold or more; ``None`` if not evaluated."""


class AccuracyCheck(BaseModel):
    """Check 3: live MAE of the last days that are over, against the backtest MAE."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluated: bool
    note: str | None
    model_version: str | None
    first_day: date | None
    last_day: date | None
    days: int
    rows: int
    live_mae: float | None
    baseline_mae: float | None
    """The weekly baseline on the same rows, for context."""
    backtest_mae: float | None
    live_to_backtest_mae: float | None
    ratio_threshold: float
    alert: bool


class Finding(BaseModel):
    """A check that crossed its threshold, and the sentence that says so."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check: str
    reason: str


class MonitoringReport(BaseModel):
    """Every check at one simulated moment. No wall-clock time: the same files, same report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    now: datetime
    """The simulated moment of the evaluation. Nothing newer was read."""
    window_days: int
    reference_days: int
    input_drift: InputDriftCheck
    prediction_drift: PredictionDriftCheck
    accuracy: AccuracyCheck
    alerts: list[Finding]
    """Accuracy below its threshold: what the retraining feature acts on."""
    warnings: list[Finding]
    """Input or prediction drift: reported, never acted on (the module docstring says why)."""
    alert: bool
    """Whether there is an alert."""
    warning: bool
    """Whether there is a warning."""


# --------------------------------------------------------------------------------------
# Drift, with Evidently
# --------------------------------------------------------------------------------------


def drift_columns(settings: Settings) -> list[str]:
    """The model inputs that are drift-tested: every one not in ``UNTESTED``."""
    return [column for column in feature_columns(settings) if column not in UNTESTED]


def drift_snapshot(window: pd.DataFrame, reference: pd.DataFrame, columns: Sequence[str]) -> Any:
    """Run Evidently's drift test on ``columns``; the result, also for its HTML page.

    Every column is treated as a number and tested with ``DRIFT_METHOD`` at
    ``DRIFT_THRESHOLD`` (the module docstring says why the test is fixed).
    """
    definition = DataDefinition(numerical_columns=list(columns))
    report = Report(
        [
            ValueDrift(column=column, method=DRIFT_METHOD, threshold=DRIFT_THRESHOLD)
            for column in columns
        ]
    )
    # A column that is the same in every row (snowfall in summer) makes Evidently's own
    # summary statistics divide by a standard deviation of 0, and numpy warns. The drift
    # test itself handles it (it divides by at least 0.001), so the warning is noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        return report.run(
            Dataset.from_pandas(window[list(columns)], data_definition=definition),
            Dataset.from_pandas(reference[list(columns)], data_definition=definition),
        )


def column_drifts(snapshot: Any) -> list[ColumnDrift]:
    """The distance and verdict of every column in a result of ``drift_snapshot``.

    A column has drifted at ``DRIFT_THRESHOLD`` or more, which is Evidently's rule for
    this test.
    """
    drifts = []
    for metric in snapshot.dict()["metrics"]:
        distance = round(float(metric["value"]), _DECIMALS)
        drifts.append(
            ColumnDrift(
                column=str(metric["config"]["column"]),
                distance=distance,
                drifted=distance >= DRIFT_THRESHOLD,
            )
        )
    return drifts


def drift_share(drifts: Sequence[ColumnDrift]) -> float:
    """The share of drifted columns: drifted / tested."""
    return round(sum(drift.drifted for drift in drifts) / len(drifts), _DECIMALS)


def holidays(inputs: pd.DataFrame) -> list[date]:
    """The days of ``inputs`` flagged as holidays, oldest first."""
    flagged = inputs.loc[inputs[IS_HOLIDAY], DAY]
    return sorted(set(flagged.tolist()))


def input_drift_check(
    window: pd.DataFrame | None,
    reference: pd.DataFrame,
    settings: Settings,
    note: str | None,
) -> tuple[InputDriftCheck, Any]:
    """Check 1, and Evidently's result for the HTML page (``None`` if not evaluated).

    Args:
        window: The model inputs of the window, with a ``day`` column; ``None`` if there
            are too few days, then ``note`` says why.
        reference: The training rows of the reference period.
        settings: For the columns and the threshold.
        note: Why ``window`` is ``None``.
    """
    threshold = settings.monitoring.drift_share_threshold
    common = {
        "reference_first_day": reference[DAY].min(),
        "reference_last_day": reference[DAY].max(),
        "reference_rows": len(reference),
        "method": DRIFT_METHOD,
        "column_threshold": DRIFT_THRESHOLD,
        "untested": UNTESTED,
        "share_threshold": threshold,
    }
    if window is None:
        check = InputDriftCheck(
            evaluated=False,
            note=note,
            window_first_day=None,
            window_last_day=None,
            window_rows=0,
            columns=[],
            drifted_columns=None,
            drift_share=None,
            holidays_in_window=[],
            drifted=None,
            **common,
        )
        return check, None
    snapshot = drift_snapshot(window, reference, drift_columns(settings))
    drifts = column_drifts(snapshot)
    share = drift_share(drifts)
    check = InputDriftCheck(
        evaluated=True,
        note=None,
        window_first_day=window[DAY].min(),
        window_last_day=window[DAY].max(),
        window_rows=len(window),
        columns=drifts,
        drifted_columns=sum(drift.drifted for drift in drifts),
        drift_share=share,
        holidays_in_window=holidays(window),
        drifted=share >= threshold,
        **common,
    )
    return check, snapshot


def prediction_drift_check(
    window: pd.DataFrame | None,
    reference: pd.DataFrame | None,
    reference_mae: float | None,
    recorded_mae: float | None,
    note: str | None,
) -> PredictionDriftCheck:
    """Check 2.

    Args:
        window: The forecasts of the window (one model version), with a ``day`` column;
            ``None`` if there are too few days, then ``note`` says why.
        reference: The recomputed backtest forecasts; ``None`` if they were not made.
        reference_mae: Their MAE.
        recorded_mae: The backtest MAE in the registry for the version of the window.
        note: Why ``window`` or ``reference`` is ``None``.
    """
    if note is None and recorded_mae is None:
        note = (
            "The registry holds no backtest MAE for this version, so the reference cannot "
            "be checked."
        )
    elif note is None and reference_mae != recorded_mae:
        note = (
            f"The recomputed backtest forecasts have an MAE of {reference_mae}; the backtest "
            f"of this version recorded {recorded_mae}. The config, the data or the code no "
            "longer match the version, so these are not the forecasts of its backtest."
        )
    distance = None
    if note is None and window is not None and reference is not None:
        (drift,) = column_drifts(drift_snapshot(window, reference, [FORECAST]))
        distance = drift.distance
    drifted = None if distance is None else distance >= DRIFT_THRESHOLD
    return PredictionDriftCheck(
        evaluated=distance is not None,
        note=note,
        model_version=None if window is None else str(window[MODEL_VERSION].iloc[0]),
        window_first_day=None if window is None else window[DAY].min(),
        window_last_day=None if window is None else window[DAY].max(),
        window_rows=0 if window is None else len(window),
        reference="the backtest's out-of-sample forecasts of its test weeks, recomputed",
        reference_rows=None if reference is None else len(reference),
        reference_mae=reference_mae,
        recorded_backtest_mae=recorded_mae,
        method=DRIFT_METHOD,
        threshold=DRIFT_THRESHOLD,
        distance=distance,
        drifted=drifted,
    )


# --------------------------------------------------------------------------------------
# Accuracy
# --------------------------------------------------------------------------------------


def accuracy_check(
    rows: pd.DataFrame | None, backtest_mae: float | None, settings: Settings, note: str | None
) -> AccuracyCheck:
    """Check 3.

    Args:
        rows: The scored rows of the window (one model version): ``day``, ``actual``,
            ``forecast``, ``baseline`` and ``model_version``; ``None`` if there are too
            few days.
        backtest_mae: The backtest MAE in the registry for that version.
        settings: For the threshold.
        note: Why ``rows`` is ``None``.
    """
    threshold = settings.monitoring.mae_degradation_ratio
    if rows is None or backtest_mae is None:
        return AccuracyCheck(
            evaluated=False,
            note=note or "The registry holds no backtest MAE for this version.",
            model_version=None if rows is None else str(rows[MODEL_VERSION].iloc[0]),
            first_day=None if rows is None else rows[DAY].min(),
            last_day=None if rows is None else rows[DAY].max(),
            days=0 if rows is None else rows[DAY].nunique(),
            rows=0 if rows is None else len(rows),
            live_mae=None,
            baseline_mae=None,
            backtest_mae=backtest_mae,
            live_to_backtest_mae=None,
            ratio_threshold=threshold,
            alert=False,
        )
    actual = rows[ACTUAL].to_numpy(dtype="float64")
    live = score(actual, rows[FORECAST].to_numpy(dtype="float64")).mae
    baseline = score(actual, rows[BASELINE].to_numpy(dtype="float64")).mae
    ratio = round(live / backtest_mae, _RATIO_DECIMALS)
    return AccuracyCheck(
        evaluated=True,
        note=None,
        model_version=str(rows[MODEL_VERSION].iloc[0]),
        first_day=rows[DAY].min(),
        last_day=rows[DAY].max(),
        days=rows[DAY].nunique(),
        rows=len(rows),
        live_mae=live,
        baseline_mae=baseline,
        backtest_mae=backtest_mae,
        live_to_backtest_mae=ratio,
        ratio_threshold=threshold,
        alert=ratio >= threshold,
    )


# --------------------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------------------


def warnings_of(inputs: InputDriftCheck, predictions: PredictionDriftCheck) -> list[Finding]:
    """One warning for each drift check whose threshold was crossed."""
    warnings = []
    if inputs.drifted:
        drifted = [drift.column for drift in inputs.columns if drift.drifted]
        warnings.append(
            Finding(
                check=INPUT_DRIFT,
                reason=(
                    f"{inputs.drifted_columns} of {len(inputs.columns)} model inputs drifted "
                    f"(share {inputs.drift_share}, threshold {inputs.share_threshold}): "
                    f"{', '.join(drifted)}."
                ),
            )
        )
    if predictions.drifted:
        warnings.append(
            Finding(
                check=PREDICTION_DRIFT,
                reason=(
                    f"The forecasts of version {predictions.model_version} drifted from the "
                    f"backtest's: distance {predictions.distance}, threshold "
                    f"{predictions.threshold}."
                ),
            )
        )
    return warnings


def alerts_of(accuracy: AccuracyCheck) -> list[Finding]:
    """The accuracy alert, if its threshold was crossed: the only check that alerts."""
    if not accuracy.alert:
        return []
    return [
        Finding(
            check=ACCURACY,
            reason=(
                f"Live MAE {accuracy.live_mae} over {accuracy.first_day} to "
                f"{accuracy.last_day} is {accuracy.live_to_backtest_mae} times the "
                f"backtest MAE {accuracy.backtest_mae} (threshold "
                f"{accuracy.ratio_threshold}); the weekly baseline had "
                f"{accuracy.baseline_mae} on the same rows."
            ),
        )
    ]


def monitoring_report(
    now: pd.Timestamp,
    settings: Settings,
    inputs: InputDriftCheck,
    predictions: PredictionDriftCheck,
    accuracy: AccuracyCheck,
) -> MonitoringReport:
    """The report of the three checks at ``now``."""
    alerts = alerts_of(accuracy)
    warnings = warnings_of(inputs, predictions)
    return MonitoringReport(
        now=now.to_pydatetime(),
        window_days=settings.monitoring.window_days,
        reference_days=settings.monitoring.reference_days,
        input_drift=inputs,
        prediction_drift=predictions,
        accuracy=accuracy,
        alerts=alerts,
        warnings=warnings,
        alert=bool(alerts),
        warning=bool(warnings),
    )
