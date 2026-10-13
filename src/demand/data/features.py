"""The one function that builds model inputs, for training and for serving.

Everything in this module is pure: dataframes go in, a dataframe comes out. It reads no
files, and it does not know whether it is called for training or for a forecast.

**Why one function.** If training built its features one way and the forecast another,
the model would be asked questions it was never trained on, and no error would be
raised: the forecasts would just be worse. So ``build_features`` is the only place
where a model input is calculated. Training, the batch forecast, the API and the drift
monitor all call it.

**Its two inputs.** A forecast is made for hours whose pickups are not known yet. The
function therefore takes two tables:

* ``history``: zone, hour and pickups, for as far back as the features need. The past.
* ``targets``: zone and hour of the rows to build, with the weather of those hours. No
  pickups are needed, so these can be hours that have not happened.

Training passes the gold demand table as both (``build_feature_table``). A forecast for
tomorrow passes the demand known so far as ``history`` and tomorrow's hours with the
weather forecast as ``targets``.

**The features**, in the order of ``feature_columns``, for a target hour ``t`` and a
forecast horizon of ``h`` hours (``features.horizon_hours``; 48 in the config of this
project, and the examples below use that value):

* ``lag_<k>h``: the pickups of the same zone ``k`` hours before ``t``, for every ``k``
  in ``features.lags``. The config refuses a lag shorter than ``h``. With ``h = 48`` the
  lags are 48 and 168: the same hour two days earlier and the same hour a week earlier.
* ``roll_mean_<w>h`` and ``roll_std_<w>h``: the mean and the standard deviation of the
  ``w`` hourly values that end at ``t - h``, that is the hours ``t - h - w + 1`` to
  ``t - h``. For ``w = 24`` and ``h = 48`` these are the hours 71 to 48 before ``t``;
  for ``w = 168`` the hours 215 to 48 before ``t``. The standard deviation is the
  sample standard deviation (divided by ``w - 1``).
* ``hour``, ``day_of_week`` (Monday is 0), ``month``, ``is_weekend``, ``is_holiday``:
  read from the local clock time of ``t``.
* The weather of ``t`` itself. It is a weather forecast, so a value for ``t`` exists
  before ``t``. (Whether the stored value is what was forecast ``h`` hours ahead is a
  limit of the data and not of this function; see ``backtest.py``.)

**No leakage.** At the moment hour ``t`` is predicted, nothing newer than ``t - h`` is
known. The function is built so that it *cannot* read anything newer, whatever it is
given: every lag is at least ``h`` and every window ends at ``t - h``. Training passes
a history that goes far beyond each row's target hour, and that is safe because of this
and not because the caller was careful. Features are also calculated zone by zone; no
value of one zone enters a feature of another.

**When a row can be built.** The newest pickups the row of hour ``t`` reads are those
of the hour that starts at ``t - h``. That hour is complete one hour later, so the row
can be built ``h - 1`` hours before ``t`` starts: 47 hours with ``h = 48``. For a whole
day the last hour decides. 23:00 on a day reads the hour that starts at 23:00 two days
earlier, and that hour is complete at 00:00 on the day before. So with ``h = 48`` all
24 rows of a day can be built at once at 00:00 on the day before it; its first hour
then starts 24 hours later and its last hour 47 hours later. (With ``h = 24`` the same
sum gives 00:00 on the day itself.) Around a clock change the moment moves by an hour;
``backtest.py`` has the details. The features of a row depend on its target hour only,
not on the moment the row is built: building it later does not make it read newer
pickups.

**"48 hours earlier" means 48 real hours.** Twice a year the clocks change, and then
"the same clock time two days ago" and "48 hours ago" are different hours. All lags and
windows here count *elapsed* hours, for three reasons. The rule that protects against
leakage is a rule about elapsed time: in the two days after the clocks go forward, "the
same clock time two days ago" is only 47 real hours back, which is newer than a horizon
of 48 hours allows. Clock time has holes, too: 02:00 does not exist on the night the
clocks go forward and 01:00 exists twice when they go back, so a clock-time lag would
need special cases. And elapsed hours are simply a fixed number of rows on the hourly
grid. The price: for the rows after a clock change (48 hours of rows for the 48 h lag,
a week for the 168 h lag) the lag points at the clock hour next to the one a person
would expect, for example at 07:00 on Friday for a target of 08:00 on the Sunday the
clocks went forward. The calendar features are *not* affected; they always describe
the local clock time of the target hour.

**Missing history is missing, not zero.** A feature that needs an hour ``history`` does
not contain is left empty (``NaN``); so is a window with a single missing hour. The
function never guesses. What happens to such rows is up to the caller:
``build_feature_table`` drops them (see ``warm_up_hours``), a forecast has to refuse.

**Holidays.** ``is_holiday`` marks the public holidays of the United States (the
federal ones) from the ``holidays`` library, including the "observed" day when a
holiday falls on a weekend and the day off moves to Friday or Monday. State holidays of
New York (Lincoln's Birthday, Election Day) are not included: most people work on them.
"""

from __future__ import annotations

import holidays
import numpy as np
import numpy.typing as npt
import pandas as pd

from demand.config import Settings
from demand.schemas import (
    DAY_OF_WEEK,
    HOUR,
    HOUR_START,
    IS_HOLIDAY,
    IS_OUTLIER,
    IS_WEEKEND,
    MONTH,
    TARGET,
    ZONE_ID,
    feature_columns,
    gold_features_schema,
    lag_column,
    rolling_mean_column,
    rolling_std_column,
)

HOLIDAY_COUNTRY = "US"
_SATURDAY = 5  # pandas counts the days of the week from Monday = 0

FloatArray = npt.NDArray[np.float64]


def weekend_flag(hours: pd.Series) -> pd.Series:
    """True for hours on a Saturday or Sunday, by the local date of each hour."""
    is_weekend: pd.Series = hours.dt.dayofweek >= _SATURDAY
    return is_weekend


def holiday_flag(hours: pd.Series) -> pd.Series:
    """True for hours on a US public holiday, by the local date of each hour."""
    local_day = hours.dt.tz_localize(None).dt.normalize()
    years = range(int(local_day.dt.year.min()), int(local_day.dt.year.max()) + 1)
    calendar = holidays.country_holidays(HOLIDAY_COUNTRY, years=years)
    is_holiday: pd.Series = local_day.isin(pd.to_datetime(list(calendar)))
    return is_holiday


def calendar_features(hours: pd.Series) -> pd.DataFrame:
    """The calendar columns for timezone-aware hours, read from their local clock time."""
    day_of_week = hours.dt.dayofweek.astype("int64")
    return pd.DataFrame(
        {
            HOUR: hours.dt.hour.astype("int64"),
            DAY_OF_WEEK: day_of_week,
            MONTH: hours.dt.month.astype("int64"),
            IS_WEEKEND: weekend_flag(hours),
            IS_HOLIDAY: holiday_flag(hours),
        }
    )


def shift_later(values: FloatArray, hours: int) -> FloatArray:
    """Move every row ``hours`` rows down, so that row ``t`` holds the value of ``t - hours``.

    The first ``hours`` rows have nothing to show and are ``NaN``.
    """
    shifted = np.full_like(values, np.nan)
    if hours < len(values):
        shifted[hours:] = values[: len(values) - hours]
    return shifted


def window_statistics(values: FloatArray, window: int) -> tuple[FloatArray, FloatArray]:
    """Mean and sample standard deviation of the ``window`` rows that end at each row.

    Row ``t`` of the result describes rows ``t - window + 1`` to ``t`` of ``values``,
    column by column. A window that is not complete, or that contains a ``NaN``, gives
    ``NaN``.

    Every window is calculated on its own from its ``window`` values. A running sum
    (add the new value, subtract the old one) would be faster, but its rounding errors
    depend on everything that came before, so the same window could give a slightly
    different number in training and in serving.
    """
    mean = np.full_like(values, np.nan)
    std = np.full_like(values, np.nan)
    if len(values) >= window:
        # A view of shape (rows, columns, window): no values are copied.
        windows = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
        mean[window - 1 :] = windows.mean(axis=-1)
        std[window - 1 :] = windows.std(axis=-1, ddof=1)
    return mean, std


def warm_up_hours(settings: Settings) -> int:
    """How many hours of history a row needs before all its features can be filled.

    The longest lag looks ``max(lags)`` hours back. The longest window ends ``horizon``
    hours back and is ``max(rolling_windows)`` long, so its oldest value is
    ``horizon + max(rolling_windows) - 1`` hours back. The larger of the two decides:
    with lags up to 168 h and a 168 h window that ends 48 h back it is
    48 + 168 - 1 = 215 hours, so the first 215 hours of every zone (one hour short of
    nine days) cannot become training rows.
    """
    features = settings.features
    return max(max(features.lags), features.horizon_hours + max(features.rolling_windows) - 1)


def _check_table(table: pd.DataFrame, columns: list[str], name: str) -> None:
    """Refuse a table that the feature function cannot work with."""
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"{name} has no column {', '.join(missing)}")
    if table.empty:
        raise ValueError(f"{name} has no rows")
    if table.duplicated(subset=[ZONE_ID, HOUR_START]).any():
        raise ValueError(f"{name} has more than one row for the same zone and hour")


def demand_features(
    history: pd.DataFrame, targets: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """The lag and rolling features of every target row, read from ``history``.

    The history is laid out as a table with one row per hour and one column per zone,
    on an unbroken hourly timeline that reaches from the first hour of the history to
    the last target hour. On that table "k hours earlier" is "k rows up". An hour the
    history lacks, and every hour after its end, is ``NaN``.

    Returns:
        One row per target row, with the index of ``targets``.
    """
    config = settings.features
    # Elapsed time is counted in UTC, where no hour is skipped or repeated.
    history_hours = history[HOUR_START].dt.tz_convert("UTC")
    target_hours = targets[HOUR_START].dt.tz_convert("UTC")
    by_zone = history.assign(**{HOUR_START: history_hours}).pivot(
        index=HOUR_START, columns=ZONE_ID, values=TARGET
    )
    last_hour = max(history_hours.max(), target_hours.max())
    timeline = pd.date_range(history_hours.min(), last_hour, freq="h")
    pickups = by_zone.reindex(timeline).to_numpy(dtype="float64")

    tables: dict[str, FloatArray] = {}
    for lag in config.lags:
        tables[lag_column(lag)] = shift_later(pickups, lag)
    # At row t this holds the pickups of t - horizon: the newest value that is known
    # when t is predicted. Every window ends there.
    newest_known = shift_later(pickups, config.horizon_hours)
    for window in config.rolling_windows:
        mean, std = window_statistics(newest_known, window)
        tables[rolling_mean_column(window)] = mean
        tables[rolling_std_column(window)] = std

    # Pick the cell of every target row. A target before the start of the history, or
    # in a zone the history does not have, finds nothing and stays NaN.
    rows = timeline.get_indexer(pd.DatetimeIndex(target_hours))
    columns = by_zone.columns.get_indexer(pd.Index(targets[ZONE_ID]))
    found = (rows >= 0) & (columns >= 0)
    features = pd.DataFrame(index=targets.index)
    for name, table in tables.items():
        values = np.full(len(targets), np.nan)
        values[found] = table[rows[found], columns[found]]
        features[name] = values
    return features


def build_features(
    history: pd.DataFrame, targets: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Build the model inputs for the rows in ``targets``.

    This is the single feature function of the project; see the module docstring.

    Args:
        history: Hourly demand to read the past from: zone, hour (timezone-aware, on the
            hour) and pickups, one row per zone and hour. Other columns are ignored. It
            may contain hours newer than a target allows; they are not read.
        targets: The rows to build: zone, hour and one column per weather variable with
            the weather of that hour. Other columns (such as pickups) are ignored, so
            the hours may lie in the future.
        settings: Project settings (lags, windows, horizon, weather variables,
            timezone).

    Returns:
        One row per row of ``targets``, in the same order and with the same index: zone,
        hour (in the project timezone) and then the model inputs, so that
        ``result[feature_columns(settings)]`` is exactly what the model is given. Lag
        and rolling columns are ``NaN`` where the history does not reach.

    Raises:
        ValueError: If a table lacks a column, is empty, or has two rows for the same
            zone and hour.
    """
    variables = settings.data.weather.variables
    _check_table(history, [ZONE_ID, HOUR_START, TARGET], "history")
    _check_table(targets, [ZONE_ID, HOUR_START, *variables], "targets")

    local_hours = targets[HOUR_START].dt.tz_convert(settings.project.timezone)
    features = pd.concat(
        [
            targets[ZONE_ID].astype("int64"),
            local_hours,
            demand_features(history, targets, settings),
            calendar_features(local_hours),
            targets[variables].astype("float64"),
        ],
        axis=1,
    )
    inputs = [column for column in feature_columns(settings) if column != ZONE_ID]
    ordered: pd.DataFrame = features[[ZONE_ID, HOUR_START, *inputs]]
    return ordered


def build_feature_table(demand: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, int]:
    """Build the gold feature table: the features of every hour, with its target.

    The demand table is passed to ``build_features`` as the history *and* as the
    targets, so training rows are built by exactly the code that serves forecasts.

    The feature schema allows no missing value. Rows whose history is not complete are
    dropped; on a complete grid these are the first ``warm_up_hours`` hours of every
    zone. Nothing is filled in: a lag of 0 for an hour that was never observed would
    teach the model something false.

    Args:
        demand: The gold demand table (``gold_demand_schema``).
        settings: Project settings.

    Returns:
        The table with the columns of ``gold_features_schema`` in the order of the
        schema, and the number of rows that were dropped for lack of history.
    """
    features = build_features(demand, demand, settings)
    features[TARGET] = demand[TARGET]
    features[IS_OUTLIER] = demand[IS_OUTLIER]
    has_history = features.notna().all(axis=1)
    columns = list(gold_features_schema(settings).columns)
    table = features.loc[has_history, columns].reset_index(drop=True)
    return table, int((~has_history).sum())
