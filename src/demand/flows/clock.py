"""The simulated clock of the replay: what "now" is, and which day is forecast next.

**Why a clock at all.** The data ends on 30 June 2025, and the model was trained on the
days up to ``data.train_end_date`` (30 April). May and June are "production": the
replay walks through them one day at a time, as a scheduler would have, and at every
step does what the system would have done at that moment. The moment is simulated; the
wall clock plays no part.

**The clock is the folder of forecasts.** There is no state file and no counter. The
newest day in ``data/predictions/forecasts/`` *is* the position of the clock: the next
step forecasts the day after it, at that day's issue moment (``issue_moment``: 00:00 on
the day before). A state file would be a second record of the same fact, and the two
could disagree after a crash between writing one and the other. A ``--day`` argument
would let anyone forecast days out of order, or one day twice. Derived from the files,
the clock cannot disagree with them, it survives a restart, and a crash halfway through
a step leaves either no forecast for the day (the step is done again) or a complete one
(the files are written whole, under a temporary name first, then renamed).

**The days must follow on from each other.** The forecasts on disk must be every day
from ``first_replay_day`` on, with none missing. A gap (a file deleted by hand) would
leave a day nobody forecast, and the clock would silently step over it; so a gap, or a
file whose name is not a day, stops the step with a message that says what to do.

**Where the replay starts: 2 May, not 1 May.** The champion was fitted on the rows up to
30 April 23:00 (``models/training.py``), so it existed at 00:00 on 1 May, not earlier.
The forecast for 1 May was due at 00:00 on 30 April, a day before the model existed; a
forecast it made for 1 May would read no future pickups (the features see to that), but
it would come from a model that could not yet have been trained. So 1 May is not
forecast at all: the first day of the replay is the first day whose issue moment is at
or after the end of the training data (``first_replay_day``), which is 2 May. Leaving it
out is simpler and more honest than writing a forecast that must then be flagged and
excluded everywhere it is read.

**The end of the data.** The last day the data can forecast is ``last_day`` of the
snapshot (30 June). After its step the clock stands at 00:00 on 29 June, so 29 and 30
June are forecast but not yet over. One more step moves the clock to the end of the
demand history (00:00 on 1 July) and forecasts nothing: it only scores the days that are
now complete. Every step after that is the same step again, and changes nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from demand.config import Settings
from demand.errors import PipelineError
from demand.schemas import HOUR_START
from demand.serving.snapshot import ONE_HOUR, DataSnapshot, day_hours, issue_moment

FORECASTS_DIR = "forecasts"
INPUTS_DIR = "inputs"
SCORED_DIR = "scored"
PARQUET = ".parquet"


class ReplayStateError(PipelineError):
    """The files in ``data/predictions`` do not describe one unbroken replay."""


@dataclass(frozen=True)
class Step:
    """One tick of the simulated clock."""

    now: pd.Timestamp
    """The simulated moment of the step, in the project timezone. Nothing newer is read."""
    day: date | None
    """The day forecast at ``now``; ``None`` once the data leaves no day to forecast."""


def day_file(folder: Path, day: date) -> Path:
    """The file of ``day`` in one of the folders of ``data/predictions``."""
    return folder / f"{day.isoformat()}{PARQUET}"


def forecasts_dir(settings: Settings) -> Path:
    """The forecasts as issued: written once, never changed."""
    return settings.paths.predictions / FORECASTS_DIR


def inputs_dir(settings: Settings) -> Path:
    """The model inputs of every forecast: the rows the model was given."""
    return settings.paths.predictions / INPUTS_DIR


def scored_dir(settings: Settings) -> Path:
    """The forecasts of the days that are over, with the pickups that were counted."""
    return settings.paths.predictions / SCORED_DIR


def stored_days(folder: Path) -> list[date]:
    """The days that have a file in ``folder``, oldest first; none if it does not exist.

    Raises:
        ReplayStateError: If the folder holds anything but ``YYYY-MM-DD.parquet`` files.
    """
    if not folder.is_dir():
        return []
    days = []
    for path in sorted(folder.iterdir()):
        try:
            if path.suffix != PARQUET or not path.is_file():
                raise ValueError(path.name)
            days.append(date.fromisoformat(path.stem))
        except ValueError as error:
            raise ReplayStateError(
                f"{path.as_posix()} is not a file of the replay (they are named "
                "YYYY-MM-DD.parquet). Remove it; if it is left over from an interrupted "
                "run, the step that wrote it is done again."
            ) from error
    return days


def day_end(day: date, settings: Settings) -> pd.Timestamp:
    """The moment ``day`` is over: 00:00 on the next day, local time."""
    end: pd.Timestamp = day_hours(day, settings)[-1] + ONE_HOUR
    return end


def first_replay_day(settings: Settings) -> date:
    """The first day whose forecast was due when the champion already existed.

    The champion learned from every hour up to the end of ``data.train_end_date``, so it
    exists from 00:00 on the next day. With a horizon of 48 hours that is 2 May.
    """
    trained_until = pd.Timestamp(settings.data.replay_start_date, tz=settings.project.timezone)
    day = settings.data.replay_start_date
    while issue_moment(day, settings) < trained_until:
        day += timedelta(days=1)
    return day


def history_end(snapshot: DataSnapshot) -> pd.Timestamp:
    """The moment the demand history ends: the end of its last hour."""
    end: pd.Timestamp = snapshot.history[HOUR_START].max() + ONE_HOUR
    return end


def next_step(settings: Settings, snapshot: DataSnapshot) -> Step:
    """Where the clock goes next, read from the forecasts already on disk.

    Raises:
        ReplayStateError: If the stored forecasts do not start at ``first_replay_day``
            or leave out a day.
    """
    first = first_replay_day(settings)
    days = stored_days(forecasts_dir(settings))
    expected = [first + timedelta(days=offset) for offset in range(len(days))]
    if days != expected:
        missing = sorted(set(expected) - set(days))
        problem = (
            f"day {missing[0].isoformat()} is missing"
            if missing and days[0] == first
            else f"they start on {days[0].isoformat()}, not on {first.isoformat()}"
        )
        raise ReplayStateError(
            f"The forecasts in {forecasts_dir(settings).as_posix()} are not one unbroken "
            f"run of days from {first.isoformat()}: {problem}. A forecast cannot be "
            "issued late, so start the replay again: delete "
            f"{settings.paths.predictions.as_posix()} and run the forecast step."
        )
    day = days[-1] + timedelta(days=1) if days else first
    if day <= snapshot.last_day:
        return Step(now=issue_moment(day, settings), day=day)
    return Step(now=history_end(snapshot), day=None)
