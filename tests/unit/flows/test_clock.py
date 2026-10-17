"""The simulated clock: where the replay starts, which day comes next, where it ends."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from demand.config import Settings
from demand.errors import PipelineError
from demand.flows.clock import (
    ReplayStateError,
    day_end,
    day_file,
    first_replay_day,
    forecasts_dir,
    next_step,
    stored_days,
)
from demand.serving.snapshot import DataSnapshot, issue_moment
from tests.unit.flows.conftest import FIRST_REPLAY_DAY, LAST_DAY


def local(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="America/New_York")


def touch(settings: Settings, *days: date) -> None:
    folder = forecasts_dir(settings)
    folder.mkdir(parents=True, exist_ok=True)
    for day in days:
        day_file(folder, day).write_bytes(b"")


def test_the_replay_of_the_project_starts_on_2_may_not_1_may(settings: Settings) -> None:
    # The champion learned from the hours up to 30 April 23:00, so it exists at 00:00 on
    # 1 May: the issue moment of 2 May. The forecast for 1 May was due a day earlier.
    assert settings.data.replay_start_date == date(2025, 5, 1)
    assert first_replay_day(settings) == date(2025, 5, 2)
    assert issue_moment(date(2025, 5, 2), settings) == local("2025-05-01 00:00")


def test_the_first_step_forecasts_the_first_replay_day_at_its_issue_moment(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    step = next_step(replay_settings, snapshot)

    assert step.day == FIRST_REPLAY_DAY
    assert step.now == local("2025-03-08 00:00")


def test_the_next_step_forecasts_the_day_after_the_newest_forecast(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    touch(replay_settings, date(2025, 3, 9), date(2025, 3, 10))

    step = next_step(replay_settings, snapshot)

    assert step.day == date(2025, 3, 11)
    assert step.now == local("2025-03-10 00:00")


def test_the_clock_does_not_move_without_a_forecast_on_disk(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    # Asking twice is the same step: only a written forecast moves the clock.
    assert next_step(replay_settings, snapshot) == next_step(replay_settings, snapshot)


def test_after_the_last_day_the_clock_stands_at_the_end_of_the_history(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    touch(replay_settings, date(2025, 3, 9), date(2025, 3, 10), date(2025, 3, 11), LAST_DAY)

    step = next_step(replay_settings, snapshot)

    assert step.day is None
    assert step.now == local("2025-03-13 00:00")
    assert snapshot.last_day == LAST_DAY


def test_a_missing_day_stops_the_clock_with_what_to_do(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    touch(replay_settings, date(2025, 3, 9), date(2025, 3, 11))

    with pytest.raises(ReplayStateError, match="day 2025-03-10 is missing") as raised:
        next_step(replay_settings, snapshot)

    assert isinstance(raised.value, PipelineError)
    assert "start the replay again" in str(raised.value)


def test_forecasts_that_start_on_another_day_are_refused(
    replay_settings: Settings, snapshot: DataSnapshot
) -> None:
    touch(replay_settings, date(2025, 3, 10))

    with pytest.raises(ReplayStateError, match="start on 2025-03-10, not on 2025-03-09"):
        next_step(replay_settings, snapshot)


@pytest.mark.parametrize("name", ["notes.txt", "2025-03-09.parquet.tmp", "2025-13-01.parquet"])
def test_a_file_that_is_not_a_day_is_refused(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_bytes(b"")

    with pytest.raises(ReplayStateError, match=r"is not a file of the replay .*Remove it"):
        stored_days(tmp_path)


def test_a_folder_inside_the_replay_folder_is_refused(tmp_path: Path) -> None:
    (tmp_path / "2025-03-09.parquet").mkdir()

    with pytest.raises(ReplayStateError, match="not a file of the replay"):
        stored_days(tmp_path)


def test_a_folder_that_does_not_exist_holds_no_day(tmp_path: Path) -> None:
    assert stored_days(tmp_path / "missing") == []


def test_stored_days_are_oldest_first(tmp_path: Path) -> None:
    for name in ["2025-03-10.parquet", "2025-03-09.parquet"]:
        (tmp_path / name).write_bytes(b"")

    assert stored_days(tmp_path) == [date(2025, 3, 9), date(2025, 3, 10)]


def test_a_day_ends_at_midnight_also_when_the_clocks_change(settings: Settings) -> None:
    # 9 March 2025 has 23 hours; it still ends at 00:00 on 10 March.
    assert day_end(date(2025, 3, 9), settings) == local("2025-03-10 00:00")
    assert day_end(date(2025, 11, 2), settings) == local("2025-11-03 00:00")
