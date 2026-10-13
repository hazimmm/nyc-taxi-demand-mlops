from __future__ import annotations

import pandas as pd
import pytest

from demand.data.aggregate import TooFewZonesError, WeatherCoverageError, busiest_zones
from demand.data.clean import RawFormatError
from demand.data.clean_weather import WeatherGapError
from demand.data.gold import MissingSilverFileError, ZonesNotSelectedError
from demand.data.ingest import DownloadError, RejectedRequestError, TransientDownloadError
from demand.data.silver import MissingRawFileError
from demand.data.weather import UnusableWeatherError
from demand.errors import PipelineError, QualityGateError

# Every error a person at the terminal can act on, with the built-in class it also
# inherits from. Code written before ``PipelineError`` existed catches the built-in one.
EXPECTED_ERRORS = [
    (MissingRawFileError, FileNotFoundError),
    (MissingSilverFileError, FileNotFoundError),
    (ZonesNotSelectedError, ValueError),
    (TooFewZonesError, ValueError),
    (WeatherCoverageError, ValueError),
    (WeatherGapError, ValueError),
    (RawFormatError, ValueError),
    (DownloadError, Exception),
    (RejectedRequestError, DownloadError),
    (TransientDownloadError, DownloadError),
    (UnusableWeatherError, DownloadError),
    (QualityGateError, Exception),
]


@pytest.mark.parametrize(("error", "older_base"), EXPECTED_ERRORS)
def test_expected_error_is_reported_in_one_line_and_keeps_its_older_base(
    error: type[Exception], older_base: type[Exception]
) -> None:
    # ``demand.cli`` prints a ``PipelineError`` as one line instead of a traceback.
    assert issubclass(error, PipelineError)
    assert issubclass(error, older_base)


def test_an_expected_error_can_still_be_caught_by_its_older_base() -> None:
    with pytest.raises(FileNotFoundError):
        raise MissingSilverFileError("a silver file is missing")


def test_too_few_zones_says_what_to_do_next() -> None:
    with pytest.raises(TooFewZonesError, match=r"Lower data\.n_zones"):
        busiest_zones(pd.Series({161: 5}), n_zones=2)
