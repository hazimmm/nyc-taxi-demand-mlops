"""Training/serving skew: the service builds exactly the rows the model was trained on.

Training rows come from ``build_feature_table``, which passes the whole gold demand table
to ``build_features`` as history and as targets. The service passes only the demand known
at the issue moment, and the weather from the silver table. For the same data the two
must give the same rows, value for value and type for type; otherwise the model is asked
questions it was never trained on and nothing raises.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from demand.config import Settings
from demand.data.features import build_feature_table
from demand.schemas import HOUR_START, ZONE_ID, feature_columns
from demand.serving.snapshot import day_features, make_snapshot
from tests.conftest import TEST_ZONES
from tests.unit.serving.conftest import CLOCKS_FORWARD, FIRST_DAY, LAST_DATA_DAY, tables


@pytest.mark.parametrize("day", [FIRST_DAY, CLOCKS_FORWARD, date(2025, 3, 10), LAST_DATA_DAY])
def test_served_rows_equal_the_training_rows_of_the_same_hours(
    settings: Settings, day: date
) -> None:
    demand, weather = tables(settings, seed=3)
    training, _ = build_feature_table(demand, settings)
    columns = [ZONE_ID, HOUR_START, *feature_columns(settings)[1:]]

    _, served = day_features(make_snapshot(demand, weather, settings), TEST_ZONES, day, settings)

    expected = (
        training[training[HOUR_START].dt.date == day]
        .sort_values([ZONE_ID, HOUR_START])
        .reset_index(drop=True)
    )
    assert len(served) == len(expected) > 0
    # The service makes the hours of a request from the calendar; training takes them
    # from the demand table. pandas may store the two with another precision (seconds
    # or microseconds). They are keys, not model inputs: the same instants is what counts.
    served = served.astype({HOUR_START: expected[HOUR_START].dtype})
    pd.testing.assert_frame_equal(
        served[columns].reset_index(drop=True), expected[columns], check_exact=True
    )
