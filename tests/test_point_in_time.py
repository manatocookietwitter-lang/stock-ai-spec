from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from stock_ai.data import DataAvailabilityError, assert_point_in_time, point_in_time_view


def test_point_in_time_view_excludes_later_records() -> None:
    frame = pd.DataFrame(
        {
            "value": [1, 2],
            "available_at": [
                "2026-08-24T01:00:00Z",
                "2026-08-24T03:00:00Z",
            ],
        }
    )
    as_of = datetime(2026, 8, 24, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    result = point_in_time_view(frame, as_of)
    assert result["value"].tolist() == [1]


def test_naive_as_of_is_rejected() -> None:
    frame = pd.DataFrame({"available_at": ["2026-08-24T01:00:00Z"]})
    with pytest.raises(DataAvailabilityError, match="timezone-aware"):
        point_in_time_view(frame, datetime(2026, 8, 24, 11, 30))


def test_assert_point_in_time_detects_future_information() -> None:
    frame = pd.DataFrame(
        {
            "available_at": ["2026-08-24T03:00:00Z"],
            "as_of": ["2026-08-24T02:30:00Z"],
        }
    )
    with pytest.raises(DataAvailabilityError, match="future information"):
        assert_point_in_time(frame)
