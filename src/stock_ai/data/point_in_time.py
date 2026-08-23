"""Utilities enforcing the ``available_at <= as_of`` invariant."""

from __future__ import annotations

from datetime import datetime

import pandas as pd


class DataAvailabilityError(ValueError):
    """Raised when a point-in-time input cannot be used safely."""


def _as_utc_timestamp(value: datetime | pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataAvailabilityError("as_of must be timezone-aware")
    return timestamp.tz_convert("UTC")


def normalize_available_at(frame: pd.DataFrame, column: str = "available_at") -> pd.DataFrame:
    if column not in frame:
        raise DataAvailabilityError(f"missing required availability column: {column}")
    normalized = frame.copy()
    for value in normalized[column].dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise DataAvailabilityError(f"{column} timestamps must be timezone-aware")
    normalized[column] = pd.to_datetime(normalized[column], utc=True, errors="raise")
    if normalized[column].isna().any():
        raise DataAvailabilityError(f"{column} cannot contain missing timestamps")
    return normalized


def point_in_time_view(
    frame: pd.DataFrame,
    as_of: datetime | pd.Timestamp,
    *,
    available_at_column: str = "available_at",
) -> pd.DataFrame:
    cutoff = _as_utc_timestamp(as_of)
    normalized = normalize_available_at(frame, available_at_column)
    return normalized.loc[normalized[available_at_column] <= cutoff].copy()


def assert_point_in_time(
    frame: pd.DataFrame,
    *,
    as_of_column: str = "as_of",
    available_at_column: str = "available_at",
) -> None:
    if as_of_column not in frame:
        raise DataAvailabilityError(f"missing required as-of column: {as_of_column}")
    normalized = normalize_available_at(frame, available_at_column)
    for value in normalized[as_of_column].dropna():
        if pd.Timestamp(value).tzinfo is None:
            raise DataAvailabilityError(f"{as_of_column} timestamps must be timezone-aware")
    as_of = pd.to_datetime(normalized[as_of_column], utc=True, errors="raise")
    if as_of.isna().any():
        raise DataAvailabilityError(f"{as_of_column} cannot contain missing timestamps")
    invalid = normalized[available_at_column] > as_of
    if invalid.any():
        examples = normalized.loc[invalid, [available_at_column, as_of_column]].head(3)
        raise DataAvailabilityError(
            f"future information detected:\n{examples.to_string(index=False)}"
        )
