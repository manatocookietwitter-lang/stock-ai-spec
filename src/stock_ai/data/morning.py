"""Provider-neutral, fail-closed contracts for 09:00--11:30 morning data."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import datetime, time
from enum import StrEnum
from types import MappingProxyType
from typing import Self
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from stock_ai.data.contracts import Capability, CapabilityStatus

JST = ZoneInfo("Asia/Tokyo")
MORNING_OPEN = time(9, 0)
MORNING_FREEZE = time(11, 30)
MORNING_CUTOFFS: tuple[time, ...] = (
    time(9, 0),
    time(9, 5),
    time(9, 15),
    time(9, 30),
    time(10, 0),
    time(11, 0),
    time(11, 30),
)


class MorningDataError(ValueError):
    """Raised when morning data cannot safely be used at the freeze."""


class MorningUniverseRole(StrEnum):
    HOLDING = "HOLDING"
    CANDIDATE = "CANDIDATE"
    HOLDING_AND_CANDIDATE = "HOLDING_AND_CANDIDATE"


class MorningUniverseMember(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    role: MorningUniverseRole


class MorningCapabilityReport(BaseModel):
    """Immutable provider capability declaration used before feature construction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str | None = None
    capabilities: Mapping[str, CapabilityStatus]
    reasons: Mapping[str, str]

    @field_validator("capabilities", mode="after")
    @classmethod
    def freeze_capabilities(
        cls, value: Mapping[str, CapabilityStatus]
    ) -> Mapping[str, CapabilityStatus]:
        return MappingProxyType(dict(value))

    @field_validator("reasons", mode="after")
    @classmethod
    def freeze_reasons(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("capabilities", "reasons")
    def serialize_mapping(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    def require(self, *names: str) -> None:
        unavailable = [
            name for name in names if self.capabilities.get(name) is not CapabilityStatus.AVAILABLE
        ]
        if unavailable:
            detail = ", ".join(
                f"{name}={self.capabilities.get(name, CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY)}"
                for name in unavailable
            )
            raise MorningDataError(f"BLOCKED_BY_DATA_CAPABILITY: {detail}")


class MorningFreezeMetadata(BaseModel):
    """Audit metadata for one exact 11:30 monitored-universe freeze."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: datetime
    provider: str = Field(min_length=1)
    source_snapshot_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    universe: tuple[MorningUniverseMember, ...]
    capability_report: MorningCapabilityReport
    is_order_instruction: bool = False

    @field_validator("as_of")
    @classmethod
    def aware_exact_freeze(cls, value: datetime) -> datetime:
        _require_aware(value)
        local = value.astimezone(JST)
        if local.time().replace(tzinfo=None) != MORNING_FREEZE:
            raise ValueError("morning freeze must be exactly 11:30 JST")
        return value

    @model_validator(mode="after")
    def valid_freeze(self) -> Self:
        if self.is_order_instruction:
            raise ValueError("morning data freezes can never be order instructions")
        if not self.source_snapshot_ids:
            raise ValueError("morning freeze requires source snapshot IDs")
        if len(self.source_snapshot_ids) != len(set(self.source_snapshot_ids)):
            raise ValueError("morning source snapshot IDs must be unique")
        if not self.source_record_ids:
            raise ValueError("morning freeze requires source record IDs")
        if len(self.source_record_ids) != len(set(self.source_record_ids)):
            raise ValueError("morning source record IDs must be unique")
        if self.capability_report.provider != self.provider:
            raise ValueError("morning freeze provider must match its capability report")
        symbols = [member.symbol for member in self.universe]
        if not symbols or len(symbols) != len(set(symbols)):
            raise ValueError("morning monitored-universe symbols must be non-empty and unique")
        return self


_CAPABILITY_COLUMNS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "morning_ohlc": frozenset({"timestamp", "price"}),
        "intraday_bars": frozenset({"timestamp", "price", "volume", "trading_value"}),
        "intraday_volume_profile": frozenset(
            {"timestamp", "volume", "trading_value", "historical_same_time_sessions"}
        ),
        "quotes": frozenset({"bid", "ask", "spread", "quote_state"}),
        "order_book": frozenset({"bid_size", "ask_size"}),
        "trade_frequency": frozenset({"trade_count", "seconds_since_last_trade"}),
    }
)


def morning_capabilities(
    *, provider: str | None, available_fields: Iterable[str]
) -> MorningCapabilityReport:
    """Declare capabilities from an explicit provider/field contract; never infer a source."""

    fields = frozenset(available_fields)
    statuses: dict[str, CapabilityStatus] = {}
    reasons: dict[str, str] = {}
    for name, required in _CAPABILITY_COLUMNS.items():
        missing = sorted(required - fields)
        if provider is None:
            statuses[name] = CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY
            reasons[name] = "no morning market-data provider is configured"
        elif missing:
            statuses[name] = CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY
            reasons[name] = f"provider contract is missing: {', '.join(missing)}"
        else:
            statuses[name] = CapabilityStatus.AVAILABLE
    return MorningCapabilityReport(provider=provider, capabilities=statuses, reasons=reasons)


def morning_capability_rows(report: MorningCapabilityReport) -> tuple[Capability, ...]:
    """Expose the provider-neutral report through the common capability DTO."""

    return tuple(
        Capability(
            name=name,
            status=status,
            reason=report.reasons.get(name),
        )
        for name, status in report.capabilities.items()
    )


def build_morning_universe(
    *, current_holdings: Iterable[str], candidates: Iterable[str]
) -> tuple[MorningUniverseMember, ...]:
    """Union holdings and candidates while retaining the role of every symbol."""

    holdings = {str(symbol).strip() for symbol in current_holdings if str(symbol).strip()}
    candidate_set = {str(symbol).strip() for symbol in candidates if str(symbol).strip()}
    symbols = sorted(holdings | candidate_set)
    if not symbols:
        raise MorningDataError("morning monitored universe cannot be empty")
    return tuple(
        MorningUniverseMember(
            symbol=symbol,
            role=(
                MorningUniverseRole.HOLDING_AND_CANDIDATE
                if symbol in holdings and symbol in candidate_set
                else MorningUniverseRole.HOLDING
                if symbol in holdings
                else MorningUniverseRole.CANDIDATE
            ),
        )
        for symbol in symbols
    )


def validate_morning_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate bar-level PIT invariants for historical or current morning sessions."""

    required = {
        "symbol",
        "timestamp",
        "available_at",
        "price",
        "volume",
        "trading_value",
        "provider",
        "source_record_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise MorningDataError(f"morning bars are missing columns: {', '.join(missing)}")
    if frame.empty:
        raise MorningDataError("morning bars cannot be empty")
    for column in ("symbol", "provider", "source_record_id"):
        if frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any():
            raise MorningDataError(f"morning {column} cannot be blank")
    output = frame.copy()
    timestamps = _aware_series(output["timestamp"], "timestamp")
    available = _aware_series(output["available_at"], "available_at")
    output["timestamp"] = timestamps
    output["available_at"] = available
    local_timestamps = timestamps.map(lambda value: value.astimezone(JST))
    session_dates = local_timestamps.map(lambda value: value.date())
    session_freezes = pd.Series(
        [pd.Timestamp(datetime.combine(day, MORNING_FREEZE, tzinfo=JST)) for day in session_dates],
        index=output.index,
    )
    local_times = local_timestamps.map(lambda value: value.time().replace(tzinfo=None))
    if ((local_times < MORNING_OPEN) | (local_times > MORNING_FREEZE)).any():
        raise MorningDataError("morning bars must be timestamped from 09:00 through 11:30 JST")
    if (available > session_freezes).any():
        raise MorningDataError("morning bar was not available by its session's 11:30 freeze")
    if (available < timestamps).any():
        raise MorningDataError("morning bars cannot be available before their observation")
    if output.duplicated(["symbol", "timestamp"]).any():
        raise MorningDataError("morning bars must be unique by symbol and timestamp")
    for column in ("price", "volume", "trading_value"):
        numeric = pd.to_numeric(output[column], errors="coerce")
        if not numeric.map(math.isfinite).all():
            raise MorningDataError(f"morning {column} must contain only finite numeric values")
        output[column] = numeric.astype(float)
    if (output["price"] <= 0).any():
        raise MorningDataError("morning prices must be positive")
    if (output[["volume", "trading_value"]] < 0).any(axis=None):
        raise MorningDataError("morning volume and trading value cannot be negative")
    for column in (
        "bid",
        "ask",
        "spread",
        "bid_size",
        "ask_size",
        "trade_count",
        "seconds_since_last_trade",
    ):
        if column not in output.columns:
            continue
        present = output[column].notna()
        numeric = pd.to_numeric(output[column], errors="coerce")
        if numeric.loc[present].isna().any() or not numeric.dropna().map(math.isfinite).all():
            raise MorningDataError(f"morning {column} must be finite when present")
        if numeric.dropna().lt(0).any():
            raise MorningDataError(f"morning {column} cannot be negative")
        output[column] = numeric.astype(float)
    if {"bid", "ask"} <= set(output.columns):
        valid_quotes = output["bid"].notna() & output["ask"].notna()
        if (output.loc[valid_quotes, "ask"] < output.loc[valid_quotes, "bid"]).any():
            raise MorningDataError("morning ask cannot be below bid")
    output["trading_date"] = pd.to_datetime(session_dates)
    return output.sort_values(["trading_date", "symbol", "timestamp"], kind="stable").reset_index(
        drop=True
    )


def assert_morning_freeze_coverage(
    frame: pd.DataFrame, metadata: MorningFreezeMetadata
) -> pd.DataFrame:
    """Require the exact declared holdings+candidates universe at a current freeze."""

    validated = validate_morning_bars(frame)
    freeze_date = pd.Timestamp(metadata.as_of.astimezone(JST).date())
    session = validated.loc[validated["trading_date"].eq(freeze_date)].copy()
    expected = {member.symbol for member in metadata.universe}
    observed = set(session["symbol"].astype(str))
    if expected != observed:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise MorningDataError(
            "BLOCKED_BY_DATA_CAPABILITY: morning freeze universe mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if set(session["provider"].astype(str)) != {metadata.provider}:
        raise MorningDataError("morning freeze provider does not match its metadata")
    if set(session["source_record_id"].astype(str)) - set(metadata.source_record_ids):
        raise MorningDataError("morning freeze source records do not match its metadata")
    return session.reset_index(drop=True)


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


def _aware_series(values: pd.Series, name: str) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    for raw in values:
        value = pd.Timestamp(raw)
        if value.tzinfo is None or value.utcoffset() is None:
            raise MorningDataError(f"morning {name} values must be timezone-aware")
        parsed.append(value)
    return pd.Series(parsed, index=values.index, dtype="object")
