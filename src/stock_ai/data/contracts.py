"""Versioned contracts for J-Quants V2 ingestion and storage."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataContractModel(BaseModel):
    """Strict immutable base model for persisted data-plane metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetName(StrEnum):
    SECURITY_MASTER = "security_master"
    DAILY_PRICES = "daily_prices"
    TRADING_CALENDAR = "trading_calendar"
    TOPIX = "topix"
    FINANCIAL_SUMMARY = "financial_summary"


class SubscriptionPlan(StrEnum):
    FREE = "free"
    LIGHT = "light"
    STANDARD = "standard"
    PREMIUM = "premium"

    @property
    def requests_per_minute(self) -> int:
        return {
            SubscriptionPlan.FREE: 5,
            SubscriptionPlan.LIGHT: 60,
            SubscriptionPlan.STANDARD: 120,
            SubscriptionPlan.PREMIUM: 500,
        }[self]

    def includes(self, required: SubscriptionPlan) -> bool:
        order = tuple(SubscriptionPlan)
        return order.index(self) >= order.index(required)


class CapabilityStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    PARTIAL = "PARTIAL"
    BLOCKED_BY_PLAN = "BLOCKED_BY_PLAN"
    BLOCKED_BY_DATA_CAPABILITY = "BLOCKED_BY_DATA_CAPABILITY"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


class Capability(DataContractModel):
    name: str
    status: CapabilityStatus
    source_endpoint: str | None = None
    minimum_plan: SubscriptionPlan | None = None
    reason: str | None = None


class EndpointSchema(DataContractModel):
    dataset: DatasetName
    endpoint: str
    minimum_plan: SubscriptionPlan
    required_columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    source_date_column: str
    schema_version: str
    individual_requests_per_minute: int | None = None


ENDPOINT_SCHEMAS: Mapping[DatasetName, EndpointSchema] = MappingProxyType({
    DatasetName.SECURITY_MASTER: EndpointSchema(
        dataset=DatasetName.SECURITY_MASTER,
        endpoint="/equities/master",
        minimum_plan=SubscriptionPlan.FREE,
        required_columns=(
            "Date",
            "Code",
            "CoName",
            "CoNameEn",
            "S17",
            "S17Nm",
            "S33",
            "S33Nm",
            "ScaleCat",
            "Mkt",
            "MktNm",
            "Mrgn",
            "MrgnNm",
        ),
        primary_key=("Date", "Code"),
        source_date_column="Date",
        schema_version="jquants-v2-eq-master-2026-08",
    ),
    DatasetName.DAILY_PRICES: EndpointSchema(
        dataset=DatasetName.DAILY_PRICES,
        endpoint="/equities/bars/daily",
        minimum_plan=SubscriptionPlan.FREE,
        required_columns=(
            "Date",
            "Code",
            "O",
            "H",
            "L",
            "C",
            "Vo",
            "Va",
            "AdjFactor",
            "AdjO",
            "AdjH",
            "AdjL",
            "AdjC",
            "AdjVo",
        ),
        primary_key=("Date", "Code"),
        source_date_column="Date",
        schema_version="jquants-v2-eq-bars-daily-2026-08",
    ),
    DatasetName.TRADING_CALENDAR: EndpointSchema(
        dataset=DatasetName.TRADING_CALENDAR,
        endpoint="/markets/calendar",
        minimum_plan=SubscriptionPlan.LIGHT,
        required_columns=("Date", "HolDiv"),
        primary_key=("Date",),
        source_date_column="Date",
        schema_version="jquants-v2-mkt-calendar-2026-08",
    ),
    DatasetName.TOPIX: EndpointSchema(
        dataset=DatasetName.TOPIX,
        endpoint="/indices/bars/daily/topix",
        minimum_plan=SubscriptionPlan.LIGHT,
        required_columns=("Date", "O", "H", "L", "C"),
        primary_key=("Date",),
        source_date_column="Date",
        schema_version="jquants-v2-topix-daily-2026-08",
    ),
    DatasetName.FINANCIAL_SUMMARY: EndpointSchema(
        dataset=DatasetName.FINANCIAL_SUMMARY,
        endpoint="/fins/summary",
        minimum_plan=SubscriptionPlan.FREE,
        required_columns=(
            "DiscDate",
            "DiscTime",
            "Code",
            "DiscNo",
            "DocType",
            "CurPerType",
            "CurPerSt",
            "CurPerEn",
            "Sales",
            "OP",
            "NP",
            "EPS",
            "TA",
            "Eq",
            "ShOutFY",
            "TrShFY",
        ),
        primary_key=("DiscNo",),
        source_date_column="DiscDate",
        schema_version="jquants-v2-fin-summary-2026-08",
        individual_requests_per_minute=60,
    ),
})


class FetchedPayload(DataContractModel):
    dataset: DatasetName
    endpoint: str
    query: tuple[tuple[str, str], ...]
    requested_at: datetime
    received_at: datetime
    pages: int = Field(ge=1)
    rows: tuple[dict[str, Any], ...]

    @field_validator("requested_at", "received_at")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_window(self) -> FetchedPayload:
        if self.received_at < self.requested_at:
            raise ValueError("received_at cannot precede requested_at")
        return self


class QualitySeverity(StrEnum):
    WARNING = "WARNING"
    ERROR = "ERROR"


class QualityIssue(DataContractModel):
    code: str
    severity: QualitySeverity
    message: str
    rows_affected: int = Field(default=0, ge=0)


class QualityReport(DataContractModel):
    dataset: DatasetName
    rows: int = Field(ge=0)
    issues: tuple[QualityIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return not any(issue.severity is QualitySeverity.ERROR for issue in self.issues)


class ObjectKind(StrEnum):
    RAW = "raw"
    NORMALIZED = "normalized"


class StoredObject(DataContractModel):
    object_id: str
    kind: ObjectKind
    dataset: DatasetName
    source_date: date
    payload_hash: str
    parquet_hash: str
    rows: int = Field(ge=0)
    schema_version: str
    provider: str = "J-Quants"
    source_endpoint: str
    received_at: datetime
    available_at: datetime
    as_of: datetime
    ingestion_run_id: str
    parquet_path: Path
    manifest_path: Path
    quality_passed: bool
    quality_issues: tuple[QualityIssue, ...] = ()

    @field_validator("received_at", "available_at", "as_of")
    @classmethod
    def object_timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def available_by_as_of(self) -> StoredObject:
        if self.available_at > self.as_of:
            raise ValueError("available_at cannot be after as_of")
        return self


class IngestionStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class IngestionResult(DataContractModel):
    ingestion_run_id: str
    source_date: date
    started_at: datetime
    completed_at: datetime
    status: IngestionStatus
    objects: tuple[StoredObject, ...]
    capabilities: tuple[Capability, ...]


def capabilities_for(plan: SubscriptionPlan) -> tuple[Capability, ...]:
    """Return an explicit, fail-closed capability table for Goal 2A."""

    capabilities: list[Capability] = []
    for name, dataset in (
        ("security_master", DatasetName.SECURITY_MASTER),
        ("daily_prices", DatasetName.DAILY_PRICES),
        ("research_adjusted_ohlcv", DatasetName.DAILY_PRICES),
        ("trading_calendar", DatasetName.TRADING_CALENDAR),
        ("topix_context", DatasetName.TOPIX),
        ("financial_summary", DatasetName.FINANCIAL_SUMMARY),
    ):
        schema = ENDPOINT_SCHEMAS[dataset]
        available = plan.includes(schema.minimum_plan)
        capabilities.append(
            Capability(
                name=name,
                status=(
                    CapabilityStatus.AVAILABLE
                    if available
                    else CapabilityStatus.BLOCKED_BY_PLAN
                ),
                source_endpoint=schema.endpoint,
                minimum_plan=schema.minimum_plan,
                reason=(
                    None
                    if available
                    else f"requires {schema.minimum_plan.value} plan or higher"
                ),
            )
        )
    capabilities.extend(
        (
            Capability(
                name="shares_outstanding",
                status=CapabilityStatus.PARTIAL,
                source_endpoint=ENDPOINT_SCHEMAS[DatasetName.FINANCIAL_SUMMARY].endpoint,
                minimum_plan=SubscriptionPlan.FREE,
                reason="fiscal-period disclosure snapshots only; not a daily share-count series",
            ),
            Capability(
                name="historical_point_in_time_universe",
                status=CapabilityStatus.PARTIAL,
                source_endpoint=ENDPOINT_SCHEMAS[DatasetName.SECURITY_MASTER].endpoint,
                minimum_plan=SubscriptionPlan.FREE,
                reason=(
                    "date-addressed master snapshots are preserved from first ingestion; "
                    "pre-ingestion correction vintages cannot be reconstructed"
                ),
            ),
            Capability(
                name="market_breadth",
                status=CapabilityStatus.PARTIAL,
                source_endpoint=ENDPOINT_SCHEMAS[DatasetName.DAILY_PRICES].endpoint,
                minimum_plan=SubscriptionPlan.FREE,
                reason="derivable only after full-universe coverage validation",
            ),
            Capability(
                name="supply_demand",
                status=(
                    CapabilityStatus.BLOCKED_BY_PLAN
                    if not plan.includes(SubscriptionPlan.STANDARD)
                    else CapabilityStatus.OUT_OF_SCOPE
                ),
                minimum_plan=SubscriptionPlan.STANDARD,
                reason="Goal 2A does not ingest Standard/Premium supply-demand endpoints",
            ),
            Capability(
                name="intraday_morning",
                status=CapabilityStatus.OUT_OF_SCOPE,
                reason="real-time 09:00-11:30 processing is outside Goal 2A",
            ),
        )
    )
    return tuple(capabilities)
