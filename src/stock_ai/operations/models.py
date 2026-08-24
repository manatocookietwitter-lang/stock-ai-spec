"""Immutable contracts for the local Goal 5 operating ledger.

These records describe decision-support operations only.  None of them can be
interpreted as, or converted into, a securities order instruction.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Self
from zoneinfo import ZoneInfo

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from stock_ai.decision.engine import DecisionEngineConfig


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("operational timestamps must be timezone-aware")
    return value


class OperationalModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class PipelineState(StrEnum):
    PRE_MARKET = "PRE_MARKET"
    MORNING_ANALYSIS = "MORNING_ANALYSIS"
    FREEZING_INPUTS = "FREEZING_INPUTS"
    GENERATING_PROPOSAL = "GENERATING_PROPOSAL"
    PROPOSAL_READY = "PROPOSAL_READY"
    USER_DECISION_SAVED = "USER_DECISION_SAVED"
    EXECUTION_PENDING = "EXECUTION_PENDING"
    EXECUTION_RECORDED = "EXECUTION_RECORDED"
    MARKET_CLOSED = "MARKET_CLOSED"
    HOLIDAY = "HOLIDAY"
    STALE_DATA = "STALE_DATA"
    DATA_ERROR = "DATA_ERROR"
    MODEL_ERROR = "MODEL_ERROR"


class AutomationStage(StrEnum):
    DATA_SYNC = "data_sync"
    CANDIDATE_SELECTION = "candidate_selection"
    MORNING_CAPTURE = "morning_capture"
    FREEZE_1130 = "freeze_1130"
    PREDICTION = "prediction"
    PROPOSAL = "proposal"
    NOTIFICATION = "notification"
    EOD_UPDATE = "eod_update"
    CHALLENGER_TRAINING = "challenger_training"


class JobStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class NotificationChannel(StrEnum):
    IN_APP = "in_app"
    WEB_PUSH = "web_push"


class NotificationStatus(StrEnum):
    PENDING = "PENDING"
    DELIVERED = "DELIVERED"
    BLOCKED_BY_CONFIGURATION = "BLOCKED_BY_CONFIGURATION"


class ImportStatus(StrEnum):
    PREVIEW = "PREVIEW"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


class ModelDriftStatus(StrEnum):
    INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
    STABLE = "STABLE"
    DEGRADED = "DEGRADED"


class PaperReadoutPeriod(StrEnum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DailyOperationStatus(OperationalModel):
    business_date: date
    pipeline_state: PipelineState
    updated_at: datetime
    data_as_of: datetime | None = None
    morning_data_as_of: datetime | None = None
    proposal_id: str | None = None
    blocking_reason: str | None = None
    is_stale: bool = False

    _validate_updated = field_validator("updated_at")(_aware)

    @field_validator("data_as_of", "morning_data_as_of")
    @classmethod
    def validate_optional_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @model_validator(mode="after")
    def fail_closed_state(self) -> Self:
        blocked = self.pipeline_state in {
            PipelineState.STALE_DATA,
            PipelineState.DATA_ERROR,
            PipelineState.MODEL_ERROR,
        }
        if blocked and not self.blocking_reason:
            raise ValueError("blocked/error pipeline states require a reason")
        proposal_states = {
            PipelineState.PROPOSAL_READY,
            PipelineState.USER_DECISION_SAVED,
            PipelineState.EXECUTION_PENDING,
            PipelineState.EXECUTION_RECORDED,
            PipelineState.MARKET_CLOSED,
        }
        if self.pipeline_state in proposal_states and not self.proposal_id:
            raise ValueError("proposal-bearing pipeline states require an archived proposal")
        return self


class AutomationJobRecord(OperationalModel):
    run_id: str = Field(min_length=1)
    idempotency_key: str = Field(min_length=1)
    business_date: date
    stage: AutomationStage
    status: JobStatus
    started_at: datetime
    finished_at: datetime | None = None
    artifact_id: str | None = None
    data_as_of: datetime | None = None
    morning_data_as_of: datetime | None = None
    reason_code: str | None = None
    detail: str | None = None

    _validate_started = field_validator("started_at")(_aware)

    @field_validator("data_as_of", "morning_data_as_of")
    @classmethod
    def validate_optional_provenance_times(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @field_validator("finished_at")
    @classmethod
    def validate_finished(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        if self.status is JobStatus.RUNNING and self.finished_at is not None:
            raise ValueError("running job cannot have finished_at")
        if self.status is not JobStatus.RUNNING and self.finished_at is None:
            raise ValueError("terminal job requires finished_at")
        if self.status in {JobStatus.BLOCKED, JobStatus.FAILED} and not self.reason_code:
            raise ValueError("blocked/failed job requires a stable reason code")
        return self


class NotificationRecord(OperationalModel):
    notification_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    channel: NotificationChannel
    status: NotificationStatus
    created_at: datetime
    delivered_at: datetime | None = None
    title: str = Field(min_length=1, max_length=120)
    body: str = Field(min_length=1, max_length=500)

    _validate_created = field_validator("created_at")(_aware)

    @field_validator("delivered_at")
    @classmethod
    def validate_delivered(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)


class DecisionReview(OperationalModel):
    proposal_id: str = Field(min_length=1)
    selected_targets: Mapping[str, int]
    estimated_cash_after: Mapping[str, Decimal]
    estimated_buy_value: Decimal = Field(ge=0)
    estimated_sell_value: Decimal = Field(ge=0)
    estimated_transaction_cost: Decimal = Field(ge=0)
    estimated_tax_effect: Decimal
    resulting_positions: int = Field(ge=0)
    constraint_violations: tuple[str, ...]
    is_order_instruction: bool = False

    @field_validator("selected_targets", "estimated_cash_after", mode="after")
    @classmethod
    def freeze_mapping(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(value))

    @field_serializer("selected_targets", "estimated_cash_after")
    def serialize_mapping(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @model_validator(mode="after")
    def decision_support_only(self) -> Self:
        if self.is_order_instruction:
            raise ValueError("decision review cannot be an order instruction")
        return self


class DecisionPolicySnapshot(OperationalModel):
    """Exact Decision Engine configuration used to create one archived proposal."""

    proposal_id: str = Field(min_length=1)
    captured_at: datetime
    config: DecisionEngineConfig

    _validate_captured_at = field_validator("captured_at")(_aware)


class RankingRecord(OperationalModel):
    ranking_id: str = Field(min_length=1)
    as_of: datetime
    symbol: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    rank_type: str = Field(min_length=1)
    rank: int = Field(ge=1)
    total_universe: int = Field(ge=1)
    candidate_status: str = Field(min_length=1)
    portfolio_action: str | None = None
    model_bundle_version: str = Field(min_length=1)
    data_snapshot_id: str = Field(min_length=1)

    _validate_as_of = field_validator("as_of")(_aware)

    @model_validator(mode="after")
    def rank_within_universe(self) -> Self:
        if self.rank > self.total_universe:
            raise ValueError("rank cannot exceed total universe")
        return self


class PaperCalendarSnapshot(OperationalModel):
    """Immutable, content-addressed JPX session calendar used by Paper outcomes."""

    source_snapshot_id: str = Field(min_length=1)
    session_dates: tuple[date, ...]
    created_at: datetime

    _validate_created_at = field_validator("created_at")(_aware)

    @field_validator("session_dates")
    @classmethod
    def valid_sessions(cls, value: tuple[date, ...]) -> tuple[date, ...]:
        if len(value) < 2 or tuple(sorted(set(value))) != value:
            raise ValueError("Paper calendar sessions must be unique and increasing")
        return value

    @property
    def calendar_snapshot_id(self) -> str:
        material = "|".join(
            (
                self.source_snapshot_id,
                self.created_at.isoformat(),
                *(item.isoformat() for item in self.session_dates),
            )
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
        return f"paper-calendar-{digest}"


class PaperOutcome(OperationalModel):
    outcome_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    horizon_sessions: int = Field(gt=0)
    horizon_session_dates: tuple[date, ...]
    label_end_at: datetime
    label_available_at: datetime
    observed_at: datetime
    proposal_return: float
    benchmark_return: float
    estimated_cost: Decimal = Field(ge=0)
    actual_cost: Decimal | None = Field(default=None, ge=0)
    estimated_tax_effect: Decimal
    audited_tax_effect: Decimal | None = None
    champion_version: str = Field(min_length=1)
    champion_absolute_error: float = Field(ge=0)
    challenger_version: str | None = None
    challenger_absolute_error: float | None = Field(default=None, ge=0)
    calendar_snapshot_id: str = Field(min_length=1)
    source_snapshot_ids: tuple[str, ...]

    _validate_times = field_validator("label_end_at", "label_available_at", "observed_at")(_aware)

    @field_validator(
        "proposal_return",
        "benchmark_return",
        "champion_absolute_error",
        "challenger_absolute_error",
    )
    @classmethod
    def finite_metric(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("paper metrics must be finite")
        return value

    @model_validator(mode="after")
    def source_lineage_required(self) -> Self:
        if not self.source_snapshot_ids or len(self.source_snapshot_ids) != len(
            set(self.source_snapshot_ids)
        ):
            raise ValueError("paper outcome requires unique source snapshot IDs")
        if self.calendar_snapshot_id not in self.source_snapshot_ids:
            raise ValueError("paper outcome calendar must be part of source lineage")
        if len(self.horizon_session_dates) != self.horizon_sessions:
            raise ValueError("paper outcome session path must match its horizon")
        if tuple(sorted(set(self.horizon_session_dates))) != self.horizon_session_dates:
            raise ValueError("paper outcome session dates must be unique and increasing")
        end_date = self.label_end_at.astimezone(ZoneInfo("Asia/Tokyo")).date()
        if self.horizon_session_dates[-1] != end_date:
            raise ValueError("paper label endpoint must match its final session date")
        if self.label_available_at < self.label_end_at:
            raise ValueError("paper label cannot be available before its endpoint")
        if self.observed_at < self.label_available_at:
            raise ValueError("paper observation cannot precede label availability")
        if (self.challenger_version is None) != (self.challenger_absolute_error is None):
            raise ValueError("challenger version and error must be recorded together")
        return self


class PaperSummary(OperationalModel):
    observations: int = Field(ge=0)
    proposal_return: float | None
    benchmark_return: float | None
    excess_return: float | None
    maximum_drawdown: float | None
    mean_cost_error: Decimal | None
    mean_tax_error: Decimal | None
    champion_mean_absolute_error: float | None
    challenger_mean_absolute_error: float | None
    challenger_better_rate: float | None
    champion_version: str | None
    champion_observations: int = Field(ge=0)
    challenger_version: str | None
    challenger_observations: int = Field(ge=0)
    drift_status: ModelDriftStatus
    drift_ratio: float | None
    drift_window: int = Field(gt=0)
    minimum_observation_count: int = Field(gt=0)
    is_decision_ready: bool
    model_monitoring_ready: bool
    updated_at: datetime | None

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)


class PaperReadout(OperationalModel):
    period: PaperReadoutPeriod
    period_key: str = Field(min_length=1)
    period_start: date
    period_end: date
    observations: int = Field(gt=0)
    proposal_return: float
    benchmark_return: float
    excess_return: float
    mean_cost_error: Decimal | None
    mean_tax_error: Decimal | None
    champion_mean_absolute_error: float
    challenger_mean_absolute_error: float | None
    challenger_better_rate: float | None
    champion_versions: tuple[str, ...]
    challenger_versions: tuple[str, ...]
    updated_at: datetime

    _validate_updated = field_validator("updated_at")(_aware)

    @field_validator(
        "proposal_return",
        "benchmark_return",
        "excess_return",
        "champion_mean_absolute_error",
        "challenger_mean_absolute_error",
        "challenger_better_rate",
    )
    @classmethod
    def finite_optional_metric(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("paper readout metrics must be finite")
        return value

    @model_validator(mode="after")
    def valid_period(self) -> Self:
        if self.period_end < self.period_start:
            raise ValueError("paper readout period end precedes start")
        if not self.champion_versions:
            raise ValueError("paper readout requires champion provenance")
        return self


class PaperSeriesPoint(OperationalModel):
    observed_at: datetime
    observations: int = Field(gt=0)
    proposal_return: float
    benchmark_return: float
    excess_return: float

    _validate_observed = field_validator("observed_at")(_aware)

    @field_validator("proposal_return", "benchmark_return", "excess_return")
    @classmethod
    def finite_series_metric(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("paper series metrics must be finite")
        return value


class ImportConflict(OperationalModel):
    conflict_id: str = Field(min_length=1)
    row_number: int = Field(ge=2)
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    record_identity: str | None = Field(default=None, min_length=1)


class ExecutionImportPreview(OperationalModel):
    preview_id: str = Field(min_length=1)
    source_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime
    status: ImportStatus = ImportStatus.PREVIEW
    execution_payloads: tuple[str, ...]
    conflicts: tuple[ImportConflict, ...]

    _validate_created = field_validator("created_at")(_aware)


class PositionDifference(OperationalModel):
    symbol: str = Field(min_length=1)
    account_bucket_id: str = Field(min_length=1)
    ledger_shares: int = Field(ge=0)
    imported_shares: int = Field(ge=0)
    ledger_average_price: Decimal | None = Field(default=None, ge=0)
    imported_average_price: Decimal | None = Field(default=None, ge=0)
    ledger_market_price: Decimal | None = Field(default=None, ge=0)
    imported_market_price: Decimal | None = Field(default=None, ge=0)


class CashDifference(OperationalModel):
    account_bucket_id: str = Field(min_length=1)
    ledger_available_cash: Decimal = Field(ge=0)
    imported_available_cash: Decimal = Field(ge=0)
    ledger_reserved_cash: Decimal = Field(ge=0)
    imported_reserved_cash: Decimal = Field(ge=0)


class PositionReconciliationPreview(OperationalModel):
    preview_id: str = Field(min_length=1)
    source_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime
    status: ImportStatus = ImportStatus.PREVIEW
    imported_as_of: datetime
    portfolio_payload: str = Field(min_length=2)
    differences: tuple[PositionDifference, ...]
    cash_differences: tuple[CashDifference, ...]

    _validate_times = field_validator("created_at", "imported_as_of")(_aware)
