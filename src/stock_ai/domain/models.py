"""Immutable domain models for proposals, decisions, and actual executions.

Money is represented by :class:`decimal.Decimal`.  Research features and model
outputs remain floating point because they are statistical quantities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

Money = Annotated[Decimal, Field(ge=0)]
SignedMoney = Decimal
Shares = Annotated[int, Field(ge=0)]


class DomainModel(BaseModel):
    """Strict immutable base model used for auditable records."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class AccountType(StrEnum):
    NISA = "nisa"
    TAXABLE_SPECIFIED = "taxable_specified"
    TAXABLE_GENERAL = "taxable_general"
    UNKNOWN_MANUAL = "unknown_manual"


class WithholdingMode(StrEnum):
    WITHHOLDING = "withholding"
    NO_WITHHOLDING = "no_withholding"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class ProposalAction(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"
    SKIP = "SKIP"


class TradeSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class ExecutionStatus(StrEnum):
    NOT_ORDERED = "not_ordered"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


class Security(DomainModel):
    symbol: str = Field(min_length=1, max_length=16)
    company_name: str = Field(min_length=1)
    sector: str = Field(min_length=1)
    market_segment: str = "TSE Prime"
    lot_size: int = Field(default=100, gt=0)


class Account(DomainModel):
    account_id: str = Field(min_length=1)
    broker: str = Field(min_length=1)
    display_name: str = Field(min_length=1)


class AccountBucket(DomainModel):
    bucket_id: str = Field(min_length=1)
    account_id: str = Field(min_length=1)
    account_type: AccountType
    withholding_mode: WithholdingMode
    fee_policy_id: str = Field(min_length=1)
    tax_policy_id: str = Field(min_length=1)


class Position(DomainModel):
    symbol: str = Field(min_length=1)
    account_bucket_id: str = Field(min_length=1)
    shares: Shares
    average_acquisition_price: Money
    market_price: Money

    @property
    def key(self) -> tuple[str, str]:
        return (self.symbol, self.account_bucket_id)

    @property
    def market_value(self) -> Decimal:
        return self.market_price * self.shares

    @property
    def book_value(self) -> Decimal:
        return self.average_acquisition_price * self.shares


class CashState(DomainModel):
    account_bucket_id: str = Field(min_length=1)
    available_cash: Money
    reserved_cash: Money = Decimal("0")

    @property
    def deployable_cash(self) -> Decimal:
        return self.available_cash - self.reserved_cash

    @model_validator(mode="after")
    def reserved_does_not_exceed_available(self) -> Self:
        if self.reserved_cash > self.available_cash:
            raise ValueError("reserved cash cannot exceed available cash")
        return self


class TaxState(DomainModel):
    account_bucket_id: str = Field(min_length=1)
    tax_year: int = Field(ge=2000, le=2200)
    realized_gain_ytd: Money = Decimal("0")
    realized_loss_ytd: Money = Decimal("0")
    loss_carryforward_user_input: Money = Decimal("0")
    nisa_annual_capacity_user_input: Money | None = None
    nisa_lifetime_capacity_user_input: Money | None = None


class MarketSnapshot(DomainModel):
    snapshot_id: str = Field(min_length=1)
    as_of: datetime
    available_at: datetime
    prices: Mapping[str, Money]
    source: str = Field(min_length=1)

    _validate_as_of = field_validator("as_of", "available_at")(_require_aware)

    @field_validator("prices", mode="after")
    @classmethod
    def freeze_prices(cls, value: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
        return MappingProxyType(dict(value))

    @field_serializer("prices")
    def serialize_prices(self, value: Mapping[str, Decimal]) -> dict[str, Decimal]:
        return dict(value)

    @model_validator(mode="after")
    def available_before_as_of(self) -> Self:
        if self.available_at > self.as_of:
            raise ValueError("market snapshot is not available at as_of")
        return self


class PredictionUncertainty(DomainModel):
    standard_error: float = Field(ge=0)
    model_disagreement: float = Field(default=0.0, ge=0)
    coverage_warning: str | None = None

    @field_validator("standard_error", "model_disagreement")
    @classmethod
    def finite_uncertainty(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("uncertainty values must be finite")
        return value

    @property
    def combined(self) -> float:
        return self.standard_error + self.model_disagreement


class Prediction(DomainModel):
    symbol: str = Field(min_length=1)
    as_of: datetime
    expected_return_1d: float
    expected_return_5d: float
    expected_return_20d: float
    downside_quantile: float
    large_loss_probability: float = Field(ge=0, le=1)
    uncertainty: PredictionUncertainty
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    data_snapshot_id: str = Field(min_length=1)

    _validate_as_of = field_validator("as_of")(_require_aware)

    @field_validator(
        "expected_return_1d",
        "expected_return_5d",
        "expected_return_20d",
        "downside_quantile",
        "large_loss_probability",
    )
    @classmethod
    def finite_prediction_values(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("prediction values must be finite")
        return value


class TransactionCostEstimate(DomainModel):
    commission: Money = Decimal("0")
    spread: Money = Decimal("0")
    slippage: Money = Decimal("0")
    market_impact: Money = Decimal("0")
    policy_version: str = Field(min_length=1)
    assumptions: tuple[str, ...] = ()

    @property
    def total(self) -> Decimal:
        return self.commission + self.spread + self.slippage + self.market_impact


class PortfolioState(DomainModel):
    portfolio_id: str = Field(min_length=1)
    as_of: datetime
    accounts: tuple[Account, ...]
    account_buckets: tuple[AccountBucket, ...]
    positions: tuple[Position, ...]
    cash: tuple[CashState, ...]
    tax_states: tuple[TaxState, ...]
    applied_execution_ids: tuple[str, ...] = ()

    _validate_as_of = field_validator("as_of")(_require_aware)

    @model_validator(mode="after")
    def unique_and_referenced_keys(self) -> Self:
        account_ids = [account.account_id for account in self.accounts]
        bucket_ids = [bucket.bucket_id for bucket in self.account_buckets]
        position_keys = [position.key for position in self.positions]
        cash_bucket_ids = [item.account_bucket_id for item in self.cash]
        tax_bucket_ids = [item.account_bucket_id for item in self.tax_states]
        if len(self.applied_execution_ids) != len(set(self.applied_execution_ids)):
            raise ValueError("applied execution IDs must be unique")
        if len(account_ids) != len(set(account_ids)):
            raise ValueError("account_id must be unique")
        if len(bucket_ids) != len(set(bucket_ids)):
            raise ValueError("bucket_id must be unique")
        if len(position_keys) != len(set(position_keys)):
            raise ValueError("symbol/account bucket position keys must be unique")
        if len(cash_bucket_ids) != len(set(cash_bucket_ids)):
            raise ValueError("cash state must be unique by account bucket")
        if len(tax_bucket_ids) != len(set(tax_bucket_ids)):
            raise ValueError("tax state must be unique by account bucket")
        if any(bucket.account_id not in account_ids for bucket in self.account_buckets):
            raise ValueError("account bucket references an unknown account")
        if any(position.account_bucket_id not in bucket_ids for position in self.positions):
            raise ValueError("position references an unknown account bucket")
        if any(item.account_bucket_id not in bucket_ids for item in self.cash):
            raise ValueError("cash state references an unknown account bucket")
        if any(item.account_bucket_id not in bucket_ids for item in self.tax_states):
            raise ValueError("tax state references an unknown account bucket")
        if set(cash_bucket_ids) != set(bucket_ids):
            raise ValueError("every account bucket requires exactly one cash state")
        if set(tax_bucket_ids) != set(bucket_ids):
            raise ValueError("every account bucket requires exactly one tax state")
        return self

    def position_map(self) -> dict[tuple[str, str], Position]:
        return {position.key: position for position in self.positions}

    def cash_map(self) -> dict[str, CashState]:
        return {cash.account_bucket_id: cash for cash in self.cash}

    def tax_state_map(self) -> dict[str, TaxState]:
        return {state.account_bucket_id: state for state in self.tax_states}


class TargetPosition(DomainModel):
    symbol: str = Field(min_length=1)
    account_bucket_id: str = Field(min_length=1)
    target_shares: Shares

    @property
    def key(self) -> tuple[str, str]:
        return (self.symbol, self.account_bucket_id)


class ProposalLine(DomainModel):
    line_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    company_name: str = Field(min_length=1)
    account_bucket_id: str = Field(min_length=1)
    current_shares: Shares
    recommended_shares: Shares
    share_difference: int
    action: ProposalAction
    reference_price: Money
    current_market_value: Money
    recommended_market_value: Money
    estimated_cash_required_or_released: SignedMoney
    hold_expected_value: SignedMoney
    proposed_expected_value: SignedMoney
    transaction_cost: TransactionCostEstimate
    estimated_tax_effect: SignedMoney
    estimated_nisa_opportunity_cost: Money
    estimated_realized_pnl: SignedMoney
    estimated_tax_cash_withholding: Money
    tax_policy_version: str = Field(min_length=1)
    tax_is_estimate: bool
    tax_assumptions: tuple[str, ...]
    net_expected_improvement: SignedMoney
    downside_risk: float
    uncertainty: float = Field(ge=0)
    reason_codes: tuple[str, ...]
    human_readable_reasons: tuple[str, ...]

    @model_validator(mode="after")
    def share_difference_matches(self) -> Self:
        expected = self.recommended_shares - self.current_shares
        if self.share_difference != expected:
            raise ValueError("share_difference does not match recommended - current")
        expected_action = (
            ProposalAction.SKIP
            if self.current_shares == 0 and self.recommended_shares == 0
            else ProposalAction.SELL
            if self.current_shares > 0 and self.recommended_shares == 0
            else ProposalAction.BUY
            if self.recommended_shares > self.current_shares
            else ProposalAction.REDUCE
            if self.recommended_shares < self.current_shares
            else ProposalAction.HOLD
        )
        if self.action is not expected_action:
            raise ValueError("proposal action does not match current/recommended shares")
        if self.current_market_value != self.reference_price * self.current_shares:
            raise ValueError("current market value does not match shares and reference price")
        if self.recommended_market_value != self.reference_price * self.recommended_shares:
            raise ValueError("recommended market value does not match shares and reference price")
        return self


class PortfolioProposal(DomainModel):
    proposal_id: str = Field(min_length=1)
    as_of: datetime
    generated_at: datetime
    current_portfolio_id: str = Field(min_length=1)
    targets: tuple[TargetPosition, ...]
    lines: tuple[ProposalLine, ...]
    hold_utility: SignedMoney
    proposed_utility: SignedMoney
    net_improvement: SignedMoney
    estimated_cash_after: Mapping[str, SignedMoney]
    model_bundle_version: str = Field(min_length=1)
    decision_engine_version: str = Field(min_length=1)
    cost_policy_id: str = Field(min_length=1)
    cost_policy_version: str = Field(min_length=1)
    tax_policy_id: str = Field(min_length=1)
    tax_policy_version: str = Field(min_length=1)
    no_trade_reason: str | None = None
    is_order_instruction: bool = False

    _validate_times = field_validator("as_of", "generated_at")(_require_aware)

    @field_validator("estimated_cash_after", mode="after")
    @classmethod
    def freeze_cash_after(cls, value: Mapping[str, Decimal]) -> Mapping[str, Decimal]:
        return MappingProxyType(dict(value))

    @field_serializer("estimated_cash_after")
    def serialize_cash_after(self, value: Mapping[str, Decimal]) -> dict[str, Decimal]:
        return dict(value)

    @model_validator(mode="after")
    def decision_support_only(self) -> Self:
        if self.is_order_instruction:
            raise ValueError("portfolio proposals can never be order instructions")
        if self.generated_at < self.as_of:
            raise ValueError("proposal generated_at cannot precede its as_of")
        if self.net_improvement != self.proposed_utility - self.hold_utility:
            raise ValueError("proposal net improvement must equal proposed minus HOLD utility")
        target_keys = [target.key for target in self.targets]
        line_keys = [(line.symbol, line.account_bucket_id) for line in self.lines]
        line_ids = [line.line_id for line in self.lines]
        if len(target_keys) != len(set(target_keys)):
            raise ValueError("proposal targets must be unique by symbol/account bucket")
        if len(line_keys) != len(set(line_keys)) or len(line_ids) != len(set(line_ids)):
            raise ValueError("proposal lines must have unique keys and IDs")
        if set(target_keys) != set(line_keys):
            raise ValueError("proposal targets and lines must describe the same keys")
        target_map = {target.key: target.target_shares for target in self.targets}
        if any(
            target_map[(line.symbol, line.account_bucket_id)] != line.recommended_shares
            for line in self.lines
        ):
            raise ValueError("proposal target shares must match proposal lines")
        return self


class UserDecisionLine(DomainModel):
    proposal_line_id: str = Field(min_length=1)
    selected_target_shares: Shares
    note: str | None = None


class UserDecision(DomainModel):
    decision_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    saved_at: datetime
    lines: tuple[UserDecisionLine, ...]
    confirms_manual_order_only: bool = True

    _validate_saved_at = field_validator("saved_at")(_require_aware)

    @model_validator(mode="after")
    def never_authorizes_automatic_orders(self) -> Self:
        if not self.confirms_manual_order_only:
            raise ValueError("a user decision cannot authorize automatic order submission")
        line_ids = [line.proposal_line_id for line in self.lines]
        if len(line_ids) != len(set(line_ids)):
            raise ValueError("user decision line IDs must be unique")
        return self


class ExecutionRecord(DomainModel):
    execution_id: str = Field(min_length=1)
    decision_id: str = Field(min_length=1)
    executed_at: datetime
    symbol: str = Field(min_length=1)
    account_bucket_id: str = Field(min_length=1)
    status: ExecutionStatus
    side: TradeSide
    ordered_shares: Shares
    filled_shares: Shares
    average_fill_price: Money | None = None
    actual_commission: Money = Decimal("0")
    actual_other_cost: Money = Decimal("0")
    tax_withheld: Money = Decimal("0")
    source: str = Field(default="manual", min_length=1)

    _validate_executed_at = field_validator("executed_at")(_require_aware)

    @model_validator(mode="after")
    def valid_fill(self) -> Self:
        if self.filled_shares > self.ordered_shares:
            raise ValueError("filled shares cannot exceed ordered shares")
        if self.filled_shares > 0 and self.average_fill_price is None:
            raise ValueError("filled executions require an average fill price")
        if self.filled_shares == 0 and (
            self.average_fill_price is not None
            or self.actual_commission != 0
            or self.actual_other_cost != 0
            or self.tax_withheld != 0
        ):
            raise ValueError("zero-fill executions cannot contain fill price, costs, or tax")
        if self.status in {ExecutionStatus.NOT_ORDERED, ExecutionStatus.OPEN}:
            if self.filled_shares != 0:
                raise ValueError("not-ordered/open executions cannot contain fills")
        elif self.status is ExecutionStatus.PARTIALLY_FILLED:
            if not 0 < self.filled_shares < self.ordered_shares:
                raise ValueError("partially-filled executions require a partial quantity")
        elif self.status is ExecutionStatus.FILLED:
            if self.ordered_shares == 0 or self.filled_shares != self.ordered_shares:
                raise ValueError("filled executions require the full ordered quantity")
        elif self.status in {ExecutionStatus.CANCELLED, ExecutionStatus.EXPIRED}:
            if self.ordered_shares > 0 and self.filled_shares == self.ordered_shares:
                raise ValueError("fully filled executions must use FILLED status")
        return self
