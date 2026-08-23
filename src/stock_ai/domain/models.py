"""Immutable domain models for proposals, decisions, and actual executions.

Money is represented by :class:`decimal.Decimal`.  Research features and model
outputs remain floating point because they are statistical quantities.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    realized_gain_ytd: Money = Decimal("0")
    realized_loss_ytd: Money = Decimal("0")
    loss_carryforward_user_input: Money = Decimal("0")
    nisa_annual_capacity_user_input: Money | None = None
    nisa_lifetime_capacity_user_input: Money | None = None


class MarketSnapshot(DomainModel):
    snapshot_id: str = Field(min_length=1)
    as_of: datetime
    available_at: datetime
    prices: dict[str, Money]
    source: str = Field(min_length=1)

    _validate_as_of = field_validator("as_of", "available_at")(_require_aware)

    @model_validator(mode="after")
    def available_before_as_of(self) -> Self:
        if self.available_at > self.as_of:
            raise ValueError("market snapshot is not available at as_of")
        return self


class PredictionUncertainty(DomainModel):
    standard_error: float = Field(ge=0)
    model_disagreement: float = Field(default=0.0, ge=0)
    coverage_warning: str | None = None

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

    _validate_as_of = field_validator("as_of")(_require_aware)

    @model_validator(mode="after")
    def unique_and_referenced_keys(self) -> Self:
        account_ids = [account.account_id for account in self.accounts]
        bucket_ids = [bucket.bucket_id for bucket in self.account_buckets]
        position_keys = [position.key for position in self.positions]
        if len(account_ids) != len(set(account_ids)):
            raise ValueError("account_id must be unique")
        if len(bucket_ids) != len(set(bucket_ids)):
            raise ValueError("bucket_id must be unique")
        if len(position_keys) != len(set(position_keys)):
            raise ValueError("symbol/account bucket position keys must be unique")
        if any(bucket.account_id not in account_ids for bucket in self.account_buckets):
            raise ValueError("account bucket references an unknown account")
        if any(position.account_bucket_id not in bucket_ids for position in self.positions):
            raise ValueError("position references an unknown account bucket")
        if any(item.account_bucket_id not in bucket_ids for item in self.cash):
            raise ValueError("cash state references an unknown account bucket")
        if any(item.account_bucket_id not in bucket_ids for item in self.tax_states):
            raise ValueError("tax state references an unknown account bucket")
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
    estimated_cash_required_or_released: SignedMoney
    hold_expected_value: SignedMoney
    proposed_expected_value: SignedMoney
    transaction_cost: TransactionCostEstimate
    estimated_tax_effect: SignedMoney
    net_expected_improvement: SignedMoney
    downside_risk: float
    uncertainty: float = Field(ge=0)
    reason_codes: tuple[str, ...]

    @model_validator(mode="after")
    def share_difference_matches(self) -> Self:
        expected = self.recommended_shares - self.current_shares
        if self.share_difference != expected:
            raise ValueError("share_difference does not match recommended - current")
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
    estimated_cash_after: dict[str, SignedMoney]
    model_bundle_version: str = Field(min_length=1)
    decision_engine_version: str = Field(min_length=1)
    cost_policy_id: str = Field(min_length=1)
    tax_policy_id: str = Field(min_length=1)
    no_trade_reason: str | None = None
    is_order_instruction: bool = False

    _validate_times = field_validator("as_of", "generated_at")(_require_aware)

    @model_validator(mode="after")
    def decision_support_only(self) -> Self:
        if self.is_order_instruction:
            raise ValueError("portfolio proposals can never be order instructions")
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
        return self
