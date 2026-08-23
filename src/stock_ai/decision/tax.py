"""Versioned, explicitly approximate Japanese cash-equity tax interface."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from stock_ai.domain import AccountBucket, AccountType, Position, TaxState


class TaxPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    effective_from: date
    taxable_rate: Decimal = Field(default=Decimal("0.20315"), ge=0, le=1)
    nisa_opportunity_cost_rate: Decimal = Field(default=Decimal("0"), ge=0, le=1)


class TaxEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    realized_pnl: Decimal
    immediate_tax_effect: Decimal
    policy_version: str
    is_estimate: bool = True
    assumptions: tuple[str, ...]


class TaxEngine(Protocol):
    policy: TaxPolicy

    def estimate_sale(
        self,
        *,
        account_bucket: AccountBucket,
        position: Position,
        sell_shares: int,
        expected_sell_price: Decimal,
        tax_state: TaxState,
    ) -> TaxEstimate: ...


class SimpleJapanTaxEngine:
    """A conservative decision estimate, not a tax-return calculator."""

    def __init__(self, policy: TaxPolicy) -> None:
        self.policy = policy

    def estimate_sale(
        self,
        *,
        account_bucket: AccountBucket,
        position: Position,
        sell_shares: int,
        expected_sell_price: Decimal,
        tax_state: TaxState,
    ) -> TaxEstimate:
        if sell_shares < 0 or sell_shares > position.shares:
            raise ValueError("sell shares must be within the held quantity")
        realized_pnl = (expected_sell_price - position.average_acquisition_price) * sell_shares
        if account_bucket.account_type is AccountType.NISA:
            opportunity_cost = (
                expected_sell_price * sell_shares * self.policy.nisa_opportunity_cost_rate
            )
            return TaxEstimate(
                realized_pnl=realized_pnl,
                immediate_tax_effect=opportunity_cost,
                policy_version=self.policy.version,
                assumptions=(
                    "NISA realized gains use zero immediate income-tax effect",
                    "NISA losses are not offset against taxable-account gains",
                ),
            )

        net_realized_ytd = tax_state.realized_gain_ytd - tax_state.realized_loss_ytd
        available_loss_offset = tax_state.loss_carryforward_user_input + max(
            Decimal("0"), -net_realized_ytd
        )
        assumptions = ["estimated tax only; final filing and broker withholding may differ"]
        if realized_pnl >= 0:
            taxable_gain = max(Decimal("0"), realized_pnl - available_loss_offset)
            tax_effect = taxable_gain * self.policy.taxable_rate
        else:
            previously_taxable_gain = max(
                Decimal("0"), net_realized_ytd - tax_state.loss_carryforward_user_input
            )
            loss_used_now = min(-realized_pnl, previously_taxable_gain)
            tax_effect = -(loss_used_now * self.policy.taxable_rate)
            assumptions.append("loss benefit is capped by estimated taxable gains realized YTD")
        if account_bucket.account_type is AccountType.UNKNOWN_MANUAL:
            assumptions.append("unknown account type treated conservatively as taxable")
        return TaxEstimate(
            realized_pnl=realized_pnl,
            immediate_tax_effect=tax_effect,
            policy_version=self.policy.version,
            assumptions=tuple(assumptions),
        )
