"""Versioned, explicitly approximate Japanese cash-equity tax interface."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stock_ai.domain import AccountBucket, AccountType, Position, TaxState, WithholdingMode


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
    nisa_opportunity_cost: Decimal = Field(default=Decimal("0"), ge=0)
    estimated_cash_withholding: Decimal = Field(ge=0)
    policy_version: str
    is_estimate: bool = True
    assumptions: tuple[str, ...]


class SaleTaxInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allocation_id: str = Field(min_length=1)
    position: Position
    sell_shares: int = Field(gt=0)
    expected_sell_price: Decimal = Field(ge=0)
    estimated_deductible_cost: Decimal = Field(default=Decimal("0"), ge=0)

    @model_validator(mode="after")
    def valid_quantity(self) -> Self:
        if self.sell_shares > self.position.shares:
            raise ValueError("sell shares must be within the held quantity")
        return self


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

    def estimate_sales(
        self,
        *,
        account_bucket: AccountBucket,
        sales: tuple[SaleTaxInput, ...],
        tax_state: TaxState,
    ) -> dict[str, TaxEstimate]: ...


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
        if sell_shares == 0:
            return TaxEstimate(
                realized_pnl=Decimal("0"),
                immediate_tax_effect=Decimal("0"),
                nisa_opportunity_cost=Decimal("0"),
                estimated_cash_withholding=Decimal("0"),
                policy_version=self.policy.version,
                assumptions=("no sale",),
            )
        results = self.estimate_sales(
            account_bucket=account_bucket,
            sales=(
                SaleTaxInput(
                    allocation_id="single-sale",
                    position=position,
                    sell_shares=sell_shares,
                    expected_sell_price=expected_sell_price,
                ),
            ),
            tax_state=tax_state,
        )
        return results["single-sale"]

    def estimate_sales(
        self,
        *,
        account_bucket: AccountBucket,
        sales: tuple[SaleTaxInput, ...],
        tax_state: TaxState,
    ) -> dict[str, TaxEstimate]:
        if not sales:
            return {}
        if len({sale.allocation_id for sale in sales}) != len(sales):
            raise ValueError("tax allocation IDs must be unique")
        if tax_state.account_bucket_id != account_bucket.bucket_id:
            raise ValueError("tax state and account bucket must match")
        if any(sale.position.account_bucket_id != account_bucket.bucket_id for sale in sales):
            raise ValueError("every sale must belong to the evaluated account bucket")
        realized = {
            sale.allocation_id: (sale.expected_sell_price - sale.position.average_acquisition_price)
            * sale.sell_shares
            - sale.estimated_deductible_cost
            for sale in sales
        }
        proceeds = {
            sale.allocation_id: sale.expected_sell_price * sale.sell_shares for sale in sales
        }
        realized_pnl = sum(realized.values(), Decimal("0"))
        total_proceeds = sum(proceeds.values(), Decimal("0"))
        if account_bucket.account_type is AccountType.NISA:
            total_tax_effect = Decimal("0")
            total_opportunity_cost = total_proceeds * self.policy.nisa_opportunity_cost_rate
            assumptions: tuple[str, ...] = (
                "NISA realized gains use zero immediate income-tax effect",
                "NISA losses are not offset against taxable-account gains",
            )
            allocation_weights = proceeds
        else:
            total_opportunity_cost = Decimal("0")
            net_realized_ytd = tax_state.realized_gain_ytd - tax_state.realized_loss_ytd
            if net_realized_ytd >= 0:
                available_loss_offset = max(
                    Decimal("0"), tax_state.loss_carryforward_user_input - net_realized_ytd
                )
            else:
                available_loss_offset = tax_state.loss_carryforward_user_input - net_realized_ytd
            assumption_list = ["estimated tax only; final filing and withholding may differ"]
            if realized_pnl >= 0:
                taxable_gain = max(Decimal("0"), realized_pnl - available_loss_offset)
                total_tax_effect = taxable_gain * self.policy.taxable_rate
                allocation_weights = {
                    key: max(value, Decimal("0")) for key, value in realized.items()
                }
            else:
                previously_taxable_gain = max(
                    Decimal("0"), net_realized_ytd - tax_state.loss_carryforward_user_input
                )
                loss_used_now = min(-realized_pnl, previously_taxable_gain)
                total_tax_effect = -(loss_used_now * self.policy.taxable_rate)
                assumption_list.append(
                    "loss benefit is capped by estimated taxable gains realized YTD"
                )
                allocation_weights = {
                    key: max(-value, Decimal("0")) for key, value in realized.items()
                }
            if account_bucket.account_type is AccountType.UNKNOWN_MANUAL:
                assumption_list.append("unknown account type treated conservatively as taxable")
            assumptions = tuple(assumption_list)

        weight_total = sum(allocation_weights.values(), Decimal("0"))
        if weight_total == 0:
            allocation_weights = {key: proceeds[key] for key in realized}
            weight_total = sum(allocation_weights.values(), Decimal("0"))
        ordered_ids = sorted(realized)
        allocated: dict[str, Decimal] = {}
        remaining = total_tax_effect
        for allocation_id in ordered_ids[:-1]:
            share = (
                total_tax_effect * allocation_weights[allocation_id] / weight_total
                if weight_total
                else Decimal("0")
            )
            allocated[allocation_id] = share
            remaining -= share
        allocated[ordered_ids[-1]] = remaining
        allocated_opportunity = {
            allocation_id: (
                total_opportunity_cost * proceeds[allocation_id] / total_proceeds
                if total_proceeds
                else Decimal("0")
            )
            for allocation_id in ordered_ids
        }
        cash_withholding = {
            allocation_id: (
                max(Decimal("0"), allocated[allocation_id])
                if account_bucket.withholding_mode is WithholdingMode.WITHHOLDING
                else Decimal("0")
            )
            for allocation_id in ordered_ids
        }
        if account_bucket.withholding_mode is WithholdingMode.UNKNOWN:
            cash_withholding = {
                allocation_id: max(Decimal("0"), allocated[allocation_id])
                for allocation_id in ordered_ids
            }
        return {
            allocation_id: TaxEstimate(
                realized_pnl=realized[allocation_id],
                immediate_tax_effect=allocated[allocation_id],
                nisa_opportunity_cost=allocated_opportunity[allocation_id],
                estimated_cash_withholding=cash_withholding[allocation_id],
                policy_version=self.policy.version,
                assumptions=(*assumptions, "tax effect allocated from aggregate bucket sales"),
            )
            for allocation_id in ordered_ids
        }
