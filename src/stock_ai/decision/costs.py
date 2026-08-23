"""Configurable commission, spread, slippage, and market-impact estimates."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from stock_ai.domain import TransactionCostEstimate

BPS = Decimal("10000")


class CostPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    commission_fixed: Decimal = Field(default=Decimal("0"), ge=0)
    commission_bps: Decimal = Field(default=Decimal("0"), ge=0)
    minimum_commission: Decimal = Field(default=Decimal("0"), ge=0)
    full_spread_bps: Decimal = Field(default=Decimal("10"), ge=0)
    slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    impact_bps_at_full_adv: Decimal = Field(default=Decimal("25"), ge=0)


class TransactionCostEngine:
    def __init__(self, policy: CostPolicy) -> None:
        self.policy = policy

    def estimate(
        self,
        *,
        shares: int,
        price: Decimal,
        average_daily_trading_value: Decimal | None,
    ) -> TransactionCostEstimate:
        if shares < 0 or price < 0:
            raise ValueError("shares and price must be non-negative")
        notional = price * shares
        if notional == 0:
            return TransactionCostEstimate(policy_version=self.policy.version)
        variable_commission = notional * self.policy.commission_bps / BPS
        commission = max(
            self.policy.minimum_commission,
            self.policy.commission_fixed + variable_commission,
        )
        spread = notional * (self.policy.full_spread_bps / Decimal("2")) / BPS
        slippage = notional * self.policy.slippage_bps / BPS
        assumptions: list[str] = []
        if average_daily_trading_value is None or average_daily_trading_value <= 0:
            participation = Decimal("1")
            assumptions.append("market impact uses conservative 100% ADV participation")
        else:
            participation = min(Decimal("1"), notional / average_daily_trading_value)
        impact_rate_bps = self.policy.impact_bps_at_full_adv * participation.sqrt()
        market_impact = notional * impact_rate_bps / BPS
        return TransactionCostEstimate(
            commission=commission,
            spread=spread,
            slippage=slippage,
            market_impact=market_impact,
            policy_version=self.policy.version,
            assumptions=tuple(assumptions),
        )
