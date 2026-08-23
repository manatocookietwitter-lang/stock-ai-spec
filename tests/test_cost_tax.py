from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_ai.decision import (
    CostPolicy,
    SaleTaxInput,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
)
from stock_ai.domain import AccountType, Position, TaxState, WithholdingMode
from tests.conftest import account_components


def test_transaction_cost_components_are_all_included() -> None:
    engine = TransactionCostEngine(
        CostPolicy(
            policy_id="policy",
            version="v1",
            commission_fixed=Decimal("100"),
            full_spread_bps=Decimal("10"),
            slippage_bps=Decimal("5"),
            impact_bps_at_full_adv=Decimal("100"),
        )
    )
    estimate = engine.estimate(
        shares=100,
        price=Decimal("1000"),
        average_daily_trading_value=Decimal("100000"),
    )
    assert estimate.commission == Decimal("100")
    assert estimate.spread == Decimal("50")
    assert estimate.slippage == Decimal("50")
    assert estimate.market_impact == Decimal("1000")
    assert estimate.total == Decimal("1200")


def test_missing_liquidity_uses_disclosed_conservative_impact_assumption() -> None:
    engine = TransactionCostEngine(
        CostPolicy(
            policy_id="policy",
            version="v1",
            zero_commission_confirmed=True,
            impact_bps_at_full_adv=Decimal("25"),
        )
    )
    estimate = engine.estimate(shares=100, price=Decimal("1000"), average_daily_trading_value=None)
    assert estimate.market_impact == Decimal("250")
    assert "conservative" in estimate.assumptions[0]


def test_taxable_and_nisa_sales_use_different_tax_interfaces() -> None:
    policy = TaxPolicy(
        policy_id="tax",
        version="2026-v1",
        effective_from=date(2026, 1, 1),
        taxable_rate=Decimal("0.20"),
    )
    engine = SimpleJapanTaxEngine(policy)
    _, taxable_bucket, _, taxable_state = account_components()
    _, nisa_bucket, _, nisa_state = account_components(nisa=True)
    assert taxable_bucket.account_type is AccountType.TAXABLE_SPECIFIED
    position = Position(
        symbol="A",
        account_bucket_id=taxable_bucket.bucket_id,
        shares=100,
        average_acquisition_price=Decimal("500"),
        market_price=Decimal("1000"),
    )
    taxable = engine.estimate_sale(
        account_bucket=taxable_bucket,
        position=position,
        sell_shares=100,
        expected_sell_price=Decimal("1000"),
        tax_state=taxable_state,
    )
    nisa = engine.estimate_sale(
        account_bucket=nisa_bucket,
        position=position.model_copy(update={"account_bucket_id": nisa_bucket.bucket_id}),
        sell_shares=100,
        expected_sell_price=Decimal("1000"),
        tax_state=nisa_state,
    )
    assert taxable.immediate_tax_effect == Decimal("10000.00")
    assert nisa.immediate_tax_effect == Decimal("0")
    assert taxable.is_estimate and nisa.is_estimate


def test_tax_loss_benefit_is_capped_by_realized_gain_ytd() -> None:
    engine = SimpleJapanTaxEngine(
        TaxPolicy(
            policy_id="tax",
            version="v1",
            effective_from=date(2026, 1, 1),
            taxable_rate=Decimal("0.20"),
        )
    )
    _, bucket, _, _ = account_components()
    state = TaxState(
        account_bucket_id=bucket.bucket_id,
        tax_year=2026,
        realized_gain_ytd=Decimal("10000"),
    )
    position = Position(
        symbol="A",
        account_bucket_id=bucket.bucket_id,
        shares=100,
        average_acquisition_price=Decimal("1000"),
        market_price=Decimal("500"),
    )
    estimate = engine.estimate_sale(
        account_bucket=bucket,
        position=position,
        sell_shares=100,
        expected_sell_price=Decimal("500"),
        tax_state=state,
    )
    assert estimate.realized_pnl == Decimal("-50000")
    assert estimate.immediate_tax_effect == Decimal("-2000.00")


def test_loss_offset_is_allocated_once_across_all_bucket_sales() -> None:
    engine = SimpleJapanTaxEngine(
        TaxPolicy(
            policy_id="tax",
            version="v1",
            effective_from=date(2026, 1, 1),
            taxable_rate=Decimal("0.20"),
        )
    )
    _, bucket, _, state = account_components()
    state = state.model_copy(update={"loss_carryforward_user_input": Decimal("10000")})
    sales = tuple(
        SaleTaxInput(
            allocation_id=symbol,
            position=Position(
                symbol=symbol,
                account_bucket_id=bucket.bucket_id,
                shares=100,
                average_acquisition_price=Decimal("1000"),
                market_price=Decimal("1100"),
            ),
            sell_shares=100,
            expected_sell_price=Decimal("1100"),
        )
        for symbol in ("A", "B")
    )
    estimates = engine.estimate_sales(
        account_bucket=bucket,
        sales=sales,
        tax_state=state,
    )
    assert sum((item.immediate_tax_effect for item in estimates.values()), Decimal("0")) == Decimal(
        "2000.00"
    )


def test_withholding_and_nisa_opportunity_are_separate_from_economic_tax() -> None:
    policy = TaxPolicy(
        policy_id="tax",
        version="v1",
        effective_from=date(2026, 1, 1),
        taxable_rate=Decimal("0.20"),
        nisa_opportunity_cost_rate=Decimal("0.10"),
    )
    engine = SimpleJapanTaxEngine(policy)
    _, taxable, _, state = account_components()
    no_withholding = taxable.model_copy(update={"withholding_mode": WithholdingMode.NO_WITHHOLDING})
    position = Position(
        symbol="A",
        account_bucket_id=taxable.bucket_id,
        shares=100,
        average_acquisition_price=Decimal("500"),
        market_price=Decimal("1000"),
    )
    taxable_estimate = engine.estimate_sale(
        account_bucket=no_withholding,
        position=position,
        sell_shares=100,
        expected_sell_price=Decimal("1000"),
        tax_state=state,
    )
    _, nisa, _, nisa_state = account_components(nisa=True)
    nisa_estimate = engine.estimate_sale(
        account_bucket=nisa,
        position=position.model_copy(update={"account_bucket_id": nisa.bucket_id}),
        sell_shares=100,
        expected_sell_price=Decimal("1000"),
        tax_state=nisa_state,
    )
    assert taxable_estimate.immediate_tax_effect == Decimal("10000.00")
    assert taxable_estimate.estimated_cash_withholding == 0
    assert nisa_estimate.immediate_tax_effect == 0
    assert nisa_estimate.estimated_cash_withholding == 0
    assert nisa_estimate.nisa_opportunity_cost == Decimal("10000.00")


def test_zero_commission_policy_must_be_explicitly_confirmed() -> None:
    with pytest.raises(ValueError, match="zero commission requires an explicit confirmed policy"):
        CostPolicy(policy_id="zero", version="v1")
