from __future__ import annotations

from datetime import date
from decimal import Decimal

from stock_ai.decision import CostPolicy, SimpleJapanTaxEngine, TaxPolicy, TransactionCostEngine
from stock_ai.domain import AccountType, Position, TaxState
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
        CostPolicy(policy_id="policy", version="v1", impact_bps_at_full_adv=Decimal("25"))
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
    state = TaxState(account_bucket_id=bucket.bucket_id, realized_gain_ytd=Decimal("10000"))
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
