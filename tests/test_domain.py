from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from stock_ai.domain import (
    Account,
    AccountBucket,
    AccountType,
    CashState,
    MarketSnapshot,
    PortfolioProposal,
    PortfolioState,
    Position,
    TaxState,
    WithholdingMode,
)


def test_same_symbol_can_exist_in_multiple_account_buckets() -> None:
    as_of = datetime(2026, 8, 24, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    account = Account(account_id="a", broker="SBI", display_name="Main")
    buckets = (
        AccountBucket(
            bucket_id="nisa",
            account_id="a",
            account_type=AccountType.NISA,
            withholding_mode=WithholdingMode.NOT_APPLICABLE,
            fee_policy_id="f",
            tax_policy_id="t",
        ),
        AccountBucket(
            bucket_id="taxable",
            account_id="a",
            account_type=AccountType.TAXABLE_SPECIFIED,
            withholding_mode=WithholdingMode.WITHHOLDING,
            fee_policy_id="f",
            tax_policy_id="t",
        ),
    )
    state = PortfolioState(
        portfolio_id="p",
        as_of=as_of,
        accounts=(account,),
        account_buckets=buckets,
        positions=(
            Position(
                symbol="7203",
                account_bucket_id="nisa",
                shares=100,
                average_acquisition_price=Decimal("2000"),
                market_price=Decimal("2500"),
            ),
            Position(
                symbol="7203",
                account_bucket_id="taxable",
                shares=200,
                average_acquisition_price=Decimal("2200"),
                market_price=Decimal("2500"),
            ),
        ),
        cash=(
            CashState(account_bucket_id="nisa", available_cash=Decimal("100000")),
            CashState(account_bucket_id="taxable", available_cash=Decimal("200000")),
        ),
        tax_states=(
            TaxState(account_bucket_id="nisa", tax_year=2026),
            TaxState(account_bucket_id="taxable", tax_year=2026),
        ),
    )
    assert set(state.position_map()) == {("7203", "nisa"), ("7203", "taxable")}


def test_duplicate_symbol_bucket_is_rejected() -> None:
    as_of = datetime(2026, 8, 24, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    account = Account(account_id="a", broker="SBI", display_name="Main")
    bucket = AccountBucket(
        bucket_id="taxable",
        account_id="a",
        account_type=AccountType.TAXABLE_SPECIFIED,
        withholding_mode=WithholdingMode.WITHHOLDING,
        fee_policy_id="f",
        tax_policy_id="t",
    )
    position = Position(
        symbol="7203",
        account_bucket_id="taxable",
        shares=100,
        average_acquisition_price=Decimal("2200"),
        market_price=Decimal("2500"),
    )
    with pytest.raises(ValidationError, match="position keys must be unique"):
        PortfolioState(
            portfolio_id="p",
            as_of=as_of,
            accounts=(account,),
            account_buckets=(bucket,),
            positions=(position, position),
            cash=(CashState(account_bucket_id="taxable", available_cash=Decimal("1")),),
            tax_states=(TaxState(account_bucket_id="taxable", tax_year=2026),),
        )


def test_portfolio_proposal_cannot_be_order_instruction() -> None:
    as_of = datetime(2026, 8, 24, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    with pytest.raises(ValidationError, match="never be order instructions"):
        PortfolioProposal(
            proposal_id="p",
            as_of=as_of,
            generated_at=as_of,
            current_portfolio_id="portfolio",
            targets=(),
            lines=(),
            hold_utility=Decimal("0"),
            proposed_utility=Decimal("0"),
            net_improvement=Decimal("0"),
            estimated_cash_after={},
            model_bundle_version="m",
            decision_engine_version="e",
            cost_policy_id="c",
            cost_policy_version="c-v1",
            tax_policy_id="t",
            tax_policy_version="t-v1",
            is_order_instruction=True,
        )


def test_market_snapshot_nested_price_mapping_is_immutable() -> None:
    as_of = datetime(2026, 8, 24, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    snapshot = MarketSnapshot(
        snapshot_id="snapshot",
        as_of=as_of,
        available_at=as_of,
        prices={"7203": Decimal("2500")},
        source="fixture",
    )
    with pytest.raises(TypeError):
        snapshot.prices["7203"] = Decimal("1")  # type: ignore[index]
