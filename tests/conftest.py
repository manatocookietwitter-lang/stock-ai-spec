from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from stock_ai.decision import (
    CostPolicy,
    DailyPortfolioDecisionEngine,
    DecisionCandidate,
    DecisionEngineConfig,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
)
from stock_ai.domain import (
    Account,
    AccountBucket,
    AccountType,
    CashState,
    PortfolioProposal,
    PortfolioState,
    Position,
    Prediction,
    PredictionUncertainty,
    Security,
    TaxState,
    WithholdingMode,
)
from stock_ai.operations.models import DecisionPolicySnapshot

JST = ZoneInfo("Asia/Tokyo")
AS_OF = datetime(2026, 8, 24, 11, 30, tzinfo=JST)


def prediction(symbol: str, expected: float) -> Prediction:
    return Prediction(
        symbol=symbol,
        as_of=AS_OF,
        expected_return_1d=expected,
        expected_return_5d=expected,
        expected_return_20d=expected,
        downside_quantile=0.0,
        large_loss_probability=0.0,
        uncertainty=PredictionUncertainty(standard_error=0.0),
        model_version="test-model-v1",
        feature_version="test-features-v1",
        data_snapshot_id="test-snapshot",
    )


def account_components(
    *, cash: Decimal = Decimal("200000"), nisa: bool = False
) -> tuple[Account, AccountBucket, CashState, TaxState]:
    account = Account(account_id="account", broker="fixture", display_name="Fixture")
    bucket = AccountBucket(
        bucket_id="bucket",
        account_id=account.account_id,
        account_type=AccountType.NISA if nisa else AccountType.TAXABLE_SPECIFIED,
        withholding_mode=(WithholdingMode.NOT_APPLICABLE if nisa else WithholdingMode.WITHHOLDING),
        fee_policy_id="cost-v1",
        tax_policy_id="tax-v1",
    )
    return (
        account,
        bucket,
        CashState(account_bucket_id=bucket.bucket_id, available_cash=cash),
        TaxState(account_bucket_id=bucket.bucket_id, tax_year=AS_OF.year),
    )


def portfolio(
    positions: tuple[Position, ...] = (), *, cash: Decimal = Decimal("200000")
) -> PortfolioState:
    account, bucket, cash_state, tax_state = account_components(cash=cash)
    return PortfolioState(
        portfolio_id="portfolio",
        as_of=AS_OF,
        accounts=(account,),
        account_buckets=(bucket,),
        positions=positions,
        cash=(cash_state,),
        tax_states=(tax_state,),
    )


def candidate(
    symbol: str,
    expected: float,
    *,
    price: Decimal = Decimal("1000"),
    sector: str = "Sector",
    bucket: str = "bucket",
) -> DecisionCandidate:
    return DecisionCandidate(
        security=Security(symbol=symbol, company_name=f"Company {symbol}", sector=sector),
        account_bucket_id=bucket,
        price=price,
        average_daily_trading_value=Decimal("100000000"),
        prediction=prediction(symbol, expected),
    )


def decision_engine(
    *,
    cost_policy: CostPolicy | None = None,
    tax_rate: Decimal = Decimal("0"),
    threshold: Decimal = Decimal("0"),
) -> DailyPortfolioDecisionEngine:
    return DailyPortfolioDecisionEngine(
        config=DecisionEngineConfig(
            maximum_symbol_weight=Decimal("1"),
            maximum_sector_weight=Decimal("1"),
            minimum_cash_ratio=Decimal("0"),
            maximum_turnover_ratio=Decimal("2"),
            minimum_improvement_yen=threshold,
            downside_penalty_weight=Decimal("0"),
            uncertainty_penalty_weight=Decimal("0"),
        ),
        cost_engine=TransactionCostEngine(
            cost_policy
            or CostPolicy(
                policy_id="cost-v1",
                version="cost-v1",
                zero_commission_confirmed=True,
                full_spread_bps=Decimal("0"),
                slippage_bps=Decimal("0"),
                impact_bps_at_full_adv=Decimal("0"),
            )
        ),
        tax_engine=SimpleJapanTaxEngine(
            TaxPolicy(
                policy_id="tax-v1",
                version="tax-v1",
                effective_from=date(2026, 1, 1),
                taxable_rate=tax_rate,
            )
        ),
    )


def decision_policy_snapshot(
    proposal: PortfolioProposal,
    config: DecisionEngineConfig,
) -> DecisionPolicySnapshot:
    return DecisionPolicySnapshot(
        proposal_id=proposal.proposal_id,
        captured_at=proposal.generated_at,
        config=config,
    )


@pytest.fixture
def held_position() -> Position:
    return Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=100,
        average_acquisition_price=Decimal("900"),
        market_price=Decimal("1000"),
    )
