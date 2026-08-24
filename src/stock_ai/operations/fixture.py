"""Explicit deterministic Goal 5 fixture bootstrap; never a production fallback."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal

from stock_ai.decision import (
    CostPolicy,
    DailyPortfolioDecisionEngine,
    DecisionCandidate,
    DecisionEngineConfig,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
)
from stock_ai.domain import Prediction, PredictionUncertainty, Security
from stock_ai.fixtures import portfolio_fixture
from stock_ai.operations.models import (
    DailyOperationStatus,
    DecisionPolicySnapshot,
    PipelineState,
    RankingRecord,
)
from stock_ai.operations.store import OperationalStore


def bootstrap_goal5_fixture(store: OperationalStore, *, as_of: datetime) -> str:
    """Populate a local ledger with one clearly labelled fixture proposal."""
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("fixture as_of must be timezone-aware")
    prices = {
        "7203": Decimal("2870"),
        "9432": Decimal("168"),
        "6758": Decimal("14250"),
        "8306": Decimal("2080"),
    }
    portfolio = portfolio_fixture(as_of, prices)
    store.append_portfolio(portfolio, created_at=as_of)
    predictions = {
        "7203": -0.025,
        "9432": 0.004,
        "6758": 0.045,
        "8306": 0.032,
    }
    securities = {
        "7203": ("トヨタ自動車", "輸送用機器"),
        "9432": ("NTT", "情報・通信業"),
        "6758": ("ソニーグループ", "電気機器"),
        "8306": ("三菱UFJ", "銀行業"),
    }
    buckets = {
        "7203": ("sbi-taxable", "sbi-nisa"),
        "9432": ("sbi-taxable",),
        "6758": ("sbi-nisa",),
        "8306": ("sbi-taxable",),
    }
    candidates: list[DecisionCandidate] = []
    for symbol, expected in predictions.items():
        for bucket in buckets[symbol]:
            name, sector = securities[symbol]
            prediction = Prediction(
                symbol=symbol,
                as_of=as_of,
                expected_return_1d=expected / 3,
                expected_return_5d=expected,
                expected_return_20d=expected * 1.5,
                downside_quantile=min(-0.01, expected - 0.03),
                large_loss_probability=0.08 if expected >= 0 else 0.25,
                uncertainty=PredictionUncertainty(standard_error=0.015),
                model_version="fixture-goal5-ensemble-v1",
                feature_version="fixture-v2-plus-morning",
                data_snapshot_id="fixture-goal5-snapshot",
            )
            candidates.append(
                DecisionCandidate(
                    security=Security(symbol=symbol, company_name=name, sector=sector),
                    account_bucket_id=bucket,
                    price=prices[symbol],
                    average_daily_trading_value=Decimal("5000000000"),
                    prediction=prediction,
                )
            )
    cost_policy = CostPolicy(
        policy_id="fixture-cost-v1",
        version="fixture-cost-v1",
        zero_commission_confirmed=True,
        full_spread_bps=Decimal("8"),
        slippage_bps=Decimal("4"),
        impact_bps_at_full_adv=Decimal("10"),
    )
    tax_policy = TaxPolicy(
        policy_id="fixture-tax-v1",
        version="fixture-tax-v1",
        effective_from=date(as_of.year, 1, 1),
    )
    decision_config = DecisionEngineConfig(
        maximum_symbol_weight=Decimal("0.50"),
        maximum_sector_weight=Decimal("0.70"),
        minimum_cash_ratio=Decimal("0.10"),
        maximum_turnover_ratio=Decimal("0.80"),
        minimum_improvement_yen=Decimal("300"),
        uncertainty_penalty_weight=Decimal("0.30"),
    )
    engine = DailyPortfolioDecisionEngine(
        config=decision_config,
        cost_engine=TransactionCostEngine(cost_policy),
        tax_engine=SimpleJapanTaxEngine(tax_policy),
    )
    proposal = engine.propose(
        portfolio=portfolio,
        candidates=tuple(candidates),
        generated_at=as_of + timedelta(minutes=6),
        model_bundle_version="fixture-goal5-ensemble-v1",
    )
    store.archive_proposal(
        proposal,
        archived_at=proposal.generated_at,
        decision_policy=DecisionPolicySnapshot(
            proposal_id=proposal.proposal_id,
            captured_at=proposal.generated_at,
            config=decision_config,
        ),
    )
    store.set_metadata("runtime_mode", "DETERMINISTIC_FIXTURE_ONLY")
    store.set_daily_status(
        DailyOperationStatus(
            business_date=as_of.date(),
            pipeline_state=PipelineState.PROPOSAL_READY,
            updated_at=proposal.generated_at,
            data_as_of=as_of.replace(hour=1, minute=30),
            morning_data_as_of=as_of,
            proposal_id=proposal.proposal_id,
        )
    )
    rankings = tuple(
        RankingRecord(
            ranking_id=f"fixture-overall-{as_of.date()}-{symbol}",
            as_of=as_of,
            symbol=symbol,
            company_name=securities[symbol][0],
            rank_type="overall",
            rank=rank,
            total_universe=len(predictions),
            candidate_status="候補" if rank <= 3 else "監視",
            portfolio_action=next(
                (
                    line.action.value
                    for line in proposal.lines
                    if line.symbol == symbol and line.share_difference != 0
                ),
                next(
                    (line.action.value for line in proposal.lines if line.symbol == symbol),
                    None,
                ),
            ),
            model_bundle_version=proposal.model_bundle_version,
            data_snapshot_id="fixture-goal5-snapshot",
        )
        for rank, symbol in enumerate(
            sorted(predictions, key=lambda item: predictions[item], reverse=True), start=1
        )
    )
    store.append_rankings(rankings)
    return proposal.proposal_id
