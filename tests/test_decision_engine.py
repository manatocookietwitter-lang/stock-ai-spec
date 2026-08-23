from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from stock_ai.decision import (
    CostPolicy,
    DailyPortfolioDecisionEngine,
    DecisionEngineConfig,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
    classify_action,
)
from stock_ai.domain import (
    Account,
    AccountBucket,
    AccountType,
    CashState,
    PortfolioState,
    Position,
    ProposalAction,
    TaxState,
    WithholdingMode,
)
from tests.conftest import AS_OF, candidate, decision_engine, portfolio


def test_buy_hold_reduce_sell_skip_classification() -> None:
    assert classify_action(0, 100) is ProposalAction.BUY
    assert classify_action(100, 100) is ProposalAction.HOLD
    assert classify_action(300, 100) is ProposalAction.REDUCE
    assert classify_action(100, 0) is ProposalAction.SELL
    assert classify_action(0, 0) is ProposalAction.SKIP


def test_sell_and_reduce_are_distinguished_at_zero() -> None:
    assert classify_action(100, 0) is ProposalAction.SELL
    assert classify_action(200, 100) is ProposalAction.REDUCE


def test_insufficient_cash_results_in_skip() -> None:
    state = portfolio(cash=Decimal("99999"))
    proposal = decision_engine().propose(
        portfolio=state,
        candidates=(candidate("B", 0.20),),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    assert proposal.lines[0].action is ProposalAction.SKIP
    assert proposal.lines[0].recommended_shares == 0


def test_target_is_rounded_to_100_share_lots_without_exceeding_cash() -> None:
    state = portfolio(cash=Decimal("250000"))
    proposal = decision_engine().propose(
        portfolio=state,
        candidates=(candidate("B", 0.20),),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    line = proposal.lines[0]
    assert line.action is ProposalAction.BUY
    assert line.recommended_shares == 200
    assert line.recommended_shares % 100 == 0
    assert proposal.estimated_cash_after["bucket"] == Decimal("50000")


def test_negative_holding_can_be_sold_to_all_cash() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=100,
        average_acquisition_price=Decimal("900"),
        market_price=Decimal("1000"),
    )
    proposal = decision_engine().propose(
        portfolio=portfolio((held,), cash=Decimal("0")),
        candidates=(candidate("A", -0.10),),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    assert proposal.lines[0].action is ProposalAction.SELL
    assert proposal.estimated_cash_after["bucket"] == Decimal("100000")


def test_tax_offset_curve_can_produce_reduce_not_sell() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=300,
        average_acquisition_price=Decimal("900"),
        market_price=Decimal("1000"),
    )
    state = portfolio((held,), cash=Decimal("0"))
    state = state.model_copy(
        update={
            "tax_states": (
                state.tax_states[0].model_copy(
                    update={"loss_carryforward_user_input": Decimal("10000")}
                ),
            )
        }
    )
    engine = decision_engine(tax_rate=Decimal("0.20"))
    proposal = engine.propose(
        portfolio=state,
        candidates=(candidate("A", -0.01),),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    assert proposal.lines[0].action is ProposalAction.REDUCE
    assert proposal.lines[0].recommended_shares == 200


def test_higher_ranked_replacement_is_rejected_after_cost_and_tax() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=100,
        average_acquisition_price=Decimal("500"),
        market_price=Decimal("1000"),
    )
    costly = CostPolicy(
        policy_id="cost-v1",
        version="cost-v1",
        commission_fixed=Decimal("500"),
        full_spread_bps=Decimal("20"),
        slippage_bps=Decimal("10"),
        impact_bps_at_full_adv=Decimal("0"),
    )
    engine = decision_engine(cost_policy=costly, tax_rate=Decimal("0.20"))
    proposal = engine.propose(
        portfolio=portfolio((held,), cash=Decimal("0")),
        candidates=(candidate("A", 0.01), candidate("B", 0.08)),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    by_symbol = {line.symbol: line for line in proposal.lines}
    assert by_symbol["A"].action is ProposalAction.HOLD
    assert by_symbol["B"].action is ProposalAction.SKIP
    assert proposal.no_trade_reason is not None


def test_lower_ranked_current_holding_remains_hold_when_gain_is_too_small() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=100,
        average_acquisition_price=Decimal("1000"),
        market_price=Decimal("1000"),
    )
    proposal = decision_engine(threshold=Decimal("500")).propose(
        portfolio=portfolio((held,), cash=Decimal("0")),
        candidates=(candidate("A", 0.010), candidate("B", 0.011)),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    by_symbol = {line.symbol: line for line in proposal.lines}
    assert by_symbol["A"].action is ProposalAction.HOLD
    assert by_symbol["B"].action is ProposalAction.SKIP


def test_same_symbol_in_taxable_and_nisa_gets_separate_lines() -> None:
    account = Account(account_id="account", broker="fixture", display_name="Fixture")
    taxable = AccountBucket(
        bucket_id="taxable",
        account_id="account",
        account_type=AccountType.TAXABLE_SPECIFIED,
        withholding_mode=WithholdingMode.WITHHOLDING,
        fee_policy_id="cost-v1",
        tax_policy_id="tax-v1",
    )
    nisa = AccountBucket(
        bucket_id="nisa",
        account_id="account",
        account_type=AccountType.NISA,
        withholding_mode=WithholdingMode.NOT_APPLICABLE,
        fee_policy_id="cost-v1",
        tax_policy_id="tax-v1",
    )
    positions = tuple(
        Position(
            symbol="7203",
            account_bucket_id=bucket,
            shares=100,
            average_acquisition_price=Decimal("1000"),
            market_price=Decimal("1000"),
        )
        for bucket in ("taxable", "nisa")
    )
    state = PortfolioState(
        portfolio_id="p",
        as_of=AS_OF,
        accounts=(account,),
        account_buckets=(taxable, nisa),
        positions=positions,
        cash=(
            CashState(account_bucket_id="taxable", available_cash=Decimal("0")),
            CashState(account_bucket_id="nisa", available_cash=Decimal("0")),
        ),
        tax_states=(
            TaxState(account_bucket_id="taxable", tax_year=2026),
            TaxState(account_bucket_id="nisa", tax_year=2026),
        ),
    )
    cost = TransactionCostEngine(
        CostPolicy(
            policy_id="cost-v1",
            version="cost-v1",
            zero_commission_confirmed=True,
            full_spread_bps=Decimal("0"),
            slippage_bps=Decimal("0"),
            impact_bps_at_full_adv=Decimal("0"),
        )
    )
    tax = SimpleJapanTaxEngine(
        TaxPolicy(
            policy_id="tax-v1",
            version="tax-v1",
            effective_from=date(2026, 1, 1),
            taxable_rate=Decimal("0"),
        )
    )
    engine = DailyPortfolioDecisionEngine(
        config=DecisionEngineConfig(
            maximum_symbol_weight=Decimal("1"),
            maximum_sector_weight=Decimal("1"),
            minimum_cash_ratio=Decimal("0"),
            downside_penalty_weight=Decimal("0"),
            uncertainty_penalty_weight=Decimal("0"),
            minimum_improvement_yen=Decimal("0"),
        ),
        cost_engine=cost,
        tax_engine=tax,
    )
    proposal = engine.propose(
        portfolio=state,
        candidates=(
            candidate("7203", 0.0, bucket="taxable"),
            candidate("7203", 0.0, bucket="nisa"),
        ),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    assert {(line.symbol, line.account_bucket_id) for line in proposal.lines} == {
        ("7203", "taxable"),
        ("7203", "nisa"),
    }
    assert all(line.action is ProposalAction.HOLD for line in proposal.lines)


def test_existing_odd_lot_only_allows_round_lot_trade_deltas() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=150,
        average_acquisition_price=Decimal("1000"),
        market_price=Decimal("1000"),
    )
    proposal = decision_engine().propose(
        portfolio=portfolio((held,), cash=Decimal("0")),
        candidates=(candidate("A", -0.10),),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    line = proposal.lines[0]
    assert abs(line.share_difference) % 100 == 0
    assert line.recommended_shares == 50
    assert line.action is ProposalAction.REDUCE


def test_candidate_order_does_not_change_proposal_or_identity() -> None:
    state = portfolio(cash=Decimal("200000"))
    candidates = (candidate("B", 0.10), candidate("A", 0.10))
    engine = decision_engine()
    first = engine.propose(
        portfolio=state,
        candidates=candidates,
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    second = engine.propose(
        portfolio=state,
        candidates=tuple(reversed(candidates)),
        generated_at=AS_OF,
        model_bundle_version="test-model-v1",
    )
    assert first.targets == second.targets
    assert first.proposal_id == second.proposal_id


def test_mixed_prediction_provenance_is_rejected() -> None:
    first = candidate("A", 0.10)
    second = candidate("B", 0.10)
    second = second.model_copy(
        update={
            "prediction": second.prediction.model_copy(
                update={"data_snapshot_id": "different-snapshot"}
            )
        }
    )
    with pytest.raises(ValueError, match="coherent prediction bundle"):
        decision_engine().propose(
            portfolio=portfolio(cash=Decimal("200000")),
            candidates=(first, second),
            generated_at=AS_OF,
            model_bundle_version="test-model-v1",
        )


def test_all_cash_empty_universe_can_propose_no_trades() -> None:
    proposal = decision_engine().propose(
        portfolio=portfolio(cash=Decimal("200000")),
        candidates=(),
        generated_at=AS_OF,
        model_bundle_version="empty-universe-v1",
    )
    assert proposal.lines == ()
    assert proposal.targets == ()
    assert proposal.net_improvement == 0
