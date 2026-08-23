from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from stock_ai.decision import apply_executions
from stock_ai.domain import (
    ExecutionRecord,
    ExecutionStatus,
    Position,
    TradeSide,
    UserDecision,
    UserDecisionLine,
)
from tests.conftest import AS_OF, portfolio


def test_proposal_decision_and_execution_state_are_separate() -> None:
    state = portfolio(cash=Decimal("200000"))
    decision = UserDecision(
        decision_id="decision",
        proposal_id="proposal",
        version=1,
        saved_at=AS_OF,
        lines=(UserDecisionLine(proposal_line_id="B:bucket", selected_target_shares=100),),
    )
    unchanged = apply_executions(
        state,
        (),
        next_as_of=AS_OF + timedelta(days=1),
        next_portfolio_id="unchanged",
    )
    assert unchanged.positions == ()
    assert unchanged.cash[0].available_cash == Decimal("200000")

    execution = ExecutionRecord(
        execution_id="execution",
        decision_id=decision.decision_id,
        executed_at=AS_OF + timedelta(hours=1),
        symbol="B",
        account_bucket_id="bucket",
        status=ExecutionStatus.PARTIALLY_FILLED,
        side=TradeSide.BUY,
        ordered_shares=200,
        filled_shares=100,
        average_fill_price=Decimal("1000"),
        actual_commission=Decimal("100"),
        source="manual",
    )
    updated = apply_executions(
        state,
        (execution,),
        next_as_of=AS_OF + timedelta(days=1),
        next_portfolio_id="next",
    )
    assert updated.position_map()[("B", "bucket")].shares == 100
    assert updated.position_map()[("B", "bucket")].average_acquisition_price == Decimal("1001")
    assert updated.cash_map()["bucket"].available_cash == Decimal("99900")
    assert decision.lines[0].selected_target_shares == 100


def test_sell_execution_updates_position_cash_and_realized_gain() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=200,
        average_acquisition_price=Decimal("800"),
        market_price=Decimal("1000"),
    )
    state = portfolio((held,), cash=Decimal("0"))
    execution = ExecutionRecord(
        execution_id="sell",
        decision_id="decision",
        executed_at=AS_OF,
        symbol="A",
        account_bucket_id="bucket",
        status=ExecutionStatus.FILLED,
        side=TradeSide.SELL,
        ordered_shares=100,
        filled_shares=100,
        average_fill_price=Decimal("1000"),
        actual_commission=Decimal("100"),
        tax_withheld=Decimal("4000"),
        source="manual",
    )
    updated = apply_executions(
        state,
        (execution,),
        next_as_of=AS_OF + timedelta(days=1),
        next_portfolio_id="next",
    )
    assert updated.position_map()[("A", "bucket")].shares == 100
    assert updated.cash_map()["bucket"].available_cash == Decimal("95900")
    assert updated.tax_state_map()["bucket"].realized_gain_ytd == Decimal("19900")


def test_execution_cannot_create_cash_or_shares_from_invalid_records() -> None:
    state = portfolio(cash=Decimal("1000"))
    buy = ExecutionRecord(
        execution_id="bad-buy",
        decision_id="decision",
        executed_at=AS_OF,
        symbol="B",
        account_bucket_id="bucket",
        status=ExecutionStatus.FILLED,
        side=TradeSide.BUY,
        ordered_shares=100,
        filled_shares=100,
        average_fill_price=Decimal("1000"),
        source="manual",
    )
    with pytest.raises(ValueError, match="exceeds available cash"):
        apply_executions(
            state,
            (buy,),
            next_as_of=AS_OF + timedelta(days=1),
            next_portfolio_id="bad",
        )


def test_terminal_cancel_record_applies_its_partial_fill_once() -> None:
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=100,
        average_acquisition_price=Decimal("800"),
        market_price=Decimal("1000"),
    )
    state = portfolio((held,), cash=Decimal("0"))
    execution = ExecutionRecord(
        execution_id="cancelled-partial",
        decision_id="decision",
        executed_at=AS_OF,
        symbol="A",
        account_bucket_id="bucket",
        status=ExecutionStatus.CANCELLED,
        side=TradeSide.SELL,
        ordered_shares=100,
        filled_shares=50,
        average_fill_price=Decimal("1000"),
        source="manual",
    )
    updated = apply_executions(
        state,
        (execution,),
        next_as_of=AS_OF + timedelta(days=1),
        next_portfolio_id="next",
    )
    assert updated.position_map()[("A", "bucket")].shares == 50
    with pytest.raises(ValueError, match="already applied"):
        apply_executions(
            updated,
            (execution,),
            next_as_of=AS_OF + timedelta(days=2),
            next_portfolio_id="replayed",
        )


def test_tax_ytd_resets_at_execution_year_boundary_in_timestamp_order() -> None:
    jst = ZoneInfo("Asia/Tokyo")
    start = datetime(2026, 12, 31, 11, 30, tzinfo=jst)
    held = Position(
        symbol="A",
        account_bucket_id="bucket",
        shares=200,
        average_acquisition_price=Decimal("800"),
        market_price=Decimal("1000"),
    )
    base = portfolio((held,), cash=Decimal("0"))
    state = base.model_copy(
        update={
            "as_of": start,
            "tax_states": (base.tax_states[0].model_copy(update={"tax_year": 2026}),),
        }
    )
    executions = tuple(
        ExecutionRecord(
            execution_id=f"sell-{year}",
            decision_id="decision",
            executed_at=executed_at,
            symbol="A",
            account_bucket_id="bucket",
            status=ExecutionStatus.FILLED,
            side=TradeSide.SELL,
            ordered_shares=100,
            filled_shares=100,
            average_fill_price=Decimal("1000"),
            actual_commission=Decimal("100"),
            source="manual",
        )
        for year, executed_at in (
            (2026, datetime(2026, 12, 31, 12, 30, tzinfo=jst)),
            (2027, datetime(2027, 1, 1, 12, 30, tzinfo=jst)),
        )
    )
    updated = apply_executions(
        state,
        executions,
        next_as_of=datetime(2027, 1, 2, 11, 30, tzinfo=jst),
        next_portfolio_id="2027-state",
    )
    tax_state = updated.tax_state_map()["bucket"]
    assert tax_state.tax_year == 2027
    assert tax_state.realized_gain_ytd == Decimal("19900")
