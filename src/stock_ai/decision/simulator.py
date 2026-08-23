"""Apply manually recorded executions to produce the next operational state."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from stock_ai.domain import (
    CashState,
    ExecutionRecord,
    PortfolioState,
    Position,
    TaxState,
    TradeSide,
)


def _start_tax_year(state: TaxState, year: int) -> TaxState:
    """Reset year-to-date fields while preserving explicit user inputs."""
    return TaxState(
        account_bucket_id=state.account_bucket_id,
        tax_year=year,
        realized_gain_ytd=Decimal("0"),
        realized_loss_ytd=Decimal("0"),
        loss_carryforward_user_input=state.loss_carryforward_user_input,
        nisa_annual_capacity_user_input=state.nisa_annual_capacity_user_input,
        nisa_lifetime_capacity_user_input=state.nisa_lifetime_capacity_user_input,
    )


def apply_executions(
    portfolio: PortfolioState,
    executions: tuple[ExecutionRecord, ...],
    *,
    next_as_of: datetime,
    next_portfolio_id: str,
) -> PortfolioState:
    """Carry actual fills forward; proposals and decisions never change holdings."""
    if next_as_of.tzinfo is None or next_as_of.utcoffset() is None:
        raise ValueError("next_as_of must be timezone-aware")
    if next_as_of <= portfolio.as_of:
        raise ValueError("next_as_of must be later than the current portfolio as_of")
    execution_ids = [execution.execution_id for execution in executions]
    if len(execution_ids) != len(set(execution_ids)):
        raise ValueError("execution IDs must be unique")
    replayed = set(execution_ids) & set(portfolio.applied_execution_ids)
    if replayed:
        raise ValueError(f"execution IDs were already applied: {sorted(replayed)}")
    if any(
        execution.executed_at < portfolio.as_of or execution.executed_at > next_as_of
        for execution in executions
    ):
        raise ValueError("execution timestamps must fall within the state transition window")
    positions = portfolio.position_map()
    cash = {item.account_bucket_id: item for item in portfolio.cash}
    tax_states = portfolio.tax_state_map()
    applied_execution_ids = list(portfolio.applied_execution_ids)
    for execution in sorted(executions, key=lambda item: (item.executed_at, item.execution_id)):
        if execution.filled_shares == 0:
            continue
        if execution.source not in {"manual", "csv_import", "statement_import"}:
            raise ValueError("execution source must be manual or an approved import path")
        key = (execution.symbol, execution.account_bucket_id)
        existing_cash = cash[execution.account_bucket_id]
        price = execution.average_fill_price
        if price is None:
            raise ValueError("filled execution requires a fill price")
        notional = price * execution.filled_shares
        expenses = execution.actual_commission + execution.actual_other_cost
        position = positions.get(key)
        tax_state = tax_states[execution.account_bucket_id]
        if execution.executed_at.year != tax_state.tax_year:
            tax_state = _start_tax_year(tax_state, execution.executed_at.year)
            tax_states[execution.account_bucket_id] = tax_state
        if execution.side is TradeSide.BUY:
            total_outflow = notional + expenses + execution.tax_withheld
            if total_outflow > existing_cash.available_cash:
                raise ValueError("recorded buy execution exceeds available cash")
            acquisition_outflow = notional + expenses
            if position is None:
                positions[key] = Position(
                    symbol=execution.symbol,
                    account_bucket_id=execution.account_bucket_id,
                    shares=execution.filled_shares,
                    average_acquisition_price=acquisition_outflow / execution.filled_shares,
                    market_price=price,
                )
            else:
                new_shares = position.shares + execution.filled_shares
                average_price = (
                    position.average_acquisition_price * position.shares + acquisition_outflow
                ) / new_shares
                positions[key] = Position(
                    symbol=position.symbol,
                    account_bucket_id=position.account_bucket_id,
                    shares=new_shares,
                    average_acquisition_price=average_price,
                    market_price=price,
                )
            cash[execution.account_bucket_id] = CashState(
                account_bucket_id=existing_cash.account_bucket_id,
                available_cash=existing_cash.available_cash - total_outflow,
                reserved_cash=max(Decimal("0"), existing_cash.reserved_cash - total_outflow),
            )
        else:
            if position is None or execution.filled_shares > position.shares:
                raise ValueError("recorded sell execution exceeds held shares")
            realized_pnl = (
                price - position.average_acquisition_price
            ) * execution.filled_shares - expenses
            remaining = position.shares - execution.filled_shares
            if remaining:
                positions[key] = Position(
                    symbol=position.symbol,
                    account_bucket_id=position.account_bucket_id,
                    shares=remaining,
                    average_acquisition_price=position.average_acquisition_price,
                    market_price=price,
                )
            else:
                del positions[key]
            inflow = notional - expenses - execution.tax_withheld
            cash[execution.account_bucket_id] = CashState(
                account_bucket_id=existing_cash.account_bucket_id,
                available_cash=existing_cash.available_cash + inflow,
                reserved_cash=existing_cash.reserved_cash,
            )
            if realized_pnl >= 0:
                tax_states[execution.account_bucket_id] = TaxState(
                    **{
                        **tax_state.model_dump(),
                        "realized_gain_ytd": tax_state.realized_gain_ytd + realized_pnl,
                    }
                )
            else:
                tax_states[execution.account_bucket_id] = TaxState(
                    **{
                        **tax_state.model_dump(),
                        "realized_loss_ytd": tax_state.realized_loss_ytd - realized_pnl,
                    }
                )
        applied_execution_ids.append(execution.execution_id)

    # A state dated in the new year must not carry the prior year's YTD totals,
    # even when the transition contained only prior-year executions.
    tax_states = {
        bucket_id: (
            state if state.tax_year == next_as_of.year else _start_tax_year(state, next_as_of.year)
        )
        for bucket_id, state in tax_states.items()
    }

    return PortfolioState(
        portfolio_id=next_portfolio_id,
        as_of=next_as_of,
        accounts=portfolio.accounts,
        account_buckets=portfolio.account_buckets,
        positions=tuple(sorted(positions.values(), key=lambda item: item.key)),
        cash=tuple(sorted(cash.values(), key=lambda item: item.account_bucket_id)),
        tax_states=tuple(sorted(tax_states.values(), key=lambda item: item.account_bucket_id)),
        applied_execution_ids=tuple(applied_execution_ids),
    )
