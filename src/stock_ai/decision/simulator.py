"""Apply manually recorded executions to produce the next operational state."""

from __future__ import annotations

from datetime import datetime

from stock_ai.domain import (
    ExecutionRecord,
    ExecutionStatus,
    PortfolioState,
    Position,
    TradeSide,
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
    positions = portfolio.position_map()
    cash = {item.account_bucket_id: item for item in portfolio.cash}
    tax_states = portfolio.tax_state_map()
    allowed_statuses = {ExecutionStatus.PARTIALLY_FILLED, ExecutionStatus.FILLED}
    for execution in sorted(executions, key=lambda item: (item.executed_at, item.execution_id)):
        if execution.status not in allowed_statuses or execution.filled_shares == 0:
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
        if execution.side is TradeSide.BUY:
            total_outflow = notional + expenses + execution.tax_withheld
            if total_outflow > existing_cash.deployable_cash:
                raise ValueError("recorded buy execution exceeds deployable cash")
            if position is None:
                positions[key] = Position(
                    symbol=execution.symbol,
                    account_bucket_id=execution.account_bucket_id,
                    shares=execution.filled_shares,
                    average_acquisition_price=price,
                    market_price=price,
                )
            else:
                new_shares = position.shares + execution.filled_shares
                average_price = (
                    position.average_acquisition_price * position.shares + notional
                ) / new_shares
                positions[key] = position.model_copy(
                    update={
                        "shares": new_shares,
                        "average_acquisition_price": average_price,
                        "market_price": price,
                    }
                )
            cash[execution.account_bucket_id] = existing_cash.model_copy(
                update={"available_cash": existing_cash.available_cash - total_outflow}
            )
        else:
            if position is None or execution.filled_shares > position.shares:
                raise ValueError("recorded sell execution exceeds held shares")
            realized_pnl = (price - position.average_acquisition_price) * execution.filled_shares
            remaining = position.shares - execution.filled_shares
            if remaining:
                positions[key] = position.model_copy(
                    update={"shares": remaining, "market_price": price}
                )
            else:
                del positions[key]
            inflow = notional - expenses - execution.tax_withheld
            cash[execution.account_bucket_id] = existing_cash.model_copy(
                update={"available_cash": existing_cash.available_cash + inflow}
            )
            if realized_pnl >= 0:
                tax_states[execution.account_bucket_id] = tax_state.model_copy(
                    update={"realized_gain_ytd": tax_state.realized_gain_ytd + realized_pnl}
                )
            else:
                tax_states[execution.account_bucket_id] = tax_state.model_copy(
                    update={"realized_loss_ytd": tax_state.realized_loss_ytd - realized_pnl}
                )

    return PortfolioState(
        portfolio_id=next_portfolio_id,
        as_of=next_as_of,
        accounts=portfolio.accounts,
        account_buckets=portfolio.account_buckets,
        positions=tuple(sorted(positions.values(), key=lambda item: item.key)),
        cash=tuple(sorted(cash.values(), key=lambda item: item.account_bucket_id)),
        tax_states=tuple(sorted(tax_states.values(), key=lambda item: item.account_bucket_id)),
    )
