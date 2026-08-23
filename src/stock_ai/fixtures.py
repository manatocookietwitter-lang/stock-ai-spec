"""Deterministic verification fixtures.

These records are available only through the explicitly named fixture demo and
tests.  They are never selected as a fallback for absent production data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_ai.domain import (
    Account,
    AccountBucket,
    AccountType,
    CashState,
    PortfolioState,
    Position,
    TaxState,
    WithholdingMode,
)

JST = ZoneInfo("Asia/Tokyo")


def market_fixture(periods: int = 340) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2024-01-04", periods=periods)
    definitions = (
        ("7203", "Transport Equipment", 2400.0, 0.85, 26.0, 1_500_000.0),
        ("6758", "Electric Appliances", 11800.0, 2.4, 105.0, 500_000.0),
        ("8306", "Banks", 1180.0, -0.10, 18.0, 2_200_000.0),
        ("9432", "Information & Communication", 175.0, 0.025, 2.8, 7_000_000.0),
    )
    rows: list[dict[str, object]] = []
    for symbol_index, (symbol, sector, base, trend, wave, shares) in enumerate(definitions):
        for index, trading_date in enumerate(dates):
            seasonal = wave * np.sin((index + symbol_index * 3) / 13)
            close = base + trend * index + seasonal
            open_price = close * (1 + 0.0015 * np.sin(index / 5 + symbol_index))
            high = max(open_price, close) * 1.007
            low = min(open_price, close) * 0.993
            volume = 650_000 + symbol_index * 75_000 + (index % 17) * 11_000
            available_at = trading_date.to_pydatetime().replace(tzinfo=JST) + timedelta(
                days=1, hours=8
            )
            rows.append(
                {
                    "symbol": symbol,
                    "sector": sector,
                    "trading_date": trading_date,
                    "available_at": available_at,
                    "adjusted_open": open_price,
                    "adjusted_high": high,
                    "adjusted_low": low,
                    "adjusted_close": close,
                    "adjusted_volume": float(volume),
                    "trading_value": close * volume,
                    "shares_outstanding": shares * 1_000,
                }
            )
    daily = pd.DataFrame(rows)

    market_rows: list[dict[str, object]] = []
    for index, trading_date in enumerate(dates):
        topix_close = 2250 + 0.65 * index + 17 * np.sin(index / 19)
        market_rows.append(
            {
                "trading_date": trading_date,
                "available_at": trading_date.to_pydatetime().replace(tzinfo=JST)
                + timedelta(days=1, hours=8),
                "topix_close": topix_close,
            }
        )
    market = pd.DataFrame(market_rows)

    financial_rows: list[dict[str, object]] = []
    for index, (symbol, _, _, _, _, _) in enumerate(definitions):
        offsets = tuple(dict.fromkeys((min(20, periods - 1), min(180, periods - 1))))
        for revision, offset in enumerate(offsets):
            disclosure_date = dates[offset]
            financial_rows.append(
                {
                    "symbol": symbol,
                    "available_at": disclosure_date.to_pydatetime().replace(tzinfo=JST)
                    + timedelta(hours=17),
                    "per": 10.0 + index * 2 + revision * 0.4,
                    "pbr": 0.9 + index * 0.25,
                    "roe": 0.08 + index * 0.015,
                    "operating_margin": 0.06 + index * 0.012,
                    "revenue_growth_yoy": 0.03 + revision * 0.01 + index * 0.004,
                    "operating_profit_growth_yoy": 0.04 + revision * 0.015 + index * 0.003,
                    "forecast_revision": -0.01 + index * 0.01 + revision * 0.012,
                }
            )
    return daily, market, pd.DataFrame(financial_rows)


def portfolio_fixture(as_of: datetime, latest_prices: dict[str, Decimal]) -> PortfolioState:
    account = Account(account_id="acct-sbi", broker="SBI", display_name="Fixture SBI")
    taxable = AccountBucket(
        bucket_id="sbi-taxable",
        account_id=account.account_id,
        account_type=AccountType.TAXABLE_SPECIFIED,
        withholding_mode=WithholdingMode.WITHHOLDING,
        fee_policy_id="fixture-cost-v1",
        tax_policy_id="fixture-tax-v1",
    )
    nisa = AccountBucket(
        bucket_id="sbi-nisa",
        account_id=account.account_id,
        account_type=AccountType.NISA,
        withholding_mode=WithholdingMode.NOT_APPLICABLE,
        fee_policy_id="fixture-cost-v1",
        tax_policy_id="fixture-tax-v1",
    )
    return PortfolioState(
        portfolio_id="fixture-current",
        as_of=as_of,
        accounts=(account,),
        account_buckets=(taxable, nisa),
        positions=(
            Position(
                symbol="7203",
                account_bucket_id=taxable.bucket_id,
                shares=200,
                average_acquisition_price=Decimal("2450"),
                market_price=latest_prices["7203"],
            ),
            Position(
                symbol="7203",
                account_bucket_id=nisa.bucket_id,
                shares=100,
                average_acquisition_price=Decimal("2350"),
                market_price=latest_prices["7203"],
            ),
            Position(
                symbol="9432",
                account_bucket_id=taxable.bucket_id,
                shares=500,
                average_acquisition_price=Decimal("168"),
                market_price=latest_prices["9432"],
            ),
        ),
        cash=(
            CashState(account_bucket_id=taxable.bucket_id, available_cash=Decimal("900000")),
            CashState(account_bucket_id=nisa.bucket_id, available_cash=Decimal("450000")),
        ),
        tax_states=(
            TaxState(
                account_bucket_id=taxable.bucket_id,
                realized_gain_ytd=Decimal("80000"),
                realized_loss_ytd=Decimal("15000"),
                loss_carryforward_user_input=Decimal("0"),
            ),
            TaxState(account_bucket_id=nisa.bucket_id),
        ),
    )


def next_business_morning(last_trading_date: pd.Timestamp) -> datetime:
    day = last_trading_date.date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime(day.year, day.month, day.day, 11, 30, tzinfo=JST)
