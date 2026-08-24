"""Deterministic verification fixtures.

These records are available only through the explicitly named fixture demo and
tests.  They are never selected as a fallback for absent production data.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_ai.data.morning import (
    MorningFreezeMetadata,
    build_morning_universe,
    morning_capabilities,
)
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


def market_fixture(
    periods: int = 340,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
                    "close": close,
                    "adjusted_volume": float(volume),
                    "trading_value": close * volume,
                    "shares_outstanding": shares * 1_000,
                }
            )
    daily = pd.DataFrame(rows)
    daily["__return_1d"] = daily.groupby("symbol", sort=False)["adjusted_close"].pct_change(
        fill_method=None
    )

    market_rows: list[dict[str, object]] = []
    for index, trading_date in enumerate(dates):
        topix_close = 2250 + 0.65 * index + 17 * np.sin(index / 19)
        date_returns = daily.loc[daily["trading_date"] == trading_date, "__return_1d"]
        market_rows.append(
            {
                "trading_date": trading_date,
                "available_at": trading_date.to_pydatetime().replace(tzinfo=JST)
                + timedelta(days=1, hours=8),
                "topix_close": topix_close,
                "advancing_issues": int((date_returns > 0).sum()),
                "declining_issues": int((date_returns < 0).sum()),
            }
        )
    market = pd.DataFrame(market_rows)

    sector_context = (
        daily.groupby(["sector", "trading_date"], as_index=False, observed=True)
        .agg(sector_return_1d=("__return_1d", "mean"), available_at=("available_at", "max"))
        .sort_values(["sector", "trading_date"])
        .reset_index(drop=True)
    )
    daily = daily.drop(columns="__return_1d")

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
    return daily, market, sector_context, pd.DataFrame(financial_rows)


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
                tax_year=as_of.year,
                realized_gain_ytd=Decimal("80000"),
                realized_loss_ytd=Decimal("15000"),
                loss_carryforward_user_input=Decimal("0"),
            ),
            TaxState(account_bucket_id=nisa.bucket_id, tax_year=as_of.year),
        ),
    )


def next_business_morning(last_trading_date: pd.Timestamp) -> datetime:
    day = last_trading_date.date() + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return datetime(day.year, day.month, day.day, 11, 30, tzinfo=JST)


def morning_research_fixture(
    periods: int = 75,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[MorningFreezeMetadata, ...],
    pd.DatetimeIndex,
]:
    """Explicit synchronized morning-history fixture; never a production fallback."""

    if periods < 50:
        raise ValueError("morning research fixture requires at least 50 sessions")
    calendar = _fixture_jpx_calendar(periods + 20)
    dates = calendar[:periods]
    symbols = ("7203", "6758", "8306", "9432")
    sectors = {
        "7203": "Transport Equipment",
        "6758": "Electric Appliances",
        "8306": "Banks",
        "9432": "Information & Communication",
    }
    cutoffs = (
        time(9),
        time(9, 5),
        time(9, 15),
        time(9, 30),
        time(10),
        time(11),
        time(11, 30),
    )
    stock_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_number, trading_date in enumerate(dates):
        day = trading_date.date()
        signal = 0.0015 + 0.0008 * np.sin(day_number / 7)
        for symbol_number, symbol in enumerate(symbols):
            base = 1_000.0 + symbol_number * 250.0
            symbol_signal = signal + symbol_number * 0.0003
            for cutoff_number, cutoff in enumerate(cutoffs):
                fraction = cutoff_number / (len(cutoffs) - 1)
                timestamp = datetime.combine(day, cutoff, tzinfo=JST)
                price = base * (1.0 + symbol_signal * fraction)
                volume = (1_000.0 + symbol_number * 100.0) * (1.0 + day_number * 0.005)
                stock_rows.append(
                    _morning_bar(
                        symbol=symbol,
                        timestamp=timestamp,
                        price=price,
                        volume=volume,
                        source_id=f"stock-{day}-{symbol}-{cutoff}",
                    )
                )
            prior_values = {
                horizon: (len(symbols) - symbol_number) * 0.0001 * (1 + horizon / 20)
                for horizon in (1, 5, 20)
            }
            context_rows.append(
                {
                    "symbol": symbol,
                    "trading_date": trading_date,
                    "sector": sectors[symbol],
                    "prior_close": base * 0.995,
                    "average_daily_trading_value": 100_000_000.0,
                    "prior_expected_return_1d": prior_values[1],
                    "prior_expected_return_5d": prior_values[5],
                    "prior_expected_return_20d": prior_values[20],
                    "prior_downside_quantile": -0.02,
                    "prior_large_loss_probability": 0.10,
                    "prior_uncertainty": 0.02,
                    "prior_rank_pct": (symbol_number + 1) / len(symbols),
                    "prior_model_version": "fixture-daily-v1",
                    "prior_feature_version": "fixture-daily-feature-v2",
                    "prior_data_snapshot_id": "fixture-daily-snapshot",
                    "prior_prediction_as_of": datetime.combine(day, time(8, 50), tzinfo=JST),
                    "is_current_holding": symbol in {"7203", "8306"},
                    "is_candidate": symbol != "7203",
                    "available_at": datetime.combine(day, time(8, 50), tzinfo=JST),
                }
            )
            label: dict[str, object] = {"symbol": symbol, "trading_date": trading_date}
            for horizon in (1, 5, 20):
                end_date = calendar[day_number + horizon]
                label[f"target_return_{horizon}d"] = (
                    prior_values[horizon]
                    + symbol_signal * (1 + horizon / 20)
                    + 0.0002 * np.sin(day_number / 3 + symbol_number)
                )
                label[f"label_entry_at_{horizon}d"] = datetime.combine(
                    day, time(12, 30), tzinfo=JST
                )
                label[f"label_end_date_{horizon}d"] = end_date
                label[f"label_end_at_{horizon}d"] = datetime.combine(
                    end_date.date(), time(15, 30), tzinfo=JST
                )
                label[f"label_available_at_{horizon}d"] = datetime.combine(
                    end_date.date(), time(16), tzinfo=JST
                )
                label[f"label_status_{horizon}d"] = "AVAILABLE"
            label_rows.append(label)
        for cutoff_number, cutoff in enumerate(cutoffs):
            fraction = cutoff_number / (len(cutoffs) - 1)
            timestamp = datetime.combine(day, cutoff, tzinfo=JST)
            market_rows.append(
                _morning_bar(
                    symbol="TOPIX",
                    timestamp=timestamp,
                    price=2_000.0 * (1.0 + signal * 0.2 * fraction),
                    volume=10_000.0,
                    source_id=f"topix-{day}-{cutoff}",
                )
            )
            for sector_name in sectors.values():
                sector_rows.append(
                    _morning_bar(
                        symbol=sector_name,
                        timestamp=timestamp,
                        price=500.0 * (1.0 + signal * 0.4 * fraction),
                        volume=5_000.0,
                        source_id=f"sector-{sector_name}-{day}-{cutoff}",
                    )
                )
    stock = pd.DataFrame(stock_rows)
    market = pd.DataFrame(market_rows)
    sector = pd.DataFrame(sector_rows)
    capability_report = morning_capabilities(
        provider="deterministic-fixture",
        available_fields=(
            "timestamp",
            "price",
            "volume",
            "trading_value",
            "historical_same_time_sessions",
        ),
    )
    all_bars = pd.concat((stock, market, sector), ignore_index=True)
    freeze_metadata: list[MorningFreezeMetadata] = []
    universe = build_morning_universe(
        current_holdings=("7203", "8306"),
        candidates=("6758", "8306", "9432"),
    )
    for trading_date in dates:
        day = trading_date.date()
        day_rows = all_bars.loc[
            all_bars["timestamp"].map(lambda value: pd.Timestamp(value).date()).eq(day)
        ]
        freeze_metadata.append(
            MorningFreezeMetadata(
                as_of=datetime.combine(day, time(11, 30), tzinfo=JST),
                provider="deterministic-fixture",
                source_snapshot_ids=(f"fixture-morning-snapshot-{day}",),
                source_record_ids=tuple(sorted(day_rows["source_record_id"].astype(str).unique())),
                universe=universe,
                capability_report=capability_report,
            )
        )
    return (
        stock,
        pd.DataFrame(context_rows),
        market,
        sector,
        pd.DataFrame(label_rows),
        tuple(freeze_metadata),
        calendar,
    )


def _morning_bar(
    *, symbol: str, timestamp: datetime, price: float, volume: float, source_id: str
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "available_at": timestamp,
        "price": price,
        "volume": volume,
        "trading_value": price * volume,
        "provider": "deterministic-fixture",
        "source_record_id": source_id,
    }


def _fixture_jpx_calendar(periods: int) -> pd.DatetimeIndex:
    """Small explicit JPX-session fixture; never a production calendar fallback."""

    fixture_holidays = {
        date(2026, 1, 12),
        date(2026, 2, 11),
        date(2026, 2, 23),
        date(2026, 3, 20),
        date(2026, 4, 29),
        date(2026, 5, 4),
        date(2026, 5, 5),
        date(2026, 5, 6),
    }
    candidates = pd.bdate_range("2026-01-06", periods=periods + len(fixture_holidays) + 10)
    sessions = pd.DatetimeIndex(
        [value for value in candidates if value.date() not in fixture_holidays]
    )
    return sessions[:periods]
