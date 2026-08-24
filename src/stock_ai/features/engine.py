"""Causal FeatureSet V0/V1 Core calculation over completed daily bars."""

from __future__ import annotations

from datetime import datetime
from typing import Final

import numpy as np
import pandas as pd

from stock_ai.data.point_in_time import point_in_time_view
from stock_ai.features.catalog import V1_CORE_MANIFEST
from stock_ai.features.indicators import (
    atr,
    bollinger,
    cross_flags,
    days_since_event,
    directional_movement,
    ema,
    macd,
    mfi,
    obv,
    rsi,
    sma,
)
from stock_ai.features.registry import FeatureSetManifest

DAILY_REQUIRED: Final = {
    "symbol",
    "sector",
    "trading_date",
    "available_at",
    "adjusted_high",
    "adjusted_low",
    "adjusted_close",
    "adjusted_volume",
    "close",
    "trading_value",
}
MARKET_REQUIRED: Final = {
    "trading_date",
    "available_at",
    "topix_close",
    "advancing_issues",
    "declining_issues",
}
SECTOR_REQUIRED: Final = {"sector", "trading_date", "available_at", "sector_return_1d"}
FINANCIAL_COLUMNS: Final = (
    "per",
    "pbr",
    "roe",
    "operating_margin",
    "revenue_growth_yoy",
    "operating_profit_growth_yoy",
    "forecast_revision",
)


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _require_finite_numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    label: str,
    *,
    allow_missing: bool = False,
) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = ~np.isfinite(numeric.to_numpy(dtype=float))
        if allow_missing:
            invalid &= numeric.notna().to_numpy()
        if invalid.any():
            raise ValueError(f"{label} column {column} must contain finite numeric values")


def _compute_symbol(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("trading_date").copy()
    if "trading_session_index" not in group:
        return _compute_contiguous_symbol(group)
    session_index = pd.to_numeric(group["trading_session_index"], errors="coerce")
    if session_index.isna().any():
        raise ValueError("trading_session_index must be present for every production row")
    segment = session_index.diff().ne(1).cumsum()
    return pd.concat(
        [_compute_contiguous_symbol(part) for _, part in group.groupby(segment, sort=False)],
        ignore_index=True,
    )


def _compute_contiguous_symbol(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("trading_date").copy()
    close = group["adjusted_close"].astype(float)
    high = group["adjusted_high"].astype(float)
    low = group["adjusted_low"].astype(float)
    volume = group["adjusted_volume"].astype(float)
    trading_value = group["trading_value"].astype(float)

    output = group[["symbol", "sector", "trading_date", "available_at"]].copy()
    output["close"] = group["close"].astype(float)
    output["adjusted_close"] = close
    one_day_return = close.pct_change(fill_method=None)
    output["__return_1d"] = one_day_return
    for window in (1, 5, 20, 60, 120):
        output[f"price.return_{window}d"] = close.pct_change(window, fill_method=None)
    for window in (20, 60):
        output[f"risk.realized_vol_{window}d"] = one_day_return.rolling(
            window, min_periods=window
        ).std(ddof=1) * np.sqrt(252)

    volume_mean_20 = volume.rolling(20, min_periods=20).mean()
    output["volume.ratio_20d"] = volume / volume_mean_20
    output["liquidity.trading_value"] = trading_value
    output["liquidity.trading_value_mean_20d"] = trading_value.rolling(20, min_periods=20).mean()
    output["price.distance_52w_high"] = close / high.rolling(250, min_periods=250).max() - 1
    output["price.distance_52w_low"] = close / low.rolling(250, min_periods=250).min() - 1

    sma_20 = sma(close, 20)
    sma_60 = sma(close, 60)
    sma_200 = sma(close, 200)
    for window, average in ((20, sma_20), (60, sma_60), (200, sma_200)):
        output[f"trend.sma_distance_{window}d"] = close / average - 1
    output["trend.sma_slope_20d_5d"] = (sma_20 / sma_20.shift(5) - 1) / 5
    output["trend.sma_slope_60d_5d"] = (sma_60 / sma_60.shift(5) - 1) / 5
    output["trend.sma_spread_20_60"] = sma_20 / sma_60 - 1
    sma_gc, sma_dc = cross_flags(sma_20, sma_60)
    output["trend.sma_gc_20_60"] = sma_gc
    output["trend.sma_dc_20_60"] = sma_dc
    output["trend.sma_days_since_gc_20_60"] = days_since_event(sma_gc)
    output["trend.sma_days_since_dc_20_60"] = days_since_event(sma_dc)

    for window in (12, 26):
        output[f"trend.ema_distance_{window}d"] = close / ema(close, window) - 1
    macd_value, signal, histogram = macd(close)
    output["macd.value_pct_price"] = macd_value / close
    output["macd.signal_pct_price"] = signal / close
    output["macd.histogram_pct_price"] = histogram / close
    macd_gc, macd_dc = cross_flags(macd_value, signal)
    output["macd.golden_cross"] = macd_gc
    output["macd.dead_cross"] = macd_dc
    output["macd.days_since_golden_cross"] = days_since_event(macd_gc)
    output["macd.days_since_dead_cross"] = days_since_event(macd_dc)
    output["rsi.14"] = rsi(close, 14)

    _, _, _, percent_b, width = bollinger(close, 20, 2)
    output["bollinger.percent_b_20_2"] = percent_b
    output["bollinger.width_20_2"] = width
    plus_di, minus_di, adx = directional_movement(high, low, close, 14)
    output["trend.plus_di_14"] = plus_di
    output["trend.minus_di_14"] = minus_di
    output["trend.adx_14"] = adx
    average_true_range = atr(high, low, close, 14)
    output["risk.atr_14"] = average_true_range
    output["risk.natr_14"] = 100 * average_true_range / close
    output["volume.ratio_5_60"] = (
        volume.rolling(5, min_periods=5).mean() / volume.rolling(60, min_periods=60).mean()
    )
    if "shares_outstanding" in group:
        shares = pd.to_numeric(group["shares_outstanding"], errors="coerce")
    else:
        shares = pd.Series(np.nan, index=group.index, dtype="float64")
    turnover = volume / shares
    output["liquidity.turnover_20d"] = turnover.rolling(20, min_periods=20).mean()
    on_balance_volume = obv(close, volume)
    output["volume.obv_slope_5d"] = (
        (on_balance_volume - on_balance_volume.shift(5)) / volume_mean_20 / 5
    )
    output["volume.mfi_14"] = mfi(high, low, close, volume, 14)
    return output


def _merge_financials(features: pd.DataFrame, financials: pd.DataFrame | None) -> pd.DataFrame:
    output_parts: list[pd.DataFrame] = []
    if financials is None:
        raise ValueError("BLOCKED_BY_DATA_CAPABILITY: financial_summary is required")

    required = {"symbol", "available_at", *FINANCIAL_COLUMNS}
    _require_columns(financials, required, "financial data")
    _require_finite_numeric(
        financials,
        FINANCIAL_COLUMNS,
        "financial data",
        allow_missing=True,
    )
    right = financials.copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
    for symbol, left_group in features.groupby("symbol", sort=False):
        right_group = right.loc[right["symbol"] == symbol, ["available_at", *FINANCIAL_COLUMNS]]
        left_group = left_group.sort_values("available_at")
        if right_group.empty:
            merged = left_group.copy()
            for column in FINANCIAL_COLUMNS:
                merged[column] = np.nan
        else:
            merged = pd.merge_asof(
                left_group,
                right_group.sort_values("available_at"),
                on="available_at",
                direction="backward",
                allow_exact_matches=True,
            )
        output_parts.append(merged)
    output = pd.concat(output_parts, ignore_index=True)
    return output.rename(columns={column: f"fundamental.{column}" for column in FINANCIAL_COLUMNS})


class FeatureEngine:
    """Computes only causal daily features for the explicitly supplied data."""

    def __init__(self, manifest: FeatureSetManifest = V1_CORE_MANIFEST) -> None:
        self.manifest = manifest

    def transform(
        self,
        daily: pd.DataFrame,
        market_context: pd.DataFrame,
        sector_context: pd.DataFrame,
        *,
        financials: pd.DataFrame | None = None,
        as_of: datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        _require_columns(daily, DAILY_REQUIRED, "daily market data")
        _require_columns(market_context, MARKET_REQUIRED, "market context data")
        _require_columns(sector_context, SECTOR_REQUIRED, "sector context data")
        safe_daily = daily.copy()
        safe_market = market_context.copy()
        safe_sector = sector_context.copy()
        safe_daily["available_at"] = pd.to_datetime(safe_daily["available_at"], utc=True)
        safe_market["available_at"] = pd.to_datetime(safe_market["available_at"], utc=True)
        safe_sector["available_at"] = pd.to_datetime(safe_sector["available_at"], utc=True)
        safe_daily["trading_date"] = pd.to_datetime(safe_daily["trading_date"]).dt.normalize()
        safe_market["trading_date"] = pd.to_datetime(safe_market["trading_date"]).dt.normalize()
        safe_sector["trading_date"] = pd.to_datetime(safe_sector["trading_date"]).dt.normalize()
        if as_of is not None:
            safe_daily = point_in_time_view(safe_daily, as_of)
            safe_market = point_in_time_view(safe_market, as_of)
            safe_sector = point_in_time_view(safe_sector, as_of)
            if financials is not None:
                financials = point_in_time_view(financials, as_of)
            cutoff = pd.Timestamp(as_of)
            if cutoff.tzinfo is None:
                raise ValueError("as_of must be timezone-aware")
            local_day = cutoff.tz_convert("Asia/Tokyo").tz_localize(None).normalize()
            safe_daily = safe_daily.loc[safe_daily["trading_date"] < local_day]
            safe_market = safe_market.loc[safe_market["trading_date"] < local_day]
            safe_sector = safe_sector.loc[safe_sector["trading_date"] < local_day]
        if safe_daily.duplicated(["symbol", "trading_date"]).any():
            raise ValueError("daily data contains duplicate symbol/trading_date rows")
        if safe_market.duplicated(["trading_date"]).any():
            raise ValueError("market context contains duplicate trading_date rows")
        if safe_sector.duplicated(["sector", "trading_date"]).any():
            raise ValueError("sector context contains duplicate sector/trading_date rows")
        if safe_daily.empty:
            raise ValueError(
                "BLOCKED_BY_DATA_CAPABILITY: no completed daily bars are available at as_of"
            )
        requires_shares = "liquidity.turnover_20d" in self.manifest.feature_names
        if requires_shares and "shares_outstanding" not in safe_daily:
            raise ValueError(
                "BLOCKED_BY_DATA_CAPABILITY: V1 turnover requires shares_outstanding history"
            )
        daily_numeric = (
            "adjusted_high",
            "adjusted_low",
            "adjusted_close",
            "adjusted_volume",
            "close",
            "trading_value",
        )
        _require_finite_numeric(safe_daily, daily_numeric, "daily market data")
        if "shares_outstanding" in safe_daily:
            _require_finite_numeric(
                safe_daily,
                ("shares_outstanding",),
                "daily market data",
                allow_missing=True,
            )
            observed_shares = pd.to_numeric(safe_daily["shares_outstanding"], errors="coerce")
            if (observed_shares.dropna() <= 0).any():
                raise ValueError("observed shares outstanding must be strictly positive")
            if requires_shares and observed_shares.notna().sum() == 0:
                raise ValueError(
                    "BLOCKED_BY_DATA_CAPABILITY: no point-in-time shares outstanding are available"
                )
        positive_columns = (
            "adjusted_high",
            "adjusted_low",
            "adjusted_close",
            "close",
        )
        if any((pd.to_numeric(safe_daily[column]) <= 0).any() for column in positive_columns):
            raise ValueError("daily prices and shares outstanding must be strictly positive")
        if (
            pd.to_numeric(safe_daily["adjusted_high"]) < pd.to_numeric(safe_daily["adjusted_low"])
        ).any():
            raise ValueError("daily adjusted high cannot be below adjusted low")
        _require_finite_numeric(
            safe_market,
            ("topix_close", "advancing_issues", "declining_issues"),
            "market context data",
        )
        if (pd.to_numeric(safe_market["topix_close"]) <= 0).any():
            raise ValueError("TOPIX close must be strictly positive")
        if any(
            (pd.to_numeric(safe_market[column]) < 0).any()
            for column in ("advancing_issues", "declining_issues")
        ):
            raise ValueError("market breadth issue counts cannot be negative")
        _require_finite_numeric(
            safe_sector,
            ("sector_return_1d",),
            "sector context data",
            allow_missing=True,
        )
        missing_market_dates = set(safe_daily["trading_date"]) - set(safe_market["trading_date"])
        if missing_market_dates:
            raise ValueError(
                "BLOCKED_BY_DATA_CAPABILITY: TOPIX/market context coverage is incomplete"
            )
        daily_sector_keys = set(zip(safe_daily["sector"], safe_daily["trading_date"], strict=True))
        sector_keys = set(zip(safe_sector["sector"], safe_sector["trading_date"], strict=True))
        if daily_sector_keys - sector_keys:
            raise ValueError("BLOCKED_BY_DATA_CAPABILITY: sector context coverage is incomplete")

        parts = [_compute_symbol(group) for _, group in safe_daily.groupby("symbol", sort=True)]
        computed = pd.concat(parts, ignore_index=True)

        sector_daily = safe_sector.sort_values(["sector", "trading_date"]).copy()
        for window in (5, 20, 60):
            sector_daily[f"__sector_{window}d"] = sector_daily.groupby("sector")[
                "sector_return_1d"
            ].transform(
                lambda values, size=window: (
                    (1 + values).rolling(size, min_periods=size).apply(np.prod, raw=True) - 1
                )
            )
        sector_feature_columns = [
            "sector",
            "trading_date",
            "available_at",
            *[f"__sector_{window}d" for window in (5, 20, 60)],
        ]
        computed = computed.merge(
            sector_daily[sector_feature_columns],
            on=["sector", "trading_date"],
            how="left",
            suffixes=("", "_sector"),
        )
        sector_is_late = computed["available_at_sector"] > computed["available_at"]
        for window in (5, 20, 60):
            computed.loc[sector_is_late, f"__sector_{window}d"] = np.nan
            computed[f"relative.sector_{window}d"] = (
                computed[f"price.return_{window}d"] - computed[f"__sector_{window}d"]
            )

        safe_market = safe_market.sort_values("trading_date").copy()
        market_return_1d = safe_market["topix_close"].astype(float).pct_change(fill_method=None)
        safe_market["__topix_return_1d"] = market_return_1d
        for window in (5, 20, 60):
            safe_market[f"__topix_{window}d"] = (
                safe_market["topix_close"].astype(float).pct_change(window, fill_method=None)
            )
        safe_market["market.volatility_20d"] = market_return_1d.rolling(20, min_periods=20).std(
            ddof=1
        ) * np.sqrt(252)
        non_flat = safe_market["advancing_issues"] + safe_market["declining_issues"]
        safe_market["market.breadth"] = safe_market["advancing_issues"] / non_flat
        safe_market.loc[non_flat == 0, "market.breadth"] = np.nan
        market_columns = [
            "trading_date",
            "available_at",
            "market.volatility_20d",
            "market.breadth",
            *[f"__topix_{window}d" for window in (5, 20, 60)],
        ]
        computed = computed.merge(
            safe_market[market_columns],
            on="trading_date",
            how="left",
            suffixes=("", "_market"),
        )
        market_is_late = computed["available_at_market"] > computed["available_at"]
        for column in [
            "market.volatility_20d",
            "market.breadth",
            *[f"__topix_{w}d" for w in (5, 20, 60)],
        ]:
            computed.loc[market_is_late, column] = np.nan
        for window in (5, 20, 60):
            computed[f"relative.topix_{window}d"] = (
                computed[f"price.return_{window}d"] - computed[f"__topix_{window}d"]
            )

        computed = _merge_financials(computed, financials)
        selected = [
            "symbol",
            "sector",
            "trading_date",
            "available_at",
            "close",
            "adjusted_close",
            *self.manifest.feature_names,
        ]
        missing_outputs = set(self.manifest.feature_names) - set(computed.columns)
        if missing_outputs:
            raise RuntimeError(
                f"feature implementation is missing manifest columns: {missing_outputs}"
            )
        return computed[selected].sort_values(["trading_date", "symbol"]).reset_index(drop=True)

    def latest_snapshot(
        self,
        daily: pd.DataFrame,
        market_context: pd.DataFrame,
        sector_context: pd.DataFrame,
        *,
        as_of: datetime | pd.Timestamp,
        financials: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        history = self.transform(
            daily,
            market_context,
            sector_context,
            financials=financials,
            as_of=as_of,
        )
        return history.groupby("symbol", as_index=False, sort=True).tail(1).reset_index(drop=True)
