"""Goal 1 FeatureSet V0 and V1 Core definitions.

The manifest intentionally enables 58 candidates.  Indicators remain model
inputs; none of the definitions emit BUY/SELL actions.
"""

from __future__ import annotations

from stock_ai.features.registry import FeatureDefinition, FeatureRegistry


def _feature(
    name: str,
    family: str,
    formula: str,
    warmup: int,
    *,
    stage: str = "v1_core",
    inputs: tuple[str, ...] = ("adjusted_close",),
    parameters: dict[str, int | float | str] | None = None,
    unit: str = "ratio",
    capabilities: tuple[str, ...] = ("daily_adjusted_ohlcv",),
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        family=family,
        version=1,
        stage=stage,
        inputs=inputs,
        parameters=parameters or {},
        formula=formula,
        warmup_period=warmup,
        output_unit=unit,
        required_capabilities=capabilities,
    )


V0_DEFINITIONS = (
    *(
        _feature(
            f"price.return_{window}d",
            "returns",
            f"adjusted_close / adjusted_close.shift({window}) - 1",
            window + 1,
            stage="v0",
            parameters={"window": window},
        )
        for window in (1, 5, 20, 60, 120)
    ),
    *(
        _feature(
            f"risk.realized_vol_{window}d",
            "volatility",
            f"std(return_1d, {window}) * sqrt(252)",
            window + 1,
            stage="v0",
            parameters={"window": window, "annualization": 252},
        )
        for window in (20, 60)
    ),
    _feature(
        "volume.ratio_20d",
        "volume",
        "adjusted_volume / mean(adjusted_volume, 20)",
        20,
        stage="v0",
        inputs=("adjusted_volume",),
    ),
    _feature(
        "liquidity.trading_value",
        "liquidity",
        "provider trading_value",
        1,
        stage="v0",
        inputs=("trading_value",),
        unit="JPY",
        capabilities=("daily_prices",),
    ),
    _feature(
        "liquidity.trading_value_mean_20d",
        "liquidity",
        "mean(trading_value, 20)",
        20,
        stage="v0",
        inputs=("trading_value",),
        unit="JPY",
        capabilities=("daily_prices",),
    ),
    _feature(
        "price.distance_52w_high",
        "price_position",
        "adjusted_close / rolling_max(adjusted_high, 250) - 1",
        250,
        stage="v0",
        inputs=("adjusted_close", "adjusted_high"),
        parameters={"window": 250},
    ),
    *(
        _feature(
            f"relative.topix_{window}d",
            "relative_strength",
            f"stock_return_{window}d - topix_return_{window}d",
            window + 1,
            stage="v0",
            parameters={"window": window},
            capabilities=("daily_adjusted_ohlcv", "topix_history"),
        )
        for window in (5, 20, 60)
    ),
    *(
        _feature(
            f"relative.sector_{window}d",
            "relative_strength",
            f"stock_return_{window}d - compounded_sector_return_{window}d",
            window + 1,
            stage="v0",
            parameters={"window": window},
            capabilities=("daily_adjusted_ohlcv", "sector_history"),
        )
        for window in (5, 20, 60)
    ),
    *(
        _feature(
            f"fundamental.{column}",
            "fundamental",
            f"latest {column} with available_at <= feature as_of",
            0,
            stage="v0",
            inputs=(column, "available_at"),
            unit="raw",
            capabilities=("financial_summary",),
        )
        for column in (
            "per",
            "pbr",
            "roe",
            "operating_margin",
            "revenue_growth_yoy",
            "operating_profit_growth_yoy",
            "forecast_revision",
        )
    ),
)


V1_ADDITIONAL_DEFINITIONS = (
    *(
        _feature(
            f"trend.sma_distance_{window}d",
            "moving_average",
            f"adjusted_close / SMA({window}) - 1",
            window,
            parameters={"window": window},
        )
        for window in (20, 60, 200)
    ),
    *(
        _feature(
            f"trend.sma_slope_{window}d_5d",
            "moving_average",
            f"(SMA({window}) / SMA({window}).shift(5) - 1) / 5",
            window + 5,
            parameters={"window": window, "slope_window": 5},
        )
        for window in (20, 60)
    ),
    _feature(
        "trend.sma_spread_20_60",
        "moving_average",
        "SMA(20) / SMA(60) - 1",
        60,
        parameters={"short": 20, "long": 60},
    ),
    _feature(
        "trend.sma_gc_20_60",
        "moving_average_cross",
        "SMA20_t > SMA60_t and SMA20_t-1 <= SMA60_t-1",
        61,
        parameters={"short": 20, "long": 60},
        unit="flag",
    ),
    _feature(
        "trend.sma_dc_20_60",
        "moving_average_cross",
        "SMA20_t < SMA60_t and SMA20_t-1 >= SMA60_t-1",
        61,
        parameters={"short": 20, "long": 60},
        unit="flag",
    ),
    _feature(
        "trend.sma_days_since_gc_20_60",
        "moving_average_cross",
        "trading rows since latest SMA20/SMA60 golden cross",
        61,
        parameters={"short": 20, "long": 60},
        unit="days",
    ),
    _feature(
        "trend.sma_days_since_dc_20_60",
        "moving_average_cross",
        "trading rows since latest SMA20/SMA60 dead cross",
        61,
        parameters={"short": 20, "long": 60},
        unit="days",
    ),
    *(
        _feature(
            f"trend.ema_distance_{window}d",
            "moving_average",
            f"adjusted_close / EMA({window}) - 1",
            window,
            parameters={"window": window},
        )
        for window in (12, 26)
    ),
    *(
        _feature(
            f"macd.{name}",
            "macd",
            formula,
            warmup,
            parameters={"fast": 12, "slow": 26, "signal": 9},
        )
        for name, formula, warmup in (
            ("value_pct_price", "(EMA12 - EMA26) / adjusted_close", 26),
            ("signal_pct_price", "EMA9(MACD) / adjusted_close", 34),
            ("histogram_pct_price", "(MACD - Signal) / adjusted_close", 34),
        )
    ),
    _feature(
        "macd.golden_cross",
        "macd",
        "MACD_t > Signal_t and MACD_t-1 <= Signal_t-1",
        35,
        parameters={"fast": 12, "slow": 26, "signal": 9},
        unit="flag",
    ),
    _feature(
        "macd.dead_cross",
        "macd",
        "MACD_t < Signal_t and MACD_t-1 >= Signal_t-1",
        35,
        parameters={"fast": 12, "slow": 26, "signal": 9},
        unit="flag",
    ),
    _feature(
        "macd.days_since_golden_cross",
        "macd",
        "trading rows since latest MACD golden cross",
        35,
        parameters={"fast": 12, "slow": 26, "signal": 9},
        unit="days",
    ),
    _feature(
        "macd.days_since_dead_cross",
        "macd",
        "trading rows since latest MACD dead cross",
        35,
        parameters={"fast": 12, "slow": 26, "signal": 9},
        unit="days",
    ),
    _feature("rsi.14", "rsi", "Wilder RSI(14)", 15, parameters={"window": 14}, unit="index"),
    _feature(
        "bollinger.percent_b_20_2",
        "bollinger",
        "(close - lower) / (upper - lower), SMA20 +/- 2 population std",
        20,
        parameters={"window": 20, "standard_deviations": 2},
        unit="index",
    ),
    _feature(
        "bollinger.width_20_2",
        "bollinger",
        "(upper - lower) / middle",
        20,
        parameters={"window": 20, "standard_deviations": 2},
    ),
    *(
        _feature(
            f"trend.{name}_14",
            "directional_movement",
            formula,
            warmup,
            inputs=("adjusted_high", "adjusted_low", "adjusted_close"),
            parameters={"window": 14},
            unit="index",
        )
        for name, formula, warmup in (
            ("plus_di", "+DI using aligned Wilder TR/DM smoothing", 15),
            ("minus_di", "-DI using aligned Wilder TR/DM smoothing", 15),
            ("adx", "Wilder average of DX after aligned DI seed", 28),
        )
    ),
    _feature(
        "risk.atr_14",
        "volatility",
        "Wilder average of true range",
        15,
        inputs=("adjusted_high", "adjusted_low", "adjusted_close"),
        parameters={"window": 14},
        unit="price",
    ),
    _feature(
        "risk.natr_14",
        "volatility",
        "100 * ATR14 / adjusted_close",
        15,
        inputs=("adjusted_high", "adjusted_low", "adjusted_close"),
        parameters={"window": 14},
        unit="percent",
    ),
    _feature(
        "volume.ratio_5_60",
        "volume",
        "mean(adjusted_volume, 5) / mean(adjusted_volume, 60)",
        60,
        inputs=("adjusted_volume",),
    ),
    _feature(
        "liquidity.turnover_20d",
        "liquidity",
        "mean(adjusted_volume / point_in_time_shares_outstanding, 20)",
        20,
        inputs=("adjusted_volume", "shares_outstanding"),
        capabilities=("daily_adjusted_ohlcv", "shares_outstanding_history"),
    ),
    _feature(
        "volume.obv_slope_5d",
        "money_flow",
        "(OBV_t - OBV_t-5) / mean(adjusted_volume, 20) / 5",
        20,
        inputs=("adjusted_close", "adjusted_volume"),
    ),
    _feature(
        "volume.mfi_14",
        "money_flow",
        "Money Flow Index(14)",
        15,
        inputs=("adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume"),
        parameters={"window": 14},
        unit="index",
    ),
    _feature(
        "price.distance_52w_low",
        "price_position",
        "adjusted_close / rolling_min(adjusted_low, 250) - 1",
        250,
        inputs=("adjusted_close", "adjusted_low"),
        parameters={"window": 250},
    ),
    _feature(
        "market.volatility_20d",
        "market_context",
        "std(topix_return_1d, 20) * sqrt(252)",
        21,
        inputs=("topix_close",),
        capabilities=("topix_history",),
    ),
    _feature(
        "market.breadth",
        "market_context",
        "advancing issues / non-flat observed issues",
        2,
        inputs=("point_in_time_universe_returns",),
        unit="ratio",
        capabilities=("market_breadth",),
    ),
)


V2_EXTENDED_DEFINITIONS = (
    _feature(
        "risk.downside_vol_20d",
        "volatility",
        "sqrt(mean(min(return_1d, 0)^2, 20)) * sqrt(252)",
        21,
        stage="v2_extended",
        parameters={"window": 20, "annualization": 252},
    ),
    *(
        _feature(
            f"risk.max_drawdown_{window}d",
            "volatility",
            f"minimum(close / cumulative_max(close) - 1) within trailing {window} rows",
            window,
            stage="v2_extended",
            parameters={"window": window},
        )
        for window in (60, 120)
    ),
    _feature(
        "risk.return_skew_60d",
        "volatility",
        "sample skew(return_1d, 60)",
        61,
        stage="v2_extended",
        parameters={"window": 60},
        unit="index",
    ),
    _feature(
        "risk.return_quantile_10_60d",
        "volatility",
        "10th percentile(return_1d, 60)",
        61,
        stage="v2_extended",
        parameters={"window": 60, "quantile": 0.10},
    ),
    _feature(
        "volume.zscore_20d",
        "volume",
        "(adjusted_volume - mean_20) / sample_std_20",
        20,
        stage="v2_extended",
        inputs=("adjusted_volume",),
        parameters={"window": 20},
        unit="zscore",
    ),
    _feature(
        "liquidity.trading_value_ratio_20_60",
        "liquidity",
        "mean(trading_value, 20) / mean(trading_value, 60)",
        60,
        stage="v2_extended",
        inputs=("trading_value",),
    ),
    _feature(
        "volume.obv_slope_20d",
        "money_flow",
        "(OBV_t - OBV_t-20) / mean(adjusted_volume, 20) / 20",
        21,
        stage="v2_extended",
        inputs=("adjusted_close", "adjusted_volume"),
    ),
    _feature(
        "volume.cmf_20",
        "money_flow",
        "sum(((2*close-high-low)/(high-low))*volume,20) / sum(volume,20); "
        "zero-range multiplier=0; zero-volume window=missing",
        20,
        stage="v2_extended",
        inputs=("adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume"),
        parameters={"window": 20},
    ),
    *(
        _feature(
            name,
            family,
            formula,
            warmup,
            stage="v2_extended",
            inputs=("adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"),
            unit=unit,
        )
        for name, family, formula, warmup, unit in (
            ("candle.body_pct_close", "candle", "(close-open)/close", 1, "ratio"),
            (
                "candle.body_to_range",
                "candle",
                "abs(close-open)/(high-low); zero-range=0",
                1,
                "ratio",
            ),
            (
                "candle.upper_wick_ratio",
                "candle",
                "(high-max(open,close))/(high-low); zero-range=0",
                1,
                "ratio",
            ),
            (
                "candle.lower_wick_ratio",
                "candle",
                "(min(open,close)-low)/(high-low); zero-range=0",
                1,
                "ratio",
            ),
            (
                "candle.close_location",
                "candle",
                "(close-low)/(high-low); zero-range=0.5",
                1,
                "ratio",
            ),
            ("candle.gap_pct", "candle", "open/previous_close-1", 2, "ratio"),
        )
    ),
    *(
        _feature(
            f"breakout.{direction}_{window}d_{extreme}",
            "breakout",
            formula,
            window + 1,
            stage="v2_extended",
            inputs=("adjusted_high", "adjusted_low", "adjusted_close"),
            parameters={"window": window},
            unit="flag",
        )
        for window in (20, 60)
        for direction, extreme, formula in (
            ("above", "high", f"close > previous rolling {window}d high"),
            ("below", "low", f"close < previous rolling {window}d low"),
        )
    ),
)


FEATURE_REGISTRY = FeatureRegistry(
    (*V0_DEFINITIONS, *V1_ADDITIONAL_DEFINITIONS, *V2_EXTENDED_DEFINITIONS)
)
V0_NAMES = tuple(definition.name for definition in V0_DEFINITIONS)
V1_CORE_NAMES = (*V0_NAMES, *(definition.name for definition in V1_ADDITIONAL_DEFINITIONS))
V2_EXTENDED_NAMES = (
    *V1_CORE_NAMES,
    *(definition.name for definition in V2_EXTENDED_DEFINITIONS),
)
V0_MANIFEST = FEATURE_REGISTRY.manifest("featureset-v0", "1.0.0", V0_NAMES)
V1_CORE_MANIFEST = FEATURE_REGISTRY.manifest("featureset-v1-core", "1.0.0", V1_CORE_NAMES)
V2_EXTENDED_MANIFEST = FEATURE_REGISTRY.manifest(
    "featureset-v2-extended", "2.0.0", V2_EXTENDED_NAMES
)
