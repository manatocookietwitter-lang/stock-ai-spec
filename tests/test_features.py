from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from stock_ai.features import V0_MANIFEST, V1_CORE_MANIFEST, V2_EXTENDED_MANIFEST, FeatureEngine
from stock_ai.features.indicators import obv
from stock_ai.fixtures import market_fixture


def test_feature_manifests_are_explicit_and_versioned() -> None:
    assert 20 <= len(V0_MANIFEST.feature_names) < len(V1_CORE_MANIFEST.feature_names)
    assert 40 <= len(V1_CORE_MANIFEST.feature_names) <= 60
    assert len(V2_EXTENDED_MANIFEST.feature_names) > len(V1_CORE_MANIFEST.feature_names)
    assert len(V1_CORE_MANIFEST.manifest_hash) == 64
    assert "rsi.14" in V1_CORE_MANIFEST.feature_names
    assert "macd.golden_cross" in V1_CORE_MANIFEST.feature_names
    assert "market.breadth" in V1_CORE_MANIFEST.feature_names
    original_hash = V1_CORE_MANIFEST.manifest_hash
    with pytest.raises(TypeError):
        V1_CORE_MANIFEST.feature_definition_hashes["rsi.14"] = "changed"  # type: ignore[index]
    assert V1_CORE_MANIFEST.manifest_hash == original_hash


def test_v2_extended_features_have_exact_causal_daily_formulas() -> None:
    daily, market, sectors, financials = market_fixture(periods=270)
    features = FeatureEngine(V2_EXTENDED_MANIFEST).transform(
        daily, market, sectors, financials=financials
    )
    source = daily.loc[daily["symbol"] == "7203"].sort_values("trading_date")
    observed = features.loc[features["symbol"] == "7203"].iloc[-1]
    latest = source.iloc[-1]
    price_range = float(latest["adjusted_high"] - latest["adjusted_low"])
    expected_body = abs(float(latest["adjusted_close"] - latest["adjusted_open"])) / price_range
    trailing = source["adjusted_close"].astype(float).iloc[-60:].to_numpy()
    expected_drawdown = float((trailing / np.maximum.accumulate(trailing) - 1.0).min())
    trailing_120 = source["adjusted_close"].astype(float).iloc[-120:].to_numpy()
    expected_drawdown_120 = float(
        (trailing_120 / np.maximum.accumulate(trailing_120) - 1.0).min()
    )
    close = source["adjusted_close"].astype(float)
    high = source["adjusted_high"].astype(float)
    low = source["adjusted_low"].astype(float)
    volume = source["adjusted_volume"].astype(float)
    returns = close.pct_change(fill_method=None)
    downside = returns.clip(upper=0.0).pow(2).iloc[-20:].mean() ** 0.5 * np.sqrt(252)
    expected_skew = returns.iloc[-60:].skew()
    expected_tail = returns.iloc[-60:].quantile(0.10)
    expected_volume_z = (volume.iloc[-1] - volume.iloc[-20:].mean()) / volume.iloc[
        -20:
    ].std(ddof=1)
    trading_value = source["trading_value"].astype(float)
    expected_value_ratio = trading_value.iloc[-20:].mean() / trading_value.iloc[-60:].mean()
    balance_volume = obv(close.reset_index(drop=True), volume.reset_index(drop=True))
    expected_obv_slope = (
        (balance_volume.iloc[-1] - balance_volume.iloc[-21])
        / volume.iloc[-20:].mean()
        / 20
    )
    multiplier = (2 * close - high - low) / (high - low)
    multiplier = multiplier.mask((high - low) == 0, 0.0)
    expected_cmf = (multiplier * volume).iloc[-20:].sum() / volume.iloc[-20:].sum()
    prior_high_20 = high.iloc[-21:-1].max()
    prior_low_20 = low.iloc[-21:-1].min()
    prior_high_60 = high.iloc[-61:-1].max()
    prior_low_60 = low.iloc[-61:-1].min()
    adjusted_open = source["adjusted_open"].astype(float)
    expected_body_pct_close = (close.iloc[-1] - adjusted_open.iloc[-1]) / close.iloc[-1]
    expected_upper_wick = (
        high.iloc[-1] - max(adjusted_open.iloc[-1], close.iloc[-1])
    ) / price_range
    expected_lower_wick = (
        min(adjusted_open.iloc[-1], close.iloc[-1]) - low.iloc[-1]
    ) / price_range
    expected_close_location = (close.iloc[-1] - low.iloc[-1]) / price_range
    expected_gap = adjusted_open.iloc[-1] / close.iloc[-2] - 1.0

    assert observed["candle.body_pct_close"] == pytest.approx(expected_body_pct_close)
    assert observed["candle.body_to_range"] == pytest.approx(expected_body)
    assert observed["risk.max_drawdown_60d"] == pytest.approx(expected_drawdown)
    assert observed["risk.max_drawdown_120d"] == pytest.approx(expected_drawdown_120)
    assert observed["risk.downside_vol_20d"] == pytest.approx(downside)
    assert observed["risk.return_skew_60d"] == pytest.approx(expected_skew)
    assert observed["risk.return_quantile_10_60d"] == pytest.approx(expected_tail)
    assert observed["volume.zscore_20d"] == pytest.approx(expected_volume_z)
    assert observed["liquidity.trading_value_ratio_20_60"] == pytest.approx(
        expected_value_ratio
    )
    assert observed["volume.obv_slope_20d"] == pytest.approx(expected_obv_slope)
    assert observed["volume.cmf_20"] == pytest.approx(expected_cmf)
    assert observed["candle.upper_wick_ratio"] == pytest.approx(expected_upper_wick)
    assert observed["candle.lower_wick_ratio"] == pytest.approx(expected_lower_wick)
    assert observed["candle.close_location"] == pytest.approx(expected_close_location)
    assert observed["candle.gap_pct"] == pytest.approx(expected_gap)
    assert observed["breakout.above_20d_high"] == (close.iloc[-1] > prior_high_20)
    assert observed["breakout.below_20d_low"] == (close.iloc[-1] < prior_low_20)
    assert observed["breakout.above_60d_high"] == (close.iloc[-1] > prior_high_60)
    assert observed["breakout.below_60d_low"] == (close.iloc[-1] < prior_low_60)
    assert set(V1_CORE_MANIFEST.feature_names) < set(V2_EXTENDED_MANIFEST.feature_names)

    with pytest.raises(ValueError, match="V2 candle features require adjusted open"):
        FeatureEngine(V2_EXTENDED_MANIFEST).transform(
            daily.drop(columns="adjusted_open"),
            market,
            sectors,
            financials=financials,
        )

    symbol_features = features.loc[features["symbol"] == "7203"].reset_index(drop=True)
    assert symbol_features["risk.downside_vol_20d"].first_valid_index() == 20
    assert symbol_features["risk.max_drawdown_60d"].first_valid_index() == 59
    assert symbol_features["risk.max_drawdown_120d"].first_valid_index() == 119
    assert symbol_features["risk.return_skew_60d"].first_valid_index() == 60
    assert symbol_features["risk.return_quantile_10_60d"].first_valid_index() == 60
    assert symbol_features["volume.zscore_20d"].first_valid_index() == 19
    assert symbol_features["liquidity.trading_value_ratio_20_60"].first_valid_index() == 59
    assert symbol_features["volume.obv_slope_20d"].first_valid_index() == 20
    assert symbol_features["volume.cmf_20"].first_valid_index() == 19
    assert symbol_features["candle.gap_pct"].first_valid_index() == 1
    assert symbol_features["breakout.above_20d_high"].first_valid_index() == 20
    assert symbol_features["breakout.above_60d_high"].first_valid_index() == 60


def test_v2_candle_zero_range_and_bearish_body_conventions_are_exact() -> None:
    daily, market, sectors, financials = market_fixture(periods=80)
    symbol_rows = daily.index[daily["symbol"] == "7203"]
    last_index = symbol_rows[-1]
    bearish = daily.copy()
    bearish.loc[last_index, "adjusted_open"] = bearish.loc[last_index, "adjusted_high"]
    bearish_features = FeatureEngine(V2_EXTENDED_MANIFEST).transform(
        bearish, market, sectors, financials=financials
    )
    bearish_row = bearish_features.loc[bearish_features["symbol"] == "7203"].iloc[-1]
    expected = abs(
        float(bearish.loc[last_index, "adjusted_close"])
        - float(bearish.loc[last_index, "adjusted_open"])
    ) / (
        float(bearish.loc[last_index, "adjusted_high"])
        - float(bearish.loc[last_index, "adjusted_low"])
    )
    assert bearish_row["candle.body_to_range"] == pytest.approx(expected)

    flat = daily.copy()
    flat_price = float(flat.loc[last_index, "adjusted_close"])
    flat.loc[
        last_index,
        ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"],
    ] = flat_price
    flat_features = FeatureEngine(V2_EXTENDED_MANIFEST).transform(
        flat, market, sectors, financials=financials
    )
    flat_row = flat_features.loc[flat_features["symbol"] == "7203"].iloc[-1]
    assert flat_row["candle.body_to_range"] == 0.0
    assert flat_row["candle.upper_wick_ratio"] == 0.0
    assert flat_row["candle.lower_wick_ratio"] == 0.0
    assert flat_row["candle.close_location"] == 0.5


def test_v2_zero_volume_denominators_are_explicitly_missing_not_infinite() -> None:
    daily, market, sectors, financials = market_fixture(periods=80)
    zero_volume = daily.copy()
    symbol_rows = zero_volume["symbol"] == "7203"
    zero_volume.loc[symbol_rows, ["adjusted_volume", "trading_value"]] = 0.0

    features = FeatureEngine(V2_EXTENDED_MANIFEST).transform(
        zero_volume, market, sectors, financials=financials
    )
    observed = features.loc[features["symbol"] == "7203"].iloc[-1]

    for feature_name in (
        "volume.zscore_20d",
        "liquidity.trading_value_ratio_20_60",
        "volume.obv_slope_20d",
        "volume.cmf_20",
    ):
        assert pd.isna(observed[feature_name])
    numeric = features.loc[:, V2_EXTENDED_MANIFEST.feature_names].apply(
        pd.to_numeric, errors="coerce"
    )
    assert not np.isinf(numeric.to_numpy(dtype=float)).any()


def test_feature_generation_has_core_values_and_preserves_warmup_missingness() -> None:
    daily, market, sectors, financials = market_fixture(periods=270)
    features = FeatureEngine().transform(daily, market, sectors, financials=financials)
    first = features.loc[features["symbol"] == "7203"].iloc[0]
    last = features.loc[features["symbol"] == "7203"].iloc[-1]
    assert pd.isna(first["price.return_20d"])
    assert pd.isna(first["trend.sma_distance_200d"])
    assert pd.notna(last["trend.sma_distance_200d"])
    assert pd.notna(last["macd.histogram_pct_price"])
    assert pd.notna(last["rsi.14"])
    assert pd.notna(last["fundamental.roe"])
    assert not features[list(V1_CORE_MANIFEST.feature_names)].eq(0).all(axis=None)


def test_future_price_mutation_does_not_change_past_features() -> None:
    daily, market, sectors, financials = market_fixture(periods=270)
    engine = FeatureEngine(V2_EXTENDED_MANIFEST)
    original = engine.transform(daily, market, sectors, financials=financials)
    cutoff = pd.Timestamp(daily["trading_date"].sort_values().unique()[250])
    mutated = daily.copy()
    future = pd.to_datetime(mutated["trading_date"]) > cutoff
    mutated.loc[
        future, ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close"]
    ] *= 10
    changed = engine.transform(mutated, market, sectors, financials=financials)
    columns = ["symbol", "trading_date", *V2_EXTENDED_MANIFEST.feature_names]
    assert_frame_equal(
        original.loc[original["trading_date"] <= cutoff, columns].reset_index(drop=True),
        changed.loc[changed["trading_date"] <= cutoff, columns].reset_index(drop=True),
        check_exact=True,
    )


def test_late_financial_correction_does_not_rewrite_earlier_feature() -> None:
    daily, market, sectors, financials = market_fixture(periods=270)
    correction_time = pd.Timestamp(financials["available_at"].max()) + timedelta(days=10)
    correction = financials.iloc[[-1]].copy()
    correction["symbol"] = "7203"
    correction["available_at"] = correction_time
    correction["roe"] = 9.99
    with_correction = pd.concat([financials, correction], ignore_index=True)
    before = FeatureEngine().transform(daily, market, sectors, financials=financials)
    after = FeatureEngine().transform(daily, market, sectors, financials=with_correction)
    earlier = before["available_at"] < correction_time
    assert_frame_equal(
        before.loc[earlier, ["symbol", "trading_date", "fundamental.roe"]].reset_index(drop=True),
        after.loc[earlier, ["symbol", "trading_date", "fundamental.roe"]].reset_index(drop=True),
        check_exact=True,
    )


def test_sector_and_breadth_features_are_independent_of_candidate_subset() -> None:
    daily, market, sectors, financials = market_fixture(periods=80)
    all_features = FeatureEngine().transform(daily, market, sectors, financials=financials)
    subset_features = FeatureEngine().transform(
        daily.loc[daily["symbol"] == "7203"],
        market,
        sectors,
        financials=financials,
    )
    columns = [
        "symbol",
        "trading_date",
        "relative.sector_20d",
        "market.breadth",
    ]
    assert_frame_equal(
        all_features.loc[all_features["symbol"] == "7203", columns].reset_index(drop=True),
        subset_features[columns].reset_index(drop=True),
        check_exact=True,
    )


def test_market_breadth_uses_non_flat_denominator() -> None:
    daily, market, sectors, financials = market_fixture(periods=30)
    market.loc[market.index[-1], ["advancing_issues", "declining_issues"]] = [1, 1]
    features = FeatureEngine().transform(daily, market, sectors, financials=financials)
    latest = features.loc[features["trading_date"] == features["trading_date"].max()]
    assert (latest["market.breadth"] == 0.5).all()


def test_missing_required_financial_capability_fails_closed() -> None:
    daily, market, sectors, _ = market_fixture(periods=30)
    with pytest.raises(ValueError, match="BLOCKED_BY_DATA_CAPABILITY"):
        FeatureEngine().transform(daily, market, sectors)


def test_same_day_bar_is_excluded_at_1130_even_if_timestamp_claims_available() -> None:
    daily, market, sectors, financials = market_fixture(periods=30)
    last_date = pd.Timestamp(daily["trading_date"].max())
    as_of = last_date.to_pydatetime().replace(hour=11, minute=30, tzinfo=ZoneInfo("Asia/Tokyo"))
    same_day_available = as_of - timedelta(minutes=30)
    daily.loc[daily["trading_date"] == last_date, "available_at"] = same_day_available
    market.loc[market["trading_date"] == last_date, "available_at"] = same_day_available
    sectors.loc[sectors["trading_date"] == last_date, "available_at"] = same_day_available
    features = FeatureEngine().transform(
        daily,
        market,
        sectors,
        financials=financials,
        as_of=as_of,
    )
    assert features["trading_date"].max() < last_date


def test_invalid_or_empty_daily_capability_fails_with_domain_error() -> None:
    daily, market, sectors, financials = market_fixture(periods=30)
    broken = daily.copy()
    broken.loc[broken.index[0], "shares_outstanding"] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        FeatureEngine().transform(broken, market, sectors, financials=financials)

    before_history = (
        pd.Timestamp(daily["trading_date"].min())
        .to_pydatetime()
        .replace(hour=11, minute=30, tzinfo=ZoneInfo("Asia/Tokyo"))
    )
    with pytest.raises(ValueError, match="no completed daily bars"):
        FeatureEngine().transform(
            daily,
            market,
            sectors,
            financials=financials,
            as_of=before_history,
        )
