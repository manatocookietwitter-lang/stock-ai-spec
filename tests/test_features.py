from __future__ import annotations

from datetime import timedelta

import pandas as pd
from pandas.testing import assert_frame_equal

from stock_ai.features import V0_MANIFEST, V1_CORE_MANIFEST, FeatureEngine
from stock_ai.fixtures import market_fixture


def test_feature_manifests_are_explicit_and_versioned() -> None:
    assert 20 <= len(V0_MANIFEST.feature_names) < len(V1_CORE_MANIFEST.feature_names)
    assert 40 <= len(V1_CORE_MANIFEST.feature_names) <= 60
    assert len(V1_CORE_MANIFEST.manifest_hash) == 64
    assert "rsi.14" in V1_CORE_MANIFEST.feature_names
    assert "macd.golden_cross" in V1_CORE_MANIFEST.feature_names
    assert "market.breadth" in V1_CORE_MANIFEST.feature_names


def test_feature_generation_has_core_values_and_preserves_warmup_missingness() -> None:
    daily, market, financials = market_fixture(periods=270)
    features = FeatureEngine().transform(daily, market, financials=financials)
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
    daily, market, financials = market_fixture(periods=270)
    engine = FeatureEngine()
    original = engine.transform(daily, market, financials=financials)
    cutoff = pd.Timestamp(daily["trading_date"].sort_values().unique()[250])
    mutated = daily.copy()
    future = pd.to_datetime(mutated["trading_date"]) > cutoff
    mutated.loc[future, ["adjusted_high", "adjusted_low", "adjusted_close"]] *= 10
    changed = engine.transform(mutated, market, financials=financials)
    columns = ["symbol", "trading_date", *V1_CORE_MANIFEST.feature_names]
    assert_frame_equal(
        original.loc[original["trading_date"] <= cutoff, columns].reset_index(drop=True),
        changed.loc[changed["trading_date"] <= cutoff, columns].reset_index(drop=True),
        check_exact=True,
    )


def test_late_financial_correction_does_not_rewrite_earlier_feature() -> None:
    daily, market, financials = market_fixture(periods=270)
    correction_time = pd.Timestamp(financials["available_at"].max()) + timedelta(days=10)
    correction = financials.iloc[[-1]].copy()
    correction["symbol"] = "7203"
    correction["available_at"] = correction_time
    correction["roe"] = 9.99
    with_correction = pd.concat([financials, correction], ignore_index=True)
    before = FeatureEngine().transform(daily, market, financials=financials)
    after = FeatureEngine().transform(daily, market, financials=with_correction)
    earlier = before["available_at"] < correction_time
    assert_frame_equal(
        before.loc[earlier, ["symbol", "trading_date", "fundamental.roe"]].reset_index(drop=True),
        after.loc[earlier, ["symbol", "trading_date", "fundamental.roe"]].reset_index(drop=True),
        check_exact=True,
    )
