from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
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
    original_hash = V1_CORE_MANIFEST.manifest_hash
    with pytest.raises(TypeError):
        V1_CORE_MANIFEST.feature_definition_hashes["rsi.14"] = "changed"  # type: ignore[index]
    assert V1_CORE_MANIFEST.manifest_hash == original_hash


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
    engine = FeatureEngine()
    original = engine.transform(daily, market, sectors, financials=financials)
    cutoff = pd.Timestamp(daily["trading_date"].sort_values().unique()[250])
    mutated = daily.copy()
    future = pd.to_datetime(mutated["trading_date"]) > cutoff
    mutated.loc[future, ["adjusted_high", "adjusted_low", "adjusted_close"]] *= 10
    changed = engine.transform(mutated, market, sectors, financials=financials)
    columns = ["symbol", "trading_date", *V1_CORE_MANIFEST.feature_names]
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
