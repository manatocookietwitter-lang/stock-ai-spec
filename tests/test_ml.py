from __future__ import annotations

import inspect
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

import stock_ai.ml as public_ml
from stock_ai.features import V0_MANIFEST, FeatureEngine
from stock_ai.fixtures import market_fixture
from stock_ai.ml import (
    MomentumRegressor,
    PurgedExpandingWindowSplitter,
    RidgeRegressor,
    build_supervised_dataset,
    walk_forward_validate,
    write_dataset_snapshot,
)


def test_one_five_twenty_day_targets_are_forward_only() -> None:
    dates = pd.bdate_range("2026-01-01", periods=25)
    frame = pd.DataFrame(
        {
            "symbol": ["A"] * 25,
            "trading_date": dates,
            "available_at": [
                value.to_pydatetime().replace(tzinfo=ZoneInfo("Asia/Tokyo"))
                + timedelta(days=1, hours=8)
                for value in dates
            ],
            "adjusted_close": np.arange(100.0, 125.0),
            "price.return_20d": np.arange(25) / 100,
        }
    )
    dataset = build_supervised_dataset(frame)
    assert dataset.loc[0, "target_return_1d"] == pytest.approx(101 / 100 - 1)
    assert dataset.loc[0, "target_return_5d"] == pytest.approx(105 / 100 - 1)
    assert dataset.loc[0, "target_return_20d"] == pytest.approx(120 / 100 - 1)
    assert pd.isna(dataset.loc[24, "target_return_1d"])
    assert dataset.loc[0, "label_end_date_5d"] == dates[5]


def test_snapshot_is_deterministic_and_immutable(tmp_path: object) -> None:
    from pathlib import Path

    assert isinstance(tmp_path, Path)
    daily, market, financials = market_fixture(periods=80)
    features = FeatureEngine(V0_MANIFEST).transform(daily, market, financials=financials)
    dataset = build_supervised_dataset(features)
    as_of = datetime(2026, 8, 24, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    created = datetime(2026, 8, 24, 11, 31, tzinfo=ZoneInfo("Asia/Tokyo"))
    first = write_dataset_snapshot(
        dataset, tmp_path, manifest=V0_MANIFEST, as_of=as_of, created_at=created
    )
    second = write_dataset_snapshot(
        dataset, tmp_path, manifest=V0_MANIFEST, as_of=as_of, created_at=created
    )
    assert first.snapshot_id == second.snapshot_id
    assert first.parquet_path.read_bytes() == second.parquet_path.read_bytes()


def test_purged_expanding_split_has_gap_and_no_overlapping_labels() -> None:
    dates = pd.bdate_range("2026-01-01", periods=20)
    frame = pd.DataFrame(
        {
            "trading_date": dates,
            "label_end": pd.Series(dates).shift(-3),
        }
    )
    splitter = PurgedExpandingWindowSplitter(
        initial_train_periods=5,
        validation_periods=3,
        step_periods=3,
        purge_periods=2,
        embargo_periods=1,
    )
    folds = tuple(splitter.split(frame, label_end_column="label_end"))
    assert len(folds) >= 2
    for fold in folds:
        train = frame.iloc[fold.train_indices]
        validation = frame.iloc[fold.validation_indices]
        assert train["label_end"].max() < validation["trading_date"].min()
        assert train["trading_date"].max() < validation["trading_date"].min()
    assert len(folds[1].train_indices) > len(folds[0].train_indices)


def test_public_validation_api_exposes_no_random_split() -> None:
    public_names = set(public_ml.__all__)
    assert not any("random" in name.lower() or "shuffle" in name.lower() for name in public_names)
    signature = inspect.signature(PurgedExpandingWindowSplitter)
    assert "shuffle" not in signature.parameters
    assert "random_state" not in signature.parameters


def test_momentum_and_ridge_baselines_and_walk_forward_are_deterministic() -> None:
    dates = pd.bdate_range("2025-01-01", periods=80)
    x = np.linspace(-1, 1, 80)
    frame = pd.DataFrame(
        {
            "trading_date": dates,
            "feature": x,
            "price.return_20d": x / 10,
            "target": 0.02 + 0.3 * x,
            "label_end": pd.Series(dates).shift(-2),
        }
    )
    momentum = MomentumRegressor().fit(frame, frame["target"])
    assert momentum.predict(frame.iloc[[-1]])[0] == pytest.approx(0.1)
    ridge = RidgeRegressor(alpha=0.01).fit(frame[["feature"]], frame["target"])
    assert ridge.predict(frame.loc[[79], ["feature"]])[0] == pytest.approx(0.32, abs=0.01)
    splitter = PurgedExpandingWindowSplitter(
        initial_train_periods=30,
        validation_periods=10,
        purge_periods=1,
        embargo_periods=1,
    )
    first = walk_forward_validate(
        frame,
        feature_names=("feature",),
        target_column="target",
        label_end_column="label_end",
        splitter=splitter,
        model_factory=lambda: RidgeRegressor(alpha=0.01),
    )
    second = walk_forward_validate(
        frame,
        feature_names=("feature",),
        target_column="target",
        label_end_column="label_end",
        splitter=splitter,
        model_factory=lambda: RidgeRegressor(alpha=0.01),
    )
    assert first == second
    assert len(first) >= 3
