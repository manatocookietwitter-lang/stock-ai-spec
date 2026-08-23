from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

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


def test_sma_and_ema_known_values() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert sma(values, 3).iloc[-1] == pytest.approx(4.0)
    assert ema(values, 3).iloc[-1] == pytest.approx(4.0625)
    assert sma(values, 6).isna().all()


def test_macd_constant_series_known_zero() -> None:
    close = pd.Series(np.full(60, 100.0))
    value, signal, histogram = macd(close, 12, 26, 9)
    assert value.iloc[-1] == pytest.approx(0.0)
    assert signal.iloc[-1] == pytest.approx(0.0)
    assert histogram.iloc[-1] == pytest.approx(0.0)


def test_rsi_known_edge_values() -> None:
    rising = pd.Series(np.arange(1.0, 31.0))
    flat = pd.Series(np.full(31, 10.0))
    assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert rsi(flat, 14).iloc[-1] == pytest.approx(50.0)
    assert rsi(rising.iloc[:14], 14).isna().all()


def test_bollinger_known_values() -> None:
    close = pd.Series(np.arange(1.0, 21.0))
    middle, upper, lower, percent_b, width = bollinger(close, 20, 2)
    expected_std = math.sqrt(33.25)
    assert middle.iloc[-1] == pytest.approx(10.5)
    assert upper.iloc[-1] == pytest.approx(10.5 + 2 * expected_std)
    assert lower.iloc[-1] == pytest.approx(10.5 - 2 * expected_std)
    assert percent_b.iloc[-1] == pytest.approx(
        (20 - (10.5 - 2 * expected_std)) / (4 * expected_std)
    )
    assert width.iloc[-1] == pytest.approx((4 * expected_std) / 10.5)


def test_atr_natr_adx_and_di_known_trend() -> None:
    close = pd.Series(np.arange(1.0, 61.0))
    high = close + 1
    low = close - 1
    average_true_range = atr(high, low, close, 14)
    plus_di, minus_di, adx = directional_movement(high, low, close, 14)
    assert average_true_range.iloc[-1] == pytest.approx(2.0)
    assert 100 * average_true_range.iloc[-1] / close.iloc[-1] == pytest.approx(10 / 3)
    assert plus_di.iloc[-1] == pytest.approx(50.0)
    assert minus_di.iloc[-1] == pytest.approx(0.0)
    assert adx.iloc[-1] == pytest.approx(100.0)
    assert average_true_range.first_valid_index() == 14
    assert plus_di.first_valid_index() == 14
    assert minus_di.first_valid_index() == 14
    assert adx.first_valid_index() == 27


def test_obv_and_mfi_known_values() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    volume = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    assert obv(close, volume).tolist() == [0.0, 20.0, 50.0, 90.0, 140.0]
    value = mfi(close + 1, close - 1, close, volume, period=3)
    assert value.iloc[-1] == pytest.approx(100.0)


def test_cross_boundary_and_warmup() -> None:
    short = pd.Series([1.0, 1.0, 3.0, 1.0])
    long = pd.Series([2.0, 2.0, 2.0, 2.0])
    golden, dead = cross_flags(short, long)
    assert math.isnan(golden.iloc[0])
    assert golden.iloc[2] == 1.0
    assert dead.iloc[3] == 1.0
    assert golden.iloc[1] == 0.0
    elapsed = days_since_event(golden)
    assert pd.isna(elapsed.iloc[1])
    assert elapsed.iloc[2:].tolist() == [0.0, 1.0]
