"""Small, deterministic indicator implementations used by FeatureSet V0/V1 Core."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(values: pd.Series, window: int) -> pd.Series:
    return values.rolling(window=window, min_periods=window).mean()


def ema(values: pd.Series, span: int) -> pd.Series:
    return values.ewm(span=span, adjust=False, min_periods=span).mean()


def cross_flags(short: pd.Series, long: pd.Series) -> tuple[pd.Series, pd.Series]:
    valid = short.notna() & long.notna() & short.shift(1).notna() & long.shift(1).notna()
    golden = ((short > long) & (short.shift(1) <= long.shift(1))).astype(float).where(valid)
    dead = ((short < long) & (short.shift(1) >= long.shift(1))).astype(float).where(valid)
    return golden, dead


def days_since_event(flag: pd.Series) -> pd.Series:
    """Trading-row count since the most recent true event; missing before the first."""
    result = np.full(len(flag), np.nan, dtype=float)
    elapsed: int | None = None
    for position, value in enumerate(flag.to_numpy(dtype=float)):
        if pd.isna(value):
            continue
        if value == 1:
            elapsed = 0
        elif elapsed is not None:
            elapsed += 1
        if elapsed is not None:
            result[position] = float(elapsed)
    return pd.Series(result, index=flag.index, dtype=float)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_line = ema(close, fast)
    slow_line = ema(close, slow)
    value = fast_line - slow_line
    signal_line = ema(value, signal)
    return value, signal_line, value - signal_line


def _wilder_average(values: pd.Series, period: int) -> pd.Series:
    data = values.astype(float).to_numpy()
    result = np.full(data.shape, np.nan, dtype=float)
    valid_positions = np.flatnonzero(~np.isnan(data))
    if len(valid_positions) < period:
        return pd.Series(result, index=values.index, dtype=float)
    seed_positions = valid_positions[:period]
    seed_index = int(seed_positions[-1])
    previous = float(np.mean(data[seed_positions]))
    result[seed_index] = previous
    for index in range(seed_index + 1, len(data)):
        value = data[index]
        if np.isnan(value):
            continue
        previous = ((period - 1) * previous + value) / period
        result[index] = previous
    return pd.Series(result, index=values.index, dtype=float)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = _wilder_average(gain, period)
    average_loss = _wilder_average(loss, period)
    relative_strength = average_gain / average_loss
    value = 100 - (100 / (1 + relative_strength))
    value = value.mask((average_loss == 0) & (average_gain > 0), 100.0)
    return value.mask((average_loss == 0) & (average_gain == 0), 50.0)


def bollinger(
    close: pd.Series, window: int = 20, standard_deviations: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    middle = sma(close, window)
    deviation = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + standard_deviations * deviation
    lower = middle - standard_deviations * deviation
    width = (upper - lower) / middle
    percent_b = (close - lower) / (upper - lower)
    percent_b = percent_b.mask((upper - lower) == 0, 0.5)
    return middle, upper, lower, percent_b, width


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    components = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    )
    result = components.max(axis=1, skipna=True)
    if not result.empty:
        result.iloc[0] = np.nan
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return _wilder_average(true_range(high, low, close), period)


def directional_movement(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where((upward > downward) & (upward > 0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0), 0.0)
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan
    average_true_range = atr(high, low, close, period)
    plus_di = 100 * _wilder_average(plus_dm, period) / average_true_range
    minus_di = 100 * _wilder_average(minus_dm, period) / average_true_range
    denominator = plus_di + minus_di
    dx = 100 * (plus_di - minus_di).abs() / denominator
    dx = dx.mask(denominator == 0, 0.0)
    adx = _wilder_average(dx, period)
    return plus_di, minus_di, adx


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    difference = close.diff()
    direction = difference.gt(0).astype(float) - difference.lt(0).astype(float)
    return pd.Series((direction * volume).cumsum(), index=close.index, dtype=float)


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    typical_price = (high + low + close) / 3
    raw_flow = typical_price * volume
    direction = typical_price.diff()
    positive = raw_flow.where(direction > 0, 0.0)
    negative = raw_flow.where(direction < 0, 0.0)
    positive.iloc[0] = np.nan
    negative.iloc[0] = np.nan
    positive_sum = positive.rolling(period, min_periods=period).sum()
    negative_sum = negative.rolling(period, min_periods=period).sum()
    ratio = positive_sum / negative_sum
    value = 100 - 100 / (1 + ratio)
    value = value.mask((negative_sum == 0) & (positive_sum > 0), 100.0)
    return value.mask((negative_sum == 0) & (positive_sum == 0), 50.0)
