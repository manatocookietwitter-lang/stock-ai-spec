"""F13/F14 point-in-time features for the 09:00--11:30 morning session."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time
from types import MappingProxyType
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_ai.data.contracts import CapabilityStatus
from stock_ai.data.morning import (
    MORNING_CUTOFFS,
    MORNING_FREEZE,
    MorningCapabilityReport,
    MorningDataError,
    MorningFreezeMetadata,
    assert_morning_freeze_coverage,
    validate_morning_bars,
)
from stock_ai.features.registry import FeatureDefinition, FeatureRegistry, FeatureSetManifest

JST = ZoneInfo("Asia/Tokyo")
_PROFILE_SESSIONS = 20
_RETURN_CUTOFFS = MORNING_CUTOFFS[1:]


def _suffix(value: time) -> str:
    return f"{value.hour:02d}{value.minute:02d}"


def _definition(
    name: str,
    family: str,
    formula: str,
    *,
    warmup: int = 1,
    unit: str = "ratio",
    capabilities: tuple[str, ...] = ("intraday_bars",),
    stage: str = "morning_core",
    inputs: tuple[str, ...] = ("price", "timestamp", "available_at"),
    parameters: Mapping[str, int | float | str] | None = None,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        family=family,
        version=1,
        stage=stage,
        inputs=inputs,
        parameters=parameters or {},
        formula=formula,
        implementation="stock_ai.features.morning@1.0.0",
        warmup_period=warmup,
        output_unit=unit,
        availability_rule="source available_at <= exact 11:30 JST freeze",
        required_capabilities=capabilities,
    )


_RETURN_DEFINITIONS = tuple(
    _definition(
        f"morning.return_0900_{_suffix(cutoff)}",
        "morning_return",
        f"price({_suffix(cutoff)}) / price(0900) - 1",
        parameters={"cutoff": _suffix(cutoff)},
    )
    for cutoff in _RETURN_CUTOFFS
)
_RELATIVE_DEFINITIONS = tuple(
    definition
    for cutoff in _RETURN_CUTOFFS
    for definition in (
        _definition(
            f"morning.topix_relative_{_suffix(cutoff)}",
            "morning_relative",
            f"stock_return_0900_{_suffix(cutoff)} - topix_return_0900_{_suffix(cutoff)}",
            capabilities=("intraday_bars", "morning_ohlc"),
            parameters={"cutoff": _suffix(cutoff), "context": "TOPIX"},
        ),
        _definition(
            f"morning.sector_relative_{_suffix(cutoff)}",
            "morning_relative",
            f"stock_return_0900_{_suffix(cutoff)} - sector_return_0900_{_suffix(cutoff)}",
            capabilities=("intraday_bars", "morning_ohlc"),
            parameters={"cutoff": _suffix(cutoff), "context": "sector"},
        ),
    )
)
_VOLUME_DEFINITIONS = tuple(
    definition
    for cutoff in _RETURN_CUTOFFS
    for definition in (
        _definition(
            f"morning.cumulative_volume_{_suffix(cutoff)}",
            "morning_volume",
            f"sum(bar_volume, 0900..{_suffix(cutoff)})",
            unit="shares",
            inputs=("volume", "timestamp", "available_at"),
            parameters={"cutoff": _suffix(cutoff)},
        ),
        _definition(
            f"morning.volume_progress_{_suffix(cutoff)}_20d",
            "morning_volume_profile",
            f"cumulative_volume_{_suffix(cutoff)} / prior_20_session_same_time_mean",
            warmup=_PROFILE_SESSIONS + 1,
            inputs=("volume", "timestamp", "available_at"),
            capabilities=("intraday_bars", "intraday_volume_profile"),
            parameters={"cutoff": _suffix(cutoff), "sessions": _PROFILE_SESSIONS},
        ),
        _definition(
            f"morning.trading_value_progress_{_suffix(cutoff)}_20d",
            "morning_volume_profile",
            f"cumulative_trading_value_{_suffix(cutoff)} / prior_20_session_same_time_mean",
            warmup=_PROFILE_SESSIONS + 1,
            inputs=("trading_value", "timestamp", "available_at"),
            capabilities=("intraday_bars", "intraday_volume_profile"),
            parameters={"cutoff": _suffix(cutoff), "sessions": _PROFILE_SESSIONS},
        ),
    )
)

MORNING_CORE_DEFINITIONS = (
    _definition(
        "morning.gap_prev_close_to_0900",
        "morning_return",
        "price(0900) / prior_adjusted_close - 1",
        inputs=("price", "prior_close", "timestamp", "available_at"),
    ),
    *_RETURN_DEFINITIONS,
    *_RELATIVE_DEFINITIONS,
    _definition("morning.high", "morning_range", "max(price, 0900..1130)", unit="JPY"),
    _definition("morning.low", "morning_range", "min(price, 0900..1130)", unit="JPY"),
    _definition(
        "morning.range_pct_open",
        "morning_range",
        "(morning_high - morning_low) / price(0900)",
    ),
    _definition(
        "morning.realized_volatility",
        "morning_range",
        "sqrt(sum(log(price_t / price_t-1)^2, 0900..1130))",
    ),
    _definition(
        "morning.vwap",
        "morning_vwap",
        "sum(trading_value) / sum(volume); missing when cumulative volume is zero",
        unit="JPY",
        inputs=("volume", "trading_value", "available_at"),
    ),
    _definition(
        "morning.price_to_vwap",
        "morning_vwap",
        "price(1130) / morning_vwap - 1",
        inputs=("price", "volume", "trading_value", "available_at"),
    ),
    _definition(
        "morning.close_location",
        "morning_range",
        "(price(1130) - low) / (high - low); 0.5 when high equals low",
    ),
    _definition(
        "morning.drop_from_high",
        "morning_range",
        "price(1130) / morning_high - 1",
    ),
    _definition(
        "morning.rebound_from_low",
        "morning_range",
        "price(1130) / morning_low - 1",
    ),
    *_VOLUME_DEFINITIONS,
    _definition(
        "morning.monitored_volume_rank_pct",
        "morning_cross_section",
        "within monitored universe percentile rank of cumulative volume at 11:30",
    ),
    _definition(
        "morning.candidate_volume_rank_pct",
        "morning_cross_section",
        "within candidate subset percentile rank of cumulative volume at 11:30",
    ),
    _definition(
        "morning.is_current_holding",
        "morning_role",
        "1 when symbol is a current holding at the session's monitoring freeze",
        unit="flag",
        inputs=("is_current_holding",),
    ),
    _definition(
        "morning.is_candidate",
        "morning_role",
        "1 when symbol is a daily-model candidate at the monitoring freeze",
        unit="flag",
        inputs=("is_candidate",),
    ),
    *(
        _definition(
            f"morning.prior_{name}",
            "morning_prior_forecast",
            f"daily model {name} frozen before morning inference",
            inputs=(f"prior_{name}", "available_at"),
            unit="rank" if name == "rank_pct" else "ratio",
            capabilities=(),
        )
        for name in (
            "expected_return_1d",
            "expected_return_5d",
            "expected_return_20d",
            "downside_quantile",
            "large_loss_probability",
            "uncertainty",
            "rank_pct",
        )
    ),
)

MORNING_MICROSTRUCTURE_DEFINITIONS = (
    _definition(
        "morning.micro_spread_bps",
        "morning_microstructure",
        "spread(1130) / price(1130) * 10000",
        unit="bps",
        capabilities=("quotes",),
        stage="morning_microstructure",
        inputs=("spread", "price", "available_at"),
    ),
    _definition(
        "morning.micro_price_to_midpoint",
        "morning_microstructure",
        "price(1130) / ((bid(1130) + ask(1130)) / 2) - 1",
        capabilities=("quotes",),
        stage="morning_microstructure",
        inputs=("bid", "ask", "price", "available_at"),
    ),
    _definition(
        "morning.micro_order_book_imbalance",
        "morning_microstructure",
        "(bid_size - ask_size) / (bid_size + ask_size); missing at zero depth",
        capabilities=("order_book",),
        stage="morning_microstructure",
        inputs=("bid_size", "ask_size", "available_at"),
    ),
    _definition(
        "morning.micro_trade_frequency_per_minute",
        "morning_microstructure",
        "sum(trade_count, 0900..1130) / 150",
        unit="trades/minute",
        capabilities=("trade_frequency",),
        stage="morning_microstructure",
        inputs=("trade_count", "available_at"),
    ),
    _definition(
        "morning.micro_no_trade_seconds",
        "morning_microstructure",
        "seconds_since_last_trade at 11:30",
        unit="seconds",
        capabilities=("trade_frequency",),
        stage="morning_microstructure",
        inputs=("seconds_since_last_trade", "available_at"),
    ),
)

MORNING_FEATURE_REGISTRY = FeatureRegistry(
    (*MORNING_CORE_DEFINITIONS, *MORNING_MICROSTRUCTURE_DEFINITIONS)
)
MORNING_CORE_MANIFEST = MORNING_FEATURE_REGISTRY.manifest(
    "morning-core", "morning-core-v1", tuple(item.name for item in MORNING_CORE_DEFINITIONS)
)
MORNING_MICROSTRUCTURE_MANIFEST = MORNING_FEATURE_REGISTRY.manifest(
    "morning-microstructure",
    "morning-microstructure-v1",
    tuple(item.name for item in (*MORNING_CORE_DEFINITIONS, *MORNING_MICROSTRUCTURE_DEFINITIONS)),
)


def morning_feature_manifest(
    available_microstructure_features: Iterable[str] = (),
) -> FeatureSetManifest:
    """Create an exact versioned F13 or capability-specific F14 manifest."""

    requested = frozenset(available_microstructure_features)
    ordered_micro = tuple(
        item.name for item in MORNING_MICROSTRUCTURE_DEFINITIONS if item.name in requested
    )
    if requested != set(ordered_micro):
        unknown = sorted(requested - set(ordered_micro))
        raise ValueError(f"unknown morning microstructure features: {', '.join(unknown)}")
    if not ordered_micro:
        return MORNING_CORE_MANIFEST
    feature_names = (*MORNING_CORE_MANIFEST.feature_names, *ordered_micro)
    suffix = hashlib.sha256("\n".join(ordered_micro).encode()).hexdigest()[:12]
    return MORNING_FEATURE_REGISTRY.manifest(
        "morning-microstructure",
        f"morning-microstructure-v1-{suffix}",
        feature_names,
    )


def morning_feature_manifest_for_capabilities(
    report: MorningCapabilityReport,
) -> FeatureSetManifest:
    """Return the one exact versioned manifest implied by a capability report."""

    available, _ = _microstructure_status(report)
    return morning_feature_manifest(available)


@dataclass(frozen=True)
class MorningFeatureOutput:
    frame: pd.DataFrame
    manifest: FeatureSetManifest
    available_microstructure_features: tuple[str, ...]
    blocked_microstructure_features: tuple[str, ...]
    capability_report: MorningCapabilityReport


def build_morning_features(
    bars: pd.DataFrame,
    *,
    daily_context: pd.DataFrame,
    market_bars: pd.DataFrame,
    sector_bars: pd.DataFrame,
    capability_report: MorningCapabilityReport,
    freeze_metadata: Iterable[MorningFreezeMetadata],
) -> MorningFeatureOutput:
    """Build historical F13/F14 rows without using future sessions or post-freeze values."""

    capability_report.require("morning_ohlc", "intraday_bars", "intraday_volume_profile")
    if capability_report.provider is None:
        raise MorningDataError("BLOCKED_BY_DATA_CAPABILITY: morning provider is not configured")
    stock = validate_morning_bars(bars)
    market = validate_morning_bars(market_bars)
    sector = validate_morning_bars(sector_bars)
    context = _validate_daily_context(daily_context)
    metadata_by_date = _freeze_metadata_by_date(freeze_metadata)
    observed_dates = set(pd.to_datetime(stock["trading_date"]).dt.date)
    if observed_dates != set(metadata_by_date):
        raise MorningDataError("morning features require one freeze metadata record per session")
    _validate_freeze_roles(context, metadata_by_date)
    _require_microstructure_values(stock, capability_report)
    feature_rows: list[dict[str, object]] = []
    market_sessions = _session_map(market)
    sector_sessions = _session_map(sector)
    for local_date, metadata in metadata_by_date.items():
        session_date = pd.Timestamp(local_date)
        stock_date = stock.loc[stock["trading_date"].eq(session_date)]
        market_date = market.loc[market["trading_date"].eq(session_date)]
        sector_date = sector.loc[sector["trading_date"].eq(session_date)]
        _validate_session_lineage(
            stock_date,
            market_session=market_date,
            sector_session=sector_date,
            metadata=metadata,
            capability_report=capability_report,
        )
    context_map = {
        (str(row.symbol), _timestamp(row.trading_date)): row
        for row in context.itertuples(index=False)
    }
    stock_keys = {
        (str(symbol), _timestamp(trading_date))
        for symbol, trading_date in stock[["symbol", "trading_date"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if stock_keys != set(context_map):
        raise MorningDataError("daily context and morning bars must contain identical symbol-dates")
    for key, session in stock.groupby(["symbol", "trading_date"], sort=True):
        symbol, trading_date_raw = key
        trading_date = _timestamp(trading_date_raw)
        prior = context_map[(str(symbol), trading_date)]
        metadata = metadata_by_date[trading_date.date()]
        market_session = market_sessions.get(("TOPIX", trading_date))
        sector_session = sector_sessions.get((str(prior.sector), trading_date))
        if market_session is None or sector_session is None:
            raise MorningDataError(
                f"BLOCKED_BY_DATA_CAPABILITY: missing TOPIX/sector morning context for {key}"
            )
        feature_rows.append(
            _session_features(
                session,
                market_session=market_session,
                sector_session=sector_session,
                prior=prior,
                trading_date=trading_date,
                metadata=metadata,
            )
        )
    output = pd.DataFrame(feature_rows).sort_values(["trading_date", "symbol"], kind="stable")
    output = _add_historical_profiles(output, sessions=_PROFILE_SESSIONS)
    output["morning.monitored_volume_rank_pct"] = output.groupby("trading_date", sort=False)[
        "morning.cumulative_volume_1130"
    ].rank(method="average", pct=True)
    output["morning.candidate_volume_rank_pct"] = np.nan
    candidate_mask = output["morning.is_candidate"].eq(1.0)
    output.loc[candidate_mask, "morning.candidate_volume_rank_pct"] = (
        output.loc[candidate_mask]
        .groupby("trading_date", sort=False)["morning.cumulative_volume_1130"]
        .rank(method="average", pct=True)
    )
    available_micro, blocked_micro = _microstructure_status(capability_report)
    if missing_micro := sorted(set(available_micro) - set(output.columns)):
        raise MorningDataError(
            "BLOCKED_BY_DATA_CAPABILITY: declared morning microstructure fields are absent: "
            + ", ".join(missing_micro)
        )
    output = output.drop(
        columns=[name for name in blocked_micro if name in output.columns], errors="ignore"
    )
    expected_core = set(MORNING_CORE_MANIFEST.feature_names)
    if missing_core := sorted(expected_core - set(output.columns)):
        raise RuntimeError(f"morning core implementation omitted: {', '.join(missing_core)}")
    if not np.isfinite(output[["as_of"]].assign(value=1.0)["value"].to_numpy(dtype=float)).all():
        raise RuntimeError("morning feature identity unexpectedly became non-finite")
    return MorningFeatureOutput(
        frame=output.reset_index(drop=True),
        manifest=morning_feature_manifest(available_micro),
        available_microstructure_features=available_micro,
        blocked_microstructure_features=blocked_micro,
        capability_report=capability_report,
    )


def _validate_daily_context(frame: pd.DataFrame) -> pd.DataFrame:
    prior_columns = {
        "symbol",
        "trading_date",
        "sector",
        "prior_close",
        "average_daily_trading_value",
        "prior_expected_return_1d",
        "prior_expected_return_5d",
        "prior_expected_return_20d",
        "prior_downside_quantile",
        "prior_large_loss_probability",
        "prior_uncertainty",
        "prior_rank_pct",
        "prior_model_version",
        "prior_feature_version",
        "prior_data_snapshot_id",
        "prior_prediction_as_of",
        "is_current_holding",
        "is_candidate",
        "available_at",
    }
    missing = sorted(prior_columns - set(frame.columns))
    if missing:
        raise MorningDataError(f"daily morning context is missing columns: {', '.join(missing)}")
    output = frame.copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.normalize()
    if output.duplicated(["symbol", "trading_date"]).any():
        raise MorningDataError("daily morning context must be unique by symbol-date")
    for row in output.itertuples(index=False):
        available = _timestamp(row.available_at)
        if available.tzinfo is None or available.utcoffset() is None:
            raise MorningDataError("daily context available_at must be timezone-aware")
        freeze = pd.Timestamp(
            datetime.combine(_timestamp(row.trading_date).date(), MORNING_FREEZE, tzinfo=JST)
        )
        if available > freeze:
            raise MorningDataError("daily context was not available by the 11:30 freeze")
        prediction_as_of = _timestamp(row.prior_prediction_as_of)
        if prediction_as_of.tzinfo is None or prediction_as_of.utcoffset() is None:
            raise MorningDataError("prior prediction as_of must be timezone-aware")
        if prediction_as_of > freeze:
            raise MorningDataError("prior prediction was not frozen by the 11:30 cutoff")
        if not bool(row.is_current_holding) and not bool(row.is_candidate):
            raise MorningDataError("every morning context row must be a holding or candidate")
    numeric_columns = [
        "prior_close",
        "prior_expected_return_1d",
        "prior_expected_return_5d",
        "prior_expected_return_20d",
        "prior_downside_quantile",
        "prior_large_loss_probability",
        "prior_uncertainty",
        "prior_rank_pct",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(output[column], errors="coerce")
        if not values.map(math.isfinite).all():
            raise MorningDataError(f"daily context {column} must be finite")
        output[column] = values.astype(float)
    if (output["prior_close"] <= 0).any():
        raise MorningDataError("daily context prior_close must be positive")
    if (output["average_daily_trading_value"] <= 0).any():
        raise MorningDataError("daily context average_daily_trading_value must be positive")
    if not output["prior_large_loss_probability"].between(0.0, 1.0).all():
        raise MorningDataError("prior large-loss probability must be in [0, 1]")
    return output.sort_values(["trading_date", "symbol"], kind="stable").reset_index(drop=True)


def _session_map(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], pd.DataFrame]:
    return {
        (str(symbol), _timestamp(trading_date)): session.copy()
        for (symbol, trading_date), session in frame.groupby(["symbol", "trading_date"], sort=False)
    }


def _session_features(
    session: pd.DataFrame,
    *,
    market_session: pd.DataFrame,
    sector_session: pd.DataFrame,
    prior: Any,
    trading_date: pd.Timestamp,
    metadata: MorningFreezeMetadata,
) -> dict[str, object]:
    stock_prices = _cutoff_prices(session)
    market_prices = _cutoff_prices(market_session)
    sector_prices = _cutoff_prices(sector_session)
    open_price = stock_prices[time(9, 0)]
    close_price = stock_prices[MORNING_FREEZE]
    high = float(session["price"].max())
    low = float(session["price"].min())
    ordered_prices = session.sort_values("timestamp", kind="stable")["price"].to_numpy(dtype=float)
    log_returns = np.diff(np.log(ordered_prices))
    realized_vol = float(np.sqrt(np.sum(np.square(log_returns))))
    volume_sum = float(session["volume"].sum())
    trading_value_sum = float(session["trading_value"].sum())
    vwap = trading_value_sum / volume_sum if volume_sum > 0.0 else math.nan
    row: dict[str, object] = {
        "symbol": str(prior.symbol),
        "sector": str(prior.sector),
        "trading_date": trading_date,
        "as_of": datetime.combine(trading_date.date(), MORNING_FREEZE, tzinfo=JST),
        "available_at": max(
            [
                *(pd.Timestamp(value).to_pydatetime() for value in session["available_at"]),
                *(pd.Timestamp(value).to_pydatetime() for value in market_session["available_at"]),
                *(pd.Timestamp(value).to_pydatetime() for value in sector_session["available_at"]),
                pd.Timestamp(prior.available_at).to_pydatetime(),
            ]
        ),
        "source_record_ids": tuple(
            sorted(
                {
                    *(str(value) for value in session["source_record_id"]),
                    *(str(value) for value in market_session["source_record_id"]),
                    *(str(value) for value in sector_session["source_record_id"]),
                }
            )
        ),
        "source_snapshot_ids": tuple(metadata.source_snapshot_ids),
        "provider": metadata.provider,
        "reference_price_1130": close_price,
        "average_daily_trading_value": float(prior.average_daily_trading_value),
        "prior_model_version": str(prior.prior_model_version),
        "prior_feature_version": str(prior.prior_feature_version),
        "prior_data_snapshot_id": str(prior.prior_data_snapshot_id),
        "prior_prediction_as_of": pd.Timestamp(prior.prior_prediction_as_of).to_pydatetime(),
        "morning.gap_prev_close_to_0900": open_price / float(prior.prior_close) - 1.0,
        "morning.high": high,
        "morning.low": low,
        "morning.range_pct_open": (high - low) / open_price,
        "morning.realized_volatility": realized_vol,
        "morning.vwap": vwap,
        "morning.price_to_vwap": close_price / vwap - 1.0 if math.isfinite(vwap) else math.nan,
        "morning.close_location": (close_price - low) / (high - low) if high > low else 0.5,
        "morning.drop_from_high": close_price / high - 1.0,
        "morning.rebound_from_low": close_price / low - 1.0,
        "morning.is_current_holding": float(bool(prior.is_current_holding)),
        "morning.is_candidate": float(bool(prior.is_candidate)),
    }
    for name in (
        "expected_return_1d",
        "expected_return_5d",
        "expected_return_20d",
        "downside_quantile",
        "large_loss_probability",
        "uncertainty",
        "rank_pct",
    ):
        row[f"morning.prior_{name}"] = float(getattr(prior, f"prior_{name}"))
    for cutoff in _RETURN_CUTOFFS:
        suffix = _suffix(cutoff)
        stock_return = stock_prices[cutoff] / open_price - 1.0
        market_return = market_prices[cutoff] / market_prices[time(9, 0)] - 1.0
        sector_return = sector_prices[cutoff] / sector_prices[time(9, 0)] - 1.0
        row[f"morning.return_0900_{suffix}"] = stock_return
        row[f"morning.topix_relative_{suffix}"] = stock_return - market_return
        row[f"morning.sector_relative_{suffix}"] = stock_return - sector_return
        cutoff_mask = pd.Series(
            [_local_time(value) <= cutoff for value in session["timestamp"]],
            index=session.index,
        )
        row[f"morning.cumulative_volume_{suffix}"] = float(session.loc[cutoff_mask, "volume"].sum())
        row[f"_cumulative_trading_value_{suffix}"] = float(
            session.loc[cutoff_mask, "trading_value"].sum()
        )
    _add_microstructure_values(row, session)
    return row


def _cutoff_prices(session: pd.DataFrame) -> dict[time, float]:
    local_times = session["timestamp"].map(
        lambda value: pd.Timestamp(value).astimezone(JST).time().replace(tzinfo=None)
    )
    output: dict[time, float] = {}
    for cutoff in MORNING_CUTOFFS:
        exact = session.loc[local_times == cutoff, "price"]
        if len(exact) != 1:
            raise MorningDataError(
                f"BLOCKED_BY_DATA_CAPABILITY: exact {_suffix(cutoff)} bar is required"
            )
        output[cutoff] = float(exact.iloc[0])
    return output


def _add_historical_profiles(frame: pd.DataFrame, *, sessions: int) -> pd.DataFrame:
    output = frame.sort_values(["symbol", "trading_date"], kind="stable").copy()
    for cutoff in _RETURN_CUTOFFS:
        suffix = _suffix(cutoff)
        pairs = (
            (f"morning.cumulative_volume_{suffix}", f"morning.volume_progress_{suffix}_20d"),
            (
                f"_cumulative_trading_value_{suffix}",
                f"morning.trading_value_progress_{suffix}_20d",
            ),
        )
        for source, destination in pairs:
            prior_mean = output.groupby("symbol", sort=False)[source].transform(
                lambda values: values.shift(1).rolling(sessions, min_periods=sessions).mean()
            )
            output[destination] = output[source] / prior_mean.replace(0.0, np.nan)
    private = [name for name in output.columns if name.startswith("_cumulative_trading_value_")]
    return output.drop(columns=private)


def _add_microstructure_values(row: dict[str, object], session: pd.DataFrame) -> None:
    local_times = session["timestamp"].map(
        lambda value: pd.Timestamp(value).astimezone(JST).time().replace(tzinfo=None)
    )
    close = session.loc[local_times == MORNING_FREEZE].iloc[0]
    price = float(close["price"])
    if {"bid", "ask", "spread"} <= set(session.columns):
        bid, ask, spread = float(close["bid"]), float(close["ask"]), float(close["spread"])
        midpoint = (bid + ask) / 2.0
        row["morning.micro_spread_bps"] = spread / price * 10_000.0
        row["morning.micro_price_to_midpoint"] = (
            price / midpoint - 1.0 if midpoint > 0 else math.nan
        )
    if {"bid_size", "ask_size"} <= set(session.columns):
        bid_size, ask_size = float(close["bid_size"]), float(close["ask_size"])
        depth = bid_size + ask_size
        row["morning.micro_order_book_imbalance"] = (
            (bid_size - ask_size) / depth if depth > 0 else math.nan
        )
    if "trade_count" in session.columns:
        row["morning.micro_trade_frequency_per_minute"] = float(session["trade_count"].sum()) / 150
    if "seconds_since_last_trade" in session.columns:
        row["morning.micro_no_trade_seconds"] = float(close["seconds_since_last_trade"])


def _microstructure_status(
    report: MorningCapabilityReport,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requirements = MappingProxyType(
        {
            "morning.micro_spread_bps": "quotes",
            "morning.micro_price_to_midpoint": "quotes",
            "morning.micro_order_book_imbalance": "order_book",
            "morning.micro_trade_frequency_per_minute": "trade_frequency",
            "morning.micro_no_trade_seconds": "trade_frequency",
        }
    )
    available = tuple(
        name
        for name, capability in requirements.items()
        if report.capabilities.get(capability) is CapabilityStatus.AVAILABLE
    )
    blocked = tuple(name for name in requirements if name not in available)
    return available, blocked


def _freeze_metadata_by_date(
    values: Iterable[MorningFreezeMetadata],
) -> dict[date, MorningFreezeMetadata]:
    output: dict[date, MorningFreezeMetadata] = {}
    for metadata in values:
        local_date = metadata.as_of.astimezone(JST).date()
        if local_date in output:
            raise MorningDataError("morning freeze metadata dates must be unique")
        output[local_date] = metadata
    if not output:
        raise MorningDataError("morning freeze metadata cannot be empty")
    return output


def _validate_session_lineage(
    session: pd.DataFrame,
    *,
    market_session: pd.DataFrame,
    sector_session: pd.DataFrame,
    metadata: MorningFreezeMetadata,
    capability_report: MorningCapabilityReport,
) -> None:
    assert_morning_freeze_coverage(session, metadata)
    providers = {
        *(str(value) for value in session["provider"]),
        *(str(value) for value in market_session["provider"]),
        *(str(value) for value in sector_session["provider"]),
    }
    if providers != {metadata.provider} or capability_report.provider != metadata.provider:
        raise MorningDataError("morning provider lineage does not match the freeze metadata")
    if capability_report.model_dump(mode="json") != metadata.capability_report.model_dump(
        mode="json"
    ):
        raise MorningDataError("morning capability report does not match the freeze metadata")
    source_ids = {
        *(str(value) for value in session["source_record_id"]),
        *(str(value) for value in market_session["source_record_id"]),
        *(str(value) for value in sector_session["source_record_id"]),
    }
    if source_ids != set(metadata.source_record_ids):
        raise MorningDataError("morning source record lineage does not match the freeze metadata")


def _validate_freeze_roles(
    context: pd.DataFrame,
    metadata_by_date: Mapping[date, MorningFreezeMetadata],
) -> None:
    for local_date, metadata in metadata_by_date.items():
        session = context.loc[context["trading_date"].dt.date.eq(local_date)]
        expected = {member.symbol: member.role for member in metadata.universe}
        for row in session.itertuples(index=False):
            holding = bool(row.is_current_holding)
            candidate = bool(row.is_candidate)
            observed = (
                "HOLDING_AND_CANDIDATE"
                if holding and candidate
                else "HOLDING"
                if holding
                else "CANDIDATE"
                if candidate
                else "NEITHER"
            )
            if expected.get(str(row.symbol)) is None or expected[str(row.symbol)].value != observed:
                raise MorningDataError("morning context roles do not match the freeze metadata")


def _require_microstructure_values(
    frame: pd.DataFrame, report: MorningCapabilityReport
) -> None:
    requirements = {
        "quotes": ("bid", "ask", "spread", "quote_state"),
        "order_book": ("bid_size", "ask_size"),
        "trade_frequency": ("trade_count", "seconds_since_last_trade"),
    }
    for capability, columns in requirements.items():
        if report.capabilities.get(capability) is not CapabilityStatus.AVAILABLE:
            continue
        if missing := sorted(set(columns) - set(frame.columns)):
            raise MorningDataError(
                "BLOCKED_BY_DATA_CAPABILITY: declared morning microstructure fields are absent: "
                + ", ".join(missing)
            )
        for _, session in frame.groupby(["symbol", "trading_date"], sort=False):
            local_times = session["timestamp"].map(_local_time)
            close = session.loc[local_times == MORNING_FREEZE]
            if len(close) != 1:
                raise MorningDataError(
                    "BLOCKED_BY_DATA_CAPABILITY: exact 1130 microstructure row is required"
                )
            required_rows = session if capability == "trade_frequency" else close
            if required_rows.loc[:, list(columns)].isna().any(axis=None):
                raise MorningDataError(
                    "BLOCKED_BY_DATA_CAPABILITY: declared morning microstructure values are missing"
                )
            if capability == "quotes":
                quote = close.iloc[0]
                if float(quote["bid"]) <= 0 or float(quote["ask"]) <= 0:
                    raise MorningDataError("morning quote prices must be positive")
                if not str(quote["quote_state"]).strip():
                    raise MorningDataError("morning quote state cannot be blank")


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value)


def _local_time(value: Any) -> time:
    return _timestamp(value).astimezone(JST).time().replace(tzinfo=None)
