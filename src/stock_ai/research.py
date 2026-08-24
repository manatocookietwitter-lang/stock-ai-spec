"""Research-only bridge from real-data Ridge predictions to the decision engine."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from statistics import NormalDist

import numpy as np
import pandas as pd

from stock_ai.decision import DailyPortfolioDecisionEngine, DecisionCandidate
from stock_ai.domain import (
    PortfolioProposal,
    PortfolioState,
    Prediction,
    PredictionUncertainty,
    Security,
)
from stock_ai.features import V1_CORE_MANIFEST
from stock_ai.ml.models import RidgeRegressor
from stock_ai.ml.production import ProductionBaselineReport


@dataclass(frozen=True)
class ResearchDecisionResult:
    proposal: PortfolioProposal
    predictions: tuple[Prediction, ...]
    candidate_count: int
    reference_price_rule: str


def run_research_decision_e2e(
    *,
    dataset: pd.DataFrame,
    latest_features: pd.DataFrame,
    universe: pd.DataFrame,
    data_snapshot_id: str,
    baseline_report: ProductionBaselineReport,
    portfolio: PortfolioState,
    engine: DailyPortfolioDecisionEngine,
    candidate_limit: int = 8,
) -> ResearchDecisionResult:
    """Generate a paper proposal; never create an order or execution record."""

    if candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")
    if baseline_report.data_snapshot_id != data_snapshot_id:
        raise ValueError("baseline report and dataset snapshot provenance do not match")
    if baseline_report.feature_set_version != V1_CORE_MANIFEST.feature_set_version:
        raise ValueError("baseline report uses a different feature-set version")
    if baseline_report.target_column != "target_return_5d":
        raise ValueError("research E2E requires the audited 5d absolute-return baseline")
    feature_names = V1_CORE_MANIFEST.feature_names
    required = {
        "symbol",
        "trading_date",
        "available_at",
        "close",
        "liquidity.trading_value_mean_20d",
        *feature_names,
    }
    _require_columns(latest_features, required, "latest production features")
    latest = latest_features.copy()
    latest["available_at"] = pd.to_datetime(latest["available_at"], utc=True)
    portfolio_as_of = pd.Timestamp(portfolio.as_of).tz_convert("UTC")
    if (latest["available_at"] > portfolio_as_of).any():
        raise ValueError("latest feature observation is after the portfolio as_of")
    latest_date = pd.to_datetime(latest["trading_date"]).max().normalize()
    latest = latest.loc[pd.to_datetime(latest["trading_date"]).dt.normalize() == latest_date].copy()
    if latest.empty:
        raise ValueError("latest production feature snapshot is empty")

    training = dataset.loc[
        pd.to_datetime(dataset["trading_date"])
        < pd.Timestamp(baseline_report.locked_holdout_start)
    ].copy()
    holdout_start = pd.Timestamp(baseline_report.locked_holdout_start)
    predictions = latest[["symbol"]].copy()
    for horizon in (1, 5, 20):
        target = f"target_return_{horizon}d"
        status = f"label_status_{horizon}d"
        label_end = f"label_end_date_{horizon}d"
        _require_columns(
            training,
            {target, status, label_end, *feature_names},
            "production training data",
        )
        valid = (
            training[target].notna()
            & training[status].eq("AVAILABLE")
            & (pd.to_datetime(training[label_end]) < holdout_start)
        )
        if not valid.any():
            raise ValueError(f"BLOCKED_BY_VALIDATION: no training target for {horizon}d")
        model = RidgeRegressor(alpha=5.0).fit(
            training.loc[valid, list(feature_names)],
            training.loc[valid, target],
        )
        predictions[f"prediction_{horizon}d"] = model.predict(latest[list(feature_names)])

    ridge_summary = next(
        (model for model in baseline_report.models if model.model_name == "Ridge"),
        None,
    )
    if ridge_summary is None:
        raise ValueError("baseline report does not contain Ridge validation metrics")
    standard_error = max(float(np.sqrt(ridge_summary.mean_squared_error)), 1e-6)
    bundle_version = "ridge-real-v1-" + hashlib.sha256(
        f"{data_snapshot_id}|{baseline_report.report_id}".encode()
    ).hexdigest()[:16]
    prediction_models: list[Prediction] = []
    for _, row in predictions.iterrows():
        expected_5d = float(row["prediction_5d"])
        probability = NormalDist(mu=expected_5d, sigma=standard_error).cdf(-0.10)
        prediction_models.append(
            Prediction(
                symbol=str(row["symbol"]),
                as_of=portfolio.as_of,
                expected_return_1d=float(row["prediction_1d"]),
                expected_return_5d=expected_5d,
                expected_return_20d=float(row["prediction_20d"]),
                downside_quantile=expected_5d - 1.645 * standard_error,
                large_loss_probability=float(min(1.0, max(0.0, probability))),
                uncertainty=PredictionUncertainty(
                    standard_error=standard_error,
                    coverage_warning=(
                        "OOS 5d fold RMSE is reused for all horizons in the Goal 2 baseline"
                    ),
                ),
                model_version=bundle_version,
                feature_version=V1_CORE_MANIFEST.feature_set_version,
                data_snapshot_id=data_snapshot_id,
            )
        )
    prediction_by_symbol = {prediction.symbol: prediction for prediction in prediction_models}

    latest = latest.merge(predictions, on="symbol", how="inner", validate="one_to_one")
    held_symbols = {position.symbol for position in portfolio.positions}
    missing_holdings = held_symbols - set(latest["symbol"].astype(str))
    if missing_holdings:
        raise ValueError(
            "BLOCKED_BY_DATA_CAPABILITY: holdings lack current production features: "
            + ", ".join(sorted(missing_holdings))
        )
    ranked = latest.sort_values(
        ["prediction_5d", "liquidity.trading_value_mean_20d", "symbol"],
        ascending=[False, False, True],
    )
    selected_symbols = set(ranked.head(candidate_limit)["symbol"].astype(str)) | held_symbols
    selected = ranked.loc[ranked["symbol"].astype(str).isin(selected_symbols)].copy()

    universe_latest = universe.loc[
        pd.to_datetime(universe["effective_date"]).dt.normalize() == latest_date
    ].set_index("symbol")
    bucket_ids = tuple(bucket.bucket_id for bucket in portfolio.account_buckets)
    if len(bucket_ids) != 1:
        raise ValueError("research E2E currently requires exactly one explicit account bucket")
    bucket_id = bucket_ids[0]
    candidates: list[DecisionCandidate] = []
    for _, row in selected.iterrows():
        symbol = str(row["symbol"])
        if symbol not in universe_latest.index:
            raise ValueError(f"selected symbol {symbol} is outside the PIT universe")
        security_row = universe_latest.loc[symbol]
        price = _decimal(row["close"])
        trading_value = pd.to_numeric(
            pd.Series([row["liquidity.trading_value_mean_20d"]]), errors="coerce"
        ).iloc[0]
        candidates.append(
            DecisionCandidate(
                security=Security(
                    symbol=symbol,
                    company_name=str(security_row.get("company_name", symbol)),
                    sector=str(security_row["sector_33_code"]),
                    market_segment=str(security_row.get("market_name", "TSE")),
                ),
                account_bucket_id=bucket_id,
                price=price,
                average_daily_trading_value=(
                    _decimal(trading_value)
                    if pd.notna(trading_value) and trading_value > 0
                    else None
                ),
                prediction=prediction_by_symbol[symbol],
            )
        )
    proposal = engine.propose(
        portfolio=portfolio,
        candidates=tuple(candidates),
        generated_at=portfolio.as_of,
        model_bundle_version=bundle_version,
    )
    return ResearchDecisionResult(
        proposal=proposal,
        predictions=tuple(sorted(prediction_models, key=lambda item: item.symbol)),
        candidate_count=len(candidates),
        reference_price_rule=(
            "previous completed raw close research proxy; exact 12:30 reference is "
            "BLOCKED_BY_DATA_CAPABILITY"
        ),
    )


def _decimal(value: object) -> Decimal:
    numeric = float(str(value))
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("decision reference price/value must be finite and positive")
    return Decimal(str(round(numeric, 4)))


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")
