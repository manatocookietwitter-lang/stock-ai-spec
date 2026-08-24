from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from typer.testing import CliRunner

import stock_ai.cli as stock_cli
from stock_ai.cli import app
from stock_ai.data import DatasetName, HistoricalRevisionPolicy, SubscriptionPlan
from stock_ai.data.production import build_production_data
from stock_ai.decision import (
    CostPolicy,
    DailyPortfolioDecisionEngine,
    DecisionEngineConfig,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
)
from stock_ai.domain import (
    Account,
    AccountBucket,
    AccountType,
    CashState,
    PortfolioState,
    TaxState,
    WithholdingMode,
)
from stock_ai.features import V0_MANIFEST, V1_CORE_MANIFEST
from stock_ai.ml.production import (
    build_production_feature_sets,
    build_production_supervised_dataset,
    load_production_build_manifest,
    load_production_dataset_snapshot,
    load_production_feature_snapshot,
    run_production_walk_forward_baselines,
    write_production_baseline_report,
    write_production_build_manifest,
    write_production_dataset_snapshot,
    write_production_feature_snapshot,
)
from stock_ai.research import run_research_decision_e2e

JST = ZoneInfo("Asia/Tokyo")


class FrameCatalog:
    def __init__(self, frames: dict[DatasetName, pd.DataFrame]) -> None:
        self.frames = frames

    def point_in_time(self, dataset: DatasetName, _as_of: datetime) -> pd.DataFrame:
        return self.frames[dataset].copy()


class ContextFrameCatalog(FrameCatalog):
    def __enter__(self) -> ContextFrameCatalog:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_production_pipeline_is_point_in_time_and_uses_fixed_calendar_labels(
    tmp_path: Path,
) -> None:
    frames, dates = _source_frames()
    source_as_of = datetime(2026, 8, 24, tzinfo=UTC)
    bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(frames),
        source_snapshot_as_of=source_as_of,
        minimum_market_coverage=0.60,
    )

    historical_a = bundle.universe.loc[bundle.universe["symbol"] == "A"]
    assert not historical_a.empty
    assert historical_a["effective_date"].min() == dates[0]
    assert historical_a["delisting_date"].dropna().max() == dates[-10]
    assert "A" not in set(
        bundle.universe.loc[bundle.universe["effective_date"] == dates[-1], "symbol"]
    )

    b_financial = bundle.financials.loc[bundle.financials["symbol"] == "B"].sort_values(
        "trading_date"
    )
    assert pd.isna(b_financial.iloc[0]["financial_available_at"])
    assert pd.notna(b_financial.iloc[1]["financial_available_at"])
    assert (
        pd.to_datetime(bundle.financials["financial_available_at"], utc=True).dropna()
        <= pd.to_datetime(
            bundle.financials.loc[
                bundle.financials["financial_available_at"].notna(), "available_at"
            ],
            utc=True,
        )
    ).all()

    assert len(bundle.corporate_actions) == 1
    assert bundle.corporate_actions.iloc[0]["symbol"] == "A"
    assert bundle.corporate_actions.iloc[0]["ratio"] == 2.0

    feature_sets = build_production_feature_sets(bundle)
    assert set(V0_MANIFEST.feature_names) <= set(feature_sets.v0)
    assert set(V1_CORE_MANIFEST.feature_names) <= set(feature_sets.v1_core)
    assert len(feature_sets.v0) == len(feature_sets.v1_core)
    assert feature_sets.v1_core["shares_outstanding_is_missing"].any()

    dataset, label_1230_status = build_production_supervised_dataset(
        feature_sets.v1_core,
        bundle,
        plan=SubscriptionPlan.STANDARD,
    )
    assert label_1230_status.value == "BLOCKED_BY_DATA_CAPABILITY"
    assert not any(column.startswith("target_1230_return") for column in dataset)
    available_labels = dataset.loc[dataset["label_status_5d"] == "AVAILABLE"]
    label_as_of_dates = pd.to_datetime(available_labels["as_of"], utc=True).dt.tz_convert(JST)
    assert (
        pd.to_datetime(available_labels["label_entry_date"]).dt.date
        == label_as_of_dates.dt.date
    ).all()
    assert (label_as_of_dates.dt.hour == 11).all()
    assert (label_as_of_dates.dt.minute == 30).all()

    delisted = dataset.loc[
        (dataset["symbol"] == "A") & (dataset["trading_date"] == dates[-15])
    ].iloc[0]
    assert delisted["label_entry_date"] == dates[-14]
    assert delisted["label_end_date_5d"] == dates[-9]
    assert delisted["label_status_5d"] == "DELISTED_NO_EXIT_PRICE"
    assert pd.isna(delisted["target_return_5d"])

    suspended = dataset.loc[
        (dataset["symbol"] == "C") & (dataset["trading_date"] == dates[194])
    ].iloc[0]
    assert suspended["label_entry_date"] == dates[195]
    assert suspended["label_end_date_5d"] == dates[200]
    assert suspended["label_status_5d"] == "SUSPENDED_NO_EXIT_PRICE"
    assert pd.isna(suspended["target_return_5d"])

    snapshot = write_production_dataset_snapshot(
        dataset,
        tmp_path,
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
        created_at=datetime(2026, 8, 25, 0, 1, tzinfo=UTC),
        label_1230_status=label_1230_status,
    )
    repeated = write_production_dataset_snapshot(
        dataset,
        tmp_path,
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
        created_at=datetime(2026, 8, 25, 0, 1, tzinfo=UTC),
        label_1230_status=label_1230_status,
    )
    assert repeated.snapshot_id == snapshot.snapshot_id
    assert snapshot.parquet_path.is_file()
    assert snapshot.metadata_path.is_file()
    loaded_snapshot, loaded = load_production_dataset_snapshot(snapshot.parquet_path)
    assert loaded_snapshot.snapshot_id == snapshot.snapshot_id
    assert len(loaded) == snapshot.rows
    persisted = pd.read_parquet(snapshot.parquet_path)
    persisted_delisted = persisted.loc[
        (persisted["symbol"] == "A") & (persisted["trading_date"] == dates[-15])
    ].iloc[0]
    assert persisted_delisted["label_status_5d"] == "DELISTED_NO_EXIT_PRICE"
    feature_snapshot = write_production_feature_snapshot(
        feature_sets.v1_core,
        tmp_path / "features",
        manifest=V1_CORE_MANIFEST,
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
        created_at=datetime(2026, 8, 25, 0, 1, tzinfo=UTC),
    )
    assert feature_snapshot.parquet_path.is_file()
    loaded_feature_snapshot, loaded_features = load_production_feature_snapshot(
        feature_snapshot.parquet_path
    )
    assert loaded_feature_snapshot.snapshot_id == feature_snapshot.snapshot_id
    assert len(loaded_features) == feature_snapshot.rows
    v0_snapshot = write_production_feature_snapshot(
        feature_sets.v0,
        tmp_path / "features-v0",
        manifest=V0_MANIFEST,
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
        created_at=datetime(2026, 8, 25, 0, 1, tzinfo=UTC),
    )
    build_manifest = write_production_build_manifest(
        v0_snapshot,
        feature_snapshot,
        snapshot,
        tmp_path / "builds",
        created_at=datetime(2026, 8, 25, 0, 2, tzinfo=UTC),
    )
    assert load_production_build_manifest(build_manifest.manifest_path) == build_manifest

    renamed = dataset.rename(columns={"market.breadth": "market.breadth.renamed"})
    renamed_snapshot = write_production_dataset_snapshot(
        renamed,
        tmp_path,
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=datetime(2026, 8, 25, tzinfo=UTC),
        created_at=datetime(2026, 8, 25, 0, 1, tzinfo=UTC),
        label_1230_status=label_1230_status,
    )
    assert renamed_snapshot.snapshot_id != snapshot.snapshot_id

    with snapshot.parquet_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(RuntimeError, match="Parquet hash"):
        load_production_dataset_snapshot(snapshot.parquet_path)


def test_future_price_mutation_does_not_change_prior_production_features() -> None:
    frames, dates = _source_frames()
    bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(frames),
        source_snapshot_as_of=datetime(2026, 8, 24, tzinfo=UTC),
        minimum_market_coverage=0.60,
    )
    original = build_production_feature_sets(bundle).v1_core
    changed_daily = bundle.daily.copy()
    final_date = changed_daily["trading_date"].max()
    changed_daily.loc[changed_daily["trading_date"] == final_date, "adjusted_close"] *= 10
    changed = build_production_feature_sets(
        replace(bundle, daily=changed_daily)
    ).v1_core
    prior_columns = ["symbol", "trading_date", *V1_CORE_MANIFEST.feature_names]
    original_prior = original.loc[original["trading_date"] < final_date, prior_columns].reset_index(
        drop=True
    )
    changed_prior = changed.loc[changed["trading_date"] < final_date, prior_columns].reset_index(
        drop=True
    )
    pdt.assert_frame_equal(original_prior, changed_prior)
    assert dates[-1] > final_date


def test_production_momentum_and_ridge_walk_forward_keep_holdout_locked(
    tmp_path: Path,
) -> None:
    frames, _dates = _source_frames()
    source_as_of = datetime(2026, 8, 24, tzinfo=UTC)
    bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(frames),
        source_snapshot_as_of=source_as_of,
        minimum_market_coverage=0.60,
    )
    features = build_production_feature_sets(bundle).v1_core
    dataset, _status = build_production_supervised_dataset(
        features,
        bundle,
        plan=SubscriptionPlan.STANDARD,
    )
    report = run_production_walk_forward_baselines(
        dataset,
        data_snapshot_id="d" * 64,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        code_commit="test-commit",
        initial_train_periods=120,
        validation_periods=20,
        step_periods=20,
        holdout_periods=30,
    )
    assert [model.model_name for model in report.models] == ["CASH", "Momentum", "Ridge"]
    assert all(model.folds >= 1 for model in report.models)
    assert report.models[0].top_decile_mean_target == 0
    assert report.models[0].cost_scenarios == ((10, 0.0), (20, 0.0), (30, 0.0), (50, 0.0))
    assert report.locked_holdout_start
    assert not report.adoption_eligible
    assert report.historical_revision_status.value == "PARTIAL"
    path = write_production_baseline_report(report, tmp_path)
    assert write_production_baseline_report(report, tmp_path) == path
    assert path.is_file()

    mutated = dataset.copy()
    holdout_start = pd.Timestamp(report.locked_holdout_start)
    crossing = pd.to_datetime(mutated["label_end_date_5d"]) >= holdout_start
    mutated.loc[crossing, "target_return_5d"] = 999.0
    repeated_report = run_production_walk_forward_baselines(
        mutated,
        data_snapshot_id="d" * 64,
        created_at=datetime(2026, 8, 26, tzinfo=UTC),
        code_commit="test-commit",
        initial_train_periods=120,
        validation_periods=20,
        step_periods=20,
        holdout_periods=30,
    )
    assert repeated_report.report_id == report.report_id
    assert repeated_report.models == report.models


def test_real_data_research_predictions_reach_decision_engine_without_order_path() -> None:
    frames, _dates = _source_frames()
    source_as_of = datetime(2026, 8, 24, tzinfo=UTC)
    bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(frames),
        source_snapshot_as_of=source_as_of,
        minimum_market_coverage=0.60,
    )
    feature_sets = build_production_feature_sets(bundle)
    dataset, _status = build_production_supervised_dataset(
        feature_sets.v1_core,
        bundle,
        plan=SubscriptionPlan.STANDARD,
    )
    report = run_production_walk_forward_baselines(
        dataset,
        data_snapshot_id="d" * 64,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        code_commit="test-commit",
        initial_train_periods=120,
        validation_periods=20,
        step_periods=20,
        holdout_periods=30,
    )
    latest_date = feature_sets.v1_core["trading_date"].max()
    latest = feature_sets.v1_core.loc[
        feature_sets.v1_core["trading_date"] == latest_date
    ].copy()
    as_of = pd.Timestamp(latest["available_at"].max()).to_pydatetime()
    account = Account(account_id="research", broker="manual", display_name="Research")
    bucket = AccountBucket(
        bucket_id="research-taxable",
        account_id=account.account_id,
        account_type=AccountType.TAXABLE_SPECIFIED,
        withholding_mode=WithholdingMode.WITHHOLDING,
        fee_policy_id="research-cost-v1",
        tax_policy_id="research-tax-v1",
    )
    portfolio = PortfolioState(
        portfolio_id="research-cash",
        as_of=as_of,
        accounts=(account,),
        account_buckets=(bucket,),
        positions=(),
        cash=(CashState(account_bucket_id=bucket.bucket_id, available_cash=Decimal("1000000")),),
        tax_states=(TaxState(account_bucket_id=bucket.bucket_id, tax_year=as_of.year),),
    )
    engine = DailyPortfolioDecisionEngine(
        config=DecisionEngineConfig(
            maximum_positions=2,
            maximum_symbol_weight=Decimal("0.5"),
            maximum_sector_weight=Decimal("1"),
            minimum_cash_ratio=Decimal("0"),
            minimum_improvement_yen=Decimal("0"),
        ),
        cost_engine=TransactionCostEngine(
            CostPolicy(
                policy_id="research-cost-v1",
                version="research-cost-v1",
                zero_commission_confirmed=True,
                full_spread_bps=Decimal("10"),
                slippage_bps=Decimal("5"),
                impact_bps_at_full_adv=Decimal("10"),
            )
        ),
        tax_engine=SimpleJapanTaxEngine(
            TaxPolicy(
                policy_id="research-tax-v1",
                version="research-tax-v1",
                effective_from=as_of.date().replace(month=1, day=1),
            )
        ),
    )
    result = run_research_decision_e2e(
        dataset=dataset,
        latest_features=latest,
        universe=bundle.universe,
        data_snapshot_id="d" * 64,
        baseline_report=report,
        portfolio=portfolio,
        engine=engine,
        candidate_limit=2,
    )
    assert result.candidate_count == 2
    assert len(result.predictions) == 2
    assert result.proposal.current_portfolio_id == portfolio.portfolio_id
    assert result.proposal.model_bundle_version.startswith("ridge-real-v1-")
    assert "BLOCKED_BY_DATA_CAPABILITY" in result.reference_price_rule

    mutated = dataset.copy()
    holdout_start = pd.Timestamp(report.locked_holdout_start)
    for horizon in (1, 5, 20):
        crossing = pd.to_datetime(mutated[f"label_end_date_{horizon}d"]) >= holdout_start
        mutated.loc[crossing, f"target_return_{horizon}d"] = 999.0
    repeated = run_research_decision_e2e(
        dataset=mutated,
        latest_features=latest,
        universe=bundle.universe,
        data_snapshot_id="d" * 64,
        baseline_report=report,
        portfolio=portfolio,
        engine=engine,
        candidate_limit=2,
    )
    assert repeated.predictions == result.predictions
    mismatched = report.model_copy(update={"data_snapshot_id": "e" * 64})
    with pytest.raises(ValueError, match="dataset snapshot"):
        run_research_decision_e2e(
            dataset=dataset,
            latest_features=latest,
            universe=bundle.universe,
            data_snapshot_id="d" * 64,
            baseline_report=mismatched,
            portfolio=portfolio,
            engine=engine,
            candidate_limit=2,
        )


def test_revision_policy_and_missing_share_components_fail_closed() -> None:
    frames, _dates = _source_frames()
    source_as_of = datetime(2026, 8, 24, tzinfo=UTC)
    future_frames = {key: value.copy() for key, value in frames.items()}
    future_frames[DatasetName.DAILY_PRICES].loc[0, "available_at"] = pd.Timestamp(
        "2026-08-25T00:00:00Z"
    )
    with pytest.raises(ValueError, match="after source_snapshot_as_of"):
        build_production_data(  # type: ignore[arg-type]
            FrameCatalog(future_frames),
            source_snapshot_as_of=source_as_of,
            minimum_market_coverage=0.60,
        )

    missing_component = {key: value.copy() for key, value in frames.items()}
    missing_component[DatasetName.FINANCIAL_SUMMARY].loc[
        missing_component[DatasetName.FINANCIAL_SUMMARY]["symbol"] == "B",
        "treasury_shares_fy",
    ] = np.nan
    component_bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(missing_component),
        source_snapshot_as_of=source_as_of,
        minimum_market_coverage=0.60,
    )
    b_daily = component_bundle.daily.loc[component_bundle.daily["symbol"] == "B"]
    assert b_daily["shares_outstanding"].isna().all()
    assert set(b_daily["shares_outstanding_missing_reason"].dropna()) == {
        "fiscal treasury shares is missing"
    }

    strict_frames = {key: value.copy() for key, value in frames.items()}
    for dataset, frame in strict_frames.items():
        if dataset is not DatasetName.SECURITY_MASTER:
            frame["available_at"] = pd.Timestamp("2026-08-23T00:00:00Z")
    strict_bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(strict_frames),
        source_snapshot_as_of=source_as_of,
        minimum_market_coverage=0.60,
        revision_policy=HistoricalRevisionPolicy.STRICT_AS_KNOWN,
    )
    strict_features = build_production_feature_sets(strict_bundle).v1_core
    assert (
        pd.to_datetime(strict_features["source_revision_available_at"], utc=True)
        <= pd.to_datetime(strict_features["available_at"], utc=True)
    ).all()
    strict_dataset, _ = build_production_supervised_dataset(
        strict_features,
        strict_bundle,
        plan=SubscriptionPlan.STANDARD,
    )
    assert strict_dataset["target_return_5d"].isna().all()
    assert set(strict_dataset["label_status_5d"]) == {"BLOCKED_BY_REVISION_HISTORY"}


def test_premium_1230_labels_are_partial_and_snapshot_maturity_is_enforced(
    tmp_path: Path,
) -> None:
    frames, dates = _source_frames()
    price_frame = frames[DatasetName.DAILY_PRICES].copy()
    price_frame["research_afternoon_open"] = price_frame["research_close"] * 1.001
    price_frame.loc[price_frame.index[0], "research_afternoon_open"] = np.nan
    frames[DatasetName.DAILY_PRICES] = price_frame
    source_as_of = datetime(2026, 8, 24, tzinfo=UTC)
    bundle = build_production_data(  # type: ignore[arg-type]
        FrameCatalog(frames),
        source_snapshot_as_of=source_as_of,
        minimum_market_coverage=0.60,
    )
    features = build_production_feature_sets(bundle).v1_core
    dataset, status = build_production_supervised_dataset(
        features,
        bundle,
        plan=SubscriptionPlan.PREMIUM,
    )
    assert status.value == "PARTIAL"
    assert "label_1230_available_at_5d" in dataset
    cutoff_row = dataset.loc[
        (dataset["symbol"] == "B") & (dataset["trading_date"] == dates[99])
    ].iloc[0]
    cutoff = pd.Timestamp(cutoff_row["as_of"]).to_pydatetime()
    assert cutoff_row["label_1230_available_at_5d"] > pd.Timestamp(cutoff)
    snapshot = write_production_dataset_snapshot(
        dataset,
        tmp_path,
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=cutoff,
        created_at=datetime(2026, 8, 25, tzinfo=UTC),
        label_1230_status=status,
    )
    persisted = pd.read_parquet(snapshot.parquet_path)
    stored = persisted.loc[
        (persisted["symbol"] == "B") & (persisted["trading_date"] == dates[99])
    ].iloc[0]
    assert pd.isna(stored["target_1230_return_5d"])
    assert stored["label_1230_status_5d"] == "HORIZON_NOT_MATURE"


def test_production_build_orchestration_publishes_one_verifiable_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames, _dates = _source_frames()
    monkeypatch.setattr(
        stock_cli,
        "DuckDBCatalog",
        lambda _path: ContextFrameCatalog(frames),
    )
    source_as_of = datetime(2026, 8, 24, tzinfo=UTC)
    artifacts = stock_cli._build_production_artifacts(
        data_root=tmp_path,
        catalog_path=tmp_path / "unused.duckdb",
        source_snapshot_as_of=source_as_of,
        plan=SubscriptionPlan.STANDARD,
        minimum_market_coverage=0.60,
        revision_policy=HistoricalRevisionPolicy.SINGLE_VINTAGE_AS_REVISED,
    )
    assert artifacts.snapshot.source_snapshot_ids == artifacts.bundle.source_snapshot_ids
    assert len(tuple(tmp_path.glob("builds/production/*/*.json"))) == 1

    verified = CliRunner().invoke(app, ["data", "verify", "--data-root", str(tmp_path)])
    assert verified.exit_code == 0
    assert "feature_snapshots=2 dataset_snapshots=1 builds=1 status=OK" in verified.stdout


def _source_frames() -> tuple[dict[DatasetName, pd.DataFrame], pd.DatetimeIndex]:
    dates = pd.bdate_range("2025-01-06", periods=280)
    revision_at = pd.Timestamp("2026-08-24T00:00:00Z")
    master_rows: list[dict[str, object]] = []
    price_rows: list[dict[str, object]] = []
    topix_rows: list[dict[str, object]] = []
    for position, day in enumerate(dates):
        symbols = ("B", "C") if position >= len(dates) - 10 else ("A", "B", "C")
        for symbol in symbols:
            sector = "10" if symbol in ("A", "B") else "20"
            master_rows.append(
                {
                    "effective_date": day,
                    "provider_code": f"{symbol}0000",
                    "symbol": symbol,
                    "company_name": f"Company {symbol}",
                    "sector_33_code": sector,
                    "sector_33_name": f"Sector {sector}",
                    "market_code": "0111",
                    "market_name": "Prime",
                    "available_at": revision_at,
                }
            )
            if symbol == "C" and position == 200:
                continue
            base = 1000 + position * (1 + ord(symbol) - ord("A"))
            adjustment = 0.5 if symbol == "A" and position == 100 else 1.0
            raw_close = base * 2 if adjustment == 0.5 else base
            price_rows.append(
                {
                    "trading_date": day,
                    "provider_code": f"{symbol}0000",
                    "symbol": symbol,
                    "raw_close": raw_close,
                    "trading_value": base * 100_000,
                    "adjustment_factor": adjustment,
                    "research_high": base * 1.01,
                    "research_low": base * 0.99,
                    "research_close": float(base),
                    "research_volume": 100_000.0,
                    "research_afternoon_open": np.nan,
                    "adjustment_version": "adjustment-v1",
                    "available_at": revision_at,
                }
            )
        topix = 2500 + position * 2
        topix_rows.append(
            {
                "trading_date": day,
                "open": topix - 1,
                "high": topix + 3,
                "low": topix - 3,
                "close": float(topix),
                "available_at": revision_at,
            }
        )

    financial_rows = []
    for symbol in ("A", "B", "C"):
        announcement_day = dates[1] if symbol == "B" else dates[0]
        announcement_hour = 15 if symbol == "B" else 9
        financial_rows.append(
            {
                "symbol": symbol,
                "disclosure_number": f"disc-{symbol}",
                "announced_at": pd.Timestamp(
                    datetime.combine(
                        announcement_day.date(),
                        datetime.min.time().replace(hour=announcement_hour),
                        JST,
                    )
                ).tz_convert("UTC"),
                "period_type": "FY",
                "period_end": pd.Timestamp("2024-12-31"),
                "sales": 1_000_000.0,
                "operating_profit": 100_000.0,
                "eps": 100.0,
                "equity": 5_000_000.0,
                "shares_outstanding_fy": 100_000.0,
                "treasury_shares_fy": 0.0,
                "forecast_eps": 110.0,
                "bps": 50.0,
                "provider_roe": 0.10,
                "available_at": revision_at,
            }
        )
    calendar = pd.DataFrame(
        {
            "trading_date": dates,
            "is_equity_business_day": True,
            "available_at": revision_at,
        }
    )
    frames = {
        DatasetName.SECURITY_MASTER: pd.DataFrame(master_rows),
        DatasetName.DAILY_PRICES: pd.DataFrame(price_rows),
        DatasetName.TRADING_CALENDAR: calendar,
        DatasetName.TOPIX: pd.DataFrame(topix_rows),
        DatasetName.FINANCIAL_SUMMARY: pd.DataFrame(financial_rows),
    }
    return frames, dates
