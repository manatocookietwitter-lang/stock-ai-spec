"""CLI for an explicit deterministic fixture end-to-end demonstration."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd
import typer

from stock_ai.data import (
    DatasetName,
    DuckDBCatalog,
    ImmutableParquetStore,
    JQuantsError,
    JQuantsV2Client,
    JQuantsV2Config,
    JQuantsV2Ingestor,
    StorageIntegrityError,
    SubscriptionPlan,
    capabilities_for,
)
from stock_ai.decision import (
    CostPolicy,
    DailyPortfolioDecisionEngine,
    DecisionCandidate,
    DecisionEngineConfig,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
    apply_executions,
)
from stock_ai.domain import (
    ExecutionRecord,
    ExecutionStatus,
    Prediction,
    PredictionUncertainty,
    Security,
    TradeSide,
    UserDecision,
    UserDecisionLine,
)
from stock_ai.features import V1_CORE_MANIFEST, FeatureEngine
from stock_ai.fixtures import market_fixture, next_business_morning, portfolio_fixture
from stock_ai.ml import (
    BaselinePredictionBundle,
    PurgedExpandingWindowSplitter,
    RidgeRegressor,
    build_supervised_dataset,
    reserve_locked_final_holdout,
    walk_forward_validate,
    write_dataset_snapshot,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Japanese-stock decision-support research commands (never submits orders).",
)
data_app = typer.Typer(
    no_args_is_help=True,
    help="Acquire, verify, and inspect immutable J-Quants V2 data.",
)
app.add_typer(data_app, name="data")


@app.callback()
def main() -> None:
    """Run explicit research and fixture-only decision-support workflows."""


def _decimal(value: float) -> Decimal:
    return Decimal(str(round(value, 4)))


@data_app.command("capabilities")
def data_capabilities(
    plan: Annotated[
        SubscriptionPlan,
        typer.Option(help="Declared J-Quants subscription plan; no API call is made."),
    ] = SubscriptionPlan.FREE,
) -> None:
    """Print the fail-closed Goal 2A source-capability table."""

    typer.echo("capability status minimum_plan endpoint reason")
    for item in capabilities_for(plan):
        minimum_plan = "-" if item.minimum_plan is None else item.minimum_plan.value
        endpoint = item.source_endpoint or "-"
        reason = item.reason or "-"
        typer.echo(f"{item.name} {item.status.value} {minimum_plan} {endpoint} {reason}")


@data_app.command("sync")
def data_sync(
    source_date: Annotated[
        str,
        typer.Option("--date", help="Provider source date (YYYY-MM-DD)."),
    ],
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing immutable raw/normalized object directories."),
    ] = Path("data"),
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="DuckDB catalog path; defaults below data-root."),
    ] = None,
    plan: Annotated[
        SubscriptionPlan,
        typer.Option(help="Declared subscription plan used for throttling and fail-closed access."),
    ] = SubscriptionPlan.FREE,
    datasets: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated datasets: security_master,daily_prices,"
                "financial_summary,trading_calendar,topix."
            )
        ),
    ] = "security_master,daily_prices,financial_summary",
) -> None:
    """Fetch one date from J-Quants V2; never falls back to fixture data."""

    selected = _parse_datasets(datasets)
    try:
        source_day = date.fromisoformat(source_date)
    except ValueError as exc:
        raise typer.BadParameter("--date must use YYYY-MM-DD") from exc
    target_catalog = catalog_path or data_root / "catalog.duckdb"
    try:
        with (
            JQuantsV2Client.from_env(config=JQuantsV2Config(plan=plan)) as client,
            DuckDBCatalog(target_catalog) as catalog,
        ):
            result = JQuantsV2Ingestor(
                client=client,
                store=ImmutableParquetStore(data_root),
                catalog=catalog,
            ).sync_date(source_day, datasets=selected)
    except JQuantsError as exc:
        typer.echo(f"ingestion blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"run={result.ingestion_run_id} status={result.status.value} "
        f"source_date={result.source_date.isoformat()} objects={len(result.objects)}"
    )
    typer.echo("No fixture fallback was used. No securities order was submitted.")


@data_app.command("verify")
def data_verify(
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing immutable raw/normalized object directories."),
    ] = Path("data"),
) -> None:
    """Verify every published object manifest and Parquet hash."""

    store = ImmutableParquetStore(data_root)
    manifests = sorted(data_root.glob("*/jquants_v2/*/source_date=*/*/manifest.json"))
    verified = 0
    try:
        for manifest in manifests:
            store.verify(manifest.parent)
            verified += 1
    except StorageIntegrityError as exc:
        typer.echo(f"verification failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(f"verified_objects={verified} status=OK")


def _parse_datasets(value: str) -> tuple[DatasetName, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise typer.BadParameter("at least one dataset is required")
    try:
        selected = tuple(DatasetName(name) for name in names)
    except ValueError as exc:
        allowed = ",".join(item.value for item in DatasetName)
        raise typer.BadParameter(f"unknown dataset; allowed values: {allowed}") from exc
    return tuple(dict.fromkeys(selected))


@app.command("fixture-demo")
def fixture_demo(
    snapshot_dir: Annotated[
        Path,
        typer.Option(help="Directory for the immutable fixture dataset snapshot."),
    ] = Path(".demo-artifacts/datasets"),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit the final proposal as JSON."),
    ] = False,
) -> None:
    """Run fixture -> features -> Ridge -> costs/tax -> proposal -> next state."""
    daily, market, sectors, financials = market_fixture()
    features = FeatureEngine(V1_CORE_MANIFEST).transform(
        daily, market, sectors, financials=financials
    )
    dataset = build_supervised_dataset(features)
    last_date = pd.Timestamp(features["trading_date"].max())
    as_of = next_business_morning(last_date)
    snapshot = write_dataset_snapshot(
        dataset,
        snapshot_dir,
        manifest=V1_CORE_MANIFEST,
        as_of=as_of,
        created_at=as_of + timedelta(minutes=1),
    )

    feature_names = V1_CORE_MANIFEST.feature_names
    locked_holdout = reserve_locked_final_holdout(dataset, holdout_periods=20)
    development = dataset.iloc[list(locked_holdout.development_indices)].copy()
    training = development.copy()
    bundle = BaselinePredictionBundle(feature_names, alpha=5.0).fit(training)
    latest = features.loc[features["trading_date"] == last_date].copy()
    predicted = bundle.predict(latest).set_index("symbol")
    splitter = PurgedExpandingWindowSplitter(
        initial_train_periods=220,
        validation_periods=30,
        step_periods=30,
        purge_periods=5,
        embargo_periods=5,
        label_horizon_periods=5,
    )
    validation = walk_forward_validate(
        development.dropna(subset=["target_return_5d"]),
        feature_names=feature_names,
        target_column="target_return_5d",
        label_end_column="label_end_date_5d",
        splitter=splitter,
        model_factory=lambda: RidgeRegressor(alpha=5.0),
    )

    price_lookup: dict[str, Decimal] = {}
    for _, latest_row in latest.iterrows():
        latest_symbol = str(latest_row["symbol"])
        price_lookup[latest_symbol] = _decimal(cast(float, latest_row["close"]))
    portfolio = portfolio_fixture(as_of, price_lookup)
    security_data = {
        "7203": ("Toyota", "Transport Equipment"),
        "6758": ("Sony Group", "Electric Appliances"),
        "8306": ("MUFG", "Banks"),
        "9432": ("NTT", "Information & Communication"),
    }
    predictions: dict[str, Prediction] = {}
    for symbol_value, row in predicted.iterrows():
        symbol = str(symbol_value)
        one = float(np.clip(cast(float, row["prediction_1d"]), -0.04, 0.04))
        five = float(np.clip(cast(float, row["prediction_5d"]), -0.08, 0.08))
        twenty = float(np.clip(cast(float, row["prediction_20d"]), -0.12, 0.12))
        standard_error = float(min(0.02, max(0.001, cast(float, row["uncertainty_5d"]) * 0.25)))
        predictions[symbol] = Prediction(
            symbol=symbol,
            as_of=as_of,
            expected_return_1d=one,
            expected_return_5d=five,
            expected_return_20d=twenty,
            downside_quantile=min(-0.005, five - 0.025),
            large_loss_probability=0.08,
            uncertainty=PredictionUncertainty(standard_error=standard_error),
            model_version="ridge-fixture-v1",
            feature_version=V1_CORE_MANIFEST.feature_set_version,
            data_snapshot_id=snapshot.snapshot_id,
        )

    candidates: list[DecisionCandidate] = []
    candidate_buckets = {
        ("7203", "sbi-taxable"),
        ("7203", "sbi-nisa"),
        ("9432", "sbi-taxable"),
        ("6758", "sbi-nisa"),
        ("8306", "sbi-taxable"),
    }
    latest_by_symbol = latest.set_index("symbol")
    for symbol, bucket in sorted(candidate_buckets):
        name, sector = security_data[symbol]
        candidates.append(
            DecisionCandidate(
                security=Security(symbol=symbol, company_name=name, sector=sector),
                account_bucket_id=bucket,
                price=price_lookup[symbol],
                average_daily_trading_value=_decimal(
                    cast(
                        float,
                        latest_by_symbol.loc[symbol, "liquidity.trading_value_mean_20d"],
                    )
                ),
                prediction=predictions[symbol],
            )
        )

    cost_policy = CostPolicy(
        policy_id="fixture-cost-v1",
        version="fixture-cost-v1",
        commission_fixed=Decimal("80"),
        full_spread_bps=Decimal("8"),
        slippage_bps=Decimal("4"),
        impact_bps_at_full_adv=Decimal("20"),
    )
    tax_policy = TaxPolicy(
        policy_id="fixture-tax-v1",
        version="fixture-tax-v1",
        effective_from=date(2025, 1, 1),
    )
    engine = DailyPortfolioDecisionEngine(
        config=DecisionEngineConfig(
            maximum_symbol_weight=Decimal("0.50"),
            maximum_sector_weight=Decimal("0.70"),
            minimum_cash_ratio=Decimal("0.10"),
            maximum_turnover_ratio=Decimal("0.80"),
            minimum_improvement_yen=Decimal("300"),
            uncertainty_penalty_weight=Decimal("0.30"),
        ),
        cost_engine=TransactionCostEngine(cost_policy),
        tax_engine=SimpleJapanTaxEngine(tax_policy),
    )
    proposal = engine.propose(
        portfolio=portfolio,
        candidates=tuple(candidates),
        generated_at=as_of + timedelta(minutes=6),
        model_bundle_version="ridge-fixture-v1",
    )

    decision = UserDecision(
        decision_id="fixture-decision-v1",
        proposal_id=proposal.proposal_id,
        version=1,
        saved_at=as_of + timedelta(minutes=12),
        lines=tuple(
            UserDecisionLine(
                proposal_line_id=line.line_id,
                selected_target_shares=line.recommended_shares,
            )
            for line in proposal.lines
        ),
    )
    changed = next((line for line in proposal.lines if line.share_difference != 0), None)
    executions: tuple[ExecutionRecord, ...] = ()
    if changed is not None:
        fill_shares = min(100, abs(changed.share_difference))
        side = TradeSide.BUY if changed.share_difference > 0 else TradeSide.SELL
        executions = (
            ExecutionRecord(
                execution_id="fixture-manual-fill-v1",
                decision_id=decision.decision_id,
                executed_at=as_of.replace(hour=12, minute=31),
                symbol=changed.symbol,
                account_bucket_id=changed.account_bucket_id,
                status=ExecutionStatus.FILLED,
                side=side,
                ordered_shares=fill_shares,
                filled_shares=fill_shares,
                average_fill_price=changed.reference_price,
                actual_commission=Decimal("80"),
                source="manual",
            ),
        )
    next_state = apply_executions(
        portfolio,
        executions,
        next_as_of=as_of + timedelta(days=1),
        next_portfolio_id="fixture-next-day",
    )

    if json_output:
        typer.echo(proposal.model_dump_json(indent=2))
        return
    typer.echo("DETERMINISTIC FIXTURE ONLY - never production fallback, never an order")
    typer.echo(
        f"snapshot={snapshot.snapshot_id[:12]} features={len(feature_names)} "
        f"rows={snapshot.rows} validation_folds={len(validation)} "
        f"locked_holdout_start={locked_holdout.holdout_start.date()}"
    )
    typer.echo("baseline predictions (fixture research proxy): symbol p1d p5d p20d")
    for symbol_value, row in predicted.sort_index().iterrows():
        typer.echo(
            f"{symbol_value} {row['prediction_1d']:.6f} "
            f"{row['prediction_5d']:.6f} {row['prediction_20d']:.6f}"
        )
    typer.echo("symbol bucket current recommended action cost tax")
    for line in proposal.lines:
        typer.echo(
            f"{line.symbol} {line.account_bucket_id} {line.current_shares} "
            f"{line.recommended_shares} {line.action.value} "
            f"{line.transaction_cost.total:.2f} {line.estimated_tax_effect:.2f}"
        )
    typer.echo(
        f"net_improvement={proposal.net_improvement:.2f} "
        f"manual_executions={len(executions)} next_positions={len(next_state.positions)}"
    )
    typer.echo("No securities order was submitted. The execution record is a local fixture record.")


if __name__ == "__main__":
    app()
