"""CLI for an explicit deterministic fixture end-to-end demonstration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import typer

from stock_ai.data import (
    DataQualityError,
    DatasetName,
    DuckDBCatalog,
    HistoricalRevisionPolicy,
    ImmutableParquetStore,
    JQuantsError,
    JQuantsV2Client,
    JQuantsV2Config,
    JQuantsV2HistoryIngestor,
    JQuantsV2Ingestor,
    ProductionDataBundle,
    StorageIntegrityError,
    SubscriptionPlan,
    build_production_data,
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
    Account,
    AccountBucket,
    AccountType,
    CashState,
    ExecutionRecord,
    ExecutionStatus,
    PortfolioState,
    Prediction,
    PredictionUncertainty,
    Security,
    TaxState,
    TradeSide,
    UserDecision,
    UserDecisionLine,
    WithholdingMode,
)
from stock_ai.features import V0_MANIFEST, V1_CORE_MANIFEST, V2_EXTENDED_MANIFEST, FeatureEngine
from stock_ai.fixtures import market_fixture, next_business_morning, portfolio_fixture
from stock_ai.ml import (
    AdvancedFoldResult,
    AdvancedResearchConfig,
    AdvancedResearchExecutionError,
    AdvancedResearchRun,
    BaselinePredictionBundle,
    ProductionDatasetSnapshot,
    ProductionFeatureSets,
    PurgedExpandingWindowSplitter,
    RidgeRegressor,
    TrialAudit,
    TuningSearchError,
    build_production_feature_sets,
    build_production_supervised_dataset,
    build_supervised_dataset,
    load_advanced_research_run,
    load_production_build_manifest,
    load_production_dataset_snapshot,
    load_production_feature_snapshot,
    reserve_locked_final_holdout,
    run_advanced_research,
    run_production_walk_forward_baselines,
    walk_forward_validate,
    write_advanced_research_run,
    write_dataset_snapshot,
    write_production_baseline_report,
    write_production_build_manifest,
    write_production_dataset_snapshot,
    write_production_feature_snapshot,
)
from stock_ai.ml.experiments import ExperimentRecord, ExperimentRegistry
from stock_ai.research import run_research_decision_e2e

app = typer.Typer(
    no_args_is_help=True,
    help="Japanese-stock decision-support research commands (never submits orders).",
)
data_app = typer.Typer(
    no_args_is_help=True,
    help="Acquire, verify, and inspect immutable J-Quants V2 data.",
)
app.add_typer(data_app, name="data")
research_app = typer.Typer(
    no_args_is_help=True,
    help="Build immutable real-data datasets and research-only reports/proposals.",
)
app.add_typer(research_app, name="research")


@dataclass(frozen=True)
class _ProductionArtifacts:
    bundle: ProductionDataBundle
    features: ProductionFeatureSets
    dataset: pd.DataFrame
    snapshot: ProductionDatasetSnapshot


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
    except (JQuantsError, DataQualityError, StorageIntegrityError, ValueError, RuntimeError) as exc:
        typer.echo(f"ingestion blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"run={result.ingestion_run_id} status={result.status.value} "
        f"source_date={result.source_date.isoformat()} objects={len(result.objects)}"
    )
    typer.echo("No fixture fallback was used. No securities order was submitted.")


@data_app.command("history")
def data_history(
    start: Annotated[str, typer.Option(help="Inclusive source start date (YYYY-MM-DD).")],
    end: Annotated[str, typer.Option(help="Inclusive source end date (YYYY-MM-DD).")],
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
        typer.Option(help="Declared plan used for Bulk access and throttling."),
    ] = SubscriptionPlan.STANDARD,
    datasets: Annotated[
        str,
        typer.Option(
            help=(
                "Comma-separated Bulk datasets: security_master,daily_prices,"
                "financial_summary,trading_calendar,topix."
            )
        ),
    ] = "security_master,daily_prices,financial_summary,trading_calendar,topix",
    resume: Annotated[
        bool,
        typer.Option("--resume/--no-resume", help="Skip successfully checkpointed Bulk files."),
    ] = True,
) -> None:
    """Acquire an official V2 Bulk history with durable file-level resume checkpoints."""

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as exc:
        raise typer.BadParameter("--start and --end must use YYYY-MM-DD") from exc
    selected = _parse_datasets(datasets)
    target_catalog = catalog_path or data_root / "catalog.duckdb"
    try:
        with (
            JQuantsV2Client.from_env(config=JQuantsV2Config(plan=plan)) as client,
            DuckDBCatalog(target_catalog) as catalog,
        ):
            store = ImmutableParquetStore(data_root)
            daily_ingestor = JQuantsV2Ingestor(
                client=client,
                store=store,
                catalog=catalog,
            )
            result = JQuantsV2HistoryIngestor(
                client=client,
                daily_ingestor=daily_ingestor,
                catalog=catalog,
            ).sync_history(
                start_date,
                end_date,
                datasets=selected,
                resume=resume,
            )
    except (JQuantsError, DataQualityError, StorageIntegrityError, ValueError, RuntimeError) as exc:
        typer.echo(f"history ingestion blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"history={result.start.isoformat()}..{result.end.isoformat()} "
        f"listed={result.listed_files} downloaded={result.downloaded_files} "
        f"skipped={result.skipped_files} source_dates={result.ingested_source_dates} "
        f"objects={result.objects}"
    )
    typer.echo("No fixture fallback was used. No securities order was submitted.")


@data_app.command("verify")
def data_verify(
    data_root: Annotated[
        Path,
        typer.Option(help="Root containing immutable raw/normalized object directories."),
    ] = Path("data"),
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="DuckDB catalog path; defaults below data-root."),
    ] = None,
    allow_empty: Annotated[
        bool,
        typer.Option(help="Explicitly allow a new store with no published objects."),
    ] = False,
) -> None:
    """Reconcile the catalog and every immutable object; empty is blocked by default."""

    store = ImmutableParquetStore(data_root)
    target_catalog = catalog_path or data_root / "catalog.duckdb"
    try:
        immutable_manifests = tuple(data_root.glob("*/jquants_v2/*/source_date=*/*/manifest.json"))
        if target_catalog.is_file():
            with DuckDBCatalog(target_catalog) as catalog:
                verified = catalog.verify_integrity(store, allow_empty=True)
        elif immutable_manifests:
            raise StorageIntegrityError("immutable objects exist but the catalog is missing")
        else:
            verified = 0
        feature_paths = sorted(data_root.glob("features/*/*/*.parquet"))
        dataset_paths = sorted(data_root.glob("datasets/production/*/*.parquet"))
        build_paths = sorted(data_root.glob("builds/production/*/*.json"))
        for path in feature_paths:
            load_production_feature_snapshot(path)
        for path in dataset_paths:
            load_production_dataset_snapshot(path)
        builds = [load_production_build_manifest(path) for path in build_paths]
        referenced_features = {
            snapshot_path.resolve()
            for build in builds
            for snapshot_path in (
                build.v0_parquet_path,
                build.v1_parquet_path,
                build.v2_parquet_path,
            )
            if snapshot_path is not None
        }
        referenced_datasets = {build.dataset_parquet_path.resolve() for build in builds}
        if {path.resolve() for path in feature_paths} != referenced_features:
            raise StorageIntegrityError("production feature snapshots are not batch-complete")
        if {path.resolve() for path in dataset_paths} != referenced_datasets:
            raise StorageIntegrityError("production dataset snapshots are not batch-complete")
        total = verified + len(feature_paths) + len(dataset_paths) + len(build_paths)
        if total == 0 and not allow_empty:
            raise StorageIntegrityError("no immutable objects or production snapshots were found")
    except (StorageIntegrityError, RuntimeError, ValueError) as exc:
        typer.echo(f"verification failed: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"verified_objects={verified} feature_snapshots={len(feature_paths)} "
        f"dataset_snapshots={len(dataset_paths)} builds={len(build_paths)} status=OK"
    )


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


def _parse_aware_timestamp(value: str, option: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise typer.BadParameter(f"{option} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise typer.BadParameter(f"{option} must include a timezone offset")
    return parsed


def _build_production_artifacts(
    *,
    data_root: Path,
    catalog_path: Path,
    source_snapshot_as_of: datetime,
    plan: SubscriptionPlan,
    minimum_market_coverage: float,
    revision_policy: HistoricalRevisionPolicy,
) -> _ProductionArtifacts:
    with DuckDBCatalog(catalog_path) as catalog:
        bundle = build_production_data(
            catalog,
            source_snapshot_as_of=source_snapshot_as_of,
            minimum_market_coverage=minimum_market_coverage,
            revision_policy=revision_policy,
        )
    features = build_production_feature_sets(bundle)
    if features.v2_extended is None:
        raise RuntimeError("BLOCKED_BY_DATA_CAPABILITY: V2 Extended features were not built")
    dataset, label_1230_status = build_production_supervised_dataset(
        features.v2_extended,
        bundle,
        plan=plan,
    )
    created_at = datetime.now(UTC)
    v0_snapshot = write_production_feature_snapshot(
        features.v0,
        data_root / "features" / V0_MANIFEST.feature_set_version,
        manifest=V0_MANIFEST,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=source_snapshot_as_of,
        created_at=created_at,
    )
    v1_snapshot = write_production_feature_snapshot(
        features.v1_core,
        data_root / "features" / V1_CORE_MANIFEST.feature_set_version,
        manifest=V1_CORE_MANIFEST,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=source_snapshot_as_of,
        created_at=created_at,
    )
    v2_snapshot = write_production_feature_snapshot(
        features.v2_extended,
        data_root / "features" / V2_EXTENDED_MANIFEST.feature_set_version,
        manifest=V2_EXTENDED_MANIFEST,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=source_snapshot_as_of,
        created_at=created_at,
    )
    snapshot = write_production_dataset_snapshot(
        dataset,
        data_root / "datasets" / "production",
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=bundle.source_snapshot_ids,
        as_of=source_snapshot_as_of,
        created_at=created_at,
        label_1230_status=label_1230_status,
        manifests=(V0_MANIFEST, V1_CORE_MANIFEST, V2_EXTENDED_MANIFEST),
    )
    write_production_build_manifest(
        v0_snapshot,
        v1_snapshot,
        v2_snapshot,
        snapshot,
        data_root / "builds" / "production",
        created_at=created_at,
    )
    return _ProductionArtifacts(
        bundle=bundle,
        features=features,
        dataset=dataset,
        snapshot=snapshot,
    )


@research_app.command("build")
def research_build(
    as_of: Annotated[
        str,
        typer.Option(help="Immutable source-vintage cutoff with timezone."),
    ],
    data_root: Annotated[Path, typer.Option(help="Data lake root.")] = Path("data"),
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="DuckDB catalog; defaults below data-root."),
    ] = None,
    plan: Annotated[
        SubscriptionPlan,
        typer.Option(help="Confirmed J-Quants plan for capability decisions."),
    ] = SubscriptionPlan.STANDARD,
    minimum_market_coverage: Annotated[
        float,
        typer.Option(help="Minimum PIT-universe coverage for market/sector context."),
    ] = 0.95,
    revision_policy: Annotated[
        HistoricalRevisionPolicy,
        typer.Option(
            help=(
                "Historical revision policy. single-vintage results remain research-only; "
                "strict mode blocks historical labels without provider vintages."
            )
        ),
    ] = HistoricalRevisionPolicy.SINGLE_VINTAGE_AS_REVISED,
) -> None:
    """Build immutable V0/V1 and Production Research Dataset snapshots."""

    cutoff = _parse_aware_timestamp(as_of, "--as-of")
    try:
        artifacts = _build_production_artifacts(
            data_root=data_root,
            catalog_path=catalog_path or data_root / "catalog.duckdb",
            source_snapshot_as_of=cutoff,
            plan=plan,
            minimum_market_coverage=minimum_market_coverage,
            revision_policy=revision_policy,
        )
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"production build blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    v2_rows = 0 if artifacts.features.v2_extended is None else len(artifacts.features.v2_extended)
    typer.echo(
        f"v0_rows={len(artifacts.features.v0)} v1_rows={len(artifacts.features.v1_core)} "
        f"v2_rows={v2_rows} "
        f"dataset={artifacts.snapshot.snapshot_id} rows={artifacts.snapshot.rows} "
        f"period={artifacts.snapshot.data_start}..{artifacts.snapshot.data_end} "
        f"label_1230={artifacts.snapshot.label_1230_status.value}"
    )


@research_app.command("baseline")
def research_baseline(
    dataset_parquet: Annotated[
        Path,
        typer.Option(exists=True, dir_okay=False, help="Production Dataset Parquet path."),
    ],
    code_commit: Annotated[str, typer.Option(help="Exact source commit for audit provenance.")],
    report_root: Annotated[
        Path,
        typer.Option(help="Immutable baseline report directory."),
    ] = Path("artifacts/reports/baselines"),
) -> None:
    """Run HOLD/Momentum/Ridge real-data walk-forward without opening the holdout."""

    try:
        snapshot, dataset = load_production_dataset_snapshot(dataset_parquet)
        report = run_production_walk_forward_baselines(
            dataset,
            data_snapshot_id=snapshot.snapshot_id,
            created_at=datetime.now(UTC),
            code_commit=code_commit,
        )
        path = write_production_baseline_report(report, report_root)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"baseline blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"report={report.report_id} holdout_start={report.locked_holdout_start} path={path}")
    for model in report.models:
        rank_ic = "NA" if model.mean_daily_rank_ic is None else f"{model.mean_daily_rank_ic:.6f}"
        typer.echo(
            f"model={model.model_name} folds={model.folds} mse={model.mean_squared_error:.8f} "
            f"rank_ic={rank_ic} rank_dates={model.rank_ic_dates}"
        )


@research_app.command("advanced")
def research_advanced(
    build_manifest: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Authenticated V0/V1/V2/Dataset Production Build Manifest.",
        ),
    ],
    code_commit: Annotated[str, typer.Option(help="Exact source commit for audit provenance.")],
    report_root: Annotated[
        Path,
        typer.Option(help="Content-addressed Goal 3 report root."),
    ] = Path("artifacts/reports/advanced"),
    target_family: Annotated[
        str,
        typer.Option(help="return, topix_excess, sector_excess, or beta_residual."),
    ] = "return",
    horizons: Annotated[
        str,
        typer.Option(help="Comma-separated subset of 1,5,20."),
    ] = "5",
    model_families: Annotated[
        str,
        typer.Option(help="Comma-separated lightgbm,xgboost,catboost."),
    ] = "lightgbm",
    seeds: Annotated[
        str,
        typer.Option(help="Comma-separated deterministic seeds."),
    ] = "17",
    tuning_trials: Annotated[
        int,
        typer.Option(min=1, max=100, help="Hard bound on Optuna trials per model/horizon."),
    ] = 20,
    tuning_timeout_seconds: Annotated[
        int,
        typer.Option(min=10, max=7200, help="Hard tuning timeout per model/horizon."),
    ] = 900,
    estimator_count: Annotated[
        int,
        typer.Option(min=5, max=2000, help="Bounded boosting iterations."),
    ] = 300,
    initial_train_periods: Annotated[
        int,
        typer.Option(min=20, help="Initial expanding training sessions."),
    ] = 500,
    validation_periods: Annotated[
        int,
        typer.Option(min=1, help="Validation sessions per fold."),
    ] = 60,
    step_periods: Annotated[
        int,
        typer.Option(min=1, help="Walk-forward step sessions."),
    ] = 60,
    holdout_periods: Annotated[
        int,
        typer.Option(min=1, help="Locked final holdout sessions."),
    ] = 120,
    run_ablations: Annotated[
        bool,
        typer.Option(help="Run chronological incremental feature-family ablations."),
    ] = True,
    run_diagnostics: Annotated[
        bool,
        typer.Option(help="Run OOS permutation diagnostics."),
    ] = True,
    max_materialized_oof_rows: Annotated[
        int,
        typer.Option(min=1000, help="Fail-closed bound for in-memory OOF rows."),
    ] = 5_000_000,
    max_model_fits: Annotated[
        int,
        typer.Option(min=10, help="Fail-closed bound for total boosting fits."),
    ] = 5_000,
    experiment_registry: Annotated[
        Path,
        typer.Option(help="Append-only success/failure experiment registry."),
    ] = Path("artifacts/experiments/advanced.jsonl"),
) -> None:
    """Run leakage-safe GBDT/LTR/downside/OOF research; never open the holdout."""

    created_at = datetime.now(UTC)
    config: AdvancedResearchConfig | None = None
    snapshot: ProductionDatasetSnapshot | None = None
    run: AdvancedResearchRun | None = None
    authenticated_build_id: str | None = None
    authenticated_feature_snapshot_id: str | None = None
    raw_config: dict[str, object] = {
        "horizons": horizons,
        "target_family": target_family,
        "model_families": model_families,
        "seeds": seeds,
        "tuning_trials": tuning_trials,
        "tuning_timeout_seconds": tuning_timeout_seconds,
        "estimator_count": estimator_count,
        "initial_train_periods": initial_train_periods,
        "validation_periods": validation_periods,
        "step_periods": step_periods,
        "holdout_periods": holdout_periods,
        "run_ablations": run_ablations,
        "run_diagnostics": run_diagnostics,
        "max_materialized_oof_rows": max_materialized_oof_rows,
        "max_model_fits": max_model_fits,
    }
    try:
        config = AdvancedResearchConfig.model_validate(
            {
                "horizons": tuple(
                    int(value.strip()) for value in horizons.split(",") if value.strip()
                ),
                "target_family": target_family,
                "model_families": tuple(
                    value.strip() for value in model_families.split(",") if value.strip()
                ),
                "seeds": tuple(int(value.strip()) for value in seeds.split(",") if value.strip()),
                "tuning_trials": tuning_trials,
                "tuning_timeout_seconds": tuning_timeout_seconds,
                "estimator_count": estimator_count,
                "initial_train_periods": initial_train_periods,
                "validation_periods": validation_periods,
                "step_periods": step_periods,
                "holdout_periods": holdout_periods,
                "run_ablations": run_ablations,
                "run_diagnostics": run_diagnostics,
                "max_materialized_oof_rows": max_materialized_oof_rows,
                "max_model_fits": max_model_fits,
            }
        )
        build = load_production_build_manifest(build_manifest)
        authenticated_build_id = build.build_id
        snapshot, dataset = load_production_dataset_snapshot(build.dataset_parquet_path)
        v2_snapshot, _ = load_production_feature_snapshot(build.v2_parquet_path)
        authenticated_feature_snapshot_id = v2_snapshot.snapshot_id
        run = run_advanced_research(
            dataset,
            data_snapshot_id=snapshot.snapshot_id,
            created_at=created_at,
            code_commit=code_commit,
            config=config,
            feature_snapshot_id=v2_snapshot.snapshot_id,
            feature_manifest_hash=v2_snapshot.manifest_hash,
        )
        metadata_path, oof_path = write_advanced_research_run(run, report_root)
        # Read through the authenticated boundary before declaring publication complete.
        load_advanced_research_run(oof_path)
        ExperimentRegistry(experiment_registry).append(
            _advanced_experiment_record(
                run,
                experiment_id=f"advanced-{uuid4()}",
                build_id=build.build_id,
                metadata_path=metadata_path,
                oof_path=oof_path,
            )
        )
    except (ValueError, RuntimeError, OSError) as exc:
        execution_error = (
            exc if isinstance(exc, AdvancedResearchExecutionError) else None
        )
        completed_trial_contexts = (
            tuple(
                (tuning.horizon, tuning.model_family, trial)
                for tuning in run.report.tuning_results
                for trial in tuning.trials
            )
            if run is not None
            else ()
        )
        ExperimentRegistry(experiment_registry).append(
            _failed_advanced_experiment_record(
                experiment_id=f"advanced-{uuid4()}",
                created_at=created_at,
                code_commit=code_commit,
                config=config,
                raw_config=raw_config,
                data_snapshot_id=(
                    snapshot.snapshot_id
                    if snapshot is not None
                    else f"UNVERIFIED_BUILD:{build_manifest.stem}"
                ),
                reason=f"{type(exc).__name__}: {str(exc)[:500]}",
                trial_contexts=(
                    execution_error.trial_contexts
                    if execution_error is not None
                    else (
                        completed_trial_contexts
                        or (exc.trial_contexts if isinstance(exc, TuningSearchError) else ())
                    )
                ),
                fold_results=(
                    execution_error.fold_results
                    if execution_error is not None
                    else (run.report.fold_results if run is not None else ())
                ),
                build_id=authenticated_build_id,
                feature_snapshot_id=authenticated_feature_snapshot_id,
                report_id=(run.report.report_id if run is not None else None),
            )
        )
        typer.echo(f"advanced research blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    assert run is not None
    typer.echo(
        f"report={run.report.report_id} oof_rows={run.report.oof_rows} "
        f"holdout_start={run.report.locked_holdout_start} "
        f"adoption_eligible={str(run.report.adoption_eligible).lower()} "
        f"metadata={metadata_path} oof={oof_path}"
    )


def _advanced_experiment_record(
    run: AdvancedResearchRun,
    *,
    experiment_id: str,
    build_id: str,
    metadata_path: Path,
    oof_path: Path,
) -> ExperimentRecord:
    report = run.report
    return ExperimentRecord(
        experiment_id=experiment_id,
        created_at=report.created_at,
        hypothesis=report.hypothesis,
        data_snapshot_id=report.data_snapshot_id,
        feature_set_version=report.feature_set_version,
        preprocessing_version=report.preprocessing_version,
        feature_definition_hashes=report.feature_definition_hashes,
        code_commit=report.code_commit,
        config_hash=report.config_hash,
        model_type="advanced_gbdt_ltr_downside_oof",
        parameters={
            "target_family": report.prediction_semantics,
            "horizons": ",".join(str(value) for value in report.config.horizons),
            "model_families": ",".join(report.config.model_families),
            "seeds": ",".join(str(value) for value in report.config.seeds),
            "estimator_count": report.config.estimator_count,
            "config_json": json.dumps(
                report.config.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ),
            "report_id": report.report_id,
            "report_metadata_path": str(metadata_path.resolve()),
            "oof_path": str(oof_path.resolve()),
            "build_id": build_id,
            "feature_snapshot_id": report.feature_snapshot_id,
            "tax_policy_version": report.tax_policy_version,
            "decision_engine_version": report.decision_engine_version,
            "cost_scenarios_bps": ",".join(
                str(value) for value in report.cost_scenarios_bps
            ),
        },
        seed=None,
        fold_results=tuple(_fold_audit_row(fold) for fold in report.fold_results),
        trial_results=tuple(
            _trial_audit_row(
                trial,
                horizon=tuning.horizon,
                model_family=tuning.model_family,
            )
            for tuning in report.tuning_results
            for trial in tuning.trials
        ),
        aggregate_results={
            "oof_rows": report.oof_rows,
            "adoption_eligible": str(report.adoption_eligible).lower(),
            "cost_evaluation_status": report.cost_evaluation_status.value,
            "historical_revision_policy": report.historical_revision_policy,
            "historical_revision_status": report.historical_revision_status.value,
            "adoption_blocking_reasons": " | ".join(report.adoption_blocking_reasons),
        },
        decision="research_only",
        rejection_reason=None,
        locked_holdout_accessed=False,
    )


def _failed_advanced_experiment_record(
    *,
    experiment_id: str,
    created_at: datetime,
    code_commit: str,
    config: AdvancedResearchConfig | None,
    raw_config: Mapping[str, object],
    data_snapshot_id: str,
    reason: str,
    trial_contexts: tuple[tuple[int, str, TrialAudit], ...] = (),
    fold_results: tuple[AdvancedFoldResult, ...] = (),
    build_id: str | None = None,
    feature_snapshot_id: str | None = None,
    report_id: str | None = None,
) -> ExperimentRecord:
    serialized_raw_config = json.dumps(
        dict(raw_config), sort_keys=True, separators=(",", ":"), default=str
    )
    config_hash = (
        config.config_hash
        if config is not None
        else hashlib.sha256(serialized_raw_config.encode()).hexdigest()
    )
    return ExperimentRecord(
        experiment_id=experiment_id,
        created_at=created_at,
        hypothesis=(
            config.hypothesis
            if config is not None
            else "Advanced research command should complete under the declared configuration"
        ),
        data_snapshot_id=data_snapshot_id,
        feature_set_version=V2_EXTENDED_MANIFEST.feature_set_version,
        preprocessing_version=V2_EXTENDED_MANIFEST.preprocessing_version,
        feature_definition_hashes=V2_EXTENDED_MANIFEST.feature_definition_hashes,
        code_commit=code_commit.strip() or "UNSET",
        config_hash=config_hash,
        model_type="advanced_gbdt_ltr_downside_oof",
        parameters={
            "raw_config_json": serialized_raw_config,
            **({"build_id": build_id} if build_id is not None else {}),
            **(
                {"feature_snapshot_id": feature_snapshot_id}
                if feature_snapshot_id is not None
                else {}
            ),
            **({"report_id": report_id} if report_id is not None else {}),
            **(
                {
                    "target_family": config.target_family,
                    "horizons": ",".join(str(value) for value in config.horizons),
                    "model_families": ",".join(config.model_families),
                    "seeds": ",".join(str(value) for value in config.seeds),
                    "estimator_count": config.estimator_count,
                }
                if config is not None
                else {}
            ),
        },
        seed=None,
        fold_results=tuple(_fold_audit_row(fold) for fold in fold_results),
        trial_results=tuple(
            _trial_audit_row(
                trial,
                horizon=horizon,
                model_family=model_family,
            )
            for horizon, model_family, trial in trial_contexts
        ),
        aggregate_results={"status": "FAILED"},
        decision="rejected",
        rejection_reason=reason,
        locked_holdout_accessed=False,
    )


def _fold_audit_row(fold: AdvancedFoldResult) -> dict[str, int | float | str]:
    return {
        "horizon": fold.horizon,
        "fold": fold.fold,
        "model_family": fold.model_family,
        "task": fold.task,
        "seed": fold.seed,
        "validation_start": fold.validation_start,
        "validation_end": fold.validation_end,
        "rows": fold.rows,
        "mean_squared_error": fold.mean_squared_error,
        **(
            {"mean_daily_rank_ic": fold.mean_daily_rank_ic}
            if fold.mean_daily_rank_ic is not None
            else {}
        ),
    }


def _trial_audit_row(
    trial: TrialAudit,
    *,
    horizon: int | None,
    model_family: str | None,
) -> dict[str, int | float | str]:
    return {
        **({"horizon": horizon} if horizon is not None else {}),
        **({"model_family": model_family} if model_family is not None else {}),
        "number": trial.number,
        "state": trial.state,
        "parameters": json.dumps(
            dict(trial.parameters), sort_keys=True, separators=(",", ":")
        ),
        **({"value": trial.value} if trial.value is not None else {}),
        **(
            {"duration_seconds": trial.duration_seconds}
            if trial.duration_seconds is not None
            else {}
        ),
        **(
            {"failure_reason": trial.failure_reason}
            if trial.failure_reason is not None
            else {}
        ),
    }


@research_app.command("e2e")
def research_e2e(
    as_of: Annotated[str, typer.Option(help="Immutable source-vintage cutoff with timezone.")],
    code_commit: Annotated[str, typer.Option(help="Exact source commit for audit provenance.")],
    data_root: Annotated[Path, typer.Option(help="Data lake root.")] = Path("data"),
    catalog_path: Annotated[
        Path | None,
        typer.Option(help="DuckDB catalog; defaults below data-root."),
    ] = None,
    report_root: Annotated[
        Path,
        typer.Option(help="Research report root."),
    ] = Path("artifacts/reports"),
    plan: Annotated[SubscriptionPlan, typer.Option(help="Confirmed J-Quants plan.")] = (
        SubscriptionPlan.STANDARD
    ),
    cash_yen: Annotated[
        float,
        typer.Option(help="Explicit paper cash balance; no broker connection."),
    ] = 1_000_000.0,
    candidate_limit: Annotated[
        int,
        typer.Option(help="Bounded candidate count for exact portfolio search."),
    ] = 8,
    minimum_market_coverage: Annotated[
        float,
        typer.Option(help="Minimum PIT-universe context coverage."),
    ] = 0.95,
    revision_policy: Annotated[
        HistoricalRevisionPolicy,
        typer.Option(help="Explicit historical source-revision policy."),
    ] = HistoricalRevisionPolicy.SINGLE_VINTAGE_AS_REVISED,
) -> None:
    """Run real data -> baseline -> paper proposal; never submit or simulate an order."""

    cutoff = _parse_aware_timestamp(as_of, "--as-of")
    try:
        artifacts = _build_production_artifacts(
            data_root=data_root,
            catalog_path=catalog_path or data_root / "catalog.duckdb",
            source_snapshot_as_of=cutoff,
            plan=plan,
            minimum_market_coverage=minimum_market_coverage,
            revision_policy=revision_policy,
        )
        report = run_production_walk_forward_baselines(
            artifacts.dataset,
            data_snapshot_id=artifacts.snapshot.snapshot_id,
            created_at=datetime.now(UTC),
            code_commit=code_commit,
        )
        baseline_path = write_production_baseline_report(
            report,
            report_root / "baselines",
        )
        latest_date = artifacts.features.v1_core["trading_date"].max()
        latest = artifacts.features.v1_core.loc[
            artifacts.features.v1_core["trading_date"] == latest_date
        ].copy()
        portfolio_as_of = pd.Timestamp(latest["available_at"].max()).to_pydatetime()
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
            portfolio_id=f"paper-cash-{artifacts.snapshot.snapshot_id[:12]}",
            as_of=portfolio_as_of,
            accounts=(account,),
            account_buckets=(bucket,),
            positions=(),
            cash=(
                CashState(
                    account_bucket_id=bucket.bucket_id,
                    available_cash=Decimal(str(cash_yen)),
                ),
            ),
            tax_states=(
                TaxState(account_bucket_id=bucket.bucket_id, tax_year=portfolio_as_of.year),
            ),
        )
        engine = DailyPortfolioDecisionEngine(
            config=DecisionEngineConfig(maximum_positions=min(candidate_limit, 10)),
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
                    effective_from=portfolio_as_of.date().replace(month=1, day=1),
                )
            ),
        )
        decision = run_research_decision_e2e(
            dataset=artifacts.dataset,
            latest_features=latest,
            universe=artifacts.bundle.universe,
            data_snapshot_id=artifacts.snapshot.snapshot_id,
            baseline_report=report,
            portfolio=portfolio,
            engine=engine,
            candidate_limit=candidate_limit,
        )
        proposal_directory = report_root / "proposals"
        proposal_directory.mkdir(parents=True, exist_ok=True)
        proposal_path = proposal_directory / f"{decision.proposal.proposal_id}.research.json"
        envelope = {
            "data_snapshot_id": artifacts.snapshot.snapshot_id,
            "baseline_report_id": report.report_id,
            "reference_price_rule": decision.reference_price_rule,
            "is_order_instruction": False,
            "proposal": decision.proposal.model_dump(mode="json"),
        }
        proposal_payload = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if proposal_path.exists():
            if proposal_path.read_text(encoding="utf-8") != proposal_payload:
                raise RuntimeError("existing research proposal identity collision")
        else:
            temporary = proposal_directory / f".{proposal_path.name}.{uuid4().hex}.tmp"
            temporary.write_text(proposal_payload, encoding="utf-8")
            temporary.replace(proposal_path)
    except (ValueError, RuntimeError) as exc:
        typer.echo(f"research E2E blocked: {exc}", err=True)
        raise typer.Exit(code=2) from None
    typer.echo(
        f"dataset={artifacts.snapshot.snapshot_id} baseline={report.report_id} "
        f"proposal={decision.proposal.proposal_id} candidates={decision.candidate_count}"
    )
    typer.echo(f"baseline_path={baseline_path} proposal_path={proposal_path}")
    typer.echo(decision.reference_price_rule)
    typer.echo("RESEARCH ONLY - no order or execution record was created or submitted.")


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
