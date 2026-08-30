"""Production research features, labels, snapshots, and baseline reports."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stock_ai.data.contracts import (
    CapabilityStatus,
    HistoricalRevisionPolicy,
    SubscriptionPlan,
)
from stock_ai.data.production import ProductionDataBundle
from stock_ai.features import V0_MANIFEST, V1_CORE_MANIFEST, V2_EXTENDED_MANIFEST, FeatureEngine
from stock_ai.features.registry import FeatureSetManifest
from stock_ai.ml.dataset import HORIZONS
from stock_ai.ml.models import MomentumRegressor, Regressor, RidgeRegressor
from stock_ai.ml.validation import PurgedExpandingWindowSplitter, reserve_locked_final_holdout


@dataclass(frozen=True)
class ProductionFeatureSets:
    v0: pd.DataFrame
    v1_core: pd.DataFrame
    v2_extended: pd.DataFrame | None = None


class ProductionDatasetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    snapshot_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    as_of: datetime
    source_snapshot_as_of: datetime
    source_snapshot_ids: tuple[tuple[str, str], ...]
    feature_manifests: tuple[tuple[str, str], ...]
    target_definition: str
    label_1230_status: CapabilityStatus
    historical_revision_policy: str
    historical_revision_status: CapabilityStatus
    rows: int = Field(ge=0)
    data_start: str
    data_end: str
    parquet_path: Path
    metadata_path: Path

    @field_validator("created_at", "as_of", "source_snapshot_as_of")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("production snapshot timestamps must be timezone-aware")
        return value


class ProductionFeatureSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    snapshot_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    as_of: datetime
    source_snapshot_as_of: datetime
    source_snapshot_ids: tuple[tuple[str, str], ...]
    feature_set_id: str
    feature_set_version: str
    manifest_hash: str = Field(min_length=64, max_length=64)
    historical_revision_policy: str
    historical_revision_status: CapabilityStatus
    rows: int = Field(ge=0)
    parquet_path: Path
    metadata_path: Path

    @field_validator("created_at", "as_of", "source_snapshot_as_of")
    @classmethod
    def feature_timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("production feature timestamps must be timezone-aware")
        return value


class ProductionBuildManifest(BaseModel):
    """Atomic publication marker for a complete V0/V1/V2/Dataset production build."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    build_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    source_snapshot_as_of: datetime
    source_snapshot_ids: tuple[tuple[str, str], ...]
    v0_snapshot_id: str = Field(min_length=64, max_length=64)
    v1_snapshot_id: str = Field(min_length=64, max_length=64)
    v2_snapshot_id: str = Field(min_length=64, max_length=64)
    dataset_snapshot_id: str = Field(min_length=64, max_length=64)
    v0_parquet_path: Path
    v1_parquet_path: Path
    v2_parquet_path: Path
    dataset_parquet_path: Path
    manifest_path: Path

    @field_validator("created_at", "source_snapshot_as_of")
    @classmethod
    def build_timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("production build timestamps must be timezone-aware")
        return value


class BaselineModelSummary(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    model_name: str
    folds: int = Field(ge=1)
    validation_rows: int = Field(ge=1)
    mean_squared_error: float = Field(ge=0)
    mean_daily_rank_ic: float | None
    rank_ic_dates: int = Field(ge=0)
    top_decile_mean_target: float
    cost_scenarios: tuple[tuple[int, float], ...]


class ProductionBaselineReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    report_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    code_commit: str = Field(min_length=1)
    config_hash: str = Field(min_length=64, max_length=64)
    data_snapshot_id: str
    feature_set_version: str
    target_column: str
    locked_holdout_start: str
    historical_revision_policy: str
    historical_revision_status: CapabilityStatus
    adoption_eligible: bool
    adoption_blocking_reason: str | None
    label_status_counts: tuple[tuple[str, int], ...]
    models: tuple[BaselineModelSummary, ...]

    @field_validator("created_at")
    @classmethod
    def report_timestamp_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("baseline report created_at must be timezone-aware")
        return value


def build_production_feature_sets(bundle: ProductionDataBundle) -> ProductionFeatureSets:
    """Compute both frozen V0 and V1 Core manifests from the same real-data inputs."""

    _assert_financial_timing(bundle.financials)
    incomplete_market = ~bundle.market_context["coverage_complete"].astype(bool)
    if incomplete_market.any():
        first = pd.to_datetime(bundle.market_context.loc[incomplete_market, "trading_date"]).min()
        raise ValueError(
            "BLOCKED_BY_DATA_CAPABILITY: market breadth coverage is below the configured "
            f"threshold on {first.date()}"
        )
    v2 = FeatureEngine(V2_EXTENDED_MANIFEST).transform(
        bundle.daily,
        bundle.market_context,
        bundle.sector_context,
        financials=bundle.financials,
    )
    # V2 is a strict superset. Projecting one shared pass guarantees identical observations.
    v1 = v2[
        [
            "symbol",
            "sector",
            "trading_date",
            "available_at",
            "close",
            "adjusted_close",
            *V1_CORE_MANIFEST.feature_names,
        ]
    ].copy()
    v0 = v1[
        [
            "symbol",
            "sector",
            "trading_date",
            "available_at",
            "close",
            "adjusted_close",
            *V0_MANIFEST.feature_names,
        ]
    ].copy()

    audit = bundle.market_context[
        [
            "trading_date",
            "coverage_ratio",
            "coverage_complete",
            "revision_available_at",
        ]
    ].rename(
        columns={
            "coverage_ratio": "market_coverage_ratio",
            "coverage_complete": "market_coverage_complete",
            "revision_available_at": "market_revision_available_at",
        }
    )
    sector_lineage = bundle.sector_context[
        ["sector", "trading_date", "revision_available_at"]
    ].rename(columns={"revision_available_at": "sector_revision_available_at"})
    daily_lineage = bundle.daily[["symbol", "trading_date", "revision_available_at"]].rename(
        columns={"revision_available_at": "daily_revision_available_at"}
    )
    financial_lineage = bundle.financials[
        ["symbol", "trading_date", "financial_revision_available_at"]
    ]
    master_lineage = bundle.universe[["symbol", "effective_date", "revision_available_at"]].rename(
        columns={
            "effective_date": "trading_date",
            "revision_available_at": "master_revision_available_at",
        }
    )
    shares = bundle.daily[
        [
            "symbol",
            "trading_date",
            "shares_outstanding",
            "shares_outstanding_missing_reason",
        ]
    ].copy()
    for frame in (v0, v1, v2):
        frame["source_snapshot_as_of"] = pd.Timestamp(bundle.source_snapshot_as_of)
    v0 = v0.merge(audit, on="trading_date", how="left", validate="many_to_one")
    v1 = v1.merge(audit, on="trading_date", how="left", validate="many_to_one")
    v2 = v2.merge(audit, on="trading_date", how="left", validate="many_to_one")
    for name, frame in (("v0", v0), ("v1", v1), ("v2", v2)):
        frame = frame.merge(
            sector_lineage,
            on=["sector", "trading_date"],
            how="left",
            validate="many_to_one",
        )
        frame = frame.merge(
            daily_lineage,
            on=["symbol", "trading_date"],
            how="left",
            validate="one_to_one",
        )
        frame = frame.merge(
            financial_lineage,
            on=["symbol", "trading_date"],
            how="left",
            validate="one_to_one",
        )
        frame = frame.merge(
            master_lineage,
            on=["symbol", "trading_date"],
            how="left",
            validate="one_to_one",
        )
        revision_columns = [
            "daily_revision_available_at",
            "master_revision_available_at",
            "market_revision_available_at",
            "sector_revision_available_at",
            "financial_revision_available_at",
        ]
        # A lineage source can be entirely absent for an observation range (for
        # example before the first financial disclosure). Parquet/Pandas may
        # then materialize that merged column as float NaN beside timezone-aware
        # timestamps. Coerce every component to one datetime dtype before the
        # row-wise maximum so missing lineage remains NaT.
        for column in revision_columns:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce").astype(
                "datetime64[ns, UTC]"
            )
        frame["source_revision_available_at"] = frame[revision_columns].max(axis=1)
        if bundle.revision_policy is HistoricalRevisionPolicy.STRICT_AS_KNOWN:
            frame["available_at"] = frame[["available_at", "source_revision_available_at"]].max(
                axis=1
            )
        frame["historical_revision_policy"] = bundle.revision_policy.value
        frame["historical_revision_status"] = bundle.historical_revision_status.value
        if name == "v0":
            v0 = frame
        elif name == "v1":
            v1 = frame
        else:
            v2 = frame
    for name, frame in (("v1", v1), ("v2", v2)):
        frame = frame.merge(
            shares,
            on=["symbol", "trading_date"],
            how="left",
            validate="one_to_one",
        )
        frame["shares_outstanding_is_missing"] = frame["shares_outstanding"].isna()
        if name == "v1":
            v1 = frame
        else:
            v2 = frame
    missing_without_reason = (
        v1["shares_outstanding_is_missing"] & v1["shares_outstanding_missing_reason"].isna()
    )
    v1.loc[
        missing_without_reason,
        "shares_outstanding_missing_reason",
    ] = "no fiscal disclosure available at observation cutoff"
    v2.loc[
        v2["shares_outstanding_is_missing"] & v2["shares_outstanding_missing_reason"].isna(),
        "shares_outstanding_missing_reason",
    ] = "no fiscal disclosure available at observation cutoff"
    return ProductionFeatureSets(v0=v0, v1_core=v1, v2_extended=v2)


def build_production_supervised_dataset(
    features: pd.DataFrame,
    bundle: ProductionDataBundle,
    *,
    plan: SubscriptionPlan,
) -> tuple[pd.DataFrame, CapabilityStatus]:
    """Create fixed-calendar 1/5/20d absolute, TOPIX, sector, and beta labels."""

    required = {"symbol", "sector", "trading_date", "available_at", "adjusted_close"}
    _require_columns(features, required, "production features")
    output = features.copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.normalize()
    output["as_of"] = pd.to_datetime(output["available_at"], utc=True)
    if output.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("production features contain duplicate symbol/trading_date rows")

    calendar = pd.DatetimeIndex(
        pd.to_datetime(bundle.universe["effective_date"]).sort_values().unique()
    )
    date_position = {value: index for index, value in enumerate(calendar)}
    daily = bundle.daily[
        [
            "symbol",
            "sector",
            "trading_date",
            "adjusted_close",
            "available_at",
            "trading_session_index",
            "research_afternoon_open",
        ]
    ].copy()
    daily["trading_date"] = pd.to_datetime(daily["trading_date"]).dt.normalize()
    market = bundle.market_context[["trading_date", "topix_close"]].copy()
    market["trading_date"] = pd.to_datetime(market["trading_date"]).dt.normalize()
    topix_lookup = market.set_index("trading_date")["topix_close"]
    label_maturity_lookup = bundle.market_context.assign(
        trading_date=pd.to_datetime(bundle.market_context["trading_date"]).dt.normalize()
    ).set_index("trading_date")["available_at"]
    sector_index = _sector_cumulative_index(bundle.sector_context)
    universe_keys = set(
        zip(
            bundle.universe["symbol"].astype(str),
            pd.to_datetime(bundle.universe["effective_date"]).dt.normalize(),
            strict=True,
        )
    )
    beta = _causal_beta(daily, market)
    output = output.merge(
        beta,
        on=["symbol", "trading_date"],
        how="left",
        validate="one_to_one",
    )

    for horizon in HORIZONS:
        output = _add_horizon_labels(
            output,
            horizon=horizon,
            calendar=calendar,
            date_position=date_position,
            daily=daily,
            topix_lookup=topix_lookup,
            label_maturity_lookup=label_maturity_lookup,
            sector_index=sector_index,
            universe_keys=universe_keys,
        )

    revision_policy, _revision_status = _revision_contract(output)
    if revision_policy == HistoricalRevisionPolicy.STRICT_AS_KNOWN.value:
        for horizon in HORIZONS:
            value_columns = [
                f"target_return_{horizon}d",
                f"target_topix_excess_{horizon}d",
                f"target_sector_excess_{horizon}d",
                f"target_beta_residual_{horizon}d",
                f"target_large_loss_{horizon}d",
            ]
            status_columns = [
                f"label_status_{horizon}d",
                f"label_status_topix_excess_{horizon}d",
                f"label_status_sector_excess_{horizon}d",
                f"label_status_beta_residual_{horizon}d",
            ]
            output[value_columns] = pd.NA
            output[status_columns] = "BLOCKED_BY_REVISION_HISTORY"

    afternoon_values = pd.to_numeric(daily["research_afternoon_open"], errors="coerce")
    afternoon_rows = int(afternoon_values.notna().sum())
    afternoon_available = plan.includes(SubscriptionPlan.PREMIUM) and afternoon_rows > 0
    if afternoon_available and revision_policy != HistoricalRevisionPolicy.STRICT_AS_KNOWN.value:
        output = _add_exact_afternoon_open_labels(
            output,
            daily=daily,
            calendar=calendar,
            date_position=date_position,
            label_maturity_lookup=label_maturity_lookup,
            universe_keys=universe_keys,
        )
        label_1230_status = (
            CapabilityStatus.AVAILABLE if afternoon_rows == len(daily) else CapabilityStatus.PARTIAL
        )
    else:
        label_1230_status = CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY
    return output.sort_values(["trading_date", "symbol"]).reset_index(drop=True), label_1230_status


def write_production_feature_snapshot(
    features: pd.DataFrame,
    destination: Path,
    *,
    manifest: FeatureSetManifest,
    source_snapshot_as_of: datetime,
    source_snapshot_ids: tuple[tuple[str, str], ...],
    as_of: datetime,
    created_at: datetime,
) -> ProductionFeatureSnapshot:
    for value in (source_snapshot_as_of, as_of, created_at):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("production feature timestamps must be timezone-aware")
    safe = features.loc[
        pd.to_datetime(features["available_at"], utc=True) <= pd.Timestamp(as_of)
    ].copy()
    revision_policy, revision_status = _revision_contract(safe)
    canonical = safe.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
    schema = _logical_frame_schema(canonical)
    metadata_identity = json.dumps(
        {
            "manifest_hash": manifest.manifest_hash,
            "feature_set_id": manifest.feature_set_id,
            "source_snapshot_as_of": source_snapshot_as_of.isoformat(),
            "source_snapshot_ids": source_snapshot_ids,
            "as_of": as_of.isoformat(),
            "historical_revision_policy": revision_policy,
            "historical_revision_status": revision_status.value,
            "schema": schema,
        },
        sort_keys=True,
    ).encode()
    content_digest = _stable_frame_content_digest(canonical)
    snapshot_id = hashlib.sha256(metadata_identity + content_digest).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_directory = destination / snapshot_id
    parquet_path = snapshot_directory / f"{snapshot_id}.parquet"
    metadata_path = snapshot_directory / f"{snapshot_id}.json"
    metadata = {
        "snapshot_id": snapshot_id,
        "created_at": created_at.isoformat(),
        "as_of": as_of.isoformat(),
        "source_snapshot_as_of": source_snapshot_as_of.isoformat(),
        "source_snapshot_ids": [list(item) for item in source_snapshot_ids],
        "feature_set_id": manifest.feature_set_id,
        "feature_set_version": manifest.feature_set_version,
        "manifest_hash": manifest.manifest_hash,
        "historical_revision_policy": revision_policy,
        "historical_revision_status": revision_status.value,
        "rows": len(safe),
        "parquet_path": str(parquet_path),
    }
    snapshot_created_at = created_at
    if snapshot_directory.exists():
        _validate_snapshot_bundle(
            snapshot_directory,
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            expected_metadata=metadata,
        )
        existing = pd.read_parquet(parquet_path)
        existing_content = _stable_frame_content_digest(
            existing.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
        )
        if hashlib.sha256(metadata_identity + existing_content).hexdigest() != snapshot_id:
            raise RuntimeError("existing production feature snapshot failed content validation")
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
        snapshot_created_at = datetime.fromisoformat(str(observed["created_at"]))
    else:
        _publish_snapshot_bundle(
            canonical,
            destination=destination,
            snapshot_id=snapshot_id,
            metadata=metadata,
        )
    return ProductionFeatureSnapshot(
        snapshot_id=snapshot_id,
        created_at=snapshot_created_at,
        as_of=as_of,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        feature_set_id=manifest.feature_set_id,
        feature_set_version=manifest.feature_set_version,
        manifest_hash=manifest.manifest_hash,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
        rows=len(safe),
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )


def load_production_feature_snapshot(
    parquet_path: Path,
) -> tuple[ProductionFeatureSnapshot, pd.DataFrame]:
    """Load and authenticate a Production Feature Snapshot bundle."""

    parquet_path = parquet_path.resolve()
    snapshot_id = parquet_path.stem
    metadata_path = parquet_path.with_suffix(".json")
    if parquet_path.parent.name != snapshot_id:
        raise RuntimeError("production feature path is not a content-addressed bundle")
    try:
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("production feature metadata is missing or invalid") from None
    if not isinstance(observed, dict) or observed.get("snapshot_id") != snapshot_id:
        raise RuntimeError("production feature snapshot identity mismatch")
    if Path(str(observed.get("parquet_path", ""))).resolve() != parquet_path:
        raise RuntimeError("production feature Parquet path metadata mismatch")
    expected = {
        key: value
        for key, value in observed.items()
        if key not in {"metadata_hash", "parquet_sha256"}
    }
    _validate_snapshot_bundle(
        parquet_path.parent,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        expected_metadata=expected,
    )
    frame = pd.read_parquet(parquet_path)
    manifests = {
        V0_MANIFEST.feature_set_id: V0_MANIFEST,
        V1_CORE_MANIFEST.feature_set_id: V1_CORE_MANIFEST,
        V2_EXTENDED_MANIFEST.feature_set_id: V2_EXTENDED_MANIFEST,
    }
    feature_set_id = str(observed["feature_set_id"])
    feature_set_version = str(observed["feature_set_version"])
    manifest = manifests.get(feature_set_id)
    if (
        manifest is None
        or feature_set_version != manifest.feature_set_version
        or observed.get("manifest_hash") != manifest.manifest_hash
    ):
        raise RuntimeError("production feature manifest is unknown or inconsistent")
    source_snapshot_as_of = datetime.fromisoformat(str(observed["source_snapshot_as_of"]))
    source_snapshot_ids = tuple(
        (str(item[0]), str(item[1])) for item in observed["source_snapshot_ids"]
    )
    as_of = datetime.fromisoformat(str(observed["as_of"]))
    revision_policy = str(observed["historical_revision_policy"])
    revision_status = CapabilityStatus(str(observed["historical_revision_status"]))
    canonical = frame.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
    schema = _logical_frame_schema(canonical)
    identity = json.dumps(
        {
            "manifest_hash": manifest.manifest_hash,
            "feature_set_id": manifest.feature_set_id,
            "source_snapshot_as_of": source_snapshot_as_of.isoformat(),
            "source_snapshot_ids": source_snapshot_ids,
            "as_of": as_of.isoformat(),
            "historical_revision_policy": revision_policy,
            "historical_revision_status": revision_status.value,
            "schema": schema,
        },
        sort_keys=True,
    ).encode()
    content_digest = _stable_frame_content_digest(canonical)
    if hashlib.sha256(identity + content_digest).hexdigest() != snapshot_id:
        raise RuntimeError("production feature content identity mismatch")
    snapshot = ProductionFeatureSnapshot(
        snapshot_id=snapshot_id,
        created_at=datetime.fromisoformat(str(observed["created_at"])),
        as_of=as_of,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        feature_set_id=feature_set_id,
        feature_set_version=feature_set_version,
        manifest_hash=manifest.manifest_hash,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
        rows=int(observed["rows"]),
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )
    if snapshot.rows != len(frame) or frame.empty:
        raise RuntimeError("production feature row-count metadata mismatch")
    return snapshot, frame


def write_production_dataset_snapshot(
    dataset: pd.DataFrame,
    destination: Path,
    *,
    source_snapshot_as_of: datetime,
    source_snapshot_ids: tuple[tuple[str, str], ...],
    as_of: datetime,
    created_at: datetime,
    label_1230_status: CapabilityStatus,
    manifests: tuple[FeatureSetManifest, ...] = (V0_MANIFEST, V1_CORE_MANIFEST),
) -> ProductionDatasetSnapshot:
    """Publish an immutable Production Research Dataset and complete audit metadata."""

    for value in (source_snapshot_as_of, as_of, created_at):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("production snapshot timestamps must be timezone-aware")
    safe = dataset.loc[pd.to_datetime(dataset["as_of"], utc=True) <= pd.Timestamp(as_of)].copy()
    revision_policy, revision_status = _revision_contract(safe)
    cutoff = pd.Timestamp(as_of)
    for horizon in HORIZONS:
        availability_column = f"label_available_at_{horizon}d"
        target_columns = [
            f"target_return_{horizon}d",
            f"target_topix_excess_{horizon}d",
            f"target_sector_excess_{horizon}d",
            f"target_beta_residual_{horizon}d",
            f"target_large_loss_{horizon}d",
        ]
        audit_columns = [
            f"label_end_date_{horizon}d",
            availability_column,
        ]
        status_columns = [
            f"label_status_{horizon}d",
            f"label_status_topix_excess_{horizon}d",
            f"label_status_sector_excess_{horizon}d",
            f"label_status_beta_residual_{horizon}d",
        ]
        _require_columns(
            safe,
            set(target_columns + audit_columns + status_columns),
            "production dataset",
        )
        label_available = pd.to_datetime(safe[availability_column], utc=True)
        not_mature = label_available.isna() | (label_available > cutoff)
        safe.loc[not_mature, target_columns] = pd.NA
        safe.loc[not_mature, status_columns] = "HORIZON_NOT_MATURE"
        exact_target = f"target_1230_return_{horizon}d"
        if exact_target in safe:
            exact_availability = f"label_1230_available_at_{horizon}d"
            exact_status = f"label_1230_status_{horizon}d"
            exact_end = f"label_1230_end_date_{horizon}d"
            _require_columns(
                safe,
                {exact_target, exact_availability, exact_status, exact_end},
                "production 12:30 labels",
            )
            exact_available = pd.to_datetime(safe[exact_availability], utc=True)
            exact_not_mature = exact_available.isna() | (exact_available > cutoff)
            safe.loc[exact_not_mature, exact_target] = pd.NA
            safe.loc[exact_not_mature, exact_status] = "HORIZON_NOT_MATURE"

    feature_manifests = tuple(
        (manifest.feature_set_version, manifest.manifest_hash) for manifest in manifests
    )
    target_definition = (
        "fixed-JPX-calendar next-session-close entry to 1/5/20d adjusted-close exit; "
        "TOPIX/sector excess and 60d causal-beta residual; entry is strictly after the "
        "11:30 feature as_of; missing suspension/delisting prices are never shifted"
    )
    snapshot_id = _production_frame_hash(
        safe,
        feature_manifests=feature_manifests,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        as_of=as_of,
        target_definition=target_definition,
        label_1230_status=label_1230_status,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
    )
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_directory = destination / snapshot_id
    parquet_path = snapshot_directory / f"{snapshot_id}.parquet"
    metadata_path = snapshot_directory / f"{snapshot_id}.json"
    data_start = str(pd.to_datetime(safe["trading_date"]).min().date())
    data_end = str(pd.to_datetime(safe["trading_date"]).max().date())
    metadata = {
        "snapshot_id": snapshot_id,
        "created_at": created_at.isoformat(),
        "as_of": as_of.isoformat(),
        "source_snapshot_as_of": source_snapshot_as_of.isoformat(),
        "source_snapshot_ids": [list(item) for item in source_snapshot_ids],
        "feature_manifests": [list(item) for item in feature_manifests],
        "target_definition": target_definition,
        "label_1230_status": label_1230_status.value,
        "historical_revision_policy": revision_policy,
        "historical_revision_status": revision_status.value,
        "rows": len(safe),
        "data_start": data_start,
        "data_end": data_end,
        "parquet_path": str(parquet_path),
    }
    snapshot_created_at = created_at
    if snapshot_directory.exists():
        _validate_snapshot_bundle(
            snapshot_directory,
            parquet_path=parquet_path,
            metadata_path=metadata_path,
            expected_metadata=metadata,
        )
        existing = pd.read_parquet(parquet_path)
        existing_id = _production_frame_hash(
            existing,
            feature_manifests=feature_manifests,
            source_snapshot_as_of=source_snapshot_as_of,
            source_snapshot_ids=source_snapshot_ids,
            as_of=as_of,
            target_definition=target_definition,
            label_1230_status=label_1230_status,
            historical_revision_policy=revision_policy,
            historical_revision_status=revision_status,
        )
        if existing_id != snapshot_id:
            raise RuntimeError("existing production snapshot failed content validation")
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        snapshot_created_at = datetime.fromisoformat(str(existing_metadata["created_at"]))
    else:
        canonical = safe.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
        _publish_snapshot_bundle(
            canonical,
            destination=destination,
            snapshot_id=snapshot_id,
            metadata=metadata,
        )
    return ProductionDatasetSnapshot(
        snapshot_id=snapshot_id,
        created_at=snapshot_created_at,
        as_of=as_of,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        feature_manifests=feature_manifests,
        target_definition=target_definition,
        label_1230_status=label_1230_status,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
        rows=len(safe),
        data_start=data_start,
        data_end=data_end,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )


def load_production_dataset_snapshot(
    parquet_path: Path,
) -> tuple[ProductionDatasetSnapshot, pd.DataFrame]:
    """Load and authenticate a Production Dataset bundle before research use."""

    parquet_path = parquet_path.resolve()
    snapshot_id = parquet_path.stem
    metadata_path = parquet_path.with_suffix(".json")
    if parquet_path.parent.name != snapshot_id:
        raise RuntimeError("production dataset path is not a content-addressed bundle")
    try:
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("production dataset metadata is missing or invalid") from None
    if not isinstance(observed, dict):
        raise RuntimeError("production dataset metadata is invalid")
    if observed.get("snapshot_id") != snapshot_id:
        raise RuntimeError("production dataset snapshot identity mismatch")
    if Path(str(observed.get("parquet_path", ""))).resolve() != parquet_path:
        raise RuntimeError("production dataset Parquet path metadata mismatch")
    expected = {
        key: value
        for key, value in observed.items()
        if key not in {"metadata_hash", "parquet_sha256"}
    }
    _validate_snapshot_bundle(
        parquet_path.parent,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        expected_metadata=expected,
    )
    frame = pd.read_parquet(parquet_path)
    feature_manifests = tuple(
        (str(item[0]), str(item[1])) for item in observed["feature_manifests"]
    )
    source_snapshot_as_of = datetime.fromisoformat(str(observed["source_snapshot_as_of"]))
    source_snapshot_ids = tuple(
        (str(item[0]), str(item[1])) for item in observed["source_snapshot_ids"]
    )
    as_of = datetime.fromisoformat(str(observed["as_of"]))
    label_status = CapabilityStatus(str(observed["label_1230_status"]))
    revision_policy = str(observed["historical_revision_policy"])
    revision_status = CapabilityStatus(str(observed["historical_revision_status"]))
    recomputed = _production_frame_hash(
        frame,
        feature_manifests=feature_manifests,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        as_of=as_of,
        target_definition=str(observed["target_definition"]),
        label_1230_status=label_status,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
    )
    if recomputed != snapshot_id:
        raise RuntimeError("production dataset content identity mismatch")
    snapshot = ProductionDatasetSnapshot(
        snapshot_id=snapshot_id,
        created_at=datetime.fromisoformat(str(observed["created_at"])),
        as_of=as_of,
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        feature_manifests=feature_manifests,
        target_definition=str(observed["target_definition"]),
        label_1230_status=label_status,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
        rows=int(observed["rows"]),
        data_start=str(observed["data_start"]),
        data_end=str(observed["data_end"]),
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )
    if snapshot.rows != len(frame):
        raise RuntimeError("production dataset row-count metadata mismatch")
    if frame.empty:
        raise RuntimeError("production dataset cannot be empty")
    observed_start = str(pd.to_datetime(frame["trading_date"]).min().date())
    observed_end = str(pd.to_datetime(frame["trading_date"]).max().date())
    if observed_start != snapshot.data_start or observed_end != snapshot.data_end:
        raise RuntimeError("production dataset date-range metadata mismatch")
    return snapshot, frame


def write_production_build_manifest(
    v0: ProductionFeatureSnapshot,
    v1: ProductionFeatureSnapshot,
    v2: ProductionFeatureSnapshot,
    dataset: ProductionDatasetSnapshot,
    destination: Path,
    *,
    created_at: datetime,
) -> ProductionBuildManifest:
    """Publish the final atomic marker only after all four snapshots authenticate."""

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("production build created_at must be timezone-aware")
    if v0.feature_set_id != V0_MANIFEST.feature_set_id:
        raise ValueError("production build V0 manifest mismatch")
    if v1.feature_set_id != V1_CORE_MANIFEST.feature_set_id:
        raise ValueError("production build V1 manifest mismatch")
    if v2.feature_set_id != V2_EXTENDED_MANIFEST.feature_set_id:
        raise ValueError("production build V2 manifest mismatch")
    snapshots = (v0, v1, v2)
    if any(
        item.source_snapshot_as_of != dataset.source_snapshot_as_of for item in snapshots
    ) or any(item.source_snapshot_ids != dataset.source_snapshot_ids for item in snapshots):
        raise ValueError("production build snapshots do not share one source lineage")
    _validate_build_observation_contract(v0, v1, v2, dataset, error_type=ValueError)
    identity = {
        "source_snapshot_as_of": dataset.source_snapshot_as_of.isoformat(),
        "source_snapshot_ids": dataset.source_snapshot_ids,
        "v0_snapshot_id": v0.snapshot_id,
        "v1_snapshot_id": v1.snapshot_id,
        "v2_snapshot_id": v2.snapshot_id,
        "dataset_snapshot_id": dataset.snapshot_id,
    }
    required_dataset_manifests = {
        (V0_MANIFEST.feature_set_version, V0_MANIFEST.manifest_hash),
        (V1_CORE_MANIFEST.feature_set_version, V1_CORE_MANIFEST.manifest_hash),
        (V2_EXTENDED_MANIFEST.feature_set_version, V2_EXTENDED_MANIFEST.manifest_hash),
    }
    if set(dataset.feature_manifests) != required_dataset_manifests:
        raise ValueError("production dataset does not authenticate the exact V0/V1/V2 manifests")
    _authenticate_build_snapshot_paths(
        v0,
        v1,
        v2,
        dataset,
        error_type=ValueError,
    )
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_directory = destination / build_id
    manifest_path = final_directory / f"{build_id}.json"
    metadata = {
        "build_id": build_id,
        "created_at": created_at.isoformat(),
        **identity,
        "source_snapshot_ids": [list(item) for item in dataset.source_snapshot_ids],
        "v0_parquet_path": str(v0.parquet_path.resolve()),
        "v1_parquet_path": str(v1.parquet_path.resolve()),
        "v2_parquet_path": str(v2.parquet_path.resolve()),
        "dataset_parquet_path": str(dataset.parquet_path.resolve()),
        "manifest_path": str(manifest_path),
    }
    metadata["metadata_hash"] = _metadata_hash(metadata)
    if final_directory.exists():
        return load_production_build_manifest(manifest_path)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".tmp-{build_id[:12]}-", dir=destination)
    ).resolve()
    temporary_manifest = temporary_directory / f"{build_id}.json"
    try:
        with temporary_manifest.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary_directory, final_directory)
        except OSError:
            if not final_directory.exists():
                raise
            _remove_snapshot_temporary(temporary_directory, destination)
    except Exception:
        if temporary_directory.exists():
            _remove_snapshot_temporary(temporary_directory, destination)
        raise
    return load_production_build_manifest(manifest_path)


def load_production_build_manifest(manifest_path: Path) -> ProductionBuildManifest:
    """Authenticate one build marker and every snapshot it publishes."""

    manifest_path = manifest_path.resolve()
    build_id = manifest_path.stem
    if manifest_path.parent.name != build_id:
        raise RuntimeError("production build path is not content-addressed")
    try:
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("production build manifest is missing or invalid") from None
    if not isinstance(observed, dict) or observed.get("build_id") != build_id:
        raise RuntimeError("production build identity mismatch")
    required_fields = {
        "source_snapshot_as_of",
        "source_snapshot_ids",
        "v0_snapshot_id",
        "v1_snapshot_id",
        "v2_snapshot_id",
        "dataset_snapshot_id",
        "v0_parquet_path",
        "v1_parquet_path",
        "v2_parquet_path",
        "dataset_parquet_path",
        "manifest_path",
        "metadata_hash",
    }
    if required_fields - set(observed):
        raise RuntimeError("production build manifest is incomplete")
    if Path(str(observed["manifest_path"])).resolve() != manifest_path:
        raise RuntimeError("production build manifest path metadata mismatch")
    if observed.get("metadata_hash") != _metadata_hash(observed):
        raise RuntimeError("production build metadata hash mismatch")
    source_snapshot_ids = tuple(
        (str(item[0]), str(item[1])) for item in observed["source_snapshot_ids"]
    )
    identity = {
        "source_snapshot_as_of": str(observed["source_snapshot_as_of"]),
        "source_snapshot_ids": source_snapshot_ids,
        "v0_snapshot_id": str(observed["v0_snapshot_id"]),
        "v1_snapshot_id": str(observed["v1_snapshot_id"]),
        "v2_snapshot_id": str(observed["v2_snapshot_id"]),
        "dataset_snapshot_id": str(observed["dataset_snapshot_id"]),
    }
    recomputed = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if recomputed != build_id:
        raise RuntimeError("production build content identity mismatch")
    v0_path = Path(str(observed["v0_parquet_path"]))
    v1_path = Path(str(observed["v1_parquet_path"]))
    v2_path = Path(str(observed["v2_parquet_path"]))
    dataset_path = Path(str(observed["dataset_parquet_path"]))
    v0 = _feature_snapshot_from_metadata(v0_path)
    v1 = _feature_snapshot_from_metadata(v1_path)
    v2 = _feature_snapshot_from_metadata(v2_path)
    dataset = _dataset_snapshot_from_metadata(dataset_path)
    if (v0.snapshot_id, v1.snapshot_id, v2.snapshot_id, dataset.snapshot_id) != (
        identity["v0_snapshot_id"],
        identity["v1_snapshot_id"],
        identity["v2_snapshot_id"],
        identity["dataset_snapshot_id"],
    ):
        raise RuntimeError("production build references the wrong snapshot identity")
    if (
        v0.feature_set_id != V0_MANIFEST.feature_set_id
        or v0.manifest_hash != V0_MANIFEST.manifest_hash
        or v1.feature_set_id != V1_CORE_MANIFEST.feature_set_id
        or v1.manifest_hash != V1_CORE_MANIFEST.manifest_hash
        or v2.feature_set_id != V2_EXTENDED_MANIFEST.feature_set_id
        or v2.manifest_hash != V2_EXTENDED_MANIFEST.manifest_hash
    ):
        raise RuntimeError("production build feature-set role mismatch")
    if not (
        v0.source_snapshot_ids
        == v1.source_snapshot_ids
        == v2.source_snapshot_ids
        == dataset.source_snapshot_ids
        == source_snapshot_ids
    ):
        raise RuntimeError("production build snapshot lineage mismatch")
    source_snapshot_as_of = datetime.fromisoformat(str(observed["source_snapshot_as_of"]))
    if not (
        v0.source_snapshot_as_of
        == v1.source_snapshot_as_of
        == v2.source_snapshot_as_of
        == dataset.source_snapshot_as_of
        == source_snapshot_as_of
    ):
        raise RuntimeError("production build source cutoff mismatch")
    _validate_build_observation_contract(v0, v1, v2, dataset, error_type=RuntimeError)
    required_dataset_manifests = {
        (V0_MANIFEST.feature_set_version, V0_MANIFEST.manifest_hash),
        (V1_CORE_MANIFEST.feature_set_version, V1_CORE_MANIFEST.manifest_hash),
        (V2_EXTENDED_MANIFEST.feature_set_version, V2_EXTENDED_MANIFEST.manifest_hash),
    }
    if set(dataset.feature_manifests) != required_dataset_manifests:
        raise RuntimeError("production build dataset feature lineage mismatch")
    _authenticate_build_snapshot_paths(
        v0,
        v1,
        v2,
        dataset,
        error_type=RuntimeError,
    )
    return ProductionBuildManifest(
        build_id=build_id,
        created_at=datetime.fromisoformat(str(observed["created_at"])),
        source_snapshot_as_of=source_snapshot_as_of,
        source_snapshot_ids=source_snapshot_ids,
        v0_snapshot_id=v0.snapshot_id,
        v1_snapshot_id=v1.snapshot_id,
        v2_snapshot_id=v2.snapshot_id,
        dataset_snapshot_id=dataset.snapshot_id,
        v0_parquet_path=v0.parquet_path,
        v1_parquet_path=v1.parquet_path,
        v2_parquet_path=v2.parquet_path,
        dataset_parquet_path=dataset.parquet_path,
        manifest_path=manifest_path,
    )


def _validate_build_observation_contract(
    v0: ProductionFeatureSnapshot,
    v1: ProductionFeatureSnapshot,
    v2: ProductionFeatureSnapshot,
    dataset: ProductionDatasetSnapshot,
    *,
    error_type: type[ValueError] | type[RuntimeError],
) -> None:
    artifacts = (v0, v1, v2, dataset)
    if any(item.as_of != dataset.as_of for item in artifacts[:-1]):
        raise error_type("production build snapshots do not share one observation as_of")
    if any(
        item.historical_revision_policy != dataset.historical_revision_policy
        for item in artifacts[:-1]
    ):
        raise error_type("production build snapshots do not share one revision policy")
    if any(
        item.historical_revision_status != dataset.historical_revision_status
        for item in artifacts[:-1]
    ):
        raise error_type("production build snapshots do not share one revision status")


def _feature_snapshot_from_metadata(parquet_path: Path) -> ProductionFeatureSnapshot:
    parquet_path = parquet_path.resolve()
    observed = json.loads(parquet_path.with_suffix(".json").read_text(encoding="utf-8"))
    return ProductionFeatureSnapshot(
        snapshot_id=str(observed["snapshot_id"]),
        created_at=datetime.fromisoformat(str(observed["created_at"])),
        as_of=datetime.fromisoformat(str(observed["as_of"])),
        source_snapshot_as_of=datetime.fromisoformat(str(observed["source_snapshot_as_of"])),
        source_snapshot_ids=tuple(
            (str(item[0]), str(item[1])) for item in observed["source_snapshot_ids"]
        ),
        feature_set_id=str(observed["feature_set_id"]),
        feature_set_version=str(observed["feature_set_version"]),
        manifest_hash=str(observed["manifest_hash"]),
        historical_revision_policy=str(observed["historical_revision_policy"]),
        historical_revision_status=CapabilityStatus(
            str(observed["historical_revision_status"])
        ),
        rows=int(observed["rows"]),
        parquet_path=parquet_path,
        metadata_path=parquet_path.with_suffix(".json"),
    )


def _dataset_snapshot_from_metadata(parquet_path: Path) -> ProductionDatasetSnapshot:
    parquet_path = parquet_path.resolve()
    observed = json.loads(parquet_path.with_suffix(".json").read_text(encoding="utf-8"))
    return ProductionDatasetSnapshot(
        snapshot_id=str(observed["snapshot_id"]),
        created_at=datetime.fromisoformat(str(observed["created_at"])),
        as_of=datetime.fromisoformat(str(observed["as_of"])),
        source_snapshot_as_of=datetime.fromisoformat(str(observed["source_snapshot_as_of"])),
        source_snapshot_ids=tuple(
            (str(item[0]), str(item[1])) for item in observed["source_snapshot_ids"]
        ),
        feature_manifests=tuple(
            (str(item[0]), str(item[1])) for item in observed["feature_manifests"]
        ),
        target_definition=str(observed["target_definition"]),
        label_1230_status=CapabilityStatus(str(observed["label_1230_status"])),
        historical_revision_policy=str(observed["historical_revision_policy"]),
        historical_revision_status=CapabilityStatus(
            str(observed["historical_revision_status"])
        ),
        rows=int(observed["rows"]),
        data_start=str(observed["data_start"]),
        data_end=str(observed["data_end"]),
        parquet_path=parquet_path,
        metadata_path=parquet_path.with_suffix(".json"),
    )


def _authenticate_build_snapshot_paths(
    v0: ProductionFeatureSnapshot,
    v1: ProductionFeatureSnapshot,
    v2: ProductionFeatureSnapshot,
    dataset: ProductionDatasetSnapshot,
    *,
    error_type: type[ValueError] | type[RuntimeError],
) -> None:
    for expected in (v0, v1, v2):
        try:
            loaded, frame = load_production_feature_snapshot(expected.parquet_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise error_type(str(exc)) from None
        if loaded.snapshot_id != expected.snapshot_id:
            raise error_type("production build feature path identity mismatch")
        del frame
    try:
        loaded_dataset, dataset_frame = load_production_dataset_snapshot(dataset.parquet_path)
    except (OSError, ValueError, RuntimeError) as exc:
        raise error_type(str(exc)) from None
    if loaded_dataset.snapshot_id != dataset.snapshot_id:
        raise error_type("production build dataset path identity mismatch")
    del dataset_frame
    _validate_build_feature_parquet_content(v0, v1, v2, dataset, error_type=error_type)


def _validate_build_feature_parquet_content(
    v0: ProductionFeatureSnapshot,
    v1: ProductionFeatureSnapshot,
    v2: ProductionFeatureSnapshot,
    dataset: ProductionDatasetSnapshot,
    *,
    error_type: type[ValueError] | type[RuntimeError],
    chunk_size: int = 8,
) -> None:
    keys = ["symbol", "trading_date"]
    for snapshot, manifest in (
        (v0, V0_MANIFEST),
        (v1, V1_CORE_MANIFEST),
        (v2, V2_EXTENDED_MANIFEST),
    ):
        try:
            observed_keys = pd.read_parquet(snapshot.parquet_path, columns=keys)
            embedded_keys = pd.read_parquet(dataset.parquet_path, columns=keys)
            pd.testing.assert_frame_equal(
                observed_keys,
                embedded_keys,
                check_dtype=False,
                check_exact=True,
            )
            del observed_keys, embedded_keys
            names = manifest.feature_names
            for start in range(0, len(names), chunk_size):
                columns = list(names[start : start + chunk_size])
                observed = pd.read_parquet(snapshot.parquet_path, columns=columns)
                embedded = pd.read_parquet(dataset.parquet_path, columns=columns)
                pd.testing.assert_frame_equal(
                    observed,
                    embedded,
                    check_dtype=False,
                    check_exact=True,
                )
                del observed, embedded
        except (AssertionError, OSError, ValueError) as exc:
            raise error_type(
                f"production dataset feature values do not match {manifest.feature_set_id}: {exc}"
            ) from None


def _validate_build_feature_content(
    v0: pd.DataFrame,
    v1: pd.DataFrame,
    v2: pd.DataFrame,
    dataset: pd.DataFrame,
    *,
    error_type: type[ValueError] | type[RuntimeError],
) -> None:
    for frame, manifest in (
        (v0, V0_MANIFEST),
        (v1, V1_CORE_MANIFEST),
        (v2, V2_EXTENDED_MANIFEST),
    ):
        columns = ["symbol", "trading_date", *manifest.feature_names]
        if missing := set(columns) - set(frame.columns):
            raise error_type(
                f"production {manifest.feature_set_id} snapshot is missing columns: "
                f"{sorted(missing)}"
            )
        if missing := set(columns) - set(dataset.columns):
            raise error_type(
                f"production dataset is missing {manifest.feature_set_id} columns: "
                f"{sorted(missing)}"
            )
        observed = frame.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
        embedded = dataset.loc[:, columns].sort_values(columns[:2]).reset_index(drop=True)
        try:
            pd.testing.assert_frame_equal(
                observed,
                embedded,
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError:
            raise error_type(
                f"production dataset feature values do not match {manifest.feature_set_id} snapshot"
            ) from None


def run_production_walk_forward_baselines(
    dataset: pd.DataFrame,
    *,
    data_snapshot_id: str,
    created_at: datetime,
    code_commit: str,
    feature_names: tuple[str, ...] = V1_CORE_MANIFEST.feature_names,
    feature_set_version: str = V1_CORE_MANIFEST.feature_set_version,
    target_column: str = "target_return_5d",
    initial_train_periods: int = 500,
    validation_periods: int = 60,
    step_periods: int = 60,
    holdout_periods: int = 120,
) -> ProductionBaselineReport:
    """Evaluate CASH, Momentum, and Ridge without exposing the locked holdout rows."""

    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("baseline created_at must be timezone-aware")
    if not code_commit.strip() or code_commit == "UNSET":
        raise ValueError("baseline code_commit must be explicit")
    revision_policy, revision_status = _revision_contract(dataset)
    label_horizon, label_status_column = _target_contract(target_column)
    label_end_column = f"label_end_date_{label_horizon}d"
    required = {
        "trading_date",
        target_column,
        label_end_column,
        label_status_column,
        *feature_names,
    }
    _require_columns(dataset, required, "production baseline dataset")
    label_status_counts = tuple(
        (str(status), int(count))
        for status, count in dataset[label_status_column]
        .fillna("MISSING_STATUS")
        .value_counts()
        .sort_index()
        .items()
    )
    usable = dataset.loc[
        dataset[target_column].notna() & dataset[label_status_column].eq("AVAILABLE")
    ].copy()
    if usable.empty:
        raise ValueError("BLOCKED_BY_VALIDATION: production target has no available rows")
    holdout = reserve_locked_final_holdout(usable, holdout_periods=holdout_periods)
    development = usable.iloc[list(holdout.development_indices)].copy()
    development = development.loc[
        pd.to_datetime(development[label_end_column]) < holdout.holdout_start
    ].copy()
    if development.empty:
        raise ValueError("BLOCKED_BY_VALIDATION: holdout purge removed all development labels")
    splitter = PurgedExpandingWindowSplitter(
        initial_train_periods=initial_train_periods,
        validation_periods=validation_periods,
        step_periods=step_periods,
        purge_periods=0,
        embargo_periods=label_horizon,
        label_horizon_periods=label_horizon,
    )
    factories: tuple[tuple[str, Callable[[], Regressor]], ...] = (
        ("CASH", lambda: _ZeroRegressor()),
        ("Momentum", lambda: MomentumRegressor()),
        ("Ridge", lambda: RidgeRegressor(alpha=5.0)),
    )
    summaries = tuple(
        _evaluate_baseline(
            development,
            model_name=name,
            model_factory=factory,
            feature_names=feature_names,
            target_column=target_column,
            label_end_column=label_end_column,
            splitter=splitter,
        )
        for name, factory in factories
    )
    config = {
        "data_snapshot_id": data_snapshot_id,
        "feature_set_version": feature_set_version,
        "feature_names": feature_names,
        "target_column": target_column,
        "initial_train_periods": initial_train_periods,
        "validation_periods": validation_periods,
        "step_periods": step_periods,
        "holdout_periods": holdout_periods,
        "embargo_periods": label_horizon,
        "ridge_alpha": 5.0,
        "cost_scenarios_bps": (10, 20, 30, 50),
        "historical_revision_policy": revision_policy,
        "historical_revision_status": revision_status.value,
    }
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity = {
        "config_hash": config_hash,
        "code_commit": code_commit,
        "locked_holdout_start": str(holdout.holdout_start.date()),
        "models": [summary.model_dump(mode="json") for summary in summaries],
        "label_status_counts": label_status_counts,
    }
    report_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ProductionBaselineReport(
        report_id=report_id,
        created_at=created_at,
        code_commit=code_commit,
        config_hash=config_hash,
        data_snapshot_id=data_snapshot_id,
        feature_set_version=feature_set_version,
        target_column=target_column,
        locked_holdout_start=str(holdout.holdout_start.date()),
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
        adoption_eligible=revision_status is CapabilityStatus.AVAILABLE,
        adoption_blocking_reason=(
            None
            if revision_status is CapabilityStatus.AVAILABLE
            else "historical provider revision vintages are incomplete; research-only result"
        ),
        label_status_counts=label_status_counts,
        models=summaries,
    )


def write_production_baseline_report(
    report: ProductionBaselineReport,
    destination: Path,
) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{report.report_id}.json"
    payload = report.model_dump_json(indent=2)
    if path.exists():
        try:
            existing = ProductionBaselineReport.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError:
            raise RuntimeError("existing baseline report is invalid") from None
        current_identity = report.model_dump(mode="json", exclude={"created_at"})
        existing_identity = existing.model_dump(mode="json", exclude={"created_at"})
        if existing_identity != current_identity:
            raise RuntimeError("existing baseline report identity collision")
        return path
    temporary = destination / f".{report.report_id}.{uuid4().hex}.json.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    return path


class _ZeroRegressor:
    def fit(self, features: pd.DataFrame, target: pd.Series) -> _ZeroRegressor:
        del features, target
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(features), dtype=float)


def _evaluate_baseline(
    frame: pd.DataFrame,
    *,
    model_name: str,
    model_factory: Callable[[], Regressor],
    feature_names: tuple[str, ...],
    target_column: str,
    label_end_column: str,
    splitter: PurgedExpandingWindowSplitter,
) -> BaselineModelSummary:
    squared_errors: list[float] = []
    daily_rank_ics: list[float] = []
    selected_returns: list[float] = []
    validation_rows = 0
    fold_count = 0
    for fold in splitter.split(frame, label_end_column=label_end_column):
        train = frame.iloc[list(fold.train_indices)]
        validation = frame.iloc[list(fold.validation_indices)]
        if train.empty or validation.empty:
            continue
        model = model_factory().fit(train[list(feature_names)], train[target_column])
        prediction = model.predict(validation[list(feature_names)])
        target = validation[target_column].to_numpy(dtype=float)
        squared_errors.extend(np.square(target - prediction).tolist())
        validation_rows += len(validation)
        fold_count += 1
        ranked = pd.DataFrame(
            {
                "trading_date": pd.to_datetime(validation["trading_date"]).to_numpy(),
                "prediction": prediction,
                "target": target,
            }
        )
        for _, group in ranked.groupby("trading_date", sort=True):
            if (
                len(group) >= 2
                and group["prediction"].nunique() >= 2
                and group["target"].nunique() >= 2
            ):
                rank_ic = group["prediction"].corr(group["target"], method="spearman")
                if pd.notna(rank_ic):
                    daily_rank_ics.append(float(rank_ic))
            if model_name == "CASH":
                selected_returns.append(0.0)
                continue
            selected_count = max(1, int(np.ceil(len(group) * 0.10)))
            selected_returns.extend(
                group.nlargest(selected_count, "prediction")["target"].astype(float).tolist()
            )
    if fold_count == 0 or not squared_errors or not selected_returns:
        raise ValueError(f"BLOCKED_BY_VALIDATION: no usable {model_name} walk-forward folds")
    gross = float(np.mean(selected_returns))
    cost_scenarios = tuple(
        (bps, 0.0 if model_name == "CASH" else gross - bps / 10_000) for bps in (10, 20, 30, 50)
    )
    return BaselineModelSummary(
        model_name=model_name,
        folds=fold_count,
        validation_rows=validation_rows,
        mean_squared_error=float(np.mean(squared_errors)),
        mean_daily_rank_ic=float(np.mean(daily_rank_ics)) if daily_rank_ics else None,
        rank_ic_dates=len(daily_rank_ics),
        top_decile_mean_target=gross,
        cost_scenarios=cost_scenarios,
    )


def _add_horizon_labels(
    frame: pd.DataFrame,
    *,
    horizon: int,
    calendar: pd.DatetimeIndex,
    date_position: dict[pd.Timestamp, int],
    daily: pd.DataFrame,
    topix_lookup: pd.Series,
    label_maturity_lookup: pd.Series,
    sector_index: pd.DataFrame,
    universe_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    output = frame.copy()
    entry_column = "label_entry_date"
    end_column = f"label_end_date_{horizon}d"
    output[entry_column] = [
        calendar[position + 1]
        if (position := date_position.get(pd.Timestamp(value))) is not None
        and position + 1 < len(calendar)
        else pd.NaT
        for value in output["trading_date"]
    ]
    output[end_column] = [
        calendar[position + 1 + horizon]
        if (position := date_position.get(pd.Timestamp(value))) is not None
        and position + 1 + horizon < len(calendar)
        else pd.NaT
        for value in output["trading_date"]
    ]
    output[f"label_available_at_{horizon}d"] = output[end_column].map(label_maturity_lookup)
    entry = daily[["symbol", "trading_date", "adjusted_close"]].rename(
        columns={
            "trading_date": entry_column,
            "adjusted_close": "__entry_close",
        }
    )
    output = output.merge(entry, on=["symbol", entry_column], how="left", validate="many_to_one")
    future = daily[["symbol", "trading_date", "adjusted_close", "available_at"]].rename(
        columns={
            "trading_date": end_column,
            "adjusted_close": "__future_close",
            "available_at": "__future_price_available_at",
        }
    )
    output = output.merge(future, on=["symbol", end_column], how="left", validate="many_to_one")
    target = output["__future_close"] / output["__entry_close"] - 1
    output[f"target_return_{horizon}d"] = target
    start_topix = output[entry_column].map(topix_lookup)
    end_topix = output[end_column].map(topix_lookup)
    topix_return = end_topix / start_topix - 1
    output[f"target_topix_excess_{horizon}d"] = target - topix_return

    sector_start = sector_index.rename(
        columns={"trading_date": "__start_date", "index_level": "__sector_start"}
    )
    sector_end = sector_index.rename(
        columns={
            "trading_date": end_column,
            "index_level": "__sector_end",
            "missing_count": "__sector_missing_end",
        }
    )
    sector_start = sector_start[["sector", "__start_date", "__sector_start", "missing_count"]]
    sector_start = sector_start.rename(columns={"missing_count": "__sector_missing_start"})
    output["__start_date"] = output[entry_column]
    output = output.merge(
        sector_start,
        on=["sector", "__start_date"],
        how="left",
        validate="many_to_one",
    )
    output = output.merge(
        sector_end[["sector", end_column, "__sector_end", "__sector_missing_end"]],
        on=["sector", end_column],
        how="left",
        validate="many_to_one",
    )
    sector_return = output["__sector_end"] / output["__sector_start"] - 1
    sector_missing = (output["__sector_missing_end"] - output["__sector_missing_start"]) > 0
    sector_return = sector_return.mask(sector_missing)
    output[f"target_sector_excess_{horizon}d"] = target - sector_return
    output[f"target_beta_residual_{horizon}d"] = target - output["beta_60d"] * topix_return
    large_loss = (target <= -0.10).astype("Int8")
    output[f"target_large_loss_{horizon}d"] = large_loss.mask(target.isna())

    status = pd.Series("AVAILABLE", index=output.index, dtype="string")
    status = status.mask(output[end_column].isna(), "HORIZON_NOT_MATURE")
    membership_at_entry = [
        (str(symbol), pd.Timestamp(entry_date)) in universe_keys if pd.notna(entry_date) else False
        for symbol, entry_date in zip(output["symbol"], output[entry_column], strict=True)
    ]
    missing_entry = output[entry_column].notna() & output["__entry_close"].isna()
    status = status.mask(missing_entry & ~pd.Series(membership_at_entry), "DELISTED_NO_ENTRY_PRICE")
    status = status.mask(missing_entry & pd.Series(membership_at_entry), "SUSPENDED_NO_ENTRY_PRICE")
    membership_at_end = [
        (str(symbol), pd.Timestamp(end_date)) in universe_keys if pd.notna(end_date) else False
        for symbol, end_date in zip(output["symbol"], output[end_column], strict=True)
    ]
    missing_exit = output[end_column].notna() & output["__future_close"].isna() & ~missing_entry
    status = status.mask(missing_exit & ~pd.Series(membership_at_end), "DELISTED_NO_EXIT_PRICE")
    status = status.mask(missing_exit & pd.Series(membership_at_end), "SUSPENDED_NO_EXIT_PRICE")
    output[f"label_status_{horizon}d"] = status
    topix_status = status.mask(
        status.eq("AVAILABLE") & topix_return.isna(),
        "MISSING_TOPIX_CONTEXT",
    )
    sector_status = status.mask(
        status.eq("AVAILABLE") & sector_return.isna(),
        "MISSING_SECTOR_CONTEXT",
    )
    beta_status = status.mask(
        status.eq("AVAILABLE") & (topix_return.isna() | output["beta_60d"].isna()),
        "MISSING_BETA_HISTORY",
    )
    output[f"label_status_topix_excess_{horizon}d"] = topix_status
    output[f"label_status_sector_excess_{horizon}d"] = sector_status
    output[f"label_status_beta_residual_{horizon}d"] = beta_status
    return output.drop(
        columns=[
            "__future_close",
            "__entry_close",
            "__future_price_available_at",
            "__start_date",
            "__sector_start",
            "__sector_end",
            "__sector_missing_start",
            "__sector_missing_end",
        ]
    )


def _sector_cumulative_index(sector_context: pd.DataFrame) -> pd.DataFrame:
    frame = sector_context[["sector", "trading_date", "sector_return_1d"]].copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    parts: list[pd.DataFrame] = []
    for _sector, group in frame.groupby("sector", sort=True):
        group = group.sort_values("trading_date").copy()
        missing = group["sector_return_1d"].isna()
        group["index_level"] = (1 + group["sector_return_1d"].fillna(0)).cumprod()
        group["missing_count"] = missing.cumsum()
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def _causal_beta(daily: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    topix = market.sort_values("trading_date").copy()
    topix["__market_return"] = topix["topix_close"].pct_change(fill_method=None)
    joined = daily[["symbol", "trading_date", "trading_session_index", "adjusted_close"]].merge(
        topix[["trading_date", "__market_return"]],
        on="trading_date",
        how="left",
        validate="many_to_one",
    )
    joined["__stock_return"] = joined.groupby("symbol", sort=False)["adjusted_close"].pct_change(
        fill_method=None
    )
    contiguous = joined.groupby("symbol", sort=False)["trading_session_index"].diff().eq(1)
    joined.loc[~contiguous, "__stock_return"] = np.nan
    parts: list[pd.DataFrame] = []
    for _symbol, group in joined.groupby("symbol", sort=True):
        group = group.sort_values("trading_date").copy()
        covariance = (
            group["__stock_return"].rolling(60, min_periods=60).cov(group["__market_return"])
        )
        variance = group["__market_return"].rolling(60, min_periods=60).var(ddof=1)
        group["beta_60d"] = covariance / variance
        group.loc[~np.isfinite(group["beta_60d"]), "beta_60d"] = np.nan
        parts.append(group[["symbol", "trading_date", "beta_60d"]])
    return pd.concat(parts, ignore_index=True)


def _add_exact_afternoon_open_labels(
    frame: pd.DataFrame,
    *,
    daily: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    date_position: dict[pd.Timestamp, int],
    label_maturity_lookup: pd.Series,
    universe_keys: set[tuple[str, pd.Timestamp]],
) -> pd.DataFrame:
    output = frame.copy()
    output["entry_date_1230"] = [
        calendar[position + 1]
        if (position := date_position.get(pd.Timestamp(value))) is not None
        and position + 1 < len(calendar)
        else pd.NaT
        for value in output["trading_date"]
    ]
    entry = daily[["symbol", "trading_date", "research_afternoon_open"]].rename(
        columns={"trading_date": "entry_date_1230", "research_afternoon_open": "entry_price_1230"}
    )
    output = output.merge(entry, on=["symbol", "entry_date_1230"], how="left")
    close_lookup = daily.set_index(["symbol", "trading_date"])["adjusted_close"]
    for horizon in HORIZONS:
        end_column = f"label_1230_end_date_{horizon}d"
        output[end_column] = [
            calendar[position + horizon]
            if pd.notna(entry_date)
            and (position := date_position.get(pd.Timestamp(entry_date))) is not None
            and position + horizon < len(calendar)
            else pd.NaT
            for entry_date in output["entry_date_1230"]
        ]
        exit_prices = [
            close_lookup.get((str(symbol), pd.Timestamp(exit_date)), np.nan)
            if pd.notna(exit_date)
            else np.nan
            for symbol, exit_date in zip(output["symbol"], output[end_column], strict=True)
        ]
        target_column = f"target_1230_return_{horizon}d"
        output[target_column] = (
            pd.Series(exit_prices, index=output.index) / output["entry_price_1230"] - 1
        )
        output[f"label_1230_available_at_{horizon}d"] = output[end_column].map(
            label_maturity_lookup
        )
        status = pd.Series("AVAILABLE", index=output.index, dtype="string")
        status = status.mask(output[end_column].isna(), "HORIZON_NOT_MATURE")
        entry_member = pd.Series(
            [
                (str(symbol), pd.Timestamp(entry_date)) in universe_keys
                if pd.notna(entry_date)
                else False
                for symbol, entry_date in zip(
                    output["symbol"], output["entry_date_1230"], strict=True
                )
            ],
            index=output.index,
        )
        missing_entry = output["entry_date_1230"].notna() & output["entry_price_1230"].isna()
        status = status.mask(missing_entry & ~entry_member, "DELISTED_NO_ENTRY_PRICE")
        status = status.mask(missing_entry & entry_member, "SUSPENDED_NO_ENTRY_PRICE")
        end_member = pd.Series(
            [
                (str(symbol), pd.Timestamp(end_date)) in universe_keys
                if pd.notna(end_date)
                else False
                for symbol, end_date in zip(output["symbol"], output[end_column], strict=True)
            ],
            index=output.index,
        )
        missing_exit = (
            output[end_column].notna()
            & pd.Series(exit_prices, index=output.index).isna()
            & ~missing_entry
        )
        status = status.mask(missing_exit & ~end_member, "DELISTED_NO_EXIT_PRICE")
        status = status.mask(missing_exit & end_member, "SUSPENDED_NO_EXIT_PRICE")
        output[f"label_1230_status_{horizon}d"] = status
    return output


def _production_frame_hash(
    frame: pd.DataFrame,
    *,
    feature_manifests: tuple[tuple[str, str], ...],
    source_snapshot_as_of: datetime,
    source_snapshot_ids: tuple[tuple[str, str], ...],
    as_of: datetime,
    target_definition: str,
    label_1230_status: CapabilityStatus,
    historical_revision_policy: str,
    historical_revision_status: CapabilityStatus,
) -> str:
    canonical = frame.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
    content_digest = _stable_frame_content_digest(canonical)
    schema = _logical_frame_schema(canonical)
    metadata = json.dumps(
        {
            "feature_manifests": feature_manifests,
            "source_snapshot_as_of": source_snapshot_as_of.isoformat(),
            "source_snapshot_ids": source_snapshot_ids,
            "as_of": as_of.isoformat(),
            "target_definition": target_definition,
            "label_1230_status": label_1230_status.value,
            "historical_revision_policy": historical_revision_policy,
            "historical_revision_status": historical_revision_status.value,
            "schema": schema,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(metadata + content_digest).hexdigest()


def _logical_frame_schema(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple((str(column), _logical_series_kind(frame[column])) for column in frame.columns)


def _logical_series_kind(series: pd.Series) -> str:
    dtype = series.dtype
    if isinstance(dtype, pd.DatetimeTZDtype):
        return "datetime64[ns, UTC]"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime64[ns]"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "timedelta64[ns]"
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean"
    if pd.api.types.is_integer_dtype(dtype):
        return "integer64"
    if pd.api.types.is_float_dtype(dtype):
        return "float64"
    return "string"


def _stable_frame_content_digest(frame: pd.DataFrame) -> bytes:
    """Hash logical values independently of Parquet/Pandas physical dtypes."""

    digest = hashlib.sha256()
    schema = _logical_frame_schema(frame)
    digest.update(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    digest.update(len(frame).to_bytes(8, "little", signed=False))
    for column, kind in schema:
        series = frame[column]
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        digest.update(kind.encode("ascii"))
        digest.update(b"\0")
        if kind == "datetime64[ns, UTC]":
            utc_values = (
                pd.to_datetime(series, utc=True, errors="coerce")
                .astype("int64")
                .to_numpy(dtype="<i8", copy=True)
            )
            digest.update(utc_values.tobytes())
        elif kind == "datetime64[ns]":
            datetime_values = (
                pd.to_datetime(series, errors="coerce")
                .astype("datetime64[ns]")
                .astype("int64")
                .to_numpy(dtype="<i8", copy=True)
            )
            digest.update(datetime_values.tobytes())
        elif kind == "timedelta64[ns]":
            timedelta_values = (
                pd.to_timedelta(series, errors="coerce")
                .astype("timedelta64[ns]")
                .astype("int64")
                .to_numpy(dtype="<i8", copy=True)
            )
            digest.update(timedelta_values.tobytes())
        elif kind == "float64":
            float_values = series.to_numpy(dtype="float64", na_value=np.nan, copy=True)
            float_values[np.isnan(float_values)] = np.nan
            float_values[float_values == 0.0] = 0.0
            digest.update(np.ascontiguousarray(float_values, dtype="<f8").tobytes())
        elif kind == "integer64":
            missing = series.isna().to_numpy(dtype="uint8", copy=True)
            integer_values = series.fillna(0).to_numpy(dtype="int64", copy=True)
            digest.update(missing.tobytes())
            digest.update(np.ascontiguousarray(integer_values, dtype="<i8").tobytes())
        elif kind == "boolean":
            boolean_values = series.astype("boolean").to_numpy(
                dtype="uint8", na_value=2, copy=True
            )
            digest.update(boolean_values.tobytes())
        else:
            string_values = series.astype("string[python]")
            hashes = pd.util.hash_pandas_object(string_values, index=False).to_numpy(
                dtype="<u8", copy=True
            )
            digest.update(hashes.tobytes())
    return digest.digest()


def _publish_snapshot_bundle(
    frame: pd.DataFrame,
    *,
    destination: Path,
    snapshot_id: str,
    metadata: Mapping[str, object],
) -> None:
    """Atomically publish one content-addressed Parquet/metadata directory."""

    destination = destination.resolve()
    final_directory = destination / snapshot_id
    temporary_directory = Path(
        tempfile.mkdtemp(prefix=f".tmp-{snapshot_id[:12]}-", dir=destination)
    ).resolve()
    if temporary_directory.parent != destination:
        raise RuntimeError("temporary production snapshot escaped its destination")
    parquet_path = temporary_directory / f"{snapshot_id}.parquet"
    metadata_path = temporary_directory / f"{snapshot_id}.json"
    try:
        frame.to_parquet(parquet_path, index=False, compression="zstd")
        with parquet_path.open("r+b") as stream:
            os.fsync(stream.fileno())
        stored_metadata = dict(metadata)
        stored_metadata["parquet_sha256"] = _file_sha256(parquet_path)
        stored_metadata["metadata_hash"] = _metadata_hash(stored_metadata)
        with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(
                json.dumps(stored_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.replace(temporary_directory, final_directory)
        except OSError:
            if not final_directory.exists():
                raise
            _remove_snapshot_temporary(temporary_directory, destination)
        _validate_snapshot_bundle(
            final_directory,
            parquet_path=final_directory / f"{snapshot_id}.parquet",
            metadata_path=final_directory / f"{snapshot_id}.json",
            expected_metadata=metadata,
        )
    except Exception:
        if temporary_directory.exists():
            _remove_snapshot_temporary(temporary_directory, destination)
        raise


def _validate_snapshot_bundle(
    snapshot_directory: Path,
    *,
    parquet_path: Path,
    metadata_path: Path,
    expected_metadata: Mapping[str, object],
) -> None:
    if not snapshot_directory.is_dir() or not parquet_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("immutable production snapshot bundle is incomplete")
    try:
        observed = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("immutable production snapshot metadata is invalid") from None
    if not isinstance(observed, dict):
        raise RuntimeError("immutable production snapshot metadata is invalid")
    for key, value in expected_metadata.items():
        if key != "created_at" and observed.get(key) != value:
            raise RuntimeError("existing production snapshot metadata failed validation")
    try:
        observed_created_at = datetime.fromisoformat(str(observed["created_at"]))
    except (KeyError, ValueError):
        raise RuntimeError("production snapshot created_at is invalid") from None
    if observed_created_at.tzinfo is None or observed_created_at.utcoffset() is None:
        raise RuntimeError("production snapshot created_at must be timezone-aware")
    if observed.get("metadata_hash") != _metadata_hash(observed):
        raise RuntimeError("production snapshot metadata hash mismatch")
    if observed.get("parquet_sha256") != _file_sha256(parquet_path):
        raise RuntimeError("production snapshot Parquet hash mismatch")


def _metadata_hash(metadata: Mapping[str, object]) -> str:
    payload = {key: value for key, value in metadata.items() if key != "metadata_hash"}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _remove_snapshot_temporary(path: Path, destination: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != destination.resolve() or not resolved.name.startswith(".tmp-"):
        raise RuntimeError("refused to remove an unverified production snapshot temporary path")
    shutil.rmtree(resolved)


def _assert_financial_timing(financials: pd.DataFrame) -> None:
    _require_columns(
        financials,
        {"available_at", "financial_available_at"},
        "production financial features",
    )
    feature_time = pd.to_datetime(financials["available_at"], utc=True)
    source_time = pd.to_datetime(financials["financial_available_at"], utc=True)
    if (source_time.notna() & (source_time > feature_time)).any():
        raise ValueError("financial feature uses a disclosure after its observation cutoff")


def _require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing required columns: {sorted(missing)}")


def _revision_contract(frame: pd.DataFrame) -> tuple[str, CapabilityStatus]:
    _require_columns(
        frame,
        {"historical_revision_policy", "historical_revision_status"},
        "production revision contract",
    )
    policies = tuple(frame["historical_revision_policy"].dropna().astype(str).unique())
    statuses = tuple(frame["historical_revision_status"].dropna().astype(str).unique())
    if len(policies) != 1 or len(statuses) != 1:
        raise ValueError("production rows must use one historical revision contract")
    return policies[0], CapabilityStatus(statuses[0])


def _target_contract(target_column: str) -> tuple[int, str]:
    match = re.fullmatch(
        r"target_(return|topix_excess|sector_excess|beta_residual|large_loss)_(1|5|20)d",
        target_column,
    )
    if match is None:
        raise ValueError(f"unsupported production target column: {target_column}")
    family, horizon_text = match.groups()
    horizon = int(horizon_text)
    status = (
        f"label_status_{horizon}d"
        if family in {"return", "large_loss"}
        else f"label_status_{family}_{horizon}d"
    )
    return horizon, status
