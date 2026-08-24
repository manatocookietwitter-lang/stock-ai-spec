"""Leakage-safe morning forecast revision research and Decision Engine output."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self, cast
from zoneinfo import ZoneInfo

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_ai.data import (
    CapabilityStatus,
    MorningCapabilityReport,
    MorningFreezeMetadata,
    MorningUniverseMember,
)
from stock_ai.decision.engine import DailyPortfolioDecisionEngine, DecisionCandidate
from stock_ai.domain import (
    MorningDecisionAudit,
    MorningPredictionRevision,
    PortfolioProposal,
    PortfolioState,
    Prediction,
    PredictionUncertainty,
    Security,
)
from stock_ai.features import (
    MORNING_CORE_MANIFEST,
    MORNING_MICROSTRUCTURE_MANIFEST,
    MorningFeatureOutput,
    morning_feature_manifest,
    morning_feature_manifest_for_capabilities,
)
from stock_ai.features.registry import FeatureSetManifest
from stock_ai.ml.dataset import HORIZONS
from stock_ai.ml.research_metrics import evaluate_cross_sectional_predictions
from stock_ai.ml.validation import PurgedExpandingWindowSplitter, reserve_locked_final_holdout

JST = ZoneInfo("Asia/Tokyo")
_BUNDLE_CONSTRUCTION_TOKEN = object()
MorningModelFamily = Literal["ridge", "lightgbm", "mlp"]


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class MorningModelDisposition(StrEnum):
    RESEARCH = "RESEARCH"
    REJECTED = "REJECTED"
    DISABLED = "DISABLED"
    BLOCKED_BY_DATA_CAPABILITY = "BLOCKED_BY_DATA_CAPABILITY"


class MorningResearchConfig(BaseModel):
    """Bounded Goal 4 comparison; no model is automatically promoted."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizons: tuple[int, ...] = HORIZONS
    model_families: tuple[MorningModelFamily, ...] = ("ridge", "lightgbm", "mlp")
    seeds: tuple[int, ...] = (17,)
    initial_train_periods: int = Field(default=80, ge=20)
    validation_periods: int = Field(default=20, ge=1)
    step_periods: int = Field(default=20, ge=1)
    holdout_periods: int = Field(default=40, ge=1)
    lightgbm_estimators: int = Field(default=150, ge=5, le=1_000)
    mlp_hidden_units: tuple[int, ...] = (32, 16)
    mlp_max_iterations: int = Field(default=300, ge=20, le=2_000)
    enable_neural_challenger: bool = False
    minimum_rank_ic_increment: float = Field(default=0.0, ge=0.0)
    max_rows: int = Field(default=5_000_000, ge=1_000)
    max_model_fits: int = Field(default=2_000, ge=1)
    hypothesis: str = Field(
        default=(
            "Causal 09:00--11:30 features improve frozen daily forecasts for monitored "
            "holdings and candidates without using post-freeze inputs"
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def valid_contract(self) -> Self:
        if (
            not self.horizons
            or len(self.horizons) != len(set(self.horizons))
            or not set(self.horizons) <= set(HORIZONS)
        ):
            raise ValueError("morning horizons must be a unique subset of 1, 5, and 20")
        if not self.model_families or len(self.model_families) != len(set(self.model_families)):
            raise ValueError("morning model families must be non-empty and unique")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("morning research seeds must be non-empty and unique")
        if self.step_periods < self.validation_periods:
            raise ValueError("morning validation windows cannot overlap")
        if self.enable_neural_challenger and (
            "mlp" not in self.model_families
            or len(self.seeds) < 3
            or set(self.horizons) != set(HORIZONS)
        ):
            raise ValueError(
                "morning MLP research requires 1d/5d/20d, the model family, and at least 3 seeds"
            )
        if not self.mlp_hidden_units or any(
            value < 1 or value > 256 for value in self.mlp_hidden_units
        ):
            raise ValueError("morning MLP hidden units must be in [1, 256]")
        return self

    @property
    def config_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))


class MorningDatasetSnapshot(BaseModel):
    """Authenticated F13/F14 supervised dataset publication."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    snapshot_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    publication_as_of: datetime
    provider: str = Field(min_length=1)
    source_snapshot_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    feature_set_version: str = Field(min_length=1)
    feature_manifest_hash: str = Field(min_length=64, max_length=64)
    feature_names: tuple[str, ...]
    frame_hash: str = Field(min_length=64, max_length=64)
    capability_statuses: tuple[tuple[str, str], ...]
    trading_calendar_dates: tuple[str, ...]
    rows: int = Field(ge=1)
    data_start: str
    data_end: str
    parquet_path: Path
    metadata_path: Path

    @field_validator("created_at", "publication_as_of")
    @classmethod
    def aware_snapshot_times(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("morning snapshot timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_source_lineage(self) -> Self:
        for name, values in (
            ("snapshot", self.source_snapshot_ids),
            ("record", self.source_record_ids),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"morning source {name} IDs must be non-empty and unique")
        parsed_calendar = pd.DatetimeIndex(pd.to_datetime(self.trading_calendar_dates)).normalize()
        if (
            len(parsed_calendar) < 21
            or not parsed_calendar.is_monotonic_increasing
            or parsed_calendar.has_duplicates
        ):
            raise ValueError("morning snapshot requires a sorted fixed trading calendar")
        return self


class MorningModelResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    model_family: MorningModelFamily
    seed: int
    folds: int = Field(ge=1)
    rows: int = Field(ge=1)
    dates: int = Field(ge=1)
    holdings_rows: int = Field(ge=0)
    candidate_rows: int = Field(ge=0)
    mean_squared_error: float = Field(ge=0)
    baseline_mean_squared_error: float = Field(ge=0)
    mean_daily_rank_ic: float | None
    baseline_mean_daily_rank_ic: float | None
    incremental_rank_ic: float | None
    revision_win_rate: float = Field(ge=0, le=1)
    adds_oos_value: bool
    disposition: MorningModelDisposition
    inference_timing_status: str


class MorningModelCapability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_name: str = Field(min_length=1)
    disposition: MorningModelDisposition
    reason: str = Field(min_length=1)


class MorningResearchReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    report_id: str = Field(min_length=1)
    created_at: datetime
    data_snapshot_id: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    feature_manifest_hash: str = Field(min_length=1)
    feature_names: tuple[str, ...]
    code_commit: str = Field(min_length=1)
    config: Mapping[str, object]
    config_hash: str = Field(min_length=1)
    holdout_start: str
    locked_holdout_accessed: bool = False
    results: tuple[MorningModelResult, ...]
    neural_and_sequence_capabilities: tuple[MorningModelCapability, ...]
    oof_rows: int = Field(ge=1)
    oof_sha256: str = Field(min_length=64, max_length=64)
    adoption_eligible: bool = False
    blocking_reasons: tuple[str, ...]
    is_order_instruction: bool = False

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("morning report created_at must be timezone-aware")
        return value

    @field_validator("config", mode="after")
    @classmethod
    def freeze_config(cls, value: Mapping[str, object]) -> Mapping[str, object]:
        return MappingProxyType(dict(value))

    @field_serializer("config")
    def serialize_config(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @model_validator(mode="after")
    def research_only(self) -> Self:
        if self.locked_holdout_accessed:
            raise ValueError("Goal 4 development report cannot open the locked holdout")
        if self.adoption_eligible:
            raise ValueError("Goal 4 fixture/development reports cannot auto-adopt a model")
        if self.is_order_instruction:
            raise ValueError("morning research reports cannot be order instructions")
        if _stable_hash(dict(self.config)) != self.config_hash:
            raise ValueError("morning research config hash mismatch")
        return self


@dataclass(frozen=True)
class MorningResearchRun:
    report: MorningResearchReport
    oof_predictions: pd.DataFrame


@dataclass(frozen=True)
class MorningFittedResearchBundle:
    """Deterministically refittable, research-only models for a post-history freeze."""

    _construction_token: object = field(repr=False, compare=False)
    report: MorningResearchReport
    snapshot: MorningDatasetSnapshot
    selected_family: MorningModelFamily
    selected_seed: int
    training_as_of: datetime
    models: Mapping[int, Pipeline]
    calibration_oof: pd.DataFrame
    is_research_only: bool = True
    bundle_id: str = field(init=False)
    model_hashes: Mapping[int, str] = field(init=False)

    def __post_init__(self) -> None:
        if self._construction_token is not _BUNDLE_CONSTRUCTION_TOKEN:
            raise ValueError(
                "morning fitted bundles must be created by the authenticated refit path"
            )
        models = {horizon: copy.deepcopy(model) for horizon, model in self.models.items()}
        model_hashes = {horizon: _pipeline_state_hash(model) for horizon, model in models.items()}
        object.__setattr__(self, "models", MappingProxyType(models))
        object.__setattr__(self, "model_hashes", MappingProxyType(model_hashes))
        object.__setattr__(self, "calibration_oof", self.calibration_oof.copy(deep=True))
        identity = _fitted_bundle_identity(self, model_hashes=model_hashes)
        object.__setattr__(self, "bundle_id", f"morning-fit-{_stable_hash(identity)[:24]}")


class MorningDecisionMarketData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    reference_price: Decimal = Field(gt=0)
    average_daily_trading_value: Decimal = Field(gt=0)


class MorningDecisionPredictionBatch(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: datetime
    universe_symbols: tuple[str, ...]
    universe: tuple[MorningUniverseMember, ...]
    provider: str = Field(min_length=1)
    source_snapshot_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    capability_statuses: tuple[tuple[str, str], ...]
    research_report_id: str = Field(min_length=1)
    freeze_evidence_hash: str = Field(min_length=64, max_length=64)
    evidence_kind: Literal["CURRENT_FEATURES", "HISTORICAL_OOF_REPLAY", "BLOCKED"]
    frozen_market: tuple[MorningDecisionMarketData, ...]
    predictions: tuple[Prediction, ...]
    blocked: tuple[str, ...] = ()
    is_research_only: bool = True

    @field_validator("as_of")
    @classmethod
    def aware_batch_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("morning Decision batch as_of must be timezone-aware")
        local = value.astimezone(JST)
        if (local.hour, local.minute, local.second, local.microsecond) != (11, 30, 0, 0):
            raise ValueError("morning Decision batch must use the exact 11:30 JST freeze")
        return value

    @model_validator(mode="after")
    def complete_or_blocked(self) -> Self:
        if not self.predictions and not self.blocked:
            raise ValueError("morning Decision batch must contain predictions or an explicit block")
        symbols = tuple(sorted(prediction.symbol for prediction in self.predictions))
        role_symbols = tuple(sorted(member.symbol for member in self.universe))
        if role_symbols != tuple(sorted(self.universe_symbols)):
            raise ValueError("morning Decision batch roles must cover the freeze universe")
        if self.predictions and symbols != tuple(sorted(self.universe_symbols)):
            raise ValueError("morning Decision batch predictions must cover the freeze universe")
        market_symbols = tuple(sorted(item.symbol for item in self.frozen_market))
        if len(market_symbols) != len(set(market_symbols)):
            raise ValueError("morning Decision batch market symbols must be unique")
        if self.predictions and market_symbols != tuple(sorted(self.universe_symbols)):
            raise ValueError("morning Decision batch market data must cover the freeze universe")
        if self.blocked and self.predictions:
            raise ValueError("blocked morning Decision batches cannot retain predictions")
        if self.evidence_kind == "BLOCKED" and self.frozen_market:
            raise ValueError("metadata-blocked morning Decision batches cannot retain market data")
        if self.predictions and self.evidence_kind == "BLOCKED":
            raise ValueError("complete morning Decision batches require feature or OOF evidence")
        if any(prediction.as_of != self.as_of for prediction in self.predictions):
            raise ValueError("morning Decision batch predictions must share the exact freeze")
        for name, values in (
            ("snapshot", self.source_snapshot_ids),
            ("record", self.source_record_ids),
        ):
            if not values or len(values) != len(set(values)):
                raise ValueError(f"morning Decision source {name} IDs must be non-empty and unique")
        if not self.capability_statuses:
            raise ValueError("morning Decision batch requires capability statuses")
        if not self.is_research_only:
            raise ValueError("Goal 4 morning Decision batches must remain research-only")
        return self


def build_morning_supervised_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    publication_as_of: datetime,
    trading_calendar: Iterable[date | datetime | pd.Timestamp],
) -> pd.DataFrame:
    """Join post-freeze labels and blank outcomes unavailable at publication time."""

    if publication_as_of.tzinfo is None or publication_as_of.utcoffset() is None:
        raise ValueError("morning dataset publication_as_of must be timezone-aware")
    identity = {"symbol", "trading_date", "as_of"}
    missing_features = sorted(identity - set(features.columns))
    if missing_features:
        raise ValueError(f"morning feature frame is missing: {', '.join(missing_features)}")
    label_columns = {"symbol", "trading_date"}
    calendar = _canonical_trading_calendar(trading_calendar)
    calendar_positions = {value: position for position, value in enumerate(calendar)}
    for horizon in HORIZONS:
        label_columns.update(
            {
                f"target_return_{horizon}d",
                f"label_entry_at_{horizon}d",
                f"label_end_date_{horizon}d",
                f"label_end_at_{horizon}d",
                f"label_available_at_{horizon}d",
                f"label_status_{horizon}d",
            }
        )
    missing_labels = sorted(label_columns - set(labels.columns))
    if missing_labels:
        raise ValueError(f"morning label frame is missing: {', '.join(missing_labels)}")
    if labels.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("morning labels must be unique by symbol-date")
    merged = features.merge(
        labels.loc[:, sorted(label_columns)],
        on=["symbol", "trading_date"],
        how="left",
        validate="one_to_one",
    )
    trading_dates = pd.to_datetime(merged["trading_date"]).dt.normalize()
    if not set(trading_dates) <= set(calendar):
        raise ValueError("morning features contain dates outside the fixed JPX calendar")
    if merged[[f"label_status_{horizon}d" for horizon in HORIZONS]].isna().any(axis=None):
        raise ValueError("every morning feature row requires an explicit label status")
    for horizon in HORIZONS:
        target = f"target_return_{horizon}d"
        entry_name = f"label_entry_at_{horizon}d"
        end_name = f"label_end_date_{horizon}d"
        end_at_name = f"label_end_at_{horizon}d"
        available_name = f"label_available_at_{horizon}d"
        status_name = f"label_status_{horizon}d"
        entries = _aware_timestamps(merged[entry_name], entry_name)
        endpoint_at = _aware_timestamps(merged[end_at_name], end_at_name)
        available = _aware_timestamps(merged[available_name], available_name)
        feature_as_of = _aware_timestamps(merged["as_of"], "as_of")
        if (entries <= feature_as_of).any():
            raise ValueError("morning labels must begin strictly after the 11:30 feature freeze")
        for entry, trading_date in zip(entries, trading_dates, strict=True):
            local_entry = entry.astimezone(JST)
            if (
                local_entry.date() != trading_date.date()
                or (local_entry.hour, local_entry.minute, local_entry.second) != (12, 30, 0)
            ):
                raise ValueError("morning label entry must be exact same-session 12:30 JST")
        if (available < entries).any():
            raise ValueError("morning labels cannot be available before their entry")
        merged[entry_name] = entries
        merged[end_at_name] = endpoint_at
        merged[available_name] = available
        merged[end_name] = pd.to_datetime(merged[end_name])
        end_dates = pd.to_datetime(merged[end_name]).dt.date
        entry_dates = entries.map(lambda value: value.astimezone(JST).date())
        if (end_dates < entry_dates).any():
            raise ValueError("morning label end dates cannot precede their entry")
        expected_end_dates = pd.Series(
            [
                calendar[calendar_positions[trading_date] + horizon].date()
                if calendar_positions[trading_date] + horizon < len(calendar)
                else None
                for trading_date in trading_dates
            ],
            index=merged.index,
        )
        if expected_end_dates.isna().any() or not end_dates.equals(expected_end_dates):
            raise ValueError(
                f"morning {horizon}d label endpoints must use the fixed JPX session calendar"
            )
        endpoint_dates = endpoint_at.map(lambda value: value.astimezone(JST).date())
        if (endpoint_at <= entries).any() or not endpoint_dates.equals(end_dates):
            raise ValueError("morning label endpoint timestamps must match their end session")
        if (available < endpoint_at).any():
            raise ValueError("morning labels cannot be available before their label endpoint")
        mature = available <= pd.Timestamp(publication_as_of)
        unavailable = ~mature
        merged.loc[unavailable, target] = np.nan
        merged.loc[unavailable, status_name] = "HORIZON_NOT_MATURE"
        available_status = merged[status_name].eq("AVAILABLE")
        values = pd.to_numeric(merged[target], errors="coerce")
        if values.loc[available_status].isna().any():
            raise ValueError("AVAILABLE morning labels require finite targets")
        if values.loc[available_status].map(math.isfinite).eq(False).any():
            raise ValueError("AVAILABLE morning labels require finite targets")
        if values.loc[~available_status].notna().any():
            raise ValueError("blocked morning label statuses cannot retain target values")
        merged[target] = values.astype(float)
        merged[f"revision_target_{horizon}d"] = (
            merged[target] - merged[f"morning.prior_expected_return_{horizon}d"]
        )
    return merged.sort_values(["trading_date", "symbol"], kind="stable").reset_index(drop=True)


def write_morning_dataset_snapshot(
    dataset: pd.DataFrame,
    destination: Path,
    *,
    created_at: datetime,
    publication_as_of: datetime,
    capability_report: MorningCapabilityReport,
    manifest: FeatureSetManifest,
    trading_calendar: Iterable[date | datetime | pd.Timestamp],
) -> MorningDatasetSnapshot:
    """Atomically publish a content-addressed morning supervised dataset."""

    for value in (created_at, publication_as_of):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("morning snapshot timestamps must be timezone-aware")
    provider = capability_report.provider
    if provider is None or not provider.strip():
        raise ValueError("morning snapshot provider cannot be blank")
    expected_manifest = morning_feature_manifest_for_capabilities(capability_report)
    if (
        manifest.manifest_hash != expected_manifest.manifest_hash
        or manifest.feature_names != expected_manifest.feature_names
        or manifest.feature_set_version != expected_manifest.feature_set_version
    ):
        raise ValueError("morning snapshot manifest does not exactly match provider capabilities")
    unavailable_capabilities = tuple(
        name
        for name in manifest.required_capabilities
        if capability_report.capabilities.get(name) is not CapabilityStatus.AVAILABLE
    )
    if unavailable_capabilities:
        raise ValueError(
            "morning snapshot manifest capabilities are unavailable: "
            + ", ".join(unavailable_capabilities)
        )
    calendar = _canonical_trading_calendar(trading_calendar)
    calendar_dates = tuple(str(value.date()) for value in calendar)
    _validate_snapshot_frame(
        dataset,
        publication_as_of=publication_as_of,
        feature_names=manifest.feature_names,
        trading_calendar=calendar,
    )
    canonical = _canonical_snapshot_frame(dataset)
    observed_providers = set(canonical["provider"].astype(str))
    if observed_providers != {provider}:
        raise ValueError("morning snapshot provider does not match its feature rows")
    source_snapshot_ids = tuple(
        sorted(
            {
                str(source_id)
                for values in canonical["source_snapshot_ids"]
                for source_id in values
            }
        )
    )
    source_ids = tuple(
        sorted(
            {str(source_id) for values in canonical["source_record_ids"] for source_id in values}
        )
    )
    statuses = tuple(
        sorted(
            (str(name), str(getattr(status, "value", status)))
            for name, status in capability_report.capabilities.items()
        )
    )
    frame_hash = _generic_frame_hash(canonical, sort_columns=("trading_date", "symbol"))
    rows = len(canonical)
    data_start = str(pd.to_datetime(canonical["trading_date"]).min().date())
    data_end = str(pd.to_datetime(canonical["trading_date"]).max().date())
    identity = {
        "publication_as_of": publication_as_of.isoformat(),
        "provider": provider,
        "source_snapshot_ids": source_snapshot_ids,
        "source_record_ids": source_ids,
        "feature_set_version": manifest.feature_set_version,
        "feature_manifest_hash": manifest.manifest_hash,
        "feature_names": manifest.feature_names,
        "capability_statuses": statuses,
        "trading_calendar_dates": calendar_dates,
        "frame_hash": frame_hash,
        "rows": rows,
        "data_start": data_start,
        "data_end": data_end,
    }
    snapshot_id = _stable_hash(identity)
    destination.mkdir(parents=True, exist_ok=True)
    final_directory = destination / snapshot_id
    parquet_name = f"{snapshot_id}.parquet"
    metadata_name = f"{snapshot_id}.json"
    snapshot_payload = {
        "snapshot_id": snapshot_id,
        "created_at": created_at.isoformat(),
        "publication_as_of": publication_as_of.isoformat(),
        "provider": provider,
        "source_snapshot_ids": list(source_snapshot_ids),
        "source_record_ids": list(source_ids),
        "feature_set_version": manifest.feature_set_version,
        "feature_manifest_hash": manifest.manifest_hash,
        "feature_names": list(manifest.feature_names),
        "frame_hash": frame_hash,
        "capability_statuses": [list(item) for item in statuses],
        "trading_calendar_dates": list(calendar_dates),
        "rows": rows,
        "data_start": data_start,
        "data_end": data_end,
        "parquet_path": str((final_directory / parquet_name).resolve()),
    }
    if final_directory.exists():
        observed, _ = load_morning_dataset_snapshot(final_directory / parquet_name)
        if observed.snapshot_id != snapshot_id:
            raise RuntimeError("morning dataset identity already exists with conflicts")
        return observed
    temporary = Path(tempfile.mkdtemp(prefix=".morning-dataset-", dir=destination))
    try:
        parquet_path = temporary / parquet_name
        metadata_path = temporary / metadata_name
        canonical.to_parquet(parquet_path, index=False)
        payload: dict[str, object] = {
            "snapshot": snapshot_payload,
            "identity": identity,
            "parquet_sha256": _file_sha256(parquet_path),
        }
        payload["metadata_hash"] = _stable_hash(payload)
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return MorningDatasetSnapshot.model_validate(
        {
            **snapshot_payload,
            "parquet_path": final_directory / parquet_name,
            "metadata_path": final_directory / metadata_name,
        }
    )


def load_morning_dataset_snapshot(
    parquet_path: Path,
) -> tuple[MorningDatasetSnapshot, pd.DataFrame]:
    """Authenticate a morning dataset before it can be used for research."""

    parquet_path = parquet_path.resolve()
    snapshot_id = parquet_path.stem
    if parquet_path.parent.name != snapshot_id:
        raise RuntimeError("morning dataset path is not content-addressed")
    metadata_path = parquet_path.with_suffix(".json")
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("morning dataset metadata is missing or invalid") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("snapshot"), dict):
        raise RuntimeError("morning dataset metadata is invalid")
    authenticated = {key: value for key, value in payload.items() if key != "metadata_hash"}
    if payload.get("metadata_hash") != _stable_hash(authenticated):
        raise RuntimeError("morning dataset metadata hash mismatch")
    snapshot_payload = dict(payload["snapshot"])
    if snapshot_payload.get("snapshot_id") != snapshot_id:
        raise RuntimeError("morning dataset snapshot identity mismatch")
    if Path(str(snapshot_payload.get("parquet_path", ""))).resolve() != parquet_path:
        raise RuntimeError("morning dataset Parquet path mismatch")
    if _file_sha256(parquet_path) != str(payload.get("parquet_sha256", "")):
        raise RuntimeError("morning dataset Parquet hash mismatch")
    frame = _canonical_snapshot_frame(pd.read_parquet(parquet_path))
    identity = payload.get("identity")
    if not isinstance(identity, dict) or _stable_hash(identity) != snapshot_id:
        raise RuntimeError("morning dataset content identity mismatch")
    frame_hash = _generic_frame_hash(frame, sort_columns=("trading_date", "symbol"))
    if identity.get("frame_hash") != frame_hash:
        raise RuntimeError("morning dataset frame hash mismatch")
    snapshot_feature_names = tuple(str(value) for value in snapshot_payload["feature_names"])
    expected_manifest = morning_feature_manifest(
        name for name in snapshot_feature_names if name not in MORNING_CORE_MANIFEST.feature_names
    )
    if (
        snapshot_payload.get("feature_manifest_hash") != expected_manifest.manifest_hash
        or snapshot_payload.get("feature_set_version") != expected_manifest.feature_set_version
    ):
        raise RuntimeError("morning dataset feature manifest mismatch")
    snapshot = MorningDatasetSnapshot.model_validate(
        {
            **snapshot_payload,
            "parquet_path": parquet_path,
            "metadata_path": metadata_path,
        }
    )
    if len(frame) != snapshot.rows:
        raise RuntimeError("morning dataset row count mismatch")
    _verify_snapshot_frame_metadata(snapshot, frame)
    if snapshot.frame_hash != frame_hash:
        raise RuntimeError("morning dataset snapshot frame identity mismatch")
    status_report = MorningCapabilityReport(
        provider=snapshot.provider,
        capabilities={
            name: CapabilityStatus(status) for name, status in snapshot.capability_statuses
        },
        reasons={},
    )
    capability_manifest = morning_feature_manifest_for_capabilities(status_report)
    if (
        snapshot.feature_names != capability_manifest.feature_names
        or snapshot.feature_manifest_hash != capability_manifest.manifest_hash
        or snapshot.feature_set_version != capability_manifest.feature_set_version
    ):
        raise RuntimeError("morning dataset capabilities do not imply its feature manifest")
    expected_identity_fields = {
        "publication_as_of": snapshot.publication_as_of.isoformat(),
        "provider": snapshot.provider,
        "source_snapshot_ids": list(snapshot.source_snapshot_ids),
        "source_record_ids": list(snapshot.source_record_ids),
        "feature_set_version": snapshot.feature_set_version,
        "feature_manifest_hash": snapshot.feature_manifest_hash,
        "feature_names": list(snapshot.feature_names),
        "capability_statuses": [list(item) for item in snapshot.capability_statuses],
        "trading_calendar_dates": list(snapshot.trading_calendar_dates),
        "frame_hash": snapshot.frame_hash,
        "rows": snapshot.rows,
        "data_start": snapshot.data_start,
        "data_end": snapshot.data_end,
    }
    if any(identity.get(key) != value for key, value in expected_identity_fields.items()):
        raise RuntimeError("morning dataset metadata diverges from its content identity")
    observed_source_snapshots = tuple(
        sorted(
            {
                str(source_id)
                for values in frame["source_snapshot_ids"]
                for source_id in values
            }
        )
    )
    observed_source_records = tuple(
        sorted(
            {str(source_id) for values in frame["source_record_ids"] for source_id in values}
        )
    )
    if (
        set(frame["provider"].astype(str)) != {snapshot.provider}
        or observed_source_snapshots != snapshot.source_snapshot_ids
        or observed_source_records != snapshot.source_record_ids
    ):
        raise RuntimeError("morning dataset source lineage mismatch")
    _validate_snapshot_frame(
        frame,
        publication_as_of=snapshot.publication_as_of,
        feature_names=snapshot.feature_names,
        trading_calendar=pd.DatetimeIndex(pd.to_datetime(snapshot.trading_calendar_dates)),
    )
    return snapshot, frame


def run_morning_research(
    snapshot: MorningDatasetSnapshot,
    dataset: pd.DataFrame,
    *,
    created_at: datetime,
    code_commit: str,
    config: MorningResearchConfig | None = None,
) -> MorningResearchRun:
    """Compare no-update, Ridge, GBDT, and optional MLP on development OOF only."""

    active_config = config or MorningResearchConfig()
    _verify_snapshot_identity(snapshot)
    canonical = _canonical_snapshot_frame(dataset)
    _verify_snapshot_frame_metadata(snapshot, canonical)
    observed_frame_hash = _generic_frame_hash(
        canonical, sort_columns=("trading_date", "symbol")
    )
    if observed_frame_hash != snapshot.frame_hash:
        raise ValueError("morning research frame does not match its authenticated snapshot")
    _validate_snapshot_frame(
        canonical,
        publication_as_of=snapshot.publication_as_of,
        feature_names=snapshot.feature_names,
        trading_calendar=pd.DatetimeIndex(pd.to_datetime(snapshot.trading_calendar_dates)),
    )
    feature_names = snapshot.feature_names
    _validate_research_input(dataset, feature_names=feature_names, config=active_config)
    holdout = reserve_locked_final_holdout(dataset, holdout_periods=active_config.holdout_periods)
    development = dataset.iloc[list(holdout.development_indices)].copy()
    oof_parts: list[pd.DataFrame] = []
    model_results: list[MorningModelResult] = []
    horizon_frames: dict[int, pd.DataFrame] = {}
    for horizon in active_config.horizons:
        label_end = f"label_end_date_{horizon}d"
        label_available = f"label_available_at_{horizon}d"
        target = f"target_return_{horizon}d"
        revision_target = f"revision_target_{horizon}d"
        holdout_as_of = datetime.combine(
            holdout.holdout_start.date(), time(11, 30), tzinfo=JST
        )
        label_available_values = _aware_timestamps(
            development[label_available], label_available
        )
        horizon_frame = development.loc[
            development[target].notna()
            & pd.to_datetime(development[label_end]).notna()
            & development[label_available].notna()
            & (label_available_values < holdout_as_of)
            & (pd.to_datetime(development[label_end]) < holdout.holdout_start)
        ].copy()
        horizon_frames[horizon] = horizon_frame
    enabled_families = tuple(
        family
        for family in active_config.model_families
        if family != "mlp" or active_config.enable_neural_challenger
    )
    planned_fits = sum(
        sum(
            1
            for _ in _morning_splitter(active_config, horizon).split(
                horizon_frames[horizon], label_end_column=f"label_end_date_{horizon}d"
            )
        )
        * len(enabled_families)
        * len(active_config.seeds)
        for horizon in active_config.horizons
    )
    if planned_fits > active_config.max_model_fits:
        raise ValueError("BLOCKED_BY_RESOURCE_CAPABILITY: morning model-fit bound exceeded")
    for horizon in active_config.horizons:
        label_end = f"label_end_date_{horizon}d"
        label_available = f"label_available_at_{horizon}d"
        target = f"target_return_{horizon}d"
        revision_target = f"revision_target_{horizon}d"
        horizon_frame = horizon_frames[horizon]
        for family in active_config.model_families:
            if family == "mlp" and not active_config.enable_neural_challenger:
                continue
            for seed in active_config.seeds:
                oof = _generate_morning_oof(
                    horizon_frame,
                    feature_names=feature_names,
                    target_column=revision_target,
                    realized_return_column=target,
                    label_end_column=label_end,
                    label_available_column=label_available,
                    horizon=horizon,
                    family=family,
                    seed=seed,
                    config=active_config,
                )
                oof_parts.append(oof)
                model_results.append(_summarize_model(oof, config=active_config))
    if not oof_parts:
        raise ValueError("BLOCKED_BY_VALIDATION: morning research produced no OOF rows")
    combined = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["horizon", "model_family", "seed", "trading_date", "symbol"], kind="stable"
    )
    identity = ["symbol", "trading_date", "horizon", "model_family", "seed"]
    if combined.duplicated(identity).any():
        raise RuntimeError("morning OOF predictions contain duplicate identities")
    capabilities = _model_capabilities(active_config, tuple(model_results))
    oof_hash = _frame_hash(combined)
    report = MorningResearchReport(
        report_id="PENDING",
        created_at=created_at,
        data_snapshot_id=snapshot.snapshot_id,
        feature_set_version=snapshot.feature_set_version,
        feature_manifest_hash=snapshot.feature_manifest_hash,
        feature_names=feature_names,
        code_commit=code_commit,
        config=active_config.model_dump(mode="json"),
        config_hash=active_config.config_hash,
        holdout_start=str(holdout.holdout_start.date()),
        results=tuple(model_results),
        neural_and_sequence_capabilities=capabilities,
        oof_rows=len(combined),
        oof_sha256=oof_hash,
        blocking_reasons=(
            "live morning provider and synchronized historical coverage are not configured",
            "locked final holdout remains unopened",
            "development evidence cannot auto-adopt a production model",
        ),
    )
    report = report.model_copy(
        update={"report_id": f"morning-{_stable_hash(_report_identity(report))[:24]}"}
    )
    return MorningResearchRun(report=report, oof_predictions=combined.reset_index(drop=True))


def fit_morning_research_bundle(
    run: MorningResearchRun,
    snapshot: MorningDatasetSnapshot,
    dataset: pd.DataFrame,
    *,
    selected_family: MorningModelFamily,
    selected_seed: int,
) -> MorningFittedResearchBundle:
    """Refit one explicitly selected research model without opening the locked holdout."""

    _verify_snapshot_identity(snapshot)
    if (
        run.report.data_snapshot_id != snapshot.snapshot_id
        or run.report.feature_set_version != snapshot.feature_set_version
        or run.report.feature_manifest_hash != snapshot.feature_manifest_hash
        or run.report.feature_names != snapshot.feature_names
    ):
        raise ValueError("morning fitted bundle report/snapshot mismatch")
    if run.report.report_id != f"morning-{_stable_hash(_report_identity(run.report))[:24]}":
        raise ValueError("morning fitted bundle report identity is invalid")
    if len(run.oof_predictions) != run.report.oof_rows or _frame_hash(
        run.oof_predictions
    ) != run.report.oof_sha256:
        raise ValueError("morning fitted bundle OOF is not authenticated by its report")
    canonical = _canonical_snapshot_frame(dataset)
    _verify_snapshot_frame_metadata(snapshot, canonical)
    if (
        _generic_frame_hash(canonical, sort_columns=("trading_date", "symbol"))
        != snapshot.frame_hash
    ):
        raise ValueError("morning fitted bundle dataset is not authenticated")
    selected_results = tuple(
        result
        for result in run.report.results
        if result.model_family == selected_family and result.seed == selected_seed
    )
    if set(result.horizon for result in selected_results) != set(HORIZONS) or any(
        result.disposition is not MorningModelDisposition.RESEARCH for result in selected_results
    ):
        raise ValueError("morning fitted bundle requires research-positive 1d/5d/20d evidence")
    config = MorningResearchConfig.model_validate(dict(run.report.config))
    holdout = reserve_locked_final_holdout(dataset, holdout_periods=config.holdout_periods)
    development = dataset.iloc[list(holdout.development_indices)].copy()
    training_as_of = datetime.combine(holdout.holdout_start.date(), time(11, 30), tzinfo=JST)
    models: dict[int, Pipeline] = {}
    for horizon in HORIZONS:
        target = f"revision_target_{horizon}d"
        label_end = f"label_end_date_{horizon}d"
        label_available = f"label_available_at_{horizon}d"
        available = _aware_timestamps(development[label_available], label_available)
        train = development.loc[
            development[target].notna()
            & (pd.to_datetime(development[label_end]) < holdout.holdout_start)
            & (available < training_as_of)
        ].copy()
        if train.empty:
            raise ValueError(f"BLOCKED_BY_VALIDATION: no refit rows for {horizon}d")
        model = _morning_regressor(selected_family, seed=selected_seed, config=config)
        model.fit(
            train.loc[:, list(snapshot.feature_names)].replace([np.inf, -np.inf], np.nan),
            train[target].astype(float),
        )
        models[horizon] = model
    return MorningFittedResearchBundle(
        _construction_token=_BUNDLE_CONSTRUCTION_TOKEN,
        report=run.report,
        snapshot=snapshot,
        selected_family=selected_family,
        selected_seed=selected_seed,
        training_as_of=training_as_of,
        models=models,
        calibration_oof=run.oof_predictions,
    )


def infer_current_morning_predictions(
    bundle: MorningFittedResearchBundle,
    features: MorningFeatureOutput,
    freeze_metadata: MorningFreezeMetadata,
    *,
    decision_horizon: int = 5,
    minimum_calibration_rows: int = 20,
    large_loss_threshold: float = -0.08,
) -> MorningDecisionPredictionBatch:
    """Run a research-only post-history 11:30 inference without current outcomes."""

    if decision_horizon not in HORIZONS or minimum_calibration_rows < 1:
        raise ValueError("invalid morning current-inference calibration contract")
    _verify_fitted_bundle(bundle)
    if (
        features.manifest.feature_names != bundle.snapshot.feature_names
        or features.manifest.manifest_hash != bundle.snapshot.feature_manifest_hash
    ):
        raise ValueError("current morning features do not match the fitted manifest")
    if _frame_hash(bundle.calibration_oof) != bundle.report.oof_sha256:
        raise ValueError("morning fitted bundle calibration OOF was mutated")
    freeze_date = pd.Timestamp(freeze_metadata.as_of.astimezone(JST).date())
    if freeze_date <= pd.Timestamp(bundle.snapshot.data_end):
        raise ValueError("current morning inference must be after the research snapshot")
    current = features.frame.loc[
        pd.to_datetime(features.frame["trading_date"]).dt.normalize().eq(freeze_date)
    ].copy()
    if current.empty:
        raise ValueError("current morning feature frame has no requested freeze")
    current_as_of = _aware_timestamps(current["as_of"], "as_of")
    current_available = _aware_timestamps(current["available_at"], "available_at")
    if any(value != freeze_metadata.as_of for value in current_as_of):
        raise ValueError("current morning features do not share the requested exact freeze")
    if (current_available > current_as_of).any():
        raise ValueError("current morning features contain post-freeze source availability")
    observed_source_snapshots = {
        str(source_id)
        for values in current["source_snapshot_ids"]
        for source_id in values
    }
    observed_source_records = {
        str(source_id) for values in current["source_record_ids"] for source_id in values
    }
    if (
        features.capability_report.model_dump(mode="json")
        != freeze_metadata.capability_report.model_dump(mode="json")
        or features.capability_report.provider != freeze_metadata.provider
        or set(current["provider"].astype(str)) != {freeze_metadata.provider}
        or observed_source_snapshots != set(freeze_metadata.source_snapshot_ids)
        or observed_source_records != set(freeze_metadata.source_record_ids)
    ):
        raise ValueError("current morning feature lineage does not match its freeze metadata")
    expected_symbols = tuple(sorted(member.symbol for member in freeze_metadata.universe))
    if tuple(sorted(current["symbol"].astype(str).unique())) != expected_symbols:
        return _blocked_morning_batch(
            freeze_metadata,
            research_report_id=bundle.report.report_id,
            reason="BLOCKED_BY_DATA_CAPABILITY: incomplete current freeze universe",
        )
    _require_current_freeze_roles(current, freeze_metadata)
    required_values = current.loc[:, list(bundle.snapshot.feature_names)].apply(
        pd.to_numeric, errors="coerce"
    )
    usable = np.isfinite(required_values.to_numpy(dtype=float))
    candidate_rank = "morning.candidate_volume_rank_pct"
    if candidate_rank in required_values:
        column_number = required_values.columns.get_loc(candidate_rank)
        structural_missing = ~current["morning.is_candidate"].astype(bool).to_numpy()
        usable[structural_missing, column_number] = True
    if not usable.all():
        return _blocked_morning_batch(
            freeze_metadata,
            research_report_id=bundle.report.report_id,
            reason=(
                "BLOCKED_BY_DATA_CAPABILITY: current freeze lacks usable profile or feature history"
            ),
        )
    if any(
        current[field].astype(str).nunique(dropna=False) != 1
        for field in (
            "prior_model_version",
            "prior_feature_version",
            "prior_data_snapshot_id",
            "prior_prediction_as_of",
        )
    ):
        raise ValueError("current morning inference has mixed prior prediction provenance")
    revisions: dict[int, np.ndarray] = {}
    current_x = current.loc[:, list(bundle.snapshot.feature_names)].replace(
        [np.inf, -np.inf], np.nan
    )
    for horizon, model in bundle.models.items():
        values = np.asarray(model.predict(current_x), dtype=float)
        if values.ndim != 1 or len(values) != len(current) or not np.isfinite(values).all():
            raise RuntimeError("current morning model emitted invalid predictions")
        revisions[horizon] = values
    calibration = bundle.calibration_oof.loc[
        bundle.calibration_oof["model_family"].eq(bundle.selected_family)
        & bundle.calibration_oof["seed"].eq(bundle.selected_seed)
        & bundle.calibration_oof["horizon"].eq(decision_horizon)
    ].copy()
    available = _aware_timestamps(calibration["label_available_at"], "label_available_at")
    calibration_end = pd.to_datetime(calibration["label_end"]).dt.normalize()
    calibration = calibration.loc[
        (available < freeze_metadata.as_of) & (calibration_end < freeze_date)
    ]
    errors = (
        calibration["target"].astype(float) - calibration["final_prediction"].astype(float)
    ).replace([np.inf, -np.inf], np.nan).dropna()
    if len(errors) < minimum_calibration_rows:
        return _blocked_morning_batch(
            freeze_metadata,
            research_report_id=bundle.report.report_id,
            reason="BLOCKED_BY_CALIBRATION_HISTORY",
        )
    downside_error = float(errors.quantile(0.10))
    standard_error = float(np.sqrt(np.mean(np.square(errors))))
    large_loss_probability = float(
        (calibration["target"].astype(float) <= large_loss_threshold).mean()
    )
    predictions: list[Prediction] = []
    for row_number, (_, row) in enumerate(current.iterrows()):
        prior_values = {
            horizon: float(row[f"morning.prior_expected_return_{horizon}d"])
            for horizon in HORIZONS
        }
        final_values = {
            horizon: prior_values[horizon] + float(revisions[horizon][row_number])
            for horizon in HORIZONS
        }
        revision = MorningPredictionRevision(
            as_of=freeze_metadata.as_of,
            prior_model_version=str(row["prior_model_version"]),
            prior_feature_version=str(row["prior_feature_version"]),
            prior_data_snapshot_id=str(row["prior_data_snapshot_id"]),
            return_revision_1d=float(revisions[1][row_number]),
            return_revision_5d=float(revisions[5][row_number]),
            return_revision_20d=float(revisions[20][row_number]),
            downside_quantile_revision=(
                final_values[decision_horizon]
                + downside_error
                - float(row["morning.prior_downside_quantile"])
            ),
            large_loss_probability_revision=(
                large_loss_probability
                - float(row["morning.prior_large_loss_probability"])
            ),
            revised_standard_error=standard_error,
            revised_model_disagreement=0.0,
            calibration_history_rows=len(errors),
            model_version=bundle.bundle_id,
            feature_version=bundle.snapshot.feature_set_version,
            data_snapshot_id=bundle.snapshot.snapshot_id,
            capability_status="PARTIAL",
        )
        prior = Prediction(
            symbol=str(row["symbol"]),
            as_of=_aware_datetime(row["prior_prediction_as_of"], "prior_prediction_as_of"),
            expected_return_1d=prior_values[1],
            expected_return_5d=prior_values[5],
            expected_return_20d=prior_values[20],
            downside_quantile=float(row["morning.prior_downside_quantile"]),
            large_loss_probability=float(row["morning.prior_large_loss_probability"]),
            uncertainty=PredictionUncertainty(
                standard_error=float(row["morning.prior_uncertainty"])
            ),
            model_version=str(row["prior_model_version"]),
            feature_version=str(row["prior_feature_version"]),
            data_snapshot_id=str(row["prior_data_snapshot_id"]),
        )
        predictions.append(apply_morning_revision(prior, revision))
    return MorningDecisionPredictionBatch(
        as_of=freeze_metadata.as_of,
        universe_symbols=expected_symbols,
        universe=freeze_metadata.universe,
        provider=freeze_metadata.provider,
        source_snapshot_ids=freeze_metadata.source_snapshot_ids,
        source_record_ids=freeze_metadata.source_record_ids,
        capability_statuses=_capability_statuses(freeze_metadata.capability_report),
        research_report_id=bundle.report.report_id,
        freeze_evidence_hash=_generic_frame_hash(
            current, sort_columns=("trading_date", "symbol")
        ),
        evidence_kind="CURRENT_FEATURES",
        frozen_market=_frozen_market_from_frame(current),
        predictions=tuple(predictions),
    )


def write_morning_research_run(run: MorningResearchRun, destination: Path) -> tuple[Path, Path]:
    """Atomically publish an authenticated morning report and its row-level OOF."""

    expected_id = f"morning-{_stable_hash(_report_identity(run.report))[:24]}"
    if run.report.report_id != expected_id:
        raise RuntimeError("morning research report content identity mismatch")
    if len(run.oof_predictions) != run.report.oof_rows:
        raise RuntimeError("morning research OOF row count mismatch")
    if _frame_hash(run.oof_predictions) != run.report.oof_sha256:
        raise RuntimeError("morning research OOF content identity mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    final_directory = destination / run.report.report_id
    parquet_name = f"{run.report.report_id}.oof.parquet"
    metadata_name = f"{run.report.report_id}.json"
    if final_directory.exists():
        observed_report, observed_oof = load_morning_research_run(final_directory / parquet_name)
        if (
            observed_report.report_id != run.report.report_id
            or _frame_hash(observed_oof) != run.report.oof_sha256
        ):
            raise RuntimeError("morning research identity already exists with conflicts")
        return final_directory / metadata_name, final_directory / parquet_name
    temporary = Path(tempfile.mkdtemp(prefix=".morning-", dir=destination))
    try:
        parquet_path = temporary / parquet_name
        metadata_path = temporary / metadata_name
        run.oof_predictions.to_parquet(parquet_path, index=False)
        payload: dict[str, object] = {
            "report": run.report.model_dump(mode="json"),
            "parquet_path": str((final_directory / parquet_name).resolve()),
            "parquet_sha256": _file_sha256(parquet_path),
        }
        payload["metadata_hash"] = _stable_hash(payload)
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_directory / metadata_name, final_directory / parquet_name


def load_morning_research_run(
    parquet_path: Path,
) -> tuple[MorningResearchReport, pd.DataFrame]:
    """Load only a content-addressed morning bundle whose metadata and OOF authenticate."""

    parquet_path = parquet_path.resolve()
    report_id = parquet_path.name.removesuffix(".oof.parquet")
    if parquet_path.parent.name != report_id:
        raise RuntimeError("morning research path is not content-addressed")
    metadata_path = parquet_path.parent / f"{report_id}.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("morning research metadata is missing or invalid") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("report"), dict):
        raise RuntimeError("morning research metadata is invalid")
    authenticated = {key: value for key, value in payload.items() if key != "metadata_hash"}
    if payload.get("metadata_hash") != _stable_hash(authenticated):
        raise RuntimeError("morning research metadata hash mismatch")
    if Path(str(payload.get("parquet_path", ""))).resolve() != parquet_path:
        raise RuntimeError("morning research Parquet path mismatch")
    if _file_sha256(parquet_path) != str(payload.get("parquet_sha256", "")):
        raise RuntimeError("morning research Parquet hash mismatch")
    report = MorningResearchReport.model_validate(payload["report"])
    if report.report_id != report_id:
        raise RuntimeError("morning research report identity mismatch")
    expected_id = f"morning-{_stable_hash(_report_identity(report))[:24]}"
    if report.report_id != expected_id:
        raise RuntimeError("morning research report content identity mismatch")
    frame = pd.read_parquet(parquet_path)
    if len(frame) != report.oof_rows or _frame_hash(frame) != report.oof_sha256:
        raise RuntimeError("morning research OOF content identity mismatch")
    return report, frame


def build_morning_decision_predictions(
    oof: pd.DataFrame,
    *,
    report: MorningResearchReport,
    snapshot: MorningDatasetSnapshot,
    freeze_metadata: MorningFreezeMetadata,
    selected_family: MorningModelFamily,
    selected_seed: int,
    decision_horizon: int = 5,
    minimum_calibration_rows: int = 20,
    large_loss_threshold: float = -0.08,
) -> MorningDecisionPredictionBatch:
    """Create typed predictions using only residuals matured before each 11:30 row."""

    if decision_horizon not in HORIZONS:
        raise ValueError("decision horizon must be 1, 5, or 20")
    if minimum_calibration_rows < 1:
        raise ValueError("minimum calibration rows must be positive")
    _verify_snapshot_identity(snapshot)
    if (
        report.data_snapshot_id != snapshot.snapshot_id
        or report.feature_set_version != snapshot.feature_set_version
        or report.feature_manifest_hash != snapshot.feature_manifest_hash
        or report.feature_names != snapshot.feature_names
    ):
        raise ValueError("morning Decision report does not match its authenticated dataset")
    expected_report_id = f"morning-{_stable_hash(_report_identity(report))[:24]}"
    if report.report_id != expected_report_id:
        raise ValueError("morning Decision report content identity is invalid")
    if len(oof) != report.oof_rows or _frame_hash(oof) != report.oof_sha256:
        raise ValueError("morning Decision OOF does not match its authenticated report")
    if (
        freeze_metadata.provider != snapshot.provider
        or not set(freeze_metadata.source_snapshot_ids) <= set(snapshot.source_snapshot_ids)
        or not set(freeze_metadata.source_record_ids) <= set(snapshot.source_record_ids)
        or _capability_statuses(freeze_metadata.capability_report) != snapshot.capability_statuses
    ):
        raise ValueError("morning Decision freeze lineage does not match the dataset snapshot")
    selected_results = tuple(
        result
        for result in report.results
        if result.model_family == selected_family and result.seed == selected_seed
    )
    if set(result.horizon for result in selected_results) != set(HORIZONS) or any(
        result.disposition is not MorningModelDisposition.RESEARCH for result in selected_results
    ):
        raise ValueError("morning Decision model must have research-positive 1d/5d/20d evidence")
    all_selected = oof.loc[
        oof["model_family"].eq(selected_family) & oof["seed"].eq(selected_seed)
    ].copy()
    if all_selected.empty:
        raise ValueError("selected morning model family/seed has no OOF rows")
    prediction_date = pd.Timestamp(freeze_metadata.as_of.astimezone(JST).date())
    selected = all_selected.loc[
        pd.to_datetime(all_selected["trading_date"]).dt.normalize().eq(prediction_date)
    ].copy()
    if selected.empty:
        raise ValueError("selected morning model has no OOF replay at the requested freeze")
    required = {
        "symbol",
        "trading_date",
        "as_of",
        "provider",
        "source_snapshot_ids",
        "source_record_ids",
        "reference_price_1130",
        "average_daily_trading_value",
        "horizon",
        "target",
        "label_end",
        "label_available_at",
        "prior_prediction",
        "final_prediction",
        "prior_downside_quantile",
        "prior_large_loss_probability",
        "prior_uncertainty",
        "prior_model_version",
        "prior_feature_version",
        "prior_data_snapshot_id",
        "is_current_holding",
        "is_candidate",
    }
    if missing := sorted(required - set(selected.columns)):
        raise ValueError(f"morning OOF is missing Decision fields: {', '.join(missing)}")
    predictions: list[Prediction] = []
    blocked: list[str] = []
    group_columns = ["symbol", "trading_date"]
    expected_symbols = tuple(sorted(member.symbol for member in freeze_metadata.universe))
    observed_symbols = tuple(sorted(selected["symbol"].astype(str).unique()))
    if observed_symbols != expected_symbols:
        return _blocked_morning_batch(
            freeze_metadata,
            research_report_id=report.report_id,
            reason="BLOCKED_BY_DATA_CAPABILITY: incomplete freeze prediction universe",
        )
    if set(selected["provider"].astype(str)) != {freeze_metadata.provider}:
        raise ValueError("morning Decision OOF provider does not match the freeze")
    selected_snapshot_ids = {
        str(source_id) for values in selected["source_snapshot_ids"] for source_id in values
    }
    selected_record_ids = {
        str(source_id) for values in selected["source_record_ids"] for source_id in values
    }
    if (
        selected_snapshot_ids != set(freeze_metadata.source_snapshot_ids)
        or selected_record_ids != set(freeze_metadata.source_record_ids)
    ):
        raise ValueError("morning Decision OOF source lineage does not match the freeze")
    _require_current_freeze_roles(selected, freeze_metadata)
    prior_bundle_fields = (
        "prior_model_version",
        "prior_feature_version",
        "prior_data_snapshot_id",
        "prior_prediction_as_of",
    )
    if any(selected[field].astype(str).nunique(dropna=False) != 1 for field in prior_bundle_fields):
        return _blocked_morning_batch(
            freeze_metadata,
            research_report_id=report.report_id,
            reason="BLOCKED_BY_PROVENANCE_COHERENCE: mixed daily prior bundle",
        )
    model_version = f"{report.report_id}:{selected_family}:seed-{selected_seed}"
    for key, group in selected.groupby(group_columns, sort=True):
        if len(group) != len(HORIZONS) or set(group["horizon"].astype(int)) != set(HORIZONS):
            blocked.append(f"{key}: incomplete 1d/5d/20d morning horizons")
            continue
        by_horizon: dict[int, Any] = {}
        for raw_row in group.itertuples(index=False):
            row = cast(Any, raw_row)
            by_horizon[int(row.horizon)] = row
        reference = by_horizon[decision_horizon]
        coherence_fields = (
            "as_of",
            "provider",
            "source_snapshot_ids",
            "source_record_ids",
            "reference_price_1130",
            "average_daily_trading_value",
            "prior_prediction_as_of",
            "prior_model_version",
            "prior_feature_version",
            "prior_data_snapshot_id",
        )
        if any(group[field].astype(str).nunique(dropna=False) != 1 for field in coherence_fields):
            blocked.append(f"{key}: BLOCKED_BY_PROVENANCE_COHERENCE")
            continue
        reference_as_of = _aware_datetime(reference.as_of, "as_of")
        if reference_as_of != freeze_metadata.as_of:
            blocked.append(f"{key}: BLOCKED_BY_FREEZE_COHERENCE")
            continue
        calibration_available = _aware_timestamps(
            all_selected["label_available_at"], "label_available_at"
        )
        calibration_end = pd.to_datetime(all_selected["label_end"]).dt.normalize()
        calibration = all_selected.loc[
            all_selected["horizon"].eq(decision_horizon)
            & (calibration_available < reference_as_of)
            & (calibration_end < prediction_date)
        ].copy()
        calibration_error = (
            (calibration["target"].astype(float) - calibration["final_prediction"].astype(float))
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(calibration_error) < minimum_calibration_rows:
            blocked.append(f"{key}: BLOCKED_BY_CALIBRATION_HISTORY")
            continue
        final_by_horizon = {
            horizon: float(by_horizon[horizon].final_prediction) for horizon in HORIZONS
        }
        downside = final_by_horizon[decision_horizon] + float(calibration_error.quantile(0.10))
        large_loss_probability = float(
            (calibration["target"].astype(float) <= large_loss_threshold).mean()
        )
        standard_error = float(np.sqrt(np.mean(np.square(calibration_error))))
        comparison = oof.loc[
            oof["symbol"].eq(reference.symbol)
            & pd.to_datetime(oof["trading_date"]).eq(prediction_date)
            & oof["horizon"].eq(decision_horizon),
            "final_prediction",
        ].astype(float)
        disagreement = float(comparison.std(ddof=0)) if len(comparison) > 1 else 0.0
        revision = MorningPredictionRevision(
            as_of=_aware_datetime(reference.as_of, "as_of"),
            prior_model_version=str(reference.prior_model_version),
            prior_feature_version=str(reference.prior_feature_version),
            prior_data_snapshot_id=str(reference.prior_data_snapshot_id),
            return_revision_1d=final_by_horizon[1] - float(by_horizon[1].prior_prediction),
            return_revision_5d=final_by_horizon[5] - float(by_horizon[5].prior_prediction),
            return_revision_20d=final_by_horizon[20] - float(by_horizon[20].prior_prediction),
            downside_quantile_revision=downside - float(reference.prior_downside_quantile),
            large_loss_probability_revision=(
                large_loss_probability - float(reference.prior_large_loss_probability)
            ),
            revised_standard_error=standard_error,
            revised_model_disagreement=disagreement,
            calibration_history_rows=len(calibration_error),
            model_version=model_version,
            feature_version=report.feature_set_version,
            data_snapshot_id=snapshot.snapshot_id,
            capability_status="PARTIAL",
        )
        prior = Prediction(
            symbol=str(reference.symbol),
            as_of=_aware_datetime(reference.prior_prediction_as_of, "prior_prediction_as_of"),
            expected_return_1d=float(by_horizon[1].prior_prediction),
            expected_return_5d=float(by_horizon[5].prior_prediction),
            expected_return_20d=float(by_horizon[20].prior_prediction),
            downside_quantile=float(reference.prior_downside_quantile),
            large_loss_probability=float(reference.prior_large_loss_probability),
            uncertainty=PredictionUncertainty(standard_error=float(reference.prior_uncertainty)),
            model_version=str(reference.prior_model_version),
            feature_version=str(reference.prior_feature_version),
            data_snapshot_id=str(reference.prior_data_snapshot_id),
        )
        predictions.append(apply_morning_revision(prior, revision))
    if blocked:
        predictions = []
    return MorningDecisionPredictionBatch(
        as_of=freeze_metadata.as_of,
        universe_symbols=expected_symbols,
        universe=freeze_metadata.universe,
        provider=freeze_metadata.provider,
        source_snapshot_ids=freeze_metadata.source_snapshot_ids,
        source_record_ids=freeze_metadata.source_record_ids,
        capability_statuses=_capability_statuses(freeze_metadata.capability_report),
        research_report_id=report.report_id,
        freeze_evidence_hash=_generic_frame_hash(
            selected, sort_columns=("trading_date", "symbol", "horizon")
        ),
        evidence_kind="HISTORICAL_OOF_REPLAY",
        frozen_market=_frozen_market_from_frame(selected),
        predictions=tuple(predictions),
        blocked=tuple(blocked),
    )


def apply_morning_revision(prior: Prediction, revision: MorningPredictionRevision) -> Prediction:
    """Apply an audited revision; this transforms forecasts and never chooses an Action."""

    if (
        prior.model_version != revision.prior_model_version
        or prior.feature_version != revision.prior_feature_version
        or prior.data_snapshot_id != revision.prior_data_snapshot_id
    ):
        raise ValueError("morning revision does not reference the supplied prior prediction")
    if revision.as_of < prior.as_of:
        raise ValueError("morning revision cannot precede its prior prediction")
    local = revision.as_of.astimezone(JST)
    if (local.hour, local.minute, local.second, local.microsecond) != (11, 30, 0, 0):
        raise ValueError("morning revision must use the exact 11:30 JST freeze")
    revised_probability = prior.large_loss_probability + revision.large_loss_probability_revision
    if not 0.0 <= revised_probability <= 1.0:
        raise ValueError("revised large-loss probability must remain in [0, 1]")
    return Prediction(
        symbol=prior.symbol,
        as_of=revision.as_of,
        expected_return_1d=prior.expected_return_1d + revision.return_revision_1d,
        expected_return_5d=prior.expected_return_5d + revision.return_revision_5d,
        expected_return_20d=prior.expected_return_20d + revision.return_revision_20d,
        downside_quantile=prior.downside_quantile + revision.downside_quantile_revision,
        large_loss_probability=revised_probability,
        uncertainty=PredictionUncertainty(
            standard_error=revision.revised_standard_error,
            model_disagreement=revision.revised_model_disagreement,
            coverage_warning="past-matured morning OOF residual calibration; research only",
        ),
        model_version=revision.model_version,
        feature_version=revision.feature_version,
        data_snapshot_id=revision.data_snapshot_id,
        morning_revision=revision,
    )


def propose_from_morning_batch(
    engine: DailyPortfolioDecisionEngine,
    *,
    batch: MorningDecisionPredictionBatch,
    features: MorningFeatureOutput,
    freeze_metadata: MorningFreezeMetadata,
    portfolio: PortfolioState,
    securities: Mapping[str, Security],
    account_bucket_ids_by_symbol: Mapping[str, tuple[str, ...]],
    generated_at: datetime,
) -> PortfolioProposal:
    """Bridge one complete research-only freeze into the Decision Engine safely."""

    if batch.blocked or not batch.predictions:
        raise ValueError("blocked morning Decision batch cannot produce a proposal")
    if batch.evidence_kind != "CURRENT_FEATURES":
        raise ValueError("morning proposals require current-feature freeze evidence")
    if portfolio.as_of != batch.as_of or batch.as_of != freeze_metadata.as_of:
        raise ValueError("morning batch, portfolio, and freeze must share exact 11:30 as_of")
    expected_roles = {member.symbol: member.role.value for member in freeze_metadata.universe}
    if (
        batch.provider != freeze_metadata.provider
        or batch.source_snapshot_ids != freeze_metadata.source_snapshot_ids
        or batch.source_record_ids != freeze_metadata.source_record_ids
        or batch.universe != freeze_metadata.universe
        or batch.capability_statuses != _capability_statuses(freeze_metadata.capability_report)
        or {member.symbol: member.role.value for member in batch.universe} != expected_roles
    ):
        raise ValueError("morning Decision batch lineage does not match the freeze")
    if (
        features.capability_report.model_dump(mode="json")
        != freeze_metadata.capability_report.model_dump(mode="json")
        or features.manifest.feature_set_version != batch.predictions[0].feature_version
    ):
        raise ValueError("morning Decision feature capability or manifest lineage is inconsistent")
    expected_symbols = set(batch.universe_symbols)
    if set(securities) != expected_symbols or set(account_bucket_ids_by_symbol) != expected_symbols:
        raise ValueError("morning Decision adapter mappings must cover the exact freeze universe")
    current = features.frame.loc[
        pd.to_datetime(features.frame["trading_date"])
        .dt.normalize()
        .eq(pd.Timestamp(batch.as_of.astimezone(JST).date()))
    ].copy()
    if set(current["symbol"].astype(str)) != expected_symbols:
        raise ValueError("morning Decision adapter requires every frozen market row")
    if (
        _generic_frame_hash(current, sort_columns=("trading_date", "symbol"))
        != batch.freeze_evidence_hash
    ):
        raise ValueError("morning Decision current feature evidence does not match the batch")
    current_as_of = _aware_timestamps(current["as_of"], "as_of")
    current_available = _aware_timestamps(current["available_at"], "available_at")
    current_snapshots = {
        str(source_id) for values in current["source_snapshot_ids"] for source_id in values
    }
    current_records = {
        str(source_id) for values in current["source_record_ids"] for source_id in values
    }
    if (
        any(value != batch.as_of for value in current_as_of)
        or (current_available > current_as_of).any()
        or set(current["provider"].astype(str)) != {batch.provider}
        or current_snapshots != set(batch.source_snapshot_ids)
        or current_records != set(batch.source_record_ids)
    ):
        raise ValueError("morning Decision frozen market rows do not match batch lineage")
    _require_current_freeze_roles(current, freeze_metadata)
    expected_holding_symbols = {
        member.symbol
        for member in freeze_metadata.universe
        if member.role.value in {"HOLDING", "HOLDING_AND_CANDIDATE"}
    }
    portfolio_holding_symbols = {
        position.symbol for position in portfolio.positions if position.shares > 0
    }
    if portfolio_holding_symbols != expected_holding_symbols:
        raise ValueError("morning freeze holding roles do not match the current portfolio")
    prediction_map = {prediction.symbol: prediction for prediction in batch.predictions}
    frozen_market = {item.symbol: item for item in batch.frozen_market}
    if any(
        position.market_price != frozen_market[position.symbol].reference_price
        for position in portfolio.positions
        if position.shares > 0
    ):
        raise ValueError("current holding prices do not match the frozen 11:30 market")
    candidates: list[DecisionCandidate] = []
    for row in current.sort_values("symbol", kind="stable").itertuples(index=False):
        symbol = str(row.symbol)
        security = securities[symbol]
        if security.symbol != symbol or security.sector != str(row.sector):
            raise ValueError("morning Decision security mapping has a symbol or sector mismatch")
        bucket_ids = account_bucket_ids_by_symbol[symbol]
        if not bucket_ids or len(bucket_ids) != len(set(bucket_ids)):
            raise ValueError(
                "morning Decision account-bucket mappings must be non-empty and unique"
            )
        raw_price = float(cast(Any, row.reference_price_1130))
        raw_adv = float(cast(Any, row.average_daily_trading_value))
        if not math.isfinite(raw_price) or not math.isfinite(raw_adv):
            raise ValueError("morning Decision market price and liquidity must be finite")
        reference_price = Decimal(str(raw_price))
        average_daily_trading_value = Decimal(str(raw_adv))
        if reference_price <= 0 or average_daily_trading_value <= 0:
            raise ValueError("morning Decision market price and liquidity must be positive")
        frozen = frozen_market[symbol]
        if (
            reference_price != frozen.reference_price
            or average_daily_trading_value != frozen.average_daily_trading_value
        ):
            raise ValueError("morning Decision market data does not match the frozen batch")
        candidates.extend(
            DecisionCandidate(
                security=security,
                account_bucket_id=bucket_id,
                price=reference_price,
                average_daily_trading_value=average_daily_trading_value,
                prediction=prediction_map[symbol],
            )
            for bucket_id in bucket_ids
        )
    audit = MorningDecisionAudit(
        as_of=batch.as_of,
        provider=batch.provider,
        source_snapshot_ids=batch.source_snapshot_ids,
        source_record_ids=batch.source_record_ids,
        universe_roles=expected_roles,
        capability_statuses=dict(batch.capability_statuses),
        research_report_id=batch.research_report_id,
        freeze_evidence_hash=batch.freeze_evidence_hash,
        reference_prices={item.symbol: item.reference_price for item in batch.frozen_market},
        average_daily_trading_values={
            item.symbol: item.average_daily_trading_value for item in batch.frozen_market
        },
    )
    return engine.propose(
        portfolio=portfolio,
        candidates=tuple(candidates),
        generated_at=generated_at,
        model_bundle_version=batch.predictions[0].model_version,
        morning_audit=audit,
    )


def _validate_research_input(
    dataset: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    config: MorningResearchConfig,
) -> None:
    if len(dataset) > config.max_rows:
        raise ValueError("BLOCKED_BY_RESOURCE_CAPABILITY: morning row bound exceeded")
    try:
        manifest_hash = _manifest_hash_for(feature_names)
    except ValueError:
        manifest_hash = ""
    if not manifest_hash:
        raise ValueError("morning research requires the authenticated F13 or F14 manifest")
    required = {
        "symbol",
        "trading_date",
        "as_of",
        "prior_prediction_as_of",
        "prior_model_version",
        "prior_feature_version",
        "prior_data_snapshot_id",
        "morning.is_current_holding",
        "morning.is_candidate",
        *feature_names,
    }
    for horizon in config.horizons:
        required.update(
            {
                f"target_return_{horizon}d",
                f"revision_target_{horizon}d",
                f"label_end_date_{horizon}d",
                f"label_available_at_{horizon}d",
            }
        )
    if missing := sorted(required - set(dataset.columns)):
        raise ValueError(f"morning research dataset is missing: {', '.join(missing)}")
    if dataset.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("morning research rows must be unique by symbol-date")
    if not dataset["morning.is_current_holding"].eq(1.0).any():
        raise ValueError("morning research must include current-holding rows")
    if not dataset["morning.is_candidate"].eq(1.0).any():
        raise ValueError("morning research must include candidate rows")
    for raw in dataset["as_of"]:
        value = _aware_datetime(raw, "as_of")
        local = value.astimezone(JST)
        if (local.hour, local.minute, local.second, local.microsecond) != (11, 30, 0, 0):
            raise ValueError("morning research as_of must be exactly 11:30 JST")


def _morning_splitter(
    config: MorningResearchConfig, horizon: int
) -> PurgedExpandingWindowSplitter:
    return PurgedExpandingWindowSplitter(
        initial_train_periods=config.initial_train_periods,
        validation_periods=config.validation_periods,
        step_periods=config.step_periods,
        purge_periods=0,
        embargo_periods=horizon,
        label_horizon_periods=horizon,
    )


def _generate_morning_oof(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    target_column: str,
    realized_return_column: str,
    label_end_column: str,
    label_available_column: str,
    horizon: int,
    family: MorningModelFamily,
    seed: int,
    config: MorningResearchConfig,
) -> pd.DataFrame:
    splitter = _morning_splitter(config, horizon)
    outputs: list[pd.DataFrame] = []
    for fold in splitter.split(frame, label_end_column=label_end_column):
        train = frame.iloc[list(fold.train_indices)].copy()
        validation = frame.iloc[list(fold.validation_indices)].copy()
        validation_as_of = min(_aware_timestamps(validation["as_of"], "as_of"))
        train_available = _aware_timestamps(
            train[label_available_column], label_available_column
        )
        train = train.loc[train[target_column].notna()]
        train = train.loc[train_available < validation_as_of]
        validation = validation.loc[validation[target_column].notna()]
        if train.empty or validation.empty:
            continue
        model = _morning_regressor(family, seed=seed, config=config)
        train_x = train.loc[:, list(feature_names)].replace([np.inf, -np.inf], np.nan)
        validation_x = validation.loc[:, list(feature_names)].replace([np.inf, -np.inf], np.nan)
        model.fit(train_x, train[target_column].astype(float))
        revision = np.asarray(model.predict(validation_x), dtype=float)
        if revision.ndim != 1 or len(revision) != len(validation):
            raise RuntimeError("morning model emitted a prediction vector with the wrong shape")
        if not np.isfinite(revision).all():
            raise RuntimeError("morning model emitted non-finite predictions")
        prior_name = f"morning.prior_expected_return_{horizon}d"
        prior_prediction = validation[prior_name].to_numpy(dtype=float)
        final_prediction = prior_prediction + revision
        output_columns = [
            "symbol",
            "trading_date",
            "as_of",
            "provider",
            "source_snapshot_ids",
            "source_record_ids",
            "reference_price_1130",
            "average_daily_trading_value",
            "prior_prediction_as_of",
            "prior_model_version",
            "prior_feature_version",
            "prior_data_snapshot_id",
            "morning.prior_downside_quantile",
            "morning.prior_large_loss_probability",
            "morning.prior_uncertainty",
            "morning.is_current_holding",
            "morning.is_candidate",
            realized_return_column,
            label_end_column,
            label_available_column,
        ]
        output = (
            validation.loc[:, output_columns]
            .copy()
            .rename(
                columns={
                    "morning.prior_downside_quantile": "prior_downside_quantile",
                    "morning.prior_large_loss_probability": "prior_large_loss_probability",
                    "morning.prior_uncertainty": "prior_uncertainty",
                    "morning.is_current_holding": "is_current_holding",
                    "morning.is_candidate": "is_candidate",
                    realized_return_column: "target",
                    label_end_column: "label_end",
                    label_available_column: "label_available_at",
                }
            )
        )
        output["horizon"] = horizon
        output["model_family"] = family
        output["seed"] = seed
        output["fold"] = fold.fold_number
        output["prior_prediction"] = prior_prediction
        output["predicted_revision"] = revision
        output["final_prediction"] = final_prediction
        output["revision_improved_abs_error"] = np.abs(
            output["target"] - final_prediction
        ) < np.abs(output["target"] - prior_prediction)
        outputs.append(output)
    if not outputs:
        raise ValueError(f"BLOCKED_BY_VALIDATION: no morning OOF rows for {family}/{horizon}d")
    return pd.concat(outputs, ignore_index=True)


def _morning_regressor(
    family: MorningModelFamily, *, seed: int, config: MorningResearchConfig
) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    ]
    if family == "ridge":
        steps.extend((("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))))
    elif family == "lightgbm":
        steps.append(
            (
                "model",
                lgb.LGBMRegressor(
                    objective="regression",
                    n_estimators=config.lightgbm_estimators,
                    learning_rate=0.03,
                    num_leaves=15,
                    max_depth=5,
                    min_child_samples=20,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    random_state=seed,
                    n_jobs=1,
                    verbosity=-1,
                ),
            )
        )
    else:
        steps.extend(
            (
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=config.mlp_hidden_units,
                        activation="relu",
                        solver="lbfgs",
                        alpha=0.001,
                        max_iter=config.mlp_max_iterations,
                        random_state=seed,
                    ),
                ),
            )
        )
    return Pipeline(steps)


def _summarize_model(oof: pd.DataFrame, *, config: MorningResearchConfig) -> MorningModelResult:
    model = evaluate_cross_sectional_predictions(
        dates=oof["trading_date"], target=oof["target"], prediction=oof["final_prediction"]
    )
    baseline = evaluate_cross_sectional_predictions(
        dates=oof["trading_date"], target=oof["target"], prediction=oof["prior_prediction"]
    )
    increment = (
        model.mean_daily_rank_ic - baseline.mean_daily_rank_ic
        if model.mean_daily_rank_ic is not None and baseline.mean_daily_rank_ic is not None
        else None
    )
    adds_value = (
        model.mean_squared_error < baseline.mean_squared_error
        and increment is not None
        and increment > config.minimum_rank_ic_increment
    )
    family = str(oof["model_family"].iloc[0])
    return MorningModelResult(
        horizon=int(oof["horizon"].iloc[0]),
        model_family=family,  # type: ignore[arg-type]
        seed=int(oof["seed"].iloc[0]),
        folds=int(oof["fold"].nunique()),
        rows=len(oof),
        dates=int(pd.to_datetime(oof["trading_date"]).nunique()),
        holdings_rows=int(oof["is_current_holding"].astype(bool).sum()),
        candidate_rows=int(oof["is_candidate"].astype(bool).sum()),
        mean_squared_error=model.mean_squared_error,
        baseline_mean_squared_error=baseline.mean_squared_error,
        mean_daily_rank_ic=model.mean_daily_rank_ic,
        baseline_mean_daily_rank_ic=baseline.mean_daily_rank_ic,
        incremental_rank_ic=increment,
        revision_win_rate=float(oof["revision_improved_abs_error"].mean()),
        adds_oos_value=adds_value,
        disposition=(
            MorningModelDisposition.RESEARCH if adds_value else MorningModelDisposition.REJECTED
        ),
        inference_timing_status="UNMEASURED_LIVE_RESEARCH_ONLY",
    )


def _model_capabilities(
    config: MorningResearchConfig,
    results: tuple[MorningModelResult, ...],
) -> tuple[MorningModelCapability, ...]:
    mlp_results = tuple(result for result in results if result.model_family == "mlp")
    if not config.enable_neural_challenger:
        mlp_disposition = MorningModelDisposition.DISABLED
        mlp_reason = "disabled by default until explicitly enabled for research"
    elif mlp_results and all(result.adds_oos_value for result in mlp_results):
        mlp_disposition = MorningModelDisposition.RESEARCH
        mlp_reason = (
            "bounded tabular challenger improved every configured horizon/seed; "
            "live inference timing remains unmeasured and adoption is prohibited"
        )
    else:
        mlp_disposition = MorningModelDisposition.REJECTED
        mlp_reason = "multi-seed OOF stability/value requirement was not met"
    return (
        MorningModelCapability(
            model_name="small_mlp",
            disposition=mlp_disposition,
            reason=mlp_reason,
        ),
        *(
            MorningModelCapability(
                model_name=name,
                disposition=MorningModelDisposition.BLOCKED_BY_DATA_CAPABILITY,
                reason="synchronized fixed-frequency morning sequence history is unavailable",
            )
            for name in ("1d_cnn", "tcn", "gru", "small_transformer")
        ),
    )


def _manifest_hash_for(feature_names: tuple[str, ...]) -> str:
    micro = tuple(name for name in feature_names if name not in MORNING_CORE_MANIFEST.feature_names)
    manifest = morning_feature_manifest(micro)
    if manifest.feature_names != feature_names:
        raise ValueError("unknown morning feature manifest")
    return str(manifest.manifest_hash)


def _validate_snapshot_frame(
    dataset: pd.DataFrame,
    *,
    publication_as_of: datetime,
    feature_names: tuple[str, ...],
    trading_calendar: pd.DatetimeIndex,
) -> None:
    required = {
        "symbol",
        "trading_date",
        "as_of",
        "available_at",
        "provider",
        "source_snapshot_ids",
        "source_record_ids",
        "prior_prediction_as_of",
        "prior_model_version",
        "prior_feature_version",
        "prior_data_snapshot_id",
        *feature_names,
    }
    for horizon in HORIZONS:
        required.update(
            {
                f"target_return_{horizon}d",
                f"label_entry_at_{horizon}d",
                f"label_end_date_{horizon}d",
                f"label_end_at_{horizon}d",
                f"label_available_at_{horizon}d",
                f"label_status_{horizon}d",
                f"revision_target_{horizon}d",
            }
        )
    if missing := sorted(required - set(dataset.columns)):
        raise ValueError(f"morning snapshot frame is missing: {', '.join(missing)}")
    if dataset.empty or dataset.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("morning snapshot rows must be non-empty and unique by symbol-date")
    registered_micro = set(MORNING_MICROSTRUCTURE_MANIFEST.feature_names) - set(
        MORNING_CORE_MANIFEST.feature_names
    )
    if unexpected_micro := sorted((set(dataset.columns) & registered_micro) - set(feature_names)):
        raise ValueError(
            "morning snapshot contains microstructure fields outside its manifest: "
            + ", ".join(unexpected_micro)
        )
    calendar = _canonical_trading_calendar(trading_calendar)
    calendar_positions = {value: position for position, value in enumerate(calendar)}
    trading_dates = pd.to_datetime(dataset["trading_date"]).dt.normalize()
    if not set(trading_dates) <= set(calendar):
        raise ValueError("morning snapshot dates are outside its fixed JPX calendar")
    publication = pd.Timestamp(publication_as_of)
    as_of = _aware_timestamps(dataset["as_of"], "as_of")
    available_at = _aware_timestamps(dataset["available_at"], "available_at")
    if (as_of > publication).any():
        raise ValueError("morning snapshot contains a feature after publication as_of")
    if (available_at > publication).any():
        raise ValueError("morning snapshot contains a source after publication as_of")
    if (available_at > as_of).any():
        raise ValueError("morning snapshot contains a source received after its 11:30 freeze")
    prior_as_of = _aware_timestamps(dataset["prior_prediction_as_of"], "prior_prediction_as_of")
    if (prior_as_of > as_of).any():
        raise ValueError("morning snapshot contains a daily prior after its 11:30 freeze")
    for value in as_of:
        local = value.astimezone(JST)
        if (local.hour, local.minute, local.second, local.microsecond) != (11, 30, 0, 0):
            raise ValueError("morning snapshot rows must use the exact 11:30 JST freeze")
    for column in ("source_snapshot_ids", "source_record_ids"):
        for source_ids in dataset[column]:
            if isinstance(source_ids, str) or not tuple(source_ids):
                raise ValueError(f"morning snapshot rows require non-empty {column}")
    for horizon in HORIZONS:
        target = f"target_return_{horizon}d"
        entry = _aware_timestamps(
            dataset[f"label_entry_at_{horizon}d"], f"label_entry_at_{horizon}d"
        )
        end = pd.to_datetime(dataset[f"label_end_date_{horizon}d"])
        endpoint_at = _aware_timestamps(
            dataset[f"label_end_at_{horizon}d"], f"label_end_at_{horizon}d"
        )
        available = _aware_timestamps(
            dataset[f"label_available_at_{horizon}d"],
            f"label_available_at_{horizon}d",
        )
        immature = available > publication
        if dataset.loc[immature, target].notna().any():
            raise ValueError("morning snapshot retains an outcome unavailable at publication")
        if (entry <= as_of).any():
            raise ValueError("morning snapshot labels must begin after the 11:30 freeze")
        for entry_value, trading_date in zip(entry, trading_dates, strict=True):
            local_entry = entry_value.astimezone(JST)
            if (
                local_entry.date() != trading_date.date()
                or (local_entry.hour, local_entry.minute, local_entry.second) != (12, 30, 0)
            ):
                raise ValueError("morning snapshot label entry must be exact same-session 12:30")
        expected_end = pd.Series(
            [
                calendar[calendar_positions[trading_date] + horizon].date()
                if calendar_positions[trading_date] + horizon < len(calendar)
                else None
                for trading_date in trading_dates
            ],
            index=dataset.index,
        )
        if expected_end.isna().any() or not end.dt.date.equals(expected_end):
            raise ValueError("morning snapshot label endpoints violate the fixed JPX calendar")
        if (available < entry).any():
            raise ValueError("morning snapshot labels cannot be available before entry")
        endpoint_dates = endpoint_at.map(lambda value: value.astimezone(JST).date())
        if (endpoint_at <= entry).any() or not endpoint_dates.equals(end.dt.date):
            raise ValueError("morning snapshot label endpoint timestamps are inconsistent")
        if (available < endpoint_at).any():
            raise ValueError("morning snapshot labels cannot be available before their endpoint")
        values = pd.to_numeric(dataset[target], errors="coerce")
        revision = pd.to_numeric(dataset[f"revision_target_{horizon}d"], errors="coerce")
        available_status = dataset[f"label_status_{horizon}d"].eq("AVAILABLE")
        if values.loc[available_status].isna().any() or values.loc[~available_status].notna().any():
            raise ValueError("morning snapshot label status/value contract is inconsistent")
        prior = pd.to_numeric(
            dataset[f"morning.prior_expected_return_{horizon}d"], errors="coerce"
        )
        expected_revision = values - prior
        if (
            revision.loc[available_status].isna().any()
            or revision.loc[~available_status].notna().any()
            or not np.allclose(
                revision.loc[available_status].to_numpy(dtype=float),
                expected_revision.loc[available_status].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError("morning snapshot revision targets are inconsistent")


def _canonical_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["trading_date"] = pd.to_datetime(output["trading_date"]).dt.normalize()
    timestamp_columns = ["as_of", "available_at"]
    for horizon in HORIZONS:
        timestamp_columns.extend(
            [
                f"label_entry_at_{horizon}d",
                f"label_end_at_{horizon}d",
                f"label_available_at_{horizon}d",
            ]
        )
        output[f"label_end_date_{horizon}d"] = pd.to_datetime(
            output[f"label_end_date_{horizon}d"]
        ).dt.normalize()
    for column in timestamp_columns:
        output[column] = pd.to_datetime(output[column], utc=True)
    for column in ("source_snapshot_ids", "source_record_ids"):
        output[column] = output[column].map(lambda values: tuple(str(value) for value in values))
    return output.sort_values(["trading_date", "symbol"], kind="stable").reset_index(drop=True)


def _report_identity(report: MorningResearchReport) -> dict[str, object]:
    payload = report.model_dump(mode="json")
    return {key: value for key, value in payload.items() if key not in {"report_id", "created_at"}}


def _capability_statuses(report: MorningCapabilityReport) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (str(name), str(getattr(status, "value", status)))
            for name, status in report.capabilities.items()
        )
    )


def _blocked_morning_batch(
    freeze_metadata: MorningFreezeMetadata,
    *,
    research_report_id: str,
    reason: str,
) -> MorningDecisionPredictionBatch:
    return MorningDecisionPredictionBatch(
        as_of=freeze_metadata.as_of,
        universe_symbols=tuple(sorted(member.symbol for member in freeze_metadata.universe)),
        universe=freeze_metadata.universe,
        provider=freeze_metadata.provider,
        source_snapshot_ids=freeze_metadata.source_snapshot_ids,
        source_record_ids=freeze_metadata.source_record_ids,
        capability_statuses=_capability_statuses(freeze_metadata.capability_report),
        research_report_id=research_report_id,
        freeze_evidence_hash=_stable_hash(
            {
                "as_of": freeze_metadata.as_of.isoformat(),
                "provider": freeze_metadata.provider,
                "source_snapshot_ids": freeze_metadata.source_snapshot_ids,
                "source_record_ids": freeze_metadata.source_record_ids,
                "universe": tuple(
                    (member.symbol, member.role.value) for member in freeze_metadata.universe
                ),
                "reason": reason,
            }
        ),
        evidence_kind="BLOCKED",
        frozen_market=(),
        predictions=(),
        blocked=(reason,),
    )


def _frozen_market_from_frame(frame: pd.DataFrame) -> tuple[MorningDecisionMarketData, ...]:
    required = {"symbol", "reference_price_1130", "average_daily_trading_value"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(
            f"morning Decision evidence is missing market fields: {', '.join(missing)}"
        )
    market: list[MorningDecisionMarketData] = []
    for symbol, rows in frame.groupby(frame["symbol"].astype(str), sort=True):
        prices = pd.to_numeric(rows["reference_price_1130"], errors="coerce").drop_duplicates()
        liquidity = pd.to_numeric(
            rows["average_daily_trading_value"], errors="coerce"
        ).drop_duplicates()
        if (
            len(prices) != 1
            or len(liquidity) != 1
            or not math.isfinite(float(prices.iloc[0]))
            or not math.isfinite(float(liquidity.iloc[0]))
        ):
            raise ValueError("morning Decision market evidence must be finite and consistent")
        market.append(
            MorningDecisionMarketData(
                symbol=str(symbol),
                reference_price=Decimal(str(float(prices.iloc[0]))),
                average_daily_trading_value=Decimal(str(float(liquidity.iloc[0]))),
            )
        )
    return tuple(market)


def _require_current_freeze_roles(
    frame: pd.DataFrame, freeze_metadata: MorningFreezeMetadata
) -> None:
    expected = {member.symbol: member.role.value for member in freeze_metadata.universe}
    observed: dict[str, str] = {}
    holding_column = (
        "morning.is_current_holding"
        if "morning.is_current_holding" in frame.columns
        else "is_current_holding"
    )
    candidate_column = (
        "morning.is_candidate" if "morning.is_candidate" in frame.columns else "is_candidate"
    )
    for _, row in frame.iterrows():
        holding = bool(row[holding_column])
        candidate = bool(row[candidate_column])
        role = (
            "HOLDING_AND_CANDIDATE"
            if holding and candidate
            else "HOLDING"
            if holding
            else "CANDIDATE"
            if candidate
            else "NEITHER"
        )
        symbol = str(row["symbol"])
        if symbol in observed and observed[symbol] != role:
            raise ValueError("morning freeze roles differ across horizon rows")
        observed[symbol] = role
    if observed != expected:
        raise ValueError("morning feature roles do not match the freeze metadata")


def _fitted_bundle_identity(
    bundle: MorningFittedResearchBundle, *, model_hashes: Mapping[int, str]
) -> dict[str, object]:
    return {
        "report_id": bundle.report.report_id,
        "snapshot_id": bundle.snapshot.snapshot_id,
        "selected_family": bundle.selected_family,
        "selected_seed": bundle.selected_seed,
        "training_as_of": bundle.training_as_of.isoformat(),
        "model_hashes": dict(sorted(model_hashes.items())),
    }


def _verify_fitted_bundle(bundle: MorningFittedResearchBundle) -> None:
    if bundle._construction_token is not _BUNDLE_CONSTRUCTION_TOKEN:
        raise ValueError("morning fitted bundle bypassed the authenticated refit path")
    if not bundle.is_research_only:
        raise ValueError("morning fitted bundle must remain research-only")
    expected_report_id = f"morning-{_stable_hash(_report_identity(bundle.report))[:24]}"
    if bundle.report.report_id != expected_report_id:
        raise ValueError("morning fitted bundle report identity is invalid")
    snapshot = bundle.snapshot
    snapshot_identity = _morning_snapshot_identity(snapshot)
    if snapshot.snapshot_id != _stable_hash(snapshot_identity):
        raise ValueError("morning fitted bundle snapshot identity is invalid")
    if (
        bundle.report.data_snapshot_id != snapshot.snapshot_id
        or bundle.report.feature_manifest_hash != snapshot.feature_manifest_hash
        or bundle.report.feature_names != snapshot.feature_names
        or bundle.report.feature_set_version != snapshot.feature_set_version
    ):
        raise ValueError("morning fitted bundle report/snapshot provenance mismatch")
    selected_results = tuple(
        result
        for result in bundle.report.results
        if result.model_family == bundle.selected_family and result.seed == bundle.selected_seed
    )
    if set(result.horizon for result in selected_results) != set(HORIZONS) or any(
        result.disposition is not MorningModelDisposition.RESEARCH for result in selected_results
    ):
        raise ValueError("morning fitted bundle selection lacks 1d/5d/20d research evidence")
    expected_training_as_of = datetime.combine(
        pd.Timestamp(bundle.report.holdout_start).date(), time(11, 30), tzinfo=JST
    )
    if bundle.training_as_of != expected_training_as_of:
        raise ValueError("morning fitted bundle training boundary is invalid")
    if set(bundle.models) != set(HORIZONS) or set(bundle.model_hashes) != set(HORIZONS):
        raise ValueError("morning fitted bundle requires one model for every horizon")
    config = MorningResearchConfig.model_validate(dict(bundle.report.config))
    for model in bundle.models.values():
        _verify_pipeline_contract(
            model,
            family=bundle.selected_family,
            seed=bundle.selected_seed,
            config=config,
        )
    observed_hashes = {
        horizon: _pipeline_state_hash(model) for horizon, model in bundle.models.items()
    }
    if observed_hashes != dict(bundle.model_hashes):
        raise ValueError("morning fitted bundle model state was mutated")
    identity = _fitted_bundle_identity(bundle, model_hashes=observed_hashes)
    if bundle.bundle_id != f"morning-fit-{_stable_hash(identity)[:24]}":
        raise ValueError("morning fitted bundle identity is invalid")


def _verify_pipeline_contract(
    model: Pipeline,
    *,
    family: MorningModelFamily,
    seed: int,
    config: MorningResearchConfig,
) -> None:
    expected_steps = (
        ("imputer", "scaler", "model")
        if family in {"ridge", "mlp"}
        else ("imputer", "model")
    )
    if tuple(model.named_steps) != expected_steps:
        raise ValueError("morning fitted bundle pipeline does not match its selected family")
    estimator = model.named_steps["model"]
    if family == "ridge":
        valid = isinstance(estimator, Ridge) and estimator.alpha == 1.0
    elif family == "lightgbm":
        valid = (
            isinstance(estimator, lgb.LGBMRegressor)
            and estimator.random_state == seed
            and estimator.n_estimators == config.lightgbm_estimators
        )
    else:
        valid = (
            isinstance(estimator, MLPRegressor)
            and estimator.random_state == seed
            and estimator.hidden_layer_sizes == config.mlp_hidden_units
            and estimator.max_iter == config.mlp_max_iterations
        )
    if not valid:
        raise ValueError("morning fitted bundle estimator does not match its research evidence")


def _morning_snapshot_identity(snapshot: MorningDatasetSnapshot) -> dict[str, object]:
    return {
        "publication_as_of": snapshot.publication_as_of.isoformat(),
        "provider": snapshot.provider,
        "source_snapshot_ids": snapshot.source_snapshot_ids,
        "source_record_ids": snapshot.source_record_ids,
        "feature_set_version": snapshot.feature_set_version,
        "feature_manifest_hash": snapshot.feature_manifest_hash,
        "feature_names": snapshot.feature_names,
        "capability_statuses": snapshot.capability_statuses,
        "trading_calendar_dates": snapshot.trading_calendar_dates,
        "frame_hash": snapshot.frame_hash,
        "rows": snapshot.rows,
        "data_start": snapshot.data_start,
        "data_end": snapshot.data_end,
    }


def _pipeline_state_hash(model: Pipeline) -> str:
    return hashlib.sha256(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _verify_snapshot_identity(snapshot: MorningDatasetSnapshot) -> None:
    if snapshot.snapshot_id != _stable_hash(_morning_snapshot_identity(snapshot)):
        raise ValueError("morning dataset snapshot identity is invalid")


def _verify_snapshot_frame_metadata(
    snapshot: MorningDatasetSnapshot, frame: pd.DataFrame
) -> None:
    trading_dates = pd.to_datetime(frame["trading_date"])
    if (
        len(frame) != snapshot.rows
        or str(trading_dates.min().date()) != snapshot.data_start
        or str(trading_dates.max().date()) != snapshot.data_end
    ):
        raise ValueError("morning dataset snapshot range metadata is invalid")


def _frame_hash(frame: pd.DataFrame) -> str:
    return _generic_frame_hash(
        frame,
        sort_columns=("horizon", "model_family", "seed", "trading_date", "symbol"),
    )


def _generic_frame_hash(frame: pd.DataFrame, *, sort_columns: tuple[str, ...]) -> str:
    canonical = frame.sort_values(list(sort_columns), kind="stable").reset_index(drop=True)
    schema = tuple((str(column), str(dtype)) for column, dtype in canonical.dtypes.items())
    schema_bytes = json.dumps(schema, separators=(",", ":")).encode()
    hashable = canonical.copy()
    for column in hashable.select_dtypes(include="object").columns:
        hashable[column] = hashable[column].map(_hashable_cell)
    row_bytes = (
        pd.util.hash_pandas_object(hashable, index=False).to_numpy(dtype=np.uint64).tobytes()
    )
    return hashlib.sha256(schema_bytes + row_bytes).hexdigest()


def _hashable_cell(value: object) -> object:
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aware_timestamps(values: pd.Series, name: str) -> pd.Series:
    return pd.Series(
        [_aware_datetime(value, name) for value in values], index=values.index, dtype="object"
    )


def _canonical_trading_calendar(
    values: Iterable[date | datetime | pd.Timestamp],
) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex(pd.to_datetime(tuple(values))).normalize()
    if len(calendar) < 21 or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("morning research requires a sorted unique fixed JPX trading calendar")
    return calendar


def _aware_datetime(value: Any, name: str) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"morning {name} must be timezone-aware")
    return parsed.to_pydatetime()
