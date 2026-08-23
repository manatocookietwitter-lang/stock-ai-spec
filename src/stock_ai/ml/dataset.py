"""Deterministic point-in-time supervised dataset snapshots."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from stock_ai.data.point_in_time import assert_point_in_time
from stock_ai.features.registry import FeatureSetManifest

HORIZONS: Final = (1, 5, 20)


class DatasetSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    as_of: datetime
    feature_set_id: str
    feature_set_version: str
    feature_manifest_hash: str
    target_definition: str
    horizons: tuple[int, ...]
    rows: int = Field(ge=0)
    parquet_path: Path
    metadata_path: Path

    @field_validator("created_at", "as_of")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot timestamps must be timezone-aware")
        return value


def build_supervised_dataset(feature_history: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "trading_date", "available_at", "adjusted_close"}
    missing = required - set(feature_history.columns)
    if missing:
        raise ValueError(f"feature history missing dataset columns: {sorted(missing)}")
    frame = feature_history.copy()
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.normalize()
    frame["available_at"] = pd.to_datetime(frame["available_at"], utc=True)
    frame["as_of"] = frame["available_at"]
    frame = frame.sort_values(["symbol", "trading_date"]).reset_index(drop=True)
    if frame.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("feature history contains duplicate symbol/trading_date rows")
    calendars = [
        tuple(group["trading_date"])
        for _, group in frame.groupby("symbol", sort=True, observed=True)
    ]
    if calendars and any(calendar != calendars[0] for calendar in calendars[1:]):
        raise ValueError(
            "symbols must share one explicit trading calendar; "
            "suspension/delisting policy is required"
        )
    grouped_close = frame.groupby("symbol", sort=False)["adjusted_close"]
    grouped_date = frame.groupby("symbol", sort=False)["trading_date"]
    grouped_availability = frame.groupby("symbol", sort=False)["available_at"]
    for horizon in HORIZONS:
        future_close = grouped_close.shift(-horizon)
        frame[f"target_return_{horizon}d"] = future_close / frame["adjusted_close"] - 1
        frame[f"label_end_date_{horizon}d"] = grouped_date.shift(-horizon)
        frame[f"label_available_at_{horizon}d"] = grouped_availability.shift(-horizon)
    assert_point_in_time(frame)
    return frame.sort_values(["trading_date", "symbol"]).reset_index(drop=True)


def _frame_hash(frame: pd.DataFrame, manifest: FeatureSetManifest, as_of: datetime) -> str:
    canonical = frame.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
    row_hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy().tobytes()
    metadata = json.dumps(
        {"manifest_hash": manifest.manifest_hash, "as_of": as_of.isoformat(), "horizons": HORIZONS},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(metadata + row_hashes).hexdigest()


def write_dataset_snapshot(
    dataset: pd.DataFrame,
    destination: Path,
    *,
    manifest: FeatureSetManifest,
    as_of: datetime,
    created_at: datetime,
) -> DatasetSnapshot:
    if as_of.tzinfo is None or created_at.tzinfo is None:
        raise ValueError("snapshot timestamps must be timezone-aware")
    safe = dataset.loc[pd.to_datetime(dataset["as_of"], utc=True) <= pd.Timestamp(as_of)].copy()
    cutoff = pd.Timestamp(as_of)
    for horizon in HORIZONS:
        availability_column = f"label_available_at_{horizon}d"
        required = {
            f"target_return_{horizon}d",
            f"label_end_date_{horizon}d",
            availability_column,
        }
        missing = required - set(safe.columns)
        if missing:
            raise ValueError(f"dataset is missing label audit columns: {sorted(missing)}")
        label_available = pd.to_datetime(safe[availability_column], utc=True)
        not_mature = label_available.isna() | (label_available > cutoff)
        safe.loc[not_mature, list(required)] = pd.NA
    snapshot_id = _frame_hash(safe, manifest, as_of)
    destination.mkdir(parents=True, exist_ok=True)
    parquet_path = destination / f"{snapshot_id}.parquet"
    metadata_path = destination / f"{snapshot_id}.json"
    target_definition = "adjusted-close to adjusted-close absolute return; fixture research proxy"
    snapshot_created_at = created_at
    if parquet_path.exists() or metadata_path.exists():
        if not (parquet_path.exists() and metadata_path.exists()):
            raise RuntimeError("partial immutable snapshot already exists")
        existing = pd.read_parquet(parquet_path)
        if _frame_hash(existing, manifest, as_of) != snapshot_id:
            raise RuntimeError("existing immutable snapshot content failed hash validation")
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "snapshot_id": snapshot_id,
            "as_of": as_of.isoformat(),
            "feature_set_id": manifest.feature_set_id,
            "feature_set_version": manifest.feature_set_version,
            "feature_manifest_hash": manifest.manifest_hash,
            "target_definition": target_definition,
            "horizons": list(HORIZONS),
            "rows": len(safe),
            "parquet_path": str(parquet_path),
        }
        if any(metadata_payload.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError("existing immutable snapshot metadata failed validation")
        snapshot_created_at = datetime.fromisoformat(str(metadata_payload["created_at"]))
    else:
        unique = uuid4().hex
        parquet_tmp = destination / f".{snapshot_id}.{unique}.parquet.tmp"
        metadata_tmp = destination / f".{snapshot_id}.{unique}.json.tmp"
        safe.to_parquet(parquet_tmp, index=False)
        metadata_payload = {
            "snapshot_id": snapshot_id,
            "created_at": created_at.isoformat(),
            "as_of": as_of.isoformat(),
            "feature_set_id": manifest.feature_set_id,
            "feature_set_version": manifest.feature_set_version,
            "feature_manifest_hash": manifest.manifest_hash,
            "target_definition": target_definition,
            "horizons": list(HORIZONS),
            "rows": len(safe),
            "parquet_path": str(parquet_path),
        }
        metadata_tmp.write_text(
            json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        parquet_tmp.replace(parquet_path)
        metadata_tmp.replace(metadata_path)
    return DatasetSnapshot(
        snapshot_id=snapshot_id,
        created_at=snapshot_created_at,
        as_of=as_of,
        feature_set_id=manifest.feature_set_id,
        feature_set_version=manifest.feature_set_version,
        feature_manifest_hash=manifest.manifest_hash,
        target_definition=target_definition,
        horizons=HORIZONS,
        rows=len(safe),
        parquet_path=parquet_path,
        metadata_path=metadata_path,
    )
