"""Durable, interruption-safe manifests for bounded Goal 3 research batches."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from stock_ai.features import V2_EXTENDED_MANIFEST
from stock_ai.ml.advanced import AdvancedResearchConfig, load_advanced_research_run
from stock_ai.ml.checkpoint import checkpoint_id_for_provenance, read_checkpoint_status


class CampaignBatchStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INTERRUPTED = "INTERRUPTED"


class ResearchCampaignBatch(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    batch_id: str = Field(min_length=1)
    horizon: int
    model_family: str = Field(min_length=1)
    seed: int | None = None
    config_hash: str = Field(min_length=64, max_length=64)
    feature_names_hash: str = Field(min_length=64, max_length=64)
    status: CampaignBatchStatus = CampaignBatchStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    child_pid: int | None = Field(default=None, ge=1)
    report_id: str | None = None
    oof_path: str | None = None
    log_path: str | None = None
    last_error: str | None = None


class ResearchCampaignManifest(BaseModel):
    """Mutable resume state whose plan identity excludes execution timestamps."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: str = "research-campaign-manifest-v1"
    campaign_id: str = Field(min_length=64, max_length=64)
    build_id: str = Field(min_length=64, max_length=64)
    build_manifest_path: str = Field(min_length=1)
    code_commit: str = Field(min_length=1)
    report_root: str = Field(min_length=1)
    experiment_registry: str = Field(min_length=1)
    checkpoint_root: str | None = None
    common_config: dict[str, object]
    created_at: datetime
    updated_at: datetime
    batches: list[ResearchCampaignBatch]

    @model_validator(mode="after")
    def coherent_plan(self) -> ResearchCampaignManifest:
        if len({batch.batch_id for batch in self.batches}) != len(self.batches):
            raise ValueError("research campaign batch IDs must be unique")
        if self.campaign_id != campaign_identity(self):
            raise ValueError("research campaign identity mismatch")
        if self.schema_version == "research-campaign-manifest-v2":
            if self.checkpoint_root is None or any(batch.seed is None for batch in self.batches):
                raise ValueError("v2 campaign requires checkpoint root and per-seed batches")
        elif self.schema_version != "research-campaign-manifest-v1":
            raise ValueError("unsupported research campaign schema")
        return self


def create_campaign_manifest(
    *,
    build_id: str,
    build_manifest_path: Path,
    code_commit: str,
    report_root: Path,
    experiment_registry: Path,
    horizons: tuple[int, ...],
    model_families: tuple[str, ...],
    common_config: dict[str, object],
    checkpoint_root: Path | None = None,
    now: datetime | None = None,
) -> ResearchCampaignManifest:
    """Create recoverable batches, optionally split down to one deterministic seed."""

    timestamp = now or datetime.now(UTC)
    batches: list[ResearchCampaignBatch] = []
    seeds = tuple(
        int(cast(int, value)) for value in cast(tuple[object, ...], common_config["seeds"])
    )
    seed_groups: tuple[int | None, ...] = seeds if checkpoint_root is not None else (None,)
    for horizon in horizons:
        for family in model_families:
            for seed in seed_groups:
                batch_seeds = seeds if seed is None else (seed,)
                feature_names = tuple(
                    str(name) for name in cast(tuple[object, ...], common_config["feature_names"])
                )
                config = AdvancedResearchConfig.model_validate(
                    {
                        **{
                            key: value
                            for key, value in common_config.items()
                            if key != "feature_names"
                        },
                        "horizons": (horizon,),
                        "model_families": (family,),
                        "seeds": batch_seeds,
                    }
                )
                batches.append(
                    ResearchCampaignBatch(
                        batch_id=(
                            f"h{horizon}-{family}"
                            if seed is None
                            else f"h{horizon}-{family}-s{seed}"
                        ),
                        horizon=horizon,
                        model_family=family,
                        seed=seed,
                        config_hash=config.config_hash,
                        feature_names_hash=_stable_hash(feature_names),
                    )
                )
    schema_version = (
        "research-campaign-manifest-v2"
        if checkpoint_root is not None
        else "research-campaign-manifest-v1"
    )
    plan_id = _campaign_plan_identity(
        schema_version=schema_version,
        build_id=build_id,
        build_manifest_path=build_manifest_path,
        code_commit=code_commit,
        report_root=report_root,
        experiment_registry=experiment_registry,
        common_config=common_config,
        batches=batches,
        checkpoint_root=checkpoint_root,
    )
    values: dict[str, object] = {
        "schema_version": schema_version,
        "campaign_id": plan_id,
        "build_id": build_id,
        "build_manifest_path": str(build_manifest_path.resolve()),
        "code_commit": code_commit,
        "report_root": str(report_root.resolve()),
        "experiment_registry": str(experiment_registry.resolve()),
        "checkpoint_root": (
            None if checkpoint_root is None else str(checkpoint_root.resolve())
        ),
        "common_config": common_config,
        "created_at": timestamp,
        "updated_at": timestamp,
        "batches": batches,
    }
    return ResearchCampaignManifest.model_validate(values)


def campaign_identity(manifest: ResearchCampaignManifest) -> str:
    return _campaign_plan_identity(
        schema_version=manifest.schema_version,
        build_id=manifest.build_id,
        build_manifest_path=Path(manifest.build_manifest_path),
        code_commit=manifest.code_commit,
        report_root=Path(manifest.report_root),
        experiment_registry=Path(manifest.experiment_registry),
        common_config=manifest.common_config,
        batches=manifest.batches,
        checkpoint_root=(
            None if manifest.checkpoint_root is None else Path(manifest.checkpoint_root)
        ),
    )


def _campaign_plan_identity(
    *,
    schema_version: str,
    build_id: str,
    build_manifest_path: Path,
    code_commit: str,
    report_root: Path,
    experiment_registry: Path,
    common_config: dict[str, object],
    batches: list[ResearchCampaignBatch],
    checkpoint_root: Path | None,
) -> str:
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "build_id": build_id,
        "build_manifest_path": str(build_manifest_path.resolve()),
        "code_commit": code_commit,
        "report_root": str(report_root.resolve()),
        "experiment_registry": str(experiment_registry.resolve()),
        "common_config": common_config,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "horizon": batch.horizon,
                "model_family": batch.model_family,
                "config_hash": batch.config_hash,
                "feature_names_hash": batch.feature_names_hash,
                **({"seed": batch.seed} if schema_version.endswith("-v2") else {}),
            }
            for batch in batches
        ],
    }
    if schema_version.endswith("-v2"):
        payload["checkpoint_root"] = (
            None if checkpoint_root is None else str(checkpoint_root.resolve())
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_campaign_manifest(manifest: ResearchCampaignManifest, path: Path) -> None:
    """Atomically replace only the small resume marker after fsync."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.updated_at = datetime.now(UTC)
    payload = json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, indent=2, allow_nan=False
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_campaign_manifest(path: Path) -> ResearchCampaignManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("research campaign manifest is missing or invalid") from None
    return ResearchCampaignManifest.model_validate(payload)


def load_campaign_build_id(manifest_path: Path) -> str:
    """Authenticate the small build marker; each child authenticates all Parquet content."""

    return str(_load_campaign_build_payload(manifest_path)["build_id"])


def _load_campaign_build_payload(manifest_path: Path) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    build_id = manifest_path.stem
    if len(build_id) != 64 or manifest_path.parent.name != build_id:
        raise RuntimeError("production build path is not content-addressed")
    try:
        observed = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("production build manifest is missing or invalid") from None
    if not isinstance(observed, dict) or observed.get("build_id") != build_id:
        raise RuntimeError("production build identity mismatch")
    if Path(str(observed.get("manifest_path", ""))).resolve() != manifest_path:
        raise RuntimeError("production build manifest path metadata mismatch")
    metadata = {key: value for key, value in observed.items() if key != "metadata_hash"}
    metadata_hash = hashlib.sha256(
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if observed.get("metadata_hash") != metadata_hash:
        raise RuntimeError("production build metadata hash mismatch")
    return observed


def campaign_batch_checkpoint_path(
    manifest: ResearchCampaignManifest,
    batch: ResearchCampaignBatch,
) -> Path | None:
    """Derive a v2 batch checkpoint path without creating or changing progress."""

    if manifest.schema_version != "research-campaign-manifest-v2":
        return None
    if manifest.checkpoint_root is None or batch.seed is None:
        raise RuntimeError("v2 campaign lacks checkpoint or seed identity")
    build = _load_campaign_build_payload(Path(manifest.build_manifest_path))
    if str(build["build_id"]) != manifest.build_id:
        raise RuntimeError("campaign and Production Build identities differ")
    feature_names = tuple(
        str(name)
        for name in cast(tuple[object, ...], manifest.common_config["feature_names"])
    )
    config = AdvancedResearchConfig.model_validate(
        {
            **{
                key: value
                for key, value in manifest.common_config.items()
                if key != "feature_names"
            },
            "horizons": (batch.horizon,),
            "model_families": (batch.model_family,),
            "seeds": (batch.seed,),
        }
    )
    provenance: Mapping[str, object] = {
        "data_snapshot_id": str(build["dataset_snapshot_id"]),
        "feature_snapshot_id": str(build["v2_snapshot_id"]),
        "feature_manifest_hash": V2_EXTENDED_MANIFEST.manifest_hash,
        "feature_names": list(feature_names),
        "code_commit": manifest.code_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash,
        "locked_holdout_accessed": False,
    }
    checkpoint_id = checkpoint_id_for_provenance(provenance)
    return Path(manifest.checkpoint_root).resolve() / checkpoint_id


def read_campaign_status(manifest_path: Path) -> dict[str, object]:
    """Read batch and granular fold state once without reconciling or mutating it."""

    manifest = load_campaign_manifest(manifest_path.resolve())
    batches: list[dict[str, object]] = []
    for batch in manifest.batches:
        worker_alive = bool(
            batch.status is CampaignBatchStatus.RUNNING
            and batch.child_pid is not None
            and _process_is_running(batch.child_pid)
        )
        effective_status = batch.status.value
        if batch.status is CampaignBatchStatus.RUNNING and not worker_alive:
            effective_status = CampaignBatchStatus.INTERRUPTED.value
        checkpoint_path = campaign_batch_checkpoint_path(manifest, batch)
        checkpoint_summary: dict[str, object] | None = None
        if checkpoint_path is not None and checkpoint_path.is_dir():
            checkpoint = read_checkpoint_status(checkpoint_path)
            units = cast(dict[str, dict[str, Any]], checkpoint["units"])
            latest = max(
                units.values(),
                key=lambda item: str(item.get("updated_at", "")),
                default=None,
            )
            active_units = tuple(
                value
                for value in units.values()
                if value.get("status") == "RUNNING"
            )
            current = max(
                active_units or tuple(units.values()),
                key=lambda item: str(item.get("updated_at", "")),
                default=None,
            )
            checkpoint_summary = {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "updated_at": checkpoint["updated_at"],
                "unit_counts": checkpoint["unit_counts"],
                "active_units": active_units,
                "current_unit": current,
                "latest_unit": latest,
            }
        batches.append(
            {
                "batch_id": batch.batch_id,
                "horizon": batch.horizon,
                "model_family": batch.model_family,
                "seed": batch.seed,
                "stored_status": batch.status.value,
                "effective_status": effective_status,
                "attempts": batch.attempts,
                "worker_alive": worker_alive,
                "report_id": batch.report_id,
                "checkpoint": checkpoint_summary,
            }
        )
    current_batch = next(
        (
            item
            for item in batches
            if item["effective_status"] == CampaignBatchStatus.RUNNING.value
        ),
        next(
            (
                item
                for item in batches
                if item["effective_status"] != CampaignBatchStatus.SUCCEEDED.value
            ),
            None,
        ),
    )
    return {
        "schema_version": "research-campaign-status-v2",
        "checked_at": datetime.now(UTC).isoformat(),
        "campaign_id": manifest.campaign_id,
        "manifest": str(manifest_path.resolve()),
        "locked_holdout_accessed": False,
        "current_batch": current_batch,
        "batches": tuple(batches),
    }


def authenticate_batch_artifact(
    batch: ResearchCampaignBatch,
    *,
    code_commit: str,
    oof_path: Path,
) -> tuple[str, str]:
    """Authenticate a completed batch and require the exact immutable batch config."""

    report, _ = load_advanced_research_run(oof_path)
    if report.config_hash != batch.config_hash:
        raise RuntimeError("research campaign batch config hash mismatch")
    if report.code_commit != code_commit:
        raise RuntimeError("research campaign batch code commit mismatch")
    if report.config.horizons != (batch.horizon,):
        raise RuntimeError("research campaign batch horizon mismatch")
    if report.config.model_families != (batch.model_family,):
        raise RuntimeError("research campaign batch model family mismatch")
    if batch.seed is not None and report.config.seeds != (batch.seed,):
        raise RuntimeError("research campaign batch seed mismatch")
    if _stable_hash(report.feature_names) != batch.feature_names_hash:
        raise RuntimeError("research campaign batch feature names mismatch")
    return report.report_id, str(oof_path.resolve())


def discover_batch_artifact(
    batch: ResearchCampaignBatch,
    *,
    code_commit: str,
    report_root: Path,
) -> tuple[str, str] | None:
    """Recover a child-published result even if the parent died before updating its manifest."""

    for candidate in sorted(report_root.glob("*/*.oof.parquet")):
        report_id = candidate.name.removesuffix(".oof.parquet")
        metadata_path = candidate.with_name(f"{report_id}.json")
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            report_payload = payload["report"]
            if not isinstance(report_payload, dict):
                continue
            if report_payload.get("config_hash") != batch.config_hash:
                continue
            if report_payload.get("code_commit") != code_commit:
                continue
            return authenticate_batch_artifact(batch, code_commit=code_commit, oof_path=candidate)
        except (KeyError, RuntimeError, TypeError, ValueError, OSError):
            continue
    return None


def reconcile_campaign(
    manifest: ResearchCampaignManifest, *, batch_ids: frozenset[str] | None = None
) -> ResearchCampaignManifest:
    """Validate successes and recover or mark stale in-flight work after interruption."""

    report_root = Path(manifest.report_root)
    for batch in manifest.batches:
        if batch_ids is not None and batch.batch_id not in batch_ids:
            continue
        authenticated: tuple[str, str] | None = None
        if batch.oof_path is not None:
            try:
                authenticated = authenticate_batch_artifact(
                    batch,
                    code_commit=manifest.code_commit,
                    oof_path=Path(batch.oof_path),
                )
            except (RuntimeError, ValueError, OSError):
                authenticated = None
        if authenticated is None:
            authenticated = discover_batch_artifact(
                batch, code_commit=manifest.code_commit, report_root=report_root
            )
        if authenticated is not None:
            batch.report_id, batch.oof_path = authenticated
            batch.status = CampaignBatchStatus.SUCCEEDED
            batch.completed_at = batch.completed_at or datetime.now(UTC)
            batch.child_pid = None
            batch.last_error = None
        elif batch.status is CampaignBatchStatus.SUCCEEDED:
            raise RuntimeError(f"authenticated artifact is missing for {batch.batch_id}")
        elif batch.status is CampaignBatchStatus.RUNNING:
            if batch.child_pid is not None and _process_is_running(batch.child_pid):
                continue
            batch.status = CampaignBatchStatus.INTERRUPTED
            batch.child_pid = None
            batch.last_error = "parent or child stopped before authenticated publication"
    return manifest


def _process_is_running(pid: int) -> bool:
    """Check an exact child PID without terminating or modifying it."""

    try:
        os.kill(pid, signal.SIG_DFL if os.name == "nt" else 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()
