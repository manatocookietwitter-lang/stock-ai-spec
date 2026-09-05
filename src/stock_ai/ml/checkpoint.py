"""Authenticated fold and Optuna checkpoints for interruption-safe research."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, cast

import pandas as pd

_CHECKPOINT_SCHEMA = "advanced-research-checkpoint-v1"
_PROGRESS_SCHEMA = "advanced-research-progress-v1"
_FOLD_SCHEMA = "advanced-research-fold-v1"


def stable_hash(value: object) -> str:
    """Hash a JSON-serializable identity without accepting implicit coercions."""

    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def checkpoint_id_for_provenance(provenance: Mapping[str, object]) -> str:
    """Derive the exact content-addressed checkpoint namespace without creating it."""

    return stable_hash(
        {"schema_version": _CHECKPOINT_SCHEMA, "provenance": dict(provenance)}
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_hash(payload: Mapping[str, object]) -> str:
    return stable_hash({key: value for key, value in payload.items() if key != "metadata_hash"})


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _load_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError(f"{description} is missing or invalid") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return value


def _authenticate_metadata(payload: Mapping[str, object], *, description: str) -> None:
    observed = payload.get("metadata_hash")
    if not isinstance(observed, str) or observed != _metadata_hash(payload):
        raise RuntimeError(f"{description} metadata hash mismatch")


class _ExclusiveLock:
    """Small cross-platform OS lock released automatically after worker death."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: IO[bytes] = path.open("a+b")
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"\0")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        self._stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = cast(Any, __import__("fcntl"))
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._stream.close()
            raise RuntimeError("another worker owns the research checkpoint") from None
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = cast(Any, __import__("fcntl"))
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._closed = True


class ResearchCheckpointStore:
    """Persist and authenticate successful OOF folds under one research identity."""

    def __init__(self, root: Path, *, provenance: Mapping[str, object]) -> None:
        self.provenance = dict(provenance)
        self.checkpoint_id = checkpoint_id_for_provenance(self.provenance)
        self.path = root.resolve() / self.checkpoint_id
        self.path.mkdir(parents=True, exist_ok=True)
        self._lock = _ExclusiveLock(self.path / "writer.lock")
        try:
            self._authenticate_or_create_namespace()
            self._progress = self._load_or_create_progress()
            self._mark_abandoned_units_interrupted()
        except Exception:
            self._lock.close()
            raise

    @property
    def optuna_database_path(self) -> Path:
        return self.path / "optuna.sqlite3"

    def close(self) -> None:
        self._lock.close()

    def __enter__(self) -> ResearchCheckpointStore:
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def unit_id(self, evidence: Mapping[str, object]) -> str:
        return stable_hash(
            {
                "schema_version": _FOLD_SCHEMA,
                "checkpoint_id": self.checkpoint_id,
                "evidence": dict(evidence),
            }
        )

    def load_fold(
        self,
        evidence: Mapping[str, object],
        *,
        frame_hash: Callable[[pd.DataFrame], str],
    ) -> pd.DataFrame | None:
        """Return an authenticated completed fold, or None when it was never published."""

        unit_id = self.unit_id(evidence)
        artifact = self.path / "folds" / unit_id
        progress = self._unit_progress(unit_id)
        if not artifact.exists():
            if progress is not None and progress.get("status") == "SUCCEEDED":
                raise RuntimeError("successful fold checkpoint artifact is missing")
            return None
        frame = self._authenticate_fold(
            artifact,
            unit_id=unit_id,
            evidence=evidence,
            frame_hash=frame_hash,
        )
        if progress is None or progress.get("status") != "SUCCEEDED":
            self._set_unit_progress(
                unit_id,
                evidence=evidence,
                status="SUCCEEDED",
                artifact=str(artifact),
                error_code=None,
                preserve_attempts=True,
            )
        return frame

    def begin_fold(self, evidence: Mapping[str, object]) -> str:
        unit_id = self.unit_id(evidence)
        progress = self._unit_progress(unit_id)
        if progress is not None and progress.get("status") == "SUCCEEDED":
            raise RuntimeError("successful fold must be authenticated before reuse")
        self._set_unit_progress(
            unit_id,
            evidence=evidence,
            status="RUNNING",
            artifact=None,
            error_code=None,
            preserve_attempts=False,
        )
        return unit_id

    def fail_fold(self, evidence: Mapping[str, object], *, error_code: str) -> None:
        self._set_unit_progress(
            self.unit_id(evidence),
            evidence=evidence,
            status="FAILED",
            artifact=None,
            error_code=error_code,
            preserve_attempts=True,
        )

    def publish_fold(
        self,
        evidence: Mapping[str, object],
        frame: pd.DataFrame,
        *,
        frame_hash: Callable[[pd.DataFrame], str],
    ) -> Path:
        unit_id = self.unit_id(evidence)
        folds_root = self.path / "folds"
        folds_root.mkdir(parents=True, exist_ok=True)
        destination = folds_root / unit_id
        if destination.exists():
            self._authenticate_fold(
                destination,
                unit_id=unit_id,
                evidence=evidence,
                frame_hash=frame_hash,
            )
            self._set_unit_progress(
                unit_id,
                evidence=evidence,
                status="SUCCEEDED",
                artifact=str(destination),
                error_code=None,
                preserve_attempts=True,
            )
            return destination

        temporary = Path(tempfile.mkdtemp(prefix=f".{unit_id}.", dir=folds_root))
        try:
            parquet = temporary / "oof.parquet"
            frame.to_parquet(parquet, index=False, compression="zstd")
            metadata: dict[str, object] = {
                "schema_version": _FOLD_SCHEMA,
                "checkpoint_id": self.checkpoint_id,
                "unit_id": unit_id,
                "status": "SUCCEEDED",
                "evidence": dict(evidence),
                "rows": len(frame),
                "columns": list(frame.columns),
                "frame_sha256": frame_hash(frame),
                "parquet_sha256": _file_sha256(parquet),
            }
            metadata["metadata_hash"] = _metadata_hash(metadata)
            _atomic_write_json(temporary / "manifest.json", metadata)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        self._set_unit_progress(
            unit_id,
            evidence=evidence,
            status="SUCCEEDED",
            artifact=str(destination),
            error_code=None,
            preserve_attempts=True,
        )
        return destination

    def _authenticate_or_create_namespace(self) -> None:
        manifest_path = self.path / "manifest.json"
        if manifest_path.exists():
            observed = _load_json(manifest_path, description="research checkpoint manifest")
            _authenticate_metadata(observed, description="research checkpoint manifest")
            if observed.get("schema_version") != _CHECKPOINT_SCHEMA:
                raise RuntimeError("research checkpoint schema mismatch")
            if observed.get("checkpoint_id") != self.checkpoint_id:
                raise RuntimeError("research checkpoint identity mismatch")
            if observed.get("provenance") != self.provenance:
                raise RuntimeError("research checkpoint provenance mismatch")
            return
        payload: dict[str, object] = {
            "schema_version": _CHECKPOINT_SCHEMA,
            "checkpoint_id": self.checkpoint_id,
            "provenance": self.provenance,
            "created_at": datetime.now(UTC).isoformat(),
        }
        payload["metadata_hash"] = _metadata_hash(payload)
        _atomic_write_json(manifest_path, payload)

    def _load_or_create_progress(self) -> dict[str, Any]:
        path = self.path / "progress.json"
        if not path.exists():
            created_payload: dict[str, Any] = {
                "schema_version": _PROGRESS_SCHEMA,
                "checkpoint_id": self.checkpoint_id,
                "updated_at": datetime.now(UTC).isoformat(),
                "units": {},
            }
            created_payload["metadata_hash"] = _metadata_hash(created_payload)
            _atomic_write_json(path, created_payload)
            return created_payload
        payload = _load_json(path, description="research checkpoint progress")
        _authenticate_metadata(payload, description="research checkpoint progress")
        if payload.get("schema_version") != _PROGRESS_SCHEMA:
            raise RuntimeError("research checkpoint progress schema mismatch")
        if payload.get("checkpoint_id") != self.checkpoint_id:
            raise RuntimeError("research checkpoint progress identity mismatch")
        if not isinstance(payload.get("units"), dict):
            raise RuntimeError("research checkpoint unit map is invalid")
        return payload

    def _write_progress(self) -> None:
        self._progress["updated_at"] = datetime.now(UTC).isoformat()
        self._progress["metadata_hash"] = _metadata_hash(self._progress)
        _atomic_write_json(self.path / "progress.json", self._progress)

    def _mark_abandoned_units_interrupted(self) -> None:
        changed = False
        units = self._progress["units"]
        assert isinstance(units, dict)
        for value in units.values():
            if isinstance(value, dict) and value.get("status") == "RUNNING":
                value["status"] = "INTERRUPTED"
                value["error_code"] = "WORKER_TERMINATED_BEFORE_AUTHENTICATED_PUBLICATION"
                value["updated_at"] = datetime.now(UTC).isoformat()
                changed = True
        if changed:
            self._write_progress()

    def _unit_progress(self, unit_id: str) -> dict[str, Any] | None:
        units = self._progress["units"]
        assert isinstance(units, dict)
        value = units.get(unit_id)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("research checkpoint unit state is invalid")
        return value

    def _set_unit_progress(
        self,
        unit_id: str,
        *,
        evidence: Mapping[str, object],
        status: str,
        artifact: str | None,
        error_code: str | None,
        preserve_attempts: bool,
    ) -> None:
        units = self._progress["units"]
        assert isinstance(units, dict)
        prior = self._unit_progress(unit_id)
        attempts = int(prior.get("attempts", 0)) if prior is not None else 0
        if not preserve_attempts:
            attempts += 1
        units[unit_id] = {
            "unit_id": unit_id,
            "evidence": dict(evidence),
            "status": status,
            "attempts": attempts,
            "artifact": artifact,
            "error_code": error_code,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._write_progress()

    def _authenticate_fold(
        self,
        artifact: Path,
        *,
        unit_id: str,
        evidence: Mapping[str, object],
        frame_hash: Callable[[pd.DataFrame], str],
    ) -> pd.DataFrame:
        if artifact.name != unit_id or artifact.parent.name != "folds":
            raise RuntimeError("fold checkpoint path identity mismatch")
        metadata = _load_json(artifact / "manifest.json", description="fold checkpoint")
        _authenticate_metadata(metadata, description="fold checkpoint")
        if metadata.get("schema_version") != _FOLD_SCHEMA:
            raise RuntimeError("fold checkpoint schema mismatch")
        if metadata.get("checkpoint_id") != self.checkpoint_id:
            raise RuntimeError("fold checkpoint research identity mismatch")
        if metadata.get("unit_id") != unit_id or metadata.get("evidence") != dict(evidence):
            raise RuntimeError("fold checkpoint unit identity mismatch")
        parquet = artifact / "oof.parquet"
        if not parquet.is_file() or metadata.get("parquet_sha256") != _file_sha256(parquet):
            raise RuntimeError("fold checkpoint Parquet hash mismatch")
        frame = pd.read_parquet(parquet)
        if metadata.get("rows") != len(frame) or metadata.get("columns") != list(frame.columns):
            raise RuntimeError("fold checkpoint shape mismatch")
        if metadata.get("frame_sha256") != frame_hash(frame):
            raise RuntimeError("fold checkpoint content hash mismatch")
        return frame


def read_checkpoint_status(checkpoint_path: Path) -> dict[str, object]:
    """Read authenticated granular progress without locking or mutating it."""

    checkpoint_path = checkpoint_path.resolve()
    namespace = _load_json(
        checkpoint_path / "manifest.json", description="research checkpoint manifest"
    )
    _authenticate_metadata(namespace, description="research checkpoint manifest")
    progress = _load_json(
        checkpoint_path / "progress.json", description="research checkpoint progress"
    )
    _authenticate_metadata(progress, description="research checkpoint progress")
    if namespace.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise RuntimeError("research checkpoint schema mismatch")
    if progress.get("schema_version") != _PROGRESS_SCHEMA:
        raise RuntimeError("research checkpoint progress schema mismatch")
    if namespace.get("checkpoint_id") != checkpoint_path.name:
        raise RuntimeError("research checkpoint directory identity mismatch")
    if progress.get("checkpoint_id") != namespace.get("checkpoint_id"):
        raise RuntimeError("research checkpoint status identity mismatch")
    units = progress.get("units")
    if not isinstance(units, dict):
        raise RuntimeError("research checkpoint unit map is invalid")
    if not isinstance(namespace.get("provenance"), dict):
        raise RuntimeError("research checkpoint provenance is invalid")
    counts: dict[str, int] = {}
    for value in units.values():
        if not isinstance(value, dict) or not isinstance(value.get("status"), str):
            raise RuntimeError("research checkpoint unit state is invalid")
        status = str(value["status"])
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": "advanced-research-status-v1",
        "verification": "AUTHENTICATED_PROGRESS_METADATA_ONLY",
        "checkpoint_id": namespace["checkpoint_id"],
        "provenance": namespace["provenance"],
        "updated_at": progress["updated_at"],
        "unit_counts": counts,
        "units": units,
    }
