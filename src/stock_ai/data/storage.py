"""Immutable Parquet objects and a transactional DuckDB ingestion catalog."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from stock_ai.data.contracts import (
    BulkFileDescriptor,
    DatasetName,
    IngestionStatus,
    ObjectKind,
    QualityReport,
    StoredObject,
)


class StorageIntegrityError(RuntimeError):
    """Raised when a content-addressed object fails verification."""


class ImmutableParquetStore:
    """Publish complete object directories atomically; never replace valid objects."""

    def __init__(
        self,
        root: Path,
        *,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self.root = root.resolve()
        self._fault_hook = fault_hook or (lambda _stage: None)

    def write(
        self,
        *,
        kind: ObjectKind,
        dataset: DatasetName,
        source_date: date,
        frame: pd.DataFrame,
        payload_hash: str,
        schema_version: str,
        source_endpoint: str,
        received_at: datetime,
        available_at: datetime,
        as_of: datetime,
        ingestion_run_id: str,
        quality: QualityReport,
    ) -> StoredObject:
        object_id = _object_id(kind, dataset, source_date, payload_hash, schema_version)
        parent = (
            self.root
            / kind.value
            / "jquants_v2"
            / dataset.value
            / f"source_date={source_date.isoformat()}"
        )
        parent.mkdir(parents=True, exist_ok=True)
        final_directory = parent / object_id
        if final_directory.exists():
            return self.verify(final_directory)

        temporary_directory = Path(tempfile.mkdtemp(prefix=".tmp-", dir=parent)).resolve()
        if temporary_directory.parent != parent.resolve():
            raise StorageIntegrityError("temporary object directory escaped its partition")
        try:
            temporary_parquet = temporary_directory / "data.parquet"
            frame.to_parquet(temporary_parquet, index=False, compression="zstd")
            parquet_hash = _file_sha256(temporary_parquet)
            self._fault_hook("after_parquet")
            stored = StoredObject(
                object_id=object_id,
                kind=kind,
                dataset=dataset,
                source_date=source_date,
                payload_hash=payload_hash,
                parquet_hash=parquet_hash,
                rows=len(frame),
                schema_version=schema_version,
                source_endpoint=source_endpoint,
                received_at=received_at,
                available_at=available_at,
                as_of=as_of,
                ingestion_run_id=ingestion_run_id,
                parquet_path=final_directory / "data.parquet",
                manifest_path=final_directory / "manifest.json",
                quality_passed=quality.passed,
                quality_issues=quality.issues,
            )
            temporary_manifest = temporary_directory / "manifest.json"
            with temporary_manifest.open("wb") as stream:
                stream.write(stored.model_dump_json(indent=2).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            self._fault_hook("before_publish")
            try:
                os.replace(temporary_directory, final_directory)
            except FileExistsError:
                # Another writer won the content-addressed race. It must match.
                _remove_verified_temporary_directory(temporary_directory, parent)
                return self.verify(final_directory)
            return self.verify(final_directory)
        except Exception:
            if temporary_directory.exists():
                _remove_verified_temporary_directory(temporary_directory, parent)
            raise

    def verify(self, object_directory: Path) -> StoredObject:
        directory = object_directory.resolve()
        try:
            directory.relative_to(self.root)
        except ValueError:
            raise StorageIntegrityError("object is outside the immutable store") from None
        manifest_path = directory / "manifest.json"
        parquet_path = directory / "data.parquet"
        if not manifest_path.is_file() or not parquet_path.is_file():
            raise StorageIntegrityError("immutable object is incomplete")
        try:
            stored = StoredObject.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise StorageIntegrityError("immutable object manifest is invalid") from None
        if stored.object_id != directory.name:
            raise StorageIntegrityError("immutable object identity mismatch")
        expected_object_id = _object_id(
            stored.kind,
            stored.dataset,
            stored.source_date,
            stored.payload_hash,
            stored.schema_version,
        )
        if stored.object_id != expected_object_id:
            raise StorageIntegrityError("immutable object manifest identity mismatch")
        if stored.manifest_path.resolve() != manifest_path:
            raise StorageIntegrityError("immutable object manifest path mismatch")
        if stored.parquet_path.resolve() != parquet_path:
            raise StorageIntegrityError("immutable object Parquet path mismatch")
        if _file_sha256(parquet_path) != stored.parquet_hash:
            raise StorageIntegrityError("immutable object Parquet hash mismatch")
        try:
            parquet_rows = pq.ParquetFile(parquet_path).metadata.num_rows
        except (OSError, ValueError):
            raise StorageIntegrityError("immutable object Parquet metadata is invalid") from None
        if parquet_rows != stored.rows:
            raise StorageIntegrityError("immutable object row count mismatch")
        return stored


class DuckDBCatalog:
    """Transactional catalog of runs, immutable objects, and PIT analytical reads."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(self.path))
        self._initialize()

    def __enter__(self) -> DuckDBCatalog:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def begin_run(
        self,
        run_id: str,
        source_date: date,
        started_at: datetime,
        *,
        bulk_fingerprint: str | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO ingestion_runs
            (ingestion_run_id, source_date, started_at, status, bulk_fingerprint)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                run_id,
                source_date,
                started_at,
                IngestionStatus.RUNNING.value,
                bulk_fingerprint,
            ],
        )

    def record_object(self, run_id: str, stored: StoredObject) -> None:
        self._connection.execute("BEGIN TRANSACTION")
        try:
            self._connection.execute(
                """
                INSERT INTO stored_objects VALUES
                (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (object_id) DO NOTHING
                """,
                [
                    stored.object_id,
                    stored.kind.value,
                    stored.dataset.value,
                    stored.source_date,
                    stored.payload_hash,
                    stored.parquet_hash,
                    stored.rows,
                    stored.schema_version,
                    stored.source_endpoint,
                    stored.received_at,
                    stored.available_at,
                    stored.as_of,
                    str(stored.parquet_path),
                    str(stored.manifest_path),
                    stored.quality_passed,
                ],
            )
            self._connection.execute(
                """
                INSERT INTO ingestion_run_objects VALUES (?, ?)
                ON CONFLICT DO NOTHING
                """,
                [run_id, stored.object_id],
            )
            for issue in stored.quality_issues:
                self._connection.execute(
                    """
                    INSERT INTO quality_issues VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [
                        stored.object_id,
                        issue.code,
                        issue.severity.value,
                        issue.message,
                        issue.rows_affected,
                        run_id,
                    ],
                )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    def finish_run(
        self,
        run_id: str,
        status: IngestionStatus,
        completed_at: datetime,
        *,
        error_code: str | None = None,
    ) -> None:
        if status is IngestionStatus.RUNNING:
            raise ValueError("a completed run cannot remain RUNNING")
        self._connection.execute(
            """
            UPDATE ingestion_runs
            SET completed_at = ?, status = ?, error_code = ?
            WHERE ingestion_run_id = ?
            """,
            [completed_at, status.value, error_code, run_id],
        )

    def object_count(self, dataset: DatasetName, kind: ObjectKind) -> int:
        result = self._connection.execute(
            "SELECT count(*) FROM stored_objects WHERE dataset = ? AND kind = ?",
            [dataset.value, kind.value],
        ).fetchone()
        if result is None:
            return 0
        return int(result[0])

    def point_in_time(self, dataset: DatasetName, as_of: datetime) -> pd.DataFrame:
        """Read the latest record version known by ``as_of`` without rewriting history."""

        rows = self._connection.execute(
            """
            SELECT DISTINCT stored.parquet_path
            FROM stored_objects AS stored
            JOIN ingestion_run_objects AS link ON link.object_id = stored.object_id
            JOIN ingestion_runs AS run ON run.ingestion_run_id = link.ingestion_run_id
            LEFT JOIN bulk_file_checkpoints AS bulk
                ON bulk.fingerprint = run.bulk_fingerprint
            WHERE stored.dataset = ?
              AND stored.kind = ?
              AND stored.quality_passed
              AND stored.available_at <= ?
              AND run.status = ?
              AND (
                    run.bulk_fingerprint IS NULL
                    OR bulk.status = ?
                  )
            ORDER BY stored.parquet_path
            """,
            [
                dataset.value,
                ObjectKind.NORMALIZED.value,
                as_of,
                IngestionStatus.SUCCEEDED.value,
                IngestionStatus.SUCCEEDED.value,
            ],
        ).fetchall()
        paths = [str(row[0]) for row in rows]
        if not paths:
            return pd.DataFrame()
        keys = {
            DatasetName.SECURITY_MASTER: ("effective_date", "provider_code"),
            DatasetName.DAILY_PRICES: ("trading_date", "provider_code"),
            DatasetName.TRADING_CALENDAR: ("trading_date",),
            DatasetName.TOPIX: ("trading_date",),
            DatasetName.FINANCIAL_SUMMARY: ("disclosure_number",),
        }[dataset]
        partition = ", ".join(_quote_identifier(column) for column in keys)
        query = f"""
            SELECT * FROM read_parquet(?, union_by_name = true)
            WHERE available_at <= ?
            QUALIFY row_number() OVER (
                PARTITION BY {partition}
                ORDER BY available_at DESC, received_at DESC, payload_hash DESC
            ) = 1
            ORDER BY {partition}
        """
        return self._connection.execute(query, [paths, as_of]).fetchdf()

    def run_status(self, run_id: str) -> tuple[str, str | None] | None:
        row = self._connection.execute(
            "SELECT status, error_code FROM ingestion_runs WHERE ingestion_run_id = ?",
            [run_id],
        ).fetchone()
        if row is None:
            return None
        return str(row[0]), None if row[1] is None else str(row[1])

    def bulk_file_completed(self, descriptor: BulkFileDescriptor) -> bool:
        row = self._connection.execute(
            """
            SELECT checkpoint.status,
                   checkpoint.objects,
                   count(DISTINCT link.object_id) AS linked_objects
            FROM bulk_file_checkpoints AS checkpoint
            LEFT JOIN ingestion_runs AS run
                ON run.bulk_fingerprint = checkpoint.fingerprint
               AND run.status = ?
            LEFT JOIN ingestion_run_objects AS link
                ON link.ingestion_run_id = run.ingestion_run_id
            WHERE checkpoint.fingerprint = ?
            GROUP BY checkpoint.status, checkpoint.objects
            """,
            [IngestionStatus.SUCCEEDED.value, descriptor.fingerprint],
        ).fetchone()
        if row is None or str(row[0]) != IngestionStatus.SUCCEEDED.value:
            return False
        expected_objects = int(row[1])
        linked_objects = int(row[2])
        if expected_objects <= 0 or linked_objects != expected_objects:
            return False
        paths = self._connection.execute(
            """
            SELECT DISTINCT stored.parquet_path, stored.manifest_path
            FROM stored_objects AS stored
            JOIN ingestion_run_objects AS link ON link.object_id = stored.object_id
            JOIN ingestion_runs AS run ON run.ingestion_run_id = link.ingestion_run_id
            WHERE run.bulk_fingerprint = ? AND run.status = ?
            """,
            [descriptor.fingerprint, IngestionStatus.SUCCEEDED.value],
        ).fetchall()
        return len(paths) == expected_objects and all(
            Path(str(parquet)).is_file() and Path(str(manifest)).is_file()
            for parquet, manifest in paths
        )

    def record_bulk_file(
        self,
        descriptor: BulkFileDescriptor,
        *,
        status: IngestionStatus,
        updated_at: datetime,
        source_dates: int = 0,
        objects: int = 0,
        error_code: str | None = None,
    ) -> None:
        """Upsert a non-secret, resumable Bulk file checkpoint."""

        if status is IngestionStatus.RUNNING and error_code is not None:
            raise ValueError("a running bulk checkpoint cannot have an error code")
        self._connection.execute(
            """
            INSERT INTO bulk_file_checkpoints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (fingerprint) DO UPDATE SET
                status = excluded.status,
                updated_at = excluded.updated_at,
                source_dates = excluded.source_dates,
                objects = excluded.objects,
                error_code = excluded.error_code
            """,
            [
                descriptor.fingerprint,
                descriptor.dataset.value,
                descriptor.endpoint,
                descriptor.key_hash,
                descriptor.size,
                descriptor.last_modified,
                status.value,
                updated_at,
                source_dates,
                objects,
                error_code,
            ],
        )

    def bulk_checkpoint_counts(self) -> dict[str, int]:
        rows = self._connection.execute(
            "SELECT status, count(*) FROM bulk_file_checkpoints GROUP BY status"
        ).fetchall()
        return {str(status): int(count) for status, count in rows}

    def verify_integrity(
        self,
        store: ImmutableParquetStore,
        *,
        allow_empty: bool = False,
    ) -> int:
        """Reconcile catalog rows, immutable objects, and successful Bulk checkpoints."""

        rows = self._connection.execute(
            """
            SELECT object_id,
                   kind,
                   dataset,
                   source_date,
                   payload_hash,
                   parquet_hash,
                   rows,
                   schema_version,
                   source_endpoint,
                   received_at,
                   available_at,
                   as_of,
                   parquet_path,
                   manifest_path,
                   quality_passed
            FROM stored_objects
            ORDER BY object_id
            """
        ).fetchall()
        if not rows and not allow_empty:
            raise StorageIntegrityError("catalog contains no immutable objects")

        referenced_directories: set[Path] = set()
        for row in rows:
            (
                object_id,
                kind,
                dataset,
                source_date,
                payload_hash,
                parquet_hash,
                expected_rows,
                schema_version,
                source_endpoint,
                received_at,
                available_at,
                as_of,
                parquet_path,
                manifest_path,
                quality_passed,
            ) = row
            parquet = Path(str(parquet_path)).resolve()
            manifest = Path(str(manifest_path)).resolve()
            if parquet.parent != manifest.parent:
                raise StorageIntegrityError("catalog object paths do not share one directory")
            stored = store.verify(parquet.parent)
            catalog_paths_match = (
                parquet == stored.parquet_path.resolve()
                and manifest == stored.manifest_path.resolve()
            )
            if not catalog_paths_match:
                raise StorageIntegrityError("catalog paths do not match immutable object manifest")
            if (
                stored.object_id != str(object_id)
                or stored.kind.value != str(kind)
                or stored.dataset.value != str(dataset)
                or stored.source_date != source_date
                or stored.payload_hash != str(payload_hash)
                or stored.parquet_hash != str(parquet_hash)
                or stored.rows != int(expected_rows)
                or stored.schema_version != str(schema_version)
                or stored.source_endpoint != str(source_endpoint)
                or pd.Timestamp(stored.received_at) != pd.Timestamp(received_at)
                or pd.Timestamp(stored.available_at) != pd.Timestamp(available_at)
                or pd.Timestamp(stored.as_of) != pd.Timestamp(as_of)
                or stored.quality_passed is not bool(quality_passed)
            ):
                raise StorageIntegrityError("catalog metadata does not match immutable object")
            referenced_directories.add(parquet.parent)

        published_directories = {
            manifest.parent.resolve()
            for manifest in store.root.glob("*/jquants_v2/*/source_date=*/*/manifest.json")
        }
        if published_directories != referenced_directories:
            raise StorageIntegrityError("catalog and immutable object directories are inconsistent")

        checkpoints = self._connection.execute(
            """
            SELECT checkpoint.fingerprint,
                   checkpoint.objects,
                   checkpoint.source_dates,
                   count(DISTINCT link.object_id) AS linked_objects,
                   count(DISTINCT run.source_date) AS linked_source_dates
            FROM bulk_file_checkpoints AS checkpoint
            LEFT JOIN ingestion_runs AS run
                ON run.bulk_fingerprint = checkpoint.fingerprint
               AND run.status = ?
            LEFT JOIN ingestion_run_objects AS link
                ON link.ingestion_run_id = run.ingestion_run_id
            WHERE checkpoint.status = ?
            GROUP BY checkpoint.fingerprint, checkpoint.objects, checkpoint.source_dates
            """,
            [IngestionStatus.SUCCEEDED.value, IngestionStatus.SUCCEEDED.value],
        ).fetchall()
        for _fingerprint, objects, source_dates, linked_objects, linked_source_dates in checkpoints:
            if (
                int(objects) <= 0
                or int(source_dates) <= 0
                or int(objects) != int(linked_objects)
                or int(source_dates) != int(linked_source_dates)
            ):
                raise StorageIntegrityError("successful Bulk checkpoint is incomplete")
        return len(rows)

    def _initialize(self) -> None:
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS ingestion_runs (
                ingestion_run_id VARCHAR PRIMARY KEY,
                source_date DATE NOT NULL,
                started_at TIMESTAMPTZ NOT NULL,
                completed_at TIMESTAMPTZ,
                status VARCHAR NOT NULL,
                error_code VARCHAR,
                bulk_fingerprint VARCHAR
            );
            CREATE TABLE IF NOT EXISTS stored_objects (
                object_id VARCHAR PRIMARY KEY,
                kind VARCHAR NOT NULL,
                dataset VARCHAR NOT NULL,
                source_date DATE NOT NULL,
                payload_hash VARCHAR NOT NULL,
                parquet_hash VARCHAR NOT NULL,
                rows BIGINT NOT NULL,
                schema_version VARCHAR NOT NULL,
                source_endpoint VARCHAR NOT NULL,
                received_at TIMESTAMPTZ NOT NULL,
                available_at TIMESTAMPTZ NOT NULL,
                as_of TIMESTAMPTZ NOT NULL,
                parquet_path VARCHAR NOT NULL,
                manifest_path VARCHAR NOT NULL,
                quality_passed BOOLEAN NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ingestion_run_objects (
                ingestion_run_id VARCHAR NOT NULL,
                object_id VARCHAR NOT NULL,
                PRIMARY KEY (ingestion_run_id, object_id)
            );
            CREATE TABLE IF NOT EXISTS quality_issues (
                object_id VARCHAR NOT NULL,
                code VARCHAR NOT NULL,
                severity VARCHAR NOT NULL,
                message VARCHAR NOT NULL,
                rows_affected BIGINT NOT NULL,
                ingestion_run_id VARCHAR NOT NULL,
                PRIMARY KEY (object_id, code, ingestion_run_id)
            );
            CREATE TABLE IF NOT EXISTS bulk_file_checkpoints (
                fingerprint VARCHAR PRIMARY KEY,
                dataset VARCHAR NOT NULL,
                source_endpoint VARCHAR NOT NULL,
                key_hash VARCHAR NOT NULL,
                size BIGINT NOT NULL,
                last_modified TIMESTAMPTZ NOT NULL,
                status VARCHAR NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                source_dates BIGINT NOT NULL,
                objects BIGINT NOT NULL,
                error_code VARCHAR
            );
            """
        )
        self._connection.execute(
            "ALTER TABLE ingestion_runs ADD COLUMN IF NOT EXISTS bulk_fingerprint VARCHAR"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object_id(
    kind: ObjectKind,
    dataset: DatasetName,
    source_date: date,
    payload_hash: str,
    schema_version: str,
) -> str:
    return hashlib.sha256(
        "|".join(
            (
                kind.value,
                dataset.value,
                source_date.isoformat(),
                payload_hash,
                schema_version,
            )
        ).encode("utf-8")
    ).hexdigest()


def _remove_verified_temporary_directory(path: Path, expected_parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != expected_parent.resolve() or not resolved.name.startswith(".tmp-"):
        raise StorageIntegrityError("refused to remove an unverified temporary directory")
    shutil.rmtree(resolved)


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'
