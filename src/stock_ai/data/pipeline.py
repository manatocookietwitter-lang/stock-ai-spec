"""End-to-end J-Quants V2 ingestion orchestration."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from uuid import uuid4

from stock_ai.data.contracts import (
    BULK_REQUIRED_COLUMNS,
    ENDPOINT_SCHEMAS,
    DatasetName,
    FetchedPayload,
    IngestionResult,
    IngestionStatus,
    ObjectKind,
    StoredObject,
    capabilities_for,
)
from stock_ai.data.jquants_v2 import JQuantsV2Client
from stock_ai.data.normalize import canonical_payload_hash, normalize_payload, raw_frame
from stock_ai.data.quality import require_quality, validate_rows
from stock_ai.data.storage import DuckDBCatalog, ImmutableParquetStore

DEFAULT_DATASETS = (
    DatasetName.SECURITY_MASTER,
    DatasetName.DAILY_PRICES,
    DatasetName.FINANCIAL_SUMMARY,
)

ALL_DATASETS = (
    *DEFAULT_DATASETS,
    DatasetName.TRADING_CALENDAR,
    DatasetName.TOPIX,
)


class JQuantsV2Ingestor:
    """Fetch, validate, publish, and catalog complete immutable daily objects."""

    def __init__(
        self,
        *,
        client: JQuantsV2Client,
        store: ImmutableParquetStore,
        catalog: DuckDBCatalog,
        now: Callable[[], datetime] | None = None,
        run_id_factory: Callable[[date], str] | None = None,
    ) -> None:
        self.client = client
        self.store = store
        self.catalog = catalog
        self._now = now or (lambda: datetime.now(UTC))
        self._run_id_factory = run_id_factory or (
            lambda day: f"jqv2-{day.isoformat()}-{uuid4().hex}"
        )

    def sync_date(
        self,
        source_date: date,
        *,
        datasets: Iterable[DatasetName] = DEFAULT_DATASETS,
    ) -> IngestionResult:
        selected = self._selected_datasets(datasets)
        return self._sync_payload_provider(
            source_date,
            selected,
            self.client.fetch_date,
        )

    def ingest_payloads(
        self,
        source_date: date,
        payloads: Iterable[FetchedPayload],
        *,
        bulk_fingerprint: str | None = None,
    ) -> IngestionResult:
        """Ingest already-downloaded Bulk payload slices through the same safe path."""

        payload_by_dataset: dict[DatasetName, FetchedPayload] = {}
        for payload in payloads:
            if payload.dataset in payload_by_dataset:
                raise ValueError(f"duplicate payload for {payload.dataset.value}")
            expected_endpoint = ENDPOINT_SCHEMAS[payload.dataset].endpoint
            if payload.endpoint != expected_endpoint:
                raise ValueError(f"unexpected source endpoint for {payload.dataset.value}")
            payload_by_dataset[payload.dataset] = payload
        selected = self._selected_datasets(payload_by_dataset)
        return self._sync_payload_provider(
            source_date,
            selected,
            lambda dataset, _day: payload_by_dataset[dataset],
            bulk_fingerprint=bulk_fingerprint,
        )

    def _sync_payload_provider(
        self,
        source_date: date,
        selected: tuple[DatasetName, ...],
        payload_provider: Callable[[DatasetName, date], FetchedPayload],
        *,
        bulk_fingerprint: str | None = None,
    ) -> IngestionResult:
        run_id = self._run_id_factory(source_date)
        started_at = self._aware_now()
        self.catalog.begin_run(
            run_id,
            source_date,
            started_at,
            bulk_fingerprint=bulk_fingerprint,
        )
        stored_objects: list[StoredObject] = []
        try:
            for dataset in selected:
                payload = payload_provider(dataset, source_date)
                observed_as_of = max(self._aware_now(), payload.received_at)
                payload_hash = canonical_payload_hash(payload.rows)
                quality = validate_rows(
                    dataset,
                    payload.rows,
                    requested_source_date=source_date,
                    required_columns=(
                        BULK_REQUIRED_COLUMNS[dataset]
                        if bulk_fingerprint is not None
                        else None
                    ),
                )
                source_schema = ENDPOINT_SCHEMAS[dataset]
                source_schema_version = source_schema.schema_version
                if bulk_fingerprint is not None:
                    source_schema_version += "+bulk-file-v1"
                raw = raw_frame(
                    payload,
                    ingestion_run_id=run_id,
                    as_of=observed_as_of,
                    payload_hash=payload_hash,
                )
                raw_object = self.store.write(
                    kind=ObjectKind.RAW,
                    dataset=dataset,
                    source_date=source_date,
                    frame=raw,
                    payload_hash=payload_hash,
                    schema_version=source_schema_version,
                    source_endpoint=payload.endpoint,
                    received_at=payload.received_at,
                    available_at=payload.received_at,
                    as_of=observed_as_of,
                    ingestion_run_id=run_id,
                    quality=quality,
                )
                self.catalog.record_object(run_id, raw_object)
                stored_objects.append(raw_object)

                require_quality(quality)
                normalized_schema_version = f"goal2a-normalized-v2+{source_schema_version}"
                normalized = normalize_payload(
                    payload,
                    ingestion_run_id=run_id,
                    as_of=observed_as_of,
                    payload_hash=payload_hash,
                    raw_object_id=raw_object.object_id,
                )
                normalized_object = self.store.write(
                    kind=ObjectKind.NORMALIZED,
                    dataset=dataset,
                    source_date=source_date,
                    frame=normalized,
                    payload_hash=payload_hash,
                    schema_version=normalized_schema_version,
                    source_endpoint=payload.endpoint,
                    received_at=payload.received_at,
                    available_at=payload.received_at,
                    as_of=observed_as_of,
                    ingestion_run_id=run_id,
                    quality=quality,
                )
                self.catalog.record_object(run_id, normalized_object)
                stored_objects.append(normalized_object)

            completed_at = self._aware_now()
            self.catalog.finish_run(run_id, IngestionStatus.SUCCEEDED, completed_at)
            return IngestionResult(
                ingestion_run_id=run_id,
                source_date=source_date,
                started_at=started_at,
                completed_at=completed_at,
                status=IngestionStatus.SUCCEEDED,
                objects=tuple(stored_objects),
                capabilities=capabilities_for(self.client.config.plan),
            )
        except Exception as exc:
            self.catalog.finish_run(
                run_id,
                IngestionStatus.FAILED,
                self._aware_now(),
                error_code=type(exc).__name__,
            )
            raise

    @staticmethod
    def _selected_datasets(datasets: Iterable[DatasetName]) -> tuple[DatasetName, ...]:
        selected = tuple(dict.fromkeys(datasets))
        if not selected:
            raise ValueError("at least one dataset must be selected")
        return selected

    def sync_range(
        self,
        start: date,
        end: date,
        *,
        datasets: Iterable[DatasetName] = DEFAULT_DATASETS,
    ) -> tuple[IngestionResult, ...]:
        if end < start:
            raise ValueError("end date cannot precede start date")
        results: list[IngestionResult] = []
        current = start
        while current <= end:
            results.append(self.sync_date(current, datasets=datasets))
            current += timedelta(days=1)
        return tuple(results)

    def _aware_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("ingestion clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
