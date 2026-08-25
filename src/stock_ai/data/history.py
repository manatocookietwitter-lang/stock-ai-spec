"""Resumable J-Quants V2 Bulk history acquisition."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime

import pandas as pd

from stock_ai.data.contracts import (
    ENDPOINT_SCHEMAS,
    DatasetName,
    FetchedPayload,
    HistorySyncResult,
    IngestionStatus,
)
from stock_ai.data.jquants_v2 import JQuantsSchemaError, JQuantsV2Client
from stock_ai.data.pipeline import DEFAULT_DATASETS, JQuantsV2Ingestor
from stock_ai.data.storage import DuckDBCatalog


class JQuantsV2HistoryIngestor:
    """Download Bulk CSVs once and checkpoint only after every date slice is durable."""

    def __init__(
        self,
        *,
        client: JQuantsV2Client,
        daily_ingestor: JQuantsV2Ingestor,
        catalog: DuckDBCatalog,
    ) -> None:
        self.client = client
        self.daily_ingestor = daily_ingestor
        self.catalog = catalog

    def sync_history(
        self,
        start: date,
        end: date,
        *,
        datasets: Iterable[DatasetName] = DEFAULT_DATASETS,
        resume: bool = True,
    ) -> HistorySyncResult:
        if end < start:
            raise ValueError("history end date cannot precede start date")
        selected = tuple(dict.fromkeys(datasets))
        if not selected:
            raise ValueError("at least one history dataset must be selected")

        listed = 0
        downloaded = 0
        skipped = 0
        source_dates = 0
        object_count = 0
        for dataset in selected:
            descriptors = self.client.list_bulk_files(dataset, start=start, end=end)
            if not descriptors:
                raise JQuantsSchemaError(
                    f"J-Quants V2 Bulk List returned no files for {dataset.value}"
                )
            listed += len(descriptors)
            for descriptor in descriptors:
                if resume and self.catalog.bulk_file_completed(descriptor):
                    skipped += 1
                    continue
                updated_at = datetime.now(UTC)
                self.catalog.record_bulk_file(
                    descriptor,
                    status=IngestionStatus.RUNNING,
                    updated_at=updated_at,
                )
                try:
                    payload = self.client.fetch_bulk_file(descriptor)
                    downloaded += 1
                    slices = _split_payload_by_source_date(payload, start=start, end=end)
                    if not slices:
                        raise JQuantsSchemaError(
                            "J-Quants V2 Bulk CSV contains no rows for the requested range"
                        )
                    file_objects = 0
                    for source_date, sliced_payload in slices:
                        result = self.daily_ingestor.ingest_payloads(
                            source_date,
                            (sliced_payload,),
                            bulk_fingerprint=descriptor.fingerprint,
                        )
                        file_objects += len(result.objects)
                    source_dates += len(slices)
                    object_count += file_objects
                    self.catalog.record_bulk_file(
                        descriptor,
                        status=IngestionStatus.SUCCEEDED,
                        updated_at=datetime.now(UTC),
                        source_dates=len(slices),
                        objects=file_objects,
                    )
                except Exception as exc:
                    self.catalog.record_bulk_file(
                        descriptor,
                        status=IngestionStatus.FAILED,
                        updated_at=datetime.now(UTC),
                        error_code=type(exc).__name__,
                    )
                    raise
        return HistorySyncResult(
            start=start,
            end=end,
            datasets=selected,
            listed_files=listed,
            downloaded_files=downloaded,
            skipped_files=skipped,
            ingested_source_dates=source_dates,
            objects=object_count,
        )


def _split_payload_by_source_date(
    payload: FetchedPayload,
    *,
    start: date,
    end: date,
) -> tuple[tuple[date, FetchedPayload], ...]:
    if not payload.rows:
        return ()
    schema = ENDPOINT_SCHEMAS[payload.dataset]
    frame = pd.DataFrame.from_records(payload.rows)
    if schema.source_date_column not in frame:
        raise JQuantsSchemaError(
            f"J-Quants V2 Bulk CSV is missing {schema.source_date_column}"
        )
    parsed = pd.to_datetime(frame[schema.source_date_column], errors="coerce")
    if parsed.isna().any():
        raise JQuantsSchemaError("J-Quants V2 Bulk CSV contains an invalid source date")
    frame["__source_date"] = parsed.dt.date
    outside = (frame["__source_date"] < start) | (frame["__source_date"] > end)
    if outside.any() and payload.dataset is not DatasetName.TRADING_CALENDAR:
        raise JQuantsSchemaError("J-Quants V2 Bulk CSV contains rows outside the requested range")
    if payload.dataset is DatasetName.TRADING_CALENDAR:
        frame = frame.loc[~outside].copy()

    slices: list[tuple[date, FetchedPayload]] = []
    for source_date, group in frame.groupby("__source_date", sort=True):
        rows = tuple(
            {str(key): value for key, value in record.items()}
            for record in group.drop(columns="__source_date").to_dict(orient="records")
        )
        day = source_date
        if not isinstance(day, date):
            raise AssertionError("parsed bulk source date is not a date")
        slices.append(
            (
                day,
                FetchedPayload(
                    dataset=payload.dataset,
                    endpoint=payload.endpoint,
                    query=payload.query,
                    requested_at=payload.requested_at,
                    received_at=payload.received_at,
                    pages=payload.pages,
                    rows=rows,
                ),
            )
        )
    return tuple(slices)
