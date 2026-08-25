from __future__ import annotations

import csv
import gzip
import io
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from stock_ai.data import (
    DatasetName,
    DuckDBCatalog,
    ImmutableParquetStore,
    JQuantsRequestError,
    JQuantsSchemaError,
    JQuantsV2Client,
    JQuantsV2Config,
    JQuantsV2HistoryIngestor,
    JQuantsV2Ingestor,
    ObjectKind,
    SubscriptionPlan,
)
from stock_ai.data.contracts import FetchedPayload
from stock_ai.data.history import _split_payload_by_source_date


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 24, tzinfo=UTC)
        self.tick = 0.0

    def now(self) -> datetime:
        self.value += timedelta(seconds=1)
        return self.value

    def monotonic(self) -> float:
        self.tick += 100.0
        return self.tick


def test_bulk_history_is_credential_safe_and_resumes_from_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = "ephemeral-history-test-key"
    monkeypatch.setenv("JQUANTS_API_KEY", credential)
    archive = _daily_archive()
    api_calls: list[str] = []
    download_headers: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "download.example":
            download_headers.append(request.headers.get("x-api-key"))
            return httpx.Response(200, content=archive)
        api_calls.append(request.url.path)
        assert request.headers.get("x-api-key") == credential
        if request.url.path == "/v2/bulk/list":
            assert request.url.params["endpoint"] == "/equities/bars/daily"
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "Key": "safe-provider-key",
                            "Size": len(archive),
                            "LastModified": "2026-08-24T00:00:00Z",
                        }
                    ]
                },
            )
        if request.url.path == "/v2/bulk/get":
            return httpx.Response(200, json={"url": "https://download.example/bars.csv.gz"})
        raise AssertionError(f"unexpected request: {request.url.path}")

    clock = Clock()
    data_root = tmp_path / "data"
    with (
        JQuantsV2Client.from_env(
            config=JQuantsV2Config(plan=SubscriptionPlan.LIGHT),
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(data_root / "catalog.duckdb") as catalog,
    ):
        daily = JQuantsV2Ingestor(
            client=client,
            store=ImmutableParquetStore(data_root),
            catalog=catalog,
            now=clock.now,
        )
        history = JQuantsV2HistoryIngestor(
            client=client,
            daily_ingestor=daily,
            catalog=catalog,
        )
        first = history.sync_history(
            date(2026, 8, 20),
            date(2026, 8, 21),
            datasets=(DatasetName.DAILY_PRICES,),
        )
        second = history.sync_history(
            date(2026, 8, 20),
            date(2026, 8, 21),
            datasets=(DatasetName.DAILY_PRICES,),
        )

        assert first.downloaded_files == 1
        assert first.ingested_source_dates == 2
        assert first.objects == 4
        assert second.downloaded_files == 0
        assert second.skipped_files == 1
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.RAW) == 2
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.NORMALIZED) == 2
        assert catalog.bulk_checkpoint_counts() == {"SUCCEEDED": 1}

    assert api_calls == ["/v2/bulk/list", "/v2/bulk/get", "/v2/bulk/list"]
    assert download_headers == [None]
    for path in data_root.rglob("*"):
        if path.is_file():
            assert credential.encode() not in path.read_bytes()


def test_official_bulk_daily_schema_without_adjusted_fields_is_ingested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", "temporary-key")
    archive = _daily_archive(include_adjusted=False)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "download.example":
            return httpx.Response(200, content=archive)
        if request.url.path == "/v2/bulk/list":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "Key": "official-file-schema",
                            "Size": len(archive),
                            "LastModified": "2026-08-24T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"url": "https://download.example/bars.csv.gz"})

    clock = Clock()
    with (
        JQuantsV2Client.from_env(
            config=JQuantsV2Config(plan=SubscriptionPlan.LIGHT),
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(tmp_path / "catalog.duckdb") as catalog,
    ):
        history = JQuantsV2HistoryIngestor(
            client=client,
            daily_ingestor=JQuantsV2Ingestor(
                client=client,
                store=ImmutableParquetStore(tmp_path),
                catalog=catalog,
                now=clock.now,
            ),
            catalog=catalog,
        )
        result = history.sync_history(
            date(2026, 8, 20),
            date(2026, 8, 21),
            datasets=(DatasetName.DAILY_PRICES,),
        )
        observed = catalog.point_in_time(DatasetName.DAILY_PRICES, clock.now())

    assert result.downloaded_files == 1
    assert observed["research_close"].isna().all()
    assert set(observed["adjustment_source"]) == {"bulk_adjfactor_only"}


def test_complete_calendar_bulk_file_is_scoped_to_requested_dates() -> None:
    received = datetime(2026, 8, 24, tzinfo=UTC)
    payload = FetchedPayload(
        dataset=DatasetName.TRADING_CALENDAR,
        endpoint="/markets/calendar",
        query=(("bulk_fingerprint", "safe"),),
        requested_at=received,
        received_at=received,
        pages=1,
        rows=(
            {"Date": "2026-08-19", "HolDiv": "1"},
            {"Date": "2026-08-20", "HolDiv": "1"},
            {"Date": "2026-08-21", "HolDiv": "1"},
        ),
    )
    slices = _split_payload_by_source_date(
        payload,
        start=date(2026, 8, 20),
        end=date(2026, 8, 20),
    )

    assert [day for day, _ in slices] == [date(2026, 8, 20)]


def test_bulk_history_fails_closed_on_rows_outside_requested_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", "temporary-key")
    archive = _daily_archive()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "download.example":
            return httpx.Response(200, content=archive)
        if request.url.path == "/v2/bulk/list":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "Key": "range-key",
                            "Size": len(archive),
                            "LastModified": "2026-08-24T00:00:00+00:00",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"url": "https://download.example/bars.csv.gz"})

    clock = Clock()
    with (
        JQuantsV2Client.from_env(
            config=JQuantsV2Config(plan=SubscriptionPlan.LIGHT, max_retries=0),
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(tmp_path / "catalog.duckdb") as catalog,
    ):
        daily = JQuantsV2Ingestor(
            client=client,
            store=ImmutableParquetStore(tmp_path),
            catalog=catalog,
            now=clock.now,
        )
        history = JQuantsV2HistoryIngestor(
            client=client,
            daily_ingestor=daily,
            catalog=catalog,
        )
        with pytest.raises(Exception, match="outside the requested range"):
            history.sync_history(
                date(2026, 8, 20),
                date(2026, 8, 20),
                datasets=(DatasetName.DAILY_PRICES,),
            )
        assert catalog.bulk_checkpoint_counts() == {"FAILED": 1}
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.RAW) == 0


def test_partial_bulk_file_is_invisible_until_resume_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", "temporary-key")
    archive = _daily_archive()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "download.example":
            return httpx.Response(200, content=archive)
        if request.url.path == "/v2/bulk/list":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "Key": "resume-key",
                            "Size": len(archive),
                            "LastModified": "2026-08-24T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"url": "https://download.example/bars.csv.gz"})

    class FlakyIngestor:
        def __init__(self, delegate: JQuantsV2Ingestor) -> None:
            self.delegate = delegate
            self.calls = 0
            self.fail = True

        def ingest_payloads(self, *args: object, **kwargs: object) -> object:
            self.calls += 1
            if self.fail and self.calls == 2:
                raise RuntimeError("simulated second-slice interruption")
            return self.delegate.ingest_payloads(*args, **kwargs)  # type: ignore[arg-type]

    clock = Clock()
    with (
        JQuantsV2Client.from_env(
            config=JQuantsV2Config(plan=SubscriptionPlan.LIGHT),
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(tmp_path / "catalog.duckdb") as catalog,
    ):
        delegate = JQuantsV2Ingestor(
            client=client,
            store=ImmutableParquetStore(tmp_path / "lake"),
            catalog=catalog,
            now=clock.now,
        )
        flaky = FlakyIngestor(delegate)
        history = JQuantsV2HistoryIngestor(
            client=client,
            daily_ingestor=flaky,  # type: ignore[arg-type]
            catalog=catalog,
        )
        with pytest.raises(RuntimeError, match="second-slice"):
            history.sync_history(
                date(2026, 8, 20),
                date(2026, 8, 21),
                datasets=(DatasetName.DAILY_PRICES,),
            )
        assert catalog.bulk_checkpoint_counts() == {"FAILED": 1}
        assert catalog.point_in_time(DatasetName.DAILY_PRICES, clock.now()).empty

        flaky.fail = False
        flaky.calls = 0
        result = history.sync_history(
            date(2026, 8, 20),
            date(2026, 8, 21),
            datasets=(DatasetName.DAILY_PRICES,),
        )
        assert result.downloaded_files == 1
        assert catalog.bulk_checkpoint_counts() == {"SUCCEEDED": 1}
        assert len(catalog.point_in_time(DatasetName.DAILY_PRICES, clock.now())) == 2


def test_bulk_csv_requires_rows_and_respects_uncompressed_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", "temporary-key")
    archive = _daily_archive()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "download.example":
            return httpx.Response(200, content=archive)
        if request.url.path == "/v2/bulk/list":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "Key": "bounded-key",
                            "Size": len(archive),
                            "LastModified": "2026-08-24T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"url": "https://download.example/bars.csv.gz"})

    clock = Clock()
    with JQuantsV2Client.from_env(
        config=JQuantsV2Config(
            plan=SubscriptionPlan.LIGHT,
            maximum_bulk_uncompressed_bytes=10,
        ),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        descriptors = client.list_bulk_files(
            DatasetName.DAILY_PRICES,
            start=date(2026, 8, 20),
            end=date(2026, 8, 21),
        )
        with pytest.raises(JQuantsRequestError, match="uncompressed size limit"):
            client.fetch_bulk_file(descriptors[0])

    empty_archive = _daily_archive(include_rows=False)

    def empty_handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "download.example":
            return httpx.Response(200, content=empty_archive)
        if request.url.path == "/v2/bulk/list":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "Key": "empty-key",
                            "Size": len(empty_archive),
                            "LastModified": "2026-08-24T00:00:00Z",
                        }
                    ]
                },
            )
        return httpx.Response(200, json={"url": "https://download.example/empty.csv.gz"})

    with JQuantsV2Client.from_env(
        config=JQuantsV2Config(plan=SubscriptionPlan.LIGHT),
        transport=httpx.MockTransport(empty_handler),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        descriptor = client.list_bulk_files(
            DatasetName.DAILY_PRICES,
            start=date(2026, 8, 20),
            end=date(2026, 8, 21),
        )[0]
        with pytest.raises(JQuantsSchemaError, match="CSV payload"):
            client.fetch_bulk_file(descriptor)


def _daily_archive(*, include_rows: bool = True, include_adjusted: bool = True) -> bytes:
    base_columns = (
        "Date",
        "Code",
        "O",
        "H",
        "L",
        "C",
        "Vo",
        "Va",
        "AdjFactor",
    )
    adjusted_columns = (
        "AdjO",
        "AdjH",
        "AdjL",
        "AdjC",
        "AdjVo",
    )
    columns = base_columns + adjusted_columns if include_adjusted else base_columns
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    days = ("2026-08-20", "2026-08-21") if include_rows else ()
    for index, day in enumerate(days):
        close = 2500 + index * 10
        row: dict[str, object] = {
                "Date": day,
                "Code": "72030",
                "O": close - 10,
                "H": close + 20,
                "L": close - 20,
                "C": close,
                "Vo": 1_000_000,
                "Va": close * 1_000_000,
                "AdjFactor": 1,
        }
        if include_adjusted:
            row.update(
                {
                "AdjO": close - 10,
                "AdjH": close + 20,
                "AdjL": close - 20,
                "AdjC": close,
                "AdjVo": 1_000_000,
                }
            )
        writer.writerow(row)
    return gzip.compress(stream.getvalue().encode("utf-8"), mtime=0)
