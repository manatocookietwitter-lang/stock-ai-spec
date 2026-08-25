from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import duckdb
import httpx
import pandas as pd
import pytest

from stock_ai.data import (
    ALL_DATASETS,
    CapabilityStatus,
    DataQualityError,
    DatasetName,
    DuckDBCatalog,
    ImmutableParquetStore,
    JQuantsV2Client,
    JQuantsV2Config,
    JQuantsV2Ingestor,
    ObjectKind,
    StorageIntegrityError,
    SubscriptionPlan,
    capabilities_for,
    validate_rows,
)
from stock_ai.data.contracts import QualityReport
from stock_ai.data.normalize import canonical_payload_hash


class AdvancingClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 8, 24, 0, 0, tzinfo=UTC)
        self.tick = 0.0

    def now(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current

    def monotonic(self) -> float:
        self.tick += 100.0
        return self.tick


def _credential() -> str:
    return "".join(("temporary", "-", "test", "-", "credential"))


def test_fixture_integration_publishes_all_goal2a_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential()
    monkeypatch.setenv("JQUANTS_API_KEY", credential)
    clock = AdvancingClock()
    rows_by_path = _all_endpoint_rows()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": rows_by_path[request.url.path]})

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
        store = ImmutableParquetStore(tmp_path / "lake")
        ingestor = JQuantsV2Ingestor(
            client=client,
            store=store,
            catalog=catalog,
            now=clock.now,
            run_id_factory=lambda _day: "integration-run",
        )
        result = ingestor.sync_date(date(2026, 8, 21), datasets=ALL_DATASETS)

        assert result.status.value == "SUCCEEDED"
        assert len(result.objects) == 2 * len(ALL_DATASETS)
        for dataset in ALL_DATASETS:
            assert catalog.object_count(dataset, ObjectKind.RAW) == 1
            assert catalog.object_count(dataset, ObjectKind.NORMALIZED) == 1
            point_in_time = catalog.point_in_time(dataset, result.completed_at)
            assert len(point_in_time) == 1

        daily = catalog.point_in_time(DatasetName.DAILY_PRICES, result.completed_at)
        assert daily.loc[0, "raw_close"] == 2500.0
        assert daily.loc[0, "research_close"] == 1250.0
        assert daily.loc[0, "symbol"] == "7203"
        assert daily.loc[0, "available_at"] == daily.loc[0, "received_at"]
        assert catalog.verify_integrity(store) == 2 * len(ALL_DATASETS)

    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert credential.encode() not in path.read_bytes()


def test_refetch_is_idempotent_and_correction_is_not_retroactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    clock = AdvancingClock()
    fetch_number = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal fetch_number
        fetch_number += 1
        close = 2500.0 if fetch_number <= 2 else 2550.0
        return httpx.Response(200, json={"data": [_daily_row(close)]})

    run_ids = iter(("run-1", "run-2", "run-3"))
    with (
        JQuantsV2Client.from_env(
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(tmp_path / "catalog.duckdb") as catalog,
    ):
        ingestor = JQuantsV2Ingestor(
            client=client,
            store=ImmutableParquetStore(tmp_path / "lake"),
            catalog=catalog,
            now=clock.now,
            run_id_factory=lambda _day: next(run_ids),
        )
        first = ingestor.sync_date(
            date(2026, 8, 21), datasets=(DatasetName.DAILY_PRICES,)
        )
        second = ingestor.sync_date(
            date(2026, 8, 21), datasets=(DatasetName.DAILY_PRICES,)
        )
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.RAW) == 1
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.NORMALIZED) == 1
        assert first.objects[0].object_id == second.objects[0].object_id

        corrected = ingestor.sync_date(
            date(2026, 8, 21), datasets=(DatasetName.DAILY_PRICES,)
        )
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.NORMALIZED) == 2
        first_available = first.objects[1].available_at
        corrected_available = corrected.objects[1].available_at
        assert catalog.point_in_time(
            DatasetName.DAILY_PRICES, first_available - timedelta(microseconds=1)
        ).empty
        before_correction = catalog.point_in_time(
            DatasetName.DAILY_PRICES,
            corrected_available - timedelta(microseconds=1),
        )
        after_correction = catalog.point_in_time(
            DatasetName.DAILY_PRICES,
            corrected_available,
        )
        assert before_correction.loc[0, "raw_close"] == 2500.0
        assert after_correction.loc[0, "raw_close"] == 2550.0


def test_failed_publish_preserves_existing_object_and_cleans_temporary_data(
    tmp_path: Path,
) -> None:
    root = tmp_path / "lake"
    quality = QualityReport(dataset=DatasetName.DAILY_PRICES, rows=1)
    fixed = datetime(2026, 8, 24, tzinfo=UTC)
    first_store = ImmutableParquetStore(root)
    first = first_store.write(
        kind=ObjectKind.RAW,
        dataset=DatasetName.DAILY_PRICES,
        source_date=date(2026, 8, 21),
        frame=pd.DataFrame({"value": [1]}),
        payload_hash="a" * 64,
        schema_version="test-v1",
        source_endpoint="/equities/bars/daily",
        received_at=fixed,
        available_at=fixed,
        as_of=fixed,
        ingestion_run_id="run-1",
        quality=quality,
    )

    def fail(stage: str) -> None:
        if stage == "after_parquet":
            raise RuntimeError("simulated interruption")

    failing_store = ImmutableParquetStore(root, fault_hook=fail)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        failing_store.write(
            kind=ObjectKind.RAW,
            dataset=DatasetName.DAILY_PRICES,
            source_date=date(2026, 8, 21),
            frame=pd.DataFrame({"value": [2]}),
            payload_hash="b" * 64,
            schema_version="test-v1",
            source_endpoint="/equities/bars/daily",
            received_at=fixed,
            available_at=fixed,
            as_of=fixed,
            ingestion_run_id="run-2",
            quality=quality,
        )

    assert first_store.verify(first.parquet_path.parent).object_id == first.object_id
    partition = first.parquet_path.parent.parent
    assert sorted(path.name for path in partition.iterdir()) == [first.object_id]


def test_quality_failure_keeps_raw_and_does_not_publish_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    clock = AdvancingClock()
    invalid = _daily_row(2500.0)
    del invalid["AdjC"]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [invalid]})

    with (
        JQuantsV2Client.from_env(
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(tmp_path / "catalog.duckdb") as catalog,
    ):
        ingestor = JQuantsV2Ingestor(
            client=client,
            store=ImmutableParquetStore(tmp_path / "lake"),
            catalog=catalog,
            now=clock.now,
            run_id_factory=lambda _day: "bad-run",
        )
        with pytest.raises(DataQualityError, match="MISSING_REQUIRED_COLUMNS"):
            ingestor.sync_date(
                date(2026, 8, 21),
                datasets=(DatasetName.DAILY_PRICES,),
            )
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.RAW) == 1
        assert catalog.object_count(DatasetName.DAILY_PRICES, ObjectKind.NORMALIZED) == 0
        assert catalog.run_status("bad-run") == ("FAILED", "DataQualityError")


def test_failed_multi_dataset_run_is_not_visible_to_point_in_time_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    clock = AdvancingClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/equities/bars/daily":
            return httpx.Response(200, json={"data": [_daily_row(2500.0)]})
        return httpx.Response(200, json={"data": [{"Date": "2026-08-21"}]})

    run_ids = iter(("partial-run", "recovery-run"))
    with (
        JQuantsV2Client.from_env(
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(tmp_path / "catalog.duckdb") as catalog,
    ):
        ingestor = JQuantsV2Ingestor(
            client=client,
            store=ImmutableParquetStore(tmp_path / "lake"),
            catalog=catalog,
            now=clock.now,
            run_id_factory=lambda _day: next(run_ids),
        )
        with pytest.raises(DataQualityError):
            ingestor.sync_date(
                date(2026, 8, 21),
                datasets=(DatasetName.DAILY_PRICES, DatasetName.SECURITY_MASTER),
            )
        assert catalog.point_in_time(DatasetName.DAILY_PRICES, clock.now()).empty

        ingestor.sync_date(date(2026, 8, 21), datasets=(DatasetName.DAILY_PRICES,))
        assert len(catalog.point_in_time(DatasetName.DAILY_PRICES, clock.now())) == 1


def test_store_detects_tampering(tmp_path: Path) -> None:
    fixed = datetime(2026, 8, 24, tzinfo=UTC)
    store = ImmutableParquetStore(tmp_path / "lake")
    stored = store.write(
        kind=ObjectKind.RAW,
        dataset=DatasetName.TOPIX,
        source_date=date(2026, 8, 21),
        frame=pd.DataFrame({"close": [3100.0]}),
        payload_hash="c" * 64,
        schema_version="test-v1",
        source_endpoint="/indices/bars/daily/topix",
        received_at=fixed,
        available_at=fixed,
        as_of=fixed,
        ingestion_run_id="run-1",
        quality=QualityReport(dataset=DatasetName.TOPIX, rows=1),
    )
    with stored.parquet_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(StorageIntegrityError, match="hash mismatch"):
        store.verify(stored.parquet_path.parent)


def test_store_detects_manifest_identity_and_row_count_tampering(tmp_path: Path) -> None:
    fixed = datetime(2026, 8, 24, tzinfo=UTC)
    store = ImmutableParquetStore(tmp_path / "lake")
    stored = store.write(
        kind=ObjectKind.RAW,
        dataset=DatasetName.TOPIX,
        source_date=date(2026, 8, 21),
        frame=pd.DataFrame({"close": [3100.0]}),
        payload_hash="d" * 64,
        schema_version="test-v1",
        source_endpoint="/indices/bars/daily/topix",
        received_at=fixed,
        available_at=fixed,
        as_of=fixed,
        ingestion_run_id="run-identity",
        quality=QualityReport(dataset=DatasetName.TOPIX, rows=1),
    )
    original = stored.manifest_path.read_text(encoding="utf-8")
    identity_tamper = stored.model_copy(update={"schema_version": "other-v1"})
    stored.manifest_path.write_text(identity_tamper.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(StorageIntegrityError, match="manifest identity mismatch"):
        store.verify(stored.parquet_path.parent)

    stored.manifest_path.write_text(original, encoding="utf-8")
    row_tamper = stored.model_copy(update={"rows": 2})
    stored.manifest_path.write_text(row_tamper.model_dump_json(indent=2), encoding="utf-8")
    with pytest.raises(StorageIntegrityError, match="row count mismatch"):
        store.verify(stored.parquet_path.parent)


def test_catalog_path_tampering_fails_integrity_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    clock = AdvancingClock()
    catalog_path = tmp_path / "lake" / "catalog.duckdb"
    store = ImmutableParquetStore(tmp_path / "lake")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [_daily_row(2500.0)]})

    with (
        JQuantsV2Client.from_env(
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
            monotonic=clock.monotonic,
            now=clock.now,
        ) as client,
        DuckDBCatalog(catalog_path) as catalog,
    ):
        JQuantsV2Ingestor(
            client=client,
            store=store,
            catalog=catalog,
            now=clock.now,
        ).sync_date(date(2026, 8, 21), datasets=(DatasetName.DAILY_PRICES,))
        assert catalog.verify_integrity(store) == 2

    with duckdb.connect(str(catalog_path)) as connection:
        current = connection.execute(
            "SELECT parquet_path FROM stored_objects ORDER BY object_id LIMIT 1"
        ).fetchone()
        assert current is not None
        wrong_path = str(Path(str(current[0])).with_name("wrong.parquet"))
        connection.execute(
            "UPDATE stored_objects SET parquet_path = ? WHERE object_id = "
            "(SELECT object_id FROM stored_objects ORDER BY object_id LIMIT 1)",
            [wrong_path],
        )

    with DuckDBCatalog(catalog_path) as catalog:
        with pytest.raises(StorageIntegrityError, match="catalog paths"):
            catalog.verify_integrity(store)


def test_quality_checks_block_bad_schema_duplicates_and_prices() -> None:
    missing = _daily_row(2500.0)
    del missing["AdjC"]
    report = validate_rows(
        DatasetName.DAILY_PRICES,
        (missing,),
        requested_source_date=date(2026, 8, 21),
    )
    assert not report.passed
    assert report.issues[0].code == "MISSING_REQUIRED_COLUMNS"

    invalid = _daily_row(2500.0)
    invalid["H"] = 2400.0
    duplicate_report = validate_rows(
        DatasetName.DAILY_PRICES,
        (invalid, invalid.copy()),
        requested_source_date=date(2026, 8, 21),
    )
    codes = {issue.code for issue in duplicate_report.issues}
    assert {"DUPLICATE_PRIMARY_KEY", "INCONSISTENT_RAW_OHLC"} <= codes
    with pytest.raises(DataQualityError):
        from stock_ai.data import require_quality

        require_quality(duplicate_report)

    nonnumeric_volume = _daily_row(2500.0)
    nonnumeric_volume["Vo"] = "not-a-number"
    volume_report = validate_rows(
        DatasetName.DAILY_PRICES,
        (nonnumeric_volume,),
        requested_source_date=date(2026, 8, 21),
    )
    assert "INVALID_VOLUME_OR_TRADING_VALUE" in {
        issue.code for issue in volume_report.issues
    }


def test_capabilities_fail_closed_by_declared_plan() -> None:
    capabilities = {item.name: item for item in capabilities_for(SubscriptionPlan.FREE)}
    assert capabilities["daily_prices"].status is CapabilityStatus.AVAILABLE
    assert capabilities["topix_context"].status is CapabilityStatus.BLOCKED_BY_PLAN
    assert capabilities["shares_outstanding"].status is CapabilityStatus.PARTIAL
    assert capabilities["intraday_morning"].status is CapabilityStatus.OUT_OF_SCOPE
    assert capabilities["bulk_history"].status is CapabilityStatus.BLOCKED_BY_PLAN
    standard = {
        item.name: item for item in capabilities_for(SubscriptionPlan.STANDARD)
    }
    assert standard["bulk_history"].status is CapabilityStatus.AVAILABLE
    assert (
        standard["exact_1230_entry_label"].status is CapabilityStatus.BLOCKED_BY_PLAN
    )
    premium = {item.name: item for item in capabilities_for(SubscriptionPlan.PREMIUM)}
    assert premium["exact_1230_entry_label"].status is CapabilityStatus.AVAILABLE


def test_payload_identity_is_independent_of_response_row_order() -> None:
    first = _daily_row(2500.0)
    second = dict(first, Code="67580")
    assert canonical_payload_hash((first, second)) == canonical_payload_hash((second, first))


def _all_endpoint_rows() -> dict[str, list[dict[str, object]]]:
    return {
        "/v2/equities/master": [
            {
                "Date": "2026-08-21",
                "Code": "72030",
                "CoName": "トヨタ自動車",
                "CoNameEn": "TOYOTA MOTOR CORPORATION",
                "S17": "6",
                "S17Nm": "自動車・輸送機",
                "S33": "3700",
                "S33Nm": "輸送用機器",
                "ScaleCat": "TOPIX Large70",
                "Mkt": "0111",
                "MktNm": "プライム",
                "Mrgn": "1",
                "MrgnNm": "貸借",
            }
        ],
        "/v2/equities/bars/daily": [_daily_row(2500.0, adjusted_close=1250.0)],
        "/v2/markets/calendar": [{"Date": "2026-08-21", "HolDiv": "1"}],
        "/v2/indices/bars/daily/topix": [
            {"Date": "2026-08-21", "O": 3100.0, "H": 3120.0, "L": 3090.0, "C": 3110.0}
        ],
        "/v2/fins/summary": [_financial_row()],
    }


def _daily_row(close: float, *, adjusted_close: float | None = None) -> dict[str, object]:
    adjusted = close if adjusted_close is None else adjusted_close
    return {
        "Date": "2026-08-21",
        "Code": "72030",
        "O": close - 10,
        "H": close + 20,
        "L": close - 20,
        "C": close,
        "Vo": 1_000_000,
        "Va": close * 1_000_000,
        "AdjFactor": 1.0,
        "AdjO": adjusted - 5,
        "AdjH": adjusted + 10,
        "AdjL": adjusted - 10,
        "AdjC": adjusted,
        "AdjVo": 2_000_000,
    }


def _financial_row() -> dict[str, object]:
    return {
        "DiscDate": "2026-08-21",
        "DiscTime": "15:00",
        "Code": "72030",
        "DiscNo": "202608210001",
        "DocType": "FYFinancialStatements_Consolidated_JP",
        "CurPerType": "FY",
        "CurPerSt": "2025-04-01",
        "CurPerEn": "2026-03-31",
        "Sales": "48000000000000",
        "OP": "5000000000000",
        "OdP": "6000000000000",
        "NP": "4500000000000",
        "EPS": "300.0",
        "TA": "90000000000000",
        "Eq": "35000000000000",
        "ShOutFY": "16000000000",
        "TrShFY": "100000000",
        "FSales": "50000000000000",
        "FOP": "5200000000000",
        "FNP": "4600000000000",
        "FEPS": "310.0",
    }
