"""Canonical normalization for Goal 2A J-Quants V2 datasets."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from stock_ai.data.contracts import ENDPOINT_SCHEMAS, DatasetName, FetchedPayload

_JST = ZoneInfo("Asia/Tokyo")
_METADATA_COLUMNS = (
    "provider",
    "source_endpoint",
    "source_date",
    "received_at",
    "available_at",
    "as_of",
    "payload_hash",
    "schema_version",
    "ingestion_run_id",
    "source_record_hash",
)


def canonical_payload_hash(rows: tuple[dict[str, Any], ...]) -> str:
    canonical_rows = sorted(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        for row in rows
    )
    payload = ("[" + ",".join(canonical_rows) + "]").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def raw_frame(
    payload: FetchedPayload,
    *,
    ingestion_run_id: str,
    as_of: datetime,
    payload_hash: str,
) -> pd.DataFrame:
    """Preserve provider fields and append the mandatory record metadata."""

    schema = ENDPOINT_SCHEMAS[payload.dataset]
    frame = pd.DataFrame.from_records(payload.rows)
    source_dates: pd.Series[Any]
    if frame.empty:
        frame = pd.DataFrame(columns=list(schema.required_columns))
    source_dates = pd.to_datetime(frame[schema.source_date_column], errors="raise").dt.date
    frame = _add_metadata(
        frame,
        payload=payload,
        ingestion_run_id=ingestion_run_id,
        as_of=as_of,
        payload_hash=payload_hash,
        source_dates=source_dates,
    )
    return frame


def normalize_payload(
    payload: FetchedPayload,
    *,
    ingestion_run_id: str,
    as_of: datetime,
    payload_hash: str,
    raw_object_id: str,
) -> pd.DataFrame:
    """Normalize one validated provider response without inventing unavailable fields."""

    frame = pd.DataFrame.from_records(payload.rows)
    if payload.dataset is DatasetName.SECURITY_MASTER:
        normalized = _security_master(frame)
    elif payload.dataset is DatasetName.DAILY_PRICES:
        normalized = _daily_prices(frame, payload_hash)
    elif payload.dataset is DatasetName.TRADING_CALENDAR:
        normalized = _trading_calendar(frame)
    elif payload.dataset is DatasetName.TOPIX:
        normalized = _topix(frame)
    elif payload.dataset is DatasetName.FINANCIAL_SUMMARY:
        normalized = _financial_summary(frame)
    else:
        raise ValueError(f"unsupported dataset: {payload.dataset.value}")

    schema = ENDPOINT_SCHEMAS[payload.dataset]
    source_dates: pd.Series[Any]
    if frame.empty:
        source_dates = pd.Series([], dtype="object")
    else:
        source_dates = pd.to_datetime(frame[schema.source_date_column], errors="raise").dt.date
    normalized = _add_metadata(
        normalized,
        payload=payload,
        ingestion_run_id=ingestion_run_id,
        as_of=as_of,
        payload_hash=payload_hash,
        source_dates=source_dates,
    )
    normalized["raw_object_id"] = raw_object_id
    return normalized


def _add_metadata(
    frame: pd.DataFrame,
    *,
    payload: FetchedPayload,
    ingestion_run_id: str,
    as_of: datetime,
    payload_hash: str,
    source_dates: pd.Series,
) -> pd.DataFrame:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    if as_of.astimezone(UTC) < payload.received_at.astimezone(UTC):
        raise ValueError("ingestion as_of cannot precede received_at")
    result = frame.copy()
    result["provider"] = "J-Quants"
    result["source_endpoint"] = payload.endpoint
    result["source_date"] = source_dates
    result["received_at"] = pd.Timestamp(payload.received_at)
    # The API exposes current values, not historical correction timestamps. Using
    # receipt time prevents an initial backfill or later correction from leaking
    # into a point-in-time query that predates this immutable observation.
    result["available_at"] = pd.Timestamp(payload.received_at)
    result["as_of"] = pd.Timestamp(as_of)
    result["payload_hash"] = payload_hash
    result["schema_version"] = ENDPOINT_SCHEMAS[payload.dataset].schema_version
    result["ingestion_run_id"] = ingestion_run_id
    if payload.rows:
        result["source_record_hash"] = [
            _record_hash(row) for row in payload.rows
        ]
    else:
        result["source_record_hash"] = pd.Series([], dtype="string")
    return result


def _security_master(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "effective_date",
                "provider_code",
                "symbol",
                "company_name",
                "company_name_en",
                "sector_17_code",
                "sector_17_name",
                "sector_33_code",
                "sector_33_name",
                "size_category",
                "market_code",
                "market_name",
                "margin_code",
                "margin_name",
                "round_lot_size",
                "round_lot_missing_reason",
            ]
        )
    codes = frame["Code"].astype(str)
    return pd.DataFrame(
        {
            "effective_date": pd.to_datetime(frame["Date"]).dt.date,
            "provider_code": codes,
            "symbol": codes.str[:-1],
            "company_name": frame["CoName"].astype("string"),
            "company_name_en": frame["CoNameEn"].astype("string"),
            "sector_17_code": frame["S17"].astype("string"),
            "sector_17_name": frame["S17Nm"].astype("string"),
            "sector_33_code": frame["S33"].astype("string"),
            "sector_33_name": frame["S33Nm"].astype("string"),
            "size_category": frame["ScaleCat"].astype("string"),
            "market_code": frame["Mkt"].astype("string"),
            "market_name": frame["MktNm"].astype("string"),
            "margin_code": frame["Mrgn"].astype("string"),
            "margin_name": frame["MrgnNm"].astype("string"),
            "round_lot_size": pd.Series([pd.NA] * len(frame), dtype="Int64"),
            "round_lot_missing_reason": "not provided by /v2/equities/master",
        }
    )


def _daily_prices(frame: pd.DataFrame, payload_hash: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "trading_date",
                "provider_code",
                "symbol",
                "raw_open",
                "raw_high",
                "raw_low",
                "raw_close",
                "raw_volume",
                "trading_value",
                "adjustment_factor",
                "research_open",
                "research_high",
                "research_low",
                "research_close",
                "research_volume",
                "market_cap_million_yen",
                "ex_rights_type",
                "adjustment_source",
                "morning_close",
                "research_morning_close",
                "afternoon_open",
                "research_afternoon_open",
                "adjustment_version",
            ]
        )
    codes = frame["Code"].astype(str)
    def numeric(column: str) -> pd.Series[Any]:
        if column not in frame:
            return pd.Series([float("nan")] * len(frame), dtype="float64")
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.DataFrame(
        {
            "trading_date": pd.to_datetime(frame["Date"]).dt.date,
            "provider_code": codes,
            "symbol": codes.str[:-1],
            "raw_open": numeric("O"),
            "raw_high": numeric("H"),
            "raw_low": numeric("L"),
            "raw_close": numeric("C"),
            "raw_volume": numeric("Vo"),
            "trading_value": numeric("Va"),
            "adjustment_factor": numeric("AdjFactor"),
            "research_open": numeric("AdjO"),
            "research_high": numeric("AdjH"),
            "research_low": numeric("AdjL"),
            "research_close": numeric("AdjC"),
            "research_volume": numeric("AdjVo"),
            "market_cap_million_yen": numeric("MktCap"),
            "ex_rights_type": (
                frame["ExRT"].astype("string")
                if "ExRT" in frame
                else pd.Series([pd.NA] * len(frame), dtype="string")
            ),
            "adjustment_source": (
                "provider_adjusted_fields"
                if all(column in frame for column in ("AdjO", "AdjH", "AdjL", "AdjC", "AdjVo"))
                else "bulk_adjfactor_only"
            ),
            "morning_close": numeric("MC"),
            "research_morning_close": numeric("MAdjC"),
            "afternoon_open": numeric("AO"),
            "research_afternoon_open": numeric("AAdjO"),
            "adjustment_version": payload_hash,
        }
    )


def _trading_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["trading_date", "holiday_division", "is_equity_business_day"]
        )
    division = pd.to_numeric(frame["HolDiv"], errors="raise").astype("int8")
    return pd.DataFrame(
        {
            "trading_date": pd.to_datetime(frame["Date"]).dt.date,
            "holiday_division": division,
            "is_equity_business_day": division.isin((1, 2)),
        }
    )


def _topix(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=["trading_date", "open", "high", "low", "close"]
        )
    return pd.DataFrame(
        {
            "trading_date": pd.to_datetime(frame["Date"]).dt.date,
            "open": pd.to_numeric(frame["O"], errors="coerce"),
            "high": pd.to_numeric(frame["H"], errors="coerce"),
            "low": pd.to_numeric(frame["L"], errors="coerce"),
            "close": pd.to_numeric(frame["C"], errors="coerce"),
        }
    )


def _financial_summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "disclosed_date",
        "disclosed_time",
        "announced_at",
        "provider_code",
        "symbol",
        "disclosure_number",
        "document_type",
        "period_type",
        "period_start",
        "period_end",
        "sales",
        "operating_profit",
        "ordinary_profit",
        "net_income",
        "eps",
        "total_assets",
        "equity",
        "bps",
        "provider_roe",
        "shares_outstanding_fy",
        "treasury_shares_fy",
        "forecast_sales",
        "forecast_operating_profit",
        "forecast_net_income",
        "forecast_eps",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    disclosed_date = pd.to_datetime(frame["DiscDate"]).dt.date
    announced_at = [
        datetime.combine(day, time.fromisoformat(str(value)), _JST)
        for day, value in zip(disclosed_date, frame["DiscTime"], strict=True)
    ]
    codes = frame["Code"].astype(str)

    def numeric(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series([float("nan")] * len(frame), dtype="float64")
        return pd.to_numeric(frame[column].replace("", pd.NA), errors="coerce")

    return pd.DataFrame(
        {
            "disclosed_date": disclosed_date,
            "disclosed_time": frame["DiscTime"].astype("string"),
            "announced_at": pd.to_datetime(announced_at, utc=True),
            "provider_code": codes,
            "symbol": codes.str[:-1],
            "disclosure_number": frame["DiscNo"].astype("string"),
            "document_type": frame["DocType"].astype("string"),
            "period_type": frame["CurPerType"].astype("string"),
            "period_start": pd.to_datetime(frame["CurPerSt"], errors="coerce").dt.date,
            "period_end": pd.to_datetime(frame["CurPerEn"], errors="coerce").dt.date,
            "sales": numeric("Sales"),
            "operating_profit": numeric("OP"),
            "ordinary_profit": numeric("OdP"),
            "net_income": numeric("NP"),
            "eps": numeric("EPS"),
            "total_assets": numeric("TA"),
            "equity": numeric("Eq"),
            "bps": numeric("BPS"),
            "provider_roe": numeric("ROE"),
            "shares_outstanding_fy": numeric("ShOutFY"),
            "treasury_shares_fy": numeric("TrShFY"),
            "forecast_sales": numeric("FSales"),
            "forecast_operating_profit": numeric("FOP"),
            "forecast_net_income": numeric("FNP"),
            "forecast_eps": numeric("FEPS"),
        }
    )[columns]


def _record_hash(row: dict[str, Any]) -> str:
    encoded = json.dumps(
        row,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


MANDATORY_METADATA_COLUMNS = _METADATA_COLUMNS
