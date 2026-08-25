"""Fail-closed quality checks for J-Quants V2 response rows."""

from __future__ import annotations

import re
from collections.abc import Collection
from datetime import date
from typing import cast

import numpy as np
import pandas as pd

from stock_ai.data.contracts import (
    ENDPOINT_SCHEMAS,
    DatasetName,
    QualityIssue,
    QualityReport,
    QualitySeverity,
)

_ISSUE_CODE = re.compile(r"^[0-9A-Z]{5}$")


class DataQualityError(ValueError):
    """Raised when source rows cannot safely enter normalized storage."""


def validate_rows(
    dataset: DatasetName,
    rows: tuple[dict[str, object], ...],
    *,
    requested_source_date: date,
    required_columns: Collection[str] | None = None,
) -> QualityReport:
    """Validate schema, keys, dates, and numeric market-data invariants."""

    schema = ENDPOINT_SCHEMAS[dataset]
    issues: list[QualityIssue] = []
    if not rows:
        return QualityReport(
            dataset=dataset,
            rows=0,
            issues=(
                QualityIssue(
                    code="EMPTY_RESPONSE",
                    severity=QualitySeverity.WARNING,
                    message="endpoint returned no rows for the requested source date",
                ),
            ),
        )

    frame = pd.DataFrame.from_records(rows)
    active_required = schema.required_columns if required_columns is None else required_columns
    missing = tuple(column for column in active_required if column not in frame)
    if missing:
        issues.append(
            QualityIssue(
                code="MISSING_REQUIRED_COLUMNS",
                severity=QualitySeverity.ERROR,
                message=f"missing required columns: {', '.join(missing)}",
                rows_affected=len(frame),
            )
        )
        return QualityReport(dataset=dataset, rows=len(frame), issues=tuple(issues))

    key_frame = frame.loc[:, list(schema.primary_key)]
    missing_key = key_frame.isna() | key_frame.astype(str).isin([""])
    invalid_key_rows = missing_key.any(axis=1)
    if invalid_key_rows.any():
        issues.append(
            QualityIssue(
                code="MISSING_PRIMARY_KEY",
                severity=QualitySeverity.ERROR,
                message=f"primary key values are required: {', '.join(schema.primary_key)}",
                rows_affected=int(invalid_key_rows.sum()),
            )
        )

    duplicated = frame.duplicated(list(schema.primary_key), keep=False)
    if duplicated.any():
        issues.append(
            QualityIssue(
                code="DUPLICATE_PRIMARY_KEY",
                severity=QualitySeverity.ERROR,
                message=f"duplicate primary key: {', '.join(schema.primary_key)}",
                rows_affected=int(duplicated.sum()),
            )
        )

    parsed_dates = pd.to_datetime(frame[schema.source_date_column], errors="coerce")
    bad_dates = parsed_dates.isna()
    if bad_dates.any():
        issues.append(
            QualityIssue(
                code="INVALID_SOURCE_DATE",
                severity=QualitySeverity.ERROR,
                message=f"invalid values in {schema.source_date_column}",
                rows_affected=int(bad_dates.sum()),
            )
        )
    else:
        date_mismatch = parsed_dates.dt.date != requested_source_date
        if date_mismatch.any():
            issues.append(
                QualityIssue(
                    code="SOURCE_DATE_MISMATCH",
                    severity=QualitySeverity.ERROR,
                    message="response contains rows outside the requested source date",
                    rows_affected=int(date_mismatch.sum()),
                )
            )

    if "Code" in frame:
        invalid_codes = ~frame["Code"].astype(str).map(
            lambda value: bool(_ISSUE_CODE.fullmatch(value))
        )
        if invalid_codes.any():
            issues.append(
                QualityIssue(
                    code="INVALID_ISSUE_CODE",
                    severity=QualitySeverity.ERROR,
                    message=(
                        "issue codes must contain exactly five uppercase "
                        "alphanumeric characters"
                    ),
                    rows_affected=int(invalid_codes.sum()),
                )
            )

    if dataset is DatasetName.DAILY_PRICES:
        issues.extend(
            _validate_daily_prices(
                frame,
                adjusted_required=required_columns is None,
            )
        )
    elif dataset is DatasetName.TOPIX:
        issues.extend(_validate_ohlc(frame, prefix=""))
    elif dataset is DatasetName.FINANCIAL_SUMMARY:
        issues.extend(_validate_disclosure_times(frame))

    return QualityReport(dataset=dataset, rows=len(frame), issues=tuple(issues))


def require_quality(report: QualityReport) -> None:
    if report.passed:
        return
    codes = ", ".join(
        issue.code
        for issue in report.issues
        if issue.severity is QualitySeverity.ERROR
    )
    raise DataQualityError(f"{report.dataset.value} failed data quality checks: {codes}")


def _numeric(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce"),
    )


def _validate_daily_prices(
    frame: pd.DataFrame,
    *,
    adjusted_required: bool,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    prefixes = ("", "Adj") if adjusted_required else ("",)
    for prefix in prefixes:
        issues.extend(_validate_ohlc(frame, prefix=prefix))

    volume_columns = tuple(
        column for column in ("Vo", "AdjVo", "Va") if column in frame
    )
    volume_values = _numeric(frame, volume_columns)
    volume_provided = frame.loc[:, list(volume_columns)].notna() & (
        frame.loc[:, list(volume_columns)] != ""
    )
    invalid_volume = (
        (volume_provided & volume_values.isna()).any(axis=1)
        | (volume_values.notna() & ~np.isfinite(volume_values)).any(axis=1)
        | (volume_values < 0).any(axis=1)
    )
    factor = pd.to_numeric(frame["AdjFactor"], errors="coerce")
    if invalid_volume.any():
        issues.append(
            QualityIssue(
                code="INVALID_VOLUME_OR_TRADING_VALUE",
                severity=QualitySeverity.ERROR,
                message="raw/adjusted volume and trading value must be finite and non-negative",
                rows_affected=int(invalid_volume.sum()),
            )
        )
    invalid_factor = factor.isna() | ~np.isfinite(factor) | (factor <= 0)
    if invalid_factor.any():
        issues.append(
            QualityIssue(
                code="INVALID_ADJUSTMENT_FACTOR",
                severity=QualitySeverity.ERROR,
                message="adjustment factor must be finite and positive",
                rows_affected=int(invalid_factor.sum()),
            )
        )
    return issues


def _validate_ohlc(frame: pd.DataFrame, *, prefix: str) -> list[QualityIssue]:
    names = tuple(f"{prefix}{suffix}" for suffix in ("O", "H", "L", "C"))
    values = _numeric(frame, names)
    provided = frame.loc[:, list(names)].notna() & (frame.loc[:, list(names)] != "")
    failed_numeric = provided & values.isna()
    non_finite = values.notna() & ~np.isfinite(values)
    non_positive = values.notna() & (values <= 0)
    invalid_numeric = failed_numeric.any(axis=1) | non_finite.any(axis=1) | non_positive.any(axis=1)
    label = "adjusted" if prefix else "raw"
    issues: list[QualityIssue] = []
    if invalid_numeric.any():
        issues.append(
            QualityIssue(
                code=f"INVALID_{label.upper()}_OHLC",
                severity=QualitySeverity.ERROR,
                message=f"{label} OHLC values must be finite and positive when present",
                rows_affected=int(invalid_numeric.sum()),
            )
        )

    complete = values.notna().all(axis=1)
    open_col, high_col, low_col, close_col = names
    inconsistent = complete & (
        (values[high_col] < values[[open_col, close_col]].max(axis=1))
        | (values[low_col] > values[[open_col, close_col]].min(axis=1))
        | (values[high_col] < values[low_col])
    )
    if inconsistent.any():
        issues.append(
            QualityIssue(
                code=f"INCONSISTENT_{label.upper()}_OHLC",
                severity=QualitySeverity.ERROR,
                message=f"{label} OHLC high/low relationships are inconsistent",
                rows_affected=int(inconsistent.sum()),
            )
        )
    return issues


def _validate_disclosure_times(frame: pd.DataFrame) -> list[QualityIssue]:
    combined = frame["DiscDate"].astype(str) + " " + frame["DiscTime"].astype(str)
    invalid = pd.to_datetime(combined, errors="coerce").isna()
    if not invalid.any():
        return []
    return [
        QualityIssue(
            code="INVALID_DISCLOSURE_TIMESTAMP",
            severity=QualitySeverity.ERROR,
            message="financial disclosure date and time must form a valid timestamp",
            rows_affected=int(invalid.sum()),
        )
    ]
