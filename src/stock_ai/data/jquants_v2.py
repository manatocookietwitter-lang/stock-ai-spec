"""Credential-safe client for the official J-Quants API V2."""

from __future__ import annotations

import csv
import gzip
import io
import os
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from stock_ai.data.contracts import (
    ENDPOINT_SCHEMAS,
    BulkFileDescriptor,
    DatasetName,
    FetchedPayload,
    SubscriptionPlan,
)

JQUANTS_V2_BASE_URL = "https://api.jquants.com/v2"
JQUANTS_API_KEY_ENV = "JQUANTS_API_KEY"


class JQuantsError(RuntimeError):
    """Base error that deliberately excludes request headers and response bodies."""


class JQuantsCredentialError(JQuantsError):
    """Raised when the environment credential is unavailable."""


class JQuantsRequestError(JQuantsError):
    """Raised after a sanitized V2 request failure."""


class JQuantsSchemaError(JQuantsError):
    """Raised when the response envelope is invalid."""


class JQuantsPlanError(JQuantsError):
    """Raised before requesting an endpoint unavailable on the declared plan."""


class JQuantsV2Config(BaseModel):
    """Non-secret client configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: SubscriptionPlan = SubscriptionPlan.FREE
    timeout_seconds: float = Field(default=30.0, gt=0, le=120)
    bulk_timeout_seconds: float = Field(default=300.0, gt=0, le=900)
    maximum_bulk_file_bytes: int = Field(default=1_073_741_824, ge=1)
    maximum_bulk_uncompressed_bytes: int = Field(default=536_870_912, ge=1)
    max_retries: int = Field(default=3, ge=0, le=8)
    maximum_retry_wait_seconds: float = Field(default=60.0, gt=0, le=300)
    maximum_pages: int = Field(default=10_000, ge=1)
    user_agent: str = "stock-ai-goal2a/0.1"


class JQuantsV2Client:
    """Small V2-only HTTP adapter with pagination, throttling, and safe errors."""

    def __init__(
        self,
        *,
        config: JQuantsV2Config,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        value = os.environ.get(JQUANTS_API_KEY_ENV, "")
        if not value.strip():
            raise JQuantsCredentialError(
                f"{JQUANTS_API_KEY_ENV} is required for live J-Quants V2 ingestion"
            )
        self._api_key = SecretStr(value)
        self.config = config
        self._sleep = sleep
        self._monotonic = monotonic
        self._now = now or (lambda: datetime.now(UTC))
        self._last_request_at: float | None = None
        self._http = httpx.Client(
            base_url=JQUANTS_V2_BASE_URL,
            transport=transport,
            timeout=config.timeout_seconds,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "User-Agent": config.user_agent,
            },
        )

    @classmethod
    def from_env(
        cls,
        *,
        config: JQuantsV2Config | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] | None = None,
    ) -> JQuantsV2Client:
        """Build a client using the only supported credential source."""

        return cls(
            config=config or JQuantsV2Config(),
            transport=transport,
            sleep=sleep,
            monotonic=monotonic,
            now=now,
        )

    def __enter__(self) -> JQuantsV2Client:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"JQuantsV2Client(plan={self.config.plan.value!r}, base_url={JQUANTS_V2_BASE_URL!r})"

    def close(self) -> None:
        self._http.close()

    def fetch_date(self, dataset: DatasetName, source_date: date) -> FetchedPayload:
        schema = ENDPOINT_SCHEMAS[dataset]
        if not self.config.plan.includes(schema.minimum_plan):
            raise JQuantsPlanError(
                f"{dataset.value} requires {schema.minimum_plan.value} plan or higher"
            )

        requested_at = self._aware_now()
        date_text = source_date.isoformat()
        if dataset in (DatasetName.TRADING_CALENDAR, DatasetName.TOPIX):
            query = {"from": date_text, "to": date_text}
        else:
            query = {"date": date_text}

        rows: list[dict[str, Any]] = []
        pagination_key: str | None = None
        seen_keys: set[str] = set()
        pages = 0
        while True:
            page_query = dict(query)
            if pagination_key is not None:
                page_query["pagination_key"] = pagination_key
            response = self._request(
                schema.endpoint,
                page_query,
                schema.individual_requests_per_minute,
            )
            pages += 1
            if response.status_code == 210:
                break
            payload = self._json_object(response, schema.endpoint)
            page_rows = payload.get("data")
            if not isinstance(page_rows, list) or not all(
                isinstance(row, dict) for row in page_rows
            ):
                raise JQuantsSchemaError(
                    f"invalid J-Quants V2 data envelope for {schema.endpoint}"
                )
            rows.extend(page_rows)
            next_key = payload.get("pagination_key")
            if next_key is None or next_key == "":
                break
            if not isinstance(next_key, str):
                raise JQuantsSchemaError(
                    f"invalid J-Quants V2 pagination key for {schema.endpoint}"
                )
            if next_key in seen_keys:
                raise JQuantsSchemaError(
                    f"repeated J-Quants V2 pagination key for {schema.endpoint}"
                )
            seen_keys.add(next_key)
            pagination_key = next_key
            if pages >= self.config.maximum_pages:
                raise JQuantsSchemaError(
                    f"J-Quants V2 pagination exceeded safety limit for {schema.endpoint}"
                )

        return FetchedPayload(
            dataset=dataset,
            endpoint=schema.endpoint,
            query=tuple(sorted(query.items())),
            requested_at=requested_at,
            received_at=self._aware_now(),
            pages=max(1, pages),
            rows=tuple(rows),
        )

    def list_bulk_files(
        self,
        dataset: DatasetName,
        *,
        start: date,
        end: date,
    ) -> tuple[BulkFileDescriptor, ...]:
        """List official V2 Bulk files for one endpoint and a bounded date range."""

        if end < start:
            raise ValueError("bulk end date cannot precede start date")
        if not self.config.plan.includes(SubscriptionPlan.LIGHT):
            raise JQuantsPlanError("J-Quants V2 Bulk API requires light plan or higher")
        schema = ENDPOINT_SCHEMAS[dataset]
        query = {
            "endpoint": schema.endpoint,
            "from": start.isoformat(),
            "to": end.isoformat(),
        }
        response = self._request("/bulk/list", query, None)
        if response.status_code == 210:
            return ()
        payload = self._json_object(response, "/bulk/list")
        data = payload.get("data")
        if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
            raise JQuantsSchemaError("invalid J-Quants V2 data envelope for /bulk/list")
        descriptors: list[BulkFileDescriptor] = []
        for row in data:
            try:
                key = str(row["Key"])
                size = int(row["Size"])
                modified_text = str(row["LastModified"]).replace("Z", "+00:00")
                last_modified = datetime.fromisoformat(modified_text)
                if last_modified.tzinfo is None or last_modified.utcoffset() is None:
                    raise ValueError
            except (KeyError, TypeError, ValueError, OverflowError):
                raise JQuantsSchemaError("invalid J-Quants V2 Bulk List record") from None
            descriptors.append(
                BulkFileDescriptor(
                    dataset=dataset,
                    endpoint=schema.endpoint,
                    key=key,
                    size=size,
                    last_modified=last_modified.astimezone(UTC),
                )
            )
        fingerprints = [descriptor.fingerprint for descriptor in descriptors]
        if len(fingerprints) != len(set(fingerprints)):
            raise JQuantsSchemaError("duplicate J-Quants V2 Bulk List record")
        return tuple(sorted(descriptors, key=lambda item: (item.last_modified, item.key_hash)))

    def fetch_bulk_file(self, descriptor: BulkFileDescriptor) -> FetchedPayload:
        """Download and parse one signed V2 Bulk CSV without retaining its URL."""

        if descriptor.size > self.config.maximum_bulk_file_bytes:
            raise JQuantsRequestError("J-Quants V2 Bulk file exceeds configured size limit")
        requested_at = self._aware_now()
        response = self._request("/bulk/get", {"key": descriptor.key}, None)
        if response.status_code == 210:
            raise JQuantsSchemaError("J-Quants V2 Bulk Get returned no download URL")
        payload = self._json_object(response, "/bulk/get")
        signed_url = payload.get("url")
        if not isinstance(signed_url, str) or not signed_url.startswith("https://"):
            raise JQuantsSchemaError("invalid J-Quants V2 Bulk Get response")
        body = self._download_bulk_bytes(signed_url)
        if descriptor.size and len(body) != descriptor.size:
            raise JQuantsRequestError("J-Quants V2 Bulk download size mismatch")
        try:
            uncompressed = self._bounded_bulk_uncompress(body)
            text = uncompressed.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError
            required = set(ENDPOINT_SCHEMAS[descriptor.dataset].required_columns)
            if not required <= set(reader.fieldnames):
                raise ValueError
            parsed_rows: list[dict[str, str]] = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError
                parsed_rows.append({str(key): str(value) for key, value in row.items()})
            if not parsed_rows:
                raise ValueError
            rows = tuple(parsed_rows)
        except (OSError, UnicodeError, csv.Error, ValueError):
            raise JQuantsSchemaError("invalid J-Quants V2 Bulk CSV payload") from None
        return FetchedPayload(
            dataset=descriptor.dataset,
            endpoint=descriptor.endpoint,
            query=(("bulk_fingerprint", descriptor.fingerprint),),
            requested_at=requested_at,
            received_at=self._aware_now(),
            pages=1,
            rows=rows,
        )

    def _bounded_bulk_uncompress(self, body: bytes) -> bytes:
        limit = self.config.maximum_bulk_uncompressed_bytes
        if not body.startswith(b"\x1f\x8b"):
            if len(body) > limit:
                raise JQuantsRequestError(
                    "J-Quants V2 Bulk CSV exceeds configured uncompressed size limit"
                )
            return body
        output = io.BytesIO()
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as archive:
                while True:
                    chunk = archive.read(min(1024 * 1024, limit - output.tell() + 1))
                    if not chunk:
                        break
                    output.write(chunk)
                    if output.tell() > limit:
                        raise JQuantsRequestError(
                            "J-Quants V2 Bulk CSV exceeds configured uncompressed size limit"
                        )
        except OSError:
            raise JQuantsSchemaError("invalid J-Quants V2 Bulk gzip payload") from None
        return output.getvalue()

    def _download_bulk_bytes(self, signed_url: str) -> bytes:
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(self.config.max_retries + 1):
            try:
                response = self._http.get(
                    signed_url,
                    timeout=self.config.bulk_timeout_seconds,
                    follow_redirects=True,
                )
            except httpx.TransportError:
                if attempt >= self.config.max_retries:
                    raise JQuantsRequestError(
                        "J-Quants V2 Bulk download transport failed after retries"
                    ) from None
                self._sleep(self._retry_wait(attempt, None))
                continue
            if response.status_code == 200:
                body = response.content
                if len(body) > self.config.maximum_bulk_file_bytes:
                    raise JQuantsRequestError(
                        "J-Quants V2 Bulk file exceeds configured size limit"
                    )
                return body
            if response.status_code in retryable and attempt < self.config.max_retries:
                self._sleep(self._retry_wait(attempt, response.headers.get("Retry-After")))
                continue
            raise JQuantsRequestError(
                f"J-Quants V2 Bulk download failed with HTTP {response.status_code}"
            )
        raise AssertionError("unreachable retry state")

    def _request(
        self,
        endpoint: str,
        query: dict[str, str],
        individual_limit: int | None,
    ) -> httpx.Response:
        retryable = {429, 500, 502, 503, 504}
        for attempt in range(self.config.max_retries + 1):
            self._throttle(individual_limit)
            try:
                response = self._http.get(
                    endpoint,
                    params=query,
                    headers={"x-api-key": self._api_key.get_secret_value()},
                )
            except httpx.TransportError:
                if attempt >= self.config.max_retries:
                    raise JQuantsRequestError(
                        f"J-Quants V2 transport failed for {endpoint} after retries"
                    ) from None
                self._sleep(self._retry_wait(attempt, None))
                continue

            if response.status_code in (200, 210):
                return response
            if response.status_code in retryable and attempt < self.config.max_retries:
                self._sleep(self._retry_wait(attempt, response.headers.get("Retry-After")))
                continue
            raise JQuantsRequestError(
                f"J-Quants V2 request failed with HTTP {response.status_code} for {endpoint}"
            )
        raise AssertionError("unreachable retry state")

    def _throttle(self, individual_limit: int | None) -> None:
        requests_per_minute = self.config.plan.requests_per_minute
        if individual_limit is not None:
            requests_per_minute = min(requests_per_minute, individual_limit)
        interval = 60.0 / requests_per_minute
        now = self._monotonic()
        if self._last_request_at is not None:
            remaining = interval - (now - self._last_request_at)
            if remaining > 0:
                self._sleep(remaining)
                now = self._monotonic()
        self._last_request_at = now

    def _retry_wait(self, attempt: int, retry_after: str | None) -> float:
        wait = float(2**attempt)
        if retry_after is not None:
            try:
                provider_wait = float(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None or retry_at.utcoffset() is None:
                        raise ValueError
                    provider_wait = max(
                        0.0,
                        (retry_at.astimezone(UTC) - self._aware_now()).total_seconds(),
                    )
                except (TypeError, ValueError, OverflowError):
                    provider_wait = 0.0
            if provider_wait > self.config.maximum_retry_wait_seconds:
                raise JQuantsRequestError(
                    "J-Quants Retry-After exceeds configured maximum; retry aborted"
                )
            wait = max(wait, provider_wait)
        return min(wait, self.config.maximum_retry_wait_seconds)

    def _aware_now(self) -> datetime:
        value = self._now()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("client clock must return a timezone-aware datetime")
        return value.astimezone(UTC)

    @staticmethod
    def _json_object(response: httpx.Response, endpoint: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            raise JQuantsSchemaError(
                f"invalid JSON from J-Quants V2 endpoint {endpoint}"
            ) from None
        if not isinstance(payload, dict):
            raise JQuantsSchemaError(
                f"invalid J-Quants V2 response object for {endpoint}"
            )
        return payload
