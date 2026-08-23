"""Credential-safe client for the official J-Quants API V2."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from stock_ai.data.contracts import (
    ENDPOINT_SCHEMAS,
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
                wait = max(wait, float(retry_after))
            except ValueError:
                pass
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
