from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from stock_ai.data import (
    DatasetName,
    JQuantsCredentialError,
    JQuantsRequestError,
    JQuantsSchemaError,
    JQuantsV2Client,
    JQuantsV2Config,
    SubscriptionPlan,
)


class AdvancingClock:
    def __init__(self) -> None:
        self.current = datetime(2026, 8, 24, 9, 0, tzinfo=UTC)
        self.ticks = 0.0

    def now(self) -> datetime:
        self.current += timedelta(seconds=1)
        return self.current

    def monotonic(self) -> float:
        self.ticks += 100.0
        return self.ticks


def _credential() -> str:
    return "-".join(("ephemeral", "unit", "credential"))


def test_v2_client_uses_environment_header_and_paginates(monkeypatch: pytest.MonkeyPatch) -> None:
    credential = _credential()
    monkeypatch.setenv("JQUANTS_API_KEY", credential)
    requests: list[tuple[str, bool, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        pagination_key = request.url.params.get("pagination_key")
        requests.append(
            (
                request.url.path,
                request.headers.get("x-api-key") == credential,
                pagination_key,
            )
        )
        row = _daily_row("72030", 2500.0 if pagination_key is None else 2510.0)
        if pagination_key is None:
            return httpx.Response(200, json={"data": [row], "pagination_key": "next"})
        return httpx.Response(200, json={"data": [row]})

    clock = AdvancingClock()
    with JQuantsV2Client.from_env(
        config=JQuantsV2Config(plan=SubscriptionPlan.FREE),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        payload = client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21))
        representation = repr(client)

    assert payload.pages == 2
    assert len(payload.rows) == 2
    assert requests == [
        ("/v2/equities/bars/daily", True, None),
        ("/v2/equities/bars/daily", True, "next"),
    ]
    assert credential not in representation


def test_v2_client_retries_without_exposing_response_or_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential = _credential()
    monkeypatch.setenv("JQUANTS_API_KEY", credential)
    calls = 0
    waits: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, json={"message": credential})
        return httpx.Response(403, json={"message": credential})

    clock = AdvancingClock()
    with JQuantsV2Client.from_env(
        transport=httpx.MockTransport(handler),
        sleep=waits.append,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        with pytest.raises(JQuantsRequestError) as caught:
            client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21))

    assert calls == 2
    assert waits == [7.0]
    assert credential not in str(caught.value)
    assert "HTTP 403" in str(caught.value)


def test_v2_client_requires_environment_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
    with pytest.raises(JQuantsCredentialError, match="JQUANTS_API_KEY is required"):
        JQuantsV2Client.from_env()


def test_v2_client_rejects_repeated_pagination_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "data": [_daily_row("72030", 2500.0)],
                "pagination_key": "same",
            },
        )

    clock = AdvancingClock()
    with JQuantsV2Client.from_env(
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        with pytest.raises(JQuantsSchemaError, match="repeated"):
            client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21))


def _daily_row(code: str, close: float) -> dict[str, object]:
    return {
        "Date": "2026-08-21",
        "Code": code,
        "O": close - 10,
        "H": close + 20,
        "L": close - 20,
        "C": close,
        "Vo": 1_000_000,
        "Va": close * 1_000_000,
        "AdjFactor": 1.0,
        "AdjO": close - 10,
        "AdjH": close + 20,
        "AdjL": close - 20,
        "AdjC": close,
        "AdjVo": 1_000_000,
    }
