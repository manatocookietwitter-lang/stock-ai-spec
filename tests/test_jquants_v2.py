from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from stock_ai.data import (
    BulkFileDescriptor,
    DatasetName,
    JQuantsCredentialError,
    JQuantsPlanError,
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


@pytest.mark.parametrize(
    "retry_after",
    ["120", "Mon, 24 Aug 2026 00:02:00 GMT"],
)
def test_v2_client_aborts_when_retry_after_exceeds_safe_wait(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: str,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    waits: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": retry_after})

    clock = AdvancingClock()
    with JQuantsV2Client.from_env(
        transport=httpx.MockTransport(handler),
        sleep=waits.append,
        monotonic=clock.monotonic,
        now=lambda: datetime(2026, 8, 24, tzinfo=UTC),
    ) as client:
        with pytest.raises(JQuantsRequestError, match="retry aborted"):
            client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21))
    assert waits == []


def test_v2_client_fails_closed_on_plan_and_malformed_envelopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    clock = AdvancingClock()
    with JQuantsV2Client.from_env(
        transport=httpx.MockTransport(lambda _request: httpx.Response(210)),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        with pytest.raises(JQuantsPlanError, match="light plan"):
            client.fetch_date(DatasetName.TOPIX, date(2026, 8, 21))
        assert client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21)).rows == ()
        with pytest.raises(ValueError, match="cannot precede"):
            client.list_bulk_files(
                DatasetName.DAILY_PRICES,
                start=date(2026, 8, 22),
                end=date(2026, 8, 21),
            )
        with pytest.raises(JQuantsPlanError, match="Bulk API"):
            client.list_bulk_files(
                DatasetName.DAILY_PRICES,
                start=date(2026, 8, 21),
                end=date(2026, 8, 21),
            )

    responses = iter(
        (
            httpx.Response(200, json={"data": "not-a-list"}),
            httpx.Response(200, json={"data": [], "pagination_key": 123}),
        )
    )
    with JQuantsV2Client.from_env(
        transport=httpx.MockTransport(lambda _request: next(responses)),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        with pytest.raises(JQuantsSchemaError, match="data envelope"):
            client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21))
        with pytest.raises(JQuantsSchemaError, match="pagination key"):
            client.fetch_date(DatasetName.DAILY_PRICES, date(2026, 8, 21))


def test_v2_bulk_rejects_invalid_list_get_size_and_gzip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JQUANTS_API_KEY", _credential())
    clock = AdvancingClock()
    responses = iter(
        (
            httpx.Response(200, json={"data": "invalid"}),
            httpx.Response(200, json={"data": [{"Key": "missing-fields"}]}),
            httpx.Response(210),
            httpx.Response(200, json={"url": "http://not-https.example/file"}),
            httpx.Response(200, json={"url": "https://download.example/file"}),
            httpx.Response(200, content=b"short"),
            httpx.Response(200, json={"url": "https://download.example/file"}),
            httpx.Response(200, content=b"\x1f\x8b-invalid"),
        )
    )
    with JQuantsV2Client.from_env(
        config=JQuantsV2Config(plan=SubscriptionPlan.LIGHT, max_retries=0),
        transport=httpx.MockTransport(lambda _request: next(responses)),
        sleep=lambda _seconds: None,
        monotonic=clock.monotonic,
        now=clock.now,
    ) as client:
        for match in ("data envelope", "Bulk List record"):
            with pytest.raises(JQuantsSchemaError, match=match):
                client.list_bulk_files(
                    DatasetName.DAILY_PRICES,
                    start=date(2026, 8, 21),
                    end=date(2026, 8, 21),
                )
        descriptor = BulkFileDescriptor(
            dataset=DatasetName.DAILY_PRICES,
            endpoint="/equities/bars/daily",
            key="invalid-key",
            size=6,
            last_modified=datetime(2026, 8, 24, tzinfo=UTC),
        )
        with pytest.raises(JQuantsSchemaError, match="no download URL"):
            client.fetch_bulk_file(descriptor)
        with pytest.raises(JQuantsSchemaError, match="Bulk Get response"):
            client.fetch_bulk_file(descriptor)
        with pytest.raises(JQuantsRequestError, match="size mismatch"):
            client.fetch_bulk_file(descriptor)
        gzip_descriptor = descriptor.model_copy(update={"size": len(b"\x1f\x8b-invalid")})
        with pytest.raises(JQuantsSchemaError, match="gzip payload"):
            client.fetch_bulk_file(gzip_descriptor)


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
