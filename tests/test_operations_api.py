from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from stock_ai.cli import app
from stock_ai.decision import DailyPortfolioDecisionEngine
from stock_ai.domain import PortfolioProposal
from stock_ai.operations import (
    DailyOperationStatus,
    OperationalIntegrityError,
    OperationalStore,
    PipelineState,
    bootstrap_goal5_fixture,
    create_app,
)
from tests.conftest import (
    AS_OF,
    candidate,
    decision_engine,
    decision_policy_snapshot,
    portfolio,
)

runner = CliRunner()


def _client(tmp_path: Path) -> tuple[TestClient, PortfolioProposal]:
    database = tmp_path / "operations.sqlite3"
    store = OperationalStore(database)
    state = portfolio(cash=Decimal("250000"))
    engine = decision_engine()
    proposal = engine.propose(
        portfolio=state,
        candidates=(candidate("B", 0.20),),
        generated_at=AS_OF + timedelta(minutes=1),
        model_bundle_version="test-model-v1",
    )
    store.append_portfolio(state, created_at=AS_OF)
    store.archive_proposal(
        proposal,
        archived_at=AS_OF + timedelta(minutes=2),
        decision_policy=decision_policy_snapshot(proposal, engine.config),
    )
    store.set_daily_status(
        DailyOperationStatus(
            business_date=AS_OF.date(),
            pipeline_state=PipelineState.PROPOSAL_READY,
            updated_at=AS_OF + timedelta(minutes=2),
            data_as_of=AS_OF - timedelta(hours=10),
            morning_data_as_of=AS_OF,
            proposal_id=proposal.proposal_id,
        )
    )
    return TestClient(create_app(database), base_url="http://127.0.0.1"), proposal


def test_today_and_home_keep_proposal_and_actual_portfolio_separate(tmp_path: Path) -> None:
    client, raw_proposal = _client(tmp_path)
    proposal = raw_proposal
    today = client.get(f"/api/v1/today?businessDate={AS_OF.date().isoformat()}")
    assert today.status_code == 200
    payload = today.json()
    assert payload["proposal"]["proposalId"] == proposal.proposal_id
    assert payload["proposal"]["isOrderInstruction"] is False
    assert payload["proposal"]["lines"][0]["recommendedShares"] == 200

    home = client.get(f"/api/v1/home?businessDate={AS_OF.date().isoformat()}").json()
    assert home["portfolio"]["cashValue"] == "250000"
    assert home["holdings"] == []
    validation = client.get("/api/v1/validation?mode=paper").json()
    assert validation["drift_status"] == "INSUFFICIENT_OBSERVATIONS"
    assert validation["weekly_readouts"] == []
    assert validation["monthly_readouts"] == []


def test_manual_intent_header_is_required_and_saving_never_submits_order(
    tmp_path: Path,
) -> None:
    client, raw_proposal = _client(tmp_path)
    proposal = raw_proposal
    line = proposal.lines[0]
    body = {
        "version": 1,
        "savedAt": (AS_OF + timedelta(minutes=3)).isoformat(),
        "lines": [
            {
                "proposalLineId": line.line_id,
                "selectedTargetShares": line.recommended_shares,
            }
        ],
        "confirmsManualOrderOnly": True,
    }
    rejected = client.post(f"/api/v1/proposals/{proposal.proposal_id}/decisions", json=body)
    assert rejected.status_code == 428
    saved = client.post(
        f"/api/v1/proposals/{proposal.proposal_id}/decisions",
        json=body,
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert saved.status_code == 201
    assert saved.json()["confirms_manual_order_only"] is True
    duplicate_version = client.post(
        f"/api/v1/proposals/{proposal.proposal_id}/decisions",
        json=body,
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert duplicate_version.status_code == 409
    status = client.get(f"/api/v1/status?businessDate={AS_OF.date().isoformat()}").json()
    assert status["pipelineState"] == "USER_DECISION_SAVED"
    assert status["orderSubmissionAvailable"] is False

    paths = set(client.get("/api/openapi.json").json()["paths"])
    assert not any(path.startswith("/api/v1/orders") for path in paths)
    assert not any("submit-order" in path or "cancel-order" in path for path in paths)


def test_manual_fill_is_recorded_then_applied_to_next_day(tmp_path: Path) -> None:
    client, raw_proposal = _client(tmp_path)
    proposal = raw_proposal
    line = proposal.lines[0]
    decision_response = client.post(
        f"/api/v1/proposals/{proposal.proposal_id}/decisions",
        json={
            "decisionId": "api-decision",
            "version": 1,
            "savedAt": (AS_OF + timedelta(minutes=3)).isoformat(),
            "lines": [
                {
                    "proposalLineId": line.line_id,
                    "selectedTargetShares": line.recommended_shares,
                }
            ],
        },
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert decision_response.status_code == 201
    execution = client.post(
        "/api/v1/decisions/api-decision/executions",
        json={
            "executionId": "api-fill",
            "executedAt": (AS_OF + timedelta(hours=1)).isoformat(),
            "symbol": line.symbol,
            "accountBucketId": line.account_bucket_id,
            "status": "filled",
            "side": "BUY",
            "orderedShares": 200,
            "filledShares": 200,
            "averageFillPrice": "1000",
        },
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert execution.status_code == 201
    applied = client.post(
        "/api/v1/portfolio/apply-executions",
        json={
            "nextAsOf": (AS_OF + timedelta(days=1)).isoformat(),
            "nextPortfolioId": "api-next-day",
        },
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert applied.status_code == 200
    assert applied.json()["positions"][0]["shares"] == 200


def test_changed_decision_partial_fill_reaches_next_home_state(tmp_path: Path) -> None:
    database = tmp_path / "partial-flow.sqlite3"
    store = OperationalStore(database)
    state = portfolio(cash=Decimal("500000"))
    base_engine = decision_engine()
    engine = DailyPortfolioDecisionEngine(
        config=base_engine.config.model_copy(update={"maximum_symbol_weight": Decimal("0.50")}),
        cost_engine=base_engine.cost_engine,
        tax_engine=base_engine.tax_engine,
    )
    proposal = engine.propose(
        portfolio=state,
        candidates=(candidate("B", 0.20), candidate("C", 0.19)),
        generated_at=AS_OF + timedelta(minutes=1),
        model_bundle_version="test-model-v1",
    )
    assert sum(line.share_difference > 0 for line in proposal.lines) == 2
    store.append_portfolio(state, created_at=AS_OF)
    store.archive_proposal(
        proposal,
        archived_at=AS_OF + timedelta(minutes=2),
        decision_policy=decision_policy_snapshot(proposal, engine.config),
    )
    store.set_daily_status(
        DailyOperationStatus(
            business_date=AS_OF.date(),
            pipeline_state=PipelineState.PROPOSAL_READY,
            updated_at=AS_OF + timedelta(minutes=2),
            proposal_id=proposal.proposal_id,
        )
    )
    client = TestClient(create_app(database), base_url="http://127.0.0.1")
    buy_lines = [line for line in proposal.lines if line.share_difference > 0]
    selected = {
        buy_lines[0].line_id: buy_lines[0].recommended_shares,
        buy_lines[1].line_id: buy_lines[1].current_shares,
    }
    review = client.post(
        f"/api/v1/proposals/{proposal.proposal_id}/review",
        json={"selectedTargets": selected},
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert review.status_code == 200
    assert review.json()["constraint_violations"] == []
    saved = client.post(
        f"/api/v1/proposals/{proposal.proposal_id}/decisions",
        json={
            "decisionId": "partial-decision",
            "version": 1,
            "savedAt": (AS_OF + timedelta(minutes=3)).isoformat(),
            "lines": [
                {"proposalLineId": line_id, "selectedTargetShares": target}
                for line_id, target in selected.items()
            ],
        },
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert saved.status_code == 201
    chosen = buy_lines[0]
    recorded = client.post(
        "/api/v1/decisions/partial-decision/executions",
        json={
            "executionId": "partial-fill",
            "executedAt": (AS_OF + timedelta(hours=1)).isoformat(),
            "symbol": chosen.symbol,
            "accountBucketId": chosen.account_bucket_id,
            "status": "partially_filled",
            "side": "BUY",
            "orderedShares": chosen.recommended_shares,
            "filledShares": 100,
            "averageFillPrice": str(chosen.reference_price),
            "actualCommission": "25",
            "actualOtherCost": "5",
            "taxWithheld": "0",
        },
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert recorded.status_code == 201
    next_as_of = AS_OF + timedelta(days=1)
    applied = client.post(
        "/api/v1/portfolio/apply-executions",
        json={"nextAsOf": next_as_of.isoformat(), "nextPortfolioId": "partial-next"},
        headers={"X-Stock-AI-Intent": "manual-record"},
    )
    assert applied.status_code == 200
    next_home = client.get(f"/api/v1/home?businessDate={next_as_of.date().isoformat()}").json()
    assert next_home["holdings"][0]["symbol"] == chosen.symbol
    assert next_home["holdings"][0]["shares"] == 100


def test_stale_day_blocks_every_proposal_surface_and_secret_is_never_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, proposal = _client(tmp_path)
    store = client.app.state.store
    store.set_daily_status(
        DailyOperationStatus(
            business_date=AS_OF.date(),
            pipeline_state=PipelineState.STALE_DATA,
            updated_at=AS_OF + timedelta(minutes=5),
            blocking_reason="前場データが不足しています",
            is_stale=True,
        )
    )
    today = client.get(f"/api/v1/today?businessDate={AS_OF.date().isoformat()}").json()
    assert today["proposal"] is None
    assert today["isStale"] is True
    assert client.get(f"/api/v1/proposals/{proposal.proposal_id}").status_code == 409
    assert (
        client.get(f"/api/v1/stocks/B?businessDate={AS_OF.date().isoformat()}").status_code
        == 404
    )
    assert (
        client.post(
            f"/api/v1/proposals/{proposal.proposal_id}/review",
            json={"selectedTargets": {proposal.lines[0].line_id: 0}},
            headers={"X-Stock-AI-Intent": "manual-record"},
        ).status_code
        == 409
    )
    assert (
        client.post(
            f"/api/v1/proposals/{proposal.proposal_id}/decisions",
            json={
                "version": 1,
                "savedAt": (AS_OF + timedelta(minutes=6)).isoformat(),
                "lines": [
                    {
                        "proposalLineId": proposal.lines[0].line_id,
                        "selectedTargetShares": 0,
                    }
                ],
            },
            headers={"X-Stock-AI-Intent": "manual-record"},
        ).status_code
        == 409
    )

    monkeypatch.setenv("JQUANTS_API_KEY", "never-return-this-secret")
    raw = client.get("/api/v1/settings").text
    assert "never-return-this-secret" not in raw
    assert client.get("/api/v1/settings").json()["data"]["jQuantsApiKeyConfigured"] is True


def test_read_only_api_states_security_headers_and_pwa_fallback(tmp_path: Path) -> None:
    client, proposal = _client(tmp_path)
    health = client.get("/api/v1/health")
    assert health.status_code == 200
    assert health.headers["x-frame-options"] == "DENY"
    assert health.json()["orderSubmissionAvailable"] is False
    assert client.get(f"/api/v1/proposals/{proposal.proposal_id}").status_code == 200
    assert client.get("/api/v1/proposals/missing").status_code == 404
    assert client.get(f"/api/v1/proposals/{proposal.proposal_id}/decision").json() is None
    assert client.get("/api/v1/decisions/missing/executions").json() == []
    assert client.get("/api/v1/ranking").json() == []
    business_date = AS_OF.date().isoformat()
    assert client.get(f"/api/v1/stocks/B?businessDate={business_date}").status_code == 200
    assert client.get(f"/api/v1/stocks/UNKNOWN?businessDate={business_date}").status_code == 404
    assert client.get("/api/v1/validation?mode=live").json()["blockingReason"]
    assert client.get("/api/v1/validation?mode=historical").json()["blockingReason"]
    assert client.get("/api/v1/notifications").json() == []
    assert client.get("/api/v1/health", headers={"Host": "evil.example"}).status_code == 421
    assert (
        client.post(
            "/api/v1/imports/executions/preview",
            json={"csvText": "x"},
            headers={
                "Host": "127.0.0.1",
                "Origin": "https://evil.example",
                "X-Stock-AI-Intent": "manual-record",
            },
        ).status_code
        == 403
    )

    empty_database = tmp_path / "empty.sqlite3"
    empty_client = TestClient(create_app(empty_database), base_url="http://127.0.0.1")
    assert empty_client.get("/api/v1/home").json()["blockingReason"] == "保有未登録"
    with pytest.raises(OperationalIntegrityError, match="missing proposal"):
        empty_client.app.state.store.set_daily_status(
            DailyOperationStatus(
                business_date=AS_OF.date(),
                pipeline_state=PipelineState.PROPOSAL_READY,
                updated_at=AS_OF,
                proposal_id="missing-proposal",
            )
        )

    static = tmp_path / "dist"
    assets = static / "assets"
    assets.mkdir(parents=True)
    (static / "index.html").write_text("<main>shell</main>", encoding="utf-8")
    (assets / "app.js").write_text("export {};", encoding="utf-8")
    static_client = TestClient(
        create_app(tmp_path / "static.sqlite3", static_dir=static),
        base_url="http://127.0.0.1",
    )
    assert static_client.get("/ranking").text == "<main>shell</main>"
    assert static_client.get("/assets/app.js").text == "export {};"


def test_fixture_settings_are_bound_to_the_archived_decision_policy(tmp_path: Path) -> None:
    database = tmp_path / "settings.sqlite3"
    store = OperationalStore(database)
    jst = ZoneInfo("Asia/Tokyo")
    as_of = datetime.now(jst).replace(hour=11, minute=30, second=0, microsecond=0)
    proposal_id = bootstrap_goal5_fixture(store, as_of=as_of)

    snapshot = store.decision_policy_snapshot(proposal_id)
    assert snapshot is not None
    assert snapshot.config.minimum_cash_ratio == Decimal("0.10")
    assert store.verify_integrity()["decision_policy_snapshots"] == 1

    client = TestClient(create_app(database), base_url="http://127.0.0.1")
    settings = client.get("/api/v1/settings").json()
    assert settings["capital"]["minimumCashRatio"] == "0.10"
    assert settings["decisionPolicies"] == {
        "decisionEngineVersion": "decision-engine-v1",
        "costPolicyId": "fixture-cost-v1",
        "costPolicyVersion": "fixture-cost-v1",
        "taxPolicyId": "fixture-tax-v1",
        "taxPolicyVersion": "fixture-tax-v1",
        "roundLotShares": 100,
        "maximumPositions": 10,
        "maximumSymbolWeight": "0.50",
        "maximumSectorWeight": "0.70",
        "maximumTurnoverRatio": "0.80",
        "maximumTradeAdvRatio": "0.05",
        "minimumImprovementYen": "300",
        "uncertaintyBufferYen": "0",
    }


def test_api_stops_on_tampering_and_uses_status_bound_proposal(tmp_path: Path) -> None:
    client, proposal = _client(tmp_path)
    store = client.app.state.store
    newer = proposal.model_copy(
        update={
            "proposal_id": "newer-orphan",
            "generated_at": proposal.generated_at + timedelta(minutes=1),
        }
    )
    policy = store.decision_policy_snapshot(proposal.proposal_id)
    assert policy is not None
    store.archive_proposal(
        newer,
        archived_at=newer.generated_at + timedelta(minutes=1),
        decision_policy=decision_policy_snapshot(newer, policy.config),
    )
    today = client.get(f"/api/v1/today?businessDate={AS_OF.date().isoformat()}")
    assert today.status_code == 200
    assert today.json()["proposal"]["proposalId"] == proposal.proposal_id
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE proposals SET payload = '{}' WHERE proposal_id = ?",
            (proposal.proposal_id,),
        )
        connection.commit()
    assert client.get(f"/api/v1/today?businessDate={AS_OF.date().isoformat()}").status_code == 503


def test_ops_cli_fixture_verify_scheduler_and_live_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "fixture.sqlite3"
    bootstrap = runner.invoke(
        app,
        [
            "ops",
            "fixture-bootstrap",
            "--database",
            str(database),
            "--as-of",
            AS_OF.isoformat(),
        ],
    )
    assert bootstrap.exit_code == 0, bootstrap.output
    assert "DETERMINISTIC_FIXTURE_ONLY" in bootstrap.output
    verify = runner.invoke(app, ["ops", "verify", "--database", str(database)])
    assert verify.exit_code == 0
    assert "status=OK" in verify.output
    apply_result = runner.invoke(
        app,
        [
            "ops",
            "apply-executions",
            "--database",
            str(database),
            "--next-as-of",
            (AS_OF + timedelta(days=1)).isoformat(),
            "--portfolio-id",
            "cli-next-state",
        ],
    )
    assert apply_result.exit_code == 0, apply_result.output
    assert "order_submission=OUT_OF_SCOPE" in apply_result.output
    backup = tmp_path / "backups" / "fixture.sqlite3"
    backup_result = runner.invoke(
        app,
        ["ops", "backup", "--database", str(database), "--destination", str(backup)],
    )
    assert backup_result.exit_code == 0, backup_result.output
    restored = tmp_path / "restored.sqlite3"
    blocked_restore = runner.invoke(
        app,
        ["ops", "restore", "--backup", str(backup), "--database", str(restored)],
    )
    assert blocked_restore.exit_code == 2
    restored_result = runner.invoke(
        app,
        [
            "ops",
            "restore",
            "--backup",
            str(backup),
            "--database",
            str(restored),
            "--confirm-replace",
        ],
    )
    assert restored_result.exit_code == 0, restored_result.output
    assert OperationalStore(restored).verify_integrity()["proposals"] == 1
    missing_static = runner.invoke(
        app,
        [
            "ops",
            "serve",
            "--database",
            str(database),
            "--static-dir",
            str(tmp_path / "missing-dist"),
        ],
    )
    assert missing_static.exit_code == 2
    assert "run npm build" in missing_static.output
    script = runner.invoke(
        app,
        ["ops", "scheduler-script", "--database", str(database)],
    )
    assert script.exit_code == 0
    assert "Register-ScheduledTask" in script.output
    assert "challenger_training" in script.output
    assert "cannot submit orders" in script.output

    gated_database = tmp_path / "live-gated.sqlite3"
    live = runner.invoke(
        app,
        [
            "ops",
            "run-daily",
            "--database",
            str(gated_database),
            "--business-date",
            AS_OF.date().isoformat(),
        ],
    )
    assert live.exit_code == 2
    assert "BLOCKED" in live.output
    assert OperationalStore(gated_database).latest_proposal(AS_OF.date()) is None
