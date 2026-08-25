from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from stock_ai.domain import (
    ExecutionRecord,
    ExecutionStatus,
    PortfolioProposal,
    TradeSide,
    UserDecision,
    UserDecisionLine,
)
from stock_ai.operations import (
    AutomationBlockedError,
    AutomationContext,
    AutomationStage,
    DailyAutomation,
    DailyOperationStatus,
    JobStatus,
    OperationalConflictError,
    OperationalIntegrityError,
    OperationalStore,
    PaperCalendarSnapshot,
    PaperOutcome,
    PaperReadoutPeriod,
    PipelineState,
    StageResult,
)
from tests.conftest import (
    AS_OF,
    candidate,
    decision_engine,
    decision_policy_snapshot,
    portfolio,
)

PAPER_CALENDAR_SOURCE_ID = "jpx-calendar-source-snapshot"


def _proposal_store(tmp_path: Path) -> tuple[OperationalStore, PortfolioProposal]:
    store = OperationalStore(tmp_path / "operations.sqlite3")
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
    return store, proposal


def _register_paper_calendar(store: OperationalStore) -> PaperCalendarSnapshot:
    calendar = PaperCalendarSnapshot(
        source_snapshot_id=PAPER_CALENDAR_SOURCE_ID,
        session_dates=tuple(
            (AS_OF + timedelta(days=offset)).date()
            for offset in range(50)
            if (AS_OF + timedelta(days=offset)).weekday() < 5
        ),
        created_at=AS_OF - timedelta(minutes=1),
    )
    assert store.append_paper_calendar(calendar)
    return calendar


def test_proposal_decision_execution_and_next_day_are_separate(tmp_path: Path) -> None:
    store, raw_proposal = _proposal_store(tmp_path)
    proposal = raw_proposal
    line = proposal.lines[0]
    decision = UserDecision(
        decision_id="decision-1",
        proposal_id=proposal.proposal_id,
        version=1,
        saved_at=AS_OF + timedelta(minutes=3),
        lines=(
            UserDecisionLine(
                proposal_line_id=line.line_id,
                selected_target_shares=line.recommended_shares,
            ),
        ),
    )
    assert store.save_decision(decision)
    assert store.proposal(proposal.proposal_id) == proposal

    execution = ExecutionRecord(
        execution_id="manual-fill-1",
        decision_id=decision.decision_id,
        executed_at=AS_OF + timedelta(hours=1),
        symbol=line.symbol,
        account_bucket_id=line.account_bucket_id,
        status=ExecutionStatus.FILLED,
        side=TradeSide.BUY,
        ordered_shares=line.recommended_shares,
        filled_shares=line.recommended_shares,
        average_fill_price=line.reference_price,
        source="manual",
    )
    with pytest.raises(OperationalConflictError, match="timestamp cannot precede"):
        store.append_execution(
            execution.model_copy(
                update={
                    "execution_id": "impossible-fill-time",
                    "executed_at": decision.saved_at - timedelta(seconds=1),
                }
            ),
            confirm_difference=True,
        )
    assert store.append_execution(execution)
    next_state = store.apply_unapplied_executions(
        next_as_of=AS_OF + timedelta(days=1),
        next_portfolio_id="next-day",
        created_at=AS_OF + timedelta(days=1, minutes=1),
    )
    assert next_state.positions[0].shares == line.recommended_shares
    assert next_state.applied_execution_ids == (execution.execution_id,)
    assert store.latest_portfolio() == next_state


def test_immutable_identity_and_decision_constraints_fail_closed(tmp_path: Path) -> None:
    store, raw_proposal = _proposal_store(tmp_path)
    proposal = raw_proposal
    line = proposal.lines[0]
    with pytest.raises(OperationalConflictError, match="different content"):
        store.archive_proposal(
            proposal.model_copy(update={"model_bundle_version": "forged"}),
            archived_at=AS_OF + timedelta(minutes=4),
            decision_policy=decision_policy_snapshot(proposal, decision_engine().config),
        )
    review = store.review_decision(proposal.proposal_id, {line.line_id: 150})
    assert review.constraint_violations
    invalid = UserDecision(
        decision_id="decision-invalid",
        proposal_id=proposal.proposal_id,
        version=1,
        saved_at=AS_OF + timedelta(minutes=3),
        lines=(UserDecisionLine(proposal_line_id=line.line_id, selected_target_shares=150),),
    )
    with pytest.raises(OperationalConflictError, match="constraint"):
        store.save_decision(invalid)


def test_proposal_policy_snapshot_is_atomic_and_required_by_integrity(tmp_path: Path) -> None:
    store = OperationalStore(tmp_path / "policy-atomic.sqlite3")
    state = portfolio(cash=Decimal("250000"))
    engine = decision_engine()
    proposal = engine.propose(
        portfolio=state,
        candidates=(candidate("B", 0.20),),
        generated_at=AS_OF + timedelta(minutes=1),
        model_bundle_version="test-model-v1",
    )
    store.append_portfolio(state, created_at=AS_OF)
    policy = decision_policy_snapshot(proposal, engine.config)
    with pytest.raises(OperationalIntegrityError, match="capture time"):
        store.archive_proposal(
            proposal,
            archived_at=AS_OF + timedelta(minutes=2),
            decision_policy=policy.model_copy(
                update={"captured_at": proposal.generated_at + timedelta(seconds=1)}
            ),
        )
    assert store.latest_proposal(AS_OF.date()) is None

    assert store.archive_proposal(
        proposal,
        archived_at=AS_OF + timedelta(minutes=2),
        decision_policy=policy,
    )
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM decision_policy_snapshots WHERE proposal_id = ?",
            (proposal.proposal_id,),
        )
        connection.commit()
    with pytest.raises(OperationalIntegrityError, match="missing its decision policy"):
        store.verify_integrity()


def test_execution_csv_requires_preview_and_conflict_confirmation(tmp_path: Path) -> None:
    store, raw_proposal = _proposal_store(tmp_path)
    proposal = raw_proposal
    line = proposal.lines[0]
    decision = UserDecision(
        decision_id="decision-csv",
        proposal_id=proposal.proposal_id,
        version=1,
        saved_at=AS_OF + timedelta(minutes=3),
        lines=(
            UserDecisionLine(
                proposal_line_id=line.line_id,
                selected_target_shares=line.recommended_shares,
            ),
        ),
    )
    store.save_decision(decision)
    csv_text = "\n".join(
        (
            "execution_id,decision_id,executed_at,symbol,account_bucket_id,status,side,"
            "ordered_shares,filled_shares,average_fill_price,actual_commission",
            f"csv-1,{decision.decision_id},{(AS_OF + timedelta(hours=1)).isoformat()},"
            f"{line.symbol},{line.account_bucket_id},filled,BUY,200,200,1000,0",
        )
    )
    preview = store.preview_execution_csv(
        csv_text,
        preview_id="preview-1",
        created_at=AS_OF + timedelta(hours=2),
    )
    assert preview.conflicts == ()
    assert store.executions() == ()
    assert store.confirm_execution_import(preview.preview_id) == (1, 0)
    assert store.executions()[0].source == "csv_import"


def test_execution_csv_surfaces_decision_mismatch_before_confirmation(tmp_path: Path) -> None:
    store, proposal = _proposal_store(tmp_path)
    line = proposal.lines[0]
    decision = UserDecision(
        decision_id="decision-csv-mismatch",
        proposal_id=proposal.proposal_id,
        version=1,
        saved_at=AS_OF + timedelta(minutes=3),
        lines=(
            UserDecisionLine(
                proposal_line_id=line.line_id,
                selected_target_shares=line.recommended_shares,
            ),
        ),
    )
    store.save_decision(decision)
    csv_text = "\n".join(
        (
            "execution_id,decision_id,executed_at,symbol,account_bucket_id,status,side,"
            "ordered_shares,filled_shares,average_fill_price",
            f"wrong-side,{decision.decision_id},{(AS_OF + timedelta(hours=1)).isoformat()},"
            f"{line.symbol},{line.account_bucket_id},filled,SELL,200,200,1000",
        )
    )
    preview = store.preview_execution_csv(
        csv_text, preview_id="preview-mismatch", created_at=AS_OF + timedelta(hours=2)
    )
    assert [item.code for item in preview.conflicts] == ["DECISION_MISMATCH"]
    with pytest.raises(OperationalConflictError, match="explicit review"):
        store.confirm_execution_import(preview.preview_id)
    conflict_id = preview.conflicts[0].conflict_id
    assert store.confirm_execution_import(
        preview.preview_id, accepted_conflict_ids=(conflict_id,)
    ) == (1, 0)
    assert store.executions()[0].side is TradeSide.SELL


def test_paper_results_never_mutate_proposal_and_track_minimum_count(tmp_path: Path) -> None:
    store, raw_proposal = _proposal_store(tmp_path)
    calendar = _register_paper_calendar(store)
    proposal = raw_proposal
    original = proposal.model_dump_json()
    outcome = PaperOutcome(
        outcome_id="paper-1",
        proposal_id=proposal.proposal_id,
        horizon_sessions=1,
        horizon_session_dates=((AS_OF + timedelta(days=1)).date(),),
        label_end_at=AS_OF + timedelta(days=1, hours=4),
        label_available_at=AS_OF + timedelta(days=1, hours=4, minutes=30),
        observed_at=AS_OF + timedelta(days=1, hours=4, minutes=31),
        proposal_return=0.02,
        benchmark_return=0.01,
        estimated_cost=Decimal("100"),
        actual_cost=Decimal("120"),
        estimated_tax_effect=Decimal("50"),
        audited_tax_effect=Decimal("45"),
        champion_version="test-model-v1",
        champion_absolute_error=0.01,
        challenger_version="research-only-v2-challenger",
        challenger_absolute_error=0.012,
        calendar_snapshot_id=calendar.calendar_snapshot_id,
        source_snapshot_ids=(
            "future-price-snapshot",
            calendar.calendar_snapshot_id,
            calendar.source_snapshot_id,
        ),
    )
    extra_proposals = tuple(
        proposal.model_copy(
            update={
                "proposal_id": f"paper-proposal-{index}",
                "as_of": proposal.as_of + timedelta(days=7 * (index - 1)),
                "generated_at": proposal.generated_at + timedelta(days=7 * (index - 1)),
            }
        )
        for index in range(2, 5)
    )
    for item in extra_proposals:
        store.archive_proposal(
            item,
            archived_at=item.generated_at + timedelta(minutes=1),
            decision_policy=decision_policy_snapshot(item, decision_engine().config),
        )

    def later_outcome(
        index: int,
        proposal_item: PortfolioProposal,
        champion_error: float,
        challenger_error: float,
    ) -> PaperOutcome:
        endpoint = proposal_item.as_of + timedelta(days=1, hours=4)
        return outcome.model_copy(
            update={
                "outcome_id": f"paper-{index}",
                "proposal_id": proposal_item.proposal_id,
                "horizon_session_dates": (endpoint.date(),),
                "label_end_at": endpoint,
                "label_available_at": endpoint + timedelta(minutes=30),
                "observed_at": endpoint + timedelta(minutes=31),
                "champion_absolute_error": champion_error,
                "challenger_absolute_error": challenger_error,
            }
        )

    outcomes = (
        outcome,
        later_outcome(2, extra_proposals[0], 0.011, 0.009),
        later_outcome(3, extra_proposals[1], 0.03, 0.025),
        later_outcome(4, extra_proposals[2], 0.04, 0.05),
    )
    for item in outcomes:
        store.append_paper_outcome(item)
    summary = store.paper_summary(minimum_observations=5, drift_window=2)
    assert summary.observations == 4
    assert summary.excess_return is not None and summary.excess_return > 0
    assert summary.mean_cost_error == Decimal("20")
    assert summary.challenger_better_rate == pytest.approx(0.5)
    assert summary.champion_version == "test-model-v1"
    assert summary.champion_observations == 4
    assert summary.challenger_observations == 4
    assert summary.drift_status == "DEGRADED"
    assert not summary.is_decision_ready
    assert store.paper_readouts(PaperReadoutPeriod.WEEKLY)
    assert store.paper_readouts(PaperReadoutPeriod.MONTHLY)[0].champion_versions == (
        "test-model-v1",
    )
    assert store.paper_series()[-1].observations == 4
    with pytest.raises(OperationalConflictError, match="one immutable Paper outcome"):
        store.append_paper_outcome(
            outcome.model_copy(update={"outcome_id": "paper-duplicate-horizon"})
        )
    with pytest.raises(OperationalConflictError, match="immutable paper_outcomes identity"):
        store.append_paper_outcome(outcome.model_copy(update={"proposal_return": 0.99}))
    assert store.proposal(proposal.proposal_id).model_dump_json() == original


def test_paper_outcome_maturity_and_provenance_fail_closed(tmp_path: Path) -> None:
    store, proposal = _proposal_store(tmp_path)
    calendar = _register_paper_calendar(store)
    endpoint = AS_OF + timedelta(days=1, hours=4)
    payload = {
        "outcome_id": "paper-safe",
        "proposal_id": proposal.proposal_id,
        "horizon_sessions": 1,
        "horizon_session_dates": (endpoint.date(),),
        "label_end_at": endpoint,
        "label_available_at": endpoint + timedelta(minutes=30),
        "observed_at": endpoint + timedelta(minutes=31),
        "proposal_return": 0.01,
        "benchmark_return": 0.005,
        "estimated_cost": Decimal("10"),
        "estimated_tax_effect": Decimal("0"),
        "champion_version": proposal.model_bundle_version,
        "champion_absolute_error": 0.01,
        "calendar_snapshot_id": calendar.calendar_snapshot_id,
        "source_snapshot_ids": (
            "prices",
            calendar.calendar_snapshot_id,
            calendar.source_snapshot_id,
        ),
    }
    safe = PaperOutcome(**payload)
    with pytest.raises(ValueError, match="available before"):
        PaperOutcome(**{**payload, "label_available_at": endpoint - timedelta(seconds=1)})
    with pytest.raises(ValueError, match="observation cannot precede"):
        PaperOutcome(**{**payload, "observed_at": endpoint})
    with pytest.raises(ValueError, match="calendar must be part"):
        PaperOutcome(**{**payload, "source_snapshot_ids": ("prices",)})
    with pytest.raises(ValueError, match="session path"):
        PaperOutcome(**{**payload, "horizon_sessions": 2})
    with pytest.raises(OperationalConflictError, match="champion version"):
        store.append_paper_outcome(safe.model_copy(update={"champion_version": "unbound-model"}))
    archive_time = AS_OF + timedelta(minutes=2)
    with pytest.raises(ValueError, match="before proposal archival"):
        store.append_paper_outcome(
            safe.model_copy(
                update={
                    "label_end_at": archive_time - timedelta(minutes=3),
                    "label_available_at": archive_time - timedelta(minutes=2),
                    "observed_at": archive_time - timedelta(minutes=1),
                    "horizon_session_dates": ((AS_OF + timedelta(days=1)).date(),),
                }
            )
        )
    with pytest.raises(OperationalIntegrityError, match="authenticated calendar"):
        store.append_paper_outcome(
            safe.model_copy(
                update={
                    "outcome_id": "paper-fake-calendar",
                    "calendar_snapshot_id": "paper-calendar-not-registered",
                    "source_snapshot_ids": (
                        "prices",
                        "paper-calendar-not-registered",
                        calendar.source_snapshot_id,
                    ),
                }
            )
        )
    gapped_endpoint = AS_OF + timedelta(days=10, hours=4)
    with pytest.raises(ValueError, match="authenticated JPX calendar"):
        store.append_paper_outcome(
            safe.model_copy(
                update={
                    "outcome_id": "paper-gapped-calendar",
                    "horizon_sessions": 2,
                    "horizon_session_dates": (
                        (AS_OF + timedelta(days=1)).date(),
                        gapped_endpoint.date(),
                    ),
                    "label_end_at": gapped_endpoint,
                    "label_available_at": gapped_endpoint + timedelta(minutes=30),
                    "observed_at": gapped_endpoint + timedelta(minutes=31),
                }
            )
        )
    late_store = OperationalStore(tmp_path / "late-archive.sqlite3")
    late_state = portfolio(cash=Decimal("250000"))
    late_engine = decision_engine()
    late_proposal = late_engine.propose(
        portfolio=late_state,
        candidates=(candidate("B", 0.20),),
        generated_at=AS_OF + timedelta(minutes=1),
        model_bundle_version="test-model-v1",
    )
    late_store.append_portfolio(late_state, created_at=AS_OF)
    late_store.archive_proposal(
        late_proposal,
        archived_at=endpoint + timedelta(hours=1),
        decision_policy=decision_policy_snapshot(late_proposal, late_engine.config),
    )
    late_store.append_paper_calendar(calendar)
    with pytest.raises(ValueError, match="before its outcome endpoint"):
        late_store.append_paper_outcome(
            safe.model_copy(update={"observed_at": endpoint + timedelta(hours=1, minutes=1)})
        )
    assert store.append_paper_outcome(safe)


@pytest.mark.parametrize(
    ("challenger_versions", "expected_version", "expected_observations"),
    [
        (("challenger-c1", "challenger-c1", None), None, 0),
        (("challenger-c1", "challenger-c2", "challenger-c1"), "challenger-c1", 1),
    ],
)
def test_paper_challenger_uses_only_the_active_contiguous_cohort(
    tmp_path: Path,
    challenger_versions: tuple[str | None, ...],
    expected_version: str | None,
    expected_observations: int,
) -> None:
    store, first = _proposal_store(tmp_path)
    calendar = _register_paper_calendar(store)
    proposals = (
        first,
        *(
            first.model_copy(
                update={
                    "proposal_id": f"challenger-proposal-{index}",
                    "as_of": first.as_of + timedelta(days=7 * index),
                    "generated_at": first.generated_at + timedelta(days=7 * index),
                }
            )
            for index in range(1, len(challenger_versions))
        ),
    )
    for item in proposals[1:]:
        store.archive_proposal(
            item,
            archived_at=item.generated_at + timedelta(minutes=1),
            decision_policy=decision_policy_snapshot(item, decision_engine().config),
        )
    for index, (proposal_item, challenger_version) in enumerate(
        zip(proposals, challenger_versions, strict=True), start=1
    ):
        endpoint = proposal_item.as_of + timedelta(days=1, hours=4)
        store.append_paper_outcome(
            PaperOutcome(
                outcome_id=f"challenger-outcome-{index}",
                proposal_id=proposal_item.proposal_id,
                horizon_sessions=1,
                horizon_session_dates=(endpoint.date(),),
                label_end_at=endpoint,
                label_available_at=endpoint + timedelta(minutes=30),
                observed_at=endpoint + timedelta(minutes=31),
                proposal_return=0.01,
                benchmark_return=0.005,
                estimated_cost=Decimal("10"),
                estimated_tax_effect=Decimal("0"),
                champion_version=proposal_item.model_bundle_version,
                champion_absolute_error=0.01,
                challenger_version=challenger_version,
                challenger_absolute_error=(0.009 if challenger_version is not None else None),
                calendar_snapshot_id=calendar.calendar_snapshot_id,
                source_snapshot_ids=(
                    "prices",
                    calendar.calendar_snapshot_id,
                    calendar.source_snapshot_id,
                ),
            )
        )
    summary = store.paper_summary(minimum_observations=2)
    assert summary.challenger_version == expected_version
    assert summary.challenger_observations == expected_observations
    assert not summary.model_monitoring_ready


def test_backup_and_hash_verification(tmp_path: Path) -> None:
    store, proposal = _proposal_store(tmp_path)
    counts = store.verify_integrity()
    assert counts["portfolio_states"] == 1
    assert counts["proposals"] == 1
    backup = store.backup(tmp_path / "backup" / "operations.sqlite3")
    with pytest.raises(OperationalConflictError, match="already exists"):
        store.backup(backup)
    backup_store = OperationalStore(backup)
    assert backup_store.verify_integrity()["proposals"] == 1
    restored = tmp_path / "restored" / "operations.sqlite3"
    with pytest.raises(OperationalConflictError, match="explicit replacement"):
        OperationalStore.restore_backup(backup, restored, confirm_replace=False)
    OperationalStore.restore_backup(backup, restored, confirm_replace=True)
    assert OperationalStore(restored).verify_integrity()["proposals"] == 1
    with sqlite3.connect(restored) as connection:
        connection.execute(
            "UPDATE proposals SET business_date = '2099-01-01' WHERE proposal_id = ?",
            (proposal.proposal_id,),
        )
        connection.commit()
    with pytest.raises(OperationalIntegrityError, match="catalog identity mismatch"):
        OperationalStore(restored).verify_integrity()
    with sqlite3.connect(restored) as connection:
        connection.execute(
            "UPDATE proposals SET business_date = ? WHERE proposal_id = ?",
            (proposal.as_of.date().isoformat(), proposal.proposal_id),
        )
        connection.commit()
    with sqlite3.connect(restored) as connection:
        connection.execute(
            "UPDATE proposals SET payload = '{}' WHERE proposal_id = ?",
            (proposal.proposal_id,),
        )
        connection.commit()
    with pytest.raises(OperationalIntegrityError, match="content hash mismatch"):
        OperationalStore(restored).verify_integrity()


def test_backup_publish_never_replaces_a_concurrently_claimed_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, _ = _proposal_store(tmp_path)
    destination = tmp_path / "claimed.sqlite3"
    real_link = os.link

    def claim_before_link(
        source: str | bytes | os.PathLike[str], target: str | bytes | os.PathLike[str]
    ) -> None:
        Path(target).write_text("other-process", encoding="utf-8")
        real_link(source, target)

    monkeypatch.setattr(os, "link", claim_before_link)
    with pytest.raises(OperationalConflictError, match="already exists"):
        store.backup(destination)
    assert destination.read_text(encoding="utf-8") == "other-process"


def test_daily_status_requires_an_archived_same_day_proposal(tmp_path: Path) -> None:
    store = OperationalStore(tmp_path / "status.sqlite3")
    status = DailyOperationStatus(
        business_date=AS_OF.date(),
        pipeline_state=PipelineState.PROPOSAL_READY,
        updated_at=AS_OF,
        proposal_id="missing-proposal",
    )
    with pytest.raises(OperationalIntegrityError, match="missing proposal"):
        store.set_daily_status(status)

    payload = status.model_dump_json()
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO daily_status VALUES (?, ?, ?, ?)",
            (
                status.business_date.isoformat(),
                payload,
                hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                status.updated_at.isoformat(),
            ),
        )
    with pytest.raises(OperationalIntegrityError, match="missing proposal"):
        store.verify_integrity()
    with pytest.raises(ValueError, match="proposal-bearing"):
        DailyOperationStatus(
            business_date=AS_OF.date(),
            pipeline_state=PipelineState.USER_DECISION_SAVED,
            updated_at=AS_OF,
        )


def test_position_reconciliation_never_overwrites_without_confirmation(tmp_path: Path) -> None:
    store, _ = _proposal_store(tmp_path)
    imported_as_of = AS_OF + timedelta(hours=4)
    csv_text = "\n".join(
        (
            "as_of,record_type,symbol,account_bucket_id,shares,"
            "average_acquisition_price,market_price,available_cash,reserved_cash",
            f"{imported_as_of.isoformat()},POSITION,B,bucket,100,1000,1010,,",
            f"{imported_as_of.isoformat()},CASH,,bucket,,,,300000,1000",
        )
    )
    preview = store.preview_position_reconciliation_csv(
        csv_text,
        preview_id="positions-1",
        created_at=imported_as_of,
    )
    assert preview.differences[0].ledger_shares == 0
    assert preview.differences[0].imported_shares == 100
    assert preview.cash_differences[0].imported_available_cash == Decimal("300000")
    assert preview.cash_differences[0].imported_reserved_cash == Decimal("1000")
    assert store.latest_portfolio().positions == ()
    with pytest.raises(OperationalConflictError, match="manual confirmation"):
        store.confirm_position_reconciliation(
            preview.preview_id,
            next_portfolio_id="reconciled",
            confirm_all_differences=False,
            created_at=imported_as_of,
        )
    reconciled = store.confirm_position_reconciliation(
        preview.preview_id,
        next_portfolio_id="reconciled",
        confirm_all_differences=True,
        created_at=imported_as_of,
    )
    assert reconciled.positions[0].symbol == "B"
    assert reconciled.cash[0].available_cash == Decimal("300000")
    assert store.latest_portfolio() == reconciled


def test_daily_automation_archives_before_notification_and_is_idempotent(
    tmp_path: Path,
) -> None:
    store, raw_proposal = _proposal_store(tmp_path)
    # Use a fresh proposal identity because _proposal_store has already archived its proposal.
    proposal = raw_proposal.model_copy(
        update={
            "proposal_id": "automation-proposal",
            "generated_at": raw_proposal.generated_at + timedelta(minutes=5),
        }
    )
    calls: list[str] = []

    def proposal_handler(_context: object) -> StageResult:
        calls.append("proposal")
        return StageResult(
            proposal=proposal,
            decision_policy=decision_policy_snapshot(proposal, decision_engine().config),
        )

    def notification_handler(_context: object) -> StageResult:
        calls.append("notification")
        assert store.proposal(proposal.proposal_id) == proposal
        return StageResult()

    automation = DailyAutomation(store)
    run_now = datetime.now(AS_OF.tzinfo)
    first = automation.run(
        business_date=AS_OF.date(),
        now=run_now,
        handlers={
            AutomationStage.PROPOSAL: proposal_handler,
            AutomationStage.NOTIFICATION: notification_handler,
        },
        stages=(AutomationStage.PROPOSAL, AutomationStage.NOTIFICATION),
        owner="test-owner",
        require_upstream=False,
    )
    assert [item.status for item in first] == [JobStatus.SUCCEEDED, JobStatus.SUCCEEDED]
    assert calls == ["proposal", "notification"]
    assert len(store.notifications()) == 1
    assert store.daily_status(AS_OF.date()).pipeline_state is PipelineState.PROPOSAL_READY

    second = automation.run(
        business_date=AS_OF.date(),
        now=run_now + timedelta(minutes=1),
        handlers={},
        stages=(AutomationStage.PROPOSAL, AutomationStage.NOTIFICATION),
        owner="test-owner-2",
        require_upstream=False,
    )
    assert [item.run_id for item in second] == [item.run_id for item in first]
    assert calls == ["proposal", "notification"]
    upgraded = DailyAutomation(store, workflow_version="goal5-v2").run(
        business_date=AS_OF.date(),
        now=run_now + timedelta(minutes=2),
        handlers={
            AutomationStage.PROPOSAL: proposal_handler,
            AutomationStage.NOTIFICATION: notification_handler,
        },
        stages=(AutomationStage.PROPOSAL, AutomationStage.NOTIFICATION),
        owner="test-owner-3",
        require_upstream=False,
    )
    assert [item.status for item in upgraded] == [
        JobStatus.SUCCEEDED,
        JobStatus.SUCCEEDED,
    ]
    assert calls == ["proposal", "notification", "proposal", "notification"]


def test_automation_blocks_and_never_reuses_a_prior_proposal(tmp_path: Path) -> None:
    store, _ = _proposal_store(tmp_path)
    next_day = AS_OF + timedelta(days=1)
    automation = DailyAutomation(store)

    def blocked(_context: object) -> StageResult:
        raise AutomationBlockedError("BLOCKED_BY_DATA_CAPABILITY", "11:30 provider is unavailable")

    jobs = automation.run(
        business_date=next_day.date(),
        now=datetime.now(AS_OF.tzinfo),
        handlers={AutomationStage.MORNING_CAPTURE: blocked},
        stages=(AutomationStage.MORNING_CAPTURE, AutomationStage.PROPOSAL),
        require_upstream=False,
    )
    assert len(jobs) == 1
    assert jobs[0].status is JobStatus.BLOCKED
    status = store.daily_status(next_day.date())
    assert status is not None
    assert status.pipeline_state is PipelineState.STALE_DATA
    assert status.proposal_id is None
    assert store.latest_proposal(next_day.date()) is None


def test_automation_persists_started_state_and_never_exception_text(tmp_path: Path) -> None:
    store = OperationalStore(tmp_path / "automation.sqlite3")
    marker = "never-persist-api-key-value"
    run_now = datetime.now(AS_OF.tzinfo)

    def failed(context: AutomationContext) -> StageResult:
        assert context.idempotency_key.endswith("goal5-v1")
        assert context.heartbeat()
        raise RuntimeError(marker)

    jobs = DailyAutomation(store).run(
        business_date=AS_OF.date(),
        now=run_now,
        handlers={AutomationStage.DATA_SYNC: failed},
        stages=(AutomationStage.DATA_SYNC,),
        owner="secret-test",
    )
    assert jobs[0].status is JobStatus.FAILED
    assert marker not in jobs[0].model_dump_json()
    status = store.daily_status(AS_OF.date())
    assert status is not None and marker not in status.model_dump_json()
    assert store.jobs()[0].finished_at is not None


def test_automation_retry_preserves_freshness_and_reused_stage_provenance(
    tmp_path: Path,
) -> None:
    store = OperationalStore(tmp_path / "freshness.sqlite3")
    run_now = datetime.now(AS_OF.tzinfo)
    data_as_of = run_now - timedelta(hours=2)
    morning_as_of = run_now - timedelta(minutes=30)
    automation = DailyAutomation(store)

    first = automation.run(
        business_date=run_now.date(),
        now=run_now,
        handlers={
            AutomationStage.DATA_SYNC: lambda _context: StageResult(data_as_of=data_as_of),
            AutomationStage.MORNING_CAPTURE: lambda _context: StageResult(
                morning_data_as_of=morning_as_of
            ),
        },
        stages=(AutomationStage.DATA_SYNC, AutomationStage.MORNING_CAPTURE),
        owner="freshness-owner-1",
        require_upstream=False,
    )
    assert all(item.status is JobStatus.SUCCEEDED for item in first)

    def failed(_context: AutomationContext) -> StageResult:
        raise RuntimeError("provider detail must not persist")

    automation.run(
        business_date=run_now.date(),
        now=run_now + timedelta(minutes=1),
        handlers={AutomationStage.PREDICTION: failed},
        stages=(AutomationStage.PREDICTION,),
        owner="freshness-owner-2",
        require_upstream=False,
    )
    failed_status = store.daily_status(run_now.date())
    assert failed_status is not None
    assert failed_status.data_as_of == data_as_of
    assert failed_status.morning_data_as_of == morning_as_of

    reused = automation.run(
        business_date=run_now.date(),
        now=run_now + timedelta(minutes=2),
        handlers={},
        stages=(AutomationStage.DATA_SYNC, AutomationStage.MORNING_CAPTURE),
        owner="freshness-owner-3",
        require_upstream=False,
    )
    assert all(item.status is JobStatus.SUCCEEDED for item in reused)
    repaired = store.daily_status(run_now.date())
    assert repaired is not None
    assert repaired.data_as_of == data_as_of
    assert repaired.morning_data_as_of == morning_as_of


def test_automation_cannot_succeed_after_losing_its_process_lock(tmp_path: Path) -> None:
    store = OperationalStore(tmp_path / "lost-lock.sqlite3")
    run_now = datetime.now(AS_OF.tzinfo)
    lock_name = f"daily:{run_now.date().isoformat()}"

    def lose_lock(context: AutomationContext) -> StageResult:
        store.release_lock(lock_name, owner="first-owner")
        second = DailyAutomation(store).run(
            business_date=run_now.date(),
            now=context.now,
            handlers={
                AutomationStage.DATA_SYNC: lambda _context: StageResult(data_as_of=context.now)
            },
            stages=(AutomationStage.DATA_SYNC,),
            owner="second-owner",
        )
        assert second[0].status is JobStatus.SUCCEEDED
        return StageResult(data_as_of=context.now)

    jobs = DailyAutomation(store).run(
        business_date=run_now.date(),
        now=run_now,
        handlers={AutomationStage.DATA_SYNC: lose_lock},
        stages=(AutomationStage.DATA_SYNC,),
        owner="first-owner",
    )
    assert jobs[0].status is JobStatus.BLOCKED
    assert jobs[0].reason_code == "LOCK_OWNERSHIP_LOST"
    assert sum(item.status is JobStatus.SUCCEEDED for item in store.jobs()) == 1
    status = store.daily_status(run_now.date())
    assert status is not None
    assert status.pipeline_state is PipelineState.PRE_MARKET


def test_scheduler_stage_requires_same_workflow_upstream_success(tmp_path: Path) -> None:
    store = OperationalStore(tmp_path / "upstream.sqlite3")
    called = False

    def should_not_run(_context: AutomationContext) -> StageResult:
        nonlocal called
        called = True
        return StageResult()

    jobs = DailyAutomation(store).run(
        business_date=AS_OF.date(),
        now=AS_OF,
        handlers={AutomationStage.CANDIDATE_SELECTION: should_not_run},
        stages=(AutomationStage.CANDIDATE_SELECTION,),
    )
    assert not called
    assert jobs[0].reason_code == "UPSTREAM_NOT_SUCCEEDED"
