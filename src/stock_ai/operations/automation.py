"""Restart-safe orchestration for the local decision-support workflow."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from uuid import uuid4

from stock_ai.domain import PortfolioProposal
from stock_ai.operations.models import (
    AutomationJobRecord,
    AutomationStage,
    DailyOperationStatus,
    DecisionPolicySnapshot,
    JobStatus,
    NotificationChannel,
    NotificationRecord,
    NotificationStatus,
    PipelineState,
)
from stock_ai.operations.store import OperationalStore


class AutomationBlockedError(RuntimeError):
    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(detail)
        self.reason_code = reason_code


def _safe_reason_code(value: str, *, fallback: str) -> str:
    normalized = value.strip().upper()
    return normalized if re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", normalized) else fallback


def _safe_detail(stage: AutomationStage, reason_code: str) -> str:
    """Return stable operational detail without persisting provider exceptions."""

    return f"{stage.value}:{reason_code}"


@dataclass(frozen=True)
class AutomationContext:
    business_date: date
    now: datetime
    store: OperationalStore
    idempotency_key: str
    heartbeat: Callable[[], bool]


@dataclass(frozen=True)
class StageResult:
    artifact_id: str | None = None
    data_as_of: datetime | None = None
    morning_data_as_of: datetime | None = None
    proposal: PortfolioProposal | None = None
    decision_policy: DecisionPolicySnapshot | None = None
    detail: str | None = None


StageHandler = Callable[[AutomationContext], StageResult]

DAILY_STAGES: tuple[AutomationStage, ...] = (
    AutomationStage.DATA_SYNC,
    AutomationStage.CANDIDATE_SELECTION,
    AutomationStage.MORNING_CAPTURE,
    AutomationStage.FREEZE_1130,
    AutomationStage.PREDICTION,
    AutomationStage.PROPOSAL,
    AutomationStage.NOTIFICATION,
    AutomationStage.EOD_UPDATE,
)


def _job_run_id(idempotency_key: str) -> str:
    return "job-" + hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()[:24]


class DailyAutomation:
    """Run explicitly configured stages without any broker or order integration."""

    def __init__(self, store: OperationalStore, *, workflow_version: str = "goal5-v1") -> None:
        self.store = store
        self.workflow_version = workflow_version

    def run(
        self,
        *,
        business_date: date,
        now: datetime,
        handlers: Mapping[AutomationStage, StageHandler],
        stages: Sequence[AutomationStage] = DAILY_STAGES,
        owner: str | None = None,
        require_upstream: bool = True,
    ) -> tuple[AutomationJobRecord, ...]:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("automation time must be timezone-aware")
        process_owner = owner or f"process-{uuid4()}"
        lock_name = f"daily:{business_date.isoformat()}"
        if not self.store.acquire_lock(lock_name, owner=process_owner, now=now):
            raise RuntimeError("daily workflow is already running")
        completed: list[AutomationJobRecord] = []
        try:
            for stage in stages:
                existing = self._successful_job(business_date, stage)
                if existing is not None:
                    self._set_success_status(
                        business_date,
                        existing.finished_at or now,
                        stage,
                        StageResult(
                            data_as_of=existing.data_as_of,
                            morning_data_as_of=existing.morning_data_as_of,
                        ),
                        existing.artifact_id,
                    )
                    completed.append(existing)
                    continue
                attempt = self._attempt_count(business_date, stage) + 1
                logical_stage_key = (
                    f"{business_date.isoformat()}:{stage.value}:{self.workflow_version}"
                )
                idempotency_key = f"{logical_stage_key}:attempt-{attempt}"
                handler = handlers.get(stage)
                if require_upstream and not self._upstream_succeeded(business_date, stage):
                    detail = _safe_detail(stage, "UPSTREAM_NOT_SUCCEEDED")
                    record = self._terminal_job(
                        business_date=business_date,
                        stage=stage,
                        now=now,
                        idempotency_key=idempotency_key,
                        status=JobStatus.BLOCKED,
                        reason_code="UPSTREAM_NOT_SUCCEEDED",
                        detail=detail,
                    )
                    self.store.append_job(record)
                    self._set_blocked_status(business_date, now, stage, detail)
                    completed.append(record)
                    break
                if handler is None:
                    record = self._terminal_job(
                        business_date=business_date,
                        stage=stage,
                        now=now,
                        idempotency_key=idempotency_key,
                        status=JobStatus.BLOCKED,
                        reason_code="BLOCKED_BY_DATA_CAPABILITY",
                        detail=f"{stage.value} provider/handler is not configured",
                    )
                    self.store.append_job(record)
                    self._set_blocked_status(business_date, now, stage, record.detail or "blocked")
                    completed.append(record)
                    break
                try:
                    running = AutomationJobRecord(
                        run_id=_job_run_id(idempotency_key),
                        idempotency_key=idempotency_key,
                        business_date=business_date,
                        stage=stage,
                        status=JobStatus.RUNNING,
                        started_at=now,
                    )
                    self.store.append_job(running)
                    result = handler(
                        AutomationContext(
                            business_date=business_date,
                            now=now,
                            store=self.store,
                            idempotency_key=logical_stage_key,
                            heartbeat=lambda: self.store.refresh_lock(
                                lock_name,
                                owner=process_owner,
                                now=datetime.now(now.tzinfo),
                            ),
                        )
                    )
                    if not self.store.refresh_lock(
                        lock_name,
                        owner=process_owner,
                        now=datetime.now(now.tzinfo),
                    ):
                        raise AutomationBlockedError(
                            "LOCK_OWNERSHIP_LOST",
                            "workflow lock expired or moved to another process",
                        )
                    artifact_id = self._persist_stage_result(
                        stage, result, now, business_date=business_date
                    )
                    record = self._terminal_job(
                        business_date=business_date,
                        stage=stage,
                        now=now,
                        idempotency_key=idempotency_key,
                        status=JobStatus.SUCCEEDED,
                        artifact_id=artifact_id,
                        data_as_of=result.data_as_of,
                        morning_data_as_of=result.morning_data_as_of,
                        detail=_safe_detail(stage, "SUCCEEDED"),
                    )
                    self.store.finish_job(
                        record,
                        required_lock=(
                            lock_name,
                            process_owner,
                            datetime.now(now.tzinfo),
                        ),
                    )
                    self._set_success_status(business_date, now, stage, result, artifact_id)
                    completed.append(record)
                except AutomationBlockedError as exc:
                    reason_code = _safe_reason_code(exc.reason_code, fallback="STAGE_BLOCKED")
                    detail = _safe_detail(stage, reason_code)
                    record = self._terminal_job(
                        business_date=business_date,
                        stage=stage,
                        now=now,
                        idempotency_key=idempotency_key,
                        status=JobStatus.BLOCKED,
                        reason_code=reason_code,
                        detail=detail,
                    )
                    self.store.finish_job(record)
                    if self.store.refresh_lock(
                        lock_name,
                        owner=process_owner,
                        now=datetime.now(now.tzinfo),
                    ):
                        self._set_blocked_status(business_date, now, stage, detail)
                    completed.append(record)
                    break
                except Exception:
                    detail = _safe_detail(stage, "STAGE_FAILED")
                    record = self._terminal_job(
                        business_date=business_date,
                        stage=stage,
                        now=now,
                        idempotency_key=idempotency_key,
                        status=JobStatus.FAILED,
                        reason_code="STAGE_FAILED",
                        detail=detail,
                    )
                    self.store.finish_job(record)
                    if self.store.refresh_lock(
                        lock_name,
                        owner=process_owner,
                        now=datetime.now(now.tzinfo),
                    ):
                        self._set_failed_status(
                            business_date, now, stage, record.detail or "failed"
                        )
                    completed.append(record)
                    break
        finally:
            self.store.release_lock(lock_name, owner=process_owner)
        return tuple(completed)

    def _successful_job(
        self, business_date: date, stage: AutomationStage
    ) -> AutomationJobRecord | None:
        jobs = self.store.jobs(business_date=business_date, stage=stage)
        marker = f":{self.workflow_version}:"
        return next(
            (
                item
                for item in reversed(jobs)
                if item.status is JobStatus.SUCCEEDED and marker in item.idempotency_key
            ),
            None,
        )

    def _attempt_count(self, business_date: date, stage: AutomationStage) -> int:
        return len(self.store.jobs(business_date=business_date, stage=stage))

    def _upstream_succeeded(self, business_date: date, stage: AutomationStage) -> bool:
        if stage in {AutomationStage.DATA_SYNC, AutomationStage.CHALLENGER_TRAINING}:
            return True
        if stage not in DAILY_STAGES:
            return False
        index = DAILY_STAGES.index(stage)
        return self._successful_job(business_date, DAILY_STAGES[index - 1]) is not None

    @staticmethod
    def _terminal_job(
        *,
        business_date: date,
        stage: AutomationStage,
        now: datetime,
        idempotency_key: str,
        status: JobStatus,
        artifact_id: str | None = None,
        data_as_of: datetime | None = None,
        morning_data_as_of: datetime | None = None,
        reason_code: str | None = None,
        detail: str | None = None,
    ) -> AutomationJobRecord:
        return AutomationJobRecord(
            run_id=_job_run_id(idempotency_key),
            idempotency_key=idempotency_key,
            business_date=business_date,
            stage=stage,
            status=status,
            started_at=now,
            finished_at=now,
            artifact_id=artifact_id,
            data_as_of=data_as_of,
            morning_data_as_of=morning_data_as_of,
            reason_code=reason_code,
            detail=detail,
        )

    def _persist_stage_result(
        self,
        stage: AutomationStage,
        result: StageResult,
        now: datetime,
        *,
        business_date: date,
    ) -> str | None:
        if stage is AutomationStage.PROPOSAL:
            if result.proposal is None:
                raise ValueError("proposal stage must return a proposal")
            if result.decision_policy is None:
                raise ValueError("proposal stage must return its exact decision policy")
            self.store.archive_proposal(
                result.proposal,
                archived_at=now,
                decision_policy=result.decision_policy,
            )
            return result.proposal.proposal_id
        if stage is AutomationStage.NOTIFICATION:
            if result.proposal is not None:
                raise ValueError("notification stage must reference the archived daily proposal")
            proposal = self.store.latest_proposal(business_date)
            if proposal is None:
                raise ValueError("notification cannot run before proposal archival")
            notification = NotificationRecord(
                notification_id=f"in-app-{proposal.proposal_id}",
                proposal_id=proposal.proposal_id,
                channel=NotificationChannel.IN_APP,
                status=NotificationStatus.DELIVERED,
                created_at=proposal.generated_at,
                delivered_at=proposal.generated_at,
                title="今日の提案が完成しました",
                body="判断を保存しても証券会社への注文は行われません。",
            )
            self.store.append_notification(notification)
            return notification.notification_id
        return result.artifact_id

    def _set_success_status(
        self,
        business_date: date,
        now: datetime,
        stage: AutomationStage,
        result: StageResult,
        artifact_id: str | None,
    ) -> None:
        previous = self.store.daily_status(business_date)
        if stage is AutomationStage.CHALLENGER_TRAINING and previous is None:
            return
        state = {
            AutomationStage.DATA_SYNC: PipelineState.PRE_MARKET,
            AutomationStage.CANDIDATE_SELECTION: PipelineState.PRE_MARKET,
            AutomationStage.MORNING_CAPTURE: PipelineState.MORNING_ANALYSIS,
            AutomationStage.FREEZE_1130: PipelineState.FREEZING_INPUTS,
            AutomationStage.PREDICTION: PipelineState.GENERATING_PROPOSAL,
            AutomationStage.PROPOSAL: PipelineState.PROPOSAL_READY,
            AutomationStage.NOTIFICATION: PipelineState.PROPOSAL_READY,
            AutomationStage.EOD_UPDATE: PipelineState.MARKET_CLOSED,
            AutomationStage.CHALLENGER_TRAINING: (
                previous.pipeline_state if previous else PipelineState.MARKET_CLOSED
            ),
        }[stage]
        proposal_id = (
            artifact_id
            if stage is AutomationStage.PROPOSAL
            else previous.proposal_id
            if previous is not None
            else None
        )
        self.store.set_daily_status(
            DailyOperationStatus(
                business_date=business_date,
                pipeline_state=state,
                updated_at=now,
                data_as_of=result.data_as_of
                or (previous.data_as_of if previous is not None else None),
                morning_data_as_of=result.morning_data_as_of
                or (previous.morning_data_as_of if previous is not None else None),
                proposal_id=proposal_id,
            )
        )

    def _set_blocked_status(
        self, business_date: date, now: datetime, stage: AutomationStage, detail: str
    ) -> None:
        state = (
            PipelineState.STALE_DATA
            if stage
            in {
                AutomationStage.DATA_SYNC,
                AutomationStage.CANDIDATE_SELECTION,
                AutomationStage.MORNING_CAPTURE,
                AutomationStage.FREEZE_1130,
            }
            else PipelineState.MODEL_ERROR
        )
        previous = self.store.daily_status(business_date)
        self.store.set_daily_status(
            DailyOperationStatus(
                business_date=business_date,
                pipeline_state=state,
                updated_at=now,
                data_as_of=previous.data_as_of if previous is not None else None,
                morning_data_as_of=(previous.morning_data_as_of if previous is not None else None),
                proposal_id=previous.proposal_id if previous is not None else None,
                blocking_reason=detail,
                is_stale=state is PipelineState.STALE_DATA,
            )
        )

    def _set_failed_status(
        self, business_date: date, now: datetime, stage: AutomationStage, detail: str
    ) -> None:
        state = (
            PipelineState.DATA_ERROR
            if stage
            in {
                AutomationStage.DATA_SYNC,
                AutomationStage.CANDIDATE_SELECTION,
                AutomationStage.MORNING_CAPTURE,
                AutomationStage.FREEZE_1130,
            }
            else PipelineState.MODEL_ERROR
        )
        previous = self.store.daily_status(business_date)
        self.store.set_daily_status(
            DailyOperationStatus(
                business_date=business_date,
                pipeline_state=state,
                updated_at=now,
                data_as_of=previous.data_as_of if previous is not None else None,
                morning_data_as_of=(previous.morning_data_as_of if previous is not None else None),
                proposal_id=previous.proposal_id if previous is not None else None,
                blocking_reason=detail,
            )
        )


def windows_task_scheduler_script(
    *,
    executable: str,
    database_path: str,
    task_name: str = "StockAIDecisionSupport",
) -> str:
    """Return an explicit user-run registration script; never execute it here."""
    escaped_executable = executable.replace("'", "''")
    escaped_database = database_path.replace("'", "''")
    return "\n".join(
        (
            "$ErrorActionPreference = 'Stop'",
            f'$taskName = "{task_name}"',
            f"$executable = '{escaped_executable}'",
            f"$database = '{escaped_database}'",
            "$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable",
            "$stages = @(",
            "  @{ Suffix='DataSync'; Time='01:30'; Stage='data_sync' },",
            "  @{ Suffix='Candidates'; Time='08:30'; Stage='candidate_selection' },",
            "  @{ Suffix='Morning'; Time='09:00'; Stage='morning_capture' },",
            "  @{ Suffix='Freeze'; Time='11:30'; Stage='freeze_1130' },",
            "  @{ Suffix='Prediction'; Time='11:31'; Stage='prediction' },",
            "  @{ Suffix='Proposal'; Time='11:34'; Stage='proposal' },",
            "  @{ Suffix='Notification'; Time='11:36'; Stage='notification' },",
            "  @{ Suffix='EOD'; Time='16:00'; Stage='eod_update' }",
            ")",
            "foreach ($item in $stages) {",
            "  $arguments = 'ops run-daily --database \"' + $database + "
            "'\" --business-date today --stage ' + $item.Stage",
            "  $action = New-ScheduledTaskAction -Execute $executable -Argument $arguments",
            "  $trigger = New-ScheduledTaskTrigger -Daily -At $item.Time",
            "  Register-ScheduledTask -Force -TaskName ($taskName + '-' + $item.Suffix) "
            "-Action $action -Trigger $trigger -Settings $settings",
            "}",
            "$monthly = '\"' + $executable + '\" ops run-daily --database \"' + "
            "$database + '\" --business-date today --stage challenger_training'",
            "schtasks.exe /Create /F /TN ($taskName + '-Challenger') /SC MONTHLY "
            "/D 1 /ST 02:30 /TR $monthly",
            "# These tasks update decision-support records only; they cannot submit orders.",
        )
    )
