"""Local-only FastAPI surface for the mobile PWA.

The mutation surface records user intent and actual fills.  It deliberately has
no broker connection, order submission, order cancellation, or order amendment
endpoint.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from stock_ai.domain import (
    ExecutionRecord,
    ExecutionStatus,
    PortfolioProposal,
    UserDecision,
    UserDecisionLine,
)
from stock_ai.operations.models import (
    DailyOperationStatus,
    PaperReadoutPeriod,
    PipelineState,
)
from stock_ai.operations.store import (
    OperationalConflictError,
    OperationalIntegrityError,
    OperationalStore,
)

JST = ZoneInfo("Asia/Tokyo")
MANUAL_INTENT_HEADER = "manual-record"
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
PROPOSAL_DISPLAY_STATES = frozenset(
    {
        PipelineState.PROPOSAL_READY,
        PipelineState.USER_DECISION_SAVED,
        PipelineState.EXECUTION_PENDING,
        PipelineState.EXECUTION_RECORDED,
        PipelineState.MARKET_CLOSED,
    }
)
PROPOSAL_ACTION_STATES = frozenset(
    {PipelineState.PROPOSAL_READY, PipelineState.USER_DECISION_SAVED}
)


def _camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.title() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=_camel,
        populate_by_name=True,
        extra="forbid",
        str_strip_whitespace=True,
    )


class DecisionLineInput(ApiModel):
    proposal_line_id: str = Field(min_length=1)
    selected_target_shares: int = Field(ge=0)
    note: str | None = None


class DecisionReviewInput(ApiModel):
    selected_targets: Mapping[str, int]


class SaveDecisionInput(ApiModel):
    decision_id: str = Field(default_factory=lambda: f"decision-{uuid4()}")
    version: int = Field(ge=1)
    saved_at: datetime
    lines: tuple[DecisionLineInput, ...]
    confirms_manual_order_only: bool = True


class ExecutionInput(ApiModel):
    execution_id: str = Field(default_factory=lambda: f"execution-{uuid4()}")
    executed_at: datetime
    symbol: str = Field(min_length=1)
    account_bucket_id: str = Field(min_length=1)
    status: str = Field(min_length=1)
    side: str = Field(min_length=1)
    ordered_shares: int = Field(ge=0)
    filled_shares: int = Field(ge=0)
    average_fill_price: Decimal | None = Field(default=None, ge=0)
    actual_commission: Decimal = Field(default=Decimal("0"), ge=0)
    actual_other_cost: Decimal = Field(default=Decimal("0"), ge=0)
    tax_withheld: Decimal = Field(default=Decimal("0"), ge=0)
    confirm_difference: bool = False


class ApplyExecutionsInput(ApiModel):
    next_as_of: datetime
    next_portfolio_id: str = Field(min_length=1)


class CsvPreviewInput(ApiModel):
    csv_text: str = Field(min_length=1)
    preview_id: str = Field(default_factory=lambda: f"import-{uuid4()}")


class ConfirmImportInput(ApiModel):
    accepted_conflict_ids: tuple[str, ...] = ()


class ConfirmPositionReconciliationInput(ApiModel):
    next_portfolio_id: str = Field(min_length=1)
    confirm_all_differences: bool = False


def _require_manual_intent(
    intent: Annotated[str | None, Header(alias="X-Stock-AI-Intent")] = None,
) -> None:
    if intent != MANUAL_INTENT_HEADER:
        raise HTTPException(
            status_code=428,
            detail="manual record confirmation header is required; this does not submit an order",
        )


def _line_view(line: Any) -> dict[str, object]:
    return {
        "lineId": line.line_id,
        "symbol": line.symbol,
        "companyName": line.company_name,
        "accountBucketId": line.account_bucket_id,
        "currentShares": line.current_shares,
        "recommendedShares": line.recommended_shares,
        "shareDifference": line.share_difference,
        "action": line.action.value,
        "referencePrice": str(line.reference_price),
        "currentMarketValue": str(line.current_market_value),
        "recommendedMarketValue": str(line.recommended_market_value),
        "estimatedRequiredOrReleasedCash": str(line.estimated_cash_required_or_released),
        "holdExpectedValue": str(line.hold_expected_value),
        "proposedExpectedValue": str(line.proposed_expected_value),
        "estimatedTransactionCost": str(line.transaction_cost.total),
        "estimatedTaxEffect": str(line.estimated_tax_effect),
        "netExpectedImprovement": str(line.net_expected_improvement),
        "downsideLevel": line.downside_risk,
        "uncertaintyLevel": line.uncertainty,
        "positiveReasons": list(line.human_readable_reasons[:3]),
        "negativeReasons": list(line.human_readable_reasons[3:6]),
    }


def _proposal_view(proposal: PortfolioProposal) -> dict[str, object]:
    lines = [_line_view(line) for line in proposal.lines]
    changed = [line for line in proposal.lines if line.share_difference != 0]
    estimated_sell = sum(
        (
            -line.estimated_cash_required_or_released
            for line in proposal.lines
            if line.share_difference < 0
        ),
        Decimal("0"),
    )
    estimated_buy = sum(
        (
            line.estimated_cash_required_or_released
            for line in proposal.lines
            if line.share_difference > 0
        ),
        Decimal("0"),
    )
    return {
        "proposalId": proposal.proposal_id,
        "asOf": proposal.as_of.isoformat(),
        "generatedAt": proposal.generated_at.isoformat(),
        "modelBundleVersion": proposal.model_bundle_version,
        "decisionEngineVersion": proposal.decision_engine_version,
        "status": "RESEARCH_ONLY" if proposal.is_research_only else "READY",
        "isResearchOnly": proposal.is_research_only,
        "isOrderInstruction": False,
        "lines": lines,
        "estimatedSellValue": str(estimated_sell),
        "estimatedBuyValue": str(estimated_buy),
        "estimatedTransactionCost": str(
            sum((line.transaction_cost.total for line in proposal.lines), Decimal("0"))
        ),
        "estimatedTaxEffect": str(
            sum((line.estimated_tax_effect for line in proposal.lines), Decimal("0"))
        ),
        "estimatedCashAfter": {
            key: str(value) for key, value in proposal.estimated_cash_after.items()
        },
        "changeCount": len(changed),
        "holdExpectedValue": str(proposal.hold_utility),
        "proposedExpectedValue": str(proposal.proposed_utility),
        "netImprovement": str(proposal.net_improvement),
        "noTradeReason": proposal.no_trade_reason,
    }


def _active_proposal(
    store: OperationalStore, business_date: date, *, actionable: bool = False
) -> PortfolioProposal | None:
    status = store.daily_status(business_date)
    if status is None or status.is_stale:
        return None
    allowed = PROPOSAL_ACTION_STATES if actionable else PROPOSAL_DISPLAY_STATES
    if status.pipeline_state not in allowed:
        return None
    if status.proposal_id is None:
        raise OperationalIntegrityError("active daily status has no proposal identity")
    try:
        proposal = store.proposal(status.proposal_id)
    except KeyError as exc:
        raise OperationalIntegrityError("daily status references a missing proposal") from exc
    if proposal.as_of.astimezone(JST).date() != business_date:
        raise OperationalIntegrityError("daily status proposal belongs to another business date")
    return proposal


def _require_active_proposal(
    store: OperationalStore, proposal_id: str, *, actionable: bool
) -> PortfolioProposal:
    try:
        proposal = store.proposal(proposal_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="proposal not found") from exc
    business_date = proposal.as_of.astimezone(JST).date()
    active = _active_proposal(store, business_date, actionable=actionable)
    if active is None or active.proposal_id != proposal_id:
        raise OperationalConflictError("proposal is not active for its business date")
    return proposal


def _today(store: OperationalStore, business_date: date) -> dict[str, object]:
    status = store.daily_status(business_date)
    proposal = _active_proposal(store, business_date)
    if status is None:
        state = PipelineState.PRE_MARKET
        blocking_reason = None
        is_stale = False
    else:
        state = status.pipeline_state
        blocking_reason = status.blocking_reason
        is_stale = status.is_stale
    return {
        "businessDate": business_date.isoformat(),
        "pipelineState": state.value,
        "isStale": is_stale,
        "blockingReason": blocking_reason,
        "runtimeMode": store.metadata("runtime_mode") or "UNCONFIGURED",
        "proposal": (_proposal_view(proposal) if proposal is not None else None),
    }


def create_app(
    database_path: Path,
    *,
    static_dir: Path | None = None,
    clock: Callable[[], datetime] | None = None,
) -> FastAPI:
    store = OperationalStore(database_path)

    def current_time() -> datetime:
        value = datetime.now(JST) if clock is None else clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RuntimeError("operational API clock must be timezone-aware")
        return value.astimezone(JST)

    app = FastAPI(
        title="Stock AI Decision Support",
        version="0.1.0",
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        description="Local decision-support API. It cannot submit securities orders.",
    )
    app.state.store = store

    @app.middleware("http")
    async def security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        host = request.url.hostname
        if host is None or host.lower() not in LOCAL_HOSTS:
            return JSONResponse(status_code=421, content={"detail": "local host required"})
        origin = request.headers.get("origin")
        if origin is not None and request.method not in {"GET", "HEAD", "OPTIONS"}:
            parsed_origin = urlsplit(origin)
            if (
                parsed_origin.scheme not in {"http", "https"}
                or parsed_origin.hostname is None
                or parsed_origin.hostname.lower() not in LOCAL_HOSTS
            ):
                return JSONResponse(status_code=403, content={"detail": "local origin required"})
        if request.url.path.startswith("/api/"):
            try:
                store.verify_integrity()
            except OperationalIntegrityError:
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "operational data integrity check failed; proposal display stopped"
                        )
                    },
                )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(OperationalConflictError)
    async def conflict_handler(_request: Request, exc: OperationalConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(OperationalIntegrityError)
    async def integrity_handler(_request: Request, _exc: OperationalIntegrityError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"detail": "operational data integrity check failed; proposal display stopped"},
        )

    @app.get("/api/v1/health")
    def health() -> dict[str, object]:
        counts = store.verify_integrity()
        return {
            "status": "ok",
            "orderSubmissionAvailable": False,
            "runtimeMode": store.metadata("runtime_mode") or "UNCONFIGURED",
            "records": sum(counts.values()),
        }

    @app.get("/api/v1/status")
    def status(
        business_date: Annotated[date | None, Query(alias="businessDate")] = None,
    ) -> dict[str, object]:
        selected = business_date or current_time().date()
        payload = _today(store, selected)
        proposal = _active_proposal(store, selected) if payload["proposal"] is not None else None
        portfolio = store.latest_portfolio()
        daily = store.daily_status(selected)
        return {
            "businessDate": selected.isoformat(),
            "marketState": "UNKNOWN",
            "pipelineState": payload["pipelineState"],
            "dataAsOf": None if daily is None or daily.data_as_of is None else daily.data_as_of,
            "morningDataAsOf": (
                None
                if daily is None or daily.morning_data_as_of is None
                else daily.morning_data_as_of
            ),
            "proposalGeneratedAt": None if proposal is None else proposal.generated_at,
            "portfolioUpdatedAt": None if portfolio is None else portfolio.as_of,
            "modelBundleVersion": None if proposal is None else proposal.model_bundle_version,
            "isStale": payload["isStale"],
            "blockingReason": payload["blockingReason"],
            "orderSubmissionAvailable": False,
        }

    @app.get("/api/v1/home")
    def home(
        business_date: Annotated[date | None, Query(alias="businessDate")] = None,
    ) -> dict[str, object]:
        selected = business_date or current_time().date()
        portfolio = store.latest_portfolio()
        if portfolio is None:
            return {"portfolio": None, "holdings": [], "blockingReason": "保有未登録"}
        proposal = _active_proposal(store, selected)
        proposal_lines = (
            {(line.symbol, line.account_bucket_id): line for line in proposal.lines}
            if proposal is not None
            else {}
        )
        holdings = []
        for position in portfolio.positions:
            line = proposal_lines.get(position.key)
            holdings.append(
                {
                    "symbol": position.symbol,
                    "companyName": line.company_name if line is not None else position.symbol,
                    "accountBucketId": position.account_bucket_id,
                    "shares": position.shares,
                    "averageAcquisitionPrice": str(position.average_acquisition_price),
                    "currentPrice": str(position.market_price),
                    "marketValue": str(position.market_value),
                    "unrealizedPnlAmount": str(position.market_value - position.book_value),
                    "latestAction": None if line is None else line.action.value,
                    "recommendedShares": None if line is None else line.recommended_shares,
                    "proposalId": None if proposal is None else proposal.proposal_id,
                }
            )
        cash_value = sum((item.available_cash for item in portfolio.cash), Decimal("0"))
        equity_value = sum((item.market_value for item in portfolio.positions), Decimal("0"))
        return {
            "runtimeMode": store.metadata("runtime_mode") or "UNCONFIGURED",
            "portfolio": {
                "asOf": portfolio.as_of,
                "totalAssets": str(cash_value + equity_value),
                "cashValue": str(cash_value),
                "equityValue": str(equity_value),
                "holdingsCount": len(portfolio.positions),
            },
            "holdings": holdings,
            "blockingReason": None,
        }

    @app.get("/api/v1/today")
    def today(
        business_date: Annotated[date | None, Query(alias="businessDate")] = None,
    ) -> dict[str, object]:
        return _today(store, business_date or current_time().date())

    @app.get("/api/v1/proposals/{proposal_id}")
    def get_proposal(proposal_id: str) -> dict[str, object]:
        return _proposal_view(_require_active_proposal(store, proposal_id, actionable=False))

    @app.post("/api/v1/proposals/{proposal_id}/review")
    def review_decision(
        proposal_id: str,
        body: DecisionReviewInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        _require_active_proposal(store, proposal_id, actionable=True)
        return jsonable_encoder(
            store.review_decision(proposal_id, body.selected_targets), by_alias=True
        )

    @app.post("/api/v1/proposals/{proposal_id}/decisions", status_code=201)
    def save_decision(
        proposal_id: str,
        body: SaveDecisionInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        proposal = _require_active_proposal(store, proposal_id, actionable=True)
        business_date = proposal.as_of.astimezone(JST).date()
        if body.saved_at.astimezone(JST).date() != business_date:
            raise OperationalConflictError(
                "decision timestamp must match the proposal business date"
            )
        decision = UserDecision(
            decision_id=body.decision_id,
            proposal_id=proposal_id,
            version=body.version,
            saved_at=body.saved_at,
            lines=tuple(
                UserDecisionLine(
                    proposal_line_id=line.proposal_line_id,
                    selected_target_shares=line.selected_target_shares,
                    note=line.note,
                )
                for line in body.lines
            ),
            confirms_manual_order_only=body.confirms_manual_order_only,
        )
        store.save_decision(decision)
        previous = store.daily_status(business_date)
        store.set_daily_status(
            DailyOperationStatus(
                business_date=business_date,
                pipeline_state=PipelineState.USER_DECISION_SAVED,
                updated_at=decision.saved_at,
                data_as_of=None if previous is None else previous.data_as_of,
                morning_data_as_of=None if previous is None else previous.morning_data_as_of,
                proposal_id=proposal_id,
            )
        )
        return jsonable_encoder(decision, by_alias=True)

    @app.get("/api/v1/proposals/{proposal_id}/decision")
    def latest_decision(proposal_id: str) -> object:
        _require_active_proposal(store, proposal_id, actionable=False)
        decision = store.latest_decision(proposal_id)
        return None if decision is None else jsonable_encoder(decision, by_alias=True)

    @app.post("/api/v1/decisions/{decision_id}/executions", status_code=201)
    def record_execution(
        decision_id: str,
        body: ExecutionInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        execution = ExecutionRecord.model_validate(
            {
                **body.model_dump(exclude={"confirm_difference"}),
                "decision_id": decision_id,
                "source": "manual",
            }
        )
        store.append_execution(execution, confirm_difference=body.confirm_difference)
        proposal = store.proposal(store.decision(decision_id).proposal_id)
        execution_state = (
            PipelineState.EXECUTION_PENDING
            if execution.status
            in {
                ExecutionStatus.NOT_ORDERED,
                ExecutionStatus.OPEN,
                ExecutionStatus.PARTIALLY_FILLED,
            }
            else PipelineState.EXECUTION_RECORDED
        )
        previous = store.daily_status(proposal.as_of.astimezone(JST).date())
        store.set_daily_status(
            DailyOperationStatus(
                business_date=proposal.as_of.astimezone(JST).date(),
                pipeline_state=execution_state,
                updated_at=execution.executed_at,
                data_as_of=None if previous is None else previous.data_as_of,
                morning_data_as_of=None if previous is None else previous.morning_data_as_of,
                proposal_id=proposal.proposal_id,
            )
        )
        return jsonable_encoder(execution, by_alias=True)

    @app.get("/api/v1/decisions/{decision_id}/executions")
    def executions(decision_id: str) -> object:
        return jsonable_encoder(store.executions(decision_id=decision_id), by_alias=True)

    @app.post("/api/v1/portfolio/apply-executions")
    def apply_recorded_executions(
        body: ApplyExecutionsInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        state = store.apply_unapplied_executions(
            next_as_of=body.next_as_of,
            next_portfolio_id=body.next_portfolio_id,
            created_at=current_time(),
        )
        return jsonable_encoder(state, by_alias=True)

    @app.post("/api/v1/imports/executions/preview")
    def preview_import(
        body: CsvPreviewInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        preview = store.preview_execution_csv(
            body.csv_text,
            preview_id=body.preview_id,
            created_at=current_time(),
        )
        return jsonable_encoder(preview, by_alias=True)

    @app.post("/api/v1/imports/executions/{preview_id}/confirm")
    def confirm_import(
        preview_id: str,
        body: ConfirmImportInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> dict[str, int]:
        imported, skipped = store.confirm_execution_import(
            preview_id, accepted_conflict_ids=body.accepted_conflict_ids
        )
        return {"imported": imported, "skipped": skipped}

    @app.post("/api/v1/imports/positions/preview")
    def preview_positions(
        body: CsvPreviewInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        preview = store.preview_position_reconciliation_csv(
            body.csv_text,
            preview_id=body.preview_id,
            created_at=current_time(),
        )
        return jsonable_encoder(preview, by_alias=True)

    @app.post("/api/v1/imports/positions/{preview_id}/confirm")
    def confirm_positions(
        preview_id: str,
        body: ConfirmPositionReconciliationInput,
        _intent: Annotated[None, Depends(_require_manual_intent)],
    ) -> object:
        state = store.confirm_position_reconciliation(
            preview_id,
            next_portfolio_id=body.next_portfolio_id,
            confirm_all_differences=body.confirm_all_differences,
            created_at=current_time(),
        )
        return jsonable_encoder(state, by_alias=True)

    @app.get("/api/v1/ranking")
    def ranking(
        rank_type: Annotated[str, Query(alias="rankType")] = "overall",
        limit: Annotated[int, Query(ge=1, le=50)] = 20,
    ) -> object:
        records = store.latest_rankings(rank_type, limit=limit)
        return [
            {
                **jsonable_encoder(item, by_alias=True),
                "percentile": item.rank / item.total_universe,
            }
            for item in records
        ]

    @app.get("/api/v1/stocks/{symbol}")
    def stock_detail(
        symbol: str,
        business_date: Annotated[date | None, Query(alias="businessDate")] = None,
    ) -> dict[str, object]:
        portfolio = store.latest_portfolio()
        proposal_date = business_date or current_time().date()
        today_proposal = _active_proposal(store, proposal_date)
        positions = (
            []
            if portfolio is None
            else [
                jsonable_encoder(item, by_alias=True)
                for item in portfolio.positions
                if item.symbol == symbol
            ]
        )
        lines = (
            []
            if today_proposal is None
            else [_line_view(item) for item in today_proposal.lines if item.symbol == symbol]
        )
        if not positions and not lines:
            raise HTTPException(status_code=404, detail="symbol not found")
        return {
            "symbol": symbol,
            "positions": positions,
            "proposalLines": lines,
            "proposal": None if today_proposal is None else _proposal_view(today_proposal),
        }

    @app.get("/api/v1/validation")
    def validation(
        mode: Annotated[str, Query(pattern="^(paper|live|historical)$")] = "paper",
    ) -> object:
        if mode == "paper":
            summary = jsonable_encoder(store.paper_summary(), by_alias=True)
            summary["weekly_readouts"] = jsonable_encoder(
                store.paper_readouts(PaperReadoutPeriod.WEEKLY), by_alias=True
            )
            summary["monthly_readouts"] = jsonable_encoder(
                store.paper_readouts(PaperReadoutPeriod.MONTHLY), by_alias=True
            )
            summary["series"] = jsonable_encoder(store.paper_series(), by_alias=True)
            summary["horizon_sessions"] = 1
            return summary
        if mode == "live":
            executions_count = len(store.executions())
            return {
                "mode": "live",
                "observations": executions_count,
                "blockingReason": (None if executions_count else "実運用データはまだありません"),
            }
        return {
            "mode": "historical",
            "blockingReason": "認証済み研究reportを運用台帳へ登録してください",
        }

    @app.get("/api/v1/notifications")
    def notifications() -> object:
        return jsonable_encoder(store.notifications(), by_alias=True)

    @app.get("/api/v1/settings")
    def settings() -> dict[str, object]:
        portfolio = store.latest_portfolio()
        proposal = _active_proposal(store, current_time().date())
        policy_snapshot = (
            None
            if proposal is None
            else store.decision_policy_snapshot(proposal.proposal_id)
        )
        decision_config = None if policy_snapshot is None else policy_snapshot.config
        tax_states = {} if portfolio is None else portfolio.tax_state_map()
        return {
            "runtime": {"remoteAccess": "LOCALHOST_ONLY", "orderSubmission": "OUT_OF_SCOPE"},
            "runtimeMode": store.metadata("runtime_mode") or "UNCONFIGURED",
            "data": {
                "jQuantsApiKeyConfigured": bool(os.environ.get("JQUANTS_API_KEY")),
                "credentialValueExposed": False,
            },
            "capital": {
                "availableCash": None
                if portfolio is None
                else str(sum((item.available_cash for item in portfolio.cash), Decimal("0"))),
                "reservedCash": None
                if portfolio is None
                else str(sum((item.reserved_cash for item in portfolio.cash), Decimal("0"))),
                "asOf": None if portfolio is None else portfolio.as_of,
                "minimumCashRatio": (
                    None
                    if decision_config is None
                    else str(decision_config.minimum_cash_ratio)
                ),
                "dailyProposalLimit": None,
            },
            "accounts": []
            if portfolio is None
            else [
                {
                    "bucketId": item.bucket_id,
                    "accountId": item.account_id,
                    "accountType": item.account_type.value,
                    "withholdingMode": item.withholding_mode.value,
                    "feePolicyId": item.fee_policy_id,
                    "taxPolicyId": item.tax_policy_id,
                    "taxYear": tax_states[item.bucket_id].tax_year,
                    "realizedGainYtd": str(tax_states[item.bucket_id].realized_gain_ytd),
                    "realizedLossYtd": str(tax_states[item.bucket_id].realized_loss_ytd),
                    "lossCarryforward": str(
                        tax_states[item.bucket_id].loss_carryforward_user_input
                    ),
                    "nisaAnnualCapacity": (
                        None
                        if tax_states[item.bucket_id].nisa_annual_capacity_user_input is None
                        else str(tax_states[item.bucket_id].nisa_annual_capacity_user_input)
                    ),
                    "nisaLifetimeCapacity": (
                        None
                        if tax_states[item.bucket_id].nisa_lifetime_capacity_user_input is None
                        else str(tax_states[item.bucket_id].nisa_lifetime_capacity_user_input)
                    ),
                }
                for item in portfolio.account_buckets
            ],
            "decisionPolicies": {
                "decisionEngineVersion": None
                if proposal is None
                else proposal.decision_engine_version,
                "costPolicyId": None if proposal is None else proposal.cost_policy_id,
                "costPolicyVersion": None if proposal is None else proposal.cost_policy_version,
                "taxPolicyId": None if proposal is None else proposal.tax_policy_id,
                "taxPolicyVersion": None if proposal is None else proposal.tax_policy_version,
                "roundLotShares": 100 if decision_config is None else decision_config.lot_size,
                "maximumPositions": (
                    None if decision_config is None else decision_config.maximum_positions
                ),
                "maximumSymbolWeight": (
                    None
                    if decision_config is None
                    else str(decision_config.maximum_symbol_weight)
                ),
                "maximumSectorWeight": (
                    None
                    if decision_config is None
                    else str(decision_config.maximum_sector_weight)
                ),
                "maximumTurnoverRatio": (
                    None
                    if decision_config is None
                    else str(decision_config.maximum_turnover_ratio)
                ),
                "maximumTradeAdvRatio": (
                    None
                    if decision_config is None
                    else str(decision_config.maximum_trade_adv_ratio)
                ),
                "minimumImprovementYen": (
                    None
                    if decision_config is None
                    else str(decision_config.minimum_improvement_yen)
                ),
                "uncertaintyBufferYen": (
                    None
                    if decision_config is None
                    else str(decision_config.uncertainty_buffer_yen)
                ),
            },
            "notifications": {
                "inApp": "AVAILABLE",
                "webPush": "BLOCKED_BY_CONFIGURATION",
            },
            "model": {
                "morningChampion": "BLOCKED_BY_DATA_CAPABILITY",
                "trainedAt": None,
                "trainingDataEnd": None,
                "validationStatus": "BLOCKED_BY_DATA_CAPABILITY",
                "automaticPromotion": False,
            },
            "decision": {
                "freezeTime": "11:30 JST",
                "recommendedTradeTime": "12:30 JST 後場寄り",
                "method": "Daily Portfolio Decision Engine",
            },
        }

    if static_dir is not None and static_dir.resolve().is_dir():
        resolved_static = static_dir.resolve()
        assets = resolved_static / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def pwa_fallback(full_path: str) -> FileResponse:
            requested = (resolved_static / full_path).resolve()
            if requested.is_relative_to(resolved_static) and requested.is_file():
                return FileResponse(requested)
            return FileResponse(resolved_static / "index.html")

    return app
