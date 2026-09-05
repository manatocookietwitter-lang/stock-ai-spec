"""One-shot, resumable locked-holdout evaluation after development choices are frozen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import IO, Any, Literal, Self, cast

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from stock_ai.features import V2_EXTENDED_MANIFEST
from stock_ai.ml.advanced import (
    AdvancedModelMetrics,
    fit_predict_frozen_model,
    summarize_oof_predictions,
)
from stock_ai.ml.campaign import load_campaign_build_id
from stock_ai.ml.production import load_production_dataset_snapshot
from stock_ai.ml.research_metrics import (
    evaluate_cross_sectional_predictions,
    within_date_rank_standardize,
)
from stock_ai.ml.selection import (
    DevelopmentSelectionArtifact,
    FrozenModelComponent,
    HorizonDevelopmentSelection,
    load_development_selection,
)
from stock_ai.ml.validation import reserve_locked_final_holdout

_LEDGER_SCHEMA = "locked-holdout-ledger-v1"
_REPORT_SCHEMA = "locked-holdout-report-v1"


class HoldoutComponentStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"


class HoldoutEvaluationStatus(StrEnum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"


class HoldoutComponentCheckpoint(BaseModel):
    """Mutable execution state; successful artifacts never contain target values."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    component_id: str = Field(min_length=64, max_length=64)
    component_key: str = Field(min_length=1)
    horizon: int
    component_name: str = Field(min_length=1)
    status: HoldoutComponentStatus = HoldoutComponentStatus.PENDING
    attempts: int = Field(default=0, ge=0)
    prediction_path: str | None = None
    parquet_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    frame_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    rows: int | None = Field(default=None, ge=1)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_code: str | None = None


class LockedHoldoutLedger(BaseModel):
    """Durable proof that holdout access began only after an immutable selection."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal["locked-holdout-ledger-v1"] = "locked-holdout-ledger-v1"
    ledger_id: str = Field(min_length=64, max_length=64)
    selection_id: str = Field(min_length=64, max_length=64)
    build_id: str = Field(min_length=64, max_length=64)
    data_snapshot_id: str = Field(min_length=64, max_length=64)
    evaluator_code_commit: str = Field(min_length=1)
    locked_holdout_start: str
    holdout_accessed: Literal[True] = True
    status: HoldoutEvaluationStatus = HoldoutEvaluationStatus.RUNNING
    started_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    report_path: str | None = None
    components: list[HoldoutComponentCheckpoint]

    @model_validator(mode="after")
    def coherent_identity(self) -> Self:
        if self.ledger_id != _stable_hash(_ledger_identity(self)):
            raise ValueError("locked holdout ledger identity mismatch")
        if len({item.component_id for item in self.components}) != len(self.components):
            raise ValueError("locked holdout component identities must be unique")
        if self.status is HoldoutEvaluationStatus.COMPLETED:
            if self.report_path is None or self.completed_at is None:
                raise ValueError("completed holdout ledger requires its immutable report")
            if any(
                item.status is not HoldoutComponentStatus.SUCCEEDED
                for item in self.components
            ):
                raise ValueError("completed holdout ledger has unfinished components")
        return self


class HoldoutComponentResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    component_key: str = Field(min_length=1)
    component: FrozenModelComponent
    metrics: AdvancedModelMetrics
    prediction_sha256: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def coherent_result(self) -> Self:
        if self.component.horizon != self.metrics.horizon:
            raise ValueError("holdout component result horizon mismatch")
        if (
            self.component.model_family != self.metrics.model_family
            or self.component.task != self.metrics.task
            or self.component.seed != self.metrics.seed
        ):
            raise ValueError("holdout component result model identity mismatch")
        return self


class HoldoutEnsembleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    adopted_on_development: bool
    component_names: tuple[str, ...]
    weights: tuple[float, ...]
    rows: int = Field(ge=1)
    dates: int = Field(ge=1)
    mean_squared_error_rank_space: float = Field(ge=0)
    mean_daily_rank_ic: float | None
    rank_icir: float | None
    ndcg_at_5: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_10: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_20: float | None = Field(default=None, ge=0, le=1)
    empirical_coverage_80: float = Field(ge=0, le=1)
    empirical_coverage_90: float = Field(ge=0, le=1)
    disagreement_error_correlation: float | None

    @model_validator(mode="after")
    def valid_simplex(self) -> Self:
        if len(self.component_names) != len(self.weights) or not self.weights:
            raise ValueError("holdout ensemble names and weights must align")
        if any(value < -1e-10 for value in self.weights) or not math.isclose(
            sum(self.weights), 1.0, abs_tol=1e-7
        ):
            raise ValueError("holdout ensemble weights must be a non-negative simplex")
        return self


class LockedHoldoutReport(BaseModel):
    """Immutable result of the sole post-selection final evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["locked-holdout-report-v1"] = "locked-holdout-report-v1"
    report_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    selection_id: str = Field(min_length=64, max_length=64)
    ledger_id: str = Field(min_length=64, max_length=64)
    build_id: str = Field(min_length=64, max_length=64)
    data_snapshot_id: str = Field(min_length=64, max_length=64)
    evaluator_code_commit: str = Field(min_length=1)
    locked_holdout_start: str
    locked_holdout_end: str
    locked_holdout_accessed: Literal[True] = True
    feature_definition_hashes: Mapping[str, str]
    component_results: tuple[HoldoutComponentResult, ...]
    ensemble_results: tuple[HoldoutEnsembleResult, ...]
    selection_was_frozen_before_access: Literal[True] = True
    model_choices_changed_after_access: Literal[False] = False
    adoption_eligible: Literal[False] = False
    adoption_blocking_reasons: tuple[str, ...]

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("holdout report created_at must be timezone-aware")
        return value

    @field_validator("feature_definition_hashes", mode="after")
    @classmethod
    def freeze_feature_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("feature_definition_hashes")
    def serialize_feature_hashes(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def coherent_report(self) -> Self:
        if tuple(item.horizon for item in self.ensemble_results) != (1, 5, 20):
            raise ValueError("holdout report requires ordered 1d/5d/20d ensemble results")
        if not self.component_results or not self.adoption_blocking_reasons:
            raise ValueError("holdout report cannot claim live adoption")
        if not self.feature_definition_hashes or any(
            not name or not value for name, value in self.feature_definition_hashes.items()
        ):
            raise ValueError("holdout report requires selected feature definition hashes")
        if self.report_id != _stable_hash(_report_identity(self)):
            raise ValueError("locked holdout report content identity mismatch")
        return self


@dataclass(frozen=True)
class LockedHoldoutEvaluation:
    report: LockedHoldoutReport
    report_path: Path
    resumed: bool


@dataclass(frozen=True)
class _ComponentPlan:
    component_id: str
    component_key: str
    horizon_selection: HorizonDevelopmentSelection
    component: FrozenModelComponent


class _ExclusiveHoldoutLock:
    """OS lock released automatically after a worker exits or the host restarts."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: IO[bytes] = path.open("a+b")
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"\0")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        self._stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = cast(Any, __import__("fcntl"))
                fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._stream.close()
            raise RuntimeError("another worker owns the locked holdout evaluation") from None
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = cast(Any, __import__("fcntl"))
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        self._stream.close()
        self._closed = True

    def __enter__(self) -> _ExclusiveHoldoutLock:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def evaluate_locked_holdout(
    *,
    selection_path: Path,
    build_manifest_path: Path,
    evaluation_root: Path,
    evaluator_code_commit: str,
    created_at: datetime | None = None,
) -> LockedHoldoutEvaluation:
    """Evaluate the frozen selection once, resuming prediction-only checkpoints safely."""

    if not evaluator_code_commit.strip() or evaluator_code_commit == "UNSET":
        raise ValueError("locked holdout evaluator code commit must be explicit")
    selection = load_development_selection(selection_path)
    build_id = load_campaign_build_id(build_manifest_path)
    if build_id != selection.build_id:
        raise ValueError("locked holdout build differs from frozen development selection")
    timestamp = created_at or datetime.now(UTC)
    evaluation_directory = (evaluation_root / selection.selection_id).resolve()
    ledger_path = evaluation_directory / "ledger.json"
    _authorize_single_evaluation(
        selection_path=selection_path,
        selection=selection,
        build_id=build_id,
        evaluator_code_commit=evaluator_code_commit,
        ledger_path=ledger_path,
        now=timestamp,
    )
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    plans = _component_plans(selection)
    with _ExclusiveHoldoutLock(evaluation_directory / ".evaluation.lock"):
        resumed = ledger_path.exists()
        ledger = (
            _load_ledger(ledger_path)
            if resumed
            else _new_ledger(
                selection,
                build_id=build_id,
                evaluator_code_commit=evaluator_code_commit,
                plans=plans,
                now=timestamp,
            )
        )
        _require_ledger_plan(
            ledger,
            selection=selection,
            build_id=build_id,
            evaluator_code_commit=evaluator_code_commit,
            plans=plans,
        )
        if ledger.status is HoldoutEvaluationStatus.COMPLETED:
            assert ledger.report_path is not None
            _authenticate_completed_predictions(ledger)
            report = load_locked_holdout_report(Path(ledger.report_path))
            _require_completed_report(
                report,
                ledger=ledger,
                selection=selection,
                build_id=build_id,
            )
            return LockedHoldoutEvaluation(
                report=report,
                report_path=Path(ledger.report_path),
                resumed=True,
            )
        for state in ledger.components:
            if state.status is HoldoutComponentStatus.RUNNING:
                state.status = HoldoutComponentStatus.INTERRUPTED
                state.failure_code = "WORKER_EXITED_BEFORE_PREDICTION_PUBLICATION"
        _write_ledger(ledger, ledger_path)
        dataset_path = _dataset_path_from_build_marker(build_manifest_path)
        snapshot, dataset = load_production_dataset_snapshot(dataset_path)
        if snapshot.snapshot_id != selection.data_snapshot_id:
            raise RuntimeError("locked holdout dataset differs from frozen selection")
        holdout_frames = _holdout_frames(dataset, selection)
        state_by_id = {item.component_id: item for item in ledger.components}
        plan_by_id = {item.component_id: item for item in plans}
        for component_id in (item.component_id for item in plans):
            plan = plan_by_id[component_id]
            state = state_by_id[component_id]
            training, holdout = holdout_frames[plan.component.horizon]
            if state.status is HoldoutComponentStatus.SUCCEEDED:
                _load_prediction(state, expected=holdout)
                continue
            state.status = HoldoutComponentStatus.RUNNING
            state.attempts += 1
            state.started_at = datetime.now(UTC)
            state.completed_at = None
            state.failure_code = None
            _write_ledger(ledger, ledger_path)
            try:
                prediction = fit_predict_frozen_model(
                    training,
                    holdout,
                    feature_names=plan.horizon_selection.feature_names,
                    target_column=f"target_return_{plan.component.horizon}d",
                    horizon=plan.component.horizon,
                    family=plan.component.model_family,
                    task=plan.component.task,
                    seed=plan.component.seed,
                    parameters=plan.component.parameters,
                    config=plan.component.source_config,
                )
                artifact = holdout.loc[:, ["symbol", "trading_date"]].copy()
                artifact["prediction"] = prediction
                _publish_prediction(
                    state,
                    artifact,
                    evaluation_directory / "predictions",
                )
                state.status = HoldoutComponentStatus.SUCCEEDED
                state.completed_at = datetime.now(UTC)
                _write_ledger(ledger, ledger_path)
            except Exception as exc:
                state.status = HoldoutComponentStatus.FAILED
                state.completed_at = datetime.now(UTC)
                state.failure_code = type(exc).__name__
                _write_ledger(ledger, ledger_path)
                raise
        report = _build_holdout_report(
            selection,
            ledger=ledger,
            plans=plans,
            holdout_frames=holdout_frames,
            created_at=timestamp,
        )
        report_path = write_locked_holdout_report(report, evaluation_root / "reports")
        ledger.status = HoldoutEvaluationStatus.COMPLETED
        ledger.completed_at = datetime.now(UTC)
        ledger.report_path = str(report_path.resolve())
        _write_ledger(ledger, ledger_path)
        return LockedHoldoutEvaluation(report=report, report_path=report_path, resumed=resumed)


def read_locked_holdout_status(evaluation_directory: Path) -> Mapping[str, object]:
    """Read authenticated progress without changing ledger, process, or component state."""

    ledger = _load_ledger(evaluation_directory.resolve() / "ledger.json")
    return MappingProxyType(
        {
            "ledger_id": ledger.ledger_id,
            "selection_id": ledger.selection_id,
            "status": ledger.status.value,
            "holdout_accessed": True,
            "components": tuple(
                {
                    "component_key": item.component_key,
                    "status": item.status.value,
                    "attempts": item.attempts,
                }
                for item in ledger.components
            ),
        }
    )


def write_locked_holdout_report(report: LockedHoldoutReport, destination: Path) -> Path:
    """Atomically publish the content-addressed final evaluation report."""

    if report.report_id != _stable_hash(_report_identity(report)):
        raise RuntimeError("locked holdout report content identity mismatch")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_directory = destination / report.report_id
    final_path = final_directory / f"{report.report_id}.json"
    payload: dict[str, object] = {
        "report": report.model_dump(mode="json"),
        "report_path": str(final_path),
    }
    payload["metadata_hash"] = _stable_hash(payload)
    if final_directory.exists():
        observed = load_locked_holdout_report(final_path)
        if observed.report_id != report.report_id:
            raise RuntimeError("locked holdout report identity collision")
        return final_path
    temporary = Path(tempfile.mkdtemp(prefix=".holdout-report-", dir=destination))
    try:
        temporary_path = temporary / final_path.name
        _write_json(temporary_path, payload)
        os.replace(temporary, final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_path


def load_locked_holdout_report(path: Path) -> LockedHoldoutReport:
    path = path.resolve()
    report_id = path.name.removesuffix(".json")
    if path.parent.name != report_id:
        raise RuntimeError("locked holdout report path is not content-addressed")
    payload = _read_json(path, description="locked holdout report")
    _authenticate_envelope(payload, description="locked holdout report")
    if Path(str(payload.get("report_path", ""))).resolve() != path:
        raise RuntimeError("locked holdout report path metadata mismatch")
    report = LockedHoldoutReport.model_validate(payload.get("report"))
    if report.report_id != report_id:
        raise RuntimeError("locked holdout report directory identity mismatch")
    return report


def _new_ledger(
    selection: DevelopmentSelectionArtifact,
    *,
    build_id: str,
    evaluator_code_commit: str,
    plans: tuple[_ComponentPlan, ...],
    now: datetime,
) -> LockedHoldoutLedger:
    values: dict[str, object] = {
        "schema_version": _LEDGER_SCHEMA,
        "ledger_id": "0" * 64,
        "selection_id": selection.selection_id,
        "build_id": build_id,
        "data_snapshot_id": selection.data_snapshot_id,
        "evaluator_code_commit": evaluator_code_commit,
        "locked_holdout_start": selection.locked_holdout_start,
        "holdout_accessed": True,
        "status": HoldoutEvaluationStatus.RUNNING,
        "started_at": now,
        "updated_at": now,
        "components": [
            HoldoutComponentCheckpoint(
                component_id=plan.component_id,
                component_key=plan.component_key,
                horizon=plan.component.horizon,
                component_name=plan.component.component_name,
            )
            for plan in plans
        ],
    }
    provisional = LockedHoldoutLedger.model_construct(**cast(Any, values))
    values["ledger_id"] = _stable_hash(_ledger_identity(provisional))
    return LockedHoldoutLedger.model_validate(values)


def _authorize_single_evaluation(
    *,
    selection_path: Path,
    selection: DevelopmentSelectionArtifact,
    build_id: str,
    evaluator_code_commit: str,
    ledger_path: Path,
    now: datetime,
) -> None:
    """Bind one selection artifact to one ledger path before any holdout data is read."""

    authorization_path = selection_path.resolve().parent / "holdout-access.json"
    identity = {
        "schema_version": "locked-holdout-access-v1",
        "selection_id": selection.selection_id,
        "build_id": build_id,
        "data_snapshot_id": selection.data_snapshot_id,
        "evaluator_code_commit": evaluator_code_commit,
        "ledger_path": str(ledger_path.resolve()),
    }
    if authorization_path.exists():
        payload = _read_json(
            authorization_path,
            description="locked holdout access authorization",
        )
        _authenticate_envelope(payload, description="locked holdout access authorization")
        observed = {
            key: value
            for key, value in payload.items()
            if key not in {"created_at", "metadata_hash"}
        }
        if observed != identity:
            raise RuntimeError("locked holdout access authorization provenance mismatch")
        return
    authorization: dict[str, object] = {**identity, "created_at": now.isoformat()}
    authorization["metadata_hash"] = _stable_hash(authorization)
    encoded = (
        json.dumps(
            authorization,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    try:
        descriptor = os.open(
            authorization_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0),
            0o600,
        )
    except FileExistsError:
        # A concurrent contender won. Authenticate it rather than overwriting its choice.
        _authorize_single_evaluation(
            selection_path=selection_path,
            selection=selection,
            build_id=build_id,
            evaluator_code_commit=evaluator_code_commit,
            ledger_path=ledger_path,
            now=now,
        )
        return
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _component_plans(
    selection: DevelopmentSelectionArtifact,
) -> tuple[_ComponentPlan, ...]:
    plans: list[_ComponentPlan] = []
    for horizon in selection.horizons:
        components = {
            item.component_name: item
            for item in (
                horizon.expected_return_component,
                horizon.rank_component,
                horizon.downside_quantile_component,
                horizon.large_loss_component,
                *horizon.ensemble_components,
            )
        }
        for name in sorted(components):
            component = components[name]
            component_key = f"h{horizon.horizon}:{name}"
            component_id = _stable_hash(
                {
                    "selection_id": selection.selection_id,
                    "component_key": component_key,
                    "feature_names": horizon.feature_names,
                    "component": component.model_dump(mode="json"),
                }
            )
            plans.append(
                _ComponentPlan(
                    component_id=component_id,
                    component_key=component_key,
                    horizon_selection=horizon,
                    component=component,
                )
            )
    return tuple(plans)


def _require_ledger_plan(
    ledger: LockedHoldoutLedger,
    *,
    selection: DevelopmentSelectionArtifact,
    build_id: str,
    evaluator_code_commit: str,
    plans: tuple[_ComponentPlan, ...],
) -> None:
    if (
        ledger.selection_id != selection.selection_id
        or ledger.build_id != build_id
        or ledger.data_snapshot_id != selection.data_snapshot_id
        or ledger.evaluator_code_commit != evaluator_code_commit
        or ledger.locked_holdout_start != selection.locked_holdout_start
    ):
        raise RuntimeError("locked holdout ledger provenance mismatch")
    expected = tuple((item.component_id, item.component_key) for item in plans)
    observed = tuple((item.component_id, item.component_key) for item in ledger.components)
    if observed != expected:
        raise RuntimeError("locked holdout component plan mismatch")


def _dataset_path_from_build_marker(path: Path) -> Path:
    load_campaign_build_id(path)
    payload = _read_json(path.resolve(), description="Production Build marker")
    value = payload.get("dataset_parquet_path")
    if not isinstance(value, str) or not value:
        raise RuntimeError("Production Build marker lacks its dataset path")
    return Path(value).resolve()


def _holdout_frames(
    dataset: pd.DataFrame,
    selection: DevelopmentSelectionArtifact,
) -> Mapping[int, tuple[pd.DataFrame, pd.DataFrame]]:
    output: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    boundary = pd.Timestamp(selection.locked_holdout_start)
    date_values = pd.to_datetime(dataset["trading_date"])
    for horizon_selection in selection.horizons:
        horizon = horizon_selection.horizon
        configs = {
            item.source_config.holdout_periods
            for item in (
                horizon_selection.expected_return_component,
                horizon_selection.rank_component,
                horizon_selection.downside_quantile_component,
                horizon_selection.large_loss_component,
                *horizon_selection.ensemble_components,
            )
        }
        if len(configs) != 1:
            raise ValueError("frozen holdout period differs across components")
        reserved = reserve_locked_final_holdout(dataset, holdout_periods=configs.pop())
        if reserved.holdout_start != boundary:
            raise RuntimeError("frozen selection holdout boundary differs from dataset")
        target = f"target_return_{horizon}d"
        status = f"label_status_{horizon}d"
        endpoint = f"label_end_date_{horizon}d"
        required = {
            "symbol",
            "trading_date",
            target,
            status,
            endpoint,
            *horizon_selection.feature_names,
        }
        if missing := sorted(required - set(dataset.columns)):
            raise ValueError(f"locked holdout dataset is missing: {', '.join(missing)}")
        usable = dataset[target].notna() & dataset[status].eq("AVAILABLE")
        training = dataset.loc[
            usable & (date_values < boundary) & (pd.to_datetime(dataset[endpoint]) < boundary)
        ].copy()
        holdout = dataset.loc[usable & (date_values >= boundary)].copy()
        training = training.sort_values(["trading_date", "symbol"], kind="stable").reset_index(
            drop=True
        )
        holdout = holdout.sort_values(["trading_date", "symbol"], kind="stable").reset_index(
            drop=True
        )
        if training.empty or holdout.empty:
            raise ValueError(f"locked holdout {horizon}d training or evaluation frame is empty")
        output[horizon] = (training, holdout)
    return MappingProxyType(output)


def _publish_prediction(
    state: HoldoutComponentCheckpoint,
    frame: pd.DataFrame,
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    path = (destination / f"{state.component_id}.parquet").resolve()
    temporary = destination / f".{state.component_id}.{os.getpid()}.tmp.parquet"
    try:
        frame.to_parquet(temporary, index=False)
        if path.exists():
            observed = pd.read_parquet(path)
            if _frame_hash(observed) != _frame_hash(frame):
                raise RuntimeError("holdout prediction identity already exists with conflicts")
            temporary.unlink(missing_ok=True)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    state.prediction_path = str(path)
    state.parquet_sha256 = _file_sha256(path)
    state.frame_sha256 = _frame_hash(frame)
    state.rows = len(frame)


def _load_prediction(
    state: HoldoutComponentCheckpoint,
    *,
    expected: pd.DataFrame,
) -> pd.DataFrame:
    frame = _load_prediction_artifact(state)
    if len(frame) != len(expected):
        raise RuntimeError("holdout prediction row count differs from locked rows")
    if not frame["symbol"].astype(str).equals(expected["symbol"].astype(str)):
        raise RuntimeError("holdout prediction symbol identity mismatch")
    if not pd.to_datetime(frame["trading_date"]).equals(
        pd.to_datetime(expected["trading_date"])
    ):
        raise RuntimeError("holdout prediction date identity mismatch")
    return frame


def _load_prediction_artifact(state: HoldoutComponentCheckpoint) -> pd.DataFrame:
    if (
        state.prediction_path is None
        or state.parquet_sha256 is None
        or state.frame_sha256 is None
        or state.rows is None
    ):
        raise RuntimeError("successful holdout component lacks prediction provenance")
    path = Path(state.prediction_path).resolve()
    if _file_sha256(path) != state.parquet_sha256:
        raise RuntimeError("holdout prediction Parquet hash mismatch")
    frame = pd.read_parquet(path)
    if len(frame) != state.rows or _frame_hash(frame) != state.frame_sha256:
        raise RuntimeError("holdout prediction logical content mismatch")
    if tuple(frame.columns) != ("symbol", "trading_date", "prediction"):
        raise RuntimeError("holdout prediction schema mismatch")
    if not np.isfinite(frame["prediction"].to_numpy(dtype=float)).all():
        raise RuntimeError("holdout prediction contains non-finite values")
    return frame


def _authenticate_completed_predictions(ledger: LockedHoldoutLedger) -> None:
    for state in ledger.components:
        if state.status is not HoldoutComponentStatus.SUCCEEDED:
            raise RuntimeError("completed holdout ledger has an unfinished component")
        _load_prediction_artifact(state)


def _build_holdout_report(
    selection: DevelopmentSelectionArtifact,
    *,
    ledger: LockedHoldoutLedger,
    plans: tuple[_ComponentPlan, ...],
    holdout_frames: Mapping[int, tuple[pd.DataFrame, pd.DataFrame]],
    created_at: datetime,
) -> LockedHoldoutReport:
    state_by_id = {item.component_id: item for item in ledger.components}
    predictions: dict[str, pd.DataFrame] = {}
    component_results: list[HoldoutComponentResult] = []
    for plan in plans:
        _, holdout = holdout_frames[plan.component.horizon]
        state = state_by_id[plan.component_id]
        prediction = _load_prediction(state, expected=holdout)
        predictions[plan.component_key] = prediction
        metric_frame = prediction.copy()
        metric_frame["target"] = holdout[
            f"target_return_{plan.component.horizon}d"
        ].to_numpy(dtype=float)
        metric_frame["horizon"] = plan.component.horizon
        metric_frame["model_family"] = plan.component.model_family
        metric_frame["task"] = plan.component.task
        metric_frame["seed"] = plan.component.seed
        metric_frame["fold"] = 0
        metrics = summarize_oof_predictions(
            metric_frame,
            quantile_alpha=plan.component.source_config.quantile_alpha,
            large_loss_threshold=plan.component.source_config.large_loss_threshold,
        )
        assert state.parquet_sha256 is not None
        component_results.append(
            HoldoutComponentResult(
                component_key=plan.component_key,
                component=plan.component,
                metrics=metrics,
                prediction_sha256=state.parquet_sha256,
            )
        )
    ensemble_results = tuple(
        _evaluate_holdout_ensemble(
            horizon,
            predictions=predictions,
            holdout=holdout_frames[horizon.horizon][1],
        )
        for horizon in selection.horizons
    )
    holdout_end = max(
        pd.to_datetime(frame[1]["trading_date"]).max()
        for frame in holdout_frames.values()
    )
    values: dict[str, object] = {
        "schema_version": _REPORT_SCHEMA,
        "report_id": "0" * 64,
        "created_at": created_at,
        "selection_id": selection.selection_id,
        "ledger_id": ledger.ledger_id,
        "build_id": ledger.build_id,
        "data_snapshot_id": ledger.data_snapshot_id,
        "evaluator_code_commit": ledger.evaluator_code_commit,
        "locked_holdout_start": selection.locked_holdout_start,
        "locked_holdout_end": str(pd.Timestamp(holdout_end).date()),
        "locked_holdout_accessed": True,
        "feature_definition_hashes": _expected_feature_definition_hashes(selection),
        "component_results": tuple(component_results),
        "ensemble_results": ensemble_results,
        "selection_was_frozen_before_access": True,
        "model_choices_changed_after_access": False,
        "adoption_eligible": False,
        "adoption_blocking_reasons": (
            "historical provider revision vintages remain incomplete",
            "live out-of-sample evidence and explicit approval remain required",
            "holdout result cannot be used for further tuning or reselection",
        ),
    }
    provisional = LockedHoldoutReport.model_construct(**cast(Any, values))
    values["report_id"] = _stable_hash(_report_identity(provisional))
    return LockedHoldoutReport.model_validate(values)


def _expected_feature_definition_hashes(
    selection: DevelopmentSelectionArtifact,
) -> dict[str, str]:
    selected_features = {
        name for horizon in selection.horizons for name in horizon.feature_names
    }
    unknown = selected_features - set(V2_EXTENDED_MANIFEST.feature_names)
    if unknown:
        raise RuntimeError("selection contains a feature outside the frozen V2 manifest")
    return {
        name: V2_EXTENDED_MANIFEST.feature_definition_hashes[name]
        for name in V2_EXTENDED_MANIFEST.feature_names
        if name in selected_features
    }


def _require_completed_report(
    report: LockedHoldoutReport,
    *,
    ledger: LockedHoldoutLedger,
    selection: DevelopmentSelectionArtifact,
    build_id: str,
) -> None:
    expected_components = {
        item.component_key: item.parquet_sha256 for item in ledger.components
    }
    reported_components = {
        item.component_key: item.prediction_sha256 for item in report.component_results
    }
    if (
        report.selection_id != selection.selection_id
        or report.ledger_id != ledger.ledger_id
        or report.build_id != build_id
        or report.data_snapshot_id != selection.data_snapshot_id
        or report.evaluator_code_commit != ledger.evaluator_code_commit
        or report.locked_holdout_start != selection.locked_holdout_start
        or dict(report.feature_definition_hashes)
        != _expected_feature_definition_hashes(selection)
        or reported_components != expected_components
    ):
        raise RuntimeError("completed holdout report provenance mismatch")


def _evaluate_holdout_ensemble(
    horizon: HorizonDevelopmentSelection,
    *,
    predictions: Mapping[str, pd.DataFrame],
    holdout: pd.DataFrame,
) -> HoldoutEnsembleResult:
    matrix_columns: list[np.ndarray] = []
    dates = pd.Series(pd.to_datetime(holdout["trading_date"]), index=range(len(holdout)))
    target = pd.Series(
        holdout[f"target_return_{horizon.horizon}d"].to_numpy(dtype=float),
        index=range(len(holdout)),
    )
    for name in horizon.ensemble.component_names:
        key = f"h{horizon.horizon}:{name}"
        frame = predictions[key]
        ranked = (
            frame.groupby("trading_date", sort=False)["prediction"].rank(
                method="average", pct=True
            )
            - 0.5
        ) * 2.0
        matrix_columns.append(ranked.to_numpy(dtype=float))
    matrix = np.column_stack(matrix_columns)
    weights = np.asarray(horizon.ensemble.weights, dtype=float)
    ensemble_prediction = matrix @ weights
    metrics = evaluate_cross_sectional_predictions(
        dates=dates,
        target=target,
        prediction=pd.Series(ensemble_prediction),
    )
    standardized_target = within_date_rank_standardize(target, dates)
    residual = np.abs(standardized_target - ensemble_prediction)
    disagreement = np.std(matrix, axis=1)
    return HoldoutEnsembleResult(
        horizon=horizon.horizon,
        adopted_on_development=horizon.ensemble_adopted,
        component_names=horizon.ensemble.component_names,
        weights=horizon.ensemble.weights,
        rows=metrics.rows,
        dates=metrics.dates,
        mean_squared_error_rank_space=float(
            np.mean(np.square(standardized_target - ensemble_prediction))
        ),
        mean_daily_rank_ic=metrics.mean_daily_rank_ic,
        rank_icir=metrics.rank_icir,
        ndcg_at_5=metrics.ndcg_at_5,
        ndcg_at_10=metrics.ndcg_at_10,
        ndcg_at_20=metrics.ndcg_at_20,
        empirical_coverage_80=float(
            np.mean(residual <= horizon.uncertainty.residual_quantile_80)
        ),
        empirical_coverage_90=float(
            np.mean(residual <= horizon.uncertainty.residual_quantile_90)
        ),
        disagreement_error_correlation=_finite_correlation(disagreement, residual),
    )


def _write_ledger(ledger: LockedHoldoutLedger, path: Path) -> None:
    ledger.updated_at = datetime.now(UTC)
    payload: dict[str, object] = {
        "ledger": ledger.model_dump(mode="json"),
        "ledger_path": str(path.resolve()),
    }
    payload["metadata_hash"] = _stable_hash(payload)
    _write_json(path, payload)


def _load_ledger(path: Path) -> LockedHoldoutLedger:
    path = path.resolve()
    payload = _read_json(path, description="locked holdout ledger")
    _authenticate_envelope(payload, description="locked holdout ledger")
    if Path(str(payload.get("ledger_path", ""))).resolve() != path:
        raise RuntimeError("locked holdout ledger path metadata mismatch")
    return LockedHoldoutLedger.model_validate(payload.get("ledger"))


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    )
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError(f"{description} is missing or invalid") from None
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} is not a JSON object")
    return value


def _authenticate_envelope(payload: Mapping[str, object], *, description: str) -> None:
    authenticated = {key: value for key, value in payload.items() if key != "metadata_hash"}
    if payload.get("metadata_hash") != _stable_hash(authenticated):
        raise RuntimeError(f"{description} metadata hash mismatch")


def _ledger_identity(ledger: LockedHoldoutLedger) -> dict[str, object]:
    return {
        "schema_version": ledger.schema_version,
        "selection_id": ledger.selection_id,
        "build_id": ledger.build_id,
        "data_snapshot_id": ledger.data_snapshot_id,
        "evaluator_code_commit": ledger.evaluator_code_commit,
        "locked_holdout_start": ledger.locked_holdout_start,
        "holdout_accessed": ledger.holdout_accessed,
        "components": [
            {
                "component_id": item.component_id,
                "component_key": item.component_key,
                "horizon": item.horizon,
                "component_name": item.component_name,
            }
            for item in ledger.components
        ],
    }


def _report_identity(report: LockedHoldoutReport) -> dict[str, object]:
    return report.model_dump(mode="json", exclude={"report_id", "created_at"})


def _frame_hash(frame: pd.DataFrame) -> str:
    schema = [(str(name), str(dtype)) for name, dtype in frame.dtypes.items()]
    values = pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(values.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None
