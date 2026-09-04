"""Authenticated development-only selection and immutable Champion-candidate freezes.

This module deliberately has no final-holdout evaluation entry point.  It consumes only
authenticated development OOF reports and publishes the complete set of choices that a
separate, one-shot holdout evaluator will be allowed to use.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self, cast

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
from scipy.optimize import minimize

from stock_ai.data.contracts import CapabilityStatus
from stock_ai.features import V0_MANIFEST, V2_EXTENDED_MANIFEST
from stock_ai.ml.advanced import (
    AdvancedModelMetrics,
    AdvancedResearchConfig,
    AdvancedResearchReport,
    EnsembleResult,
    ModelFamily,
    ModelTask,
    UncertaintyCalibration,
    load_advanced_research_run,
)
from stock_ai.ml.campaign import (
    CampaignBatchStatus,
    ResearchCampaignBatch,
    ResearchCampaignManifest,
    load_campaign_build_id,
    load_campaign_manifest,
)
from stock_ai.ml.dataset import HORIZONS
from stock_ai.ml.research_metrics import (
    evaluate_cross_sectional_predictions,
    within_date_rank_standardize,
)

_MODEL_FAMILIES: tuple[ModelFamily, ...] = ("lightgbm", "xgboost", "catboost")
_MODEL_TASKS: tuple[ModelTask, ...] = ("regression", "ranking", "quantile", "large_loss")
_SELECTION_SCHEMA = "goal3-development-selection-v1"


class FrozenModelComponent(BaseModel):
    """One fully specified model choice made without using the locked holdout."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    component_name: str = Field(min_length=1)
    horizon: int
    model_family: ModelFamily
    task: ModelTask
    seed: int
    parameters: Mapping[str, int | float]
    source_config: AdvancedResearchConfig
    source_report_id: str = Field(min_length=64, max_length=64)
    source_config_hash: str = Field(min_length=64, max_length=64)
    selection_metric_name: str = Field(min_length=1)
    selection_metric_value: float

    @field_validator("parameters", mode="after")
    @classmethod
    def freeze_parameters(
        cls, value: Mapping[str, int | float]
    ) -> Mapping[str, int | float]:
        return MappingProxyType(dict(value))

    @field_serializer("parameters")
    def serialize_parameters(
        self, value: Mapping[str, int | float]
    ) -> dict[str, int | float]:
        return dict(value)

    @model_validator(mode="after")
    def coherent_source_config(self) -> Self:
        if self.source_config.config_hash != self.source_config_hash:
            raise ValueError("frozen component source config hash mismatch")
        if self.source_config.horizons != (self.horizon,):
            raise ValueError("frozen component source config horizon mismatch")
        if self.source_config.model_families != (self.model_family,):
            raise ValueError("frozen component source config family mismatch")
        if self.source_config.seeds != (self.seed,):
            raise ValueError("frozen component source config seed mismatch")
        return self


class FeatureFamilySelection(BaseModel):
    """A multi-seed tuning-period-only feature-family vote."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    family_id: str = Field(pattern=r"^F(?:[1-9]|1[0-2])$")
    family_name: str = Field(min_length=1)
    seeds: tuple[int, ...]
    seed_votes_selected: tuple[int, ...]
    selected: bool
    selected_features: tuple[str, ...]
    evidence_reports: int = Field(ge=0)
    blocked_reports: int = Field(ge=0)
    rule: Literal["at_least_two_thirds_of_seeds_tuning_only"] = (
        "at_least_two_thirds_of_seeds_tuning_only"
    )

    @model_validator(mode="after")
    def coherent_vote(self) -> Self:
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("feature selection requires at least three unique seeds")
        if not set(self.seed_votes_selected) <= set(self.seeds):
            raise ValueError("feature vote references an undeclared seed")
        required = math.ceil(len(self.seeds) * 2 / 3)
        if self.selected != (len(self.seed_votes_selected) >= required):
            raise ValueError("feature selection vote does not match its frozen rule")
        if not self.selected and self.selected_features:
            raise ValueError("rejected feature family cannot retain selected features")
        if self.selected and not self.selected_features:
            raise ValueError("selected feature family must retain its exact features")
        return self


class HorizonDevelopmentSelection(BaseModel):
    """Every frozen choice needed for one prediction horizon."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    feature_names: tuple[str, ...]
    feature_names_hash: str = Field(min_length=64, max_length=64)
    feature_families: tuple[FeatureFamilySelection, ...]
    expected_return_component: FrozenModelComponent
    rank_component: FrozenModelComponent
    downside_quantile_component: FrozenModelComponent
    large_loss_component: FrozenModelComponent
    ensemble_components: tuple[FrozenModelComponent, ...]
    ensemble: EnsembleResult
    ensemble_adopted: bool
    ensemble_best_component_rank_ic: float
    uncertainty: UncertaintyCalibration

    @model_validator(mode="after")
    def coherent_horizon(self) -> Self:
        if self.horizon not in HORIZONS:
            raise ValueError("selection horizon must be 1, 5, or 20")
        if not self.feature_names or _stable_hash(self.feature_names) != self.feature_names_hash:
            raise ValueError("selected feature identity mismatch")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("selected features must be unique")
        expected_family_ids = tuple(f"F{number}" for number in range(1, 13))
        if tuple(item.family_id for item in self.feature_families) != expected_family_ids:
            raise ValueError("feature-family evidence must contain ordered F1..F12")
        if any(item.horizon != self.horizon for item in self.feature_families):
            raise ValueError("feature-family evidence horizon mismatch")
        components = (
            self.expected_return_component,
            self.rank_component,
            self.downside_quantile_component,
            self.large_loss_component,
        )
        if any(component.horizon != self.horizon for component in components):
            raise ValueError("frozen component horizon mismatch")
        if self.expected_return_component.task != "regression":
            raise ValueError("expected-return choice must be a regression model")
        if self.rank_component.task not in ("regression", "ranking"):
            raise ValueError("rank choice must be a stackable model")
        if self.downside_quantile_component.task != "quantile":
            raise ValueError("downside choice must be a quantile model")
        if self.large_loss_component.task != "large_loss":
            raise ValueError("large-loss choice must be a classifier")
        if self.ensemble.horizon != self.horizon or self.uncertainty.horizon != self.horizon:
            raise ValueError("ensemble or uncertainty horizon mismatch")
        if tuple(item.component_name for item in self.ensemble_components) != (
            self.ensemble.component_names
        ):
            raise ValueError("frozen ensemble component identity mismatch")
        expected_adoption = (
            self.ensemble.mean_daily_rank_ic is not None
            and self.ensemble.mean_daily_rank_ic > self.ensemble_best_component_rank_ic
        )
        if self.ensemble_adopted != expected_adoption:
            raise ValueError("ensemble adoption does not match frozen development evidence")
        return self


class DevelopmentSelectionArtifact(BaseModel):
    """Content-addressed proof that every Goal 3 choice preceded holdout access."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal["goal3-development-selection-v1"] = (
        "goal3-development-selection-v1"
    )
    selection_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    build_id: str = Field(min_length=64, max_length=64)
    data_snapshot_id: str = Field(min_length=64, max_length=64)
    feature_snapshot_id: str = Field(min_length=64, max_length=64)
    feature_manifest_hash: str = Field(min_length=64, max_length=64)
    candidate_campaign_ids: tuple[str, ...]
    ablation_campaign_ids: tuple[str, ...]
    source_report_ids: tuple[str, ...]
    source_code_commits: tuple[str, ...]
    seeds: tuple[int, ...]
    locked_holdout_start: str
    locked_holdout_accessed: Literal[False] = False
    selection_basis: Literal["purged_expanding_development_oof_only"] = (
        "purged_expanding_development_oof_only"
    )
    feature_selection_complete: Literal[True] = True
    model_selection_complete: Literal[True] = True
    hyperparameter_selection_complete: Literal[True] = True
    ensemble_selection_complete: Literal[True] = True
    horizons: tuple[HorizonDevelopmentSelection, ...]
    adoption_eligible: Literal[False] = False
    adoption_blocking_reasons: tuple[str, ...]

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("selection created_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def complete_before_holdout(self) -> Self:
        if tuple(item.horizon for item in self.horizons) != HORIZONS:
            raise ValueError("development selection requires ordered 1d/5d/20d horizons")
        if len(self.seeds) < 3 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("development selection requires at least three unique seeds")
        if not self.source_report_ids or len(self.source_report_ids) != len(
            set(self.source_report_ids)
        ):
            raise ValueError("development selection report identities must be unique")
        for label, identities in (
            ("candidate campaign", self.candidate_campaign_ids),
            ("ablation campaign", self.ablation_campaign_ids),
            ("source code commit", self.source_code_commits),
        ):
            if not identities or len(identities) != len(set(identities)):
                raise ValueError(f"development selection {label} identities must be unique")
        if not self.adoption_blocking_reasons:
            raise ValueError("development-only Champion candidate cannot claim live adoption")
        if self.selection_id != _stable_hash(_selection_identity(self)):
            raise ValueError("development selection content identity mismatch")
        return self


@dataclass(frozen=True)
class _AuthenticatedRun:
    campaign_id: str
    build_id: str
    batch: ResearchCampaignBatch
    report: AdvancedResearchReport
    oof_path: Path


@dataclass(frozen=True)
class _FeatureVoteEvidence:
    campaign_ids: tuple[str, ...]
    report_ids: tuple[str, ...]
    code_commits: tuple[str, ...]
    data_snapshot_id: str
    feature_snapshot_id: str
    feature_manifest_hash: str
    locked_holdout_start: str
    by_horizon: Mapping[int, tuple[FeatureFamilySelection, ...]]
    feature_names_by_horizon: Mapping[int, tuple[str, ...]]


def freeze_development_selection(
    *,
    ablation_campaign_paths: Sequence[Path],
    candidate_campaign_paths: Sequence[Path],
    created_at: datetime | None = None,
) -> DevelopmentSelectionArtifact:
    """Authenticate complete campaigns and freeze all choices without touching holdout rows."""

    if not ablation_campaign_paths or not candidate_campaign_paths:
        raise ValueError("selection requires ablation and final-candidate campaigns")
    ablation_runs = _load_completed_campaigns(ablation_campaign_paths, require_ablations=True)
    feature_votes = _derive_feature_votes(ablation_runs)
    del ablation_runs
    candidate_runs = _load_completed_campaigns(candidate_campaign_paths, require_ablations=False)
    _require_candidate_matrix(candidate_runs)
    build_ids = {run.build_id for run in candidate_runs}
    if len(build_ids) != 1:
        raise ValueError("candidate campaigns must use one Production Build")
    build_id = next(iter(build_ids))
    if any(run.build_id != build_id for run in _load_campaign_headers(ablation_campaign_paths)):
        raise ValueError("ablation and candidate campaigns must use one Production Build")
    seeds = tuple(sorted({cast(int, run.batch.seed) for run in candidate_runs}))
    reference = candidate_runs[0].report
    if (
        feature_votes.data_snapshot_id != reference.data_snapshot_id
        or feature_votes.feature_snapshot_id != reference.feature_snapshot_id
        or feature_votes.feature_manifest_hash != reference.feature_manifest_hash
        or feature_votes.locked_holdout_start != reference.locked_holdout_start
    ):
        raise ValueError("ablation and candidate evidence do not share one data contract")
    horizons: list[HorizonDevelopmentSelection] = []
    for horizon in HORIZONS:
        horizon_runs = tuple(run for run in candidate_runs if run.batch.horizon == horizon)
        expected_features = feature_votes.feature_names_by_horizon[horizon]
        if any(run.report.feature_names != expected_features for run in horizon_runs):
            raise ValueError(
                f"candidate {horizon}d reports do not use the tuning-only selected features"
            )
        horizons.append(
            _select_horizon(
                horizon_runs,
                feature_families=feature_votes.by_horizon[horizon],
                feature_names=expected_features,
            )
        )
    source_report_ids = tuple(
        sorted(
            {
                *(run.report.report_id for run in candidate_runs),
                *feature_votes.report_ids,
            }
        )
    )
    values: dict[str, object] = {
        "schema_version": _SELECTION_SCHEMA,
        "selection_id": "0" * 64,
        "created_at": created_at or datetime.now(UTC),
        "build_id": build_id,
        "data_snapshot_id": reference.data_snapshot_id,
        "feature_snapshot_id": reference.feature_snapshot_id,
        "feature_manifest_hash": reference.feature_manifest_hash,
        "candidate_campaign_ids": tuple(
            sorted({run.campaign_id for run in candidate_runs})
        ),
        "ablation_campaign_ids": feature_votes.campaign_ids,
        "source_report_ids": source_report_ids,
        "source_code_commits": tuple(
            sorted(
                {
                    *(run.report.code_commit for run in candidate_runs),
                    *feature_votes.code_commits,
                }
            )
        ),
        "seeds": seeds,
        "locked_holdout_start": reference.locked_holdout_start,
        "locked_holdout_accessed": False,
        "horizons": tuple(horizons),
        "adoption_eligible": False,
        "adoption_blocking_reasons": (
            "locked final holdout has not yet been evaluated",
            "live out-of-sample evidence and explicit model approval are still required",
            "historical provider revision vintages are incomplete",
        ),
    }
    provisional = DevelopmentSelectionArtifact.model_construct(**cast(Any, values))
    values["selection_id"] = _stable_hash(_selection_identity(provisional))
    return DevelopmentSelectionArtifact.model_validate(values)


def write_development_selection(
    artifact: DevelopmentSelectionArtifact, destination: Path
) -> Path:
    """Publish one immutable content-addressed development selection."""

    if artifact.selection_id != _stable_hash(_selection_identity(artifact)):
        raise RuntimeError("development selection content identity mismatch")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    final_directory = destination / artifact.selection_id
    final_path = final_directory / f"{artifact.selection_id}.json"
    payload: dict[str, object] = {"selection": artifact.model_dump(mode="json")}
    payload["selection_path"] = str(final_path)
    payload["metadata_hash"] = _stable_hash(payload)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if final_directory.exists():
        observed = load_development_selection(final_path)
        if observed.selection_id != artifact.selection_id:
            raise RuntimeError("development selection identity already exists with conflicts")
        return final_path
    temporary = Path(tempfile.mkdtemp(prefix=".selection-", dir=destination))
    try:
        temporary_path = temporary / final_path.name
        with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_path


def load_development_selection(path: Path) -> DevelopmentSelectionArtifact:
    """Authenticate a frozen selection before any holdout evaluator may use it."""

    path = path.resolve()
    selection_id = path.name.removesuffix(".json")
    if path.parent.name != selection_id:
        raise RuntimeError("development selection path is not content-addressed")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("development selection metadata is missing or invalid") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("selection"), dict):
        raise RuntimeError("development selection metadata is invalid")
    authenticated = {key: value for key, value in payload.items() if key != "metadata_hash"}
    if payload.get("metadata_hash") != _stable_hash(authenticated):
        raise RuntimeError("development selection metadata hash mismatch")
    if Path(str(payload.get("selection_path", ""))).resolve() != path:
        raise RuntimeError("development selection path metadata mismatch")
    artifact = DevelopmentSelectionArtifact.model_validate(payload["selection"])
    if artifact.selection_id != selection_id:
        raise RuntimeError("development selection directory identity mismatch")
    return artifact


def _load_campaign_headers(paths: Sequence[Path]) -> tuple[ResearchCampaignManifest, ...]:
    manifests = tuple(load_campaign_manifest(path) for path in paths)
    if any(manifest.schema_version != "research-campaign-manifest-v2" for manifest in manifests):
        raise ValueError("selection requires granular v2 campaign manifests")
    return manifests


def _load_completed_campaigns(
    paths: Sequence[Path], *, require_ablations: bool
) -> tuple[_AuthenticatedRun, ...]:
    manifests = _load_campaign_headers(paths)
    output: list[_AuthenticatedRun] = []
    seen_batches: set[tuple[str, int, str, int]] = set()
    for manifest in manifests:
        if any(batch.status is not CampaignBatchStatus.SUCCEEDED for batch in manifest.batches):
            raise ValueError(f"campaign is not fully successful: {manifest.campaign_id}")
        for batch in manifest.batches:
            if batch.seed is None or batch.oof_path is None or batch.report_id is None:
                raise ValueError("successful selection batch lacks seed or artifact identity")
            identity = (manifest.build_id, batch.horizon, batch.model_family, batch.seed)
            if identity in seen_batches:
                raise ValueError("selection campaigns contain a duplicate model/horizon/seed batch")
            seen_batches.add(identity)
            report, oof = load_advanced_research_run(Path(batch.oof_path))
            _authenticate_batch_report(manifest, batch, report)
            if report.config.run_ablations != require_ablations:
                role = "ablation" if require_ablations else "final-candidate"
                raise ValueError(f"{role} campaign has the wrong ablation mode")
            output.append(
                _AuthenticatedRun(
                    campaign_id=manifest.campaign_id,
                    build_id=manifest.build_id,
                    batch=batch,
                    report=report,
                    oof_path=Path(batch.oof_path).resolve(),
                )
            )
            del oof
    if not output:
        raise ValueError("selection campaign set is empty")
    return tuple(output)


def _authenticate_batch_report(
    manifest: ResearchCampaignManifest,
    batch: ResearchCampaignBatch,
    report: AdvancedResearchReport,
) -> None:
    build_path = Path(manifest.build_manifest_path)
    if load_campaign_build_id(build_path) != manifest.build_id:
        raise RuntimeError("campaign Production Build identity mismatch")
    try:
        build_payload = json.loads(build_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("campaign Production Build marker is missing or invalid") from None
    if (
        report.data_snapshot_id != str(build_payload.get("dataset_snapshot_id", ""))
        or report.feature_snapshot_id != str(build_payload.get("v2_snapshot_id", ""))
    ):
        raise RuntimeError("campaign report snapshot lineage differs from Production Build")
    if report.report_id != batch.report_id:
        raise RuntimeError("campaign report identity mismatch")
    if report.config_hash != batch.config_hash or report.code_commit != manifest.code_commit:
        raise RuntimeError("campaign report code or config mismatch")
    if report.config.horizons != (batch.horizon,):
        raise RuntimeError("campaign report horizon mismatch")
    if report.config.model_families != (batch.model_family,):
        raise RuntimeError("campaign report model family mismatch")
    if report.config.seeds != (batch.seed,):
        raise RuntimeError("campaign report seed mismatch")
    if report.locked_holdout_accessed or report.adoption_eligible:
        raise RuntimeError("selection source must be development-only")
    if report.prediction_semantics != "return":
        raise ValueError("Champion candidate requires absolute-return reports")


def _derive_feature_votes(runs: Sequence[_AuthenticatedRun]) -> _FeatureVoteEvidence:
    build_ids = {run.build_id for run in runs}
    if len(build_ids) != 1:
        raise ValueError("ablation campaigns must use one Production Build")
    reference = runs[0].report
    for run in runs:
        if (
            run.report.data_snapshot_id != reference.data_snapshot_id
            or run.report.feature_snapshot_id != reference.feature_snapshot_id
            or run.report.feature_manifest_hash != reference.feature_manifest_hash
            or run.report.locked_holdout_start != reference.locked_holdout_start
        ):
            raise ValueError("ablation reports do not share one authenticated data contract")
    by_horizon: dict[int, tuple[FeatureFamilySelection, ...]] = {}
    features_by_horizon: dict[int, tuple[str, ...]] = {}
    canonical_plan = dict(_canonical_ablation_plan())
    for horizon in HORIZONS:
        horizon_runs = tuple(run for run in runs if run.batch.horizon == horizon)
        seeds = tuple(sorted({cast(int, run.batch.seed) for run in horizon_runs}))
        if len(seeds) < 3:
            raise ValueError(f"{horizon}d feature ablation requires at least three seeds")
        selections: list[FeatureFamilySelection] = []
        selected_names = set(V0_MANIFEST.feature_names)
        for family_id in (f"F{number}" for number in range(1, 13)):
            seed_votes: list[int] = []
            evidence_reports = 0
            blocked_reports = 0
            for seed in seeds:
                seed_rows = []
                for run in horizon_runs:
                    if run.batch.seed != seed:
                        continue
                    matching = tuple(
                        item for item in run.report.ablations if item.family_id == family_id
                    )
                    if len(matching) != 1:
                        raise ValueError(
                            f"ablation report must contain exactly one {family_id} result"
                        )
                    seed_rows.append(matching[0])
                available = [
                    item for item in seed_rows if item.status is CapabilityStatus.AVAILABLE
                ]
                blocked_reports += len(seed_rows) - len(available)
                evidence_reports += len(available)
                selected_count = sum(
                    bool(item.selected_on_tuning_period) for item in available
                )
                if (
                    available
                    and len(available) == len(seed_rows)
                    and selected_count * 2 > len(available)
                ):
                    seed_votes.append(seed)
            plan = canonical_plan[family_id]
            required_votes = math.ceil(len(seeds) * 2 / 3)
            selected = len(seed_votes) >= required_votes
            family_features = plan[1] if selected else ()
            selected_names.update(family_features)
            selections.append(
                FeatureFamilySelection(
                    horizon=horizon,
                    family_id=family_id,
                    family_name=plan[0],
                    seeds=seeds,
                    seed_votes_selected=tuple(seed_votes),
                    selected=selected,
                    selected_features=family_features,
                    evidence_reports=evidence_reports,
                    blocked_reports=blocked_reports,
                )
            )
        ordered = tuple(
            name for name in V2_EXTENDED_MANIFEST.feature_names if name in selected_names
        )
        if not ordered:
            raise ValueError(f"{horizon}d feature vote selected no authenticated features")
        by_horizon[horizon] = tuple(selections)
        features_by_horizon[horizon] = ordered
    return _FeatureVoteEvidence(
        campaign_ids=tuple(sorted({run.campaign_id for run in runs})),
        report_ids=tuple(sorted(run.report.report_id for run in runs)),
        code_commits=tuple(sorted({run.report.code_commit for run in runs})),
        data_snapshot_id=reference.data_snapshot_id,
        feature_snapshot_id=reference.feature_snapshot_id,
        feature_manifest_hash=reference.feature_manifest_hash,
        locked_holdout_start=reference.locked_holdout_start,
        by_horizon=MappingProxyType(by_horizon),
        feature_names_by_horizon=MappingProxyType(features_by_horizon),
    )


def _canonical_ablation_plan() -> tuple[tuple[str, tuple[str, tuple[str, ...]]], ...]:
    from stock_ai.ml.advanced import feature_family_ablation_plan

    return tuple(
        (
            item.family_id,
            (item.family_name, item.added_features),
        )
        for item in feature_family_ablation_plan(V2_EXTENDED_MANIFEST.feature_names)
        if item.family_id != "F0"
    )


def _require_candidate_matrix(runs: Sequence[_AuthenticatedRun]) -> None:
    seeds = tuple(sorted({cast(int, run.batch.seed) for run in runs}))
    if len(seeds) < 3:
        raise ValueError("final candidate comparison requires at least three seeds")
    observed = {
        (run.batch.horizon, run.batch.model_family, cast(int, run.batch.seed)) for run in runs
    }
    expected = {
        (horizon, family, seed)
        for horizon in HORIZONS
        for family in _MODEL_FAMILIES
        for seed in seeds
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "final candidate matrix is incomplete or unexpected; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )
    reference = runs[0].report
    for run in runs:
        report = run.report
        if (
            report.data_snapshot_id != reference.data_snapshot_id
            or report.feature_snapshot_id != reference.feature_snapshot_id
            or report.feature_manifest_hash != reference.feature_manifest_hash
            or report.locked_holdout_start != reference.locked_holdout_start
            or report.historical_revision_policy != reference.historical_revision_policy
            or report.historical_revision_status != reference.historical_revision_status
        ):
            raise ValueError("candidate reports do not share one authenticated data contract")
        tasks = {(item.task, item.seed) for item in report.model_metrics}
        if tasks != {(task, cast(int, run.batch.seed)) for task in _MODEL_TASKS}:
            raise ValueError("candidate report model task evidence is incomplete")
        if len(report.tuning_results) != 1:
            raise ValueError("candidate report must freeze exactly one tuning result")


def _select_horizon(
    runs: Sequence[_AuthenticatedRun],
    *,
    feature_families: tuple[FeatureFamilySelection, ...],
    feature_names: tuple[str, ...],
) -> HorizonDevelopmentSelection:
    horizon = runs[0].batch.horizon
    metrics = [item for run in runs for item in run.report.model_metrics]
    reports_by_component: dict[str, _AuthenticatedRun] = {}
    for run in runs:
        for task in _MODEL_TASKS:
            name = _component_name(run.batch.model_family, task, cast(int, run.batch.seed))
            reports_by_component[name] = run
    expected_return = _best_component(metrics, task="regression", reports=reports_by_component)
    quantile = _best_component(metrics, task="quantile", reports=reports_by_component)
    large_loss = _best_component(metrics, task="large_loss", reports=reports_by_component)
    wide = _aligned_rank_component_frame(runs)
    ensemble, uncertainty, component_rank_ics = _fit_selection_ensemble(wide, horizon=horizon)
    best_rank_name = max(
        component_rank_ics,
        key=lambda name: (component_rank_ics[name], name),
    )
    best_component_rank_ic = component_rank_ics[best_rank_name]
    best_rank = _component_from_name(
        metrics,
        component_name=best_rank_name,
        reports=reports_by_component,
        metric_name="meta_evaluation_mean_daily_rank_ic",
        metric_value=best_component_rank_ic,
    )
    ensemble_components = tuple(
        _component_from_name(
            metrics,
            component_name=name,
            reports=reports_by_component,
            metric_name="meta_evaluation_mean_daily_rank_ic",
            metric_value=component_rank_ics[name],
        )
        for name in ensemble.component_names
    )
    ensemble_adopted = (
        ensemble.mean_daily_rank_ic is not None
        and ensemble.mean_daily_rank_ic > best_component_rank_ic
    )
    return HorizonDevelopmentSelection(
        horizon=horizon,
        feature_names=feature_names,
        feature_names_hash=_stable_hash(feature_names),
        feature_families=feature_families,
        expected_return_component=expected_return,
        rank_component=best_rank,
        downside_quantile_component=quantile,
        large_loss_component=large_loss,
        ensemble_components=ensemble_components,
        ensemble=ensemble,
        ensemble_adopted=ensemble_adopted,
        ensemble_best_component_rank_ic=best_component_rank_ic,
        uncertainty=uncertainty,
    )


def _best_component(
    metrics: Sequence[AdvancedModelMetrics],
    *,
    task: ModelTask | None,
    reports: Mapping[str, _AuthenticatedRun],
) -> FrozenModelComponent:
    candidates = [item for item in metrics if task is None or item.task == task]
    if not candidates:
        raise ValueError(f"no development metric for task {task}")
    if task == "quantile":
        usable = [item for item in candidates if item.pinball_loss is not None]
        metric_name = "pinball_loss"
        if not usable:
            raise ValueError(f"no finite development selection metric for task {task}")
        chosen = min(
            usable,
            key=lambda item: (cast(float, item.pinball_loss), _metric_component_name(item)),
        )
        metric_value = cast(float, chosen.pinball_loss)
    elif task == "large_loss":
        usable = [item for item in candidates if item.brier_score is not None]
        metric_name = "brier_score"
        if not usable:
            raise ValueError(f"no finite development selection metric for task {task}")
        chosen = min(
            usable,
            key=lambda item: (cast(float, item.brier_score), _metric_component_name(item)),
        )
        metric_value = cast(float, chosen.brier_score)
    else:
        usable = [item for item in candidates if item.mean_daily_rank_ic is not None]
        metric_name = "mean_daily_rank_ic"
        if not usable:
            raise ValueError(f"no finite development selection metric for task {task}")
        chosen = max(
            usable,
            key=lambda item: (cast(float, item.mean_daily_rank_ic), _metric_component_name(item)),
        )
        metric_value = cast(float, chosen.mean_daily_rank_ic)
    component_name = _metric_component_name(chosen)
    source = reports[component_name]
    tuning = source.report.tuning_results[0]
    return FrozenModelComponent(
        component_name=component_name,
        horizon=chosen.horizon,
        model_family=chosen.model_family,
        task=chosen.task,
        seed=chosen.seed,
        parameters=tuning.best_parameters,
        source_config=source.report.config,
        source_report_id=source.report.report_id,
        source_config_hash=source.report.config_hash,
        selection_metric_name=metric_name,
        selection_metric_value=metric_value,
    )


def _component_from_name(
    metrics: Sequence[AdvancedModelMetrics],
    *,
    component_name: str,
    reports: Mapping[str, _AuthenticatedRun],
    metric_name: str,
    metric_value: float,
) -> FrozenModelComponent:
    matching = tuple(item for item in metrics if _metric_component_name(item) == component_name)
    if len(matching) != 1:
        raise ValueError("frozen component does not map to exactly one model result")
    chosen = matching[0]
    source = reports[component_name]
    return FrozenModelComponent(
        component_name=component_name,
        horizon=chosen.horizon,
        model_family=chosen.model_family,
        task=chosen.task,
        seed=chosen.seed,
        parameters=source.report.tuning_results[0].best_parameters,
        source_config=source.report.config,
        source_report_id=source.report.report_id,
        source_config_hash=source.report.config_hash,
        selection_metric_name=metric_name,
        selection_metric_value=metric_value,
    )


def _aligned_rank_component_frame(runs: Sequence[_AuthenticatedRun]) -> pd.DataFrame:
    identity_frame: pd.DataFrame | None = None
    component_values: dict[str, np.ndarray] = {}
    for run in sorted(runs, key=lambda item: (item.batch.model_family, cast(int, item.batch.seed))):
        report, oof = load_advanced_research_run(run.oof_path)
        if report.report_id != run.report.report_id:
            raise RuntimeError("candidate report changed between authentication passes")
        stackable = oof.loc[oof["task"].isin(("regression", "ranking"))].copy()
        del oof
        if stackable.empty:
            raise ValueError("candidate report has no stackable OOF rows")
        for task, part in stackable.groupby("task", sort=True):
            name = _component_name(
                run.batch.model_family,
                cast(ModelTask, task),
                cast(int, run.batch.seed),
            )
            part = part.sort_values(["trading_date", "symbol"], kind="stable").reset_index(
                drop=True
            )
            if part.duplicated(["symbol", "trading_date"]).any():
                raise RuntimeError("candidate OOF component contains duplicate rows")
            ranked = (
                part.groupby("trading_date", sort=False)["prediction"].rank(
                    method="average", pct=True
                )
                - 0.5
            ) * 2.0
            if identity_frame is None:
                identity_frame = part.loc[
                    :, ["symbol", "trading_date", "target", "label_end"]
                ].copy()
                component_values[name] = ranked.to_numpy(dtype=float)
                continue
            if len(part) != len(identity_frame):
                raise ValueError("candidate OOF components are not exactly aligned")
            if not part["symbol"].astype(str).equals(identity_frame["symbol"].astype(str)):
                raise ValueError("candidate OOF symbols are not exactly aligned")
            if not pd.to_datetime(part["trading_date"]).equals(
                pd.to_datetime(identity_frame["trading_date"])
            ):
                raise ValueError("candidate OOF dates are not exactly aligned")
            if not pd.to_datetime(part["label_end"]).equals(
                pd.to_datetime(identity_frame["label_end"])
            ):
                raise ValueError("candidate OOF label endpoints are not exactly aligned")
            if not np.array_equal(
                part["target"].to_numpy(dtype=float),
                identity_frame["target"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                raise ValueError("candidate OOF targets are not exactly aligned")
            component_values[name] = ranked.to_numpy(dtype=float)
        del part, report, stackable
    if identity_frame is None or len(component_values) < 2:
        raise ValueError("candidate OOF ensemble requires at least two aligned components")
    return pd.concat(
        [identity_frame, pd.DataFrame(component_values, index=identity_frame.index)],
        axis="columns",
    )


def _fit_selection_ensemble(
    wide: pd.DataFrame, *, horizon: int
) -> tuple[EnsembleResult, UncertaintyCalibration, Mapping[str, float]]:
    names = tuple(
        name
        for name in wide.columns
        if name not in {"symbol", "trading_date", "target", "label_end"}
    )
    if len(names) < 2:
        raise ValueError("candidate OOF ensemble requires at least two components")
    matrix = wide.loc[:, list(names)].to_numpy(dtype=float)
    dates = pd.Series(pd.to_datetime(wide["trading_date"]), index=range(len(wide)))
    label_end = pd.Series(pd.to_datetime(wide["label_end"]), index=range(len(wide)))
    target = pd.Series(wide["target"].to_numpy(dtype=float), index=range(len(wide)))
    standardized = within_date_rank_standardize(target, dates)
    meta_fit, calibration, evaluation = _chronological_meta_masks(dates, label_end)

    def objective(weights: np.ndarray) -> float:
        return float(np.mean(np.square(standardized[meta_fit] - matrix[meta_fit] @ weights)))

    initial = np.full(len(names), 1.0 / len(names), dtype=float)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(names),
        constraints={"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"OOF simplex optimization failed: {result.message}")
    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, 1.0)
    weights /= weights.sum()
    ensemble_prediction = matrix @ weights
    ensemble_metrics = evaluate_cross_sectional_predictions(
        dates=dates.loc[evaluation].reset_index(drop=True),
        target=target.loc[evaluation].reset_index(drop=True),
        prediction=pd.Series(ensemble_prediction[evaluation]),
    )
    component_rank_ics: dict[str, float] = {}
    for index, name in enumerate(names):
        metrics = evaluate_cross_sectional_predictions(
            dates=dates.loc[evaluation].reset_index(drop=True),
            target=target.loc[evaluation].reset_index(drop=True),
            prediction=pd.Series(matrix[evaluation, index]),
        )
        if metrics.mean_daily_rank_ic is None:
            raise ValueError(f"component has no rank evidence in meta-evaluation: {name}")
        component_rank_ics[name] = metrics.mean_daily_rank_ic
    evaluation_matrix = matrix[evaluation]
    correlations = pd.DataFrame(evaluation_matrix, columns=names).corr()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
    disagreement = np.std(matrix, axis=1)
    residual = np.abs(standardized - ensemble_prediction)
    ensemble = EnsembleResult(
        horizon=horizon,
        component_names=names,
        weights=tuple(float(value) for value in weights),
        mean_daily_rank_ic=ensemble_metrics.mean_daily_rank_ic,
        mean_pairwise_correlation=float(upper.mean()) if len(upper) else None,
        mean_disagreement=float(np.mean(disagreement[evaluation])),
        uncertainty_error_correlation=_finite_correlation(
            disagreement[evaluation], residual[evaluation]
        ),
        meta_fit_rows=int(np.sum(meta_fit)),
        meta_evaluation_rows=int(np.sum(evaluation)),
    )
    q80 = float(np.quantile(residual[calibration], 0.80))
    q90 = float(np.quantile(residual[calibration], 0.90))
    uncertainty = UncertaintyCalibration(
        horizon=horizon,
        residual_quantile_80=q80,
        residual_quantile_90=q90,
        empirical_coverage_80=float(np.mean(residual[evaluation] <= q80)),
        empirical_coverage_90=float(np.mean(residual[evaluation] <= q90)),
        disagreement_error_correlation=_finite_correlation(
            disagreement[evaluation], residual[evaluation]
        ),
        calibration_rows=int(np.sum(calibration)),
        evaluation_rows=int(np.sum(evaluation)),
    )
    return ensemble, uncertainty, MappingProxyType(component_rank_ics)


def _chronological_meta_masks(
    dates: pd.Series, label_end: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    unique_dates = pd.DatetimeIndex(pd.to_datetime(dates).sort_values().unique())
    if len(unique_dates) < 6:
        raise ValueError("OOF selection requires at least six dates for disjoint meta stages")
    fit_boundary = unique_dates[max(1, int(len(unique_dates) * 0.50))]
    calibration_boundary = unique_dates[max(2, int(len(unique_dates) * 0.75))]
    date_values = pd.to_datetime(dates)
    endpoint_values = pd.to_datetime(label_end)
    fit = (date_values < fit_boundary) & (endpoint_values < fit_boundary)
    calibration = (
        (date_values >= fit_boundary)
        & (date_values < calibration_boundary)
        & (endpoint_values < calibration_boundary)
    )
    evaluation = date_values >= calibration_boundary
    if not fit.any() or not calibration.any() or not evaluation.any():
        raise ValueError("OOF selection meta stages are empty after endpoint purge")
    return fit.to_numpy(), calibration.to_numpy(), evaluation.to_numpy()


def _component_name(family: str, task: ModelTask, seed: int) -> str:
    return f"{family}:{task}:seed={seed}"


def _metric_component_name(metric: AdvancedModelMetrics) -> str:
    return _component_name(metric.model_family, metric.task, metric.seed)


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _selection_identity(artifact: DevelopmentSelectionArtifact) -> dict[str, object]:
    return artifact.model_dump(mode="json", exclude={"selection_id", "created_at"})


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
