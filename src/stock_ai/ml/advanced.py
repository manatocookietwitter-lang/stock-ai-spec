"""Leakage-safe GBDT, ranking, downside, tuning, and OOF ensemble research."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Self

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor
from optuna.samplers import TPESampler
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, log_loss, mean_pinball_loss

from stock_ai.data.contracts import CapabilityStatus
from stock_ai.domain import Prediction, PredictionUncertainty
from stock_ai.features import FEATURE_REGISTRY, V0_MANIFEST, V2_EXTENDED_MANIFEST
from stock_ai.ml.dataset import HORIZONS
from stock_ai.ml.research_metrics import (
    RankingMetrics,
    cross_sectional_relevance,
    evaluate_cross_sectional_predictions,
    within_date_rank_standardize,
)
from stock_ai.ml.validation import PurgedExpandingWindowSplitter, reserve_locked_final_holdout

ModelFamily = Literal["lightgbm", "xgboost", "catboost"]
ModelTask = Literal["regression", "ranking", "quantile", "large_loss"]
TargetFamily = Literal["return", "topix_excess", "sector_excess", "beta_residual"]

_MODEL_FAMILIES: tuple[ModelFamily, ...] = ("lightgbm", "xgboost", "catboost")
_TASKS: tuple[ModelTask, ...] = ("regression", "ranking", "quantile", "large_loss")


class AdvancedResearchConfig(BaseModel):
    """A bounded and serializable Goal 3 research configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizons: tuple[int, ...] = HORIZONS
    target_family: TargetFamily = "return"
    model_families: tuple[ModelFamily, ...] = _MODEL_FAMILIES
    seeds: tuple[int, ...] = (17, 29, 43)
    initial_train_periods: int = Field(default=500, ge=20)
    validation_periods: int = Field(default=60, ge=1)
    step_periods: int = Field(default=60, ge=1)
    holdout_periods: int = Field(default=120, ge=1)
    estimator_count: int = Field(default=300, ge=5, le=2_000)
    tuning_trials: int = Field(default=20, ge=1, le=100)
    tuning_timeout_seconds: int = Field(default=900, ge=10, le=7_200)
    correlation_threshold: float = Field(default=0.98, gt=0.0, le=1.0)
    clip_lower_quantile: float = Field(default=0.005, ge=0.0, lt=0.5)
    clip_upper_quantile: float = Field(default=0.995, gt=0.5, le=1.0)
    quantile_alpha: float = Field(default=0.10, gt=0.0, lt=0.5)
    large_loss_threshold: float = Field(default=-0.08, lt=0.0)
    ablation_families: tuple[str, ...] = tuple(f"F{number}" for number in range(1, 13))
    run_ablations: bool = True
    run_diagnostics: bool = True
    diagnostic_feature_limit: int = Field(default=100, ge=1, le=200)
    max_materialized_oof_rows: int = Field(default=5_000_000, ge=1_000)
    max_model_fits: int = Field(default=5_000, ge=10)
    hypothesis: str = Field(
        default=(
            "Leakage-safe GBDT/ranking/downside models add development evidence "
            "beyond Goal 2 baselines without opening the locked holdout"
        ),
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if (
            not self.horizons
            or len(self.horizons) != len(set(self.horizons))
            or not set(self.horizons) <= set(HORIZONS)
        ):
            raise ValueError("horizons must be a unique subset of 1, 5, and 20")
        if not self.model_families or len(self.model_families) != len(set(self.model_families)):
            raise ValueError("model families must be non-empty and unique")
        if len(self.seeds) != len(set(self.seeds)) or not self.seeds:
            raise ValueError("research seeds must be non-empty and unique")
        valid_ablations = {f"F{number}" for number in range(1, 13)}
        if (
            len(self.ablation_families) != len(set(self.ablation_families))
            or not set(self.ablation_families) <= valid_ablations
        ):
            raise ValueError("ablation families must be a unique subset of F1..F12")
        return self

    @property
    def config_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))


class AdvancedModelMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    model_family: ModelFamily
    task: ModelTask
    seed: int
    folds: int = Field(ge=1)
    rows: int = Field(ge=1)
    dates: int = Field(ge=1)
    mean_squared_error: float | None = Field(default=None, ge=0)
    mean_daily_rank_ic: float | None = None
    rank_ic_standard_deviation: float | None = Field(default=None, ge=0)
    rank_icir: float | None = None
    ndcg_at_5: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_10: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_20: float | None = Field(default=None, ge=0, le=1)
    precision_at_5: float | None = Field(default=None, ge=0, le=1)
    precision_at_10: float | None = Field(default=None, ge=0, le=1)
    precision_at_20: float | None = Field(default=None, ge=0, le=1)
    top_5_mean_target: float | None = None
    top_10_mean_target: float | None = None
    top_20_mean_target: float | None = None
    pinball_loss: float | None = Field(default=None, ge=0)
    lower_tail_rate: float | None = Field(default=None, ge=0, le=1)
    brier_score: float | None = Field(default=None, ge=0, le=1)
    log_loss: float | None = Field(default=None, ge=0)
    expected_calibration_error: float | None = Field(default=None, ge=0, le=1)


class TrialAudit(BaseModel):
    """Immutable Optuna trial record, including non-successful outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    number: int = Field(ge=0)
    state: Literal["COMPLETE", "PRUNED", "FAIL", "RUNNING", "WAITING"]
    parameters: Mapping[str, int | float]
    value: float | None = None
    duration_seconds: float | None = Field(default=None, ge=0)
    failure_reason: str | None = None

    @field_validator("parameters", mode="after")
    @classmethod
    def freeze_trial_parameters(
        cls, value: Mapping[str, int | float]
    ) -> Mapping[str, int | float]:
        return MappingProxyType(dict(value))

    @field_serializer("parameters")
    def serialize_trial_parameters(
        self, value: Mapping[str, int | float]
    ) -> dict[str, int | float]:
        return dict(value)


class TuningSearchError(ValueError):
    """Tuning failure that preserves every attempted trial for audit persistence."""

    def __init__(
        self,
        message: str,
        *,
        horizon: int,
        model_family: ModelFamily,
        trials: tuple[TrialAudit, ...],
        trial_contexts: tuple[tuple[int, ModelFamily, TrialAudit], ...] | None = None,
    ) -> None:
        super().__init__(message)
        self.horizon = horizon
        self.model_family = model_family
        self.trials = trials
        self.trial_contexts = trial_contexts or tuple(
            (horizon, model_family, trial) for trial in trials
        )


class AdvancedResearchExecutionError(ValueError):
    """Run failure carrying all completed research progress for durable rejection audit."""

    def __init__(
        self,
        message: str,
        *,
        trial_contexts: tuple[tuple[int, ModelFamily, TrialAudit], ...],
        fold_results: tuple[AdvancedFoldResult, ...],
    ) -> None:
        super().__init__(message)
        self.trial_contexts = trial_contexts
        self.fold_results = fold_results


class TuningResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    model_family: ModelFamily
    trials_completed: int = Field(ge=1)
    timeout_seconds: int = Field(ge=1)
    best_value: float
    best_parameters: Mapping[str, int | float]
    trials: tuple[TrialAudit, ...]

    @field_validator("best_parameters", mode="after")
    @classmethod
    def freeze_parameters(cls, value: Mapping[str, int | float]) -> Mapping[str, int | float]:
        return MappingProxyType(dict(value))

    @field_serializer("best_parameters")
    def serialize_parameters(self, value: Mapping[str, int | float]) -> dict[str, int | float]:
        return dict(value)


class ResearchStageBoundary(BaseModel):
    """Chronological boundary separating selection from model evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    tuning_end: str
    model_evaluation_start: str
    tuning_rows: int = Field(ge=1)
    evaluation_candidate_rows: int = Field(ge=1)


class AdvancedFoldResult(BaseModel):
    """Per-fold evaluation evidence retained in the research artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    model_family: ModelFamily
    task: ModelTask
    seed: int
    fold: int = Field(ge=0)
    validation_start: str
    validation_end: str
    rows: int = Field(ge=1)
    mean_squared_error: float = Field(ge=0)
    mean_daily_rank_ic: float | None = None


class DecisionPredictionBlock(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    symbol: str = Field(min_length=1)
    as_of: datetime
    reason: str = Field(min_length=1)

    @field_validator("as_of")
    @classmethod
    def block_as_of_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("blocked prediction as_of must be timezone-aware")
        return value


class DecisionCompatiblePredictionBatch(BaseModel):
    """Typed Decision Engine inputs plus explicitly blocked historical rows."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    predictions: tuple[Prediction, ...]
    blocked: tuple[DecisionPredictionBlock, ...]


class FeatureAblationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    family_id: str
    family_name: str
    horizon: int | None = None
    status: CapabilityStatus
    added_features: tuple[str, ...]
    mean_daily_rank_ic: float | None = None
    incremental_rank_ic: float | None = None
    selection_rank_ic: float | None = None
    selected_on_tuning_period: bool | None = None
    blocking_reason: str | None = None


class EnsembleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    component_names: tuple[str, ...]
    weights: tuple[float, ...]
    mean_daily_rank_ic: float | None
    mean_pairwise_correlation: float | None
    mean_disagreement: float
    uncertainty_error_correlation: float | None
    meta_fit_rows: int = Field(ge=1)
    meta_evaluation_rows: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_simplex(self) -> Self:
        if len(self.component_names) != len(self.weights) or not self.weights:
            raise ValueError("ensemble names and weights must be non-empty and aligned")
        if any(weight < -1e-10 for weight in self.weights):
            raise ValueError("ensemble weights must be non-negative")
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-7):
            raise ValueError("ensemble weights must sum to one")
        return self


class UncertaintyCalibration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    score_space: Literal["within_date_rank"] = "within_date_rank"
    residual_quantile_80: float = Field(ge=0)
    residual_quantile_90: float = Field(ge=0)
    empirical_coverage_80: float = Field(ge=0, le=1)
    empirical_coverage_90: float = Field(ge=0, le=1)
    disagreement_error_correlation: float | None
    calibration_rows: int = Field(ge=1)
    evaluation_rows: int = Field(ge=1)


class FeatureDiagnostic(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    model_family: ModelFamily
    feature_name: str
    missing_rate: float = Field(ge=0, le=1)
    retained_fold_fraction: float = Field(ge=0, le=1)
    oos_permutation_mse_increase: float | None
    permutation_standard_deviation: float | None = Field(default=None, ge=0)


class SeedStability(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    horizon: int
    model_family: ModelFamily
    task: Literal["regression", "ranking"]
    seeds: tuple[int, ...]
    rank_ic_standard_deviation_across_seeds: float | None = Field(default=None, ge=0)
    mean_pairwise_prediction_correlation: float | None


class AdvancedResearchReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    report_id: str = Field(min_length=64, max_length=64)
    created_at: datetime
    code_commit: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    config: AdvancedResearchConfig
    config_hash: str = Field(min_length=64, max_length=64)
    data_snapshot_id: str = Field(min_length=64, max_length=64)
    feature_snapshot_id: str = Field(min_length=64, max_length=64)
    feature_set_id: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    feature_manifest_hash: str = Field(min_length=64, max_length=64)
    feature_definition_hashes: Mapping[str, str]
    feature_names: tuple[str, ...]
    prediction_semantics: TargetFamily
    locked_holdout_start: str
    locked_holdout_accessed: bool = False
    historical_revision_policy: str
    historical_revision_status: CapabilityStatus
    adoption_eligible: bool
    adoption_blocking_reasons: tuple[str, ...]
    cost_scenarios_bps: tuple[int, ...]
    cost_evaluation_status: CapabilityStatus
    tax_policy_version: str = Field(min_length=1)
    decision_engine_version: str = Field(min_length=1)
    library_versions: tuple[tuple[str, str], ...]
    stage_boundaries: tuple[ResearchStageBoundary, ...]
    tuning_results: tuple[TuningResult, ...]
    fold_results: tuple[AdvancedFoldResult, ...]
    model_metrics: tuple[AdvancedModelMetrics, ...]
    ablations: tuple[FeatureAblationResult, ...]
    ensembles: tuple[EnsembleResult, ...]
    uncertainty: tuple[UncertaintyCalibration, ...]
    feature_diagnostics: tuple[FeatureDiagnostic, ...]
    seed_stability: tuple[SeedStability, ...]
    oof_rows: int = Field(ge=1)
    oof_sha256: str = Field(min_length=64, max_length=64)

    @field_validator("created_at")
    @classmethod
    def created_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("advanced research created_at must be timezone-aware")
        return value

    @field_validator("feature_definition_hashes", mode="after")
    @classmethod
    def freeze_feature_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("feature_definition_hashes")
    def serialize_feature_hashes(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def development_report_never_claims_adoption(self) -> Self:
        if self.locked_holdout_accessed:
            raise ValueError("development research report cannot access the locked holdout")
        if self.adoption_eligible or not self.adoption_blocking_reasons:
            raise ValueError("development research report must remain research-only")
        if self.config.config_hash != self.config_hash or self.config.hypothesis != self.hypothesis:
            raise ValueError("advanced research config provenance is incoherent")
        if set(self.feature_names) != set(self.feature_definition_hashes):
            raise ValueError("advanced research feature definition provenance is incomplete")
        return self


@dataclass(frozen=True)
class AdvancedResearchRun:
    report: AdvancedResearchReport
    oof_predictions: pd.DataFrame


@dataclass
class _AdvancedResearchProgress:
    tuning_results: list[TuningResult] = field(default_factory=list)
    oof_parts: list[pd.DataFrame] = field(default_factory=list)


class FoldPreprocessor:
    """Training-fold-only clipping, imputation, and correlation pruning."""

    def __init__(
        self,
        feature_names: Sequence[str],
        *,
        lower_quantile: float,
        upper_quantile: float,
        correlation_threshold: float,
    ) -> None:
        self.feature_names = tuple(feature_names)
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.correlation_threshold = correlation_threshold
        self._lower: pd.Series | None = None
        self._upper: pd.Series | None = None
        self._median: pd.Series | None = None
        self._retained: tuple[str, ...] = ()

    @property
    def retained_features(self) -> tuple[str, ...]:
        if not self._retained:
            raise RuntimeError("fold preprocessor has not been fitted")
        return self._retained

    def fit(self, frame: pd.DataFrame) -> FoldPreprocessor:
        numeric = _finite_numeric(frame.loc[:, list(self.feature_names)])
        lower = numeric.quantile(self.lower_quantile)
        upper = numeric.quantile(self.upper_quantile)
        clipped = numeric.clip(lower=lower, upper=upper, axis="columns")
        median = clipped.median().fillna(0.0)
        filled = clipped.fillna(median)
        retained: list[str] = []
        for name in self.feature_names:
            if filled[name].nunique(dropna=False) <= 1:
                continue
            if not retained:
                retained.append(name)
                continue
            correlations = filled.loc[:, retained].corrwith(filled[name]).abs()
            if correlations.fillna(0.0).max() <= self.correlation_threshold:
                retained.append(name)
        self._lower = lower
        self._upper = upper
        self._median = median
        self._retained = tuple(retained)
        if not self._retained:
            raise ValueError("BLOCKED_BY_DATA_CAPABILITY: every fold feature is constant")
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._lower is None or self._upper is None or self._median is None:
            raise RuntimeError("fold preprocessor must be fitted before transform")
        numeric = _finite_numeric(frame.loc[:, list(self.feature_names)])
        clipped = numeric.clip(lower=self._lower, upper=self._upper, axis="columns")
        return clipped.fillna(self._median).loc[:, list(self.retained_features)]


def generate_oof_predictions(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    target_column: str,
    label_end_column: str,
    horizon: int,
    family: ModelFamily,
    task: ModelTask,
    seed: int,
    parameters: Mapping[str, int | float],
    config: AdvancedResearchConfig,
    validation_not_before: pd.Timestamp | None = None,
    progress_sink: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Generate one model's OOF predictions using only each fold's training rows."""

    required = {
        "symbol",
        "trading_date",
        target_column,
        label_end_column,
        *feature_names,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"advanced OOF frame is missing columns: {', '.join(missing)}")
    splitter = PurgedExpandingWindowSplitter(
        initial_train_periods=config.initial_train_periods,
        validation_periods=config.validation_periods,
        step_periods=config.step_periods,
        purge_periods=0,
        embargo_periods=horizon,
        label_horizon_periods=horizon,
    )
    outputs: list[pd.DataFrame] = []
    for fold in splitter.split(frame, label_end_column=label_end_column):
        train = frame.iloc[list(fold.train_indices)].copy()
        validation = frame.iloc[list(fold.validation_indices)].copy()
        train = train.loc[train[target_column].notna()].sort_values(
            ["trading_date", "symbol"], kind="stable"
        )
        validation = validation.loc[validation[target_column].notna()].sort_values(
            ["trading_date", "symbol"], kind="stable"
        )
        if validation_not_before is not None:
            validation = validation.loc[
                pd.to_datetime(validation["trading_date"]) >= validation_not_before
            ]
        if train.empty or validation.empty:
            continue
        preprocessor = FoldPreprocessor(
            feature_names,
            lower_quantile=config.clip_lower_quantile,
            upper_quantile=config.clip_upper_quantile,
            correlation_threshold=config.correlation_threshold,
        ).fit(train)
        train_x = preprocessor.transform(train)
        validation_x = preprocessor.transform(validation)
        train_target = train[target_column].astype(float)
        prediction = _fit_predict(
            family=family,
            task=task,
            train_x=train_x,
            train_target=train_target,
            train_dates=train["trading_date"],
            validation_x=validation_x,
            seed=seed,
            parameters=parameters,
            config=config,
        )
        prediction_array = np.asarray(prediction, dtype=float)
        if prediction_array.ndim != 1 or len(prediction_array) != len(validation):
            raise RuntimeError("model emitted a prediction vector with the wrong row count")
        if not np.isfinite(prediction_array).all():
            raise RuntimeError("model emitted non-finite fold predictions")
        identity_columns = ["symbol", "trading_date"]
        if "as_of" in validation.columns:
            identity_columns.append("as_of")
        output = validation.loc[
            :, [*identity_columns, target_column, label_end_column]
        ].copy()
        output = output.rename(columns={target_column: "target", label_end_column: "label_end"})
        output["horizon"] = horizon
        output["model_family"] = family
        output["task"] = task
        output["seed"] = seed
        output["fold"] = fold.fold_number
        output["prediction"] = prediction_array
        output["retained_feature_count"] = len(preprocessor.retained_features)
        outputs.append(output)
        if progress_sink is not None:
            progress_sink.append(output)
    if not outputs:
        raise ValueError(f"BLOCKED_BY_VALIDATION: no OOF rows for {family}/{task}/{horizon}d")
    combined = pd.concat(outputs, ignore_index=True)
    key = ["symbol", "trading_date", "horizon", "model_family", "task", "seed"]
    if combined.duplicated(key).any():
        raise RuntimeError("OOF predictions contain duplicate model-row identities")
    if not np.isfinite(combined["prediction"].to_numpy(dtype=float)).all():
        raise RuntimeError("model emitted non-finite OOF predictions")
    return combined.sort_values(key, kind="stable").reset_index(drop=True)


def bounded_optuna_search(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    target_column: str,
    label_end_column: str,
    horizon: int,
    family: ModelFamily,
    config: AdvancedResearchConfig,
) -> TuningResult:
    """Tune on development OOF only; this function receives no holdout rows."""

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        try:
            parameters = _suggest_parameters(trial, family)
            oof = generate_oof_predictions(
                frame,
                feature_names=feature_names,
                target_column=target_column,
                label_end_column=label_end_column,
                horizon=horizon,
                family=family,
                task="regression",
                seed=config.seeds[0],
                parameters=parameters,
                config=config,
            )
            metrics = evaluate_cross_sectional_predictions(
                dates=oof["trading_date"],
                target=oof["target"],
                prediction=oof["prediction"],
            )
            return (
                metrics.mean_daily_rank_ic if metrics.mean_daily_rank_ic is not None else -1.0
            )
        except Exception as exc:
            trial.set_user_attr(
                "failure_reason", f"{type(exc).__name__}: {str(exc)[:500]}"
            )
            raise

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.seeds[0]),
    )
    study.optimize(
        objective,
        n_trials=config.tuning_trials,
        timeout=config.tuning_timeout_seconds,
        n_jobs=1,
        show_progress_bar=False,
        gc_after_trial=True,
        catch=(Exception,),
    )
    trial_audits = tuple(
        TrialAudit(
            number=trial.number,
            state=trial.state.name,  # type: ignore[arg-type]
            parameters={key: value for key, value in trial.params.items()},
            value=float(trial.value) if trial.value is not None else None,
            duration_seconds=(
                trial.duration.total_seconds() if trial.duration is not None else None
            ),
            failure_reason=(
                str(trial.user_attrs.get("failure_reason"))
                if trial.user_attrs.get("failure_reason") is not None
                else None
            ),
        )
        for trial in study.trials
    )
    completed = [trial for trial in study.trials if trial.state is optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise TuningSearchError(
            f"BLOCKED_BY_VALIDATION: tuning failed for {family}/{horizon}d",
            horizon=horizon,
            model_family=family,
            trials=trial_audits,
        )
    return TuningResult(
        horizon=horizon,
        model_family=family,
        trials_completed=len(completed),
        timeout_seconds=config.tuning_timeout_seconds,
        best_value=float(study.best_value),
        best_parameters={key: value for key, value in study.best_params.items()},
        trials=trial_audits,
    )


def _fit_predict(
    *,
    family: ModelFamily,
    task: ModelTask,
    train_x: pd.DataFrame,
    train_target: pd.Series,
    train_dates: pd.Series,
    validation_x: pd.DataFrame,
    seed: int,
    parameters: Mapping[str, int | float],
    config: AdvancedResearchConfig,
) -> np.ndarray:
    if task == "ranking":
        relevance = cross_sectional_relevance(train_target, train_dates)
        group_sizes = tuple(
            int(value)
            for value in pd.to_datetime(train_dates).groupby(pd.to_datetime(train_dates)).size()
        )
        if family == "lightgbm":
            constructor: Any = lgb.LGBMRanker
            model = constructor(
                **{
                    "objective": "lambdarank",
                    "n_estimators": config.estimator_count,
                    "random_state": seed,
                    "n_jobs": 1,
                    "verbosity": -1,
                    "deterministic": True,
                    "force_col_wise": True,
                    **_model_parameters(family, parameters),
                }
            )
            model.fit(train_x, relevance, group=list(group_sizes))
            return np.asarray(model.predict(validation_x), dtype=float)
        if family == "xgboost":
            qid = pd.factorize(pd.to_datetime(train_dates), sort=True)[0]
            model = xgb.XGBRanker(
                objective="rank:ndcg",
                n_estimators=config.estimator_count,
                random_state=seed,
                n_jobs=1,
                tree_method="hist",
                **_model_parameters(family, parameters),
            )
            model.fit(train_x, relevance, qid=qid, verbose=False)
            return np.asarray(model.predict(validation_x), dtype=float)
        group_id = pd.factorize(pd.to_datetime(train_dates), sort=True)[0]
        model = CatBoostRanker(
            loss_function="YetiRank",
            iterations=config.estimator_count,
            random_seed=seed,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            **_model_parameters(family, parameters),
        )
        model.fit(train_x, relevance, group_id=group_id)
        return np.asarray(model.predict(validation_x), dtype=float)

    if task == "large_loss":
        binary_target = (train_target <= config.large_loss_threshold).astype(int)
        if binary_target.nunique() < 2:
            return np.full(len(validation_x), float(binary_target.iloc[0]), dtype=float)
        model = _classifier(family, seed=seed, parameters=parameters, config=config)
        model.fit(train_x, binary_target)
        probability = model.predict_proba(validation_x)
        return np.asarray(probability[:, 1], dtype=float)

    model = _regressor(family, task=task, seed=seed, parameters=parameters, config=config)
    model.fit(train_x, train_target)
    return np.asarray(model.predict(validation_x), dtype=float)


def _regressor(
    family: ModelFamily,
    *,
    task: Literal["regression", "quantile"],
    seed: int,
    parameters: Mapping[str, int | float],
    config: AdvancedResearchConfig,
) -> Any:
    common = _model_parameters(family, parameters)
    if family == "lightgbm":
        objective = "quantile" if task == "quantile" else "regression"
        constructor: Any = lgb.LGBMRegressor
        return constructor(
            **{
                "objective": objective,
                "alpha": config.quantile_alpha if task == "quantile" else 0.9,
                "n_estimators": config.estimator_count,
                "random_state": seed,
                "n_jobs": 1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
                **common,
            }
        )
    if family == "xgboost":
        objective = "reg:quantileerror" if task == "quantile" else "reg:squarederror"
        kwargs: dict[str, Any] = {
            "objective": objective,
            "n_estimators": config.estimator_count,
            "random_state": seed,
            "n_jobs": 1,
            "tree_method": "hist",
            **common,
        }
        if task == "quantile":
            kwargs["quantile_alpha"] = config.quantile_alpha
        return xgb.XGBRegressor(**kwargs)
    loss = f"Quantile:alpha={config.quantile_alpha}" if task == "quantile" else "RMSE"
    return CatBoostRegressor(
        loss_function=loss,
        iterations=config.estimator_count,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        **common,
    )


def _classifier(
    family: ModelFamily,
    *,
    seed: int,
    parameters: Mapping[str, int | float],
    config: AdvancedResearchConfig,
) -> Any:
    common = _model_parameters(family, parameters)
    if family == "lightgbm":
        constructor: Any = lgb.LGBMClassifier
        return constructor(
            **{
                "objective": "binary",
                "n_estimators": config.estimator_count,
                "random_state": seed,
                "n_jobs": 1,
                "verbosity": -1,
                "deterministic": True,
                "force_col_wise": True,
                **common,
            }
        )
    if family == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=config.estimator_count,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            **common,
        )
    return CatBoostClassifier(
        loss_function="Logloss",
        iterations=config.estimator_count,
        random_seed=seed,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        **common,
    )


def _suggest_parameters(trial: optuna.Trial, family: ModelFamily) -> dict[str, int | float]:
    values: dict[str, int | float] = {
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "depth": trial.suggest_int("depth", 2, 6),
        "l2": trial.suggest_float("l2", 1e-3, 10.0, log=True),
    }
    if family != "catboost":
        values["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
        values["column_fraction"] = trial.suggest_float("column_fraction", 0.6, 1.0)
    return values


def _model_parameters(
    family: ModelFamily, parameters: Mapping[str, int | float]
) -> dict[str, int | float]:
    learning_rate = float(parameters.get("learning_rate", 0.05))
    depth = int(parameters.get("depth", 4))
    l2 = float(parameters.get("l2", 1.0))
    if family == "lightgbm":
        return {
            "learning_rate": learning_rate,
            "max_depth": depth,
            "num_leaves": min(2**depth, 63),
            "reg_lambda": l2,
            "subsample": float(parameters.get("subsample", 0.9)),
            "colsample_bytree": float(parameters.get("column_fraction", 0.9)),
        }
    if family == "xgboost":
        return {
            "learning_rate": learning_rate,
            "max_depth": depth,
            "reg_lambda": l2,
            "subsample": float(parameters.get("subsample", 0.9)),
            "colsample_bytree": float(parameters.get("column_fraction", 0.9)),
        }
    return {"learning_rate": learning_rate, "depth": depth, "l2_leaf_reg": l2}


def summarize_oof_predictions(
    oof: pd.DataFrame,
    *,
    quantile_alpha: float,
    large_loss_threshold: float,
) -> AdvancedModelMetrics:
    identity_columns = ["horizon", "model_family", "task", "seed"]
    identities = oof.loc[:, identity_columns].drop_duplicates()
    if len(identities) != 1:
        raise ValueError("OOF summary requires exactly one model identity")
    identity = identities.iloc[0]
    task = str(identity["task"])
    common = {
        "horizon": int(identity["horizon"]),
        "model_family": str(identity["model_family"]),
        "task": task,
        "seed": int(identity["seed"]),
        "folds": int(oof["fold"].nunique()),
        "rows": len(oof),
        "dates": int(pd.to_datetime(oof["trading_date"]).nunique()),
    }
    if task in {"regression", "ranking"}:
        metrics = evaluate_cross_sectional_predictions(
            dates=oof["trading_date"],
            target=oof["target"],
            prediction=oof["prediction"],
        )
        return _ranking_summary(common, metrics)
    if task == "quantile":
        target = oof["target"].to_numpy(dtype=float)
        prediction = oof["prediction"].to_numpy(dtype=float)
        return AdvancedModelMetrics.model_validate(
            {
                **common,
                "pinball_loss": float(mean_pinball_loss(target, prediction, alpha=quantile_alpha)),
                "lower_tail_rate": float(np.mean(target < prediction)),
            }
        )
    target_binary = (oof["target"].to_numpy(dtype=float) <= large_loss_threshold).astype(int)
    probability = np.clip(oof["prediction"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8)
    return AdvancedModelMetrics.model_validate(
        {
            **common,
            "brier_score": float(brier_score_loss(target_binary, probability)),
            "log_loss": float(log_loss(target_binary, probability, labels=[0, 1])),
            "expected_calibration_error": _expected_calibration_error(target_binary, probability),
        }
    )


def _summarize_fold_results(
    oof: pd.DataFrame, *, large_loss_threshold: float
) -> tuple[AdvancedFoldResult, ...]:
    results: list[AdvancedFoldResult] = []
    group_columns = ["horizon", "model_family", "task", "seed", "fold"]
    for identity, group in oof.groupby(group_columns, sort=True):
        horizon, family, task, seed, fold = identity
        target = group["target"].to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        if task == "large_loss":
            target = (target <= large_loss_threshold).astype(float)
        rank_ic: float | None = None
        if task in {"regression", "ranking"}:
            rank_ic = evaluate_cross_sectional_predictions(
                dates=group["trading_date"],
                target=group["target"],
                prediction=group["prediction"],
            ).mean_daily_rank_ic
        dates = pd.to_datetime(group["trading_date"])
        results.append(
            AdvancedFoldResult.model_validate(
                {
                    "horizon": horizon,
                    "model_family": family,
                    "task": task,
                    "seed": seed,
                    "fold": fold,
                    "validation_start": str(dates.min().date()),
                    "validation_end": str(dates.max().date()),
                    "rows": len(group),
                    "mean_squared_error": float(np.mean(np.square(target - prediction))),
                    "mean_daily_rank_ic": rank_ic,
                }
            )
        )
    return tuple(results)


def fit_oof_ensemble(oof: pd.DataFrame, *, horizon: int) -> EnsembleResult:
    """Fit non-negative, sum-one rank stacking from OOF predictions only."""

    components = oof.loc[
        (oof["horizon"] == horizon) & oof["task"].isin(["regression", "ranking"])
    ].copy()
    if components.empty:
        raise ValueError(f"no stackable OOF components for {horizon}d")
    components["component"] = (
        components["model_family"].astype(str)
        + ":"
        + components["task"].astype(str)
        + ":seed="
        + components["seed"].astype(str)
    )
    components["rank_prediction"] = (
        components.groupby(["component", "trading_date"], sort=False)["prediction"].rank(
            method="average", pct=True
        )
        - 0.5
    ) * 2.0
    key = ["symbol", "trading_date", "target", "label_end"]
    wide = components.pivot_table(
        index=key,
        columns="component",
        values="rank_prediction",
        aggfunc="first",
    ).dropna()
    if wide.empty or wide.shape[1] < 2:
        raise ValueError(f"OOF ensemble for {horizon}d requires at least two aligned components")
    names = tuple(str(name) for name in wide.columns)
    prediction_matrix = wide.to_numpy(dtype=float)
    dates = pd.Series(wide.index.get_level_values("trading_date"), index=range(len(wide)))
    label_end = pd.Series(wide.index.get_level_values("label_end"), index=range(len(wide)))
    target = pd.Series(wide.index.get_level_values("target"), index=range(len(wide)), dtype=float)
    standardized_target = within_date_rank_standardize(target, dates)
    meta_fit, _, meta_evaluation = _chronological_meta_masks(dates, label_end=label_end)

    def objective(weights: np.ndarray) -> float:
        return float(
            np.mean(
                np.square(standardized_target[meta_fit] - prediction_matrix[meta_fit] @ weights)
            )
        )

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
    weights = weights / weights.sum()
    ensemble_prediction = prediction_matrix[meta_evaluation] @ weights
    rank_metrics = evaluate_cross_sectional_predictions(
        dates=dates.loc[meta_evaluation].reset_index(drop=True),
        target=target.loc[meta_evaluation].reset_index(drop=True),
        prediction=pd.Series(ensemble_prediction),
    )
    evaluation_matrix = prediction_matrix[meta_evaluation]
    correlations = pd.DataFrame(evaluation_matrix, columns=names).corr()
    upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
    pairwise = float(upper.mean()) if len(upper) else None
    disagreement = np.std(evaluation_matrix, axis=1)
    absolute_error = np.abs(standardized_target[meta_evaluation] - ensemble_prediction)
    uncertainty_error = _finite_correlation(disagreement, absolute_error)
    return EnsembleResult(
        horizon=horizon,
        component_names=names,
        weights=tuple(float(weight) for weight in weights),
        mean_daily_rank_ic=rank_metrics.mean_daily_rank_ic,
        mean_pairwise_correlation=pairwise,
        mean_disagreement=float(np.mean(disagreement)),
        uncertainty_error_correlation=uncertainty_error,
        meta_fit_rows=int(np.sum(meta_fit)),
        meta_evaluation_rows=int(np.sum(meta_evaluation)),
    )


def calibrate_oof_uncertainty(
    oof: pd.DataFrame, *, horizon: int, ensemble: EnsembleResult
) -> UncertaintyCalibration:
    components = oof.loc[
        (oof["horizon"] == horizon) & oof["task"].isin(["regression", "ranking"])
    ].copy()
    components["component"] = (
        components["model_family"].astype(str)
        + ":"
        + components["task"].astype(str)
        + ":seed="
        + components["seed"].astype(str)
    )
    components["rank_prediction"] = (
        components.groupby(["component", "trading_date"], sort=False)["prediction"].rank(
            method="average", pct=True
        )
        - 0.5
    ) * 2.0
    wide = components.pivot_table(
        index=["symbol", "trading_date", "target", "label_end"],
        columns="component",
        values="rank_prediction",
        aggfunc="first",
    ).dropna()
    if tuple(str(name) for name in wide.columns) != ensemble.component_names:
        raise RuntimeError("uncertainty component identity does not match OOF ensemble")
    matrix = wide.to_numpy(dtype=float)
    prediction = matrix @ np.asarray(ensemble.weights, dtype=float)
    dates = pd.Series(wide.index.get_level_values("trading_date"), index=range(len(wide)))
    label_end = pd.Series(wide.index.get_level_values("label_end"), index=range(len(wide)))
    target = pd.Series(wide.index.get_level_values("target"), index=range(len(wide)), dtype=float)
    standardized_target = within_date_rank_standardize(target, dates)
    residual = np.abs(standardized_target - prediction)
    _, calibration, evaluation = _chronological_meta_masks(dates, label_end=label_end)
    q80 = float(np.quantile(residual[calibration], 0.80))
    q90 = float(np.quantile(residual[calibration], 0.90))
    disagreement = np.std(matrix, axis=1)
    return UncertaintyCalibration(
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


def oos_permutation_diagnostics(
    frame: pd.DataFrame,
    *,
    feature_names: tuple[str, ...],
    target_column: str,
    label_end_column: str,
    horizon: int,
    family: ModelFamily,
    parameters: Mapping[str, int | float],
    config: AdvancedResearchConfig,
    validation_not_before: pd.Timestamp | None = None,
) -> tuple[FeatureDiagnostic, ...]:
    """Measure feature damage only on validation rows, never on fitted rows."""

    selected_names = feature_names[: config.diagnostic_feature_limit]
    importance: dict[str, list[float]] = {name: [] for name in selected_names}
    retained_count = dict.fromkeys(selected_names, 0)
    splitter = PurgedExpandingWindowSplitter(
        initial_train_periods=config.initial_train_periods,
        validation_periods=config.validation_periods,
        step_periods=config.step_periods,
        purge_periods=0,
        embargo_periods=horizon,
        label_horizon_periods=horizon,
    )
    fold_count = 0
    for fold in splitter.split(frame, label_end_column=label_end_column):
        train = frame.iloc[list(fold.train_indices)].copy()
        validation = frame.iloc[list(fold.validation_indices)].copy()
        train = train.loc[train[target_column].notna()].sort_values(
            ["trading_date", "symbol"], kind="stable"
        )
        validation = validation.loc[validation[target_column].notna()].sort_values(
            ["trading_date", "symbol"], kind="stable"
        )
        if validation_not_before is not None:
            validation = validation.loc[
                pd.to_datetime(validation["trading_date"]) >= validation_not_before
            ]
        if train.empty or validation.empty:
            continue
        processor = FoldPreprocessor(
            feature_names,
            lower_quantile=config.clip_lower_quantile,
            upper_quantile=config.clip_upper_quantile,
            correlation_threshold=config.correlation_threshold,
        ).fit(train)
        train_x = processor.transform(train)
        validation_x = processor.transform(validation)
        model = _regressor(
            family,
            task="regression",
            seed=config.seeds[0],
            parameters=parameters,
            config=config,
        )
        model.fit(train_x, train[target_column].astype(float))
        target = validation[target_column].to_numpy(dtype=float)
        baseline = np.asarray(model.predict(validation_x), dtype=float)
        baseline_mse = float(np.mean(np.square(target - baseline)))
        for feature_number, name in enumerate(selected_names):
            if name not in processor.retained_features:
                continue
            retained_count[name] += 1
            permuted = validation_x.copy()
            random = np.random.default_rng(
                config.seeds[0] + fold.fold_number * 10_000 + feature_number
            )
            permuted[name] = random.permutation(permuted[name].to_numpy())
            prediction = np.asarray(model.predict(permuted), dtype=float)
            permuted_mse = float(np.mean(np.square(target - prediction)))
            importance[name].append(permuted_mse - baseline_mse)
        fold_count += 1
    if fold_count == 0:
        raise ValueError("BLOCKED_BY_VALIDATION: no folds for OOS permutation diagnostics")
    diagnostics: list[FeatureDiagnostic] = []
    for name in selected_names:
        values = importance[name]
        diagnostics.append(
            FeatureDiagnostic(
                horizon=horizon,
                model_family=family,
                feature_name=name,
                missing_rate=float(
                    pd.to_numeric(frame[name], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .isna()
                    .mean()
                ),
                retained_fold_fraction=retained_count[name] / fold_count,
                oos_permutation_mse_increase=float(np.mean(values)) if values else None,
                permutation_standard_deviation=(
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                ),
            )
        )
    return tuple(diagnostics)


def summarize_seed_stability(
    oof: pd.DataFrame,
    metrics: tuple[AdvancedModelMetrics, ...],
) -> tuple[SeedStability, ...]:
    summaries: list[SeedStability] = []
    stackable = oof.loc[oof["task"].isin(["regression", "ranking"])].copy()
    for identity, group in stackable.groupby(["horizon", "model_family", "task"], sort=True):
        horizon, family, task = identity
        seed_values = tuple(int(value) for value in sorted(group["seed"].unique()))
        rank_ic_values = [
            item.mean_daily_rank_ic
            for item in metrics
            if item.horizon == int(str(horizon))
            and item.model_family == str(family)
            and item.task == str(task)
            and item.mean_daily_rank_ic is not None
        ]
        wide = group.pivot_table(
            index=["symbol", "trading_date"],
            columns="seed",
            values="prediction",
            aggfunc="first",
        ).dropna()
        correlations = wide.corr()
        upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)).stack()
        summaries.append(
            SeedStability.model_validate(
                {
                    "horizon": int(str(horizon)),
                    "model_family": str(family),
                    "task": str(task),
                    "seeds": seed_values,
                    "rank_ic_standard_deviation_across_seeds": (
                        float(np.std(rank_ic_values, ddof=1)) if len(rank_ic_values) > 1 else None
                    ),
                    "mean_pairwise_prediction_correlation": (
                        float(upper.mean()) if len(upper) else None
                    ),
                }
            )
        )
    return tuple(summaries)


def build_decision_compatible_oof_predictions(
    oof: pd.DataFrame,
    *,
    model_version: str,
    feature_version: str,
    data_snapshot_id: str,
    target_family: TargetFamily = "return",
    decision_horizon: int = 5,
) -> DecisionCompatiblePredictionBatch:
    """Build typed, causal, absolute-return Decision Engine predictions from OOF rows."""

    if target_family != "return":
        raise ValueError(
            "Decision Engine predictions require absolute returns, not benchmark-excess targets"
        )
    required = {
        "symbol",
        "trading_date",
        "horizon",
        "task",
        "target",
        "prediction",
        "as_of",
        "label_end",
    }
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"decision-compatible OOF input is missing: {', '.join(missing)}")
    observed_horizons = {int(value) for value in oof["horizon"].unique()}
    if observed_horizons != set(HORIZONS):
        raise ValueError("Decision Engine prediction bundle requires complete 1d/5d/20d horizons")
    if decision_horizon not in HORIZONS:
        raise ValueError("decision downside horizon must be one of 1, 5, and 20")
    outputs: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        horizon_frame = oof.loc[oof["horizon"] == horizon]
        key = ["symbol", "trading_date"]
        regression = horizon_frame.loc[horizon_frame["task"] == "regression"]
        quantile = horizon_frame.loc[horizon_frame["task"] == "quantile"]
        large_loss = horizon_frame.loc[horizon_frame["task"] == "large_loss"]
        if regression.empty or quantile.empty or large_loss.empty:
            raise ValueError(f"OOF output contract is incomplete for {horizon}d")
        key_sets = [
            set(map(tuple, task_frame.loc[:, key].drop_duplicates().to_numpy()))
            for task_frame in (regression, quantile, large_loss)
        ]
        if not (key_sets[0] == key_sets[1] == key_sets[2]):
            raise ValueError(f"OOF task rows are not aligned for {horizon}d")
        if regression.groupby(key, sort=False)["as_of"].nunique(dropna=False).max() != 1:
            raise ValueError(f"OOF as_of provenance is incoherent for {horizon}d")
        if regression.groupby(key, sort=False)["label_end"].nunique(dropna=False).max() != 1:
            raise ValueError(f"OOF label_end provenance is incoherent for {horizon}d")
        expected = regression.groupby(key, sort=True)["prediction"].agg(["mean", "std"])
        target = regression.groupby(key, sort=True)["target"].first()
        as_of = regression.groupby(key, sort=True)["as_of"].first()
        label_end = regression.groupby(key, sort=True)["label_end"].first()
        residual = target - expected["mean"]
        residual_frame = residual.rename("residual").reset_index()
        residual_frame = residual_frame.merge(
            label_end.rename("label_end").reset_index(),
            on=key,
            how="inner",
            validate="one_to_one",
        )
        residual_frame["squared"] = residual_frame["residual"].pow(2)
        maturity = residual_frame.groupby("label_end", sort=True)["squared"].agg(
            ["sum", "count"]
        )
        maturity.index = pd.to_datetime(maturity.index)
        maturity = maturity.sort_index()
        cumulative_sum = maturity["sum"].cumsum()
        cumulative_count = maturity["count"].cumsum()
        prediction_dates = pd.DatetimeIndex(
            pd.to_datetime(residual_frame["trading_date"]).sort_values().unique()
        )
        scale_by_date: dict[pd.Timestamp, float] = {}
        for prediction_date in prediction_dates:
            matured = maturity.index < prediction_date
            if not matured.any():
                continue
            last_position = int(np.flatnonzero(matured)[-1])
            scale_by_date[prediction_date] = math.sqrt(
                float(cumulative_sum.iloc[last_position])
                / float(cumulative_count.iloc[last_position])
            )
        output = expected.rename(
            columns={"mean": f"expected_return_{horizon}d", "std": "__disagreement"}
        )
        output = output.join(as_of.rename("as_of"))
        output[f"uncertainty_standard_error_{horizon}d"] = (
            pd.to_datetime(output.reset_index()["trading_date"])
            .map(scale_by_date)
            .to_numpy(dtype=float)
        )
        output[f"model_disagreement_{horizon}d"] = output["__disagreement"].fillna(0.0)
        output[f"uncertainty_status_{horizon}d"] = np.where(
            output[f"uncertainty_standard_error_{horizon}d"].notna(),
            CapabilityStatus.AVAILABLE.value,
            "BLOCKED_BY_CALIBRATION_HISTORY",
        )
        output[f"observed_return_{horizon}d"] = target
        output[f"downside_quantile_{horizon}d"] = quantile.groupby(key, sort=True)[
            "prediction"
        ].mean()
        output[f"large_loss_probability_{horizon}d"] = large_loss.groupby(key, sort=True)[
            "prediction"
        ].mean()
        output = output.drop(columns="__disagreement").reset_index()
        probability = output[f"large_loss_probability_{horizon}d"]
        if not probability.between(0.0, 1.0).all():
            raise RuntimeError("large-loss OOF probability is outside [0, 1]")
        outputs.append(output)
    horizon_key_sets = [
        set(map(tuple, output.loc[:, ["symbol", "trading_date"]].to_numpy()))
        for output in outputs
    ]
    common_keys = set.intersection(*horizon_key_sets)
    if not common_keys:
        raise ValueError("OOF horizons have no common Decision Engine prediction rows")
    union_keys = set.union(*horizon_key_sets)
    missing_horizon_blocks: list[DecisionPredictionBlock] = []
    if union_keys != common_keys:
        provenance = (
            oof.sort_values(["trading_date", "symbol"])
            .groupby(["symbol", "trading_date"], sort=True)["as_of"]
            .first()
        )
        missing_keys = sorted(
            union_keys - common_keys, key=lambda key: (key[1], key[0])
        )
        for symbol, trading_date in missing_keys:
            timestamp = pd.Timestamp(provenance.loc[(symbol, trading_date)])
            if timestamp.tzinfo is None:
                raise ValueError("Decision Engine OOF as_of must be timezone-aware")
            missing_horizon_blocks.append(
                DecisionPredictionBlock(
                    symbol=str(symbol),
                    as_of=timestamp.to_pydatetime(),
                    reason="BLOCKED_BY_HORIZON_ALIGNMENT: 1d/5d/20d OOF rows do not overlap",
                )
            )

    def common_rows(frame: pd.DataFrame) -> pd.DataFrame:
        keys = pd.MultiIndex.from_frame(frame.loc[:, ["symbol", "trading_date"]])
        return frame.loc[keys.isin(common_keys)].copy()

    combined = common_rows(outputs[0])
    for raw_output in outputs[1:]:
        output = common_rows(raw_output)
        as_of_check = combined.loc[:, ["symbol", "trading_date", "as_of"]].merge(
            output.loc[:, ["symbol", "trading_date", "as_of"]],
            on=["symbol", "trading_date"],
            how="inner",
            suffixes=("_existing", "_new"),
            validate="one_to_one",
        )
        if not pd.to_datetime(as_of_check["as_of_existing"], utc=True).equals(
            pd.to_datetime(as_of_check["as_of_new"], utc=True)
        ):
            raise ValueError("OOF horizon as_of provenance is not aligned")
        output = output.drop(columns="as_of")
        combined = combined.merge(
            output,
            on=["symbol", "trading_date"],
            how="inner",
            validate="one_to_one",
        )
    combined = combined.sort_values(["trading_date", "symbol"], kind="stable").reset_index(
        drop=True
    )
    predictions: list[Prediction] = []
    blocked: list[DecisionPredictionBlock] = list(missing_horizon_blocks)
    status_columns = [f"uncertainty_status_{horizon}d" for horizon in HORIZONS]
    for row in combined.to_dict(orient="records"):
        timestamp = pd.Timestamp(row["as_of"])
        if timestamp.tzinfo is None:
            raise ValueError("Decision Engine OOF as_of must be timezone-aware")
        as_of_value = timestamp.to_pydatetime()
        unavailable = [column for column in status_columns if row[column] != "AVAILABLE"]
        if unavailable:
            blocked.append(
                DecisionPredictionBlock(
                    symbol=str(row["symbol"]),
                    as_of=as_of_value,
                    reason="BLOCKED_BY_CALIBRATION_HISTORY: no strictly prior OOF residuals",
                )
            )
            continue
        predictions.append(
            Prediction(
                symbol=str(row["symbol"]),
                as_of=as_of_value,
                expected_return_1d=float(row["expected_return_1d"]),
                expected_return_5d=float(row["expected_return_5d"]),
                expected_return_20d=float(row["expected_return_20d"]),
                downside_quantile=float(row[f"downside_quantile_{decision_horizon}d"]),
                large_loss_probability=float(
                    row[f"large_loss_probability_{decision_horizon}d"]
                ),
                uncertainty=PredictionUncertainty(
                    standard_error=float(
                        row[f"uncertainty_standard_error_{decision_horizon}d"]
                    ),
                    model_disagreement=float(
                        row[f"model_disagreement_{decision_horizon}d"]
                    ),
                    coverage_warning="expanding past-only OOF residual calibration",
                ),
                model_version=model_version,
                feature_version=feature_version,
                data_snapshot_id=data_snapshot_id,
            )
        )
    if not predictions:
        raise ValueError("BLOCKED_BY_CALIBRATION_HISTORY: no typed prediction row is usable")
    return DecisionCompatiblePredictionBatch(
        predictions=tuple(predictions), blocked=tuple(blocked)
    )


_FEATURE_ABLATION_FAMILIES: Mapping[str, tuple[str, tuple[str, ...]]] = MappingProxyType(
    {
        "F1": ("MA / long-term position", ("moving_average", "price_position")),
        "F2": ("MACD", ("macd",)),
        "F3": ("RSI / oscillator", ("rsi",)),
        "F4": ("Bollinger", ("bollinger",)),
        "F5": ("ADX / DI", ("directional_movement",)),
        "F6": ("Volatility / downside", ("volatility",)),
        "F7": ("Volume / money flow", ("volume", "money_flow", "liquidity")),
        "F8": ("Valuation / quality / growth", ("fundamental",)),
        "F9": ("Revision / surprise", ("forecast_revision", "earnings_surprise")),
        "F10": ("Relative / market context", ("relative_strength", "market_context")),
        "F11": ("Supply / demand", ("supply_demand",)),
        "F12": ("Candle / breakout", ("candle", "breakout")),
    }
)


def feature_family_ablation_plan(
    feature_names: tuple[str, ...] = V2_EXTENDED_MANIFEST.feature_names,
) -> tuple[FeatureAblationResult, ...]:
    selected = set(feature_names)
    definitions = {
        name: FEATURE_REGISTRY.definition(name) for name in feature_names if name in selected
    }
    output: list[FeatureAblationResult] = []
    for family_id, (family_name, catalog_families) in _FEATURE_ABLATION_FAMILIES.items():
        added = tuple(
            name
            for name, definition in definitions.items()
            if definition.family in set(catalog_families)
            or any(token in name for token in catalog_families)
        )
        if family_id == "F8":
            added = tuple(
                name
                for name in added
                if "forecast_revision" not in name and "surprise" not in name
            )
        available = bool(added)
        output.append(
            FeatureAblationResult(
                family_id=family_id,
                family_name=family_name,
                status=(
                    CapabilityStatus.AVAILABLE
                    if available
                    else CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY
                ),
                added_features=added,
                blocking_reason=None if available else "feature family is absent from V1 Core",
            )
        )
    return tuple(output)


@dataclass(frozen=True)
class _ResearchStageFrames:
    tuning: pd.DataFrame
    model_evaluation_start: pd.Timestamp
    boundary: ResearchStageBoundary


def _split_research_stages(
    development: pd.DataFrame,
    *,
    horizon: int,
    label_end_column: str,
    config: AdvancedResearchConfig,
) -> _ResearchStageFrames:
    """Reserve a later outer OOF period never seen by hyperparameter selection."""

    dates = pd.DatetimeIndex(pd.to_datetime(development["trading_date"]).sort_values().unique())
    minimum_index = (
        config.initial_train_periods
        + config.validation_periods
        + (2 * horizon)
    )
    evaluation_index = max(minimum_index, math.ceil(len(dates) * 0.60))
    if evaluation_index + config.validation_periods > len(dates):
        raise ValueError(
            f"BLOCKED_BY_VALIDATION: {horizon}d development history cannot separate "
            "tuning from outer model evaluation"
        )
    evaluation_start = dates[evaluation_index]
    trading_date = pd.to_datetime(development["trading_date"])
    label_end = pd.to_datetime(development[label_end_column])
    tuning = development.loc[
        (trading_date < evaluation_start)
        & label_end.notna()
        & (label_end < evaluation_start)
    ].copy()
    if tuning.empty:
        raise ValueError(f"BLOCKED_BY_VALIDATION: no causal tuning rows for {horizon}d")
    tuning_dates = pd.DatetimeIndex(pd.to_datetime(tuning["trading_date"]).sort_values().unique())
    required_tuning_dates = (
        config.initial_train_periods + horizon + config.validation_periods
    )
    if len(tuning_dates) < required_tuning_dates:
        raise ValueError(
            f"BLOCKED_BY_VALIDATION: {horizon}d tuning history has "
            f"{len(tuning_dates)} dates; requires at least {required_tuning_dates}"
        )
    evaluation_rows = int((trading_date >= evaluation_start).sum())
    return _ResearchStageFrames(
        tuning=tuning,
        model_evaluation_start=evaluation_start,
        boundary=ResearchStageBoundary(
            horizon=horizon,
            tuning_end=str(tuning_dates[-1].date()),
            model_evaluation_start=str(evaluation_start.date()),
            tuning_rows=len(tuning),
            evaluation_candidate_rows=evaluation_rows,
        ),
    )


def _estimated_fold_count(
    frame: pd.DataFrame, *, horizon: int, config: AdvancedResearchConfig
) -> int:
    date_count = pd.to_datetime(frame["trading_date"]).nunique()
    first_validation = config.initial_train_periods + horizon
    if first_validation + config.validation_periods > date_count:
        return 0
    return 1 + (
        date_count - first_validation - config.validation_periods
    ) // config.step_periods


def run_advanced_research(
    dataset: pd.DataFrame,
    *,
    data_snapshot_id: str,
    created_at: datetime,
    code_commit: str,
    config: AdvancedResearchConfig,
    feature_snapshot_id: str,
    feature_manifest_hash: str,
    feature_names: tuple[str, ...] = V2_EXTENDED_MANIFEST.feature_names,
) -> AdvancedResearchRun:
    """Run Goal 3 research and preserve partial progress if a later stage fails."""

    progress = _AdvancedResearchProgress()
    try:
        return _run_advanced_research(
            dataset,
            data_snapshot_id=data_snapshot_id,
            created_at=created_at,
            code_commit=code_commit,
            config=config,
            feature_snapshot_id=feature_snapshot_id,
            feature_manifest_hash=feature_manifest_hash,
            feature_names=feature_names,
            progress=progress,
        )
    except Exception as exc:
        if isinstance(exc, AdvancedResearchExecutionError):
            raise
        trial_contexts = (
            exc.trial_contexts
            if isinstance(exc, TuningSearchError)
            else tuple(
                (result.horizon, result.model_family, trial)
                for result in progress.tuning_results
                for trial in result.trials
            )
        )
        fold_results: tuple[AdvancedFoldResult, ...] = ()
        if progress.oof_parts:
            try:
                partial_oof = pd.concat(progress.oof_parts, ignore_index=True)
                fold_results = _summarize_fold_results(
                    partial_oof,
                    large_loss_threshold=config.large_loss_threshold,
                )
            except Exception:
                fold_results = ()
        raise AdvancedResearchExecutionError(
            str(exc),
            trial_contexts=trial_contexts,
            fold_results=fold_results,
        ) from exc


def _run_advanced_research(
    dataset: pd.DataFrame,
    *,
    data_snapshot_id: str,
    created_at: datetime,
    code_commit: str,
    config: AdvancedResearchConfig,
    feature_snapshot_id: str,
    feature_manifest_hash: str,
    feature_names: tuple[str, ...],
    progress: _AdvancedResearchProgress,
) -> AdvancedResearchRun:
    """Internal implementation with progress containers owned by the public boundary."""

    _validate_run_inputs(
        dataset,
        data_snapshot_id=data_snapshot_id,
        created_at=created_at,
        code_commit=code_commit,
        feature_names=feature_names,
        feature_snapshot_id=feature_snapshot_id,
        feature_manifest_hash=feature_manifest_hash,
    )
    revision_policy, revision_status = _revision_contract(dataset)
    holdout = reserve_locked_final_holdout(dataset, holdout_periods=config.holdout_periods)
    tuning_results = progress.tuning_results
    oof_parts = progress.oof_parts
    ablation_results: list[FeatureAblationResult] = []
    feature_diagnostics: list[FeatureDiagnostic] = []
    stage_boundaries: list[ResearchStageBoundary] = []
    estimated_oof_rows = 0
    estimated_model_fits = 0

    for horizon in config.horizons:
        target_column, status_column = _target_contract(config.target_family, horizon)
        label_end_column = f"label_end_date_{horizon}d"
        required = {target_column, status_column, label_end_column}
        missing = sorted(required - set(dataset.columns))
        if missing:
            raise ValueError(f"production research dataset is missing: {', '.join(missing)}")
        usable = dataset.loc[
            dataset[target_column].notna() & dataset[status_column].eq("AVAILABLE")
        ].copy()
        development = usable.loc[
            (pd.to_datetime(usable["trading_date"]) < holdout.holdout_start)
            & (pd.to_datetime(usable[label_end_column]) < holdout.holdout_start)
        ].copy()
        if development.empty:
            raise ValueError(f"BLOCKED_BY_VALIDATION: holdout purge removed all {horizon}d labels")
        stages = _split_research_stages(
            development,
            horizon=horizon,
            label_end_column=label_end_column,
            config=config,
        )
        stage_boundaries.append(stages.boundary)
        estimated_oof_rows += (
            len(development)
            * len(config.model_families)
            * len(_TASKS)
            * len(config.seeds)
        )
        if estimated_oof_rows > config.max_materialized_oof_rows:
            raise ValueError(
                "BLOCKED_BY_RESOURCE_CAPABILITY: estimated OOF materialization "
                f"{estimated_oof_rows:,} rows exceeds configured bound "
                f"{config.max_materialized_oof_rows:,}; run one horizon/model family batch "
                "or raise the explicit bound after a measured scale test"
            )
        tuning_folds = _estimated_fold_count(stages.tuning, horizon=horizon, config=config)
        development_folds = _estimated_fold_count(
            development, horizon=horizon, config=config
        )
        estimated_model_fits += len(config.model_families) * (
            config.tuning_trials * tuning_folds
            + len(_TASKS) * len(config.seeds) * development_folds
        )
        if config.run_ablations:
            estimated_model_fits += (1 + len(config.ablation_families)) * (
                tuning_folds + development_folds
            )
        if config.run_diagnostics:
            estimated_model_fits += development_folds
        if estimated_model_fits > config.max_model_fits:
            raise ValueError(
                "BLOCKED_BY_RESOURCE_CAPABILITY: estimated model fits "
                f"{estimated_model_fits:,} exceed configured bound {config.max_model_fits:,}; "
                "run a smaller horizon/model-family batch or raise the explicit bound after "
                "a measured scale test"
            )
        best_by_family: dict[ModelFamily, Mapping[str, int | float]] = {}
        for family in config.model_families:
            try:
                tuning = bounded_optuna_search(
                    stages.tuning,
                    feature_names=feature_names,
                    target_column=target_column,
                    label_end_column=label_end_column,
                    horizon=horizon,
                    family=family,
                    config=config,
                )
            except TuningSearchError as exc:
                prior_trials = tuple(
                    (result.horizon, result.model_family, trial)
                    for result in tuning_results
                    for trial in result.trials
                )
                raise TuningSearchError(
                    str(exc),
                    horizon=exc.horizon,
                    model_family=exc.model_family,
                    trials=exc.trials,
                    trial_contexts=(*prior_trials, *exc.trial_contexts),
                ) from exc
            tuning_results.append(tuning)
            best_by_family[family] = tuning.best_parameters
            for task in _TASKS:
                for seed in config.seeds:
                    task_target = (
                        f"target_return_{horizon}d" if task == "large_loss" else target_column
                    )
                    if task_target not in development:
                        raise ValueError(f"production research dataset is missing: {task_target}")
                    generate_oof_predictions(
                        development,
                        feature_names=feature_names,
                        target_column=task_target,
                        label_end_column=label_end_column,
                        horizon=horizon,
                        family=family,
                        task=task,
                        seed=seed,
                        parameters=tuning.best_parameters,
                        config=config,
                        validation_not_before=stages.model_evaluation_start,
                        progress_sink=oof_parts,
                    )
        if config.run_ablations:
            ablation_results.extend(
                _run_ablations(
                    development,
                    selection_frame=stages.tuning,
                    target_column=target_column,
                    label_end_column=label_end_column,
                    horizon=horizon,
                    feature_names=feature_names,
                    parameters={},
                    config=config,
                    validation_not_before=stages.model_evaluation_start,
                )
            )
        if config.run_diagnostics:
            feature_diagnostics.extend(
                oos_permutation_diagnostics(
                    development,
                    feature_names=feature_names,
                    target_column=target_column,
                    label_end_column=label_end_column,
                    horizon=horizon,
                    family=config.model_families[0],
                    parameters=best_by_family[config.model_families[0]],
                    config=config,
                    validation_not_before=stages.model_evaluation_start,
                )
            )

    oof = pd.concat(oof_parts, ignore_index=True)
    metrics = tuple(
        summarize_oof_predictions(
            group,
            quantile_alpha=config.quantile_alpha,
            large_loss_threshold=config.large_loss_threshold,
        )
        for _, group in oof.groupby(["horizon", "model_family", "task", "seed"], sort=True)
    )
    ensembles = tuple(fit_oof_ensemble(oof, horizon=horizon) for horizon in config.horizons)
    seed_stability = summarize_seed_stability(oof, metrics)
    uncertainty = tuple(
        calibrate_oof_uncertainty(oof, horizon=ensemble.horizon, ensemble=ensemble)
        for ensemble in ensembles
    )
    oof_hash = _frame_hash(oof)
    fold_results = _summarize_fold_results(
        oof, large_loss_threshold=config.large_loss_threshold
    )
    library_versions = tuple(
        (name, version(name)) for name in ("lightgbm", "xgboost", "catboost", "optuna")
    )
    blocking_reasons: list[str] = []
    if revision_status is not CapabilityStatus.AVAILABLE:
        blocking_reasons.append("historical provider revision vintages are incomplete")
    if config.target_family != "return":
        blocking_reasons.append(
            "benchmark-excess predictions are not absolute-return Decision Engine inputs"
        )
    blocking_reasons.append(
        "locked final holdout remains unopened; this is a development research result"
    )
    report = AdvancedResearchReport(
        report_id="0" * 64,
        created_at=created_at,
        code_commit=code_commit,
        hypothesis=config.hypothesis,
        config=config,
        config_hash=config.config_hash,
        data_snapshot_id=data_snapshot_id,
        feature_snapshot_id=feature_snapshot_id,
        feature_set_id=V2_EXTENDED_MANIFEST.feature_set_id,
        feature_set_version=V2_EXTENDED_MANIFEST.feature_set_version,
        preprocessing_version=V2_EXTENDED_MANIFEST.preprocessing_version,
        feature_manifest_hash=feature_manifest_hash,
        feature_definition_hashes={
            name: V2_EXTENDED_MANIFEST.feature_definition_hashes[name]
            for name in feature_names
        },
        feature_names=feature_names,
        prediction_semantics=config.target_family,
        locked_holdout_start=str(holdout.holdout_start.date()),
        locked_holdout_accessed=False,
        historical_revision_policy=revision_policy,
        historical_revision_status=revision_status,
        adoption_eligible=False,
        adoption_blocking_reasons=tuple(blocking_reasons),
        cost_scenarios_bps=(0, 10, 25, 50),
        cost_evaluation_status=CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY,
        tax_policy_version="NOT_APPLIED_MODEL_RESEARCH",
        decision_engine_version="decision-engine-v1",
        library_versions=library_versions,
        stage_boundaries=tuple(stage_boundaries),
        tuning_results=tuple(tuning_results),
        fold_results=fold_results,
        model_metrics=metrics,
        ablations=tuple(ablation_results),
        ensembles=ensembles,
        uncertainty=uncertainty,
        feature_diagnostics=tuple(feature_diagnostics),
        seed_stability=seed_stability,
        oof_rows=len(oof),
        oof_sha256=oof_hash,
    )
    report = report.model_copy(update={"report_id": _stable_hash(_report_identity(report))})
    return AdvancedResearchRun(report=report, oof_predictions=oof)


def write_advanced_research_run(
    run: AdvancedResearchRun,
    destination: Path,
) -> tuple[Path, Path]:
    """Atomically publish an authenticated report + OOF Parquet directory."""

    if _stable_hash(_report_identity(run.report)) != run.report.report_id:
        raise RuntimeError("advanced research report content identity mismatch")
    if (
        len(run.oof_predictions) != run.report.oof_rows
        or _frame_hash(run.oof_predictions) != run.report.oof_sha256
    ):
        raise RuntimeError("advanced research OOF content identity mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    final_directory = destination / run.report.report_id
    parquet_name = f"{run.report.report_id}.oof.parquet"
    metadata_name = f"{run.report.report_id}.json"
    if final_directory.exists():
        observed_report, observed_oof = load_advanced_research_run(final_directory / parquet_name)
        if observed_report.report_id != run.report.report_id or _frame_hash(
            observed_oof
        ) != _frame_hash(run.oof_predictions):
            raise RuntimeError("advanced research report identity already exists with conflicts")
        return final_directory / metadata_name, final_directory / parquet_name

    temporary = Path(tempfile.mkdtemp(prefix=".advanced-", dir=destination))
    try:
        parquet_path = temporary / parquet_name
        metadata_path = temporary / metadata_name
        run.oof_predictions.to_parquet(parquet_path, index=False)
        parquet_sha256 = _file_sha256(parquet_path)
        payload = {
            "report": run.report.model_dump(mode="json"),
            "parquet_path": str((final_directory / parquet_name).resolve()),
            "parquet_sha256": parquet_sha256,
        }
        payload["metadata_hash"] = _stable_hash(payload)
        metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, final_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return final_directory / metadata_name, final_directory / parquet_name


def load_advanced_research_run(
    parquet_path: Path,
) -> tuple[AdvancedResearchReport, pd.DataFrame]:
    """Authenticate an advanced report bundle before downstream use."""

    parquet_path = parquet_path.resolve()
    report_id = parquet_path.name.removesuffix(".oof.parquet")
    if parquet_path.parent.name != report_id:
        raise RuntimeError("advanced research path is not content-addressed")
    metadata_path = parquet_path.parent / f"{report_id}.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise RuntimeError("advanced research metadata is missing or invalid") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("report"), dict):
        raise RuntimeError("advanced research metadata is invalid")
    authenticated_metadata = {
        key: value for key, value in payload.items() if key != "metadata_hash"
    }
    if payload.get("metadata_hash") != _stable_hash(authenticated_metadata):
        raise RuntimeError("advanced research metadata hash mismatch")
    if Path(str(payload.get("parquet_path", ""))).resolve() != parquet_path:
        raise RuntimeError("advanced research Parquet path metadata mismatch")
    if _file_sha256(parquet_path) != str(payload.get("parquet_sha256", "")):
        raise RuntimeError("advanced research Parquet hash mismatch")
    report = AdvancedResearchReport.model_validate(payload["report"])
    if report.report_id != report_id:
        raise RuntimeError("advanced research report identity mismatch")
    frame = pd.read_parquet(parquet_path)
    if len(frame) != report.oof_rows or _frame_hash(frame) != report.oof_sha256:
        raise RuntimeError("advanced research OOF content identity mismatch")
    if _stable_hash(_report_identity(report)) != report.report_id:
        raise RuntimeError("advanced research report content identity mismatch")
    return report, frame


def _run_ablations(
    development: pd.DataFrame,
    *,
    selection_frame: pd.DataFrame,
    target_column: str,
    label_end_column: str,
    horizon: int,
    feature_names: tuple[str, ...],
    parameters: Mapping[str, int | float],
    config: AdvancedResearchConfig,
    validation_not_before: pd.Timestamp,
) -> tuple[FeatureAblationResult, ...]:
    plans = {
        item.family_id: item
        for item in feature_family_ablation_plan(feature_names)
        if item.family_id in config.ablation_families
    }
    baseline_features = tuple(name for name in V0_MANIFEST.feature_names if name in feature_names)
    if not baseline_features:
        raise ValueError("feature ablation baseline has no available features")
    family = config.model_families[0]
    baseline_selection_oof = generate_oof_predictions(
        selection_frame,
        feature_names=baseline_features,
        target_column=target_column,
        label_end_column=label_end_column,
        horizon=horizon,
        family=family,
        task="regression",
        seed=config.seeds[0],
        parameters=parameters,
        config=config,
    )
    baseline_selection_metric = evaluate_cross_sectional_predictions(
        dates=baseline_selection_oof["trading_date"],
        target=baseline_selection_oof["target"],
        prediction=baseline_selection_oof["prediction"],
    ).mean_daily_rank_ic
    baseline_evaluation_oof = generate_oof_predictions(
        development,
        feature_names=baseline_features,
        target_column=target_column,
        label_end_column=label_end_column,
        horizon=horizon,
        family=family,
        task="regression",
        seed=config.seeds[0],
        parameters=parameters,
        config=config,
        validation_not_before=validation_not_before,
    )
    baseline_evaluation_metric = evaluate_cross_sectional_predictions(
        dates=baseline_evaluation_oof["trading_date"],
        target=baseline_evaluation_oof["target"],
        prediction=baseline_evaluation_oof["prediction"],
    ).mean_daily_rank_ic
    results: list[FeatureAblationResult] = [
        FeatureAblationResult(
            family_id="F0",
            family_name="FeatureSet V0 baseline",
            horizon=horizon,
            status=CapabilityStatus.AVAILABLE,
            added_features=baseline_features,
            mean_daily_rank_ic=baseline_evaluation_metric,
            incremental_rank_ic=None,
            selection_rank_ic=baseline_selection_metric,
            selected_on_tuning_period=True,
        )
    ]
    champion_features = baseline_features
    champion_selection_metric = baseline_selection_metric
    champion_evaluation_metric = baseline_evaluation_metric
    for family_id in config.ablation_families:
        plan = plans[family_id]
        if plan.status is not CapabilityStatus.AVAILABLE:
            results.append(plan.model_copy(update={"horizon": horizon}))
            continue
        new_features = tuple(name for name in plan.added_features if name not in champion_features)
        if not new_features:
            results.append(
                plan.model_copy(
                    update={
                        "horizon": horizon,
                        "status": CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY,
                        "blocking_reason": "family adds no feature beyond the preceding champion",
                    }
                )
            )
            continue
        experiment_features = tuple(dict.fromkeys((*champion_features, *new_features)))
        selection_oof = generate_oof_predictions(
            selection_frame,
            feature_names=experiment_features,
            target_column=target_column,
            label_end_column=label_end_column,
            horizon=horizon,
            family=family,
            task="regression",
            seed=config.seeds[0],
            parameters=parameters,
            config=config,
        )
        selection_score = evaluate_cross_sectional_predictions(
            dates=selection_oof["trading_date"],
            target=selection_oof["target"],
            prediction=selection_oof["prediction"],
        ).mean_daily_rank_ic
        selected = selection_score is not None and (
            champion_selection_metric is None
            or selection_score > champion_selection_metric
        )
        evaluation_oof = generate_oof_predictions(
            development,
            feature_names=experiment_features,
            target_column=target_column,
            label_end_column=label_end_column,
            horizon=horizon,
            family=family,
            task="regression",
            seed=config.seeds[0],
            parameters=parameters,
            config=config,
            validation_not_before=validation_not_before,
        )
        evaluation_score = evaluate_cross_sectional_predictions(
            dates=evaluation_oof["trading_date"],
            target=evaluation_oof["target"],
            prediction=evaluation_oof["prediction"],
        ).mean_daily_rank_ic
        incremental = (
            evaluation_score - champion_evaluation_metric
            if evaluation_score is not None and champion_evaluation_metric is not None
            else None
        )
        results.append(
            plan.model_copy(
                update={
                    "horizon": horizon,
                    "added_features": new_features,
                    "mean_daily_rank_ic": evaluation_score,
                    "incremental_rank_ic": incremental,
                    "selection_rank_ic": selection_score,
                    "selected_on_tuning_period": selected,
                }
            )
        )
        if selected:
            champion_features = experiment_features
            champion_selection_metric = selection_score
            champion_evaluation_metric = evaluation_score
    return tuple(results)


def _ranking_summary(common: Mapping[str, object], metrics: RankingMetrics) -> AdvancedModelMetrics:
    return AdvancedModelMetrics.model_validate(
        {
            **common,
            "mean_squared_error": metrics.mean_squared_error,
            "mean_daily_rank_ic": metrics.mean_daily_rank_ic,
            "rank_ic_standard_deviation": metrics.rank_ic_standard_deviation,
            "rank_icir": metrics.rank_icir,
            "ndcg_at_5": metrics.ndcg_at_5,
            "ndcg_at_10": metrics.ndcg_at_10,
            "ndcg_at_20": metrics.ndcg_at_20,
            "precision_at_5": metrics.precision_at_5,
            "precision_at_10": metrics.precision_at_10,
            "precision_at_20": metrics.precision_at_20,
            "top_5_mean_target": metrics.top_5_mean_target,
            "top_10_mean_target": metrics.top_10_mean_target,
            "top_20_mean_target": metrics.top_20_mean_target,
        }
    )


def _validate_run_inputs(
    dataset: pd.DataFrame,
    *,
    data_snapshot_id: str,
    created_at: datetime,
    code_commit: str,
    feature_names: tuple[str, ...],
    feature_snapshot_id: str,
    feature_manifest_hash: str,
) -> None:
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("advanced research created_at must be timezone-aware")
    if len(data_snapshot_id) != 64:
        raise ValueError("advanced research requires an authenticated data snapshot ID")
    if len(feature_snapshot_id) != 64:
        raise ValueError("advanced research requires an authenticated V2 feature snapshot ID")
    if len(feature_manifest_hash) != 64 or any(
        character not in "0123456789abcdef" for character in feature_manifest_hash
    ):
        raise ValueError("advanced research requires an authenticated V2 feature manifest hash")
    if not code_commit.strip() or code_commit == "UNSET":
        raise ValueError("advanced research code_commit must be explicit")
    if dataset.empty:
        raise ValueError("advanced research dataset cannot be empty")
    required = {
        "symbol",
        "trading_date",
        "historical_revision_policy",
        "historical_revision_status",
        *feature_names,
    }
    missing = sorted(required - set(dataset.columns))
    if missing:
        raise ValueError(f"advanced research dataset is missing: {', '.join(missing)}")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("advanced research feature names must be unique")
    if not set(feature_names) <= set(V2_EXTENDED_MANIFEST.feature_names):
        raise ValueError("advanced research feature names must belong to V2 Extended")
    if dataset.duplicated(["symbol", "trading_date"]).any():
        raise ValueError("advanced research dataset has duplicate symbol-date rows")


def _target_contract(target_family: TargetFamily, horizon: int) -> tuple[str, str]:
    if target_family == "return":
        return f"target_return_{horizon}d", f"label_status_{horizon}d"
    return (
        f"target_{target_family}_{horizon}d",
        f"label_status_{target_family}_{horizon}d",
    )


def _revision_contract(dataset: pd.DataFrame) -> tuple[str, CapabilityStatus]:
    policies = tuple(
        str(value) for value in dataset["historical_revision_policy"].dropna().unique()
    )
    statuses = tuple(
        str(value) for value in dataset["historical_revision_status"].dropna().unique()
    )
    if len(policies) != 1 or len(statuses) != 1:
        raise ValueError("production research revision contract must be coherent")
    return policies[0], CapabilityStatus(statuses[0])


def _report_identity(report: AdvancedResearchReport) -> dict[str, object]:
    payload = report.model_dump(mode="json", exclude={"report_id", "created_at"})
    tuning_results = payload.get("tuning_results", [])
    if isinstance(tuning_results, list):
        for tuning in tuning_results:
            if not isinstance(tuning, dict):
                continue
            trials = tuning.get("trials", [])
            if isinstance(trials, list):
                for trial in trials:
                    if isinstance(trial, dict):
                        trial.pop("duration_seconds", None)
    return payload


def _finite_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce").astype(float)
    return pd.DataFrame(numeric.replace([np.inf, -np.inf], np.nan))


def _chronological_meta_masks(
    dates: pd.Series,
    *,
    label_end: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = pd.to_datetime(dates)
    normalized_label_end = pd.to_datetime(label_end)
    unique_dates = pd.DatetimeIndex(normalized.sort_values().unique())
    if len(unique_dates) < 3:
        raise ValueError("OOF meta fitting requires at least three validation dates")
    fit_date_count = min(len(unique_dates) - 2, max(1, math.ceil(len(unique_dates) * 0.50)))
    calibration_end = min(
        len(unique_dates) - 1,
        max(fit_date_count + 1, math.ceil(len(unique_dates) * 0.75)),
    )
    fit_dates = set(unique_dates[:fit_date_count])
    calibration_dates = set(unique_dates[fit_date_count:calibration_end])
    calibration_start = unique_dates[fit_date_count]
    evaluation_start = unique_dates[calibration_end]
    fit = (
        normalized.isin(fit_dates) & (normalized_label_end < calibration_start)
    ).to_numpy(dtype=bool)
    calibration = (
        normalized.isin(calibration_dates) & (normalized_label_end < evaluation_start)
    ).to_numpy(dtype=bool)
    evaluation = (normalized >= evaluation_start).to_numpy(dtype=bool)
    if not fit.any() or not calibration.any() or not evaluation.any():
        raise RuntimeError("OOF meta fit/calibration/evaluation split is empty")
    return fit, calibration, evaluation


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray) -> float:
    bins = np.linspace(0.0, 1.0, 11)
    total = len(target)
    error = 0.0
    for index in range(len(bins) - 1):
        lower = bins[index]
        upper = bins[index + 1]
        selected = (probability >= lower) & (
            probability <= upper if index == len(bins) - 2 else probability < upper
        )
        if not selected.any():
            continue
        observed = float(np.mean(target[selected]))
        predicted = float(np.mean(probability[selected]))
        error += float(np.sum(selected)) / total * abs(observed - predicted)
    return error


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


def _stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
