"""Baseline datasets, models, and time-series validation."""

from stock_ai.ml.dataset import DatasetSnapshot, build_supervised_dataset, write_dataset_snapshot
from stock_ai.ml.models import BaselinePredictionBundle, MomentumRegressor, RidgeRegressor
from stock_ai.ml.production import (
    BaselineModelSummary,
    ProductionBaselineReport,
    ProductionBuildManifest,
    ProductionDatasetSnapshot,
    ProductionFeatureSets,
    ProductionFeatureSnapshot,
    build_production_feature_sets,
    build_production_supervised_dataset,
    load_production_build_manifest,
    load_production_dataset_snapshot,
    load_production_feature_snapshot,
    run_production_walk_forward_baselines,
    write_production_baseline_report,
    write_production_build_manifest,
    write_production_dataset_snapshot,
    write_production_feature_snapshot,
)
from stock_ai.ml.validation import (
    LockedFinalHoldout,
    PurgedExpandingWindowSplitter,
    reserve_locked_final_holdout,
    walk_forward_validate,
)

__all__ = [
    "BaselineModelSummary",
    "BaselinePredictionBundle",
    "DatasetSnapshot",
    "LockedFinalHoldout",
    "MomentumRegressor",
    "ProductionBaselineReport",
    "ProductionBuildManifest",
    "ProductionDatasetSnapshot",
    "ProductionFeatureSets",
    "ProductionFeatureSnapshot",
    "PurgedExpandingWindowSplitter",
    "RidgeRegressor",
    "build_production_feature_sets",
    "build_production_supervised_dataset",
    "build_supervised_dataset",
    "load_production_build_manifest",
    "load_production_dataset_snapshot",
    "load_production_feature_snapshot",
    "reserve_locked_final_holdout",
    "run_production_walk_forward_baselines",
    "walk_forward_validate",
    "write_dataset_snapshot",
    "write_production_baseline_report",
    "write_production_build_manifest",
    "write_production_dataset_snapshot",
    "write_production_feature_snapshot",
]
