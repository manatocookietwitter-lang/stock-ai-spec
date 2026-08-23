"""Baseline datasets, models, and time-series validation."""

from stock_ai.ml.dataset import DatasetSnapshot, build_supervised_dataset, write_dataset_snapshot
from stock_ai.ml.models import BaselinePredictionBundle, MomentumRegressor, RidgeRegressor
from stock_ai.ml.validation import (
    LockedFinalHoldout,
    PurgedExpandingWindowSplitter,
    reserve_locked_final_holdout,
    walk_forward_validate,
)

__all__ = [
    "BaselinePredictionBundle",
    "DatasetSnapshot",
    "LockedFinalHoldout",
    "MomentumRegressor",
    "PurgedExpandingWindowSplitter",
    "RidgeRegressor",
    "build_supervised_dataset",
    "reserve_locked_final_holdout",
    "walk_forward_validate",
    "write_dataset_snapshot",
]
