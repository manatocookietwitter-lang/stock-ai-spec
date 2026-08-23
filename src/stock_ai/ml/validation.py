"""Purged expanding walk-forward validation with no random-split API."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_ai.ml.models import Regressor


@dataclass(frozen=True)
class TimeSeriesFold:
    fold_number: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class FoldMetrics:
    fold_number: int
    train_rows: int
    validation_rows: int
    mean_squared_error: float
    spearman_rank_ic: float | None
    rank_ic_dates: int
    rank_ic_standard_deviation: float | None


@dataclass(frozen=True)
class LockedFinalHoldout:
    """A development-only view plus an auditable boundary for the untouched holdout."""

    development_indices: tuple[int, ...]
    holdout_start: pd.Timestamp
    holdout_periods: int


def reserve_locked_final_holdout(
    frame: pd.DataFrame,
    *,
    holdout_periods: int,
    date_column: str = "trading_date",
) -> LockedFinalHoldout:
    """Reserve final dates without returning their row indices to validation callers."""
    if holdout_periods < 1:
        raise ValueError("holdout periods must be positive")
    dates = pd.DatetimeIndex(pd.to_datetime(frame[date_column]).sort_values().unique())
    if len(dates) <= holdout_periods:
        raise ValueError("locked holdout must leave at least one development period")
    holdout_start = dates[-holdout_periods]
    development = pd.to_datetime(frame[date_column]) < holdout_start
    return LockedFinalHoldout(
        development_indices=tuple(int(value) for value in np.flatnonzero(development.to_numpy())),
        holdout_start=holdout_start,
        holdout_periods=holdout_periods,
    )


class PurgedExpandingWindowSplitter:
    """Expanding date split with an embargo gap and overlap-label purge."""

    def __init__(
        self,
        *,
        initial_train_periods: int,
        validation_periods: int,
        step_periods: int | None = None,
        purge_periods: int = 0,
        embargo_periods: int = 0,
        label_horizon_periods: int = 1,
    ) -> None:
        if initial_train_periods < 1 or validation_periods < 1:
            raise ValueError("train and validation periods must be positive")
        if step_periods is not None and step_periods < 1:
            raise ValueError("step periods must be positive")
        if purge_periods < 0 or embargo_periods < 0:
            raise ValueError("purge and embargo periods cannot be negative")
        if label_horizon_periods < 1:
            raise ValueError("label horizon periods must be positive")
        if embargo_periods < label_horizon_periods:
            raise ValueError("embargo periods must be at least the label horizon")
        self.initial_train_periods = initial_train_periods
        self.validation_periods = validation_periods
        self.step_periods = step_periods if step_periods is not None else validation_periods
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods
        self.label_horizon_periods = label_horizon_periods

    def split(
        self,
        frame: pd.DataFrame,
        *,
        date_column: str = "trading_date",
        label_end_column: str,
    ) -> Iterator[TimeSeriesFold]:
        dates = pd.DatetimeIndex(pd.to_datetime(frame[date_column]).sort_values().unique())
        gap = self.purge_periods + self.embargo_periods
        validation_start_index = self.initial_train_periods + gap
        fold_number = 0
        while validation_start_index + self.validation_periods <= len(dates):
            training_date_count = validation_start_index - gap
            train_dates = dates[:training_date_count]
            validation_dates = dates[
                validation_start_index : validation_start_index + self.validation_periods
            ]
            validation_start = validation_dates[0]
            train_mask = pd.to_datetime(frame[date_column]).isin(train_dates)
            label_end = pd.to_datetime(frame[label_end_column])
            train_mask &= label_end.notna() & (label_end < validation_start)
            validation_mask = pd.to_datetime(frame[date_column]).isin(validation_dates)
            train_indices = tuple(int(value) for value in np.flatnonzero(train_mask.to_numpy()))
            validation_indices = tuple(
                int(value) for value in np.flatnonzero(validation_mask.to_numpy())
            )
            if len(train_indices) and len(validation_indices):
                yield TimeSeriesFold(
                    fold_number=fold_number,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    train_end=train_dates[-1],
                    validation_start=validation_start,
                    validation_end=validation_dates[-1],
                )
                fold_number += 1
            validation_start_index += self.step_periods


def walk_forward_validate(
    frame: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    target_column: str,
    label_end_column: str,
    splitter: PurgedExpandingWindowSplitter,
    model_factory: Callable[[], Regressor],
    require_folds: bool = True,
) -> tuple[FoldMetrics, ...]:
    reports: list[FoldMetrics] = []
    for fold in splitter.split(frame, label_end_column=label_end_column):
        train = frame.iloc[list(fold.train_indices)]
        validation = frame.iloc[list(fold.validation_indices)]
        valid_validation = validation[target_column].notna()
        validation = validation.loc[valid_validation]
        if validation.empty:
            continue
        model = model_factory().fit(train.loc[:, list(feature_names)], train[target_column])
        prediction = model.predict(validation.loc[:, list(feature_names)])
        target = validation[target_column].to_numpy(dtype=float)
        mse = float(np.mean(np.square(target - prediction)))
        rank_frame = pd.DataFrame(
            {
                "trading_date": pd.to_datetime(validation["trading_date"]).to_numpy(),
                "target": target,
                "prediction": prediction,
            }
        )
        daily_rank_ic: list[float] = []
        for _, date_group in rank_frame.groupby("trading_date", sort=True):
            if (
                len(date_group) < 2
                or date_group["target"].nunique() < 2
                or date_group["prediction"].nunique() < 2
            ):
                continue
            value = date_group["target"].corr(date_group["prediction"], method="spearman")
            if pd.notna(value):
                daily_rank_ic.append(float(value))
        rank_ic = float(np.mean(daily_rank_ic)) if daily_rank_ic else None
        rank_ic_std = float(np.std(daily_rank_ic, ddof=1)) if len(daily_rank_ic) > 1 else None
        reports.append(
            FoldMetrics(
                fold_number=fold.fold_number,
                train_rows=len(train),
                validation_rows=len(validation),
                mean_squared_error=mse,
                spearman_rank_ic=rank_ic,
                rank_ic_dates=len(daily_rank_ic),
                rank_ic_standard_deviation=rank_ic_std,
            )
        )
    if require_folds and not reports:
        raise ValueError("BLOCKED_BY_VALIDATION: no usable walk-forward folds were produced")
    return tuple(reports)
