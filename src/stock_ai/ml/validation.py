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
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class FoldMetrics:
    fold_number: int
    train_rows: int
    validation_rows: int
    mean_squared_error: float
    spearman_rank_ic: float


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
    ) -> None:
        if initial_train_periods < 1 or validation_periods < 1:
            raise ValueError("train and validation periods must be positive")
        if purge_periods < 0 or embargo_periods < 0:
            raise ValueError("purge and embargo periods cannot be negative")
        self.initial_train_periods = initial_train_periods
        self.validation_periods = validation_periods
        self.step_periods = step_periods or validation_periods
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods

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
            train_indices = np.flatnonzero(train_mask.to_numpy())
            validation_indices = np.flatnonzero(validation_mask.to_numpy())
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
) -> tuple[FoldMetrics, ...]:
    reports: list[FoldMetrics] = []
    for fold in splitter.split(frame, label_end_column=label_end_column):
        train = frame.iloc[fold.train_indices]
        validation = frame.iloc[fold.validation_indices]
        valid_validation = validation[target_column].notna()
        validation = validation.loc[valid_validation]
        if validation.empty:
            continue
        model = model_factory().fit(train.loc[:, list(feature_names)], train[target_column])
        prediction = model.predict(validation.loc[:, list(feature_names)])
        target = validation[target_column].to_numpy(dtype=float)
        mse = float(np.mean(np.square(target - prediction)))
        rank_ic = float(pd.Series(target).corr(pd.Series(prediction), method="spearman"))
        reports.append(
            FoldMetrics(
                fold_number=fold.fold_number,
                train_rows=len(train),
                validation_rows=len(validation),
                mean_squared_error=mse,
                spearman_rank_ic=rank_ic,
            )
        )
    return tuple(reports)
