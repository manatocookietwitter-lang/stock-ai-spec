"""Deterministic momentum and Ridge baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stock_ai.ml.dataset import HORIZONS


class Regressor(Protocol):
    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self: ...

    def predict(self, features: pd.DataFrame) -> np.ndarray: ...


class MomentumRegressor:
    """A transparent baseline using only a causal trailing-return feature."""

    def __init__(self, feature_name: str = "price.return_20d", scale: float = 1.0) -> None:
        self.feature_name = feature_name
        self.scale = scale
        self._fitted = False
        self._fallback = 0.0

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        values = features[self.feature_name].astype(float)
        valid = values.notna() & target.notna()
        self._fallback = float(target.loc[valid].mean()) if valid.any() else 0.0
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("momentum baseline must be fitted before prediction")
        values = features[self.feature_name].astype(float).fillna(self._fallback)
        return (values * self.scale).to_numpy(dtype=float)


class RidgeRegressor:
    """Fold-local imputation, scaling, and Ridge regression baseline."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        self._fitted = False

    def fit(self, features: pd.DataFrame, target: pd.Series) -> Self:
        valid_target = target.notna()
        if not valid_target.any():
            raise ValueError("Ridge baseline requires at least one non-missing target")
        self._pipeline.fit(features.loc[valid_target], target.loc[valid_target])
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Ridge baseline must be fitted before prediction")
        result = self._pipeline.predict(features)
        return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class HorizonModel:
    horizon: int
    model: RidgeRegressor
    residual_standard_error: float


class BaselinePredictionBundle:
    """Independent 1/5/20-day Ridge models sharing one explicit feature manifest."""

    def __init__(self, feature_names: tuple[str, ...], alpha: float = 1.0) -> None:
        self.feature_names = feature_names
        self.alpha = alpha
        self._models: dict[int, HorizonModel] = {}

    def fit(self, dataset: pd.DataFrame) -> BaselinePredictionBundle:
        for horizon in HORIZONS:
            target_name = f"target_return_{horizon}d"
            valid = dataset[target_name].notna()
            model = RidgeRegressor(alpha=self.alpha).fit(
                dataset.loc[valid, list(self.feature_names)], dataset.loc[valid, target_name]
            )
            residual = dataset.loc[valid, target_name].to_numpy() - model.predict(
                dataset.loc[valid, list(self.feature_names)]
            )
            standard_error = float(np.sqrt(np.mean(np.square(residual))))
            self._models[horizon] = HorizonModel(horizon, model, standard_error)
        return self

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        if set(self._models) != set(HORIZONS):
            raise RuntimeError("all horizon models must be fitted before prediction")
        output = features[["symbol", "trading_date", "available_at"]].copy()
        for horizon in HORIZONS:
            fitted = self._models[horizon]
            output[f"prediction_{horizon}d"] = fitted.model.predict(
                features.loc[:, list(self.feature_names)]
            )
            output[f"uncertainty_{horizon}d"] = fitted.residual_standard_error
        return output
