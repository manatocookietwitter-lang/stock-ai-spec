"""Cross-sectional research metrics with explicit date grouping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankingMetrics:
    rows: int
    dates: int
    mean_squared_error: float
    mean_daily_rank_ic: float | None
    rank_ic_standard_deviation: float | None
    rank_icir: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    ndcg_at_20: float | None
    precision_at_5: float | None
    precision_at_10: float | None
    precision_at_20: float | None
    top_5_mean_target: float | None
    top_10_mean_target: float | None
    top_20_mean_target: float | None


def cross_sectional_relevance(target: pd.Series, dates: pd.Series) -> np.ndarray:
    """Map each date's continuous target to deterministic relevance labels 0..4."""

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "target": pd.to_numeric(target, errors="coerce").to_numpy(dtype=float),
        },
        index=target.index,
    )
    output = np.zeros(len(frame), dtype=np.int32)
    for _, positions in frame.groupby("date", sort=False).indices.items():
        location = np.asarray(positions, dtype=int)
        values = frame.iloc[location]["target"]
        percentile = values.rank(method="average", pct=True)
        output[location] = np.minimum(np.ceil(percentile.to_numpy() * 5.0) - 1, 4).astype(np.int32)
    return output


def within_date_rank_standardize(values: pd.Series, dates: pd.Series) -> np.ndarray:
    """Produce comparable zero-centered within-date percentile ranks."""

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "value": pd.to_numeric(values, errors="coerce").to_numpy(dtype=float),
        },
        index=values.index,
    )
    ranked = frame.groupby("date", sort=False)["value"].rank(method="average", pct=True)
    return (ranked.to_numpy(dtype=float) - 0.5) * 2.0


def evaluate_cross_sectional_predictions(
    *,
    dates: pd.Series,
    target: pd.Series,
    prediction: pd.Series,
) -> RankingMetrics:
    """Evaluate predictions by date; pooled time-series correlation is never used."""

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).to_numpy(),
            "target": pd.to_numeric(target, errors="coerce").to_numpy(dtype=float),
            "prediction": pd.to_numeric(prediction, errors="coerce").to_numpy(dtype=float),
        }
    ).replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("prediction metrics require at least one finite observation")

    daily_ic: list[float] = []
    ndcg: dict[int, list[float]] = {5: [], 10: [], 20: []}
    precision: dict[int, list[float]] = {5: [], 10: [], 20: []}
    top_target: dict[int, list[float]] = {5: [], 10: [], 20: []}
    for _, group in frame.groupby("date", sort=True):
        if (
            len(group) >= 2
            and group["target"].nunique() >= 2
            and group["prediction"].nunique() >= 2
        ):
            value = group["target"].corr(group["prediction"], method="spearman")
            if pd.notna(value):
                daily_ic.append(float(value))
        relevance = cross_sectional_relevance(group["target"], group["date"])
        target_order = np.argsort(-group["target"].to_numpy(dtype=float), kind="stable")
        prediction_order = np.argsort(-group["prediction"].to_numpy(dtype=float), kind="stable")
        for k in (5, 10, 20):
            size = min(k, len(group))
            if size < 2:
                continue
            ideal = target_order[:size]
            selected = prediction_order[:size]
            discounts = 1.0 / np.log2(np.arange(size, dtype=float) + 2.0)
            gain = np.power(2.0, relevance[selected].astype(float)) - 1.0
            ideal_gain = np.power(2.0, relevance[ideal].astype(float)) - 1.0
            denominator = float(np.sum(ideal_gain * discounts))
            if denominator > 0:
                ndcg[k].append(float(np.sum(gain * discounts) / denominator))
            precision[k].append(float(len(set(selected) & set(ideal)) / size))
            top_target[k].append(float(group.iloc[selected]["target"].mean()))

    ic_mean = float(np.mean(daily_ic)) if daily_ic else None
    ic_std = float(np.std(daily_ic, ddof=1)) if len(daily_ic) > 1 else None
    icir = (
        ic_mean / ic_std if ic_mean is not None and ic_std is not None and ic_std != 0.0 else None
    )
    mse = float(np.mean(np.square(frame["target"] - frame["prediction"])))
    return RankingMetrics(
        rows=len(frame),
        dates=int(frame["date"].nunique()),
        mean_squared_error=mse,
        mean_daily_rank_ic=ic_mean,
        rank_ic_standard_deviation=ic_std,
        rank_icir=icir,
        ndcg_at_5=_mean_or_none(ndcg[5]),
        ndcg_at_10=_mean_or_none(ndcg[10]),
        ndcg_at_20=_mean_or_none(ndcg[20]),
        precision_at_5=_mean_or_none(precision[5]),
        precision_at_10=_mean_or_none(precision[10]),
        precision_at_20=_mean_or_none(precision[20]),
        top_5_mean_target=_mean_or_none(top_target[5]),
        top_10_mean_target=_mean_or_none(top_target[10]),
        top_20_mean_target=_mean_or_none(top_target[20]),
    )


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None
