"""Training-only factor combination."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def fit_static_weights(ic: pd.DataFrame, ridge: float = 1e-6) -> pd.Series:
    """Estimate genuine long-only, fully-invested maximum-IR weights."""
    clean = ic.dropna()
    if len(clean) < 2:
        raise ValueError("At least two complete IC observations are required")
    mean = clean.mean().to_numpy()
    covariance = clean.cov().to_numpy() + ridge * np.eye(clean.shape[1])

    def objective(weights: np.ndarray) -> float:
        variance = float(weights @ covariance @ weights)
        if variance <= 0:
            return 1e12
        return -float(weights @ mean) / np.sqrt(variance)

    count = clean.shape[1]
    initial = np.full(count, 1.0 / count)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 1_000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Factor-weight optimization failed: {result.message}")
    weights = np.clip(result.x, 0.0, 1.0)
    weights /= weights.sum()
    return pd.Series(weights, index=clean.columns, name="weight")


def add_composite(data: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    result = data.copy()
    missing = set(weights.index).difference(result.columns)
    if missing:
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    result["composite"] = sum(
        result[column] * weight for column, weight in weights.items()
    )
    return result


def add_equal_composite(
    data: pd.DataFrame, signals: tuple[str, ...], name: str
) -> pd.DataFrame:
    """Add an equal-weight composite, requiring every component to be present."""
    if not signals:
        raise ValueError("At least one signal is required")
    missing = set(signals).difference(data.columns)
    if missing:
        raise ValueError(f"Missing signal columns: {sorted(missing)}")
    result = data.copy()
    result[name] = result.loc[:, list(signals)].mean(axis=1, skipna=False)
    return result


def add_rolling_composite(
    data: pd.DataFrame,
    ic: pd.DataFrame,
    window: int = 252,
    min_periods: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add a composite using only IC observations available by each decision date.

    ``ic`` is indexed by signal date and must contain an ``available_date`` column.
    For a close-to-next-open-to-open label, availability is the label exit date,
    not merely the day after the signal date.
    """
    if window < 2 or min_periods < 2 or min_periods > window:
        raise ValueError("Require 2 <= min_periods <= window")
    if "available_date" not in ic.columns:
        raise ValueError("IC observations require an available_date column")
    ordered_ic = ic.sort_index().copy()
    ordered_ic["available_date"] = pd.to_datetime(ordered_ic["available_date"])
    signal_columns = ordered_ic.columns.drop("available_date")
    weight_rows: list[pd.Series] = []
    decision_dates = pd.Index(sorted(pd.to_datetime(data["Date"]).unique()))
    for date in decision_dates:
        history = ordered_ic.loc[
            ordered_ic["available_date"].le(date), signal_columns
        ].tail(window).dropna()
        if len(history) < min_periods:
            weights = pd.Series(np.nan, index=signal_columns, name=date)
        else:
            weights = fit_static_weights(history)
            weights.name = date
        weight_rows.append(weights)
    rolling_weights = pd.DataFrame(weight_rows)
    rolling_weights.index.name = "Date"

    result = data.copy()
    dated_weights = rolling_weights.add_suffix("_rolling_weight").reset_index()
    result = result.merge(dated_weights, on="Date", how="left", validate="many_to_one")
    result["rolling_composite"] = sum(
        result[column] * result[f"{column}_rolling_weight"]
        for column in signal_columns
    )
    return result, rolling_weights
