"""Sample isolation and factor diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import Period


def select_period(
    data: pd.DataFrame,
    period: Period,
    *,
    require_label_within: bool = False,
    horizon: int = 1,
) -> pd.DataFrame:
    """Select a sample, optionally requiring its complete label to stay inside.

    Label containment purges boundary observations whose entry or exit price
    belongs to a later sample.
    """
    dates = pd.to_datetime(data["Date"])
    selected = dates.between(period.start, period.end)
    if require_label_within:
        entry_col = f"entry_date_{horizon}d"
        exit_col = f"exit_date_{horizon}d"
        missing = {entry_col, exit_col}.difference(data.columns)
        if missing:
            raise ValueError(f"Missing label-date columns: {sorted(missing)}")
        entry_dates = pd.to_datetime(data[entry_col])
        exit_dates = pd.to_datetime(data[exit_col])
        selected &= entry_dates.between(period.start, period.end)
        selected &= exit_dates.between(period.start, period.end)
    return data.loc[selected].copy()


def daily_ic(
    data: pd.DataFrame,
    signal_col: str,
    return_col: str = "forward_return_1d",
    method: str = "spearman",
) -> pd.Series:
    sample = data[["Date", signal_col, return_col]].dropna()
    return (
        sample.groupby("Date")
        .apply(
            lambda group: group[signal_col].corr(group[return_col], method=method),
            include_groups=False,
        )
        .rename(signal_col)
    )


def newey_west_mean_test(values: pd.Series, max_lags: int = 5) -> dict[str, float]:
    values = pd.Series(values).dropna().astype(float).to_numpy()
    count = len(values)
    if count < 2:
        return {
            "mean": np.nan,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    centered = values - values.mean()
    long_run_variance = np.dot(centered, centered) / count
    usable_lags = min(max_lags, count - 1)
    for lag in range(1, usable_lags + 1):
        weight = 1 - lag / (usable_lags + 1)
        autocovariance = np.dot(centered[lag:], centered[:-lag]) / count
        long_run_variance += 2 * weight * autocovariance
    standard_error = np.sqrt(max(long_run_variance, 0) / count)
    t_stat = values.mean() / standard_error if standard_error else np.nan
    return {
        "mean": float(values.mean()),
        "standard_error": float(standard_error),
        "t_stat": float(t_stat),
        "p_value": float(2 * stats.norm.sf(abs(t_stat))),
    }
