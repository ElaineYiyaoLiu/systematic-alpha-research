"""Point-in-time signal construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cross_sectionally(
    data: pd.DataFrame,
    column: str,
    lower: float = 0.01,
    upper: float = 0.99,
    eligibility_col: str = "eligible",
) -> pd.Series:
    eligible = _eligibility_mask(data, eligibility_col)
    source = data[column].where(eligible)
    grouped = source.groupby(data["Date"])
    low = grouped.transform(lambda values: values.quantile(lower))
    high = grouped.transform(lambda values: values.quantile(upper))
    return source.clip(lower=low, upper=high)


def zscore_cross_sectionally(
    data: pd.DataFrame, column: str, eligibility_col: str = "eligible"
) -> pd.Series:
    eligible = _eligibility_mask(data, eligibility_col)
    source = data[column].where(eligible)
    grouped = source.groupby(data["Date"])
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((source - mean) / std).where(eligible)


def _eligibility_mask(data: pd.DataFrame, column: str) -> pd.Series:
    """Return the point-in-time selection mask without dropping exit prices."""
    if column not in data.columns:
        return pd.Series(True, index=data.index, dtype=bool)
    return data[column].fillna(False).astype(bool)


def generate_signals(
    data: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
    alpha52_low_window: int = 5,
    alpha52_low_delay: int = 5,
    alpha52_momentum_recent_skip: int = 20,
    alpha52_momentum_lookback: int = 240,
    alpha52_volume_rank_window: int = 5,
) -> pd.DataFrame:
    """Generate four point-in-time price-volume signals.

    Alpha101 follows Kakushadze (2016), with zero-range observations treated as
    unavailable instead of adding a price-scale-dependent constant. Alpha52 is a
    disclosed robust adaptation: its five-day low move is expressed as a return
    and its 12-1 momentum leg uses compounded price performance.
    """
    result = data.sort_values(["Ticker", "Date"]).copy()
    grouped = result.groupby("Ticker", group_keys=False)
    close_return = grouped["close_adj"].pct_change(fill_method=None)
    volume_delta = grouped["volume_adj"].diff()
    result["reversal_raw"] = -close_return
    # Scale-free adaptation of Formulaic Alpha012. Absolute adjusted-price
    # differences are not comparable across stocks and can change when a later
    # corporate action changes the historical back-adjustment scale.
    result["alpha012_robust_raw"] = np.sign(volume_delta) * -close_return

    intraday_range = result["high_adj"] - result["low_adj"]
    result["alpha101_raw"] = (
        result["close_adj"] - result["open_adj"]
    ) / intraday_range.where(intraday_range > 0)

    rolling_low = grouped["low_adj"].transform(
        lambda values: values.rolling(
            alpha52_low_window, min_periods=alpha52_low_window
        ).min()
    )
    previous_low = rolling_low.groupby(result["Ticker"]).shift(alpha52_low_delay)
    # The original Alpha52 low-price leg is previous rolling low minus the
    # current rolling low.  Express it as a return while preserving that sign.
    low_improvement = (previous_low - rolling_low) / previous_low
    momentum_12_1 = (
        grouped["close_adj"].shift(alpha52_momentum_recent_skip)
        / grouped["close_adj"].shift(alpha52_momentum_lookback)
        - 1.0
    )
    eligible = _eligibility_mask(result, "eligible")
    momentum_rank = (
        momentum_12_1.where(eligible).groupby(result["Date"]).rank(pct=True)
    )
    volume_rank_5 = grouped["volume_adj"].transform(
        lambda values: values.rolling(
            alpha52_volume_rank_window,
            min_periods=alpha52_volume_rank_window,
        ).rank(pct=True)
    )
    result["alpha52_robust_raw"] = (
        low_improvement * momentum_rank * volume_rank_5
    )

    for raw, final in (
        ("reversal_raw", "reversal"),
        ("alpha012_robust_raw", "alpha012_robust"),
        ("alpha101_raw", "alpha101"),
        ("alpha52_robust_raw", "alpha52_robust"),
    ):
        winsorized = f"{raw}_winsorized"
        result[winsorized] = winsorize_cross_sectionally(result, raw, lower, upper)
        result[final] = zscore_cross_sectionally(result, winsorized)
    return result
