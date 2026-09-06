"""Portfolio performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def performance_statistics(
    returns: pd.Series, annualization: int = 252
) -> dict[str, float]:
    clean = returns.dropna()
    if clean.empty:
        return {
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    wealth = (1 + clean).cumprod()
    volatility = clean.std() * np.sqrt(annualization)
    return {
        "annualized_return": float(wealth.iloc[-1] ** (annualization / len(clean)) - 1),
        "annualized_volatility": float(volatility),
        "sharpe": float(clean.mean() / clean.std() * np.sqrt(annualization))
        if clean.std()
        else np.nan,
        "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }
