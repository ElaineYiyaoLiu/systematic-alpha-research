"""Weight-based turnover and transaction costs."""

from __future__ import annotations

import pandas as pd


def calculate_turnover(weights: pd.DataFrame) -> pd.Series:
    matrix = weights.pivot(index="Date", columns="Ticker", values="weight").fillna(0.0)
    previous = matrix.shift(1).fillna(0.0)
    return matrix.sub(previous).abs().sum(axis=1).rename("turnover")


def apply_linear_costs(
    gross_returns: pd.Series, turnover: pd.Series, cost_bps: float
) -> pd.DataFrame:
    result = pd.concat([gross_returns, turnover], axis=1).fillna(0.0)
    result["transaction_cost"] = result["turnover"] * cost_bps / 10_000
    result["net_return"] = result["gross_return"] - result["transaction_cost"]
    return result
