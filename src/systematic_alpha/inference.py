"""Dependence-aware bootstrap inference for portfolio results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _moving_block_indices(
    count: int,
    block_length: int,
    simulations: int,
    seed: int,
) -> np.ndarray:
    if count < 2:
        raise ValueError("At least two observations are required")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if simulations < 1:
        raise ValueError("simulations must be positive")
    length = min(block_length, count)
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(count / length))
    starts = rng.integers(0, count, size=(simulations, blocks))
    offsets = np.arange(length)
    return ((starts[..., None] + offsets) % count).reshape(simulations, -1)[:, :count]


def bootstrap_performance_interval(
    returns: pd.Series,
    *,
    annualization: int = 252,
    block_length: int = 21,
    simulations: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Return moving-block bootstrap intervals for mean return and Sharpe."""
    clean = pd.Series(returns).dropna().astype(float).to_numpy()
    if len(clean) < 2:
        return {
            "annualized_mean_ci_lower": np.nan,
            "annualized_mean_ci_upper": np.nan,
            "sharpe_ci_lower": np.nan,
            "sharpe_ci_upper": np.nan,
            "probability_sharpe_positive": np.nan,
        }
    samples = clean[
        _moving_block_indices(len(clean), block_length, simulations, seed)
    ]
    means = samples.mean(axis=1) * annualization
    standard_deviations = samples.std(axis=1, ddof=1)
    sharpes = np.divide(
        samples.mean(axis=1) * np.sqrt(annualization),
        standard_deviations,
        out=np.full(simulations, np.nan),
        where=standard_deviations > 0,
    )
    return {
        "annualized_mean_ci_lower": float(np.quantile(means, 0.025)),
        "annualized_mean_ci_upper": float(np.quantile(means, 0.975)),
        "sharpe_ci_lower": float(np.nanquantile(sharpes, 0.025)),
        "sharpe_ci_upper": float(np.nanquantile(sharpes, 0.975)),
        "probability_sharpe_positive": float(np.nanmean(sharpes > 0)),
    }


def bootstrap_paired_difference(
    candidate: pd.Series,
    baseline: pd.Series,
    *,
    annualization: int = 252,
    block_length: int = 21,
    simulations: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Estimate paired incremental-return and Sharpe-difference intervals."""
    aligned = pd.concat(
        [candidate.rename("candidate"), baseline.rename("baseline")], axis=1
    ).dropna()
    if len(aligned) < 2:
        return {
            "annualized_excess_mean": np.nan,
            "annualized_excess_mean_ci_lower": np.nan,
            "annualized_excess_mean_ci_upper": np.nan,
            "sharpe_difference": np.nan,
            "sharpe_difference_ci_lower": np.nan,
            "sharpe_difference_ci_upper": np.nan,
            "probability_excess_mean_positive": np.nan,
        }
    values = aligned.to_numpy(dtype=float)
    indices = _moving_block_indices(
        len(values), block_length, simulations, seed
    )
    samples = values[indices]
    excess_means = (samples[:, :, 0] - samples[:, :, 1]).mean(axis=1)
    means = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=1)
    sharpes = np.divide(
        means * np.sqrt(annualization),
        std,
        out=np.full_like(means, np.nan),
        where=std > 0,
    )
    sharpe_differences = sharpes[:, 0] - sharpes[:, 1]
    observed_std = values.std(axis=0, ddof=1)
    observed_sharpe = np.divide(
        values.mean(axis=0) * np.sqrt(annualization),
        observed_std,
        out=np.full(2, np.nan),
        where=observed_std > 0,
    )
    return {
        "annualized_excess_mean": float(
            (values[:, 0] - values[:, 1]).mean() * annualization
        ),
        "annualized_excess_mean_ci_lower": float(
            np.quantile(excess_means * annualization, 0.025)
        ),
        "annualized_excess_mean_ci_upper": float(
            np.quantile(excess_means * annualization, 0.975)
        ),
        "sharpe_difference": float(observed_sharpe[0] - observed_sharpe[1]),
        "sharpe_difference_ci_lower": float(
            np.nanquantile(sharpe_differences, 0.025)
        ),
        "sharpe_difference_ci_upper": float(
            np.nanquantile(sharpe_differences, 0.975)
        ),
        "probability_excess_mean_positive": float(np.mean(excess_means > 0)),
    }
