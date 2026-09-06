import numpy as np
import pandas as pd

from systematic_alpha.inference import (
    bootstrap_paired_difference,
    bootstrap_performance_interval,
)


def test_bootstrap_performance_is_deterministic_and_ordered():
    returns = pd.Series(np.linspace(-0.01, 0.02, 100))
    first = bootstrap_performance_interval(
        returns, simulations=100, block_length=10, seed=7
    )
    second = bootstrap_performance_interval(
        returns, simulations=100, block_length=10, seed=7
    )
    assert first == second
    assert first["sharpe_ci_lower"] <= first["sharpe_ci_upper"]


def test_paired_bootstrap_detects_identical_series():
    returns = pd.Series(np.linspace(-0.01, 0.02, 100))
    result = bootstrap_paired_difference(
        returns, returns, simulations=100, block_length=10
    )
    assert result["annualized_excess_mean"] == 0.0
    assert result["annualized_excess_mean_ci_lower"] == 0.0
    assert result["annualized_excess_mean_ci_upper"] == 0.0
    assert result["sharpe_difference"] == 0.0
