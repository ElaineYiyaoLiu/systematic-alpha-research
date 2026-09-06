import pandas as pd
import pytest

from systematic_alpha.combination import (
    add_equal_composite,
    add_rolling_composite,
    fit_static_weights,
)


def test_factor_weights_are_long_only_and_sum_to_one():
    ic = pd.DataFrame({"reversal": [0.1, 0.2, -0.1], "alpha012": [0.02, 0.03, 0.01]})
    weights = fit_static_weights(ic)
    assert (weights >= 0).all()
    assert abs(weights.sum() - 1) < 1e-12


def test_rolling_composite_uses_only_available_ic():
    dates = pd.date_range("2024-01-01", periods=5)
    ic = pd.DataFrame(
        {
            "reversal": [1.0, 1.0, 1.0, -100.0, -100.0],
            "alpha012": [0.1] * 5,
            "available_date": dates + pd.offsets.Day(2),
        },
        index=dates,
    )
    data = pd.DataFrame({"Date": dates, "reversal": [1.0] * 5, "alpha012": [0.0] * 5})
    _, weights = add_rolling_composite(data, ic, window=3, min_periods=2)
    baseline = weights.loc[dates[3]].copy()
    changed = ic.copy()
    changed.loc[dates[2] :, "reversal"] = 1_000_000.0
    _, changed_weights = add_rolling_composite(data, changed, window=3, min_periods=2)
    pd.testing.assert_series_equal(baseline, changed_weights.loc[dates[3]])


def test_rolling_composite_requires_explicit_availability_date():
    dates = pd.date_range("2024-01-01", periods=3)
    ic = pd.DataFrame({"a": [0.1, 0.2, 0.3]}, index=dates)
    data = pd.DataFrame({"Date": dates, "a": 1.0})
    with pytest.raises(ValueError, match="available_date"):
        add_rolling_composite(data, ic, window=2, min_periods=2)


def test_equal_composite_requires_all_components():
    data = pd.DataFrame({"a": [1.0, 1.0], "b": [3.0, float("nan")]})
    result = add_equal_composite(data, ("a", "b"), "equal")
    assert result.loc[0, "equal"] == 2.0
    assert pd.isna(result.loc[1, "equal"])


def test_static_optimizer_respects_long_only_simplex():
    ic = pd.DataFrame(
        {
            "good": [0.03, 0.02, 0.04, 0.03],
            "bad": [-0.02, -0.01, -0.03, -0.02],
        }
    )
    weights = fit_static_weights(ic)
    assert weights.ge(0).all()
    assert weights.le(1).all()
    assert weights.sum() == pytest.approx(1.0)
    assert weights["good"] > weights["bad"]
