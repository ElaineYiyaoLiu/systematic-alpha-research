import pandas as pd
import pytest

from systematic_alpha.costs import apply_linear_costs, calculate_turnover
from systematic_alpha.portfolio import (
    build_weights,
    portfolio_returns,
    simulate_portfolio,
)


def test_full_replacement_turnover_and_cost():
    weights = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4),
            "Ticker": ["A", "B", "C", "D"] * 2,
            "weight": [0.5, -0.5, 0, 0, 0, 0, 0.5, -0.5],
        }
    )
    turnover = calculate_turnover(weights)
    assert turnover.iloc[0] == 1.0
    assert turnover.iloc[1] == 2.0
    net = apply_linear_costs(
        pd.Series([0.0, 0.0], index=turnover.index, name="gross_return"), turnover, 5
    )
    assert net["transaction_cost"].tolist() == [0.0005, 0.001]


def test_missing_active_return_is_not_silently_treated_as_zero():
    weights = pd.DataFrame(
        {"Date": [pd.Timestamp("2024-01-01")], "Ticker": ["A"], "weight": [1.0]}
    )
    returns = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "Ticker": ["A", "B"],
            "forward_return_1d": [float("nan"), 0.01],
        }
    )
    with pytest.raises(ValueError, match="Missing forward returns"):
        portfolio_returns(weights, returns)


def test_no_rebalance_has_zero_turnover():
    weights = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "Ticker": ["A", "A", "B", "B"],
            "weight": [0.5, 0.5, -0.5, -0.5],
        }
    )
    assert calculate_turnover(weights).tolist() == [1.0, 0.0]


def test_built_portfolio_is_dollar_neutral_with_unit_gross_exposure():
    tickers = list("ABCDEFGHIJ")
    data = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-01-01"),
            "Ticker": tickers,
            "signal": range(len(tickers)),
        }
    )
    weights = build_weights(data, "signal", rebalance_frequency=1)
    assert weights["weight"].sum() == pytest.approx(0.0)
    assert weights["weight"].abs().sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("long_exposure", "short_exposure"),
    [(0.5, 0.5), (-0.5, -0.5), (0.6, -0.5)],
)
def test_invalid_exposure_configuration_is_rejected(long_exposure, short_exposure):
    data = pd.DataFrame(columns=["Date", "Ticker", "signal"])
    with pytest.raises(ValueError):
        build_weights(
            data,
            "signal",
            rebalance_frequency=1,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
        )


def test_non_rebalance_weights_drift_without_free_trading():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    targets = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Ticker": ["A", "B", "A", "B"],
            "weight": [0.5, -0.5, 0.5, -0.5],
            "is_rebalance": [True, True, False, False],
        }
    )
    returns = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Ticker": ["A", "B", "A", "B"],
            "forward_return_1d": [0.10, 0.0, 0.0, 0.0],
        }
    )
    result = simulate_portfolio(targets, returns)
    assert result.loc[dates[0], "turnover"] == pytest.approx(1.0)
    assert result.loc[dates[1], "turnover"] == pytest.approx(0.0)
    assert result.loc[dates[0], "net_exposure_end"] != pytest.approx(0.0)


def test_rebalance_turnover_is_measured_from_drifted_weights():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    targets = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Ticker": ["A", "B", "A", "B"],
            "weight": [0.5, -0.5, 0.5, -0.5],
            "is_rebalance": [True, True, True, True],
        }
    )
    returns = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Ticker": ["A", "B", "A", "B"],
            "forward_return_1d": [0.10, 0.0, 0.0, 0.0],
        }
    )
    result = simulate_portfolio(targets, returns)
    drifted_a = 0.5 * 1.10 / 1.05
    drifted_b = -0.5 / 1.05
    expected = abs(0.5 - drifted_a) + abs(-0.5 - drifted_b)
    assert result.loc[dates[1], "turnover"] == pytest.approx(expected)


def test_ineligible_holding_is_forced_out_and_charged_turnover():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    targets = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Ticker": ["A", "B", "A", "B"],
            "weight": [0.5, -0.5, 0.5, -0.5],
            "is_rebalance": [True, True, False, False],
        }
    )
    returns = pd.DataFrame(
        {
            "Date": [dates[0], dates[0], dates[1], dates[1]],
            "Ticker": ["A", "B", "A", "B"],
            "forward_return_1d": [0.0, 0.0, 0.0, 0.0],
            "eligible": [True, True, False, True],
        }
    )
    result = simulate_portfolio(targets, returns)
    assert result.loc[dates[1], "forced_exit_turnover"] == pytest.approx(0.5)
    assert result.loc[dates[1], "turnover"] == pytest.approx(0.5)


def test_build_weights_never_selects_ineligible_security():
    data = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-01-01"),
            "Ticker": list("ABCDE"),
            "signal": [100, 4, 3, 2, 1],
            "eligible": [False, True, True, True, True],
        }
    )
    weights = build_weights(data, "signal", rebalance_frequency=1, quantiles=2)
    assert weights.loc[weights["Ticker"] == "A", "weight"].iloc[0] == 0.0


def test_build_weights_enforces_minimum_eligible_cross_section():
    data = pd.DataFrame(
        {
            "Date": pd.Timestamp("2024-01-01"),
            "Ticker": list("ABCDE"),
            "signal": range(5),
            "selection_eligible": [True, True, False, False, False],
        }
    )
    weights = build_weights(
        data,
        "signal",
        rebalance_frequency=1,
        quantiles=2,
        minimum_cross_section_size=3,
        selection_eligibility_col="selection_eligible",
    )
    assert weights["weight"].eq(0).all()
    assert not weights["is_rebalance"].any()
