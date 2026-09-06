import pandas as pd

from systematic_alpha.data import (
    apply_point_in_time_universe,
    mark_point_in_time_eligibility,
)


def test_point_in_time_membership_filters_each_date():
    market = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
            "Ticker": ["A", "A", "B", "B"],
            "value": [1, 2, 3, 4],
        }
    )
    membership = pd.DataFrame(
        {
            "Ticker": ["A", "B"],
            "StartDate": ["2024-01-02", "2024-01-01"],
            "EndDate": [None, "2024-01-01"],
        }
    )
    result = apply_point_in_time_universe(market, membership)
    observed = set(result[["Date", "Ticker"]].itertuples(index=False, name=None))
    assert observed == {
        (pd.Timestamp("2024-01-02"), "A"),
        (pd.Timestamp("2024-01-01"), "B"),
    }


def test_membership_marker_retains_prices_outside_membership():
    market = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "Ticker": ["A", "A"],
            "value": [1, 2],
        }
    )
    membership = pd.DataFrame(
        {"Ticker": ["A"], "StartDate": ["2024-01-01"], "EndDate": ["2024-01-01"]}
    )
    result = mark_point_in_time_eligibility(market, membership)
    assert len(result) == 2
    assert result["eligible"].tolist() == [True, False]
