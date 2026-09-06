import pandas as pd
import pytest

from systematic_alpha.returns import add_forward_returns


def test_forward_return_is_next_open_to_following_open():
    data = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "Ticker": ["A"] * 3,
            "open_adj": [100.0, 110.0, 121.0],
        }
    )
    result = add_forward_returns(data, horizons=(1,))
    assert result.loc[0, "forward_return_1d"] == pytest.approx(0.1)
    assert result.loc[0, "entry_date_1d"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "exit_date_1d"] == pd.Timestamp("2024-01-03")
    assert pd.isna(result.loc[1, "forward_return_1d"])


def test_forward_return_does_not_skip_a_missing_market_date():
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-03",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "Ticker": ["A", "A", "B", "B", "B"],
            "open_adj": [100.0, 121.0, 50.0, 55.0, 60.5],
        }
    )
    result = add_forward_returns(data, horizons=(1,))
    first_a = result[(result["Ticker"] == "A") & (result["Date"] == "2024-01-01")]
    first_b = result[(result["Ticker"] == "B") & (result["Date"] == "2024-01-01")]
    assert first_a["forward_return_1d"].isna().all()
    assert first_b["forward_return_1d"].iloc[0] == pytest.approx(0.1)
