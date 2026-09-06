import pandas as pd
import pytest

from systematic_alpha.data import (
    adjust_ohlcv,
    validate_processed_data,
    validate_raw_data,
)


def test_split_adjustment_removes_mechanical_price_jump():
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=2),
            "Ticker": ["A", "A"],
            "Open": [100.0, 50.0],
            "High": [101.0, 51.0],
            "Low": [99.0, 49.0],
            "Close": [100.0, 50.0],
            "Adj Close": [50.0, 50.0],
            "Volume": [1_000.0, 2_000.0],
        }
    )
    with pytest.warns(UserWarning, match="Stock Splits"):
        adjusted = adjust_ohlcv(raw)
    assert adjusted["open_adj"].tolist() == [50.0, 50.0]
    assert adjusted["volume_adj"].tolist() == [2_000.0, 2_000.0]


def test_explicit_splits_do_not_use_dividend_adjustment_for_volume():
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=3),
            "Ticker": ["A"] * 3,
            "Open": [100.0, 50.0, 51.0],
            "High": [101.0, 51.0, 52.0],
            "Low": [99.0, 49.0, 50.0],
            "Close": [100.0, 50.0, 51.0],
            "Adj Close": [48.0, 49.0, 51.0],
            "Volume": [1_000.0, 2_000.0, 2_100.0],
            "Stock Splits": [0.0, 2.0, 0.0],
        }
    )
    adjusted = adjust_ohlcv(raw)
    assert adjusted["volume_adj"].tolist() == [2_000.0, 2_000.0, 2_100.0]


def test_raw_data_rejects_duplicate_security_dates():
    row = {
        "Date": "2024-01-01",
        "Ticker": "A",
        "Open": 10.0,
        "High": 11.0,
        "Low": 9.0,
        "Close": 10.0,
        "Adj Close": 10.0,
        "Volume": 1_000.0,
    }
    with pytest.raises(ValueError, match="Duplicate"):
        validate_raw_data(pd.DataFrame([row, row]))


@pytest.mark.parametrize("column", ["Date", "Ticker"])
def test_processed_data_rejects_missing_identifiers(column):
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Ticker": ["A"],
            "open_adj": [10.0],
            "high_adj": [11.0],
            "low_adj": [9.0],
            "close_adj": [10.0],
            "volume_adj": [1_000.0],
        }
    )
    frame.loc[0, column] = None
    with pytest.raises(ValueError, match="must not be missing"):
        validate_processed_data(frame)


def test_processed_data_rejects_nonpositive_values():
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-01"],
            "Ticker": ["A"],
            "open_adj": [0.0],
            "high_adj": [11.0],
            "low_adj": [9.0],
            "close_adj": [10.0],
            "volume_adj": [1_000.0],
        }
    )
    with pytest.raises(ValueError, match="must be positive"):
        validate_processed_data(frame)
