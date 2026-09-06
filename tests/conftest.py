from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def market_data() -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", "2023-12-29")
    frames = []
    for number, ticker in enumerate("ABCDEFGHIJ"):
        time = np.arange(len(dates), dtype=float)
        base = 50 + number * 5 + 0.02 * time
        seasonal = np.sin(time / (7 + number)) * (1 + number / 20)
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "open_adj": base + seasonal,
                    "high_adj": base + seasonal + 1,
                    "low_adj": base + seasonal - 1,
                    "close_adj": base + np.roll(seasonal, 1),
                    "volume_adj": 1_000_000 + number * 10_000 + time * (number + 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
