import numpy as np
import pandas as pd

from systematic_alpha.signals import generate_signals


def _panel(periods: int = 260) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=periods)
    frames = []
    for number, ticker in enumerate("ABCDE"):
        time = np.arange(periods, dtype=float)
        close = 20.0 + number * 5 + time * (0.01 + number * 0.001)
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "open_adj": close - 0.2 + number * 0.02,
                    "high_adj": close + 0.5,
                    "low_adj": close - 0.5,
                    "close_adj": close,
                    "volume_adj": 1_000 + number * 100 + time * (number + 1),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_alpha101_direction_and_zero_range_handling():
    data = _panel(2)
    first_date = data["Date"].min()
    same_day = data["Date"].eq(first_date)
    data.loc[same_day & data["Ticker"].eq("A"), "open_adj"] = (
        data.loc[same_day & data["Ticker"].eq("A"), "close_adj"] + 0.4
    )
    data.loc[same_day & data["Ticker"].eq("B"), "high_adj"] = data.loc[
        same_day & data["Ticker"].eq("B"), "low_adj"
    ]
    result = generate_signals(data)
    rows = result.loc[result["Date"].eq(first_date)].set_index("Ticker")
    assert rows.loc["A", "alpha101_raw"] < 0
    assert pd.isna(rows.loc["B", "alpha101_raw"])


def test_alpha52_has_240_day_warmup_and_is_price_scale_invariant():
    data = _panel()
    baseline = generate_signals(data)
    scaled = data.copy()
    price_columns = ["open_adj", "high_adj", "low_adj", "close_adj"]
    scaled[price_columns] *= 10
    changed = generate_signals(scaled)
    first_ticker = baseline["Ticker"].eq("A")
    assert baseline.loc[first_ticker, "alpha52_robust_raw"].iloc[:240].isna().all()
    pd.testing.assert_series_equal(
        baseline["alpha52_robust_raw"],
        changed["alpha52_robust_raw"],
        check_names=False,
    )


def test_future_data_cannot_change_current_new_signals():
    data = _panel()
    cutoff = data["Date"].sort_values().unique()[245]
    baseline = generate_signals(data)
    changed_input = data.copy()
    future = changed_input["Date"] > cutoff
    changed_input.loc[future, ["open_adj", "high_adj", "low_adj", "close_adj"]] *= 3
    changed = generate_signals(changed_input)
    columns = ["alpha101_raw", "alpha52_robust_raw"]
    pd.testing.assert_frame_equal(
        baseline.loc[baseline["Date"] <= cutoff, columns].reset_index(drop=True),
        changed.loc[changed["Date"] <= cutoff, columns].reset_index(drop=True),
    )


def test_alpha52_preserves_original_low_leg_direction():
    dates = pd.bdate_range("2024-01-01", periods=8)
    frames = []
    for number, ticker in enumerate("ABCDE"):
        time = np.arange(len(dates), dtype=float)
        close = 20.0 + number + time
        frames.append(
            pd.DataFrame(
                {
                    "Date": dates,
                    "Ticker": ticker,
                    "open_adj": close - 0.1,
                    "high_adj": close + 1.0,
                    "low_adj": 15.0 + number - time,
                    "close_adj": close,
                    "volume_adj": 1_000 + number + time,
                }
            )
        )
    result = generate_signals(
        pd.concat(frames, ignore_index=True),
        alpha52_low_window=2,
        alpha52_low_delay=2,
        alpha52_momentum_recent_skip=1,
        alpha52_momentum_lookback=3,
        alpha52_volume_rank_window=2,
    )
    final = result.loc[result["Date"].eq(dates[-1]), "alpha52_robust_raw"]
    assert final.gt(0).all()


def test_ineligible_extreme_value_does_not_change_eligible_signals():
    data = _panel(3)
    data["eligible"] = True
    baseline = generate_signals(data)

    extra = data.loc[data["Ticker"].eq("A")].copy()
    extra["Ticker"] = "Z"
    extra["eligible"] = False
    extra[["open_adj", "high_adj", "low_adj", "close_adj"]] *= 1_000_000
    combined = generate_signals(pd.concat([data, extra], ignore_index=True))

    columns = ["reversal", "alpha012_robust", "alpha101"]
    pd.testing.assert_frame_equal(
        baseline[columns].reset_index(drop=True),
        combined.loc[combined["Ticker"].ne("Z"), columns].reset_index(drop=True),
    )
    assert combined.loc[combined["Ticker"].eq("Z"), columns].isna().all().all()


def test_alpha012_robust_is_invariant_to_price_rescaling():
    data = _panel(5)
    baseline = generate_signals(data)
    scaled = data.copy()
    price_columns = ["open_adj", "high_adj", "low_adj", "close_adj"]
    scaled.loc[scaled["Ticker"].eq("A"), price_columns] *= 10
    changed = generate_signals(scaled)
    pd.testing.assert_series_equal(
        baseline["alpha012_robust_raw"],
        changed["alpha012_robust_raw"],
        check_names=False,
    )
