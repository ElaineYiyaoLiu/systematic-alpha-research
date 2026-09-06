"""Executable forward-return labels."""

from __future__ import annotations

import pandas as pd


def add_forward_returns(
    data: pd.DataFrame, horizons: tuple[int, ...] = (1, 5, 10, 21, 42)
) -> pd.DataFrame:
    """Add calendar-exact next-open-to-open returns.

    Horizons are measured on the shared market calendar, never by advancing to
    the next available row for an individual security.  A suspension or a
    missing quote therefore produces a missing label instead of a return over
    an accidentally longer holding period.
    """
    result = data.sort_values(["Ticker", "Date"]).copy()
    if result.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker observations found")
    if any(horizon < 1 for horizon in horizons):
        raise ValueError("Forward-return horizons must be positive")

    result["Date"] = pd.to_datetime(result["Date"])
    calendar = pd.DataFrame({"Date": sorted(result["Date"].unique())})
    open_prices = result[["Ticker", "Date", "open_adj"]]

    for horizon in horizons:
        dates = calendar.copy()
        entry_col = f"entry_date_{horizon}d"
        exit_col = f"exit_date_{horizon}d"
        dates[entry_col] = dates["Date"].shift(-1)
        dates[exit_col] = dates["Date"].shift(-(horizon + 1))
        result = result.merge(dates, on="Date", how="left", validate="many_to_one")

        entry_prices = open_prices.rename(
            columns={"Date": entry_col, "open_adj": "_entry_price"}
        )
        exit_prices = open_prices.rename(
            columns={"Date": exit_col, "open_adj": "_exit_price"}
        )
        result = result.merge(
            entry_prices,
            on=["Ticker", entry_col],
            how="left",
            validate="many_to_one",
        ).merge(
            exit_prices,
            on=["Ticker", exit_col],
            how="left",
            validate="many_to_one",
        )
        result[f"forward_return_{horizon}d"] = (
            result["_exit_price"] / result["_entry_price"] - 1
        )
        result = result.drop(columns=["_entry_price", "_exit_price"])
    return result
