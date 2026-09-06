"""Market-data normalization and quality checks."""

from __future__ import annotations

import urllib.request
import warnings
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_RAW_COLUMNS = {
    "Date",
    "Ticker",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
}


def current_sp500_tickers() -> list[str]:
    """Fetch current constituents. This universe has known survivorship bias."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    table = pd.read_html(StringIO(html))[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


def download_current_sp500(
    start: str, end: str, raw_path: str | Path, processed_path: str | Path
) -> pd.DataFrame:
    """Download the disclosed, biased research universe and prepare adjusted data."""
    import yfinance as yf

    tickers = current_sp500_tickers()
    downloaded = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        actions=True,
        threads=True,
    )
    frames = []
    available = set(downloaded.columns.get_level_values(0))
    for ticker in tickers:
        if ticker not in available:
            continue
        ticker_data = downloaded[ticker].copy().reset_index()
        ticker_data["Ticker"] = ticker
        frames.append(ticker_data)
    if not frames:
        raise RuntimeError("The data provider returned no usable ticker data")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.dropna(subset=["Open", "High", "Low", "Close", "Adj Close"], how="all")
    raw_destination = Path(raw_path)
    processed_destination = Path(processed_path)
    raw_destination.parent.mkdir(parents=True, exist_ok=True)
    processed_destination.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_destination, index=False)
    processed = adjust_ohlcv(raw)
    processed.to_csv(processed_destination, index=False)
    return processed


def validate_raw_data(data: pd.DataFrame) -> None:
    """Raise a clear error when raw market data violates the input contract."""
    missing = REQUIRED_RAW_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing raw columns: {sorted(missing)}")
    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker observations found")
    if data[["Date", "Ticker"]].isna().any().any():
        raise ValueError("Date and Ticker must not be missing")
    prices = data[["Open", "High", "Low", "Close", "Adj Close"]]
    if (prices.dropna() <= 0).any().any():
        raise ValueError("Prices must be positive")


def adjust_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Create internally consistent split/dividend-adjusted OHLCV columns.

    The adjustment factor follows the provider's Adj Close definition. Volume is
    inversely adjusted so price-volume scale changes do not create split signals.
    """
    validate_raw_data(data)
    result = data.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    factor = result["Adj Close"] / result["Close"]
    factor = factor.replace([np.inf, -np.inf], np.nan)
    for source, target in (
        ("Open", "open_adj"),
        ("High", "high_adj"),
        ("Low", "low_adj"),
        ("Close", "close_adj"),
    ):
        result[target] = result[source] * factor
    if "Stock Splits" in result.columns:
        result = result.sort_values(["Ticker", "Date"])
        ratios = result["Stock Splits"].fillna(0).replace(0, 1.0).astype(float)
        future_product = ratios.groupby(result["Ticker"]).transform(
            lambda values: values.iloc[::-1].cumprod().iloc[::-1]
        )
        split_adjustment = 1.0 / (future_product / ratios)
        result["volume_adj"] = result["Volume"] / split_adjustment
    else:
        warnings.warn(
            "Stock Splits column is absent; volume uses the adjusted-close factor. "
            "Supply explicit corporate actions for split-only volume adjustment.",
            UserWarning,
            stacklevel=2,
        )
        result["volume_adj"] = result["Volume"] / factor
    columns = [
        "Date",
        "Ticker",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "volume_adj",
    ]
    return result[columns].sort_values(["Ticker", "Date"]).reset_index(drop=True)


def validate_processed_data(data: pd.DataFrame) -> dict[str, float | int | str]:
    required = {
        "Date",
        "Ticker",
        "open_adj",
        "high_adj",
        "low_adj",
        "close_adj",
        "volume_adj",
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing processed columns: {sorted(missing)}")
    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker observations found")
    if data[["Date", "Ticker"]].isna().any().any():
        raise ValueError("Date and Ticker must not be missing")
    dates = pd.to_datetime(data["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Date contains invalid values")
    numeric = ["open_adj", "high_adj", "low_adj", "close_adj", "volume_adj"]
    if (data[numeric].dropna() <= 0).any().any():
        raise ValueError("Adjusted prices and volume must be positive")
    return {
        "rows": len(data),
        "tickers": int(data["Ticker"].nunique()),
        "start": str(dates.min().date()),
        "end": str(dates.max().date()),
        "missing_fraction": float(data[list(required)].isna().mean().mean()),
    }


def apply_point_in_time_universe(
    market_data: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Filter observations using inclusive point-in-time membership intervals."""
    required = {"Ticker", "StartDate", "EndDate"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Missing membership columns: {sorted(missing)}")
    intervals = membership.copy()
    intervals["StartDate"] = pd.to_datetime(intervals["StartDate"], errors="coerce")
    intervals["EndDate"] = pd.to_datetime(intervals["EndDate"], errors="coerce")
    if intervals["StartDate"].isna().any():
        raise ValueError("Membership StartDate contains invalid values")
    if intervals.duplicated(["Ticker", "StartDate", "EndDate"]).any():
        raise ValueError("Duplicate membership intervals found")
    data = market_data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    merged = data.merge(intervals, on="Ticker", how="inner", validate="many_to_many")
    active = merged["Date"].ge(merged["StartDate"]) & (
        merged["EndDate"].isna() | merged["Date"].le(merged["EndDate"])
    )
    result = merged.loc[active, market_data.columns].copy()
    if result.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Overlapping membership intervals create duplicate observations")
    return result.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def mark_point_in_time_eligibility(
    market_data: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    """Retain all prices while marking whether each security may be selected.

    Keeping post-membership prices lets the simulator close an existing position
    instead of losing the price series as soon as a constituent leaves the index.
    """
    required = {"Ticker", "StartDate", "EndDate"}
    missing = required.difference(membership.columns)
    if missing:
        raise ValueError(f"Missing membership columns: {sorted(missing)}")
    intervals = membership.copy()
    intervals["StartDate"] = pd.to_datetime(intervals["StartDate"], errors="coerce")
    intervals["EndDate"] = pd.to_datetime(intervals["EndDate"], errors="coerce")
    if intervals["StartDate"].isna().any():
        raise ValueError("Membership StartDate contains invalid values")
    if intervals.duplicated(["Ticker", "StartDate", "EndDate"]).any():
        raise ValueError("Duplicate membership intervals found")
    data = market_data.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    keyed = data[["Date", "Ticker"]].merge(
        intervals, on="Ticker", how="left", validate="many_to_many"
    )
    keyed["eligible"] = keyed["Date"].ge(keyed["StartDate"]) & (
        keyed["EndDate"].isna() | keyed["Date"].le(keyed["EndDate"])
    )
    eligibility = keyed.groupby(["Date", "Ticker"])["eligible"].sum()
    if eligibility.gt(1).any():
        raise ValueError("Overlapping membership intervals found")
    result = data.merge(
        eligibility.gt(0).rename("eligible").reset_index(),
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    result["eligible"] = result["eligible"].fillna(False).astype(bool)
    return result.sort_values(["Ticker", "Date"]).reset_index(drop=True)
