"""Dollar-neutral portfolio construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_weights(
    data: pd.DataFrame,
    signal_col: str,
    rebalance_frequency: int,
    quantiles: int = 5,
    long_exposure: float = 0.5,
    short_exposure: float = -0.5,
    minimum_cross_section_size: int | None = None,
    selection_eligibility_col: str = "eligible",
) -> pd.DataFrame:
    """Build target weights and mark dates on which they are reset."""
    if rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be positive")
    if quantiles < 2:
        raise ValueError("quantiles must be at least two")
    if long_exposure <= 0 or short_exposure >= 0:
        raise ValueError("long_exposure must be positive and short_exposure negative")
    if not np.isclose(long_exposure + short_exposure, 0.0):
        raise ValueError("long and short exposures must be dollar neutral")
    if data.duplicated(["Date", "Ticker"]).any():
        raise ValueError("Duplicate Date/Ticker observations found")
    minimum_size = minimum_cross_section_size or quantiles
    if minimum_size < quantiles:
        raise ValueError("minimum_cross_section_size must be at least quantiles")
    columns = ["Date", "Ticker", signal_col]
    if selection_eligibility_col in data.columns:
        columns.append(selection_eligibility_col)
    frame = data[columns].copy()
    if selection_eligibility_col not in frame.columns:
        frame[selection_eligibility_col] = True
    frame["Date"] = pd.to_datetime(frame["Date"])
    dates = np.array(sorted(frame["Date"].unique()))
    rebalance_dates = set(dates[::rebalance_frequency])
    universe = sorted(frame["Ticker"].unique())
    current = pd.Series(0.0, index=universe)
    rows: list[pd.DataFrame] = []
    for date in dates:
        daily = frame.loc[
            (frame["Date"] == date) & frame[selection_eligibility_col]
        ].dropna(subset=[signal_col])
        rebalanced = False
        if date in rebalance_dates and len(daily) >= minimum_size:
            # Average ranks keep membership independent of input or ticker
            # ordering when a signal contains ties.
            ranks = daily[signal_col].rank(method="average", pct=True)
            long_names = daily.loc[ranks > 1 - 1 / quantiles, "Ticker"]
            short_names = daily.loc[ranks <= 1 / quantiles, "Ticker"]
            if len(long_names) and len(short_names):
                current = pd.Series(0.0, index=universe)
                current.loc[long_names] = long_exposure / len(long_names)
                current.loc[short_names] = short_exposure / len(short_names)
                rebalanced = True
        rows.append(
            pd.DataFrame(
                {
                    "Date": date,
                    "Ticker": universe,
                    "weight": current.values,
                    "is_rebalance": rebalanced,
                }
            )
        )
    return (
        pd.concat(rows, ignore_index=True)
        if rows
        else pd.DataFrame(columns=["Date", "Ticker", "weight", "is_rebalance"])
    )


def simulate_portfolio(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    return_col: str = "forward_return_1d",
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Simulate returns, natural weight drift and trades back to targets.

    A target is installed before the date's executable forward return. Between
    target dates, risky-asset weights drift with their individual returns. This
    avoids the old combination of constant weights and zero rebalancing cost.
    """
    required = {"Date", "Ticker", "weight", "is_rebalance"}
    missing = required.difference(target_weights.columns)
    if missing:
        raise ValueError(f"Missing target-weight columns: {sorted(missing)}")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")

    weight_matrix = target_weights.pivot(
        index="Date", columns="Ticker", values="weight"
    ).fillna(0.0)
    rebalance_flags = target_weights.groupby("Date")["is_rebalance"].any()
    return_matrix = returns.pivot(index="Date", columns="Ticker", values=return_col)
    return_matrix = return_matrix.reindex(
        index=weight_matrix.index, columns=weight_matrix.columns
    )
    if "eligible" in returns.columns:
        eligibility = returns.pivot(
            index="Date", columns="Ticker", values="eligible"
        ).reindex(index=weight_matrix.index, columns=weight_matrix.columns)
        # A missing row is a data gap, not evidence that a holding left the
        # investable universe. Preserve NA so it cannot become a free exit.
        eligibility = eligibility.astype("boolean")
    else:
        eligibility = pd.DataFrame(True, index=weight_matrix.index, columns=weight_matrix.columns)

    current = pd.Series(0.0, index=weight_matrix.columns)
    rows: list[dict[str, object]] = []
    for date in weight_matrix.index:
        target = weight_matrix.loc[date]
        active_unknown_eligibility = current.ne(0) & eligibility.loc[date].isna()
        if active_unknown_eligibility.any():
            tickers = ", ".join(
                active_unknown_eligibility[active_unknown_eligibility].index[:5]
            )
            raise ValueError(
                "Missing eligibility for active positions; refusing to infer a "
                f"forced exit. Examples: {tickers}@{pd.Timestamp(date).date()}"
            )
        explicitly_ineligible = eligibility.loc[date].fillna(False).eq(False)
        ineligible_position = current.ne(0) & explicitly_ineligible
        forced_exit_turnover = float(current.loc[ineligible_position].abs().sum())
        current.loc[ineligible_position] = 0.0
        if bool(rebalance_flags.loc[date]):
            scheduled_turnover = float(target.sub(current).abs().sum())
            current = target.copy()
        else:
            scheduled_turnover = 0.0
        turnover = forced_exit_turnover + scheduled_turnover
        transaction_cost = turnover * cost_bps / 10_000

        daily_returns = return_matrix.loc[date]
        active_missing = current.ne(0) & daily_returns.isna()
        if active_missing.any():
            tickers = ", ".join(active_missing[active_missing].index[:5])
            raise ValueError(
                "Missing forward returns for active positions; refusing to assume "
                f"a zero return. Examples: {tickers}@{pd.Timestamp(date).date()}"
            )
        filled_returns = daily_returns.fillna(0.0)
        gross_return = float(current.mul(filled_returns).sum())
        net_return = gross_return - transaction_cost
        denominator = 1.0 + net_return
        if denominator <= 0:
            raise ValueError("Portfolio wealth became non-positive")
        current = current.mul(1.0 + filled_returns).div(denominator)
        rows.append(
            {
                "Date": date,
                "gross_return": gross_return,
                "turnover": turnover,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "scheduled_turnover": scheduled_turnover,
                "forced_exit_turnover": forced_exit_turnover,
                "gross_exposure_end": float(current.abs().sum()),
                "net_exposure_end": float(current.sum()),
            }
        )
    return pd.DataFrame(rows).set_index("Date")


def portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    return_col: str = "forward_return_1d",
) -> pd.Series:
    merged = weights.merge(
        returns[["Date", "Ticker", return_col]], on=["Date", "Ticker"], how="left"
    )
    valid_calendar_dates = returns.loc[returns[return_col].notna(), "Date"].unique()
    merged = merged.loc[merged["Date"].isin(valid_calendar_dates)].copy()
    active_missing = merged[return_col].isna() & merged["weight"].ne(0)
    if active_missing.any():
        examples = merged.loc[active_missing, ["Date", "Ticker"]].head(5)
        labels = ", ".join(
            f"{row.Ticker}@{pd.Timestamp(row.Date).date()}"
            for row in examples.itertuples(index=False)
        )
        raise ValueError(
            "Missing forward returns for active positions; refusing to assume a "
            f"zero return. Examples: {labels}"
        )
    merged["contribution"] = merged["weight"] * merged[return_col].fillna(0.0)
    return merged.groupby("Date")["contribution"].sum().rename("gross_return")
