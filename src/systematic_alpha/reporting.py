"""Human-readable research report generation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_research_report(
    summary: pd.DataFrame, factor_weights: pd.Series, destination: str | Path
) -> None:
    """Write a compact report separating selection from final evaluation."""
    selected = summary.loc[
        (summary["sample"] == "validation") & summary["selected_on_validation"]
    ]
    test = summary.loc[summary["sample"] == "historical_holdout"]
    lines = [
        "# Reproduced research report",
        "",
        "The baseline selects one validation rebalance frequency for every model.",
        "The historical holdout is",
        "evaluated with the frozen specification and is not used for selection.",
        "",
        "## Training-period factor weights",
        "",
        "| Signal | Weight |",
        "|---|---:|",
    ]
    for signal, weight in factor_weights.items():
        lines.append(f"| {signal} | {weight:.2%} |")
    lines.extend(
        [
            "",
            "## Frozen-specification performance",
            "",
            "| Signal | Sample | Rebalance | Net Sharpe | Net annualized return |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in pd.concat([selected, test]).iterrows():
        lines.append(
            f"| {row['signal']} | {row['sample']} | "
            f"{int(row['rebalance_frequency'])}D | {row['sharpe']:.2f} | "
            f"{row['annualized_return']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- Holdout results are not used to revise weights or rebalance frequency.",
            "- One baseline-selected frequency is applied to every model comparison.",
            "- Bootstrap uncertainty tables use dependence-aware moving blocks.",
            "- Costs use full weight-based turnover and include initial entry.",
            "- The current-constituent universe has survivorship bias.",
            "- Linear costs omit market impact and short-borrow constraints.",
            "",
        ]
    )
    Path(destination).write_text("\n".join(lines), encoding="utf-8")


def write_core_figures(
    performance: pd.DataFrame,
    returns_by_signal: dict[str, pd.DataFrame],
    ic_by_signal: pd.DataFrame,
    rolling_weights: pd.DataFrame,
    destination: str | Path,
) -> None:
    """Generate the figures referenced by the reproducible report."""
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)

    selected = performance.loc[
        (performance["sample"] == "validation") & performance["selected_on_validation"]
    ]
    test = performance.loc[performance["sample"] == "historical_holdout"]
    comparison = pd.concat([selected, test])
    pivot = comparison.pivot(index="signal", columns="sample", values="sharpe")
    ax = pivot.plot.bar(figsize=(9, 5), color=["#2F6B8A", "#D08C60"])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(title="Frozen-specification net Sharpe", ylabel="Sharpe ratio", xlabel="")
    ax.legend(title="Sample")
    ax.figure.tight_layout()
    ax.figure.savefig(output / "validation_vs_test_sharpe.png", dpi=160)
    plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(10, 5))
    for signal, frame in returns_by_signal.items():
        wealth = (1 + frame["net_return"]).cumprod()
        ax.plot(wealth.index, wealth, label=signal)
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set(title="Historical-holdout net wealth", ylabel="Growth of 1", xlabel="")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "historical_holdout_net_wealth.png", dpi=160)
    plt.close(fig)

    if not ic_by_signal.empty:
        ax = (
            ic_by_signal.rolling(21, min_periods=10)
            .mean()
            .plot(figsize=(10, 5), title="21-day rolling mean rank IC")
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(ylabel="Rank IC", xlabel="")
        ax.figure.tight_layout()
        ax.figure.savefig(output / "rolling_rank_ic.png", dpi=160)
        plt.close(ax.figure)

    if not rolling_weights.dropna(how="all").empty:
        ax = rolling_weights.plot.area(
            figsize=(10, 5), title="Trailing-only rolling factor weights"
        )
        ax.set(ylim=(0, 1), ylabel="Weight", xlabel="")
        ax.figure.tight_layout()
        ax.figure.savefig(output / "rolling_factor_weights.png", dpi=160)
        plt.close(ax.figure)
