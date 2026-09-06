"""End-to-end research orchestration with frozen test specifications."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .combination import (
    add_composite,
    add_equal_composite,
    add_rolling_composite,
    fit_static_weights,
)
from .config import ResearchConfig
from .costs import apply_linear_costs
from .data import validate_processed_data
from .inference import bootstrap_paired_difference, bootstrap_performance_interval
from .metrics import performance_statistics
from .portfolio import build_weights, simulate_portfolio
from .reporting import write_core_figures, write_research_report
from .returns import add_forward_returns
from .signals import generate_signals
from .validation import daily_ic, newey_west_mean_test, select_period

SIGNALS = ("reversal", "alpha012_robust", "alpha101", "alpha52_robust")
EQUAL_MODELS = {
    "baseline_equal": ("reversal", "alpha012_robust"),
    "plus_alpha101": ("reversal", "alpha012_robust", "alpha101"),
    "plus_alpha52": ("reversal", "alpha012_robust", "alpha52_robust"),
    "four_alpha_equal": SIGNALS,
}


def build_common_sample(
    data: pd.DataFrame,
    signal_columns: tuple[str, ...],
    minimum_cross_section_size: int,
) -> pd.DataFrame:
    """Return the ex-ante security-date sample used for portfolio selection.

    Selection must depend only on information available on the signal date.
    In particular, whether a future return label can be computed must never
    decide whether a security is allowed into the cross-sectional ranking.
    """
    required = [*signal_columns]
    missing = set(required).difference(data.columns)
    if missing:
        raise ValueError(f"Missing common-sample columns: {sorted(missing)}")
    eligible = (
        data["eligible"].fillna(False)
        if "eligible" in data.columns
        else pd.Series(True, index=data.index)
    )
    common = data.loc[data[required].notna().all(axis=1) & eligible].copy()
    sizes = common.groupby("Date").size()
    valid_dates = sizes[sizes >= minimum_cross_section_size].index
    return common.loc[common["Date"].isin(valid_dates)].copy()


def build_evaluation_sample(
    selection_sample: pd.DataFrame,
    minimum_cross_section_size: int,
    return_col: str = "forward_return_1d",
) -> pd.DataFrame:
    """Return the label-complete sample used only for IC and inference."""
    if return_col not in selection_sample.columns:
        raise ValueError(f"Missing evaluation column: {return_col}")
    evaluation = selection_sample.loc[selection_sample[return_col].notna()].copy()
    sizes = evaluation.groupby("Date").size()
    valid_dates = sizes[sizes >= minimum_cross_section_size].index
    return evaluation.loc[evaluation["Date"].isin(valid_dates)].copy()


def _evaluate(
    data: pd.DataFrame,
    signal: str,
    frequency: int,
    config: ResearchConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    usable = data.loc[
        data.groupby("Date")["forward_return_1d"].transform("count").gt(0)
    ].copy()
    weights = build_weights(
        usable,
        signal,
        frequency,
        config.quantiles,
        config.long_exposure,
        config.short_exposure,
        minimum_cross_section_size=config.minimum_cross_section_size,
        selection_eligibility_col="selection_eligible",
    )
    simulation = simulate_portfolio(weights, usable)
    result = apply_linear_costs(
        simulation["gross_return"],
        simulation["turnover"],
        config.transaction_cost_bps,
    )
    result[["gross_exposure_end", "net_exposure_end"]] = simulation[
        ["gross_exposure_end", "net_exposure_end"]
    ]
    metrics = performance_statistics(result["net_return"], config.annualization_factor)
    average_turnover = float(result["turnover"].mean())
    metrics["average_daily_turnover"] = average_turnover
    metrics["annualized_turnover"] = average_turnover * config.annualization_factor
    metrics["break_even_cost_bps"] = (
        float(result["gross_return"].mean() / average_turnover * 10_000)
        if average_turnover > 0
        else float("nan")
    )
    return result, metrics


def run_pipeline(
    processed_data: pd.DataFrame,
    config: ResearchConfig,
    output_dir: str | Path,
    membership_sha256: str | None = None,
) -> pd.DataFrame:
    """Run research once, selecting parameters only on the validation sample."""
    output = Path(output_dir)
    tables = output / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    validate_processed_data(processed_data)
    data = generate_signals(
        processed_data,
        config.winsor_lower,
        config.winsor_upper,
        config.alpha52_low_window,
        config.alpha52_low_delay,
        config.alpha52_momentum_recent_skip,
        config.alpha52_momentum_lookback,
        config.alpha52_volume_rank_window,
    )
    data = add_forward_returns(data)

    # Keep every price row for exits, but mark an identical eligible security-date
    # sample for all model comparisons. Non-members never affect cross-sectional
    # transforms, ICs, cross-section counts or new portfolio selections.
    selection_sample = build_common_sample(
        data, SIGNALS, config.minimum_cross_section_size
    )
    evaluation_sample = build_evaluation_sample(
        selection_sample, config.minimum_cross_section_size
    )
    common_keys = selection_sample[["Date", "Ticker"]].assign(
        selection_eligible=True
    )
    data = data.merge(
        common_keys,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    data["selection_eligible"] = data["selection_eligible"].fillna(False)
    evaluation_keys = evaluation_sample[["Date", "Ticker"]].assign(
        evaluation_eligible=True
    )
    data = data.merge(
        evaluation_keys,
        on=["Date", "Ticker"],
        how="left",
        validate="one_to_one",
    )
    data["evaluation_eligible"] = data["evaluation_eligible"].fillna(False)

    train = select_period(
        evaluation_sample, config.train, require_label_within=True
    )
    validation = select_period(
        evaluation_sample, config.validation, require_label_within=True
    )
    test = select_period(
        evaluation_sample, config.test, require_label_within=True
    )
    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test samples must all be non-empty")

    train_ic = pd.concat(
        [daily_ic(train, signal, method=config.ic_method) for signal in SIGNALS],
        axis=1,
        sort=True,
    )
    factor_weights = fit_static_weights(train_ic)
    data = add_composite(data, factor_weights)
    data = data.rename(columns={"composite": "optimized_composite"})
    for model_name, components in EQUAL_MODELS.items():
        data = add_equal_composite(data, components, model_name)
    all_ic = pd.concat(
        [
            daily_ic(evaluation_sample, signal, method=config.ic_method)
            for signal in SIGNALS
        ],
        axis=1,
        sort=True,
    )
    availability = (
        evaluation_sample[["Date", "exit_date_1d"]]
        .drop_duplicates()
        .set_index("Date")["exit_date_1d"]
        .rename("available_date")
    )
    if availability.index.duplicated().any():
        raise ValueError("A signal date maps to multiple IC availability dates")
    all_ic = all_ic.join(availability, how="left")
    data, rolling_weights = add_rolling_composite(
        data,
        all_ic,
        window=config.rolling_window,
        min_periods=min(60, config.rolling_window),
    )
    research_data = data.loc[data["evaluation_eligible"]].copy()
    train = select_period(research_data, config.train, require_label_within=True)
    validation_research = select_period(
        research_data, config.validation, require_label_within=True
    )
    test_research = select_period(research_data, config.test, require_label_within=True)
    validation = select_period(data, config.validation, require_label_within=True)
    test = select_period(data, config.test, require_label_within=True)

    evaluation_signals = (
        *SIGNALS,
        *EQUAL_MODELS,
        "optimized_composite",
        "rolling_composite",
    )
    validation_evaluation = validation
    test_evaluation = test
    if validation_evaluation.empty or test_evaluation.empty:
        raise ValueError("No common four-alpha evaluation sample is available")

    selector = config.frequency_selection_signal
    if selector not in evaluation_signals:
        raise ValueError(f"Unknown frequency_selection_signal: {selector}")
    selector_candidates = []
    for frequency in config.rebalance_frequencies:
        _, selector_metrics = _evaluate(
            validation_evaluation, selector, frequency, config
        )
        selector_candidates.append((frequency, selector_metrics))
    selected_frequency, _ = max(
        selector_candidates,
        key=lambda item: (
            item[1]["sharpe"] if pd.notna(item[1]["sharpe"]) else float("-inf")
        ),
    )

    summary_rows: list[dict[str, object]] = []
    validation_returns_by_signal: dict[str, pd.DataFrame] = {}
    test_returns_by_signal: dict[str, pd.DataFrame] = {}
    for signal in evaluation_signals:
        candidates: list[tuple[int, dict[str, float]]] = []
        for frequency in config.rebalance_frequencies:
            validation_returns, metrics = _evaluate(
                validation_evaluation, signal, frequency, config
            )
            if frequency == selected_frequency:
                validation_returns_by_signal[signal] = validation_returns
            candidates.append((frequency, metrics))
            summary_rows.append(
                {
                    "signal": signal,
                    "sample": "validation",
                    "rebalance_frequency": frequency,
                    "selected_on_validation": False,
                    **metrics,
                }
            )
        for row in summary_rows:
            if (
                row["signal"] == signal
                and row["sample"] == "validation"
                and row["rebalance_frequency"] == selected_frequency
            ):
                row["selected_on_validation"] = True
        test_returns, test_metrics = _evaluate(
            test_evaluation, signal, selected_frequency, config
        )
        test_returns_by_signal[signal] = test_returns
        test_returns.to_csv(tables / f"{signal}_historical_holdout_returns.csv")
        summary_rows.append(
            {
                "signal": signal,
                "sample": "historical_holdout",
                "rebalance_frequency": selected_frequency,
                "selected_on_validation": True,
                **test_metrics,
            }
        )

    ic_rows = []
    for sample_name, sample in (
        ("train", train),
        ("validation", validation_research),
        ("historical_holdout", test_research),
    ):
        for signal in evaluation_signals:
            values = daily_ic(sample, signal, method=config.ic_method)
            ic_rows.append(
                {
                    "sample": sample_name,
                    "signal": signal,
                    **newey_west_mean_test(values, config.hac_lags),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(tables / "performance_summary.csv", index=False)
    _write_bootstrap_inference(
        validation_returns_by_signal,
        test_returns_by_signal,
        config,
        tables,
    )
    _write_cost_sensitivity(
        validation_returns_by_signal,
        test_returns_by_signal,
        config,
        tables / "cost_sensitivity.csv",
    )
    pd.DataFrame(ic_rows).to_csv(tables / "ic_summary.csv", index=False)
    signal_correlation = (
        data.groupby("Date")[list(SIGNALS)]
        .corr(method="spearman")
        .groupby(level=1)
        .mean()
    )
    signal_correlation.to_csv(tables / "mean_signal_rank_correlation.csv")
    factor_weights.rename_axis("signal").reset_index().to_csv(
        tables / "training_factor_weights.csv", index=False
    )
    rolling_weights.rename_axis("Date").reset_index().to_csv(
        tables / "rolling_factor_weights.csv", index=False
    )
    _write_sample_audit(data, config, tables / "sample_exclusion_summary.csv")
    write_research_report(summary, factor_weights, output / "research_report.md")
    figure_ic = pd.concat(
        [
            daily_ic(data, signal, method=config.ic_method)
            for signal in evaluation_signals
        ],
        axis=1,
        sort=True,
    )
    write_core_figures(
        summary,
        test_returns_by_signal,
        figure_ic,
        rolling_weights,
        output / "figures",
    )
    _write_metadata(output, config, processed_data, membership_sha256)
    return summary


def _write_sample_audit(
    data: pd.DataFrame, config: ResearchConfig, path: Path
) -> None:
    """Record the observations purged at research-sample boundaries."""
    rows: list[dict[str, object]] = []
    for sample_name, period in (
        ("train", config.train),
        ("validation", config.validation),
        ("historical_holdout", config.test),
    ):
        dated = select_period(data, period)
        contained = select_period(data, period, require_label_within=True)
        rows.append(
            {
                "sample": sample_name,
                "signal_date_observations": len(dated),
                "label_contained_observations": len(contained),
                "boundary_or_missing_label_exclusions": len(dated) - len(contained),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_bootstrap_inference(
    validation_returns: dict[str, pd.DataFrame],
    test_returns: dict[str, pd.DataFrame],
    config: ResearchConfig,
    tables: Path,
) -> None:
    interval_rows: list[dict[str, object]] = []
    incremental_rows: list[dict[str, object]] = []
    for sample_name, frames in (
        ("validation", validation_returns),
        ("historical_holdout", test_returns),
    ):
        baseline = frames["baseline_equal"]["net_return"]
        for offset, (signal, frame) in enumerate(frames.items()):
            series = frame["net_return"]
            interval_rows.append(
                {
                    "sample": sample_name,
                    "signal": signal,
                    **bootstrap_performance_interval(
                        series,
                        annualization=config.annualization_factor,
                        block_length=config.bootstrap_block_length,
                        simulations=config.bootstrap_simulations,
                        seed=config.bootstrap_seed + offset,
                    ),
                }
            )
            if signal != "baseline_equal":
                incremental_rows.append(
                    {
                        "sample": sample_name,
                        "candidate": signal,
                        "baseline": "baseline_equal",
                        **bootstrap_paired_difference(
                            series,
                            baseline,
                            annualization=config.annualization_factor,
                            block_length=config.bootstrap_block_length,
                            simulations=config.bootstrap_simulations,
                            seed=config.bootstrap_seed + offset,
                        ),
                    }
                )
    pd.DataFrame(interval_rows).to_csv(
        tables / "bootstrap_performance_intervals.csv", index=False
    )
    pd.DataFrame(incremental_rows).to_csv(
        tables / "incremental_model_comparison.csv", index=False
    )


def _write_cost_sensitivity(
    validation_returns: dict[str, pd.DataFrame],
    test_returns: dict[str, pd.DataFrame],
    config: ResearchConfig,
    path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    for sample_name, frames in (
        ("validation", validation_returns),
        ("historical_holdout", test_returns),
    ):
        for signal, frame in frames.items():
            for cost_bps in config.cost_scenarios_bps:
                net = apply_linear_costs(
                    frame["gross_return"], frame["turnover"], cost_bps
                )
                rows.append(
                    {
                        "sample": sample_name,
                        "signal": signal,
                        "cost_bps": cost_bps,
                        **performance_statistics(
                            net["net_return"], config.annualization_factor
                        ),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_metadata(
    output: Path,
    config: ResearchConfig,
    data: pd.DataFrame,
    membership_sha256: str | None = None,
) -> None:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    canonical = data.sort_values(["Date", "Ticker"]).to_csv(index=False).encode()
    validation = validate_processed_data(data)
    tickers = sorted(data["Ticker"].astype(str).unique())
    eligibility_sha256 = None
    if "eligible" in data.columns:
        eligibility = data[["Date", "Ticker", "eligible"]].sort_values(
            ["Date", "Ticker"]
        )
        eligibility_sha256 = hashlib.sha256(
            eligibility.to_csv(index=False).encode()
        ).hexdigest()
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "rows": len(data),
        "tickers": int(data["Ticker"].nunique()),
        "universe_tickers": tickers,
        "universe_sha256": hashlib.sha256("\n".join(tickers).encode()).hexdigest(),
        "research_spec_sha256": hashlib.sha256(
            json.dumps(asdict(config), sort_keys=True).encode()
        ).hexdigest(),
        "data_sha256": hashlib.sha256(canonical).hexdigest(),
        "data_start": validation["start"],
        "data_end": validation["end"],
        "missing_fraction": validation["missing_fraction"],
        "universe_mode": config.universe_mode,
        "membership_path": config.membership_path,
        "membership_sha256": membership_sha256,
        "eligibility_sha256": eligibility_sha256,
        "return_mode": "next_adjusted_open_to_open",
        "return_alignment": "shared_market_calendar_exact_date_join",
        "sample_boundary_policy": "entry_and_exit_dates_must_remain_within_sample",
        "selection_sample_policy": "signal_complete_ex_ante_no_forward_label_filter",
        "evaluation_sample_policy": "selection_sample_plus_complete_forward_label",
        "transaction_cost_bps": config.transaction_cost_bps,
        "cost_scenarios_bps": config.cost_scenarios_bps,
        "gross_exposure": config.long_exposure + abs(config.short_exposure),
        "historical_holdout_used_for_selection": False,
        "frequency_selection_signal": config.frequency_selection_signal,
        "frequency_selected_once_for_all_models": True,
        "bootstrap_method": "circular_moving_block",
        "bootstrap_simulations": config.bootstrap_simulations,
        "bootstrap_block_length": config.bootstrap_block_length,
        "result_status": (
            "point_in_time_research"
            if config.universe_mode == "point_in_time"
            else "survivorship_biased_demo"
        ),
        "active_missing_return_policy": "raise",
        "rolling_ic_policy": "available_date_lte_decision_date",
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
