"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class Period:
    start: str
    end: str


@dataclass(frozen=True)
class ResearchConfig:
    train: Period
    validation: Period
    test: Period
    winsor_lower: float
    winsor_upper: float
    rolling_window: int
    quantiles: int
    long_exposure: float
    short_exposure: float
    rebalance_frequencies: tuple[int, ...]
    transaction_cost_bps: float
    ic_method: str
    hac_lags: int
    annualization_factor: int
    alpha52_low_window: int = 5
    alpha52_low_delay: int = 5
    alpha52_momentum_recent_skip: int = 20
    alpha52_momentum_lookback: int = 240
    alpha52_volume_rank_window: int = 5
    minimum_cross_section_size: int = 5
    universe_mode: str = "current_constituents"
    membership_path: str | None = None
    frequency_selection_signal: str = "baseline_equal"
    bootstrap_simulations: int = 500
    bootstrap_block_length: int = 21
    bootstrap_seed: int = 42
    cost_scenarios_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 20.0)


def load_config(path: str | Path) -> ResearchConfig:
    """Load and validate the research configuration."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    research = raw["research"]
    signals = raw["signals"]
    portfolio = raw["portfolio"]
    statistics = raw["statistics"]
    data_config = raw.get("data", {})
    alpha52 = signals.get("alpha52_robust", {})
    config = ResearchConfig(
        train=Period(**research["train"]),
        validation=Period(**research["validation"]),
        test=Period(**research["test"]),
        winsor_lower=float(signals["winsor_lower"]),
        winsor_upper=float(signals["winsor_upper"]),
        rolling_window=int(signals["rolling_window"]),
        quantiles=int(portfolio["quantiles"]),
        long_exposure=float(portfolio["long_exposure"]),
        short_exposure=float(portfolio["short_exposure"]),
        rebalance_frequencies=tuple(portfolio["rebalance_frequencies"]),
        transaction_cost_bps=float(portfolio["transaction_cost_bps"]),
        ic_method=str(statistics["ic_method"]),
        hac_lags=int(statistics["hac_lags"]),
        annualization_factor=int(statistics["annualization_factor"]),
        alpha52_low_window=int(alpha52.get("low_window", 5)),
        alpha52_low_delay=int(alpha52.get("low_delay", 5)),
        alpha52_momentum_recent_skip=int(alpha52.get("momentum_recent_skip", 20)),
        alpha52_momentum_lookback=int(alpha52.get("momentum_lookback", 240)),
        alpha52_volume_rank_window=int(alpha52.get("volume_rank_window", 5)),
        minimum_cross_section_size=int(research.get("minimum_cross_section_size", 5)),
        universe_mode=str(data_config.get("universe_mode", "current_constituents")),
        membership_path=data_config.get("membership_path"),
        frequency_selection_signal=str(
            research.get("frequency_selection_signal", "baseline_equal")
        ),
        bootstrap_simulations=int(statistics.get("bootstrap_simulations", 500)),
        bootstrap_block_length=int(statistics.get("bootstrap_block_length", 21)),
        bootstrap_seed=int(statistics.get("bootstrap_seed", 42)),
        cost_scenarios_bps=tuple(
            float(value)
            for value in portfolio.get("cost_scenarios_bps", [0, 2, 5, 10, 20])
        ),
    )
    _validate(config)
    return config


def _validate(config: ResearchConfig) -> None:
    if not 0 <= config.winsor_lower < config.winsor_upper <= 1:
        raise ValueError("Winsorization limits must satisfy 0 <= lower < upper <= 1")
    if config.long_exposure <= 0 or config.short_exposure >= 0:
        raise ValueError("Long exposure must be positive and short exposure negative")
    if abs(config.long_exposure + config.short_exposure) > 1e-12:
        raise ValueError("The portfolio must be dollar neutral")
    if min(config.rebalance_frequencies) < 1:
        raise ValueError("Rebalance frequencies must be positive")
    if config.rolling_window < 2:
        raise ValueError("Rolling window must be at least two observations")
    if config.minimum_cross_section_size < config.quantiles:
        raise ValueError(
            "minimum_cross_section_size must be at least the number of quantiles"
        )
    if config.bootstrap_simulations < 100:
        raise ValueError("bootstrap_simulations must be at least 100")
    if config.bootstrap_block_length < 1:
        raise ValueError("bootstrap_block_length must be positive")
    if not config.cost_scenarios_bps or min(config.cost_scenarios_bps) < 0:
        raise ValueError("cost_scenarios_bps must be non-empty and non-negative")
    if config.universe_mode not in {"current_constituents", "point_in_time"}:
        raise ValueError("universe_mode must be current_constituents or point_in_time")
    if (
        min(
            config.alpha52_low_window,
            config.alpha52_low_delay,
            config.alpha52_momentum_recent_skip,
            config.alpha52_volume_rank_window,
        )
        < 1
    ):
        raise ValueError("Alpha52 windows and delays must be positive")
    if config.alpha52_momentum_lookback <= config.alpha52_momentum_recent_skip:
        raise ValueError("Alpha52 lookback must exceed its recent-period skip")
    periods = (config.train, config.validation, config.test)
    bounds = [(pd.Timestamp(item.start), pd.Timestamp(item.end)) for item in periods]
    if any(start > end for start, end in bounds):
        raise ValueError("Each research period must start on or before it ends")
    if not (bounds[0][1] < bounds[1][0] and bounds[1][1] < bounds[2][0]):
        raise ValueError(
            "Train, validation, and test periods must be ordered and disjoint"
        )
