from pathlib import Path

from systematic_alpha.config import Period, ResearchConfig
from systematic_alpha.pipeline import run_pipeline


def test_pipeline_generates_frozen_test_outputs(market_data, tmp_path: Path):
    config = ResearchConfig(
        train=Period("2018-01-01", "2019-12-31"),
        validation=Period("2020-01-01", "2021-12-31"),
        test=Period("2022-01-01", "2023-12-31"),
        winsor_lower=0.01,
        winsor_upper=0.99,
        rolling_window=252,
        quantiles=5,
        long_exposure=0.5,
        short_exposure=-0.5,
        rebalance_frequencies=(1, 5, 10),
        transaction_cost_bps=5,
        ic_method="spearman",
        hac_lags=5,
        annualization_factor=252,
        bootstrap_simulations=100,
    )
    summary = run_pipeline(market_data, config, tmp_path)
    assert set(summary["sample"]) == {"validation", "historical_holdout"}
    test_rows = summary[summary["sample"] == "historical_holdout"]
    assert len(test_rows) == 10
    assert test_rows["selected_on_validation"].all()
    assert (tmp_path / "run_metadata.json").exists()
    assert (tmp_path / "research_report.md").exists()
    assert (tmp_path / "tables" / "performance_summary.csv").exists()
    assert (tmp_path / "tables" / "rolling_factor_weights.csv").exists()
    assert (tmp_path / "figures" / "validation_vs_test_sharpe.png").exists()
    assert (tmp_path / "figures" / "historical_holdout_net_wealth.png").exists()
    assert (tmp_path / "tables" / "mean_signal_rank_correlation.csv").exists()
    assert (tmp_path / "tables" / "sample_exclusion_summary.csv").exists()
    assert (tmp_path / "tables" / "bootstrap_performance_intervals.csv").exists()
    assert (tmp_path / "tables" / "incremental_model_comparison.csv").exists()
    assert (tmp_path / "tables" / "cost_sensitivity.csv").exists()
    selected = summary.loc[
        (summary["sample"] == "validation") & summary["selected_on_validation"]
    ]
    assert selected["rebalance_frequency"].nunique() == 1
    assert selected["break_even_cost_bps"].notna().all()
