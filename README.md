# Systematic Alpha Research

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB)](https://www.python.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-D7FF64)](https://docs.astral.sh/ruff/)

A leakage-aware cross-sectional equity research pipeline for testing price-volume
signals under executable timing, chronological model selection, portfolio turnover,
and transaction costs.

The project asks a simple question: does a statistically detectable signal remain
economically useful once it is converted into a portfolio under realistic research
constraints? In the included case study, it does not. That negative result is the
central finding, not a result to be hidden or optimized away.

## Key finding

The case study finds small but statistically detectable rank predictability, but
portfolio performance does not remain stable out of sample. At 5 basis points per
dollar traded:

| Strategy | Sample | Rebalance | Turnover | Gross Sharpe | Net Sharpe | Net annualized return |
|---|---|---:|---:|---:|---:|---:|
| Reversal | Validation | 10D | 0.154 | 1.61 | 1.38 | 12.15% |
| Reversal | Historical evaluation | 10D | 0.155 | 0.27 | 0.04 | Approximately 0.00% |
| Static composite | Validation | 10D | 0.145 | 0.99 | 0.57 | 2.40% |
| Static composite | Historical evaluation | 10D | 0.147 | 0.30 | -0.10 | -0.57% |

![Validation versus historical-evaluation Sharpe](results/case_study/final_figures/validation_vs_test_sharpe.png)

The deterioration is evidence against treating either specification as validated
alpha. A positive information coefficient or an attractive validation Sharpe is not
enough to support an investment claim.

## What the project demonstrates

- Exact close-to-next-open signal and return alignment
- Chronological training, validation, and historical evaluation
- Purging of labels that cross sample boundaries
- Point-in-time universe eligibility support
- Identical security-date samples for model comparisons
- Training-only and trailing-only factor weighting
- Stateful portfolio simulation with natural weight drift and forced exits
- Complete turnover accounting and linear transaction-cost sensitivity
- Newey-West IC inference and paired moving-block bootstrap inference
- Locked dependencies, continuous integration, and assumption-focused tests
- Explicit documentation of negative results and remaining limitations

## Research design

| Period | Role | Permitted use |
|---|---|---|
| 2015-01-01 to 2018-12-31 | Training | Estimate static factor weights |
| 2019-01-01 to 2021-12-31 | Validation | Select one rebalance frequency |
| 2022-01-01 to 2025-07-01 | Historical evaluation | Evaluate frozen choices |

The final interval has already been inspected and is therefore called a historical
evaluation rather than an untouched test. Data after 2025-07 should be used for a
genuinely prospective evaluation.

Signals observed after the close on day `t` are paired with adjusted open-to-open
returns from `t+1` to `t+2`:

```text
Close(t)     Observe signal
Open(t+1)    Enter position
Open(t+2)    Observe one-day return
```

Entry and exit dates come from the shared observed market calendar and are joined
on exact security-date keys. A missing quote produces a missing label rather than
silently extending the holding period. Labels crossing a research-period boundary
are purged.

## Signals and comparison set

The pipeline implements four scale-aware price-volume signals:

| Signal | Horizon | Interpretation |
|---|---:|---|
| Reversal | 1 day | Short-term close-to-close mean reversion |
| Alpha012 robust | 1 day | Scale-free reversal confirmed by volume direction |
| Alpha101 | Intraday | Directional close location within the daily range |
| Alpha52 robust | 20 to 240 days | Momentum with low-price and volume confirmation |

Raw signals are winsorized daily and standardized cross-sectionally. Alpha012 uses
returns instead of absolute price changes so arbitrary price scaling does not change
ranks. Alpha101 treats zero-range observations as unavailable. Alpha52 is explicitly
named `alpha52_robust` because its low-price movement is normalized and its momentum
leg uses compounded price performance.

The prespecified comparison set contains:

- a two-signal equal-weight baseline;
- the baseline plus Alpha101;
- the baseline plus Alpha52 robust;
- a four-signal equal-weight model;
- a training-only optimized model; and
- a trailing-only rolling model.

The baseline selects one rebalance frequency on validation. The same frequency is
then applied to every candidate. All models use an identical four-alpha-valid
security-date sample so availability differences cannot create an artificial
advantage.

The committed empirical snapshot reports the baseline case study. Its vendor input
cannot be redistributed, so those numbers are preserved as an audit artifact rather
than presented as exactly reproducible from a fresh public download. Fresh runs
write to `results/reproduced/` and record their data and specification fingerprints.

## Portfolio simulation

At each target date, the simulator holds the highest and lowest signal quintiles
with long exposure `0.5`, short exposure `-0.5`, net exposure `0.0`, and gross
exposure `1.0`. Individual asset weights drift naturally between rebalances.

```text
turnover_t = sum(abs(target_weight_t - pre_trade_weight_t))
cost_t = turnover_t * cost_bps / 10,000
net_return_t = gross_return_t - cost_t
```

Turnover includes initial entry, scheduled rebalancing, and forced exits after loss
of eligibility. An active position with a missing forward return raises an error
instead of receiving a favorable zero return.

Portfolio selection uses only point-in-time membership and signal availability.
Forward-return completeness is tracked separately for IC estimation and inference,
so future quote availability never determines whether a security may be selected.

## Statistical inference

Mean daily IC uses Newey-West standard errors. Portfolio uncertainty uses a circular
moving-block bootstrap. Candidate and baseline return series share the same resampled
blocks, preserving paired market-period variation when estimating excess-return and
Sharpe-difference intervals.

The result should still be interpreted conservatively. Chronological separation
reduces researcher degrees of freedom, but it does not eliminate multiple testing or
selection uncertainty created by repeated experimentation.

## Reproduce the pipeline

Install the locked environment and run all checks:

```bash
uv sync --extra dev --frozen
make check
```

Run with processed market data and point-in-time membership:

```bash
uv run systematic-alpha run \
  --config configs/research.yaml \
  --data data/processed/market_data.csv \
  --membership data/membership/sp500_membership.csv \
  --output results/reproduced
```

The membership file must contain `Ticker`, `StartDate`, and `EndDate`. See
[`data/README.md`](data/README.md) for the full data contract.

A current-constituent demonstration is also available:

```bash
uv run systematic-alpha run \
  --config configs/research.yaml \
  --download-if-missing \
  --output results/reproduced
```

This fallback uses current S&P 500 constituents and therefore has survivorship bias.
It can exercise the full pipeline, but it cannot support an unbiased historical
investment claim.

## Tests

The test suite focuses on assumptions that can silently invalidate a backtest:

- next-open-to-open return alignment and sample-boundary purging;
- duplicate and missing quote handling;
- future-data and price-scale invariance;
- eligibility-aware transforms and common-sample construction;
- training-only and label-availability-aware weights;
- natural weight drift, turnover, and forced exits; and
- deterministic moving-block bootstrap output.

Continuous integration runs Ruff and pytest on every push and pull request.

## Known limitations

1. **Universe and delistings.** Current constituents introduce survivorship bias.
   Membership intervals alone do not provide complete delisting returns.
2. **Data provenance.** Public adjusted data may be revised. Exact reproduction
   requires a matching input fingerprint.
3. **Execution costs.** The linear model omits nonlinear market impact, time-varying
   spreads, opening-auction constraints, short borrow, and capacity.
4. **Risk exposures.** Dollar neutrality does not guarantee beta, sector, size,
   volatility, momentum, or liquidity neutrality.
5. **Selection uncertainty.** A chronological holdout does not remove researcher
   degrees of freedom across repeated iterations.
6. **Prospective status.** Stronger claims require genuinely new data.

See [`docs/methodology.md`](docs/methodology.md) for implementation details and
[`docs/research_decisions.md`](docs/research_decisions.md) for the rationale behind
the main research choices.

## Repository structure

```text
configs/                         Frozen research specification
data/README.md                   Input-data and membership contracts
docs/                            Methodology and research decisions
results/case_study/              Curated empirical case-study artifacts
src/systematic_alpha/            Research and backtest implementation
tests/                           Assumption-focused regression tests
```

## Disclaimer

This repository is an educational research project. It is not investment advice and
does not represent a production-ready trading strategy.

## License

MIT
