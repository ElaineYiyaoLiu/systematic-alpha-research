# Methodology and research controls

## Research objective

This project asks whether Alpha101 and a robust adaptation of Alpha52 add
independent predictive information to two existing cross-sectional price-volume
signals after executable timing, turnover and linear transaction costs. It is a
demonstration of a defensible research process, not a production trading strategy.

## Information timeline

Signals are computed using adjusted observations available after the close on day
`t`. A position is assumed to enter at the adjusted open on `t+1`; its one-day label
is the adjusted open-to-open return from `t+1` to `t+2`. This convention prevents a
signal using the close from being filled at that same close.

Dates are taken from the shared observed market calendar. Entry and exit prices are
joined on exact security-date keys. The implementation never substitutes the next
available row of an individual security for a missing target-date quote. The shared
calendar cannot detect a session missing for every security without an external
exchange calendar.

## Sample isolation

| Period | Purpose | Allowed decisions |
|---|---|---|
| 2015-01-01 to 2018-12-31 | Training | Estimate static factor weights |
| 2019-01-01 to 2021-12-31 | Validation | Select rebalance frequency |
| 2022-01-01 to 2025-07-01 | Historical holdout | Evaluate frozen choices only |

The last interval has already been inspected and is therefore not called untouched.
Specifications are frozen before evaluating that historical holdout. Observations
after 2025-07 are reserved for prospective work.

Rolling composite weights use at most 252 information-coefficient observations
whose labels have completed by the decision date. Each IC carries an explicit
`available_date` equal to its forward-return exit date. Tests verify that changing
an IC whose label is not yet complete cannot alter the current weight.

An observation belongs to a research sample only when its signal, entry and exit
dates all fall inside that sample. This label-containment rule purges boundary rows
that would otherwise allow training labels to use validation prices or validation
labels to use historical-evaluation prices.

## Signals

- **Reversal:** negative one-day adjusted close return.
- **Alpha012 robust:** the sign of the one-day adjusted-volume change multiplied by
  the negative adjusted-close return. Return scaling makes the signal invariant to
  stock price level and constant historical back-adjustments.
- **Alpha101:** the directional close-to-open move scaled by the daily high-low
  range; zero-range observations are unavailable.
- **Alpha52 robust:** the percentage change in the rolling five-day low multiplied
  by 12-1 momentum rank and five-day time-series volume rank. This scale-free,
  compounded-return implementation is disclosed as an adaptation of Alpha52.
- **Optimized composite:** long-only MVO-IR weights fitted only on training IC.
- **Rolling composite:** trailing-only long-only MVO-IR weights with a 60-observation
  warm-up.

Prespecified equal-weight ablations compare the original two signals, each new signal
added separately and all four signals together. Every model ranks the same ex-ante,
four-alpha-valid security-date rows. A separate label-complete sample is used for IC
and inference; missing future quotes never determine portfolio selection eligibility.

Raw signals are winsorized each day and standardized cross-sectionally. The
portfolio holds the top and bottom signal quantiles at 0.5 long and 0.5 short,
giving zero net and 1.0 gross exposure.

With point-in-time membership, only eligible securities enter cross-sectional
ranks, winsorization, standardization, IC estimation and minimum-universe counts.
Prices outside membership remain available solely to value and close a pre-existing
position. A missing membership state is never interpreted as an exit. An explicit
loss of eligibility may trigger a forced exit, while a missing security-date state
for an active holding raises an error.

## Costs and validation

Between target dates, asset weights drift with realized returns. Turnover is measured
from those drifted holdings to the next target, so weights are not maintained for
free. Initial entry is charged. Net return equals gross return minus turnover
multiplied by the configured cost in basis points. Transaction costs also reduce the
wealth denominator used to calculate end-of-period portfolio weights. The
equal-weight baseline selects
once among 1, 5, 10 and 21-day rebalancing at 5 bps. That single frequency is applied
to every model and frozen for the historical holdout. Circular moving-block bootstrap
intervals quantify performance uncertainty and paired improvement over the baseline.

## Failure-safe assumptions

- An active position with a missing forward return raises an error instead of being
  assigned a favorable zero return.
- Portfolio selection eligibility uses only signal-date information. Future-label
  completeness is tracked separately for IC and inference.
- Train, validation and historical-holdout periods must be chronological and disjoint.
- Duplicate security-date rows and invalid prices are rejected.
- Each run records date coverage, universe membership, missingness, the Git commit
  when available and SHA-256 fingerprints of the input and universe.
- Each run writes a sample-exclusion audit showing boundary or missing-label purges.
- Point-in-time membership controls selection eligibility without deleting prices
  required to close an existing holding. Loss of eligibility triggers a charged
  forced exit at the next simulated open.

## Interpretation

The published snapshot exhibits small, statistically detectable rank IC, but the
validation Sharpe does not persist in the historical holdout after costs. The correct
conclusion is weak predictive information without evidence of production-ready
alpha. Likely contributors include signal decay, regime instability, trading costs
and researcher degrees of freedom.

## Known limitations

The default fallback universe uses current S&P 500 constituents and therefore has
survivorship bias. It demonstrates the pipeline but cannot support an unbiased
historical investment claim without point-in-time membership and delisting returns.
The cost model omits nonlinear impact, variable spreads, short
borrow and capacity. Dollar neutrality does not imply beta, sector, size or
liquidity neutrality. Exact historical reproduction requires the original data
fingerprint.
