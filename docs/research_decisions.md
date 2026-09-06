# Research decisions

This document records choices that materially affect the credibility or
interpretation of the research.

## Use executable forward returns

Signals observed after the close on day `t` cannot be executed at that same
close without a stronger assumption. The pipeline enters at the adjusted open
on `t+1` and exits at the adjusted open on `t+2`. Exact calendar joins prevent a
missing quote from silently extending the intended holding period.

## Purge labels at period boundaries

A signal belongs to a research period only when its signal, entry and exit dates
remain inside that period. This prevents labels from carrying information across
training, validation and historical-evaluation boundaries.

## Rescale Alpha012

An absolute price-difference formula is sensitive to price level and constant
back-adjustments. The robust implementation uses an adjusted close return,
preserving reversal direction while making ranks invariant to price scaling.

## Treat Alpha52 as an adaptation

The implementation normalizes low-price movement and uses compounded 12-to-1
momentum. It is named `alpha52_robust` rather than presented as an exact
reproduction of the published formula.

## Use one common sample and rebalance frequency

Every candidate ranks the same ex-ante sample: point-in-time eligible observations
for which all four signals are available on the signal date. Future-return
availability never controls selection eligibility. IC and inference use a separate
label-complete evaluation sample. The baseline selects one validation-period
rebalance frequency, which is applied to every model. These restrictions prevent
sample availability or model-specific frequency searches from creating an
artificial advantage without leaking future quote availability into selection.

## Restrict factor-weight information

Static weights use training IC only. Rolling weights may use an IC only after
its forward-return label has completed. Signal-date ordering alone would make an
unrealized label available too soon.

## Simulate stateful positions

Weights drift with returns between target dates. Turnover is measured from
pre-trade weights, not prior targets. Loss of eligibility prevents new selection
but does not erase prices needed to value and close an existing holding. Missing
eligibility is treated as a data error rather than evidence of an exit. Transaction
costs reduce portfolio wealth before the next set of drifted weights is calculated.

## Use dependence-aware inference

Newey-West standard errors address dependence in daily IC. Portfolio uncertainty
uses a circular moving-block bootstrap. Candidate and baseline series share
block indices so paired differences retain common market variation.

## Separate archived and current evidence

The committed snapshot comes from the two-signal study. The current code adds a
four-signal experiment, but legacy numbers are not relabeled as evidence for it.
New runs write separately and record data and specification fingerprints.

## Avoid unsupported investment claims

The public-data fallback uses current index constituents and lacks institutional
point-in-time provenance, complete delisting returns, nonlinear impact and
borrow constraints. Its output demonstrates the pipeline rather than investable
historical performance.
