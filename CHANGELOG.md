# Changelog

## 0.6.2

- Reject missing eligibility states for active holdings instead of treating a
  missing security-date row as a free forced exit.
- Include transaction costs in the wealth used for portfolio-weight drift and
  rerun each cost-sensitivity scenario with internally consistent weights.
- Use order-independent average ranks when signals contain ties.
- Record point-in-time universe metadata whenever a membership file is supplied.
- Clarify the survivorship-biased, archived nature of the committed case study and
  replace the legacy `Untouched Test` label with `Historical evaluation`.
- Add regression tests for missing eligibility, cost-aware drift and tied signals.

## 0.6.1

- Separate ex-ante portfolio-selection eligibility from label-complete statistical
  evaluation, preventing future quote availability from changing the ranked universe.
- Add regression tests proving that missing forward labels cannot remove securities
  from the selection sample.

## 0.6.0

- Exclude point-in-time ineligible securities from every cross-sectional transform,
  IC calculation, common sample and minimum-universe count while retaining prices
  for forced exits.
- Replace absolute-price Alpha012 with a disclosed scale-free return adaptation.
- Select one validation rebalance frequency on `baseline_equal` and apply it to all
  models instead of optimizing a separate frequency for each candidate.
- Add circular moving-block bootstrap intervals and paired incremental comparisons.
- Record a normalized research-specification fingerprint and optional membership
  fingerprint in run metadata.
- Add regression tests for eligibility contamination, cross-section counts, signal
  scale invariance, unified frequency selection and bootstrap determinism.

## 0.5.0

- Attach an explicit availability date to each daily IC and prevent rolling factor
  weights from using incomplete forward-return labels.
- Preserve complete market prices when applying point-in-time membership.
- Restrict new portfolio selections to eligible securities.
- Force held securities out when they lose eligibility and report the resulting
  turnover separately.
- Add regression tests for IC publication latency, eligibility and forced exits.

## 0.4.0

- Align forward-return labels through exact security-date joins on a shared observed
  market calendar.
- Preserve missing labels when a security lacks the required entry or exit quote.
- Purge labels whose entry or exit crosses a train, validation or historical
  evaluation boundary.
- Add a generated sample-exclusion audit and regression tests for both timing bugs.

## 0.3.0

- Enforce a common security-date sample throughout factor fitting and evaluation.
- Preserve the original direction of the Alpha52 rolling-low component while
  keeping its disclosed scale-free adaptation.
- Replace clipped unconstrained factor weights with SLSQP long-only simplex
  optimization.
- Simulate natural asset-weight drift between rebalance dates and measure trades
  from drifted holdings to each new target.
- Add regression tests for common-sample filtering, weight drift, drift-aware
  turnover, optimizer constraints and the Alpha52 direction.
- Correct the documented `prepare-data --raw` command.

Archived version 0.1 figures remain unchanged as an audit trail. Reproduced 0.3
results are expected to differ because the accounting and sample definitions are
more conservative.
