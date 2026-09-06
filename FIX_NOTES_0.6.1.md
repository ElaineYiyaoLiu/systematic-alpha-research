# Look-ahead selection fix (0.6.1)

## Defect

Before 0.6.1, `build_common_sample()` required `forward_return_1d` to be
non-missing. The returned keys became `selection_eligible`, so future entry or
exit quote availability could remove a security from the signal-date ranked
universe.

## Implementation

- `build_common_sample()` is now an ex-ante selection sample. It requires only
  the configured signals and point-in-time universe eligibility.
- `build_evaluation_sample()` separately requires a completed forward-return
  label. It is used for IC estimation, factor fitting and inference, never for
  portfolio selection.
- `run_pipeline()` records both `selection_eligible` and
  `evaluation_eligible` on the full price panel.
- The simulator still raises on a missing return for an active position. This
  deliberately prevents an unknown holding return from being silently treated
  as zero.
- Run metadata records both sample policies.

## Regression coverage

`tests/test_common_sample.py` now proves that:

1. Missing future labels do not alter the selection universe.
2. Label filtering affects only the statistical evaluation sample.

Validation performed for this release:

```text
pytest: 42 passed
ruff: all checks passed
```

## Remaining research limitation

This fix removes the future-label selection leak. It does not invent a return
for a held security whose required future quote is unavailable. A production
study should additionally define and source explicit suspension, delisting and
unfilled-order policies before claiming implementable performance.
