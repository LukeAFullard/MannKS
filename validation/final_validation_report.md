# Validation Report

This report summarises the validation of `MannKS` (v0.6.0) against the reference
LWP-TRENDS R script (`LWPTrends_v2502.r`, run live through `rpy2`) and the NADA2
R scripts (`ATS.R`, `censeaken.R`), together with simulation studies of the
bootstrap and segmented-regression features.

## How to reproduce

The scenario scripts (`validation/NN_*/run_validation.py`) expect the R sources in
`Example_Files/R/LWPTrends_v2502` and `Example_Files/R/NADA2`. They also need `rpy2`
and the R packages listed in `validation/r_parity/README.md`, plus `Icens` for NADA2.
Each LWP comparison is written to `master_results.csv`, one row per test.

More detailed parity checks, with an explicit 1e-6 tolerance on every statistic
(S, Var(S), Z, p, C, Cd, tau, slope, CI, `sen_probability`), are in
[`r_parity/`](r_parity/README.md).

## LWP-TRENDS parity (`master_results.csv`)

"LWP mode" in these scripts uses:

```python
mk_test_method='lwp', ci_method='lwp', sens_slope_method='lwp', agg_method='lwp',
tau_method='a', tie_break_method='lwp', alpha=0.10
```

The R side calls `GetMoreDateInfo(..., FindDateShifts=FALSE)`, because MannKS does
not implement LWP-TRENDS' date shifting (moving end-of-month samples into an empty
following month).

| Outcome | Rows | Cases |
|---|---|---|
| Slope, p, lower and upper CI agree with R to ≤ 1e-6 (relative) | 48 | V-01 – V-05, V-07 – V-11, V-13 – V-17, V-27 |
| Small differences due to the test harness | 3 | V-06 (numeric time: R receives the decimal years converted to whole-day dates and aggregates them by month; MannKS analyses the decimal years directly) |
| R returns `NA` because a data-sufficiency check fails; MannKS computes and reports the corresponding analysis note | 10 | V-03_flat, V-18, V-19 (a season with < 3 values), V-28 (< 5 non-censored values), V-29 (all censored), V-30 (long run) |
| R errors on the input (`if (kenS > 0)` in `GetKendal`) | 3 | V-12 |

Before the v0.6.0 fixes, none of these rows agreed with R to this tolerance. The
causes were:
- Var(S) omitted tie and censoring corrections.
- The CI level differed (MannKS used 95%; LWP-TRENDS reports 90%).
- Mid-observation selection differed from R.

## Known, documented differences from LWP-TRENDS

1. **Data-sufficiency gating.** LWP-TRENDS does not compute a trend when an analysis
   note is raised. MannKS raises the same notes but still computes.
2. **Date shifting** (`FindDateShifts`) is not implemented.
3. **Equidistant same-day samples.** When two observations are equally close to an
   increment's mid-point (e.g. same-day duplicates), MannKS keeps the first. R's
   choice depends on the row order produced by `merge()`.
4. **Default estimators.** The defaults are not the LWP-TRENDS methods:
   - the slope for censored data defaults to Akritas-Theil-Sen;
   - the CI defaults to nearest rank with α = 0.05;
   - `mk_test_method` defaults to `'robust'`.

   The LWP-mode settings above reproduce LWP-TRENDS.

## NADA2 / Akritas-Theil-Sen

`sens_slope_method='ats'` returns the exact midpoint of the interval where the
Kendall score of the residuals is zero. NADA's bisection approximates the edges of
that interval, so slopes agree to within NADA's tolerance (median relative
difference about 0.3% against `NADA::cenken`; see
`r_parity/censored_slope_study.py`). In simulations with a known slope, ATS was the
least biased censored estimator:

| Censoring | ATS | LWP (ambiguous pairs = 0) | drop ambiguous pairs |
|---|---|---|---|
| 20% | +2% | +3% | +17% |
| 50% | +4% | −55% | +69% |

## Bootstrap and segmented-regression studies (V-36 – V-52)

These are simulation studies with small numbers of replicates (often 50), so their
rejection rates and coverages carry Monte Carlo errors of several percentage points.
Larger simulations (1000–2000 replicates) are in `r_parity/monte_carlo.py`. There:

- Mann-Kendall Type I error is close to 0.05, including with heavy ties and with censoring.
- Sen's slope CI coverage is 0.95–0.96.
- The block bootstrap reduces the rejection rate for AR(1) noise with ρ = 0.6 from
  0.30 to 0.15. It is slightly liberal for iid data at n = 60 (about 0.07–0.09).
- The Yue-Wang correction holds its nominal size for iid data (0.05) and keeps full
  power, but under-corrects strong autocorrelation in short series (0.18 at ρ = 0.6,
  n = 60).

Segment slopes and their confidence intervals are conditional on the estimated
breakpoints.

## Conclusion

With the LWP-mode settings, MannKS reproduces LWP-TRENDS exactly wherever
LWP-TRENDS produces a result, apart from the documented differences above. The
default settings use estimators that are better calibrated for censored data.
