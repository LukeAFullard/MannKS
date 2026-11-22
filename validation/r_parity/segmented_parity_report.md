# External Validation & Calibration Report: Segmented Trend Analysis

- **R Reference**: `R version 4.3.3 (2024-02-29)`, CRAN `segmented` `2.0.3` (Muggeo 2003, 2008, 2020)

- **MannKS Version**: `0.7.2`

- **LWP Reference**: Frozen `LWP-TRENDS v2502` fixtures (`tests/data/lwp_parity_fixtures.json`)


## 1. Cross-Package Parity vs. R `segmented` — Published Datasets

| Dataset | Mode | R `k` | MannKS BIC `k` | MannKS ESS `k` | R Breakpoint(s) | MannKS Breakpoint(s) | Max |Δbp| (% span) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| `stagnant` | fixed (k=1) | 1 | 1 | — | 0.041 | 0.041 | 0.0002 (0.01%) |
| `stagnant` | select (Kmax=2) | 2 | 2 | 2 | -0.152, 0.313 | -0.152, 0.313 | 0.0000 (0.00%) |
| `globTempAnom` | fixed (k=3) | 3 | 3 | — | 1918.782, 1941.000, 1970.509 | 1919.539, 1941.000, 1970.508 | 0.7568 (0.44%) |
| `globTempAnom` | select (Kmax=3) | 3 | 3 | 2 | 1918.268, 1941.000, 1970.642 | 1919.539, 1941.000, 1970.508 | 1.2707 (0.73%) |
| `plant_RKV` | fixed (k=2) | 2 | 2 | — | 315.164, 445.273 | 299.877, 441.923 | 15.2866 (2.69%) |
| `plant_RKV` | select (Kmax=2) | 2 | 2 | 2 | 315.164, 445.273 | 299.877, 441.923 | 15.2866 (2.69%) |
| `plant_RKW` | fixed (k=1) | 1 | 1 | — | 469.101 | 469.101 | 0.0000 (0.00%) |
| `plant_RKW` | select (Kmax=2) | 1 | 1 | 1 | 469.101 | 469.101 | 0.0000 (0.00%) |
| `plant_RWC` | fixed (k=1) | 1 | 1 | — | 331.400 | 331.400 | 0.0000 (0.00%) |
| `plant_RWC` | select (Kmax=2) | 1 | 1 | 1 | 331.400 | 331.400 | 0.0003 (0.00%) |

> **Note on `globTempAnom` and `plant_RKV`**: Under unadjusted BIC (`autocorr_penalty='none'`, matching R `selgmented(type='bic')`), MannKS matches CRAN `segmented` `k=3` on `globTempAnom` (`RSS = 1.491`, identical to R) and achieves slightly lower RSS than R on `plant_RKV` (`0.01953` vs `0.01993`). Under MannKS's default `autocorr_penalty='auto'`, `globTempAnom`'s real positive lag-1 autocorrelation (`rho_d1 ≈ 0.615`) activates the deterministic ESS penalty and conservatively selects `k=2` consistently across all random seeds.


## 2. Planted-Truth Synthetic Benchmarks (30 Gaussian Series, n=100)

### 2a. Fixed-k Breakpoint & Slope Parity

| True `k` | N | Mean |bp_MannKS - bp_R| | MannKS bp MAE | R bp MAE | MannKS Sen Slope MAE | R OLS Slope MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 10 | 0.0000 | 0.4138 | 0.4138 | 0.0089 | 0.0084 |
| 2 | 10 | 0.0027 | 1.0037 | 1.0034 | 0.0172 | 0.0170 |

### 2b. Automatic `k` Selection Accuracy (Kmax=2)

| True `k` | N | R `selgmented(bic)` Acc | MannKS `bic` Acc | MannKS `mbic` Acc | MannKS-R BIC Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| 1 | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| **Overall** | **30** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

## 3. Null False Positive Rate (FPR ± MC SE): Gaussian vs. AR(1) φ=0.6 (50 Series Each, Kmax=2)

| Noise Process | R `selgmented(bic)` | R `selgmented(score)` | R `davies.test` (p<0.05) | MannKS `bic` | MannKS `mbic` | MannKS `bic + ESS` (default) | MannKS `mbic + ESS` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gaussian i.i.d. | 0.0% ± 0.0% | 0.0% ± 0.0% | 4.0% ± 2.8% | 2.0% ± 2.0% | 0.0% ± 0.0% | 2.0% ± 2.0% | 0.0% ± 0.0% |
| AR(1) φ=0.6 | 34.0% ± 6.7% | 0.0% ± 0.0% | 52.0% ± 7.1% | 66.0% ± 6.7% | 34.0% ± 6.7% | 6.0% ± 3.4% | 0.0% ± 0.0% |

> **Key takeaway**: Both R `selgmented(bic)` and unadjusted Python BIC suffer severe false-breakpoint inflation under AR(1) autocorrelation. MannKS's deterministic ESS heuristic (`autocorr_penalty='auto'`, default) and modified BIC (`criterion='mbic'`) substantially suppress spurious breakpoints under serial correlation, though some residual FPR can remain under very strong AR(1) on short series.

## 4. Per-Segment Exact Parity vs. R `LWP-TRENDS v2502` & `trend_test`

- **Direct R `LWP-TRENDS v2502` wrapper parity**: Verified all 10 metrics directly against `7` non-seasonal R `LWP-TRENDS v2502` wrapper fixtures (`n_breakpoints=0`, LWP mode).

- **Kernel & multi-segment parity**: Verified `S`, `VarS`, and `sen` directly against `24` R `GetKendal`/`GetInterObservationSlopes` kernel fixtures (`< 1e-9` rel error), and verified all 10 metrics against MannKS `trend_test` (LWP mode) across all `24` kernel fixtures and a multi-segment (`k=1`) censored split: **PASS**.

| Metric | Max Rel. Error vs. R `LWP-TRENDS v2502` (7 Wrapper Cases) | Max Rel. Error vs. MannKS `trend_test` (24 Kernel Cases) |
| :--- | :---: | :---: |
| `s` | `0.000e+00` | `0.000e+00` |
| `var_s` | `2.149e-15` | `0.000e+00` |
| `z` | `3.212e-15` | `0.000e+00` |
| `p` | `1.623e-13` | `0.000e+00` |
| `Tau` | `1.220e-15` | `0.000e+00` |
| `C` | `4.792e-16` | `0.000e+00` |
| `Cd` | `2.013e-13` | `0.000e+00` |
| `slope` | `3.384e-15` | `0.000e+00` |
| `lower_ci` | `2.373e-15` | `0.000e+00` |
| `upper_ci` | `3.191e-15` | `0.000e+00` |

## 5. Post-Fix Known-Truth Monte Carlo Calibration

- **Scale invariance (27 decades in `t` × 18 decades in `y`)**: All `k=1` = `True`, max relative breakpoint drift = `9.718e-10`.

- **Degenerate null FPR (10 reps each)**: constant = `0/10`, noiseless linear = `0/10`, all-censored = `0/10`.


### 5a. Robustness Across Error Distributions (True `k=1`, 40 Reps Each)

| Error Distribution | `k=1` Accuracy (± MC SE) | Breakpoint RMSE | MannKS Sen Slope MAE | OLS Slope MAE |
| :--- | :---: | :---: | :---: | :---: |
| Gaussian | 95.0% ± 3.4% | 1.607 | 0.0179 | 0.0176 |
| Student-t3 | 97.5% ± 2.5% | 3.164 | 0.0343 | 0.0404 |
| 5% Outliers | 75.0% ± 6.8% | 5.379 | 0.0231 | 0.0535 |

### 5b. 95% Confidence Interval Coverage Calibration (60 Reps, True `k=1`)

| Interval Type | Empirical Coverage (± MC SE) | Nominal |
| :--- | :---: | :---: |
| Breakpoint CI — OLS Delta-Method (fixed `k=1`) | 96.7% ± 2.3% | 95% |
| Breakpoint CI — Block-Bagging Percentile (fixed `k=1`, `n_bootstrap=40`) | 90.0% ± 3.9% | 95% |
| Per-Segment Sen Slope CI — Conditional on fixed `k=1` (estimated bp) | 95.0% ± 2.0% | 95% |
| Per-Segment Sen Slope CI — Post-Selection (auto-selected `k=1` + estimated bp) | 95.8% ± 1.9% | 95% |
| Per-Segment Sen Slope CI — Oracle (known true bp) | 95.0% ± 2.0% | 95% |

> **Note**: Per-segment Sen slope CIs are conditional on the estimated breakpoints; `segmented_trend_test` emits an explicit post-selection inference warning whenever `n_breakpoints > 0`. Coverage under lower SNR or serial correlation will be below nominal.

