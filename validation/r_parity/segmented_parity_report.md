# External Validation & Calibration Report: Segmented Trend Analysis

- **R Reference**: `R version 4.3.3 (2024-02-29)`, CRAN `segmented` `2.0.3` (Muggeo 2003, 2008, 2020)

- **MannKS Version**: `0.7.1`

- **LWP Reference**: Frozen `LWP-TRENDS v2502` fixtures (`tests/data/lwp_parity_fixtures.json`)


## 1. Cross-Package Parity vs. R `segmented` — Published Datasets

| Dataset | Mode | R `k` | MannKS BIC `k` | MannKS ESS `k` | R Breakpoint(s) | MannKS Breakpoint(s) | Max |Δbp| (% span) |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- | :---: |
| `stagnant` | fixed (k=1) | 1 | 1 | — | 0.041 | 0.041 | 0.0002 (0.01%) |
| `stagnant` | select (Kmax=2) | 2 | 2 | 2 | -0.152, 0.313 | -0.152, 0.313 | 0.0000 (0.00%) |
| `globTempAnom` | fixed (k=3) | 3 | 3 | — | 1918.782, 1941.000, 1970.509 | 1911.782, 1959.567, 1965.347 | 18.5674 (10.73%) |
| `globTempAnom` | select (Kmax=3) | 3 | 2 | 2 | 1918.268, 1941.000, 1970.642 | 1908.572, 1983.276 | N/A (k differs) |
| `plant_RKV` | fixed (k=2) | 2 | 2 | — | 315.164, 445.273 | 299.877, 441.923 | 15.2866 (2.69%) |
| `plant_RKV` | select (Kmax=2) | 2 | 2 | 2 | 315.164, 445.273 | 299.877, 441.923 | 15.2866 (2.69%) |
| `plant_RKW` | fixed (k=1) | 1 | 1 | — | 469.101 | 469.101 | 0.0000 (0.00%) |
| `plant_RKW` | select (Kmax=2) | 1 | 1 | 1 | 469.101 | 469.101 | 0.0000 (0.00%) |
| `plant_RWC` | fixed (k=1) | 1 | 1 | — | 331.400 | 334.522 | 3.1223 (0.55%) |
| `plant_RWC` | select (Kmax=2) | 1 | 1 | 1 | 331.400 | 334.522 | 3.1225 (0.55%) |

## 2. Planted-Truth Synthetic Benchmarks (30 Gaussian Series, n=100)

### 2a. Fixed-k Breakpoint & Slope Parity

| True `k` | N | Mean |bp_MannKS - bp_R| | MannKS bp MAE | R bp MAE | MannKS Sen Slope MAE | R OLS Slope MAE |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 10 | 0.0000 | 0.4138 | 0.4138 | 0.0089 | 0.0084 |
| 2 | 10 | 0.0454 | 1.0273 | 1.0034 | 0.0174 | 0.0170 |

### 2b. Automatic `k` Selection Accuracy (Kmax=2)

| True `k` | N | R `selgmented(bic)` Acc | MannKS `bic` Acc | MannKS `mbic` Acc | MannKS-R BIC Agreement |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| 1 | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | 10 | 100.0% | 100.0% | 100.0% | 100.0% |
| **Overall** | **30** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

## 3. Null False Positive Rate (FPR): Gaussian vs. AR(1) φ=0.6 (30 Series Each, Kmax=2)

| Noise Process | R `selgmented(bic)` | R `selgmented(score)` | R `davies.test` (p<0.05) | MannKS `bic` | MannKS `mbic` | MannKS `bic + ESS` | MannKS `mbic + ESS` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Gaussian i.i.d. | 0.0% | 0.0% | 3.3% | 3.3% | 0.0% | 3.3% | 0.0% |
| AR(1) φ=0.6 | 36.7% | 0.0% | 63.3% | 60.0% | 33.3% | 10.0% | 0.0% |

> **Key takeaway**: Both R `selgmented(bic)` and standard Python BIC suffer severe false-breakpoint inflation under AR(1) autocorrelation. MannKS's effective-sample-size penalty (`autocorr_penalty='auto'`) and modified BIC (`criterion='mbic'`) substantially suppress spurious breakpoints under serial correlation.

## 4. Per-Segment Exact Parity vs. R `LWP-TRENDS v2502`

- Verified `24` LWP-TRENDS kernel fixtures (`n_breakpoints=0`, LWP mode) and multi-segment (`k=1`) concatenation.

- Multi-segment exact parity with `trend_test(..., mk_test_method='lwp', ci_method='lwp', sens_slope_method='lwp')`: **PASS**

| Metric | Max Relative Error vs. LWP-TRENDS |
| :--- | :---: |
| `s` | `0.000e+00` |
| `var_s` | `0.000e+00` |
| `z` | `0.000e+00` |
| `p` | `0.000e+00` |
| `Tau` | `0.000e+00` |
| `C` | `0.000e+00` |
| `Cd` | `0.000e+00` |
| `slope` | `0.000e+00` |
| `lower_ci` | `0.000e+00` |
| `upper_ci` | `0.000e+00` |

## 5. Post-Fix Known-Truth Monte Carlo Calibration

- **Scale invariance (27 decades in `t` × 18 decades in `y`)**: All `k=1` = `True`, max relative breakpoint drift = `2.132e-16`.

- **Degenerate null FPR (10 reps each)**: constant = `0/10`, noiseless linear = `0/10`, all-censored = `0/10`.


### 5a. Robustness Across Error Distributions (True `k=1`, 25 Reps Each)

| Error Distribution | `k=1` Accuracy | Breakpoint RMSE | MannKS Sen Slope MAE | OLS Slope MAE |
| :--- | :---: | :---: | :---: | :---: |
| Gaussian | 96.0% | 1.331 | 0.0186 | 0.0171 |
| Student-t3 | 84.0% | 2.322 | 0.0314 | 0.0328 |
| 5% Outliers | 60.0% | 3.683 | 0.0136 | 0.0405 |

### 5b. 95% Confidence Interval Coverage Calibration (40 Reps, True `k=1`)

| Interval Type | Empirical Coverage | Nominal |
| :--- | :---: | :---: |
| Breakpoint CI — OLS Delta-Method | 95.0% | 95% |
| Breakpoint CI — Block-Bagging Percentile (`n_bootstrap=40`) | 92.5% | 95% |
| Per-Segment Sen Slope CI — Post-Selection (estimated bp) | 95.0% | 95% |
| Per-Segment Sen Slope CI — Oracle (known true bp) | 93.8% | 95% |

> **Note**: Per-segment Sen slope CIs are conditional on the estimated breakpoints; `segmented_trend_test` emits an explicit post-selection inference warning whenever `n_breakpoints > 0`.

