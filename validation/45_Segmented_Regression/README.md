# Validation Report: Segmented Regression

| ID | Description | Success | Result |
|---|---|---|---|
| V-45-01 | No Breakpoint (Linear Trend) | PASS | 0 |
| V-45-02 | Single Hinge Detection (True BP=50) | PASS | 1 |
| V-45-03 | Censored Hinge (Robustness, True BP=50) | FAIL | 2 |
| V-45-04 | Double Jump (True BPs=30, 70) | PASS | 2 |
| V-45-05 | Probability Calibration (True BP=50, Window=[45, 55]) | PASS | Probability: 100.00% |


## Detailed Results

### V-45-01: No Breakpoint (Linear Trend)
**Success:** True

#### Model Selection Summary
|   n_breakpoints |      bic |      aic |     mbic |     rss |     sar | converged   |
|----------------:|---------:|---------:|---------:|--------:|--------:|:------------|
|               0 | -149.888 | -155.098 | -149.888 | 20.3726 | 20.3726 | True        |
|               1 | -140.939 | -151.36  | -136.334 | 20.3194 | 20.3194 | True        |
|               2 | -140.287 | -155.918 | -131.076 | 18.6528 | 18.6528 | True        |

![Plot](v45_01_plot.png)

### V-45-02: Single Hinge Detection (True BP=50)
**Success:** True

#### Model Selection Summary
|   n_breakpoints |      bic |      aic |     mbic |      rss |      sar | converged   |
|----------------:|---------:|---------:|---------:|---------:|---------:|:------------|
|               0 |  166.271 |  161.061 |  166.271 | 480.959  | 480.959  | True        |
|               1 | -143.925 | -154.345 | -139.319 |  19.7217 |  19.7217 | True        |
|               2 | -139.243 | -154.874 | -130.033 |  18.8484 |  18.8484 | True        |

**Detected Breakpoint:** 51.18134361474205

#### Warnings
- Per-segment slope confidence intervals are conditional on the estimated breakpoints (post-selection inference) and do not account for breakpoint-location uncertainty. Nominal coverage may be lower under low SNR, small slope changes, edge breakpoints, outliers, or serial correlation.

![Plot](v45_02_plot.png)

### V-45-03: Censored Hinge (Robustness, True BP=50)
**Success:** False

#### Model Selection Summary
|   n_breakpoints |       bic |       aic |      mbic |      rss |      sar | converged   |
|----------------:|----------:|----------:|----------:|---------:|---------:|:------------|
|               0 |   85.8194 |   80.6091 |   85.8194 | 215.134  | 215.134  | True        |
|               1 |  -75.8772 |  -86.2979 |  -71.272  |  38.9466 |  38.9466 | True        |
|               2 | -160.453  | -176.084  | -151.243  |  15.2461 |  15.2461 | True        |

**Detected Breakpoint:** [np.float64(50.64148984138762), np.float64(73.7428846127307)]

#### Warnings
- High proportion of censored data (38.0% > 15%). Phase-1 OLS breakpoint detection uses substituted values (lt_mult=0.5, gt_mult=1.1), which may induce spurious breakpoints or over-splitting.
- Detected a contiguous left-censored floor of 26 observations (>= 15); flat censoring shelves can induce spurious breakpoints in Phase-1 OLS.
- Denominator near zero in Tau calculation
- Variance near zero, Z-score may be unreliable
- Per-segment slope confidence intervals are conditional on the estimated breakpoints (post-selection inference) and do not account for breakpoint-location uncertainty. Nominal coverage may be lower under low SNR, small slope changes, edge breakpoints, outliers, or serial correlation.

![Plot](v45_03_plot.png)

### V-45-04: Double Jump (True BPs=30, 70)
**Success:** True

#### Model Selection Summary
|   n_breakpoints |      bic |       aic |      mbic |      rss |      sar | converged   |
|----------------:|---------:|----------:|----------:|---------:|---------:|:------------|
|               0 |   80.109 |   74.8986 |   80.109  | 203.193  | 203.193  | True        |
|               1 |   59.554 |   49.1333 |   64.1592 | 150.883  | 150.883  | True        |
|               2 | -136.662 | -152.293  | -127.452  |  19.3412 |  19.3412 | True        |
|               3 | -131.021 | -151.862  | -117.205  |  18.6631 |  18.6631 | True        |

#### Warnings
- Per-segment slope confidence intervals are conditional on the estimated breakpoints (post-selection inference) and do not account for breakpoint-location uncertainty. Nominal coverage may be lower under low SNR, small slope changes, edge breakpoints, outliers, or serial correlation.

![Plot](v45_04_plot.png)

### V-45-05: Probability Calibration (True BP=50, Window=[45, 55])
**Success:** True

**Metric:** Probability: 100.00%

#### Warnings
- Per-segment slope confidence intervals are conditional on the estimated breakpoints (post-selection inference) and do not account for breakpoint-location uncertainty. Nominal coverage may be lower under low SNR, small slope changes, edge breakpoints, outliers, or serial correlation.

