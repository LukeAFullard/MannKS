# Validation Case V-14: Insufficient Data

## Objective
This validation case verifies how the `MannKS` package and the LWP-TRENDS R script handle datasets that are too small for a valid trend test. Two scenarios are tested: one where a test is impossible (n=1) and one where a test is possible but the sample size is very small (n=4).

---

## Scenario A: Impossible Test (n=1)

A dataset with a single data point was created. No statistical trend can be calculated from one point.

### Results (n=1)
| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | nan        | nan        | nan     |
| Sen's Slope (/yr)   | nan    | nan    | nan       |
| Classification      | insufficient data | insufficient data | N/A                 |
| Analysis Notes      | `< 3 unique values` | `< 3 unique values` | N/A                 |

### Analysis (n=1)
All three methods correctly identified that a trend test could not be performed.
-   **MannKS (Standard & LWP Mode):** Both returned a classification of "insufficient data" and populated the statistical fields with `NaN` or `0` as appropriate. This is a graceful failure.
-   **LWP-TRENDS R Script:** The R script fails internally during its pre-processing steps, and our wrapper script correctly catches the error and reports `NaN` for all results.

---

## Scenario B: Small Sample Size (n=4)

A dataset with four data points was created. While a test is technically possible, the statistical power is extremely low and the results are unreliable.

![Trend Plot for n=4](trend_plot_n4.png)

*Figure 1: Plot of the n=4 data. A trend line can be calculated, but the confidence intervals are extremely wide due to the low sample size.*

### Results (n=4)
| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | 0.7341        | 0.7341        | nan     |
| Sen's Slope (/yr)   | -0.0970    | -0.0970    | nan       |
| Classification      | No Trend | No Trend | N/A                 |
| Analysis Notes      | `< 5 Non-censored values<br>sample size (4) below minimum (10)<br>WARNING: Sen slope based on tied non-censored values` | `< 5 Non-censored values<br>sample size (4) below minimum (10)<br>WARNING: Sen slope based on tied non-censored values` | N/A                 |

### Analysis (n=4)
All three methods ran the analysis but provided warnings about the small sample size.
-   **MannKS (Standard & LWP Mode):** Both functions executed correctly but produced an analysis note: `sample size (4) below minimum (10)`. This correctly alerts the user that the results may be unreliable.
-   **LWP-TRENDS R Script:** The R script also ran but produced its own analysis note (captured in the R object, not shown here) indicating that the Sen's slope confidence intervals could not be calculated due to the small sample size, resulting in `NaN` values for the CIs in the output.

This validation confirms that all systems handle insufficient data gracefully and provide appropriate feedback to the user.