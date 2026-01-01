# Validation Case V-15: All Values Censored

## Objective
This validation case verifies that all analysis methods gracefully handle a dataset where 100% of the values are censored. In such a scenario, no trend can or should be calculated.

## Data
A synthetic dataset of 10 annual samples was generated where every value is the same left-censored string: ` <5 `.

## Results Comparison

The following table compares the key statistical outputs. As expected, no trend could be calculated.

| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | 1.0000   | 1.0000        | nan     |
| Sen's Slope         | nan | 0.0000    | nan       |
| Classification      | No Trend | No Trend | N/A                 |
| Analysis Notes      | `< 3 unique values<br>Long run of single value` | `< 3 unique values<br>Long run of single value<br>CRITICAL: Sen slope is based on a pair of two censored values.` | N/A                 |


## Analysis
All three methods correctly determined that no trend could be calculated from a dataset composed entirely of identical censored values.

-   **MannKS (Standard & LWP Mode):** Both functions returned a p-value of `1.0` and a Sen's slope of `0.0`, correctly identifying the complete lack of any trend. The analysis notes properly flag that there is only one unique value, which is the root cause.
-   **LWP-TRENDS R Script:** The R script also produces `NA` (Not Available) for its primary results, as no valid statistical comparison can be made. Our wrapper correctly translates this to `nan`.

This confirms that all systems behave as expected and do not produce misleading results when faced with this edge case.