# Validation Case V-20: Numeric Time Vector

## Objective
This validation case verifies that the `MannKS` package functions correctly with a simple numeric time vector (e.g., fractional years) instead of datetime objects, and it highlights the comparative inflexibility of the LWP-TRENDS R script.

## Data
A synthetic dataset of 40 samples was generated with a clear positive trend. The time vector `t` was created as a NumPy array of floating-point numbers representing unequally spaced fractional years.

```python
import numpy as np
import MannKS as mk

# Generate Data
np.random.seed(50)
n = 40
t = 2000.0 + np.cumsum(np.random.uniform(0.1, 1.5, n))
slope = 0.5
intercept = 10
noise = np.random.normal(0, 1, n)
x = slope * (t - t[0]) + intercept + noise

# Run MannKS test
result = mk.trend_test(x, t, alpha=0.1)

print(f"Slope: {result.slope:.4f}")
print(f"P-value: {result.p:.4f}")
```

## Methodological Difference

-   **`MannKS`**: The `trend_test` function is designed to be flexible. It natively accepts numeric arrays for the time vector `t` and correctly calculates the Sen's slope in "units of x per unit of t".

-   **LWP-TRENDS R Script**: The R script is rigid and not designed for numeric time vectors. Its internal functions require a datetime column named `myDate` to generate a `Year` column, which is essential for its workflow. To make the script run, the numeric years had to be manually coerced into a date format (`as.Date(paste0(floor(mydata$time), "-01-01"))`). This workaround forces the script to treat all fractional years as the start of that year, fundamentally altering the time data and leading to different results.

## Results Comparison

The results below show that `MannKS` runs successfully on the raw numeric data. The R script runs only after the time data is modified, leading to a different (and less accurate) result.

| Metric              | `MannKS` (Standard) | LWP-TRENDS R Script (Modified Data) |
|---------------------|-------------------------|-------------------------------------|
| p-value             | 0.000000     | 0.000000                       |
| Sen's Slope         | 0.500990   | 0.492155                         |
| Lower CI (90%)      | 0.480735 | 0.470628                      |
| Upper CI (90%)      | 0.519973 | 0.518652                      |

## Conclusion
This validation case successfully demonstrates the superior flexibility of the `MannKS` package.

- **`MannKS` correctly and easily handles numeric time vectors**, providing an accurate trend analysis on the original, unmodified data.
- The **LWP-TRENDS R script is not compatible with numeric time vectors** and requires significant data modification to run, which compromises the accuracy of the results. This highlights a key advantage of `MannKS` for users with data in this common format.
