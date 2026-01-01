# Validation Case V-05: Left-Censored Increasing Trend

## Objective
This validation case verifies the handling of left-censored (`<`) data. The goal is to ensure all methods can detect a positive trend in a dataset where some lower values are censored.

## Data
A synthetic dataset of 50 annual samples was generated with a known positive slope. Values below `6.0` were converted to left-censored strings (e.g., `'<6.0'`). The generated plot from the standard `MannKS` analysis is shown below.

![Left-Censored Plot](left_censored_plot.png)

```python
import pandas as pd
import numpy as np
import MannKS as mk

# Generate Data
np.random.seed(42)
n = 50
t = pd.to_datetime(pd.date_range(start='2000-01-01', periods=n, freq='YE'))
slope = 0.1
intercept = 5
noise = np.random.normal(0, 1, n)
x = slope * np.arange(n) + intercept + noise

# Introduce left-censoring
censor_threshold = 6.0
x_censored = [f"<6.0" if val < censor_threshold else val for val in x]

# Pre-process and run MannKS
processed_data = mk.prepare_censored_data(x_censored)
mk_results = mk.trend_test(processed_data, t)
print("p-value:", mk_results.p)
```

## Results Comparison

| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | 0.000000   | 0.000000        | 0.000000     |
| Sen's Slope         | 0.144178 | 0.123828    | 0.123828       |
| Lower CI (90%)      | 0.107612 | 0.087249 | 0.087489    |
| Upper CI (90%)      | 0.171224 | 0.158159 | 0.157401    |

## Analysis
The `sens_slope_method='lwp'` parameter is key in this test. It instructs `MannKS` to set ambiguous pairwise slopes involving censored data to 0, mimicking the R script's behavior. This results in the **MannKS (LWP Mode)** slope and p-value being very close to the **LWP-TRENDS R Script**.

The **MannKS (Standard)** run uses a more robust default (`sens_slope_method='nan'`), which removes ambiguous slopes from the calculation. This can lead to a slightly different, but statistically sound, result. All methods correctly identified the significant increasing trend.