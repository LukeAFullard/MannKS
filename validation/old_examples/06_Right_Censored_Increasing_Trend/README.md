# Validation Case V-06: Right-Censored Increasing Trend

## Objective
This validation case verifies the handling of right-censored (`>`) data, which is a key point of difference between the robust and LWP-emulation methods.

## Data
A synthetic dataset of 50 annual samples was generated with a positive slope. Values above `8.0` were converted to right-censored strings (e.g., `'>8.0'`). The generated plot from the standard `MannKS` analysis is shown below.

![Right-Censored Plot](right_censored_plot.png)

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
noise = np.random.normal(0, 1.5, n)
x = slope * np.arange(n) + intercept + noise

# Introduce right-censoring
censor_threshold = 8.0
x_censored = [f">8.0" if val > censor_threshold else val for val in x]

# Pre-process and run MannKS
processed_data = mk.prepare_censored_data(x_censored)
mk_results = mk.trend_test(processed_data, t)
print("p-value:", mk_results.p)
```

## Results Comparison

| Metric              | MannKS (Standard) | MannKS (LWP Mode) | LWP-TRENDS R Script |
|---------------------|-----------------------|-----------------------|---------------------|
| p-value             | 0.181986   | 0.000002        | 0.000002     |
| Sen's Slope         | 0.076794 | 0.064708    | 0.064708       |
| Lower CI (90%)      | 0.055880 | 0.045101 | 0.045248    |
| Upper CI (90%)      | 0.096403 | 0.087545 | 0.087060    |

## Analysis
This case highlights the difference between the `mk_test_method='robust'` (Standard) and `mk_test_method='lwp'` (LWP Mode) settings.

The **LWP-TRENDS R Script** uses a heuristic that replaces all right-censored values with a single numeric value slightly larger than the maximum detection limit. The **MannKS (LWP Mode)** replicates this, resulting in nearly identical p-values and slopes.

The **MannKS (Standard)** method, however, uses a more statistically robust ranking approach that does not modify the data. It treats comparisons between uncensored values and right-censored values as ambiguous, which is a more conservative approach. This results in a higher p-value, correctly reflecting the increased uncertainty introduced by the right-censored data.