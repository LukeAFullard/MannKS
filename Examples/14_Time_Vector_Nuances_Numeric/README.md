# Example 14: Time Vector Nuances (Numeric Data)

## Goal
Explain the critical difference between using simple **integer ranks** (0, 1, 2...) versus **fractional years** (2010.0, 2010.08...) as the time vector `t` in numeric trend analysis.

## Introduction
When performing a Mann-Kendall test with a numeric time vector (as opposed to `datetime` objects), the choice of the time vector `t` directly impacts the magnitude and units of the **Sen's Slope**, even though it does *not* affect the significance of the trend.

*   **Significance (P-value, S):** The Mann-Kendall test is **rank-based**. It only cares about the relative order of data points over time. Whether your time steps are `[1, 2, 3]` or `[2010.0, 2010.1, 2010.2]`, the order is the same, so the p-value will be identical.
*   **Magnitude (Sen's Slope):** The Sen's slope is the median of slopes calculated as `(y2 - y1) / (t2 - t1)`. The denominator `(t2 - t1)` depends entirely on the units of your time vector.
    *   If `t` is index-based (0, 1, 2...), the slope is **units per step**.
    *   If `t` is fractional years (2010.0, 2010.083...), the slope is **units per year**.

This example demonstrates this by running the test on the same dataset using two different time vectors.

## Step 1: The Data
We generate 10 years of monthly data with a known increasing trend of approximately **2.4 units per year** (which is 0.2 units per month).

## Step 2: Analysis Comparison

### Method A: Integer Index (0, 1, 2...)
We create a time vector `t` that is simply the index of the data point: 0, 1, 2, ... 119.
*   **Expected Slope:** Since the trend is 0.2 per month and each step is 1 month, we expect a slope close to **0.2**.
*   **Unit:** Units per month (or "per sample").

### Method B: Fractional Years (2010.0, 2010.083...)
We create a time vector `t` representing the decimal year: `Year + (Month - 1)/12`.
*   **Expected Slope:** Since the trend is 2.4 per year and `t` is in years, we expect a slope close to **2.4**.
*   **Unit:** Units per year.

## Python Code and Results

```python
import pandas as pd
import numpy as np
import MannKS as mk

# 1. Generate synthetic data (Monthly data for 10 years)
df = create_synthetic_data()

# 2. Define two different time vectors
# Vector A: Simple integer index (0, 1, 2, ...)
t_index = np.arange(len(df))

# Vector B: Fractional years (2010.0, 2010.083, ...)
t_fractional = df['Year'] + (df['Month'] - 1) / 12.0

# 3. Run the Trend Test with Vector A (Index)
result_index = mk.trend_test(
    x=df['Value'],
    t=t_index
)

# 4. Run the Trend Test with Vector B (Fractional Year)
result_fractional = mk.trend_test(
    x=df['Value'],
    t=t_fractional,
    plot_path='trend_plot_fractional.png' # Save plot for this one
)

print("--- Comparison of Results ---")
print(f"Significance (p-value) is IDENTICAL:")
print(f"  Method A (Index):      {result_index.p:.5f}")
print(f"  Method B (Fractional): {result_fractional.p:.5f}")
print()
print(f"Sen's Slope MAGNITUDE differs based on 't' units:")
print(f"  Method A (Index):      {result_index.slope:.4f} (approx 0.2 units/step)")
print(f"  Method B (Fractional): {result_fractional.slope:.4f} (approx 2.4 units/year)")
```

### Output

```text
Generated {len(df)} months of data (10 years).
True Trend: +2.4 units/year (+0.2 units/month)

--- Comparison of Results ---
Significance (p-value) is IDENTICAL:
  Method A (Index):      0.00000
  Method B (Fractional): 0.00000

Sen's Slope MAGNITUDE differs based on 't' units:
  Method A (Index):      0.1987 (approx 0.2 units/step)
  Method B (Fractional): 2.3841 (approx 2.4 units/year)

```

## Visualizing the Trend
The plot below shows the trend line calculated using the **Fractional Year** method. The slope of this line corresponds to the "per year" rate of change.

![Trend Analysis Plot](trend_plot_fractional.png)

## Interpretation
*   **Identical Significance:** Notice that the p-value is `0.00000` in both cases. The choice of time vector units **does not change the statistical conclusion** about whether a trend exists.
*   **Different Slopes:**
    *   Method A gives a slope of **0.1987**. This is the rate of change **per month** (since the time step was 1 month).
    *   Method B gives a slope of **2.3841**. This is the rate of change **per year**.
*   **Recommendation:** When using numeric time vectors (instead of datetime objects), always construct your vector `t` in the units you want your slope to be reported in. If you want "units per year", ensure `t` is in years (e.g., 2010.5). If you just want to know if there is a trend and don't care about the rate, a simple index `np.arange(n)` is sufficient.
