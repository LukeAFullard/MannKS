import os
import io
import contextlib
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt

# --- 1. Define the Example Code as a String ---
example_code = """
import os
import numpy as np
import pandas as pd
from MannKS.segmented_trend_test import segmented_trend_test
from MannKS.plotting import plot_segmented_trend

# 1. Generate Synthetic Data with Two Breakpoints
# Scenario: Complex trend with three distinct regimes.
# Slopes are closer together to make breakpoint detection less certain.
np.random.seed(101)
n = 150
t = np.arange(n)

# Define True Trend
# Segment 1 (0-50):   Slope 0.4
# Segment 2 (50-100): Slope 0.1 (Flatter)
# Segment 3 (100-150): Slope 0.5 (Steeper)
trend = np.concatenate([
    0.4 * t[:50],
    0.4 * 50 + 0.1 * (t[50:100] - 50),
    0.4 * 50 + 0.1 * 50 + 0.5 * (t[100:] - 100)
])

# Add Moderate Noise
# Noise level adjusted to maintain ambiguity in breakpoints
noise_std = 2.0
x = trend + np.random.normal(0, noise_std, n)

# 2. Run Analysis with Varying Alpha Levels
# varying alpha affects the width of the Confidence Intervals (CI).
# Lower alpha -> Higher Confidence -> Wider Intervals.
alphas = [0.10, 0.05, 0.01]

methods = [
    {'use_bagging': True,  'name': 'Bagging',      'suffix': 'bagging'},
    {'use_bagging': False, 'name': 'Standard OLS', 'suffix': 'ols'}
]

for alpha in alphas:
    confidence_pct = int((1-alpha)*100)
    print(f"\\n{'='*60}")
    print(f"Analysis with Alpha = {alpha} ({confidence_pct}% Confidence)")
    print(f"{'='*60}")

    for method in methods:
        print(f"\\n--- Method: {method['name']} ---")

        # We fix n_breakpoints=2 since we know the structure
        result = segmented_trend_test(
            x, t,
            n_breakpoints=2,
            alpha=alpha,
            use_bagging=method['use_bagging'],
            n_bootstrap=20 # Reduced for example speed
        )

        # Print Breakpoint details
        print("Breakpoint Results:")
        if result.n_breakpoints > 0:
            bp_df = pd.DataFrame({
                'Breakpoint': result.breakpoints,
                'Lower CI': [ci[0] for ci in result.breakpoint_cis],
                'Upper CI': [ci[1] for ci in result.breakpoint_cis]
            })
            print(bp_df.to_markdown(index=False, floatfmt=".2f"))
        else:
            print("No breakpoints found.")

        # Print Segment details
        # Focus on the slope Confidence Intervals
        print("\\nSegment Results:")
        cols = ['slope', 'lower_ci', 'upper_ci']
        print(result.segments[cols].to_markdown(index=False, floatfmt=".4f"))

        # Visualize
        fname = f'segmented_plot_alpha_{alpha}_{method["suffix"]}.png'
        save_path = os.path.join(os.path.dirname(__file__), fname)
        plot_segmented_trend(result, x, t, save_path=save_path)
        print(f"Plot saved to {fname}")

"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    try:
        exec(example_code, globals(), local_scope)
    except Exception as e:
        print(f"Error executing example: {e}")
        import traceback
        traceback.print_exc()

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 33: Varying Alpha and Method Comparison

## Overview
This example demonstrates how the `alpha` parameter impacts the Segmented Trend Analysis. The `alpha` value controls the significance level for the Confidence Intervals (CI).

*   **Alpha = 0.10**: 90% Confidence Interval (Narrower)
*   **Alpha = 0.05**: 95% Confidence Interval (Standard)
*   **Alpha = 0.01**: 99% Confidence Interval (Wider)

We also compare two methods for Breakpoint Detection:
1.  **Bagging (Bootstrap Aggregating):** Robust, creates non-parametric CIs for breakpoints.
2.  **Standard OLS:** Faster, assumes normally distributed errors for breakpoint CIs.

## The Data
We simulate a time series with **2 breakpoints** (3 segments) and moderate noise (std=2.0). The slopes are chosen to be relatively close (0.4, 0.1, 0.5) to make the exact breakpoint location uncertain.

## Code & Output

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
Notice the differences in Confidence Interval widths between different Alpha levels and different Methods.

```text
{captured_output}
```

### Step 3: Visual Comparison

#### Alpha = 0.10 (90% Confidence)

| **Bagging** | **Standard OLS** |
| :---: | :---: |
| ![Bagging 0.1](segmented_plot_alpha_0.1_bagging.png) | ![OLS 0.1](segmented_plot_alpha_0.1_ols.png) |

#### Alpha = 0.05 (95% Confidence)

| **Bagging** | **Standard OLS** |
| :---: | :---: |
| ![Bagging 0.05](segmented_plot_alpha_0.05_bagging.png) | ![OLS 0.05](segmented_plot_alpha_0.05_ols.png) |

#### Alpha = 0.01 (99% Confidence)

| **Bagging** | **Standard OLS** |
| :---: | :---: |
| ![Bagging 0.01](segmented_plot_alpha_0.01_bagging.png) | ![OLS 0.01](segmented_plot_alpha_0.01_ols.png) |
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 33 generated successfully.")
