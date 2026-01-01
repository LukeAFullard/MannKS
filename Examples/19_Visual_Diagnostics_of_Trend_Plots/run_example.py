import os
import io
import contextlib
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt

# --- 1. Define the Example Code as a String ---
example_code = """
import numpy as np
import pandas as pd
import MannKS as mk
import os

# Set random seed for reproducibility
np.random.seed(42)

# Determine where to save the plots (current directory by default)
output_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'

# Common time vector for all scenarios (15 years)
t = np.arange(2005, 2020)
n = len(t)

# --- Scenario 1: The "Textbook" Trend (Clear Signal, Low Noise) ---
# A steady increase with very little random variation.
# This represents the "ideal" case where the signal dominates.
noise_low = np.random.normal(0, 0.5, n)
x_clear = 5 + 0.8 * (t - 2005) + noise_low

print("\\n--- Scenario 1: Clear Trend ---")
plot_path_clear = os.path.join(output_dir, 'plot_clear_trend.png')
result_clear = mk.trend_test(x_clear, t, plot_path=plot_path_clear)
print(f"Trend: {result_clear.trend} ({result_clear.classification})")
print(f"p-value: {result_clear.p:.4f}, Slope: {result_clear.slope:.4f}")

# --- Scenario 2: The "Ambiguous" Result (High Variance/Uncertainty) ---
# A trend exists, but the data is very noisy (scattered).
# This results in wide confidence intervals, indicating we are less certain
# about the exact rate of change, even if the direction is likely.
noise_high = np.random.normal(0, 5.0, n) # 10x more noise
x_noisy = 5 + 0.8 * (t - 2005) + noise_high

print("\\n--- Scenario 2: High Uncertainty (Wide CI) ---")
plot_path_noisy = os.path.join(output_dir, 'plot_wide_ci.png')
result_noisy = mk.trend_test(x_noisy, t, plot_path=plot_path_noisy)
print(f"Trend: {result_noisy.trend} ({result_noisy.classification})")
print(f"p-value: {result_noisy.p:.4f}, Slope: {result_noisy.slope:.4f}")

# --- Scenario 3: The "Flatline" (No Trend) ---
# Data that fluctuates around a constant mean.
# The trend line should be flat, and the confidence interval should span across zero.
x_flat = 10 + np.random.normal(0, 2.0, n)

print("\\n--- Scenario 3: No Trend ---")
plot_path_flat = os.path.join(output_dir, 'plot_no_trend.png')
result_flat = mk.trend_test(x_flat, t, plot_path=plot_path_flat)
print(f"Trend: {result_flat.trend} ({result_flat.classification})")
print(f"p-value: {result_flat.p:.4f}, Slope: {result_flat.slope:.4f}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    exec_globals = globals().copy()
    exec_globals['__file__'] = __file__
    exec(example_code, exec_globals, local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 19: Visual Diagnostics of Trend Plots

## The Art of Reading Trend Plots

A statistical result (like a p-value) gives you a "Yes/No" answer, but a plot tells the full story. The `trend_test` function generates diagnostic plots that reveal the quality of your data, the reliability of the trend, and potential issues.

This guide teaches you how to interpret the three most common visual patterns.

### Key Visual Elements
1.  **The Data Points (Blue Dots):** Show the raw distribution. Look for outliers, gaps, or clusters.
2.  **The Trend Line (Solid Red/Blue Line):** The Sen's Slope. It passes through the median of the data.
3.  **The Confidence Interval (Shaded Region):** The "Cone of Uncertainty." It shows the range of slopes that are statistically plausible.
    *   **Narrow Cone:** High precision; we are sure about the rate of change.
    *   **Wide Cone:** Low precision; the data is noisy.
    *   **Cone Crossing Zero:** We cannot be sure if the trend is increasing or decreasing (i.e., No Trend).

## The Scenarios

### 1. The "Textbook" Trend (Clear Signal)
*   **Visuals:** Data points tightly hug the trend line. The shaded confidence interval (CI) is narrow and clearly points up (or down).
*   **Meaning:** Strong evidence of a trend. The rate of change is well-defined.

### 2. The "Ambiguous" Result (High Uncertainty)
*   **Visuals:** Data points are scattered widely. The trend line points up, but the shaded CI is very wide (like a trumpet).
*   **Meaning:** There might be a trend, but the *magnitude* is uncertain. The "signal-to-noise" ratio is low. You might get a result like "Likely Increasing" but with a p-value near the borderline (e.g., 0.08).

### 3. The "Flatline" (No Trend)
*   **Visuals:** The trend line is roughly horizontal. Crucially, the shaded CI is often symmetrical and usually "swallows" the horizontal line.
*   **Meaning:** The data is just random noise around a mean. There is no consistent direction.

## The "How": Code Walkthrough

We generate synthetic data to simulate these three specific conditions.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Visual Diagnostics

### Scenario 1: Clear Trend
![Clear Trend](plot_clear_trend.png)
*   **Diagnosis:** Perfect. The CI is tight, indicating high confidence in both the *existence* and the *rate* of the trend.

### Scenario 2: High Uncertainty (Wide CI)
![Wide CI](plot_wide_ci.png)
*   **Diagnosis:** Noisy. Notice how the shaded area flares out. Even if the p-value is significant, the wide CI warns you that the "true" rate of change could be very different from the calculated slope.

### Scenario 3: No Trend
![No Trend](plot_no_trend.png)
*   **Diagnosis:** Null. The slope is near zero, and the confidence bounds likely include zero (meaning a flat line is a plausible fit for this data).
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 19 generated successfully.")
