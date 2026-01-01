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
import matplotlib.pyplot as plt

# We will generate synthetic data for 3 scenarios to demonstrate how to interpret residual plots.

# --- Helper Function for Data Generation ---
def generate_data(scenario, n=50, seasonal=False):
    t = np.arange(n)
    if not seasonal:
        if scenario == 'appropriate':
            # Linear trend + White Noise
            noise = np.random.normal(0, 1, n)
            values = 0.5 * t + noise
        elif scenario == 'poor':
            # Non-linear (Quadratic) trend
            noise = np.random.normal(0, 1, n)
            values = 0.05 * (t - 25)**2 + noise
        elif scenario == 'middle':
            # Heteroscedasticity (Variance increases with time)
            noise = np.random.normal(0, 1 + 0.1 * t, n)
            values = 0.5 * t + noise
    else:
        # Seasonal Data (Monthly over 5 years, n=60)
        dates = pd.date_range(start='2020-01-01', periods=n, freq='ME')
        t = np.arange(n)
        seasonality = 5 * np.sin(2 * np.pi * t / 12)

        if scenario == 'appropriate':
            noise = np.random.normal(0, 1, n)
            values = 0.2 * t + seasonality + noise
        elif scenario == 'poor':
            # Non-linear trend + Seasonality
            noise = np.random.normal(0, 1, n)
            values = 0.05 * (t - 30)**2 + seasonality + noise
        elif scenario == 'middle':
            # Changing Seasonality (Amplitude increases)
            seasonality_dynamic = (5 + 0.2 * t) * np.sin(2 * np.pi * t / 12)
            noise = np.random.normal(0, 1, n)
            values = 0.2 * t + seasonality_dynamic + noise
        return values, dates

    return values, t

# --- Part 1: Non-Seasonal Trend Analysis ---
scenarios = ['appropriate', 'poor', 'middle']
print("Running Non-Seasonal Diagnostics...")

for sc in scenarios:
    values, t = generate_data(sc, n=50, seasonal=False)

    # Run Trend Test with Residual Plot
    # The `residual_plot_path` argument triggers the new diagnostic plots
    filename = f"residuals_nonseasonal_{sc}.png"
    mk.trend_test(values, t, residual_plot_path=filename)
    print(f"Generated {filename}")


# --- Part 2: Seasonal Trend Analysis ---
print("\\nRunning Seasonal Diagnostics...")

for sc in scenarios:
    values, dates = generate_data(sc, n=60, seasonal=True)

    # Run Seasonal Trend Test with Residual Plot
    filename = f"residuals_seasonal_{sc}.png"
    mk.seasonal_trend_test(
        values, dates,
        period=12, season_type='month',
        residual_plot_path=filename
    )
    print(f"Generated {filename}")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

# Change to the script's directory so plots are saved there
original_cwd = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    with contextlib.redirect_stdout(output_buffer):
        local_scope = {}
        exec(example_code, globals(), local_scope)
finally:
    # Restore original CWD
    os.chdir(original_cwd)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 28: Residual Diagnostics

## The "Why": Checking Your Assumptions
The Mann-Kendall test and Sen's slope estimator are robust, non-parametric methods. However, like all statistical tests, they tell you more when the underlying assumptions are met.

The most critical assumption for the **Sen's Slope** (which describes the *magnitude* of the trend) is that the trend is **linear**. If the true trend is curved (e.g., exponential or quadratic), the Sen's slope provides a linear approximation that might be misleading.

**Residual Plots** help you diagnose:
1.  **Linearity:** Does a straight line fit the data well?
2.  **Homogeneity of Variance:** Is the noise constant over time?
3.  **Outliers:** Are there extreme values skewing the results?

## The "How": Using `residual_plot_path`

Both `trend_test` and `seasonal_trend_test` now accept an optional argument: `residual_plot_path`. When provided, the function generates a diagnostic figure with two subplots:
*   **Residuals vs. Time:** To check for patterns (non-linearity) or changing variance.
*   **Histogram of Residuals:** To check the distribution of the noise.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Output
```text
{captured_output}
```

## Interpreting the Plots

We generated data for three common scenarios. Here is how to interpret them.

### Part 1: Non-Seasonal Data

#### 1. Appropriate Fit (Linear Trend)
![Appropriate Fit](residuals_nonseasonal_appropriate.png)
*   **Observation:** The residuals are scattered randomly around the zero line (horizontal dashed line). There is no discernable pattern (no "U-shape" or waves). The spread of points is roughly constant from left to right.
*   **Conclusion:** The linear Sen's slope is a good description of this trend.

#### 2. Poor Fit (Non-Linear Trend)
![Poor Fit](residuals_nonseasonal_poor.png)
*   **Observation:** There is a clear **U-shape** (parabola) in the residuals. They start positive, go negative in the middle, and return to positive.
*   **Conclusion:** The data has a quadratic trend. The linear Sen's slope is an oversimplification and misses the curvature. You might need a different model or to split the time series.

#### 3. "Middle" Case (Heteroscedasticity)
![Middle Case](residuals_nonseasonal_middle.png)
*   **Observation:** The trend is linear (centered on zero), but the **spread** of the residuals increases significantly over time (a "fan" shape).
*   **Conclusion:** This is "Heteroscedasticity." While the Sen's slope is still a valid estimate of the *average* rate of change, the uncertainty (confidence intervals) might be less reliable. The Mann-Kendall test is generally robust to this, but it's worth noting in your report.

---

### Part 2: Seasonal Data

#### 1. Appropriate Fit (Seasonal Linear)
![Seasonal Appropriate](residuals_seasonal_appropriate.png)
*   **Observation:** Residuals are random. Note that `seasonal_trend_test` removes the *seasonality* before checking for a monotonic trend, but the residuals plotted here are against the **global** linear trend. You might see some seasonal wiggles remaining if the seasonal amplitude is very large compared to the trend, but generally, they should look stable.
*   **Conclusion:** Good fit.

#### 2. Poor Fit (Seasonal Non-Linear)
![Seasonal Poor](residuals_seasonal_poor.png)
*   **Observation:** Similar to the non-seasonal case, a clear curved pattern dominates the residuals, overriding the seasonal noise.
*   **Conclusion:** The trend is not linear.

#### 3. "Middle" Case (Changing Seasonality)
![Seasonal Middle](residuals_seasonal_middle.png)
*   **Observation:** The residuals might show a "beating" pattern or a fan shape where the seasonal waves get larger over time.
*   **Conclusion:** The assumption of "stable seasonality" might be violated. The trend result reflects the average change, but the changing seasonal dynamics are an important feature of the data to investigate separately.
"""

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 28 generated successfully.")
