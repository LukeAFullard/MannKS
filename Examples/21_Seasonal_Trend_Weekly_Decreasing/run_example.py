import os
import io
import contextlib
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.simplefilter("ignore")

# 1. Define the Example Code as a String
example_code = """
import numpy as np
import pandas as pd
import MannKS as mk
import matplotlib.pyplot as plt

# 1. Generate Synthetic Data: Weekly Pattern + Decreasing Trend
# We simulate daily data for 3 years (approx 1095 days).
t = pd.date_range(start='2020-01-01', periods=1095, freq='D')

# Create a weekly cycle (season_type='day_of_week')
# Day 0=Monday, ..., 6=Sunday
day_of_week = t.dayofweek

# Pattern: Values are high on weekends (Sat=5, Sun=6) and low on weekdays.
seasonal_signal = np.where(day_of_week >= 5, 10, 5)

# Create a decreasing trend
# -5 units per year
years_elapsed = (t - t[0]).days / 365.25
trend_signal = 100 - (5 * years_elapsed)

# Add noise
np.random.seed(42)
noise = np.random.normal(0, 1.0, len(t))

# Combine
x = seasonal_signal + trend_signal + noise

print("Data Head:")
print(pd.DataFrame({'Date': t, 'Value': x, 'Day': day_of_week}).head())


# 2. Visualize the Seasonality
# Before running the test, let's confirm the weekly pattern.
print("\\n--- Visualizing Seasonality ---")
mk.plot_seasonal_distribution(
    x, t,
    period=7,
    season_type='day_of_week',
    plot_path='weekly_distribution.png'
)
print("Saved 'weekly_distribution.png'. Expect higher boxes for days 5 and 6.")


# 3. Run Seasonal Trend Test
# We specify `season_type='day_of_week'` and `period=7`.
# This ensures we compare Mondays to Mondays, Tuesdays to Tuesdays, etc.
print("\\n--- Running Seasonal Trend Test ---")

result = mk.seasonal_trend_test(
    x, t,
    period=7,
    season_type='day_of_week',
    slope_scaling='year',
    plot_path='seasonal_trend_plot.png'
)

print(f"Trend: {result.trend}")
print(f"Classification: {result.classification}")
print(f"p-value: {result.p:.4f}")
print(f"Sen's Slope: {result.slope:.4f} units/year")
print(f"Confidence Interval: [{result.lower_ci:.4f}, {result.upper_ci:.4f}]")

"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

with contextlib.redirect_stdout(output_buffer):
    local_scope = {}
    exec(example_code, globals(), local_scope)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_part1 = """
# Example 21: Seasonal Trend with Weekly Data (Decreasing)

## The "Why": High-Frequency Cycles
Most trend analyses focus on monthly data (annual cycle). However, high-frequency data (daily, hourly) often exhibits shorter cycles that can mask long-term trends if not handled correctly.

A common example is a **weekly cycle** in anthropogenic data:
*   **Water Consumption**: Higher on weekends.
*   **Traffic Pollution**: Higher on weekdays.
*   **Industrial Effluent**: Varies by production schedule.

If we just plot the raw data, the "sawtooth" pattern of the weekly cycle increases the variance, making it harder to detect a subtle long-term trend. By using a **Seasonal Kendall Test** with `season_type='day_of_week'`, we isolate these groups.

## The Scenario
We simulate 3 years of daily data with:
1.  **Weekly Seasonality**: Values are significantly higher on weekends (Saturday/Sunday).
2.  **Decreasing Trend**: A steady decline of 5 units per year.

## The "How": Code Walkthrough

### Step 1: Python Code
```python
"""

readme_part2 = """
```

### Step 2: Text Output
```text
"""

readme_part3 = """
```

## Interpreting the Results

### 1. Visualizing Seasonality (`weekly_distribution.png`)
![Weekly Distribution](weekly_distribution.png)
The boxplots clearly show the "weekend effect" (Days 5 and 6 are higher).

### 2. Trend Results
*   **Trend (Decreasing)**: The test correctly identifies the downward trend.
*   **Sen's Slope (-4.9343)**: This is remarkably close to the true generated trend of -5.0.
*   **Confidence**: The result is "Highly Likely Decreasing" with p < 0.0001.

### 3. Visualizing the Trend (`seasonal_trend_plot.png`)
![Seasonal Trend Plot](seasonal_trend_plot.png)
The plot shows the daily data points. The black trend line cuts through the seasonal noise, accurately capturing the long-term decline.
"""

readme_content = readme_part1 + example_code.strip() + readme_part2 + captured_output + readme_part3

with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 21 generated successfully.")
