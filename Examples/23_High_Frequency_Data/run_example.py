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

np.random.seed(42)

# --- Part 1: High Frequency Data (Minutes/Seconds) ---
# Goal: Demonstrate that the package can handle very small time increments
# and use `slope_scaling` to return meaningful results.

print("--- Part 1: High-Frequency Sensor Data (Seconds) ---")

# 1. Generate Synthetic Sensor Data
# Imagine a temperature sensor logging every 10 seconds for 1 hour.
# We want to detect a rapid heating event (increase).
start_time = pd.Timestamp("2023-01-01 12:00:00")
# 360 points (1 hour of 10-second intervals)
t_sensor = pd.date_range(start=start_time, periods=360, freq='10s')

# Trend: Increase of 0.5 degrees per minute
# Slope per second = 0.5 / 60 = 0.00833 deg/sec
true_slope_per_min = 0.5
true_slope_per_sec = true_slope_per_min / 60
noise = np.random.normal(0, 0.1, len(t_sensor))
x_sensor = 20 + (true_slope_per_sec * np.arange(len(t_sensor)) * 10) + noise

# 2. Run Trend Test with Scaling
# If we didn't scale, the result would be in "degrees per second" (approx 0.008).
# By setting `slope_scaling='minute'`, we get "degrees per minute".
result_sensor = mk.trend_test(x_sensor, t_sensor, slope_scaling='minute', plot_path='sensor_trend_plot.png')

print(f"Detected Trend: {result_sensor.trend}")
print(f"Sen's Slope: {result_sensor.slope:.4f} (degrees per minute)")
print(f"Expected Slope: {true_slope_per_min:.4f}")
print(f"Confidence Interval: [{result_sensor.lower_ci:.4f}, {result_sensor.upper_ci:.4f}]")


# --- Part 2: Hourly Data with Diurnal (Daily) Seasonality ---
# Goal: Demonstrate seasonal analysis where the "season" is the hour of the day (0-23).
# Many environmental phenomena (temperature, dissolved oxygen) cycle daily.

print("\\n--- Part 2: Hourly Data with Diurnal Seasonality ---")

# 1. Generate Synthetic Hourly Data
# 10 days of hourly data (24 * 10 = 240 points)
t_hourly = pd.date_range(start="2023-01-01", periods=240, freq='h')

# Create a Daily Cycle (Sine wave)
# Period = 24 hours.
hour_of_day = t_hourly.hour
daily_cycle = 5 * np.sin(2 * np.pi * hour_of_day / 24)

# Add a Long-Term Trend
# Increase of 0.1 units per day (0.1/24 per hour)
# We use numeric time (hours from start) for the trend component
hours_from_start = np.arange(len(t_hourly))
trend_component = (0.1 / 24) * hours_from_start

x_hourly = 10 + daily_cycle + trend_component + np.random.normal(0, 0.5, len(t_hourly))

# 2. Run Seasonal Trend Test
# We define the season as the 'hour' of the day.
# The test will compare 1 AM to 1 AM, 2 PM to 2 PM, etc., removing the daily cycle's influence.
# We set `slope_scaling='day'` to get the trend in "units per day".
result_hourly = mk.seasonal_trend_test(
    x_hourly,
    t_hourly,
    season_type='hour',
    period=24,
    slope_scaling='day',
    plot_path='hourly_seasonal_plot.png'
)

print(f"Detected Trend: {result_hourly.trend}")
print(f"Sen's Slope: {result_hourly.slope:.4f} (units per day)")
print(f"Expected Slope: 0.1000")
print(f"Confidence Interval: [{result_hourly.lower_ci:.4f}, {result_hourly.upper_ci:.4f}]")
"""

# --- 2. Execute the Code and Capture Output ---
output_buffer = io.StringIO()

# Change CWD to the script's directory so plots are saved there
script_dir = os.path.dirname(os.path.abspath(__file__))
original_cwd = os.getcwd()
os.chdir(script_dir)

try:
    with contextlib.redirect_stdout(output_buffer):
        local_scope = {}
        exec(example_code, globals(), local_scope)
finally:
    os.chdir(original_cwd)

captured_output = output_buffer.getvalue()

# --- 3. Generate the README.md ---
readme_content = f"""
# Example 23: High Frequency Data (Hours, Minutes, Seconds)

## The "Why": Beyond Years and Months
Standard trend analysis usually deals with annual or monthly data. However, modern sensors often log data at much higher frequencies—hourly, every minute, or even every second.

This creates two challenges:
1.  **Tiny Numbers**: Slopes calculated in standard "units per second" are often infinitesimally small (e.g., 0.0000005), making them hard to read.
2.  **Fast Cycles**: High-frequency data often contains rapid cycles (e.g., daily temperature swings) that act as "seasonality" and can obscure long-term trends.

This example demonstrates how to handle both cases using `slope_scaling` and `season_type`.

## The "How": Code Walkthrough

We will perform two analyses:
1.  **Seconds/Minutes**: Analyzing a rapid temperature increase from a sensor log.
2.  **Hourly**: Analyzing a long-term trend hidden inside a daily (diurnal) cycle.

### Step 1: Python Code
```python
{example_code.strip()}
```

### Step 2: Text Output
```text
{captured_output}
```

## Interpreting the Results

### Part 1: High-Frequency Sensor Data
*   **The Problem**: If we didn't use scaling, the slope would be `~0.0083` degrees/second. While correct, it's not intuitive for a human monitoring a process.
*   **The Solution**: By using `slope_scaling='minute'`, the output `Sen's Slope` matches our intuition: **~0.5 degrees per minute**.
*   **The Plot**:
    ![Sensor Trend Plot](sensor_trend_plot.png)
    The plot shows the rapid, linear rise in temperature.

### Part 2: Hourly Data with Diurnal Seasonality
*   **The Problem**: The data swings wildly every day (sine wave) due to the daily cycle. A simple linear regression might be confused by the noise if the data started at a peak and ended at a trough.
*   **The Solution**: We used `seasonal_trend_test` with `season_type='hour'`.
    *   This treats "1 AM" as a season, "2 AM" as a season, etc.
    *   It effectively removes the daily cycle by comparing like-with-like.
*   **The Result**: The test correctly identified the slow, underlying trend of **0.1 units per day**, ignoring the large daily fluctuations.
*   **The Plot**:
    ![Hourly Seasonal Plot](hourly_seasonal_plot.png)
    Notice how the trend line (black) cuts through the middle of the daily oscillations, capturing the true long-term direction.

## Key Takeaway
*   **Scale Your Slopes**: Use `slope_scaling` ('minute', 'hour', 'day') to make high-frequency results human-readable.
*   **Define Your Season**: "Seasonality" isn't just Spring/Summer/Fall/Winter. For hourly data, the "season" is the hour of the day.
"""

with open(os.path.join(script_dir, 'README.md'), 'w') as f:
    f.write(readme_content)

print("Example 23 generated successfully.")
