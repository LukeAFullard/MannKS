
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib

# Ensure the local MannKS package is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import MannKS as mk

def create_synthetic_data():
    """
    Creates a synthetic dataset of daily observations over 5 years.

    The data has:
    1. A strong seasonal peak around day 100 (Spring).
    2. A gradual increasing trend over the years (+2 units/year).
    """
    np.random.seed(42)
    n_years = 5
    start_year = 2015

    dates = []
    values = []

    # Define trend: +2.0 units per year
    # (Approx +0.0055 units per day)
    slope_per_day = 2.0 / 365.25
    base_value = 10

    start_date = pd.Timestamp(f"{start_year}-01-01")

    for day_offset in range(n_years * 365):
        current_date = start_date + pd.Timedelta(days=day_offset)
        day_of_year = current_date.dayofyear

        # 1. Trend Component
        trend_val = base_value + (day_offset * slope_per_day)

        # 2. Seasonal Component (Gaussian peak around day 100, width 30 days)
        # Amplitude of 15 units
        seasonal_val = 15 * np.exp(-0.5 * ((day_of_year - 100) / 30)**2)

        # 3. Noise
        noise = np.random.normal(0, 1.5)

        val = trend_val + seasonal_val + noise

        dates.append(current_date)
        values.append(val)

    df = pd.DataFrame({
        'Date': dates,
        'Value': values
    })

    return df

def generate_readme(output_text, plot_filename, result):
    readme_content = f"""# Example 24: Advanced Seasonality with `day_of_year`

## Goal
Showcase a granular seasonal analysis using the **day of the year** (`season_type='day_of_year'`). This is particularly useful for environmental data driven by specific annual events, such as spring runoff, phenology, or temperature cycles, where "Month" is too coarse a grouping.

## Introduction
Standard seasonal analysis often groups data by month (12 seasons). However, some natural phenomena are better described by the day of the year (1-366). For example, a river's peak flow might shift by a few weeks, which could be split across March and April. Grouping by day allows us to compare "Day 100 in 2015" against "Day 100 in 2016", providing a very fine-grained control for seasonality.

**Why use `day_of_year`?**
*   **Precision:** Removes the arbitrary boundaries of calendar months.
*   **Event Alignment:** Better aligns with biological or physical cycles that don't follow the Gregorian calendar months perfectly.

**Trade-offs:**
*   **Data Requirements:** You need multi-year daily data. If you have gaps, many "seasons" (days) might have insufficient data for a trend test.
*   **Computational Load:** Instead of 12 seasons, the test analyzes up to 366 seasons.

## Step 1: The Data
We generate 5 years of daily data.
*   **Trend:** A steady increase of **+2.0 units/year**.
*   **Seasonality:** A strong peak around **Day 100** (early April) each year, simulating a spring event.

## Step 2: Analysis
We run the `seasonal_trend_test` with `season_type='day_of_year'`.
*   **Period:** Implicitly handles the annual cycle (period ~ 365/366).
*   **Comparison:** It compares Day 1 of Year 1 vs Day 1 of Year 2, Day 2 vs Day 2, etc.

## Python Code and Results

```python
import pandas as pd
import numpy as np
import MannKS as mk

# 1. Generate synthetic daily data (5 years)
# (See run_example.py for details)
df = create_synthetic_data()

print(f"Generated {{len(df)}} daily observations.")
print("True Trend: +2.0 units/year")
print()

# 2. Run Seasonal Trend Test with 'day_of_year'
# We also use slope_scaling='year' to get a readable slope unit.
result = mk.seasonal_trend_test(
    x=df['Value'],
    t=df['Date'],
    season_type='day_of_year',
    slope_scaling='year',
    plot_path='{plot_filename}'
)

print("--- Seasonal Trend Test Results ---")
print(f"Trend: {{result.trend}}")
print(f"P-value: {{result.p:.5f}}")
print(f"Sen's Slope: {{result.slope:.4f}} {{result.slope_units}}")
print(f"Significance: {{result.h}}")
```

### Output

```text
{output_text}
```

## Visualizing the Results
The plot below shows the daily data. Because `day_of_year` seasonality is high-frequency, the "seasonal pattern" is the entire annual shape (the humps). The trend line (Sen's slope) cuts through the noise and seasonality to show the underlying increase.

![Trend Analysis Plot]({plot_filename})

## Interpretation
*   **Trend Detected:** The test correctly identifies an **increasing** trend.
*   **Slope Accuracy:** The calculated slope is **{result.slope:.4f} {result.slope_units}**, which is very close to the true synthetic trend of **+2.0 units/year**.
*   **Robustness:** By comparing "apples to apples" (Day X vs Day X), the test effectively removes the massive seasonal variation (the spring peak) from the trend calculation. If we had ignored seasonality, the variance might have been too high to detect the trend, or the seasonal shape could have confounded a simple linear regression.
"""
    return readme_content

if __name__ == "__main__":
    # Setup for capturing output
    f = io.StringIO()
    result_obj = None

    # Execute the workflow
    with contextlib.redirect_stdout(f):
        # 1. Generate Data
        df = create_synthetic_data()

        print(f"Generated {{len(df)}} daily observations.")
        print("True Trend: +2.0 units/year")
        print()

        # 2. Run Analysis
        plot_filename = "seasonal_plot_doy.png"
        plot_path = os.path.join(os.path.dirname(__file__), plot_filename)

        # Note: seasonal_trend_test handles the period logic for day_of_year internally
        # (usually implies period=1 for the cycle_identifier logic in some contexts,
        # or it separates by unique season ID).
        # For 'day_of_year', the seasons are 1..366.
        result_obj = mk.seasonal_trend_test(
            x=df['Value'],
            t=df['Date'],
            season_type='day_of_year',
            slope_scaling='year',
            plot_path=plot_path
        )

        print("--- Seasonal Trend Test Results ---")
        print(f"Trend: {result_obj.trend}")
        print(f"P-value: {result_obj.p:.5f}")
        print(f"Sen's Slope: {result_obj.slope:.4f} {result_obj.slope_units}")
        print(f"Significance: {result_obj.h}")

    # Get the captured output
    output_text = f.getvalue()

    # Generate the README content
    readme_content = generate_readme(output_text, plot_filename, result_obj)

    # Write the README file
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"Example 24 complete. Artifacts generated in {os.path.dirname(__file__)}")
