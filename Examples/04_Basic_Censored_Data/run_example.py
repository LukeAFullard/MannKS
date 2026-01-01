
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
    Creates a synthetic dataset with numeric and censored values.

    The data simulates a decreasing trend where values eventually drop below
    detection limits.
    """
    np.random.seed(42)
    n = 25
    years = np.arange(2000, 2000 + n)

    # Generate a decreasing trend: start at 20, decrease by 0.8 per year
    true_values = 20 - 0.8 * np.arange(n) + np.random.normal(0, 2, n)

    # Convert to string format, applying censoring
    values = []

    # Simulate a lab with multiple detection limits over time
    for i, v in enumerate(true_values):
        # Detection limit improves over time: 10.0 -> 5.0 -> 2.0
        if i < 10:
            dl = 10.0
        elif i < 20:
            dl = 5.0
        else:
            dl = 2.0

        if v < dl:
            # Create a left-censored value string
            values.append(f"< {dl}")
        else:
            # Create a regular numeric string
            values.append(f"{v:.2f}")

    # Add a right-censored value for demonstration (e.g., instrument saturation)
    values[2] = "> 25"

    df = pd.DataFrame({
        'Year': years,
        'RawValue': values
    })

    return df

def generate_readme(output_text, plot_filename, calculated_slope):
    readme_content = f"""# Example 4: Handling Basic Censored Data

## Goal
Explain the essential workflow for performing trend analysis on data containing censored values (values below or above a detection limit, e.g., `< 5`).

## Introduction
Environmental data often contains "non-detects" or censored values. Standard statistical methods (like linear regression) cannot handle these strings directly, and substituting them with zero or half the detection limit can introduce significant bias.

The Mann-Kendall test is non-parametric and rank-based, making it naturally well-suited for censored data. However, the data must first be parsed into a format the algorithm understands.

This example demonstrates:
1.  **Parsing** mixed numeric and string data using `prepare_censored_data`.
2.  **Running** the Mann-Kendall test on the processed data.
3.  **Visualizing** the results with censored values clearly marked.

## Step 1: The Data
We generate a synthetic dataset representing a decreasing trend. Note the mixed format: standard numbers (e.g., `18.54`) and censored strings (e.g., `< 10.0`). We also include a right-censored value (`> 25`) to show versatility.

## Step 2: Preprocessing
The `prepare_censored_data` function is the bridge between your raw data and the analysis functions. It parses the strings and returns a DataFrame with three key columns:
*   `value`: The numeric portion (e.g., `10.0` for `< 10.0`).
*   `censored`: A boolean flag (`True` if censored).
*   `cen_type`: The type of censoring (`'lt'` for left, `'gt'` for right, `'not'` for uncensored).

## Step 3: Analysis
We pass the processed DataFrame to `trend_test`. By default, the test uses the `'robust'` method for handling censored data, which correctly handles ranks without needing to substitute values.

## Python Code and Results

```python
import pandas as pd
import numpy as np
import MannKS as mk

# 1. Generate synthetic data with censored values
# (See run_example.py for the full data generation function)
df = create_synthetic_data()

print("--- Raw Input Data (First 5 Rows) ---")
print(df.head())
print()

# 2. Preprocess the data
# The 'RawValue' column contains strings like '< 10.0'
processed_df = mk.prepare_censored_data(df['RawValue'])

print("--- Processed Data (First 5 Rows) ---")
print(processed_df.head())
print()

# 3. Run the Trend Test
# We pass the processed DataFrame as 'x' and the time column as 't'.
result = mk.trend_test(
    x=processed_df,
    t=df['Year'],
    plot_path='{plot_filename}' # Save the plot
)

print("--- Trend Test Results ---")
print(f"Trend: {{result.trend}}")
print(f"P-value: {{result.p:.5f}}")
print(f"Sen's Slope: {{result.slope:.4f}}")
print(f"Significance: {{result.h}}")
```

### Output

```text
{output_text}
```

## Visualizing the Results
The generated plot visualizes the censored data points.
*   **Solid circles** represent uncensored data.
*   **Open triangles** represent censored data (pointing down for `<` and up for `>`).
*   The **Trend Line** (Sen's slope) and **Confidence Intervals** show the estimated direction of change.

![Trend Analysis Plot]({plot_filename})

## Interpretation
*   **Data Handling**: The package automatically identified and handled the censored values. You didn't need to manually substitute them with 0 or MDL/2.
*   **Trend Detection**: Despite the missing exact values for the non-detects, the test correctly identified a **decreasing** trend.
*   **Sen's Slope**: The slope of **{calculated_slope:.4f}** (compared to the synthetic trend of -0.8) demonstrates the robustness of the estimator even with heavy censoring.
"""
    return readme_content

if __name__ == "__main__":
    # Setup for capturing output
    f = io.StringIO()

    # Store result for later use in readme generation
    result_obj = None

    # Execute the workflow
    with contextlib.redirect_stdout(f):
        # 1. Generate Data
        df = create_synthetic_data()

        print("--- Raw Input Data (First 5 Rows) ---")
        print(df.head())
        print()

        # 2. Preprocess
        processed_df = mk.prepare_censored_data(df['RawValue'])

        print("--- Processed Data (First 5 Rows) ---")
        print(processed_df.head())
        print()

        # 3. Run Analysis
        plot_filename = "trend_plot.png"
        plot_path = os.path.join(os.path.dirname(__file__), plot_filename)

        result_obj = mk.trend_test(
            x=processed_df,
            t=df['Year'],
            plot_path=plot_path
        )

        print("--- Trend Test Results ---")
        print(f"Trend: {result_obj.trend}")
        print(f"P-value: {result_obj.p:.5f}")
        print(f"Sen's Slope: {result_obj.slope:.4f}")
        print(f"Significance: {result_obj.h}")

    # Get the captured output
    output_text = f.getvalue()

    # Generate the README content
    readme_content = generate_readme(output_text, plot_filename, result_obj.slope)

    # Write the README file
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"Example 4 complete. Artifacts generated in {os.path.dirname(__file__)}")
