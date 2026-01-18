
import os
import subprocess
import io

def merge_examples():
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Run Bagging Script
    print("Running Bagging Example...")
    bagging_output = subprocess.check_output(
        ['python', 'run_example_bagging.py'],
        cwd=cwd,
        env={**os.environ, 'PYTHONPATH': os.path.join(cwd, '../../')}
    ).decode('utf-8')

    # Run OLS Script
    print("Running OLS Example...")
    ols_output = subprocess.check_output(
        ['python', 'run_example_ols.py'],
        cwd=cwd,
        env={**os.environ, 'PYTHONPATH': os.path.join(cwd, '../../')}
    ).decode('utf-8')

    # Combine content for README
    example_code_bagging = open(os.path.join(cwd, 'run_example_bagging.py')).read()
    example_code_ols = open(os.path.join(cwd, 'run_example_ols.py')).read()

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

## Part 1: Bagging Method

### Code
```python
{example_code_bagging.strip()}
```

### Output
```text
{bagging_output.strip()}
```

## Part 2: Standard OLS Method

### Code
```python
{example_code_ols.strip()}
```

### Output
```text
{ols_output.strip()}
```

## Visual Comparison

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

## Insights

1.  **Slope CIs:** As `alpha` decreases (from 0.10 to 0.01), the confidence intervals for the slopes (and the shaded regions in the plots) become wider. This represents the increased certainty required to capture the true value.
2.  **Breakpoint CIs:**
    *   **Standard OLS** usually provides symmetric, often narrower confidence intervals around the detected breakpoint. However, these assume well-behaved errors and can be overly optimistic or misleading in complex data.
    *   **Bagging** provides non-parametric intervals that may be wider and asymmetric. This reflects the true uncertainty and multimodality of the breakpoint location distribution, especially when noise is high or slope differences are subtle. In this example, the Bagging intervals often capture the uncertainty more realistically.
"""

    with open(os.path.join(cwd, 'README.md'), 'w') as f:
        f.write(readme_content)

    print("Example 33 README generated successfully.")

if __name__ == "__main__":
    merge_examples()
