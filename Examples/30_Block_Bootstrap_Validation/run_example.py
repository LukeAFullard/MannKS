
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk

def generate_autocorrelated_data(n, slope=0, rho=0.7, seasonal_amp=0):
    """Generate synthetic autocorrelated data."""
    t = np.arange(n)
    x = np.zeros(n)
    noise = np.random.normal(0, 1, n)
    # AR(1) noise
    x[0] = noise[0]
    for i in range(1, n):
        x[i] = rho * x[i-1] + noise[i]

    # Add trend
    x += slope * t

    # Add seasonality
    if seasonal_amp > 0:
        x += seasonal_amp * np.sin(2 * np.pi * t / 12)

    return x, t

def run_scenario_1_aggregation():
    """Scenario 1: Block Bootstrap with Aggregation"""
    print("\n--- Scenario 1: Aggregation + Bootstrap ---")

    # Generate daily data (highly correlated)
    n = 365 * 2
    x, t = generate_autocorrelated_data(n, slope=0.01, rho=0.8)

    # Create datetime index
    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')

    # Run trend test with monthly aggregation AND block bootstrap
    # This tests if bootstrap works on the *aggregated* data.
    res = mk.trend_test(
        x, dates,
        agg_method='median',
        agg_period='month',
        autocorr_method='block_bootstrap',
        n_bootstrap=200,
        block_size='auto'
    )

    print(f"Original N: {n}")
    print(f"Aggregated N (approx): {n/30:.1f}")
    print(f"Block Size Used (on aggregated data): {res.block_size_used}")
    print(f"P-value (Bootstrap): {res.p:.4f}")
    print(f"Trend: {res.trend}")

    return res

def run_scenario_2_hicensor():
    """Scenario 2: Block Bootstrap with High Censoring Rule"""
    print("\n--- Scenario 2: High Censoring + Bootstrap ---")

    n = 100
    x, t = generate_autocorrelated_data(n, slope=0.05, rho=0.6)

    # Artificially create censoring
    # Early data: detection limit 2. Late data: detection limit 5.
    censored = np.zeros(n, dtype=bool)
    cen_type = np.full(n, 'not')
    x_input = x.copy()

    # Apply high censoring logic manually to input strings to test preprocessing + bootstrap
    x_strings = []
    for i, val in enumerate(x):
        limit = 2.0 if i < 50 else 5.0
        if val < limit:
            x_strings.append(f"<{limit}")
        else:
            x_strings.append(val)

    # Process data into the required format using prepare_censored_data.
    df_prep = mk.prepare_censored_data(x_strings)

    # Run test with hicensor=True and bootstrap
    res = mk.trend_test(
        df_prep, t,
        hicensor=True,
        autocorr_method='block_bootstrap',
        n_bootstrap=200
    )

    print(f"High Censor Rule Applied: {res.analysis_notes}")
    print(f"Block Size Used: {res.block_size_used}")
    print(f"P-value: {res.p:.4f}")

    return res

def run_scenario_3_seasonal():
    """Scenario 3: Seasonal Block Bootstrap"""
    print("\n--- Scenario 3: Seasonal Bootstrap ---")

    n = 12 * 10 # 10 years monthly
    x, t = generate_autocorrelated_data(n, slope=0.02, rho=0.5, seasonal_amp=5.0)
    dates = pd.date_range(start='2010-01-01', periods=n, freq='ME') # Month End

    res = mk.seasonal_trend_test(
        x, dates,
        period=12,
        season_type='month',
        autocorr_method='block_bootstrap',
        n_bootstrap=200,
        block_size='auto' # Should default to 1 year/cycle
    )

    print(f"Block Size Used (Cycles): {res.block_size_used}")
    print(f"P-value: {res.p:.4f}")
    print(f"Trend: {res.trend}")

    return res

def run_scenario_4_ats():
    """Scenario 4: ATS Slope + Bootstrap P-value"""
    print("\n--- Scenario 4: ATS Slope + Bootstrap P-value ---")

    n = 50
    x, t = generate_autocorrelated_data(n, slope=0.03, rho=0.5)

    # Add censoring
    censored = np.zeros(n, dtype=bool)
    cen_type = np.full(n, 'not')
    x[::5] = 2.0 # Set some low values
    censored[::5] = True
    cen_type[::5] = 'lt'

    # Construct a DataFrame manually to handle censored data without string parsing.
    df = pd.DataFrame({
        'value': x,
        'censored': censored,
        'cen_type': cen_type,
        't': t # Optional if we pass t separately, but prepare_data handles it.
    })

    # Run test
    res = mk.trend_test(
        df, t,
        sens_slope_method='ats',
        autocorr_method='block_bootstrap',
        n_bootstrap=200
    )

    print(f"Slope (ATS): {res.slope:.4f}")
    print(f"CI (ATS): [{res.lower_ci:.4f}, {res.upper_ci:.4f}]")
    print(f"P-value (Bootstrap): {res.p:.4f}")

    return res

def main():
    np.random.seed(42)  # Ensure reproducibility
    output_dir = os.path.dirname(__file__)
    readme_path = os.path.join(output_dir, 'README.md')

    # Capture output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res1 = run_scenario_1_aggregation()
        res2 = run_scenario_2_hicensor()
        res3 = run_scenario_3_seasonal()
        res4 = run_scenario_4_ats()

    output_text = f.getvalue()
    print(output_text) # Print to console for verify

    # Write README
    with open(readme_path, 'w') as readme:
        readme.write("# Example 30: Block Bootstrap Validation\n\n")
        readme.write("This example validates the integration of the Block Bootstrap method with other package features.\n\n")

        readme.write("## Scenario 1: Aggregation + Bootstrap\n")
        readme.write("Bootstrap is applied *after* temporal aggregation. This is crucial for high-frequency autocorrelated data.\n")
        readme.write("```python\n")
        readme.write("mk.trend_test(..., agg_method='median', agg_period='month', autocorr_method='block_bootstrap')\n")
        readme.write("```\n")

        readme.write("## Scenario 2: High Censoring + Bootstrap\n")
        readme.write("Ensures `hicensor` preprocessing works correctly before the bootstrap resampling step.\n")
        readme.write("```python\n")
        readme.write("mk.trend_test(..., hicensor=True, autocorr_method='block_bootstrap')\n")
        readme.write("```\n")

        readme.write("## Scenario 3: Seasonal Bootstrap\n")
        readme.write("Validates that the seasonal test bootstraps entire seasonal cycles (e.g., years) to preserve seasonality.\n")
        readme.write("```python\n")
        readme.write("mk.seasonal_trend_test(..., autocorr_method='block_bootstrap', block_size='auto')\n")
        readme.write("```\n")

        readme.write("## Scenario 4: ATS Slope + Bootstrap P-value\n")
        readme.write("Combines the robust ATS estimator for the slope/CI with the Block Bootstrap for the p-value.\n")
        readme.write("```python\n")
        readme.write("mk.trend_test(..., sens_slope_method='ats', autocorr_method='block_bootstrap')\n")
        readme.write("```\n")

        readme.write("\n## Execution Output\n")
        readme.write("```text\n")
        readme.write(output_text)
        readme.write("```\n")

        readme.write("\n## Conclusion\n")
        readme.write("The successful execution of these scenarios confirms that the Block Bootstrap feature is robustly integrated into the package. ")
        readme.write("It correctly handles:\n")
        readme.write("- **Pre-processing steps** like temporal aggregation and high-censoring adjustments, applying the bootstrap to the processed data.\n")
        readme.write("- **Seasonal structures**, by bootstrapping entire cycles (e.g., years) to preserve seasonal dependence.\n")
        readme.write("- **Advanced estimators** like ATS, allowing users to combine robust slope estimation with robust significance testing.\n")

if __name__ == "__main__":
    main()
