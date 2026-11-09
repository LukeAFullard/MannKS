
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
        random_state=42,
        n_bootstrap=200,
        block_size='auto'
    )

    print(f"Original N: {n}")
    print(f"Aggregated N (approx): {n/30:.1f}")
    print(f"Block Size Used (on aggregated data): {res.block_size_used}")
    print(f"P-value (Bootstrap): {res.p:.4f}")
    print(f"Trend: {res.trend}")

    return res, x, dates

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
        random_state=42,
        n_bootstrap=200
    )

    print(f"High Censor Rule Applied: {res.analysis_notes}")
    print(f"Block Size Used: {res.block_size_used}")
    print(f"P-value: {res.p:.4f}")

    return res, df_prep, t

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
        random_state=42,
        n_bootstrap=200,
        block_size='auto' # Should default to 1 year/cycle
    )

    print(f"Block Size Used (Cycles): {res.block_size_used}")
    print(f"P-value: {res.p:.4f}")
    print(f"Trend: {res.trend}")

    return res, x, dates

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
        random_state=42,
        n_bootstrap=200
    )

    print(f"Slope (ATS): {res.slope:.4f}")
    print(f"CI (ATS): [{res.lower_ci:.4f}, {res.upper_ci:.4f}]")
    print(f"P-value (Bootstrap): {res.p:.4f}")

    return res, df, t

def main():
    np.random.seed(42)  # Ensure reproducibility
    output_dir = os.path.dirname(__file__)
    readme_path = os.path.join(output_dir, 'README.md')

    # Capture output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        res1, x1, dates1 = run_scenario_1_aggregation()
        res2, df2, t2 = run_scenario_2_hicensor()
        res3, x3, dates3 = run_scenario_3_seasonal()
        res4, df4, t4 = run_scenario_4_ats()

    output_text = f.getvalue()
    print(output_text) # Print to console for verify

    # Generate 4-panel diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    try:
        # Scenario 1: Aggregation + Bootstrap
        ax1 = axes[0, 0]
        ax1.scatter(dates1, x1, alpha=0.3, s=10, color='gray', label='Daily Raw Data')
        # Monthly medians
        s1 = pd.Series(x1, index=dates1)
        m_med = s1.resample('ME').median()
        ax1.scatter(m_med.index, m_med.values, color='steelblue', s=35, zorder=3, label='Monthly Medians')
        ts1_sec = dates1.astype('datetime64[s]').astype('int64').to_numpy()
        y1_fit = res1.slope_per_second * ts1_sec + res1.intercept
        ax1.plot(dates1, y1_fit, color='navy', linewidth=2, label=f"Fit (p={res1.p:.4f})")
        ax1.set_title(f"1. Aggregation + Bootstrap (Block Size: {res1.block_size_used})")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Value")
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Scenario 2: High Censoring + Bootstrap
        ax2 = axes[0, 1]
        c_mask = df2['censored'].to_numpy(dtype=bool)
        v2 = df2['value'].to_numpy(dtype=float)
        ax2.scatter(t2[~c_mask], v2[~c_mask], color='steelblue', s=25, label='Observed')
        ax2.scatter(t2[c_mask], v2[c_mask], color='crimson', marker='v', s=35, label='Censored')
        ax2.axhline(5.0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Max Detection Limit (5.0)')
        ax2.plot(t2, res2.slope * t2 + res2.intercept, color='darkgreen', linewidth=2,
                 label=f"Fit (p={res2.p:.4f})")
        ax2.set_title(f"2. High Censoring + Bootstrap (Block Size: {res2.block_size_used})")
        ax2.set_xlabel("Index (t)")
        ax2.set_ylabel("Value")
        ax2.legend(loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Scenario 3: Seasonal Bootstrap
        ax3 = axes[1, 0]
        ax3.scatter(dates3, x3, color='purple', alpha=0.6, s=20, label='Monthly Data')
        ts3_sec = dates3.astype('datetime64[s]').astype('int64').to_numpy()
        y3_fit = res3.slope_per_second * ts3_sec + res3.intercept
        ax3.plot(dates3, y3_fit, color='darkmagenta', linewidth=2,
                 label=f"Seasonal Fit (p={res3.p:.4f})")
        ax3.set_title(f"3. Seasonal Bootstrap (Block Size: {res3.block_size_used} cycles)")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Value")
        ax3.legend(loc='upper left', fontsize=9)
        ax3.grid(True, alpha=0.3)

        # Scenario 4: ATS Slope + Bootstrap P-value
        ax4 = axes[1, 1]
        c4_mask = df4['censored'].to_numpy(dtype=bool)
        v4 = df4['value'].to_numpy(dtype=float)
        ax4.scatter(t4[~c4_mask], v4[~c4_mask], color='seagreen', s=25, label='Observed')
        ax4.scatter(t4[c4_mask], v4[c4_mask], color='crimson', marker='v', s=35, label='Censored (<2.0)')
        y4_fit = res4.slope * t4 + res4.intercept
        ax4.plot(t4, y4_fit, color='darkgreen', linewidth=2,
                 label=f"ATS Fit (slope={res4.slope:.4f})")
        ax4.plot(t4, res4.lower_ci * t4 + (res4.intercept - (res4.slope - res4.lower_ci) * np.median(t4)),
                 color='green', linestyle=':', label=f"95% CI [{res4.lower_ci:.4f}, {res4.upper_ci:.4f}]")
        ax4.plot(t4, res4.upper_ci * t4 + (res4.intercept - (res4.slope - res4.upper_ci) * np.median(t4)),
                 color='green', linestyle=':')
        ax4.set_title(f"4. ATS Slope + Bootstrap P-value (p={res4.p:.4f})")
        ax4.set_xlabel("Index (t)")
        ax4.set_ylabel("Value")
        ax4.legend(loc='upper left', fontsize=9)
        ax4.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, 'bootstrap_validation_scenarios.png')
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
    finally:
        plt.close(fig)

    # Write README
    with open(readme_path, 'w') as readme:
        readme.write("# Example 30: Block Bootstrap Validation\n\n")
        readme.write("This example validates the integration of the Block Bootstrap method with other package features.\n\n")

        readme.write("## Scenario 1: Aggregation + Bootstrap\n")
        readme.write("Bootstrap is applied *after* temporal aggregation. This is crucial for high-frequency autocorrelated data.\n")
        readme.write("```python\n")
        readme.write("mk.trend_test(..., agg_method='median', agg_period='month', autocorr_method='block_bootstrap')\n")
        readme.write("```\n\n")

        readme.write("## Scenario 2: High Censoring + Bootstrap\n")
        readme.write("Ensures `hicensor` preprocessing works correctly before the bootstrap resampling step.\n")
        readme.write("```python\n")
        readme.write("mk.trend_test(..., hicensor=True, autocorr_method='block_bootstrap')\n")
        readme.write("```\n\n")

        readme.write("## Scenario 3: Seasonal Bootstrap\n")
        readme.write("Validates that the seasonal test bootstraps entire seasonal cycles (e.g., years) to preserve seasonality.\n")
        readme.write("```python\n")
        readme.write("mk.seasonal_trend_test(..., autocorr_method='block_bootstrap', block_size='auto')\n")
        readme.write("```\n\n")

        readme.write("## Scenario 4: ATS Slope + Bootstrap P-value\n")
        readme.write("Combines the robust ATS estimator for the slope/CI with the Block Bootstrap for the p-value.\n")
        readme.write("```python\n")
        readme.write("mk.trend_test(..., sens_slope_method='ats', autocorr_method='block_bootstrap')\n")
        readme.write("```\n\n")

        readme.write("## Visual Diagnostics\n\n")
        readme.write("![Bootstrap Validation Scenarios](bootstrap_validation_scenarios.png)\n\n")

        readme.write("## Execution Output\n")
        readme.write("```text\n")
        readme.write(output_text)
        readme.write("```\n\n")

        readme.write("## Conclusion\n")
        readme.write("The successful execution of these scenarios confirms that the Block Bootstrap feature is robustly integrated into the package. ")
        readme.write("It correctly handles:\n")
        readme.write("- **Pre-processing steps** like temporal aggregation and high-censoring adjustments, applying the bootstrap to the processed data.\n")
        readme.write("- **Seasonal structures**, by bootstrapping entire cycles (e.g., years) to preserve seasonal dependence.\n")
        readme.write("- **Advanced estimators** like ATS, allowing users to combine robust slope estimation with robust significance testing.\n")

if __name__ == "__main__":
    main()
