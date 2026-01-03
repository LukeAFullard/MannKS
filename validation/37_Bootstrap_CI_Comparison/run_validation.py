import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from MannKS import trend_test, seasonal_trend_test

def generate_linear_trend_data(n, slope, noise_sd, start_date='2000-01-01'):
    """Generates synthetic data with a linear trend and noise."""
    np.random.seed(42)
    t = pd.date_range(start=start_date, periods=n, freq='ME')
    noise = np.random.normal(0, noise_sd, n)
    trend = np.arange(n) * slope
    values = trend + noise
    return values, t

def generate_ar1_data(n, slope, noise_sd, phi, start_date='2000-01-01'):
    """Generates synthetic AR(1) data with a linear trend."""
    np.random.seed(int(slope*100 + noise_sd*10 + phi*1000))
    t = pd.date_range(start=start_date, periods=n, freq='ME')

    # Generate AR(1) noise
    noise = np.zeros(n)
    epsilon = np.random.normal(0, noise_sd, n)
    noise[0] = epsilon[0]
    for i in range(1, n):
        noise[i] = phi * noise[i-1] + epsilon[i]

    trend = np.arange(n) * slope
    values = trend + noise
    return values, t

def generate_seasonal_trend_data(n_years, slope, noise_sd, period=12, start_date='2000-01-01'):
    """Generates synthetic seasonal data."""
    n = n_years * period
    t = pd.date_range(start=start_date, periods=n, freq='ME')

    # Seasonal pattern
    seasonal_pattern = np.sin(np.linspace(0, 2 * np.pi, period, endpoint=False))
    seasonal = np.tile(seasonal_pattern, n_years)

    noise = np.random.normal(0, noise_sd, n)
    trend = np.arange(n) * slope
    values = trend + seasonal + noise
    return values, t

def run_comparison():
    # Base directory for outputs (where this script resides)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Non-Seasonal Comparison (Independent)
    print("Running Non-Seasonal Bootstrap Comparison (Independent)...")

    results = []

    # Parameters to sweep
    slopes = [0.0, 0.05, 0.1, 0.5]
    noises = [0.5, 1.0, 2.0]
    n_samples = 100

    plot_generated = False

    for slope in slopes:
        for noise in noises:
            np.random.seed(int(slope*100 + noise*10)) # unique seed per param set
            values, t = generate_linear_trend_data(n_samples, slope, noise)

            # Analytic Run
            res_analytic = trend_test(values, t, ci_method='direct', autocorr_method='none', slope_scaling='year')

            # Bootstrap Run
            res_boot = trend_test(values, t, autocorr_method='block_bootstrap', n_bootstrap=100, slope_scaling='year')

            results.append({
                'slope_param': slope,
                'noise_param': noise,
                'autocorr_phi': 0.0,
                'analytic_lower': res_analytic.lower_ci,
                'analytic_upper': res_analytic.upper_ci,
                'boot_lower': res_boot.lower_ci,
                'boot_upper': res_boot.upper_ci,
                'analytic_width': res_analytic.upper_ci - res_analytic.lower_ci,
                'boot_width': res_boot.upper_ci - res_boot.lower_ci
            })

            # Plot one example (e.g., slope=0.1, noise=1.0)
            if slope == 0.1 and noise == 1.0 and not plot_generated:
                plt.figure(figsize=(10, 6))
                plt.plot(t, values, 'o', alpha=0.5, label='Data')
                t_seconds = t.astype(np.int64) / 1e9
                plt.plot(t, res_analytic.intercept + res_analytic.slope_per_second * t_seconds,
                         'r-', label=f'Analytic Slope (yr): {res_analytic.scaled_slope:.2e}')

                plt.title(f'Comparison: Slope={slope}, Noise={noise}, Phi=0.0\nAnalytic CI: [{res_analytic.lower_ci:.2e}, {res_analytic.upper_ci:.2e}]\nBootstrap CI: [{res_boot.lower_ci:.2e}, {res_boot.upper_ci:.2e}]')
                plt.legend()
                output_path = os.path.join(base_dir, 'non_seasonal_example.png')
                plt.savefig(output_path)
                plt.close()
                plot_generated = True

    # 1b. Non-Seasonal Comparison (Autocorrelated AR(1))
    print("Running Non-Seasonal Bootstrap Comparison (AR(1))...")

    phis = [0.5, 0.8]
    plot_generated_ar = False

    for phi in phis:
        for slope in [0.1]: # Focus on one slope
             for noise in [1.0]: # Focus on one noise level
                values, t = generate_ar1_data(n_samples, slope, noise, phi)

                # Analytic Run
                res_analytic = trend_test(values, t, ci_method='direct', autocorr_method='none', slope_scaling='year')

                # Bootstrap Run
                res_boot = trend_test(values, t, autocorr_method='block_bootstrap', n_bootstrap=100, slope_scaling='year')

                results.append({
                    'slope_param': slope,
                    'noise_param': noise,
                    'autocorr_phi': phi,
                    'analytic_lower': res_analytic.lower_ci,
                    'analytic_upper': res_analytic.upper_ci,
                    'boot_lower': res_boot.lower_ci,
                    'boot_upper': res_boot.upper_ci,
                    'analytic_width': res_analytic.upper_ci - res_analytic.lower_ci,
                    'boot_width': res_boot.upper_ci - res_boot.lower_ci
                })

                if phi == 0.8 and not plot_generated_ar:
                    plt.figure(figsize=(10, 6))
                    plt.plot(t, values, 'o', alpha=0.5, label='Data (AR(1))')
                    t_seconds = t.astype(np.int64) / 1e9
                    plt.plot(t, res_analytic.intercept + res_analytic.slope_per_second * t_seconds,
                             'r-', label=f'Analytic Slope (yr): {res_analytic.scaled_slope:.2e}')

                    plt.title(f'Comparison: Slope={slope}, Noise={noise}, Phi={phi}\nAnalytic CI: [{res_analytic.lower_ci:.2e}, {res_analytic.upper_ci:.2e}]\nBootstrap CI: [{res_boot.lower_ci:.2e}, {res_boot.upper_ci:.2e}]')
                    plt.legend()
                    output_path = os.path.join(base_dir, 'non_seasonal_ar1_example.png')
                    plt.savefig(output_path)
                    plt.close()
                    plot_generated_ar = True

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(base_dir, 'non_seasonal_results.csv'), index=False)
    print("Non-Seasonal results saved.")

    # 2. Seasonal Comparison
    print("Running Seasonal Bootstrap Comparison...")
    results_seasonal = []

    n_years = 10
    plot_generated_seasonal = False

    for slope in slopes:
        for noise in noises:
            np.random.seed(int(slope*100 + noise*10 + 500))
            values, t = generate_seasonal_trend_data(n_years, slope, noise)

            # Analytic
            res_analytic = seasonal_trend_test(values, t, autocorr_method='none', slope_scaling='year')

            # Bootstrap
            res_boot = seasonal_trend_test(values, t, autocorr_method='block_bootstrap', n_bootstrap=100, slope_scaling='year')

            results_seasonal.append({
                'slope_param': slope,
                'noise_param': noise,
                'analytic_lower': res_analytic.lower_ci,
                'analytic_upper': res_analytic.upper_ci,
                'boot_lower': res_boot.lower_ci,
                'boot_upper': res_boot.upper_ci,
                'analytic_width': res_analytic.upper_ci - res_analytic.lower_ci,
                'boot_width': res_boot.upper_ci - res_boot.lower_ci
            })

            if slope == 0.1 and noise == 1.0 and not plot_generated_seasonal:
                plt.figure(figsize=(10, 6))
                plt.plot(t, values, 'o', alpha=0.5, label='Data')

                # Plot seasonal trend line
                t_seconds = t.astype(np.int64) / 1e9
                plt.plot(t, res_analytic.intercept + res_analytic.slope_per_second * t_seconds,
                         'r-', label=f'Analytic Slope (yr): {res_analytic.scaled_slope:.2e}')

                plt.title(f'Seasonal Comparison: Slope={slope}, Noise={noise}\nAnalytic CI: [{res_analytic.lower_ci:.2e}, {res_analytic.upper_ci:.2e}]\nBootstrap CI: [{res_boot.lower_ci:.2e}, {res_boot.upper_ci:.2e}]')
                plt.legend()
                output_path = os.path.join(base_dir, 'seasonal_example.png')
                plt.savefig(output_path)
                plt.close()
                plot_generated_seasonal = True

    df_seasonal = pd.DataFrame(results_seasonal)
    df_seasonal.to_csv(os.path.join(base_dir, 'seasonal_results.csv'), index=False)
    print("Seasonal results saved.")

    # Generate Markdown Report
    readme_content = f"""# Validation V-37: Bootstrap CI Comparison

This validation case compares the confidence intervals (CIs) generated by the standard analytic method (assuming independence) against those generated by the block bootstrap method (which accounts for autocorrelation).

**Objective:** Verify that bootstrap CIs are reasonable and consistent with analytic CIs for independent data, while providing a robust alternative.

## 1. Non-Seasonal Trend Test (Independent Data)

**Example Plot (Slope=0.1, Noise=1.0, Phi=0.0):**
![Non-Seasonal Example](non_seasonal_example.png)

## 2. Non-Seasonal Trend Test (AR(1) Autocorrelated Data)

**Example Plot (Slope=0.1, Noise=1.0, Phi=0.8):**
![Non-Seasonal AR(1) Example](non_seasonal_ar1_example.png)

**Results Summary (First 10 rows):**
{df_results.head(10).to_markdown()}

## 3. Seasonal Trend Test

**Example Plot (Slope=0.1, Noise=1.0):**
![Seasonal Example](seasonal_example.png)

**Results Summary (First 5 rows):**
{df_seasonal.head().to_markdown()}

## Conclusion
The comparison demonstrates the behavior of bootstrap vs analytic confidence intervals across various slope, noise, and autocorrelation conditions.
"""
    with open(os.path.join(base_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    print("README.md generated.")

if __name__ == "__main__":
    run_comparison()
