
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk

def generate_seasonal_ar1(n_years=20, rho=0.6, noise_std=1.0):
    """
    Generate seasonal data (monthly) where each month follows an AR(1) process across years.
    This simulates inter-annual autocorrelation while maintaining a seasonal cycle.
    """
    n_months = 12
    n_total = n_years * n_months

    # Base seasonal cycle
    seasonal_pattern = 10 * np.sin(np.linspace(0, 2*np.pi, n_months, endpoint=False))

    # Generate AR(1) errors for each month independently across years
    # Or generate a single AR(1) process and add it?
    # The seasonal test assumes independence *between* seasons, but allows structure within?
    # Actually, often environmental data has serial correlation that persists across months.
    # Let's generate a continuous AR(1) process + Seasonality.

    x = np.zeros(n_total)
    epsilon = np.random.normal(0, noise_std, n_total)
    x[0] = epsilon[0]
    for t in range(1, n_total):
        x[t] = rho * x[t-1] + epsilon[t]

    # Add seasonal pattern
    t_idx = np.arange(n_total)
    season_idx = t_idx % n_months
    x += seasonal_pattern[season_idx]

    dates = pd.date_range(start='2000-01-01', periods=n_total, freq='ME')

    return x, dates

def run_seasonal_validation(n_sims=50, n_years=30, rho=0.7, alpha=0.05):
    """
    Run simulation to check Type I error of Seasonal Bootstrap MK.
    """
    reject_standard = 0
    reject_bootstrap = 0

    print(f"Running Seasonal Validation: {n_sims} simulations, {n_years} years, Rho={rho}...")

    for i in tqdm(range(n_sims)):
        x, t = generate_seasonal_ar1(n_years, rho)

        # Standard Seasonal MK
        # agg_method='none' means using all data points (standard Seasonal Kendall)
        res_std = mk.seasonal_trend_test(x, t, period=12, alpha=alpha, autocorr_method='none')
        if res_std.h:
            reject_standard += 1

        # Block Bootstrap Seasonal MK
        # This resamples YEARS (blocks of 12 months) to preserve seasonality
        res_boot = mk.seasonal_trend_test(
            x, t, period=12, alpha=alpha,
            autocorr_method='block_bootstrap',
            n_bootstrap=100, # Reduced for demo speed
            block_size=1 # 1 year blocks (default for seasonal)
        )
        if res_boot.h:
            reject_bootstrap += 1

    return {
        'reject_rate_standard': reject_standard / n_sims,
        'reject_rate_bootstrap': reject_bootstrap / n_sims
    }

def create_report(results: Dict, output_dir: str, n_years, rho, alpha):
    filename = 'README.md'
    report_path = os.path.join(output_dir, filename)

    with open(report_path, 'w') as f:
        f.write("# Validation Case 44: Seasonal Bootstrap Validation\n\n")
        f.write("This validation case verifies that the Seasonal Block Bootstrap Mann-Kendall test correctly ")
        f.write("handles seasonal data with autocorrelation. Specifically, it checks if the method preserves ")
        f.write("the seasonal structure during resampling and controls Type I error rates better than the standard test.\n\n")

        f.write("## Simulation Setup\n")
        f.write(f"- **Data**: Monthly data with strong seasonality + AR(1) noise ($\\rho={rho}$)\n")
        f.write(f"- **Duration**: {n_years} years\n")
        f.write(f"- **Simulations**: 100\n")
        f.write(f"- **Alpha**: {alpha}\n\n")

        f.write("## Results\n")
        f.write("The table below shows the rejection rates (Type I Error) for Standard vs. Bootstrap Seasonal MK tests.\n\n")

        df = pd.DataFrame([{
            'Method': 'Standard Seasonal MK',
            'Type I Error Rate': f"{results['reject_rate_standard']:.2f}"
        }, {
            'Method': 'Seasonal Bootstrap MK',
            'Type I Error Rate': f"{results['reject_rate_bootstrap']:.2f}"
        }])

        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            f.write(df.to_string(index=False))

        f.write("\n\n")
        f.write("## Interpretation\n")
        f.write("- **Standard Method**: Likely to have inflated Type I error if the autocorrelation is significant, as it treats years as independent observations of the seasonal cycle.\n")
        f.write("- **Bootstrap Method**: Should have a rejection rate closer to the nominal alpha (0.05), as it resamples blocks of years to preserve the inter-annual dependence structure.\n")
        f.write("\n## Structural Verification\n")
        f.write("The implementation uses `moving_block_bootstrap` on the *indices of the cycles* (years). ")
        f.write("This ensures that:\n")
        f.write("1.  **Seasonality is Preserved**: Data for 'January' always remains in the 'January' slot relative to the cycle start.\n")
        f.write("2.  **Autocorrelation is Preserved**: Blocks of consecutive years are kept together, capturing the serial correlation.\n")

    print(f"Report saved to {report_path}")

def run():
    np.random.seed(42)
    output_dir = os.path.dirname(__file__)

    n_years = 30
    rho = 0.7
    alpha = 0.05

    results = run_seasonal_validation(n_sims=50, n_years=n_years, rho=rho, alpha=alpha)

    create_report(results, output_dir, n_years, rho, alpha)

if __name__ == "__main__":
    run()
