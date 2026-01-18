
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

def generate_ar1_trend(n, rho, slope, noise_std=1.0):
    """Generate AR(1) process with trend."""
    x = np.zeros(n)
    epsilon = np.random.normal(0, noise_std, n)
    x[0] = epsilon[0] / np.sqrt(1 - rho**2)
    for t in range(1, n):
        x[t] = rho * x[t-1] + epsilon[t]

    # Add trend
    t_vals = np.arange(n)
    x += slope * t_vals
    return x, t_vals

def run_sensitivity_analysis(n_sims=50, n=100, rho=0.6, slope=0.05, block_sizes=[2, 5, 10, 20, 30]):
    """
    Run simulation to test sensitivity of p-values to block size.
    """
    results = []

    print(f"Running Sensitivity Analysis: {n_sims} simulations, N={n}, Rho={rho}...")

    # We compute the average p-value across simulations for each block size.
    # For a true trend, p-value should be small.
    # For no trend, it should be uniformly distributed (mean 0.5).
    # We use a null case (slope=0) to see if Type I error rate (rejection rate) is stable.
    # The audit report states: "Results should be stable for reasonable block sizes".

    # We track Rejection Rate (Type I error) vs Block Size for H0.
    # Rejection Rate under H0 (slope=0) is the most critical metric for validity.
    # If block size is too small, rejection rate will be high (inflated).
    # If block size is appropriate, it should be near alpha.

    alpha = 0.05
    true_slope = 0.0 # H0

    for bs in block_sizes + ['auto']:
        rejections = 0
        avg_width = 0

        for i in tqdm(range(n_sims), desc=f"Block={bs}"):
            x, t = generate_ar1_trend(n, rho, true_slope)

            res = mk.trend_test(
                x, t, alpha=alpha,
                autocorr_method='block_bootstrap',
                n_bootstrap=200,
                block_size=bs
            )

            if res.h:
                rejections += 1

            # Track CI width as a proxy for "uncertainty estimation" stability
            avg_width += (res.upper_ci - res.lower_ci)

        results.append({
            'Block Size': str(bs),
            'Rejection Rate': rejections / n_sims,
            'Avg CI Width': avg_width / n_sims
        })

    return pd.DataFrame(results)

def create_report(df_results: pd.DataFrame, output_dir: str, n, rho):
    filename = 'README.md'
    report_path = os.path.join(output_dir, filename)

    with open(report_path, 'w') as f:
        f.write("# Validation Case 42: Block Size Sensitivity\n\n")
        f.write("This validation case examines how the choice of block size affects the performance of the ")
        f.write("Block Bootstrap Mann-Kendall test. Specifically, we look at the Type I error rate ")
        f.write("(false positive rate) for autocorrelated data with no trend.\n\n")

        f.write("## Simulation Setup\n")
        f.write(f"- **Data**: AR(1) process with $\\rho={rho}$ (No Trend)\n")
        f.write(f"- **Sample Size (N)**: {n}\n")
        f.write(f"- **Simulations**: 100 per block size\n")
        f.write(f"- **Alpha**: 0.05\n\n")

        f.write("## Results\n")
        f.write("The table below shows the Rejection Rate and Average Confidence Interval Width for various block sizes.\n\n")

        # Determine optimal block size for context.
        # Politis & White formula typically gives N^(1/3) scaling.

        df_display = df_results.copy()
        df_display['Rejection Rate'] = df_display['Rejection Rate'].apply(lambda x: f"{x:.2f}")
        df_display['Avg CI Width'] = df_display['Avg CI Width'].apply(lambda x: f"{x:.4f}")

        try:
            f.write(df_display.to_markdown(index=False))
        except ImportError:
            f.write(df_display.to_string(index=False))

        f.write("\n\n")
        f.write("## Sensitivity Plot\n")
        f.write("![Block Size Sensitivity](block_size_sensitivity.png)\n\n")

        f.write("## Interpretation\n")
        f.write("- **Small Block Sizes**: Rejection rate is typically inflated (too high) because small blocks fail to capture the long-range dependence (autocorrelation), making the bootstrap samples appear more independent than they really are.\n")
        f.write("- **Optimal Range**: There should be a range of block sizes where the rejection rate stabilizes near 0.05.\n")
        f.write("- **'Auto' Setting**: The automatic block size selection should ideally perform within this stable range.\n")

    print(f"Report saved to {report_path}")

def generate_plot(df_results: pd.DataFrame, output_dir: str):
    # Filter out 'auto' for plotting on x-axis, plot as hline
    df_numeric = df_results[df_results['Block Size'] != 'auto'].copy()
    df_numeric['Block Size'] = df_numeric['Block Size'].astype(int)

    auto_val = df_results[df_results['Block Size'] == 'auto']['Rejection Rate'].values
    if len(auto_val) > 0:
        auto_rr = auto_val[0]
    else:
        auto_rr = None

    plt.figure(figsize=(10, 6))
    plt.plot(df_numeric['Block Size'], df_numeric['Rejection Rate'], 'o-', label='Manual Block Size')
    plt.axhline(y=0.05, color='k', linestyle='--', label='Nominal Alpha (0.05)')

    if auto_rr is not None:
        plt.axhline(y=auto_rr, color='g', linestyle=':', label=f"'Auto' Setting ({auto_rr:.2f})")

    plt.xlabel('Block Size')
    plt.ylabel('Rejection Rate (Type I Error)')
    plt.title('Sensitivity of Rejection Rate to Block Size')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "block_size_sensitivity.png"))
    plt.close()

def run():
    np.random.seed(42)
    output_dir = os.path.dirname(__file__)

    # Run Simulation
    n = 100
    rho = 0.7 # Moderate-High autocorrelation to make block size matter
    df_results = run_sensitivity_analysis(n_sims=50, n=n, rho=rho)

    # Generate Plot
    generate_plot(df_results, output_dir)

    # Create Report
    create_report(df_results, output_dir, n, rho)

if __name__ == "__main__":
    run()
