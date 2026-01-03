
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

def run_consistency_simulation(n_sims=500, n=100, n_bootstrap=500):
    """
    Run simulation to compare analytical vs bootstrap p-values on independent data.
    """
    results = []

    print(f"Running {n_sims} simulations with N={n} independent observations...")

    for i in tqdm(range(n_sims)):
        # Generate independent data (white noise)
        # Trend doesn't strictly matter for p-value consistency check under H0,
        # but let's use H0 (no trend) data.
        x = np.random.normal(0, 1, n)
        t = np.arange(n)

        # Analytical Method
        res_std = mk.trend_test(x, t, autocorr_method='none')
        p_std = res_std.p

        # Bootstrap Method
        # Use a reasonably high n_bootstrap to reduce bootstrap variance
        # block_size should be small for independent data (likely 1 or auto detects 1)
        res_boot = mk.trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=n_bootstrap, block_size='auto')
        p_boot = res_boot.p

        results.append({
            'p_analytical': p_std,
            'p_bootstrap': p_boot,
            'diff': p_boot - p_std,
            'abs_diff': abs(p_boot - p_std),
            'block_size': res_boot.block_size_used
        })

    return pd.DataFrame(results)

def create_report(df_results: pd.DataFrame, output_dir: str):
    filename = 'README.md'
    report_path = os.path.join(output_dir, filename)

    mean_diff = df_results['diff'].mean()
    mean_abs_diff = df_results['abs_diff'].mean()
    correlation = df_results['p_analytical'].corr(df_results['p_bootstrap'])

    # Calculate how often they agree on significance (alpha=0.05)
    alpha = 0.05
    sig_std = df_results['p_analytical'] < alpha
    sig_boot = df_results['p_bootstrap'] < alpha
    agreement = (sig_std == sig_boot).mean()

    with open(report_path, 'w') as f:
        f.write("# Validation Case 40: Independence Consistency Check\n\n")
        f.write("This validation case checks that the Block Bootstrap Mann-Kendall test produces p-values ")
        f.write("consistent with the standard analytical Mann-Kendall test when the data is truly independent ")
        f.write("(no autocorrelation).\n\n")

        f.write("## Simulation Setup\n")
        f.write(f"- **Data**: Independent Gaussian Noise (White Noise)\n")
        f.write(f"- **Sample Size (N)**: {len(df_results)}\n") # Simulation count matches length
        f.write(f"- **Bootstrap Resamples**: 500\n\n")

        f.write("## Results\n")
        f.write(f"- **Mean Difference (Bootstrap - Analytical)**: {mean_diff:.4f}\n")
        f.write(f"- **Mean Absolute Difference**: {mean_abs_diff:.4f}\n")
        f.write(f"- **Correlation between P-values**: {correlation:.4f}\n")
        f.write(f"- **Significance Agreement (alpha=0.05)**: {agreement*100:.1f}%\n\n")

        f.write("## P-Value Comparison Scatterplot\n")
        f.write("![P-Value Comparison](p_value_scatter.png)\n\n")

        f.write("## Interpretation\n")
        f.write("- **Ideal Outcome**: P-values should be strongly correlated and centered around the y=x line.\n")
        f.write("- **Deviation**: Small deviations are expected due to finite bootstrap resampling (Monte Carlo error). ")
        f.write("Large systematic biases would indicate an implementation error.\n")

        if mean_abs_diff < 0.05 and correlation > 0.9:
            f.write("\n**PASS**: The bootstrap method produces results strongly consistent with the analytical method for independent data.\n")
        else:
            f.write("\n**WARNING**: There may be significant divergence between the methods.\n")

    print(f"Report saved to {report_path}")

def generate_plot(df_results: pd.DataFrame, output_dir: str):
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['p_analytical'], df_results['p_bootstrap'], alpha=0.5, s=10)
    plt.plot([0, 1], [0, 1], 'r--', label='Ideal (y=x)')
    plt.xlabel('Analytical P-value')
    plt.ylabel('Bootstrap P-value')
    plt.title('Consistency Check: Analytical vs Bootstrap P-values\n(Independent Data)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "p_value_scatter.png"))
    plt.close()

def run():
    np.random.seed(42)
    output_dir = os.path.dirname(__file__)

    # Run Simulation
    # N=100 observations, 200 simulations
    df_results = run_consistency_simulation(n_sims=200, n=100, n_bootstrap=500)

    # Generate Plot
    generate_plot(df_results, output_dir)

    # Create Report
    create_report(df_results, output_dir)

if __name__ == "__main__":
    run()
