
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

def generate_ar1_trend(n, rho, slope, intercept=0, noise_std=1.0):
    """
    Generate AR(1) process with a trend:
    y_t = intercept + slope * t + x_t
    x_t = rho * x_{t-1} + epsilon_t
    """
    # AR(1) error term
    x = np.zeros(n)
    epsilon = np.random.normal(0, noise_std, n)
    x[0] = epsilon[0] / np.sqrt(1 - rho**2) # Stationary initialization
    for t in range(1, n):
        x[t] = rho * x[t-1] + epsilon[t]

    # Trend
    t_vals = np.arange(n)
    y = intercept + slope * t_vals + x
    return y, t_vals

def run_coverage_simulation(n_sims=500, n=50, rho_values=[0.0, 0.5, 0.8], alpha=0.05, true_slope=0.1):
    """
    Run simulation to estimate coverage probability of Confidence Intervals.
    """
    results = []

    for rho in rho_values:
        print(f"Running {n_sims} simulations with N={n}, AR(1) rho={rho}, slope={true_slope}...")

        covered_standard = 0
        covered_bootstrap = 0

        # We can also track width
        width_standard_sum = 0
        width_bootstrap_sum = 0

        for i in tqdm(range(n_sims), desc=f"Rho={rho}"):
            y, t = generate_ar1_trend(n, rho, true_slope)

            # Standard Method (ci_method='direct', autocorr_method='none')
            # Assuming standard behavior implies standard CIs
            res_std = mk.trend_test(y, t, alpha=alpha, autocorr_method='none')
            if res_std.lower_ci <= true_slope <= res_std.upper_ci:
                covered_standard += 1
            width_standard_sum += (res_std.upper_ci - res_std.lower_ci)

            # Bootstrap Method
            # Note: We need block_bootstrap CIs. Currently trend_test returns bootstrap CIs
            # if autocorr_method='block_bootstrap' and sens_slope_method != 'ats'
            res_boot = mk.trend_test(y, t, alpha=alpha, autocorr_method='block_bootstrap', n_bootstrap=200)
            if res_boot.lower_ci <= true_slope <= res_boot.upper_ci:
                covered_bootstrap += 1
            width_bootstrap_sum += (res_boot.upper_ci - res_boot.lower_ci)

        results.append({
            'rho': rho,
            'coverage_standard': covered_standard / n_sims,
            'coverage_bootstrap': covered_bootstrap / n_sims,
            'avg_width_standard': width_standard_sum / n_sims,
            'avg_width_bootstrap': width_bootstrap_sum / n_sims
        })

    return results

def create_report(results: List[Dict], n_sims, n, alpha, true_slope, output_dir: str):
    filename = 'README.md'
    report_path = os.path.join(output_dir, filename)

    with open(report_path, 'w') as f:
        f.write("# Validation Case 39: Coverage Probability of Confidence Intervals\n\n")
        f.write("This validation case tests the coverage probability of the confidence intervals generated ")
        f.write("by the standard method versus the block bootstrap method under varying levels of autocorrelation.\n\n")

        f.write("## Simulation Setup\n")
        f.write(f"- **Data Generation**: AR(1) errors + Linear Trend\n")
        f.write(f"- **True Slope**: {true_slope}\n")
        f.write(f"- **Sample Size (N)**: {n}\n")
        f.write(f"- **Simulations**: {n_sims}\n")
        f.write(f"- **Target Confidence Level**: {1 - alpha:.2f} ($\\alpha={alpha}$)\n\n")

        f.write("## Results\n")
        f.write("The table below shows the proportion of simulations where the calculated confidence interval ")
        f.write("contained the true slope value (Coverage Probability).\n\n")

        df = pd.DataFrame(results)
        df_display = df.copy()
        df_display.columns = ['Rho (Autocorrelation)', 'Standard Coverage', 'Bootstrap Coverage', 'Avg Width (Std)', 'Avg Width (Boot)']

        # Format as percentage
        df_display['Standard Coverage'] = df_display['Standard Coverage'].apply(lambda x: f"{x*100:.1f}%")
        df_display['Bootstrap Coverage'] = df_display['Bootstrap Coverage'].apply(lambda x: f"{x*100:.1f}%")
        df_display['Avg Width (Std)'] = df_display['Avg Width (Std)'].apply(lambda x: f"{x:.4f}")
        df_display['Avg Width (Boot)'] = df_display['Avg Width (Boot)'].apply(lambda x: f"{x:.4f}")

        try:
            # Requires tabulate
            f.write(df_display.to_markdown(index=False))
        except ImportError:
            f.write(df_display.to_string(index=False))

        f.write("\n\n")
        f.write("## Interpretation\n")
        f.write("- **Ideal Outcome**: Coverage should be close to 95%.\n")
        f.write("- **Standard Method**: With high autocorrelation (rho > 0), the standard method typically produces intervals that are too narrow, leading to coverage significantly below 95%.\n")
        f.write("- **Bootstrap Method**: The block bootstrap method should account for the autocorrelation, producing wider intervals that maintain coverage closer to the nominal 95% level.\n")

    print(f"Report saved to {report_path}")

def run():
    np.random.seed(42)
    output_dir = os.path.dirname(__file__)

    n_sims = 100  # Reduced for faster execution in CI/Sandbox
    n = 50
    alpha = 0.05
    true_slope = 0.1
    rho_values = [0.0, 0.5, 0.8]

    # Run Simulation
    results = run_coverage_simulation(n_sims=n_sims, n=n, rho_values=rho_values, alpha=alpha, true_slope=true_slope)

    # Create Report
    create_report(results, n_sims, n, alpha, true_slope, output_dir)

if __name__ == "__main__":
    run()
