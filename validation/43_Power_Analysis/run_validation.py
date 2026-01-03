
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

def run_power_analysis(n_sims=500, n=50, rho=0.5, slopes=None, alpha=0.05):
    """
    Run simulation to estimate statistical power across different slope strengths.
    """
    if slopes is None:
        slopes = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    results = []

    print(f"Running Power Analysis: {n_sims} simulations, N={n}, Rho={rho}, Alpha={alpha}...")

    for slope in slopes:
        reject_standard = 0
        reject_bootstrap = 0

        for i in tqdm(range(n_sims), desc=f"Slope={slope:.2f}"):
            x, t = generate_ar1_trend(n, rho, slope)

            # Standard MK
            res_std = mk.trend_test(x, t, alpha=alpha, autocorr_method='none')
            if res_std.h:
                reject_standard += 1

            # Block Bootstrap MK
            res_boot = mk.trend_test(x, t, alpha=alpha, autocorr_method='block_bootstrap', n_bootstrap=200, block_size='auto')
            if res_boot.h:
                reject_bootstrap += 1

        results.append({
            'slope': slope,
            'power_standard': reject_standard / n_sims,
            'power_bootstrap': reject_bootstrap / n_sims
        })

    return pd.DataFrame(results)

def create_report(df_results: pd.DataFrame, output_dir: str, rho, n, alpha):
    filename = 'README.md'
    report_path = os.path.join(output_dir, filename)

    with open(report_path, 'w') as f:
        f.write("# Validation Case 43: Power Analysis\n\n")
        f.write("This validation case examines the statistical power (True Positive Rate) of the Block Bootstrap Mann-Kendall test. ")
        f.write("Power is the probability of correctly detecting a trend when one actually exists.\n\n")

        f.write("## Simulation Setup\n")
        f.write(f"- **Data Generation**: AR(1) process with $\\rho={rho}$ + Linear Trend\n")
        f.write(f"- **Sample Size (N)**: {n}\n")
        f.write(f"- **Simulations**: {50} per slope (reduced for demo)\n") # Hardcoded message matching the run() call below
        f.write(f"- **Alpha**: {alpha}\n\n")

        f.write("## Results\n")
        f.write("The table below shows the detection rate (Power) for varying trend slopes.\n\n")

        df_display = df_results.copy()
        df_display.columns = ['Slope', 'Standard Power', 'Bootstrap Power']
        df_display['Standard Power'] = df_display['Standard Power'].apply(lambda x: f"{x:.2f}")
        df_display['Bootstrap Power'] = df_display['Bootstrap Power'].apply(lambda x: f"{x:.2f}")

        try:
            f.write(df_display.to_markdown(index=False))
        except ImportError:
            f.write(df_display.to_string(index=False))

        f.write("\n\n")
        f.write("## Power Curve Plot\n")
        f.write("![Power Curve](power_curve.png)\n\n")

        f.write("## Interpretation\n")
        f.write("- **Corrected vs Uncorrected**: With autocorrelation ($\rho > 0$), the standard MK test has inflated Type I error (high rejection rate at slope=0). ")
        f.write("This makes its 'power' artificially high because it is biased towards rejection.\n")
        f.write("- **Bootstrap Performance**: The block bootstrap method should have a lower rejection rate at slope=0 (closer to alpha) ")
        f.write("but should still exhibit increasing power as the slope magnitude increases, demonstrating its ability to detect real trends ")
        f.write("while controlling for false positives.\n")
        f.write("- **Trade-off**: Correcting for autocorrelation inevitably reduces power compared to the (invalid) uncorrected test, ")
        f.write("but provides a trustworthy statistical inference.\n")

    print(f"Report saved to {report_path}")

def generate_plot(df_results: pd.DataFrame, output_dir: str, alpha):
    plt.figure(figsize=(10, 6))
    plt.plot(df_results['slope'], df_results['power_standard'], 'r-o', label='Standard MK')
    plt.plot(df_results['slope'], df_results['power_bootstrap'], 'b-s', label='Block Bootstrap MK')
    plt.axhline(y=alpha, color='k', linestyle='--', label=f'Nominal Alpha ({alpha})')

    plt.xlabel('True Trend Slope')
    plt.ylabel('Power (Detection Rate)')
    plt.title('Power Analysis: Detection Rate vs. Trend Strength')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "power_curve.png"))
    plt.close()

def run():
    np.random.seed(42)
    output_dir = os.path.dirname(__file__)

    # Run Simulation
    # Reduced n_sims=50 for speed in validation environment
    n = 50
    rho = 0.5
    alpha = 0.05
    slopes = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]

    df_results = run_power_analysis(n_sims=50, n=n, rho=rho, slopes=slopes, alpha=alpha)

    # Generate Plot
    generate_plot(df_results, output_dir, alpha)

    # Create Report
    create_report(df_results, output_dir, rho, n, alpha)

if __name__ == "__main__":
    run()
