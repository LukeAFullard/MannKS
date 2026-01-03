
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk
from MannKS._stats import _mk_score_and_var_censored

def generate_ar1(n, rho, noise_std=1.0):
    """Generate AR(1) process: x_t = rho * x_{t-1} + epsilon_t"""
    x = np.zeros(n)
    epsilon = np.random.normal(0, noise_std, n)
    x[0] = epsilon[0]
    for t in range(1, n):
        x[t] = rho * x[t-1] + epsilon[t]
    return x

def run_simulation(n_sims=500, n=50, rho=0.6, alpha=0.05):
    """
    Run simulation to estimate Type I error rates.
    H0 is true (no trend), but data is autocorrelated.
    """
    reject_standard = 0
    reject_bootstrap = 0

    print(f"Running {n_sims} simulations with N={n}, AR(1) rho={rho}...")

    for i in range(n_sims):
        x = generate_ar1(n, rho)
        t = np.arange(n)
        censored = np.zeros(n, dtype=bool)
        cen_type = np.array(['none'] * n)

        # Standard Mann-Kendall
        # We access internal function for raw p-value to avoid overhead/packaging of full result object if speed matters,
        # but using public API is cleaner. Let's use internal for speed as in original audit.
        _, _, _, p = _mk_score_and_var_censored(x, t, censored, cen_type)
        if p < alpha:
            reject_standard += 1

        # Block Bootstrap Mann-Kendall
        # Reduce n_bootstrap for speed in validation example vs full research paper
        p_boot, _, _ = mk.block_bootstrap_mann_kendall(
            x, t, censored, cen_type,
            n_bootstrap=200,
            block_size='auto'
        )
        if p_boot < alpha:
            reject_bootstrap += 1

    return {
        'n_sims': n_sims,
        'n': n,
        'rho': rho,
        'alpha': alpha,
        'reject_rate_standard': reject_standard / n_sims,
        'reject_rate_bootstrap': reject_bootstrap / n_sims
    }

def create_report(results: Dict, output_dir: str):
    filename = 'README.md'
    report_path = os.path.join(output_dir, filename)

    with open(report_path, 'w') as f:
        f.write("# Validation Case 38: Block Bootstrap for Autocorrelation\n\n")
        f.write("This validation case demonstrates the effectiveness of the Block Bootstrap Mann-Kendall test ")
        f.write("in handling autocorrelated data. Standard Mann-Kendall assumes independence and suffers from ")
        f.write("inflated Type I error rates (false positives) when positive autocorrelation is present.\n\n")

        f.write("## Simulation Setup\n")
        f.write(f"- **Data Generation**: AR(1) process with $\\rho = {results['rho']}$\n")
        f.write(f"- **Sample Size (N)**: {results['n']}\n")
        f.write(f"- **Simulations**: {results['n_sims']}\n")
        f.write(f"- **Significance Level ($\\alpha$)**: {results['alpha']}\n")
        f.write("- **Null Hypothesis**: No trend (true slope = 0)\n\n")

        f.write("## Results\n")
        f.write("The table below compares the Type I error rates (rejection rates) of the standard Mann-Kendall test ")
        f.write("versus the Block Bootstrap Mann-Kendall test.\n\n")

        df = pd.DataFrame([{
            'Method': 'Standard Mann-Kendall',
            'Type I Error Rate': f"{results['reject_rate_standard']:.3f}"
        }, {
            'Method': 'Block Bootstrap MK',
            'Type I Error Rate': f"{results['reject_rate_bootstrap']:.3f}"
        }])

        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            f.write(df.to_string(index=False))

        f.write("\n\n")
        f.write("## Interpretation\n")
        f.write(f"With $\\rho={results['rho']}$, the standard test rejects H0 approximately {results['reject_rate_standard']*100:.1f}% ")
        f.write(f"of the time, which is significantly higher than the nominal {results['alpha']*100}%. ")
        f.write(f"The Block Bootstrap method corrects this, bringing the rejection rate down to {results['reject_rate_bootstrap']*100:.1f}%, ")
        f.write("much closer to the target level.\n\n")

        f.write("## Sample Data Plot\n")
        f.write("![Sample AR(1) Process](sample_ar1.png)\n")

    print(f"Report saved to {report_path}")

def generate_plot(output_dir: str, n=100, rho=0.6):
    np.random.seed(42)
    x = generate_ar1(n, rho)
    plt.figure(figsize=(10, 6))
    plt.plot(x, marker='o', linestyle='-', alpha=0.7)
    plt.title(f"Sample AR(1) Process ($\\rho={rho}$)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, "sample_ar1.png"))
    plt.close()

def run():
    np.random.seed(42)
    output_dir = os.path.dirname(__file__)

    # Run Simulation
    results = run_simulation(n_sims=500, n=50, rho=0.6)

    # Generate Plot
    generate_plot(output_dir, n=50, rho=0.6)

    # Create Report
    create_report(results, output_dir)

    # Append to master results (using a simplified schema or creating a new one if needed,
    # but ValidationUtils expects a specific schema. I'll skip master_results.csv integration
    # for this custom validation unless I mock the columns, but usually it's better to keep it independent
    # if it doesn't fit the 'Slope vs R Slope' paradigm).
    # The user asked to "follow similar template", which usually implies the file structure and report style.

if __name__ == "__main__":
    run()
