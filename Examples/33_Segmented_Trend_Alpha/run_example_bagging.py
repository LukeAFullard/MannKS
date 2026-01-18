
import os
import numpy as np
import pandas as pd
from MannKS.segmented_trend_test import segmented_trend_test
from MannKS.plotting import plot_segmented_trend

def run_bagging():
    # -------------------------------------------------------------------------
    # 1. Generate Synthetic Data
    # -------------------------------------------------------------------------
    # We create a time series with 3 segments (2 breakpoints).
    # The slopes are relatively close (0.4, 0.1, 0.5) to make precise breakpoint
    # detection challenging, highlighting the utility of robust methods.
    np.random.seed(101)
    n = 150
    t = np.arange(n)

    # Define True Trend:
    # - Seg 1 (0-50):   Slope 0.4
    # - Seg 2 (50-100): Slope 0.1 (Flatter)
    # - Seg 3 (100-150): Slope 0.5 (Steeper)
    trend = np.concatenate([
        0.4 * t[:50],
        0.4 * 50 + 0.1 * (t[50:100] - 50),
        0.4 * 50 + 0.1 * 50 + 0.5 * (t[100:] - 100)
    ])

    # Add Moderate Noise (Gaussian)
    # With std=2.0, the signal-to-noise ratio is low enough to create
    # uncertainty about the exact breakpoint location.
    noise_std = 2.0
    x = trend + np.random.normal(0, noise_std, n)

    # -------------------------------------------------------------------------
    # 2. Run Analysis with Varying Alpha Levels
    # -------------------------------------------------------------------------
    # We test three significance levels:
    # - 0.10: 90% Confidence (Narrowest intervals)
    # - 0.05: 95% Confidence (Standard)
    # - 0.01: 99% Confidence (Widest intervals)
    alphas = [0.10, 0.05, 0.01]

    for alpha in alphas:
        confidence_pct = int((1-alpha)*100)
        print(f"\n{'='*60}")
        print(f"Bagging Analysis with Alpha = {alpha} ({confidence_pct}% Confidence)")
        print(f"{'='*60}")

        # Use Bagging (Bootstrap Aggregating)
        # -----------------------------------
        # Bagging improves robustness by resampling the data and re-fitting the
        # breakpoint model multiple times (n_bootstrap).
        # - The breakpoint location is estimated from the distribution of bootstrapped results.
        # - The Confidence Intervals (CIs) are non-parametric (percentile-based),
        #   often resulting in asymmetric intervals that better reflect reality.
        result = segmented_trend_test(
            x, t,
            n_breakpoints=2, # We fix the number of breakpoints for this example
            alpha=alpha,
            use_bagging=True,
            n_bootstrap=50, # Number of bootstrap iterations (higher is better for production)
            random_state=42 # Set seed for reproducibility
        )

        # Print Breakpoint details
        print("Breakpoint Results:")
        if result.n_breakpoints > 0:
            bp_df = pd.DataFrame({
                'Breakpoint': result.breakpoints,
                'Lower CI': [ci[0] for ci in result.breakpoint_cis],
                'Upper CI': [ci[1] for ci in result.breakpoint_cis]
            })
            print(bp_df.to_markdown(index=False, floatfmt=".2f"))
        else:
            print("No breakpoints found.")

        # Print Segment details
        # The slopes are calculated using the Sen's slope estimator on the
        # identified segments. Their CIs also respect the 'alpha' parameter.
        print("\nSegment Results:")
        cols = ['slope', 'lower_ci', 'upper_ci']
        print(result.segments[cols].to_markdown(index=False, floatfmt=".4f"))

        # Visualize
        # The plotting function automatically labels the CIs with the correct percentage.
        fname = f'segmented_plot_alpha_{alpha}_bagging.png'
        save_path = os.path.join(os.path.dirname(__file__), fname)
        plot_segmented_trend(result, x, t, save_path=save_path)
        print(f"Plot saved to {fname}")

if __name__ == "__main__":
    run_bagging()
