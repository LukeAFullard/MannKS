
import os
import numpy as np
import pandas as pd
from MannKS.segmented_trend_test import segmented_trend_test
from MannKS.plotting import plot_segmented_trend

def run_ols():
    # -------------------------------------------------------------------------
    # 1. Generate Synthetic Data
    # -------------------------------------------------------------------------
    # We use the exact same seed and data generation parameters as the bagging
    # example to ensure a fair comparison.
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

    # Moderate Noise
    noise_std = 2.0
    x = trend + np.random.normal(0, noise_std, n)

    # -------------------------------------------------------------------------
    # 2. Run Analysis with Varying Alpha Levels
    # -------------------------------------------------------------------------
    alphas = [0.10, 0.05, 0.01]

    for alpha in alphas:
        confidence_pct = int((1-alpha)*100)
        print(f"\n{'='*60}")
        print(f"Standard OLS Analysis with Alpha = {alpha} ({confidence_pct}% Confidence)")
        print(f"{'='*60}")

        # Use Standard OLS (Ordinary Least Squares)
        # -----------------------------------------
        # This method uses the `piecewise-regression` library directly without bagging.
        # - It is much faster than bagging.
        # - It assumes normally distributed errors for calculating Confidence Intervals.
        # - Breakpoint CIs are often symmetric and may be narrower than reality
        #   in noisy or complex data (like this example).
        result = segmented_trend_test(
            x, t,
            n_breakpoints=2,
            alpha=alpha,
            use_bagging=False # Disable bagging for standard OLS
        )

        # Print Breakpoint details
        print("Breakpoint Results:")
        # We manually calculate these CIs based on the standard error and t-distribution
        # for the requested alpha (see `MannKS/_segmented.py`).
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
        print("\nSegment Results:")
        cols = ['slope', 'lower_ci', 'upper_ci']
        print(result.segments[cols].to_markdown(index=False, floatfmt=".4f"))

        # Visualize
        fname = f'segmented_plot_alpha_{alpha}_ols.png'
        save_path = os.path.join(os.path.dirname(__file__), fname)
        plot_segmented_trend(result, x, t, save_path=save_path)
        print(f"Plot saved to {fname}")

if __name__ == "__main__":
    run_ols()
