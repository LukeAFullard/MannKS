"""
Example 35: Large Dataset Trend Analysis
========================================

This example demonstrates the new capabilities in MannKS v0.5.0 for handling
large datasets (n > 5,000) using optimized algorithms.

We will generate synthetic datasets with known trends and noise, and then apply
Trend, Seasonal, and Segmented analysis to verify performance and accuracy.

Modes demonstrated:
1. 'fast' mode: Stochastic Sen's slope estimation (default for n > 5,000)
2. Stratified sampling for seasonal data.
"""

import numpy as np
import pandas as pd
from MannKS import trend_test, seasonal_trend_test, segmented_trend_test
import time

def generate_data(n, slope, noise_std, seasonal_amp=0):
    """Generates synthetic data with trend, seasonality, and noise."""
    t = np.arange(n)
    trend = slope * t
    season = seasonal_amp * np.sin(2 * np.pi * t / 12) # Period 12
    noise = np.random.normal(0, noise_std, n)
    x = trend + season + noise
    return t, x

def print_result(title, result, true_slope, start_time):
    elapsed = time.time() - start_time
    print(f"\n--- {title} ---")
    print(f"Time: {elapsed:.4f}s")
    if hasattr(result, 'computation_mode'):
        print(f"Mode: {result.computation_mode}")
    if hasattr(result, 'pairs_used') and result.pairs_used is not None:
        print(f"Pairs Used: {result.pairs_used}")

    # For segmented, result.slope is not directly available, need to iterate
    if hasattr(result, 'segments'):
        print("Segments found:", len(result.segments))
        for i, row in result.segments.iterrows():
            # Check for scaled columns
            if 'slope_per_second' in row and row['slope_per_second'] != row['slope']:
                 unit_str = result.segments['slope_units'].iloc[0] if 'slope_units' in result.segments else ""
                 print(f"  Segment {i+1}: Slope={row['slope']:.4f} {unit_str}, CI=({row['lower_ci']:.4f}, {row['upper_ci']:.4f})")
            else:
                 print(f"  Segment {i+1}: Slope={row['slope']:.4f}, CI=({row['lower_ci']:.4f}, {row['upper_ci']:.4f})")
    else:
        print(f"Estimated Slope: {result.slope:.6f} {result.slope_units}")
        print(f"True Slope:      {true_slope:.6f}")
        # Only calc error if true_slope is provided and non-zero
        if not np.isnan(true_slope) and true_slope != 0:
            print(f"Error:           {abs(result.slope - true_slope)/abs(true_slope):.2%}")
        print(f"Conf. Interval:  ({result.lower_ci:.6f}, {result.upper_ci:.6f})")
        print(f"Trend:           {result.trend}")
        if result.analysis_notes:
            print(f"Notes:           {result.analysis_notes}")

def main():
    np.random.seed(42)

    print("Generating Large Datasets (n=12,000)...")

    # 1. Linear Trend (Medium Noise)
    # n=12000, slope=0.5, noise=10
    t1, x1 = generate_data(12000, 0.5, 10)

    start = time.time()
    res1 = trend_test(x1, t1, large_dataset_mode='fast', max_pairs=100000, random_state=42)
    print_result("Linear Trend (Medium Noise)", res1, 0.5, start)

    # 2. Linear Trend (High Noise)
    # n=12000, slope=0.1, noise=50 (Signal-to-Noise ratio much lower)
    t2, x2 = generate_data(12000, 0.1, 50)

    start = time.time()
    res2 = trend_test(x2, t2, large_dataset_mode='fast', random_state=42)
    print_result("Linear Trend (High Noise)", res2, 0.1, start)

    # 3. Seasonal Trend
    # n=12000, slope=0.2 (per hour), season_amp=20, noise=5
    # Use hourly frequency
    t3, x3 = generate_data(12000, 0.2, 5, seasonal_amp=20)
    dates3 = pd.date_range(start='2000-01-01', periods=12000, freq='h')

    # Re-generate x3 with 24-hour seasonality to match 'hour' season_type
    season3 = 20 * np.sin(2 * np.pi * t3 / 24)
    x3 = 0.2 * t3 + season3 + np.random.normal(0, 5, 12000)

    start = time.time()
    # Use slope_scaling='hour' so the returned slope matches the input slope of 0.2 per hour
    res3 = seasonal_trend_test(
        x3, dates3,
        season_type='hour', # Uses dates.hour (0-23)
        large_dataset_mode='fast',
        max_per_season=200,
        slope_scaling='hour',
        random_state=42
    )
    print_result("Seasonal Trend (Stratified)", res3, 0.2, start)

    # 4. Segmented Trend
    # 6000 points slope 1.0, then 6000 points slope -0.5
    print("\nGenerating Segmented Data...")
    n_seg = 6000
    t_seg1 = np.arange(n_seg)
    x_seg1 = 1.0 * t_seg1 + np.random.normal(0, 5, n_seg)

    t_seg2 = np.arange(n_seg, 2*n_seg)
    x_seg2 = x_seg1[-1] - 0.5 * (t_seg2 - n_seg) + np.random.normal(0, 5, n_seg)

    t4 = np.concatenate([t_seg1, t_seg2])
    x4 = np.concatenate([x_seg1, x_seg2])

    start = time.time()
    res4 = segmented_trend_test(
        x4, t4,
        n_breakpoints=1,
        large_dataset_mode='fast',
        random_state=42
    )

    print_result("Segmented Trend", res4, np.nan, start)
    print("True Slopes: Segment 1 = 1.0, Segment 2 = -0.5")

if __name__ == "__main__":
    main()
