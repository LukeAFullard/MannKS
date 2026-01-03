"""
Validation Script for Rolling Trend Analysis (V-36)

This script performs a comprehensive validation of the rolling trend feature,
covering correctness, feature interaction, edge cases, and robustness.
It generates a Markdown report and diagnostic plots.
"""
import os
import sys
import numpy as np
import pandas as pd
import MannKS as mk
from scipy.stats import linregress
import warnings
import matplotlib.pyplot as plt
import io
import contextlib

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Output Directory
OUTPUT_DIR = os.path.dirname(__file__)

def capture_output(func):
    """Capture stdout from a function."""
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        func()
    return f.getvalue()

def run_validation():
    report_lines = []
    report_lines.append("# Validation Report: Rolling Trend Analysis (V-36)")
    report_lines.append("")

    # --- V-36a: Basic Correctness ---
    report_lines.append("## V-36a: Basic Correctness Check")
    n = 200
    t = pd.date_range('2000-01-01', periods=n, freq='D')
    x = np.arange(n) * 0.1 + np.random.normal(0, 1, n)

    rolling_res = mk.rolling_trend_test(x, t, window='50D', step='10D', min_size=10)

    manual_slopes = []
    t_series = pd.Series(t)
    for idx, row in rolling_res.iterrows():
        start, end = row['window_start'], row['window_end']
        mask = (t_series >= start) & (t_series < end)
        res = mk.trend_test(x[mask], t[mask])
        manual_slopes.append(res.slope)

    diffs = np.abs(rolling_res['slope'] - manual_slopes)
    max_diff = np.nanmax(diffs)

    if max_diff < 1e-10:
        report_lines.append(f"PASS: Rolling slopes match manual calculation perfectly (Max diff: {max_diff:.2e})")
    else:
        report_lines.append(f"FAIL: Rolling slopes diverge from manual calculation (Max diff: {max_diff:.2e})")
    report_lines.append("")

    # --- V-36b: Interaction with Aggregation ---
    report_lines.append("## V-36b: Interaction with Aggregation")
    t_agg = pd.date_range('2000-01-01', periods=24*365, freq='h')
    x_agg = np.arange(len(t_agg)) * 0.01 + np.random.normal(0, 5, len(t_agg))

    report_lines.append("Running rolling trend with daily aggregation on hourly data...")
    res_agg = mk.rolling_trend_test(
        x_agg, t_agg,
        window='90D', step='30D',
        agg_method='median', agg_period='day'
    )

    if len(res_agg) > 0 and not res_agg['slope'].isna().all():
        report_lines.append(f"PASS: Successfully computed rolling trends with aggregation (Generated {len(res_agg)} windows)")
    else:
        report_lines.append("FAIL: Failed to compute rolling trends with aggregation")
    report_lines.append("")

    # --- V-36c: Interaction with High Censor Rule ---
    report_lines.append("## V-36c: Interaction with High Censor Rule")
    t_cen = pd.date_range('2000-01-01', periods=100, freq='ME')
    x_raw = np.random.normal(10, 2, 100)

    x_mixed = x_raw.copy()
    data_list = []
    for i in range(100):
        val = x_mixed[i]
        if i < 50:
            if i % 10 == 0: data_list.append('<1')
            else: data_list.append(val)
        else:
            if i % 5 == 0: data_list.append('<15')
            else: data_list.append(val)

    df_cen = mk.prepare_censored_data(data_list)

    res_hicensor = mk.rolling_trend_test(
        df_cen, t_cen,
        window='400D', step='100D',
        hicensor=True,
        mk_test_method='lwp'
    )

    report_lines.append("PASS: High censor rule rolling test completed without error.")
    report_lines.append("")

    # --- V-36d: Edge Cases ---
    report_lines.append("## V-36d: Edge Cases (Small Windows)")
    x_short = np.arange(20)
    t_short = np.arange(20)

    res_empty = mk.rolling_trend_test(x_short, t_short, window=5, step=1, min_size=6)
    if res_empty.empty:
        report_lines.append("PASS: min_size filtering correctly returned empty DataFrame.")
    else:
        report_lines.append(f"FAIL: min_size filtering returned {len(res_empty)} rows (expected 0).")
    report_lines.append("")

    # --- V-36e: Outlier Robustness ---
    report_lines.append("## V-36e: Outlier Robustness")
    n_out = 100
    t_out = np.arange(n_out)
    x_out = 0.5 * t_out + np.random.normal(0, 1, n_out)
    x_out[50] = 1000 # Massive outlier

    mask = (t_out >= 40) & (t_out < 60)
    x_win = x_out[mask]
    t_win = t_out[mask]

    ols_res = linregress(t_win, x_win)
    ols_slope = ols_res.slope

    mk_res = mk.trend_test(x_win, t_win)
    sen_slope = mk_res.slope

    report_lines.append(f"Window with outlier (True slope ~0.5)")
    report_lines.append(f"- OLS Slope: {ols_slope:.4f}")
    report_lines.append(f"- Sen's Slope: {sen_slope:.4f}")

    if abs(sen_slope - 0.5) < abs(ols_slope - 0.5):
        report_lines.append("PASS: Sen's slope is more robust to outlier than OLS.")
    else:
        report_lines.append("FAIL: Sen's slope was not more robust.")

    # Generate Plot for V-36e
    plt.figure(figsize=(10, 6))
    plt.scatter(t_win, x_win, label='Data (with Outlier)')
    plt.plot(t_win, ols_res.intercept + ols_slope * t_win, 'r--', label=f'OLS (Slope={ols_slope:.2f})')
    # Sen intercept approx
    sen_int = np.median(x_win - sen_slope * t_win)
    plt.plot(t_win, sen_int + sen_slope * t_win, 'g-', label=f"Sen's (Slope={sen_slope:.2f})")
    plt.title('V-36e: Outlier Robustness Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    outlier_plot_path = os.path.join(OUTPUT_DIR, 'outlier_robustness.png')
    plt.savefig(outlier_plot_path)
    plt.close()
    report_lines.append(f"\n![Outlier Robustness](outlier_robustness.png)")
    report_lines.append("")

    # --- V-36f: Synthetic Reliability ---
    report_lines.append("## V-36f: Synthetic Reliability (Sine Wave)")
    n_sine = 200
    t_sine = np.arange(n_sine)
    x_sine = 10 * np.sin(t_sine / 10)

    window_sine = 20
    res_sine = mk.rolling_trend_test(x_sine, t_sine, window=window_sine, step=1)

    centers = res_sine['window_center']
    analytical_slopes = np.cos(centers / 10)

    correlation = np.corrcoef(res_sine['slope'], analytical_slopes)[0, 1]

    report_lines.append(f"Correlation between rolling Sen's slope and analytical derivative: {correlation:.4f}")

    if correlation > 0.95:
        report_lines.append("PASS: Rolling trend accurately tracks changing signal derivative.")
    else:
        report_lines.append("FAIL: Correlation too low.")

    # Generate Plot for V-36f
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(t_sine, x_sine, 'k-', alpha=0.3, label='Signal: 10*sin(t/10)')
    ax1.set_ylabel('Signal Value', color='black')

    ax2 = ax1.twinx()
    ax2.plot(centers, res_sine['slope'], 'b-', label="Rolling Sen's Slope")
    ax2.plot(centers, analytical_slopes, 'r--', label="Analytical Derivative: cos(t/10)")
    ax2.set_ylabel('Slope', color='blue')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title('V-36f: Tracking a Sine Wave Derivative')
    sine_plot_path = os.path.join(OUTPUT_DIR, 'sine_wave_tracking.png')
    plt.savefig(sine_plot_path)
    plt.close()
    report_lines.append(f"\n![Sine Wave Tracking](sine_wave_tracking.png)")
    report_lines.append("")

    # Write Report
    report_path = os.path.join(OUTPUT_DIR, 'README.md')
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))

    # Also print to stdout
    print("\n".join(report_lines))

if __name__ == "__main__":
    run_validation()
