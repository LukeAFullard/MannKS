import os
import numpy as np
import pandas as pd
import MannKS as mk
from MannKS.segmented_trend_test import find_best_segmentation, calculate_breakpoint_probability, segmented_trend_test
from MannKS import plot_segmented_trend
import matplotlib.pyplot as plt

# Configuration
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_FILE = os.path.join(OUTPUT_DIR, 'README.md')

def generate_data(n=100, pattern='linear', noise_std=0.5, censor_level=None):
    """
    Generates synthetic data for validation.
    """
    np.random.seed(42) # Fixed seed for reproducibility of specific dataset
    t = np.arange(n)

    if pattern == 'linear':
        x = 0.1 * t
    elif pattern == 'hinge':
        # Break at 50
        x = 0.1 * t
        x[50:] = x[50] - 0.2 * (t[50:] - 50) # Downward slope
    elif pattern == 'double_jump':
        # Flat, Up, Flat
        x = np.zeros(n)
        x[30:70] = np.linspace(0, 10, 40)
        x[70:] = 10
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Add random noise (using a random seed that varies if we wanted Monte Carlo, but here fixed)
    noise = np.random.normal(0, noise_std, n)
    x = x + noise

    # Censoring
    censored = np.zeros(n, dtype=bool)
    if censor_level is not None:
        censored = x < censor_level
        # Store as string for prepare_data
        x_str = x.astype(str)
        x_str[censored] = f"<{censor_level}"
        df = mk.prepare_censored_data(x_str)
    else:
        df = pd.DataFrame({'value': x, 'censored': False, 'cen_type': 'none'})

    df['t'] = t
    return df

def run_case_v01():
    """V-45-01: No Breakpoint (Type I Error Check)"""
    print("Running V-45-01: No Breakpoint Check...")
    df = generate_data(pattern='linear', noise_std=0.5)

    # Expect n=0 to be selected
    best_result, summary = find_best_segmentation(df, df['t'], max_breakpoints=2, n_bootstrap=20)

    success = (best_result.n_breakpoints == 0)

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, 'v45_01_plot.png')
    plot_segmented_trend(best_result, df['value'], df['t'], save_path=plot_path)

    return {
        'id': 'V-45-01',
        'desc': 'No Breakpoint (Linear Trend)',
        'success': success,
        'selected_n': best_result.n_breakpoints,
        'summary': summary.to_markdown(index=False),
        'plot': 'v45_01_plot.png'
    }

def run_case_v02():
    """V-45-02: Single Hinge Detection (Type II Error / Accuracy)"""
    print("Running V-45-02: Single Hinge Detection...")
    df = generate_data(pattern='hinge', noise_std=0.5)

    # Expect n=1 to be selected
    best_result, summary = find_best_segmentation(df, df['t'], max_breakpoints=2, n_bootstrap=20)

    n_ok = (best_result.n_breakpoints == 1)

    bp_ok = False
    bp_val = None
    if n_ok:
        bp_val = best_result.breakpoints[0]
        # True break at 50. Allow tolerance +/- 5 (10% of range)
        bp_ok = 45 <= bp_val <= 55

    success = n_ok and bp_ok

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, 'v45_02_plot.png')
    plot_segmented_trend(best_result, df['value'], df['t'], save_path=plot_path)

    return {
        'id': 'V-45-02',
        'desc': 'Single Hinge Detection (True BP=50)',
        'success': success,
        'selected_n': best_result.n_breakpoints,
        'detected_bp': bp_val,
        'summary': summary.to_markdown(index=False),
        'plot': 'v45_02_plot.png'
    }

def run_case_v03():
    """V-45-03: Censored Hinge (Robustness)"""
    print("Running V-45-03: Censored Hinge...")
    # Hinge data with censoring at tails (e.g. values < 1.0)
    # Hinge goes 0->5->-5.
    # Censored at start and end.
    df = generate_data(pattern='hinge', noise_std=0.5, censor_level=1.0)

    # Expect n=1 (or maybe 2 if censoring creates artifacts, but ideally 1)
    best_result, summary = find_best_segmentation(df, df['t'], max_breakpoints=2, n_bootstrap=20)

    # We accept n=1 OR n=2 (since we know censoring can create false regimes)
    # But for "Validation", we'd prefer robust n=1.
    # Let's see what happens.
    selected_n = best_result.n_breakpoints

    success = (selected_n >= 1)

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, 'v45_03_plot.png')
    plot_segmented_trend(best_result, df['value'], df['t'], save_path=plot_path)

    return {
        'id': 'V-45-03',
        'desc': 'Censored Hinge (Robustness)',
        'success': success,
        'selected_n': selected_n,
        'summary': summary.to_markdown(index=False),
        'plot': 'v45_03_plot.png'
    }

def run_case_v04():
    """V-45-04: Multiple Breakpoints"""
    print("Running V-45-04: Multiple Breakpoints...")
    df = generate_data(pattern='double_jump', noise_std=0.5)

    # Expect n=2
    best_result, summary = find_best_segmentation(df, df['t'], max_breakpoints=3, n_bootstrap=20)

    success = (best_result.n_breakpoints == 2)

    # Plot
    plot_path = os.path.join(OUTPUT_DIR, 'v45_04_plot.png')
    plot_segmented_trend(best_result, df['value'], df['t'], save_path=plot_path)

    return {
        'id': 'V-45-04',
        'desc': 'Double Jump (True BPs=30, 70)',
        'success': success,
        'selected_n': best_result.n_breakpoints,
        'summary': summary.to_markdown(index=False),
        'plot': 'v45_04_plot.png'
    }

def run_case_v05():
    """V-45-05: Breakpoint Probability Calibration"""
    print("Running V-45-05: Probability Calibration...")
    df = generate_data(pattern='hinge', noise_std=0.5)

    # Run with n=1 fixed
    result = segmented_trend_test(df, df['t'], n_breakpoints=1, use_bagging=True, n_bootstrap=100)

    # Calculate probability that break is in [45, 55] (True is 50)
    prob = calculate_breakpoint_probability(result, 45, 55)

    # Expect high probability
    success = prob > 0.8

    return {
        'id': 'V-45-05',
        'desc': 'Probability Calibration (True BP=50, Window=[45, 55])',
        'success': success,
        'metric': f"Probability: {prob:.2%}",
        'plot': None
    }

def write_report(results):
    with open(REPORT_FILE, 'w') as f:
        f.write("# Validation Report: Segmented Regression\n\n")

        # Summary Table
        f.write("| ID | Description | Success | Result |\n")
        f.write("|---|---|---|---|\n")
        for res in results:
            status = "PASS" if res['success'] else "FAIL"
            metric = str(res.get('selected_n', res.get('metric', '')))
            f.write(f"| {res['id']} | {res['desc']} | {status} | {metric} |\n")

        f.write("\n\n## Detailed Results\n")

        for res in results:
            f.write(f"\n### {res['id']}: {res['desc']}\n")
            f.write(f"**Success:** {res['success']}\n\n")

            if 'summary' in res:
                f.write("#### Model Selection Summary\n")
                f.write(res['summary'])
                f.write("\n\n")

            if 'detected_bp' in res:
                f.write(f"**Detected Breakpoint:** {res['detected_bp']}\n\n")

            if 'metric' in res:
                f.write(f"**Metric:** {res['metric']}\n\n")

            if res.get('plot'):
                f.write(f"![Plot]({res['plot']})\n")

def main():
    results = []
    try:
        results.append(run_case_v01())
        results.append(run_case_v02())
        results.append(run_case_v03())
        results.append(run_case_v04())
        results.append(run_case_v05())
    except Exception as e:
        print(f"Validation failed with error: {e}")
        import traceback
        traceback.print_exc()

    write_report(results)
    print(f"Validation complete. Report written to {REPORT_FILE}")

if __name__ == "__main__":
    main()
