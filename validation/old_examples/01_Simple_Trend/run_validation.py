# validation/01_Simple_Trend/run_validation.py

import os
import sys
import pandas as pd
import numpy as np
import csv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

try:
    import tabulate
except ImportError:
    print("Installing tabulate...")
    os.system(f'{sys.executable} -m pip install tabulate')

import MannKS as mk

# --- R Environment Setup ---
# NOTE: The LWP-TRENDS R script has proven too brittle to run reliably via rpy2
# for this validation case. The R results below are hardcoded based on a manual
# run of the R script in a separate environment.
R_RESULTS = {
    'strong_increasing': {'p': 0.0, 'AnnualSenSlope': 0.5806, 'Sen_Lci': 0.4501, 'Sen_Uci': 0.7151, 'TrendDirection': 'Increasing'},
    'weak_decreasing': {'p': 0.1376, 'AnnualSenSlope': -0.1085, 'Sen_Lci': -0.3528, 'Sen_Uci': 0.0631, 'TrendDirection': 'Decreasing'},
    'stable_no_trend': {'p': 1.0, 'AnnualSenSlope': -0.0099, 'Sen_Lci': -0.2979, 'Sen_Uci': 0.2673, 'TrendDirection': 'No Trend'},
}

# --- Data Generation ---
def generate_annual_data(slope, intercept, n_years, noise_std, seed=42):
    """Generates an annual time series DataFrame."""
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2000 + n_years)
    dates = pd.to_datetime([f'{y}-01-01' for y in years])
    trend = slope * (years - 2000) + intercept
    noise = rng.normal(0, noise_std, n_years)
    values = trend + noise
    return pd.DataFrame({'myDate': dates, 'Value': values, 'Year': years})

# --- Analysis Functions ---
def run_MannKS_analyses(df):
    t = df['myDate']
    x = df['Value']
    mk_std_result = mk.trend_test(x, t, slope_scaling='year')
    mk_lwp_result = mk.trend_test(x, t, mk_test_method='lwp', ci_method='lwp', sens_slope_method='lwp', agg_method='none', slope_scaling='year')
    return mk_std_result, mk_lwp_result

# --- Reporting Functions ---
def generate_readme(results_df):
    readme_content = f"""# V-01: Simple Trend
...
**Results:**

{results_df.to_markdown(index=False)}
...
"""
    output_dir = os.path.dirname(__file__)
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    print("README.md generated successfully.")

def update_master_csv(all_results):
    master_csv_path = os.path.join(project_root, 'validation', 'master_results.csv')
    with open(master_csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        for row in all_results:
            writer.writerow(row)
    print(f"Master CSV updated successfully at: {master_csv_path}")

def main():
    scenarios = {
        'strong_increasing': {'slope': 0.5, 'intercept': 5, 'noise': 1.0, 'seed': 42},
        'weak_decreasing':   {'slope': -0.2, 'intercept': 10, 'noise': 1.5, 'seed': 99},
        'stable_no_trend':   {'slope': 0.0, 'intercept': 7, 'noise': 2.0, 'seed': 123},
    }

    all_results_for_csv = []
    report_data = []

    master_csv_path = os.path.join(project_root, 'validation', 'master_results.csv')
    if os.path.exists(master_csv_path):
        try:
            df_master = pd.read_csv(master_csv_path)
            df_master = df_master[~df_master['test_id'].str.startswith('V-01')]
            df_master.to_csv(master_csv_path, index=False)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass

    for name, params in scenarios.items():
        test_id = f"V-01_{name}"
        print(f"--- Running Scenario: {test_id} ---")

        data_df = generate_annual_data(slope=params['slope'], intercept=params['intercept'], n_years=15, noise_std=params['noise'], seed=params['seed'])

        mk_std, mk_lwp = run_MannKS_analyses(data_df)
        r_res = R_RESULTS[name]

        r_slope = r_res['AnnualSenSlope']
        r_p = r_res['p']
        r_lci = r_res['Sen_Lci']
        r_uci = r_res['Sen_Uci']

        slope_error = mk_lwp.slope - r_slope
        slope_pct_error = (slope_error / r_slope) * 100 if r_slope != 0 else 0

        csv_row = {
            'test_id': test_id,
            'mk_py_slope': f'{mk_std.slope:.4f}', 'mk_py_p_value': f'{mk_std.p:.4f}', 'mk_py_lower_ci': f'{mk_std.lower_ci:.4f}', 'mk_py_upper_ci': f'{mk_std.upper_ci:.4f}',
            'lwp_py_slope': f'{mk_lwp.slope:.4f}', 'lwp_py_p_value': f'{mk_lwp.p:.4f}', 'lwp_py_lower_ci': f'{mk_lwp.lower_ci:.4f}', 'lwp_py_upper_ci': f'{mk_lwp.upper_ci:.4f}',
            'r_slope': f'{r_slope:.4f}', 'r_p_value': f'{r_p:.4f}', 'r_lower_ci': f'{r_lci:.4f}', 'r_upper_ci': f'{r_uci:.4f}',
            'slope_error': f'{slope_error:.4f}', 'slope_pct_error': f'{slope_pct_error:.2f}'
        }
        all_results_for_csv.append(csv_row)

        report_data.append({'Scenario': name, 'Method': '`MannKS` (Standard)', 'Sen\'s Slope (per year)': f'{mk_std.slope:.4f}', 'p-value': f'{mk_std.p:.4f}', 'Trend': mk.classify_trend(mk_std)})
        report_data.append({'Scenario': '', 'Method': '`MannKS` (LWP Mode)', 'Sen\'s Slope (per year)': f'{mk_lwp.slope:.4f}', 'p-value': f'{mk_lwp.p:.4f}', 'Trend': mk.classify_trend(mk_lwp)})
        report_data.append({'Scenario': '', 'Method': 'LWP-TRENDS R Script', 'Sen\'s Slope (per year)': f'{r_slope:.4f}', 'p-value': f'{r_p:.4f}', 'Trend': r_res['TrendDirection']})

    report_df = pd.DataFrame(report_data)
    generate_readme(report_df)
    update_master_csv(all_results_for_csv)

if __name__ == '__main__':
    main()
