
import os
import sys
import numpy as np
import pandas as pd
import MannKS as mk
import warnings

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Import the regional utilities
# We assume regional_validation_utils.py is in validation/
# so we can import it by adding validation/ to path or relative import
sys.path.insert(0, os.path.join(repo_root, 'validation'))
from regional_validation_utils import generate_site_data, run_python_regional_test, run_r_regional_test, create_markdown_report

def run():
    print("Running V-25: Regional Trend with High Inter-site Correlation...")

    # 1. Generate Highly Correlated Data
    # By using the same random noise for all sites, we ensure correlation = 1.0
    # Then we add a small jitter to make it not perfectly singular but very high.
    np.random.seed(42)

    n_years = 10
    dates = pd.date_range(start='2000-01-01', periods=n_years*12, freq='ME')
    t = np.arange(len(dates))
    base_noise = np.random.normal(0, 1.0, len(t))

    sites = ['SiteA', 'SiteB', 'SiteC', 'SiteD']
    dfs = []

    for site in sites:
        # High correlation: Shared base noise + very small unique noise
        unique_noise = np.random.normal(0, 0.1, len(t))
        slope = 0.05 # Weak increasing trend
        values = 10 + slope * t + base_noise + unique_noise

        df = pd.DataFrame({
            'site': site,
            'time': dates,
            'value': values
        })
        dfs.append(df)

    all_data = pd.concat(dfs, ignore_index=True)
    all_data = mk.prepare_censored_data(all_data['value']).join(all_data[['site', 'time']])

    # Run Python Test
    py_res, trend_res_df = run_python_regional_test(all_data)

    # Run R Test
    r_res = run_r_regional_test(all_data, trend_res_df)

    # 2. Generate Uncorrelated Data (Control Case)
    dfs_uncorr = []
    for site in sites:
        # Unique noise for each site
        slope = 0.05
        values = 10 + slope * t + np.random.normal(0, 1.0, len(t))
        df = pd.DataFrame({
            'site': site,
            'time': dates,
            'value': values
        })
        dfs_uncorr.append(df)

    all_data_uncorr = pd.concat(dfs_uncorr, ignore_index=True)
    all_data_uncorr = mk.prepare_censored_data(all_data_uncorr['value']).join(all_data_uncorr[['site', 'time']])

    py_res_uncorr, trend_res_df_uncorr = run_python_regional_test(all_data_uncorr)
    r_res_uncorr = run_r_regional_test(all_data_uncorr, trend_res_df_uncorr)

    scenarios = [
        {'name': 'High Correlation', 'py_res': py_res, 'r_res': r_res},
        {'name': 'Uncorrelated (Control)', 'py_res': py_res_uncorr, 'r_res': r_res_uncorr}
    ]

    report = create_markdown_report(
        "V-25: Regional Trend with High Inter-site Correlation",
        "This test verifies that the Regional Kendall Test correctly calculates the variance inflation factor when sites are highly correlated. \n\n"
        "In the 'High Correlation' scenario, sites share the same underlying noise pattern, which should result in a much higher `CorrectedVarTAU` compared to the uncorrected `VarTAU`. "
        "In the 'Uncorrelated' scenario, the corrected and uncorrected variances should be similar.",
        scenarios
    )

    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
        f.write(report)

    print("V-25 Complete. Report generated.")

if __name__ == "__main__":
    run()
