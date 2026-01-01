
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

sys.path.insert(0, os.path.join(repo_root, 'validation'))
from regional_validation_utils import generate_site_data, run_python_regional_test, run_r_regional_test, create_markdown_report

def run():
    print("Running V-26: Regional Trend with Insufficient Site Data...")

    np.random.seed(42)

    # 1. Create a dataset where one site has very few data points (e.g., n=3)
    # The standard MK test usually requires n >= 4 to return a valid result/p-value,
    # or the package might return NaN/None if configured with min_size.

    # Valid Site A
    df_a = generate_site_data('SiteA', n_years=10, trend_type='increasing')

    # Valid Site B
    df_b = generate_site_data('SiteB', n_years=10, trend_type='increasing')

    # Invalid Site C (Insufficient Data)
    # Only 3 points
    dates_c = pd.to_datetime(['2000-01-01', '2000-02-01', '2000-03-01'])
    values_c = [10.0, 10.1, 10.2]
    df_c = pd.DataFrame({'site': 'SiteC', 'time': dates_c, 'value': values_c})
    df_c = mk.prepare_censored_data(df_c['value']).join(df_c[['site', 'time']])

    all_data = pd.concat([df_a, df_b, df_c], ignore_index=True)

    # Run Python Regional Test
    # This involves manually running trend_test for each site first.
    # We expect SiteC to produce a result that might be filtered out or handled.

    site_results = []
    for site in all_data['site'].unique():
        site_df = all_data[all_data['site'] == site].copy()

        # We explicitly set min_size to 4 to FORCE SiteC to fail validation
        # Wait, the current implementation of trend_test might default min_size=5 or similar.
        # Let's check trend_test's behavior. If it returns valid result for n=3 (S=3, var=...),
        # we might need fewer points or rely on the fact that p-value might be nan or 1.0.
        # Ideally, we want a case where the result is explicitly 'insufficient data' like.

        # Actually, let's look at `mk.trend_test`. It calculates S even for small n.
        # But `regional_test` requires 's' and 'C'.

        result = mk.trend_test(site_df['value'], site_df['time'])

        # If n=3, S=3 (10, 10.1, 10.2). Var = n(n-1)(2n+5)/18 = 3*2*11/18 = 3.66.
        # So it WILL return a result.

        # To simulate "Insufficient Data" in the context of LWP/Regional,
        # usually we mean sites that didn't meet the data sufficiency criteria *before* the test.
        # The `regional_test` function documentation says it checks for 'insufficient data'.
        # Let's see if we can create a result that has NaNs.

        # If we pass an empty dataframe or all NaNs, trend_test returns a dummy result with 0s or NaNs.
        # Let's try making SiteC all NaNs.

        if site == 'SiteC':
             # Force a failure result simulation if the test was skipped
             # or run on empty data
             res_c = mk.trend_test(pd.Series([], dtype=float), pd.Series([], dtype=float))
             site_results.append({
                'site': site,
                's': res_c.s, # Should be 0 or nan
                'C': res_c.C, # Should be nan or 0.5
                'p': res_c.p,
                'slope': res_c.slope
             })
        else:
            result = mk.trend_test(site_df['value'], site_df['time'])
            site_results.append({
                'site': site,
                's': result.s,
                'C': result.C,
                'p': result.p,
                'slope': result.slope
            })

    trend_res_df = pd.DataFrame(site_results)

    # Run Regional Test
    # We expect SiteC to be excluded because it has no valid S/C or insufficient data
    # The regional_test function should handle this gracefully.
    py_res = mk.regional_test(
        trend_res_df,
        all_data,
        site_col='site',
        value_col='value',
        time_col='time',
        s_col='s',
        c_col='C'
    )

    # Run R Test
    # The R script's getTAU is robust but expects valid inputs.
    # If we pass it a row with NA/NaN for S/C, we need to see if it handles it.
    # Or we verify if our Python impl matches R's handling of "bad" sites.
    # We will pass the same trend_res_df to R.
    r_res = run_r_regional_test(all_data, trend_res_df)

    scenarios = [
        {'name': 'Mixed Data Sufficiency', 'py_res': py_res, 'r_res': r_res}
    ]

    report = create_markdown_report(
        "V-26: Regional Trend with Insufficient Site Data",
        "This test checks the robustness of the regional test when one site (SiteC) provides invalid or empty results (simulated as insufficient data). \n\n"
        "The expected behavior is that the invalid site is excluded from the calculation, meaning the number of sites (M) should be 2, not 3. "
        "The regional statistics (TAU, etc.) should be calculated based only on the valid sites (A and B).",
        scenarios
    )

    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
        f.write(report)

    print("V-26 Complete. Report generated.")

if __name__ == "__main__":
    run()
