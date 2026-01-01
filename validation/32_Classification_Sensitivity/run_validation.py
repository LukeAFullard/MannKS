
import os
import sys
import numpy as np
import pandas as pd
import warnings

# Add parent directory to path to import local packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import MannKS as mk

# Add validation directory to path to import validation utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Try to import rpy2 for R execution
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("WARNING: rpy2 not found. Cannot run R script validation.")

def generate_synthetic_data(n=100, slope=0.1, noise_std=1.0, seed=None):
    """Generates synthetic time series data."""
    if seed is not None:
        np.random.seed(seed)

    t = pd.date_range(start='2000-01-01', periods=n, freq='ME')
    # Linear trend + noise
    # Calculate time in years for slope application
    years = (t - t[0]).days / 365.25
    y = slope * years + np.random.normal(0, noise_std, n)

    return pd.DataFrame({'t': t, 'value': y})

def run_r_script(data, r_script_path):
    """Runs the LWP R script on the provided data."""
    if not HAS_RPY2:
        return None

    # Source the R script
    ro.r['source'](r_script_path)

    # Prepare data for R
    r_data = data.copy()
    r_data['myDate'] = r_data['t']
    r_data['RawValue'] = r_data['value']
    r_data['Censored'] = False
    r_data['CenType'] = 'not'
    r_data['Year'] = r_data['t'].dt.year
    r_data['TimeIncr'] = r_data['t'].dt.strftime('%b') # Month name

    # Convert to R DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(r_data)

    # Call the R function
    try:
        # Define a wrapper in R to handle the call and return a list/vector
        ro.r('''
            run_analysis <- function(df) {
                suppressMessages(library(plyr))
                suppressMessages(library(lubridate))

                # Ensure myDate is Date class
                df$myDate <- as.Date(df$myDate)

                # Initialize LWP environment variables and columns
                df <- GetMoreDateInfo(df)

                # Set TimeIncr and Season for NonSeasonalTrendAnalysis
                df$TimeIncr <- df$Month
                df$Season <- df$Month

                # Run Analysis with UseMidObs=FALSE to avoid MidTimeIncrList dependency issue
                res <- NonSeasonalTrendAnalysis(df, do.plot=FALSE, ValuesToUse="RawValue", UseMidObs=FALSE)

                # Re-calculate TrendCategory because NonSeasonalTrendAnalysis deletes it
                # We can use AssignConfCat directly on the result 'res'
                cat_label <- AssignConfCat(res, CatType="Direction")

                # Extract relevant fields
                return(list(
                    slope = res$AnnualSenSlope,
                    p = res$p,
                    C = res$C,
                    TrendCategory = as.character(cat_label),
                    TrendDirection = as.character(res$TrendDirection)
                ))
            }
        ''')

        result_r = ro.r['run_analysis'](r_df)

        # Helper to safely extract scalar from R vector/list item
        def get_scalar(idx, default=np.nan, type_func=float):
            try:
                if idx < len(result_r) and result_r[idx] is not ro.rinterface.NULL:
                    # R vector access is 0-based in Python rpy2 wrapper of list
                    item = result_r[idx]
                    if hasattr(item, '__len__') and len(item) > 0:
                        return type_func(item[0])
                    return default
            except Exception:
                return default
            return default

        slope = get_scalar(0, np.nan, float)
        p = get_scalar(1, np.nan, float)
        C = get_scalar(2, np.nan, float)
        category = get_scalar(3, "", str)
        direction = get_scalar(4, "", str)

        return {
            'slope': slope,
            'p': p,
            'C': C,
            'category': category,
            'direction': direction
        }

    except Exception as e:
        print(f"R execution failed: {e}")
        return None

def run_v32():
    print("Running V-32: Classification Sensitivity Sweep")

    r_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r'))

    # Parameters for the sweep
    # Fixed n for consistency
    n = 60 # 5 years of monthly data

    # Slopes to test (positive and negative)
    slopes = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, -0.02, -0.05, -0.1, -0.2, -0.5]

    # Noise levels to modulate significance
    noise_levels = [0.5, 1.0, 2.0]

    results = []

    # Generate scenarios
    for slope in slopes:
        for noise in noise_levels:
            # Run multiple seeds to get variation
            for i in range(3):
                seed = int(abs(slope * 1000) + (noise * 100) + i)
                data = generate_synthetic_data(n=n, slope=slope, noise_std=noise, seed=seed)

                # 1. Run MannKS (LWP Mode)
                # Use standard aggregation (default 'none') since data is unique.
                # Use LWP statistics/methods.
                mk_res = mk.trend_test(
                    data['value'],
                    data['t'],
                    mk_test_method='lwp',
                    sens_slope_method='lwp',
                    ci_method='lwp',
                    tau_method='b',
                    slope_scaling='year'
                )

                # 2. Run R Script
                r_res = run_r_script(data, r_script_path)

                # 3. Compare
                if r_res:
                    py_cat = mk_res.classification.lower()

                    r_cat_str = r_res['category']
                    r_dir_str = r_res['direction']

                    # Logic to construct comparable string
                    # If R says "As likely as not", sometimes it omits direction or includes it.
                    # MannKS generally includes direction e.g. "As Likely as Not Increasing"

                    if r_cat_str.lower() == "as likely as not":
                         r_full = f"{r_cat_str} {r_dir_str}"
                    else:
                        # e.g. Likely Increasing
                        r_full = f"{r_cat_str} {r_dir_str}"

                    r_full = r_full.strip().lower()

                    match = (py_cat == r_full)

                    # Compare C values
                    if pd.isna(mk_res.C) or pd.isna(r_res['C']):
                         c_match = False
                    else:
                        c_diff = abs(mk_res.C - r_res['C'])
                        c_match = c_diff < 1e-4

                    results.append({
                        'slope_in': slope,
                        'noise_in': noise,
                        'py_p': mk_res.p,
                        'r_p': r_res['p'],
                        'py_C': mk_res.C,
                        'r_C': r_res['C'],
                        'py_class': mk_res.classification,
                        'r_class': r_full.title(),
                        'match_class': match,
                        'match_C': c_match
                    })
                else:
                    results.append({
                        'slope_in': slope,
                        'noise_in': noise,
                        'py_p': mk_res.p,
                        'r_p': np.nan,
                        'py_C': mk_res.C,
                        'r_C': np.nan,
                        'py_class': mk_res.classification,
                        'r_class': "R Failed",
                        'match_class': False,
                        'match_C': False
                    })

    # Save Results
    df_res = pd.DataFrame(results)

    # Generate README
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
        f.write("# V-32: Classification Sensitivity Sweep\n\n")
        f.write("This validation runs a parameter sweep of slope and noise to verify that `MannKS` classification logic matches the LWP R script across the full range of confidence levels.\n\n")

        f.write("## Summary\n\n")
        total = len(df_res)
        matches = df_res['match_class'].sum()
        pct = (matches/total*100) if total > 0 else 0
        f.write(f"Total Tests: {total}\n")
        f.write(f"Matches: {matches} ({pct:.1f}%)\n\n")

        f.write("## Mismatches\n\n")
        mismatches = df_res[~df_res['match_class']]
        if len(mismatches) > 0:
            f.write(mismatches.to_markdown(index=False))
        else:
            f.write("No mismatches found.\n")

        f.write("\n## Full Results (Top 20)\n\n")
        f.write(df_res.head(20).to_markdown(index=False))

    # Save CSV
    df_res.to_csv(os.path.join(os.path.dirname(__file__), 'results.csv'), index=False)
    print("V-32 Completed.")

if __name__ == "__main__":
    run_v32()
