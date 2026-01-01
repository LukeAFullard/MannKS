
import os
import sys
import numpy as np
import pandas as pd
from datetime import timedelta
import random

# Add parent directory to path to import local packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import MannKS as mk

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    HAS_RPY2 = True
except ImportError:
    HAS_RPY2 = False
    print("WARNING: rpy2 not found. Cannot run R script validation.")

def generate_censored_data(n=100, slope=0.1, noise_std=1.0, censor_prob=0.3, n_limits=2, seed=None):
    """
    Generates synthetic time series data with mixed censoring.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    t = pd.date_range(start='2000-01-01', periods=n, freq='ME')
    years = (t - t[0]).days / 365.25
    # Ensure y_raw is a numpy array to allow mutable operations
    y_raw = (slope * years + np.random.normal(0, noise_std, n)).to_numpy()

    # Create detection limits
    # We want roughly `censor_prob` of data to be censored.
    # We set limits such that the lowest values are censored.

    # Sort y_raw to find cutoffs
    sorted_y = np.sort(y_raw)

    limit_values = []

    # If we want 30% censored, and n_limits=2
    # We can split that 30% into chunks.
    # e.g. Limit 1 covers bottom 15%, Limit 2 covers next 15% (or overlaps).
    # Realistically, detection limits are fixed values.

    # Let's pick 'n_limits' random percentiles within the 0 to censor_prob range
    # to serve as detection limits.
    if censor_prob > 0:
        percentiles = np.linspace(0.05, censor_prob, n_limits)
        limits = np.percentile(y_raw, percentiles * 100)
    else:
        limits = []

    # Apply censoring
    # For each point, assign a random detection limit from the set (simulating changing labs/methods)
    # Then if value < limit, it is censored.

    # Alternatively, limits usually change over time (improve).
    # Let's simulate step changes in detection limits if n_limits > 1.
    # But "mixed censoring" usually implies we have multiple limits in the dataset.

    final_values = y_raw.copy()
    censored_flags = np.zeros(n, dtype=bool)

    if len(limits) > 0:
        # Assign limits to time periods roughly
        chunk_size = n // len(limits)

        limit_series = np.zeros(n)

        # Scenario: Detection limits decrease over time (common) or vary randomly.
        # Let's just vary randomly to test the robustness of "Mixed" handling.
        for i in range(n):
            lim = np.random.choice(limits)
            if y_raw[i] < lim:
                final_values[i] = lim # Report as < Limit
                censored_flags[i] = True
            limit_series[i] = lim

    # Create censored strings for Python (e.g. "<5")
    # For R, we pass RawValue and Censored bool

    # Python input preparation
    py_input = []
    for val, is_cen in zip(final_values, censored_flags):
        if is_cen:
            py_input.append(f"<{val:.4f}")
        else:
            py_input.append(val)

    return pd.DataFrame({
        't': t,
        'value': final_values,
        'censored': censored_flags,
        'cen_type': ['lt' if c else 'not' for c in censored_flags],
        'py_input': py_input # For passing to prepare_censored_data if needed, or mk functions handle it
    })

def run_r_script(data, r_script_path):
    if not HAS_RPY2:
        return None

    ro.r['source'](r_script_path)

    r_data = data.copy()
    r_data['myDate'] = r_data['t']
    r_data['RawValue'] = r_data['value']
    r_data['Censored'] = r_data['censored']
    r_data['CenType'] = r_data['cen_type']
    r_data['Year'] = r_data['t'].dt.year
    r_data['TimeIncr'] = r_data['t'].dt.strftime('%b')

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(r_data)

    try:
        ro.r('''
            run_analysis <- function(df) {
                suppressMessages(library(plyr))
                suppressMessages(library(lubridate))
                suppressMessages(library(NADA))

                df$myDate <- as.Date(df$myDate)
                df <- GetMoreDateInfo(df)

                df$TimeIncr <- df$Month
                df$Season <- df$Month

                # NonSeasonalTrendAnalysis
                # Using UseMidObs=FALSE for consistency with MannKS default aggregation (none/median)
                res <- NonSeasonalTrendAnalysis(df, do.plot=FALSE, ValuesToUse="RawValue", UseMidObs=FALSE)

                cat_label <- AssignConfCat(res, CatType="Direction")

                return(list(
                    slope = res$AnnualSenSlope,
                    p = res$p,
                    C = res$C,
                    Lci = res$Sen_Lci,
                    Uci = res$Sen_Uci,
                    TrendCategory = as.character(cat_label),
                    TrendDirection = as.character(res$TrendDirection)
                ))
            }
        ''')

        result_r = ro.r['run_analysis'](r_df)

        def get_scalar(idx, default=np.nan, type_func=float):
            try:
                if idx < len(result_r) and result_r[idx] is not ro.rinterface.NULL:
                    item = result_r[idx]
                    if hasattr(item, '__len__') and len(item) > 0:
                        return type_func(item[0])
                    return default
            except Exception:
                return default
            return default

        return {
            'slope': get_scalar(0),
            'p': get_scalar(1),
            'C': get_scalar(2),
            'lower_ci': get_scalar(3),
            'upper_ci': get_scalar(4),
            'category': get_scalar(5, "", str),
            'direction': get_scalar(6, "", str)
        }

    except Exception as e:
        print(f"R execution failed: {e}")
        return None

def run_v33():
    print("Running V-33: Censored Sensitivity Sweep (Non-Seasonal)")

    r_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r'))

    n = 60

    # Define pools of parameters to sample from
    slopes_pool = [0.0, 0.05, 0.1, -0.05, -0.1]
    noises_pool = [0.1, 0.5, 1.0, 2.0]
    censor_probs_pool = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    results = []

    # Run 99 tests
    for i in range(99):
        seed = 42 + i
        random.seed(seed)
        np.random.seed(seed)

        slope = random.choice(slopes_pool)
        noise = random.choice(noises_pool)
        c_prob = random.choice(censor_probs_pool)

        # Add some jitter to parameters so they aren't always exactly the same
        noise += random.uniform(-0.05, 0.05)
        if noise < 0.1: noise = 0.1

        data = generate_censored_data(n=n, slope=slope, noise_std=noise, censor_prob=c_prob, n_limits=2, seed=seed)

        processed_data = mk.prepare_censored_data(data['py_input'])

        mk_res = mk.trend_test(
            processed_data,
            data['t'],
            mk_test_method='lwp',
            sens_slope_method='lwp',
            ci_method='lwp',
            tau_method='b',
            slope_scaling='year',
            alpha=0.1,
            tie_break_method='lwp'
        )

        r_res = run_r_script(data, r_script_path)

        if r_res:
            py_cat = mk_res.classification.lower()
            r_cat_str = r_res['category']
            r_dir_str = r_res['direction']
            if r_cat_str.lower() == "as likely as not":
                    r_full = f"{r_cat_str} {r_dir_str}"
            else:
                r_full = f"{r_cat_str} {r_dir_str}"
            r_full = r_full.strip().lower()

            match_class = (py_cat == r_full)

            # Numerical comparisons
            slope_diff = abs(mk_res.scaled_slope - r_res['slope'])
            p_diff = abs(mk_res.p - r_res['p'])
            # CIs
            lci_diff = abs(mk_res.lower_ci - r_res['lower_ci'])
            uci_diff = abs(mk_res.upper_ci - r_res['upper_ci'])

            # Check for NaN matches
            if np.isnan(mk_res.scaled_slope) and np.isnan(r_res['slope']): slope_diff = 0.0
            if np.isnan(mk_res.p) and np.isnan(r_res['p']): p_diff = 0.0

            results.append({
                'iter': i+1,
                'slope_in': slope,
                'noise_in': noise,
                'censor_pct': c_prob,
                'slope_diff': slope_diff,
                'p_diff': p_diff,
                'lci_diff': lci_diff,
                'uci_diff': uci_diff,
                'match_class': match_class,
                'py_class': py_cat,
                'r_class': r_full
            })
        else:
            results.append({'iter': i+1, 'r_class': 'Failed'})

        print(f"Finished iteration {i+1}/99")

    df_res = pd.DataFrame(results)

    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
        f.write("# V-33: Censored Sensitivity Sweep (Non-Seasonal)\n\n")
        f.write("Verifies Sen's slope, p-value, CIs, and classification for non-seasonal data with mixed censoring.\n\n")

        if 'match_class' in df_res.columns:
            total = len(df_res)
            matches = df_res['match_class'].sum()
            f.write(f"Total Tests: {total}\n")
            f.write(f"Class Matches: {matches} ({matches/total*100:.1f}%)\n")
            f.write(f"Max Slope Diff: {df_res['slope_diff'].max():.6f}\n")
            f.write(f"Max P-value Diff: {df_res['p_diff'].max():.6f}\n")
            f.write(f"Max Lower CI Diff: {df_res['lci_diff'].max():.6f}\n")
            f.write(f"Max Upper CI Diff: {df_res['uci_diff'].max():.6f}\n\n")

            f.write("## Detailed Results (Top 20)\n")
            f.write(df_res.head(20).to_markdown(index=False))
        else:
            f.write("No successful comparisons.\n")

    df_res.to_csv(os.path.join(os.path.dirname(__file__), 'results.csv'), index=False)
    print("V-33 Completed.")

if __name__ == "__main__":
    run_v33()
