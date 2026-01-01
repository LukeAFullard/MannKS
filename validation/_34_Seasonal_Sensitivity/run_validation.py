
import os
import sys
import numpy as np
import pandas as pd
import warnings
import random

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

def generate_seasonal_censored_data(n_years=5, slope=0.1, noise_std=1.0, censor_prob=0.3, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    n = n_years * 12
    t = pd.date_range(start='2000-01-01', periods=n, freq='ME')
    years = (t - t[0]).days / 365.25

    # Seasonal pattern: sine wave
    seasonality = np.sin(2 * np.pi * t.month / 12)
    # Ensure y_raw is a numpy array
    y_raw = (slope * years + seasonality + np.random.normal(0, noise_std, n)).to_numpy()

    # Censoring
    if censor_prob > 0:
        limits = np.percentile(y_raw, [5, 10, 20]) # Multiple limits
        # Filter limits < censor_prob percentile roughly
        limit = np.percentile(y_raw, censor_prob * 100)
        limits = limits[limits <= limit]
        if len(limits) == 0: limits = [limit]
    else:
        limits = []

    final_values = y_raw.copy()
    censored_flags = np.zeros(n, dtype=bool)

    for i in range(n):
        if len(limits) > 0:
            lim = np.random.choice(limits)
            if y_raw[i] < lim:
                final_values[i] = lim
                censored_flags[i] = True

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
        'py_input': py_input
    })

def run_r_script_seasonal(data, r_script_path):
    if not HAS_RPY2:
        return None

    ro.r['source'](r_script_path)

    r_data = data.copy()
    if 'py_input' in r_data.columns:
        r_data = r_data.drop(columns=['py_input'])

    r_data['myDate'] = r_data['t']
    r_data['RawValue'] = r_data['value']
    r_data['Censored'] = r_data['censored']
    r_data['CenType'] = r_data['cen_type']
    r_data['Year'] = r_data['t'].dt.year
    r_data['TimeIncr'] = r_data['t'].dt.strftime('%b') # Month name as increment
    r_data['Month'] = r_data['TimeIncr']

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.py2rpy(r_data)

    try:
        ro.r('''
            run_seasonal_analysis <- function(df) {
                suppressMessages(library(plyr))
                suppressMessages(library(lubridate))
                suppressMessages(library(NADA))

                df$myDate <- as.Date(df$myDate)
                df <- GetMoreDateInfo(df)

                # Setup for SeasonalTrendAnalysis
                # It requires Season column. Usually Season = Month for monthly data.
                df$TimeIncr <- df$Month
                df$Season <- df$Month

                # SeasonalTrendAnalysis
                # Using UseMidObs=FALSE to match
                res <- SeasonalTrendAnalysis(df, do.plot=FALSE, ValuesToUse="RawValue", UseMidObs=FALSE)

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

        result_r = ro.r['run_seasonal_analysis'](r_df)

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

def run_v34():
    print("Running V-34: Seasonal Sensitivity Sweep")

    r_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r'))

    # Define pools of parameters to sample from
    slopes_pool = [0.0, 0.05, 0.1, 0.15, 0.2]
    noises_pool = [0.1, 0.5, 1.0, 1.5]
    censor_probs_pool = [0.0, 0.1, 0.2, 0.3, 0.4]

    results = []

    # Run 99 tests
    for i in range(99):
        seed = 42 + i
        random.seed(seed)
        np.random.seed(seed)

        slope = random.choice(slopes_pool)
        noise = random.choice(noises_pool)
        c_prob = random.choice(censor_probs_pool)

        # Jitter noise slightly
        noise += random.uniform(-0.05, 0.05)
        if noise < 0.1: noise = 0.1

        data = generate_seasonal_censored_data(slope=slope, noise_std=noise, censor_prob=c_prob, seed=seed)

        processed_data = mk.prepare_censored_data(data['py_input'])

        # Run Seasonal Test
        # period=12 for monthly
        mk_res = mk.seasonal_trend_test(
            processed_data,
            data['t'],
            period=12,
            mk_test_method='lwp',
            sens_slope_method='lwp',
            ci_method='lwp',
            tau_method='b',
            slope_scaling='year',
            alpha=0.1
        )

        r_res = run_r_script_seasonal(data, r_script_path)

        if r_res:
            py_cat = mk_res.classification.lower()
            r_cat_str = r_res['category']
            r_dir_str = r_res['direction']

            # Construct R full classification string
            r_full = f"{r_cat_str} {r_dir_str}".strip().lower()

            # With _stats.py updated, "no trend" should now be "indeterminate"
            # So direct comparison should work for the low confidence cases.
            match_class = (py_cat == r_full)

            slope_diff = abs(mk_res.scaled_slope - r_res['slope'])
            p_diff = abs(mk_res.p - r_res['p'])
            lci_diff = abs(mk_res.lower_ci - r_res['lower_ci'])
            uci_diff = abs(mk_res.upper_ci - r_res['upper_ci'])

            if np.isnan(mk_res.scaled_slope) and np.isnan(r_res['slope']): slope_diff = 0.0
            if np.isnan(mk_res.p) and np.isnan(r_res['p']): p_diff = 0.0

            results.append({
                'iter': i+1,
                'slope_in': slope,
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

        if (i+1) % 10 == 0:
            print(f"Finished iteration {i+1}/99")

    df_res = pd.DataFrame(results)

    with open(os.path.join(os.path.dirname(__file__), 'README.md'), 'w') as f:
        f.write("# V-34: Seasonal Sensitivity Sweep\n\n")
        f.write("Verifies metrics for seasonal data with mixed censoring.\n\n")

        if 'match_class' in df_res.columns:
            total = len(df_res)
            matches = df_res['match_class'].sum()
            f.write(f"Total Tests: {total}\n")
            f.write(f"Class Matches: {matches} ({matches/total*100:.1f}%)\n")
            f.write(f"Max Slope Diff: {df_res['slope_diff'].max():.6f}\n")
            f.write(f"Max P-value Diff: {df_res['p_diff'].max():.6f}\n")
            f.write(f"Max Lower CI Diff: {df_res['lci_diff'].max():.6f}\n")
            f.write(f"Max Upper CI Diff: {df_res['uci_diff'].max():.6f}\n\n")
            f.write(df_res.head(20).to_markdown(index=False))
        else:
            f.write("No successful comparisons.\n")

    df_res.to_csv(os.path.join(os.path.dirname(__file__), 'results.csv'), index=False)
    print("V-34 Completed.")

if __name__ == "__main__":
    run_v34()
