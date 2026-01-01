
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import contextlib
import io
import warnings

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk

# RPy2 imports
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
except ImportError:
    print("Warning: rpy2 not installed. R comparisons will be skipped.")
    ro = None

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def generate_site_data(site_id, n_years=10, trend_type='increasing', noise_std=1.0,
                       start_year=2000, missing_prob=0.0):
    """
    Generates synthetic monthly data for a single site.
    """
    dates = pd.date_range(start=f'{start_year}-01-01', periods=n_years*12, freq='ME')
    t = np.arange(len(dates))

    # Base signal
    if trend_type == 'increasing':
        slope = 0.1
    elif trend_type == 'decreasing':
        slope = -0.1
    elif trend_type == 'stable':
        slope = 0.0
    else:
        raise ValueError("trend_type must be 'increasing', 'decreasing', or 'stable'")

    values = 10 + slope * t + np.random.normal(0, noise_std, len(t))

    # Introduce missing values
    if missing_prob > 0:
        mask = np.random.rand(len(values)) < missing_prob
        values[mask] = np.nan

    df = pd.DataFrame({
        'site': site_id,
        'time': dates,
        'value': values
    })

    # Pre-process for MannKS (add censored columns even if not used)
    # For this regional test, we assume no censoring for simplicity unless needed
    df = mk.prepare_censored_data(df['value'])
    df['site'] = site_id
    df['time'] = dates

    # Remove NaNs
    df = df.dropna(subset=['value'])

    return df

def run_python_regional_test(all_site_data):
    """
    Runs MannKS trend test for each site and then the regional test.
    """
    site_results = []

    for site in all_site_data['site'].unique():
        site_df = all_site_data[all_site_data['site'] == site].copy()

        # Run standard Mann-Kendall test
        # We use agg_method='median' to match typical R behavior for monthly data if needed,
        # but for clean monthly data 'none' is fine. Let's use 'none' for simplicity first.
        result = mk.trend_test(site_df['value'], site_df['time'], agg_method='median', agg_period='month', slope_scaling='year')

        site_results.append({
            'site': site,
            's': result.s,
            'C': result.C,
            'p': result.p,
            'slope': result.slope
        })

    trend_results_df = pd.DataFrame(site_results)

    # Run regional test
    regional_result = mk.regional_test(
        trend_results_df,
        all_site_data,
        site_col='site',
        value_col='value',
        time_col='time',
        s_col='s',
        c_col='C'
    )

    return regional_result, trend_results_df

def run_r_regional_test(all_site_data, python_trend_results):
    """
    Runs the R LWP-TRENDS regional analysis (getTAU).
    """
    if ro is None: return None
    # Load R script
    r = ro.r
    # We need to source the LWP script.
    lwp_script_path = os.path.join(repo_root, 'Example_Files', 'R', 'LWPTrends_v2502', 'LWPTrends_v2502.r')

    if not os.path.exists(lwp_script_path):
        print(f"R script not found at {lwp_script_path}")
        return None

    try:
        r.source(lwp_script_path)

        # Initialize globals in R
        r("""
          if(!exists("MidTimeIncrList")) {
              MidTimeIncrList<-list()
              yearshift<-0
              for(i in 1:12){MidTimeIncrList[[i]] <- seq(365/i/2,365-365/i/2,by=365/i)-yearshift-1}
              assign("MidTimeIncrList",MidTimeIncrList,envir = .GlobalEnv)
          }

          # Also need SeasonLabs if not present, though it's defined at top level in script usually
          if(!exists("SeasonLabs")) {
            SeasonLabs<-data.frame(mySeas=c("Monthly","Bi-monthly","Quarterly","Bi-annual"),
                           SeasName=c("Month","BiMonth","Qtr","BiAnn"))
            assign("SeasonLabs",SeasonLabs,envir = .GlobalEnv)
          }
        """)

        # Prepare 'obs' DataFrame for R
        # R expects: siteID, analyte, RawValue, Year, Censored, CenType, myDate, TimeIncr
        obs_r = all_site_data.copy()
        obs_r['siteID'] = obs_r['site']
        obs_r['analyte'] = 'TestVar'
        obs_r['RawValue'] = obs_r['value']
        obs_r['Year'] = obs_r['time'].dt.year
        obs_r['myDate'] = obs_r['time']
        # Censored columns are already there from prepare_censored_data
        obs_r['Censored'] = obs_r['censored'] # boolean
        obs_r['CenType'] = obs_r['cen_type']

        obs_r['TimeIncr'] = 'Monthly'
        obs_r['Month'] = obs_r['time'].dt.strftime('%b') # Jan, Feb...

        # Add dummy columns required by ValueForTimeIncr
        obs_r['BiMonth'] = 'NA'
        obs_r['Qtr'] = 'NA'
        obs_r['BiAnn'] = 'NA'
        obs_r['Season'] = obs_r['Month'] # Use Month as Season

        # Prepare 'x' DataFrame (trend results) for R
        x_r = python_trend_results.copy()
        x_r['siteID'] = x_r['site']
        x_r['analyte'] = 'TestVar'
        x_r['S'] = x_r['s']
        x_r['C'] = x_r['C'] # Python C is confidence (0.5 to 1.0)
        x_r['TimeIncr'] = 'Monthly' # Used to determine Frequency in getTAU

        # Convert to R objects
        with (ro.default_converter + pandas2ri.converter).context():
            r_obs = ro.conversion.get_conversion().py2rpy(obs_r)
            r_x = ro.conversion.get_conversion().py2rpy(x_r)

        # Call getTAU
        get_tau = r['getTAU']

        # Capture R output to avoid clutter
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            result_list = get_tau(r_x, r_obs, SiteNameCol="siteID", VarNameCol="analyte",
                                  ValuesToUse="RawValue", Year="Year", UseMidObs=True,
                                  DirType="Decrease")

        # Result is a list: SummaryResults, ObservationCorrelations
        summary_results = result_list[0] # DataFrame

        with (ro.default_converter + pandas2ri.converter).context():
             summary_df = ro.conversion.get_conversion().rpy2py(summary_results)

        return summary_df.iloc[0] # Return the first row as a Series

    except Exception as e:
        print("R execution failed:")
        print(e)
        return None

def create_markdown_report(title, description, scenarios):
    """
    Generates the markdown content.
    scenarios is a list of dicts: {'name': str, 'py_res': RegionalTrendResult, 'r_res': Series}
    """
    md = f"# {title}\n\n"
    md += f"{description}\n\n"

    md += "## Results Comparison\n\n"

    headers = ["Scenario", "Metric", "Python (MKS)", "R (LWP)", "Match?"]
    table_rows = []

    for scen in scenarios:
        py = scen['py_res']
        r = scen['r_res']
        name = scen['name']

        if r is None:
            # Handle failed R execution
            table_rows.append([name, "Sites (M)", py.M, "Error", "❓"])
            table_rows.append(["---", "---", "---", "---", "---"])
            continue

        # M (Number of sites)
        m_match = "✅" if py.M == r['M'] else "❌"
        table_rows.append([name, "Sites (M)", py.M, r['M'], m_match])

        # TAU
        tau_match = "✅" if np.isclose(py.TAU, r['TAU'], atol=1e-4) else "❌"
        table_rows.append(["", "TAU", f"{py.TAU:.4f}", f"{r['TAU']:.4f}", tau_match])

        # VarTAU
        # R returns VarTAU (uncorrected). Python returns VarTAU.
        var_match = "✅" if np.isclose(py.VarTAU, r['VarTAU'], atol=1e-4) else "❌"
        table_rows.append(["", "VarTAU (Uncorr)", f"{py.VarTAU:.4f}", f"{r['VarTAU']:.4f}", var_match])

        # CorrectedVarTAU
        corr_var_match = "✅" if np.isclose(py.CorrectedVarTAU, r['CorrectedVarTAU'], atol=1e-4) else "❌"
        table_rows.append(["", "Corrected VarTAU", f"{py.CorrectedVarTAU:.4f}", f"{r['CorrectedVarTAU']:.4f}", corr_var_match])

        # DT (Direction)
        dt_match = "✅" if py.DT == r['DT'] else "❌"
        table_rows.append(["", "Direction (DT)", py.DT, r['DT'], dt_match])

        # CT (Confidence)
        # R output 'CT' is the confidence.
        # Python output 'CT' is confidence.
        ct_match = "✅" if np.isclose(py.CT, r['CT'], atol=1e-4) else "❌"
        table_rows.append(["", "Confidence (CT)", f"{py.CT:.4f}", f"{r['CT']:.4f}", ct_match])

        # Separator row
        table_rows.append(["---", "---", "---", "---", "---"])

    try:
        from tabulate import tabulate
        md += tabulate(table_rows, headers=headers, tablefmt="github")
    except ImportError:
        # Fallback if tabulate not available
        md += "| " + " | ".join(headers) + " |\n"
        md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in table_rows:
            md += "| " + " | ".join(str(x) for x in row) + " |\n"

    md += "\n\n"
    return md

def run_v24():
    # Set seed for reproducibility
    np.random.seed(42)

    scenarios = []

    # --- Scenario 1: Aggregate Increasing (Mixed) ---
    print("Running Scenario 1: Aggregate Increasing (Mixed)...")
    sites = []
    # 4 increasing, 1 decreasing
    for i in range(4):
        sites.append(generate_site_data(f"Site_Inc_{i+1}", trend_type='increasing'))
    sites.append(generate_site_data(f"Site_Dec_1", trend_type='decreasing'))
    data_sc1 = pd.concat(sites)

    py_res_sc1, py_trends_sc1 = run_python_regional_test(data_sc1)
    r_res_sc1 = run_r_regional_test(data_sc1, py_trends_sc1)

    scenarios.append({
        'name': 'Aggregate Increasing',
        'py_res': py_res_sc1,
        'r_res': r_res_sc1
    })

    # --- Scenario 2: Aggregate Decreasing (Mixed) ---
    print("Running Scenario 2: Aggregate Decreasing (Mixed)...")
    sites = []
    # 4 decreasing, 1 increasing
    for i in range(4):
        sites.append(generate_site_data(f"Site_Dec_{i+1}", trend_type='decreasing'))
    sites.append(generate_site_data(f"Site_Inc_1", trend_type='increasing'))
    data_sc2 = pd.concat(sites)

    py_res_sc2, py_trends_sc2 = run_python_regional_test(data_sc2)
    r_res_sc2 = run_r_regional_test(data_sc2, py_trends_sc2)

    scenarios.append({
        'name': 'Aggregate Decreasing',
        'py_res': py_res_sc2,
        'r_res': r_res_sc2
    })

    # --- Scenario 3: Aggregate Indeterminate (Balanced) ---
    print("Running Scenario 3: Aggregate Indeterminate (Balanced)...")
    sites = []
    # 2 increasing, 2 decreasing, 1 stable
    sites.append(generate_site_data("Site_Inc_1", trend_type='increasing'))
    sites.append(generate_site_data("Site_Inc_2", trend_type='increasing'))
    sites.append(generate_site_data("Site_Dec_1", trend_type='decreasing'))
    sites.append(generate_site_data("Site_Dec_2", trend_type='decreasing'))
    sites.append(generate_site_data("Site_Stab_1", trend_type='stable'))
    data_sc3 = pd.concat(sites)

    py_res_sc3, py_trends_sc3 = run_python_regional_test(data_sc3)
    r_res_sc3 = run_r_regional_test(data_sc3, py_trends_sc3)

    scenarios.append({
        'name': 'Aggregate Indeterminate',
        'py_res': py_res_sc3,
        'r_res': r_res_sc3
    })

    # Generate Report
    report = create_markdown_report(
        "V-24: Regional Trend with Mixed Directions",
        "Verification of regional trend aggregation for sites with conflicting trend directions.",
        scenarios
    )

    output_path = os.path.join(os.path.dirname(__file__), "README.md")
    with open(output_path, "w") as f:
        f.write(report)

    print(f"V-24 Complete. Report generated at {output_path}")

if __name__ == "__main__":
    run_v24()
