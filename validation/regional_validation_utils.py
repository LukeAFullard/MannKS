
import numpy as np
import pandas as pd
import MannKS as mk
import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import contextlib
import io
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Activate pandas conversion (Legacy but needed for simplicity unless we refactor to context manager everywhere)
try:
    pandas2ri.activate()
except:
    pass

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
        result = mk.trend_test(site_df['value'], site_df['time'], agg_method='median', agg_period='month')

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
    # Load R script
    r = ro.r
    # We need to source the LWP script.
    # Assuming relative path from the validation script which will be in validation/XX/
    # So we go up two levels to root, then to Example_Files...

    # But wait, this utility is imported by the run_validation.py.
    # We should use absolute path or relative to repo root.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    lwp_script_path = os.path.join(repo_root, 'Example_Files', 'R', 'LWPTrends_v2502', 'LWPTrends_v2502.r')

    if not os.path.exists(lwp_script_path):
        raise FileNotFoundError(f"R script not found at {lwp_script_path}")

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

    # Add TimeIncrYear manually to avoid R script calculating it incorrectly or if it expects it
    # Actually GetTimeIncrYear does this.

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

    try:
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

    from tabulate import tabulate
    md += tabulate(table_rows, headers=headers, tablefmt="github")
    md += "\n\n"

    return md
