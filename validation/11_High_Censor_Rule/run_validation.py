import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import sys
import warnings

# Add the repository root to the path so we can import MannKS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import MannKS as mk
from MannKS import trend_test
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
import rpy2.robjects.vectors as ro_vectors

# --- Validation Settings ---
CASE_ID = "V-11"
CASE_NAME = "High Censor Rule"
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "validation_results.csv")
MASTER_RESULTS_FILE = os.path.join(os.path.dirname(__file__), "../master_results.csv")
README_FILE = os.path.join(os.path.dirname(__file__), "README.md")
LWP_R_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Example_Files/R/LWPTrends_v2502/LWPTrends_v2102.R"))
NADA2_SCRIPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../Example_Files/R/NADA2/ATS.R"))

# --- Helper Functions ---

def generate_hicensor_data(trend_type='increasing', n_years=10):
    """
    Generates monthly data with mixed detection limits to test the High Censor Rule.

    Structure:
    - Base values follow a trend.
    - Noise is added.
    - We ensure the trend spans *across* the high censoring limit to see if it survives.
    - Some low values are censored at <1.
    - Some intermediate values are censored at <5.
    - Crucially, some OBSERVED values are between 1 and 5 (e.g. 3).
    - The High Censor Rule should treat everything <5 (including the observed 3s) as censored at <5.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2010-01-01', periods=n_years*12, freq='ME')
    t = np.arange(len(dates))

    # Base trend
    # We want values to go from below 5 to above 5 clearly for an increasing trend.
    # Start around 2, go to 10.
    if trend_type == 'increasing':
        slope = 8.0 / len(t)
        values = 2.0 + slope * t + np.random.normal(0, 1.0, len(t))
    elif trend_type == 'decreasing':
        slope = -8.0 / len(t)
        values = 10.0 + slope * t + np.random.normal(0, 1.0, len(t))
    else: # stable
        values = 6.0 + np.random.normal(0, 1.5, len(t))

    # Ensure positive values
    values = np.abs(values)

    # Create mixed censoring
    # < 1 detection limit (Low)
    # < 5 detection limit (High)

    final_values = []
    censored = []

    for v in values:
        rand_val = np.random.random()

        # 10% chance of being censored at < 5
        # 10% chance of being censored at < 1

        # If the true value is very low, it's more likely to be censored in reality,
        # but here we force some censoring to test the rule.

        if rand_val < 0.10:
            final_values.append(5.0) # Represents <5
            censored.append('<')
        elif rand_val < 0.20:
             final_values.append(1.0) # Represents <1
             censored.append('<')
        else:
            final_values.append(v)
            censored.append(None)

    df = pd.DataFrame({
        'date': dates,
        'value': final_values,
        'censored_flag': censored
    })

    # Construct the raw string column for validation input
    def make_val_str(row):
        if row['censored_flag'] == '<':
            return f"<{row['value']}"
        else:
            return f"{row['value']:.3f}"

    df['value_str'] = df.apply(make_val_str, axis=1)

    return df

def run_lwp_r_script_v2102(df):
    """Runs the specific LWP v2102 R script."""
    # Convert data to R format
    # The R script expects: myDate, RawValue, Censored, CenType, Year, Season, SeasonYear

    # Prepare data in Python first
    df_prep = mk.prepare_censored_data(df['value_str'])
    df_prep['myDate'] = df['date']
    df_prep['Year'] = df['date'].dt.year
    df_prep['Month'] = df['date'].dt.strftime('%b')
    df_prep['Season'] = df_prep['Month'] # Monthly analysis

    # Rename columns to match LWP R script expectations (PascalCase)
    df_prep = df_prep.rename(columns={
        'value': 'RawValue',
        'censored': 'Censored',
        'cen_type': 'CenType'
    })

    # Save to temp CSV to avoid type conversion issues
    temp_csv = "temp_lwp_input.csv"
    df_prep.to_csv(temp_csv, index=False)

    # R Code wrapper
    r_code = f"""
    library(plyr)
    library(NADA)

    source("{LWP_R_SCRIPT_PATH}")

    # Load data
    data <- read.csv("{temp_csv}")
    data$myDate <- as.Date(data$myDate)
    data$Censored <- as.logical(data$Censored)
    data$CenType <- as.character(data$CenType) # Ensure correct type

    # Required for v2102: SeasonString
    SeasonString <<- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    data$Season <- factor(data$Season, levels=SeasonString)

    # v2102 specific: GetMoreDateInfo and MedianForSeason are internal but NonSeasonalTrendAnalysis calls MannKendall
    # We call MannKendall directly or NonSeasonalTrendAnalysis
    # Signature: NonSeasonalTrendAnalysis(x, do.plot=F, ...)
    # Key parameter: HiCensor=TRUE

    # Note: v2102 NonSeasonalTrendAnalysis calls MannKendall then SenSlope

    result <- NonSeasonalTrendAnalysis(data, ValuesToUse="RawValue", HiCensor=TRUE, UseMidObs=TRUE)

    # Extract results
    # v2102 returns a list/dataframe with: AnnualSenSlope, p, Sen_Lci, Sen_Uci

    list(
        slope = result$AnnualSenSlope,
        p = result$p,
        lci = result$Sen_Lci,
        uci = result$Sen_Uci
    )
    """

    try:
        result = ro.r(r_code)
        # result is a list with names
        slope = result[0][0]
        p_value = result[1][0]
        lci = result[2][0]
        uci = result[3][0]

        # Handle NAs
        if np.isnan(slope): slope = 0.0

        os.remove(temp_csv)
        return slope, p_value, lci, uci
    except Exception as e:
        print(f"R Script Error: {e}")
        if os.path.exists(temp_csv): os.remove(temp_csv)
        return np.nan, np.nan, np.nan, np.nan

def run_nada_r_script(df):
    """Runs the NADA2 ATS analysis."""
    # NADA2 ATS is for non-seasonal data primarily, but here we are checking the high censor rule effect.
    # NADA2 doesn't have a specific "HiCensor" rule, so it will treat data as provided.
    # This comparison highlights the difference in methodology.

    df_prep = mk.prepare_censored_data(df['value_str'])

    # Convert to vectors for R
    # ATS.R requires numeric time, e.g., fractional years
    dates = df['date']
    frac_years = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25

    # Passing large vectors via f-string is risky/slow. Better to use rpy2 conversion.
    with (ro.default_converter + pandas2ri.converter).context():
        r_y = ro.conversion.get_conversion().py2rpy(df_prep['value'])
        r_ycen = ro.conversion.get_conversion().py2rpy(df_prep['censored'])
        r_x = ro.conversion.get_conversion().py2rpy(frac_years)

    ro.globalenv['y'] = r_y
    ro.globalenv['ycen'] = r_ycen
    ro.globalenv['x'] = r_x

    r_code_run = f"""
    source("{NADA2_SCRIPT_PATH}")
    # ATS might return an S3 object or list.
    # ATS signature: ATS(..., drawplot=TRUE, ...)
    # It returns 'ret' invisibly which is a data.frame
    res <- ATS(y, ycen, x, LOG=FALSE, drawplot=FALSE)

    # res is a data frame with columns: intercept, slope, S, tau, pval
    list(slope=res$slope[1], p=res$pval[1])
    """

    try:
        res = ro.r(r_code_run)
        slope = res[0][0]
        p_value = res[1][0]
        # NADA2 ATS script doesn't output CIs in the returned dataframe.
        return slope, p_value, np.nan, np.nan
    except Exception as e:
        print(f"NADA R Error: {e}")
        return np.nan, np.nan, np.nan, np.nan

def analyze_scenario(scenario_name, df):
    print(f"Analyzing {scenario_name}...")

    # Pre-process data for trend_test
    df_prep = mk.prepare_censored_data(df['value_str'])

    # 1. MannKS Standard (Robust, no HiCensor)
    # Scale slope to year to match R
    mk_std = trend_test(df_prep, df['date'], mk_test_method='robust', slope_scaling='year')

    # 2. MannKS LWP Mode (with HiCensor)
    mk_lwp = trend_test(df_prep, df['date'],
                        mk_test_method='lwp',
                        ci_method='lwp',
                        agg_method='lwp',
                        sens_slope_method='lwp',
                        hicensor=True, # The key parameter
                        agg_period='month',
                        slope_scaling='year')

    # 3. LWP R Script (v2102)
    r_slope, r_p, r_lci, r_uci = run_lwp_r_script_v2102(df)

    # 4. MannKS ATS Mode
    mk_ats = trend_test(df_prep, df['date'], sens_slope_method='ats', slope_scaling='year')

    # 5. NADA2 R Script
    nada_slope, nada_p, nada_lci, nada_uci = run_nada_r_script(df)

    # Compile Results
    # Flatten structure for easy DataFrame creation
    results = {
        'test_id': f"{CASE_ID}_{scenario_name.replace(' ', '_').lower()}",
    }

    # Method 1
    results['mk_py_slope'] = mk_std.slope
    results['mk_py_p_value'] = mk_std.p
    results['mk_py_lower_ci'] = mk_std.lower_ci
    results['mk_py_upper_ci'] = mk_std.upper_ci

    # Method 2
    results['lwp_py_slope'] = mk_lwp.slope
    results['lwp_py_p_value'] = mk_lwp.p
    results['lwp_py_lower_ci'] = mk_lwp.lower_ci
    results['lwp_py_upper_ci'] = mk_lwp.upper_ci

    # Method 3
    results['r_slope'] = r_slope
    results['r_p_value'] = r_p
    results['r_lower_ci'] = r_lci
    results['r_upper_ci'] = r_uci

    # Method 4
    results['ats_py_slope'] = mk_ats.slope
    results['ats_py_p_value'] = mk_ats.p
    results['ats_py_lower_ci'] = mk_ats.lower_ci
    results['ats_py_upper_ci'] = mk_ats.upper_ci

    # Method 5
    results['nada_r_slope'] = nada_slope
    results['nada_r_p_value'] = nada_p
    results['nada_r_lower_ci'] = nada_lci
    results['nada_r_upper_ci'] = nada_uci

    # Errors
    results['slope_error'] = np.nan
    results['slope_pct_error'] = np.nan
    if not np.isnan(r_slope):
        results['slope_error'] = mk_lwp.slope - r_slope
        if r_slope != 0:
            results['slope_pct_error'] = (results['slope_error'] / r_slope) * 100
        elif mk_lwp.slope == 0:
            results['slope_pct_error'] = 0.0

    return results, mk_lwp, df_prep

def generate_combined_plot(scenarios, filename, main_title):
    num_plots = len(scenarios)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))
    if num_plots == 1:
        axes = [axes]

    for ax, scen in zip(axes, scenarios):
        df = scen['df']
        title = scen.get('title', '')
        mk_result = scen.get('mk_result')

        # Prepare data for plotting
        x_plot = df['date']
        # Convert values, stripping <
        y_raw = df['value_str'].str.replace('<', '', regex=False).astype(float)

        is_censored = df['value_str'].str.contains('<')

        # Color coding: Black = Observed, Red = Censored (Original),
        # Blue = Censored (High Censor Visualization)
        # Actually, let's just show Red for any censored.

        colors = np.where(is_censored, 'red', 'black')

        ax.scatter(x_plot, y_raw, c=colors, s=30, alpha=0.7)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Observed', markerfacecolor='black', markersize=8),
            Line2D([0], [0], marker='o', color='w', label='Censored', markerfacecolor='red', markersize=8)
        ]

        if mk_result is not None and not np.isnan(mk_result.slope):
            # Plot trend line
            # Need numeric time for slope calc
            t_numeric = (x_plot - pd.Timestamp("1970-01-01")).dt.days / 365.25

            # Slope is per year (from scaling).
            # The intercept returned by trend_test depends on the time units used during calculation.
            # To ensure the plot line matches the slope correctly, we reconstruct it pivoting around the median.

            # Re-calculate line points
            y_trend = mk_result.slope * t_numeric.values + mk_result.intercept

            # MannKS intercept is calculated as y_med - slope * t_med.
            # If slope is scaled (e.g., per year), care must be taken with units.
            # We plot using the slope and pivoting around the median for consistency.

            t_seconds = (x_plot - x_plot.min()).dt.total_seconds()
            t_years = t_seconds / 31557600 # Approx

            # Re-calculate a safe trend line for plotting based on the returned slope
            # Pivot at median time and median value
            # The trend line passes through (t_med, y_med)

            # Approximate center
            t_mid = x_plot.mean()
            y_mid = y_raw.median() # Rough approx

            # Using the returned slope (units/year)
            # Delta Years from mid
            delta_years = (x_plot - t_mid).dt.total_seconds() / (365.25 * 24 * 3600)
            y_line = y_mid + mk_result.slope * delta_years

            ax.plot(x_plot, y_line, '-', color='blue', lw=2, label=f"LWP Slope: {mk_result.slope:.4f}/yr")
            legend_elements.append(Line2D([0], [0], color='blue', lw=2, label=f"LWP Slope: {mk_result.slope:.4f}/yr"))

        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend(handles=legend_elements, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.6)

    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Combined plot saved to {plot_path}")

# --- Main Execution ---

if __name__ == "__main__":
    all_results_rows = []
    scenarios = []

    # Scenario 1: Strong Increasing
    df_inc = generate_hicensor_data('increasing')
    res_inc, mk_lwp_inc, df_inc_prep = analyze_scenario("strong_increasing", df_inc)
    all_results_rows.append(res_inc)
    scenarios.append({'df': df_inc, 'title': 'Strong Increasing (LWP Mode)', 'mk_result': mk_lwp_inc})

    # Scenario 2: Weak Decreasing
    df_dec = generate_hicensor_data('decreasing')
    res_dec, mk_lwp_dec, _ = analyze_scenario("weak_decreasing", df_dec)
    all_results_rows.append(res_dec)
    scenarios.append({'df': df_dec, 'title': 'Weak Decreasing (LWP Mode)', 'mk_result': mk_lwp_dec})

    # Scenario 3: Stable
    df_stable = generate_hicensor_data('stable')
    res_stable, mk_lwp_stable, _ = analyze_scenario("stable", df_stable)
    all_results_rows.append(res_stable)
    scenarios.append({'df': df_stable, 'title': 'Stable (LWP Mode)', 'mk_result': mk_lwp_stable})

    # Generate Plot
    generate_combined_plot(scenarios, "v11_combined.png", "V-11: High Censor Rule Analysis (LWP Mode)")

    # Save Results CSV
    results_df = pd.DataFrame(all_results_rows)
    results_df.to_csv(RESULTS_FILE, index=False)

    # Append to Master CSV
    if os.path.exists(MASTER_RESULTS_FILE):
        master_df = pd.read_csv(MASTER_RESULTS_FILE)
        # Remove old entries for this case
        master_df = master_df[~master_df['test_id'].str.startswith(CASE_ID)]
        # Map columns to match master_results structure if needed
        # My result_row keys roughly match master csv columns from V-01 example
        # Need to ensure column order/existence
        master_df = pd.concat([master_df, results_df], ignore_index=True)
        master_df.to_csv(MASTER_RESULTS_FILE, index=False)
    else:
        results_df.to_csv(MASTER_RESULTS_FILE, index=False)

    # Generate Detailed README
    with open(README_FILE, 'w') as f:
        f.write(f"# {CASE_ID}: {CASE_NAME}\n\n")
        f.write("## Objective\n")
        f.write("Verify the implementation of the 'High Censor Rule' (`hicensor=True`). "
                "This rule, used in older LWP-TRENDS versions, sets all values below the highest "
                "detection limit to be censored at that highest limit. This prevents spurious trends "
                "caused solely by changing detection limits (e.g., detection limit improves from <5 to <1 over time).\n\n")

        f.write("## Plots\n")
        f.write("### v11_combined.png\n")
        f.write("![v11_combined.png](v11_combined.png)\n\n")

        f.write("## Results\n")

        # Reshape for display like V-01
        long_rows = []
        for res in all_results_rows:
            test_id = res['test_id']
            # Standard
            long_rows.append({'Test ID': test_id, 'Method': 'MannKS (Standard)', 'Slope': res['mk_py_slope'], 'P-Value': res['mk_py_p_value'], 'Lower CI': res['mk_py_lower_ci'], 'Upper CI': res['mk_py_upper_ci']})
            # LWP Mode
            long_rows.append({'Test ID': test_id, 'Method': 'MannKS (LWP Mode)', 'Slope': res['lwp_py_slope'], 'P-Value': res['lwp_py_p_value'], 'Lower CI': res['lwp_py_lower_ci'], 'Upper CI': res['lwp_py_upper_ci']})
            # R LWP
            long_rows.append({'Test ID': test_id, 'Method': 'LWP-TRENDS (R)', 'Slope': res['r_slope'], 'P-Value': res['r_p_value'], 'Lower CI': res['r_lower_ci'], 'Upper CI': res['r_upper_ci']})
            # ATS
            long_rows.append({'Test ID': test_id, 'Method': 'MannKS (ATS)', 'Slope': res['ats_py_slope'], 'P-Value': res['ats_py_p_value'], 'Lower CI': res['ats_py_lower_ci'], 'Upper CI': res['ats_py_upper_ci']})
            # NADA R
            long_rows.append({'Test ID': test_id, 'Method': 'NADA2 (R)', 'Slope': res['nada_r_slope'], 'P-Value': res['nada_r_p_value'], 'Lower CI': res['nada_r_lower_ci'], 'Upper CI': res['nada_r_upper_ci']})

        df_long = pd.DataFrame(long_rows)
        try:
            import tabulate
            f.write(df_long.to_markdown(index=False))
        except ImportError:
            f.write(df_long.to_string(index=False))
        f.write("\n\n")

        f.write("## LWP Accuracy (Python vs R)\n")
        acc_df = results_df[['test_id', 'slope_error', 'slope_pct_error']].rename(columns={'test_id': 'Test ID', 'slope_error': 'Slope Error', 'slope_pct_error': 'Slope % Error'})
        try:
            f.write(acc_df.to_markdown(index=False))
        except ImportError:
            f.write(acc_df.to_string(index=False))
        f.write("\n")

    print(f"Validation complete. Results saved to {RESULTS_FILE} and {README_FILE}")
