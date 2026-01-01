import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple, List, Union
from datetime import datetime

# Add repo root to path to ensure MannKS can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk

# RPy2 imports
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

# No global activation of pandas2ri to avoid deprecation warnings

class ValidationUtils:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.lwp_script_path = os.path.join(repo_root, 'Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r')
        self.nada2_ats_path = os.path.join(repo_root, 'Example_Files/R/NADA2/ATS.R')
        self.nada2_ken_path = os.path.join(repo_root, 'Example_Files/R/NADA2/NADA_ken.R')
        self.nada2_atsmini_path = os.path.join(repo_root, 'Example_Files/R/NADA2/ATSmini.R')
        self.nada2_censeaken_path = os.path.join(repo_root, 'Example_Files/R/NADA2/censeaken.R')
        self.nada2_computes_path = os.path.join(repo_root, 'Example_Files/R/NADA2/computeS.R')
        self.nada2_kenplot_path = os.path.join(repo_root, 'Example_Files/R/NADA2/kenplot.R')
        self.master_csv_path = os.path.join(repo_root, 'validation/master_results.csv')
        self.results = [] # Store results for this session

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _prepare_r_dataframe(self, df: pd.DataFrame, is_seasonal: bool = False) -> ro.DataFrame:
        """
        Converts Python DataFrame to R DataFrame and adds necessary columns for LWP script.
        """
        df_r_prep = df.copy()

        # Handle Missing Date if Time (Numeric Year) is present
        if 'date' not in df_r_prep.columns and 'time' in df_r_prep.columns:
            # Construct a fake date from decimal year for R script compatibility
            year = df_r_prep['time'].astype(int)
            days = ((df_r_prep['time'] - year) * 365.25).astype(int)
            # Use 'D' for days. ensure year is string for to_datetime
            df_r_prep['date'] = pd.to_datetime(year.astype(str) + '-01-01') + pd.to_timedelta(days, unit='D')

        # Ensure 'MyDate' exists (LWP script expects 'myDate')
        if 'date' in df_r_prep.columns:
            df_r_prep['myDate'] = df_r_prep['date']

        # Add LWP required columns if not present
        if 'RawValue' not in df_r_prep.columns:
            df_r_prep['RawValue'] = df_r_prep['value']

        # Handle mixed censoring for RawValue and CenType
        if df_r_prep['value'].dtype == object:
            # Check for censoring indicators
            is_left = df_r_prep['value'].astype(str).str.contains('<')
            is_right = df_r_prep['value'].astype(str).str.contains('>')

            # Clean RawValue by removing both < and >
            clean_vals = df_r_prep['value'].astype(str).str.replace('<', '', regex=False).str.replace('>', '', regex=False)
            df_r_prep['RawValue'] = clean_vals.astype(float)

            # Set Censored flag (True if either left or right)
            df_r_prep['Censored'] = is_left | is_right

            # Set CenType
            conditions = [is_left, is_right]
            choices = ['lt', 'gt']
            df_r_prep['CenType'] = np.select(conditions, choices, default='not')
        else:
            if 'Censored' not in df_r_prep.columns:
                df_r_prep['Censored'] = False
            if 'CenType' not in df_r_prep.columns:
                df_r_prep['CenType'] = 'not'

        df_r_prep['Censored'] = df_r_prep['Censored'].astype(bool)

        with localconverter(ro.default_converter + pandas2ri.converter):
            r_df = ro.conversion.py2rpy(df_r_prep)

        # FORCE Date conversion: LWP expects 'Date' (days), but rpy2 passes POSIXct (seconds).
        # This causes slope calculation error by factor of 86400.
        ro.globalenv['df_temp'] = r_df
        ro.r('df_temp$myDate <- as.Date(df_temp$myDate)')
        r_df = ro.globalenv['df_temp']

        return r_df

    def run_lwp_r_script(self, df: pd.DataFrame, seasonal: bool = False) -> Dict:
        """Runs the LWP-TRENDS R script."""
        try:
            ro.r(f'source("{self.lwp_script_path}")')

            r_df = self._prepare_r_dataframe(df)

            # Inject dataframe into R environment
            ro.globalenv['df_r'] = r_df
            ro.r('df_r$CenType <- as.factor(df_r$CenType)')

            # Critical Step: Run GetMoreDateInfo to populate MidTimeIncrList and date columns
            ro.r('df_r <- GetMoreDateInfo(df_r)')

            # Set TimeIncr explicitly.
            # If unique years == count, assume annual. Otherwise assume monthly (use Month).
            # If date was generated from time, year column should be correct.
            # But checking df['date'] might fail if df didn't have it originally.
            # Check r_df columns or reconstruct logic.
            # We know _prepare_r_dataframe adds 'date' if missing.

            # Simple heuristic: if 'time' is in df and it looks like years (e.g. 2000, 2001),
            # LWP should use Year.

            is_annual = True
            # Check repetition of years in R dataframe
            ro.r('year_counts <- table(df_r$Year)')
            max_counts = ro.r('max(year_counts)')[0]
            if max_counts > 1:
                is_annual = False

            if is_annual and not seasonal:
                 ro.r('df_r$TimeIncr <- df_r$Year')
            else:
                 ro.r('df_r$TimeIncr <- df_r$Month')

            if not seasonal:
                # Capture output
                cmd = """
                suppressWarnings(
                    result <- NonSeasonalTrendAnalysis(df_r, do.plot=FALSE, TimeIncrMed=TRUE, UseMidObs=TRUE)
                )
                """
                ro.r(cmd)
            else:
                ro.r('df_r$Season <- df_r$Month')
                cmd = """
                suppressWarnings(
                    result <- SeasonalTrendAnalysis(df_r, do.plot=FALSE, TimeIncrMed=TRUE, UseMidObs=TRUE)
                )
                """
                ro.r(cmd)

            if 'result' not in ro.globalenv:
                return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

            r_result = ro.globalenv['result']

            if r_result == ro.r('NULL'):
                    return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

            with localconverter(ro.default_converter + pandas2ri.converter):
                res_df = ro.conversion.rpy2py(r_result)

            if res_df is None or len(res_df) == 0:
                return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

            return {
                'slope': float(res_df['AnnualSenSlope'].iloc[0]),
                'p_value': float(res_df['p'].iloc[0]),
                'lower_ci': float(res_df['Sen_Lci'].iloc[0]),
                'upper_ci': float(res_df['Sen_Uci'].iloc[0])
            }

        except Exception as e:
            print(f"Error running LWP R script: {e}")
            return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

    def run_nada2_r_script(self, df: pd.DataFrame, seasonal: bool = False) -> Dict:
        """Runs the NADA2 ATS.R or censeaken.R script."""
        try:
            # Load required packages explicitly
            ro.r('library(Icens)')
            ro.r('library(survival)')

            ro.r(f'source("{self.nada2_ken_path}")')
            ro.r(f'source("{self.nada2_ats_path}")')

            if seasonal:
                ro.r(f'source("{self.nada2_atsmini_path}")')
                ro.r(f'source("{self.nada2_computes_path}")')
                ro.r(f'source("{self.nada2_kenplot_path}")')
                ro.r(f'source("{self.nada2_censeaken_path}")')

            if 'value' in df.columns and df['value'].dtype == object and (df['value'].str.contains('<').any() or df['value'].str.contains('>').any()):
                 y_vals = df['value'].astype(str).str.replace('<', '').str.replace('>', '').astype(float).values
                 # Logical OR on contains < or >
                 is_censored = df['value'].astype(str).str.contains('<') | df['value'].astype(str).str.contains('>')
                 y_cen = is_censored.values
            else:
                y_vals = df['value'].values
                y_cen = np.zeros(len(df), dtype=bool)

            # Determine x_vals (time)
            if 'time' in df.columns:
                 x_vals = df['time'].values
            elif 'date' in df.columns:
                 dates = pd.to_datetime(df['date'])
                 # Convert to decimal year
                 x_vals = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25
                 x_vals = x_vals.values
            else:
                x_vals = np.arange(len(df))

            # Explicit conversion to R vectors to avoid numpy2ri context issues
            ro.globalenv['y_vec'] = ro.FloatVector(y_vals)
            # BoolVector in R is logical
            ro.globalenv['ycen_vec'] = ro.BoolVector(y_cen)
            ro.globalenv['x_vec'] = ro.FloatVector(x_vals)

            if not seasonal:
                cmd = "res_ats <- ATS(y_vec, ycen_vec, x_vec, LOG=FALSE, printstat=FALSE, drawplot=FALSE)"
                ro.r(cmd)
                r_res = ro.globalenv['res_ats']

                with localconverter(ro.default_converter + pandas2ri.converter):
                    res_df = ro.conversion.rpy2py(r_res)

                return {
                    'slope': float(res_df['slope'].iloc[0]),
                    'p_value': float(res_df['pval'].iloc[0]),
                    'lower_ci': np.nan,
                    'upper_ci': np.nan
                }
            else:
                # Prepare 'group' variable for censeaken
                # Assuming monthly data for now, or use df['date'] to extract month
                if 'date' in df.columns:
                    dates = pd.to_datetime(df['date'])
                    groups = dates.dt.month.values
                else:
                    # Fallback or error if no date column for seasonal
                    # Or try to infer from index if it's monthly
                    groups = np.tile(np.arange(1, 13), len(df) // 12 + 1)[:len(df)]

                ro.globalenv['grp_vec'] = ro.FloatVector(groups) # censeaken expects factor or numeric

                # censeaken(time, y, y.cen, group, data = NULL, LOG = FALSE, R = 4999, nmin = 4, ...)
                # Note: censeaken uses permutation test (R=4999). It might be slow.
                # Reducing R for validation speed if allowed, but sticking to default for accuracy.
                cmd = "res_censeaken <- censeaken(x_vec, y_vec, ycen_vec, grp_vec, LOG=FALSE, R=499, seaplots=FALSE)"
                ro.r(cmd)
                r_res = ro.globalenv['res_censeaken']

                # censeaken returns a named list/dataframe with stats.
                # Based on analysis: Prints and returns an overall RESULTS dataframe containing:
                # S_SK, tau_SK, pval, intercept, slope

                with localconverter(ro.default_converter + pandas2ri.converter):
                    res_df = ro.conversion.rpy2py(r_res)

                # res_df is usually a DataFrame with 1 row
                return {
                    'slope': float(res_df['slope'].iloc[0]),
                    'p_value': float(res_df['pval'].iloc[0]),
                    'lower_ci': np.nan, # censeaken does not appear to return CIs for slope easily?
                    'upper_ci': np.nan
                }

        except Exception as e:
            print(f"Error running NADA2 R script: {e}")
            return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}


    def run_comparison(self, test_id: str, df: pd.DataFrame,
                       scenario_name: str,
                       mk_kwargs: Dict = {},
                       lwp_mode_kwargs: Dict = {},
                       true_slope: float = None,
                       seasonal: bool = False) -> Tuple[Dict, object]:

        full_test_id = f"{test_id}_{scenario_name}"
        print(f"Running comparison for: {full_test_id} (Seasonal: {seasonal})")

        # Pre-process censored data for Python if needed
        x_std = df['value']
        if df['value'].dtype == object:
             try:
                 x_std = mk.prepare_censored_data(df['value'])
             except Exception as e:
                 print(f"Warning: automatic pre-processing failed: {e}")

        # Standard MK: Use converted numeric time (decimal years)
        t_datetime = None
        t_numeric = None

        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            t_datetime = dates.to_numpy()

            # --- FIX: Use R-style numeric time calculation (Days since Epoch / 365.25) ---
            # Correctly access .dt.days via timedelta
            t_numeric = (dates - pd.Timestamp("1970-01-01")).dt.days / 365.25
            t_numeric = t_numeric.values
        else:
            t_numeric = np.arange(len(df))
            t_std = t_numeric # Default for fallback if no date

        # Pass x_std (which might be the DataFrame from prepare_censored_data)
        if 't_std' not in locals():
            t_std = t_numeric if t_numeric is not None else t_datetime

        if seasonal:
             # For seasonal, default to period 12 if not in kwargs and t is numeric/time
             # Or rely on seasonal_trend_test defaults
             mk_std = mk.seasonal_trend_test(x_std, t_datetime if t_datetime is not None else t_std, **mk_kwargs)
        else:
             mk_std = mk.trend_test(x_std, t_std, **mk_kwargs)

        # LWP Mode: Try to mimic R behavior.
        # Helper to prepare censored data if needed
        def get_input_x(dataframe):
             if dataframe['value'].dtype == object and dataframe['value'].astype(str).str.contains('<|>').any():
                 return mk.prepare_censored_data(dataframe['value'])
             return dataframe['value']

        x_input = get_input_x(df)

        lwp_defaults = {
            'mk_test_method': 'lwp',
            'ci_method': 'lwp',
            'sens_slope_method': 'lwp',
            'agg_method': 'middle_lwp'
        }
        lwp_final_kwargs = {**lwp_defaults, **lwp_mode_kwargs}

        if 'date' in df.columns:
             t_lwp = pd.to_datetime(df['date'])
             n_years = len(t_lwp.dt.year.unique())
             n_obs = len(t_lwp)

             if 'agg_period' not in lwp_final_kwargs and not seasonal:
                 if n_years == n_obs:
                     lwp_final_kwargs['agg_period'] = 'year'
                 else:
                     lwp_final_kwargs['agg_period'] = 'month' # Assumption

             # seasonal_trend_test does NOT support agg_period. It uses season_type/period.
             # So we must remove agg_period if it got added or is present when calling seasonal.
             if seasonal and 'agg_period' in lwp_final_kwargs:
                 del lwp_final_kwargs['agg_period']

             if 'slope_scaling' not in lwp_final_kwargs:
                 lwp_final_kwargs['slope_scaling'] = 'year'

             if seasonal:
                 mk_lwp = mk.seasonal_trend_test(x_std, t_lwp, **lwp_final_kwargs)
             else:
                 mk_lwp = mk.trend_test(x_std, t_lwp, **lwp_final_kwargs)
        else:
             if seasonal:
                 # Check if agg_period is in kwargs and remove it for seasonal
                 if 'agg_period' in lwp_final_kwargs:
                      del lwp_final_kwargs['agg_period']
                 mk_lwp = mk.seasonal_trend_test(x_std, t_std, **lwp_final_kwargs)
             else:
                 mk_lwp = mk.trend_test(x_std, t_std, **lwp_final_kwargs)

        r_res = self.run_lwp_r_script(df, seasonal=seasonal)

        # ATS
        # Pass t_datetime if available to ensure correct seasonality handling in ATS mode
        if t_datetime is not None:
             t_ats = t_datetime
        else:
             t_ats = t_numeric if t_numeric is not None else t_datetime

        if seasonal:
             mk_ats = mk.seasonal_trend_test(x_std, t_ats, sens_slope_method='ats', **mk_kwargs)
        else:
             mk_ats = mk.trend_test(x_std, t_ats, sens_slope_method='ats')

        # NADA2 R script
        nada_res = self.run_nada2_r_script(df, seasonal=seasonal)

        slope_error = np.nan
        slope_pct_error = np.nan

        if not np.isnan(r_res['slope']):
            slope_error = mk_lwp.slope - r_res['slope']

            if true_slope is not None and true_slope != 0:
                slope_pct_error = (slope_error / true_slope) * 100
            elif r_res['slope'] != 0:
                slope_pct_error = (slope_error / r_res['slope']) * 100
            elif slope_error == 0:
                slope_pct_error = 0.0
            else:
                slope_pct_error = np.nan

        result_row = {
            'test_id': full_test_id,
            'mk_py_slope': mk_std.slope,
            'mk_py_p_value': mk_std.p,
            'mk_py_lower_ci': mk_std.lower_ci,
            'mk_py_upper_ci': mk_std.upper_ci,
            'lwp_py_slope': mk_lwp.slope,
            'lwp_py_p_value': mk_lwp.p,
            'lwp_py_lower_ci': mk_lwp.lower_ci,
            'lwp_py_upper_ci': mk_lwp.upper_ci,
            'r_slope': r_res['slope'],
            'r_p_value': r_res['p_value'],
            'r_lower_ci': r_res['lower_ci'],
            'r_upper_ci': r_res['upper_ci'],
            'ats_py_slope': mk_ats.slope,
            'ats_py_p_value': mk_ats.p,
            'ats_py_lower_ci': mk_ats.lower_ci,
            'ats_py_upper_ci': mk_ats.upper_ci,
            'nada_r_slope': nada_res['slope'],
            'nada_r_p_value': nada_res['p_value'],
            'nada_r_lower_ci': nada_res['lower_ci'],
            'nada_r_upper_ci': nada_res['upper_ci'],
            'slope_error': slope_error,
            'slope_pct_error': slope_pct_error
        }

        self.results.append(result_row)
        self._append_to_csv(result_row)
        return result_row, mk_std

    def _append_to_csv(self, row: Dict):
        df = pd.DataFrame([row])
        if os.path.exists(self.master_csv_path) and os.stat(self.master_csv_path).st_size > 0:
             cols = pd.read_csv(self.master_csv_path, nrows=0).columns.tolist()
             df = df[cols]
             df.to_csv(self.master_csv_path, mode='a', header=False, index=False)
        else:
             df.to_csv(self.master_csv_path, mode='w', header=True, index=False)


    def _get_decimal_year(self, df: pd.DataFrame) -> np.ndarray:
        """Converts dataframe date column to decimal year, matching run_comparison logic."""
        if 'time' in df.columns:
            return df['time'].values
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            # --- FIX: Use R-style numeric time calculation in this helper too ---
            # Correctly access .dt.days via timedelta
            t = (dates - pd.Timestamp("1970-01-01")).dt.days / 365.25
            return t.values
        else:
            return np.arange(len(df))

    def generate_plot(self, df: pd.DataFrame, title: str, filename: str, mk_result=None):
        """Generates a simple plot of the data, optionally with trend line and CIs."""
        # For backward compatibility, wraps generate_combined_plot with single scenario
        self.generate_combined_plot([{'df': df, 'title': title, 'mk_result': mk_result}], filename, title)

    def generate_combined_plot(self, scenarios: List[Dict], filename: str, main_title: str):
        """
        Generates a figure with subplots (e.g. 1x3).
        scenarios: List of dicts with keys: 'df', 'title', 'mk_result'.
        """
        num_plots = len(scenarios)
        fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

        if num_plots == 1:
            axes = [axes]

        for ax, scen in zip(axes, scenarios):
            df = scen['df']
            title = scen.get('title', '')
            mk_result = scen.get('mk_result')

            if 'time' in df.columns:
                x_plot = df['time']
            elif 'date' in df.columns:
                x_plot = df['date']
            else:
                x_plot = np.arange(len(df))

            # Handle censored values for plotting (just strip < and >)
            y_plot = df['value'].astype(str).str.replace('<', '', regex=False).str.replace('>', '', regex=False).astype(float)

            # Determine color based on censoring
            colors = []
            if 'cen_type' in df.columns:
                 for c_type in df['cen_type']:
                     if c_type == 'lt': colors.append('red') # Left censored
                     elif c_type == 'gt': colors.append('orange') # Right censored
                     else: colors.append('black')
            else:
                 colors = ['black'] * len(df)

            ax.scatter(x_plot, y_plot, c=colors, label='Data')

            # Create proxy artists for legend
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Data', markerfacecolor='black', markersize=8)]
            if 'red' in colors:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Left Censored', markerfacecolor='red', markersize=8))
            if 'orange' in colors:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Right Censored', markerfacecolor='orange', markersize=8))

            # Add Trend Line and CIs if provided
            if mk_result is not None and not np.isnan(mk_result.slope):
                t_numeric = self._get_decimal_year(df)

                # Calculate Trend Line: y = slope * t + intercept
                y_trend = mk_result.slope * t_numeric + mk_result.intercept

                ax.plot(x_plot, y_trend, '-', color='blue', label=f"Sen's Slope: {mk_result.slope:.4f}")
                legend_elements.append(Line2D([0], [0], color='blue', lw=2, label=f"Slope: {mk_result.slope:.4f}"))

                # Calculate CIs (Approximation for visualization: pivot around median)
                t_med = np.median(t_numeric)
                y_med = np.median(y_plot)

                if hasattr(mk_result, 'lower_ci') and not np.isnan(mk_result.lower_ci):
                    y_lower = mk_result.lower_ci * (t_numeric - t_med) + y_med
                    ax.plot(x_plot, y_lower, '--', color='blue', alpha=0.5)
                    legend_elements.append(Line2D([0], [0], color='blue', lw=1, linestyle='--', label='90% CI'))

                if hasattr(mk_result, 'upper_ci') and not np.isnan(mk_result.upper_ci):
                    y_upper = mk_result.upper_ci * (t_numeric - t_med) + y_med
                    ax.plot(x_plot, y_upper, '--', color='blue', alpha=0.5)

                    # Shade the confidence interval
                    if hasattr(mk_result, 'lower_ci') and not np.isnan(mk_result.lower_ci):
                        ax.fill_between(x_plot, y_lower, y_upper, color='blue', alpha=0.1)

            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(handles=legend_elements, loc='best')
            ax.grid(True, linestyle=':', alpha=0.6)

        fig.suptitle(main_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Combined plot saved to {plot_path}")

    def create_report(self, filename='README.md', description=None):
        """Creates a Markdown report with results and plots in long format."""
        report_path = os.path.join(self.output_dir, filename)

        with open(report_path, 'w') as f:
            f.write(f"# Validation Report\n\n")

            if description:
                f.write(description + "\n\n")

            # Plots
            f.write("## Plots\n")
            plots = [p for p in os.listdir(self.output_dir) if p.endswith('.png')]
            for p in sorted(plots):
                f.write(f"### {p}\n")
                f.write(f"![{p}]({p})\n\n")

            # Results Table (Long Format)
            f.write("## Results\n")
            if self.results:
                long_rows = []
                for res in self.results:
                    test_id = res.get('test_id', 'Unknown')

                    # Define method mappings
                    methods = [
                        ('MannKS (Standard)', 'mk_py'),
                        ('MannKS (LWP Mode)', 'lwp_py'),
                        ('LWP-TRENDS (R)', 'r'),
                        ('MannKS (ATS)', 'ats_py'),
                        ('NADA2 (R)', 'nada_r')
                    ]

                    for method_name, prefix in methods:
                        row = {
                            'Test ID': test_id,
                            'Method': method_name,
                            'Slope': res.get(f'{prefix}_slope', np.nan),
                            'P-Value': res.get(f'{prefix}_p_value', np.nan),
                            'Lower CI': res.get(f'{prefix}_lower_ci', np.nan),
                            'Upper CI': res.get(f'{prefix}_upper_ci', np.nan)
                        }
                        long_rows.append(row)

                df_long = pd.DataFrame(long_rows)
                f.write(df_long.to_markdown(index=False))
                f.write("\n\n")

                # LWP Accuracy Table
                f.write("## LWP Accuracy (Python vs R)\n")
                accuracy_rows = []
                for res in self.results:
                    accuracy_rows.append({
                        'Test ID': res.get('test_id'),
                        'Slope Error': res.get('slope_error'),
                        'Slope % Error': res.get('slope_pct_error')
                    })
                df_acc = pd.DataFrame(accuracy_rows)
                f.write(df_acc.to_markdown(index=False))
                f.write("\n")

        print(f"Report saved to {report_path}")
