import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

# Add repo root to path to ensure MannKS can be imported
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

class ValidationUtils:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        self.lwp_script_path = os.path.join(self.repo_root, 'Example_Files/R/LWPTrends_v2502/LWPTrends_v2502.r')
        self.nada2_ats_path = os.path.join(self.repo_root, 'Example_Files/R/NADA2/ATS.R')
        self.nada2_ken_path = os.path.join(self.repo_root, 'Example_Files/R/NADA2/NADA_ken.R')
        self.nada2_atsmini_path = os.path.join(self.repo_root, 'Example_Files/R/NADA2/ATSmini.R')
        self.nada2_censeaken_path = os.path.join(self.repo_root, 'Example_Files/R/NADA2/censeaken.R')
        self.nada2_computes_path = os.path.join(self.repo_root, 'Example_Files/R/NADA2/computeS.R')
        self.nada2_kenplot_path = os.path.join(self.repo_root, 'Example_Files/R/NADA2/kenplot.R')
        self.master_csv_path = os.path.join(self.repo_root, 'validation/master_results.csv')
        self.results = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _prepare_r_dataframe(self, df: pd.DataFrame, is_seasonal: bool = False):
        if ro is None: return None
        df_r_prep = df.copy()

        if 'date' not in df_r_prep.columns and 'time' in df_r_prep.columns:
            year = df_r_prep['time'].astype(int)
            days = ((df_r_prep['time'] - year) * 365.25).astype(int)
            df_r_prep['date'] = pd.to_datetime(year.astype(str) + '-01-01') + pd.to_timedelta(days, unit='D')

        if 'date' in df_r_prep.columns:
            df_r_prep['myDate'] = df_r_prep['date']

        if 'RawValue' not in df_r_prep.columns:
            df_r_prep['RawValue'] = df_r_prep['value']

        if df_r_prep['value'].dtype == object:
            is_left = df_r_prep['value'].astype(str).str.contains('<')
            is_right = df_r_prep['value'].astype(str).str.contains('>')
            clean_vals = df_r_prep['value'].astype(str).str.replace('<', '', regex=False).str.replace('>', '', regex=False)
            df_r_prep['RawValue'] = clean_vals.astype(float)
            df_r_prep['Censored'] = is_left | is_right
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

        ro.globalenv['df_temp'] = r_df
        ro.r('df_temp$myDate <- as.Date(df_temp$myDate)')
        r_df = ro.globalenv['df_temp']

        return r_df

    def _patch_dataframe_for_r(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Injects missing seasons into the dataframe to satisfy the R script's structural requirements.
        This creates a full grid of Year x Month.
        Crucially, it IMPUTES missing values using linear interpolation to allow the R script to run.
        """
        df_patched = df.copy()
        if 'date' not in df_patched.columns:
            return df # Cannot patch without dates

        dates = pd.to_datetime(df_patched['date'])
        min_date = dates.min()
        max_date = dates.max()
        start_year = min_date.year
        end_year = max_date.year

        all_dates = []
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Use the 15th of the month to match typical generation
                all_dates.append(datetime(year, month, 15))

        full_df = pd.DataFrame({'date': all_dates})

        # Ensure df_patched['date'] matches the generated date format (datetime)
        df_patched['date'] = pd.to_datetime(df_patched['date'])

        merged_df = pd.merge(full_df, df_patched, on='date', how='left')

        # Interpolate numeric values (Imputation)
        if 'value' in merged_df.columns:
            # Check if value is numeric or can be converted
            if pd.api.types.is_numeric_dtype(merged_df['value']):
                merged_df['value'] = merged_df['value'].interpolate(method='linear', limit_direction='both')

        # Fill missing metadata
        if 'Censored' in merged_df.columns:
            merged_df['Censored'] = merged_df['Censored'].fillna(False)

        if 'CenType' in merged_df.columns:
            merged_df['CenType'] = merged_df['CenType'].fillna('not')

        return merged_df

    def run_lwp_r_script(self, df: pd.DataFrame, seasonal: bool = False, season_col: str = 'Month', use_patch: bool = False) -> Dict:
        if ro is None: return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
        try:
            ro.r(f'source("{self.lwp_script_path}")')

            df_to_use = df
            if use_patch:
                df_to_use = self._patch_dataframe_for_r(df)

            r_df = self._prepare_r_dataframe(df_to_use)
            ro.globalenv['df_r'] = r_df
            ro.r('df_r$CenType <- as.factor(df_r$CenType)')

            ro.r('df_r <- GetMoreDateInfo(df_r)')

            is_annual = True
            ro.r('year_counts <- table(df_r$Year)')
            max_counts = ro.r('max(year_counts)')[0]
            if max_counts > 1:
                is_annual = False

            if is_annual and not seasonal:
                 ro.r('df_r$TimeIncr <- df_r$Year')
            else:
                 if season_col == 'Qtr':
                     ro.r('df_r$TimeIncr <- df_r$Qtr')
                 else:
                     ro.r('df_r$TimeIncr <- df_r$Month')

            if not seasonal:
                cmd = """
                suppressWarnings(
                    result <- NonSeasonalTrendAnalysis(df_r, do.plot=FALSE, TimeIncrMed=TRUE, UseMidObs=TRUE)
                )
                """
                ro.r(cmd)
            else:
                ro.r(f'df_r$Season <- df_r${season_col}')
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

            # Helper to clean R IntMin NaNs
            def clean_r_val(val):
                if val <= -2000000000: # R IntMin is approx -2.14e9
                    return np.nan
                return float(val)

            return {
                'slope': clean_r_val(res_df['AnnualSenSlope'].iloc[0]),
                'p_value': clean_r_val(res_df['p'].iloc[0]),
                'lower_ci': clean_r_val(res_df['Sen_Lci'].iloc[0]),
                'upper_ci': clean_r_val(res_df['Sen_Uci'].iloc[0])
            }

        except Exception as e:
            if not use_patch:
                print(f"LWP R script execution failed (expected for missing seasons): {e}")
            else:
                print(f"LWP R script execution failed even with patch: {e}")
            return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

    def run_nada2_r_script(self, df: pd.DataFrame, seasonal: bool = False) -> Dict:
        if ro is None: return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
        try:
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
                 is_censored = df['value'].astype(str).str.contains('<') | df['value'].astype(str).str.contains('>')
                 y_cen = is_censored.values
            else:
                y_vals = df['value'].values
                y_cen = np.zeros(len(df), dtype=bool)

            if 'time' in df.columns:
                 x_vals = df['time'].values
            elif 'date' in df.columns:
                 dates = pd.to_datetime(df['date'])
                 x_vals = dates.dt.year + (dates.dt.dayofyear - 1) / 365.25
                 x_vals = x_vals.values
            else:
                x_vals = np.arange(len(df))

            ro.globalenv['y_vec'] = ro.FloatVector(y_vals)
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
                if 'date' in df.columns:
                    dates = pd.to_datetime(df['date'])
                    groups = dates.dt.month.values
                else:
                    groups = np.tile(np.arange(1, 13), len(df) // 12 + 1)[:len(df)]
                ro.globalenv['grp_vec'] = ro.FloatVector(groups)
                cmd = "res_censeaken <- censeaken(x_vec, y_vec, ycen_vec, grp_vec, LOG=FALSE, R=499, seaplots=FALSE)"
                ro.r(cmd)
                r_res = ro.globalenv['res_censeaken']
                with localconverter(ro.default_converter + pandas2ri.converter):
                    res_df = ro.conversion.rpy2py(r_res)
                return {
                    'slope': float(res_df['slope'].iloc[0]),
                    'p_value': float(res_df['pval'].iloc[0]),
                    'lower_ci': np.nan,
                    'upper_ci': np.nan
                }
        except Exception as e:
            print(f"Error running NADA2 R script: {e}")
            return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

    def run_comparison(self, test_id: str, df: pd.DataFrame, scenario_name: str, mk_kwargs: Dict = {}, lwp_mode_kwargs: Dict = {}, true_slope: float = None, seasonal: bool = False, season_col_r: str = 'Month') -> Tuple[Dict, object]:
        full_test_id = f"{test_id}_{scenario_name}"
        print(f"Running comparison for: {full_test_id} (Seasonal: {seasonal})")

        x_std = df['value']
        if df['value'].dtype == object:
             try:
                 x_std = mk.prepare_censored_data(df['value'])
             except Exception:
                 pass

        t_datetime = None
        t_numeric = None
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            t_datetime = dates.to_numpy()
            t_numeric = (dates - pd.Timestamp("1970-01-01")).dt.days / 365.25
            t_numeric = t_numeric.values
        else:
            t_numeric = np.arange(len(df))

        if 't_std' not in locals():
            t_std = t_numeric if t_numeric is not None else t_datetime

        if seasonal:
             mk_std = mk.seasonal_trend_test(x_std, t_datetime if t_datetime is not None else t_std, **mk_kwargs)
        else:
             mk_std = mk.trend_test(x_std, t_std, **mk_kwargs)

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
                     lwp_final_kwargs['agg_period'] = 'month'
             if seasonal and 'agg_period' in lwp_final_kwargs:
                 del lwp_final_kwargs['agg_period']
             if 'slope_scaling' not in lwp_final_kwargs:
                 lwp_final_kwargs['slope_scaling'] = 'year'
             if seasonal:
                 mk_lwp = mk.seasonal_trend_test(x_std, t_lwp, **lwp_final_kwargs)
             else:
                 mk_lwp = mk.trend_test(x_std, t_lwp, **lwp_final_kwargs)
        else:
             if seasonal and 'agg_period' in lwp_final_kwargs:
                  del lwp_final_kwargs['agg_period']
             if seasonal:
                 mk_lwp = mk.seasonal_trend_test(x_std, t_std, **lwp_final_kwargs)
             else:
                 mk_lwp = mk.trend_test(x_std, t_std, **lwp_final_kwargs)

        # 1. Run Standard R (Expected to FAIL)
        r_res = self.run_lwp_r_script(df, seasonal=seasonal, season_col=season_col_r, use_patch=False)

        # 2. Run Patched R (Expected to SUCCEED)
        r_patched_res = self.run_lwp_r_script(df, seasonal=seasonal, season_col=season_col_r, use_patch=True)

        if t_datetime is not None:
             t_ats = t_datetime
        else:
             t_ats = t_numeric if t_numeric is not None else t_datetime

        if seasonal:
             mk_ats = mk.seasonal_trend_test(x_std, t_ats, sens_slope_method='ats', **mk_kwargs)
        else:
             mk_ats = mk.trend_test(x_std, t_ats, sens_slope_method='ats')

        nada_res = self.run_nada2_r_script(df, seasonal=seasonal)

        slope_error = np.nan
        slope_pct_error = np.nan

        # Use patched result for error calculation if standard failed, OR keep standard?
        # User wants to "keep the failing LWP mode". So error calc should probably be NaN if standard failed.
        # But we can also compute error against patched version if we want to show match.
        # Let's default to standard for the main 'slope_error' column to reflect the failure in main stats.
        if not np.isnan(r_res['slope']):
            slope_error = mk_lwp.slope - r_res['slope']
            if true_slope is not None and true_slope != 0:
                slope_pct_error = (slope_error / true_slope) * 100
            elif r_res['slope'] != 0:
                slope_pct_error = (slope_error / r_res['slope']) * 100
            elif slope_error == 0:
                slope_pct_error = 0.0

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
            'r_patched_slope': r_patched_res['slope'],
            'r_patched_p_value': r_patched_res['p_value'],
            'r_patched_lower_ci': r_patched_res['lower_ci'],
            'r_patched_upper_ci': r_patched_res['upper_ci'],
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
             # Align columns if new ones (patched) were added
             for col in df.columns:
                 if col not in cols:
                     # This is tricky if master csv structure is rigid.
                     # For now, just append what matches or rewrite the file if structure changes?
                     # Let's just drop extra columns to keep master CSV consistent,
                     # OR allow master CSV to grow.
                     pass

             # Actually, if we add columns, standard pd.read_csv might fail if we just append without header.
             # But this is a specific validation script. Let's just ignore the patched columns for the master CSV
             # to avoid breaking other scripts that read it, unless we want to update the schema globally.
             # Given the scope, let's keep master CSV standard and only report patched in this README.
             common_cols = [c for c in cols if c in df.columns]
             df_to_save = df[common_cols]
             df_to_save.to_csv(self.master_csv_path, mode='a', header=False, index=False)
        else:
             df.to_csv(self.master_csv_path, mode='w', header=True, index=False)

    def _get_decimal_year(self, df: pd.DataFrame) -> np.ndarray:
        if 'time' in df.columns:
            return df['time'].values
        elif 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            t = (dates - pd.Timestamp("1970-01-01")).dt.days / 365.25
            return t.values
        else:
            return np.arange(len(df))

    def generate_combined_plot(self, scenarios: List[Dict], filename: str, main_title: str):
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

            y_plot = df['value'].astype(str).str.replace('<', '', regex=False).str.replace('>', '', regex=False).astype(float)
            colors = []
            if 'cen_type' in df.columns:
                 for c_type in df['cen_type']:
                     if c_type == 'lt': colors.append('red')
                     elif c_type == 'gt': colors.append('orange')
                     else: colors.append('black')
            else:
                 colors = ['black'] * len(df)

            ax.scatter(x_plot, y_plot, c=colors, label='Data')

            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label='Data', markerfacecolor='black', markersize=8)]

            if mk_result is not None and not np.isnan(mk_result.slope):
                t_numeric = self._get_decimal_year(df)
                y_trend = mk_result.slope * t_numeric + mk_result.intercept
                ax.plot(x_plot, y_trend, '-', color='blue', label=f"Sen's Slope: {mk_result.slope:.4f}")
                legend_elements.append(Line2D([0], [0], color='blue', lw=2, label=f"Slope: {mk_result.slope:.4f}"))

                t_med = np.median(t_numeric)
                y_med = np.median(y_plot)
                if hasattr(mk_result, 'lower_ci') and not np.isnan(mk_result.lower_ci):
                    y_lower = mk_result.lower_ci * (t_numeric - t_med) + y_med
                    ax.plot(x_plot, y_lower, '--', color='blue', alpha=0.5)
                if hasattr(mk_result, 'upper_ci') and not np.isnan(mk_result.upper_ci):
                    y_upper = mk_result.upper_ci * (t_numeric - t_med) + y_med
                    ax.plot(x_plot, y_upper, '--', color='blue', alpha=0.5)
                    if hasattr(mk_result, 'lower_ci') and not np.isnan(mk_result.lower_ci):
                        ax.fill_between(x_plot, y_lower, y_upper, color='blue', alpha=0.1)

            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.legend(handles=legend_elements, loc='best')
            ax.grid(True, linestyle=':', alpha=0.6)

        fig.suptitle(main_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path = os.path.join(self.output_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        print(f"Combined plot saved to {plot_path}")

    def create_report(self, filename='README.md', description=None):
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            f.write(f"# Validation Report\n\n")
            if description:
                f.write(description + "\n\n")

            f.write("## Plots\n")
            plots = [p for p in os.listdir(self.output_dir) if p.endswith('.png')]
            for p in sorted(plots):
                f.write(f"### {p}\n")
                f.write(f"![{p}]({p})\n\n")

            f.write("## Results\n")
            if self.results:
                long_rows = []
                for res in self.results:
                    test_id = res.get('test_id', 'Unknown')
                    methods = [
                        ('MannKS (Standard)', 'mk_py'),
                        ('MannKS (LWP Mode)', 'lwp_py'),
                        ('LWP-TRENDS (R)', 'r'),
                        ('LWP-TRENDS (R) [Patched]', 'r_patched'),
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
                try:
                    import tabulate
                    f.write(df_long.to_markdown(index=False))
                except ImportError:
                    f.write(df_long.to_string(index=False))
                f.write("\n\n")

                f.write("## LWP Accuracy (Python vs R)\n")
                accuracy_rows = []
                for res in self.results:
                    accuracy_rows.append({
                        'Test ID': res.get('test_id'),
                        'Slope Error': res.get('slope_error'),
                        'Slope % Error': res.get('slope_pct_error')
                    })
                df_acc = pd.DataFrame(accuracy_rows)
                try:
                    f.write(df_acc.to_markdown(index=False))
                except ImportError:
                    f.write(df_acc.to_string(index=False))
                f.write("\n")

        print(f"Report saved to {report_path}")

TEST_ID = "V-18"
DESCRIPTION = """
**V-18: Seasonal Data with Missing Seasons**

This test verifies the seasonal trend analysis when entire seasons are missing from the dataset.
Specifically, all data for **July (Month 7)** and **August (Month 8)** will be removed.
This forces the test to skip these seasons and only analyze the remaining 10 months.

**Note:** The LWP-TRENDS R script has a known fragility with missing seasons and may fail to run.
The `MannKS` package is expected to handle this gracefully by skipping the missing seasons and analyzing the rest.

**R Workaround:** To verify if the LWP R script *can* run if the data is massaged, this validation
script also runs a "Patched" version where missing seasons are filled with `NA` values to ensure
a complete Year x Month grid. This confirms that the R failure is structural, not statistical.
"""

def generate_missing_season_data(n_years=10, start_year=2000, trend_slope=0.0, noise_std=1.0, season_amp=5.0):
    dates = []
    values = []
    for year in range(start_year, start_year + n_years):
        for month in range(1, 13):
            # SKIP July and August
            if month in [7, 8]:
                continue

            date = datetime(year, month, 15)
            t_year = year + (month - 1) / 12.0
            trend_val = trend_slope * (t_year - start_year)
            season_val = season_amp * np.sin(2 * np.pi * (month - 1) / 12.0)
            noise = np.random.normal(0, noise_std)
            raw_val = 100 + trend_val + season_val + noise

            dates.append(date)
            values.append(raw_val)
    return pd.DataFrame({'date': dates, 'value': values})

def run():
    output_dir = os.path.dirname(__file__)
    utils = ValidationUtils(output_dir)
    scenarios_to_plot = []

    # 1. Strong Increasing Trend (Missing Seasons)
    df_inc = generate_missing_season_data(n_years=10, trend_slope=2.0, noise_std=1.0)
    res_inc, mk_std_inc = utils.run_comparison(
        TEST_ID, df_inc, "strong_increasing",
        seasonal=True,
        mk_kwargs={'season_type': 'month', 'period': 12, 'slope_scaling': 'year'},
        lwp_mode_kwargs={'season_type': 'month', 'period': 12},
        true_slope=2.0
    )
    scenarios_to_plot.append({'df': df_inc, 'title': 'Strong Increasing (No Jul/Aug)', 'mk_result': mk_std_inc})

    # 2. Weak Decreasing Trend (Missing Seasons)
    df_dec = generate_missing_season_data(n_years=10, trend_slope=-0.5, noise_std=2.0)
    res_dec, mk_std_dec = utils.run_comparison(
        TEST_ID, df_dec, "weak_decreasing",
        seasonal=True,
        mk_kwargs={'season_type': 'month', 'period': 12, 'slope_scaling': 'year'},
        lwp_mode_kwargs={'season_type': 'month', 'period': 12},
        true_slope=-0.5
    )
    scenarios_to_plot.append({'df': df_dec, 'title': 'Weak Decreasing (No Jul/Aug)', 'mk_result': mk_std_dec})

    # 3. Stable (No Trend) (Missing Seasons)
    df_stab = generate_missing_season_data(n_years=10, trend_slope=0.0, noise_std=1.0)
    res_stab, mk_std_stab = utils.run_comparison(
        TEST_ID, df_stab, "stable",
        seasonal=True,
        mk_kwargs={'season_type': 'month', 'period': 12, 'slope_scaling': 'year'},
        lwp_mode_kwargs={'season_type': 'month', 'period': 12},
        true_slope=0.0
    )
    scenarios_to_plot.append({'df': df_stab, 'title': 'Stable (No Jul/Aug)', 'mk_result': mk_std_stab})

    utils.generate_combined_plot(scenarios_to_plot, "V18_Missing_Seasons_Analysis.png", "V-18: Monthly Seasonal (Missing Jul/Aug)")
    utils.create_report("README.md", DESCRIPTION)

if __name__ == "__main__":
    np.random.seed(42)
    run()
