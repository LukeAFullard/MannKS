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

    def run_lwp_r_script(self, df: pd.DataFrame, seasonal: bool = False) -> Dict:
        if ro is None: return {'slope': np.nan, 'p_value': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
        try:
            ro.r(f'source("{self.lwp_script_path}")')
            r_df = self._prepare_r_dataframe(df)
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
                 ro.r('df_r$TimeIncr <- df_r$Month')

            if not seasonal:
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

    def run_comparison(self, test_id: str, df: pd.DataFrame, scenario_name: str, mk_kwargs: Dict = {}, lwp_mode_kwargs: Dict = {}, true_slope: float = None, seasonal: bool = False) -> Tuple[Dict, object]:
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

        r_res = self.run_lwp_r_script(df, seasonal=seasonal)
        if t_datetime is not None:
             t_ats = t_datetime
        else:
             t_ats = t_numeric if t_numeric is not None else t_datetime

        if seasonal:
             mk_ats = mk.seasonal_trend_test(x_std, t_ats, sens_slope_method='ats', **mk_kwargs)
        else:
             mk_ats = mk.trend_test(x_std, t_ats, sens_slope_method='ats', slope_scaling='year')

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

def generate_censored_data(n=24, slope=1.0, noise_std=0.5, start_year=2000, censor_pct=0.3):
    dates = []
    current_date = datetime(start_year, 1, 15)
    for i in range(n):
        dates.append(current_date)
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 15)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 15)
    t = np.arange(n)
    noise = np.random.normal(0, noise_std, n)
    base_values = slope * t + 10 + noise
    threshold = np.percentile(base_values, (1 - censor_pct) * 100)
    final_values = []
    cen_type = []
    for v in base_values:
        if v > threshold:
            final_values.append(f">{threshold:.1f}")
            cen_type.append('gt')
        else:
            final_values.append(v)
            cen_type.append('not')
    return pd.DataFrame({'date': dates, 'value': final_values, 'cen_type': cen_type})

def run():
    output_dir = os.path.dirname(__file__)
    utils = ValidationUtils(output_dir)
    scenarios = []

    lwp_kwargs = {
        'mk_test_method': 'lwp',
        'ci_method': 'lwp',
        'sens_slope_method': 'lwp',
        'lt_mult': 0.5,
        'gt_mult': 1.1,
    }

    print("Generating Scenario 1: Strong Increasing")
    df_strong = generate_censored_data(n=48, slope=2.0, noise_std=1.0, censor_pct=0.3)
    _, mk_std_strong = utils.run_comparison(
        test_id="V-13",
        df=df_strong,
        scenario_name="strong_increasing",
        lwp_mode_kwargs=lwp_kwargs,
        true_slope=2.0
    )
    scenarios.append({
        'df': df_strong,
        'title': 'Strong Increasing (Right Censored)',
        'mk_result': mk_std_strong
    })

    print("Generating Scenario 2: Weak Decreasing")
    df_weak = generate_censored_data(n=48, slope=-0.2, noise_std=1.0, censor_pct=0.3)
    _, mk_std_weak = utils.run_comparison(
        test_id="V-13",
        df=df_weak,
        scenario_name="weak_decreasing",
        lwp_mode_kwargs=lwp_kwargs,
        true_slope=-0.2
    )
    scenarios.append({
        'df': df_weak,
        'title': 'Weak Decreasing (Right Censored)',
        'mk_result': mk_std_weak
    })

    print("Generating Scenario 3: Stable")
    df_stable = generate_censored_data(n=48, slope=0.0, noise_std=1.0, censor_pct=0.3)
    _, mk_std_stable = utils.run_comparison(
        test_id="V-13",
        df=df_stable,
        scenario_name="stable",
        lwp_mode_kwargs=lwp_kwargs,
        true_slope=0.0
    )
    scenarios.append({
        'df': df_stable,
        'title': 'Stable (Right Censored)',
        'mk_result': mk_std_stable
    })

    utils.generate_combined_plot(scenarios, "v13_combined.png", "V-13: LWP Censored Compatibility Analysis")

    description = """
# Validation Case V-13: Censored LWP Compatibility Modes

This validation case focuses on verifying the "LWP Compatibility Mode" of the `MannKS` package against the original LWP-TRENDS R script, specifically for **right-censored** data.

The goal is to demonstrate that setting parameters `mk_test_method='lwp'` and `sens_slope_method='lwp'` allows Python to accurately replicate the R script's handling of censored values.

## Methodology
- **Data:** 4 years of monthly data (n=48).
- **Censoring:** Approximately 30% of data is **right-censored** (values above a threshold are marked as `>Threshold`).
    """
    utils.create_report(description=description)

if __name__ == "__main__":
    np.random.seed(123)
    run()
