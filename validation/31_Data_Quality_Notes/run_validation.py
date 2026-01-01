import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime

# Add repo root to path to ensure MannKS can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk
from MannKS.analysis_notes import get_analysis_note, get_sens_slope_analysis_note

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
        self.results = []

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def run_data_quality_check(self, test_id: str, df: pd.DataFrame, expected_note: str, **kwargs) -> Dict:
        """
        Runs the trend test and verifies if the expected analysis note is present.
        """
        print(f"Running data quality check: {test_id} - Expecting: '{expected_note}'")

        # Prepare data
        values = df['value']
        if values.dtype == object:
             values = mk.prepare_censored_data(values)

        # Run MannKS Trend Test
        try:
            if 'date' in df.columns:
                t = pd.to_datetime(df['date'])
                result = mk.trend_test(values, t, slope_scaling='year', **kwargs)
            else:
                t = np.arange(len(df))
                result = mk.trend_test(values, t, **kwargs)

            notes = result.analysis_notes
            notes_str = "; ".join(notes)
        except Exception as e:
            notes = [str(e)]
            notes_str = str(e)
            result = None

        # Check if expected note matches any of the returned notes
        # The expected note might be a substring match
        match = any(expected_note in n for n in notes) if notes else False

        if not match:
             print(f"  FAILED: Expected '{expected_note}' but got: {notes}")
        else:
             print(f"  PASSED")

        res_row = {
            'Test ID': test_id,
            'Expected Note': expected_note,
            'Actual Notes': notes_str,
            'Match': match,
            'Slope': result.slope if result else np.nan,
            'P-Value': result.p if result else np.nan
        }
        self.results.append(res_row)
        return res_row

    def create_report(self, filename='README.md'):
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            f.write(f"# V-31 Data Quality Analysis Notes Validation\n\n")
            f.write("This validation case explicitly triggers and verifies the data quality warnings defined in `analysis_notes.py`.\n\n")

            f.write("## Results\n")
            if self.results:
                df = pd.DataFrame(self.results)
                try:
                    f.write(df.to_markdown(index=False))
                except ImportError:
                    f.write(df.to_string(index=False))
        print(f"Report saved to {report_path}")

def run():
    utils = ValidationUtils(os.path.dirname(__file__))

    # Case 1: All NA values
    df_all_na = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=10, freq='YE'),
        'value': [np.nan] * 10
    })
    utils.run_data_quality_check("V-31_All_NA", df_all_na, "Data all NA values")

    # Case 2: Insufficient unique values (< 3 unique)
    df_low_unique = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=10, freq='YE'),
        'value': [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    })
    utils.run_data_quality_check("V-31_Low_Unique", df_low_unique, "< 3 unique values")

    # Case 3: Insufficient non-censored values (< 5 non-censored)
    df_low_nc = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=10, freq='YE'),
        'value': [10.0, 11.0, 12.0, 13.0, '<5', '<5', '<5', '<5', '<5', '<5']
    })
    utils.run_data_quality_check("V-31_Low_NonCensored", df_low_nc, "< 5 Non-censored values")

    # Case 4: Long run of single value (> 50% identical)
    df_long_run = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=10, freq='YE'),
        'value': [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 2.0, 3.0, 4.0]
    })
    utils.run_data_quality_check("V-31_Long_Run", df_long_run, "Long run of single value")

    # Case 5: Sen slope based on censored values (Left Censored)
    # Using 'lwp' method to ensure censored slopes are considered (as 0) if they affect median.
    df_sen_lt = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=4, freq='YE'),
        'value': ['<5', '<5', '10', '12']
    })
    utils.run_data_quality_check("V-31_Sen_Slope_LeftCensored", df_sen_lt, "influenced by left-censored values", sens_slope_method='lwp')

    # Case 6: Sen slope based on tied non-censored values
    df_tied_sen = pd.DataFrame({
        'date': pd.date_range(start='2000-01-01', periods=5, freq='YE'),
        'value': [10.0, 10.0, 10.0, 10.0, 12.0]
    })
    utils.run_data_quality_check("V-31_Sen_Slope_Tied", df_tied_sen, "based on tied non-censored values")

    # Case 8: Critical - Sen slope based on two censored values
    # Must use sens_slope_method='lwp' to treat ambiguous slopes as 0 instead of NaN.
    df_crit_cen = pd.DataFrame({
        'value': ['<5', '<5', '<5', '<5', '100']
    })
    utils.run_data_quality_check("V-31_Critical_Censored", df_crit_cen, "CRITICAL: Sen slope is based on a pair of two censored values", sens_slope_method='lwp')

    utils.create_report()

if __name__ == "__main__":
    run()
