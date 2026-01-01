import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datetime import datetime

# Add repo root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import MannKS as mk

class ValidationUtils:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.results = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def create_report(self, filename='README.md', description=None):
        report_path = os.path.join(self.output_dir, filename)
        with open(report_path, 'w') as f:
            f.write(f"# Validation Report: V-20 Seasonality Check\n\n")
            if description:
                f.write(description + "\n\n")

            f.write("## Results\n\n")
            if self.results:
                df = pd.DataFrame(self.results)
                try:
                    import tabulate
                    f.write(df.to_markdown(index=False))
                except ImportError:
                    f.write(df.to_string(index=False))
                f.write("\n\n")
        print(f"Report saved to {report_path}")

def generate_seasonal_data(n_months=60, seasonal_amp=5.0, noise_std=1.0):
    dates = pd.date_range(start='2000-01-01', periods=n_months, freq='ME')
    t = np.arange(n_months)
    # Strong seasonal pattern
    seasonality = seasonal_amp * np.sin(2 * np.pi * dates.month / 12.0)
    values = 10 + 0.05 * t + seasonality + np.random.normal(0, noise_std, n_months)
    return pd.DataFrame({'date': dates, 'value': values})

def generate_non_seasonal_data(n_months=60, noise_std=1.0):
    dates = pd.date_range(start='2000-01-01', periods=n_months, freq='ME')
    t = np.arange(n_months)
    values = 10 + 0.05 * t + np.random.normal(0, noise_std, n_months)
    return pd.DataFrame({'date': dates, 'value': values})

def run():
    utils = ValidationUtils(os.path.dirname(__file__))

    # 1. Strong Seasonality
    print("\n--- Testing Strong Seasonality ---")
    df_seasonal = generate_seasonal_data(seasonal_amp=10.0, noise_std=1.0)
    result_seasonal = mk.check_seasonality(df_seasonal['value'], pd.to_datetime(df_seasonal['date']), alpha=0.05)

    # Corrected attribute names: .p_value, .h_statistic
    print(f"Result (Seasonal Data): Is Seasonal? {result_seasonal.is_seasonal} (p={result_seasonal.p_value:.4f})")

    utils.results.append({
        'Test Case': 'Strong Seasonality',
        'Expected': True,
        'Detected': result_seasonal.is_seasonal,
        'P-Value': result_seasonal.p_value,
        'KW-Statistic': result_seasonal.h_statistic,
        'Result': 'PASS' if result_seasonal.is_seasonal else 'FAIL'
    })

    # 2. No Seasonality
    print("\n--- Testing No Seasonality ---")
    df_non_seasonal = generate_non_seasonal_data(noise_std=2.0)
    result_non_seasonal = mk.check_seasonality(df_non_seasonal['value'], pd.to_datetime(df_non_seasonal['date']), alpha=0.05)

    print(f"Result (Non-Seasonal Data): Is Seasonal? {result_non_seasonal.is_seasonal} (p={result_non_seasonal.p_value:.4f})")

    utils.results.append({
        'Test Case': 'No Seasonality',
        'Expected': False,
        'Detected': result_non_seasonal.is_seasonal,
        'P-Value': result_non_seasonal.p_value,
        'KW-Statistic': result_non_seasonal.h_statistic,
        'Result': 'PASS' if not result_non_seasonal.is_seasonal else 'FAIL'
    })

    # 3. Check with Aggregation Method 'lwp' (should work same as default 'median' for complete data)
    print("\n--- Testing Seasonality with LWP Aggregation ---")
    result_lwp = mk.check_seasonality(
        df_seasonal['value'],
        pd.to_datetime(df_seasonal['date']),
        alpha=0.05,
        agg_method='lwp',
        agg_period='month'  # Added required parameter
    )

    utils.results.append({
        'Test Case': 'Strong Seasonality (LWP Mode)',
        'Expected': True,
        'Detected': result_lwp.is_seasonal,
        'P-Value': result_lwp.p_value,
        'KW-Statistic': result_lwp.h_statistic,
        'Result': 'PASS' if result_lwp.is_seasonal else 'FAIL'
    })

    utils.create_report(description="Verifies the `check_seasonality` function using the Kruskal-Wallis test.")

if __name__ == "__main__":
    np.random.seed(42)
    run()
