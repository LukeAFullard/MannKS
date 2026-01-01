
import os
import sys
import pandas as pd
import numpy as np

# Ensure correct package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import MannKS as mk

def run_v22():
    print("Running V-22: Data Inspection Validation")

    # Set seed for reproducibility
    np.random.seed(42)

    output_dir = os.path.dirname(__file__)
    output_file = os.path.join(output_dir, 'README.md')
    plot_file = os.path.join(output_dir, 'inspection_plot.png')

    # 1. Generate Synthetic Data
    # --------------------------
    # Monthly data for 2 years (24 points)
    dates = pd.date_range(start='2020-01-01', periods=24, freq='ME')
    values = np.linspace(10, 20, 24) + np.random.normal(0, 1, 24)

    # Introduce censoring
    # Values < 12 are left-censored (<12)
    # Values > 18 are right-censored (>18)
    data = []
    for d, v in zip(dates, values):
        if v < 12:
            data.append(f"<{12}")
        elif v > 18:
            data.append(f">{18}")
        else:
            data.append(v)

    # Create DataFrame
    df = pd.DataFrame({'date': dates, 'value': data})

    # 2. Pre-process Data (Required for inspection)
    # ---------------------------------------------
    # inspect_trend_data expects raw data, but let's see if we can pass the processed DF directly
    # or if we pass the raw columns.
    # The docstring says: "x (pd.DataFrame): Input data containing 'value' and 'censored' columns
    # (output of prepare_censored_data)."
    # So we MUST run prepare_censored_data first.

    df_processed = mk.prepare_censored_data(df['value'])
    df_processed['date'] = df['date'] # Add date back

    # 3. Run Inspection
    # -----------------
    print("Running inspect_trend_data...")
    inspection_df = mk.inspect_trend_data(
        df_processed,
        time_col='date',
        value_col='value',
        plot=True,
        plot_path=plot_file
    )

    # 4. Generate Report
    # ------------------
    with open(output_file, 'w') as f:
        f.write("# V-22: Data Inspection Validation\n\n")
        f.write("This document validates the `inspect_trend_data` function using a synthetic dataset with mixed censoring.\n\n")

        f.write("## 1. Input Data Summary\n\n")
        f.write(f"- Total Points: {len(df)}\n")
        f.write(f"- Date Range: {df['date'].min()} to {df['date'].max()}\n")
        f.write(f"- Censored Values: {df_processed['censored'].sum()}\n\n")

        f.write("## 2. Inspection Output (First 5 Rows)\n\n")
        f.write(inspection_df.head().to_markdown(index=False))
        f.write("\n\n")

        f.write("## 3. Inspection Plot\n\n")
        if os.path.exists(plot_file):
            f.write(f"![Inspection Plot](inspection_plot.png)\n")
        else:
            f.write("ERROR: Plot file was not generated.\n")

    print(f"Validation complete. Report saved to {output_file}")

if __name__ == "__main__":
    run_v22()
