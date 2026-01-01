
import os
import sys
import pandas as pd
from collections import namedtuple

# Ensure correct package import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import MannKS as mk

# Mock the TrendResult namedtuple structure used by classify_trend
# Based on the actual code, it needs 'h', 'C', and 'trend' in addition to what I had.
TrendResult = namedtuple('TrendResult', ['slope', 'p', 'lower_ci', 'upper_ci', 's', 'intercept', 'tau', 'z', 'var_s', 'analysis_notes', 'sen_probability', 'sen_probability_min', 'sen_probability_max', 'h', 'C', 'trend'])

def run_v21():
    print("Running V-21: Trend Classification Validation")

    output_file = os.path.join(os.path.dirname(__file__), 'README.md')

    # 1. Test Default Classification Logic
    # ------------------------------------
    # Mock results. Note: C = 1 - p/2 is the formula typically used.

    scenarios = [
        ("Strong Increase", TrendResult(
            slope=1.5, p=0.001, lower_ci=1.0, upper_ci=2.0, s=10, intercept=0, tau=0.8, z=3.0, var_s=1, analysis_notes=[], sen_probability=0.9995, sen_probability_min=0.99, sen_probability_max=1.0,
            h=True, C=0.9995, trend='increasing'
        )),
        ("Likely Increase", TrendResult(
            slope=0.5, p=0.15, lower_ci=0.1, upper_ci=1.0, s=5, intercept=0, tau=0.4, z=1.5, var_s=1, analysis_notes=[], sen_probability=0.925, sen_probability_min=0.90, sen_probability_max=0.95,
            h=True, C=0.925, trend='increasing'
        )),
        ("Possible Decrease (but h=True)", TrendResult(
            slope=-0.2, p=0.60, lower_ci=-0.5, upper_ci=0.1, s=-2, intercept=0, tau=-0.1, z=-0.5, var_s=1, analysis_notes=[], sen_probability=0.70, sen_probability_min=0.65, sen_probability_max=0.75,
            h=True, C=0.70, trend='decreasing'
        )),
        ("Not Significant (h=False)", TrendResult(
            slope=0.01, p=0.90, lower_ci=-0.1, upper_ci=0.1, s=0, intercept=0, tau=0.0, z=0.0, var_s=1, analysis_notes=[], sen_probability=0.55, sen_probability_min=0.50, sen_probability_max=0.60,
            h=False, C=0.55, trend='no trend'
        )),
    ]

    # 2. Test Custom Map
    # ------------------
    # Simple map: 0.90 -> "High Confidence", 0.0 -> "Low Confidence"
    custom_map = {
        0.90: "High Confidence",
        0.00: "Low Confidence"
    }

    with open(output_file, 'w') as f:
        f.write("# V-21: Trend Classification Validation\n\n")
        f.write("This document validates the `classify_trend` helper function using various synthetic trend results.\n\n")

        f.write("## 1. Default Classification Map (IPCC-based)\n\n")
        f.write("| Scenario | Slope | P-Value | C (Confidence) | h (Signif) | Trend | Classification |\n")
        f.write("|---|---|---|---|---|---|---|\n")

        for name, res in scenarios:
            classification = mk.classify_trend(res)
            f.write(f"| {name} | {res.slope} | {res.p} | {res.C:.4f} | {res.h} | {res.trend} | {classification} |\n")

        f.write("\n## 2. Custom Classification Map\n\n")
        f.write("Map: `{0.90: 'High Confidence', 0.0: 'Low Confidence'}`\n\n")
        f.write("| Scenario | Classification |\n")
        f.write("|---|---|\n")

        for name, res in scenarios:
            classification = mk.classify_trend(res, category_map=custom_map)
            f.write(f"| {name} | {classification} |\n")

    print(f"Validation complete. Report saved to {output_file}")

if __name__ == "__main__":
    run_v21()
