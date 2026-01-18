
import os
import sys
import numpy as np

# Add parent dir to sys.path to import common_validation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_validation import run_validation_suite

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_data(seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_points = 60
    t = np.linspace(0, 100, n_points)

    # 1 Breakpoint
    bp_true = np.random.uniform(40, 60)

    slope1 = 0.5
    slope2 = 1.5
    intercept = 10

    y = np.zeros_like(t)
    mask = t < bp_true
    y[mask] = intercept + slope1 * t[mask]
    y[~mask] = (intercept + slope1 * bp_true) + slope2 * (t[~mask] - bp_true)

    # Heteroscedastic Noise
    # Noise increases with t
    sigma = 0.5 + 2.0 * (t / 100.0)
    noise = np.random.normal(0, sigma, n_points)
    y += noise

    return t, y, 1, [bp_true], [slope1, slope2]

if __name__ == "__main__":
    print("Script setup complete.")
    run_validation_suite(generate_data, OUTPUT_DIR, n_iterations=30)
