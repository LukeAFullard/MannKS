
import os
import sys
import numpy as np

# Add parent dir to sys.path to import common_validation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_validation import run_validation_suite

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Re-use High SNR generator but this is the comparison folder (46)
def generate_data(seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_points = 100
    t = np.linspace(0, 100, n_points)
    n_bp = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])

    bps = []
    if n_bp == 1:
        bps = [np.random.uniform(25, 75)]
    elif n_bp == 2:
        b1 = np.random.uniform(20, 60)
        b2 = np.random.uniform(b1 + 20, 90)
        bps = [b1, b2]

    slopes = [np.random.uniform(-0.5, 0.5)]
    for _ in range(n_bp):
        delta = np.random.uniform(0.2, 0.6) * np.random.choice([-1, 1])
        slopes.append(slopes[-1] + delta)

    y = np.zeros_like(t)
    y = 10 + slopes[0] * t
    for i, bp in enumerate(bps):
        mask = t >= bp
        delta = slopes[i+1] - slopes[i]
        y[mask] += delta * (t[mask] - bp)
    y += np.random.normal(0, 1.0, n_points)

    return t, y, n_bp, bps

if __name__ == "__main__":
    print("Script setup complete. Ready to run.")
    run_validation_suite(generate_data, OUTPUT_DIR, n_iterations=100)
