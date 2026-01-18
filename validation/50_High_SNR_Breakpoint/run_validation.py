
import os
import sys
import numpy as np
import argparse

# Add parent dir to sys.path to import common_validation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_validation import run_validation_suite

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_data(seed=None):
    if seed is not None:
        np.random.seed(seed)

    n_points = 100
    t = np.linspace(0, 100, n_points)

    # 0, 1, or 2 breakpoints
    n_bp = np.random.choice([0, 1, 2], p=[0.2, 0.4, 0.4])

    bps = []
    if n_bp == 1:
        bps = [np.random.uniform(25, 75)]
    elif n_bp == 2:
        b1 = np.random.uniform(20, 60)
        b2 = np.random.uniform(b1 + 20, 90)
        bps = [b1, b2]

    # Slopes
    slopes = [np.random.uniform(-0.5, 0.5)]
    for _ in range(n_bp):
        # High SNR change
        # Noise sigma will be 1.0. Change should be detectable.
        delta = np.random.uniform(0.2, 0.6) * np.random.choice([-1, 1])
        slopes.append(slopes[-1] + delta)

    # Generate Y (Continuous)
    y = np.zeros_like(t)
    # Seg 0
    y = 10 + slopes[0] * t

    for i, bp in enumerate(bps):
        mask = t >= bp
        # Add contribution of slope change
        # delta_slope * (t - bp)
        delta = slopes[i+1] - slopes[i]
        y[mask] += delta * (t[mask] - bp)

    # Noise (High SNR)
    # Signal range ~ 50. Noise 1.0. SNR 50.
    y += np.random.normal(0, 1.0, n_points)

    return t, y, n_bp, bps, slopes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations")
    args = parser.parse_args()

    # Just setup, don't run full suite
    print(f"Script setup complete. Ready to run with {args.iterations} iterations.")
    # Uncomment to run
    run_validation_suite(generate_data, OUTPUT_DIR, n_iterations=args.iterations)
