
import os
import sys
import numpy as np
import pandas as pd
import time
import logging
import warnings
import piecewise_regression
# Updated import: Use public API for consistency with examples
from MannKS import segmented_trend_test

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup Logging
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'validation_log.txt')
    # Clear old log
    if os.path.exists(log_file):
        os.remove(log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

# --- 1. OLS Piecewise Implementation (piecewise-regression) ---
def _extract_piecewise_estimates(estimates, n_bp):
    """
    Extract sorted breakpoints and per-segment slopes from a piecewise_regression
    estimates dict.

    In piecewise_regression, ``alpha1..alpha_{n_bp+1}`` are the segment slopes
    directly, while ``beta1..beta_{n_bp}`` are the slope changes at each breakpoint.
    """
    bps = sorted(
        val['estimate']
        for key, val in estimates.items()
        if key.startswith('breakpoint') and isinstance(val, dict) and 'estimate' in val
    )
    slopes = [
        estimates.get(f'alpha{i}', {}).get('estimate', np.nan)
        for i in range(1, n_bp + 2)
    ]
    return bps, slopes


def fit_piecewise_ols(t, y, max_breakpoints=2):
    """
    Fits piecewise linear regression using the piecewise_regression library.
    Searches for 0, 1, ..., max_breakpoints and selects best via BIC.
    """
    n = len(y)
    best_bic = np.inf
    best_n = 0
    best_bps = []

    # 0 Breakpoints (Linear) - Manually calculate BIC as Fit doesn't support n_breakpoints=0
    # y = a + b*t
    A = np.vstack([t, np.ones(n)]).T
    try:
        sol, resid, _, _ = np.linalg.lstsq(A, y, rcond=None)
        if resid.size > 0:
            ssr = resid[0]
        else:
            # Perfect fit
            y_pred = sol[0] * t + sol[1]
            ssr = np.sum((y - y_pred) ** 2)

        k = 2
        ssr_eff = max(float(ssr), np.finfo(float).tiny)
        bic = n * np.log(ssr_eff / n) + k * np.log(n)

        best_bic = bic
        best_n = 0
        best_bps = []
    except Exception:
        pass

    # Loop for 1 to max_breakpoints
    best_slopes = []

    for n_bp in range(1, max_breakpoints + 1):
        try:
            pw_fit = piecewise_regression.Fit(t, y, n_breakpoints=n_bp, verbose=False)

            # Access results via best_muggeo as requested for robust estimate access
            bm = getattr(pw_fit, 'best_muggeo', None)
            best_fit = getattr(bm, 'best_fit', None) if bm is not None else None
            if best_fit is not None:
                # Use the library's calculated BIC directly if available
                if hasattr(best_fit, 'bic') and best_fit.bic is not None:
                    current_bic = best_fit.bic
                else:
                    rss = getattr(best_fit, 'residual_sum_squares', None)
                    if rss is None:
                        rss = getattr(best_fit, 'rss', np.inf)
                    k = 2 * n_bp + 2
                    rss_eff = max(float(rss), np.finfo(float).tiny)
                    current_bic = n * np.log(rss_eff / n) + k * np.log(n)

                if np.isfinite(current_bic) and current_bic < best_bic:
                    best_bic = current_bic
                    best_n = n_bp
                    best_bps, best_slopes = _extract_piecewise_estimates(
                        best_fit.estimates, n_bp
                    )
            else:
                # Fallback to standard get_results if best_muggeo is unavailable.
                res = pw_fit.get_results()
                current_bic = res.get('bic')
                if (
                    bool(res.get('converged', False))
                    and current_bic is not None
                    and np.isfinite(current_bic)
                    and current_bic < best_bic
                ):
                    best_bic = current_bic
                    best_n = n_bp
                    best_bps, best_slopes = _extract_piecewise_estimates(
                        res.get('estimates', {}), n_bp
                    )

        except Exception:
            # Convergence failure or other error
            continue

    if best_n == 0:
        # Get linear slope for N=0
        try:
            p = np.polyfit(t, y, 1)
            best_slopes = [p[0]]
        except Exception:
            best_slopes = [np.nan]

    return best_n, best_bps, best_slopes

# --- 2. Generic Run Function ---
def run_validation_suite(data_generator, output_dir, n_iterations=100):
    logger = setup_logging(output_dir)
    logger.info(f"Starting validation suite with {n_iterations} iterations.")

    results = []

    # Define methods
    methods = [
        ('Piecewise_Regression', 'ols', {}),
        ('MannKS_Hybrid', 'mannks_hybrid', {})
    ]

    start_total = time.time()

    for i in range(n_iterations):
        gen_data = data_generator(seed=42+i)
        t, x, true_n, true_bps = gen_data[:4]

        true_slopes = None
        if len(gen_data) >= 5:
            true_slopes = gen_data[4]

        iter_res = {
            'iter': i,
            'true_n': true_n,
            'true_bps': str(list(true_bps))
        }

        if i % 10 == 0:
            logger.info(f"Processing iteration {i}/{n_iterations}...")

        for name, m_type, kwargs in methods:
            t0 = time.time()
            pred_slopes = []
            try:
                if m_type == 'ols':
                    np.random.seed(42 + i)
                    pred_n, pred_bps, pred_slopes = fit_piecewise_ols(t, x, max_breakpoints=2)
                elif m_type == 'mannks_hybrid':
                    # Hybrid Segmented Trend via Public API
                    # Note: segmented_trend_test takes (x, t) order
                    result = segmented_trend_test(x, t, max_breakpoints=2, random_state=42 + i)

                    pred_n = result.n_breakpoints
                    pred_bps = list(result.breakpoints)

                    # result.segments is a DataFrame in the public API
                    if result.segments is not None and not result.segments.empty:
                        pred_slopes = result.segments['slope'].tolist()
                    else:
                        pred_slopes = []

                dur = time.time() - t0

                # Store
                iter_res[f'{name}_n'] = pred_n

                # Format bps to simple list string for CSV safety
                if isinstance(pred_bps, np.ndarray):
                    bps_list = pred_bps.tolist()
                elif isinstance(pred_bps, list):
                    bps_list = pred_bps
                else:
                    bps_list = []
                iter_res[f'{name}_bps'] = str(bps_list)

                iter_res[f'{name}_time'] = dur
                iter_res[f'{name}_correct_n'] = (pred_n == true_n)

                # Location Error (if N matches)
                if pred_n == true_n and true_n > 0:
                    # Match bps
                    if isinstance(pred_bps, list):
                        p = np.sort(pred_bps)
                    else:
                        p = np.sort(list(pred_bps))

                    gt = np.sort(true_bps)
                    if len(p) == len(gt):
                        err = np.mean(np.abs(p - gt))
                        iter_res[f'{name}_loc_error'] = err
                    else:
                         iter_res[f'{name}_loc_error'] = np.nan
                else:
                    iter_res[f'{name}_loc_error'] = np.nan

                # Slope Error (if N matches and true_slopes provided)
                if pred_n == true_n and true_slopes is not None:
                    # Slopes should match segment by segment if sorted by time
                    if len(pred_slopes) == len(true_slopes):
                        s_err = np.mean(np.abs(np.array(pred_slopes) - np.array(true_slopes)))
                        iter_res[f'{name}_slope_error'] = s_err
                    else:
                        iter_res[f'{name}_slope_error'] = np.nan
                else:
                    iter_res[f'{name}_slope_error'] = np.nan

            except Exception as e:
                logger.error(f"Error in {name} iter {i}: {e}")
                iter_res[f'{name}_n'] = -1
                iter_res[f'{name}_time'] = time.time() - t0
                # Ensure keys exist to avoid DataFrame errors later
                iter_res[f'{name}_correct_n'] = False
                iter_res[f'{name}_loc_error'] = np.nan
                iter_res[f'{name}_slope_error'] = np.nan

        results.append(iter_res)

        # Incremental save every 10
        if i % 10 == 0:
            pd.DataFrame(results).to_csv(os.path.join(output_dir, 'results_partial.csv'), index=False)

    total_time = time.time() - start_total
    logger.info(f"Completed in {total_time:.2f}s")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'results_final.csv'), index=False)

    generate_summary_report(df, output_dir, methods)
    return df

def generate_summary_report(df, output_dir, methods):
    lines = []
    lines.append("# Validation Report\n")

    # 1. Accuracy of N
    lines.append("## 1. Model Selection Accuracy (Correct N)\n")
    lines.append("| Method | Accuracy | Mean Time (s) |\n")
    lines.append("| :--- | :--- | :--- |\n")

    for name, _, _ in methods:
        col = f'{name}_correct_n'
        if col in df.columns:
            acc = df[col].mean()
            tm = df[f'{name}_time'].mean()
            lines.append(f"| {name} | {acc:.1%} | {tm:.4f} |\n")
        else:
            lines.append(f"| {name} | Error | Error |\n")

    # 2. Location Accuracy
    lines.append("\n## 2. Breakpoint Location Accuracy (MAE)\n")
    lines.append("| Method | Mean Error | Std Dev | Min | Max |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- |\n")

    for name, _, _ in methods:
        # Filter for correct detections
        if f'{name}_loc_error' in df.columns:
            errs = df[f'{name}_loc_error'].dropna()
            if len(errs) > 0:
                mean_e = errs.mean()
                std_e = errs.std()
                min_e = errs.min()
                max_e = errs.max()
                lines.append(f"| {name} | {mean_e:.4f} | {std_e:.4f} | {min_e:.4f} | {max_e:.4f} |\n")
            else:
                lines.append(f"| {name} | N/A | N/A | N/A | N/A |\n")
        else:
            lines.append(f"| {name} | N/A | N/A | N/A | N/A |\n")

    # 3. Slope Estimation Accuracy (MAE)
    lines.append("\n## 3. Slope Estimation Accuracy (MAE)\n")
    lines.append("| Method | Mean Error | Std Dev | Min | Max |\n")
    lines.append("| :--- | :--- | :--- | :--- | :--- |\n")

    for name, _, _ in methods:
        col = f'{name}_slope_error'
        if col in df.columns:
            errs = df[col].dropna()
            if len(errs) > 0:
                mean_e = errs.mean()
                std_e = errs.std()
                min_e = errs.min()
                max_e = errs.max()
                lines.append(f"| {name} | {mean_e:.4f} | {std_e:.4f} | {min_e:.4f} | {max_e:.4f} |\n")
            else:
                lines.append(f"| {name} | N/A | N/A | N/A | N/A |\n")
        else:
            lines.append(f"| {name} | N/A | N/A | N/A | N/A |\n")

    # 4. Contingency Tables (Confusion Matrix)
    lines.append("\n## 3. Confusion Matrix (True N vs Predicted N)\n")

    for name, _, _ in methods:
        lines.append(f"\n### {name}\n")
        if 'true_n' in df.columns and f'{name}_n' in df.columns:
            # Create crosstab
            ct = pd.crosstab(df['true_n'], df[f'{name}_n'])
            ct.index.name = "True N"
            ct.columns.name = "Pred N"

            # Convert to markdown manually or via to_markdown if available
            try:
                # Use simple formatting
                header = "| True N \\ Pred N | " + " | ".join(map(str, ct.columns)) + " |"
                sep = "| :--- | " + " | ".join(["---"] * len(ct.columns)) + " |"
                lines.append(header + "\n")
                lines.append(sep + "\n")

                for idx, row in ct.iterrows():
                    row_str = " | ".join(map(str, row.values))
                    lines.append(f"| **{idx}** | {row_str} |\n")
            except Exception as e:
                lines.append(f"Error generating table: {e}\n")
        else:
            lines.append("Data not available for contingency table.\n")

    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.writelines(lines)
