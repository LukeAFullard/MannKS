
import numpy as np
import pandas as pd
import piecewise_regression
from scipy.stats import gaussian_kde, t as t_dist
from scipy.signal import find_peaks
import warnings

def _bootstrap_breakpoints(t, x, n_breakpoints, n_bootstrap=100, alpha_n=0.05, random_state=None):
    """
    Bootstrap the breakpoint detection to find robust breakpoint locations.

    Args:
        t: Time vector
        x: Data vector
        n_breakpoints: Number of breakpoints to find
        n_bootstrap: Number of bootstrap iterations
        alpha_n: Significance level (not used directly here but standard arg)
        random_state: Seed for random number generator.

    Returns:
        all_breakpoints: List of lists. Each inner list contains breakpoints
                         found in one bootstrap iteration.
    """
    # Explicit conversion to numpy arrays for safety
    t = np.asarray(t)
    x = np.asarray(x)

    n = len(x)
    all_breakpoints = []

    # Initialize random generator
    rng = np.random.default_rng(random_state)

    for _ in range(n_bootstrap):
        # Bootstrap Resampling
        indices = rng.choice(n, n, replace=True)
        t_boot = t[indices]
        x_boot = x[indices]

        # Sort for piecewise_regression
        sort_idx = np.argsort(t_boot)
        t_boot = t_boot[sort_idx]
        x_boot = x_boot[sort_idx]

        try:
            # We fix the number of breakpoints to what was requested/estimated
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pw_fit = piecewise_regression.Fit(t_boot, x_boot, n_breakpoints=n_breakpoints, verbose=False)

            # Extract estimates
            estimates = None
            if hasattr(pw_fit, 'best_muggeo') and hasattr(pw_fit.best_muggeo, 'best_fit'):
                 estimates = pw_fit.best_muggeo.best_fit.estimates
            elif hasattr(pw_fit, 'get_results'):
                 estimates = pw_fit.get_results().get('estimates')

            current_iter_bps = []
            if estimates:
                for key, val in estimates.items():
                    if key.startswith('breakpoint'):
                        current_iter_bps.append(val['estimate'])

            current_iter_bps.sort()
            # We append an empty list on failure to ensure the bootstrap iteration count matches.
            all_breakpoints.append(current_iter_bps)

        except Exception:
            all_breakpoints.append([])
            continue

    return all_breakpoints

def find_bagged_breakpoints(t, x, n_breakpoints, n_bootstrap=100, random_state=None):
    """
    Find robust breakpoints using Bagging and KDE per breakpoint index.

    1. Bootstrap data and find breakpoints (keeping them ordered).
    2. For each breakpoint index (1st, 2nd, ...), estimate the density of samples.
    3. Find the peak (mode) of that density to identify the robust location for that index.

    Args:
        t: Time vector
        x: Data vector
        n_breakpoints: Number of breakpoints to identify
        n_bootstrap: Number of bootstrap iterations
        random_state: Seed for random number generator.

    Returns:
        robust_breakpoints: Array of robust breakpoint locations
        all_breakpoints: List of lists of all bootstrap breakpoint samples
    """
    # Explicit conversion to numpy arrays for safety
    t = np.asarray(t)
    x = np.asarray(x)

    if n_breakpoints == 0:
        return np.array([]), []

    all_bps_structured = _bootstrap_breakpoints(t, x, n_breakpoints, n_bootstrap, random_state=random_state)

    robust_bps = []

    # Process column-wise (per breakpoint index)
    for i in range(n_breakpoints):
        # Collect samples for the i-th breakpoint
        col_samples = [bps[i] for bps in all_bps_structured if len(bps) > i]
        col_samples = np.array(col_samples)

        if len(col_samples) < 2:
            if len(col_samples) == 1:
                robust_bps.append(col_samples[0])
            else:
                robust_bps.append(np.nan)
            continue

        # KDE for this column
        try:
            kde = gaussian_kde(col_samples)

            # Grid centered around the samples
            min_val, max_val = col_samples.min(), col_samples.max()
            padding = (max_val - min_val) * 0.2
            if padding == 0: padding = 1e-5

            t_grid = np.linspace(min_val - padding, max_val + padding, 500)
            density = kde(t_grid)

            # Find global peak (mode)
            # We assume for the i-th breakpoint, the distribution should be unimodal
            # if the model is stable. If multimodal, the highest peak is the best point estimate.
            peak_idx = np.argmax(density)
            robust_bps.append(t_grid[peak_idx])

        except Exception as e:
            # Fallback to median if KDE fails (e.g. singular matrix due to identical values)
            robust_bps.append(np.median(col_samples))

    return np.array(robust_bps), all_bps_structured

class HybridSegmentedTrend:
    """
    Hybrid Segmented Regression.

    Phase 1 (Structure Discovery): Uses OLS (piecewise-regression) to find
    the number and location of breakpoints. Optionally uses bagging for robustness.

    Phase 2 (Robust Estimation): Uses Mann-Kendall / Sen's Slope on the
    identified segments to estimate robust slopes and confidence intervals.
    """

    def __init__(self, max_breakpoints=5, n_breakpoints=None, use_bagging=False, n_bootstrap=100, criterion='bic', random_state=None):
        """
        Args:
            max_breakpoints (int): Maximum number of breakpoints to search for (if n_breakpoints is None).
            n_breakpoints (int, optional): Fixed number of breakpoints to fit.
            use_bagging (bool): Use bootstrap aggregating for breakpoint location.
            n_bootstrap (int): Number of bootstrap samples if bagging is used.
            criterion (str): Model selection criterion ('bic' or 'aic'). Default 'bic'.
            random_state (int, optional): Seed for random number generator.
        """
        self.max_breakpoints = max_breakpoints
        self.n_breakpoints = n_breakpoints
        self.use_bagging = use_bagging
        self.n_bootstrap = n_bootstrap
        self.criterion = criterion.lower()
        self.random_state = random_state

        self.breakpoints_ = None
        self.breakpoint_cis_ = None
        self.segments_ = None
        self.n_breakpoints_ = None
        self.bic_ = None
        self.aic_ = None
        self.selection_summary_ = None
        self.bootstrap_samples_ = None

    def fit(self, t, x, censored=None, cen_type=None, lt_mult=0.5, gt_mult=1.1, alpha=0.05):
        t = np.asarray(t)
        x = np.asarray(x)

        # Sort data
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        x = x[sort_idx]

        if censored is not None:
            censored = np.asarray(censored)[sort_idx]
            cen_type = np.asarray(cen_type)[sort_idx]
            # OLS needs numeric x. Substitute censored values.
            x_ols = x.copy().astype(float)
            x_ols[cen_type == 'lt'] *= lt_mult
            x_ols[cen_type == 'gt'] *= gt_mult
        else:
            x_ols = x

        # --- Phase 1: Structure Discovery ---

        best_n = 0
        best_bps = []
        best_bp_cis = []
        best_score = np.inf # Tracks best BIC or AIC
        best_bic = np.inf
        best_aic = np.inf

        selection_records = []

        n_range = []
        if self.n_breakpoints is not None:
            n_range = [self.n_breakpoints]
        else:
            n_range = range(self.max_breakpoints + 1)

        for k in n_range:
            record = {'n_breakpoints': k, 'bic': np.nan, 'aic': np.nan, 'sar': np.nan, 'converged': False}
            try:
                if k == 0:
                    # Linear Fit BIC
                    p = np.polyfit(t, x_ols, 1)
                    y_pred = np.polyval(p, t)
                    rss = np.sum((x_ols - y_pred)**2)
                    n_samples = len(x)
                    if rss <= 1e-10: rss = 1e-10
                    # k_params = 2 (slope, intercept)
                    bic = n_samples * np.log(rss/n_samples) + 2 * np.log(n_samples)
                    aic = n_samples * np.log(rss/n_samples) + 2 * 2

                    record['bic'] = bic
                    record['aic'] = aic
                    record['sar'] = rss
                    record['converged'] = True

                    score = bic if self.criterion == 'bic' else aic
                    if score < best_score:
                        best_score = score
                        best_bic = bic
                        best_aic = aic
                        best_n = 0
                        best_bps = []
                        best_bp_cis = []
                else:
                    pw_fit = piecewise_regression.Fit(t, x_ols, n_breakpoints=k, verbose=False)
                    # Check BIC
                    current_bic = np.inf
                    current_aic = np.inf
                    rss = np.nan

                    if hasattr(pw_fit, 'best_muggeo') and hasattr(pw_fit.best_muggeo, 'best_fit'):
                        bf = pw_fit.best_muggeo.best_fit
                        # Try to get BIC directly
                        current_bic = getattr(bf, 'bic', np.inf)

                        # Calculate AIC if possible
                        if current_bic != np.inf:
                            k_params = 2 * k + 2
                            n_samples = len(x)
                            current_aic = current_bic - k_params * np.log(n_samples) + 2 * k_params
                        else:
                            current_aic = np.inf

                        rss = getattr(bf, 'residual_sum_squares', np.nan)
                    else:
                        res = pw_fit.get_results()
                        current_bic = res.get('bic', np.inf)
                        current_aic = current_bic # Approx if not avail
                        rss = res.get('ssr', res.get('rss', np.nan))

                    record['bic'] = current_bic
                    record['aic'] = current_aic
                    record['sar'] = rss
                    record['converged'] = True

                    score = current_bic if self.criterion == 'bic' else current_aic
                    if score < best_score:
                        best_score = score
                        best_bic = current_bic
                        best_aic = current_aic
                        best_n = k

                        # Standard Estimates
                        estimates = None
                        if hasattr(pw_fit, 'best_muggeo') and hasattr(pw_fit.best_muggeo, 'best_fit'):
                             estimates = pw_fit.best_muggeo.best_fit.estimates
                        elif hasattr(pw_fit, 'get_results'):
                             estimates = pw_fit.get_results().get('estimates')

                        bps_data = []
                        if estimates:
                            for key, val in estimates.items():
                                if key.startswith('breakpoint'):
                                    est = val['estimate']

                                    # Manually calculate CI for requested alpha
                                    # piecewise_regression defaults to 95%
                                    se = val.get('se')
                                    if se is not None and not np.isnan(se):
                                        # df = n_samples - n_params
                                        # n_params = 2 (const, beta1) + n_breakpoints (alphas) + n_breakpoints (bps)
                                        # For muggeo: roughly 2 + 2*k
                                        n_params = 2 + 2 * k
                                        df = len(x) - n_params
                                        if df > 0:
                                            # t-statistic for two-tailed alpha
                                            t_crit = t_dist.ppf(1 - alpha / 2, df)
                                            ci = (est - t_crit * se, est + t_crit * se)
                                        else:
                                            ci = (np.nan, np.nan)
                                    else:
                                        ci = val.get('confidence_interval', (np.nan, np.nan))

                                    if ci is None: ci = (np.nan, np.nan)
                                    bps_data.append((est, ci))

                        bps_data.sort(key=lambda x: x[0])
                        best_bps = [b[0] for b in bps_data]
                        best_bp_cis = [b[1] for b in bps_data]

            except Exception:
                pass

            selection_records.append(record)

        self.selection_summary_ = pd.DataFrame(selection_records)

        # 2. Refine Locations with Bagging (if enabled and N > 0)
        if self.use_bagging and best_n > 0:
            robust_bps, all_samples_structured = find_bagged_breakpoints(t, x_ols, best_n, self.n_bootstrap, random_state=self.random_state)
            self.bootstrap_samples_ = all_samples_structured

            if len(robust_bps) == best_n:
                 best_bps = robust_bps
                 best_bp_cis = []

                 # Calculate Bootstrap CIs (Column-wise)
                 for i in range(best_n):
                     # Extract column i
                     col_samples = [bps[i] for bps in all_samples_structured if len(bps) > i]

                     if len(col_samples) > 0:
                         # Filter outliers using IQR to avoid massive CIs from failed fits
                         q1 = np.percentile(col_samples, 25)
                         q3 = np.percentile(col_samples, 75)
                         iqr = q3 - q1

                         # Define acceptable range (1.5 * IQR is standard for outliers)
                         lower_bound = q1 - 1.5 * iqr
                         upper_bound = q3 + 1.5 * iqr

                         filtered_samples = [s for s in col_samples if lower_bound <= s <= upper_bound]

                         if len(filtered_samples) > 0:
                             low = np.percentile(filtered_samples, 2.5)
                             high = np.percentile(filtered_samples, 97.5)
                             best_bp_cis.append((low, high))
                         else:
                             # Fallback if filtering excludes all samples
                             low = np.percentile(col_samples, 2.5)
                             high = np.percentile(col_samples, 97.5)
                             best_bp_cis.append((low, high))
                     else:
                         best_bp_cis.append((np.nan, np.nan))
            else:
                 # If bagging returned different number of robust peaks than best_n (unexpected with fixed N logic)
                 pass

        self.n_breakpoints_ = best_n
        self.breakpoints_ = np.array(best_bps)
        self.breakpoint_cis_ = best_bp_cis
        self.bic_ = best_bic
        self.aic_ = best_aic

        # --- Phase 2: Robust Estimation (MannKS) ---
        # Import here to avoid potential circular dependencies.
        from ._stats import (
            _sens_estimator_unequal_spacing,
            _sens_estimator_censored,
            _mk_score_and_var_censored,
            _confidence_intervals
        )

        self.segments_ = []
        boundaries = np.concatenate(([t.min()], self.breakpoints_, [t.max()]))

        for i in range(len(boundaries) - 1):
            t_start = boundaries[i]
            t_end = boundaries[i+1]

            if i == len(boundaries) - 2:
                mask = (t >= t_start) & (t <= t_end)
            else:
                mask = (t >= t_start) & (t < t_end)

            t_seg = t[mask]
            x_seg = x[mask]

            if len(x_seg) < 2:
                self.segments_.append({
                    'slope': np.nan, 'intercept': np.nan,
                    'lower_ci': np.nan, 'upper_ci': np.nan,
                    'n': 0
                })
                continue

            if censored is not None:
                cen_seg = censored[mask]
                cen_type_seg = cen_type[mask]
                slopes = _sens_estimator_censored(x_seg, t_seg, cen_type_seg, lt_mult, gt_mult)
                s, var_s, _, _ = _mk_score_and_var_censored(x_seg, t_seg, cen_seg, cen_type_seg)
            else:
                slopes = _sens_estimator_unequal_spacing(x_seg, t_seg)
                dummy_cen = np.zeros(len(x_seg), dtype=bool)
                dummy_type = np.full(len(x_seg), 'not', dtype=object)
                s, var_s, _, _ = _mk_score_and_var_censored(x_seg, t_seg, dummy_cen, dummy_type)

            if len(slopes) == 0 or np.all(np.isnan(slopes)):
                slope = np.nan
            else:
                slope = np.nanmedian(slopes)
            lower_ci, upper_ci = _confidence_intervals(slopes, var_s, alpha=alpha)

            # Robust Intercept
            if censored is not None:
                uncensored_mask = ~cen_seg.astype(bool)
                if np.any(uncensored_mask):
                    intercept = np.median(x_seg[uncensored_mask] - slope * t_seg[uncensored_mask])
                else:
                    intercept = np.nan
            else:
                intercept = np.median(x_seg - slope * t_seg)

            self.segments_.append({
                'slope': slope,
                'intercept': intercept,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'n': len(x_seg)
            })

    def predict(self, t):
        t = np.asarray(t)
        y_pred = np.zeros_like(t, dtype=float)
        y_pred[:] = np.nan

        if self.segments_ is None or len(self.segments_) == 0:
            return y_pred

        bps = self.breakpoints_
        if len(bps) == 0:
             seg = self.segments_[0]
             return seg['slope'] * t + seg['intercept']

        mask = t < bps[0]
        seg = self.segments_[0]
        y_pred[mask] = seg['slope'] * t[mask] + seg['intercept']

        for i in range(len(bps) - 1):
            mask = (t >= bps[i]) & (t < bps[i+1])
            seg = self.segments_[i+1]
            y_pred[mask] = seg['slope'] * t[mask] + seg['intercept']

        mask = t >= bps[-1]
        seg = self.segments_[-1]
        y_pred[mask] = seg['slope'] * t[mask] + seg['intercept']

        return y_pred
