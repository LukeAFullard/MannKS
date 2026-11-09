
import numpy as np
import pandas as pd
import warnings
from collections import namedtuple
from typing import Union, Optional, List, Tuple

from ._segmented import HybridSegmentedTrend as _HybridSegmentedTrend
from ._datetime import _to_numeric_time, _is_datetime_like
from ._helpers import _get_slope_scaling_factor

_Segmented_Trend_Test_Tuple = namedtuple('Segmented_Trend_Test', [
    'n_breakpoints', 'breakpoints', 'breakpoint_cis', 'segments',
    'is_datetime', 'bic', 'aic', 'score', 'selection_summary', 'bootstrap_samples',
    'alpha', 'warnings', 'computation_mode'
])

class SegmentedTrendResult(_Segmented_Trend_Test_Tuple):
    def predict(self, t):
        """
        Predict values based on the segmented trend model.

        Args:
            t: Time vector (datetime or numeric).

        Returns:
            y_pred: Predicted values.
        """
        # Convert input time to numeric if needed
        t_orig = t
        if self.is_datetime:
            t = _to_numeric_time(t)
        t = np.asarray(t)

        y_pred = np.zeros_like(t, dtype=float)
        y_pred[:] = np.nan

        if self.segments is None or self.segments.empty:
            return y_pred

        # Get breakpoints in numeric format
        # self.breakpoints might be datetime. Convert back to numeric for logic.
        bps = self.breakpoints
        if self.is_datetime and len(bps) > 0:
             bps = _to_numeric_time(bps)
        bps = np.asarray(bps)

        # We need to use the RAW slope (per second) and intercept for prediction
        # because the intercept is always defined relative to the raw time axis.
        # If slope_scaling was used, 'slope' column is scaled.
        # We should look for 'slope_per_second' if it exists, otherwise 'slope'.

        segs = self.segments

        def get_params(idx):
            row = segs.iloc[idx]
            if 'slope_per_second' in row:
                s = row['slope_per_second']
            else:
                s = row['slope']
            i = row['intercept']
            return s, i

        if len(bps) == 0:
             s, i = get_params(0)
             return s * t + i

        # Segment 0
        mask = t < bps[0]
        if np.any(mask):
            s, i = get_params(0)
            y_pred[mask] = s * t[mask] + i

        for idx in range(len(bps) - 1):
            mask = (t >= bps[idx]) & (t < bps[idx+1])
            if np.any(mask):
                s, i = get_params(idx+1)
                y_pred[mask] = s * t[mask] + i

        # Last Segment
        mask = t >= bps[-1]
        if np.any(mask):
            s, i = get_params(-1)
            y_pred[mask] = s * t[mask] + i

        return y_pred

def _prepare_data(x, t, hicensor=False):
    """
    Prepare data for segmented analysis using the same rules as
    ``trend_test`` (censoring labels, NaN removal, high-censor rule), then
    sort by time.
    """
    from ._helpers import _prepare_data as _prepare_common
    is_dt = _is_datetime_like(t)
    if isinstance(x, pd.DataFrame) and 'value' in x.columns:
        df = x.copy()
        if 'censored' not in df.columns:
            df['censored'] = False
        df['censored'] = df['censored'].astype(bool)
        if 'cen_type' not in df.columns:
            if df['censored'].any():
                warnings.warn("No 'cen_type' column: censored values are assumed to be left-censored ('lt').",
                              UserWarning)
            df['cen_type'] = np.where(df['censored'], 'lt', 'not')
        x = df[['value', 'censored', 'cen_type']].reset_index(drop=True)
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            x = x.iloc[:, 0].to_numpy()
        else:
            raise ValueError("Input DataFrame must contain a 'value' column.")
    df, _ = _prepare_common(x, t, hicensor)
    df['t'] = _to_numeric_time(np.asarray(t))[df.index.to_numpy(dtype=np.intp)]
    df = df.sort_values('t', kind='stable').reset_index(drop=True)
    return df, is_dt

def segmented_trend_test(
    x: Union[np.ndarray, pd.DataFrame],
    t: np.ndarray,
    n_breakpoints: Optional[int] = None, # None means search up to max
    max_breakpoints: int = 5,
    alpha: float = 0.05,
    hicensor: Union[bool, float] = False,
    criterion: str = 'bic', # Only 'bic' is currently supported by the underlying implementation.
    use_bagging: bool = False,
    n_bootstrap: int = 100,
    slope_scaling: Optional[str] = None,
    random_state: Optional[int] = None,
    large_dataset_mode: str = 'auto',
    max_pairs: Optional[int] = None,
    **kwargs
):
    """
    Perform Hybrid Segmented Trend Analysis.

    This function is a wrapper around `HybridSegmentedTrend`. It combines:
    1. OLS-based breakpoint detection (Piecewise Regression).
    2. Robust Mann-Kendall / Sen's slope estimation on the identified segments.

    Args:
        x: Data vector or DataFrame.
        t: Time vector.
        n_breakpoints: Fixed number of breakpoints. If None, optimal number is searched.
        max_breakpoints: Maximum number of breakpoints to search (if n_breakpoints is None).
        alpha: Significance level for confidence intervals.
        hicensor: High-censor rule flag.
        criterion: Model selection criterion ('bic' or 'aic').
        use_bagging: Use bootstrap aggregating for robust breakpoint location.
        n_bootstrap: Number of bootstrap iterations if bagging is enabled.
        slope_scaling: Unit to scale the slope to (e.g. 'year'). Only for datetime t.
        random_state: Seed for random number generator.
        large_dataset_mode: Controls large dataset handling (passed to slope estimator).
        max_pairs: Max pairs for fast slope estimation.
        **kwargs: Additional arguments for trend estimation (lt_mult, gt_mult,
            sens_slope_method: 'ats' (default for censored segments), 'lwp'
            or 'drop_ambiguous').

    Note:
        Segment slopes and confidence intervals are conditional on the
        estimated breakpoints (no adjustment for breakpoint selection).

    Returns:
        namedtuple: Segmented_Trend_Test result.

    Note:
        When `slope_scaling` is used, the `slope`, `lower_ci`, and `upper_ci` columns
        in the returned `segments` DataFrame are scaled to the requested unit.
        However, the `intercept` column corresponds to the unscaled time (seconds for datetime inputs)
        and unscaled slope (units per second). If you wish to reconstruct the line using
        the scaled slope, you must adjust the time variable accordingly.
    """
    from ._stats import _normalize_sens_slope_method
    kwargs['sens_slope_method'] = _normalize_sens_slope_method(kwargs.get('sens_slope_method', 'ats'))

    # Capture warnings
    captured_warnings = []

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")

        # 1. Data Prep
        data_filtered, is_datetime = _prepare_data(x, t, hicensor)

        x_val = data_filtered['value'].to_numpy()
        t_numeric = data_filtered['t'].to_numpy()
        censored = data_filtered['censored'].to_numpy()
        cen_type = data_filtered['cen_type'].to_numpy()

        if len(x_val) > 20000:
            warnings.warn(
                f"Breakpoint search on n={len(x_val)} observations can take several minutes "
                f"(the piecewise-regression fit scales with n). Aggregating first (e.g. "
                f"daily or monthly medians) is much faster.", UserWarning)

        if len(x_val) < 2:
            warnings.warn("Insufficient data for segmented analysis.", UserWarning)
            # Collect warnings (including the one we just issued)
            for w in w_log:
                captured_warnings.append(str(w.message))

            return SegmentedTrendResult(
                n_breakpoints=0,
                breakpoints=[],
                breakpoint_cis=[],
                segments=pd.DataFrame(),
                is_datetime=is_datetime,
                bic=np.nan,
                aic=np.nan,
                score=np.nan,
                selection_summary=None,
                bootstrap_samples=None,
                alpha=alpha,
                warnings=captured_warnings,
                computation_mode='insufficient'
            )

        # 2. Fit Hybrid Model
        hybrid_model = _HybridSegmentedTrend(
            max_breakpoints=max_breakpoints,
            n_breakpoints=n_breakpoints,
            use_bagging=use_bagging,
            n_bootstrap=n_bootstrap,
            criterion=criterion,
            random_state=random_state
        )

        # Extract kwargs relevant for estimation
        lt_mult = kwargs.get('lt_mult', 0.5)
        gt_mult = kwargs.get('gt_mult', 1.1)
        sens_slope_method = kwargs['sens_slope_method']

        hybrid_model.fit(
            t_numeric, x_val, censored, cen_type, lt_mult, gt_mult,
            alpha=alpha,
            large_dataset_mode=large_dataset_mode,
            max_pairs=max_pairs,
            sens_slope_method=sens_slope_method
        )

        # 3. Format Results
        breakpoints = hybrid_model.breakpoints_
        n_bp = hybrid_model.n_breakpoints_

        # Convert breakpoints back to original time format if datetime
        if is_datetime:
            breakpoints_final = pd.to_datetime(breakpoints, unit='s')

            # Convert CIs to Datetime
            breakpoint_cis_final = []
            if hybrid_model.breakpoint_cis_:
                for ci in hybrid_model.breakpoint_cis_:
                    # ci should be [low, high] in numeric time
                    if isinstance(ci, (list, tuple, np.ndarray)) and len(ci) == 2:
                        low_val = ci[0]
                        high_val = ci[1]

                        low_dt = pd.to_datetime(low_val, unit='s') if pd.notna(low_val) else pd.NaT
                        high_dt = pd.to_datetime(high_val, unit='s') if pd.notna(high_val) else pd.NaT
                        breakpoint_cis_final.append((low_dt, high_dt))
                    else:
                        breakpoint_cis_final.append((pd.NaT, pd.NaT))
            else:
                breakpoint_cis_final = [(pd.NaT, pd.NaT)] * n_bp
        else:
            breakpoints_final = breakpoints
            if hybrid_model.breakpoint_cis_:
                breakpoint_cis_final = hybrid_model.breakpoint_cis_
            else:
                breakpoint_cis_final = [(np.nan, np.nan)] * n_bp

        # Format segments for output
        segments_list = pd.DataFrame(hybrid_model.segments_)

        # Apply Slope Scaling
        if slope_scaling and not segments_list.empty:
            if is_datetime:
                try:
                    factor = _get_slope_scaling_factor(slope_scaling)
                    segments_list['slope_per_second'] = segments_list['slope']
                    segments_list['lower_ci_per_second'] = segments_list['lower_ci']
                    segments_list['upper_ci_per_second'] = segments_list['upper_ci']

                    segments_list['slope'] *= factor
                    segments_list['lower_ci'] *= factor
                    segments_list['upper_ci'] *= factor
                    segments_list['slope_units'] = f"units per {slope_scaling}"
                except Exception as e:
                    warnings.warn(f"Slope scaling failed: {e}", UserWarning)
            else:
                 warnings.warn("slope_scaling requires datetime inputs.", UserWarning)
        elif not segments_list.empty:
             # Store raw slopes as per_second just in case plotting needs it
             segments_list['slope_per_second'] = segments_list['slope']
             segments_list['lower_ci_per_second'] = segments_list['lower_ci']
             segments_list['upper_ci_per_second'] = segments_list['upper_ci']
             segments_list['slope_units'] = "units per second" if is_datetime else "units per time"

        # Collect warnings
        for w in w_log:
            captured_warnings.append(str(w.message))

    # Re-issue warnings if desired (optional, but polite since we suppressed them)
    for w_str in captured_warnings:
        warnings.warn(w_str, UserWarning)

    return SegmentedTrendResult(
        n_breakpoints=n_bp,
        breakpoints=breakpoints_final,
        breakpoint_cis=breakpoint_cis_final,
        segments=segments_list,
        is_datetime=is_datetime,
        bic=hybrid_model.bic_,
        aic=hybrid_model.aic_,
        score=hybrid_model.bic_,
        selection_summary=hybrid_model.selection_summary_,
        bootstrap_samples=hybrid_model.bootstrap_samples_,
        alpha=alpha,
        warnings=captured_warnings,
        computation_mode='hybrid'
    )


def find_best_segmentation(x, t, max_breakpoints=5, n_bootstrap=100, alpha=0.05, random_state=None, **kwargs):
    """
    Wrapper around segmented_trend_test to perform model selection and return summary.

    Args:
        x: Data
        t: Time
        max_breakpoints: Max breakpoints to check
        n_bootstrap: Number of bootstraps (if bagging enabled via kwargs)
        alpha: Significance level
        random_state: Seed for random number generator.
        **kwargs: Passed to segmented_trend_test

    Returns:
        best_result: The optimal Segmented_Trend_Test result
        summary: DataFrame of model selection metrics (BIC, AIC, etc.)
    """
    # Ensure n_breakpoints is None to trigger search
    kwargs['n_breakpoints'] = None
    kwargs['max_breakpoints'] = max_breakpoints
    kwargs['n_bootstrap'] = n_bootstrap
    kwargs['alpha'] = alpha
    kwargs['random_state'] = random_state

    result = segmented_trend_test(x, t, **kwargs)
    return result, result.selection_summary

def calculate_breakpoint_probability(result, start_date, end_date):
    """
    Calculate the probability that a breakpoint occurred within a specific time window.
    Requires that `use_bagging=True` was used in the test.

    Args:
        result: The result from segmented_trend_test
        start_date: Start of the window (datetime or numeric)
        end_date: End of the window (datetime or numeric)

    Returns:
        prob: Probability (0.0 to 1.0)
    """
    if result.bootstrap_samples is None or len(result.bootstrap_samples) == 0:
        warnings.warn("No bootstrap samples available. Run with use_bagging=True.", UserWarning)
        return np.nan

    # Convert dates to numeric if needed
    t_start = _to_numeric_time(start_date) if result.is_datetime else start_date
    t_end = _to_numeric_time(end_date) if result.is_datetime else end_date

    samples = result.bootstrap_samples

    # samples is now a list of lists (breakpoints per bootstrap iteration)
    count_in_window = 0
    total_iter = len(samples)

    if total_iter == 0:
        return 0.0

    for iteration_bps in samples:
        # Check if ANY breakpoint in this iteration is in the window
        found = False
        for bp in iteration_bps:
            if t_start <= bp <= t_end:
                found = True
                break
        if found:
            count_in_window += 1

    return count_in_window / total_iter
