
import numpy as np
import pandas as pd
import warnings
from collections import namedtuple
from typing import Union, Optional, List, Tuple

from ._segmented import HybridSegmentedTrend as _HybridSegmentedTrend, _validate_alpha
from ._datetime import _to_numeric_time, _is_datetime_like, _extract_timezone
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
        if self.is_datetime:
            t = _to_numeric_time(t)
        elif isinstance(t, (pd.Series, pd.Index)) and pd.api.types.is_numeric_dtype(t):
            t = t.to_numpy(dtype=float, na_value=np.nan)
        else:
            t = np.asarray(t, dtype=float)

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

    @property
    def bootstrap_failure_rate(self) -> Optional[float]:
        """
        Fraction of bagging bootstrap replicates that failed to converge,
        or ``None`` if bagging was not used.
        """
        if self.bootstrap_samples is None:
            return None
        if len(self.bootstrap_samples) == 0:
            return 1.0
        n_failed = sum(1 for s in self.bootstrap_samples if len(s) == 0)
        return n_failed / len(self.bootstrap_samples)

    @property
    def ci_is_conditional(self) -> bool:
        """
        True when at least one breakpoint was estimated (``n_breakpoints > 0``),
        indicating that per-segment slope confidence intervals are conditional
        on the estimated breakpoint locations (post-selection inference).
        """
        return bool(self.n_breakpoints and self.n_breakpoints > 0)

    @property
    def analysis_notes(self) -> List[str]:
        """
        Diagnostic notes describing methodological caveats for this result
        (e.g., post-selection conditional CIs, bootstrap replicate failures).
        """
        notes: List[str] = []
        if self.ci_is_conditional:
            if self.bootstrap_samples is None:
                notes.append(
                    "Per-segment slope confidence intervals are conditional on the "
                    "estimated breakpoints (post-selection inference) and Phase-1 OLS "
                    "breakpoint CIs use an asymptotic Delta-method t-interval assuming "
                    "i.i.d. Gaussian errors; both may undercover nominal levels under "
                    "low SNR, small slope changes, edge breakpoints, outliers, or "
                    "serial correlation."
                )
            else:
                notes.append(
                    "Per-segment slope confidence intervals are conditional on the "
                    "bagged breakpoint estimates (post-selection inference) and do not "
                    "include breakpoint-location uncertainty; they may undercover "
                    "nominal levels under low SNR, small slope changes, edge "
                    "breakpoints, outliers, or serial correlation."
                )
        failure_rate = self.bootstrap_failure_rate
        if failure_rate is not None and failure_rate > 0.0:
            notes.append(
                f"Bagging bootstrap replicate failure rate: {100.0 * failure_rate:.1f}% "
                f"(failed replicates were excluded from breakpoint estimates and CIs)."
            )
        return notes

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
    df = df.sort_values('t', kind='stable').reset_index(drop=True)
    return df, is_dt

def segmented_trend_test(
    x: Union[np.ndarray, pd.DataFrame],
    t: np.ndarray,
    n_breakpoints: Optional[int] = None,  # None means search up to max
    max_breakpoints: int = 5,
    alpha: float = 0.05,
    hicensor: Union[bool, float] = False,
    criterion: str = 'bic',
    use_bagging: bool = False,
    n_bootstrap: int = 100,
    slope_scaling: Optional[str] = None,
    random_state: Optional[int] = None,
    large_dataset_mode: str = 'auto',
    max_pairs: Optional[int] = None,
    bootstrap_method: str = 'block',
    block_size: Union[str, int] = 'auto',
    inner_n_boot: int = 10,
    autocorr_penalty: Union[str, bool] = 'auto',
    use_davies_test: bool = False,
    autocorr_method: str = 'none',
    lt_mult: float = 0.5,
    gt_mult: float = 1.1,
    sens_slope_method: str = 'ats',
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
        alpha: Significance level for confidence intervals (must satisfy 0 < alpha < 1).
        hicensor: High-censor rule flag.
        criterion: Model selection criterion ('bic', 'mbic', or 'aic'). Default 'bic'.
        use_bagging: Use bootstrap aggregating for robust breakpoint locations.
            Note that bagging resamples breakpoint *locations* conditional on the
            Phase-1 selected ``n_breakpoints`` (model selection itself is not bagged,
            and returned ``bic``/``aic`` remain Phase-1 OLS values).
        n_bootstrap: Number of bootstrap iterations if bagging is enabled (must be >= 2).
        slope_scaling: Unit to scale the slope to (e.g. 'year'). Only for datetime t.
        random_state: Seed for random number generator.
        large_dataset_mode: Controls large dataset handling (passed to slope estimator).
        max_pairs: Max pairs for fast slope estimation.
        bootstrap_method: Resampling scheme when ``use_bagging=True``:
            ``'block'`` (default, autocorrelation-aware circular block bootstrap),
            ``'pairs'`` (i.i.d. pairs bootstrap), or ``'residual'`` (circular block
            residual bootstrap on the original time grid).
        block_size: Block length for ``'block'``/``'residual'`` bagging
            (``'auto'`` or integer >= 1).
        inner_n_boot: Internal Muggeo restarts per outer bootstrap replicate. Default 10.
        autocorr_penalty: Effective-sample-size (ESS) penalty for autocorrelation
            during automatic Phase-1 model selection (``'auto'``, ``'none'``, ``True``,
            or ``False``). Default ``'auto'``.
        use_davies_test: Opt-in Davies test gate (0 vs >=1 breakpoints) during
            automatic model selection. Default ``False``.
        autocorr_method: Optional Phase-2 per-segment autocorrelation adjustment
            for slope confidence intervals: ``'none'`` (default, i.i.d. Mann-Kendall
            variance), ``'auto'`` (detect per-segment autocorrelation and apply block
            bootstrap or Yue-Wang), ``'yue_wang'`` (ESS variance inflation), or
            ``'block_bootstrap'`` (circular block bootstrap percentile CIs).
        lt_mult: Multiplier for left-censored values in Phase-1 OLS substitution
            and ``'lwp'`` slope estimation. Default ``0.5``.
        gt_mult: Multiplier for right-censored values in Phase-1 OLS substitution
            and ``'lwp'`` slope estimation. Default ``1.1``.
        sens_slope_method: Slope estimator for censored segments: ``'ats'`` (default,
            Akritas-Theil-Sen), ``'lwp'``, or ``'drop_ambiguous'``.

    Note:
        **Post-Selection Conditional Confidence Intervals:**
        When one or more breakpoints are estimated (``result.ci_is_conditional``
        is ``True``), per-segment slope confidence intervals are conditional on
        the estimated breakpoint locations and do not account for breakpoint-location
        uncertainty. Phase-1 OLS breakpoint confidence intervals (when
        ``use_bagging=False``) rely on an asymptotic Delta-method t-interval assuming
        i.i.d. Gaussian errors. Both intervals may undercover nominal ``1 - alpha``
        levels under low signal-to-noise ratio, small slope changes, edge breakpoints,
        outliers, or serial correlation. Inspect ``result.ci_is_conditional``,
        ``result.analysis_notes``, and ``result.warnings`` for diagnostics, and
        consider ``use_bagging=True`` and/or ``autocorr_method='yue_wang'`` or
        ``'block_bootstrap'`` for autocorrelated series.

    Returns:
        SegmentedTrendResult: Namedtuple result with ``predict()``,
        ``bootstrap_failure_rate``, ``ci_is_conditional``, and ``analysis_notes``.

    Note:
        When `slope_scaling` is used, the `slope`, `lower_ci`, and `upper_ci` columns
        in the returned `segments` DataFrame are scaled to the requested unit.
        However, the `intercept` column corresponds to the unscaled time (seconds for datetime inputs)
        and unscaled slope (units per second). If you wish to reconstruct the line using
        the scaled slope, you must adjust the time variable accordingly.
    """
    from ._stats import _normalize_sens_slope_method
    alpha = _validate_alpha(alpha)
    sens_slope_method = _normalize_sens_slope_method(sens_slope_method)

    # Validate model hyperparameters upfront (before data prep / warning capture)
    hybrid_model = _HybridSegmentedTrend(
        max_breakpoints=max_breakpoints,
        n_breakpoints=n_breakpoints,
        use_bagging=use_bagging,
        n_bootstrap=n_bootstrap,
        criterion=criterion,
        random_state=random_state,
        bootstrap_method=bootstrap_method,
        block_size=block_size,
        inner_n_boot=inner_n_boot,
        autocorr_penalty=autocorr_penalty,
        use_davies_test=use_davies_test,
        autocorr_method=autocorr_method,
    )

    # Capture warnings
    captured_warnings = []

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")

        # 1. Data Prep
        data_filtered, is_datetime = _prepare_data(x, t, hicensor)
        orig_tz = _extract_timezone(t) if is_datetime else None

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

        # Warn on shifting left-censoring detection limits when hicensor is disabled (F-07)
        lt_mask = (cen_type == 'lt')
        if not hicensor and int(np.sum(lt_mask)) >= 5 and len(np.unique(x_val[lt_mask])) >= 2:
            warnings.warn(
                "Multiple distinct left-censoring detection limits detected with "
                "hicensor=False. Shifting detection limits can fabricate breakpoints "
                "at method-change dates; consider setting hicensor=True.",
                UserWarning,
            )

        # 2. Fit Hybrid Model
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

        # Surface post-selection conditional CI diagnostic when breakpoints were estimated (F-08)
        if n_bp > 0:
            warnings.warn(
                "Per-segment slope confidence intervals are conditional on the estimated "
                "breakpoints (post-selection inference) and do not account for breakpoint-location "
                "uncertainty. Nominal coverage may be lower under low SNR, small slope changes, "
                "edge breakpoints, outliers, or serial correlation.",
                UserWarning,
            )

        # Warn on short datetime records where unadjusted seasonality mimics breakpoints (F-07)
        if is_datetime and len(t_numeric) >= 12:
            span_days = float(t_numeric[-1] - t_numeric[0]) / 86400.0
            diffs = np.diff(t_numeric)
            med_step_days = float(np.median(diffs)) / 86400.0 if len(diffs) > 0 else np.inf
            if (
                180.0 <= span_days < 3.0 * 365.25
                and med_step_days < 100.0
                and (n_breakpoints is None or n_bp > 0)
            ):
                warnings.warn(
                    f"Short datetime record ({span_days / 365.25:.2f} years < 3 years) with "
                    f"sub-seasonal sampling (median step {med_step_days:.1f} days). "
                    f"Unadjusted seasonality can be mistaken for structural breakpoints; "
                    f"consider check_seasonality() or deseasonalization before segmented regression.",
                    UserWarning,
                )

        # Convert breakpoints back to original time format if datetime
        if is_datetime:
            breakpoints_final = pd.to_datetime(breakpoints, unit='s')
            if orig_tz is not None:
                breakpoints_final = breakpoints_final.tz_localize('UTC').tz_convert(orig_tz)

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
                        if orig_tz is not None:
                            if pd.notna(low_dt):
                                low_dt = low_dt.tz_localize('UTC').tz_convert(orig_tz)
                            if pd.notna(high_dt):
                                high_dt = high_dt.tz_localize('UTC').tz_convert(orig_tz)
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

        # Always store unscaled slopes/CIs and default units before attempting scaling (F-13e)
        if not segments_list.empty:
            segments_list['slope_per_second'] = segments_list['slope'].copy()
            segments_list['lower_ci_per_second'] = segments_list['lower_ci'].copy()
            segments_list['upper_ci_per_second'] = segments_list['upper_ci'].copy()
            segments_list['slope_units'] = "units per second" if is_datetime else "units per time"

            if slope_scaling:
                if is_datetime:
                    try:
                        factor = _get_slope_scaling_factor(slope_scaling)
                        segments_list['slope'] = segments_list['slope'] * factor
                        segments_list['lower_ci'] = segments_list['lower_ci'] * factor
                        segments_list['upper_ci'] = segments_list['upper_ci'] * factor
                        segments_list['slope_units'] = f"units per {slope_scaling}"
                    except Exception as e:
                        warnings.warn(f"Slope scaling failed: {e}", UserWarning)
                else:
                    warnings.warn("slope_scaling requires datetime inputs.", UserWarning)

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
        score=hybrid_model.score_,
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
    if 'n_breakpoints' in kwargs:
        raise TypeError(
            "find_best_segmentation() does not accept 'n_breakpoints'; "
            "use segmented_trend_test() for a fixed number of breakpoints."
        )

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
    Calculate the empirical conditional bootstrap frequency that at least one
    breakpoint occurred within ``[start_date, end_date]``.

    Requires that ``use_bagging=True`` was used in the test. Note that this is
    an empirical bootstrap frequency conditional on ``result.n_breakpoints``
    (not a Bayesian posterior probability), and for ``n_breakpoints >= 2`` it
    reflects the fraction of converged bootstrap replicates in which at least
    one breakpoint falls inside the window. Failed (empty) bootstrap replicates
    are excluded from the denominator with a warning.

    Args:
        result: The result from segmented_trend_test.
        start_date: Start of the window (datetime or numeric).
        end_date: End of the window (datetime or numeric).

    Returns:
        prob: Empirical bootstrap frequency in [0.0, 1.0], or ``np.nan`` if no
            valid bootstrap replicates are available.
    """
    if result.bootstrap_samples is None or len(result.bootstrap_samples) == 0:
        warnings.warn("No bootstrap samples available. Run with use_bagging=True.", UserWarning)
        return np.nan

    # Convert dates to numeric scalar floats if needed
    t_start = float(_to_numeric_time(start_date)[0]) if result.is_datetime else float(start_date)
    t_end = float(_to_numeric_time(end_date)[0]) if result.is_datetime else float(end_date)

    samples = result.bootstrap_samples
    valid_samples = [it for it in samples if len(it) > 0]
    n_failed = len(samples) - len(valid_samples)

    if n_failed > 0:
        warnings.warn(
            f"Excluded {n_failed}/{len(samples)} failed bootstrap replicates when "
            f"computing breakpoint probability.",
            UserWarning,
        )

    if len(valid_samples) == 0:
        warnings.warn("All bootstrap replicates failed; returning NaN.", UserWarning)
        return np.nan

    count_in_window = 0
    for iteration_bps in valid_samples:
        # Check if ANY breakpoint in this iteration is in the window
        found = False
        for bp in iteration_bps:
            if t_start <= bp <= t_end:
                found = True
                break
        if found:
            count_in_window += 1

    return count_in_window / len(valid_samples)
