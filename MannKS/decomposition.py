"""
Robust, censoring-aware seasonal-trend decomposition for irregularly sampled
data, integrated with the MannKS trend tests.

The series is split into trend + seasonal + remainder using annual harmonic
regression (seasonal), a robust local-linear smoother with a time-based window
(trend), and backfitting with one shared set of bisquare robustness weights.
Non-detects are imputed with ``ndimpute`` *conditional on the fitted model*
(on residuals), only to estimate the components; the deseasonalised series
keeps them censored.

This module is opt-in: it does not change any existing MannKS default or
result.  The Seasonal Kendall test (``seasonal_trend_test``) remains the
recommended trend test for seasonal data.
"""
import warnings
from collections import namedtuple
from numbers import Integral
from typing import Optional, Union

import numpy as np
import pandas as pd

from ._helpers import _prepare_data
from ._datetime import _to_numeric_time, _is_datetime_like
from ._autocorr import estimate_acf
from . import _decompose as _d

_FIELDS = [
    # per observation (time-sorted, transformed scale)
    'times', 't_years', 'order', 'values', 'censored', 'cen_type',
    'trend', 'seasonal', 'remainder', 'weights', 'fitted_values',
    # model
    'transform', 'n_harmonics_used', 'K_max', 'seasonal_mode_used',
    'seasonal_constancy_pvalue', 'coefficients',
    # descriptors
    'seasonal_amplitude', 'seasonal_amplitude_ratio', 'peak_day_of_year',
    'trough_day_of_year', 'seasonal_by_year',
    # uncertainty
    'trend_low', 'trend_high', 'trend_simul_low', 'trend_simul_high',
    'seasonal_low', 'seasonal_high', 'trend_imputation_sd', 'seasonal_imputation_sd',
    # bookkeeping
    'diagnostics', 'warnings', 'analysis_notes', 'settings',
]

_Decomposition_Tuple = namedtuple('Decomposition', _FIELDS)


class DecompositionResult(_Decomposition_Tuple):
    """
    Result of :func:`decompose`.

    Per-observation arrays (``trend``, ``seasonal``, ``remainder``, ``weights``,
    ...) are in time-sorted order on the transformed scale (natural log when
    ``transform='log'``).  ``order`` maps them back to the input positions
    (after removal of missing values).  ``times`` holds the original time
    values in the same sorted order and container type.
    """
    __slots__ = ()

    def predict(self, t_new):
        """
        Trend + seasonal at ``t_new`` on the transformed scale.

        The trend is interpolated linearly between observation times.  Times
        outside the data span return NaN (with a warning); there is no
        extrapolation.
        """
        s = self.settings
        if s['is_datetime']:
            if not _is_datetime_like(t_new):
                raise TypeError("The decomposition was fitted with datetime times; "
                                "pass datetime-like `t_new`.")
            ty = _d.seconds_to_decimal_years(_to_numeric_time(t_new))
        else:
            ty = np.atleast_1d(np.asarray(t_new, dtype=float))
        ty = np.atleast_1d(ty)
        out = (_d.interp_at(self.t_years, self.trend, ty)
               + s['seasonal_model'].evaluate(s['coef'], ty))
        outside = (ty < self.t_years[0]) | (ty > self.t_years[-1]) | ~np.isfinite(ty)
        if np.any(outside):
            warnings.warn(f"{int(outside.sum())} prediction time(s) lie outside the data span; "
                          "NaN returned (no extrapolation).", UserWarning, stacklevel=2)
            out[outside] = np.nan
        return out

    def to_dataframe(self):
        """One row per observation (time-sorted)."""
        df = pd.DataFrame({
            'time': np.asarray(self.times),
            't_years': self.t_years,
            'order': self.order,
            'value': self.values,
            'censored': self.censored,
            'cen_type': self.cen_type,
            'trend': self.trend,
            'seasonal': self.seasonal,
            'remainder': self.remainder,
            'weight': self.weights,
            'fitted_value': self.fitted_values,
            'deseasonalized': self.values - self.seasonal,
        })
        for name in ('trend_low', 'trend_high', 'trend_simul_low', 'trend_simul_high',
                     'seasonal_low', 'seasonal_high', 'trend_imputation_sd',
                     'seasonal_imputation_sd'):
            v = getattr(self, name)
            if v is not None:
                df[name] = v
        return df

    def plot(self, save_path=None, axes=None):
        """Diagnostic plot; see :func:`MannKS.plot_decomposition`."""
        from .plotting import plot_decomposition
        return plot_decomposition(self, save_path=save_path, axes=axes)

    def deseasonalized_series(self, scale='transformed'):
        """
        Deseasonalised series in exactly the form MannKS trend functions accept.

        Returns ``(x, t)``: ``x`` is a DataFrame with columns ``value``,
        ``censored`` and ``cen_type`` (as from ``prepare_censored_data``), and
        ``t`` the original times in sorted order.  Use as::

            x, t = res.deseasonalized_series()
            MannKS.trend_test(x, t, slope_scaling='year')

        Detected values become ``value - seasonal``; censored values stay
        censored with limit ``limit - seasonal`` (on the original scale,
        ``DL * exp(-seasonal)``), keeping their censoring type.

        Parameters
        ----------
        scale : {'transformed', 'original'}
            'transformed' (default) returns the series on the fitting scale
            (ln units when ``transform='log'``); slopes are then in ln-units
            per time unit.  'original' back-transforms (log mode only).  The
            Mann-Kendall S and p-value are identical on both scales; Sen/ATS
            slopes are not.
        """
        if scale not in ('transformed', 'original'):
            raise ValueError("`scale` must be 'transformed' or 'original'.")
        v = self.values - self.seasonal
        if scale == 'original':
            if self.transform != 'log':
                raise ValueError("scale='original' is only available with transform='log'.")
            v = np.exp(v)
        x = pd.DataFrame({
            'value': np.asarray(v, dtype=float),
            'censored': np.asarray(self.censored, dtype=bool).copy(),
            'cen_type': np.asarray(self.cen_type, dtype=object).copy(),
        })
        return x, self.times


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _take_times(t, order):
    if isinstance(t, pd.Series):
        return t.reset_index(drop=True).iloc[order].reset_index(drop=True)
    if isinstance(t, pd.Index):
        return t[order]
    return np.asarray(t)[order]


def _as_int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, (Integral, float)) or \
            not float(value).is_integer() or value < minimum:
        raise ValueError(f"`{name}` must be an integer >= {minimum}; got {value!r}.")
    return int(value)


def _check_options(censored_method, transform, n_harmonics, trend_span_years, seasonal,
                   seasonal_window_years, robust, n_outer, backfit_tol, backfit_max_iter,
                   impute_tol, impute_max_iter, n_boot, block_years, n_boot_auto, n_imputations):
    if censored_method not in ('parametric', 'ros'):
        raise ValueError("`censored_method` must be 'parametric' or 'ros'.")
    if transform not in ('log', 'none'):
        raise ValueError("`transform` must be 'log' or 'none'.")
    if n_harmonics != 'auto':
        if isinstance(n_harmonics, bool) or not isinstance(n_harmonics, Integral) or \
                not 1 <= n_harmonics <= _d.MAX_HARMONICS:
            raise ValueError("`n_harmonics` must be 'auto' or an integer from 1 to 4.")
    if not np.isfinite(trend_span_years) or trend_span_years < 2.0:
        raise ValueError(f"`trend_span_years` must be at least 2.0; got {trend_span_years}.")
    if seasonal not in ('global', 'time_varying', 'auto'):
        raise ValueError("`seasonal` must be 'global', 'time_varying' or 'auto'.")
    if not np.isfinite(seasonal_window_years) or seasonal_window_years < 3.0:
        raise ValueError(f"`seasonal_window_years` must be at least 3.0; got {seasonal_window_years}.")
    if not isinstance(robust, (bool, np.bool_)):
        raise ValueError("`robust` must be True or False.")
    out = {
        'n_outer': _as_int('n_outer', n_outer, 1),
        'backfit_max_iter': _as_int('backfit_max_iter', backfit_max_iter, 1),
        'impute_max_iter': _as_int('impute_max_iter', impute_max_iter, 0),
        'n_boot': _as_int('n_boot', n_boot, 0),
        'n_boot_auto': _as_int('n_boot_auto', n_boot_auto, 1),
        'n_imputations': _as_int('n_imputations', n_imputations, 0),
    }
    if out['n_imputations'] == 1:
        raise ValueError("`n_imputations` must be 0 or at least 2.")
    if not backfit_tol > 0 or not impute_tol > 0:
        raise ValueError("`backfit_tol` and `impute_tol` must be positive.")
    try:
        out['block_years'] = _as_int('block_years', block_years, 1)
    except ValueError:
        raise ValueError("`block_years` must be a positive integer number of years "
                         f"(keeps the seasonal phase when blocks are shifted); got {block_years!r}.")
    return out


def _collect_warnings(diag, settings, notes):
    """Brief v2 section 4.10 warning conditions.  Returns a list of messages."""
    msgs = []
    pc = diag['phase_coverage']
    if pc['K_max'] < _d.MAX_HARMONICS:
        msgs.append(f"Phase coverage limits the number of harmonics to K_max={pc['K_max']} "
                    f"(largest gap between supported phases {pc['max_gap']:.3f} yr).")
    if settings['n_harmonics_requested'] != 'auto' and settings['K'] > pc['K_max']:
        msgs.append(f"n_harmonics={settings['K']} exceeds what phase coverage supports "
                    f"(K_max={pc['K_max']}).")
    if diag['share_zero_weights'] > 0.10:
        msgs.append(f"{diag['share_zero_weights']:.0%} of detected values have zero robustness "
                    "weight (more than 10%).")
    if np.isfinite(diag['acf1_remainder']) and abs(diag['acf1_remainder']) > 0.3:
        msgs.append(f"Lag-1 autocorrelation of the remainder is {diag['acf1_remainder']:.2f}; "
                    "consider autocorr_method='yue_wang' or 'block_bootstrap' in the follow-up "
                    "trend test (burst sampling also inflates this value).")
    if diag['n_trend_widenings'] > 0:
        msgs.append(f"The trend window was widened at {diag['n_trend_widenings']} observation "
                    "time(s) to meet the minimum-support rules.")
    if diag['n_local_constant_fallbacks'] > 0:
        msgs.append(f"Local-constant trend fallback used at {diag['n_local_constant_fallbacks']} "
                    "observation time(s).")
    if diag['n_seasonal_widenings'] > 0:
        msgs.append(f"The seasonal window was widened at {diag['n_seasonal_widenings']} grid "
                    "point(s); see diagnostics['seasonal_window_used'].")
    conv = diag['convergence']
    if not conv['backfit_converged']:
        msgs.append("Backfitting did not converge within backfit_max_iter.")
    if conv['impute_converged'] and not all(conv['impute_converged']):
        msgs.append("Censored-value imputation did not converge within impute_max_iter.")
    if settings['robust'] and not conv['weights_converged'] and settings['n_outer'] > 1:
        notes.append("robustness weights still changing after n_outer iterations")
    if conv.get('impute_cycles'):
        notes.append("imputation settled on the mid-point of a period-2 cycle "
                     f"({conv['impute_cycles']} time(s); ROS rank ordering)")
    if conv['robustness_skipped']:
        notes.append("robustness skipped: residual scale is approximately zero")
    if settings['seasonal_requested'] in ('time_varying', 'auto') and \
            diag['span_years'] < 2 * settings['seasonal_window_years']:
        msgs.append(f"Record ({diag['span_years']:.1f} yr) is shorter than 2 x "
                    f"seasonal_window_years ({settings['seasonal_window_years']:g} yr).")
    nd = diag.get('ndimpute')
    if nd:
        if nd['warnings']:
            msgs.append(f"ndimpute issued {sum(nd['warnings'].values())} warning(s) during "
                        "imputation; see diagnostics['ndimpute']['warnings'].")
        if nd['n_fits_using_all_detects']:
            msgs.append("Too few detected values with non-zero weight; the residual imputation "
                        "model was fitted to all detected values.")
    bs = diag.get('bootstrap')
    if bs and bs['n_failed'] > 0:
        msgs.append(f"{bs['n_failed']} of {bs['n_requested']} bootstrap replicates failed "
                    "and were skipped.")
    au = diag.get('auto')
    if au and au['n_failed'] > 0:
        msgs.append(f"{au['n_failed']} of {au['n_requested']} auto-mode bootstrap replicates "
                    "failed and were skipped.")
    return msgs


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def decompose(
    x: Union[np.ndarray, pd.DataFrame, pd.Series, list],
    t,
    censored_method: str = 'parametric',
    transform: str = 'log',
    n_harmonics: Union[str, int] = 'auto',
    trend_span_years: float = 3.0,
    seasonal: str = 'global',
    seasonal_window_years: float = 7.0,
    robust: bool = True,
    n_outer: int = 5,
    backfit_tol: float = 1e-8,
    backfit_max_iter: int = 50,
    impute_tol: float = 1e-4,
    impute_max_iter: int = 50,
    n_boot: int = 0,
    block_years: float = 1.0,
    n_boot_auto: int = 199,
    n_imputations: int = 0,
    random_state: Optional[int] = None,
) -> DecompositionResult:
    """
    Robust, censoring-aware seasonal-trend decomposition (opt-in).

    Splits an irregularly sampled series into trend + seasonal + remainder on
    the transformed scale, by backfitting an annual harmonic regression and a
    robust local-linear smoother with one shared set of bisquare robustness
    weights.  Censored values (non-detects) are imputed with the optional
    ``ndimpute`` package from residuals around the current fit (model-aware),
    only to estimate the components; they stay censored in
    :meth:`DecompositionResult.deseasonalized_series`.

    The Seasonal Kendall test (``seasonal_trend_test``) remains the
    recommended trend test for seasonal data.  Use ``decompose`` to describe
    and visualise seasonality, to deseasonalise for segmented or rolling
    analyses, and to check robustness to outliers.

    Parameters
    ----------
    x : array-like or DataFrame
        Values, or the DataFrame returned by ``prepare_censored_data``
        (columns ``value``, ``censored``, ``cen_type``).  Multiple detection
        limits are simply different values on censored rows.
    t : array-like
        Times, datetime-like or numeric.  Numeric times are taken as decimal
        years (period 1.0).  Datetimes are converted to decimal years on a
        uniform 365.2425-day axis anchored at 2000-01-01.
    censored_method : {'parametric', 'ros'}
        ``ndimpute`` method for the residual imputation.
    transform : {'log', 'none'}
        'log' (default, for concentrations) fits ln(values); censored limits
        become ln(DL).
    n_harmonics : 'auto' or int 1..4
        Number of annual harmonics.  'auto' caps K by phase coverage and
        chooses it by weighted BIC.
    trend_span_years : float, default 3.0
        Full width of the trend window (minimum 2.0).
    seasonal : {'global', 'time_varying', 'auto'}
        'time_varying' lets the harmonic coefficients change over time
        (window ``seasonal_window_years``); 'auto' chooses between global and
        time-varying with a parametric block-bootstrap test (``n_boot_auto``
        replicates; about 2 x (n_boot_auto + 1) fits).
    seasonal_window_years : float, default 7.0
        Full width of the time-varying window (minimum 3.0).  Windows are
        widened where data are too sparse; see
        ``diagnostics['seasonal_window_used']``.
    robust : bool, default True
        Bisquare robustness weights shared by trend and seasonal.
    n_outer : int, default 5
        Maximum robustness iterations.
    backfit_tol, backfit_max_iter, impute_tol, impute_max_iter
        Convergence controls for backfitting and the censored-value
        imputation loop.
    n_boot : int, default 0
        Block-bootstrap replicates for pointwise and simultaneous 95% bands.
    block_years : int, default 1
        Bootstrap block length in whole years.
    n_boot_auto : int, default 199
        Bootstrap size for ``seasonal='auto'``.
    n_imputations : int, default 0
        With censored data and ``n_boot == 0``: number of stochastic
        re-imputations used to report ``trend_imputation_sd`` and
        ``seasonal_imputation_sd`` (a lower bound on imputation uncertainty).
    random_state : int, optional
        Seed.  None uses fresh entropy (as MannKS's block bootstrap).  The
        global NumPy random state is never used.

    Returns
    -------
    DecompositionResult
    """
    opts = _check_options(censored_method, transform, n_harmonics, trend_span_years, seasonal,
                          seasonal_window_years, robust, n_outer, backfit_tol, backfit_max_iter,
                          impute_tol, impute_max_iter, n_boot, block_years, n_boot_auto,
                          n_imputations)
    records = []
    notes = []
    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter('always')
        res = _decompose(x, t, censored_method, transform, n_harmonics, float(trend_span_years),
                         seasonal, float(seasonal_window_years), bool(robust), opts,
                         float(backfit_tol), float(impute_tol), random_state, notes)
        for w in w_log:
            records.append((w.message, w.category))

    msgs = [str(m) for m, _ in records]
    final_msgs = list(dict.fromkeys(msgs + res.warnings))
    res = res._replace(warnings=final_msgs,
                       analysis_notes=list(dict.fromkeys(res.analysis_notes + notes)))
    seen = set()
    for message, category in records:
        key = str(message)
        if key not in seen:
            seen.add(key)
            warnings.warn(message if isinstance(message, Warning) else str(message), category,
                          stacklevel=2)
    for m in res.warnings:
        if m not in seen:
            seen.add(m)
            warnings.warn(m, UserWarning, stacklevel=2)
    return res


def _decompose(x, t, censored_method, transform, n_harmonics, trend_span_years, seasonal,
               seasonal_window_years, robust, opts, backfit_tol, impute_tol, random_state, notes):
    # ---- inputs (MannKS conventions) ------------------------------------
    if isinstance(x, (pd.DataFrame, pd.Series)):
        x_work = x.reset_index(drop=True)
    else:
        x_work = x
    t_work = t.reset_index(drop=True) if isinstance(t, pd.Series) else t
    if len(x_work) != len(t_work):
        raise ValueError(f"Input vectors `x` and `t` must have the same length. "
                         f"Got {len(x_work)} and {len(t_work)}.")
    data, is_datetime = _prepare_data(x_work, t_work, hicensor=False)
    data = data.sort_values('t', kind='stable')
    order = data.index.to_numpy()

    cens = data['censored'].to_numpy(dtype=bool)
    cen_type = data['cen_type'].astype(str).to_numpy()
    if np.any(cens & ~np.isin(cen_type, ['lt', 'gt'])):
        raise ValueError("Censored rows must have cen_type 'lt' or 'gt'.")
    if np.any(~cens & (cen_type != 'not')):
        raise ValueError("Rows with censored=False must have cen_type 'not'.")
    cen_type = np.where(cens, cen_type, 'not').astype(object)
    values = data['value'].to_numpy(dtype=float)
    t_num = data['t'].to_numpy(dtype=float)
    t_years = _d.seconds_to_decimal_years(t_num) if is_datetime else t_num.copy()
    n = t_years.size
    if n == 0:
        raise ValueError("No usable observations.")
    span = float(t_years[-1] - t_years[0])
    if not is_datetime and span > 1000:
        warnings.warn("Numeric `t` spans more than 1000 units; decompose interprets numeric "
                      "times as decimal years (did you pass seconds or days?).", UserWarning)

    if transform == 'log':
        if np.any(values <= 0):
            raise ValueError("transform='log' requires all values and detection limits to be > 0; "
                             f"found {int(np.sum(values <= 0))} value(s) <= 0.")
        x_lim = np.log(values)
    else:
        x_lim = values.copy()

    if span < 3.0:
        raise ValueError(f"At least 3 years of data are required; the record spans {span:.2f} years.")
    n_det = int(np.sum(~cens))
    if n_det < 24:
        raise ValueError(f"At least 24 detected values are required; got {n_det}.")
    prop_cens = float(np.mean(cens))
    if prop_cens > 0.8:
        raise ValueError(f"{prop_cens:.0%} of values are censored; at most 80% is supported.")
    if prop_cens > 0.5:
        warnings.warn(f"{prop_cens:.0%} of values are censored (more than 50%); the "
                      "decomposition relies heavily on the imputation model.", UserWarning)
    if len(np.unique(t_years)) < n:
        notes.append('tied timestamps present without aggregation')
    if seasonal == 'auto' and opts['n_boot_auto'] < 20:
        warnings.warn(f"n_boot_auto={opts['n_boot_auto']} is below 20: the smallest attainable "
                      "p-value is 1/(n_boot_auto + 1) >= 0.05, so time-varying mode can never be "
                      "selected.", UserWarning)
    L = opts['block_years']
    if (opts['n_boot'] > 0 or seasonal == 'auto') and _d.n_full_blocks(t_years, L) < 2:
        raise ValueError("The record must contain at least two full bootstrap blocks "
                         f"(block_years={L}).")

    # ---- fitting --------------------------------------------------------
    cfg = dict(h=trend_span_years / 2.0, robust=robust, n_outer=opts['n_outer'],
               backfit_tol=backfit_tol, backfit_max_iter=opts['backfit_max_iter'],
               impute_tol=impute_tol, impute_max_iter=opts['impute_max_iter'])
    has_cens = bool(np.any(cens))
    if has_cens:
        from ._decompose_censored import ResidualImputer
        imputer = ResidualImputer(cen_type, censored_method)
    else:
        imputer = None
    ss = np.random.SeedSequence(random_state)
    ss_boot, ss_auto, ss_imp = ss.spawn(3)

    K_max, cov = _d.phase_coverage(t_years)
    H = seasonal_window_years / 2.0

    def global_factory(K, tt=None):
        tt = t_years if tt is None else tt
        return lambda w: _d.GlobalSeasonal(tt, K)

    def tv_factory(K, tt=None):
        tt = t_years if tt is None else tt
        return lambda w: _d.TimeVaryingSeasonal(tt, w, K, H)

    bics = None
    if n_harmonics == 'auto':
        fit = _d.fit_pipeline(t_years, x_lim, cens, global_factory(K_max), cfg, imputer)
        bics = {K: float(_d.bic_for_K(t_years, fit, cens, K, cfg)) for K in range(1, K_max + 1)}
        K = min(bics, key=bics.get)
        if K != K_max:
            fit = _d.fit_pipeline(t_years, x_lim, cens, global_factory(K), cfg, imputer)
    else:
        K = int(n_harmonics)
        fit = _d.fit_pipeline(t_years, x_lim, cens, global_factory(K), cfg, imputer)
    global_fit = fit
    mode_used = 'global'
    pvalue = np.nan
    auto_diag = None

    def mu_fn_of(f, tt):
        sm, coef, trend = f['seasonal_model'], f['coef'], f['trend']
        return lambda t_new: _d.interp_at(tt, trend, t_new) + sm.evaluate(coef, t_new)

    def make_imputer(ct):
        if not np.any(ct != 'not'):
            return None
        from ._decompose_censored import ResidualImputer
        return ResidualImputer(ct, censored_method)

    def rss_tv_fixed(f, tt, cc):
        sm = _d.TimeVaryingSeasonal(tt, f['w'], K, H)
        r = _d.backfit(tt, f['x'], f['w'], sm, cfg['h'], (f['trend'], f['seasonal']),
                       tol=cfg['backfit_tol'], max_iter=cfg['backfit_max_iter'])
        return _d.weighted_rss(dict(f, trend=r['trend'], seasonal=r['seasonal']), cc)[0]

    def f_stat(f, tt, cc):
        rss_g, sw, _ = _d.weighted_rss(f, cc)
        rss_t = rss_tv_fixed(f, tt, cc)
        floor = 1e-12 * max(sw, 1.0)
        return (rss_g - rss_t) / max(rss_t, floor)

    if seasonal == 'auto':
        F = f_stat(global_fit, t_years, cens)
        mu_g = mu_fn_of(global_fit, t_years)
        mu_obs = global_fit['trend'] + global_fit['seasonal']
        B = opts['n_boot_auto']
        F_star = []
        n_failed, first_err = 0, None
        for child in ss_auto.spawn(B):
            rng = np.random.default_rng(child)
            try:
                tb, xb, cb, ctb = _d.block_bootstrap_series(
                    t_years, x_lim, global_fit['x'], cens, cen_type, mu_obs, mu_g, L, rng)
                fb = _d.fit_pipeline(tb, xb, cb, global_factory(K, tb), cfg, make_imputer(ctb))
                F_star.append(f_stat(fb, tb, cb))
            except Exception as e:  # noqa: BLE001 - a failed replicate is skipped and counted
                n_failed += 1
                first_err = first_err or f"{type(e).__name__}: {e}"
        F_star = np.asarray(F_star)
        pvalue = float((1 + np.sum(F_star >= F)) / (F_star.size + 1))
        auto_diag = {'F': float(F), 'pvalue': pvalue, 'n_requested': B,
                     'n_used': int(F_star.size), 'n_failed': n_failed,
                     'first_error': first_err}
        if pvalue < 0.05:
            fit = _d.fit_pipeline(t_years, x_lim, cens, tv_factory(K), cfg, imputer)
            mode_used = 'time_varying'
    elif seasonal == 'time_varying':
        fit = _d.fit_pipeline(t_years, x_lim, cens, tv_factory(K), cfg, imputer)
        mode_used = 'time_varying'

    sm = fit['seasonal_model']
    trend, seas, w, x_work_vals = fit['trend'], fit['seasonal'], fit['w'], fit['x']

    # ---- bootstrap bands -------------------------------------------------
    bands = dict.fromkeys(['trend_low', 'trend_high', 'trend_simul_low', 'trend_simul_high',
                           'seasonal_low', 'seasonal_high'])
    boot_diag = None
    if opts['n_boot'] > 0:
        mu_f = mu_fn_of(fit, t_years)
        mu_obs = trend + seas
        tr_reps, se_reps = [], []
        n_failed, first_err = 0, None
        for child in ss_boot.spawn(opts['n_boot']):
            rng = np.random.default_rng(child)
            try:
                tb, xb, cb, ctb = _d.block_bootstrap_series(
                    t_years, x_lim, x_work_vals, cens, cen_type, mu_obs, mu_f, L, rng)
                fac = (tv_factory(K, tb) if mode_used == 'time_varying'
                       else global_factory(K, tb))
                imp_b = make_imputer(ctb)
                fb = _d.fit_pipeline(tb, xb, cb, fac, cfg, imp_b)
                if imp_b is not None:
                    seed = int(rng.integers(0, 2 ** 63 - 1))
                    new = imp_b.conditional(xb, fb['trend'] + fb['seasonal'], fb['w'],
                                            stochastic=True, seed=seed)
                    xs = fb['x'].copy()
                    xs[cb] = new
                    rb = _d.backfit(tb, xs, fb['w'], fb['seasonal_model'], cfg['h'],
                                    (fb['trend'], fb['seasonal']), tol=cfg['backfit_tol'],
                                    max_iter=cfg['backfit_max_iter'])
                    trb, coefb = rb['trend'], rb['coef']
                else:
                    trb, coefb = fb['trend'], fb['coef']
                tr_reps.append(_d.interp_at(tb, trb, t_years))
                se_reps.append(fb['seasonal_model'].evaluate(coefb, t_years))
            except Exception as e:  # noqa: BLE001 - a failed replicate is skipped and counted
                n_failed += 1
                first_err = first_err or f"{type(e).__name__}: {e}"
        boot_diag = {'n_requested': opts['n_boot'], 'n_used': len(tr_reps), 'n_failed': n_failed,
                     'block_years': L, 'first_error': first_err}
        if len(tr_reps) >= 2:
            TR, SE = np.vstack(tr_reps), np.vstack(se_reps)
            bands['trend_low'], bands['trend_high'] = np.percentile(TR, [2.5, 97.5], axis=0)
            bands['seasonal_low'], bands['seasonal_high'] = np.percentile(SE, [2.5, 97.5], axis=0)
            sd = TR.std(axis=0, ddof=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                z = np.where(sd > 0, np.abs(TR - trend) / sd, 0.0)
            c = float(np.percentile(z.max(axis=1), 95))
            bands['trend_simul_low'], bands['trend_simul_high'] = trend - c * sd, trend + c * sd
            boot_diag['simultaneous_c'] = c

    # ---- stochastic re-imputations (n_boot == 0) --------------------------
    tr_sd = se_sd = None
    if opts['n_imputations'] > 0:
        if opts['n_boot'] > 0:
            notes.append('n_imputations ignored because n_boot > 0 (bands include imputation)')
        elif not has_cens:
            notes.append('n_imputations ignored: no censored values')
        else:
            TRm, SEm = [], []
            for child in ss_imp.spawn(opts['n_imputations']):
                seed = int(np.random.default_rng(child).integers(0, 2 ** 63 - 1))
                new = imputer.conditional(x_lim, trend + seas, w, stochastic=True, seed=seed)
                xs = x_work_vals.copy()
                xs[cens] = new
                r = _d.backfit(t_years, xs, w, sm, cfg['h'], (trend, seas),
                               tol=cfg['backfit_tol'], max_iter=cfg['backfit_max_iter'])
                TRm.append(r['trend'])
                SEm.append(r['seasonal'])
            tr_sd = np.vstack(TRm).std(axis=0, ddof=1)
            se_sd = np.vstack(SEm).std(axis=0, ddof=1)

    # ---- descriptors -------------------------------------------------------
    coef = fit['coef']
    by_year = None
    if mode_used == 'global':
        amp, ratio, peak, trough = _d.curve_descriptors(coef, K, transform)
    else:
        amp = ratio = np.nan
        peak = trough = None
        rows = []
        for yr in range(int(np.floor(t_years[0])), int(np.floor(t_years[-1])) + 1):
            mid = float(np.clip(yr + 0.5, t_years[0], t_years[-1]))
            c_y = sm.coef_at(coef, [mid])[0]
            a_y, r_y, p_y, tr_y = _d.curve_descriptors(c_y, K, transform)
            rows.append({'year': yr, 'amplitude': a_y, 'amplitude_ratio': r_y,
                         'peak_day_of_year': p_y, 'trough_day_of_year': tr_y})
        by_year = pd.DataFrame(rows)

    # ---- diagnostics -------------------------------------------------------
    remainder = x_work_vals - trend - seas
    det = ~cens
    acf1 = float(estimate_acf(remainder[det], max_lag=1)[0][1]) if det.sum() > 2 else np.nan
    reason = ('phase coverage' if K_max < _d.MAX_HARMONICS else 'no cap')
    diag = {
        'phase_coverage': dict(cov, reason=reason),
        'bic': bics,
        'share_zero_weights': float(np.mean(w[det] == 0)),
        'acf1_remainder': acf1,
        'n_trend_widenings': int(np.sum(fit['widened'])),
        'n_local_constant_fallbacks': int(np.sum(fit['fallback'])),
        'n_seasonal_widenings': int(getattr(sm, 'n_widen', 0)),
        'seasonal_window_used': getattr(sm, 'window_used', None),
        'seasonal_grid': getattr(sm, 'grid', None),
        'convergence': fit['convergence'],
        'ndimpute': imputer.diagnostics() if imputer is not None else None,
        'prop_censored': prop_cens,
        'n': n,
        'span_years': span,
        'auto': auto_diag,
        'bootstrap': boot_diag,
    }
    settings = {
        'is_datetime': is_datetime, 'K': K, 'n_harmonics_requested': n_harmonics,
        'seasonal_requested': seasonal, 'seasonal_window_years': seasonal_window_years,
        'trend_span_years': trend_span_years, 'robust': robust, 'n_outer': opts['n_outer'],
        'censored_method': censored_method, 'seasonal_model': sm, 'coef': coef,
        'random_state': random_state,
    }
    msgs = _collect_warnings(diag, settings, notes)

    if mode_used == 'global':
        coefficients = coef
    else:
        coefficients = {'grid': sm.grid, 'coef': coef}

    return DecompositionResult(
        times=_take_times(t, order), t_years=t_years, order=order, values=x_lim,
        censored=cens, cen_type=cen_type, trend=trend, seasonal=seas, remainder=remainder,
        weights=w, fitted_values=np.where(cens, x_work_vals, np.nan),
        transform=transform, n_harmonics_used=K, K_max=K_max, seasonal_mode_used=mode_used,
        seasonal_constancy_pvalue=pvalue, coefficients=coefficients,
        seasonal_amplitude=amp, seasonal_amplitude_ratio=ratio, peak_day_of_year=peak,
        trough_day_of_year=trough, seasonal_by_year=by_year,
        trend_imputation_sd=tr_sd, seasonal_imputation_sd=se_sd,
        diagnostics=diag, warnings=msgs, analysis_notes=list(notes), settings=settings,
        **bands,
    )


def deseasonalized_trend_test(res: DecompositionResult, scale: str = 'transformed', **kwargs):
    """
    Run :func:`MannKS.trend_test` on ``res.deseasonalized_series(scale)``.

    Censored values are passed as censored with their shifted limits, so the
    censored Kendall test and the Akritas-Theil-Sen slope are used.  With
    censored data only ``sens_slope_method='ats'`` (the default) is allowed:
    the other methods multiply censored values by ``lt_mult``/``gt_mult``,
    which is a substitution and is wrong for shifted log-scale limits.

    On ``scale='transformed'`` with ``transform='log'`` the slope is in
    ln-units per time unit; with ``slope_scaling='year'`` and small slopes,
    100 x slope is approximately the percentage change per year.
    """
    if np.any(res.censored) and kwargs.get('sens_slope_method', 'ats') != 'ats':
        raise ValueError("With censored values, deseasonalized_trend_test requires "
                         "sens_slope_method='ats'; other methods substitute multiples of the "
                         "detection limit (lt_mult/gt_mult).")
    if kwargs.get('hicensor'):
        warnings.warn("hicensor with a deseasonalised series: the shifted detection limits "
                      "differ by observation, so the high-censor rule censors much more data "
                      "than on the original series.", UserWarning, stacklevel=2)
    from .trend_test import trend_test
    x, t = res.deseasonalized_series(scale)
    return trend_test(x, t, **kwargs)
