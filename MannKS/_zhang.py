"""
Zhang trend-free pre-whitening for the Mann-Kendall test.

Port of ``zyp::zyp.zhang`` (Bronaugh & Werner, R package *zyp*; method of
Zhang et al., 2000, and Wang & Swail, 2001), which is what USGS EGRET uses for
the serial-correlation-adjusted p-value (``pValueAdj``) that colours its
Quantile-Kendall plot.

The port follows the R code step by step, including its final significance
test, ``Kendall::Kendall``.  That routine (Fortran ``TAUK2``) works in single
precision and always uses the continuity-corrected normal approximation with
Kendall's variance for ties, so values that agree to single precision are
tied.  Both properties are reproduced here; p-values agree with R to about
single-precision accuracy (~1e-6).
"""
import numpy as np
from scipy.stats import norm

_ACF_THRESHOLD = 0.05      # zyp: pre-whiten only if lag-1 autocorrelation >= 0.05
_MAX_ITER = 500


def _acf1(x: np.ndarray) -> float:
    """
    Lag-1 autocorrelation as R's ``acf(x, lag.max = 1)$acf[2]``.

    The arithmetic follows R operation by operation (mean accumulated in
    extended precision, strictly sequential sums of products, division by
    ``sqrt(acf0)**2``).  Zhang's iteration does not always converge and is
    then sensitive to the last bit of this number, so reproducing zyp's
    stopping point requires reproducing its rounding.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = float(np.cumsum(x.astype(np.longdouble))[-1] / n)
    d = x - mean
    acf0 = float(np.cumsum(d * d)[-1]) / n
    if not acf0 > 0.0:
        return np.nan
    acf1 = float(np.cumsum(d[1:] * d[:-1])[-1]) / n
    se = np.sqrt(acf0)
    a = acf1 / (se * se)
    return float(min(1.0, max(-1.0, a)))


def _sen_slope(y: np.ndarray, t: np.ndarray) -> float:
    """``zyp.sen``: median of all finite pairwise slopes."""
    i, j = np.triu_indices(len(y), k=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        slopes = (y[i] - y[j]) / (t[i] - t[j])
    slopes = slopes[np.isfinite(slopes)]
    return float(np.median(slopes)) if len(slopes) else np.nan


def _tie_sums(v: np.ndarray):
    _, counts = np.unique(v, return_counts=True)
    c = counts.astype(float)
    return (float(np.sum(c * (c - 1.0))),
            float(np.sum(c * (c - 1.0) * (2.0 * c + 5.0))),
            float(np.sum(c * (c - 1.0) * (c - 2.0))))


def _kendall_single_precision(x: np.ndarray, y: np.ndarray):
    """
    ``Kendall::Kendall(x, y)``: (tau, two-sided p-value).

    Inputs are compared in single precision, as the R function passes them
    with ``as.single``.  The p-value is the continuity-corrected normal
    approximation with Kendall's tie-corrected variance.  If either variable
    is constant the routine exits early and R reports ``tau = 1, sl = 1``.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    n = len(x)
    if n < 2 or np.all(x == x[0]) or np.all(y == y[0]) or n <= 3:
        return 1.0, 1.0
    i, j = np.triu_indices(n, k=1)
    s = float(np.sum(np.sign(x[j] - x[i]).astype(float) * np.sign(y[j] - y[i]).astype(float)))
    cn = float(n) * float(n - 1)
    dn = 0.5 * cn
    sumt, suma1, suma2 = _tie_sums(x)
    sumu, sumb1, sumb2 = _tie_sums(y)
    d1 = dn - 0.5 * sumt
    d2 = dn - 0.5 * sumu
    if d1 <= 0.0 or d2 <= 0.0:
        return 1.0, 1.0
    tau = s / (np.sqrt(d1) * np.sqrt(d2))
    var_s = ((cn * float(2 * n + 5) - suma1 - sumb1) / 18.0
             + suma2 * sumb2 / (9.0 * cn * float(n - 2))
             + sumt * sumu / (cn + cn))
    scor = s - 1.0 if s > 0 else (s + 1.0 if s < 0 else 0.0)
    prob = float(norm.cdf(scor / np.sqrt(var_s)))
    p = 2.0 * prob if prob < 0.5 else 2.0 * (1.0 - prob)
    return float(tau), float(p)


def zhang_prewhitened_test(y, t) -> dict:
    """
    Mann-Kendall test after Zhang's iterative trend-free pre-whitening.

    Parameters
    ----------
    y : array-like
        Series values (no missing values).
    t : array-like
        Times of the values (numeric).

    Returns
    -------
    dict
        ``p`` (two-sided p-value of the pre-whitened series; zyp ``sig``),
        ``tau`` (its Kendall tau), ``trend`` (Sen slope of the pre-whitened
        series per unit of ``t``), ``acf1`` (lag-1 autocorrelation used; zyp
        ``autocor``) and ``n_iter`` (zyp ``nruns``).  All ``nan`` when zyp
        would return ``NA`` (too few values, fewer than 3 non-zero values, or
        a constant series).
    """
    data = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(data)
    out = {'p': np.nan, 'tau': np.nan, 'trend': np.nan, 'acf1': np.nan, 'n_iter': np.nan}
    if n < 3 or np.isnan(data).any() or np.isnan(t).any():
        return out

    c = _acf1(data)
    if np.isnan(c):
        return out

    if c < _ACF_THRESHOLD:
        yw, tw = data, t
    else:
        yw = data[1:] - c * data[:-1]
        tw = t[:-1]

    if len(yw) <= 3 or np.count_nonzero(yw) < 3 or len(yw) / n < 0.1:
        return out

    trend = _sen_slope(yw, tw)
    k = 1
    c0 = c
    if c >= _ACF_THRESHOLD:
        trend0 = trend
        while k < _MAX_ITER:
            c = _acf1(data - trend * t)
            if np.isnan(c):
                break
            if c < _ACF_THRESHOLD and abs(c - c0) <= 1e-4:
                break
            yw = (data[1:] - c * data[:-1]) / (1.0 - c)
            trend = _sen_slope(yw, tw)
            k += 1
            with np.errstate(divide='ignore', invalid='ignore'):
                # numpy scalars: x / 0 is inf or nan as in R, not an exception
                tpdiff = float(np.abs((np.float64(trend) - np.float64(trend0)) / np.float64(trend)))
            if not np.isnan(tpdiff) and tpdiff <= 1e-3 and abs(c - c0) <= 1e-4:
                break
            trend0 = trend
            c0 = c

    tau, p = _kendall_single_precision(tw, yw)
    out.update({'p': p, 'tau': tau, 'trend': trend, 'acf1': float(c0), 'n_iter': float(k)})
    return out
