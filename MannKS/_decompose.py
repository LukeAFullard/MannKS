"""
Numerical core of the seasonal-trend decomposition (``MannKS.decompose``).

Annual harmonic regression (seasonal) + robust local-linear smoother with a
time-based window (trend), combined by backfitting with one shared set of
bisquare robustness weights.  Censored values are handled by a separate
imputer object (``_decompose_censored.ResidualImputer``); this module never
imports ``ndimpute``.

All times are decimal years on a uniform axis (365.2425-day years).
Everything here is numpy / scipy only.
"""
import numpy as np

_EPOCH_2000 = 946_684_800.0             # 2000-01-01T00:00:00 UTC in Unix seconds
_SECONDS_PER_YEAR = 365.2425 * 86_400.0

PHASE_TOL = 1.0 / 48.0                  # +/- 7.6 days, phase-support neighbourhood
MAX_HARMONICS = 4


# ---------------------------------------------------------------------------
# Time and basis
# ---------------------------------------------------------------------------
def seconds_to_decimal_years(sec):
    """Unix seconds -> decimal years with a uniform 365.2425-day year."""
    return 2000.0 + (np.asarray(sec, dtype=float) - _EPOCH_2000) / _SECONDS_PER_YEAR


def tricube(u):
    a = np.minimum(np.abs(u), 1.0)
    k = 1.0 - a * a * a
    return k * k * k


def harmonics(t, K):
    """Columns cos(2 pi k t), sin(2 pi k t) for k = 1..K (no intercept)."""
    t = np.asarray(t, dtype=float)
    k = np.arange(1, K + 1, dtype=float)
    ang = 2.0 * np.pi * t[:, None] * k[None, :]
    out = np.empty((t.size, 2 * K))
    out[:, 0::2] = np.cos(ang)
    out[:, 1::2] = np.sin(ang)
    return out


def mad(x):
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


# ---------------------------------------------------------------------------
# Phase coverage (brief v2 section 4.2)
# ---------------------------------------------------------------------------
def phase_coverage(t_years, tol=PHASE_TOL, max_K=MAX_HARMONICS):
    """
    Largest circular gap between *supported* phases and the implied K_max.

    A phase is supported when observations from at least
    ``m = max(2, ceil(0.1 * n_years))`` distinct calendar years fall within
    +/- ``tol`` years of it.  ``K_max = clip(floor(1 / (2 * max_gap)), 1, max_K)``.
    """
    t_years = np.asarray(t_years, dtype=float)
    n = t_years.size
    info = {'max_gap': 1.0, 'n_supported': 0, 'm': 2, 'n_years': 0, 'K_max': 1}
    if n == 0:
        return 1, info
    phase = np.mod(t_years, 1.0)
    years = np.floor(t_years).astype(np.int64)
    uy, yid = np.unique(years, return_inverse=True)
    n_years = uy.size
    m = max(2, int(np.ceil(0.1 * n_years)))

    srt = np.argsort(phase, kind='stable')
    ph = phase[srt]
    yi = yid[srt]
    lo_wrap = ph < tol
    hi_wrap = ph > 1.0 - tol
    ext_ph = np.concatenate([ph[hi_wrap] - 1.0, ph, ph[lo_wrap] + 1.0])
    ext_y = np.concatenate([yi[hi_wrap], yi, yi[lo_wrap]])

    # cumulative per-year presence counts along the sorted, wrapped phases
    onehot = np.zeros((ext_ph.size, n_years), dtype=np.int32)
    onehot[np.arange(ext_ph.size), ext_y] = 1
    cum = np.zeros((ext_ph.size + 1, n_years), dtype=np.int32)
    np.cumsum(onehot, axis=0, out=cum[1:])
    lo = np.searchsorted(ext_ph, ph - tol, side='left')
    hi = np.searchsorted(ext_ph, ph + tol, side='right')
    distinct = ((cum[hi] - cum[lo]) > 0).sum(axis=1)

    sp = np.unique(ph[distinct >= m])
    if sp.size < 2:
        max_gap = 1.0
    else:
        gaps = np.diff(sp)
        max_gap = float(max(gaps.max(), sp[0] + 1.0 - sp[-1]))
    K_max = int(np.clip(np.floor(1.0 / (2.0 * max_gap) + 1e-9), 1, max_K))
    info.update(max_gap=max_gap, n_supported=int(sp.size), m=m, n_years=int(n_years), K_max=K_max)
    return K_max, info


# ---------------------------------------------------------------------------
# Trend smoother (brief v2 section 4.3)
# ---------------------------------------------------------------------------
def local_linear(t, z, w, h, min_pts=10, cond_max=1e8):
    """
    Weighted local-linear regression evaluated at every observation time.

    Kernel weights are ``w_i * tricube((t_i - tau) / h)``.  For each target the
    half-width is widened in 10 % steps (capped at the record span) until the
    window has >= ``min_pts`` non-zero weights, >= 3 distinct timestamps and a
    well-conditioned 2x2 design; if that fails at the cap, a local-constant
    (weighted mean) estimate is used.

    Returns ``(fit, hat_diag, widened_mask, fallback_mask)``.  ``t`` must be
    sorted.
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    w = np.asarray(w, dtype=float)
    n = t.size
    span = float(t[-1] - t[0]) if n > 1 else 0.0
    h_cap = max(span, h)
    fit = np.empty(n)
    hat = np.empty(n)
    widened = np.zeros(n, dtype=bool)
    fallback = np.zeros(n, dtype=bool)
    uid = np.unique(t, return_inverse=True)[1]  # non-decreasing because t is sorted
    chunk = max(1, min(64, 2_000_000 // max(n, 1)))

    for start in range(0, n, chunk):
        idx = np.arange(start, min(n, start + chunk))
        hrow = np.full(idx.size, float(h))
        todo = np.arange(idx.size)
        while todo.size:
            tgt = idx[todo]
            tau = t[tgt]
            hh = hrow[todo]
            hmax = hh.max()
            c0 = np.searchsorted(t, tau.min() - hmax, side='left')
            c1 = np.searchsorted(t, tau.max() + hmax, side='right')
            tc, zc, wc, idc = t[c0:c1], z[c0:c1], w[c0:c1], uid[c0:c1]

            d = tc[None, :] - tau[:, None]
            k = tricube(d / hh[:, None]) * wc[None, :]
            nz = k > 0
            cnt = nz.sum(axis=1)
            idm = np.where(nz, idc[None, :], -1)
            prev = np.empty_like(idm)
            prev[:, 0] = -1
            if idm.shape[1] > 1:
                prev[:, 1:] = np.maximum.accumulate(idm, axis=1)[:, :-1]
            distinct = (nz & (idc[None, :] != prev)).sum(axis=1)

            S0 = k.sum(axis=1)
            with np.errstate(invalid='ignore', divide='ignore'):
                dbar = (k * d).sum(axis=1) / S0
                dc = d - dbar[:, None]
                S2c = (k * dc ** 2).sum(axis=1)
                a, b = S0, S2c / hh ** 2
                cond = np.where(np.minimum(a, b) > 0, np.maximum(a, b) / np.minimum(a, b), np.inf)
                zbar = (k * zc[None, :]).sum(axis=1) / S0
                slope = (k * dc * zc[None, :]).sum(axis=1) / S2c

            ok = (cnt >= min_pts) & (distinct >= 3) & (cond < cond_max)
            at_cap = hh >= h_cap
            const = ~ok & at_cap
            done = ok | const

            w_t = w[tgt]
            with np.errstate(invalid='ignore', divide='ignore'):
                lin_fit = zbar - dbar * slope
                lin_hat = w_t * (1.0 / S0 + dbar ** 2 / S2c)
                const_hat = w_t / S0
            res_fit = np.where(ok, lin_fit, zbar)
            res_hat = np.where(ok, lin_hat, const_hat)
            bad = done & ~(S0 > 0)
            if np.any(bad):  # no positive weight even over the whole record
                res_fit[bad] = np.mean(z)
                res_hat[bad] = 0.0

            rows = tgt[done]
            fit[rows] = res_fit[done]
            hat[rows] = res_hat[done]
            widened[rows] = hh[done] > h * (1 + 1e-12)
            fallback[rows] = const[done]

            remaining = todo[~done]
            hrow[remaining] = np.minimum(hrow[remaining] * 1.1, h_cap)
            todo = remaining
    return fit, hat, widened, fallback


def interp_at(t_obs, values, t_new):
    """Linear interpolation of per-observation values (duplicate times allowed)."""
    tu, first = np.unique(t_obs, return_index=True)
    return np.interp(np.asarray(t_new, dtype=float), tu, np.asarray(values, dtype=float)[first])


# ---------------------------------------------------------------------------
# Seasonal models
# ---------------------------------------------------------------------------
class GlobalSeasonal:
    """Global annual harmonics, weighted least squares."""
    mode = 'global'

    def __init__(self, t, K):
        self.K = K
        self.B = harmonics(t, K)
        self.n_widen = 0
        self.window_used = None

    def fit(self, r, w):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(self.B * sw[:, None], r * sw, rcond=None)[0]
        return self.B @ coef, coef

    def evaluate(self, coef, t_new):
        return harmonics(t_new, self.K) @ coef

    def coef_at(self, coef, t_new):
        return np.tile(coef, (np.size(t_new), 1))


class TimeVaryingSeasonal:
    """
    Local harmonic fits on a grid (brief v2 section 4.7).

    Windows depend only on the times and the weights, so they are built once
    per weight vector.  Coefficients are estimated on a grid (step 0.1 year)
    and interpolated linearly.  A window widened to the record span uses the
    global weights (no kernel), which makes that grid point identical to
    global mode.
    """
    mode = 'time_varying'

    def __init__(self, t, w, K, H, c_ess=5.0, cond_max=1e6, grid_step=0.1):
        t = np.asarray(t, dtype=float)
        w = np.asarray(w, dtype=float)
        self.t = t
        self.K = K
        self.B = harmonics(t, K)
        span = float(t[-1] - t[0])
        n_grid = max(2, int(np.ceil(span / grid_step - 1e-9)) + 1)
        self.grid = np.linspace(t[0], t[-1], n_grid)
        p = 2 * K
        G = self.grid.size
        self.kernel = np.empty((G, t.size))
        self.H_used = np.empty(G)
        self.capped = np.zeros(G, dtype=bool)
        self.n_widen = 0
        for g, tau in enumerate(self.grid):
            Hg = float(H)
            widened = False
            while True:
                if Hg >= span:
                    k = w.copy()
                    Hg = span
                    self.capped[g] = True
                    break
                k = tricube((t - tau) / Hg) * w
                nz = k > 0
                ok = False
                if nz.sum() >= p:
                    kmax_win, _ = phase_coverage(t[nz])
                    ess = k.sum() ** 2 / np.sum(k ** 2)
                    if kmax_win >= K and ess >= c_ess * p:
                        Bn = self.B[nz]
                        A = Bn.T @ (Bn * k[nz, None])
                        ok = np.linalg.cond(A) < cond_max ** 2
                if ok:
                    break
                Hg = min(Hg * 1.1, span)
                widened = True
            self.kernel[g] = k
            self.H_used[g] = Hg
            self.n_widen += int(widened)
        # Gram matrices per grid point (chunked to bound memory)
        self.A = np.empty((G, p, p))
        step = max(1, 4_000_000 // max(1, t.size * p))
        for s in range(0, G, step):
            kk = self.kernel[s:s + step]
            self.A[s:s + step] = np.einsum('gn,nk,nl->gkl', kk, self.B, self.B, optimize=True)
        self.window_used = 2.0 * self.H_used

    def _solve_rows(self, rows, rhs):
        A = self.A[rows]
        try:
            return np.linalg.solve(A, rhs[..., None])[..., 0]
        except np.linalg.LinAlgError:
            out = np.empty_like(rhs)
            for i in range(rhs.shape[0]):
                out[i] = np.linalg.lstsq(A[i], rhs[i], rcond=None)[0]
            return out

    def fit(self, r, w):
        G, p = self.A.shape[0], self.A.shape[1]
        coef_grid = np.empty((G, p))
        free = ~self.capped
        if np.any(free):
            rhs = self.kernel[free] @ (self.B * r[:, None])
            coef_grid[free] = self._solve_rows(np.nonzero(free)[0], rhs)
        if np.any(self.capped):
            # window widened to the record span: identical to global mode
            sw = np.sqrt(w)
            coef_grid[self.capped] = np.linalg.lstsq(self.B * sw[:, None], r * sw, rcond=None)[0]
        seas = np.sum(self.B * self.coef_at(coef_grid, self.t), axis=1)
        return seas, coef_grid

    def coef_at(self, coef_grid, t_new):
        t_new = np.asarray(t_new, dtype=float)
        return np.column_stack([np.interp(t_new, self.grid, coef_grid[:, j])
                                for j in range(coef_grid.shape[1])])

    def evaluate(self, coef_grid, t_new):
        return np.sum(harmonics(t_new, self.K) * self.coef_at(coef_grid, t_new), axis=1)


# ---------------------------------------------------------------------------
# Backfitting (brief v2 section 4.4)
# ---------------------------------------------------------------------------
def backfit(t, x, w, seasonal, h, state=None, tol=1e-8, max_iter=50, min_pts=10):
    scale = max(1.0, mad(x))
    if state is None:
        trend = local_linear(t, x, w, h, min_pts)[0]
        seas = np.zeros_like(x)
    else:
        trend, seas = state
    converged = False
    it = 0
    coef = hat = widened = fallback = None
    for it in range(1, max_iter + 1):
        seas_new, coef = seasonal.fit(x - trend, w)
        trend_new, hat, widened, fallback = local_linear(t, x - seas_new, w, h, min_pts)
        delta = max(np.max(np.abs(trend_new - trend)), np.max(np.abs(seas_new - seas)))
        trend, seas = trend_new, seas_new
        if delta < tol * scale:
            converged = True
            break
    return {'trend': trend, 'seasonal': seas, 'coef': coef, 'hat': hat,
            'widened': widened, 'fallback': fallback, 'iters': it, 'converged': converged}


# ---------------------------------------------------------------------------
# Fitting schedule (brief v2 section 4.5)
# ---------------------------------------------------------------------------
def fit_pipeline(t, x_lim, cens, make_seasonal, cfg, imputer=None):
    """
    Robustness outer loop, imputation middle loop (to convergence), backfitting
    inner loop.  ``make_seasonal(w)`` returns a seasonal model for weights w.

    ``cfg`` keys: h, robust, n_outer, backfit_tol, backfit_max_iter,
    impute_tol, impute_max_iter.
    """
    n = t.size
    x = np.asarray(x_lim, dtype=float).copy()
    has_cens = bool(np.any(cens))
    if has_cens:
        x[cens] = imputer.start(x_lim)
    w = np.ones(n)
    loops = cfg['n_outer'] if cfg['robust'] else 1
    bf_kw = dict(tol=cfg['backfit_tol'], max_iter=cfg['backfit_max_iter'])
    state = None
    seasonal = None
    res = None
    conv = {'outer_iters': 0, 'weights_converged': not cfg['robust'], 'impute_iters': [],
            'impute_converged': [], 'backfit_iters': [], 'backfit_converged': True,
            'robustness_skipped': False, 'impute_cycles': 0}

    def _bf(seasonal_model, st):
        r = backfit(t, x, w, seasonal_model, cfg['h'], st, **bf_kw)
        conv['backfit_iters'].append(r['iters'])
        conv['backfit_converged'] &= r['converged']
        return r

    for outer in range(loops):
        conv['outer_iters'] += 1
        seasonal = make_seasonal(w)
        res = _bf(seasonal, state)
        state = (res['trend'], res['seasonal'])
        if has_cens:
            n_imp, imp_conv = 0, cfg['impute_max_iter'] == 0
            prev2 = None
            for _ in range(cfg['impute_max_iter']):
                mu = res['trend'] + res['seasonal']
                new = imputer.conditional(x_lim, mu, w)
                n_imp += 1
                prev = x[cens].copy()
                delta = np.max(np.abs(new - prev))
                cycle = (delta >= cfg['impute_tol'] and prev2 is not None
                         and np.max(np.abs(new - prev2)) < cfg['impute_tol'])
                # A period-2 limit cycle (ROS plotting positions depend on the
                # discrete rank order of residuals): settle on its mid-point.
                x[cens] = 0.5 * (new + prev) if cycle else new
                res = _bf(seasonal, state)
                state = (res['trend'], res['seasonal'])
                if delta < cfg['impute_tol'] or cycle:
                    imp_conv = True
                    conv['impute_cycles'] += int(cycle)
                    break
                prev2 = prev
            conv['impute_iters'].append(n_imp)
            conv['impute_converged'].append(imp_conv)
        if not cfg['robust'] or outer == loops - 1:
            break
        r = x - res['trend'] - res['seasonal']
        det = ~cens
        s = 6.0 * np.median(np.abs(r[det]))
        if s <= 1e-10 * max(1.0, mad(x)):
            w_new = np.ones(n)
            conv['robustness_skipped'] = True
        else:
            u = r / s
            w_new = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        w_new[cens] = 1.0
        if np.max(np.abs(w_new - w)) < 1e-3:
            conv['weights_converged'] = True
            break
        w = w_new

    return {'x': x, 'w': w, 'trend': res['trend'], 'seasonal': res['seasonal'],
            'coef': res['coef'], 'hat': res['hat'], 'widened': res['widened'],
            'fallback': res['fallback'], 'seasonal_model': seasonal, 'convergence': conv}


def weighted_rss(fit, cens):
    m = (~cens) & (fit['w'] > 0)
    r = fit['x'] - fit['trend'] - fit['seasonal']
    return float(np.sum(fit['w'][m] * r[m] ** 2)), float(np.sum(fit['w'][m])), int(m.sum())


def bic_for_K(t, fit, cens, K, cfg):
    """BIC of K harmonics, holding the weights and imputations of ``fit`` fixed."""
    seasonal = GlobalSeasonal(t, K)
    r = backfit(t, fit['x'], fit['w'], seasonal, cfg['h'], None,
                tol=cfg['backfit_tol'], max_iter=cfg['backfit_max_iter'])
    tmp = dict(fit, trend=r['trend'], seasonal=r['seasonal'])
    rss, sw, n_d = weighted_rss(tmp, cens)
    floor = 1e-12 * sw * max(1.0, mad(fit['x'])) ** 2
    p = 2 * K + float(np.sum(r['hat']))
    return n_d * np.log(max(rss, floor) / sw) + p * np.log(n_d)


# ---------------------------------------------------------------------------
# Block bootstrap series (brief v2 section 4.9)
# ---------------------------------------------------------------------------
def block_bootstrap_series(t, x_lim, x_work, cens, cen_type, mu_obs, mu_fn, L, rng):
    """
    Non-overlapping block bootstrap with time-shifted whole blocks.

    Returns ``(t_star, x_lim_star, cens_star, cen_type_star)`` sorted by time.
    Censored copies keep their residual-scale limit.
    """
    t0, t1 = float(t[0]), float(t[-1])
    span = t1 - t0
    n_full = int(np.floor(span / L + 1e-9))
    n_blocks = max(1, int(np.ceil(span / L - 1e-9)))
    blk = np.minimum(np.floor((t - t0) / L).astype(int), n_blocks - 1)
    r = x_work - mu_obs
    lim_r = x_lim - mu_obs
    sel_parts, t_parts = [], []
    src = rng.integers(0, n_full, size=n_blocks)
    for b in range(n_blocks):
        sel = np.nonzero(blk == src[b])[0]
        tn = t[sel] + (b - src[b]) * L
        keep = tn <= t1 + 1e-12
        sel_parts.append(sel[keep])
        t_parts.append(tn[keep])
    sel = np.concatenate(sel_parts)
    tn = np.concatenate(t_parts)
    o = np.argsort(tn, kind='stable')
    sel, tn = sel[o], tn[o]
    mu_new = mu_fn(tn)
    c = cens[sel]
    x_star = np.where(c, mu_new + lim_r[sel], mu_new + r[sel])
    return tn, x_star, c, cen_type[sel]


def n_full_blocks(t, L):
    return int(np.floor((t[-1] - t[0]) / L + 1e-9))


# ---------------------------------------------------------------------------
# Seasonal descriptors
# ---------------------------------------------------------------------------
def curve_descriptors(coef, K, transform):
    phase = np.linspace(0.0, 1.0, 1000, endpoint=False)
    s = harmonics(phase, K) @ coef
    amp = float(s.max() - s.min())
    peak = int(np.clip(np.floor(phase[np.argmax(s)] * 365.2425) + 1, 1, 366))
    trough = int(np.clip(np.floor(phase[np.argmin(s)] * 365.2425) + 1, 1, 366))
    ratio = float(np.exp(amp)) if transform == 'log' else np.nan
    return amp, ratio, peak, trough
