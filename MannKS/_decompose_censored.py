"""
Bridge between ``MannKS.decompose`` and the optional ``ndimpute`` package.

This is the only module that imports ``ndimpute``.  Every call goes through
``ResidualImputer._call``, which allows only ``method='parametric'`` or
``'ros'``, never substitution and never ``lwp_compat`` (no DL/2 or other
substitution of non-detects).
"""
import warnings
import numpy as np

_MIN_DETECTS = 24
_INSTALL_MSG = ("Censored input to `decompose` needs the optional dependency `ndimpute`. "
                "Install it with: pip install 'mannks[censored]'")


def require_ndimpute():
    try:
        import ndimpute
    except ImportError as e:  # pragma: no cover - exercised via sys.modules patch
        raise ImportError(_INSTALL_MSG) from e
    if ndimpute is None:  # sys.modules entry blocked
        raise ImportError(_INSTALL_MSG)
    return ndimpute


class ResidualImputer:
    """
    Imputation of censored values conditional on the fitted model.

    ``cen_type`` uses MannKS's codes ('lt', 'gt', 'not'), which ndimpute
    accepts directly as string status.
    """

    def __init__(self, cen_type, method='parametric'):
        if method not in ('parametric', 'ros'):
            raise ValueError("censored_method must be 'parametric' or 'ros'.")
        self.nd = require_ndimpute()
        self.status = np.asarray(cen_type).astype(str).astype(object)
        self.cens = self.status != 'not'
        self.method = method
        self.warning_counts = {}
        self.attrs = None
        self.n_calls = 0
        self.used_all_detects = 0

    # -- the single entry point to ndimpute ---------------------------------
    def _call(self, values, **kw):
        if kw.get('method') not in ('parametric', 'ros'):
            raise RuntimeError("Internal error: only parametric or ROS imputation is allowed.")
        if 'lwp_compat' in kw or 'strategy' in kw or 'multiplier' in kw:
            raise RuntimeError("Internal error: substitution options are not allowed.")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            df = self.nd.impute(np.asarray(values, dtype=float), status=self.status, **kw)
        for w in rec:
            msg = str(w.message)
            self.warning_counts[msg] = self.warning_counts.get(msg, 0) + 1
        self.attrs = {k: v for k, v in df.attrs.items()}
        self.n_calls += 1
        return df['imputed_value'].to_numpy(dtype=float)

    def start(self, x_lim):
        """Marginal ROS on the transformed scale; only used to start the loop."""
        imp = self._call(x_lim, method='ros', dist='normal', impute_type='mean',
                         right_impute_type='mean')
        return imp[self.cens]

    def conditional(self, x_lim, mu, w, stochastic=False, seed=None):
        """
        Impute censored residuals around ``mu``; returns new transformed-scale
        values for the censored rows.  Detected rows with zero robustness
        weight are excluded from the residual-distribution fit (passed as NaN).
        """
        resid = np.asarray(x_lim, dtype=float) - mu
        drop = (~self.cens) & (np.asarray(w) == 0)
        if np.any(drop):
            if np.sum(~self.cens & ~drop) >= _MIN_DETECTS:
                resid = resid.copy()
                resid[drop] = np.nan
            else:
                self.used_all_detects += 1
        itype = 'stochastic' if stochastic else 'mean'
        kw = dict(method=self.method, dist='normal', impute_type=itype)
        if stochastic:
            kw['random_state'] = seed
        if self.method == 'ros':
            kw['right_impute_type'] = itype
        imp = self._call(resid, **kw)
        return mu[self.cens] + imp[self.cens]

    def diagnostics(self):
        return {'method': self.method, 'n_calls': self.n_calls, 'attrs': self.attrs,
                'warnings': dict(self.warning_counts),
                'n_fits_using_all_detects': self.used_all_detects}
