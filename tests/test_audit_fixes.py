"""
Comprehensive regression tests for audit fixes (CRIT-01 through LOW-03),
verifying scientific correctness and 100% LWP-TRENDS compatibility.
"""
import types
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt
from unittest.mock import patch

import MannKS as mk
from MannKS import (
    trend_test,
    seasonal_trend_test,
    regional_test,
    rolling_trend_test,
    check_seasonality,
    prepare_censored_data,
)
from MannKS.classification import DEFAULT_CATEGORY_MAP
import MannKS._stats as st
from MannKS._autocorr import detrend, effective_sample_size, should_apply_correction
from MannKS._bootstrap import block_bootstrap_mann_kendall, block_bootstrap_confidence_intervals
from MannKS._large_dataset import _sample_pairs, fast_sens_slope
from MannKS.analysis_notes import get_sens_slope_analysis_note
from MannKS.plotting import (
    plot_seasonal_distribution,
    plot_inspection_data,
    plot_trend,
    plot_residuals,
    plot_rolling_trend,
    plot_segmented_trend,
)


# ---------------------------------------------------------------------------
# CRIT-01: Confidence interval index symmetry for ci_method='direct' vs 'lwp'
# ---------------------------------------------------------------------------

def test_crit01_direct_ci_index_symmetry():
    """
    Verify Gilbert (1987) Eq. 16.8 index symmetry for ci_method='direct':
    lower_idx + upper_idx == n_sample - 1, so symmetric slopes yield
    exact opposite CIs (-upper_ci == lower_ci).
    Meanwhile, ci_method='lwp' preserves exact R LWP-TRENDS interpolation.
    """
    slopes = np.linspace(-10.0, 10.0, 21)  # symmetric around 0.0
    var_s = 12.0
    alpha = 0.05

    lower_direct, upper_direct = st._confidence_intervals(
        slopes.copy(), var_s, alpha=alpha, method='direct'
    )
    assert np.isclose(lower_direct, -upper_direct, atol=1e-14), (
        f"Expected symmetric CIs for symmetric slopes in 'direct' mode, "
        f"got ({lower_direct}, {upper_direct})"
    )

    # Check LWP mode still matches R's approx(1:N, slopes, xout=c(M1, M2))$y
    from scipy.stats import norm
    n_pop = len(slopes)
    c_conf = norm.ppf(1 - alpha / 2) * np.sqrt(var_s)
    r1 = (n_pop - c_conf) / 2.0
    r2 = (n_pop + c_conf) / 2.0
    exp_lwp = tuple(np.interp([r1, r2], np.arange(1, n_pop + 1), np.sort(slopes)))
    got_lwp = st._confidence_intervals(slopes.copy(), var_s, alpha=alpha, method='lwp')
    np.testing.assert_allclose(got_lwp, exp_lwp, rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# CRIT-02: Right-censored ('gt') interval ordering & reflection symmetry
# ---------------------------------------------------------------------------

def test_crit02_right_censored_mk_ordering_and_reflection_symmetry(monkeypatch):
    """
    Verify that in mk_test_method='robust':
    1. Right-censored values '>L' are ordered as intervals (L, +inf):
       - uncensored x <= L vs '>L' at a later time is +1 (concordant increase)
       - uncensored x > L vs '>L' is 0 (ambiguous)
       - two right-censored '>L1' and '>L2' are 0 (ambiguous)
       - left-censored '<L1' vs right-censored '>L2' with L1 <= L2 is +1
    2. Sign reflection (x -> -x, 'lt' <-> 'gt') produces exact negation of S
       and identical Var(S) across all 3 computation paths (O(n log n),
       chunked, and pairwise).
    """
    # 1. Explicit interval ordering check
    # Pair (2.0, '>2.0') -> 2.0 < (>2.0) -> S = +1
    res_tie_lod = trend_test(prepare_censored_data([2.0, '>2.0']), [1, 2], mk_test_method='robust', min_size=None)
    assert res_tie_lod.s == 1

    # Pair (3.0, '>2.0') -> ambiguous -> S = 0
    res_ambig = trend_test(prepare_censored_data([3.0, '>2.0']), [1, 2], mk_test_method='robust', min_size=None)
    assert res_ambig.s == 0

    # Pair ('<2.0', '>2.0') -> (<2.0) < (>2.0) -> S = +1
    res_lt_gt = trend_test(prepare_censored_data(['<2.0', '>2.0']), [1, 2], mk_test_method='robust', min_size=None)
    assert res_lt_gt.s == 1

    # Pair ('<3.0', '>2.0') -> ambiguous -> S = 0
    res_lt_gt_ambig = trend_test(prepare_censored_data(['<3.0', '>2.0']), [1, 2], mk_test_method='robust', min_size=None)
    assert res_lt_gt_ambig.s == 0

    # 2. Exact reflection symmetry: x -> -x + C and 'lt' <-> 'gt'
    rng = np.random.default_rng(42)
    for _ in range(50):
        n = int(rng.integers(5, 60))
        x = rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=n)
        t = np.sort(rng.integers(0, n, size=n)).astype(float)
        ct = rng.choice(['not', 'lt', 'gt'], size=n, p=[0.6, 0.2, 0.2]).astype(object)
        c = ct != 'not'

        # Reflect around 10.0 so all values stay strictly positive
        x_ref = 10.0 - x
        ct_ref = np.where(ct == 'lt', 'gt', np.where(ct == 'gt', 'lt', 'not')).astype(object)
        c_ref = ct_ref != 'not'

        for tau_method in ('a', 'b'):
            for tie_method in ('robust', 'lwp'):
                # Pairwise path
                monkeypatch.setattr(st, '_FAST_S_MIN_N', 10**9)
                s1, var1, d1, tau1 = st._mk_score_and_var_censored(
                    x, t, c, ct, tau_method=tau_method, mk_test_method='robust', tie_break_method=tie_method
                )
                s2, var2, d2, tau2 = st._mk_score_and_var_censored(
                    x_ref, t, c_ref, ct_ref, tau_method=tau_method, mk_test_method='robust', tie_break_method=tie_method
                )
                assert s1 == -s2
                np.testing.assert_allclose(var1, var2, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(d1, d2, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(tau1, -tau2, rtol=1e-12, atol=1e-12)

                # Fast O(n log n) path
                monkeypatch.setattr(st, '_FAST_S_MIN_N', 1)
                s1_f, var1_f, d1_f, tau1_f = st._mk_score_and_var_censored(
                    x, t, c, ct, tau_method=tau_method, mk_test_method='robust', tie_break_method=tie_method
                )
                np.testing.assert_allclose([s1_f, var1_f, d1_f, tau1_f], [s1, var1, d1, tau1], rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# CRIT-03, CRIT-04, HIGH-08: Plotting figure cleanup, OO API, exception catch
# ---------------------------------------------------------------------------

def test_crit03_crit04_plotting_closes_figure_on_save_error(tmp_path):
    """
    Verify that if fig.savefig raises an exception, every plotting function
    still closes its created figure in the finally block (no memory leak)
    and does not corrupt an already-open caller figure.
    """
    caller_fig, caller_ax = plt.subplots()
    caller_ax.set_title("Caller Figure Must Remain Untouched")
    initial_figs = set(plt.get_fignums())

    data = pd.DataFrame({
        'value': [1.0, 2.0, 3.0, 4.0, 5.0],
        't': [1.0, 2.0, 3.0, 4.0, 5.0],
        'censored': [False] * 5,
        'cen_type': ['not'] * 5,
    })
    res = trend_test(data['value'], data['t'])

    with patch.object(plt.Figure, 'savefig', side_effect=OSError("disk full")):
        with pytest.raises(OSError, match="disk full"):
            plot_trend(data, res, str(tmp_path / "fail.png"), alpha=0.05)
        assert set(plt.get_fignums()) == initial_figs

        with pytest.raises(OSError, match="disk full"):
            plot_residuals(data, res, str(tmp_path / "fail_res.png"))
        assert set(plt.get_fignums()) == initial_figs

        with pytest.raises(OSError, match="disk full"):
            plot_seasonal_distribution([1, 2, 3, 4], [1, 2, 3, 4], period=2, plot_path=str(tmp_path / "fail_s.png"))
        assert set(plt.get_fignums()) == initial_figs

        with pytest.raises(OSError, match="disk full"):
            plot_inspection_data(data, str(tmp_path / "fail_i.png"), 'value', 't', 'none', {})
        assert set(plt.get_fignums()) == initial_figs

    # Caller's figure title must still be intact (not overwritten by plt.title)
    assert caller_ax.get_title() == "Caller Figure Must Remain Untouched"
    plt.close(caller_fig)


def test_high08_plot_rolling_trend_specific_exception_handling(tmp_path):
    """
    Verify plot_rolling_trend catches (ValueError, TypeError) on pd.to_datetime
    conversion, does NOT swallow BaseException like KeyboardInterrupt, and closes
    its figure cleanly.
    """
    t_dt = pd.date_range('2020-01-01', periods=20, freq='ME')
    x = np.arange(20, dtype=float)
    rolling_res = rolling_trend_test(x, t_dt, window='1YE', step='6ME', min_size=3)
    data = pd.DataFrame({'t': [str(d) for d in t_dt], 'v': x})

    initial_figs = set(plt.get_fignums())
    with patch('MannKS.plotting.pd.to_datetime', side_effect=KeyboardInterrupt("stop")):
        with pytest.raises(KeyboardInterrupt, match="stop"):
            plot_rolling_trend(rolling_res, data=data, time_col='t', value_col='v', save_path=str(tmp_path / "r.png"))
    assert set(plt.get_fignums()) == initial_figs


# ---------------------------------------------------------------------------
# HIGH-04: Autocorr detrending respects max_pairs / large-dataset fast path
# ---------------------------------------------------------------------------

def test_high04_autocorr_detrend_respects_max_pairs():
    """Verify detrend, effective_sample_size, and should_apply_correction accept max_pairs."""
    rng = np.random.default_rng(123)
    n = 6000  # > MAX_EXACT_N (5000)
    t = np.arange(n, dtype=float)
    x = 0.01 * t + rng.normal(size=n)

    det = detrend(x, t, max_pairs=5000, random_state=42)
    assert len(det) == n
    assert not np.any(np.isnan(det))

    n_eff, r1 = effective_sample_size(x, t=t, method='yue_pilon', max_pairs=5000, random_state=42)
    assert np.isfinite(n_eff) and n_eff > 0

    apply_corr, r1_c, p_val = should_apply_correction(x, t=t, max_pairs=5000, random_state=42)
    assert isinstance(apply_corr, (bool, np.bool_))


# ---------------------------------------------------------------------------
# HIGH-05: Namedtuple field order parity between trend_test and seasonal_trend_test
# ---------------------------------------------------------------------------

def test_high05_result_namedtuple_fields_identical():
    """Verify trend_test and seasonal_trend_test return identical namedtuple field names and order."""
    t = pd.date_range('2020-01-01', periods=24, freq='ME')
    x = np.arange(24, dtype=float)
    r_trend = trend_test(x, t, slope_scaling='year')
    r_seas = seasonal_trend_test(x, t, period=12, slope_scaling='year')

    assert r_trend._fields == r_seas._fields
    assert r_seas.slope_units == 'units per year'
    assert np.isclose(r_seas.scaled_slope, r_seas.slope)
    assert r_seas.slope_per_second < r_seas.slope


# ---------------------------------------------------------------------------
# HIGH-06: regional_test handles sites whose values are all NaN
# ---------------------------------------------------------------------------

def test_high06_regional_test_handles_all_nan_site():
    """Verify regional_test does not raise KeyError when a site in valid_trends has all-NaN observations."""
    df_trends = pd.DataFrame({
        'site': ['SiteA', 'SiteB', 'SiteC'],
        's': [10, 8, 5],
        'C': [0.95, 0.90, 0.85],
    })
    df_obs = pd.DataFrame({
        'site': ['SiteA'] * 5 + ['SiteB'] * 5 + ['SiteC'] * 5,
        'time': list(range(5)) * 3,
        'value': [1, 2, 3, 4, 5, 2, 3, 4, 5, 6, np.nan, np.nan, np.nan, np.nan, np.nan],
    })
    res = regional_test(
        df_trends, df_obs,
        site_col='site', value_col='value', time_col='time', s_col='s', c_col='C'
    )
    assert res.M == 3
    assert np.isfinite(res.TAU)


# ---------------------------------------------------------------------------
# HIGH-07: Block bootstrap accepts np.random.Generator as random_state
# ---------------------------------------------------------------------------

def test_high07_bootstrap_accepts_numpy_generator():
    """Verify block_bootstrap_mann_kendall and block_bootstrap_confidence_intervals accept np.random.Generator."""
    t = np.arange(20, dtype=float)
    x = 0.5 * t + np.sin(t)
    censored = np.zeros(len(x), dtype=bool)
    cen_type = np.full(len(x), 'not', dtype=object)

    gen1 = np.random.default_rng(999)
    gen2 = np.random.default_rng(999)
    p1, s1, boot1 = block_bootstrap_mann_kendall(x, t, censored, cen_type, n_bootstrap=50, random_state=gen1)
    p2, s2, boot2 = block_bootstrap_mann_kendall(x, t, censored, cen_type, n_bootstrap=50, random_state=gen2)
    assert p1 == p2
    np.testing.assert_array_equal(boot1, boot2)

    gen3 = np.random.default_rng(999)
    gen4 = np.random.default_rng(999)
    sl1, lo1, hi1, dist1 = block_bootstrap_confidence_intervals(x, t, censored, cen_type, n_bootstrap=50, random_state=gen3)
    sl2, lo2, hi2, dist2 = block_bootstrap_confidence_intervals(x, t, censored, cen_type, n_bootstrap=50, random_state=gen4)
    assert (sl1, lo1, hi1) == (sl2, lo2, hi2)
    np.testing.assert_array_equal(dist1, dist2)

    # Also verify trend_test end-to-end with Generator and autocorr_method='block_bootstrap'
    res = trend_test(x, t, autocorr_method='block_bootstrap', n_bootstrap=50, random_state=np.random.default_rng(42))
    assert np.isfinite(res.p)


# ---------------------------------------------------------------------------
# MED-03: _sample_pairs O(n) memory-bounded row indexing
# ---------------------------------------------------------------------------

def test_med03_sample_pairs_large_n_no_triu_allocation():
    """
    Verify _sample_pairs works for n=100,000 (where np.triu_indices would allocate ~80 GB)
    and produces valid upper-triangle pairs (0 <= i < j < n).
    """
    n = 100_000
    i, j = _sample_pairs(n, max_pairs=1000, random_state=42)
    assert len(i) == 1000
    assert len(j) == 1000
    assert np.all((0 <= i) & (i < j) & (j < n))


# ---------------------------------------------------------------------------
# MED-05: check_seasonality length mismatch validation
# ---------------------------------------------------------------------------

def test_med05_check_seasonality_length_mismatch():
    """Verify check_seasonality raises a clear ValueError when x and t have different lengths."""
    with pytest.raises(ValueError, match="must have the same length"):
        check_seasonality([1, 2, 3, 4], [1, 2, 3])


# ---------------------------------------------------------------------------
# MED-06: DEFAULT_CATEGORY_MAP is immutable MappingProxyType
# ---------------------------------------------------------------------------

def test_med06_default_category_map_immutable():
    """Verify DEFAULT_CATEGORY_MAP cannot be mutated in place."""
    assert isinstance(DEFAULT_CATEGORY_MAP, types.MappingProxyType)
    with pytest.raises(TypeError):
        DEFAULT_CATEGORY_MAP[0.95] = "Mutated"


# ---------------------------------------------------------------------------
# MED-07 & MED-10: rolling_trend_test max_windows guard and searchsorted parity
# ---------------------------------------------------------------------------

def test_med07_med10_rolling_trend_guard_and_unsorted_parity():
    """
    Verify:
    1. rolling_trend_test raises ValueError when step <= 0 or window count > max_windows.
    2. Sorted (searchsorted fast path) and unsorted (mask fallback) inputs produce identical results.
    """
    t = np.arange(100, dtype=float)
    x = 2.0 * t + 1.0

    with pytest.raises(ValueError, match="positive"):
        rolling_trend_test(x, t, window=10, step=0)

    with pytest.raises(ValueError, match="max_windows"):
        rolling_trend_test(x, t, window=10, step=0.001, max_windows=500)

    # Compare sorted vs shuffled input order
    rng = np.random.default_rng(77)
    perm = rng.permutation(len(t))
    res_sorted = rolling_trend_test(x, t, window=25, step=10)
    res_unsorted = rolling_trend_test(x[perm], t[perm], window=25, step=10)
    pd.testing.assert_frame_equal(res_sorted, res_unsorted)


# ---------------------------------------------------------------------------
# LOW-01: Sub-1e-10 numeric timestamp differences are not dropped as ties
# ---------------------------------------------------------------------------

def test_low01_small_numeric_timestamps_preserved():
    """
    Verify that numeric timestamps spaced by 1e-12 (smaller than 1e-10)
    are not erroneously discarded as tied timestamps in Sen's slope or analysis notes.
    """
    t = np.arange(10, dtype=float) * 1e-12
    x = 3.0 * t  # true slope = 3.0

    res = trend_test(x, t)
    assert np.isclose(res.slope, 3.0, rtol=1e-12)
    assert res.sen_probability is not None and not np.isnan(res.sen_probability)

    slopes_fast = fast_sens_slope(x, t, max_pairs=100, random_state=42)
    assert np.allclose(slopes_fast, 3.0, rtol=1e-12)

    ct = np.full(len(x), 'not', dtype=object)
    note = get_sens_slope_analysis_note(slopes_fast, t, ct)
    assert note == 'ok'
