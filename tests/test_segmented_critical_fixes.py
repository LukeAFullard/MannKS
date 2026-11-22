"""
Comprehensive regression and statistical-validity tests for critical segmented
trend fixes (F-01, F-03, F-06 a–g, F-07).
"""
import io
import time
import warnings
import numpy as np
import pandas as pd
import pytest

import MannKS as mk
import MannKS._segmented as _seg
from MannKS.segmented_trend_test import (
    SegmentedTrendResult,
    calculate_breakpoint_probability,
    segmented_trend_test,
)
from MannKS.plotting import plot_segmented_trend


# =============================================================================
# F-01: Standardization, symmetric relative RSS floor, scale invariance
# =============================================================================

def test_f01_constant_series_auto_select():
    t = np.arange(50, dtype=float)
    x = np.full(50, 5.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(x, t, n_breakpoints=None, max_breakpoints=3, random_state=42)
    assert res.n_breakpoints == 0
    assert len(res.breakpoints) == 0
    assert res.bic == -np.inf
    assert res.aic == -np.inf
    assert res.score == -np.inf
    assert res.segments['n'].sum() == 50
    assert any("constant" in w.lower() for w in res.warnings)


def test_f01_constant_series_fixed_k_fallback():
    t = np.arange(50, dtype=float)
    x = np.full(50, 5.0)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(x, t, n_breakpoints=1, random_state=42)
    assert res.n_breakpoints == 0
    assert bool(res.selection_summary.loc[0, 'converged']) is False
    assert np.isnan(res.bic)
    assert np.isnan(res.aic)
    assert np.isnan(res.score)
    assert any("constant" in w.lower() for w in res.warnings)


def test_f01_all_censored_same_dl():
    t = np.arange(50, dtype=float)
    df = mk.prepare_censored_data(['<1.0'] * 50)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(df, t, n_breakpoints=None, max_breakpoints=3, random_state=42)
    assert res.n_breakpoints == 0
    assert any("constant" in w.lower() for w in res.warnings)


def test_f01_exact_linear_numeric_and_datetime():
    t_num = np.arange(40, dtype=float)
    x_num = 2.0 * t_num + 5.0
    res_num = segmented_trend_test(x_num, t_num, n_breakpoints=None, max_breakpoints=3, random_state=42)
    assert res_num.n_breakpoints == 0
    assert np.isfinite(res_num.bic)
    assert np.isfinite(res_num.aic)
    assert len(res_num.selection_summary) == 1  # short-circuited after k=0

    t_dt = pd.date_range('2010-01-01', periods=40, freq='365D')
    x_dt = 2.0 * np.arange(40, dtype=float) + 5.0
    res_dt = segmented_trend_test(x_dt, t_dt, n_breakpoints=None, max_breakpoints=3, random_state=42)
    assert res_dt.n_breakpoints == 0
    assert np.isfinite(res_dt.bic)
    assert len(res_dt.selection_summary) == 1


def test_f01_noiseless_v_shape_auto_select():
    t = np.arange(30, dtype=float)
    x = np.where(t < 15, t, 15.0 - (t - 15.0))
    res = segmented_trend_test(x, t, n_breakpoints=None, max_breakpoints=3, random_state=42)
    assert res.n_breakpoints == 1
    assert np.isclose(res.breakpoints[0], 15.0, atol=1.0)


def test_f01_trace_scale_invariance_null_and_hinge():
    rng = np.random.default_rng(12345)
    t = np.arange(60, dtype=float)
    y_null = 0.05 * t + rng.normal(0, 1.0, len(t))

    mults = [1e-12, 1e-8, 1e-5, 1.0, 1e6, 1e12]
    bics = {}
    for m in mults:
        res = segmented_trend_test(
            y_null * m, t, n_breakpoints=None, max_breakpoints=2,
            autocorr_penalty=False, random_state=7,
        )
        assert res.n_breakpoints == 0, f"False breakpoint at scale {m}"
        assert np.isfinite(res.bic)
        bics[m] = res.bic

    n = len(t)
    for m in mults:
        expected_diff = 2.0 * n * np.log(m / 1.0)
        assert abs((bics[m] - bics[1.0]) - expected_diff) < 1e-7

    # Noisy 1-breakpoint hinge: exact scale invariance of breakpoints & CIs
    y_hinge = np.where(t < 30, 0.1 * t, 3.0 + 1.5 * (t - 30)) + rng.normal(0, 0.8, len(t))
    r_unit = segmented_trend_test(y_hinge, t, n_breakpoints=1, random_state=99)
    r_trace = segmented_trend_test(y_hinge * 1e-12, t, n_breakpoints=1, random_state=99)
    assert r_unit.n_breakpoints == 1
    assert r_trace.n_breakpoints == 1
    assert np.allclose(r_unit.breakpoints, r_trace.breakpoints, atol=1e-7, rtol=0)
    assert np.allclose(r_unit.breakpoint_cis, r_trace.breakpoint_cis, atol=1e-7, rtol=0)


def test_f01_identical_timestamps_guard():
    t = np.full(10, 5.0)
    x = np.arange(10, dtype=float)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(x, t, n_breakpoints=None, max_breakpoints=2, random_state=42)
    assert res.n_breakpoints == 0
    assert np.isnan(res.bic)
    assert bool(res.selection_summary['converged'].any()) is False
    assert any("identical" in w.lower() for w in res.warnings)


@pytest.mark.parametrize("val", [0.1, 0.3])
def test_f01_non_dyadic_constant_series_auto_and_fixed(val):
    t = np.arange(50, dtype=float)
    x = np.full(50, val)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_auto = segmented_trend_test(x, t, n_breakpoints=None, max_breakpoints=2, random_state=42)
        r_fix1 = segmented_trend_test(x, t, n_breakpoints=1, random_state=42)

    assert r_auto.n_breakpoints == 0
    assert r_auto.bic == -np.inf
    assert any("constant" in w.lower() for w in r_auto.warnings)

    assert r_fix1.n_breakpoints == 0
    assert np.isnan(r_fix1.bic)
    assert bool(r_fix1.selection_summary['converged'].iloc[0]) is False
    assert any("constant" in w.lower() for w in r_fix1.warnings)


@pytest.mark.parametrize("dl_str", ["<0.2", "<0.3"])
def test_f01_non_dyadic_all_censored_auto_and_fixed(dl_str):
    t = np.arange(50, dtype=float)
    df = mk.prepare_censored_data([dl_str] * 50)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_auto = segmented_trend_test(df, t, n_breakpoints=None, max_breakpoints=2, random_state=42)
        r_fix1 = segmented_trend_test(df, t, n_breakpoints=1, random_state=42)

    assert r_auto.n_breakpoints == 0
    assert r_auto.bic == -np.inf
    assert any("constant" in w.lower() for w in r_auto.warnings)

    assert r_fix1.n_breakpoints == 0
    assert np.isnan(r_fix1.bic)
    assert bool(r_fix1.selection_summary['converged'].iloc[0]) is False
    assert any("constant" in w.lower() for w in r_fix1.warnings)


@pytest.mark.parametrize("n_pts,t_val", [(30, 0.1), (10, 0.3)])
@pytest.mark.parametrize("n_bp", [None, 0])
def test_f01_non_dyadic_identical_timestamps(n_pts, t_val, n_bp):
    t = np.full(n_pts, t_val)
    x = np.linspace(1.0, float(n_pts), n_pts)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(
            x, t, n_breakpoints=n_bp, max_breakpoints=2, random_state=42
        )
    assert res.n_breakpoints == 0
    assert np.isnan(res.bic)
    assert (res.selection_summary['converged'] == False).all()
    assert any("identical" in w.lower() for w in res.warnings)


# =============================================================================
# F-03: Convergence validation, fallback, int(k), score, segment n, RNG isolation
# =============================================================================

def test_f03_fixed_k_non_convergence_small_n():
    t = np.arange(8, dtype=float)
    x = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(x, t, n_breakpoints=5, random_state=42)
    assert res.n_breakpoints == 0
    assert bool(res.selection_summary.loc[0, 'converged']) is False
    assert np.isnan(res.selection_summary.loc[0, 'bic'])
    assert np.isnan(res.bic)
    assert np.isnan(res.aic)
    assert np.isnan(res.score)
    assert len(res.warnings) > 0


def test_f03_numpy_int64_n_breakpoints():
    rng = np.random.default_rng(10)
    t = np.arange(60, dtype=float)
    x = np.where(t < 30, 0.1 * t, 3.0 + 1.5 * (t - 30)) + rng.normal(0, 0.5, len(t))
    res = segmented_trend_test(x, t, n_breakpoints=np.int64(1), random_state=42)
    assert res.n_breakpoints == 1
    assert np.isclose(res.breakpoints[0], 30.0, atol=2.0)


def test_f03_result_score_matches_active_criterion():
    rng = np.random.default_rng(11)
    t = np.arange(60, dtype=float)
    x = np.where(t < 30, 0.1 * t, 3.0 + 1.5 * (t - 30)) + rng.normal(0, 0.5, len(t))

    r_bic = segmented_trend_test(x, t, n_breakpoints=1, criterion='bic', random_state=42)
    r_aic = segmented_trend_test(x, t, n_breakpoints=1, criterion='aic', random_state=42)
    r_mbic = segmented_trend_test(x, t, n_breakpoints=1, criterion='mbic', random_state=42)

    assert np.isclose(r_bic.score, r_bic.bic)
    assert np.isclose(r_aic.score, r_aic.aic)
    assert not np.isclose(r_aic.score, r_aic.bic)
    expected_mbic = r_mbic.selection_summary.loc[0, 'mbic']
    assert np.isclose(r_mbic.score, expected_mbic)
    assert not np.isclose(r_mbic.score, r_mbic.bic)


def test_f03_segment_n_preserves_single_point_segment(monkeypatch):
    t = np.arange(20, dtype=float)
    x = np.where(t < 10, t, 10.0 - (t - 10.0))

    # Force a breakpoint between t=0 and t=1 so segment 0 has exactly 1 observation
    def fake_find_bagged(*args, **kwargs):
        return np.array([0.5]), [[0.5]] * 5

    monkeypatch.setattr(_seg, "find_bagged_breakpoints", fake_find_bagged)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(
            x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=5,
            min_segment_size=None, random_state=42
        )
    assert res.segments.loc[0, 'n'] == 1
    assert res.segments['n'].sum() == len(t)


def test_f03_reproducible_and_isolated_legacy_rng():
    rng = np.random.default_rng(21)
    t = np.arange(60, dtype=float)
    x = np.where(t < 30, 0.1 * t, 3.0 + 1.5 * (t - 30)) + rng.normal(0, 0.8, len(t))

    np.random.seed(111)
    expected_next = np.random.random()

    np.random.seed(111)
    r1 = segmented_trend_test(x, t, n_breakpoints=1, random_state=42)
    actual_next = np.random.random()
    assert actual_next == expected_next  # Global legacy np.random state preserved!

    np.random.seed(999999)
    r2 = segmented_trend_test(x, t, n_breakpoints=1, random_state=42)
    assert np.allclose(r1.breakpoints, r2.breakpoints, atol=0.0, rtol=0.0)
    assert np.allclose(r1.breakpoint_cis, r2.breakpoint_cis, atol=0.0, rtol=0.0)


# =============================================================================
# F-06 (a)–(g): Bagging fixes
# =============================================================================

def test_f06a_inverted_kde_modes_fallback_and_monotonic(monkeypatch):
    rng = np.random.default_rng(31)
    t = np.arange(100, dtype=float)
    x = np.piecewise(
        t,
        [t < 35, (t >= 35) & (t < 65), t >= 65],
        [lambda z: 0.5 * z, lambda z: 17.5 - 0.5 * (z - 35), lambda z: 2.5 + 0.8 * (z - 65)],
    ) + rng.normal(0, 0.5, len(t))

    # Construct ordered pairs whose per-column KDE modes invert (mode0 ~ 45 > mode1 ~ 35)
    fake_samples = []
    for _ in range(55):
        fake_samples.append([45.0 + rng.normal(0, 0.2), rng.uniform(60.0, 95.0)])
    for _ in range(45):
        fake_samples.append([rng.uniform(5.0, 25.0), 35.0 + rng.normal(0, 0.2)])

    monkeypatch.setattr(_seg, "_bootstrap_breakpoints", lambda *a, **kw: fake_samples)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(
            x, t, n_breakpoints=2, use_bagging=True, n_bootstrap=100, random_state=42
        )

    assert len(res.breakpoints) == 2
    assert np.all(np.diff(res.breakpoints) > 0)
    assert res.segments['n'].sum() == len(t)
    assert any("out of order" in w.lower() for w in res.warnings)

    buf = io.BytesIO()
    plot_segmented_trend(res, x, t, save_path=buf)
    assert buf.tell() > 0


def test_f06a_plotting_defense_in_depth_unsorted_breakpoints():
    t = np.arange(30, dtype=float)
    x = np.arange(30, dtype=float)
    segs = pd.DataFrame([
        {'slope': 1.0, 'intercept': 0.0, 'lower_ci': 0.8, 'upper_ci': 1.2, 'n': 10,
         'slope_per_second': 1.0, 'lower_ci_per_second': 0.8, 'upper_ci_per_second': 1.2},
        {'slope': 0.0, 'intercept': 10.0, 'lower_ci': -0.2, 'upper_ci': 0.2, 'n': 10,
         'slope_per_second': 0.0, 'lower_ci_per_second': -0.2, 'upper_ci_per_second': 0.2},
        {'slope': -1.0, 'intercept': 30.0, 'lower_ci': -1.2, 'upper_ci': -0.8, 'n': 10,
         'slope_per_second': -1.0, 'lower_ci_per_second': -1.2, 'upper_ci_per_second': -0.8},
    ])
    fake_res = SegmentedTrendResult(
        n_breakpoints=2,
        breakpoints=np.array([20.0, 10.0]),  # intentionally reversed
        breakpoint_cis=[(18.0, 22.0), (8.0, 12.0)],
        segments=segs,
        is_datetime=False,
        bic=1.0, aic=1.0, score=1.0,
        selection_summary=None,
        bootstrap_samples=None,
        alpha=0.05,
        warnings=[],
        computation_mode='hybrid',
    )
    buf = io.BytesIO()
    plot_segmented_trend(fake_res, x, t, save_path=buf)
    assert buf.tell() > 0


@pytest.mark.parametrize("bad_n_boot", [0, -5, 1, True, 2.5])
def test_f06b_n_bootstrap_validation(bad_n_boot):
    t = np.arange(30, dtype=float)
    x = np.arange(30, dtype=float)
    with pytest.raises(ValueError, match="n_bootstrap"):
        segmented_trend_test(x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=bad_n_boot)


def test_f06b_all_failed_bagging_retains_ols_breakpoints(monkeypatch):
    rng = np.random.default_rng(41)
    t = np.arange(60, dtype=float)
    x = np.where(t < 30, 0.1 * t, 3.0 + 1.5 * (t - 30)) + rng.normal(0, 0.5, len(t))

    r_ols = segmented_trend_test(x, t, n_breakpoints=1, use_bagging=False, random_state=42)

    monkeypatch.setattr(_seg, "_bootstrap_breakpoints", lambda *a, **kw: [[] for _ in range(10)])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_bag = segmented_trend_test(
            x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=10, random_state=42
        )

    assert np.all(np.isfinite(r_bag.breakpoints))
    assert np.allclose(r_bag.breakpoints, r_ols.breakpoints)
    assert r_bag.bootstrap_failure_rate == 1.0
    assert any("falling back to phase-1 ols" in w.lower() for w in r_bag.warnings)


def test_f06c_no_iqr_clipping_exact_percentiles(monkeypatch):
    rng = np.random.default_rng(51)
    t = np.arange(100, dtype=float)
    x = np.where(t < 50, 0.1 * t, 5.0 + 1.5 * (t - 50)) + rng.normal(0, 0.5, len(t))

    # Heavy-tailed valid samples strictly inside (0, 99): 80 tight + 20 tail points
    # 1.5*IQR would aggressively clip the 20 tail points at 15 and 85.
    raw = [50.0 + i * 0.01 for i in range(-40, 40)] + [15.0] * 10 + [85.0] * 10
    fake_samples = [[v] for v in raw]
    monkeypatch.setattr(_seg, "_bootstrap_breakpoints", lambda *a, **kw: fake_samples)

    widths = []
    for alpha in (0.10, 0.05, 0.01):
        res = segmented_trend_test(
            x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=100, alpha=alpha, random_state=42
        )
        expected_low = np.percentile(raw, 100 * alpha / 2)
        expected_high = np.percentile(raw, 100 * (1 - alpha / 2))
        assert np.isclose(res.breakpoint_cis[0][0], expected_low, atol=1e-9)
        assert np.isclose(res.breakpoint_cis[0][1], expected_high, atol=1e-9)
        widths.append(res.breakpoint_cis[0][1] - res.breakpoint_cis[0][0])

    assert widths[0] <= widths[1] <= widths[2]
    assert widths[2] > 60.0  # Unclipped 99% CI spans the heavy tails


def test_f06d_bootstrap_methods_and_validation():
    t = np.arange(60, dtype=float)
    x = np.arange(60, dtype=float)
    with pytest.raises(ValueError, match="bootstrap_method"):
        segmented_trend_test(x, t, use_bagging=True, n_bootstrap=5, bootstrap_method='invalid')
    with pytest.raises(ValueError, match="block_size"):
        segmented_trend_test(x, t, use_bagging=True, n_bootstrap=5, block_size=0)
    with pytest.raises(ValueError, match="inner_n_boot"):
        segmented_trend_test(x, t, use_bagging=True, n_bootstrap=5, inner_n_boot=-1)

    # AR(1) phi=0.6 hinged series
    rng = np.random.default_rng(61)
    n = 80
    t_ar = np.arange(n, dtype=float)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0, 1.0)
    x_ar = np.where(t_ar < 40, 0.1 * t_ar, 4.0 + 1.2 * (t_ar - 40)) + eps

    for method in ('block', 'pairs', 'residual'):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            res = segmented_trend_test(
                x_ar, t_ar, n_breakpoints=1, use_bagging=True, n_bootstrap=10,
                bootstrap_method=method, block_size='auto', inner_n_boot=5, random_state=42,
            )
        assert res.n_breakpoints == 1
        assert np.isfinite(res.breakpoints[0])
        assert res.breakpoint_cis[0][0] <= res.breakpoint_cis[0][1]


def test_f06e_bootstrap_failure_rate_and_probability():
    # Exact 13 namedtuple fields preserved
    assert len(SegmentedTrendResult._fields) == 13

    t = np.arange(30, dtype=float)
    x = 2.0 * t + 1.0
    r_nobag = segmented_trend_test(x, t, n_breakpoints=0, use_bagging=False)
    assert r_nobag.bootstrap_failure_rate is None

    res_partial = SegmentedTrendResult(
        n_breakpoints=1, breakpoints=np.array([25.0]), breakpoint_cis=[(22.0, 28.0)],
        segments=pd.DataFrame(), is_datetime=False, bic=0.0, aic=0.0, score=0.0,
        selection_summary=None,
        bootstrap_samples=[[25.0], [25.0], [], []],
        alpha=0.05, warnings=[], computation_mode='hybrid',
    )
    assert res_partial.bootstrap_failure_rate == 0.5
    with pytest.warns(UserWarning, match="Excluded 2/4 failed bootstrap"):
        prob = calculate_breakpoint_probability(res_partial, 20.0, 30.0)
    assert np.isclose(prob, 1.0)

    res_all_fail = res_partial._replace(bootstrap_samples=[[], []])
    assert res_all_fail.bootstrap_failure_rate == 1.0
    with pytest.warns(UserWarning, match="All bootstrap replicates failed"):
        prob_nan = calculate_breakpoint_probability(res_all_fail, 20.0, 30.0)
    assert np.isnan(prob_nan)


def test_f06g_no_find_peaks_and_fast_inner_n_boot():
    assert 'find_peaks' not in dir(_seg)

    rng = np.random.default_rng(71)
    t = np.arange(80, dtype=float)
    x = np.where(t < 40, 0.1 * t, 4.0 + 1.5 * (t - 40)) + rng.normal(0, 1.0, len(t))

    t0 = time.perf_counter()
    res = segmented_trend_test(
        x, t, n_breakpoints=1, use_bagging=True, n_bootstrap=15, random_state=42
    )
    elapsed = time.perf_counter() - t0
    assert res.n_breakpoints == 1
    assert np.isclose(res.breakpoints[0], 40.0, atol=3.0)
    assert elapsed < 15.0  # Was ~20-35s with inner n_boot=100


# =============================================================================
# F-07: mBIC, ESS-BIC autocorr_penalty, opt-in Davies test, diagnostic warnings
# =============================================================================

def test_f07_parameter_validation():
    t = np.arange(30, dtype=float)
    x = np.arange(30, dtype=float)
    with pytest.raises(ValueError, match="criterion"):
        segmented_trend_test(x, t, criterion='invalid')
    with pytest.raises(ValueError, match="autocorr_penalty"):
        segmented_trend_test(x, t, autocorr_penalty='bad')


def test_f07_ar1_fpr_reduction_and_warning():
    rng = np.random.default_rng(2026)
    n = 100
    t = np.arange(n, dtype=float)
    n_reps = 12

    fp_raw = 0
    fp_ess_bic = 0
    fp_ess_mbic = 0
    acf_warn_count = 0

    for rep in range(n_reps):
        eps = np.zeros(n)
        for i in range(1, n):
            eps[i] = 0.6 * eps[i - 1] + rng.normal(0, 1.0)
        x = 0.02 * t + eps

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r_raw = segmented_trend_test(
                x, t, n_breakpoints=None, max_breakpoints=1,
                criterion='bic', autocorr_penalty=False, random_state=rep,
            )
            r_ess = segmented_trend_test(
                x, t, n_breakpoints=None, max_breakpoints=1,
                criterion='bic', autocorr_penalty='auto', random_state=rep,
            )
            r_mbic = segmented_trend_test(
                x, t, n_breakpoints=None, max_breakpoints=1,
                criterion='mbic', autocorr_penalty='auto', random_state=rep,
            )

        if r_raw.n_breakpoints > 0:
            fp_raw += 1
        if r_ess.n_breakpoints > 0:
            fp_ess_bic += 1
        if r_mbic.n_breakpoints > 0:
            fp_ess_mbic += 1
        if any("autocorrelation" in w.lower() for w in r_ess.warnings):
            acf_warn_count += 1

    assert fp_ess_bic <= fp_raw
    assert fp_ess_bic / n_reps <= 0.25
    assert fp_ess_mbic / n_reps <= 0.17
    assert acf_warn_count >= int(0.8 * n_reps)


def test_f07_true_hinge_power_preserved_and_clean_no_acf_warning():
    rng = np.random.default_rng(81)
    n = 100
    t = np.arange(n, dtype=float)

    # Clean Gaussian k=1 hinge: detects breakpoint(s) and emits NO autocorrelation warning
    x_clean = np.where(t < 50, 0.05 * t, 2.5 + 0.6 * (t - 50)) + rng.normal(0, 1.0, n)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_clean = segmented_trend_test(
            x_clean, t, n_breakpoints=None, max_breakpoints=2, criterion='bic', random_state=42
        )
        r_clean_mbic = segmented_trend_test(
            x_clean, t, n_breakpoints=None, max_breakpoints=2, criterion='mbic', random_state=42
        )
    assert r_clean.n_breakpoints >= 1
    assert r_clean_mbic.n_breakpoints == 1
    assert not any("autocorrelation" in w.lower() for w in r_clean.warnings)
    assert not any("autocorrelation" in w.lower() for w in r_clean_mbic.warnings)

    # AR(1) phi=0.6 k=1 hinge: still detects breakpoint AND emits autocorrelation warning
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.6 * eps[i - 1] + rng.normal(0, 1.0)
    x_ar_hinge = np.where(t < 50, 0.05 * t, 2.5 + 0.6 * (t - 50)) + eps
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_ar = segmented_trend_test(
            x_ar_hinge, t, n_breakpoints=None, max_breakpoints=2, random_state=42
        )
    assert r_ar.n_breakpoints >= 1
    assert any("autocorrelation" in w.lower() for w in r_ar.warnings)


def test_f07_censoring_floor_and_shifting_dl_warnings():
    t = np.arange(40, dtype=float)

    # 1. Contiguous left-censored floor (>15% censored + contiguous run >= 6)
    raw_floor = ['<1.0'] * 15 + [str(1.5 + 0.2 * i) for i in range(25)]
    df_floor = mk.prepare_censored_data(raw_floor)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_floor = segmented_trend_test(df_floor, t, n_breakpoints=0, random_state=42)
    assert any("high proportion of censored" in w.lower() for w in res_floor.warnings)
    assert any("left-censored floor" in w.lower() for w in res_floor.warnings)

    # 2. Shifting left-censoring detection limits (<5.0 -> <0.5)
    raw_shift = ['<5.0'] * 6 + [str(6.0 + i * 0.1) for i in range(14)] + ['<0.5'] * 6 + [str(2.0 + i * 0.1) for i in range(14)]
    df_shift = mk.prepare_censored_data(raw_shift)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_no_hi = segmented_trend_test(df_shift, t, n_breakpoints=0, hicensor=False, random_state=42)
        res_hi = segmented_trend_test(df_shift, t, n_breakpoints=0, hicensor=True, random_state=42)
    assert any("multiple distinct left-censoring detection limits" in w.lower() for w in res_no_hi.warnings)
    assert not any("multiple distinct left-censoring detection limits" in w.lower() for w in res_hi.warnings)


def test_f07_short_record_seasonality_warning():
    rng = np.random.default_rng(91)

    # 24-month monthly record (< 3 years, >= 180 days, median step ~30 days) -> warns
    t_24m = pd.date_range('2020-01-01', periods=24, freq='MS')
    x_24m = np.sin(2 * np.pi * np.arange(24) / 12.0) + rng.normal(0, 0.2, 24)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_24m = segmented_trend_test(x_24m, t_24m, n_breakpoints=None, max_breakpoints=1, random_state=42)
    assert any("short datetime record" in w.lower() for w in r_24m.warnings)

    # 20-day daily record (< 180 days) -> no seasonality warning
    t_20d = pd.date_range('2020-01-01', periods=20, freq='D')
    x_20d = np.arange(20, dtype=float) + rng.normal(0, 0.5, 20)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_20d = segmented_trend_test(x_20d, t_20d, n_breakpoints=None, max_breakpoints=1, random_state=42)
    assert not any("short datetime record" in w.lower() for w in r_20d.warnings)

    # 10-year monthly record (>= 3 years) -> no seasonality warning
    t_10y = pd.date_range('2010-01-01', periods=120, freq='MS')
    x_10y = 0.05 * np.arange(120, dtype=float) + rng.normal(0, 0.5, 120)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_10y = segmented_trend_test(x_10y, t_10y, n_breakpoints=None, max_breakpoints=1, random_state=42)
    assert not any("short datetime record" in w.lower() for w in r_10y.warnings)


def test_f07_opt_in_davies_test_gate():
    # _safe_davies_test guards n < 10
    assert np.isnan(_seg._safe_davies_test(np.arange(4, dtype=float), np.arange(4, dtype=float)))

    # Does not crash on small n=4 or constant series
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        r_small = segmented_trend_test(
            np.array([1.0, 2.0, 3.0, 4.0]), np.arange(4, dtype=float),
            n_breakpoints=None, max_breakpoints=1, use_davies_test=True, random_state=42,
        )
        r_const = segmented_trend_test(
            np.full(20, 3.0), np.arange(20, dtype=float),
            n_breakpoints=None, max_breakpoints=1, use_davies_test=True, random_state=42,
        )
    assert r_small.n_breakpoints == 0
    assert r_const.n_breakpoints == 0

    # Passes true k=1 hinge
    rng = np.random.default_rng(101)
    t = np.arange(80, dtype=float)
    x = np.where(t < 40, 0.05 * t, 2.0 + 1.2 * (t - 40)) + rng.normal(0, 0.8, len(t))
    r_hinge = segmented_trend_test(
        x, t, n_breakpoints=None, max_breakpoints=2, use_davies_test=True, random_state=42
    )
    assert r_hinge.n_breakpoints == 1


def test_f07_negative_acf_warning_wording_with_auto_penalty():
    rng = np.random.default_rng(77)
    n = 60
    t = np.arange(n, dtype=float)
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = -0.75 * eps[i - 1] + rng.normal(0, 1.0)
    x = 0.05 * t + eps

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(
            x, t, n_breakpoints=None, max_breakpoints=2,
            autocorr_penalty='auto', random_state=42,
        )
    acf_warns = [w for w in res.warnings if "autocorrelation detected" in w.lower()]
    assert len(acf_warns) >= 1
    assert "positive autocorrelation" in acf_warns[0].lower()
    assert "consider autocorr_penalty='auto'" not in acf_warns[0].lower()

