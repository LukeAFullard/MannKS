"""
Comprehensive regression tests for High-Severity Segmented Trend Audit Fixes:
F-02, F-04, F-05, F-08, F-10, F-13, and F-15.
"""

import importlib.util
import io
import json
import os
import subprocess
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
import pandas as pd
import pytest

import MannKS as mk
from MannKS._datetime import _datetime_to_seconds, _is_datetime_like
from MannKS._segmented import HybridSegmentedTrend, _isolated_legacy_numpy_seed
from MannKS.plotting import _create_segments, plot_segmented_trend
from MannKS.segmented_trend_test import (
    SegmentedTrendResult,
    _prepare_data,
    calculate_breakpoint_probability,
    find_best_segmentation,
    segmented_trend_test,
)


# =============================================================================
# F-02: Reproducibility & Global np.random Isolation
# =============================================================================


def test_f02_fresh_subprocess_reproducibility():
    """
    Fresh Python subprocesses (no pytest conftest.py seeding) with random_state=42
    must produce bit-for-bit identical results for fixed k=2, auto-search, and bagging.
    """
    script = """
import json
import numpy as np
from MannKS import segmented_trend_test

rng = np.random.default_rng(12345)
t = np.arange(90, dtype=float)
x = np.where(t < 30, 0.3 * t, np.where(t < 60, 9.0 - 0.25 * (t - 30), 1.5 + 0.35 * (t - 60)))
eps = np.zeros(90)
for i in range(1, 90):
    eps[i] = 0.4 * eps[i - 1] + rng.normal(0, 0.4)
x = x + eps

r_fixed = segmented_trend_test(x, t, n_breakpoints=2, use_bagging=False, random_state=42)
r_search = segmented_trend_test(x, t, n_breakpoints=None, max_breakpoints=2, use_bagging=False, random_state=42)
r_bag = segmented_trend_test(x, t, n_breakpoints=2, use_bagging=True, n_bootstrap=8, random_state=42)

def pack(r):
    return {
        "n": int(r.n_breakpoints),
        "bps": [float(v) for v in r.breakpoints],
        "cis": [[float(a), float(b)] for a, b in r.breakpoint_cis],
        "bic": float(r.bic),
        "slopes": [float(v) for v in r.segments["slope"].tolist()],
    }

print(json.dumps({"fixed": pack(r_fixed), "search": pack(r_search), "bag": pack(r_bag)}))
"""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outputs = []
    for _ in range(2):
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(proc.stdout.strip()))

    assert outputs[0] == outputs[1]
    assert outputs[0]["fixed"]["n"] == 2
    assert outputs[0]["bag"]["n"] == 2


def test_f02_global_numpy_rng_state_restored_all_modes():
    """
    Caller's global np.random stream must be bit-for-bit unchanged after
    segmented_trend_test for int, Generator, RandomState, None, and on exceptions.
    """
    t = np.arange(40, dtype=float)
    x = np.where(t < 20, 0.5 * t, 10.0 - 0.4 * (t - 20)) + np.sin(t) * 0.1

    np.random.seed(987654)
    expected_next = np.random.random(5)

    for rs in (42, np.random.default_rng(42), np.random.RandomState(42), None):
        np.random.seed(987654)
        segmented_trend_test(x, t, n_breakpoints=1, random_state=rs)
        actual_next = np.random.random(5)
        np.testing.assert_array_equal(actual_next, expected_next)

    # Exception path inside _isolated_legacy_numpy_seed
    np.random.seed(987654)
    with pytest.raises(RuntimeError, match="boom"):
        with _isolated_legacy_numpy_seed(None):
            _ = np.random.random(10)
            raise RuntimeError("boom")
    np.testing.assert_array_equal(np.random.random(5), expected_next)


def test_f02_phase2_max_pairs_reproducible_with_generator_and_int():
    """
    Phase-2 subsampling (max_pairs) must be reproducible across calls with
    different global np.random states for both int and Generator seeds.
    """
    rng = np.random.default_rng(7)
    t = np.arange(60, dtype=float)
    x = np.where(t < 30, 0.4 * t, 12.0 - 0.3 * (t - 30)) + rng.normal(0, 0.3, 60)

    np.random.seed(111)
    r_int_1 = segmented_trend_test(x, t, n_breakpoints=1, max_pairs=50, random_state=42)
    np.random.seed(222)
    r_int_2 = segmented_trend_test(x, t, n_breakpoints=1, max_pairs=50, random_state=42)
    np.testing.assert_array_equal(r_int_1.segments["slope"].to_numpy(), r_int_2.segments["slope"].to_numpy())

    np.random.seed(333)
    r_gen_1 = segmented_trend_test(x, t, n_breakpoints=1, max_pairs=50, random_state=np.random.default_rng(42))
    np.random.seed(444)
    r_gen_2 = segmented_trend_test(x, t, n_breakpoints=1, max_pairs=50, random_state=np.random.default_rng(42))
    np.testing.assert_array_equal(r_gen_1.segments["slope"].to_numpy(), r_gen_2.segments["slope"].to_numpy())


# =============================================================================
# F-04: Datetime / Timezone / Nullable Pandas Dtypes
# =============================================================================


def test_f04_tz_aware_datetimeindex_and_series():
    """
    Timezone-aware DatetimeIndex and Series must not crash, must preserve the
    input timezone on breakpoints & CIs (no silent -13h UTC shift), and must
    support predict(), calculate_breakpoint_probability(), and plot_segmented_trend().
    """
    t_idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="Pacific/Auckland")
    t_ser = pd.Series(t_idx)
    idx = np.arange(30, dtype=float)
    x = np.where(idx < 15, 1.0 * idx, 15.0 - 1.0 * (idx - 15))

    for t_in in (t_idx, t_ser):
        res0 = segmented_trend_test(x, t_in, n_breakpoints=0, random_state=42)
        assert res0.is_datetime is True
        assert res0.n_breakpoints == 0

        res1 = segmented_trend_test(
            x, t_in, n_breakpoints=1, use_bagging=True, n_bootstrap=10, random_state=42
        )
        assert res1.is_datetime is True
        assert res1.n_breakpoints == 1
        assert str(res1.breakpoints.tz) == "Pacific/Auckland"

        expected_bp = pd.Timestamp("2020-01-16 00:00", tz="Pacific/Auckland")
        assert abs((res1.breakpoints[0] - expected_bp).total_seconds()) < 2 * 86400
        assert res1.breakpoint_cis[0][0].tz == res1.breakpoints.tz
        assert res1.breakpoint_cis[0][1].tz == res1.breakpoints.tz

        preds = res1.predict(t_in)
        assert preds.shape == (30,)
        assert np.all(np.isfinite(preds))
        assert np.max(np.abs(preds - x)) < 2.0

        prob = calculate_breakpoint_probability(
            res1,
            pd.Timestamp("2020-01-13", tz="Pacific/Auckland"),
            pd.Timestamp("2020-01-19", tz="Pacific/Auckland"),
        )
        assert 0.5 <= prob <= 1.0

        fig, ax = plot_segmented_trend(res1, x, t_in, save_path=None)
        plt.close(fig)


def test_f04_dst_transition_preserves_monotonic_physical_time():
    """DST fall-back hourly series must keep monotonic physical-time slope and restored tz."""
    t_dst = pd.date_range("2020-04-04 00:00", periods=48, freq="h", tz="Pacific/Auckland")
    t_hours = np.arange(48, dtype=float)
    x = 2.0 * t_hours + 5.0

    res = segmented_trend_test(x, t_dst, n_breakpoints=0, slope_scaling="hour", random_state=42)
    assert res.is_datetime is True
    assert np.isclose(res.segments["slope"].iloc[0], 2.0, atol=1e-6)


def test_f04_python_list_of_tz_aware_timestamps_and_mixed_tz():
    """Python lists of tz-aware Timestamps/datetimes preserve tz; mixed tz returns UTC."""
    t_idx = pd.date_range("2020-01-01", periods=30, freq="D", tz="Pacific/Auckland")
    idx = np.arange(30, dtype=float)
    x = np.where(idx < 15, 1.0 * idx, 15.0 - 1.0 * (idx - 15))

    for t_list in (list(t_idx), t_idx.to_pydatetime().tolist()):
        res = segmented_trend_test(x, t_list, n_breakpoints=1, random_state=42)
        assert res.is_datetime is True
        assert str(res.breakpoints.tz) == "Pacific/Auckland"

    # Mixed timezones -> UTC (not tz-naive)
    t_mixed = [
        t_idx[i].tz_convert("UTC") if i % 2 == 0 else t_idx[i]
        for i in range(len(t_idx))
    ]
    res_mixed = segmented_trend_test(x, t_mixed, n_breakpoints=1, random_state=42)
    assert res_mixed.is_datetime is True
    assert res_mixed.breakpoints.tz is not None
    assert str(res_mixed.breakpoints.tz) == "UTC"


def test_f04_nullable_int64_and_float64_series():
    """Pandas nullable Int64 and Float64 Series (with and without pd.NA) must work."""
    t_int = pd.Series(list(range(30)), dtype="Int64")
    x_flt = pd.Series(
        [float(i) if i < 15 else float(30 - i) for i in range(30)], dtype="Float64"
    )
    res = segmented_trend_test(x_flt, t_int, n_breakpoints=1, random_state=42)
    assert res.is_datetime is False
    assert res.n_breakpoints == 1
    preds = res.predict(t_int)
    assert np.all(np.isfinite(preds))

    # With pd.NA present
    t_na = t_int.copy()
    t_na.iloc[5] = pd.NA
    x_na = x_flt.copy()
    x_na.iloc[2] = pd.NA
    res_na = segmented_trend_test(x_na, t_na, n_breakpoints=1, random_state=42)
    assert res_na.is_datetime is False
    assert res_na.n_breakpoints == 1
    preds_na = res_na.predict(t_na)
    assert np.isnan(preds_na[5])
    assert np.isfinite(preds_na[0])


def test_f04_object_array_with_leading_none_and_prepare_data_contract():
    """Object arrays with leading None are detected as datetime-like; _prepare_data returns 2-tuple."""
    t_dt = pd.date_range("2020-01-01", periods=20, freq="D")
    t_obj = np.array([None] + list(t_dt), dtype=object)
    assert _is_datetime_like(t_obj) is True

    x = np.arange(21, dtype=float)
    df, is_dt = _prepare_data(x, t_obj, hicensor=False)
    assert is_dt is True
    assert len(df) == 20


# =============================================================================
# F-05: Sub-Second Precision & Overflow-Safe _datetime_to_seconds
# =============================================================================


def test_f05_subsecond_100ms_hinge_and_trend_test():
    """100ms sensor series must preserve sub-second timestamps without collapsing to integer seconds."""
    t = pd.date_range("2020-01-01", periods=50, freq="100ms")
    i = np.arange(50, dtype=float)
    x = np.where(i < 25, 0.5 * i, 12.5 - 0.4 * (i - 25))

    res = segmented_trend_test(x, t, n_breakpoints=1, random_state=42)
    assert res.is_datetime is True
    assert res.n_breakpoints == 1
    expected_bp = pd.Timestamp("2020-01-01 00:00:02.500")
    assert abs((res.breakpoints[0] - expected_bp).total_seconds()) <= 0.2
    assert np.isclose(res.segments["slope"].iloc[0], 5.0, atol=0.5)
    assert np.isclose(res.segments["slope"].iloc[1], -4.0, atol=0.5)

    # Also verify trend_test on 100ms series
    tr = mk.trend_test(np.arange(50, dtype=float), t)
    assert np.isclose(tr.slope, 10.0, atol=1e-4)


def test_f05_subsecond_1ms_100us_and_overflow_safe_far_dates():
    """_datetime_to_seconds preserves 1ms/100us precision and never overflows dates outside 1677-2262."""
    t_ms = pd.date_range("2020-01-01", periods=10, freq="1ms")
    s_ms = _datetime_to_seconds(t_ms)
    np.testing.assert_allclose(np.diff(s_ms), 1e-3, atol=1e-6)

    t_us = pd.date_range("2020-01-01", periods=10, freq="100us")
    s_us = _datetime_to_seconds(t_us)
    np.testing.assert_allclose(np.diff(s_us), 1e-4, atol=1e-6)

    for dtype in ("datetime64[us]", "datetime64[s]", "datetime64[D]"):
        far = np.array(["1600-01-01", "2500-01-01"], dtype=dtype)
        sec = _datetime_to_seconds(far)
        assert np.all(np.isfinite(sec))
        roundtrip = pd.to_datetime(sec, unit="s")
        assert roundtrip[0].year == 1600
        assert roundtrip[1].year == 2500


# =============================================================================
# F-08: Post-Selection Conditional CI Diagnostics & Phase-2 autocorr_method
# =============================================================================


def test_f08_conditional_ci_diagnostics_and_namedtuple_contract():
    """
    SegmentedTrendResult._fields must remain exact 13 fields; ci_is_conditional,
    analysis_notes, and conditional-CI warning must reflect n_breakpoints > 0 vs == 0.
    """
    expected_fields = (
        "n_breakpoints",
        "breakpoints",
        "breakpoint_cis",
        "segments",
        "is_datetime",
        "bic",
        "aic",
        "score",
        "selection_summary",
        "bootstrap_samples",
        "alpha",
        "warnings",
        "computation_mode",
    )
    assert SegmentedTrendResult._fields == expected_fields
    assert len(SegmentedTrendResult._fields) == 13

    t = np.arange(40, dtype=float)
    x_lin = 0.5 * t
    x_hinge = np.where(t < 20, 0.5 * t, 10.0 - 0.5 * (t - 20))

    res0 = segmented_trend_test(x_lin, t, n_breakpoints=0, random_state=42)
    assert res0.ci_is_conditional is False
    assert res0.analysis_notes == []
    assert not any("conditional" in w.lower() for w in res0.warnings)

    for use_bag in (False, True):
        res1 = segmented_trend_test(
            x_hinge, t, n_breakpoints=1, use_bagging=use_bag, n_bootstrap=6, random_state=42
        )
        assert res1.ci_is_conditional is True
        assert len(res1.analysis_notes) >= 1
        assert "conditional" in res1.analysis_notes[0].lower()
        assert "post-selection" in res1.analysis_notes[0].lower()
        assert any("conditional" in w.lower() for w in res1.warnings)


def test_f08_phase2_autocorr_method_widens_ar1_cis_and_validates():
    """
    autocorr_method validates inputs, widens AR(1) segment slope CIs under 'yue_wang'
    while preserving point slope, and runs cleanly for 'auto' and 'block_bootstrap'.
    """
    t = np.arange(80, dtype=float)
    rng = np.random.default_rng(101)
    eps = np.zeros(80)
    for i in range(1, 80):
        eps[i] = 0.7 * eps[i - 1] + rng.normal(0, 0.5)
    x_ar = 0.1 * t + eps

    with pytest.raises(ValueError, match="autocorr_method"):
        segmented_trend_test(x_ar, t, n_breakpoints=0, autocorr_method="bogus")

    r_none = segmented_trend_test(x_ar, t, n_breakpoints=0, autocorr_method="none", random_state=42)
    r_yw = segmented_trend_test(x_ar, t, n_breakpoints=0, autocorr_method="yue_wang", random_state=42)
    assert np.isclose(r_none.segments["slope"].iloc[0], r_yw.segments["slope"].iloc[0])
    w_none = r_none.segments["upper_ci"].iloc[0] - r_none.segments["lower_ci"].iloc[0]
    w_yw = r_yw.segments["upper_ci"].iloc[0] - r_yw.segments["lower_ci"].iloc[0]
    assert w_yw >= w_none

    # Also test k=1 with 'yue_wang', 'auto', and 'block_bootstrap' (reproducible)
    x_hinge_ar = np.where(t < 40, 0.3 * t, 12.0 - 0.3 * (t - 40)) + eps
    r_bb_1 = segmented_trend_test(
        x_hinge_ar, t, n_breakpoints=1, autocorr_method="block_bootstrap", n_bootstrap=20, random_state=42
    )
    r_bb_2 = segmented_trend_test(
        x_hinge_ar, t, n_breakpoints=1, autocorr_method="block_bootstrap", n_bootstrap=20, random_state=42
    )
    np.testing.assert_allclose(r_bb_1.segments["lower_ci"].to_numpy(), r_bb_2.segments["lower_ci"].to_numpy())
    np.testing.assert_allclose(r_bb_1.segments["upper_ci"].to_numpy(), r_bb_2.segments["upper_ci"].to_numpy())

    r_auto = segmented_trend_test(
        x_hinge_ar, t, n_breakpoints=1, autocorr_method="auto", n_bootstrap=15, random_state=42
    )
    assert len(r_auto.segments) == 2

    # Tiny segment (n_seg < 5) with block_bootstrap falls back gracefully with warning
    t_small = np.array([0.0, 1.0, 2.0, 3.0])
    x_small = np.array([1.0, 2.0, 3.0, 4.0])
    r_small = segmented_trend_test(
        x_small, t_small, n_breakpoints=0, autocorr_method="block_bootstrap", random_state=42
    )
    assert any("falling back" in w.lower() for w in r_small.warnings)
    assert np.isfinite(r_small.segments["slope"].iloc[0])


# =============================================================================
# F-10: Input Validation & Explicit Kwargs
# =============================================================================


def test_f10_infinite_times_filtered_with_warning():
    """Inf and -Inf timestamps must be filtered with a UserWarning in segmented_trend_test and trend_test."""
    t = np.arange(30, dtype=float)
    x = np.where(t < 15, 1.0 * t, 15.0 - 1.0 * (t - 15))
    t_inf = np.concatenate(([-np.inf], t, [np.inf]))
    x_inf = np.concatenate(([99.0], x, [-99.0]))

    res = segmented_trend_test(x_inf, t_inf, n_breakpoints=1, random_state=42)
    assert any("Removed 2 observation" in w and "infinite" in w.lower() for w in res.warnings)
    assert res.n_breakpoints == 1
    assert abs(res.breakpoints[0] - 15.0) < 2.0

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")
        tr = mk.trend_test(np.concatenate(([99.0], np.arange(20, dtype=float), [-99.0])),
                           np.concatenate(([-np.inf], np.arange(20, dtype=float), [np.inf])))
    assert any("Removed 2 observation" in str(w.message) for w in w_log)
    assert np.isclose(tr.slope, 1.0, atol=1e-6)


@pytest.mark.parametrize("bad_alpha", [-0.05, 0.0, 1.0, 1.5, np.nan, np.inf, True, False, "0.05"])
def test_f10_alpha_validation(bad_alpha):
    """alpha outside (0, 1) or non-finite/bool/str must raise ValueError immediately."""
    t = np.arange(20, dtype=float)
    x = 0.5 * t
    with pytest.raises(ValueError, match="alpha"):
        segmented_trend_test(x, t, n_breakpoints=0, alpha=bad_alpha)

    # Even when n < 2
    with pytest.raises(ValueError, match="alpha"):
        segmented_trend_test(np.array([1.0]), np.array([0.0]), alpha=bad_alpha)

    # Direct HybridSegmentedTrend.fit
    model = HybridSegmentedTrend(n_breakpoints=0)
    with pytest.raises(ValueError, match="alpha"):
        model.fit(t, x, alpha=bad_alpha)


def test_f10_invalid_breakpoint_counts_and_unknown_kwargs_rejected():
    """Negative/invalid breakpoint counts raise ValueError; unknown kwargs & find_best_segmentation n_breakpoints raise TypeError."""
    t = np.arange(25, dtype=float)
    x = 0.5 * t

    for bad_k in (-1, 1.5, True, "1"):
        with pytest.raises(ValueError):
            segmented_trend_test(x, t, n_breakpoints=bad_k)
        with pytest.raises(ValueError):
            segmented_trend_test(x, t, max_breakpoints=bad_k)

    # Unknown / misspelled kwargs must raise TypeError
    for bad_kw in (
        {"use_baging": True},
        {"agg_method": "median"},
        {"unknown_flag": 123},
    ):
        with pytest.raises(TypeError):
            segmented_trend_test(x, t, n_breakpoints=0, **bad_kw)
        with pytest.raises(TypeError):
            find_best_segmentation(x, t, max_breakpoints=1, **bad_kw)

    # find_best_segmentation rejects n_breakpoints
    with pytest.raises(TypeError, match="n_breakpoints"):
        find_best_segmentation(x, t, max_breakpoints=2, n_breakpoints=1)

    # Valid promoted kwargs are accepted in both functions
    res = segmented_trend_test(
        x, t, n_breakpoints=0, lt_mult=0.25, gt_mult=1.2, sens_slope_method="lwp", autocorr_method="yue_wang"
    )
    assert res.n_breakpoints == 0
    res2, summary2 = find_best_segmentation(
        x, t, max_breakpoints=1, lt_mult=0.25, gt_mult=1.2, sens_slope_method="lwp", autocorr_method="yue_wang",
        random_state=42,
    )
    assert list(summary2.columns[:5]) == ["n_breakpoints", "bic", "aic", "sar", "converged"] or set(
        ["n_breakpoints", "bic", "aic", "sar", "converged"]
    ).issubset(summary2.columns)


# =============================================================================
# F-13: Plotting Bugs (a-e)
# =============================================================================


def test_f13a_ci_band_pivots_on_fitted_line_censored_ats():
    """
    Slope CI band must pivot on (t_center, slope * t_center + intercept), eliminating
    the censored ATS offset, and handle NaN values in x_data cleanly.
    """
    t = np.arange(30, dtype=float)
    raw = 0.5 * t - 3.0
    x_str = [f"<{2.0}" if v < 2.0 else f"{v:.4f}" for v in raw]
    df_c = mk.prepare_censored_data(x_str)

    res = segmented_trend_test(df_c, t, n_breakpoints=0, random_state=42)
    fig, ax = plot_segmented_trend(res, df_c, t, save_path=None)
    try:
        polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
        assert len(polys) == 1
        verts = polys[0].get_paths()[0].vertices

        row = res.segments.iloc[0]
        slope = row["slope_per_second"]
        intercept = row["intercept"]
        t_center = float(np.median(t))
        y_center = slope * t_center + intercept

        # Check at both endpoints that trend line is bounded by CI band and
        # linear interpolation to t_center equals y_center exactly
        t0, t1 = float(t.min()), float(t.max())
        y_line_0 = slope * t0 + intercept
        y_line_1 = slope * t1 + intercept

        at_t0 = verts[np.isclose(verts[:, 0], t0)][:, 1]
        at_t1 = verts[np.isclose(verts[:, 0], t1)][:, 1]
        assert at_t0.min() - 1e-10 <= y_line_0 <= at_t0.max() + 1e-10
        assert at_t1.min() - 1e-10 <= y_line_1 <= at_t1.max() + 1e-10

        # Interpolate upper/lower boundaries to t_center: both must equal y_center
        y_lo_0 = y_center + row["lower_ci_per_second"] * (t0 - t_center)
        y_hi_0 = y_center + row["upper_ci_per_second"] * (t0 - t_center)
        assert np.isclose(at_t0.min(), min(y_lo_0, y_hi_0), atol=1e-10)
        assert np.isclose(at_t0.max(), max(y_lo_0, y_hi_0), atol=1e-10)
    finally:
        plt.close(fig)

    # Also test NaN in x_data
    x_nan = np.arange(20, dtype=float)
    x_nan[3] = np.nan
    t_nan = np.arange(20, dtype=float)
    res_nan = segmented_trend_test(x_nan, t_nan, n_breakpoints=0, random_state=42)
    fig2, ax2 = plot_segmented_trend(res_nan, x_nan, t_nan, save_path=None)
    try:
        polys2 = [c for c in ax2.collections if isinstance(c, PolyCollection)]
        assert len(polys2) == 1
    finally:
        plt.close(fig2)


def test_f13b_create_segments_and_plot_with_nan_time():
    """_create_segments and plot_segmented_trend must ignore NaN/NaT/inf timestamps."""
    assert _create_segments(np.array([np.nan, 1.0, 2.0, 3.0]), np.array([1.5])) == [
        (1.0, 1.5),
        (1.5, 3.0),
    ]
    assert _create_segments(np.array([np.nan, np.nan]), np.array([])) == []
    assert _create_segments(np.array([-np.inf, 1.0, 5.0, np.inf]), np.array([np.nan])) == [
        (1.0, 5.0)
    ]

    t = np.arange(20, dtype=float)
    x = 0.5 * t
    t_with_nan = np.concatenate(([np.nan], t))
    x_with_nan = np.concatenate(([1.0], x))
    res = segmented_trend_test(x_with_nan, t_with_nan, n_breakpoints=0, random_state=42)
    fig, ax = plot_segmented_trend(res, x_with_nan, t_with_nan, save_path=None)
    try:
        assert len(ax.lines) >= 1
        xdata = ax.lines[0].get_xdata()
        assert np.all(np.isfinite(xdata))
    finally:
        plt.close(fig)


def test_f13c_right_censored_marker_upward_triangle():
    """Left-censored ('lt') and right-censored ('gt') points must use distinct labels & markers."""
    t = np.arange(20, dtype=float)
    x_str = [f"<{2.0}" if i < 4 else (f">{15.0}" if i >= 16 else f"{float(i):.1f}") for i in range(20)]
    df_c = mk.prepare_censored_data(x_str)
    res = segmented_trend_test(df_c, t, n_breakpoints=0, random_state=42)
    fig, ax = plot_segmented_trend(res, df_c, t, save_path=None)
    try:
        _, labels = ax.get_legend_handles_labels()
        assert "Left-Censored" in labels
        assert "Right-Censored" in labels

        # Verify scatter collections for lt and gt have distinct marker paths
        lt_col = next(c for c in ax.collections if c.get_label() == "Left-Censored")
        gt_col = next(c for c in ax.collections if c.get_label() == "Right-Censored")
        assert not np.array_equal(lt_col.get_paths()[0].vertices, gt_col.get_paths()[0].vertices)
    finally:
        plt.close(fig)


def test_f13d_save_path_none_returns_fig_ax_and_no_leak_when_saved():
    """save_path=None returns (fig, ax); save_path!=None saves, closes internal fig, and returns None."""
    t = np.arange(20, dtype=float)
    x = 0.5 * t
    res = segmented_trend_test(x, t, n_breakpoints=0, random_state=42)

    ret = plot_segmented_trend(res, x, t, save_path=None)
    assert isinstance(ret, tuple) and len(ret) == 2
    fig, ax = ret
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    before_figs = set(plt.get_fignums())
    buf = io.BytesIO()
    ret2 = plot_segmented_trend(res, x, t, save_path=buf)
    assert ret2 is None
    assert buf.getvalue()[:8] == b"\x89PNG\r\n\x1a\n"
    assert set(plt.get_fignums()) == before_figs

    # Caller-provided ax stays open even when save_path is provided
    ext_fig, ext_ax = plt.subplots()
    try:
        buf2 = io.BytesIO()
        ret3 = plot_segmented_trend(res, x, t, save_path=buf2, ax=ext_ax)
        assert ret3 is None
        assert ext_fig.number in plt.get_fignums()
    finally:
        plt.close(ext_fig)


def test_f13e_invalid_and_numeric_slope_scaling_preserves_ci_bands_and_alpha_label():
    """
    Invalid datetime slope_scaling and numeric slope_scaling must preserve *_per_second
    columns and CI bands; ci_percent label must not truncate fractional/float alpha.
    """
    t_dt = pd.date_range("2020-01-01", periods=25, freq="D")
    x = np.arange(25, dtype=float)

    res_bad = segmented_trend_test(x, t_dt, n_breakpoints=0, slope_scaling="invalid_unit", random_state=42)
    assert any("Slope scaling failed" in w for w in res_bad.warnings)
    assert "lower_ci_per_second" in res_bad.segments.columns
    assert res_bad.segments["slope_units"].iloc[0] == "units per second"
    fig, ax = plot_segmented_trend(res_bad, x, t_dt, save_path=None)
    try:
        assert any(isinstance(c, PolyCollection) for c in ax.collections)
    finally:
        plt.close(fig)

    t_num = np.arange(25, dtype=float)
    res_num = segmented_trend_test(x, t_num, n_breakpoints=0, slope_scaling="year", random_state=42)
    assert any("requires datetime" in w for w in res_num.warnings)
    assert "lower_ci_per_second" in res_num.segments.columns
    assert res_num.segments["slope_units"].iloc[0] == "units per time"
    fig2, ax2 = plot_segmented_trend(res_num, x, t_num, save_path=None)
    try:
        assert any(isinstance(c, PolyCollection) for c in ax2.collections)
    finally:
        plt.close(fig2)

    # Non-truncating ci_percent formatting
    for alpha, expected_label in ((0.005, "99.5% CI"), (0.14, "86% CI"), (0.05, "95% CI")):
        r_a = segmented_trend_test(x, t_num, n_breakpoints=0, alpha=alpha, random_state=42)
        f_a, a_a = plot_segmented_trend(r_a, x, t_num, save_path=None)
        try:
            _, labels = a_a.get_legend_handles_labels()
            assert any(expected_label in lbl for lbl in labels), f"Expected {expected_label} in {labels}"
        finally:
            plt.close(f_a)


# =============================================================================
# F-15: Validation Integrity (common_validation.py & validation/45)
# =============================================================================


def _load_module_from_path(name, rel_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(repo_root, rel_path)
    spec = importlib.util.spec_from_file_location(name, full_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_f15_common_validation_slope_extraction_k1_and_k2():
    """
    fit_piecewise_ols in validation/common_validation.py must extract alpha1..alpha_{k+1}
    as segment slopes (not swapped beta1/alpha_i) for both k=1 and k=2.
    """
    cv = _load_module_from_path("common_validation_test", "validation/common_validation.py")
    rng = np.random.default_rng(42)

    # k=1: true slopes [0.5, 1.5]
    t1 = np.arange(100, dtype=float)
    y1 = np.where(t1 < 50, 0.5 * t1, 25.0 + 1.5 * (t1 - 50)) + rng.normal(0, 0.2, 100)
    np.random.seed(42)
    n1, bps1, slopes1 = cv.fit_piecewise_ols(t1, y1, max_breakpoints=2)
    assert n1 == 1
    assert len(bps1) == 1 and abs(bps1[0] - 50.0) < 3.0
    np.testing.assert_allclose(slopes1, [0.5, 1.5], atol=0.1)

    # k=2: true slopes [0.5, -0.5, 1.5]
    t2 = np.arange(120, dtype=float)
    y2 = np.where(
        t2 < 40,
        0.5 * t2,
        np.where(t2 < 80, 20.0 - 0.5 * (t2 - 40), 0.0 + 1.5 * (t2 - 80)),
    ) + rng.normal(0, 0.2, 120)
    np.random.seed(42)
    n2, bps2, slopes2 = cv.fit_piecewise_ols(t2, y2, max_breakpoints=2)
    assert n2 == 2
    assert len(bps2) == 2
    np.testing.assert_allclose(slopes2, [0.5, -0.5, 1.5], atol=0.1)

    # Direct test of _extract_piecewise_estimates helper
    mock_est = {
        "breakpoint1": {"estimate": 40.0},
        "breakpoint2": {"estimate": 80.0},
        "alpha1": {"estimate": 0.5},
        "alpha2": {"estimate": -0.5},
        "alpha3": {"estimate": 1.5},
        "beta1": {"estimate": -1.0},
        "beta2": {"estimate": 2.0},
    }
    bps_m, slopes_m = cv._extract_piecewise_estimates(mock_est, 2)
    assert bps_m == [40.0, 80.0]
    assert slopes_m == [0.5, -0.5, 1.5]


def test_f15_v45_03_enforces_exact_n_breakpoints():
    """validation/45_Segmented_Regression/run_validation.py must require exact selected_n == 1 in V-45-03."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v45_path = os.path.join(repo_root, "validation/45_Segmented_Regression/run_validation.py")
    with open(v45_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "selected_n >= 1" not in src
    assert "selected_n == 1" in src
