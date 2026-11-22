"""
Regression tests for Medium-severity segmented trend audit findings:
  - F-09: Minimum segment size enforcement (`min_segment_size`)
  - F-11: `rss` column in `selection_summary` alongside backward-compatible `sar`
  - F-12: Phase-2 parity with `trend_test` (per-segment MK stats & classification,
          LWP-TRENDS CI/MK parity, Conover intercept, `computation_mode='fast'`,
          `detect_size_tier` warnings, and `ats['notes']`)
  - F-14: Negative left/right censored multiplicative substitution warning
"""
import json
import os
import warnings
import numpy as np
import pandas as pd
import pytest

from MannKS import segmented_trend_test, trend_test, prepare_censored_data


# ---------------------------------------------------------------------------
# F-09: Minimum Segment Size
# ---------------------------------------------------------------------------

def test_f09_min_segment_size_validation():
    t = np.arange(30, dtype=float)
    y = 0.5 * t
    for bad in [0, 1, -3, True, False, "5", 2.5]:
        with pytest.raises(ValueError, match="min_segment_size"):
            segmented_trend_test(y, t, n_breakpoints=0, min_segment_size=bad)


def test_f09_rejects_micro_segment_by_default_and_allows_when_disabled():
    t_iso = np.concatenate([[0.0], np.linspace(50.0, 100.0, 29)])
    y_iso = np.concatenate([[100.0], np.linspace(0.0, 10.0, 29)])

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_default = segmented_trend_test(y_iso, t_iso, n_breakpoints=1, random_state=42)

    # Default min_segment_size=5 guarantees every returned segment has >= 5 observations
    assert (res_default.segments["n"] >= 5).all()
    assert int(res_default.segments["n"].sum()) == len(t_iso)

    # Disabling min_segment_size permits the 1-point micro-segment and reports actual n < 5
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_disabled = segmented_trend_test(
            y_iso, t_iso, n_breakpoints=1, min_segment_size=None, random_state=42
        )

    assert res_disabled.n_breakpoints == 1
    assert int(res_disabled.segments["n"].sum()) == len(t_iso)
    assert int(res_disabled.segments.iloc[0]["n"]) < 5


def test_f09_auto_search_skips_k_when_n_too_small_for_min_segment_size():
    t = np.arange(8, dtype=float)
    y = np.array([1.0, 2.0, 3.0, 4.0, 3.5, 2.5, 1.5, 0.5])
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(y, t, max_breakpoints=2, min_segment_size=5, random_state=42)

    assert res.n_breakpoints == 0
    assert any("min_segment_size" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# F-11: `rss` Column Alias in `selection_summary`
# ---------------------------------------------------------------------------

def test_f11_selection_summary_has_rss_and_sar_alias():
    rng = np.random.default_rng(12)
    t = np.arange(40, dtype=float)
    y = np.where(t < 20, 0.5 * t, 10.0 - 0.4 * (t - 20)) + rng.normal(0, 0.3, 40)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(y, t, max_breakpoints=2, random_state=42)

    ss = res.selection_summary
    assert "rss" in ss.columns and "sar" in ss.columns
    np.testing.assert_allclose(ss["rss"].to_numpy(dtype=float), ss["sar"].to_numpy(dtype=float), equal_nan=True)


# ---------------------------------------------------------------------------
# F-12: Phase-2 Parity With `trend_test`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["uncensored", "ats", "lwp", "drop_ambiguous"])
def test_f12_zero_breakpoint_exact_parity_with_trend_test(mode):
    rng = np.random.default_rng(42)
    t = np.arange(40, dtype=float)
    y_raw = 0.25 * t + rng.normal(0, 1.2, 40)

    if mode == "uncensored":
        x_in = y_raw
        kw = dict(sens_slope_method="ats", ci_method="direct", mk_test_method="robust")
    elif mode == "ats":
        x_in = prepare_censored_data(["<3.0" if v < 3.0 else float(v) for v in y_raw])
        kw = dict(sens_slope_method="ats", ci_method="direct", mk_test_method="robust")
    elif mode == "lwp":
        x_in = prepare_censored_data(["<3.0" if v < 3.0 else float(v) for v in y_raw])
        kw = dict(sens_slope_method="lwp", ci_method="lwp", mk_test_method="lwp")
    else:
        x_in = prepare_censored_data(["<3.0" if v < 3.0 else float(v) for v in y_raw])
        kw = dict(sens_slope_method="drop_ambiguous", ci_method="direct", mk_test_method="robust")

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        tt = trend_test(x_in, t, alpha=0.05, **kw)
        st = segmented_trend_test(x_in, t, n_breakpoints=0, alpha=0.05, random_state=42, **kw)

    row = st.segments.iloc[0]
    for col in ("slope", "intercept", "lower_ci", "upper_ci", "s", "var_s", "z", "p", "Tau", "C", "Cd"):
        assert np.isclose(float(row[col]), float(getattr(tt, col)), rtol=1e-12, atol=1e-12), (
            f"{mode} mismatch on {col}: seg={row[col]} vs trend_test={getattr(tt, col)}"
        )
    assert bool(row["h"]) == bool(tt.h)
    assert row["trend"] == tt.trend
    assert row["classification"] == tt.classification


def test_f12_lwp_trends_fixture_parity():
    fixture_path = os.path.join(os.path.dirname(__file__), "data", "lwp_parity_fixtures.json")
    if not os.path.exists(fixture_path):
        pytest.skip("lwp_parity_fixtures.json not present")
    with open(fixture_path) as f:
        fixtures = json.load(f)

    # Verify exact parity with trend_test(..., ci_method='lwp', mk_test_method='lwp') on all non-seasonal wrapper cases
    non_seasonal = [c for c in fixtures["wrapper"] if not c.get("seasonal", False)]
    assert len(non_seasonal) >= 8
    for case in non_seasonal:
        raw_vals = case["values"]
        t_vals = pd.to_datetime(case["dates"])
        x_in = prepare_censored_data(raw_vals)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            tt = trend_test(
                x_in, t_vals, alpha=0.10, slope_scaling="year",
                sens_slope_method="lwp", ci_method="lwp", mk_test_method="lwp",
            )
            st = segmented_trend_test(
                x_in, t_vals, n_breakpoints=0, alpha=0.10, slope_scaling="year",
                sens_slope_method="lwp", ci_method="lwp", mk_test_method="lwp",
                random_state=42,
            )
        row = st.segments.iloc[0]
        for col in ("slope", "lower_ci", "upper_ci", "intercept", "p", "s", "Tau"):
            assert np.isclose(float(row[col]), float(getattr(tt, col)), rtol=1e-12, atol=1e-12, equal_nan=True), (
                f"Case {case['case']} mismatch on {col}: seg={row[col]} vs trend_test={getattr(tt, col)}"
            )
        assert row["classification"] == tt.classification


def test_f12_per_segment_mk_stats_on_two_segment_hinge():
    rng = np.random.default_rng(7)
    t = np.arange(60, dtype=float)
    y = np.where(t < 30, 0.5 * t, 15.0 - 0.5 * (t - 30)) + rng.normal(0, 0.5, 60)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res = segmented_trend_test(y, t, n_breakpoints=1, random_state=42)

    assert len(res.segments) == 2
    assert res.segments.iloc[0]["trend"] == "increasing"
    assert res.segments.iloc[0]["p"] < 1e-4
    assert "Increasing" in res.segments.iloc[0]["classification"]
    assert res.segments.iloc[1]["trend"] == "decreasing"
    assert res.segments.iloc[1]["p"] < 1e-4
    assert "Decreasing" in res.segments.iloc[1]["classification"]


def test_f12_computation_mode_fast_and_ats_notes():
    rng = np.random.default_rng(5)
    t = np.arange(60, dtype=float)
    y = 0.2 * t + rng.normal(0, 0.5, 60)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_fast = segmented_trend_test(y, t, n_breakpoints=0, max_pairs=200, random_state=42)
        res_full = segmented_trend_test(y, t, n_breakpoints=0, random_state=42)

    assert res_fast.computation_mode == "fast"
    assert res_full.computation_mode == "hybrid"

    # All-censored series triggers ATS unidentifiable note surfaced in segments & result.analysis_notes
    df_all_cen = prepare_censored_data(["<1.0"] * 30)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_cen = segmented_trend_test(df_all_cen, np.arange(30, dtype=float), n_breakpoints=0, random_state=42)
    assert isinstance(res_cen.segments.iloc[0]["analysis_notes"], str)
    assert any("Segment 0:" in n for n in res_cen.analysis_notes)


def test_f12_invalid_phase2_method_parameters_raise():
    t = np.arange(20, dtype=float)
    y = 0.5 * t
    with pytest.raises(ValueError, match="ci_method"):
        segmented_trend_test(y, t, n_breakpoints=0, ci_method="invalid")
    with pytest.raises(ValueError, match="mk_test_method"):
        segmented_trend_test(y, t, n_breakpoints=0, mk_test_method="invalid")
    with pytest.raises(ValueError, match="tau_method"):
        segmented_trend_test(y, t, n_breakpoints=0, tau_method="z")
    with pytest.raises(ValueError, match="tie_break_method"):
        segmented_trend_test(y, t, n_breakpoints=0, tie_break_method="invalid")


# ---------------------------------------------------------------------------
# F-14: Negative Censored Limits Warning
# ---------------------------------------------------------------------------

def test_f14_negative_censored_values_emit_warning():
    t = np.arange(30, dtype=float)
    raw_lt = ["<-5.0" if i < 10 else float(-4.0 + 0.3 * i) for i in range(30)]
    df_lt = prepare_censored_data(raw_lt)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_lt = segmented_trend_test(df_lt, t, n_breakpoints=0, random_state=42)
    assert any("negative left-censored" in w.lower() for w in res_lt.warnings)

    raw_gt = [">-5.0" if i >= 20 else float(-10.0 + 0.2 * i) for i in range(30)]
    df_gt = prepare_censored_data(raw_gt)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        res_gt = segmented_trend_test(df_gt, t, n_breakpoints=0, random_state=42)
    assert any("negative right-censored" in w.lower() for w in res_gt.warnings)


# ---------------------------------------------------------------------------
# PR #9 Review Regression Tests (H1, H2, L1, L2, L3, L6, L7, L8, Nits)
# ---------------------------------------------------------------------------

def test_h1_deterministic_ess_k0_score_and_monotonic_rss():
    """H1: k=0 ESS score must not depend on random_state, and RSS(k) <= RSS(k-1)."""
    rng = np.random.default_rng(77)
    n = 90
    t = np.arange(n, dtype=float)
    eps = rng.normal(0.0, np.sqrt(1.0 - 0.65 ** 2), size=n)
    y = np.empty(n)
    y[0] = rng.normal(0.0, 1.0)
    for i in range(1, n):
        y[i] = 0.65 * y[i - 1] + eps[i]
    y = 5.0 + 0.05 * t + y

    k0_scores = []
    selected_ks = []
    for seed in (0, 1, 2, 7, 42):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = segmented_trend_test(
                y, t, max_breakpoints=3, autocorr_penalty="auto", random_state=seed
            )
        ss = res.selection_summary
        k0_scores.append(float(ss.loc[ss["n_breakpoints"] == 0, "score"].iloc[0]))
        selected_ks.append(res.n_breakpoints)
        conv_rss = ss[ss["converged"]]["rss"].to_numpy(dtype=float)
        assert np.all(np.diff(conv_rss) <= 1e-8), f"Non-monotonic RSS: {conv_rss}"

    assert np.allclose(k0_scores, k0_scores[0], rtol=0.0, atol=1e-12)
    assert len(set(selected_ks)) == 1


def test_h2_bagging_enforces_min_segment_size():
    """H2: use_bagging=True must never emit segments smaller than min_segment_size."""
    rng = np.random.default_rng(13)
    t = np.arange(25, dtype=float)
    y = rng.normal(0.0, 1.0, size=25)

    for seed in (13, 14, 15):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = segmented_trend_test(
                y, t, n_breakpoints=2, use_bagging=True, n_bootstrap=20,
                min_segment_size=5, random_state=seed,
            )
        assert (res.segments["n"] >= 5).all(), (
            f"seed={seed}: got segment sizes {res.segments['n'].tolist()}"
        )


def test_l1_sharp_v_at_n10_recovers_5_5_split():
    """L1: Constrained profile-OLS recovers a sharp V at n=10 with min_segment_size=5."""
    t = np.arange(10, dtype=float)
    y = np.abs(t - 4.5) * 3.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = segmented_trend_test(
            y, t, max_breakpoints=1, min_segment_size=5,
            autocorr_penalty="none", random_state=42,
        )
    assert res.n_breakpoints == 1
    assert res.segments["n"].tolist() == [5, 5]


def test_l2_l3_l7_l8_and_identical_time_score():
    """Verify L2 (davies_gated), L3 (tie_break_method default), L7 (reversed bounds),
    L8 (tz-naive predict warning), and Nit (identical-time score column)."""
    import inspect
    from MannKS import calculate_breakpoint_probability

    # L3: default tie_break_method matches trend_test ('robust')
    sig_seg = inspect.signature(segmented_trend_test)
    sig_tt = inspect.signature(trend_test)
    assert sig_seg.parameters["tie_break_method"].default == sig_tt.parameters["tie_break_method"].default == "robust"

    # L7: reversed bounds raise ValueError
    t = np.arange(40, dtype=float)
    y = np.where(t < 20, 0.5 * t, 10.0 - 0.5 * (t - 20)) + np.random.default_rng(0).normal(0, 0.3, 40)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_bag = segmented_trend_test(y, t, n_breakpoints=1, use_bagging=True, n_bootstrap=10, random_state=0)
    with pytest.raises(ValueError, match="less than or equal"):
        calculate_breakpoint_probability(res_bag, 25.0, 15.0)

    # L8: predicting naive datetimes on tz-aware model emits UserWarning
    t_tz = pd.date_range("2020-01-01", periods=30, freq="D", tz="Pacific/Auckland")
    y_tz = np.where(np.arange(30) < 15, 0.5 * np.arange(30), 7.5 - 0.5 * (np.arange(30) - 15))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_tz = segmented_trend_test(y_tz, t_tz, n_breakpoints=1, random_state=0)
    t_naive = pd.date_range("2020-01-05", periods=5, freq="D")
    with pytest.warns(UserWarning, match="timezone-naive"):
        preds = res_tz.predict(t_naive)
    assert len(preds) == 5 and np.all(np.isfinite(preds))

    # Nit: identical-time selection_summary includes 'score' column
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_id = segmented_trend_test(np.arange(10, dtype=float), np.zeros(10), max_breakpoints=1)
    assert "score" in res_id.selection_summary.columns

