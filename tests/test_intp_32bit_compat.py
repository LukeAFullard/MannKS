"""
Regression and cross-architecture compatibility tests for pointer-sized integer
(`np.intp`) array indexing and 32-bit environments (e.g., Pyodide/WebAssembly).
"""
import warnings
import numpy as np
import pandas as pd
import pytest

import MannKS as mk
import MannKS._stats as st
from MannKS._stats import (
    _iter_triu_blocks,
    _rle_lengths,
    _censored_pair_slopes,
    _censor_codes,
    _mk_score_and_var_censored,
    _sens_estimator_unequal_spacing,
    _sens_estimator_censored,
    _sens_estimator_adaptive,
    _sens_estimator_censored_adaptive,
)
from MannKS._ats import _pair_thresholds, _ATSScore, make_intervals, ats_slope, seasonal_ats_slope
from MannKS._large_dataset import _sample_pairs
from MannKS._bootstrap import (
    _moving_block_bootstrap_indices,
    block_bootstrap_mann_kendall,
    block_bootstrap_confidence_intervals,
)
from MannKS._autocorr import effective_sample_size
from MannKS.analysis_notes import get_sens_slope_analysis_note


@pytest.fixture
def simulated_32bit_intp(monkeypatch):
    """
    Simulate a 32-bit pointer environment (such as Pyodide/wasm32) where:
    - `np.intp` is `np.int32`
    - `np.repeat` enforces NumPy's C-level `'safe'` casting rule from `repeats`
      to `np.int32` (`NPY_INTP`), which raises:
      `TypeError: Cannot cast array data from dtype('int64') to dtype('int32') according to the rule 'safe'`
      if `repeats` is `int64`.
    """
    orig_repeat = np.repeat

    def strict_32bit_repeat(a, repeats, axis=None):
        repeats_arr = np.asarray(repeats)
        if not np.can_cast(repeats_arr.dtype, np.int32, casting='safe'):
            raise TypeError(
                f"Cannot cast array data from {repeats_arr.dtype!r} to "
                f"dtype('int32') according to the rule 'safe'"
            )
        return orig_repeat(a, repeats_arr.astype(np.int32, copy=False), axis=axis)

    monkeypatch.setattr(np, 'intp', np.int32)
    monkeypatch.setattr(np, 'repeat', strict_32bit_repeat)


def test_iter_triu_blocks_dtypes_and_values():
    """Verify `_iter_triu_blocks` yields `np.intp` arrays matching `np.triu_indices`."""
    n = 45
    blocks = list(_iter_triu_blocks(n, max_pairs_per_block=120))
    assert len(blocks) > 1
    for i, j in blocks:
        assert i.dtype == np.intp
        assert j.dtype == np.intp

    all_i = np.concatenate([b[0] for b in blocks])
    all_j = np.concatenate([b[1] for b in blocks])
    ref_i, ref_j = np.triu_indices(n, k=1)
    np.testing.assert_array_equal(all_i, ref_i)
    np.testing.assert_array_equal(all_j, ref_j)


def test_iter_triu_blocks_simulated_32bit(simulated_32bit_intp):
    """
    Regression test for `TypeError: Cannot cast array data from dtype('int64')
    to dtype('int32') according to the rule 'safe'` inside `_iter_triu_blocks`.
    """
    blocks = list(_iter_triu_blocks(40, max_pairs_per_block=100))
    assert len(blocks) > 1
    for i, j in blocks:
        assert i.dtype == np.int32
        assert j.dtype == np.int32


def test_numpy_estimators_simulated_32bit(simulated_32bit_intp, monkeypatch):
    """
    Verify all NumPy-based slope, MK, ATS, bootstrap, autocorr, and analysis-note
    functions execute cleanly when `np.intp` is `np.int32` and `np.repeat`
    rejects `int64` repeats.
    """
    # 1. _rle_lengths
    empty_rle = _rle_lengths(np.array([]))
    assert empty_rle.dtype == np.int32
    rle = _rle_lengths(np.array([1.0, 1.0, 2.0, 5.0, 5.0, 5.0]))
    assert rle.dtype == np.int32
    np.testing.assert_array_equal(rle, [2, 1, 3])

    # 2. _sample_pairs (small pool, large pool, and huge n > 46340)
    i_small, j_small = _sample_pairs(50, max_pairs=200, random_state=42)
    assert i_small.dtype == np.int32 and j_small.dtype == np.int32
    assert np.all(i_small < j_small)

    i_large, j_large = _sample_pairs(500, max_pairs=100, random_state=42)
    assert i_large.dtype == np.int32 and j_large.dtype == np.int32
    assert np.all(i_large < j_large)

    i_huge, j_huge = _sample_pairs(60000, max_pairs=500, random_state=42)
    assert i_huge.dtype == np.int32 and j_huge.dtype == np.int32
    assert np.all((i_huge >= 0) & (i_huge < j_huge) & (j_huge < 60000))

    # 3. _moving_block_bootstrap_indices
    boot_idx = _moving_block_bootstrap_indices(35, block_size=6, rng=42)
    assert boot_idx.dtype == np.int32 and len(boot_idx) == 35

    # 4. _sens_estimator_unequal_spacing & _sens_estimator_adaptive
    n = 30
    t = np.arange(n, dtype=np.int32)
    x = (t * 2 + (t % 4)).astype(np.int32)
    slopes_exact = _sens_estimator_unequal_spacing(x, t)
    assert len(slopes_exact) == n * (n - 1) // 2
    slopes_fast = _sens_estimator_adaptive(x, t, max_pairs=100, random_state=0)
    assert len(slopes_fast) <= 100

    # 5. _sens_estimator_censored & _sens_estimator_censored_adaptive & analysis_notes
    cens = np.zeros(n, dtype=bool)
    cens[[0, 3]] = True
    cen_type = np.where(cens, 'lt', 'not')
    for method in ('lwp', 'drop_ambiguous'):
        c_slopes = _sens_estimator_censored(x, t, cen_type, method=method)
        assert len(c_slopes) == n * (n - 1) // 2
        note = get_sens_slope_analysis_note(c_slopes, t, cen_type)
        assert isinstance(note, str)
        c_slopes_fast = _sens_estimator_censored_adaptive(
            x, t, cen_type, method=method, max_pairs=80, random_state=0
        )
        assert len(c_slopes_fast) <= 80

    # 6. _censored_pair_slopes & _pair_thresholds with mixed int32/int64 index inputs
    code = _censor_codes(cen_type)
    i_in = np.array([0, 1, 2], dtype=np.int32)
    j_in = np.array([2, 3, 4], dtype=np.int64)
    pair_slopes, valid = _censored_pair_slopes(x.astype(float), t.astype(float), code, i_in, j_in)
    assert len(pair_slopes) == 3 and np.all(valid)

    lo, hi = make_intervals(x, cens, cen_type=cen_type)
    a, b = _pair_thresholds(t.astype(float), lo, hi, i_in, j_in)
    assert len(a) == 3 and len(b) == 3

    # 7. ats_slope and seasonal_ats_slope (exact and sampled)
    ats_res = ats_slope(t, x, cens, cen_type=cen_type)
    assert np.isfinite(ats_res['beta'])
    ats_fast = ats_slope(t, x, cens, cen_type=cen_type, max_pairs=80, random_state=0)
    assert np.isfinite(ats_fast['beta'])
    seasons = (t % 4).astype(np.int32)
    s_ats = seasonal_ats_slope(t, x, cens, seasons, cen_type=cen_type)
    assert np.isfinite(s_ats['beta'])

    # 8. effective_sample_size and block bootstrap functions
    n_eff, acf1 = effective_sample_size(x, t, method='yue')
    assert np.isfinite(n_eff) and np.isfinite(acf1)

    p_boot, s_obs, _ = block_bootstrap_mann_kendall(
        x, t, cens, cen_type, block_size=4, n_bootstrap=10, random_state=0
    )
    assert np.isfinite(p_boot) and np.isfinite(s_obs)

    slope_b, lo_ci, hi_ci, _ = block_bootstrap_confidence_intervals(
        x, t, cens, cen_type, block_size=4, n_bootstrap=10, random_state=0
    )
    assert np.isfinite(slope_b) and np.isfinite(lo_ci) and np.isfinite(hi_ci)

    # 9. Chunked path in _mk_score_and_var_censored
    monkeypatch.setattr(st, '_FAST_S_MIN_N', 5)
    monkeypatch.setattr(st, '_censored_s_and_ties', lambda *args, **kwargs: None)
    s, var_s, d, tau = _mk_score_and_var_censored(x, t, cens, cen_type)
    assert np.isfinite(s) and var_s > 0


@pytest.mark.parametrize("int_dtype", [np.int32, np.int64])
def test_end_to_end_workflows_mixed_integer_dtypes(monkeypatch, int_dtype):
    """
    Run full MannKS workflows with `int32` and `int64` inputs while enforcing
    that `np.repeat` is always called with `np.intp` arrays.
    """
    orig_repeat = np.repeat

    def checked_repeat(a, repeats, axis=None):
        a_arr = np.asarray(a)
        r_arr = np.asarray(repeats)
        assert a_arr.dtype == np.intp, f"Expected a.dtype == np.intp, got {a_arr.dtype}"
        assert r_arr.dtype == np.intp, f"Expected repeats.dtype == np.intp, got {r_arr.dtype}"
        return orig_repeat(a, repeats, axis=axis)

    monkeypatch.setattr(np, 'repeat', checked_repeat)

    n = 36
    t_int = np.arange(n, dtype=int_dtype)
    x_int = (t_int * 2 + (t_int % 5)).astype(int_dtype)

    # 1. Uncensored trend_test (exact, fast, yue_wang, block_bootstrap)
    res_full = mk.trend_test(x_int, t_int, large_dataset_mode='full')
    assert res_full.trend == 'increasing' and np.isfinite(res_full.slope)

    res_fast = mk.trend_test(x_int, t_int, large_dataset_mode='fast', max_pairs=100, random_state=0)
    assert res_fast.trend == 'increasing' and np.isfinite(res_fast.slope)

    res_yw = mk.trend_test(x_int, t_int, autocorr_method='yue_wang')
    assert np.isfinite(res_yw.p)

    res_boot = mk.trend_test(x_int, t_int, autocorr_method='block_bootstrap', n_bootstrap=15, random_state=0)
    assert np.isfinite(res_boot.p) and np.isfinite(res_boot.lower_ci)

    # 2. Censored trend_test across all slope estimators
    raw_cens = ['<5' if v < 5 else (f'>{int(v)}' if idx == n - 1 else float(v))
                for idx, v in enumerate(x_int)]
    df_cens = mk.prepare_censored_data(raw_cens)
    for method in ('ats', 'lwp', 'drop_ambiguous'):
        res_c = mk.trend_test(df_cens, t_int, sens_slope_method=method)
        assert np.isfinite(res_c.slope)
        res_c_fast = mk.trend_test(df_cens, t_int, sens_slope_method=method,
                                   large_dataset_mode='fast', max_pairs=80, random_state=0)
        assert np.isfinite(res_c_fast.slope)

    # 3. Seasonal trend test (numeric and datetime, with bootstrap)
    res_seas = mk.seasonal_trend_test(x_int, t_int, period=12)
    assert res_seas.trend == 'increasing' and np.isfinite(res_seas.slope)

    res_seas_cens = mk.seasonal_trend_test(df_cens, t_int, period=12, sens_slope_method='lwp')
    assert np.isfinite(res_seas_cens.slope)

    res_seas_boot = mk.seasonal_trend_test(
        x_int, t_int, period=12, autocorr_method='block_bootstrap', n_bootstrap=15, random_state=0
    )
    assert np.isfinite(res_seas_boot.p)

    # 4. Check seasonality
    seas_check = mk.check_seasonality(x_int, t_int, period=12)
    assert np.isfinite(seas_check.h_statistic)

    # 5. Rolling trend and compare_periods
    roll_df = mk.rolling_trend_test(x_int, t_int, window=15, step=5, min_size=5)
    assert not roll_df.empty and np.all(np.isfinite(roll_df['slope']))

    comp = mk.compare_periods(x_int, t_int, breakpoint=18)
    assert np.isfinite(comp['slope_difference'])

    # 6. Segmented trend test
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        seg_res = mk.segmented_trend_test(
            x_int, t_int, n_breakpoints=1, use_bagging=True, n_bootstrap=10, random_state=0
        )
    assert seg_res.segments is not None
    pred = seg_res.predict(t_int)
    assert len(pred) == n
