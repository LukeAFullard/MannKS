import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from MannKS.plotting import plot_quantile_kendall, plot_quantile_trend
from MannKS.quantile_trend_test import QuantileTrendResult


def create_dummy_result(
    M=15,
    transform='none',
    slope_values=None,
    p_values=None,
    ci_low=None,
    ci_high=None,
    matrix_present=True,
    all_nan_ci=False,
    all_nan_p=False,
    slope_scaling=None,
    slope_units=None,
):
    ranks = np.arange(1, M + 1, dtype=float)
    p_grid = ranks / (M + 1.0)
    z_grid = norm.ppf(p_grid)

    if slope_values is None:
        slope = np.linspace(-2.0, 3.0, M)
    else:
        slope = np.asarray(slope_values, dtype=float)

    if transform == 'log':
        slope_pct = (np.exp(slope) - 1.0) * 100.0
    else:
        slope_pct = slope * 10.0

    if all_nan_p:
        p = np.full(M, np.nan)
    elif p_values is None:
        p = np.linspace(0.001, 0.4, M)
    else:
        p = np.asarray(p_values, dtype=float)

    if all_nan_ci:
        low_ci = np.full(M, np.nan)
        high_ci = np.full(M, np.nan)
        low_ci_pct = np.full(M, np.nan)
        high_ci_pct = np.full(M, np.nan)
    else:
        low_ci = slope - 0.5 if ci_low is None else np.asarray(ci_low, dtype=float)
        high_ci = slope + 0.5 if ci_high is None else np.asarray(ci_high, dtype=float)
        low_ci_pct = slope_pct - 5.0
        high_ci_pct = slope_pct + 5.0

    classifications = []
    for s, pv in zip(slope, p):
        if np.isnan(pv) or pv >= 0.05:
            classifications.append('No Trend')
        elif s > 0:
            classifications.append('Increasing')
        else:
            classifications.append('Decreasing')

    df_dict = {
        'rank': ranks,
        'quantile': p_grid,
        'plotting_position': p_grid,
        'z_position': z_grid,
        'slope': slope,
        'lower_ci': low_ci,
        'upper_ci': high_ci,
        'slope_pct': slope_pct,
        'lower_ci_pct': low_ci_pct,
        'upper_ci_pct': high_ci_pct,
        'p': p,
        'classification': classifications,
    }
    if slope_units is not None:
        df_dict['slope_units'] = [slope_units] * M

    df = pd.DataFrame(df_dict)

    matrix = pd.DataFrame(np.random.randn(8, M) + 5.0) if matrix_present else None

    return QuantileTrendResult(
        results=df,
        blocks=pd.DataFrame(),
        matrix=matrix,
        censored_matrix=None,
        cen_type_matrix=None,
        sampling_diagnostic={'flag': False},
        field_significance=None,
        n_ranks=M,
        target_length=M,
        block_frequency='YS',
        alpha=0.05,
        transform=transform,
        standardization='interpolate',
        is_datetime=True,
        computation_mode='matrix',
        analysis_notes=[],
        warnings=[],
    )


class BadBuffer:
    """Writable buffer that deliberately fails on write."""
    def write(self, data):
        raise IOError("Simulated disk/buffer write failure")


# =========================================================================
# SUITE 1: 100+ LOOPS MEMORY LEAK STRESS TESTING
# =========================================================================

def test_memory_leak_stress_100_loops(tmp_path):
    """
    Stress-test memory leaks across 120+ calls combining:
    - save_path=None
    - File paths (str and Path)
    - In-memory buffers
    - Deliberate save errors (bad buffer, impossible path, mock error)
    - Invalid arguments (bad x_scale, bad color_by, invalid result)
    - Empty DataFrames
    Asserts len(plt.get_fignums()) == 0 after every single call.
    """
    # Baseline assertion
    plt.close('all')
    assert len(plt.get_fignums()) == 0, "Matplotlib figures already open before test!"

    res = create_dummy_result(M=10)

    for i in range(120):
        mode = i % 10
        if mode == 0:
            # save_path=None, ax=None -> returns None immediately
            out = plot_quantile_trend(res, save_path=None, ax=None)
            assert out is None
        elif mode == 1:
            # File save using string path
            fpath = str(tmp_path / f"test_{i}.png")
            out = plot_quantile_trend(res, save_path=fpath)
            assert out == fpath
            assert os.path.exists(fpath)
        elif mode == 2:
            # Buffer save using io.BytesIO
            buf = io.BytesIO()
            out = plot_quantile_trend(res, save_path=buf, show_ci=True)
            assert out is buf
            assert buf.tell() > 0
        elif mode == 3:
            # Path object save
            p_obj = tmp_path / f"test_path_{i}.png"
            out = plot_quantile_trend(res, save_path=p_obj, show_fdc=True)
            assert out == p_obj
            assert p_obj.exists()
        elif mode == 4:
            # Deliberate failure: buffer write error
            bad_buf = BadBuffer()
            with pytest.raises(IOError, match="Simulated disk/buffer write failure"):
                plot_quantile_trend(res, save_path=bad_buf)
        elif mode == 5:
            # Deliberate failure: nonexistent directory
            impossible_path = "/nonexistent_dir_xyz_12345/subdir/leak_test.png"
            with pytest.raises((OSError, FileNotFoundError)):
                plot_quantile_trend(res, save_path=impossible_path)
        elif mode == 6:
            # Deliberate failure: mock savefig throwing OSError
            with patch.object(plt.Figure, 'savefig', side_effect=OSError("Disk full")):
                with pytest.raises(OSError, match="Disk full"):
                    plot_quantile_trend(res, save_path=str(tmp_path / f"fail_{i}.png"))
        elif mode == 7:
            # Invalid arguments
            with pytest.raises(ValueError, match="Unsupported x_scale"):
                plot_quantile_trend(res, save_path=str(tmp_path / f"inv_{i}.png"), x_scale="logarithmic")
            with pytest.raises(ValueError, match="Unsupported color_by"):
                plot_quantile_trend(res, save_path=str(tmp_path / f"inv_{i}.png"), color_by="invalid_mode")
            with pytest.raises(TypeError, match="Expected result to be"):
                plot_quantile_trend(12345, save_path=str(tmp_path / f"inv_{i}.png"))
        elif mode == 8:
            # Empty DataFrame result
            empty_res = QuantileTrendResult(
                results=pd.DataFrame(),
                blocks=pd.DataFrame(),
                matrix=None,
                censored_matrix=None,
                cen_type_matrix=None,
                sampling_diagnostic=None,
                field_significance=None,
                n_ranks=0,
                target_length=0,
                block_frequency='YS',
                alpha=0.05,
                transform='none',
                standardization='interpolate',
                is_datetime=True,
                computation_mode='matrix',
                analysis_notes=[],
                warnings=[],
            )
            buf = io.BytesIO()
            out = plot_quantile_trend(empty_res, save_path=buf)
            assert out is buf
            assert buf.tell() > 0
        elif mode == 9:
            # Alias test
            buf = io.BytesIO()
            out = plot_quantile_kendall(res, save_path=buf, color_by='classification')
            assert out is buf
            assert buf.tell() > 0

        # RIGID INVARIANT: Zero figure leaks after EVERY single iteration!
        fignums = plt.get_fignums()
        assert len(fignums) == 0, f"Memory leak detected at iteration {i}! Open fignums: {fignums}"


def test_caller_ax_figure_lifecycle():
    """
    Test figure lifecycle when caller provides an existing ax:
    1. If save_path is None: caller's figure MUST stay open.
    2. If save_path is provided: figure is saved and closed.
    """
    plt.close('all')
    res = create_dummy_result(M=5)

    # 1. External ax with save_path=None -> caller figure remains open
    fig_caller, ax_caller = plt.subplots()
    assert len(plt.get_fignums()) == 1
    returned_ax = plot_quantile_trend(res, ax=ax_caller, save_path=None)
    assert returned_ax is ax_caller
    assert len(plt.get_fignums()) == 1
    plt.close(fig_caller)
    assert len(plt.get_fignums()) == 0

    # 2. External ax with save_path provided -> figure saved and closed
    fig_caller2, ax_caller2 = plt.subplots()
    assert len(plt.get_fignums()) == 1
    buf = io.BytesIO()
    returned_val = plot_quantile_trend(res, ax=ax_caller2, save_path=buf)
    assert returned_val is buf
    assert len(plt.get_fignums()) == 0


# =========================================================================
# SUITE 2: EXTREME EDGE CASES
# =========================================================================

def test_single_quantile_M_1():
    """Extreme edge case: M=1 (single quantile)."""
    plt.close('all')
    res = create_dummy_result(M=1, matrix_present=True)

    # Normal scale
    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, x_scale='normal', show_ci=True, show_fdc=True)
    assert out is buf
    assert buf.tell() > 0
    assert len(plt.get_fignums()) == 0

    # Linear scale
    buf2 = io.BytesIO()
    out2 = plot_quantile_trend(res, save_path=buf2, x_scale='linear', color_by='classification', show_ci=True)
    assert out2 is buf2
    assert buf2.tell() > 0
    assert len(plt.get_fignums()) == 0


def test_massive_quantiles_M_1000():
    """Extreme edge case: M=1000 (1000 quantiles)."""
    plt.close('all')
    res = create_dummy_result(M=1000, matrix_present=True)

    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, x_scale='normal', show_ci=True, show_fdc=True)
    assert out is buf
    assert buf.tell() > 0
    assert len(plt.get_fignums()) == 0


def test_all_nan_p_values():
    """Extreme edge case: all p-values are NaN."""
    plt.close('all')
    res = create_dummy_result(M=15, all_nan_p=True)

    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, color_by='p')
    assert out is buf
    assert buf.tell() > 0
    assert len(plt.get_fignums()) == 0


def test_all_nan_confidence_intervals():
    """Extreme edge case: all lower/upper CI values are NaN with show_ci=True."""
    plt.close('all')
    res = create_dummy_result(M=15, all_nan_ci=True)

    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, show_ci=True)
    assert out is buf
    assert buf.tell() > 0
    assert len(plt.get_fignums()) == 0


def test_all_nan_slopes():
    """Extreme edge case: all slope values are NaN."""
    plt.close('all')
    res = create_dummy_result(M=10, slope_values=np.full(10, np.nan), all_nan_ci=True)

    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, show_ci=True)
    assert out is buf
    assert buf.tell() > 0
    assert len(plt.get_fignums()) == 0


def test_negative_slopes_under_log_transform():
    """
    Extreme edge case: negative slopes under log transform.
    Check slope_pct, percentage CIs, and axis labeling.
    """
    plt.close('all')
    # Negative log slopes: e.g. -0.5 -> exp(-0.5)-1 = -39.3%
    neg_slopes = np.linspace(-2.0, -0.05, 10)
    res = create_dummy_result(M=10, transform='log', slope_values=neg_slopes, slope_units='% per year')

    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, show_ci=True)
    assert out is buf
    assert buf.tell() > 0
    assert len(plt.get_fignums()) == 0


def test_missing_optional_fields_and_bare_dataframe():
    """
    Robustness test: missing optional fields in QuantileTrendResult,
    or passing bare pd.DataFrame with minimal columns.
    """
    plt.close('all')

    # 1. Bare DataFrame with only quantile and slope
    bare_df = pd.DataFrame({
        'quantile': [0.1, 0.5, 0.9],
        'slope': [1.2, 0.4, -0.8],
    })
    buf1 = io.BytesIO()
    out1 = plot_quantile_trend(bare_df, save_path=buf1, x_scale='normal')
    assert out1 is buf1
    assert len(plt.get_fignums()) == 0

    # 2. Bare DataFrame on linear scale
    buf2 = io.BytesIO()
    out2 = plot_quantile_trend(bare_df, save_path=buf2, x_scale='linear')
    assert out2 is buf2
    assert len(plt.get_fignums()) == 0

    # 3. Object with only .results and no other attributes
    class MinimalObj:
        def __init__(self, df):
            self.results = df

    min_res = MinimalObj(bare_df)
    buf3 = io.BytesIO()
    out3 = plot_quantile_trend(min_res, save_path=buf3, show_ci=True, show_fdc=True)
    assert out3 is buf3
    assert len(plt.get_fignums()) == 0

    # 4. Result with to_frame() method
    class ToFrameObj:
        def __init__(self, df):
            self._df = df
        def to_frame(self):
            return self._df

    tf_res = ToFrameObj(bare_df)
    buf4 = io.BytesIO()
    out4 = plot_quantile_trend(tf_res, save_path=buf4)
    assert out4 is buf4
    assert len(plt.get_fignums()) == 0


def test_fdc_mismatched_dimensions_handled_gracefully():
    """
    If result.matrix has dimensions not matching quantiles, FDC is safely omitted.
    """
    plt.close('all')
    res = create_dummy_result(M=15)
    # Alter matrix to have 5 columns instead of 15
    res = QuantileTrendResult(
        results=res.results,
        blocks=res.blocks,
        matrix=pd.DataFrame(np.ones((4, 5))),
        censored_matrix=None,
        cen_type_matrix=None,
        sampling_diagnostic=None,
        field_significance=None,
        n_ranks=15,
        target_length=15,
        block_frequency='YS',
        alpha=0.05,
        transform='none',
        standardization='interpolate',
        is_datetime=True,
        computation_mode='matrix',
        analysis_notes=[],
        warnings=[],
    )

    buf = io.BytesIO()
    out = plot_quantile_trend(res, save_path=buf, show_fdc=True)
    assert out is buf
    assert len(plt.get_fignums()) == 0


def test_legend_loc_and_title_variations():
    """Test custom titles, y_lim, and disabling/changing legend."""
    plt.close('all')
    res = create_dummy_result(M=10)

    buf = io.BytesIO()
    plot_quantile_trend(
        res,
        save_path=buf,
        title="Custom Test Title",
        y_lim=(-10.0, 10.0),
        legend_loc=None
    )
    assert len(plt.get_fignums()) == 0

    buf2 = io.BytesIO()
    plot_quantile_trend(
        res,
        save_path=buf2,
        legend_loc='none'
    )
    assert len(plt.get_fignums()) == 0


def test_invalid_input_structures_fail_cleanly_with_zero_leaks():
    """Invalid result structures should raise appropriate errors with zero figure leaks."""
    plt.close('all')
    buf = io.BytesIO()

    # None result with save_path
    with pytest.raises(TypeError):
        plot_quantile_trend(None, save_path=buf)
    assert len(plt.get_fignums()) == 0

    # Missing slope column
    bad_df = pd.DataFrame({'quantile': [0.1, 0.5, 0.9]})
    with pytest.raises(ValueError, match="must contain 'slope' or 'slope_pct'"):
        plot_quantile_trend(bad_df, save_path=buf)
    assert len(plt.get_fignums()) == 0

    # Missing quantile/position column
    bad_df2 = pd.DataFrame({'slope': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="must contain 'z_position'"):
        plot_quantile_trend(bad_df2, save_path=buf)
    assert len(plt.get_fignums()) == 0
