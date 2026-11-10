"""
Unit tests for plot_quantile_trend and plot_quantile_kendall.

Target module: MannKS.plotting
Functions:
    - plot_quantile_trend
    - plot_quantile_kendall
"""
import io
import os
import warnings
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from unittest.mock import patch

from MannKS.plotting import plot_quantile_trend, plot_quantile_kendall
from MannKS.quantile_trend_test import (
    quantile_trend_test,
    QuantileTrendResult,
)


@pytest.fixture(autouse=True)
def ensure_no_figure_leaks():
    """Ensure no matplotlib figures are leaked before or after any test."""
    plt.close('all')
    assert len(plt.get_fignums()) == 0
    yield
    plt.close('all')
    assert len(plt.get_fignums()) == 0


@pytest.fixture
def mock_quantile_result():
    """Creates a realistic mock QuantileTrendResult for plotting tests."""
    M = 15
    ranks = np.arange(1, M + 1, dtype=np.intp)
    p_grid = ranks / (M + 1.0)
    z_grid = norm.ppf(p_grid)

    df = pd.DataFrame({
        'rank': ranks,
        'quantile': p_grid,
        'plotting_position': p_grid,
        'z_position': z_grid,
        'slope': np.linspace(-2.0, 3.0, M),
        'lower_ci': np.linspace(-3.5, 1.5, M),
        'upper_ci': np.linspace(-0.5, 4.5, M),
        'slope_pct': np.linspace(-15.0, 25.0, M),
        'lower_ci_pct': np.linspace(-25.0, 10.0, M),
        'upper_ci_pct': np.linspace(-5.0, 40.0, M),
        'slope_units': ['units per year'] * M,
        'p': [0.01] * 5 + [0.07] * 5 + [0.35] * 5,
        'classification': ['Decreasing'] * 4 + ['No Trend'] * 7 + ['Increasing'] * 4,
        'h': [True] * 5 + [False] * 10,
    })

    matrix_df = pd.DataFrame(
        np.tile(np.linspace(10, 50, M), (8, 1)),
        index=pd.date_range('2010-01-01', periods=8, freq='YS'),
        columns=[f"r{r}" for r in ranks]
    )

    return QuantileTrendResult(
        results=df,
        blocks=pd.DataFrame(),
        matrix=matrix_df,
        censored_matrix=None,
        cen_type_matrix=None,
        sampling_diagnostic={'flag': False},
        field_significance=None,
        n_ranks=M,
        target_length=M,
        block_frequency='YE',
        alpha=0.05,
        transform='none',
        standardization='interpolate',
        is_datetime=True,
        computation_mode='matrix',
        analysis_notes=[],
        warnings=[],
    )


class TestPlotQuantileTrendCore:
    """Core interface and lifecycle tests for plot_quantile_trend."""

    def test_alias_identity(self):
        assert plot_quantile_kendall is plot_quantile_trend

    def test_save_path_none_and_ax_none_returns_none_zero_leak(self, mock_quantile_result):
        # When both save_path and ax are None, no figure is allocated
        res = plot_quantile_trend(mock_quantile_result, save_path=None, ax=None)
        assert res is None
        assert len(plt.get_fignums()) == 0

    def test_save_to_file_path(self, tmp_path, mock_quantile_result):
        out_file = str(tmp_path / "quantile_test.png")
        res = plot_quantile_trend(mock_quantile_result, save_path=out_file)
        assert res == out_file
        assert os.path.exists(out_file)
        assert os.path.getsize(out_file) > 1000
        assert len(plt.get_fignums()) == 0

    def test_save_to_bytes_io_buffer(self, mock_quantile_result):
        buf = io.BytesIO()
        res = plot_quantile_trend(mock_quantile_result, save_path=buf)
        assert res is buf
        assert buf.tell() > 1000
        assert len(plt.get_fignums()) == 0

    def test_caller_provided_ax(self, mock_quantile_result):
        fig, ax = plt.subplots()
        ret_ax = plot_quantile_trend(mock_quantile_result, ax=ax, save_path=None)
        assert ret_ax is ax
        plt.close(fig)
        assert len(plt.get_fignums()) == 0

    def test_figure_closes_on_save_error(self, mock_quantile_result):
        # If savefig raises an exception, the figure must still be closed cleanly in finally block
        with patch.object(plt.Figure, 'savefig', side_effect=OSError("disk full")):
            with pytest.raises(OSError, match="disk full"):
                plot_quantile_trend(mock_quantile_result, save_path='/tmp/should_fail.png')
        assert len(plt.get_fignums()) == 0


class TestPlotQuantileTrendOptions:
    """Tests for scales, colors, CIs, FDC, and transforms."""

    def test_dual_normal_scale_axes(self, mock_quantile_result):
        fig, ax = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax, x_scale='normal')

        # Check primary x-axis labels
        expected_labels = ['.001', '.01', '.05', '.10', '.25', '.50', '.75', '.90', '.95', '.99', '.999']
        actual_labels = [label.get_text() for label in ax.get_xticklabels()]
        assert actual_labels == expected_labels
        assert ax.get_xlabel() == "Non-Exceedance Probability"

        # Check secondary top axis exists and has Z labels
        children = ax.get_children()
        sec_axes = [c for c in children if isinstance(c, matplotlib.axes.Axes) or hasattr(c, 'secondary_xaxis')]
        assert len(ax.child_axes) > 0
        top_ax = ax.child_axes[0]
        assert "Standard Normal Quantile (Z)" in top_ax.get_xlabel()

        plt.close(fig)

    def test_linear_scale_axis(self, mock_quantile_result):
        fig, ax = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax, x_scale='linear')
        assert ax.get_xlabel() == "Non-Exceedance Probability"
        assert ax.get_xlim() == (-0.02, 1.02)
        plt.close(fig)

    def test_color_by_p_and_classification(self, mock_quantile_result):
        # Test color_by='p'
        fig1, ax1 = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax1, color_by='p')
        handles1, labels1 = ax1.get_legend_handles_labels()
        assert any("p < 0.05" in lbl for lbl in labels1)
        sig_colls = [c for c in ax1.collections if hasattr(c, 'get_label') and 'p < 0.05' in c.get_label()]
        assert len(sig_colls) == 1
        assert len(sig_colls[0].get_offsets()) == 5
        plt.close(fig1)

        # Test color_by='classification'
        fig2, ax2 = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax2, color_by='classification')
        handles2, labels2 = ax2.get_legend_handles_labels()
        assert any("Increasing" in lbl for lbl in labels2)
        assert any("Decreasing" in lbl for lbl in labels2)
        plt.close(fig2)

    def test_show_ci_and_fdc_overlay(self, mock_quantile_result):
        fig, ax = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax, show_ci=True, show_fdc=True)
        # Check errorbars present (ErrorbarContainer)
        assert len(ax.containers) > 0
        # Check legend has Median FDC
        handles, labels = ax.get_legend_handles_labels()
        # With twinx, FDC handles are merged into legend
        legend = ax.get_legend()
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert "Median FDC" in legend_texts
        plt.close(fig)

    def test_log_transform_axis_label(self, mock_quantile_result):
        # Create result with transform='log'
        log_res = mock_quantile_result._replace(transform='log')
        fig, ax = plt.subplots()
        plot_quantile_trend(log_res, ax=ax)
        assert "% per year" in ax.get_ylabel()
        plt.close(fig)

    def test_invalid_arguments_raise(self, mock_quantile_result):
        fig, ax = plt.subplots()
        with pytest.raises(ValueError, match="Unsupported x_scale"):
            plot_quantile_trend(mock_quantile_result, ax=ax, x_scale='invalid')

        with pytest.raises(ValueError, match="Unsupported color_by"):
            plot_quantile_trend(mock_quantile_result, ax=ax, color_by='invalid')
        plt.close(fig)

    def test_y_lim_option(self, mock_quantile_result):
        fig, ax = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax, y_lim=(-10.0, 10.0))
        assert ax.get_ylim() == (-10.0, 10.0)
        plt.close(fig)

    def test_custom_and_default_title(self, mock_quantile_result):
        fig1, ax1 = plt.subplots()
        plot_quantile_trend(mock_quantile_result, ax=ax1, title="Custom Trend Title")
        assert ax1.get_title() == "Custom Trend Title"
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        res_no_freq = mock_quantile_result._replace(block_frequency=None)
        plot_quantile_trend(res_no_freq, ax=ax2)
        assert ax2.get_title() == "Quantile Kendall Trend Analysis"
        plt.close(fig2)

    def test_plot_empty_result(self, tmp_path):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            empty_res = quantile_trend_test([1.0], pd.to_datetime(['2020-01-01']))

        fig, ax = plt.subplots()
        plot_quantile_trend(empty_res, ax=ax)
        assert any("No quantile trend results available" in t.get_text() for t in ax.texts)
        plt.close(fig)

        out_empty = str(tmp_path / "empty.png")
        plot_quantile_trend(empty_res, save_path=out_empty)
        assert os.path.exists(out_empty)
        assert os.path.getsize(out_empty) > 1000

        buf = io.BytesIO()
        plot_quantile_trend(empty_res, save_path=buf)
        assert buf.tell() > 1000

    def test_plot_dataframe_input(self):
        df_slope = pd.DataFrame({'slope': [1.0, 2.0], 'quantile': [0.25, 0.75]})
        fig, ax = plt.subplots()
        plot_quantile_trend(df_slope, ax=ax)
        assert ax.get_ylabel() == "Trend"
        plt.close(fig)

        df_pct = pd.DataFrame({'slope_pct': [5.0, 10.0], 'plotting_position': [0.25, 0.75]})
        fig2, ax2 = plt.subplots()
        plot_quantile_trend(df_pct, ax=ax2)
        assert "%" in ax2.get_ylabel()
        plt.close(fig2)

    def test_plot_invalid_result_type_raises(self):
        fig, ax = plt.subplots()
        with pytest.raises(TypeError, match="Expected result to be a QuantileTrendResult, namedtuple, or DataFrame"):
            plot_quantile_trend("invalid_string_result", ax=ax)
        plt.close(fig)


class TestPlotQuantileTrendEndToEnd:
    """End-to-end integration tests using quantile_trend_test(..., plot_path=...)."""

    def test_quantile_trend_test_with_plot_path(self, tmp_path):
        plot_file = str(tmp_path / "e2e_plot.png")
        years = 6
        dates = pd.date_range('2010-01-01', f"{2010+years-1}-12-31", freq='10D')
        rng = np.random.default_rng(42)
        vals = 10.0 + rng.normal(0, 1, len(dates))

        res = quantile_trend_test(
            vals, dates, block_frequency='YE', quantiles=5,
            min_size_per_block=20, min_blocks=5,
            plot_path=plot_file
        )
        assert os.path.exists(plot_file)
        assert os.path.getsize(plot_file) > 1000
        assert len(plt.get_fignums()) == 0
