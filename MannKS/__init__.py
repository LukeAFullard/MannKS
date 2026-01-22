"""
MannKS: A Python package for non-parametric trend analysis on unequally spaced time series data.

This package provides implementations of the Mann-Kendall test and Sen's slope
estimator, with additional support for seasonal analysis, seasonality testing,
and plotting utilities.
"""
from .trend_test import trend_test
from .seasonal_trend_test import seasonal_trend_test
from .check_seasonality import check_seasonality
from .plotting import plot_seasonal_distribution, plot_rolling_trend, plot_segmented_trend
from .inspection import inspect_trend_data
from .regional_test import regional_test
from .classification import classify_trend
from .preprocessing import prepare_censored_data
from ._bootstrap import block_bootstrap_mann_kendall, block_bootstrap_confidence_intervals
from .rolling_trend import rolling_trend_test, compare_periods
from .segmented_trend_test import (
    segmented_trend_test,
    SegmentedTrendResult,
    find_best_segmentation,
    calculate_breakpoint_probability
)

__all__ = [
    'trend_test',
    'seasonal_trend_test',
    'check_seasonality',
    'plot_seasonal_distribution',
    'plot_rolling_trend',
    'plot_segmented_trend',
    'inspect_trend_data',
    'regional_test',
    'classify_trend',
    'prepare_censored_data',
    'block_bootstrap_mann_kendall',
    'block_bootstrap_confidence_intervals',
    'rolling_trend_test',
    'compare_periods',
    'segmented_trend_test',
    'SegmentedTrendResult',
    'find_best_segmentation',
    'calculate_breakpoint_probability'
]

__version__ = "0.5.0"
