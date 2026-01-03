import numpy as np
import pandas as pd
import pytest
from MannKS import rolling_trend_test

def test_rolling_trend_offset_fallback():
    """
    Test that when window is a DateOffset (e.g., '1M') and step is None,
    the step defaults to the window size (non-overlapping), unlike Timedelta
    which defaults to window/2.
    """
    n = 60
    t = pd.date_range('2000-01-01', periods=n, freq='D')
    x = np.arange(n)

    # Case 1: Window is Timedelta-compatible string (e.g., '10D')
    # Default step should be window/2 = 5D
    results_td = rolling_trend_test(x, t, window='10D', min_size=5)

    # Check step size by looking at window starts
    starts_td = pd.to_datetime(results_td['window_start'])
    diffs_td = starts_td.diff().dropna()
    # Should be mostly 5 days
    assert (diffs_td == pd.Timedelta('5D')).all()

    # Case 2: Window is Offset string (e.g., '1ME' for month end)
    # Default step should be window size (1 month) because Offsets aren't divisible
    # Note: '1ME' is variable length, but roughly 1 month.
    # We need enough data. 60 days is approx 2 months.
    # Let's use '10D' but force it to be treated as an offset?
    # pd.tseries.frequencies.to_offset('10D') returns a FixedOffset/Day which is also a Timedelta?
    # No, '10D' parses as Timedelta first in the code.

    # Use '1ME' (Month End) which fails Timedelta parsing
    # We need more data for monthly windows
    t_long = pd.date_range('2000-01-01', periods=365*2, freq='D')
    x_long = np.arange(len(t_long))

    results_offset = rolling_trend_test(x_long, t_long, window='1ME', min_size=5)

    starts_offset = pd.to_datetime(results_offset['window_start'])
    diffs_offset = starts_offset.diff().dropna()

    # The step should be 1 Month. The diff in days will vary (28, 29, 30, 31).
    # But it should NOT be 0.5 months (~15 days).
    assert (diffs_offset > pd.Timedelta('27D')).all()

    # Check that windows are contiguous (end of one is start of next approx)
    # Actually code does: win_end = current + window_size
    # step = window_size
    # next start = current + step = current + window_size
    # So next start == current end.

    # Let's verify overlap.
    # If step == window, overlap is 0.
    # If step == window/2, overlap is 50%.

    # For Timedelta case ('10D'):
    # Window [0, 10), [5, 15). Overlap [5, 10).
    # Check first two windows
    w1_end_td = results_td.iloc[0]['window_end']
    w2_start_td = results_td.iloc[1]['window_start']
    assert w2_start_td < w1_end_td # Overlap exists

    # For Offset case ('1ME'):
    # Window 1: [Start, Start + 1ME)
    # Step: 1ME
    # Window 2: [Start + 1ME, Start + 1ME + 1ME)
    # Start 2 should equal End 1
    w1_end_off = results_offset.iloc[0]['window_end']
    w2_start_off = results_offset.iloc[1]['window_start']

    assert w2_start_off == w1_end_off # No overlap
