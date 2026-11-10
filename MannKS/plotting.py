"""
This script provides plotting utilities for the MannKS package.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Optional, Union, Tuple, Any
from scipy.stats import norm
from ._helpers import _preprocessing
from ._datetime import _get_season_func, _is_datetime_like, _get_cycle_identifier

def plot_seasonal_distribution(x, t, period=12, season_type='month', plot_path='seasonal_distribution.png'):
    """
    Generates and saves a box plot to visualize the distribution of values
    across different seasons.

    Input:
        x: a vector of data
        t: a vector of timestamps
        period: seasonal cycle (default 12)
        season_type: For datetime inputs, specifies the type of seasonality.
        plot_path: The file path to save the plot.
    Output:
        The file path where the plot was saved.
    """
    x_raw = np.asarray(x)
    t_raw = np.asarray(t)

    is_datetime = _is_datetime_like(t_raw)

    if is_datetime:
        season_func = _get_season_func(season_type, period)

    mask = ~np.isnan(x_raw)
    x, t = x_raw[mask], t_raw[mask]

    if len(x) < 2:
        print("Not enough data to generate a plot.")
        return None

    if is_datetime:
        seasons = season_func(pd.to_datetime(t))
    else:
        t_numeric = np.asarray(t, dtype=np.float64)
        # Normalize to start from 0 for consistent seasonal calculation
        t_normalized = t_numeric - np.min(t_numeric)
        seasons = (np.floor(t_normalized) % period).astype(np.intp)

    df = pd.DataFrame({'Value': x, 'Season': seasons})

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # orient='v' is implied by x/y, but explicit setting helps silence some matplotlib warnings
        sns.boxplot(x='Season', y='Value', data=df, hue='Season', legend=False, orient='v', ax=ax)
        ax.set_title('Distribution of Values Across Seasons')
        ax.set_xlabel('Season')
        ax.set_ylabel('Value')

        if hasattr(plot_path, 'write'):
            fig.savefig(plot_path, format='png')
        else:
            fig.savefig(plot_path)
    finally:
        plt.close(fig)

    return plot_path


def plot_inspection_data(data, plot_path, value_col, time_col, time_increment, increment_map):
    """
    Creates and saves a 2x2 grid of data inspection plots.
    """
    # 1. Validate data
    if 'censored' not in data.columns:
        raise ValueError(
            "Input data does not appear to be prepared for censored analysis. "
            "Please run `prepare_censored_data` first to add 'censored' and 'cen_type' columns."
        )

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    try:
        fig.suptitle(f"Data Inspection (Time Increment: {time_increment})", fontsize=16)

        # Plot 1: Time series plot (consistent with plot_trend)
        ax1 = axes[0, 0]
        is_datetime = _is_datetime_like(data[time_col].values)
        x_axis = pd.to_datetime(data[time_col]) if is_datetime else data[time_col]

        censored_mask = data['censored']
        ax1.scatter(x_axis[~censored_mask], data.loc[~censored_mask, value_col],
                    color='blue', label='Non-censored', marker='o', alpha=0.7)
        if censored_mask.any():
            ax1.scatter(x_axis[censored_mask], data.loc[censored_mask, value_col],
                        color='red', label='Censored', marker='x')
        ax1.set_title('Time Series Plot')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        # Get season and cycle columns for matrix plots
        season_col = increment_map.get(time_increment)

        if season_col is None or time_increment == 'none' or not is_datetime:
            plot_title = "Cannot generate matrix plots.\n"
            if not is_datetime:
                plot_title += "Matrix plots require datetime-like time column."
            else:
                plot_title += "No suitable time increment found."

            for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
                ax.text(0.5, 0.5, plot_title, ha='center', va='center', fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            plot_df = data.copy()
            # Dynamically determine the cycle (e.g., year for monthly, week for daily)
            plot_df['cycle'] = _get_cycle_identifier(plot_df[time_col], time_increment)

            # Plot 2: Value matrix
            ax2 = axes[0, 1]
            try:
                data_pivot = plot_df.pivot_table(
                    values=value_col, index='cycle', columns=season_col, aggfunc='median')
                sns.heatmap(data_pivot, cmap='viridis', ax=ax2, cbar_kws={'label': 'Median Value'})
                ax2.set_title(f'Median Value ({season_col.capitalize()} vs. Cycle)')
                ax2.set_xlabel(season_col.capitalize())
                ax2.set_ylabel('Cycle')
            except Exception as e:
                ax2.text(0.5, 0.5, f"Could not generate value matrix:\n{e}", ha='center', va='center')

            # Plot 3: Censoring matrix
            ax3 = axes[1, 0]
            try:
                cens_pivot = plot_df.pivot_table(
                    values='censored', index='cycle', columns=season_col,
                    aggfunc='any', fill_value=False)
                sns.heatmap(cens_pivot.astype(int), cmap='coolwarm', ax=ax3,
                           cbar_kws={'label': 'Censored (1=True)'}, vmin=0, vmax=1)
                ax3.set_title('Censoring Status')
                ax3.set_xlabel(season_col.capitalize())
                ax3.set_ylabel('Cycle')
            except Exception as e:
                ax3.text(0.5, 0.5, f"Could not generate censoring matrix:\n{e}", ha='center', va='center')

            # Plot 4: Sample count matrix
            ax4 = axes[1, 1]
            try:
                count_pivot = plot_df.pivot_table(
                    values=value_col, index='cycle', columns=season_col,
                    aggfunc='count', fill_value=0)
                sns.heatmap(count_pivot, cmap='Blues', ax=ax4,
                           cbar_kws={'label': 'Sample Count'}, annot=True, fmt='g')
                ax4.set_title('Sample Count')
                ax4.set_xlabel(season_col.capitalize())
                ax4.set_ylabel('Cycle')
            except Exception as e:
                ax4.text(0.5, 0.5, f"Could not generate count matrix:\n{e}", ha='center', va='center')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if hasattr(plot_path, 'write'):
            fig.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
        else:
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    finally:
        plt.close(fig)

    return plot_path

def plot_trend(data, results, save_path, alpha, seasonal_coloring=False):
    """
    Generates and saves a plot of the data with the calculated trend line.

    Input:
        data (pd.DataFrame): The DataFrame containing the data, including 'value',
                             'censored', 't', and optionally 't_original'.
        results (namedtuple): The results from trend_test or seasonal_trend_test.
        save_path (str): The file path to save the plot.
        alpha (float): The significance level for the confidence intervals.
        seasonal_coloring (bool): If True and 'season' is in data, points are
                                  colored by season. Default is False.
    """
    if save_path is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # Determine x-axis values (datetime or numeric)
        is_datetime = 't_original' in data.columns and _is_datetime_like(data['t_original'].values)
        x_axis = pd.to_datetime(data['t_original']) if is_datetime else data['t']

        # Scatter plot with potential seasonal coloring
        if seasonal_coloring and 'season' in data.columns:
            # Get unique seasons for colormap mapping
            unique_seasons = sorted(data['season'].unique())
            cmap = plt.get_cmap('tab10')
            colors = {season: cmap(i % 10) for i, season in enumerate(unique_seasons)}

            # Plot non-censored data (circles)
            non_censored_data = data[~data['censored']]
            if not non_censored_data.empty:
                for season in unique_seasons:
                    season_mask = non_censored_data['season'] == season
                    if season_mask.any():
                        ax.scatter(
                            x_axis[non_censored_data[season_mask].index],
                            non_censored_data.loc[season_mask, 'value'],
                            color=colors[season],
                            label=f'Season {season}',
                            marker='o'
                        )

            # Plot censored data (x markers)
            censored_data = data[data['censored']]
            if not censored_data.empty:
                for season in unique_seasons:
                    season_mask = censored_data['season'] == season
                    if season_mask.any():
                        # For legend clarity, we might not label every censored season individually
                        # if we already labeled the non-censored ones.
                        # But to keep it simple, we plot them with the same color scheme.
                        ax.scatter(
                            x_axis[censored_data[season_mask].index],
                            censored_data.loc[season_mask, 'value'],
                            color=colors[season],
                            marker='x'
                        )
        else:
            # Standard coloring (Blue=Observed, Red=Censored)
            censored_data = data[data['censored']]
            non_censored_data = data[~data['censored']]

            ax.scatter(x_axis[non_censored_data.index], non_censored_data['value'],
                       color='blue', label='Non-censored', marker='o')

            if 'cen_type' in data.columns:
                # Differentiate between left ('lt') and right ('gt') censored data
                lt_censored = censored_data[censored_data['cen_type'] == 'lt']
                gt_censored = censored_data[censored_data['cen_type'] == 'gt']
                other_censored = censored_data[~censored_data['cen_type'].isin(['lt', 'gt'])]

                if not lt_censored.empty:
                    ax.scatter(x_axis[lt_censored.index], lt_censored['value'],
                               color='red', label='Left-Censored', marker='v')
                if not gt_censored.empty:
                    ax.scatter(x_axis[gt_censored.index], gt_censored['value'],
                               color='blue', label='Right-Censored', marker='^')
                if not other_censored.empty:
                    ax.scatter(x_axis[other_censored.index], other_censored['value'],
                               color='red', label='Censored (Other)', marker='x')
            else:
                ax.scatter(x_axis[censored_data.index], censored_data['value'],
                           color='red', label='Censored', marker='x')

        # Trend line and confidence intervals
        if pd.notna(results.slope):
            t_numeric = data['t'].values
            t_min, t_max = t_numeric.min(), t_numeric.max()

            # ALWAYS use unscaled slope/CI for internal plotting calculations
            plot_slope = getattr(results, 'slope_per_second', results.slope)
            plot_lower_ci = getattr(results, 'lower_ci_per_second', results.lower_ci)
            plot_upper_ci = getattr(results, 'upper_ci_per_second', results.upper_ci)

            # All lines (trend and CI) must be pivoted around the median data point
            # to be correct when dealing with large timestamp values.
            ymed = np.nanmedian(data['value'])
            tmed = np.nanmedian(data['t'])

            # Correctly calculate intercept for the main trend line
            intercept_trend = ymed - plot_slope * tmed
            trend_line = plot_slope * np.array([t_min, t_max]) + intercept_trend

            # Confidence interval lines
            intercept_lower = ymed - plot_lower_ci * tmed
            intercept_upper = ymed - plot_upper_ci * tmed

            lower_line = plot_lower_ci * np.array([t_min, t_max]) + intercept_lower
            upper_line = plot_upper_ci * np.array([t_min, t_max]) + intercept_upper

            x_line = pd.to_datetime([t_min, t_max], unit='s') if is_datetime else [t_min, t_max]

            ax.plot(x_line, trend_line, color='black', linestyle='--', label="Sen's Slope")
            ci_label = f'{int((1 - alpha) * 100)}% CI'
            ax.fill_between(x_line, lower_line, upper_line, color='gray', alpha=0.3, label=ci_label)

        # Add statistics text box
        slope_str = f"{results.slope:.4f}"
        if results.slope_units:
            slope_str += f" ({results.slope_units})"

        # Use classification for the "Trend" line if available (Continuous Confidence mode),
        # otherwise fall back to the basic trend string.
        trend_display = getattr(results, 'classification', results.trend)

        stats_text = (f"Trend: {trend_display}\n"
                      f"Tau: {results.Tau:.4f}\n"
                      f"Slope: {slope_str}\n"
                      f"P-value: {results.p:.4f}")
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

        # Add annotation for Sen's slope interpretation
        annotation_text = "Trend line is the Sen's slope (median of all pairwise slopes)."
        fig.text(0.99, 0.01, annotation_text, ha="right", fontsize=8, style='italic')

        ax.set_title('Trend Analysis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

        # Manually set y-limits to override matplotlib's auto-scaling, which can
        # fail with the large float values of Unix timestamps on the x-axis.
        min_val = np.nanmin(data['value'])
        max_val = np.nanmax(data['value'])
        padding = (max_val - min_val) * 0.1
        ax.set_ylim(min_val - padding, max_val + padding)

        fig.tight_layout(rect=[0, 0.03, 1, 1])
        if hasattr(save_path, 'write'):
            fig.savefig(save_path, format='png')
        else:
            fig.savefig(save_path)
    finally:
        plt.close(fig)


def plot_residuals(data, results, save_path):
    """
    Generates and saves diagnostic plots for the residuals of the trend analysis.
    The figure contains two subplots:
    1. Residuals vs. Time (to check for patterns/autocorrelation)
    2. Histogram of Residuals (to check distribution)

    The residuals are calculated as:
    Residual = Observed Value - (Slope * Time + Intercept)
    where Intercept is defined as median(Observed Value - Slope * Time).

    Input:
        data (pd.DataFrame): The DataFrame containing the data ('value', 't', 'censored').
        results (namedtuple): The results from the trend test.
        save_path (str): The file path to save the plot.
    """
    if save_path is None:
        return

    # Use the unscaled slope (units per second/unit t) for calculation
    slope = getattr(results, 'slope_per_second', results.slope)

    if pd.isna(slope):
        print("Cannot plot residuals: Slope is NaN.")
        return

    t_numeric = data['t'].values
    values = data['value'].values

    # Calculate residuals according to user specification
    # Intercept = median(y - slope*t)
    residuals_raw = values - slope * t_numeric
    intercept = np.nanmedian(residuals_raw)
    residuals = residuals_raw - intercept

    # Prepare for plotting
    is_datetime = 't_original' in data.columns and _is_datetime_like(data['t_original'].values)
    x_axis = pd.to_datetime(data['t_original']) if is_datetime else data['t']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    try:
        # --- Subplot 1: Residuals vs Time ---
        non_censored = ~data['censored']
        censored = data['censored']

        ax1.scatter(x_axis[non_censored], residuals[non_censored],
                    color='blue', alpha=0.7, label='Residuals (Observed)')
        if censored.any():
            ax1.scatter(x_axis[censored], residuals[censored],
                        color='red', marker='x', alpha=0.7, label='Residuals (Censored)')

        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title('Residuals vs. Time')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Residual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        if is_datetime:
            plt.setp(ax1.get_xticklabels(), rotation=30, ha='right')

        # --- Subplot 2: Histogram of Residuals ---
        ax2.hist(residuals, bins='auto', color='green', alpha=0.6, edgecolor='black')
        ax2.axvline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Distribution of Residuals')
        ax2.set_xlabel('Residual')
        ax2.set_ylabel('Frequency')

        fig.suptitle(f"Residual Diagnostics (Slope: {results.slope:.4g} {getattr(results, 'slope_units', '')})")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if hasattr(save_path, 'write'):
            fig.savefig(save_path, format='png')
        else:
            fig.savefig(save_path)
    finally:
        plt.close(fig)

def plot_rolling_trend(rolling_results, data=None, time_col=None, value_col=None,
                      highlight_significant=True, show_global_trend=False,
                      global_result=None, change_points=None,
                      save_path=None, figsize=(14, 8),
                      trend_line_span='step'):
    """
    Visualize rolling trend analysis results.

    Args:
        rolling_results: DataFrame from rolling_trend_test
        data: Original data (optional, for background scatter)
        time_col, value_col: Column names in data
        highlight_significant: Color significant windows differently
        show_global_trend: Overlay single global trend line
        global_result: Result from trend_test on full dataset
        change_points: List of detected change points (optional)
        save_path: Path to save figure
        figsize: Figure size
        trend_line_span: How far each window's trend line extends on the x-axis.
            Options:
            - 'step' (default): Each line spans the window step interval centered
              around the window center, forming a clean local tangent trace without
              excessive overlap or overshooting beyond data boundaries.
            - 'window': Each line spans the full [window_start, window_end] range
              (clipped to data bounds).
            - float: Fraction of window width (e.g. 0.2 for 20% of window length).
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    try:
        # Determine if datetime based on window_center column
        is_datetime = pd.api.types.is_datetime64_any_dtype(rolling_results['window_center'])

        # --- Panel 1: Original data with rolling trend lines ---
        ax1 = axes[0]

        # Determine data time boundaries for clipping trend lines
        if data is not None and time_col and value_col:
            # Check if time column in data is datetime
            data_is_datetime = pd.api.types.is_datetime64_any_dtype(data[time_col])
            if is_datetime and not data_is_datetime:
                try:
                    x_data = pd.to_datetime(data[time_col])
                except (ValueError, TypeError):
                    x_data = data[time_col]
            else:
                x_data = data[time_col]

            ax1.scatter(x_data, data[value_col],
                       alpha=0.3, s=20, color='gray', label='Data')

            if is_datetime:
                data_ts_min = pd.to_datetime(data[time_col]).min().timestamp()
                data_ts_max = pd.to_datetime(data[time_col]).max().timestamp()
            else:
                data_ts_min = float(data[time_col].min())
                data_ts_max = float(data[time_col].max())
        else:
            if is_datetime:
                data_ts_min = pd.to_datetime(rolling_results['window_start']).min().timestamp()
                data_ts_max = pd.to_datetime(rolling_results['window_end']).max().timestamp()
            else:
                data_ts_min = float(rolling_results['window_start'].min())
                data_ts_max = float(rolling_results['window_end'].max())

        # Plot each window's trend line
        has_intercept = 'intercept' in rolling_results.columns
        has_unscaled = 'slope_per_second' in rolling_results.columns

        if is_datetime:
            ts_centers = np.array([pd.to_datetime(c).timestamp() for c in rolling_results['window_center']])
        else:
            ts_centers = np.array(rolling_results['window_center'], dtype=float)

        n_windows = len(rolling_results)

        for idx, (row_idx, row) in enumerate(rolling_results.iterrows()):
            # Skip failed windows (NaN slope)
            if pd.isna(row['slope']):
                continue

            if has_intercept and has_unscaled and pd.notna(row['intercept']):
                t_start = row['window_start']
                t_end = row['window_end']
                ts_start = pd.to_datetime(t_start).timestamp() if is_datetime else float(t_start)
                ts_end = pd.to_datetime(t_end).timestamp() if is_datetime else float(t_end)
                ts_cen = ts_centers[idx]

                if trend_line_span == 'window':
                    seg_start = ts_start
                    seg_end = ts_end
                elif isinstance(trend_line_span, (int, float)):
                    half_w = float(trend_line_span) * (ts_end - ts_start) / 2.0
                    seg_start = ts_cen - half_w
                    seg_end = ts_cen + half_w
                else:  # default: 'step'
                    if n_windows > 1:
                        if idx == 0:
                            half_step = (ts_centers[1] - ts_centers[0]) / 2.0
                            seg_start = ts_cen - half_step
                            seg_end = ts_cen + half_step
                        elif idx == n_windows - 1:
                            half_step = (ts_centers[-1] - ts_centers[-2]) / 2.0
                            seg_start = ts_cen - half_step
                            seg_end = ts_cen + half_step
                        else:
                            seg_start = (ts_centers[idx - 1] + ts_cen) / 2.0
                            seg_end = (ts_cen + ts_centers[idx + 1]) / 2.0
                    else:
                        seg_start = ts_start
                        seg_end = ts_end

                # Clip segment to actual data bounds to prevent overshooting into future/past
                seg_start = max(seg_start, data_ts_min)
                seg_end = min(seg_end, data_ts_max)

                if seg_end <= seg_start:
                    continue

                x_num = np.array([seg_start, seg_end])
                slope_raw = row['slope_per_second']
                y_vals = slope_raw * x_num + row['intercept']

                if is_datetime:
                    x_plot_seg = [pd.to_datetime(seg_start, unit='s'), pd.to_datetime(seg_end, unit='s')]
                else:
                    x_plot_seg = [seg_start, seg_end]

                color = 'red' if (highlight_significant and row.get('h', False)) else 'black'
                alpha_line = 0.8 if (highlight_significant and row.get('h', False)) else 0.4
                width = 2.0 if (highlight_significant and row.get('h', False)) else 1.2

                ax1.plot(x_plot_seg, y_vals, color=color, alpha=alpha_line, linewidth=width)

        if show_global_trend and global_result:
            if hasattr(global_result, 'intercept') and pd.notna(global_result.intercept):
                g_slope = getattr(global_result, 'slope_per_second', global_result.slope)
                x_num_g = np.array([data_ts_min, data_ts_max])
                y_g = g_slope * x_num_g + global_result.intercept

                if is_datetime:
                    x_g = [pd.to_datetime(data_ts_min, unit='s'), pd.to_datetime(data_ts_max, unit='s')]
                else:
                    x_g = [data_ts_min, data_ts_max]

                ax1.plot(x_g, y_g, color='blue', linestyle='--', linewidth=2, label='Global Trend')


        ax1.set_ylabel('Value')
        ax1.set_title('Original Data with Rolling Trend Lines')
        if data is not None:
            ax1.legend()
        ax1.grid(True, alpha=0.3)

        # --- Panel 2: Rolling slope with confidence intervals ---
        ax2 = axes[1]

        x_plot = rolling_results['window_center']

        # Color by significance (h)
        if highlight_significant and 'h' in rolling_results.columns:
            sig_mask = rolling_results['h'].astype(bool)
            if sig_mask.any():
                ax2.scatter(x_plot[sig_mask], rolling_results.loc[sig_mask, 'slope'],
                           color='red', s=30, label='Significant', zorder=3)
            if (~sig_mask).any():
                ax2.scatter(x_plot[~sig_mask], rolling_results.loc[~sig_mask, 'slope'],
                           color='gray', s=30, alpha=0.5, label='Not significant', zorder=3)
        else:
            ax2.scatter(x_plot, rolling_results['slope'], s=30, zorder=3)

        # Confidence intervals
        if 'lower_ci' in rolling_results.columns and 'upper_ci' in rolling_results.columns:
            ax2.fill_between(x_plot,
                             rolling_results['lower_ci'],
                             rolling_results['upper_ci'],
                             alpha=0.2, color='blue', label='95% CI')

        # Zero line
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)

        # Change points
        if change_points:
            for cp in change_points:
                ax2.axvline(cp, color='orange', linestyle=':', linewidth=2, alpha=0.7)

        ax2.set_ylabel('Slope')
        ax2.set_title('Rolling Sen\'s Slope')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # --- Panel 3: P-values and confidence ---
        ax3 = axes[2]
        ax3_twin = ax3.twinx()

        # P-values (log scale)
        if 'p_value' in rolling_results.columns:
            # Handle p=0 for log scale
            p_safe = rolling_results['p_value'].replace(0, 1e-10)
            ax3.semilogy(x_plot, p_safe,
                        color='purple', marker='o', linestyle='', label='P-value', markersize=3, alpha=0.6)
            ax3.axhline(0.05, color='red', linestyle='--', linewidth=1, label='α=0.05')

        ax3.set_ylabel('P-value', color='purple')
        ax3.tick_params(axis='y', labelcolor='purple')

        # Confidence in direction
        if 'C' in rolling_results.columns:
            ax3_twin.plot(x_plot, rolling_results['C'],
                         color='green', marker='s', linestyle='', label='Confidence', markersize=3, alpha=0.6)
            ax3_twin.set_ylabel('Confidence (C)', color='green')
            ax3_twin.tick_params(axis='y', labelcolor='green')
            ax3_twin.set_ylim([0.4, 1.0])

        ax3.set_xlabel('Time (Window Center)')
        ax3.set_title('Statistical Significance & Confidence')
        ax3.grid(True, alpha=0.3)

        # Format x-axis for datetime
        if is_datetime:
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                # Locator depends on data span, letting auto handle it is usually safest
                # ax.xaxis.set_major_locator(mdates.YearLocator())

        fig.tight_layout()

        if save_path:
            if hasattr(save_path, 'write'):
                fig.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            else:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    finally:
        plt.close(fig)

def _create_segments(t, breakpoints):
    """Internal helper to split time vector into segments"""
    t_min, t_max = np.min(t), np.max(t)
    sorted_bp = np.sort(breakpoints)
    segments = []

    if len(sorted_bp) == 0:
        return [(t_min, t_max)]

    segments.append((t_min, sorted_bp[0]))
    for i in range(len(sorted_bp) - 1):
        segments.append((sorted_bp[i], sorted_bp[i+1]))
    segments.append((sorted_bp[-1], t_max))

    return segments

def plot_segmented_trend(result, x_data, t_data, save_path=None):
    """
    Visualizes the segmented trend analysis results, including confidence intervals.

    Args:
        result: The namedtuple returned by segmented_trend_test.
        x_data: Original data values, or the DataFrame from
            ``prepare_censored_data`` (censored values are marked).
        t_data: Original time values.
        save_path: Path to save the plot.
    """
    if save_path is None:
        return

    censored_mask = None
    if isinstance(x_data, pd.DataFrame):
        if 'censored' in x_data.columns:
            censored_mask = x_data['censored'].to_numpy(dtype=bool)
        x_data = x_data['value'].to_numpy(dtype=float) if 'value' in x_data.columns \
            else x_data.iloc[:, 0].to_numpy(dtype=float)
    else:
        x_data = np.asarray(x_data, dtype=float)

    from ._datetime import _to_numeric_time

    # Determine CI label from alpha if available
    alpha_val = getattr(result, 'alpha', 0.05)
    ci_percent = int((1 - alpha_val) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        # Prepare data for plotting
        t_numeric = _to_numeric_time(t_data)
        is_datetime = result.is_datetime

        x_axis = pd.to_datetime(t_data) if is_datetime else t_data

        # Plot raw data
        if censored_mask is not None and censored_mask.any():
            x_axis_arr = np.asarray(x_axis)
            ax.scatter(x_axis_arr[~censored_mask], x_data[~censored_mask], color='gray', alpha=0.5,
                       label='Observations')
            ax.scatter(x_axis_arr[censored_mask], x_data[censored_mask], color='red', marker='v',
                       alpha=0.6, label='Censored (limit)')
        else:
            ax.scatter(x_axis, x_data, color='gray', alpha=0.5, label='Observations')

        # Re-create segments to get boundaries
        if is_datetime:
            bp_numeric = _to_numeric_time(result.breakpoints)
        else:
            bp_numeric = result.breakpoints

        segments = _create_segments(t_numeric, bp_numeric)

        # Iterate through segments dataframe in result
        for i, (start, end) in enumerate(segments):
            if i >= len(result.segments):
                break

            row = result.segments.iloc[i]

            # Calculate trend line
            slope = getattr(row, 'slope_per_second', row.slope)
            intercept = row.intercept

            if pd.isna(slope) or pd.isna(intercept):
                continue

            # Define x-range for segment.
            # Segments are plotted individually to respect the discontinuous nature of the piecewise Sen's slope.
            x_seg_numeric = np.array([start, end])
            y_seg = slope * x_seg_numeric + intercept

            if is_datetime:
                x_seg_plot = pd.to_datetime(x_seg_numeric, unit='s')
            else:
                x_seg_plot = x_seg_numeric

            ax.plot(x_seg_plot, y_seg, linewidth=2, label=f'Segment {i+1}')

            # Plot Slope CIs (if available)
            if hasattr(row, 'lower_ci_per_second') and pd.notna(row.lower_ci_per_second):
                # Calculate CI band.
                # To plot the band correctly, we need the pivot point (t_center, y_center)
                # of the segment data, as the intercepts are defined relative to this point.
                # We re-calculate the pivot point for the segment data.

                is_last = i == len(segments) - 1
                mask = (t_numeric >= start) & ((t_numeric <= end) if is_last else (t_numeric < end))
                if np.sum(mask) > 0:
                    t_seg = t_numeric[mask]
                    x_seg = x_data[mask]

                    # Ensure data is numeric for median calculation.
                    if np.issubdtype(np.asarray(x_seg).dtype, np.number):
                        t_center = np.median(t_seg)
                        x_center = np.nanmedian(x_seg)

                        slope_lower = row.lower_ci_per_second
                        slope_upper = row.upper_ci_per_second

                        # CI lines passing through (t_center, x_center)
                        y_lower = x_center + slope_lower * (x_seg_numeric - t_center)
                        y_upper = x_center + slope_upper * (x_seg_numeric - t_center)

                        ax.fill_between(x_seg_plot, y_lower, y_upper, alpha=0.2, label=f'{ci_percent}% CI (Seg {i+1})')


        # Plot Breakpoints and their CIs
        for i, bp in enumerate(result.breakpoints):
            ax.axvline(bp, color='black', linestyle='--', label='Breakpoint')

            # Plot Breakpoint CI if available
            if hasattr(result, 'breakpoint_cis') and i < len(result.breakpoint_cis):
                ci_low, ci_high = result.breakpoint_cis[i]
                # Add shaded region
                if pd.notna(ci_low) and pd.notna(ci_high):
                    ax.axvspan(ci_low, ci_high, color='orange', alpha=0.3, label=f'Breakpoint {ci_percent}% CI')

        ax.set_title('Segmented Trend Analysis')
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')

        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        if hasattr(save_path, 'write'):
            fig.savefig(save_path, format='png')
        else:
            fig.savefig(save_path)
    finally:
        plt.close(fig)


def plot_quantile_trend(
    result: Any,
    save_path: Optional[Union[str, Any]] = None,
    x_scale: str = 'normal',
    color_by: str = 'p',
    show_ci: bool = False,
    show_fdc: bool = False,
    y_lim: Optional[Tuple[float, float]] = None,
    legend_loc: Optional[str] = 'upper left',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[plt.Axes] = None,
    dpi: int = 150,
) -> Optional[Union[str, plt.Axes]]:
    """
    Plot generalized Quantile-Kendall trend results across quantiles.

    Visualizes trend slope (and optional confidence intervals) as a function of
    non-exceedance probability (quantile) on either standard normal Z-scale or
    linear scale, with color coding by significance (p-value) or trend classification,
    and optional Flow Duration Curve (FDC) overlay.

    Parameters
    ----------
    result : QuantileTrendResult, namedtuple, or pd.DataFrame
        The result object returned by `quantile_trend_test` or a DataFrame
        containing quantile trend results.
    save_path : str, Path, buffer, or None, default None
        File path or writable buffer where the plot should be saved.
        If None and ax is None, no figure is created and None is returned.
    x_scale : {'normal', 'linear'}, default 'normal'
        X-axis coordinate scaling:
        - 'normal': Quantiles mapped to standard normal quantiles (Z-scale) with
          primary tick marks labeled with standard probabilities (.001 to .999)
          and a secondary top x-axis displaying standard normal Z values [-3, 3].
        - 'linear': Quantiles plotted linearly against non-exceedance probability [0, 1].
    color_by : {'p', 'classification'}, default 'p'
        Color coding for plotted points:
        - 'p': Colored by significance: p < alpha (red/highlight),
          alpha <= p < 2*alpha (charcoal/intermediate), and p >= 2*alpha (gray).
        - 'classification': Colored by MannKS trend classification category:
          Increasing (blue/navy), Decreasing (red/orange), and No Trend / Indeterminate (gray).
    show_ci : bool, default False
        Whether to display vertical error whiskers (confidence intervals) for each quantile.
    show_fdc : bool, default False
        Whether to overlay the median Flow Duration Curve (FDC) across blocks
        on a secondary right-hand y-axis (`twinx`).
    y_lim : tuple of (float, float), optional
        Manual (ymin, ymax) limits for the primary trend slope y-axis.
    legend_loc : str or None, default 'upper left'
        Location for the legend (e.g. 'upper left', 'best', etc.). If None or 'none',
        legend is omitted.
    title : str, optional
        Custom title for the plot. If None, an informative default title is generated.
    figsize : tuple of (float, float), default (10, 6)
        Dimensions of the figure in inches when creating a new figure.
    ax : plt.Axes, optional
        Existing matplotlib Axes object to draw onto. If provided, `fig = ax.figure`.
    dpi : int, default 150
        Resolution in dots per inch for saved figure files.

    Returns
    -------
    save_path or ax or None
        Returns save_path if save_path is provided, ax if ax is provided and save_path is None,
        or None if save_path is None and ax is None.
    """
    if save_path is None and ax is None:
        return None

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    try:
        # Extract DataFrame and metadata from result
        if hasattr(result, 'results'):
            df = result.results
        elif isinstance(result, pd.DataFrame):
            df = result
        elif hasattr(result, 'to_frame'):
            df = result.to_frame()
        else:
            raise TypeError("Expected result to be a QuantileTrendResult, namedtuple, or DataFrame.")

        if df is None or df.empty:
            ax.text(
                0.5, 0.5, "No quantile trend results available",
                ha='center', va='center', transform=ax.transAxes
            )
            if save_path is not None:
                if hasattr(save_path, 'write'):
                    fig.savefig(save_path, format='png', dpi=dpi, bbox_inches='tight')
                else:
                    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                return save_path
            return ax

        if x_scale not in ('normal', 'linear'):
            raise ValueError(f"Unsupported x_scale: '{x_scale}'. Must be 'normal' or 'linear'.")
        if color_by not in ('p', 'classification'):
            raise ValueError(f"Unsupported color_by: '{color_by}'. Must be 'p' or 'classification'.")

        alpha_val = float(getattr(result, 'alpha', 0.05) or 0.05)
        transform = getattr(result, 'transform', 'none')

        # Determine y values, CIs, and percentage formatting
        is_log = (transform == 'log')
        if is_log and 'slope_pct' in df.columns and not df['slope_pct'].isna().all():
            y = df['slope_pct'].to_numpy(dtype=float)
            ci_low = df['lower_ci_pct'].to_numpy(dtype=float) if 'lower_ci_pct' in df.columns else None
            ci_high = df['upper_ci_pct'].to_numpy(dtype=float) if 'upper_ci_pct' in df.columns else None
            is_pct = True
        elif 'slope' in df.columns:
            y = df['slope'].to_numpy(dtype=float)
            ci_low = df['lower_ci'].to_numpy(dtype=float) if 'lower_ci' in df.columns else None
            ci_high = df['upper_ci'].to_numpy(dtype=float) if 'upper_ci' in df.columns else None
            is_pct = False
        elif 'slope_pct' in df.columns:
            y = df['slope_pct'].to_numpy(dtype=float)
            ci_low = df['lower_ci_pct'].to_numpy(dtype=float) if 'lower_ci_pct' in df.columns else None
            ci_high = df['upper_ci_pct'].to_numpy(dtype=float) if 'upper_ci_pct' in df.columns else None
            is_pct = True
        else:
            raise ValueError("Input result DataFrame must contain 'slope' or 'slope_pct' column.")

        # Determine x coordinates
        if x_scale == 'normal':
            if 'z_position' in df.columns and not df['z_position'].isna().all():
                x = df['z_position'].to_numpy(dtype=float)
            elif 'plotting_position' in df.columns and not df['plotting_position'].isna().all():
                x = norm.ppf(df['plotting_position'].to_numpy(dtype=float))
            elif 'quantile' in df.columns and not df['quantile'].isna().all():
                x = norm.ppf(df['quantile'].to_numpy(dtype=float))
            else:
                raise ValueError("Input result DataFrame must contain 'z_position', 'plotting_position', or 'quantile'.")
        else:  # 'linear'
            if 'plotting_position' in df.columns and not df['plotting_position'].isna().all():
                x = df['plotting_position'].to_numpy(dtype=float)
            elif 'quantile' in df.columns and not df['quantile'].isna().all():
                x = df['quantile'].to_numpy(dtype=float)
            else:
                raise ValueError("Input result DataFrame must contain 'plotting_position' or 'quantile'.")

        # Horizontal baseline at y = 0
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

        # Confidence intervals (error bars)
        if show_ci and ci_low is not None and ci_high is not None:
            yerr_low = np.where(np.isnan(ci_low) | np.isnan(y), 0.0, np.maximum(0.0, y - ci_low))
            yerr_high = np.where(np.isnan(ci_high) | np.isnan(y), 0.0, np.maximum(0.0, ci_high - y))
            valid_ci = (~np.isnan(y)) & ((~np.isnan(ci_low)) | (~np.isnan(ci_high)))
            if np.any(valid_ci):
                ax.errorbar(
                    x[valid_ci], y[valid_ci],
                    yerr=[yerr_low[valid_ci], yerr_high[valid_ci]],
                    fmt='none', ecolor='gray', elinewidth=1.0, capsize=2, alpha=0.6,
                    zorder=2
                )

        # Color coding points
        if color_by == 'p':
            p_vals = df['p'].to_numpy(dtype=float) if 'p' in df.columns else np.full(len(df), np.nan)
            mask_sig = (~np.isnan(p_vals)) & (p_vals < alpha_val)
            mask_mid = (~np.isnan(p_vals)) & (p_vals >= alpha_val) & (p_vals < 2.0 * alpha_val)
            mask_ns = np.isnan(p_vals) | (p_vals >= 2.0 * alpha_val)

            if np.any(mask_sig):
                ax.scatter(
                    x[mask_sig], y[mask_sig],
                    color='#d95f02', s=35, zorder=5,
                    label=f"p < {alpha_val:g}"
                )
            if np.any(mask_mid):
                ax.scatter(
                    x[mask_mid], y[mask_mid],
                    color='#252525', s=35, zorder=4,
                    label=f"{alpha_val:g} <= p < {2.0*alpha_val:g}"
                )
            if np.any(mask_ns):
                ax.scatter(
                    x[mask_ns], y[mask_ns],
                    color='#7f7f7f', s=35, zorder=3,
                    label=f"p >= {2.0*alpha_val:g}"
                )
        else:  # color_by == 'classification'
            if 'classification' in df.columns:
                cat_series = df['classification'].astype(str)
            elif 'trend' in df.columns:
                cat_series = df['trend'].astype(str)
            else:
                cat_series = pd.Series(['No Trend'] * len(df))

            mask_inc = cat_series.str.contains('increasing', case=False, na=False).to_numpy()
            mask_dec = cat_series.str.contains('decreasing', case=False, na=False).to_numpy()
            mask_other = ~(mask_inc | mask_dec)

            if np.any(mask_inc):
                ax.scatter(
                    x[mask_inc], y[mask_inc],
                    color='#1f78b4', s=35, zorder=5,
                    label='Increasing'
                )
            if np.any(mask_dec):
                ax.scatter(
                    x[mask_dec], y[mask_dec],
                    color='#e31a1c', s=35, zorder=4,
                    label='Decreasing'
                )
            if np.any(mask_other):
                ax.scatter(
                    x[mask_other], y[mask_other],
                    color='#7f7f7f', s=35, zorder=3,
                    label='No Trend / Indeterminate'
                )

        # X-axis configuration
        if x_scale == 'normal':
            prob_ticks = [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999]
            prob_labels = ['.001', '.01', '.05', '.10', '.25', '.50', '.75', '.90', '.95', '.99', '.999']
            z_ticks = [float(norm.ppf(p)) for p in prob_ticks]

            min_x = min(z_ticks[0], np.nanmin(x) if len(x) > 0 and not np.isnan(x).all() else z_ticks[0])
            max_x = max(z_ticks[-1], np.nanmax(x) if len(x) > 0 and not np.isnan(x).all() else z_ticks[-1])
            margin = max(0.15, (max_x - min_x) * 0.03)
            ax.set_xlim(min_x - margin, max_x + margin)

            ax.set_xticks(z_ticks)
            ax.set_xticklabels(prob_labels)
            ax.set_xlabel("Non-Exceedance Probability")

            secax = ax.secondary_xaxis('top')
            secax.set_xlabel("Standard Normal Quantile (Z)")
            secax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            secax.set_xticklabels(['-3', '-2', '-1', '0', '1', '2', '3'])
        else:  # 'linear'
            ax.set_xlabel("Non-Exceedance Probability")
            ax.set_xlim(-0.02, 1.02)

        # Y-axis configuration
        if is_pct:
            scale_unit = getattr(result, 'slope_scaling', None)
            if not scale_unit and 'slope_units' in df.columns and len(df) > 0 and pd.notna(df['slope_units'].iloc[0]):
                u_str = str(df['slope_units'].iloc[0])
                if 'per ' in u_str:
                    scale_unit = u_str.split('per ')[-1].strip()
            if not scale_unit:
                scale_unit = 'year'
            y_label = f"Trend (% per {scale_unit})"
        else:
            u_str = ""
            if 'slope_units' in df.columns and len(df) > 0 and pd.notna(df['slope_units'].iloc[0]):
                u_str = str(df['slope_units'].iloc[0]).strip()
            if u_str:
                y_label = f"Trend ({u_str})"
            else:
                y_label = "Trend"
        ax.set_ylabel(y_label)

        if y_lim is not None:
            ax.set_ylim(y_lim)

        ax.grid(True, linestyle=':', alpha=0.6)

        # Optional Flow Duration Curve overlay
        ax_fdc = None
        if show_fdc:
            matrix = getattr(result, 'matrix', None)
            if matrix is not None and hasattr(matrix, 'median') and not getattr(matrix, 'empty', True):
                median_curve = matrix.median(axis=0).to_numpy(dtype=float)
                if len(median_curve) == len(x):
                    ax_fdc = ax.twinx()
                    ax_fdc.plot(
                        x, median_curve,
                        color='navy', linestyle=':', linewidth=1.5, alpha=0.75,
                        label='Median FDC'
                    )
                    unit_str = getattr(result, 'x_unit', None) or getattr(result, 'units', None) or ''
                    if unit_str:
                        ax_fdc.set_ylabel(f"Median Concentration / Flow ({unit_str})", color='navy')
                    else:
                        ax_fdc.set_ylabel("Median Concentration / Flow", color='navy')
                    ax_fdc.tick_params(axis='y', labelcolor='navy')
                    ax_fdc.grid(False)

        # Title
        if title is not None:
            ax.set_title(title)
        else:
            freq = getattr(result, 'block_frequency', None)
            if freq:
                ax.set_title(f"Quantile Kendall Trend Analysis ({freq} blocks)")
            else:
                ax.set_title("Quantile Kendall Trend Analysis")

        # Legend
        if legend_loc is not None and str(legend_loc).lower() not in ('none', 'na', 'false'):
            handles, labels = ax.get_legend_handles_labels()
            if ax_fdc is not None:
                fdc_handles, fdc_labels = ax_fdc.get_legend_handles_labels()
                handles = handles + fdc_handles
                labels = labels + fdc_labels
            if handles:
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc=legend_loc, frameon=True)

        if save_path is not None:
            if hasattr(save_path, 'write'):
                fig.savefig(save_path, format='png', dpi=dpi, bbox_inches='tight')
            else:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            return save_path
        return ax
    finally:
        if created_fig or save_path is not None:
            plt.close(fig)


plot_quantile_kendall = plot_quantile_trend
