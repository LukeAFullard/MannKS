"""
This script provides plotting utilities for the MannKS package.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
        t_normalized = t_numeric - t_numeric[0]
        seasons = (np.floor(t_normalized) % period).astype(int)

    df = pd.DataFrame({'Value': x, 'Season': seasons})

    plt.figure(figsize=(10, 6))
    # orient='v' is implied by x/y, but explicit setting helps silence some matplotlib warnings
    sns.boxplot(x='Season', y='Value', data=df, hue='Season', legend=False, orient='v')
    plt.title('Distribution of Values Across Seasons')
    plt.xlabel('Season')
    plt.ylabel('Value')

    if hasattr(plot_path, 'write'):
        plt.savefig(plot_path, format='png')
    else:
        plt.savefig(plot_path)
    plt.close()

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

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if hasattr(plot_path, 'write'):
        plt.savefig(plot_path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path

def plot_trend(data, results, save_path, alpha, seasonal_coloring=False):
    """
    Generates and saves a plot of the data with the calculated trend line.

    Input:
        data (pd.DataFrame): The DataFrame containing the data, including 'value',
                             'censored', 't', and optionally 't_original'.
        results (namedtuple): The results from original_test or seasonal_test.
        save_path (str): The file path to save the plot.
        alpha (float): The significance level for the confidence intervals.
        seasonal_coloring (bool): If True and 'season' is in data, points are
                                  colored by season. Default is False.
    """
    if save_path is None:
        return

    fig = plt.figure(figsize=(10, 6))

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
                    plt.scatter(
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
                    plt.scatter(
                        x_axis[censored_data[season_mask].index],
                        censored_data.loc[season_mask, 'value'],
                        color=colors[season],
                        marker='x'
                    )
    else:
        # Standard coloring (Blue=Observed, Red=Censored)
        censored_data = data[data['censored']]
        non_censored_data = data[~data['censored']]

        plt.scatter(x_axis[non_censored_data.index], non_censored_data['value'],
                    color='blue', label='Non-censored', marker='o')

        if 'cen_type' in data.columns:
            # Differentiate between left ('lt') and right ('gt') censored data
            lt_censored = censored_data[censored_data['cen_type'] == 'lt']
            gt_censored = censored_data[censored_data['cen_type'] == 'gt']
            other_censored = censored_data[~censored_data['cen_type'].isin(['lt', 'gt'])]

            if not lt_censored.empty:
                plt.scatter(x_axis[lt_censored.index], lt_censored['value'],
                            color='red', label='Left-Censored', marker='v')
            if not gt_censored.empty:
                plt.scatter(x_axis[gt_censored.index], gt_censored['value'],
                            color='blue', label='Right-Censored', marker='^')
            if not other_censored.empty:
                plt.scatter(x_axis[other_censored.index], other_censored['value'],
                            color='red', label='Censored (Other)', marker='x')
        else:
            plt.scatter(x_axis[censored_data.index], censored_data['value'],
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

        plt.plot(x_line, trend_line, color='black', linestyle='--', label="Sen's Slope")
        ci_label = f'{int((1 - alpha) * 100)}% CI'
        plt.fill_between(x_line, lower_line, upper_line, color='gray', alpha=0.3, label=ci_label)


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
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

    # Add annotation for Sen's slope interpretation
    annotation_text = "Trend line is the Sen's slope (median of all pairwise slopes)."
    plt.figtext(0.99, 0.01, annotation_text, ha="right", fontsize=8, style='italic')


    plt.title('Trend Analysis')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Manually set y-limits to override matplotlib's auto-scaling, which can
    # fail with the large float values of Unix timestamps on the x-axis.
    min_val = np.nanmin(data['value'])
    max_val = np.nanmax(data['value'])
    padding = (max_val - min_val) * 0.1
    plt.ylim(min_val - padding, max_val + padding)


    plt.tight_layout(rect=[0, 0.03, 1, 1])
    if hasattr(save_path, 'write'):
        plt.savefig(save_path, format='png')
    else:
        plt.savefig(save_path)
    plt.close()


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

    plt.suptitle(f"Residual Diagnostics (Slope: {results.slope:.4g} {getattr(results, 'slope_units', '')})")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if hasattr(save_path, 'write'):
        plt.savefig(save_path, format='png')
    else:
        plt.savefig(save_path)
    plt.close()
