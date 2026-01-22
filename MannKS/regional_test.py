"""
This module provides functions for regional trend aggregation, inspired by
the LWP-TRENDS R script.
"""
import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm
from collections import namedtuple


def regional_test(
    trend_results: pd.DataFrame,
    time_series_data: pd.DataFrame,
    site_col: str = 'site',
    value_col: str = 'value',
    time_col: str = 'time',
    s_col: str = 's',
    c_col: str = 'C'
) -> namedtuple:
    """
    Performs a regional trend aggregation analysis on the results of
    multiple single-site trend tests.

    Args:
        trend_results (pd.DataFrame): A DataFrame containing the results from
            running `original_test` or `seasonal_test` on multiple sites.
            Must contain columns for the site identifier, the Mann-Kendall
            score `s`, and the confidence `C`.
        time_series_data (pd.DataFrame): A DataFrame containing the original
            time series data for all sites. Must contain columns for the
            site identifier, the timestamp, and the value.
        site_col (str): The name of the site identifier column in both DataFrames.
        value_col (str): The name of the value column in `time_series_data`.
        time_col (str): The name of the time column in `time_series_data`.
        s_col (str): The name of the Mann-Kendall score column in `trend_results`.
        c_col (str): The name of the confidence column in `trend_results`.

    Returns:
        namedtuple: A namedtuple `RegionalTrendResult` with the following fields:
            - M (int): The total number of sites.
            - TAU (float): The aggregate trend strength (proportion of sites
                           trending in the modal direction).
            - VarTAU (float): The uncorrected variance of TAU.
            - CorrectedVarTAU (float): The variance of TAU corrected for
                                       inter-site correlation.
            - DT (str): The aggregate trend direction ('Increasing' or 'Decreasing').
            - CT (float): The confidence in the aggregate trend direction.

    Statistical Assumptions:
    ----------------------
    The regional trend aggregation method is based on the work of van Belle
    and Hughes (1984) and is designed to combine results from multiple sites
    while accounting for inter-site correlation.

    1.  **Homogeneity of Trend Direction**: The method is most meaningful when
        the trend directions across the sites are relatively homogeneous. It
        calculates a "modal direction" and assesses the overall trend based
        on this direction. If sites have strongly opposing trends, the
        aggregate result may not be representative.
    2.  **Pearson Correlation for Inter-site Dependence**: The correction for
        inter-site correlation is based on the pairwise Pearson correlation
        of the raw time series data. This assumes that a linear correlation
        coefficient is a reasonable measure of the dependence between sites.
    3.  **Normal Approximation for Aggregate Trend**: The final confidence in the
        aggregate trend (`CT`) is calculated using a Z-score, which assumes that
        the aggregate trend statistic (`TAU`) is approximately normally
        distributed. This assumption is more likely to hold with a larger
        number of sites.
    4.  **Validity of Single-Site Tests**: The quality of the regional analysis
        is dependent on the validity of the individual site trend tests that
        are used as input.

    *Note on Methodology*: The use of Pearson correlation on the raw (or
    temporally aggregated) time series data is a direct replication of the
    methodology used in the reference LWP-TRENDS R script. While other
    correlation methods (e.g., Spearman's Rho) might be more robust for
    non-normal or censored data, this implementation prioritizes consistency
    with the original R script's behavior.
    """
    RegionalTrendResult = namedtuple('RegionalTrendResult',
                                     ['M', 'TAU', 'VarTAU', 'CorrectedVarTAU',
                                      'DT', 'CT', 'warnings'])

    # --- 1. Input Validation ---
    required_trend_cols = [site_col, s_col, c_col]
    if not all(col in trend_results.columns for col in required_trend_cols):
        raise ValueError(f"trend_results DataFrame must contain the "
                         f"following columns: {required_trend_cols}")

    required_ts_cols = [site_col, value_col, time_col]
    if not all(col in time_series_data.columns for col in required_ts_cols):
        raise ValueError(f"time_series_data DataFrame must contain the "
                         f"following columns: {required_ts_cols}")

    # Capture warnings
    captured_warnings = []

    with warnings.catch_warnings(record=True) as w_log:
        warnings.simplefilter("always")

        # --- 2. Determine Modal Direction and Aggregate TAU ---
        results = trend_results.dropna(subset=[s_col, c_col]).copy()
        M = len(results)

        if M == 0:
            # Collect warnings
            for w in w_log:
                captured_warnings.append(str(w.message))
            return RegionalTrendResult(0, np.nan, np.nan, np.nan, 'Insufficient Data', np.nan, captured_warnings)

        # Determine the modal direction
        s_signs = np.sign(results[s_col])
        modal_direction = np.sign(np.sum(s_signs) + np.sum(s_signs == 0) / 2)
        if modal_direction == 0:
            # Collect warnings
            for w in w_log:
                captured_warnings.append(str(w.message))
            return RegionalTrendResult(M, 0.5, np.nan, np.nan,
                                    'No Clear Direction', np.nan, captured_warnings)

        DT = 'Increasing' if modal_direction == 1 else 'Decreasing'

        # Calculate TAU
        num_in_modal_direction = (np.sum(s_signs == modal_direction) +
                                np.sum(s_signs == 0) / 2)
        TAU = num_in_modal_direction / M

        # --- 3. Inter-site Covariance Calculation ---
        # Calculate the probability of being in the modal direction for each site
        results['p_modal'] = np.where(s_signs == modal_direction,
                                    results[c_col], 1 - results[c_col])

        # Uncorrected variance
        sum_var_tau = np.sum(results['p_modal'] * (1 - results['p_modal']))
        VarTAU = (1 / M**2) * sum_var_tau

        # Site alignment validation
        sites_in_trends = set(results[site_col].unique())
        sites_in_ts = set(time_series_data[site_col].unique())

        if not sites_in_trends.issubset(sites_in_ts):
            missing_sites = sites_in_trends - sites_in_ts
            raise ValueError(f"Sites in trend_results not found in time_series_data: {missing_sites}")

        # Pivot the time series data to a wide format
        ts_wide = time_series_data.pivot_table(index=time_col,
                                            columns=site_col,
                                            values=value_col)

        # Align columns with trend_results
        sites_in_common = results[site_col].unique()
        ts_wide = ts_wide[sites_in_common]

        if ts_wide.empty:
            warnings.warn("Time series pivot resulted in empty DataFrame. Check that timestamps align across sites.", UserWarning)
            # Collect warnings
            for w in w_log:
                captured_warnings.append(str(w.message))
            return RegionalTrendResult(M, TAU, VarTAU, np.nan, DT, np.nan, captured_warnings)

        # Calculate the pairwise correlation matrix
        cor_matrix = ts_wide.corr(method='pearson')

        # Calculate the covariance term
        p_modal_series = results.set_index(site_col)['p_modal']
        site_variances = p_modal_series * (1 - p_modal_series)
        cov_matrix = np.outer(np.sqrt(site_variances), np.sqrt(site_variances))

        # Align the correlation matrix with the covariance matrix
        cor_matrix = cor_matrix.reindex(index=p_modal_series.index, columns=p_modal_series.index)

        cov_term_matrix = cov_matrix * cor_matrix

        # Sum the lower triangle of the covariance term matrix
        sum_cov_term = np.sum(np.tril(cov_term_matrix, k=-1))

        # Calculate the corrected variance
        CorrectedVarTAU = (1 / M**2) * (sum_var_tau + 2 * sum_cov_term)

        # --- 4. Calculate Final Regional Trend Confidence ---
        if CorrectedVarTAU > 0:
            z_score = (TAU - 0.5) / np.sqrt(CorrectedVarTAU)
            CT = norm.cdf(z_score)
        else:
            # If variance is zero, we are 100% confident in the direction indicated by TAU
            # provided TAU is not 0.5.
            if TAU > 0.5:
                CT = 1.0
            elif TAU < 0.5:
                # If TAU < 0.5, it contradicts the modal direction definition (TAU >= 0.5),
                # but for completeness:
                CT = 0.0
            else:
                # TAU = 0.5 means no direction
                CT = 0.5

        # Collect warnings
        for w in w_log:
            captured_warnings.append(str(w.message))

    # Re-issue warnings
    for w_str in captured_warnings:
        warnings.warn(w_str, UserWarning)

    return RegionalTrendResult(M=M, TAU=TAU, VarTAU=VarTAU,
                               CorrectedVarTAU=CorrectedVarTAU, DT=DT, CT=CT, warnings=captured_warnings)
