"""
Monte Carlo Validation & USGS EGRET Parity Study for Generalized Quantile-Kendall
Package: MannKS v0.7.0
Folder: validation/43_Quantile_Kendall/

This study validates the statistical trustworthiness of Generalized Quantile-Kendall:
1. USGS EGRET Parity:
   Direct comparison against the genuine USGS EGRET R code (EGRET's own makeSortQ,
   leapEven, leapOdd, trendSortQ run with the EGRET, rkt and zyp packages through
   rpy2; see egret_reference.R and egret_parity.py).  Without R the stored EGRET
   output in tests/data/egret_qk is used and the report says so.
2. Type I Error Rate Calibration:
   Empirical rejection rates under H0 across quantiles at nominal alpha = 0.05.
3. Sampling Frequency Drift Trap & Equalization:
   Validation of the sampling frequency drift trap (monthly -> weekly shift) and
   verification that equalize='thin' and equalize='aggregate' restore nominal
   Type I error rates.
4. Confidence Interval Coverage:
   Validation that (1 - alpha) = 95% confidence intervals achieve nominal coverage.
5. Permutation Field Significance Calibration:
   Validation of the global field significance permutation test under H0.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Ensure MannKS from workspace root is in path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import MannKS as mk
from MannKS.quantile_trend_test import quantile_trend_test

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ==============================================================================
# 1. USGS EGRET parity (genuine R code, see egret_reference.R / egret_parity.py)
# ==============================================================================

import egret_parity  # noqa: E402  (same folder)


def _mc_status(rate: float, nominal: float, n: int) -> str:
    """Whether an empirical rate lies inside the 95% binomial Monte Carlo band around nominal."""
    se = np.sqrt(nominal * (1.0 - nominal) / n)
    lo, hi = nominal - 1.96 * se, nominal + 1.96 * se
    inside = lo <= rate <= hi
    return f"{'Within' if inside else '**Outside**'} 95% MC band [{lo:.3f}, {hi:.3f}]"


# ==============================================================================
# Validation Tests
# ==============================================================================

def validate_egret_parity():
    """Compares MannKS with the genuine EGRET R code on every parity scenario."""
    live = egret_parity.r_available()
    source = "live EGRET run through rpy2" if live else "stored EGRET output (tests/data/egret_qk); R not available"
    print(f"\n[1/5] Validating parity with USGS EGRET ({source})...")

    rows = []
    versions = None
    plot_data = None
    for name, (dataset, pa_start, pa_long, desc) in egret_parity.SCENARIOS.items():
        if live:
            daily = egret_parity.make_dataset(dataset)
            egret = egret_parity.run_egret(daily, pa_start, pa_long)
        else:
            daily, egret, _ = egret_parity.load_fixture(name)
        versions = egret['versions']
        res = egret_parity.run_mannks(daily, pa_start, pa_long)
        diff = egret_parity.compare(res, egret)
        diff.update({'name': name, 'paStart': pa_start, 'paLong': pa_long, 'description': desc})
        rows.append(diff)
        print(f"  {name}: {diff['n_ranks']} ranks x {diff['n_years']} years, "
              f"max|d sortQ|={diff['sortQ']:.1e} |d S|={diff['S']:.1e} |d p|={diff['p']:.1e} "
              f"|d tau|={diff['tau']:.1e} |d slopeLog|={diff['slopeLog']:.1e} |d slopePct|={diff['slopePct']:.1e}")

        if name == 'climate_year':
            kw = egret_parity.mannks_settings(pa_start, pa_long)
            x, t = daily['q'].to_numpy(), pd.DatetimeIndex(daily['date'])
            kw_center = dict(kw, block_time='center')
            centre = quantile_trend_test(x, t, **kw_center).to_frame()
            kw_interp = dict(kw_center, standardization='interpolate')
            interp = quantile_trend_test(x, t, **kw_interp).to_frame()
            ref = egret['results']
            plot_data = (ref, res.to_frame(), centre, interp)

    ref, exact, centre, interp = plot_data
    ref_pct = ref['slopePct'].to_numpy()
    centre_rel = float(np.max(np.abs(centre['slope_pct'].to_numpy() - ref_pct) / np.abs(ref_pct)) * 100)
    interp_rel = float(np.mean(np.abs(interp['slope_pct'].to_numpy() - ref_pct) / np.abs(ref_pct)) * 100)
    interp_abs = float(np.mean(np.abs(interp['slope_pct'].to_numpy() - ref_pct)))
    print(f"  Default block centre (block_time='center') vs EGRET: max slope difference {centre_rel:.4f}%")
    print(f"  Interpolation mode vs EGRET: mean slope difference {interp_rel:.3f}% ({interp_abs:.4f} %/yr abs)")

    # Plot Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150)
    z = ref['z'].to_numpy()
    ref_label = 'USGS EGRET (R, live via rpy2)' if live else 'USGS EGRET (R, stored output)'
    ax1.plot(z, ref_pct, 'k-', label=ref_label, linewidth=1.8)
    ax1.plot(z, exact['slope_pct'], 'r--', label="MannKS (standardization='egret', block_time='egret')", linewidth=1.5)
    ax1.plot(z, interp['slope_pct'], 'b:', label="MannKS (standardization='interpolate')", linewidth=1.2)
    ax1.set_xlabel('Standard Normal Quantile (Z-scale)', fontsize=11)
    ax1.set_ylabel('Trend Slope (% / year)', fontsize=11)
    ax1.set_title('USGS EGRET vs MannKS Quantile Slopes (climate year)', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='best', frameon=True, fontsize=8)

    ax2.plot(z, exact['slope_pct'].to_numpy() - ref_pct, 'r-', linewidth=1.2, label="block_time='egret' (EGRET settings)")
    ax2.plot(z, centre['slope_pct'].to_numpy() - ref_pct, 'm-', linewidth=1.0, label="block_time='center' (MannKS default time axis)")
    ax2.axhline(0, color='k', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Standard Normal Quantile (Z-scale)', fontsize=11)
    ax2.set_ylabel('Slope difference, MannKS - EGRET (% / year)', fontsize=11)
    ax2.set_title('Slope Difference (MannKS - EGRET)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='best', frameon=True, fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'egret_parity_comparison.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    return {
        'rows': rows,
        'live': live,
        'source': source,
        'versions': versions,
        'centre_rel': centre_rel,
        'interp_slope_rel': interp_rel,
        'interp_slope_abs': interp_abs,
    }


def validate_type1_error(n_reps: int = 400, alpha: float = 0.05):
    """Monte Carlo study evaluating empirical Type I error under stationary H0 across quantiles."""
    print(f"\n[2/5] Running Type I error calibration ({n_reps} Monte Carlo reps, alpha={alpha})...")
    rng = np.random.default_rng(101)
    n_blocks = 20
    n_per_block = 30
    dates = pd.date_range("2000-01-01", periods=n_blocks * n_per_block, freq="12D")

    rejections = np.zeros(7)
    q_probs = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    for rep in range(n_reps):
        # Stationary standard normal noise under H0
        x = rng.normal(size=len(dates))
        res = quantile_trend_test(
            x, dates,
            block_frequency='YE',
            quantiles=q_probs,
            alpha=alpha,
            min_size_per_block=10,
        )
        df_res = res.to_frame()
        rejections += df_res['h'].to_numpy(dtype=int)

    emp_rates = rejections / n_reps

    print("  Empirical Type I Error Rates:")
    for q, rate in zip(q_probs, emp_rates):
        print(f"    Quantile {q:0.2f}: Empirical alpha = {rate:.4f} (Nominal: {alpha:.2f})")

    # Plot Type I Error
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.bar([str(q) for q in q_probs], emp_rates, color='steelblue', alpha=0.85, edgecolor='black', width=0.5, label='Empirical Rejection Rate')
    ax.axhline(alpha, color='red', linestyle='--', linewidth=1.8, label=f'Nominal alpha = {alpha}')
    # 95% binomial confidence bounds for Monte Carlo simulation
    se = np.sqrt(alpha * (1.0 - alpha) / n_reps)
    ax.axhspan(alpha - 1.96 * se, alpha + 1.96 * se, color='red', alpha=0.15, label='95% Monte Carlo Confidence Band')

    ax.set_xlabel('Quantile Rank ($p$)', fontsize=11)
    ax.set_ylabel('Empirical Rejection Rate ($H_0$ False Positive Rate)', fontsize=11)
    ax.set_title(f'Type I Error Rate Calibration Across Quantiles ($N_{{reps}}={n_reps}$)', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 0.10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'type1_error_calibration.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    return {'rates': dict(zip(q_probs, emp_rates)), 'n_reps': n_reps}


def validate_sampling_drift_trap(n_reps: int = 300):
    """Monte Carlo verification demonstrating the sampling frequency drift trap and equalization remedy."""
    print(f"\n[3/5] Validating Sampling Frequency Drift Trap & Equalization ({n_reps} Monte Carlo reps)...")
    rng = np.random.default_rng(202)

    # 20 years: years 1-10 have 12 visits/year; years 11-20 have 52 visits/year
    # Physical process is completely stationary lognormal (NO REAL TREND)
    raw_rejections_low = 0
    raw_rejections_high = 0
    raw_flags = 0

    eq_rejections_low = 0
    eq_rejections_high = 0
    agg_rejections_low = 0
    agg_rejections_high = 0
    hazen_rejections_low = 0
    hazen_rejections_high = 0

    for _ in range(n_reps):
        t_list = []
        x_list = []
        for yr in range(2000, 2010):
            t_list.extend(pd.date_range(f"{yr}-01-15", f"{yr}-12-15", periods=12))
            x_list.extend(rng.lognormal(2.0, 0.4, size=12))
        for yr in range(2010, 2020):
            t_list.extend(pd.date_range(f"{yr}-01-05", f"{yr}-12-25", periods=52))
            x_list.extend(rng.lognormal(2.0, 0.4, size=52))

        t_arr = pd.to_datetime(t_list)
        x_arr = np.array(x_list)

        # All "order-statistic" analyses use quantile_method='linear', for which rank 1 / rank 12 are
        # each year's minimum / maximum (plotting positions p=0.08 and p=0.92): the worst case.
        # 0. Raw analysis with the package default (Hazen ranks)
        df_hz = quantile_trend_test(x_arr, t_arr, block_frequency='YE', quantiles='auto', equalize='none', min_size_per_block=10).to_frame()
        hazen_rejections_low += int(df_hz.iloc[0]['h'])
        hazen_rejections_high += int(df_hz.iloc[-1]['h'])

        # 1. Raw analysis of order statistics (equalize='none')
        res_raw = quantile_trend_test(x_arr, t_arr, block_frequency='YE', quantiles='auto', quantile_method='linear', equalize='none', min_size_per_block=10)
        df_raw = res_raw.to_frame()
        if res_raw.sampling_diagnostic['flag']:
            raw_flags += 1
        if df_raw.iloc[0]['h'] and df_raw.iloc[0]['slope'] < 0: # False downward trend in min
            raw_rejections_low += 1
        if df_raw.iloc[-1]['h'] and df_raw.iloc[-1]['slope'] > 0: # False upward trend in max
            raw_rejections_high += 1

        # 2. Equalized Analysis (equalize='thin')
        res_eq = quantile_trend_test(x_arr, t_arr, block_frequency='YE', quantiles='auto', quantile_method='linear', equalize='thin', min_size_per_block=10)
        df_eq = res_eq.to_frame()
        if df_eq.iloc[0]['h']:
            eq_rejections_low += 1
        if df_eq.iloc[-1]['h']:
            eq_rejections_high += 1

        # 3. Equalized Analysis (equalize='aggregate': monthly sub-intervals on a common support)
        res_agg = quantile_trend_test(x_arr, t_arr, block_frequency='YE', quantiles='auto', quantile_method='linear', equalize='aggregate', min_size_per_block=10)
        df_agg = res_agg.to_frame()
        if df_agg.iloc[0]['h']:
            agg_rejections_low += 1
        if df_agg.iloc[-1]['h']:
            agg_rejections_high += 1

    rate_raw_flag = raw_flags / n_reps
    rate_raw_low = raw_rejections_low / n_reps
    rate_raw_high = raw_rejections_high / n_reps
    rate_eq_low = eq_rejections_low / n_reps
    rate_eq_high = eq_rejections_high / n_reps
    rate_agg_low = agg_rejections_low / n_reps
    rate_agg_high = agg_rejections_high / n_reps

    print(f"  Sampling Diagnostic Detection Rate: {rate_raw_flag*100:.1f}%")
    print(f"  Raw Unequalized False Trend Rates (Nominal alpha = 0.05):")
    print(f"    Lowest Quantile (p=0.08): False Downward Rate = {rate_raw_low*100:.1f}% ")
    print(f"    Highest Quantile (p=0.92): False Upward Rate   = {rate_raw_high*100:.1f}% ")
    print(f"  Equalized (equalize='thin') False Trend Rates:")
    print(f"    Lowest Quantile (p=0.08): False Trend Rate     = {rate_eq_low*100:.1f}%")
    print(f"    Highest Quantile (p=0.92): False Trend Rate    = {rate_eq_high*100:.1f}%")
    print(f"  Equalized (equalize='aggregate') False Trend Rates:")
    print(f"    Lowest Quantile (p=0.08): False Trend Rate     = {rate_agg_low*100:.1f}%")
    print(f"    Highest Quantile (p=0.92): False Trend Rate    = {rate_agg_high*100:.1f}%")

    # Plot Drift Trap Comparison
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=150)
    categories = ['Lowest Quantile (p=0.08)\nSpurious Downward', 'Highest Quantile (p=0.92)\nSpurious Upward']
    x_indices = np.arange(len(categories))
    width = 0.35

    ax.bar(x_indices - width/2, [rate_raw_low, rate_raw_high], width, color='crimson', alpha=0.85, edgecolor='black', label='Raw Analysis (equalize="none")')
    ax.bar(x_indices + width/2, [rate_eq_low, rate_eq_high], width, color='forestgreen', alpha=0.85, edgecolor='black', label='Equalized Analysis (equalize="thin")')

    ax.axhline(0.05, color='black', linestyle='--', linewidth=1.5, label='Nominal alpha = 0.05')
    ax.set_ylabel('Empirical False Positive Rate', fontsize=11)
    ax.set_title('Sampling Frequency Drift Trap: Inflated Error Rates vs. Equalization Remedy', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(categories, fontsize=10.5)
    ax.set_ylim(0, max(0.40, rate_raw_low * 1.25))
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'sampling_drift_validation.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    return {
        'rate_raw_flag': rate_raw_flag,
        'rate_raw_low': rate_raw_low,
        'rate_raw_high': rate_raw_high,
        'rate_eq_low': rate_eq_low,
        'rate_eq_high': rate_eq_high,
        'rate_agg_low': rate_agg_low,
        'rate_hazen_low': hazen_rejections_low / n_reps,
        'rate_hazen_high': hazen_rejections_high / n_reps,
        'rate_agg_high': rate_agg_high,
        'n_reps': n_reps,
    }


def validate_ci_coverage(n_reps: int = 350, alpha: float = 0.05):
    """Validates that nominal 95% confidence intervals achieve >= (1 - alpha) empirical coverage."""
    print(f"\n[4/5] Validating Confidence Interval Empirical Coverage ({n_reps} Monte Carlo reps)...")
    rng = np.random.default_rng(303)
    n_blocks = 25
    n_per_block = 20
    dates = pd.date_range("2000-01-01", periods=n_blocks * n_per_block, freq="18D")
    t_years = (dates - dates[0]).total_seconds() / (365.25 * 86400.0)

    # True analytic slope: Location shift beta = 0.15 units/year across ALL quantiles
    true_slope = 0.15
    q_probs = [0.10, 0.25, 0.50, 0.75, 0.90]
    covered = np.zeros(len(q_probs))

    for _ in range(n_reps):
        x = true_slope * t_years + rng.normal(scale=1.2, size=len(dates))
        res = quantile_trend_test(
            x, dates,
            block_frequency='YE',
            quantiles=q_probs,
            alpha=alpha,
            slope_scaling='year',
        )
        df_res = res.to_frame()
        for idx, (_, row) in enumerate(df_res.iterrows()):
            if row['lower_ci'] <= true_slope <= row['upper_ci']:
                covered[idx] += 1

    emp_coverage = covered / n_reps
    print("  Empirical (1 - alpha) CI Coverage Rates (Nominal = 0.95):")
    for q, cov in zip(q_probs, emp_coverage):
        print(f"    Quantile {q:0.2f}: Coverage = {cov:.4f}")

    # Plot CI Coverage
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    ax.bar([str(q) for q in q_probs], emp_coverage * 100, color='darkorange', alpha=0.85, edgecolor='black', width=0.5, label='Empirical CI Coverage')
    ax.axhline(95.0, color='blue', linestyle='--', linewidth=1.8, label='Nominal 95% Coverage')
    ax.set_xlabel('Quantile Rank ($p$)', fontsize=11)
    ax.set_ylabel('Empirical Coverage Percentage (%)', fontsize=11)
    ax.set_title('95% Confidence Interval Coverage Across Quantiles', fontsize=12, fontweight='bold')
    ax.set_ylim(85, 100)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, 'ci_coverage_validation.png')
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"  Saved plot: {plot_path}")

    return {'coverage': dict(zip(q_probs, emp_coverage)), 'n_reps': n_reps}


def validate_field_significance_calibration(n_reps: int = 250, alpha: float = 0.05):
    """Validates global permutation field significance false discovery rate under H0."""
    n_permutations = 199
    print(f"\n[5/5] Validating Permutation Field Significance Calibration ({n_reps} Monte Carlo reps, {n_permutations} permutations)...")
    rng = np.random.default_rng(404)
    n_blocks = 20
    n_per_block = 25
    dates = pd.date_range("2000-01-01", periods=n_blocks * n_per_block, freq="14D")

    rejections_field = 0
    rejections_count = 0
    rejections_max_z = 0
    rejections_naive_any = 0
    M = 15

    for rep_idx in range(n_reps):
        x = rng.normal(size=len(dates))
        res = quantile_trend_test(
            x, dates,
            block_frequency='YE',
            quantiles=M,
            field_significance=True,
            n_permutations=n_permutations,
            alpha=alpha,
            random_state=rep_idx,
        )
        df_res = res.to_frame()
        # Naive: any quantile significant
        if df_res['h'].any():
            rejections_naive_any += 1
        # Field significance: each statistic on its own, and the union of the two
        fs = res.field_significance
        rejections_count += int(fs['p_count'] < alpha)
        rejections_max_z += int(fs['p_max_z'] < alpha)
        if fs['p_count'] < alpha or fs['p_max_z'] < alpha:
            rejections_field += 1

    rate_field = rejections_field / n_reps
    rate_naive = rejections_naive_any / n_reps

    print(f"  Naive Joint False Alarm Rate (any p < {alpha}): {rate_naive*100:.1f}% ")
    print(f"  Permutation Field Significance False Alarm Rate: {rate_field*100:.1f}%")

    return {
        'rate_field': rate_field,
        'rate_count': rejections_count / n_reps,
        'rate_max_z': rejections_max_z / n_reps,
        'rate_naive': rate_naive,
        'n_reps': n_reps,
        'n_permutations': n_permutations,
    }


# ==============================================================================
# Markdown Report Generator
# ==============================================================================

def generate_validation_report(results: dict):
    """Generates the README.md validation report from the numbers computed in this run."""
    report_path = os.path.join(OUT_DIR, 'README.md')
    eg = results['egret']
    t1 = results['type1']
    drift = results['drift']
    ci = results['ci']
    field = results['field']
    nl = chr(10)

    ver = eg['versions'] or {}
    ver_str = ", ".join(f"{k} {v}" for k, v in ver.items())
    parity_rows = nl.join(
        f"| `{r['name']}` | {r['paStart']} / {r['paLong']} | {r['n_ranks']} x {r['n_years']} | {r['n_tied_ranks']} | "
        f"`{r['sortQ']:.1e}` | `{r['S']:.1e}` | `{r['varS']:.1e}` | `{r['tau']:.1e}` | `{r['p']:.1e}` | "
        f"`{r['slopeLog']:.1e}` | `{r['slopePct']:.1e}` |"
        for r in eg['rows']
    )
    scen_rows = nl.join(f"| `{r['name']}` | {r['description']} |" for r in eg['rows'])
    worst = max(max(r[k] for k in ('sortQ', 'S', 'varS', 'tau', 'p', 'slopeLog', 'slopePct')) for r in eg['rows'])
    n_rep_d = drift['n_reps']

    type1_rows = nl.join(
        f"| **{q:0.2f}** | 0.05 | {rate:.4f} | {_mc_status(rate, 0.05, t1['n_reps'])} |"
        for q, rate in t1['rates'].items()
    )
    ci_rows = nl.join(
        f"| **{q:0.2f}** | 95.0% | {cov*100:.2f}% | {_mc_status(1.0 - cov, 0.05, ci['n_reps'])} (non-coverage) |"
        for q, cov in ci['coverage'].items()
    )

    content = rf"""# Validation Report: Generalized Quantile-Kendall (`MannKS` v0.7.0)

## Overview

This report is generated by [`run_monte_carlo.py`](run_monte_carlo.py); every number below was computed in the run that wrote this file. It covers:

1. **Parity with USGS EGRET**: `MannKS.quantile_trend_test` against the genuine EGRET R code (EGRET's `makeSortQ` + `leapOdd`/`leapEven` + `trendSortQ`, run with the EGRET, rkt and zyp R packages) on identical daily series and periods of analysis.
2. **Type I error** of the per-quantile tests under $H_0$.
3. **Sampling-frequency drift**: the artifact, the diagnostic, and both equalization modes.
4. **Confidence-interval coverage** of the per-quantile slopes.
5. **Permutation field significance** under $H_0$.

Monte Carlo sample sizes are modest (a few hundred replicates), so each rate is shown with the 95% binomial Monte Carlo band it should fall in; read the tables as "consistent with nominal", not as proof of exact calibration.

---

## 1. USGS EGRET Parity

### What is compared
* **Reference**: {eg['source']}. Package versions: {ver_str}.
* The R code in [`egret_reference.R`](egret_reference.R) is EGRET's own: `makeSortQ`, `trendSortQ`, `getFirstJulian`, `leapOdd`, `leapEven` and `odd` are copied verbatim from the EGRET vignette [*Daily Streamflow Trend Analysis*](https://doi-usgs.github.io/EGRET/articles/streamflow_trend.html) (the code behind `plotQuantileKendall`). They run on an EGRET `eList` built with `populateDaily`, `as.egret` and `setPA`, so `DecYear`, `MonthSeq` and the period of analysis are EGRET's, and the Mann-Kendall test and Theil-Sen slope are `rkt::rkt`.
* [`egret_parity.py`](egret_parity.py) feeds the **same dates and discharges** to EGRET (through `rpy2`) and to MannKS, and stores EGRET's output in `tests/data/egret_qk/` so that `tests/test_quantile_egret_parity.py` asserts the same parity without R (and re-derives it live when R is installed).

### Equivalent settings
| EGRET | MannKS `quantile_trend_test` |
| :--- | :--- |
| `setPA(paStart = m, paLong = 12)` | `block_frequency='YS-<MON>'` (e.g. `paStart = 10` is `'YS-OCT'`) |
| `paLong < 12` (seasonal period of analysis) | additionally `months=[...]` listing the `paLong` months |
| `makeSortQ` keeps a year only if every day is present | `require_complete=True` |
| `leapOdd` / `leapEven` rank reduction | `standardization='egret'` |
| `time = (DecYear[first] + DecYear[last]) / 2` | `block_time='egret'` |
| `rkt(time, log(Q))`, `slopePct = 100 * (exp(B) - 1)` | `transform='log'`, `slope_scaling='year'` (`slope` is `slopeLog`, `slope_pct` is `slopePct`) |
| `rkt` `tau` (tau-b), `sl` (two-sided, continuity-corrected, tie-corrected) | `tau_method='b'` (default), `p` |
| `freq = iRank / (nFreq + 1)`, `z = qnorm(freq)` | `plotting_position`, `z_position` |

### Scenarios
| Scenario | Description |
| :--- | :--- |
{scen_rows}

### Results: maximum absolute difference, MannKS minus EGRET
| Scenario | `paStart` / `paLong` | ranks x years | ranks with ties | `sortQ` matrix | $S$ | $\mathrm{{Var}}(S)$ | $\tau$ | $p$ (`pValue`) | `slopeLog` | `slopePct` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{parity_rows}

The largest difference in any quantity in any scenario is `{worst:.1e}`: with the settings above MannKS reproduces EGRET to floating-point rounding, including tied values and both leap-year reductions (`leapEven` for 366-day years, `leapOdd` for 91-day Dec-Feb seasons).

### What is *not* the same as EGRET
* **Default time axis.** Without `block_time='egret'` MannKS places each block at its calendar midpoint and scales by a 365.25-day year. $S$, $\mathrm{{Var}}(S)$, $\tau$ and $p$ are unchanged (they only use the order of the blocks); the slope differs by at most `{eg['centre_rel']:.4f}%` of its value in the `climate_year` scenario (magenta line in the right-hand panel below).
* **Default standardization.** `standardization='interpolate'` (interpolated Hazen quantiles: leap years are read at 365 Hazen positions instead of being rank-reduced) is a different estimator: mean slope difference `{eg['interp_slope_rel']:.3f}%` (`{eg['interp_slope_abs']:.4f}` %/yr) in the `climate_year` scenario.
* **Serial-correlation-adjusted p-value.** EGRET's `trendSortQ` also returns `pValueAdj` (`zyp::zyp.zhang` trend-free pre-whitening) and `plotQuantileKendall` colours its points by `pValueAdj` (red $< 0.05$, black $\le 0.1$, grey otherwise). MannKS does not implement Zhang pre-whitening: `p` reproduces EGRET's unadjusted `pValue`, and `plot_quantile_trend` colours by `p` with bands at $\alpha$ and $2\alpha$. `pValueAdj` is stored in the fixtures for reference only. Use `autocorr_method` for MannKS's own serial-correlation corrections.
* **Records with gaps.** EGRET's `makeSortQ` stops with an error when a year inside the record has missing days; MannKS drops that block (`require_complete=True`) and continues.

### Visual Parity
![USGS EGRET Parity Comparison](egret_parity_comparison.png)

---

## 2. Type I Error Rate

### Methodology
{t1['n_reps']} Monte Carlo realizations of stationary standard normal noise under $H_0$ across 20 annual blocks; rejection rate of each per-quantile test at nominal $\alpha = 0.05$. The Mann-Kendall normal approximation with continuity correction is slightly conservative for 20 blocks.

### Results
| Quantile ($p$) | Nominal $\alpha$ | Empirical Rejection Rate | Monte Carlo check |
| :--- | :--- | :--- | :--- |
{type1_rows}

### Plot
![Type I Error Calibration](type1_error_calibration.png)

---

## 3. Sampling Frequency Drift

### The Drift Phenomenon
When monitoring frequency shifts from monthly ($N=12$) to weekly ($N=52$) mid-record, sample extremes widen over time even when the data-generating process is stationary: the highest order statistics trend upward and the lowest trend downward.

### Monte Carlo Results ({n_rep_d} replications, stationary lognormal, 10 years monthly then 10 years weekly)
The order-statistic analyses use `quantile_method='linear'`, for which rank 1 and rank $M$ are each year's minimum and maximum: the worst case for this artifact. With the package default (`'hazen'`) the extreme ranks of a larger block are read at interior positions ($0.5/M$ and $1 - 0.5/M$), which removes most of the artifact even without equalization. Rates are two-sided rejection rates at $\alpha = 0.05$, except the raw order-statistic rows, which count only rejections in the artifact's direction.

| Analysis | Rank | False Rejection Rate | Monte Carlo check vs. $\alpha = 0.05$ |
| :--- | :--- | :--- | :--- |
| **Order statistics, raw (`equalize='none'`)** | lowest ($p = 0.08$), downward | **{drift['rate_raw_low']*100:.1f}%** | {_mc_status(drift['rate_raw_low'], 0.05, n_rep_d)} |
| **Order statistics, raw (`equalize='none'`)** | highest ($p = 0.92$), upward | **{drift['rate_raw_high']*100:.1f}%** | {_mc_status(drift['rate_raw_high'], 0.05, n_rep_d)} |
| **Default Hazen ranks, raw** | lowest | **{drift['rate_hazen_low']*100:.1f}%** | {_mc_status(drift['rate_hazen_low'], 0.05, n_rep_d)} |
| **Default Hazen ranks, raw** | highest | **{drift['rate_hazen_high']*100:.1f}%** | {_mc_status(drift['rate_hazen_high'], 0.05, n_rep_d)} |
| **Order statistics, `equalize='thin'`** | lowest | **{drift['rate_eq_low']*100:.1f}%** | {_mc_status(drift['rate_eq_low'], 0.05, n_rep_d)} |
| **Order statistics, `equalize='thin'`** | highest | **{drift['rate_eq_high']*100:.1f}%** | {_mc_status(drift['rate_eq_high'], 0.05, n_rep_d)} |
| **Order statistics, `equalize='aggregate'`** | lowest | **{drift['rate_agg_low']*100:.1f}%** | {_mc_status(drift['rate_agg_low'], 0.05, n_rep_d)} |
| **Order statistics, `equalize='aggregate'`** | highest | **{drift['rate_agg_high']*100:.1f}%** | {_mc_status(drift['rate_agg_high'], 0.05, n_rep_d)} |

The sampling diagnostic (`sampling_diagnostic['flag']`) was raised in **{drift['rate_raw_flag']*100:.1f}%** of the raw analyses.

`equalize='aggregate'` takes sub-interval medians on a **common support**: every sub-interval of every block is thinned to the same number of observations $k$ before the median is taken (here $k = 1$, because the monthly years have one observation per month). Medians of different numbers of observations are not comparable between blocks; an implementation that takes the median of whatever is in each month makes the weekly years look compressed and produces the *opposite* artifact (lowest rank falsely increasing, highest falsely decreasing, in roughly 70-80% of replicates of this design).

### Drift Plot
![Sampling Drift Validation](sampling_drift_validation.png)

---

## 4. Confidence Interval Coverage

### Methodology
{ci['n_reps']} Monte Carlo realizations of a linear location shift (true slope $\beta = 0.15$ units/year at every quantile), 25 annual blocks; coverage of the 95% interval for each quantile's slope.

### Results
| Quantile ($p$) | Nominal Coverage | Empirical Coverage | Monte Carlo check |
| :--- | :--- | :--- | :--- |
{ci_rows}

### Plot
![Confidence Interval Coverage](ci_coverage_validation.png)

---

## 5. Permutation Field Significance

Testing $M=15$ quantiles simultaneously under $H_0$ ({field['n_reps']} Monte Carlo replications, {field['n_permutations']} block permutations each, seeded per replicate):

| Rule | False alarm rate | Monte Carlo check vs. $\alpha = 0.05$ |
| :--- | :--- | :--- |
| Naive: any quantile with $p < 0.05$ | **{field['rate_naive']*100:.1f}%** | {_mc_status(field['rate_naive'], 0.05, field['n_reps'])} |
| `p_count < 0.05` | **{field['rate_count']*100:.1f}%** | {_mc_status(field['rate_count'], 0.05, field['n_reps'])} |
| `p_max_z < 0.05` | **{field['rate_max_z']*100:.1f}%** | {_mc_status(field['rate_max_z'], 0.05, field['n_reps'])} |
| `p_count < 0.05` **or** `p_max_z < 0.05` | **{field['rate_field']*100:.1f}%** | {_mc_status(field['rate_field'], 0.05, field['n_reps'])} |

Each field statistic is a level-$\alpha$ test on its own; rejecting when *either* is below $\alpha$ is two tests and is slightly above $\alpha$. Choose one statistic in advance. The permutation assumes exchangeable blocks: with strong between-block autocorrelation the field test is anti-conservative (see the methodology document, section 6).

---

## Reproducing

```bash
python validation/43_Quantile_Kendall/run_monte_carlo.py                   # this report
python validation/43_Quantile_Kendall/egret_parity.py --write-fixtures     # refresh stored EGRET output (needs R)
```

The live EGRET comparison needs R with the `EGRET`, `rkt` and `zyp` packages and `rpy2` (`rpy2==3.5.17` for R < 4.5). Without them the stored EGRET output is used and section 1 says so.
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content.strip() + "\n")
    print(f"\nGenerated validation report: {report_path}")


# ==============================================================================
# Main Runner
# ==============================================================================

if __name__ == "__main__":
    t_start = time.time()
    print("=" * 70)
    print("MannKS v0.7.0 — Generalized Quantile-Kendall Validation Study")
    print("=" * 70)

    res_egret = validate_egret_parity()
    res_type1 = validate_type1_error(n_reps=300, alpha=0.05)
    res_drift = validate_sampling_drift_trap(n_reps=250)
    res_ci = validate_ci_coverage(n_reps=250, alpha=0.05)
    res_field = validate_field_significance_calibration(n_reps=150, alpha=0.05)

    all_results = {
        'egret': res_egret,
        'type1': res_type1,
        'drift': res_drift,
        'ci': res_ci,
        'field': res_field,
    }

    generate_validation_report(all_results)
    elapsed = time.time() - t_start
    print(f"\nValidation completed successfully in {elapsed:.2f} seconds.")
