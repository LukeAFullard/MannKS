"""
Large dataset optimizations for MannKS.

Implements fast approximations of O(n²) operations while preserving
statistical validity.

- Fast mode: Sen's slope from a random sample of pairs.  The Mann-Kendall
  test itself (S, Var(S), p) is always exact, in O(n log n).
- 'aggregate' tier (n > 50,000): as fast mode, with a warning recommending
  temporal aggregation; it does not aggregate.
- Stratified sampling: caps observations per season in very large seasonal
  data sets.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Union, Dict

# Size thresholds
SIZE_TIER_FULL = 5000      # Use full algorithms
SIZE_TIER_FAST = 50000     # Use fast approximations
SIZE_TIER_AGGREGATE = 200000  # Force aggregation

# Largest n for exact pairwise slopes: n(n-1)/2 slopes at 8 bytes each
# already need ~8.6 GB at this size.
MAX_EXACT_N = 46340

# Default sampling parameters
DEFAULT_MAX_PAIRS = 100000  # For Sen's slope
DEFAULT_BLOCK_SIZE = 1000   # For MK score windowing


def detect_size_tier(n: int,
                     user_mode: Optional[str] = None,
                     force_tier: Optional[int] = None) -> Dict:
    """
    Determine which algorithms to use based on dataset size.

    Args:
        n: Sample size
        user_mode: User override ('full', 'fast', 'aggregate', 'auto')
        force_tier: Direct tier specification (1=full, 2=fast, 3=aggregate)

    Returns:
        dict with keys:
            - tier: int (1, 2, or 3)
            - strategy: str description
            - max_pairs: int for Sen's slope
            - use_aggregation: bool
            - warnings: list of warning messages
    """
    warnings_list = []

    # User explicit override
    if user_mode == 'full':
        tier = 1
    elif user_mode == 'fast':
        tier = 2
    elif user_mode == 'aggregate':
        tier = 3
    elif force_tier is not None:
        tier = force_tier
    else:
        # Automatic detection
        if n <= SIZE_TIER_FULL:
            tier = 1
        elif n <= SIZE_TIER_FAST:
            tier = 2
            warnings_list.append(
                f"Large dataset (n={n}): Using fast approximations. "
                f"Set large_dataset_mode='full' to force exact calculations."
            )
        else:
            tier = 3
            warnings_list.append(
                f"Very large dataset (n={n}): Temporal aggregation recommended. "
                f"Results are approximations. See documentation for details."
            )

    if tier == 1 and n > MAX_EXACT_N:
        tier = 2
        warnings_list.append(
            f"large_dataset_mode='full' is not possible for n={n} (exact slopes need "
            f"all {n * (n - 1) // 2:,} pairs, over 8 GB of memory); using fast mode."
        )

    # Configure based on tier
    if tier == 1:
        return {
            'tier': 1,
            'strategy': 'full',
            'max_pairs': None,  # Use all pairs
            'use_aggregation': False,
            'warnings': warnings_list
        }
    elif tier == 2:
        return {
            'tier': 2,
            'strategy': 'fast',
            'max_pairs': DEFAULT_MAX_PAIRS,
            'use_aggregation': False,
            'warnings': warnings_list
        }
    else:  # tier == 3
        return {
            'tier': 3,
            'strategy': 'aggregate',
            'max_pairs': DEFAULT_MAX_PAIRS,
            'use_aggregation': True,
            'warnings': warnings_list
        }


def _sample_pairs(n, max_pairs, random_state):
    """Uniformly sample distinct index pairs (i < j).  A fixed default seed is
    used when ``random_state`` is None so results are reproducible."""
    rng = np.random.default_rng(0 if random_state is None else random_state)
    n_pairs = n * (n - 1) // 2
    if n_pairs <= 20 * max_pairs and n_pairs <= 20_000_000:
        # Small pool: sample distinct pairs without replacement and invert flat
        # upper-triangular indices in O(n + max_pairs) memory instead of O(n^2).
        k = np.asarray(rng.choice(n_pairs, size=min(max_pairs, n_pairs), replace=False), dtype=np.int64)
        idx = np.sort(k)
        row_starts = np.empty(n, dtype=np.int64)
        row_starts[0] = 0
        np.cumsum(np.arange(n - 1, 0, -1, dtype=np.int64), out=row_starts[1:])
        ii = np.searchsorted(row_starts, idx, side='right') - 1
        jj = ii + 1 + (idx - row_starts[ii])
        return np.asarray(ii, dtype=np.intp), np.asarray(jj, dtype=np.intp)
    i_idx = rng.integers(0, n, size=max_pairs)
    j_idx = rng.integers(0, n, size=max_pairs)
    lo = np.minimum(i_idx, j_idx).astype(np.int64)
    hi = np.maximum(i_idx, j_idx).astype(np.int64)
    keep = lo < hi
    # Distinct pairs in (lo, hi) lexicographic order, via a 1-D key.
    key = np.unique(lo[keep] * np.int64(n) + hi[keep])
    return np.asarray(key // n, dtype=np.intp), np.asarray(key % n, dtype=np.intp)


def sampled_median_error(slopes, z: float = 1.959964) -> Optional[float]:
    """
    Approximate 95% half-width of the sampling error of a median computed
    from K sampled slopes: half the distance between the order statistics at
    ranks K/2 -/+ z*sqrt(K)/2 (distribution-free binomial interval for the
    median; valid with tied or clumped slopes).
    """
    v = np.asarray(slopes, dtype=float)
    v = v[~np.isnan(v)]
    k = len(v)
    if k < 2:
        return None
    half = z * np.sqrt(k) / 2.0
    lo = int(np.clip(np.floor(k / 2.0 - half), 0, k - 1))
    hi = int(np.clip(np.ceil(k / 2.0 + half), 0, k - 1))
    v.partition([lo, hi] if lo != hi else [lo])
    return float((v[hi] - v[lo]) / 2.0)


def fast_sens_slope(x: np.ndarray,
                    t: np.ndarray,
                    max_pairs: int = DEFAULT_MAX_PAIRS,
                    random_state: Optional[int] = None) -> np.ndarray:
    """
    Estimate Sen's slope by sampling random pairs instead of all pairs.

    Sen's slope is the median of all pairwise slopes; the median of K
    uniformly sampled pairwise slopes estimates it with standard error of
    roughly IQR(slopes) / sqrt(K).

    Args:
        x: Data values
        t: Time values
        max_pairs: Maximum number of pairs to sample.  If there are no more
            possible pairs than this, all pairs are used (exact).
        random_state: Seed for reproducibility (default: a fixed seed).

    Returns:
        Array of sampled slopes.  Pairs with equal times are returned as NaN
        so that the fraction of usable pairs is preserved for rank scaling.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(x)
    if n * (n - 1) // 2 <= max_pairs:
        from ._stats import _sens_estimator_unequal_spacing
        return _sens_estimator_unequal_spacing(x, t)

    i, j = _sample_pairs(n, max_pairs, random_state)
    dt = t[j] - t[i]
    slopes = np.full(len(i), np.nan)
    ok = dt != 0.0
    slopes[ok] = (x[j][ok] - x[i][ok]) / dt[ok]
    return slopes


def fast_sens_slope_censored(x: np.ndarray,
                             t: np.ndarray,
                             cen_type: np.ndarray,
                             max_pairs: int = DEFAULT_MAX_PAIRS,
                             lt_mult: float = 0.5,
                             gt_mult: float = 1.1,
                             method: str = 'drop_ambiguous',
                             random_state: Optional[int] = None) -> np.ndarray:
    """
    Censored Sen's slopes for a random sample of pairs.

    The LWP-TRENDS censoring rules depend only on the two members of a pair,
    so they are applied to each sampled pair exactly as in the full
    calculation.  Pairs with equal times are returned as NaN.
    """
    from ._stats import _sens_estimator_censored, _censor_codes, _censored_pair_slopes
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(x)
    if n * (n - 1) // 2 <= max_pairs:
        return _sens_estimator_censored(x, t, cen_type, lt_mult=lt_mult,
                                        gt_mult=gt_mult, method=method)

    i, j = _sample_pairs(n, max_pairs, random_state)
    slopes, valid = _censored_pair_slopes(x, t, _censor_codes(cen_type), i, j,
                                          lt_mult, gt_mult, method)
    slopes[~valid] = np.nan
    return slopes


def stratified_seasonal_sampling(data: pd.DataFrame,
                                 season_col: str,
                                 max_per_season: int = 1000,
                                 random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Sample data while maintaining seasonal balance.

    Critical for seasonal trend tests where we need equal representation
    from all seasons.

    Args:
        data: DataFrame with season column
        season_col: Column name for seasons
        max_per_season: Target samples per season
        random_state: For reproducibility

    Returns:
        Stratified sample maintaining season proportions

    Theory:
        Seasonal Mann-Kendall requires S = Σ S_i where S_i is the score
        for season i. Random sampling could deplete some seasons, biasing
        the result. Stratified sampling ensures each season contributes
        proportionally.
    """
    rng = np.random.default_rng(0 if random_state is None else random_state)

    sampled_groups = []
    for season, group in data.groupby(season_col):
        n_season = len(group)
        if n_season <= max_per_season:
            sampled_groups.append(group)
        else:
            # Sample without replacement
            sample_idx = rng.choice(
                group.index,
                size=max_per_season,
                replace=False
            )
            sampled_groups.append(group.loc[sample_idx])

    return pd.concat(sampled_groups).sort_values('t', kind='stable')
