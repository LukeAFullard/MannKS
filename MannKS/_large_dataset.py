"""
Large dataset optimizations for MannKS.

Implements fast approximations of O(n²) operations while preserving
statistical validity.

Features (v0.5.0):
- Fast Mode: Stochastic Sen's slope and O(N log N) Mann-Kendall score.
- Aggregate Mode: Automatic temporal aggregation for massive datasets.
- Stratified Sampling: Preserves seasonal structure in large seasonal data.
"""

import numpy as np
import pandas as pd
import warnings
from typing import Tuple, Optional, Union, Dict

# Size thresholds
SIZE_TIER_FULL = 5000      # Use full algorithms
SIZE_TIER_FAST = 50000     # Use fast approximations
SIZE_TIER_AGGREGATE = 200000  # Force aggregation

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
        # Small pool: sample distinct pairs without replacement.
        k = rng.choice(n_pairs, size=min(max_pairs, n_pairs), replace=False)
        ii, jj = np.triu_indices(n, k=1)
        return ii[np.sort(k)], jj[np.sort(k)]
    i_idx = rng.integers(0, n, size=max_pairs)
    j_idx = rng.integers(0, n, size=max_pairs)
    lo = np.minimum(i_idx, j_idx)
    hi = np.maximum(i_idx, j_idx)
    pairs = np.unique(np.column_stack([lo, hi]), axis=0)
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    return pairs[:, 0], pairs[:, 1]


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
    ok = np.abs(dt) > 1e-10
    slopes[ok] = (x[j][ok] - x[i][ok]) / dt[ok]
    return slopes


def fast_sens_slope_censored(x: np.ndarray,
                             t: np.ndarray,
                             cen_type: np.ndarray,
                             max_pairs: int = DEFAULT_MAX_PAIRS,
                             lt_mult: float = 0.5,
                             gt_mult: float = 1.1,
                             method: str = 'unbiased',
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
