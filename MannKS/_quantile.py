"""
Core quantile blocking, length standardization, and sampling diagnostics
for Generalized Quantile-Kendall Analysis (MannKS v0.7.0).
"""
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm, kendalltau
from ._datetime import _is_datetime_like, _to_numeric_time
from ._stats import _rank_of_zero

try:
    from pandas.tseries.offsets import to_offset
except ImportError:
    try:
        from pandas.tseries.frequencies import to_offset
    except ImportError:
        to_offset = None

def _parse_block_frequency(block_frequency, is_datetime):
    if is_datetime:
        if to_offset is None:
            raise ImportError("Could not import to_offset from pandas.")
        if block_frequency is None:
            raise ValueError(f"Invalid block_frequency '{block_frequency}' for datetime data. Use a pandas frequency string (e.g., 'YE', '5YE', 'QE').")
        try:
            offset = to_offset(block_frequency)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid block_frequency '{block_frequency}' for datetime data. Use a pandas frequency string (e.g., 'YE', '5YE', 'QE').")
        if offset is None:
            raise ValueError(f"Invalid block_frequency '{block_frequency}' for datetime data. Use a pandas frequency string (e.g., 'YE', '5YE', 'QE').")
        probe = pd.Timestamp('2000-01-01')
        try:
            if (probe + offset) <= probe:
                raise ValueError(
                    f"Invalid block_frequency '{block_frequency}' for datetime data. Frequency step must be positive."
                )
        except TypeError:
            raise ValueError(f"Invalid block_frequency '{block_frequency}' for datetime data.")
        return offset
    else:
        if isinstance(block_frequency, (bool, np.bool_)):
            raise ValueError(f"Invalid block_frequency '{block_frequency}' for numeric data. Must be a positive number (finite).")
        try:
            width = float(block_frequency)
            if not np.isfinite(width) or width <= 0:
                raise ValueError
            return width
        except (ValueError, TypeError):
            raise ValueError(f"Invalid block_frequency '{block_frequency}' for numeric data. Must be a positive number (finite).")

def _normalize_start_freq(offset_or_str) -> str:
    s_raw = str(offset_or_str).strip()
    try:
        fs = getattr(offset_or_str, 'freqstr', None)
        if fs:
            s_raw = str(fs).strip()
        else:
            rc = getattr(offset_or_str, 'rule_code', None)
            if rc:
                s_raw = str(rc).strip()
    except Exception:
        pass
    s = s_raw.upper()
    end_month_map = {
        'JAN': 'FEB', 'FEB': 'MAR', 'MAR': 'APR', 'APR': 'MAY',
        'MAY': 'JUN', 'JUN': 'JUL', 'JUL': 'AUG', 'AUG': 'SEP',
        'SEP': 'OCT', 'OCT': 'NOV', 'NOV': 'DEC', 'DEC': 'JAN',
    }
    if 'YE-' in s:
        parts = s.split('YE-')
        prefix = parts[0]
        m_end = parts[1][:3]
        m_start = end_month_map.get(m_end, 'JAN')
        return f"{prefix}YS-{m_start}"
    if s.endswith('YE') or s == 'Y' or s == 'A':
        prefix = s[:-2] if s.endswith('YE') else ''
        return f"{prefix}YS"
    if 'QE-' in s:
        parts = s.split('QE-')
        prefix = parts[0]
        m_end = parts[1][:3]
        m_start = end_month_map.get(m_end, 'JAN')
        return f"{prefix}QS-{m_start}"
    if s.endswith('QE') or s == 'Q':
        prefix = s[:-2] if s.endswith('QE') else ''
        return f"{prefix}QS"
    if s.endswith('ME') or s == 'M':
        prefix = s[:-2] if s.endswith('ME') else ''
        return f"{prefix}MS"
    # Preserve lowercase sub-daily units required by pandas >= 2.2 without altering 'MS' (MonthStart)
    unit_part = s_raw.lstrip('0123456789')
    if unit_part in ('h', 'H', 'min', 's', 'ms', 'us', 'ns') or unit_part.lower() in ('h', 'min', 'us', 'ns'):
        prefix = s_raw[:len(s_raw) - len(unit_part)]
        return f"{prefix}{unit_part.lower()}"
    return s

def _block_edges(t_min, t_max, block_frequency, block_origin, is_datetime):
    if is_datetime:
        _parse_block_frequency(block_frequency, True)
        start_freq = _normalize_start_freq(block_frequency)
        start_offset = to_offset(start_freq)

        t_min_ts = pd.Timestamp(t_min)
        t_max_ts = pd.Timestamp(t_max)
        if t_min_ts.tzinfo is not None:
            t_min_ts = t_min_ts.tz_convert(None)
        if t_max_ts.tzinfo is not None:
            t_max_ts = t_max_ts.tz_convert(None)

        if block_origin is not None:
            origin = pd.Timestamp(block_origin)
            if origin.tzinfo is not None:
                origin = origin.tz_convert(None)
            while origin > t_min_ts:
                prev = origin - start_offset
                if prev >= origin:
                    break
                origin = prev
            if isinstance(start_offset, pd.offsets.Tick) and (origin + start_offset) <= t_min_ts:
                step_td = pd.Timedelta(start_offset)
                if step_td > pd.Timedelta(0):
                    k_fwd = int((t_min_ts - origin) // step_td)
                    if k_fwd > 0:
                        origin = origin + k_fwd * step_td
                        while origin > t_min_ts:
                            origin = origin - step_td
        else:
            unit_freq = start_freq.lstrip('0123456789')
            unit_offset = to_offset(unit_freq)
            # Case-sensitive check: keep lowercase 's'/'ms'/'us'/'ns' distinct from 'MS' (MonthStart)
            if unit_freq in ('s', 'ms', 'us', 'ns'):
                origin = unit_offset.rollback(t_min_ts)
            else:
                origin = unit_offset.rollback(t_min_ts.normalize())

        edges_dt = pd.date_range(start=origin, end=t_max_ts + start_offset, freq=start_offset)
        while len(edges_dt) < 2 or edges_dt[-1] <= t_max_ts:
            prev_last = pd.Timestamp(edges_dt[-1]) if len(edges_dt) > 0 else None
            edges_dt = pd.date_range(start=origin, periods=len(edges_dt) + 1, freq=start_offset)
            if prev_last is not None and pd.Timestamp(edges_dt[-1]) <= prev_last:
                break
        return edges_dt.to_numpy()
    else:
        width = float(block_frequency)
        t_min_f, t_max_f = float(t_min), float(t_max)
        if block_origin is not None:
            origin = float(block_origin)
            if origin > t_min_f:
                k_back = max(1, int(np.ceil((origin - t_min_f) / width)))
                origin = origin - k_back * width
                while origin > t_min_f:
                    nxt = origin - width
                    if nxt >= origin:
                        break
                    origin = nxt
            elif (origin + width) <= t_min_f:
                k_fwd = int(np.floor((t_min_f - origin) / width))
                if k_fwd > 0:
                    origin = origin + k_fwd * width
                    while origin > t_min_f:
                        nxt = origin - width
                        if nxt >= origin:
                            break
                        origin = nxt
        else:
            origin = np.floor(t_min_f / width) * width
            while origin > t_min_f:
                nxt = origin - width
                if nxt >= origin:
                    break
                origin = nxt
        n_steps = max(2, int(np.ceil((t_max_f - origin) / width)) + 2)
        edges = origin + np.arange(n_steps, dtype=float) * width
        while edges[-1] <= t_max_f:
            nxt = origin + len(edges) * width
            if nxt <= edges[-1]:
                break
            edges = np.append(edges, nxt)
        return edges

def _expected_block_seconds(b_start_ts: pd.Timestamp, b_end_ts: pd.Timestamp, months=None) -> float:
    nominal_sec = (b_end_ts - b_start_ts).total_seconds()
    if months is None or nominal_sec <= 0.0:
        return nominal_sec
    if nominal_sec < 86400.0:
        end_probe = b_end_ts - pd.Timedelta(nanoseconds=1)
        if b_start_ts.month in months and end_probe.month in months:
            return nominal_sec
        if b_start_ts.month not in months and end_probe.month not in months:
            return 0.0
        boundary = end_probe.normalize()
        sec = 0.0
        if b_start_ts.month in months:
            sec += max(0.0, (boundary - b_start_ts).total_seconds())
        if end_probe.month in months:
            sec += max(0.0, (b_end_ts - boundary).total_seconds())
        return sec
    day_starts = pd.date_range(b_start_ts.normalize(), (b_end_ts - pd.Timedelta(nanoseconds=1)).normalize(), freq='D')
    total = 0.0
    for d in day_starts:
        if d.month in months:
            seg_start = max(b_start_ts, d)
            seg_end = min(b_end_ts, d + pd.Timedelta(days=1))
            total += max(0.0, (seg_end - seg_start).total_seconds())
    return total

def _block_coverage(block_start, block_end, t_block_obs, is_datetime, months=None):
    if len(t_block_obs) == 0:
        return 0.0
    if is_datetime:
        b_start_ts = pd.Timestamp(block_start)
        b_end_ts = pd.Timestamp(block_end)
        duration_sec = _expected_block_seconds(b_start_ts, b_end_ts, months=months)

        if duration_sec <= 0:
            return 1.0
        t_sec = _to_numeric_time(t_block_obs)
        t_first, t_last = t_sec.min(), t_sec.max()
        diffs = np.diff(np.sort(t_sec))
        pos_diffs = diffs[diffs > 0]
        median_spacing = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 86400.0
        span = (t_last - t_first) + median_spacing
        return float(min(1.0, max(0.0, span / duration_sec)))
    else:
        duration = float(block_end - block_start)
        if duration <= 0:
            return 1.0
        t_arr = np.asarray(t_block_obs, dtype=float)
        t_first, t_last = t_arr.min(), t_arr.max()
        diffs = np.diff(np.sort(t_arr))
        pos_diffs = diffs[diffs > 0]
        median_spacing = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 0.0
        span = (t_last - t_first) + median_spacing
        return float(min(1.0, max(0.0, span / duration)))

def _assign_blocks(
    data: pd.DataFrame,
    block_frequency,
    block_origin,
    months,
    min_size_per_block: int,
    min_coverage: float,
    require_complete: bool,
    is_datetime: bool,
):
    t_vals = data['t_original'].to_numpy() if is_datetime else data['t'].to_numpy()
    if len(t_vals) == 0:
        return [], pd.DataFrame(), np.array([])

    _parse_block_frequency(block_frequency, is_datetime)
    t_min, t_max = t_vals.min(), t_vals.max()
    edges = _block_edges(t_min, t_max, block_frequency, block_origin, is_datetime)

    active_data = data.copy()
    if months is not None and is_datetime:
        dt_series = pd.to_datetime(active_data['t_original'])
        active_mask = dt_series.dt.month.isin(months)
        active_data = active_data[active_mask].reset_index(drop=True)

    t_active = active_data['t_original'].to_numpy() if is_datetime else active_data['t'].to_numpy()
    if len(t_active) == 0:
        return [], pd.DataFrame(), np.array([])

    if is_datetime:
        t_nums = _to_numeric_time(t_active)
        edge_nums = _to_numeric_time(edges)
    else:
        t_nums = np.asarray(t_active, dtype=float)
        edge_nums = np.asarray(edges, dtype=float)

    all_diffs = np.diff(np.unique(t_nums))
    pos_all_diffs = all_diffs[all_diffs > 0]
    global_step_sec = float(np.median(pos_all_diffs)) if len(pos_all_diffs) > 0 else (86400.0 if is_datetime else np.nan)

    indices = (np.searchsorted(edge_nums, t_nums, side='right') - 1).astype(np.intp, copy=False)

    summary_records = []
    valid_groups = []
    block_centers = []

    for i in range(len(edges) - 1):
        b_start, b_end = edges[i], edges[i + 1]
        mask = indices == i
        group_df = active_data[mask]
        n_obs = len(group_df)
        if n_obs == 0 and (i == 0 or i == len(edges) - 2):
            continue

        n_censored = int(group_df['censored'].sum()) if 'censored' in group_df.columns else 0
        obs_times = group_df['t_original'].to_numpy() if is_datetime else group_df['t'].to_numpy()
        coverage = _block_coverage(b_start, b_end, obs_times, is_datetime, months=months)

        valid = True
        reason = "ok"
        if n_obs < min_size_per_block:
            valid = False
            reason = f"below min_size_per_block ({n_obs} < {min_size_per_block})"
        elif coverage < min_coverage:
            valid = False
            reason = f"coverage {coverage:.2f} < {min_coverage}"
        elif require_complete:
            if is_datetime:
                expected_sec = _expected_block_seconds(pd.Timestamp(b_start), pd.Timestamp(b_end), months=months)
                if np.isfinite(global_step_sec) and global_step_sec > 0:
                    expected_n = int(round(expected_sec / global_step_sec))
                else:
                    expected_n = 1 if expected_sec > 0 else 0
            else:
                duration = float(b_end) - float(b_start)
                if np.isfinite(global_step_sec) and global_step_sec > 0:
                    expected_n = int(round(duration / global_step_sec))
                else:
                    expected_n = n_obs + 1 if duration > 0 else 0
            if n_obs < expected_n:
                valid = False
                reason = f"incomplete block ({n_obs} < {expected_n} expected)"

        if is_datetime:
            b_center = pd.Timestamp(b_start) + (pd.Timestamp(b_end) - pd.Timestamp(b_start)) / 2
        else:
            b_center = 0.5 * (float(b_start) + float(b_end))

        summary_records.append({
            'block_start': b_start,
            'block_end': b_end,
            'block_center': b_center,
            'n_obs': n_obs,
            'n_censored': n_censored,
            'coverage': coverage,
            'valid': valid,
            'reason': reason,
        })

        if valid and n_obs > 0:
            valid_groups.append(group_df.sort_values('t', kind='stable').reset_index(drop=True))
            block_centers.append(b_center)

    summary_df = pd.DataFrame(summary_records)
    return valid_groups, summary_df, np.array(block_centers)

def _plotting_positions(M: int):
    ranks = np.arange(1, M + 1, dtype=float)
    p_grid = ranks / (M + 1.0)
    z_grid = norm.ppf(p_grid)
    return ranks, p_grid, z_grid

def _standardize_block(
    group_df: pd.DataFrame,
    M: int,
    standardization: str,
    quantile_method: str,
    censor_quantile_method: str,
    lt_mult: float,
    gt_mult: float,
    u_grid: np.ndarray = None,
):
    vals = group_df['value'].to_numpy(dtype=float).copy()
    censored = (
        group_df['censored'].to_numpy(dtype=bool).copy()
        if 'censored' in group_df.columns
        else np.zeros(len(vals), dtype=bool)
    )
    cen_type = (
        group_df['cen_type'].to_numpy(dtype=object).copy()
        if 'cen_type' in group_df.columns
        else np.full(len(vals), 'not', dtype=object)
    )

    if censor_quantile_method == 'error' and censored.any():
        raise ValueError("Censored observations encountered with censor_quantile_method='error'.")
    if censor_quantile_method == 'substitute' and censored.any():
        vals[censored & (cen_type == 'lt')] *= lt_mult
        vals[censored & (cen_type == 'gt')] *= gt_mult
        censored[:] = False
        cen_type[:] = 'not'

    from ._ats import make_intervals
    lo, hi = make_intervals(vals, censored, cen_type=cen_type, nudge=True)
    sort_key = np.where(censored & (cen_type == 'lt'), hi, np.where(censored & (cen_type == 'gt'), lo, vals))
    sort_idx = np.argsort(sort_key, kind='stable').astype(np.intp, copy=False)
    s_vals = vals[sort_idx]
    s_cens = censored[sort_idx]
    s_type = cen_type[sort_idx]

    n_obs = len(s_vals)
    if standardization == 'egret':
        if n_obs == M:
            std_vals = s_vals.copy()
            std_cens = s_cens.copy()
            std_type = s_type.copy()
        elif n_obs == M + 1:
            mid = n_obs // 2
            if n_obs % 2 == 1:
                keep = np.ones(n_obs, dtype=bool)
                keep[mid] = False
                std_vals = s_vals[keep]
                std_cens = s_cens[keep]
                std_type = s_type[keep]
            else:
                merged_val = 0.5 * (s_vals[mid - 1] + s_vals[mid])
                merged_cens = bool(s_cens[mid - 1] or s_cens[mid])
                merged_type = s_type[mid - 1] if s_type[mid - 1] != 'not' else s_type[mid]
                keep = np.ones(n_obs, dtype=bool)
                keep[mid] = False
                std_vals = s_vals[keep].copy()
                std_cens = s_cens[keep].copy()
                std_type = s_type[keep].copy()
                std_vals[mid - 1] = merged_val
                std_cens[mid - 1] = merged_cens
                std_type[mid - 1] = merged_type
        else:
            raise ValueError(
                f"standardization='egret' requires block sample size n_obs ({n_obs}) "
                f"to equal target_length ({M}) or target_length + 1 ({M + 1})."
            )
    else:
        if u_grid is None:
            u_grid = np.linspace(0.0, 1.0, M) if M > 1 else np.array([0.5])
        std_vals = np.quantile(s_vals, u_grid, method=quantile_method)
        std_cens = np.zeros(M, dtype=bool)
        std_type = np.full(M, 'not', dtype=object)

        if s_cens.any() and censor_quantile_method == 'propagate':
            lt_mask = s_cens & (s_type == 'lt')
            gt_mask = s_cens & (s_type == 'gt')

            v_pos = u_grid * (n_obs - 1)
            lo_idx = np.clip(np.floor(v_pos).astype(np.intp), 0, n_obs - 1)
            hi_idx = np.clip(np.ceil(v_pos).astype(np.intp), 0, n_obs - 1)
            neighbor_lt = lt_mask[lo_idx] | lt_mask[hi_idx]
            neighbor_gt = gt_mask[lo_idx] | gt_mask[hi_idx]

            for k in range(M):
                if neighbor_lt[k]:
                    std_cens[k] = True
                    std_type[k] = 'lt'
                elif neighbor_gt[k]:
                    std_cens[k] = True
                    std_type[k] = 'gt'

    return std_vals, std_cens, std_type

def _sampling_diagnostic(block_centers, block_sizes):
    if len(block_sizes) < 3:
        return {'tau_n_vs_time': 0.0, 'p_n_vs_time': 1.0, 'n_min': int(block_sizes.min()) if len(block_sizes)>0 else 0, 'n_max': int(block_sizes.max()) if len(block_sizes)>0 else 0, 'ratio_max_min': 1.0, 'flag': False}

    t_num = _to_numeric_time(block_centers)
    res_tau = kendalltau(t_num, block_sizes)
    tau_val = float(res_tau.statistic) if not np.isnan(res_tau.statistic) else 0.0
    p_val = float(res_tau.pvalue) if not np.isnan(res_tau.pvalue) else 1.0

    n_min = int(block_sizes.min())
    n_max = int(block_sizes.max())
    ratio = float(n_max / n_min) if n_min > 0 else 1.0
    flag = bool(p_val < 0.10 or ratio > 1.5)

    return {
        'tau_n_vs_time': tau_val,
        'p_n_vs_time': p_val,
        'n_min': n_min,
        'n_max': n_max,
        'ratio_max_min': ratio,
        'flag': flag
    }

def _equalize_blocks(valid_group_dfs, mode: str, is_datetime: bool = True):
    if mode == 'none' or not valid_group_dfs:
        return valid_group_dfs
    elif mode == 'thin':
        min_n = min(len(df) for df in valid_group_dfs)
        thinned = []
        for df in valid_group_dfs:
            n_b = len(df)
            if n_b == min_n:
                thinned.append(df)
            else:
                idx = np.round(np.linspace(0, n_b - 1, min_n)).astype(np.intp)
                thinned.append(df.iloc[idx].reset_index(drop=True))
        return thinned
    elif mode == 'aggregate':
        from ._helpers import _aggregate_by_group
        aggregated = []
        agg_unit = None
        if is_datetime:
            spans_days = []
            for df in valid_group_dfs:
                dt_col = pd.to_datetime(df['t_original'])
                spans_days.append((dt_col.max() - dt_col.min()).total_seconds() / 86400.0)
            median_span_days = float(np.median(spans_days)) if spans_days else 0.0
            if median_span_days > 45.0:
                agg_unit = 'month'
            elif median_span_days > 2.0:
                agg_unit = 'day'
            elif median_span_days > 2.0 / 24.0:
                agg_unit = 'hour'
            else:
                agg_unit = 'minute'

        for df in valid_group_dfs:
            if is_datetime:
                dt_col = pd.to_datetime(df['t_original'])
                if getattr(dt_col.dt, 'tz', None) is not None:
                    dt_col = dt_col.dt.tz_convert(None)
                if agg_unit == 'month':
                    sub_key = dt_col.dt.year * 100 + dt_col.dt.month
                    freq_str = 'ME'
                elif agg_unit == 'day':
                    sub_key = dt_col.dt.floor('D')
                    freq_str = 'D'
                elif agg_unit == 'hour':
                    sub_key = dt_col.dt.floor('h')
                    freq_str = 'h'
                else:
                    sub_key = dt_col.dt.floor('min')
                    freq_str = 'min'
                if pd.Series(sub_key).nunique() < 2 and len(df) >= 2:
                    t_vals = df['t'].to_numpy(dtype=float)
                    span = t_vals.max() - t_vals.min()
                    if span > 0:
                        step = span / 12.0
                        sub_key = np.floor((t_vals - t_vals.min()) / step).astype(np.intp)
                        freq_str = None
            else:
                t_vals = df['t'].to_numpy(dtype=float)
                span = t_vals.max() - t_vals.min()
                step = span / 12.0 if span > 0 else 1.0
                sub_key = np.floor((t_vals - t_vals.min()) / step).astype(np.intp)
                freq_str = None
            df_tmp = df.assign(_sub_key=np.asarray(sub_key))
            agg_df = pd.DataFrame([
                _aggregate_by_group(
                    g.drop(columns='_sub_key'),
                    'robust_median',
                    is_datetime,
                    period=freq_str,
                    return_dict=True,
                )
                for _, g in df_tmp.groupby('_sub_key', sort=True)
            ])
            aggregated.append(agg_df.sort_values('t', kind='stable').reset_index(drop=True))
        return aggregated
    else:
        raise ValueError(f"Unknown equalize mode: '{mode}'.")

def _mk_s_matrix(M_matrix: np.ndarray, t_numeric: np.ndarray, tau_method: str = 'b'):
    B, K = M_matrix.shape
    if B < 2:
        return np.zeros(K), np.ones(K), np.zeros(K)
    
    i_idx, j_idx = np.triu_indices(B, k=1)
    dt = t_numeric[j_idx] - t_numeric[i_idx]
    valid_t = dt != 0.0
    if not np.any(valid_t):
        return np.zeros(K), np.ones(K), np.zeros(K)

    ii, jj = i_idx[valid_t], j_idx[valid_t]
    dM = M_matrix[jj, :] - M_matrix[ii, :]
    dt_sign = np.sign(t_numeric[jj] - t_numeric[ii])[:, np.newaxis]
    
    s_scores = np.sum(dt_sign * np.sign(dM), axis=0)

    # Variance and Tau adjusted for ties in values per column
    base_var = B * (B - 1) * (2 * B + 5) / 18.0
    n0 = B * (B - 1) / 2.0
    var_s = np.empty(K, dtype=float)
    tau = np.empty(K, dtype=float)

    for k in range(K):
        _, counts = np.unique(M_matrix[:, k], return_counts=True)
        tie_counts = counts[counts > 1]
        if len(tie_counts) > 0:
            tie_adj = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)) / 18.0
            var_s[k] = max(0.0, base_var - tie_adj)
            if tau_method == 'b':
                n_ties = np.sum(tie_counts * (tie_counts - 1)) / 2.0
                denom = np.sqrt(max(0.0, n0 - n_ties)) * np.sqrt(n0)
                tau[k] = s_scores[k] / denom if denom > 0 else 0.0
            else:
                tau[k] = s_scores[k] / n0 if n0 > 0 else 0.0
        else:
            var_s[k] = base_var
            tau[k] = s_scores[k] / n0 if n0 > 0 else 0.0

    return s_scores, var_s, tau
