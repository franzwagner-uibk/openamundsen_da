from __future__ import annotations

"""Time-series helpers used by visualization and analysis modules.

Functions focus on light pandas operations that are commonly repeated across
plotters: windowing, resampling + smoothing, and hydrological-year cumulative
precipitation.
"""

from datetime import datetime
from typing import Optional, Dict

import numpy as np
import pandas as pd


def apply_window(df: pd.DataFrame, start: Optional[datetime], end: Optional[datetime]) -> pd.DataFrame:
    """Return a view of df limited to [start, end] if index is datetime-like."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df
    if start is not None:
        out = out[out.index >= start]
    if end is not None:
        out = out[out.index <= end]
    return out


def resample_and_smooth(
    df: pd.DataFrame,
    rule: Optional[str],
    agg: Optional[Dict[str, str]] = None,
    rolling: Optional[int] = None,
) -> pd.DataFrame:
    """Resample by rule using provided aggregation per column, then smooth.

    - rule: pandas offset alias (e.g., 'D'), or None to skip resampling
    - agg: mapping column -> aggregation ('mean', 'sum', 'first', 'last', ...)
    - rolling: window length (samples) for simple moving average per column
    """
    out = df
    if isinstance(out.index, pd.DatetimeIndex) and rule:
        out = out.resample(rule).agg(agg or {}) if (agg is not None) else out.resample(rule).mean()
    if rolling and rolling > 1:
        for c in out.columns:
            out[c] = out[c].rolling(rolling, min_periods=1).mean()
    return out


def hydro_year_index(idx: pd.DatetimeIndex, start_month: int, start_day: int) -> np.ndarray:
    """Return hydrological year integer per timestamp given month/day start."""
    before = (idx.month < start_month) | ((idx.month == start_month) & (idx.day < start_day))
    return (idx.year - before.astype(int)).astype(int, copy=False)


def cumulative_hydro(series: pd.Series, start_month: int, start_day: int) -> pd.Series:
    """Return cumulative precipitation within hydrological years for a Series.

    Negative values are clipped at zero; NaNs treated as zero for accumulation.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        return pd.Series(dtype=float)
    pr = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    hy = hydro_year_index(pr.index, start_month, start_day)
    return pr.groupby(hy).cumsum()

