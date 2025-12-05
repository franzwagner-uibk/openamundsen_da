from __future__ import annotations

"""Time-series helpers used by visualization and analysis modules.

Functions focus on light pandas operations that are commonly repeated across
plotters: windowing, resampling + smoothing, and hydrological-year cumulative
precipitation.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Iterable, List, Sequence

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


# ---- Shared parsing/loading helpers ----------------------------------------


def parse_time_column(series: pd.Series) -> pd.DatetimeIndex:
    """Parse mixed date/datetime columns robustly.

    Tries common explicit formats before falling back to pandas' parser.
    """
    text = series.astype(str)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            parsed = pd.to_datetime(text, format=fmt, errors="coerce")
        except Exception:
            continue
        if not parsed.isna().all():
            return pd.DatetimeIndex(parsed)
    return pd.DatetimeIndex(pd.to_datetime(text, errors="coerce"))


def collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate timestamps by mean if index is DatetimeIndex.

    Returns the input unchanged if index is not DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    out = df.sort_index()
    if out.index.is_unique:
        return out
    return out.groupby(level=0).mean(numeric_only=True)


def read_timeseries_csv(csv_path: Path, time_col: str, columns: Sequence[str]) -> pd.DataFrame:
    """Read one or more numeric columns with a parsed datetime index.

    - Ensures `time_col` exists and columns exist; coerces to numeric.
    - Drops rows with NaT index; collapses duplicate timestamps by mean.
    """
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"Missing time column '{time_col}' in {Path(csv_path).name}")
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {Path(csv_path).name}")
    idx = parse_time_column(df[time_col])
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in columns})
    out.index = idx
    out = out[~out.index.isna()]
    return collapse_duplicates(out)


def concat_series(series_list: List[pd.Series]) -> pd.Series:
    """Concatenate multiple series along the time axis and collapse duplicates."""
    if not series_list:
        return pd.Series(dtype=float)
    df = pd.concat(series_list, axis=0)
    df = collapse_duplicates(df.to_frame("v") if not isinstance(df, pd.DataFrame) else df)
    if isinstance(df, pd.DataFrame):
        # return the single column if present
        if df.shape[1] == 1:
            return df.iloc[:, 0]
    # already a Series
    return df


def parse_datetime_opt(text: str | None) -> "pd.Timestamp | None":
    """Best-effort datetime parser for config values (YYYY-MM-DD[_HH:MM:SS] allowed)."""
    if not text:
        return None
    t = str(text).strip().replace("_", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.to_datetime(t, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(t)
    except Exception:
        return None
