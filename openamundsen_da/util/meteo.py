"""Shared helpers for meteo CSV perturbation and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from openamundsen_da.core.constants import (
    DEFAULT_PRECIP_COL,
    DEFAULT_REL_HUM_COL,
    DEFAULT_SW_IN_COL,
    DEFAULT_TEMP_COL,
    DEFAULT_TIME_COL,
    STATIONS_CSV,
)


def filter_and_write_meteo(
    src_dir: Path,
    dst_dir: Path,
    start,
    end,
    *,
    delta_t: float = 0.0,
    f_p: float = 1.0,
    delta_rh: float = 0.0,
    f_sw: float = 1.0,
) -> None:
    """Filter meteo CSVs to [start..end], apply perturbations, and write to dst_dir.

    - Uses the first column as datetime index (name flexible).
    - Applies additive delta_t to temp (if present).
    - Applies additive delta_rh to rel_hum with clipping to [0, 100] (if present).
    - Applies multiplicative f_p to positive precip values (if present).
    - Applies multiplicative f_sw to positive sw_in values only (if present).
    - Copies stations.csv unchanged.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    stations_csv = src_dir / STATIONS_CSV
    if stations_csv.exists():
        (dst_dir / STATIONS_CSV).write_bytes(stations_csv.read_bytes())

    for src in sorted(p for p in src_dir.glob("*.csv") if p.name != STATIONS_CSV):
        df = pd.read_csv(src, parse_dates=True, index_col=0)
        time_col = df.index.name or DEFAULT_TIME_COL
        df = _inclusive_filter(df, start, end)
        df.index = _normalize_datetime_index(df.index)
        if (delta_t != 0.0) and (DEFAULT_TEMP_COL in df.columns):
            df[DEFAULT_TEMP_COL] = pd.to_numeric(df[DEFAULT_TEMP_COL], errors="coerce") + delta_t
        if (delta_rh != 0.0) and (DEFAULT_REL_HUM_COL in df.columns):
            rh = pd.to_numeric(df[DEFAULT_REL_HUM_COL], errors="coerce") + delta_rh
            df[DEFAULT_REL_HUM_COL] = rh.clip(lower=0.0, upper=100.0)
        if (f_p != 1.0) and (DEFAULT_PRECIP_COL in df.columns):
            precip = pd.to_numeric(df[DEFAULT_PRECIP_COL], errors="coerce")
            mask = precip > 0.0
            precip.loc[mask] = precip.loc[mask] * f_p
            df[DEFAULT_PRECIP_COL] = precip
        if (f_sw != 1.0) and (DEFAULT_SW_IN_COL in df.columns):
            sw_in = pd.to_numeric(df[DEFAULT_SW_IN_COL], errors="coerce")
            mask = sw_in > 0.0
            sw_in.loc[mask] = sw_in.loc[mask] * f_sw
            df[DEFAULT_SW_IN_COL] = sw_in.clip(lower=0.0)
        idx_col_name = df.index.name or "index"
        df_out = df.reset_index().rename(columns={idx_col_name: time_col})
        dst_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(dst_dir / src.name, index=False)


def _inclusive_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start = _strip_timezone(start)
    end = _strip_timezone(end)
    dt_idx = _normalize_datetime_index(df.index)
    mask = (dt_idx >= start) & (dt_idx <= end)
    out = df.loc[mask].copy()
    out.index = dt_idx[mask]
    return out


def _normalize_datetime_index(idx: Iterable) -> pd.DatetimeIndex:
    dt_idx = pd.to_datetime(idx, errors="coerce")
    if getattr(dt_idx, "tz", None) is not None:
        dt_idx = dt_idx.tz_convert("UTC").tz_localize(None)
    return dt_idx


def _strip_timezone(ts: pd.Timestamp) -> pd.Timestamp:
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC")
    try:
        return ts.tz_localize(None)
    except Exception:
        return ts
