"""Shared helpers for meteo CSV perturbation and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from openamundsen_da.core.constants import (
    DEFAULT_PRECIP_COL,
    DEFAULT_REL_HUM_COL,
    DEFAULT_SW_IN_COL,
    DEFAULT_TEMP_COL,
    DEFAULT_TIME_COL,
    STATIONS_CSV,
)

MAGNUS_A = 17.62
MAGNUS_B_DEG_C = 243.12
KELVIN_OFFSET_DEG_C = 273.15
MIN_REL_HUM_PERCENT = 1e-6


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
    - Applies additive delta_rh as a dew-point temperature offset before
      recalculating rel_hum (if temp and rel_hum are present).
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
        has_temp = DEFAULT_TEMP_COL in df.columns
        has_rel_hum = DEFAULT_REL_HUM_COL in df.columns
        temp = _numeric_perturbation_series(df[DEFAULT_TEMP_COL]) if has_temp else None
        if (delta_t != 0.0 or delta_rh != 0.0) and has_temp and has_rel_hum:
            df[DEFAULT_REL_HUM_COL] = perturb_relative_humidity_via_dew_point(
                temp,
                df[DEFAULT_REL_HUM_COL],
                delta_t=delta_t,
                delta_dew_point=delta_rh,
            )
        if (delta_t != 0.0) and has_temp:
            df[DEFAULT_TEMP_COL] = temp + delta_t
        if (f_p != 1.0) and (DEFAULT_PRECIP_COL in df.columns):
            precip = _numeric_perturbation_series(df[DEFAULT_PRECIP_COL])
            mask = precip > 0.0
            precip = precip.where(~mask, precip * f_p)
            df[DEFAULT_PRECIP_COL] = precip
        if (f_sw != 1.0) and (DEFAULT_SW_IN_COL in df.columns):
            sw_in = _numeric_perturbation_series(df[DEFAULT_SW_IN_COL])
            mask = sw_in > 0.0
            sw_in = sw_in.where(~mask, sw_in * f_sw)
            df[DEFAULT_SW_IN_COL] = sw_in.clip(lower=0.0)
        idx_col_name = df.index.name or "index"
        df_out = df.reset_index().rename(columns={idx_col_name: time_col})
        dst_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(dst_dir / src.name, index=False)


def perturb_relative_humidity_via_dew_point(
    temp_k: pd.Series,
    rel_hum_percent: pd.Series,
    *,
    delta_t: float,
    delta_dew_point: float,
) -> pd.Series:
    """Perturb relative humidity through dew-point temperature.

    The meteo forcing stores air temperature in Kelvin. ``delta_t`` and
    ``delta_dew_point`` are additive temperature offsets, so their numerical
    values are identical in Kelvin and degrees Celsius.
    """
    temp_c = _numeric_perturbation_series(temp_k) - KELVIN_OFFSET_DEG_C
    rel_hum = _numeric_perturbation_series(rel_hum_percent).clip(
        lower=MIN_REL_HUM_PERCENT,
        upper=100.0,
    )
    rh_fraction = rel_hum / 100.0

    gamma = np.log(rh_fraction) + _magnus_exponent(temp_c)
    dew_point_c = MAGNUS_B_DEG_C * gamma / (MAGNUS_A - gamma)
    perturbed_temp_c = temp_c + delta_t
    perturbed_dew_point_c = (dew_point_c + delta_dew_point).clip(upper=perturbed_temp_c)

    perturbed_rh = 100.0 * np.exp(
        _magnus_exponent(perturbed_dew_point_c) - _magnus_exponent(perturbed_temp_c)
    )
    return perturbed_rh.clip(lower=0.0, upper=100.0)


def _magnus_exponent(temp_c: pd.Series) -> pd.Series:
    return (MAGNUS_A * temp_c) / (MAGNUS_B_DEG_C + temp_c)


def _inclusive_filter(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start = _strip_timezone(start)
    end = _strip_timezone(end)
    dt_idx = _normalize_datetime_index(df.index)
    mask = (dt_idx >= start) & (dt_idx <= end)
    out = df.loc[mask].copy()
    out.index = dt_idx[mask]
    return out


def _numeric_perturbation_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


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
