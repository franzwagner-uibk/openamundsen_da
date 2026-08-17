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
from openamundsen_da.util.humidity import perturb_relative_humidity_via_dew_point
from openamundsen_da.util.storage_policy import apply_meteo_csv_precision


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
    - Applies humidity perturbation as an additive dew-point temperature offset.
    - Applies multiplicative f_p to positive precip values (if present).
    - Applies multiplicative f_sw to positive sw_in values only (if present).
    - Writes only stations that have at least one row in the selected window.
    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    stations_csv = src_dir / STATIONS_CSV
    written_station_ids: list[str] = []

    for src in sorted(p for p in src_dir.glob("*.csv") if p.name != STATIONS_CSV):
        df = pd.read_csv(src, parse_dates=True, index_col=0)
        time_col = df.index.name or DEFAULT_TIME_COL
        df = _inclusive_filter(df, start, end)
        if df.empty:
            continue
        df.index = _normalize_datetime_index(df.index)
        has_temp = DEFAULT_TEMP_COL in df.columns
        has_rel_hum = DEFAULT_REL_HUM_COL in df.columns
        temp = _numeric_perturbation_series(df[DEFAULT_TEMP_COL]) if has_temp else None
        if has_rel_hum:
            rh = _numeric_perturbation_series(df[DEFAULT_REL_HUM_COL])
            if (delta_t != 0.0 or delta_rh != 0.0) and temp is None:
                raise ValueError(
                    f"{src} contains '{DEFAULT_REL_HUM_COL}' but no '{DEFAULT_TEMP_COL}'; "
                    "dew-point humidity perturbation requires air temperature"
                )
            if delta_t != 0.0 or delta_rh != 0.0:
                df[DEFAULT_REL_HUM_COL] = perturb_relative_humidity_via_dew_point(
                    temp.to_numpy(),
                    rh.to_numpy(),
                    delta_rh,
                    delta_t=delta_t,
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
        df_out = apply_meteo_csv_precision(df_out)
        dst_dir.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(dst_dir / src.name, index=False)
        written_station_ids.append(src.stem)

    if stations_csv.exists():
        metadata = pd.read_csv(stations_csv, dtype={"id": "string"})
        if "id" not in metadata.columns:
            raise ValueError(f"Meteo station metadata has no 'id' column: {stations_csv}")
        station_ids = metadata["id"].astype("string")
        selected = metadata.loc[station_ids.isin(written_station_ids)].copy()
        selected_ids = set(selected["id"].dropna().astype(str))
        missing = sorted(set(written_station_ids) - selected_ids)
        if missing:
            raise ValueError(
                "Meteo forcing files have no matching stations.csv row: "
                + ", ".join(missing)
            )
        selected.to_csv(dst_dir / STATIONS_CSV, index=False)


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
