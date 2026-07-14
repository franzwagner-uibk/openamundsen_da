from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from openamundsen_da.methods.wet_snow.wsl import (
    compute_wet_snow_line_from_fraction_grid,
    load_wet_snow_line_config,
)

if TYPE_CHECKING:
    from openamundsen_da.methods.viz.maps.data import StaticContext


def finite_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[column], errors="coerce").dropna()


def first_finite_value(df: pd.DataFrame, columns: list[str]) -> float | None:
    for column in columns:
        values = finite_numeric_column(df, column)
        if not values.empty:
            return float(values.iloc[0])
    return None


def wsl_prior_summary_from_weights_df(df: pd.DataFrame) -> dict[str, float | int | None] | None:
    model_values = finite_numeric_column(df, "value_model")
    if model_values.empty:
        return None
    return {
        "mean": float(model_values.mean()),
        "q05": float(model_values.quantile(0.05)),
        "q95": float(model_values.quantile(0.95)),
        "median": float(model_values.median()),
        "min": float(model_values.min()),
        "max": float(model_values.max()),
        "obs": first_finite_value(df, ["value_obs", "wet_snow_line_obs"]),
        "n_members": int(model_values.size),
    }


def elevation_band_fraction_map(
    *,
    context: StaticContext,
    wet_fraction: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    cfg = load_wet_snow_line_config(context.project_dir)
    dem = np.asarray(context.dem, dtype=float)
    wet_fraction = np.asarray(wet_fraction, dtype=float)
    valid = (
        np.asarray(valid_mask, dtype=bool)
        & np.asarray(context.roi_mask, dtype=bool)
        & np.isfinite(dem)
        & np.isfinite(wet_fraction)
    )
    out = np.full(dem.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out

    valid_elev = dem[valid]
    band = float(cfg.elevation_band_size_m)
    low = np.floor(np.nanmin(valid_elev) / band) * band
    high = np.ceil(np.nanmax(valid_elev) / band) * band
    if np.isclose(low, high):
        high = low + band
    edges = np.arange(low, high + band, band, dtype=float)
    if edges.size < 2:
        edges = np.array([low, low + band], dtype=float)

    for idx in range(len(edges) - 1):
        band_low = float(edges[idx])
        band_high = float(edges[idx + 1])
        if idx == len(edges) - 2:
            band_mask = valid & (dem >= band_low) & (dem <= band_high)
        else:
            band_mask = valid & (dem >= band_low) & (dem < band_high)
        if not np.any(band_mask):
            continue
        out[band_mask] = float(np.nanmean(wet_fraction[band_mask]))
    return out


def wet_snow_line_from_fraction(
    *,
    context: StaticContext,
    wet_fraction: np.ndarray,
    threshold: float | None = None,
) -> float | None:
    return compute_wet_snow_line_from_fraction_grid(
        project_dir=context.project_dir,
        dem=context.dem,
        roi_mask=context.roi_mask,
        wet_fraction=wet_fraction,
        threshold=threshold,
    )
