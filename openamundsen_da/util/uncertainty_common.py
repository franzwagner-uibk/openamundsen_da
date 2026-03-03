"""Shared uncertainty parsing and NetCDF/raster consistency helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from openamundsen_da.util.config_validators import require_nonempty_str


INGEST_MODES = {"product_layer", "companion_layer", "generated_layer"}
SIGMA_MODES = {"formula", "uncertainty_layer"}


def parse_ingest_block(
    ingest: dict[str, object],
    *,
    path: str,
    value_variable_key: str,
) -> tuple[str, str | None, str | None, str | None]:
    """Parse generic uncertainty ingest config block."""
    mode = require_nonempty_str(ingest, "mode", path=path).lower()
    if mode not in INGEST_MODES:
        raise ValueError(
            f"{path}.mode must be one of: product_layer, companion_layer, generated_layer"
        )

    value_variable = None
    uncertainty_variable = None
    time_variable = None
    if mode == "product_layer":
        value_variable = require_nonempty_str(ingest, value_variable_key, path=path)
        uncertainty_variable = require_nonempty_str(ingest, "uncertainty_variable", path=path)
        time_variable = require_nonempty_str(ingest, "time_variable", path=path)

    return mode, value_variable, uncertainty_variable, time_variable


def parse_assimilation_block(assim: dict[str, object], *, path: str) -> tuple[str, str]:
    """Parse uncertainty assimilation block for sigma mode + aggregate metric."""
    sigma_mode = require_nonempty_str(assim, "sigma_mode", path=path).lower()
    if sigma_mode not in SIGMA_MODES:
        raise ValueError(f"{path}.sigma_mode must be one of: formula, uncertainty_layer")
    aggregate_metric = require_nonempty_str(assim, "aggregate_metric", path=path)
    return sigma_mode, aggregate_metric


def normalize_netcdf_times(time_values: object, *, source_name: str) -> pd.DatetimeIndex:
    """Normalize arbitrary time values to unique day-keyed UTC index."""
    times = pd.to_datetime(time_values, utc=True, errors="raise")
    if np.asarray(times).ndim == 0:
        times = pd.DatetimeIndex([times])
    else:
        times = pd.DatetimeIndex(times)
    day_keys = [ts.date().isoformat() for ts in times]
    if len(day_keys) != len(set(day_keys)):
        raise ValueError(
            f"Ambiguous NetCDF timesteps in {source_name}: multiple timesteps map to the same day"
        )
    return times


def assert_same_grid(
    src: rasterio.DatasetReader,
    other: rasterio.DatasetReader,
    *,
    left: Path | str,
    right: Path | str,
) -> None:
    """Validate CRS/transform/shape equality between source and uncertainty rasters."""
    left_name = Path(str(left)).name
    right_name = Path(str(right)).name
    if src.crs != other.crs:
        raise ValueError(f"CRS mismatch between {left_name} and {right_name}")
    if src.transform != other.transform:
        raise ValueError(f"Transform mismatch between {left_name} and {right_name}")
    if src.shape != other.shape:
        raise ValueError(f"Shape mismatch between {left_name} and {right_name}")
