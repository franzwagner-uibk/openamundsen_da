"""Internal storage precision policy for DA-owned persisted artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr


METEO_CSV_DECIMALS: dict[str, int] = {
    "temp": 1,
    "precip": 2,
    "sw_in": 0,
    "rel_hum": 2,
    "wind_speed": 2,
    "wind_dir": 1,
}

DA_SUMMARY_NC_FILL_VALUE = np.int16(-32768)
PERCENT_UINT8_NODATA = np.uint8(255)


@dataclass(frozen=True)
class NetcdfEncodingSpec:
    suffix: str
    scale_factor: float


DA_SUMMARY_NC_SPECS: tuple[NetcdfEncodingSpec, ...] = (
    NetcdfEncodingSpec("snowdepth_daily", 0.001),
    NetcdfEncodingSpec("swe_daily", 1.0),
    NetcdfEncodingSpec("liquid_water_content", 1.0),
)


def apply_meteo_csv_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with known meteo columns formatted to storage precision."""
    out = df.copy()
    for column, decimals in METEO_CSV_DECIMALS.items():
        if column in out.columns:
            out[column] = _format_numeric_series(out[column], decimals)
    return out


def _format_numeric_series(series: pd.Series, decimals: int) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    finite = numeric.notna()
    out = series.astype(object).copy()
    if not finite.any():
        return out

    fmt = f"{{:.{decimals}f}}"
    rounded = numeric.loc[finite].round(decimals)
    out.loc[finite] = rounded.map(lambda value: fmt.format(_normalize_zero(float(value), decimals)))
    return out


def _normalize_zero(value: float, decimals: int) -> float:
    threshold = 0.5 * (10.0 ** (-decimals))
    if abs(value) < threshold:
        return 0.0
    return value


def da_summary_grid_scale_factor(name: str) -> float | None:
    """Return the compact storage scale factor for a DA summary variable."""
    for spec in DA_SUMMARY_NC_SPECS:
        if name == spec.suffix or name.endswith(f"_{spec.suffix}"):
            return spec.scale_factor
    return None


def da_summary_netcdf_encoding(ds: "xr.Dataset") -> dict[str, dict]:
    """Build NetCDF encodings for DA summary grids and validate int16 ranges."""
    encoding: dict[str, dict] = {}
    for name, da in ds.data_vars.items():
        is_grid_payload = "y" in da.dims and "x" in da.dims
        if not is_grid_payload:
            continue
        scale_factor = da_summary_grid_scale_factor(name)
        if scale_factor is None:
            encoding[name] = _float32_netcdf_encoding()
            continue
        _validate_int16_scaled_range(name=name, values=da.values, scale_factor=scale_factor)
        encoding[name] = _scaled_int16_netcdf_encoding(scale_factor)
    return encoding


def preserved_netcdf_encoding(source_encoding: dict) -> dict:
    """Return writer-safe storage keys from an existing xarray encoding."""
    allowed = {
        "_FillValue",
        "add_offset",
        "chunksizes",
        "complevel",
        "contiguous",
        "dtype",
        "fletcher32",
        "scale_factor",
        "shuffle",
        "zlib",
    }
    return {key: value for key, value in source_encoding.items() if key in allowed and value is not None}


def percent_to_uint8_nodata(values: np.ndarray, *, nodata_value: float | int | None = None) -> np.ndarray:
    """Encode a 0..100 percent raster as uint8 with 255 nodata."""
    arr = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(arr)
    if nodata_value is not None and np.isfinite(float(nodata_value)):
        valid &= arr != float(nodata_value)

    out = np.full(arr.shape, PERCENT_UINT8_NODATA, dtype=np.uint8)
    if np.any(valid):
        out[valid] = np.rint(np.clip(arr[valid], 0.0, 100.0)).astype(np.uint8)
    return out


def _float32_netcdf_encoding() -> dict:
    return {
        "dtype": "float32",
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "_FillValue": np.float32(-9999.0),
    }


def _scaled_int16_netcdf_encoding(scale_factor: float) -> dict:
    return {
        "dtype": "int16",
        "scale_factor": np.float32(scale_factor),
        "add_offset": np.float32(0.0),
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "_FillValue": DA_SUMMARY_NC_FILL_VALUE,
    }


def _validate_int16_scaled_range(*, name: str, values: np.ndarray, scale_factor: float) -> None:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(arr)
    if not np.any(valid):
        return

    scaled = np.rint(arr[valid] / float(scale_factor))
    min_allowed = int(np.iinfo(np.int16).min) + 1
    max_allowed = int(np.iinfo(np.int16).max)
    scaled_min = float(np.nanmin(scaled))
    scaled_max = float(np.nanmax(scaled))
    if scaled_min < min_allowed or scaled_max > max_allowed:
        raise ValueError(
            f"{name} exceeds compact int16 NetCDF storage range with "
            f"scale_factor={scale_factor:g}: scaled range {scaled_min:g}..{scaled_max:g}, "
            f"allowed {min_allowed}..{max_allowed}"
        )
