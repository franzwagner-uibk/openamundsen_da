"""Raster helpers for observation products.

The observation pipeline primarily works with rasterio datasets. Some NetCDF
products, including EURAC SnowFLAKES v3 files, expose CRS information through
CF/xarray metadata while GDAL subdatasets report no CRS. This module
materializes those NetCDF variables as in-memory GeoTIFF datasets so existing
ROI masking, reprojection and merging code can keep using rasterio.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio.io import DatasetReader, MemoryFile
from rasterio.transform import from_bounds


_NETCDF_SUFFIXES = {".nc", ".nc4", ".netcdf"}
_SPATIAL_DIMS = {"x", "y"}


def is_netcdf_path(path: str | Path) -> bool:
    """Return True when *path* has a supported NetCDF suffix."""

    return Path(path).suffix.lower() in _NETCDF_SUFFIXES


def crs_from_netcdf(ds: xr.Dataset, *, variable: str | None = None) -> CRS | None:
    """Return CRS metadata from a CF-style NetCDF dataset."""

    candidates: list[str] = []
    if variable is not None and variable in ds:
        grid_mapping = ds[variable].attrs.get("grid_mapping")
        if grid_mapping:
            candidates.append(str(grid_mapping))
    candidates.extend(["crs", "spatial_ref"])
    candidates.extend(str(name) for name in ds.data_vars if str(name) not in candidates)

    for name in candidates:
        if name not in ds:
            continue
        attrs = ds[name].attrs
        try:
            return CRS.from_cf(attrs)
        except Exception:
            pass
        for key in ("crs_wkt", "spatial_ref"):
            value = attrs.get(key)
            if not value:
                continue
            try:
                return CRS.from_user_input(value)
            except Exception:
                continue
    return None


def _slice_dim(da: xr.DataArray) -> str | None:
    dims = [str(dim) for dim in da.dims if str(dim) not in _SPATIAL_DIMS]
    if not dims:
        return None
    non_singleton = [dim for dim in dims if int(da.sizes[dim]) > 1]
    if len(non_singleton) > 1:
        raise ValueError(f"NetCDF variable '{da.name}' has multiple non-spatial dimensions: {non_singleton}")
    return non_singleton[0] if non_singleton else dims[0]


def netcdf_variable_slice_count(path: str | Path, variable: str) -> int:
    """Return the number of raster slices for one NetCDF variable."""

    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in {Path(path).name}")
        da = ds[variable]
        dim = _slice_dim(da)
        return int(da.sizes[dim]) if dim is not None else 1


def netcdf_band_index_for_token(
    path: str | Path,
    *,
    token: str,
    time_variable: str | None,
) -> int:
    """Return a 1-based NetCDF band index selected by a source token timestamp."""

    if "@" not in str(token):
        return 1
    if not time_variable:
        return 1

    target_raw = str(token).split("@", 1)[1].strip()
    if not target_raw:
        return 1
    target = pd.Timestamp(target_raw)
    if target.tzinfo is not None:
        target = target.tz_convert(None)

    with xr.open_dataset(path) as ds:
        if time_variable not in ds:
            return 1
        times = pd.to_datetime(ds[time_variable].values)
        if len(times) == 0:
            return 1
        normalized = []
        for value in times:
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            normalized.append(ts)

    for idx, ts in enumerate(normalized, start=1):
        if ts == target:
            return idx
    target_day = target.normalize()
    matches = [idx for idx, ts in enumerate(normalized, start=1) if ts.normalize() == target_day]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous NetCDF timestamp token {target_raw!r} in {Path(path).name}")
    raise ValueError(f"NetCDF timestamp token {target_raw!r} not found in {Path(path).name}")


def _nodata_value(da: xr.DataArray) -> float | int | None:
    for source in (da.encoding, da.attrs):
        for key in ("_FillValue", "missing_value"):
            if key not in source:
                continue
            value = source[key]
            if np.ndim(value) > 0:
                value = np.asarray(value).reshape(-1)[0]
            try:
                return float(value)
            except Exception:
                return None
    return None


def _netcdf_slice(path: str | Path, variable: str, *, band_index: int) -> tuple[np.ndarray, object, CRS | None, float | int | None]:
    with xr.open_dataset(path) as ds:
        if variable not in ds:
            raise ValueError(f"Variable '{variable}' not found in {Path(path).name}")
        da = ds[variable]
        dim = _slice_dim(da)
        if dim is not None:
            if band_index < 1 or band_index > int(da.sizes[dim]):
                raise IndexError(
                    f"Band index {band_index} out of range for variable '{variable}' in {Path(path).name}"
                )
            da = da.isel({dim: band_index - 1})
        if "y" not in da.dims or "x" not in da.dims:
            raise ValueError(f"Variable '{variable}' in {Path(path).name} must have 'y' and 'x' dimensions")
        da = da.transpose("y", "x")
        data = np.asarray(da.values, dtype=np.float32)
        x = np.asarray(ds["x"].values, dtype=float)
        y = np.asarray(ds["y"].values, dtype=float)
        if x.size < 2 or y.size < 2:
            raise ValueError(f"Insufficient x/y coordinate metadata in {Path(path).name}")

        if x[0] > x[-1]:
            data = data[:, ::-1]
            x = x[::-1]
        if y[0] < y[-1]:
            data = data[::-1, :]
            y = y[::-1]

        dx = float(abs(np.mean(np.diff(x))))
        dy = float(abs(np.mean(np.diff(y))))
        transform = from_bounds(
            float(x.min() - dx / 2.0),
            float(y.min() - dy / 2.0),
            float(x.max() + dx / 2.0),
            float(y.max() + dy / 2.0),
            data.shape[1],
            data.shape[0],
        )
        crs = crs_from_netcdf(ds, variable=variable)
        nodata = _nodata_value(da)

    return data, transform, crs, nodata


@contextmanager
def open_netcdf_variable_raster(
    path: str | Path,
    *,
    variable: str,
    band_index: int = 1,
) -> Iterator[DatasetReader]:
    """Open one NetCDF variable slice as an in-memory rasterio dataset."""

    data, transform, crs, nodata = _netcdf_slice(path, variable, band_index=band_index)
    profile = {
        "driver": "GTiff",
        "height": int(data.shape[0]),
        "width": int(data.shape[1]),
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
    }
    memfile = MemoryFile()
    try:
        with memfile.open(**profile) as dst:
            dst.write(data.astype(np.float32, copy=False), 1)
        with memfile.open() as src:
            yield src
    finally:
        memfile.close()


@contextmanager
def open_raster_or_netcdf_variable(
    path: str | Path,
    *,
    variable: str | None = None,
    band_index: int = 1,
) -> Iterator[DatasetReader]:
    """Open a GeoTIFF/raster or NetCDF variable as a rasterio dataset."""

    path = Path(path)
    if is_netcdf_path(path):
        if variable is None:
            raise ValueError(f"NetCDF source requires an explicit variable name: {path}")
        with open_netcdf_variable_raster(path, variable=variable, band_index=band_index) as src:
            yield src
        return

    with rasterio.open(path) as src:
        yield src

