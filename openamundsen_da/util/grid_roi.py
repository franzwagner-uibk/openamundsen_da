"""Grid-to-ROI helpers shared across model diagnostics and plotting.

This module centralizes the common logic for:

- opening daily grid slices from GeoTIFF or NetCDF-backed ``GridSlice`` inputs,
- clipping them to a ROI polygon,
- optionally applying the configured land-cover exclusions, and
- returning a masked array ready for downstream AOI/ROI statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
import xarray as xr
from pyproj import CRS
from rasterio import features
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from openamundsen_da.io.paths import GridSlice
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig, apply_landcover_mask
from openamundsen_da.util.roi import read_single_roi


def _crs_from_netcdf(ds: xr.Dataset) -> CRS | None:
    """Return CRS metadata from an openAMUNDSEN-style NetCDF dataset."""
    if "crs" not in ds:
        return None

    crs_var = ds["crs"]
    try:
        return CRS.from_cf(crs_var.attrs)
    except Exception:
        pass

    for key in ("crs_wkt", "spatial_ref"):
        value = crs_var.attrs.get(key)
        if value:
            try:
                return CRS.from_user_input(value)
            except Exception:
                continue
    return None


def _open_netcdf_slice(gs: GridSlice) -> MemoryFile:
    """Materialize one NetCDF grid slice into an in-memory raster."""
    if not gs.nc_var:
        raise ValueError("NetCDF grid slice is missing nc_var")

    with xr.open_dataset(gs.path) as ds:
        if gs.nc_var not in ds:
            raise FileNotFoundError(f"Variable {gs.nc_var} not found in {gs.path}")
        da = ds[gs.nc_var]
        time_dims = [d for d in da.dims if d.startswith("time")]
        if time_dims:
            da = da.isel({time_dims[0]: gs.band - 1})
        data = np.asarray(da.values, dtype=np.float32)
        if data.ndim > 2:
            data = data.reshape(data.shape[-2], data.shape[-1])
        x = np.asarray(ds["x"].values)
        y = np.asarray(ds["y"].values)
        if x.size < 2 or y.size < 2:
            raise ValueError("Insufficient coordinate metadata in NetCDF grid")
        dx = float(np.abs(np.mean(np.diff(x))))
        dy = float(np.abs(np.mean(np.diff(y))))
        transform = from_bounds(
            float(x.min() - dx / 2),
            float(y.min() - dy / 2),
            float(x.max() + dx / 2),
            float(y.max() + dy / 2),
            data.shape[1],
            data.shape[0],
        )
        crs = _crs_from_netcdf(ds)
        nodata = da.encoding.get("_FillValue")

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
    }
    memfile = MemoryFile()
    with memfile.open(**profile) as dst:
        dst.write(data.astype(np.float32), 1)
    return memfile


def valid_mask(data: np.ma.MaskedArray) -> np.ndarray:
    """Return boolean mask of valid (non-masked, finite) pixels."""
    arr = np.ma.array(data, copy=False)
    return (~arr.mask) & np.isfinite(arr)


def read_grid_slice_roi_masked_array(
    raster: Path | GridSlice,
    roi_path: Path,
    *,
    landcover_cfg: LandcoverMaskConfig | None = None,
) -> np.ma.MaskedArray:
    """Read one grid slice and return its ROI-clipped masked array.

    Parameters
    ----------
    raster : Path or GridSlice
        GeoTIFF path or logical grid-slice descriptor resolved by
        ``find_member_daily_grid_slice``.
    roi_path : Path
        ROI vector file. Multiple features are unioned internally.
    landcover_cfg : LandcoverMaskConfig, optional
        When provided and enabled, apply land-cover exclusions after ROI
        clipping. When omitted, the full ROI footprint is retained.
    """

    mem: MemoryFile | None = None
    src_ctx = None
    if isinstance(raster, GridSlice):
        if raster.kind == "netcdf":
            mem = _open_netcdf_slice(raster)
            src_ctx = mem.open()
            url = None
            indexes = 1
        else:
            url = str(raster.path)
            indexes = 1
    else:
        url = str(raster)
        indexes = 1

    src_mgr = rasterio.open(url) if url is not None else src_ctx  # type: ignore[arg-type]
    try:
        with src_mgr as src:
            if src.crs is None:
                raise ValueError("Raster has no CRS; unable to align with ROI")
            gdf, _ = read_single_roi(
                roi_path,
                required_field=None,
                to_crs=src.crs,
            )
            shapes: Iterable = gdf.geometry
            roi_mask = features.geometry_mask(
                shapes,
                out_shape=(src.height, src.width),
                transform=src.transform,
                invert=True,
            )
            raw = src.read(indexes, masked=False)
            if raw.ndim == 3:
                raw = raw[0]
            mask = ~roi_mask
            if src.nodata is not None:
                mask = mask | (raw == src.nodata)
            arr = np.ma.array(raw, mask=mask, copy=False)
            if landcover_cfg is not None and landcover_cfg.enabled:
                arr, _ = apply_landcover_mask(
                    arr,
                    transform=src.transform,
                    target_crs=src.crs,
                    roi_mask=roi_mask,
                    lc_cfg=landcover_cfg,
                )
            if not np.any(valid_mask(arr)):
                raise ValueError(f"ROI contains no valid pixels for raster {getattr(raster, 'path', raster)}")
            return arr
    finally:
        if mem is not None:
            mem.close()
