"""Merge per-subregion outputs back to a global mosaic."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import shutil

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from loguru import logger

from openamundsen_da.batch.manifest import BatchManifest, SubregionMeta


def _load_roi(sub: SubregionMeta) -> np.ndarray:
    """Return ROI mask as boolean array."""
    with rasterio.open(sub.roi_path) as ds:
        return ds.read(1).astype(bool)


def _window_slices(sub: SubregionMeta, data_shape: Tuple[int, int], global_shape: Tuple[int, int]) -> Tuple[slice, slice]:
    """Return row/col slices placing subregion data into the global array."""
    if data_shape == global_shape:
        return (slice(0, global_shape[0]), slice(0, global_shape[1]))
    return (
        slice(sub.window.row_off, sub.window.row_off + data_shape[0]),
        slice(sub.window.col_off, sub.window.col_off + data_shape[1]),
    )


def merge_grids(
    *,
    manifest_path: Path,
    mode: str = "hard_clip",
    subregions: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """Merge gridded outputs (GeoTIFF or NetCDF) across subregions."""
    if mode not in {"hard_clip", "blend"}:
        raise ValueError("mode must be 'hard_clip' or 'blend'")

    manifest = BatchManifest.load(manifest_path)
    selected_ids = list(subregions) if subregions else list(manifest.subregions.keys())
    global_shape = (manifest.grid_rows, manifest.grid_cols)
    global_transform = Affine(*manifest.grid_transform)

    out_base = out_dir or (manifest_path.parent / "merged" / "grids")
    out_base.mkdir(parents=True, exist_ok=True)

    # Determine available outputs
    tif_groups: Dict[str, List[Tuple[SubregionMeta, Path]]] = {}
    nc_paths: List[Tuple[SubregionMeta, Path]] = []
    for sid in selected_ids:
        sub = manifest.subregions[sid]
        res_dir = sub.results_dir
        if not res_dir.is_dir():
            logger.warning("Results dir missing for {}: {}", sid, res_dir)
            continue
        for tif in sorted(res_dir.glob("*.tif")):
            tif_groups.setdefault(tif.name, []).append((sub, tif))
        for nc in sorted(res_dir.glob("*.nc")):
            if nc.name.startswith("output_grids"):
                nc_paths.append((sub, nc))

    written: List[Path] = []
    if tif_groups:
        written.extend(_merge_tifs(tif_groups, global_shape, global_transform, manifest.crs, mode, out_base))
    if nc_paths:
        written.append(_merge_netcdf(nc_paths, global_shape, manifest, mode, out_base))
    return written


def _merge_tifs(
    tif_groups: Dict[str, List[Tuple[SubregionMeta, Path]]],
    global_shape: Tuple[int, int],
    transform: Affine,
    crs: Optional[str],
    mode: str,
    out_dir: Path,
) -> List[Path]:
    outputs: List[Path] = []
    for fname, entries in sorted(tif_groups.items()):
        # use dtype from first file
        with rasterio.open(entries[0][1]) as ds0:
            dtype = ds0.dtypes[0]
            nodata = ds0.nodata

        data_global = np.full(global_shape, np.nan, dtype=float)
        weight_global = np.zeros(global_shape, dtype=float) if mode == "blend" else None

        for sub, tif_path in entries:
            roi = _load_roi(sub)
            with rasterio.open(tif_path) as ds:
                arr = ds.read(1).astype(float)
                if ds.nodata is not None:
                    arr[arr == ds.nodata] = np.nan
            sl_r, sl_c = _window_slices(sub, arr.shape, global_shape)
            mask = roi
            if mask.shape != arr.shape:
                # Resize mask if it is global-sized while data is cropped
                if mask.shape == global_shape:
                    mask = mask[sl_r, sl_c]
                else:
                    logger.warning("ROI shape {} does not match data {} for {}", mask.shape, arr.shape, sub.id)
                    mask = np.ones_like(arr, dtype=bool)

            arr = np.where(mask, arr, np.nan)

            if mode == "hard_clip":
                dest = data_global[sl_r, sl_c]
                replace = np.isnan(dest) & ~np.isnan(arr)
                dest[replace] = arr[replace]
                data_global[sl_r, sl_c] = dest
            else:
                dest = data_global[sl_r, sl_c]
                wdest = weight_global[sl_r, sl_c] if weight_global is not None else None
                valid = ~np.isnan(arr)
                dest[valid] += arr[valid]
                if wdest is not None:
                    wdest[valid] += 1.0
                data_global[sl_r, sl_c] = dest
                if weight_global is not None:
                    weight_global[sl_r, sl_c] = wdest

        if mode == "blend" and weight_global is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                data_global = np.where(weight_global > 0, data_global / np.maximum(weight_global, 1e-6), np.nan)

        out_path = out_dir / fname
        nd = nodata if nodata is not None else (-9999.0 if np.issubdtype(np.dtype(dtype), np.floating) else -9999)
        meta = {
            "driver": "GTiff",
            "dtype": dtype,
            "nodata": nd,
            "width": global_shape[1],
            "height": global_shape[0],
            "count": 1,
            "crs": crs,
            "transform": transform,
            "compress": "lzw",
        }
        with rasterio.open(out_path, "w", **meta) as dst:
            write_arr = data_global
            if np.issubdtype(np.dtype(dtype), np.integer):
                fill_val = nd if nd is not None else -9999
                write_arr = np.where(np.isnan(data_global), fill_val, np.round(data_global)).astype(dtype)
                dst.write(write_arr, 1)
                dst.nodata = fill_val
            else:
                dst.write(np.where(np.isnan(data_global), nd, data_global).astype(dtype), 1)
        outputs.append(out_path)
        logger.info("Wrote merged grid {}", out_path)
    return outputs


def _merge_netcdf(
    nc_paths: List[Tuple[SubregionMeta, Path]],
    global_shape: Tuple[int, int],
    manifest: BatchManifest,
    mode: str,
    out_dir: Path,
) -> Path:
    base_sub, base_nc = nc_paths[0]
    ds_template = xr.open_dataset(base_nc)

    rows, cols = global_shape
    transform = Affine(*manifest.grid_transform)
    x_range, y_range = rasterio.transform.xy(transform, [0, rows - 1], [0, cols - 1])
    xs = np.linspace(x_range[0], x_range[1], cols)
    ys = np.linspace(y_range[0], y_range[1], rows)

    coords = dict(ds_template.coords)
    coords["x"] = ("x", xs)
    coords["y"] = ("y", ys)

    data_vars: Dict[str, Dict] = {}
    weights: Dict[str, np.ndarray] = {}

    for name, da in ds_template.data_vars.items():
        y_idx = da.dims.index("y")
        x_idx = da.dims.index("x")
        shape = list(da.shape)
        shape[y_idx] = rows
        shape[x_idx] = cols
        fill_raw = da.attrs.get("_FillValue")
        fill = np.nan if fill_raw is None else fill_raw
        if not np.issubdtype(da.dtype, np.floating) and (isinstance(fill, float) and np.isnan(fill)):
            fill = -9999
        data_vars[name] = {
            "array": np.full(shape, np.nan, dtype=np.float32),
            "fill": fill,
            "dims": da.dims,
            "attrs": da.attrs,
            "dtype": da.dtype,
            "y_idx": y_idx,
            "x_idx": x_idx,
        }
        if mode == "blend":
            weights[name] = np.zeros(shape, dtype=np.float32)

    for sub, nc_path in nc_paths:
        roi = _load_roi(sub)
        ds = xr.open_dataset(nc_path)
        for name, da in ds.data_vars.items():
            if name not in data_vars:
                continue
            info = data_vars[name]
            arr = da.values.astype(np.float32)
            fill_val_raw = da.attrs.get("_FillValue")
            fill_val = np.nan if fill_val_raw is None else fill_val_raw
            if not np.issubdtype(info["dtype"], np.floating) and (isinstance(fill_val, float) and np.isnan(fill_val)):
                fill_val = info["fill"]
            if not (isinstance(fill_val, float) and np.isnan(fill_val)):
                arr = np.where(arr == fill_val, np.nan, arr)

            y_idx = info["y_idx"]
            x_idx = info["x_idx"]
            sl_r, sl_c = _window_slices(sub, (arr.shape[y_idx], arr.shape[x_idx]), global_shape)

            mask = roi
            if mask.shape != (arr.shape[y_idx], arr.shape[x_idx]):
                if mask.shape == global_shape:
                    mask = mask[sl_r, sl_c]
                else:
                    mask = np.ones((arr.shape[y_idx], arr.shape[x_idx]), dtype=bool)

            mask_nd = mask
            while mask_nd.ndim < arr.ndim:
                mask_nd = np.expand_dims(mask_nd, axis=0)
            mask_nd = np.broadcast_to(mask_nd, arr.shape)
            arr = np.where(mask_nd, arr, np.nan)

            slice_obj = [slice(None)] * arr.ndim
            slice_obj[y_idx] = slice(sl_r.start, sl_r.stop)
            slice_obj[x_idx] = slice(sl_c.start, sl_c.stop)

            target = info["array"]
            dest = target[tuple(slice_obj)]

            if mode == "hard_clip":
                replace = np.isnan(dest) & ~np.isnan(arr)
                dest[replace] = arr[replace]
            else:
                valid = ~np.isnan(arr)
                dest_valid = np.where(np.isnan(dest), 0.0, dest)
                dest_valid[valid] += arr[valid]
                dest = dest_valid
                w = weights[name]
                w_slice = w[tuple(slice_obj)]
                w_slice[valid] += 1.0
                w[tuple(slice_obj)] = w_slice
                weights[name] = w

            target[tuple(slice_obj)] = dest
            info["array"] = target
        ds.close()

    if mode == "blend":
        for name, w in weights.items():
            arr = data_vars[name]["array"]
            data_vars[name]["array"] = np.where(w > 0, arr / np.maximum(w, 1e-6), arr)

    out_data_vars = {}
    for name, info in data_vars.items():
        arr = info["array"]
        fill_val = info["fill"]
        if not (isinstance(fill_val, float) and np.isnan(fill_val)):
            arr = np.where(np.isnan(arr), fill_val, arr)
        out_data_vars[name] = xr.DataArray(
            arr.astype(info["dtype"]),
            dims=info["dims"],
            attrs=info["attrs"],
        )

    merged = xr.Dataset(data_vars=out_data_vars, coords=coords, attrs=ds_template.attrs)
    for name, info in data_vars.items():
        fill_val = info["fill"]
        if not np.isnan(fill_val):
            merged[name].encoding["_FillValue"] = fill_val
    out_path = out_dir / "output_grids.nc"
    merged.to_netcdf(out_path)
    ds_template.close()
    logger.info("Wrote merged NetCDF {}", out_path)
    return out_path


def merge_points(
    *,
    manifest_path: Path,
    subregions: Optional[Iterable[str]] = None,
    out_dir: Optional[Path] = None,
) -> List[Path]:
    """Collect point (timeseries) outputs into a common directory."""
    manifest = BatchManifest.load(manifest_path)
    selected_ids = list(subregions) if subregions else list(manifest.subregions.keys())
    out_base = out_dir or (manifest_path.parent / "merged" / "points")
    out_base.mkdir(parents=True, exist_ok=True)
    obs_out = out_base / "obs" / "stations"
    obs_out.mkdir(parents=True, exist_ok=True)

    copied: List[Path] = []

    # Merge station metadata (best-effort)
    stations_frames = []
    for sid in selected_ids:
        sub = manifest.subregions[sid]
        meta_path = sub.meteo_dir / "stations.csv"
        if meta_path.is_file():
            try:
                df = pd.read_csv(meta_path)
                stations_frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read stations.csv for {}: {}", sid, exc)
    if stations_frames:
        all_stations = pd.concat(stations_frames, ignore_index=True)
        all_stations = all_stations.drop_duplicates(subset="id")
        stations_out = out_base / "stations.csv"
        all_stations.to_csv(stations_out, index=False)
        copied.append(stations_out)

    # Merge observation station metadata (snow depth)
    obs_meta_frames = []
    for sid in selected_ids:
        sub = manifest.subregions[sid]
        meta_path = sub.obs_dir / "stations_snow_depth.csv"
        if meta_path.is_file():
            try:
                df = pd.read_csv(meta_path)
                obs_meta_frames.append(df)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read stations_snow_depth.csv for {}: {}", sid, exc)
    if obs_meta_frames:
        obs_meta = pd.concat(obs_meta_frames, ignore_index=True).drop_duplicates(subset="id")
        meta_out = obs_out / "stations_snow_depth.csv"
        meta_out.parent.mkdir(parents=True, exist_ok=True)
        obs_meta.to_csv(meta_out, index=False)
        copied.append(meta_out)

    seen_points: set[str] = set()
    for sid in selected_ids:
        sub = manifest.subregions[sid]
        res_dir = sub.results_dir
        if not res_dir.is_dir():
            continue
        for csv in sorted(res_dir.glob("point_*.csv")):
            if csv.name in seen_points:
                continue
            target = out_base / csv.name
            shutil.copy2(csv, target)
            seen_points.add(csv.name)
            copied.append(target)
        for nc in sorted(res_dir.glob("output_timeseries*.nc")):
            if nc.name in seen_points:
                continue
            target = out_base / nc.name
            shutil.copy2(nc, target)
            seen_points.add(nc.name)
            copied.append(target)
        # copy obs station files
        if sub.obs_dir.is_dir():
            for obs_file in sorted(sub.obs_dir.glob("*.csv")):
                target = obs_out / obs_file.name
                if target.exists():
                    continue
                shutil.copy2(obs_file, target)
                copied.append(target)

    logger.info("Merged point outputs into {}", out_base)
    return copied
