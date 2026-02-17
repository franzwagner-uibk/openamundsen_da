"""DA output-grid summaries and compact retention helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr
from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml


def output_retention_mode(project_dir: Path) -> str:
    """Return output retention mode from project YAML, defaulting to compact."""
    try:
        cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
        da_cfg = cfg.get("data_assimilation") or {}
        out_cfg = da_cfg.get("output") or {}
        mode = str(out_cfg.get("retention", "compact")).strip().lower()
        if mode in {"compact", "full"}:
            return mode
        logger.warning(
            "Unknown data_assimilation.output.retention='{}' in {}; using 'compact'",
            mode,
            project_dir,
        )
        return "compact"
    except Exception:
        return "compact"


def _as_nan_array(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values, dtype=np.float64)
    fill = da.attrs.get("_FillValue", np.nan)
    if not (isinstance(fill, float) and np.isnan(fill)):
        arr = np.where(arr == fill, np.nan, arr)
    return arr


def write_da_output_grids(
    *,
    open_loop_nc: Path,
    member_ncs: Sequence[Path],
    output_nc: Path,
) -> Path | None:
    """Write compact DA summary grids into a single NetCDF file."""
    if not open_loop_nc.is_file():
        logger.warning("DA output summary skipped: open_loop NetCDF not found at {}", open_loop_nc)
        return None
    member_files = [Path(p) for p in member_ncs if Path(p).is_file()]
    if not member_files:
        logger.warning("DA output summary skipped: no member NetCDF files provided")
        return None

    with xr.open_dataset(open_loop_nc) as ds_ol:
        out_vars: dict[str, xr.DataArray] = {}
        encoding: dict[str, dict] = {}
        n_members = int(len(member_files))
        grid_var_names = []
        for var_name, da_ol in ds_ol.data_vars.items():
            if "y" not in da_ol.dims or "x" not in da_ol.dims:
                continue
            grid_var_names.append(var_name)
            shape = tuple(int(s) for s in da_ol.shape)
            arr_sum = np.zeros(shape, dtype=np.float64)
            arr_sum_sq = np.zeros(shape, dtype=np.float64)
            arr_count = np.zeros(shape, dtype=np.int32)
            arr_min = np.full(shape, np.nan, dtype=np.float64)
            arr_max = np.full(shape, np.nan, dtype=np.float64)

            for nc_path in member_files:
                with xr.open_dataset(nc_path) as ds_m:
                    if var_name not in ds_m.data_vars:
                        logger.warning("Variable {} missing in {}", var_name, nc_path)
                        continue
                    da_m = ds_m[var_name]
                    if tuple(int(s) for s in da_m.shape) != shape:
                        logger.warning(
                            "Variable {} shape mismatch for {} (expected {}, got {})",
                            var_name,
                            nc_path,
                            shape,
                            tuple(int(s) for s in da_m.shape),
                        )
                        continue
                    arr = _as_nan_array(da_m)
                    valid = np.isfinite(arr)
                    if not np.any(valid):
                        continue
                    arr_sum[valid] += arr[valid]
                    arr_sum_sq[valid] += arr[valid] * arr[valid]
                    arr_count[valid] += 1
                    arr_min = np.where(np.isnan(arr_min), arr, np.fmin(arr_min, arr))
                    arr_max = np.where(np.isnan(arr_max), arr, np.fmax(arr_max, arr))

            with np.errstate(invalid="ignore", divide="ignore"):
                arr_mean = np.full(shape, np.nan, dtype=np.float64)
                np.divide(arr_sum, arr_count, out=arr_mean, where=arr_count > 0)
                arr_second_moment = np.full(shape, np.nan, dtype=np.float64)
                np.divide(arr_sum_sq, arr_count, out=arr_second_moment, where=arr_count > 0)
                arr_var = arr_second_moment - (arr_mean * arr_mean)
            arr_mean = np.where(arr_count > 0, arr_mean, np.nan)
            arr_var = np.where(arr_count > 0, arr_var, np.nan)
            arr_var = np.where(arr_var < 0, 0.0, arr_var)
            arr_std = np.sqrt(arr_var)
            arr_ol = _as_nan_array(da_ol)
            arr_inc = arr_mean - arr_ol

            dims = da_ol.dims
            coords = {d: ds_ol.coords[d] for d in dims if d in ds_ol.coords}
            out_vars[f"open_loop_{var_name}"] = xr.DataArray(arr_ol.astype(np.float32), dims=dims, coords=coords)
            out_vars[f"da_mean_{var_name}"] = xr.DataArray(arr_mean.astype(np.float32), dims=dims, coords=coords)
            out_vars[f"da_std_{var_name}"] = xr.DataArray(arr_std.astype(np.float32), dims=dims, coords=coords)
            out_vars[f"da_min_{var_name}"] = xr.DataArray(arr_min.astype(np.float32), dims=dims, coords=coords)
            out_vars[f"da_max_{var_name}"] = xr.DataArray(arr_max.astype(np.float32), dims=dims, coords=coords)
            out_vars[f"da_increment_{var_name}"] = xr.DataArray(arr_inc.astype(np.float32), dims=dims, coords=coords)
            for out_name in (
                f"open_loop_{var_name}",
                f"da_mean_{var_name}",
                f"da_std_{var_name}",
                f"da_min_{var_name}",
                f"da_max_{var_name}",
                f"da_increment_{var_name}",
            ):
                encoding[out_name] = {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": -9999.0}

        if not out_vars:
            logger.warning("DA output summary skipped: no grid variables with x/y dims in {}", open_loop_nc)
            return None

        out_ds = xr.Dataset(
            data_vars=out_vars,
            coords=ds_ol.coords,
            attrs={
                **(dict(ds_ol.attrs) if ds_ol.attrs is not None else {}),
                "da_output_version": "1",
                "source_open_loop_nc": str(open_loop_nc),
                "source_member_count": str(n_members),
                "source_member_weighting": "uniform",
                "source_grid_variables": ",".join(grid_var_names),
            },
        )
        output_nc.parent.mkdir(parents=True, exist_ok=True)
        out_ds.to_netcdf(output_nc, encoding=encoding)
        logger.info("Wrote DA output summary NetCDF {}", output_nc)
        return output_nc


def delete_files(paths: Iterable[Path]) -> tuple[int, int]:
    """Delete files best-effort, returning (count, bytes_freed)."""
    deleted = 0
    bytes_freed = 0
    seen: set[Path] = set()
    for p in paths:
        path = Path(p)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        try:
            path.unlink()
            deleted += 1
            bytes_freed += size
        except Exception as exc:
            logger.warning("Could not delete {}: {}", path, exc)
    return deleted, bytes_freed


def collect_project_grid_artifacts(project_dir: Path) -> list[Path]:
    """Collect step-level grid artifacts for compact retention cleanup."""
    project_dir = Path(project_dir)
    patterns = (
        "steps/step_*/ensembles/prior/member_*/results/output_grids*.nc",
        "steps/step_*/ensembles/prior/open_loop/results/output_grids*.nc",
        "steps/step_*/ensembles/prior/member_*/results/**/*.tif",
        "steps/step_*/ensembles/prior/open_loop/results/**/*.tif",
        "steps/step_*/ensembles/posterior/member_*/results/output_grids*.nc",
        "steps/step_*/ensembles/posterior/member_*/results/**/*.tif",
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for patt in patterns:
        for path in project_dir.glob(patt):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def collect_subdomain_grid_artifacts(project_dir: Path) -> list[Path]:
    """Collect sub-domain step-level grid artifacts for compact retention."""
    project_dir = Path(project_dir)
    patterns = (
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/member_*/results/output_grids*.nc",
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/open_loop/results/output_grids*.nc",
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/member_*/results/**/*.tif",
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/open_loop/results/**/*.tif",
        "subdomains/*/projects/*/steps/step_*/ensembles/posterior/member_*/results/output_grids*.nc",
        "subdomains/*/projects/*/steps/step_*/ensembles/posterior/member_*/results/**/*.tif",
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for patt in patterns:
        for path in project_dir.glob(patt):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return files
