"""openamundsen_da.methods.h_of_x.model_scf

Model-based Snow Cover Fraction (SCF) operator H(x).

This module derives an areal SCF in a given AOI from openAMUNDSEN
distributed outputs (snow depth/HS or SWE) for a single date and member.

Two supported per-pixel operators (both averaged over the AOI):

- Depth threshold (deterministic)
  I_snow = 1 if X > h0 else 0, SCF = mean(I_snow)

- Logistic (probabilistic)
  p_snow = 1 / (1 + exp(-k * (X - h0))), SCF = mean(p_snow)

Where X is either snow depth (HS, meters) or SWE (e.g., mm). The parameters
"h0" and "k" are interpreted in the same units as X and its inverse.

Notes
- This operator intentionally mirrors simple satellite SCF processing
  (thresholding) but stays flexible via the logistic variant for stability
  in data assimilation.
- AOI geometry is reprojected to the raster CRS if needed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Tuple
import concurrent.futures as cf

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from loguru import logger
from rasterio import features
from rasterio.mask import mask as rio_mask
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds
from pyproj import CRS

from openamundsen_da.core.env import ensure_gdal_proj_from_conda, _read_yaml_file
from openamundsen_da.core.constants import (
    LOGURU_FORMAT,
    VAR_HS,
    VAR_SWE,
    HOFX_BLOCK,
    HOFX_METHOD,
    HOFX_VARIABLE,
    HOFX_PARAMS,
    HOFX_PARAM_H0,
    HOFX_PARAM_K,
    DA_BLOCK,
)
from openamundsen_da.io.paths import (
    GridSlice,
    find_member_daily_grid_slice,
    member_id_from_results_dir,
    find_setup_yaml,
    find_project_yaml,
    infer_project_dir,
    infer_setup_dir_from_project,
    read_step_config,
    list_member_dirs,
    open_loop_dir,
    list_step_dirs,
)
from openamundsen_da.util.landcover_mask import (
    LandcoverMaskConfig,
    apply_landcover_mask,
    resolve_landcover_mask,
)
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.stats import sigmoid
from openamundsen_da.methods.daily_aoi_series import (
    compute_step_daily_series_for_all_members,
    step_start_end,
)


Variable = Literal["hs", "swe"]
Method = Literal["depth_threshold", "logistic"]


@dataclass
class SCFParams:
    """SCF operator parameters.

    Attributes
    ----------
    h0 : float
        Midpoint/threshold in the same units as the input variable (m for HS,
        mm for SWE if your SWE raster uses mm). For the threshold method, this
        is the cut-off; for logistic, it's the 50% point.
    k : float
        Slope in 1/units of X. Larger k means a sharper transition around h0.
        Ignored by the depth_threshold method.
    """

    h0: float = 0.05
    k: float = 80.0


def _parse_hofx_block(hofx: dict[str, Any]) -> tuple[str, str, SCFParams]:
    """Return H(x) settings from a config block, applying defaults where keys are missing."""
    method = str(hofx.get(HOFX_METHOD, "depth_threshold"))
    variable = str(hofx.get(HOFX_VARIABLE, "hs"))
    params = SCFParams()
    plist = hofx.get(HOFX_PARAMS) or {}
    if isinstance(plist, dict):
        if HOFX_PARAM_H0 in plist:
            params.h0 = float(plist[HOFX_PARAM_H0])
        if HOFX_PARAM_K in plist:
            params.k = float(plist[HOFX_PARAM_K])
    return method, variable, params


def load_hofx_from_project(project_dir: Path) -> tuple[str, str, SCFParams]:
    """Read required H(x) configuration from project YAML."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    da = cfg.get(DA_BLOCK, {}) if isinstance(cfg, dict) else {}
    hofx = da.get(HOFX_BLOCK)
    if not isinstance(hofx, dict):
        hofx = {}
    if not hofx:
        raise ValueError(f"Missing '{DA_BLOCK}.{HOFX_BLOCK}' section in {project_yaml}")
    return _parse_hofx_block(hofx)


def load_hofx_from_setup(project_dir: Path) -> tuple[str, str, SCFParams]:
    """Backward-compatible alias; delegated to project-level config."""
    return load_hofx_from_project(project_dir)


def _grid_format_from_setup(setup_dir: Path) -> str | None:
    """Return output_data.grids.format from setup YAML (lowercase) if present."""
    try:
        proj = find_setup_yaml(setup_dir)
        cfg = _read_yaml_file(proj) or {}
        out_cfg = cfg.get("output_data", {}).get("grids", {})
        fmt = out_cfg.get("format")
        if fmt:
            f = str(fmt).lower().strip()
            if f in {"geotiff", "netcdf"}:
                return f
            if f == "ascii":
                # ASCII grids are not supported for DA readers; fall back to autodetect.
                return None
        return None
    except Exception:
        return None


def _serialize_lc_cfg(lc_cfg: LandcoverMaskConfig) -> dict[str, object]:
    """Return a dict-safe representation of the land-cover config for pickling."""
    return {
        "enabled": lc_cfg.enabled,
        "path": str(lc_cfg.path) if lc_cfg.path else None,
        "classes": tuple(lc_cfg.classes),
        "project_crs_wkt": lc_cfg.project_crs.to_wkt(),
    }


def _deserialize_lc_cfg(data: Any) -> LandcoverMaskConfig | None:
    """Reconstruct LandcoverMaskConfig from serialized form."""
    if isinstance(data, LandcoverMaskConfig):
        return data
    if not isinstance(data, dict):
        return None
    path_val = data.get("path")
    project_crs_wkt = data.get("project_crs_wkt")
    if project_crs_wkt is None:
        return None
    return LandcoverMaskConfig(
        enabled=bool(data.get("enabled", False)),
        path=Path(path_val) if path_val else None,
        classes=tuple(data.get("classes") or ()),
        project_crs=CRS.from_wkt(str(project_crs_wkt)),
    )


def _read_masked_array(
    raster: Path | GridSlice,
    aoi_path: Path,
    lc_cfg: LandcoverMaskConfig,
) -> np.ma.MaskedArray:
    """Read raster and mask by AOI geometry, applying land-cover exclusions."""
    def _open_netcdf_slice(gs: GridSlice):
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
            dx = float(np.mean(np.diff(x)))
            dy = float(np.mean(np.diff(y)))
            transform = from_bounds(
                float(x.min() - dx / 2),
                float(y.min() - dy / 2),
                float(x.max() + dx / 2),
                float(y.max() + dy / 2),
                data.shape[1],
                data.shape[0],
            )
            crs = None
            try:
                crs = CRS.from_cf(ds["crs"].attrs)
            except Exception:
                pass
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
    with src_mgr as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; unable to align with AOI")
        gdf, _ = read_single_roi(
            aoi_path,
            required_field=None,
            to_crs=src.crs,
        )
        shapes: Iterable = gdf.geometry
        geom_mask = features.geometry_mask(
            shapes,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        raw = src.read(indexes, masked=False)
        if raw.ndim == 3:
            raw = raw[0]
        mask = ~geom_mask
        if src.nodata is not None:
            mask = mask | (raw == src.nodata)
        arr = np.ma.array(raw, mask=mask, copy=False)
        arr, _ = apply_landcover_mask(
            arr,
            transform=src.transform,
            target_crs=src.crs,
            roi_mask=geom_mask,
            lc_cfg=lc_cfg,
        )
        valid = _valid_mask(arr)
        if not np.any(valid):
            logger.warning("AOI mask empty for %s; falling back to full grid.", url)
            arr = np.ma.array(raw, copy=False)
    if isinstance(raster, GridSlice) and raster.kind == "netcdf":
        mem.close()
    return arr


def _valid_mask(x: np.ma.MaskedArray) -> np.ndarray:
    """Return boolean mask of valid (non-masked, finite) pixels."""
    data = np.ma.array(x, copy=False)
    return (~data.mask) & np.isfinite(data)


def _scf_depth_threshold(x: np.ma.MaskedArray, h0: float) -> Tuple[int, int, float]:
    """Compute SCF using deterministic threshold (I = 1 if x > h0)."""
    valid = np.ma.array(_valid_mask(x), copy=False)
    n_valid = int(np.ma.filled(valid, False).sum())
    if n_valid == 0:
        raise ValueError("AOI contains no valid pixels for SCF computation")
    snow = np.ma.array(valid & (x > h0), copy=False)
    n_snow = int(np.ma.filled(snow, False).sum())
    scf = float(n_snow) / float(n_valid)
    return n_valid, n_snow, scf


def _scf_logistic(x: np.ma.MaskedArray, h0: float, k: float) -> Tuple[int, float]:
    """Compute SCF using logistic probability: mean(sigmoid(k * (x - h0)))."""
    valid = np.ma.array(_valid_mask(x), copy=False)
    n_valid = int(np.ma.filled(valid, False).sum())
    if n_valid == 0:
        raise ValueError("AOI contains no valid pixels for SCF computation")
    dx = np.clip((x - h0), a_min=-1e6, a_max=1e6)
    p = sigmoid(k * dx)
    scf = float(np.ma.array(p, copy=False)[valid].mean())
    return n_valid, scf


def compute_model_scf(
    *,
    setup_dir: Path,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    date: datetime,
    variable: Variable = "hs",
    method: Method = "depth_threshold",
    params: SCFParams | None = None,
) -> dict:
    """Compute model SCF for one member/date within an AOI.

    Parameters
    ----------
    results_dir : Path
        Path to member results directory (e.g., .../member_001/results).
    aoi_path : Path
        Vector file with single AOI polygon; reprojected to raster CRS if needed.
    date : datetime
        Date for which to read the raster (daily outputs expected).
    variable : {"hs","swe"}
        Use snow depth (HS) or SWE raster for computation.
    method : {"depth_threshold","logistic"}
        SCF operator: deterministic threshold or logistic probability.
    params : SCFParams, optional
        Parameters for the operator; h0 in units of the variable, k in 1/units.

    Returns
    -------
    dict
        Dict with keys: date, member_id, variable, method, h0, k, n_valid, scf.
    """
    ensure_gdal_proj_from_conda()
    params = params or SCFParams()
    if landcover_cfg is not None:
        lc_cfg = landcover_cfg
    else:
        lc_cfg = resolve_landcover_mask(Path(setup_dir), Path(project_dir))

    var = variable if variable in (VAR_HS, VAR_SWE) else VAR_HS
    preferred_format = _grid_format_from_setup(Path(setup_dir))
    slice_ = find_member_daily_grid_slice(
        Path(results_dir),
        var,
        date.strftime("%Y-%m-%d"),
        preferred_format=preferred_format,
    )
    arr = _read_masked_array(slice_, Path(aoi_path), lc_cfg=lc_cfg)

    if method == "depth_threshold":
        n_valid, n_snow, scf = _scf_depth_threshold(arr, float(params.h0))
    elif method == "logistic":
        n_valid, scf = _scf_logistic(arr, float(params.h0), float(params.k))
        n_snow = int(round(scf * n_valid))  # pseudo-count for reporting only
    else:
        raise ValueError(f"Unknown method: {method}")

    member_id = member_id_from_results_dir(Path(results_dir))
    return {
        "date": date.strftime("%Y-%m-%d"),
        "member_id": member_id,
        "variable": var,
        "method": method,
        "h0": float(params.h0),
        "k": float(params.k),
        "n_valid": int(n_valid),
        "n_snow": int(n_snow),
        "scf": float(scf),
        "raster": Path(slice_.path).name,
    }


def compute_member_scf_daily(
    *,
    setup_dir: Path,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Compute daily AOI-mean SCF for one member over a date range.

    Uses the same H(x) configuration as assimilation (data_assimilation.h_of_x)
    and reuses :func:`compute_model_scf` for each day where a daily raster is
    available. Missing rasters for particular days are skipped.

    Returns a DataFrame with columns ``time`` (datetime) and ``scf`` sorted by
    time.
    """
    method, variable, params = load_hofx_from_project(Path(project_dir))
    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(setup_dir), Path(project_dir))

    # Normalize variable name for internal use
    var = variable if variable in (VAR_HS, VAR_SWE) else VAR_HS

    # Build daily date range (inclusive, based on calendar days)
    start_day = datetime(start.year, start.month, start.day)
    end_day = datetime(end.year, end.month, end.day)
    if end_day < start_day:
        return pd.DataFrame(columns=["time", "scf"])
    dates = pd.date_range(start_day, end_day, freq="D").to_pydatetime()

    times: list[datetime] = []
    scfs: list[float] = []
    for dt in dates:
        try:
            out = compute_model_scf(
                setup_dir=Path(setup_dir),
                project_dir=Path(project_dir),
                results_dir=Path(results_dir),
                aoi_path=Path(aoi_path),
                landcover_cfg=lc_cfg,
                date=dt,
                variable=var,  # type: ignore[arg-type]
                method=method,  # type: ignore[arg-type]
                params=params,
            )
        except FileNotFoundError:
            # No daily raster for this date -> skip
            continue
        except Exception as exc:
            logger.warning("SCF daily computation failed for {} at {}: {}", results_dir, dt.date(), exc)
            continue
        times.append(dt)
        scfs.append(float(out["scf"]))

    if not times:
        return pd.DataFrame(columns=["time", "scf"])
    df = pd.DataFrame({"time": times, "scf": scfs})
    return df.sort_values("time")


def _compute_member_scf_for_step_worker(
    results_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
    out_csv: Path,
    overwrite: bool,
    extra: Dict[str, Any],
) -> bool:
    """Worker: compute SCF daily series for a single member results dir."""
    lc_cfg = _deserialize_lc_cfg(extra.get("landcover_cfg"))
    setup_dir = Path(extra["setup_dir"])
    project_dir = Path(extra["project_dir"])
    df = compute_member_scf_daily(
        setup_dir=setup_dir,
        project_dir=project_dir,
        results_dir=results_dir,
        aoi_path=aoi_path,
        landcover_cfg=lc_cfg,
        start=start,
        end=end,
    )
    if df.empty:
        return False
    if out_csv.exists() and not overwrite:
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return True


def compute_step_scf_daily_for_all_members(
    *,
    setup_dir: Path,
    project_dir: Path,
    step_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    max_workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Compute daily model SCF for all prior members in a step.

    For the given ``step_dir``, this function:
    - Reads the step start/end dates from its YAML.
    - Discovers prior ensemble members (including open_loop when present).
    - In parallel across members, computes daily AOI-mean SCF time series
      using :func:`compute_member_scf_daily`.
    - Writes the result to ``<member>/results/point_scf_roi.csv`` for each
      member, which can then be used by the setup plotting utilities via
      ``var_col='scf'`` and ``station='point_scf_roi.csv'``.

    Existing CSVs are skipped unless ``overwrite=True``.
    """
    step_dir = Path(step_dir)
    setup_dir = Path(setup_dir)
    project_dir = Path(project_dir)
    aoi_path = Path(aoi_path)
    resolved_project = infer_project_dir(step_dir)
    if resolved_project.resolve() != project_dir.resolve():
        logger.warning(
            "Step {} resolves to project {}; overriding provided project_dir {}",
            step_dir,
            resolved_project,
            project_dir,
        )
        project_dir = resolved_project
    resolved_setup = infer_setup_dir_from_project(project_dir)
    if resolved_setup.resolve() != setup_dir.resolve():
        logger.warning(
            "Project {} resolves to setup {}; overriding provided setup_dir {}",
            project_dir,
            resolved_setup,
            setup_dir,
        )
        setup_dir = resolved_setup
    lc_cfg = landcover_cfg or resolve_landcover_mask(setup_dir, project_dir)

    start, end = step_start_end(step_dir)

    compute_step_daily_series_for_all_members(
        step_dir=step_dir,
        aoi_path=aoi_path,
        start=start,
        end=end,
        csv_name="point_scf_roi.csv",
        worker=_compute_member_scf_for_step_worker,
        ensemble="prior",
        include_open_loop=True,
        max_workers=max_workers,
        overwrite=overwrite,
        worker_kwargs={
            "setup_dir": str(setup_dir),
            "project_dir": str(project_dir),
            "landcover_cfg": _serialize_lc_cfg(lc_cfg),
        },
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI for computing model SCF per member/date.

    Examples
    --------
    oa-da-model-scf \
      --setup-dir C:/.../examples/test-project \
      --project-dir C:/.../examples/test-project/projects/project_2022_2023 \
      --member-results C:/.../member_001/results \
      --aoi examples/test-project/env/GMBA_Inventory_L8_15422.gpkg \
      --date 2017-12-10
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-scf",
        description=(
            "Compute model-derived Snow Cover Fraction (SCF) from openAMUNDSEN "
            "daily outputs (HS/SWE) within an AOI for one member/date using the "
            "project-level H(x) configuration."
        ),
    )
    parser.add_argument("--setup-dir", type=Path, required=True, help="Setup root containing setup YAML")
    parser.add_argument("--project-dir", type=Path, help="Project directory (auto-inferred from --member-results when omitted)")
    parser.add_argument("--member-results", type=Path, required=True, help="Path to member results directory")
    parser.add_argument("--aoi", "--roi", dest="aoi", type=Path, required=True, help="Path to ROI vector file")
    parser.add_argument("--date", type=str, required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--output", type=Path, default=None, help="Optional output CSV path")
    parser.add_argument("--region-id-field", type=str, default="region_id", help="AOI field name for region_id")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (INFO, DEBUG, ...)")

    args = parser.parse_args(argv)

    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    # Parse date
    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
    except Exception as e:
        logger.error(f"Invalid --date format (expected YYYY-MM-DD): {args.date}")
        return 2

    try:
        project_dir = Path(args.project_dir) if args.project_dir is not None else infer_project_dir(Path(args.member_results))
        resolved_setup = infer_setup_dir_from_project(project_dir)
        if resolved_setup.resolve() != Path(args.setup_dir).resolve():
            logger.warning(
                "Project {} belongs to setup {}; overriding provided setup {}",
                project_dir,
                resolved_setup,
                args.setup_dir,
            )
        setup_dir = resolved_setup
        method, variable, prm = load_hofx_from_project(project_dir)
        lc_cfg = resolve_landcover_mask(setup_dir, project_dir)
    except Exception as e:
        logger.error("Failed to read configuration: {}", e)
        return 2

    # Compute SCF
    try:
        out = compute_model_scf(
            setup_dir=setup_dir,
            project_dir=project_dir,
            results_dir=Path(args.member_results),
            aoi_path=Path(args.aoi),
            landcover_cfg=lc_cfg,
            date=dt,
            variable=variable,  # type: ignore[arg-type]
            method=("logistic" if method == "logistic" else "depth_threshold"),  # type: ignore[arg-type]
            params=prm,
        )
    except Exception as e:
        logger.error(f"Model SCF computation failed: {e}")
        return 1

    # Attempt to extract region_id for reporting (optional)
    region_id = None
    try:
        gdf = gpd.read_file(args.aoi)
        if len(gdf) == 1 and args.region_id_field in gdf.columns:
            region_id = str(gdf.iloc[0][args.region_id_field])
    except Exception:
        pass

    # Prepare CSV
    df = pd.DataFrame({
        "date": [out["date"]],
        "member_id": [out["member_id"]],
        "region_id": [region_id if region_id is not None else ""],
        "variable": [out["variable"]],
        "method": [out["method"]],
        "h0": [out["h0"]],
        "k": [out["k"]],
        "n_valid": [out["n_valid"]],
        "scf_model": [round(out["scf"], 4)],
        "raster": [out["raster"]],
    })

    # Output path default: <member_results>/model_scf_YYYYMMDD.csv
    out_csv = args.output
    if out_csv is None:
        out_csv = Path(args.member_results) / f"model_scf_{dt.strftime('%Y%m%d')}.csv"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    logger.info(
        "MODEL_SCF | raster={} member={} region={} var={} method={} h0={} k={} valid={} scf={:.3f} -> {}".format(
            out["raster"], out["member_id"], region_id if region_id else "", out["variable"], out["method"], out["h0"], out["k"], out["n_valid"], out["scf"], out_csv.name
        )
    )
    return 0


def cli_project_daily(argv: list[str] | None = None) -> int:
    """CLI: compute daily model SCF for all members and steps in a project.

    Example
    -------
    oa-da-model-scf-project-daily \\
      --setup-dir C:/.../examples/test-project \\
      --project-dir C:/.../examples/test-project/projects/project_2022_2023 \\
      --aoi C:/.../env/GMBA_Inventory_L8_15422.gpkg \\
      --max-workers 8
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-scf-project-daily",
        description=(
            "Compute daily model SCF (AOI-mean) for all prior members in each "
            "step of a project, writing point_scf_roi.csv per member."
        ),
    )
    parser.add_argument("--setup-dir", type=Path, required=True, help="Setup root containing setup YAML")
    parser.add_argument("--project-dir", type=Path, required=True, help="Project directory containing steps/step_*")
    parser.add_argument("--aoi", "--roi", dest="aoi", type=Path, required=True, help="Single-feature ROI vector (same as used by assimilation)")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel workers per step")
    parser.add_argument("--overwrite", action="store_true", help="Recompute SCF even if point_scf_roi.csv exists")
    parser.add_argument("--log-level", type=str, default="INFO", help="Log level (INFO, DEBUG, ...)")

    args = parser.parse_args(argv)

    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    project_dir = Path(args.project_dir)
    setup_dir = Path(args.setup_dir)
    resolved_setup = infer_setup_dir_from_project(project_dir)
    if resolved_setup.resolve() != setup_dir.resolve():
        logger.warning(
            "Project {} belongs to setup {}; overriding provided setup {}",
            project_dir,
            resolved_setup,
            setup_dir,
        )
        setup_dir = resolved_setup
    steps = list_step_dirs(project_dir)
    if not steps:
        logger.error("No step directories found under {}", project_dir)
        return 1

    logger.info("Computing daily model SCF for project: {} ({} step(s))", project_dir.name, len(steps))
    for step in steps:
        try:
            compute_step_scf_daily_for_all_members(
                setup_dir=setup_dir,
                project_dir=project_dir,
                step_dir=step,
                aoi_path=Path(args.aoi),
                max_workers=int(args.max_workers or 4),
                overwrite=bool(args.overwrite),
            )
        except Exception as exc:
            logger.error("SCF daily computation failed for step {}: {}", step.name, exc)
            return 2
    logger.info("Project-wide model SCF daily computation complete for {}", project_dir)
    return 0


# Backward-compatible alias for transitional references.
cli_setup_daily = cli_project_daily


if __name__ == "__main__":
    sys.exit(cli_main())





