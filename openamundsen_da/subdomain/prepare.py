"""Preparation utilities for sub-domain DA workflows."""

from __future__ import annotations

import copy
import io
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.enums import Resampling
from rasterio import features, windows
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from openamundsen_da.io.paths import (
    abspath_relative_to,
    find_plain_setup_yaml,
    find_project_yaml,
    find_setup_yaml,
)
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec
from openamundsen_da.subdomain.status import record_stage
from openamundsen_da.util.yaml_utils import read_yaml_mapping
from openamundsen_da.util.roi_grid import ensure_setup_roi_grid, load_setup_roi_mask
from openamundsen_da.util.run_mode import ensure_run_mode
from openamundsen_da.util.station_da import (
    STATION_DA_METADATA_FILENAME,
    STATION_SNOW_DEPTH_METADATA_FILENAME,
    is_station_metadata_file,
    normalize_station_id_series,
)


@dataclass
class GridPaths:
    dem: Path
    svf: Optional[Path]
    srf: Optional[Path]
    lc: Optional[Path]


def _nested_dir(cfg: dict, keys: tuple[str, ...], default_rel: str) -> Path:
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict):
            cur = None
            break
        cur = cur.get(key)
    if isinstance(cur, str) and cur.strip():
        return Path(cur)
    return Path(default_rel)


def _to_yaml_text(data: dict) -> str:
    import ruamel.yaml as _yaml

    y = _yaml.YAML()
    y.default_flow_style = False
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()


def _sanitize_id(raw: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))
    return clean.strip("_") or "subdomain"


def _find_grid(grids_dir: Path, prefix: str, domain: str, resolution: str) -> Optional[Path]:
    patt = f"{prefix}_{domain}_{resolution}"
    candidates = sorted(grids_dir.glob(f"{patt}*.asc"))
    if candidates:
        return candidates[0]
    candidates = sorted(grids_dir.glob(f"{patt}*.tif"))
    return candidates[0] if candidates else None


def _check_no_overlap(geoms: Iterable[BaseGeometry], *, area_tol: float = 0.0) -> None:
    geoms = list(geoms)
    for i, g1 in enumerate(geoms):
        for g2 in geoms[i + 1 :]:
            if not g1.intersects(g2):
                continue
            inter_area = g1.intersection(g2).area
            if inter_area > area_tol:
                raise ValueError(
                    f"Detected overlapping sub-domains (overlap area {inter_area:.3f} m^2 "
                    f"exceeds tolerance {area_tol} m^2)."
                )


def _dem_metadata(dem_path: Path) -> Tuple[int, int, rasterio.Affine, float, str | None]:
    with rasterio.open(dem_path) as ds:
        rows, cols = ds.height, ds.width
        transform = ds.transform
        res_x, res_y = ds.res
        if not math.isclose(abs(res_x), abs(res_y)):
            raise ValueError(f"DEM resolution not square: {res_x} vs {res_y}")
        crs = ds.crs.to_string() if ds.crs else None
    return rows, cols, transform, float(res_x), crs


def _window_for_geometry(
    geom: BaseGeometry,
    transform: rasterio.Affine,
    raster_shape: Tuple[int, int],
) -> windows.Window:
    minx, miny, maxx, maxy = geom.bounds
    win = windows.from_bounds(minx, miny, maxx, maxy, transform=transform, precision=6)
    win = win.round_offsets().round_shape()
    full = windows.Window(col_off=0, row_off=0, width=raster_shape[1], height=raster_shape[0])
    return win.intersection(full)


def _window_for_mask(mask: np.ndarray) -> windows.Window:
    rows, cols = np.where(mask.astype(bool))
    if len(rows) == 0:
        raise ValueError("Cannot derive window from empty mask")
    row0 = int(rows.min())
    row1 = int(rows.max()) + 1
    col0 = int(cols.min())
    col1 = int(cols.max()) + 1
    return windows.Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _union_windows(
    a: windows.Window,
    b: windows.Window,
    raster_shape: Tuple[int, int],
) -> windows.Window:
    row0 = min(int(a.row_off), int(b.row_off))
    col0 = min(int(a.col_off), int(b.col_off))
    row1 = max(int(a.row_off + a.height), int(b.row_off + b.height))
    col1 = max(int(a.col_off + a.width), int(b.col_off + b.width))

    row0 = max(0, row0)
    col0 = max(0, col0)
    row1 = min(int(raster_shape[0]), row1)
    col1 = min(int(raster_shape[1]), col1)
    if row1 <= row0 or col1 <= col0:
        raise ValueError("Union window is empty after clipping to raster extent")
    return windows.Window(col_off=col0, row_off=row0, width=col1 - col0, height=row1 - row0)


def _crop_grid(src: Path, dst: Path, win: windows.Window, fill_value=None) -> None:
    with rasterio.open(src) as ds:
        fv = fill_value if fill_value is not None else ds.nodata
        if fv is None or (isinstance(fv, (float, np.floating)) and np.isnan(fv)):
            fv = -9999.0
        data = ds.read(1, window=win, boundless=True, fill_value=fv)
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isnan(data), fv, data)
        meta = ds.meta.copy()
        meta.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": windows.transform(win, ds.transform),
            }
        )
        if fv is not None:
            meta["nodata"] = fv
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, "w", **meta) as out:
            out.write(data, 1)


def _crop_grid_to_template(
    src: Path,
    dst: Path,
    *,
    template_shape: Tuple[int, int],
    template_transform: rasterio.Affine,
    fill_value=None,
    resampling: Resampling = Resampling.nearest,
) -> None:
    """Crop/reproject a raster to a target grid definition.

    This is used for optional grids (SVF/SRF/LC) that may differ in extent from
    DEM while still sharing the same CRS/resolution.
    """
    t_rows, t_cols = int(template_shape[0]), int(template_shape[1])
    tpl_window = windows.Window(col_off=0, row_off=0, width=t_cols, height=t_rows)
    left, bottom, right, top = windows.bounds(tpl_window, template_transform)

    with rasterio.open(src) as ds:
        fv = fill_value if fill_value is not None else ds.nodata
        if fv is None or (isinstance(fv, (float, np.floating)) and np.isnan(fv)):
            fv = -9999.0
        src_win = windows.from_bounds(left, bottom, right, top, transform=ds.transform)
        data = ds.read(
            1,
            window=src_win,
            out_shape=(t_rows, t_cols),
            boundless=True,
            fill_value=fv,
            resampling=resampling,
        )
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isnan(data), fv, data)
        meta = ds.meta.copy()
        meta.update(
            {
                "height": t_rows,
                "width": t_cols,
                "transform": template_transform,
            }
        )
        if fv is not None:
            meta["nodata"] = fv
        dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, "w", **meta) as out:
            out.write(data, 1)


def _write_roi_mask(
    mask: np.ndarray,
    out_path: Path,
    transform: rasterio.Affine,
    *,
    crs: str | None,
) -> None:
    mask = mask.astype(bool)
    meta = {
        "driver": "AAIGrid",
        "dtype": "uint8",
        "nodata": 0,
        "width": mask.shape[1],
        "height": mask.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype("uint8"), 1)


def _mask_grid_outside_mask(grid_path: Path, mask: np.ndarray) -> None:
    """Set pixels outside a boolean mask to nodata."""
    with rasterio.open(grid_path, "r+") as ds:
        arr = ds.read(1)
        inside = mask.astype(bool)
        if inside.shape != arr.shape:
            raise ValueError(f"Mask shape mismatch for {grid_path}: {inside.shape} vs {arr.shape}")

        nodata = ds.nodata
        out = arr.copy()
        if nodata is None and np.issubdtype(out.dtype, np.floating):
            out[~inside] = np.nan
        elif nodata is not None:
            out[~inside] = nodata
        else:
            # Fallback for integer rasters without nodata metadata.
            out[~inside] = 0
        ds.write(out, 1)


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def _write_subdomain_setup_yaml(
    *,
    source_cfg: dict,
    sub_setup_dir: Path,
    domain: str,
    grids_dir: Path,
    meteo_dir: Path,
    roi_geom: BaseGeometry | None = None,
) -> Path:
    sub_setup_dir.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(source_cfg)
    cfg["domain"] = domain
    cfg.setdefault("input_data", {}).setdefault("grids", {})
    cfg["input_data"]["grids"]["dir"] = grids_dir.resolve().relative_to(sub_setup_dir.resolve()).as_posix()
    cfg.setdefault("input_data", {}).setdefault("meteo", {})
    cfg["input_data"]["meteo"]["dir"] = meteo_dir.resolve().relative_to(sub_setup_dir.resolve()).as_posix()
    # Sub-domains may rely on nearby stations outside the clipped grid extent.
    # Use global station bounds to avoid dropping all stations in small tiles.
    cfg["input_data"]["meteo"]["bounds"] = "global"
    _filter_output_timeseries_points(cfg, roi_geom=roi_geom)
    cfg["results_dir"] = str((sub_setup_dir / "results").resolve())
    out_yaml = sub_setup_dir / f"{sub_setup_dir.name}.yml"
    out_yaml.write_text(_to_yaml_text(cfg), encoding="utf-8")
    return out_yaml


def _point_xy(point_cfg: object) -> tuple[float, float] | None:
    if not isinstance(point_cfg, dict):
        return None
    for x_key, y_key in (("x", "y"), ("lon", "lat"), ("longitude", "latitude")):
        if x_key not in point_cfg or y_key not in point_cfg:
            continue
        try:
            return float(point_cfg[x_key]), float(point_cfg[y_key])
        except Exception:
            return None
    if "coords" in point_cfg:
        coords = point_cfg.get("coords")
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            try:
                return float(coords[0]), float(coords[1])
            except Exception:
                return None
    return None


def _normalize_output_point_cfg(point_cfg: object) -> object:
    if isinstance(point_cfg, dict) and "name" in point_cfg and point_cfg["name"] is not None:
        point_cfg = copy.deepcopy(point_cfg)
        point_cfg["name"] = str(point_cfg["name"])
    return point_cfg


def _filter_output_timeseries_points(cfg: dict, *, roi_geom: BaseGeometry | None) -> None:
    """Keep configured point outputs that fall inside the sub-domain ROI."""
    if roi_geom is None or roi_geom.is_empty:
        return
    timeseries_cfg = ((cfg.get("output_data") or {}).get("timeseries") or {})
    points = timeseries_cfg.get("points")
    if not isinstance(points, list):
        return

    kept: list[object] = []
    dropped = 0
    for point_cfg in points:
        xy = _point_xy(point_cfg)
        if xy is None:
            kept.append(_normalize_output_point_cfg(point_cfg))
            continue
        if roi_geom.covers(Point(xy)):
            kept.append(_normalize_output_point_cfg(point_cfg))
        else:
            dropped += 1

    timeseries_cfg["points"] = kept
    if dropped:
        logger.debug("Dropped {} configured point output(s) outside sub-domain ROI", dropped)


def _copy_project_dir(source_project_dir: Path, target_project_dir: Path) -> Path:
    ignore_names = {"steps", "plots", "ensembles", "assim", "subdomains", "results"}
    if target_project_dir.exists():
        shutil.rmtree(target_project_dir)
    shutil.copytree(
        source_project_dir,
        target_project_dir,
        ignore=shutil.ignore_patterns(*ignore_names),
    )
    return find_project_yaml(target_project_dir)


def _write_roi_vector(*, roi_geom: BaseGeometry, crs: str | None, out_path: Path, region_label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attrs = pd.DataFrame({"id": pd.Series([str(region_label)], dtype="object")})
    gdf = gpd.GeoDataFrame(attrs, geometry=[roi_geom], crs=crs)
    gdf.to_file(out_path, driver="GPKG")


def _prepare_grids(
    *,
    grid_paths: GridPaths,
    grids_out: Path,
    clip_mode: str,
    domain: str,
    new_domain: str,
    resolution: str,
    window: windows.Window,
    transform: rasterio.Affine,
    global_shape: Tuple[int, int],
    global_transform: rasterio.Affine,
    roi_mask: np.ndarray,
    crs: str | None,
) -> Path:
    dem_src = grid_paths.dem
    if clip_mode == "roi-symlink":
        dem_dst = grids_out / f"dem_{new_domain}_{resolution}.asc"
        _copy_or_link(dem_src, dem_dst)
        if grid_paths.svf:
            _copy_or_link(grid_paths.svf, grids_out / f"svf_{new_domain}_{resolution}.asc")
        if grid_paths.srf:
            _copy_or_link(grid_paths.srf, grids_out / f"srf_{new_domain}_{resolution}.asc")
        if grid_paths.lc:
            _copy_or_link(grid_paths.lc, grids_out / f"lc_{new_domain}_{resolution}.asc")
        roi_dst = grids_out / f"roi_{new_domain}_{resolution}.asc"
        if roi_mask.shape != (int(global_shape[0]), int(global_shape[1])):
            expected_shape = (int(global_shape[0]), int(global_shape[1]))
            raise ValueError(
                f"ROI mask shape mismatch for roi-symlink mode: {roi_mask.shape} vs {expected_shape}"
            )
        _write_roi_mask(roi_mask, roi_dst, global_transform, crs=crs)
        return roi_dst

    dem_out = grids_out / f"dem_{new_domain}_{resolution}.asc"
    _crop_grid(dem_src, dem_out, window, fill_value=-9999.0)
    target_shape = (int(window.height), int(window.width))
    if roi_mask.shape != target_shape:
        raise ValueError(f"ROI mask shape mismatch for window mode: {roi_mask.shape} vs {target_shape}")
    svf_out = grids_out / f"svf_{new_domain}_{resolution}.asc"
    srf_out = grids_out / f"srf_{new_domain}_{resolution}.asc"
    lc_out = grids_out / f"lc_{new_domain}_{resolution}.asc"
    if grid_paths.svf:
        _crop_grid_to_template(
            grid_paths.svf,
            svf_out,
            template_shape=target_shape,
            template_transform=transform,
            fill_value=-9999.0,
            resampling=Resampling.bilinear,
        )
    if grid_paths.srf:
        _crop_grid_to_template(
            grid_paths.srf,
            srf_out,
            template_shape=target_shape,
            template_transform=transform,
            fill_value=-9999.0,
            resampling=Resampling.bilinear,
        )
    if grid_paths.lc:
        _crop_grid_to_template(
            grid_paths.lc,
            lc_out,
            template_shape=target_shape,
            template_transform=transform,
            fill_value=None,
            resampling=Resampling.nearest,
        )

    roi_dst = grids_out / f"roi_{new_domain}_{resolution}.asc"
    _write_roi_mask(roi_mask, roi_dst, transform, crs=crs)
    return roi_dst


def _prepare_meteo(
    *,
    meteo_dir: Path,
    out_dir: Path,
    geom: Polygon,
    buffer_m: float,
    crs: Optional[str],
) -> list[str]:
    stations_path = meteo_dir / "stations.csv"
    if not stations_path.is_file():
        raise FileNotFoundError(f"Missing stations.csv in {meteo_dir}")
    stations = pd.read_csv(stations_path)
    if not {"x", "y"}.issubset(stations.columns):
        raise ValueError("stations.csv must contain 'x' and 'y' columns (project CRS)")
    gdf = gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(stations["x"], stations["y"]),
        crs=crs,
    )
    buffered = geom.buffer(buffer_m) if buffer_m and buffer_m > 0 else geom
    selected = stations.loc[gdf.geometry.within(buffered)].copy()
    if selected.empty:
        raise ValueError(f"No meteo stations found within buffer {buffer_m} m for region")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "stations.csv", index=False)

    for sid in selected["id"]:
        src_csv = meteo_dir / f"{sid}.csv"
        src_nc = meteo_dir / f"{sid}.nc"
        src = src_csv if src_csv.exists() else src_nc if src_nc.exists() else None
        if src is None:
            logger.warning("No meteo file found for station {}", sid)
            continue
        _copy_or_link(src, out_dir / src.name)

    if (meteo_dir / "meteo_format.txt").is_file():
        _copy_or_link(meteo_dir / "meteo_format.txt", out_dir / "meteo_format.txt")
    return [str(sid) for sid in selected["id"].astype(str)]


def _prepare_obs_station_subset(
    *,
    obs_dir: Path,
    out_dir: Path,
    geom: Polygon,
    buffer_m: float,
    crs: Optional[str],
    station_ids: Optional[Iterable[str]] = None,
) -> dict[str, int]:
    stats = {
        "obs_stations_selected": 0,
        "obs_stations_inside_grid": 0,
        "obs_stations_da_active": 0,
        "obs_stations_benchmark_active": 0,
        "obs_station_series_copied": 0,
    }

    def role_enabled(raw: object) -> bool:
        if isinstance(raw, (bool, np.bool_)):
            return bool(raw)
        return str(raw).strip().lower() not in {"false", "0", "no", "n", "off"}
    if not obs_dir.is_dir():
        logger.info("Obs directory {} not found; skipping station subset", obs_dir)
        return stats
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = {
        sid
        for sid in normalize_station_id_series(pd.Series(list(station_ids or [])))
        if sid
    }
    selected_ids: set[str] = set()
    inside_grid_ids: set[str] = set()
    stations_meta = obs_dir / STATION_SNOW_DEPTH_METADATA_FILENAME
    da_meta = obs_dir / STATION_DA_METADATA_FILENAME
    coordinate_meta: pd.DataFrame | None = None
    coordinate_id_column: str | None = None
    used_coordinate_filter = False
    if stations_meta.is_file():
        meta_df = pd.read_csv(stations_meta, dtype={"id": "string"})
        if "id" not in meta_df.columns:
            raise ValueError(f"{stations_meta} must contain an 'id' column")
        coordinate_meta = meta_df
        coordinate_id_column = "id"
    elif da_meta.is_file():
        meta_df = pd.read_csv(da_meta, dtype={"station_id": "string"})
        if {"station_id", "x", "y"}.issubset(meta_df.columns):
            coordinate_meta = meta_df
            coordinate_id_column = "station_id"

    if coordinate_meta is not None and coordinate_id_column is not None:
        if not {"x", "y"}.issubset(coordinate_meta.columns):
            logger.warning("{} missing x/y columns; trying ID-based station fallback", stations_meta.name)
        else:
            used_coordinate_filter = True
            try:
                x_values = pd.to_numeric(coordinate_meta["x"], errors="raise")
                y_values = pd.to_numeric(coordinate_meta["y"], errors="raise")
            except Exception as exc:
                raise ValueError(
                    f"Station coordinate metadata contains invalid x/y values: "
                    f"{stations_meta if stations_meta.is_file() else da_meta}"
                ) from exc
            if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
                raise ValueError(
                    f"Station coordinate metadata contains non-finite x/y values: "
                    f"{stations_meta if stations_meta.is_file() else da_meta}"
                )
            gdf = gpd.GeoDataFrame(
                coordinate_meta,
                geometry=gpd.points_from_xy(x_values, y_values),
                crs=crs,
            )
            buffered = geom.buffer(buffer_m) if buffer_m and buffer_m > 0 else geom
            inside_mask = gdf.geometry.apply(geom.covers)
            selected_mask = gdf.geometry.within(buffered)
            coordinate_meta = coordinate_meta.loc[selected_mask].copy()
            inside_ids = coordinate_meta.loc[
                inside_mask.reindex(coordinate_meta.index, fill_value=False),
                coordinate_id_column,
            ]
            selected_ids = {
                str(sid).strip()
                for sid in coordinate_meta[coordinate_id_column].dropna().astype(str)
                if str(sid).strip()
            }
            inside_grid_ids = {
                sid for sid in normalize_station_id_series(pd.Series(inside_ids)) if sid
            }
            if stations_meta.is_file():
                coordinate_meta.to_csv(out_dir / STATION_SNOW_DEPTH_METADATA_FILENAME, index=False)

    if not selected_ids and not used_coordinate_filter:
        series_by_id = {
            path.stem.strip().lower(): path.stem
            for pattern in ("*.csv", "*.nc")
            for path in sorted(obs_dir.glob(pattern))
            if path.is_file() and not is_station_metadata_file(path)
        }
        if requested_ids:
            missing_requested = sorted(requested_ids - set(series_by_id))
            if missing_requested:
                raise ValueError(
                    "Cannot select snow-station observations without coordinate metadata: "
                    "requested station IDs have no same-ID observation series: "
                    + ", ".join(missing_requested)
                )
            selected_ids = {series_by_id[sid] for sid in requested_ids}
            inside_grid_ids = set(requested_ids)
        elif series_by_id:
            selected_ids = set(series_by_id.values())
            inside_grid_ids = set(series_by_id)
        else:
            raise ValueError(
                f"Cannot select snow-station observations in {obs_dir}: provide "
                f"{STATION_SNOW_DEPTH_METADATA_FILENAME} with id/x/y, "
                f"{STATION_DA_METADATA_FILENAME} with station_id/x/y, or same-ID station series."
            )

    selected_ids_lower = {sid.strip().lower() for sid in selected_ids if sid.strip()}
    inside_grid_ids_lower = {sid.strip().lower() for sid in inside_grid_ids if sid.strip()}
    stats["obs_stations_selected"] = len(selected_ids_lower)
    stats["obs_stations_inside_grid"] = len(inside_grid_ids_lower)

    if da_meta.is_file():
        da_df = pd.read_csv(da_meta, dtype={"station_id": "string"})
        if selected_ids_lower and "station_id" in da_df.columns:
            station_keys = normalize_station_id_series(da_df["station_id"])
            keep_mask = station_keys.isin(selected_ids_lower)
            da_df = da_df.loc[keep_mask].copy()
            inside_mask = normalize_station_id_series(da_df["station_id"]).isin(inside_grid_ids_lower)
            for role_col in ("use_for_da", "use_for_benchmark"):
                if role_col not in da_df.columns:
                    da_df[role_col] = True
                da_df.loc[~inside_mask, role_col] = False
            stats["obs_stations_da_active"] = sum(role_enabled(value) for value in da_df["use_for_da"])
            stats["obs_stations_benchmark_active"] = sum(role_enabled(value) for value in da_df["use_for_benchmark"])
        da_df.to_csv(out_dir / STATION_DA_METADATA_FILENAME, index=False)

    if not selected_ids:
        return stats

    for sid in sorted(selected_ids):
        copied = False
        for ext in (".csv", ".nc"):
            src = obs_dir / f"{sid}{ext}"
            if not src.exists():
                matches = sorted(
                    p
                    for p in obs_dir.glob(f"*{ext}")
                    if p.stem.strip().lower() == sid.strip().lower()
                )
                src = matches[0] if matches else src
            if src.exists():
                _copy_or_link(src, out_dir / src.name)
                copied = True
                stats["obs_station_series_copied"] += 1
                break
        if not copied:
            logger.debug("No station obs series found for id {}", sid)
    return stats


def prepare_subdomains(
    *,
    setup_dir: Path,
    project_dir: Path | None,
    regions_path: Path,
    subdomain_root: Path | None = None,
    id_field: str = "id",
    clip_mode: str = "window",
    station_buffer_m: float = 50_000.0,
    roi_buffer_m: float = 0.0,
    grid_buffer_m: Optional[float] = None,
    obs_stations_dir: Optional[Path] = None,
    overlap_area_tol_m2: float = 100.0,
    sliver_fix_m: float = 0.0,
    overwrite: bool = False,
    model_mode: bool = False,
) -> SubdomainManifest:
    """Prepare per-sub-domain setups under `<project>/subdomains/<id>`."""
    logger.debug("Preparing sub-domains for setup={} regions={}", setup_dir, regions_path)
    if clip_mode not in {"window", "roi-symlink"}:
        raise ValueError("clip_mode must be 'window' or 'roi-symlink'")

    setup_dir = Path(setup_dir).resolve()
    if model_mode:
        project_dir = Path(project_dir).resolve() if project_dir is not None else None
    else:
        if project_dir is None:
            raise TypeError("project_dir is required for DA sub-domain preparation")
        project_dir = Path(project_dir).resolve()
        ensure_run_mode(project_dir, expected="subdomain", write_if_missing=True)
    setup_yaml = find_plain_setup_yaml(setup_dir) if model_mode else find_setup_yaml(setup_dir)
    setup_cfg = read_yaml_mapping(setup_yaml, error_cls=ValueError, context="Setup YAML root")
    if model_mode:
        missing_dates = [
            key
            for key in ("start_date", "end_date")
            if setup_cfg.get(key) is None or str(setup_cfg.get(key)).strip() == ""
        ]
        if missing_dates:
            raise ValueError(
                "Model sub-domain mode requires the source setup YAML to define "
                f"{', '.join(missing_dates)}."
            )
        project_yaml = setup_yaml
        project_cfg: dict = {}
    else:
        project_yaml = find_project_yaml(project_dir)
        project_cfg = read_yaml_mapping(project_yaml, error_cls=ValueError, context="Project YAML root")

    grid_buffer_m = float(grid_buffer_m if grid_buffer_m is not None else 0.0)
    station_buffer_m = float(station_buffer_m)
    roi_buffer_m = float(roi_buffer_m)

    if "meteo" not in (setup_cfg.get("input_data") or {}) or "grids" not in (setup_cfg.get("input_data") or {}):
        raise ValueError("Setup config input_data must define both 'grids' and 'meteo' sections.")

    grids_rel = _nested_dir(setup_cfg, ("input_data", "grids", "dir"), "grids")
    meteo_rel = _nested_dir(setup_cfg, ("input_data", "meteo", "dir"), "meteo")
    if model_mode:
        obs_station_rel = Path("obs/stations")
        raw_snowcover_rel = Path("obs/snowcover")
        raw_wetsnow_rel = Path("obs/wetsnow")
    else:
        obs_station_rel = _nested_dir(project_cfg, ("obs", "stations", "dir"), "obs/stations")
        raw_snowcover_rel = _nested_dir(project_cfg, ("obs", "snowcover", "dir"), "obs/snowcover")
        raw_wetsnow_rel = _nested_dir(project_cfg, ("obs", "wetsnow", "dir"), "obs/wetsnow")

    grids_dir = Path(abspath_relative_to(setup_dir, grids_rel))
    meteo_dir = Path(abspath_relative_to(setup_dir, meteo_rel))
    obs_dir = Path(obs_stations_dir) if obs_stations_dir else Path(abspath_relative_to(setup_dir, obs_station_rel))
    raw_snowcover_dir = Path(abspath_relative_to(setup_dir, raw_snowcover_rel))
    raw_wetsnow_dir = Path(abspath_relative_to(setup_dir, raw_wetsnow_rel))
    if model_mode:
        logger.info("Resolved model data dirs grids={} meteo={}", grids_dir, meteo_dir)
    else:
        logger.info(
            "Resolved data dirs grids={} meteo={} obs_stations={} snowcover_raw={} wetsnow_raw={}",
            grids_dir,
            meteo_dir,
            obs_dir,
            raw_snowcover_dir,
            raw_wetsnow_dir,
        )

    domain = str(setup_cfg["domain"])
    resolution = str(setup_cfg["resolution"])
    crs_expected = setup_cfg.get("crs")

    dem_path = _find_grid(grids_dir, "dem", domain, resolution)
    if dem_path is None:
        raise FileNotFoundError(f"DEM not found in {grids_dir} for domain={domain}, resolution={resolution}")
    rows, cols, transform, res_val, dem_crs = _dem_metadata(dem_path)
    if crs_expected and dem_crs and crs_expected.lower() != dem_crs.lower():
        raise ValueError(f"CRS mismatch: setup {crs_expected} vs DEM {dem_crs}")
    crs_str = crs_expected or dem_crs

    gdf = gpd.read_file(regions_path)
    gdf["geometry"] = gdf.geometry.buffer(0)
    if sliver_fix_m and sliver_fix_m > 0:
        gdf["geometry"] = gdf.geometry.buffer(-sliver_fix_m).buffer(sliver_fix_m)
    effective_id_field = id_field
    if effective_id_field not in gdf.columns:
        raise KeyError(f"Regions file missing id field '{id_field}'")
    if gdf.crs and crs_str and gdf.crs.to_string().lower() != crs_str.lower():
        raise ValueError(f"CRS mismatch between regions ({gdf.crs}) and setup ({crs_str})")
    if len(gdf) < 2:
        raise ValueError(
            f"Sub-domain mode requires at least 2 polygons in {regions_path} (got {len(gdf)})."
        )
    _check_no_overlap(gdf.geometry, area_tol=float(overlap_area_tol_m2))
    ensure_setup_roi_grid(setup_dir, roi_vector_path=regions_path, overwrite=True)
    setup_roi_mask, roi_spec, setup_roi_grid_path = load_setup_roi_mask(setup_dir, ensure_grid=False)
    if (roi_spec.rows, roi_spec.cols) != (rows, cols):
        raise ValueError(
            f"Setup ROI grid shape mismatch: {(roi_spec.rows, roi_spec.cols)} vs DEM {(rows, cols)}"
        )
    if tuple(roi_spec.transform) != tuple(transform):
        raise ValueError("Setup ROI grid transform mismatch with DEM transform")
    logger.info("Using setup ROI grid {}", setup_roi_grid_path)

    if model_mode:
        subdomain_root = (Path(subdomain_root) if subdomain_root else (setup_dir / "subdomains" / "model")).resolve()
        manifest_project_dir = subdomain_root
        project_name = "model"
    else:
        subdomain_root = (Path(subdomain_root) if subdomain_root else (project_dir / "subdomains")).resolve()
        manifest_project_dir = project_dir
        project_name = project_dir.name
    if overwrite:
        derived_dirs = (
            (subdomain_root,)
            if model_mode
            else (subdomain_root, project_dir / "results", project_dir / "plots")
        )
        for derived_dir in derived_dirs:
            if derived_dir.is_dir():
                shutil.rmtree(derived_dir)
    subdomain_root.mkdir(parents=True, exist_ok=True)

    lc_path = _find_grid(grids_dir, "lc", domain, resolution)
    svf_path = _find_grid(grids_dir, "svf", domain, resolution)
    srf_path = _find_grid(grids_dir, "srf", domain, resolution)
    grid_paths = GridPaths(dem=dem_path, svf=svf_path, srf=srf_path, lc=lc_path)

    manifest = SubdomainManifest(
        run_mode="model" if model_mode else "subdomain",
        setup_dir=setup_dir,
        project_dir=manifest_project_dir,
        project_name=project_name,
        setup_yaml=setup_yaml,
        project_yaml=project_yaml,
        subdomain_root=subdomain_root,
        regions_path=regions_path.resolve(),
        id_field=effective_id_field,
        crs=crs_str,
        grid_rows=rows,
        grid_cols=cols,
        grid_transform=tuple(transform),
        grid_resolution=res_val,
        grid_domain=domain,
        clip_mode=clip_mode,
        station_buffer_m=station_buffer_m,
        roi_buffer_m=roi_buffer_m,
        grid_buffer_m=grid_buffer_m,
        raw_snowcover_dir=raw_snowcover_dir,
        raw_wetsnow_dir=raw_wetsnow_dir,
    )

    entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for _, row in gdf.iterrows():
        raw_id = row[effective_id_field]
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError(f"Region {raw_id} has empty geometry")
        if geom.geom_type == "MultiPolygon":
            geom = unary_union(geom)
        if geom.geom_type not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Region {raw_id} must be polygonal, got {geom.geom_type}")

        clean_id = _sanitize_id(str(raw_id))
        label = str(raw_id)
        if clean_id in seen_ids:
            raise ValueError(f"Duplicate sub-domain id after sanitization: {clean_id}")
        seen_ids.add(clean_id)
        geom_roi = geom.buffer(roi_buffer_m) if roi_buffer_m else geom
        geom_extent = geom.buffer(grid_buffer_m) if grid_buffer_m else geom
        entries.append(
            {
                "clean_id": clean_id,
                "label": label,
                "geom": geom,
                "geom_roi": geom_roi,
                "geom_extent": geom_extent,
            }
        )

    owner_shapes: list[tuple[BaseGeometry, int]] = []
    coverage_count = np.zeros((rows, cols), dtype=np.uint16)
    for owner_code, entry in enumerate(entries, start=1):
        geom_roi = entry["geom_roi"]
        entry["owner_code"] = owner_code
        owner_shapes.append((geom_roi, owner_code))
        region_mask = features.rasterize(
            [(geom_roi, 1)],
            out_shape=(rows, cols),
            transform=transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)
        region_mask &= setup_roi_mask
        if int(np.count_nonzero(region_mask)) == 0:
            raise ValueError(f"Sub-domain {entry['clean_id']} has no pixels inside setup ROI grid")
        coverage_count[region_mask] += 1

    overlap_count = int(np.count_nonzero((coverage_count > 1) & setup_roi_mask))
    if overlap_count > 0:
        raise ValueError(
            f"Rasterized sub-domain overlap detected: {overlap_count} overlapping pixel(s) inside setup ROI. "
            "Adjust subdomain geometries or reduce roi_buffer_m."
        )

    owner = features.rasterize(
        owner_shapes,
        out_shape=(rows, cols),
        transform=transform,
        fill=0,
        dtype="int32",
    )
    owner = np.where(setup_roi_mask, owner, 0)

    unassigned = setup_roi_mask & (owner == 0)
    unassigned_count = int(np.count_nonzero(unassigned))
    if unassigned_count > 0:
        raise ValueError(
            f"Sub-domain polygons do not cover setup ROI grid: {unassigned_count} pixel(s) uncovered. "
            "Ensure regions fully cover the setup ROI."
        )

    for entry in entries:
        clean_id = str(entry["clean_id"])
        label = str(entry["label"])
        geom = entry["geom"]
        geom_roi = entry["geom_roi"]
        geom_extent = entry["geom_extent"]
        owner_code = int(entry["owner_code"])

        geom_window = _window_for_geometry(geom_extent, transform, (rows, cols))
        owner_window = _window_for_mask(owner == owner_code)
        win = _union_windows(geom_window, owner_window, (rows, cols))
        sub_transform = windows.transform(win, transform)
        sub_rows, sub_cols = int(win.height), int(win.width)
        window_spec = WindowSpec(
            row_off=int(win.row_off),
            col_off=int(win.col_off),
            height=sub_rows,
            width=sub_cols,
        )

        sub_setup_dir = subdomain_root / clean_id
        if sub_setup_dir.exists() and not overwrite:
            raise FileExistsError(f"{sub_setup_dir} already exists. Use --overwrite to rebuild.")
        if sub_setup_dir.exists() and overwrite:
            shutil.rmtree(sub_setup_dir)

        grids_out = sub_setup_dir / "grids"
        meteo_out = sub_setup_dir / "meteo"
        obs_out = sub_setup_dir / "obs" / "stations"
        env_out = sub_setup_dir / "env"
        project_out = sub_setup_dir if model_mode else (sub_setup_dir / "projects" / project_name)
        setup_dirs = (
            (grids_out, meteo_out, env_out)
            if model_mode
            else (grids_out, meteo_out, obs_out, env_out, project_out)
        )
        for d in setup_dirs:
            d.mkdir(parents=True, exist_ok=True)

        sub_domain = f"{domain}_{clean_id}"
        if clip_mode == "roi-symlink":
            roi_mask = owner == owner_code
        else:
            r0 = int(win.row_off)
            r1 = int(win.row_off + win.height)
            c0 = int(win.col_off)
            c1 = int(win.col_off + win.width)
            roi_mask = owner[r0:r1, c0:c1] == owner_code
        if int(np.count_nonzero(roi_mask)) == 0:
            raise ValueError(f"Sub-domain {clean_id} has no ROI pixels in selected window")

        roi_raster_path = _prepare_grids(
            grid_paths=grid_paths,
            grids_out=grids_out,
            clip_mode=clip_mode,
            domain=domain,
            new_domain=sub_domain,
            resolution=resolution,
            window=win,
            transform=sub_transform,
            global_shape=(rows, cols),
            global_transform=transform,
            roi_mask=roi_mask,
            crs=crs_str,
        )

        selected_station_ids = _prepare_meteo(
            meteo_dir=meteo_dir,
            out_dir=meteo_out,
            geom=geom,
            buffer_m=station_buffer_m,
            crs=crs_str,
        )
        if not model_mode:
            station_counts = _prepare_obs_station_subset(
                obs_dir=obs_dir,
                out_dir=obs_out,
                geom=geom,
                buffer_m=station_buffer_m,
                crs=crs_str,
                station_ids=selected_station_ids,
            )
        else:
            station_counts = {}
        roi_vector_path = env_out / "roi.gpkg"
        _write_roi_vector(
            roi_geom=geom_roi,
            crs=crs_str,
            out_path=roi_vector_path,
            region_label=label,
        )

        sub_setup_yaml = _write_subdomain_setup_yaml(
            source_cfg=setup_cfg,
            sub_setup_dir=sub_setup_dir,
            domain=sub_domain,
            grids_dir=grids_out,
            meteo_dir=meteo_out,
            roi_geom=geom_roi,
        )
        sub_project_yaml = sub_setup_yaml if model_mode else _copy_project_dir(project_dir, project_out)

        manifest.subdomains[clean_id] = SubdomainMeta(
            id=clean_id,
            label=label,
            setup_dir=sub_setup_dir,
            setup_yaml=sub_setup_yaml,
            project_dir=project_out,
            project_yaml=sub_project_yaml,
            project_name=project_name,
            grids_dir=grids_out,
            meteo_dir=meteo_out,
            obs_stations_dir=obs_out,
            roi_raster_path=roi_raster_path,
            roi_vector_path=roi_vector_path,
            window=window_spec,
            transform=tuple(sub_transform),
            bounds=geom.bounds,
            crs=crs_str,
            status="pending",
            station_counts=station_counts,
        )
        logger.info(
            "Prepared sub-domain {} ({}x{}, window r{} c{})",
            clean_id,
            sub_rows,
            sub_cols,
            win.row_off,
            win.col_off,
        )

    if not model_mode:
        support_rows = [
            {"subdomain_id": sid, **dict(meta.station_counts or {})}
            for sid, meta in sorted(manifest.subdomains.items())
        ]
        support_path = subdomain_root / "observation_support_by_subdomain.csv"
        pd.DataFrame(support_rows).to_csv(support_path, index=False)
        logger.info("Wrote observation support table -> {}", support_path)

    manifest_path = subdomain_root / "subdomain_manifest.json"
    record_stage(manifest, "prepare", "completed", outputs=(manifest_path,))
    manifest.save(manifest_path)
    logger.info("Wrote manifest -> {}", manifest_path)
    return manifest


def prepare_model_subdomains(
    *,
    setup_dir: Path,
    regions_path: Path,
    subdomain_root: Path | None = None,
    id_field: str = "id",
    clip_mode: str = "window",
    station_buffer_m: float = 50_000.0,
    roi_buffer_m: float = 0.0,
    grid_buffer_m: Optional[float] = None,
    overlap_area_tol_m2: float = 100.0,
    sliver_fix_m: float = 0.0,
    overwrite: bool = False,
) -> SubdomainManifest:
    """Prepare plain openAMUNDSEN sub-domain setups under `<setup>/subdomains/model`."""
    return prepare_subdomains(
        setup_dir=setup_dir,
        project_dir=None,
        regions_path=regions_path,
        subdomain_root=subdomain_root,
        id_field=id_field,
        clip_mode=clip_mode,
        station_buffer_m=station_buffer_m,
        roi_buffer_m=roi_buffer_m,
        grid_buffer_m=grid_buffer_m,
        obs_stations_dir=None,
        overlap_area_tol_m2=overlap_area_tol_m2,
        sliver_fix_m=sliver_fix_m,
        overwrite=overwrite,
        model_mode=True,
    )
