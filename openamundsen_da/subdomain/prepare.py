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
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from openamundsen_da.io.paths import (
    abspath_relative_to,
    find_project_yaml,
    find_setup_yaml,
)
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec
from openamundsen_da.util.run_mode import ensure_run_mode


@dataclass
class GridPaths:
    dem: Path
    svf: Optional[Path]
    srf: Optional[Path]
    lc: Optional[Path]


def _read_yaml(path: Path) -> dict:
    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML(typ="safe")
        with Path(path).open("r", encoding="utf-8") as f:
            data = y.load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping in {path}")
        return data
    except Exception as exc:
        raise ValueError(f"Could not parse config at {path}: {exc}") from exc


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


def _rasterize_roi(
    geom: BaseGeometry,
    out_path: Path,
    shape: Tuple[int, int],
    transform: rasterio.Affine,
) -> None:
    mask = features.rasterize(
        [(geom, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    meta = {
        "driver": "AAIGrid",
        "dtype": "uint8",
        "nodata": 0,
        "width": mask.shape[1],
        "height": mask.shape[0],
        "count": 1,
        "crs": None,
        "transform": transform,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype("uint8"), 1)


def _mask_grid_outside_geometry(grid_path: Path, geom: BaseGeometry) -> None:
    """Set pixels outside the sub-domain polygon to nodata."""
    with rasterio.open(grid_path, "r+") as ds:
        arr = ds.read(1)
        inside = features.rasterize(
            [(geom, 1)],
            out_shape=(ds.height, ds.width),
            transform=ds.transform,
            fill=0,
            dtype="uint8",
        ).astype(bool)

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
) -> Path:
    cfg = copy.deepcopy(source_cfg)
    cfg["domain"] = domain
    cfg.setdefault("input_data", {}).setdefault("grids", {})
    cfg["input_data"]["grids"]["dir"] = str(grids_dir.resolve())
    cfg.setdefault("input_data", {}).setdefault("meteo", {})
    cfg["input_data"]["meteo"]["dir"] = str(meteo_dir.resolve())
    # Sub-domains may rely on nearby stations outside the clipped grid extent.
    # Use global station bounds to avoid dropping all stations in small tiles.
    cfg["input_data"]["meteo"]["bounds"] = "global"
    cfg["results_dir"] = str((sub_setup_dir / "results").resolve())
    out_yaml = sub_setup_dir / f"{sub_setup_dir.name}.yml"
    out_yaml.write_text(_to_yaml_text(cfg), encoding="utf-8")
    return out_yaml


def _copy_project_dir(source_project_dir: Path, target_project_dir: Path) -> Path:
    ignore_names = {"steps", "plots", "ensembles", "assim", "subdomains", "merged"}
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
    geom_roi: Polygon,
    transform: rasterio.Affine,
    global_shape: Tuple[int, int],
    global_transform: rasterio.Affine,
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
        _rasterize_roi(geom_roi, roi_dst, (int(global_shape[0]), int(global_shape[1])), global_transform)
        return roi_dst

    dem_out = grids_out / f"dem_{new_domain}_{resolution}.asc"
    _crop_grid(dem_src, dem_out, window, fill_value=-9999.0)
    target_shape = (int(window.height), int(window.width))
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

    # Enforce polygonal clipping (not only bounding window clipping).
    _mask_grid_outside_geometry(dem_out, geom_roi)
    if grid_paths.svf:
        _mask_grid_outside_geometry(svf_out, geom_roi)
    if grid_paths.srf:
        _mask_grid_outside_geometry(srf_out, geom_roi)
    if grid_paths.lc:
        _mask_grid_outside_geometry(lc_out, geom_roi)

    roi_dst = grids_out / f"roi_{new_domain}_{resolution}.asc"
    _rasterize_roi(geom_roi, roi_dst, (int(window.height), int(window.width)), transform)
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
) -> None:
    if not obs_dir.is_dir():
        logger.info("Obs directory {} not found; skipping station subset", obs_dir)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = {str(sid) for sid in (station_ids or []) if str(sid)}
    selected_ids: set[str] = set()
    stations_meta = obs_dir / "stations_snow_depth.csv"
    if stations_meta.is_file():
        meta_df = pd.read_csv(stations_meta)
        if {"x", "y"}.issubset(meta_df.columns):
            gdf = gpd.GeoDataFrame(
                meta_df,
                geometry=gpd.points_from_xy(meta_df["x"], meta_df["y"]),
                crs=crs,
            )
            buffered = geom.buffer(buffer_m) if buffer_m and buffer_m > 0 else geom
            meta_df = meta_df.loc[gdf.geometry.within(buffered)].copy()
        else:
            logger.warning("stations_snow_depth.csv missing x/y columns; skipping spatial filter")
            if requested_ids and "id" in meta_df.columns:
                meta_df = meta_df.loc[meta_df["id"].astype(str).isin(requested_ids)].copy()
        meta_df.to_csv(out_dir / "stations_snow_depth.csv", index=False)
        if "id" in meta_df.columns:
            selected_ids = {str(sid) for sid in meta_df["id"].dropna().astype(str)}

    if not selected_ids and requested_ids:
        selected_ids = set(requested_ids)

    if selected_ids:
        for sid in sorted(selected_ids):
            copied = False
            for ext in (".csv", ".nc"):
                src = obs_dir / f"{sid}{ext}"
                if src.exists():
                    _copy_or_link(src, out_dir / src.name)
                    copied = True
                    break
            if not copied:
                logger.debug("No station obs series found for id {}", sid)
        return

    copied_any = False
    for pattern in ("*.csv", "*.nc"):
        for src in sorted(obs_dir.glob(pattern)):
            if src.name == "stations_snow_depth.csv":
                continue
            _copy_or_link(src, out_dir / src.name)
            copied_any = True
    if not copied_any:
        logger.info("No station obs files found in {}", obs_dir)


def prepare_subdomains(
    *,
    setup_dir: Path,
    project_dir: Path,
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
) -> SubdomainManifest:
    """Prepare per-sub-domain setups under `<project>/subdomains/<id>`."""
    logger.debug("Preparing sub-domains for setup={} regions={}", setup_dir, regions_path)
    if clip_mode not in {"window", "roi-symlink"}:
        raise ValueError("clip_mode must be 'window' or 'roi-symlink'")

    setup_dir = Path(setup_dir).resolve()
    project_dir = Path(project_dir).resolve()
    ensure_run_mode(project_dir, expected="subdomain", write_if_missing=True)
    setup_yaml = find_setup_yaml(setup_dir)
    project_yaml = find_project_yaml(project_dir)
    setup_cfg = _read_yaml(setup_yaml)
    project_cfg = _read_yaml(project_yaml)

    grid_buffer_m = float(grid_buffer_m if grid_buffer_m is not None else 0.0)
    station_buffer_m = float(station_buffer_m)
    roi_buffer_m = float(roi_buffer_m)

    if "meteo" not in (setup_cfg.get("input_data") or {}) or "grids" not in (setup_cfg.get("input_data") or {}):
        raise ValueError("Setup config input_data must define both 'grids' and 'meteo' sections.")

    grids_rel = _nested_dir(setup_cfg, ("input_data", "grids", "dir"), "grids")
    meteo_rel = _nested_dir(setup_cfg, ("input_data", "meteo", "dir"), "meteo")
    obs_station_rel = _nested_dir(project_cfg, ("obs", "stations", "dir"), "obs/stations")
    raw_snowcover_rel = _nested_dir(project_cfg, ("obs", "snowcover", "dir"), "obs/snowcover")
    raw_wetsnow_rel = _nested_dir(project_cfg, ("obs", "wetsnow", "dir"), "obs/wetsnow")

    grids_dir = Path(abspath_relative_to(setup_dir, grids_rel))
    meteo_dir = Path(abspath_relative_to(setup_dir, meteo_rel))
    obs_dir = Path(obs_stations_dir) if obs_stations_dir else Path(abspath_relative_to(setup_dir, obs_station_rel))
    raw_snowcover_dir = Path(abspath_relative_to(setup_dir, raw_snowcover_rel))
    raw_wetsnow_dir = Path(abspath_relative_to(setup_dir, raw_wetsnow_rel))
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
        if id_field == "id" and "region_id" in gdf.columns:
            effective_id_field = "region_id"
            logger.info("Regions file has no 'id' field; using fallback id field 'region_id'.")
        else:
            raise KeyError(f"Regions file missing id field '{id_field}'")
    if gdf.crs and crs_str and gdf.crs.to_string().lower() != crs_str.lower():
        raise ValueError(f"CRS mismatch between regions ({gdf.crs}) and setup ({crs_str})")
    if len(gdf) < 2:
        raise ValueError(
            f"Sub-domain mode requires at least 2 polygons in {regions_path} (got {len(gdf)})."
        )
    _check_no_overlap(gdf.geometry, area_tol=float(overlap_area_tol_m2))

    subdomain_root = (Path(subdomain_root) if subdomain_root else (project_dir / "subdomains")).resolve()
    if overwrite:
        for derived_dir in (subdomain_root, project_dir / "merged", project_dir / "plots"):
            if derived_dir.is_dir():
                shutil.rmtree(derived_dir)
    subdomain_root.mkdir(parents=True, exist_ok=True)

    project_name = project_dir.name
    lc_path = _find_grid(grids_dir, "lc", domain, resolution)
    svf_path = _find_grid(grids_dir, "svf", domain, resolution)
    srf_path = _find_grid(grids_dir, "srf", domain, resolution)
    grid_paths = GridPaths(dem=dem_path, svf=svf_path, srf=srf_path, lc=lc_path)

    manifest = SubdomainManifest(
        run_mode="subdomain",
        setup_dir=setup_dir,
        project_dir=project_dir,
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
        geom_roi = geom.buffer(roi_buffer_m) if roi_buffer_m else geom
        geom_extent = geom.buffer(grid_buffer_m) if grid_buffer_m else geom

        win = _window_for_geometry(geom_extent, transform, (rows, cols))
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
        project_out = sub_setup_dir / "projects" / project_name
        for d in (grids_out, meteo_out, obs_out, env_out, project_out):
            d.mkdir(parents=True, exist_ok=True)

        sub_domain = f"{domain}_{clean_id}"
        roi_raster_path = _prepare_grids(
            grid_paths=grid_paths,
            grids_out=grids_out,
            clip_mode=clip_mode,
            domain=domain,
            new_domain=sub_domain,
            resolution=resolution,
            window=win,
            geom_roi=geom_roi,
            transform=sub_transform,
            global_shape=(rows, cols),
            global_transform=transform,
        )

        selected_station_ids = _prepare_meteo(
            meteo_dir=meteo_dir,
            out_dir=meteo_out,
            geom=geom,
            buffer_m=station_buffer_m,
            crs=crs_str,
        )
        _prepare_obs_station_subset(
            obs_dir=obs_dir,
            out_dir=obs_out,
            geom=geom,
            buffer_m=station_buffer_m,
            crs=crs_str,
            station_ids=selected_station_ids,
        )
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
        )
        sub_project_yaml = _copy_project_dir(project_dir, project_out)

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
        )
        logger.info(
            "Prepared sub-domain {} ({}x{}, window r{} c{})",
            clean_id,
            sub_rows,
            sub_cols,
            win.row_off,
            win.col_off,
        )

    manifest_path = subdomain_root / "subdomain_manifest.json"
    manifest.save(manifest_path)
    logger.info("Wrote manifest -> {}", manifest_path)
    return manifest
