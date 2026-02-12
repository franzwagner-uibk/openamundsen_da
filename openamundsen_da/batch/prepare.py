"""Preparation utilities for sub-domain open-loop runs."""

from __future__ import annotations

import copy
import math
import shutil
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio import features, windows
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from openamundsen_da.io.paths import abspath_relative_to
from openamundsen_da.batch.manifest import BatchManifest, SubregionMeta, WindowSpec


# ---- Helper dataclasses -----------------------------------------------------

@dataclass
class GridPaths:
    dem: Path
    svf: Optional[Path]
    srf: Optional[Path]
    lc: Optional[Path]


# ---- Generic helpers --------------------------------------------------------

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


def _to_yaml_text(data: dict) -> str:
    import ruamel.yaml as _yaml

    y = _yaml.YAML()
    y.default_flow_style = False
    buf = io.StringIO()
    y.dump(data, buf)
    return buf.getvalue()

def _sanitize_id(raw: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(raw))
    return clean.strip("_") or "subregion"


def _read_project_config(path: Path) -> dict:
    cfg = _read_yaml(path)
    required = ("domain", "resolution", "input_data")
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config {path} missing required key '{key}'")
    return cfg


def _find_grid(grids_dir: Path, prefix: str, domain: str, resolution: str) -> Optional[Path]:
    patt = f"{prefix}_{domain}_{resolution}"
    candidates = sorted(grids_dir.glob(f"{patt}*.asc"))
    if candidates:
        return candidates[0]
    candidates = sorted(grids_dir.glob(f"{patt}*.tif"))
    return candidates[0] if candidates else None


def _check_no_overlap(geoms: Iterable[BaseGeometry], *, area_tol: float = 0.0) -> None:
    """Raise if polygons overlap above a tolerance (m²)."""
    geoms = list(geoms)
    for i, g1 in enumerate(geoms):
        for g2 in geoms[i + 1 :]:
            if not g1.intersects(g2):
                continue
            inter_area = g1.intersection(g2).area
            if inter_area > area_tol:
                raise ValueError(
                    f"Detected overlapping subregions (overlap area {inter_area:.3f} m² exceeds tolerance {area_tol} m²); please fix geometry or raise tolerance."
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
    win = win.intersection(full)
    return win


def _crop_grid(src: Path, dst: Path, win: windows.Window, fill_value=None) -> None:
    with rasterio.open(src) as ds:
        fv = fill_value if fill_value is not None else ds.nodata
        data = ds.read(1, window=win, boundless=True, fill_value=fv)
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


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists():
            dst.unlink()
        dst.symlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


# ---- Public API -------------------------------------------------------------

def prepare_batch(
    *,
    base_config: Path,
    regions_path: Path,
    batch_root: Path,
    id_field: str = "id",
    clip_mode: str = "window",
    station_buffer_m: float = 50_000.0,
    roi_buffer_m: float = 0.0,
    grid_buffer_m: Optional[float] = None,
    obs_stations_dir: Optional[Path] = None,
    overlap_area_tol_m2: float = 100.0,
    sliver_fix_m: float = 0.0,
    overwrite: bool = False,
) -> BatchManifest:
    """Prepare per-sub-domain setups and return a manifest."""
    logger.debug("Preparing batch with base_config={} regions={}", base_config, regions_path)
    if clip_mode not in {"window", "roi-symlink"}:
        raise ValueError("clip_mode must be 'window' or 'roi-symlink'")
    cfg = _read_project_config(base_config)
    base_dir = base_config.parent
    if "meteo" not in (cfg.get("input_data") or {}) or "grids" not in (cfg.get("input_data") or {}):
        raise ValueError("Config input_data must define both 'grids' and 'meteo' sections.")

    grid_buffer_m = float(grid_buffer_m if grid_buffer_m is not None else station_buffer_m)
    station_buffer_m = float(station_buffer_m)
    roi_buffer_m = float(roi_buffer_m)

    grids_dir = Path(abspath_relative_to(base_dir, cfg["input_data"]["grids"]["dir"]))
    meteo_dir = Path(abspath_relative_to(base_dir, cfg["input_data"]["meteo"]["dir"]))
    obs_dir = Path(obs_stations_dir) if obs_stations_dir else (base_dir / "obs" / "stations")

    domain = str(cfg["domain"])
    resolution = str(cfg["resolution"])
    crs_expected = cfg.get("crs")

    dem_path = _find_grid(grids_dir, "dem", domain, resolution)
    if dem_path is None:
        raise FileNotFoundError(f"DEM not found in {grids_dir} for domain={domain}, resolution={resolution}")
    rows, cols, transform, res_val, dem_crs = _dem_metadata(dem_path)
    if crs_expected and dem_crs and crs_expected.lower() != dem_crs.lower():
        raise ValueError(f"CRS mismatch: config {crs_expected} vs DEM {dem_crs}")
    crs_str = crs_expected or dem_crs

    gdf = gpd.read_file(regions_path)
    # Clean geometries (buffer(0)) and optional sliver smoothing
    gdf["geometry"] = gdf.geometry.buffer(0)
    if sliver_fix_m and sliver_fix_m > 0:
        gdf["geometry"] = gdf.geometry.buffer(-sliver_fix_m).buffer(sliver_fix_m)
    if id_field not in gdf.columns:
        raise KeyError(f"Region file missing id field '{id_field}'")
    if gdf.crs and crs_str and gdf.crs.to_string().lower() != crs_str.lower():
        raise ValueError(f"CRS mismatch between regions ({gdf.crs}) and config ({crs_str})")
    _check_no_overlap(gdf.geometry, area_tol=float(overlap_area_tol_m2))

    batch_root = Path(batch_root)
    setups_root = batch_root / "setups"
    merged_root = batch_root / "merged"
    for d in (setups_root, merged_root):
        d.mkdir(parents=True, exist_ok=True)

    lc_path = _find_grid(grids_dir, "lc", domain, resolution)
    svf_path = _find_grid(grids_dir, "svf", domain, resolution)
    srf_path = _find_grid(grids_dir, "srf", domain, resolution)
    grid_paths = GridPaths(dem=dem_path, svf=svf_path, srf=srf_path, lc=lc_path)

    manifest = BatchManifest(
        batch_name=batch_root.name,
        base_config=base_config.resolve(),
        regions_path=regions_path.resolve(),
        id_field=id_field,
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
    )

    for _, row in gdf.iterrows():
        raw_id = row[id_field]
        geom = row.geometry
        if geom is None or geom.is_empty:
            raise ValueError(f"Region {raw_id} has empty geometry")
        if geom.geom_type == "MultiPolygon":
            geom = unary_union(geom)
        clean_id = _sanitize_id(raw_id)
        label = str(raw_id)

        if not isinstance(geom, Polygon):
            raise ValueError(f"Region {raw_id} is not a polygon")

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

        setup_dir = setups_root / clean_id
        grids_out = setup_dir / "grids"
        meteo_out = setup_dir / "meteo"
        obs_out = setup_dir / "obs" / "stations"
        results_dir = setup_dir / "results"

        if setup_dir.exists() and not overwrite:
            raise FileExistsError(f"{setup_dir} already exists. Use --overwrite to rebuild.")

        if setup_dir.exists() and overwrite:
            shutil.rmtree(setup_dir)

        grids_out.mkdir(parents=True, exist_ok=True)
        meteo_out.mkdir(parents=True, exist_ok=True)
        obs_out.mkdir(parents=True, exist_ok=True)

        # Copy/crop grids
        _prepare_grids(
            grid_paths=grid_paths,
            grids_out=grids_out,
            clip_mode=clip_mode,
            domain=domain,
            new_domain=f"{domain}_{clean_id}",
            resolution=resolution,
            window=win,
            geom_roi=geom_roi,
            transform=sub_transform,
            global_shape=(rows, cols),
            global_transform=transform,
        )

        # Meteo subset
        _prepare_meteo(
            meteo_dir=meteo_dir,
            out_dir=meteo_out,
            geom=geom,
            buffer_m=station_buffer_m,
            crs=crs_str,
        )

        # Obs subset (best-effort)
        _prepare_obs(
            obs_dir=obs_dir,
            out_dir=obs_out,
            geom=geom,
            buffer_m=station_buffer_m,
            crs=crs_str,
        )

        # Config for subregion
        cfg_out = setup_dir / "config.yml"
        _write_subregion_config(
            base_cfg=cfg,
            dest=cfg_out,
            domain=f"{domain}_{clean_id}",
            grids_dir=grids_out,
            meteo_dir=meteo_out,
            results_dir=results_dir,
            geom=geom,
            station_buffer_m=station_buffer_m,
            crs=crs_str,
        )

        # ROI path recorded for merging
        roi_path = grids_out / f"roi_{domain}_{clean_id}_{resolution}.asc"
        if not roi_path.exists():
            roi_path = grids_out / f"roi_{domain}_{resolution}.asc"

        meta = SubregionMeta(
            id=clean_id,
            label=label,
            setup_dir=setup_dir,
            config_path=cfg_out,
            grids_dir=grids_out,
            meteo_dir=meteo_out,
            obs_dir=obs_out,
            results_dir=results_dir,
            roi_path=roi_path,
            window=window_spec,
            transform=tuple(sub_transform),
            bounds=geom.bounds,
            crs=crs_str,
            status="pending",
        )
        manifest.subregions[clean_id] = meta
        logger.info("Prepared subregion {} ({}x{}, window r{} c{})", clean_id, sub_rows, sub_cols, win.row_off, win.col_off)

    manifest_path = batch_root / "batch_manifest.json"
    manifest.save(manifest_path)
    logger.info("Wrote manifest -> {}", manifest_path)
    return manifest


# ---- Internal steps --------------------------------------------------------

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
) -> None:
    """Copy/crop DEM + ancillary grids and build ROI."""
    # DEM is required
    dem_src = grid_paths.dem
    if clip_mode == "roi-symlink":
        dem_dst = grids_out / f"dem_{new_domain}_{resolution}.asc"
        _copy_or_link(dem_src, dem_dst)
        svf_src = grid_paths.svf
        if svf_src:
            svf_dst = grids_out / f"svf_{new_domain}_{resolution}.asc"
            _copy_or_link(svf_src, svf_dst)
        if grid_paths.srf:
            srf_dst = grids_out / f"srf_{new_domain}_{resolution}.asc"
            _copy_or_link(grid_paths.srf, srf_dst)
        if grid_paths.lc:
            lc_dst = grids_out / f"lc_{new_domain}_{resolution}.asc"
            _copy_or_link(grid_paths.lc, lc_dst)
        roi_dst = grids_out / f"roi_{new_domain}_{resolution}.asc"
        shape = (int(global_shape[0]), int(global_shape[1]))
        _rasterize_roi(geom_roi, roi_dst, shape, global_transform)
        return

    dem_dst = grids_out / f"dem_{new_domain}_{resolution}.asc"
    _crop_grid(dem_src, dem_dst, window, fill_value=np.nan)

    if grid_paths.svf:
        svf_dst = grids_out / f"svf_{new_domain}_{resolution}.asc"
        _crop_grid(grid_paths.svf, svf_dst, window, fill_value=np.nan)
    if grid_paths.srf:
        srf_dst = grids_out / f"srf_{new_domain}_{resolution}.asc"
        _crop_grid(grid_paths.srf, srf_dst, window, fill_value=np.nan)
    if grid_paths.lc:
        lc_dst = grids_out / f"lc_{new_domain}_{resolution}.asc"
        _crop_grid(grid_paths.lc, lc_dst, window, fill_value=0)

    roi_dst = grids_out / f"roi_{new_domain}_{resolution}.asc"
    _rasterize_roi(geom_roi, roi_dst, (int(window.height), int(window.width)), transform)


def _prepare_meteo(
    *,
    meteo_dir: Path,
    out_dir: Path,
    geom: Polygon,
    buffer_m: float,
    crs: Optional[str],
) -> None:
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
    sel = gdf.geometry.within(buffered)
    selected = stations.loc[sel].copy()
    if selected.empty:
        raise ValueError(f"No meteo stations found within buffer {buffer_m} m for region.")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(out_dir / "stations.csv", index=False)

    # copy files per station id
    for sid in selected["id"]:
        src_csv = meteo_dir / f"{sid}.csv"
        src_nc = meteo_dir / f"{sid}.nc"
        src = src_csv if src_csv.exists() else src_nc if src_nc.exists() else None
        if src is None:
            logger.warning("No meteo file found for station {}", sid)
            continue
        _copy_or_link(src, out_dir / src.name)
    # keep format info
    if (meteo_dir / "meteo_format.txt").is_file():
        _copy_or_link(meteo_dir / "meteo_format.txt", out_dir / "meteo_format.txt")


def _prepare_obs(
    *,
    obs_dir: Path,
    out_dir: Path,
    geom: Polygon,
    buffer_m: float,
    crs: Optional[str],
) -> None:
    if not obs_dir.is_dir():
        logger.info("Obs directory {} not found; skipping obs subset", obs_dir)
        return
    stations_meta = obs_dir / "stations_snow_depth.csv"
    if stations_meta.is_file():
        meta_df = pd.read_csv(stations_meta)
        if not {"x", "y"}.issubset(meta_df.columns):
            logger.warning("stations_snow_depth.csv missing x/y columns; skipping spatial filter")
        else:
            gdf = gpd.GeoDataFrame(
                meta_df,
                geometry=gpd.points_from_xy(meta_df["x"], meta_df["y"]),
                crs=crs,
            )
            buffered = geom.buffer(buffer_m) if buffer_m and buffer_m > 0 else geom
            sel = gdf.geometry.within(buffered)
            meta_df = meta_df.loc[sel].copy()
        meta_df.to_csv(out_dir / "stations_snow_depth.csv", index=False)
        ids = set(meta_df["id"].astype(str))
    else:
        ids = set()

    out_dir.mkdir(parents=True, exist_ok=True)
    if not ids:
        return

    for sid in ids:
        for ext in (".csv", ".nc"):
            src = obs_dir / f"{sid}{ext}"
            if src.exists():
                _copy_or_link(src, out_dir / src.name)
                break


def _write_subregion_config(
    *,
    base_cfg: dict,
    dest: Path,
    domain: str,
    grids_dir: Path,
    meteo_dir: Path,
    results_dir: Path,
    geom: Polygon,
    station_buffer_m: float,
    crs: Optional[str],
) -> None:
    cfg = copy.deepcopy(base_cfg)
    cfg["domain"] = domain
    cfg["results_dir"] = str(results_dir.resolve())
    cfg["extend_roi_with_stations"] = True

    cfg.setdefault("input_data", {}).setdefault("grids", {})
    cfg["input_data"]["grids"]["dir"] = str(grids_dir.resolve())
    cfg.setdefault("input_data", {}).setdefault("meteo", {})
    cfg["input_data"]["meteo"]["dir"] = str(meteo_dir.resolve())

    # Filter explicit point outputs if present
    ts_cfg = ((cfg.get("output_data") or {}).get("timeseries") or {})
    if ts_cfg.get("points"):
        pts = ts_cfg["points"]
        df = pd.DataFrame(pts)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs=crs)
        buffered = geom.buffer(station_buffer_m) if station_buffer_m and station_buffer_m > 0 else geom
        mask = gdf.geometry.within(buffered)
        filtered = df.loc[mask]
        cfg["output_data"]["timeseries"]["points"] = filtered.to_dict(orient="records")

    # Normalize grid output format casing (common user typo)
    grids_cfg = ((cfg.get("output_data") or {}).get("grids") or {})
    if isinstance(grids_cfg, dict):
        fmt = grids_cfg.get("format")
        if isinstance(fmt, str):
            grids_cfg["format"] = fmt.lower()
            cfg["output_data"]["grids"] = grids_cfg

    dest.write_text(_to_yaml_text(cfg), encoding="utf-8")
