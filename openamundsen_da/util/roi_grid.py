"""ROI raster/vector helpers aligned to setup grid naming conventions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio import features
from shapely.geometry import shape as shp_shape
from shapely.ops import unary_union

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import abspath_relative_to, find_plain_setup_yaml, find_setup_yaml


@dataclass(frozen=True)
class SetupGridSpec:
    setup_dir: Path
    grids_dir: Path
    domain: str
    resolution: str
    crs: str | None
    rows: int
    cols: int
    transform: Affine
    dem_path: Path
    roi_grid_path: Path


def _format_resolution(resolution: object) -> str:
    if isinstance(resolution, (int, float)):
        if float(resolution).is_integer():
            return str(int(resolution))
    return str(resolution).strip()


def _find_grid_file(grids_dir: Path, prefix: str, domain: str, resolution: str) -> Path:
    base = f"{prefix}_{domain}_{resolution}"
    ext_order = (".asc", ".tif", ".tiff")

    for ext in ext_order:
        exact = grids_dir / f"{base}{ext}"
        if exact.is_file():
            return exact

    suffix_candidates: list[Path] = []
    for ext in ext_order:
        suffix_candidates.extend(sorted(grids_dir.glob(f"{base}_*{ext}")))
    if len(suffix_candidates) == 1:
        return suffix_candidates[0]
    if len(suffix_candidates) > 1:
        names = ", ".join(p.name for p in suffix_candidates)
        raise FileExistsError(f"Multiple {prefix} grids match {base}_* under {grids_dir}: {names}")

    if prefix == "dem":
        fallback_names = ("dem.asc", "dem.tif", "dem.tiff")
        for name in fallback_names:
            cand = grids_dir / name
            if cand.is_file():
                return cand
        fallback_candidates: list[Path] = []
        for ext in ext_order:
            fallback_candidates.extend(sorted(grids_dir.glob(f"dem_*{ext}")))
        if len(fallback_candidates) == 1:
            return fallback_candidates[0]
        if len(fallback_candidates) > 1:
            names = ", ".join(p.name for p in fallback_candidates)
            raise FileExistsError(f"Multiple fallback DEM grids found under {grids_dir}: {names}")

    raise FileNotFoundError(
        f"Grid not found for {prefix}: expected {base}.asc (or .tif/.tiff) under {grids_dir}"
    )


def _find_setup_yaml_for_roi(setup_dir: Path) -> Path:
    try:
        return find_setup_yaml(setup_dir)
    except FileNotFoundError as exc:
        if "missing projects/" not in str(exc):
            raise
    return find_plain_setup_yaml(setup_dir)


def resolve_setup_grid_spec(setup_dir: Path) -> SetupGridSpec:
    setup_dir = Path(setup_dir).resolve()
    setup_yaml = _find_setup_yaml_for_roi(setup_dir)
    cfg = _read_yaml_file(setup_yaml) or {}

    domain = cfg.get("domain")
    resolution_raw = cfg.get("resolution")
    if not domain:
        raise ValueError(f"{setup_yaml} missing required key 'domain'")
    if resolution_raw is None:
        raise ValueError(f"{setup_yaml} missing required key 'resolution'")

    resolution = _format_resolution(resolution_raw)
    grids_rel = (((cfg.get("input_data") or {}).get("grids") or {}).get("dir")) or "grids"
    grids_dir = Path(abspath_relative_to(setup_dir, Path(grids_rel)))
    if not grids_dir.is_dir():
        raise FileNotFoundError(f"Grids directory not found: {grids_dir}")

    dem_path = _find_grid_file(grids_dir, "dem", str(domain), resolution)
    with rasterio.open(dem_path) as dem:
        rows = int(dem.height)
        cols = int(dem.width)
        transform = dem.transform
        dem_crs = dem.crs.to_string() if dem.crs else None

    setup_crs = cfg.get("crs")
    crs = str(setup_crs) if setup_crs is not None else dem_crs

    roi_grid_path = grids_dir / f"roi_{domain}_{resolution}.asc"
    return SetupGridSpec(
        setup_dir=setup_dir,
        grids_dir=grids_dir,
        domain=str(domain),
        resolution=resolution,
        crs=crs,
        rows=rows,
        cols=cols,
        transform=transform,
        dem_path=dem_path,
        roi_grid_path=roi_grid_path,
    )


def discover_setup_roi_vector(setup_dir: Path) -> Path | None:
    env_dir = Path(setup_dir) / "env"
    if not env_dir.is_dir():
        return None

    priority = ["roi.gpkg", "subdomains.gpkg", "roi.shp", "subdomains.shp"]
    for name in priority:
        cand = env_dir / name
        if cand.is_file():
            return cand

    candidates = sorted(list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp")))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(p.name for p in candidates)
    raise FileExistsError(f"Multiple ROI vector candidates found under {env_dir}: {names}")


def _read_mask_from_grid(roi_grid_path: Path, spec: SetupGridSpec) -> np.ndarray:
    with rasterio.open(roi_grid_path) as ds:
        arr = ds.read(1)
        if int(ds.height) != spec.rows or int(ds.width) != spec.cols:
            raise ValueError(
                f"ROI grid shape mismatch for {roi_grid_path}: {(ds.height, ds.width)} vs {(spec.rows, spec.cols)}"
            )
        if tuple(ds.transform) != tuple(spec.transform):
            raise ValueError(f"ROI grid transform mismatch for {roi_grid_path}")
        mask = arr.astype(bool)
    if int(np.count_nonzero(mask)) == 0:
        raise ValueError(f"ROI grid is empty: {roi_grid_path}")
    return mask


def _write_mask_grid(mask: np.ndarray, out_path: Path, *, transform: Affine, crs: str | None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "driver": "AAIGrid",
        "dtype": "uint8",
        "nodata": 0,
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "count": 1,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask.astype("uint8"), 1)


def _union_geometry(roi_vector_path: Path, target_crs: str | None) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(roi_vector_path)
    if gdf.empty:
        raise ValueError(f"ROI vector has no features: {roi_vector_path}")
    if target_crs:
        if gdf.crs is None:
            raise ValueError(f"ROI vector has no CRS: {roi_vector_path}")
        if gdf.crs.to_string().lower() != str(target_crs).lower():
            gdf = gdf.to_crs(target_crs)
    geom = unary_union([geom for geom in gdf.geometry if geom is not None and not geom.is_empty])
    if geom is None or geom.is_empty:
        raise ValueError(f"ROI geometry is empty after union: {roi_vector_path}")
    attrs = {"id": pd.Series(["roi"], dtype="object")}
    return gpd.GeoDataFrame(attrs, geometry=[geom], crs=gdf.crs)


def ensure_setup_roi_grid(
    setup_dir: Path,
    *,
    roi_vector_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    spec = resolve_setup_grid_spec(setup_dir)
    roi_grid_path = spec.roi_grid_path

    if roi_grid_path.is_file() and not overwrite:
        _read_mask_from_grid(roi_grid_path, spec)
        return roi_grid_path

    source_vector = Path(roi_vector_path) if roi_vector_path is not None else discover_setup_roi_vector(spec.setup_dir)
    if source_vector is None:
        raise FileNotFoundError(
            f"Missing ROI grid {roi_grid_path} and no ROI vector under {spec.setup_dir / 'env'}."
        )
    if not source_vector.is_file():
        raise FileNotFoundError(f"ROI vector not found: {source_vector}")

    roi_gdf = _union_geometry(source_vector, spec.crs)
    mask = features.rasterize(
        [(roi_gdf.geometry.iloc[0], 1)],
        out_shape=(spec.rows, spec.cols),
        transform=spec.transform,
        fill=0,
        dtype="uint8",
    ).astype(bool)
    if int(np.count_nonzero(mask)) == 0:
        raise ValueError(f"Generated ROI grid would be empty from {source_vector}")

    _write_mask_grid(mask, roi_grid_path, transform=spec.transform, crs=spec.crs)
    _read_mask_from_grid(roi_grid_path, spec)
    return roi_grid_path


def load_setup_roi_mask(
    setup_dir: Path,
    *,
    ensure_grid: bool = False,
    roi_vector_path: Optional[Path] = None,
) -> tuple[np.ndarray, SetupGridSpec, Path]:
    spec = resolve_setup_grid_spec(setup_dir)
    roi_grid_path = spec.roi_grid_path
    if ensure_grid:
        roi_grid_path = ensure_setup_roi_grid(setup_dir, roi_vector_path=roi_vector_path)
    if not roi_grid_path.is_file():
        raise FileNotFoundError(f"ROI grid not found: {roi_grid_path}")
    mask = _read_mask_from_grid(roi_grid_path, spec)
    return mask, spec, roi_grid_path


def ensure_setup_roi_vector(setup_dir: Path) -> Path:
    setup_dir = Path(setup_dir).resolve()
    existing = discover_setup_roi_vector(setup_dir)
    if existing is not None and existing.is_file():
        return existing

    mask, spec, _ = load_setup_roi_mask(setup_dir, ensure_grid=True)
    shapes = [
        shp_shape(geom)
        for geom, value in features.shapes(mask.astype("uint8"), mask=mask, transform=spec.transform)
        if int(value) == 1
    ]
    if not shapes:
        raise ValueError(f"Cannot generate ROI vector from empty ROI grid: {spec.roi_grid_path}")
    geom = unary_union(shapes)
    if geom is None or geom.is_empty:
        raise ValueError(f"Cannot generate ROI vector from {spec.roi_grid_path}: geometry empty after union")

    env_dir = setup_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    roi_path = env_dir / "roi.gpkg"
    attrs = {"id": pd.Series(["roi"], dtype="object")}
    gdf = gpd.GeoDataFrame(attrs, geometry=[geom], crs=spec.crs)
    gdf.to_file(roi_path, driver="GPKG")
    return roi_path


__all__ = [
    "SetupGridSpec",
    "discover_setup_roi_vector",
    "ensure_setup_roi_grid",
    "ensure_setup_roi_vector",
    "load_setup_roi_mask",
    "resolve_setup_grid_spec",
]
