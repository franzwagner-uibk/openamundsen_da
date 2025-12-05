"""Helpers for masking AOIs with glacier outlines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import geopandas as gpd
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.roi import read_single_roi

GLACIER_FILENAME = "glaciers.gpkg"
ROI_FILENAME = "roi.gpkg"
GLACIER_MASK_BLOCK = "glacier_mask"
GLACIER_MASK_ENABLED = "enabled"
GLACIER_MASK_PATH = "path"


@dataclass(frozen=True)
class GlacierMaskConfig:
    enabled: bool
    path: Path | None


def _default_glacier_path(project_dir: Path) -> Path:
    return Path(project_dir) / "env" / GLACIER_FILENAME


def default_roi_path(project_dir: Path) -> Path:
    """Return the conventional ROI path under env/."""
    return Path(project_dir) / "env" / ROI_FILENAME


def resolve_glacier_mask(project_dir: Path) -> GlacierMaskConfig:
    """Return effective glacier mask config from project.yml and filesystem."""
    project_dir = Path(project_dir)
    try:
        proj_yaml = find_project_yaml(project_dir)
        proj_cfg = _read_yaml_file(proj_yaml) or {}
        da_cfg = proj_cfg.get("data_assimilation") or {}
        gm_cfg = da_cfg.get(GLACIER_MASK_BLOCK) or {}
        enabled_cfg = gm_cfg.get(GLACIER_MASK_ENABLED)
        path_cfg = gm_cfg.get(GLACIER_MASK_PATH)
    except Exception:
        enabled_cfg = None
        path_cfg = None

    # Determine requested enabled flag
    if enabled_cfg is False:
        return GlacierMaskConfig(enabled=False, path=None)

    # Resolve path (config override or default)
    if path_cfg:
        gp = Path(path_cfg)
        if not gp.is_absolute():
            gp = project_dir / gp
    else:
        gp = _default_glacier_path(project_dir)

    if gp.is_file():
        # If explicitly enabled or implicitly via presence, use it.
        return GlacierMaskConfig(enabled=True, path=gp)

    # File missing: only allow if not explicitly enabled
    if enabled_cfg:
        # Explicitly enabled but missing -> disabled with None path
        return GlacierMaskConfig(enabled=False, path=None)

    # Implicit default: disabled when file absent
    return GlacierMaskConfig(enabled=False, path=None)


def find_glacier_layer(project_dir: Path) -> Path | None:
    """Backward-compatible finder; returns path when enabled."""
    cfg = resolve_glacier_mask(project_dir)
    return cfg.path if cfg.enabled else None


def mask_aoi_with_glaciers(
    aoi_path: Path,
    *,
    glacier_path: Path | None = None,
    required_field: str | None = None,
    to_crs: object | None = None,
    allow_missing_glaciers: bool = True,
) -> Tuple[gpd.GeoDataFrame, str]:
    """Read a single-feature AOI and subtract glaciers when provided.

    If ``glacier_path`` exists, all glacier geometries are unioned (in the AOI
    CRS) and subtracted from the AOI polygon. Raises ``ValueError`` when the
    difference is empty. When ``glacier_path`` is missing and
    ``allow_missing_glaciers`` is False, a ``FileNotFoundError`` is raised.
    """

    gdf, region_id = read_single_roi(
        aoi_path,
        required_field=required_field,
        to_crs=None,  # defer reprojection until after masking
    )

    if glacier_path is not None:
        gp = Path(glacier_path)
        if not gp.exists():
            if not allow_missing_glaciers:
                raise FileNotFoundError(f"Glacier file not found: {gp}")
        else:
            glaciers = gpd.read_file(gp)
            if not glaciers.empty:
                if glaciers.crs is None:
                    raise ValueError(f"Glacier layer {gp} has no CRS")
                if gdf.crs is None:
                    raise ValueError("AOI has no CRS; unable to align with glacier layer")
                glaciers = glaciers.to_crs(gdf.crs)
                # Only glaciers intersecting the ROI are used.
                glaciers = glaciers[glaciers.intersects(gdf.geometry.iloc[0])]
                union_geom: BaseGeometry = unary_union(glaciers.geometry.values)
                if not union_geom.is_empty:
                    geom = gdf.geometry.iloc[0]
                    diff = geom.difference(union_geom)
                    if diff.is_empty:
                        raise ValueError(f"AOI fully masked by glaciers in {gp}")
                    if not diff.is_valid:
                        diff = diff.buffer(0)
                    gdf = gdf.copy()
                    gdf.geometry = [diff]

    if to_crs is not None:
        if gdf.crs is None:
            raise ValueError("AOI has no CRS; unable to reproject to target CRS")
        gdf = gdf.to_crs(to_crs)

    return gdf, region_id


__all__ = [
    "GlacierMaskConfig",
    "GLACIER_FILENAME",
    "resolve_glacier_mask",
    "default_roi_path",
    "find_glacier_layer",
    "mask_aoi_with_glaciers",
]
