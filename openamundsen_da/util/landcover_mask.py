"""Land-cover based masking for DA grids and observations.

This module replaces the former glacier mask logic. It resolves the land-cover
file from project.yml, reads mask settings, reprojects the mask to a target
grid, and applies the requested exclusions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import rasterio
from affine import Affine
from loguru import logger
from pyproj import CRS
from rasterio.warp import Resampling, reproject

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml

LC_MASK_BLOCK = "landcover_mask"
LC_MASK_ENABLED = "enabled"
LC_MASK_CLASSES = "classes_to_exclude"


@dataclass(frozen=True)
class LandcoverMaskConfig:
    enabled: bool
    path: Path | None
    classes: tuple[int, ...]
    project_crs: CRS


def _format_resolution(resolution: object) -> str:
    if isinstance(resolution, (int, float)):
        if float(resolution).is_integer():
            return str(int(resolution))
    return str(resolution).strip()


def _derive_landcover_path(project_dir: Path, domain: str, resolution: object) -> Path:
    """Find land-cover file matching lc_<domain>_<resolution>*.asc under grids/."""
    grids_dir = Path(project_dir) / "grids"
    base = f"lc_{domain}_{_format_resolution(resolution)}"
    candidates = sorted(grids_dir.glob(f"{base}*.asc"))
    if not candidates:
        raise FileNotFoundError(f"No land-cover file matching {base}*.asc under {grids_dir}")
    return candidates[0]


def resolve_landcover_mask(project_dir: Path) -> LandcoverMaskConfig:
    """Return land-cover mask configuration from project.yml."""
    proj_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(proj_yaml) or {}
    da_cfg = cfg.get("data_assimilation") or {}
    lc_cfg = da_cfg.get(LC_MASK_BLOCK) or {}
    enabled = bool(lc_cfg.get(LC_MASK_ENABLED, False))
    classes_raw = lc_cfg.get(LC_MASK_CLASSES) or []
    classes: tuple[int, ...] = tuple(int(c) for c in classes_raw) if enabled else tuple()

    crs_val = cfg.get("crs")
    if not crs_val:
        raise ValueError(f"{proj_yaml} missing required 'crs' for land-cover masking")
    project_crs = CRS.from_user_input(crs_val)

    if not enabled:
        return LandcoverMaskConfig(enabled=False, path=None, classes=tuple(), project_crs=project_crs)

    if not classes:
        raise ValueError(f"Land-cover mask enabled in {proj_yaml} but 'classes_to_exclude' is empty")

    domain = cfg.get("domain")
    resolution = cfg.get("resolution")
    if domain is None:
        raise ValueError(f"{proj_yaml} missing 'domain' required for land-cover filename derivation")
    if resolution is None:
        raise ValueError(f"{proj_yaml} missing 'resolution' required for land-cover filename derivation")

    lc_path = _derive_landcover_path(Path(project_dir), str(domain), resolution)
    if not lc_path.is_file():
        raise FileNotFoundError(f"Land-cover mask enabled but file not found: {lc_path}")

    return LandcoverMaskConfig(
        enabled=True,
        path=lc_path,
        classes=classes,
        project_crs=project_crs,
    )


@lru_cache(maxsize=16)
def _cached_exclusion_mask(
    lc_path: str,
    classes_key: tuple[int, ...],
    target_crs_wkt: str,
    transform_tuple: tuple[float, ...],
    shape: tuple[int, int],
    project_crs_wkt: str,
    lc_nodata: float | int | None,
) -> np.ndarray:
    """Return boolean mask of excluded pixels on target grid."""
    target_crs = CRS.from_wkt(target_crs_wkt)
    project_crs = CRS.from_wkt(project_crs_wkt)
    transform = Affine(*transform_tuple)

    with rasterio.open(lc_path) as src:
        src_crs = src.crs
        if src_crs is None:
            src_crs = project_crs
        elif src_crs != project_crs:
            raise ValueError(f"Land-cover CRS {src_crs} differs from project CRS {project_crs}")
        nodata = src.nodata if src.nodata is not None else lc_nodata
        dst = np.full(shape, nodata if nodata is not None else np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            dst_nodata=nodata,
        )

    excluded = np.zeros(shape, dtype=bool)
    if nodata is not None:
        excluded |= dst == nodata
    excluded |= ~np.isfinite(dst)
    if classes_key:
        excluded |= np.isin(dst, classes_key)
    return excluded


def apply_landcover_mask(
    arr: np.ma.MaskedArray,
    *,
    transform: Affine,
    target_crs: CRS,
    roi_mask: np.ndarray,
    lc_cfg: LandcoverMaskConfig,
    warn_threshold: float = 50.0,
) -> tuple[np.ma.MaskedArray, float]:
    """Apply land-cover exclusions to a masked array and return (array, excluded_pct)."""
    if not lc_cfg.enabled:
        return arr, 0.0
    if lc_cfg.path is None:
        raise ValueError("Land-cover mask enabled but no path resolved")
    if target_crs is None:
        raise ValueError("Target CRS is required for land-cover masking")
    if roi_mask.shape != arr.shape:
        raise ValueError("ROI mask shape does not match raster shape for land-cover masking")

    excluded = _cached_exclusion_mask(
        str(lc_cfg.path),
        tuple(sorted(lc_cfg.classes)),
        target_crs.to_wkt(),
        tuple(transform),
        (arr.shape[0], arr.shape[1]),
        lc_cfg.project_crs.to_wkt(),
        None,
    )

    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels == 0:
        raise ValueError("ROI mask is empty after raster alignment")
    excluded_roi = int(np.count_nonzero(excluded & roi_mask))
    excluded_pct = (excluded_roi / roi_pixels) * 100.0

    if excluded_pct >= 100.0:
        raise ValueError(f"Land-cover mask excludes entire ROI ({excluded_pct:.1f}%)")
    if excluded_pct > warn_threshold:
        logger.warning("Land-cover mask excludes {:.1f}% of the ROI", excluded_pct)

    combined_mask = np.ma.getmaskarray(arr) | excluded
    masked = np.ma.array(np.ma.getdata(arr), mask=combined_mask, copy=False)
    return masked, excluded_pct


__all__ = [
    "LandcoverMaskConfig",
    "resolve_landcover_mask",
    "apply_landcover_mask",
]
