"""Land-cover based masking for DA grids and observations.

This module replaces the former glacier mask logic. It resolves the land-cover
file from setup YAML, reads DA mask settings from project YAML, reprojects the
mask to a target grid, and applies the requested exclusions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
import rasterio
from affine import Affine
from loguru import logger
from pyproj import CRS
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, reproject

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import abspath_relative_to, find_setup_yaml, find_project_yaml
from openamundsen_da.util.roi import read_single_roi

LC_MASK_BLOCK = "landcover_mask"
LC_MASK_ENABLED = "enabled"
LC_MASK_CLASSES = "classes_to_exclude"
_LC_CLASS_NAMES = {
    1: "rock",
    2: "ice",
    3: "water",
    4: "grassland",
    5: "shrubland",
    6: "farmland",
    7: "transitional",
    8: "deciduous 30-60",
    9: "deciduous 60-100",
    10: "mixed forest",
    11: "coniferous 30-60",
    12: "coniferous 60-100",
    13: "built-up",
}


@dataclass(frozen=True)
class LandcoverMaskConfig:
    enabled: bool
    path: Path | None
    classes: tuple[int, ...]
    project_crs: CRS


@dataclass(frozen=True)
class LandcoverMaskClassSummary:
    code: str
    name: str
    cells: int
    area_km2: float
    percent_of_roi: float | None


@dataclass(frozen=True)
class LandcoverMaskSummary:
    roi_area_km2: float | None
    total_roi_cells: int
    pixel_area_m2: float
    masked_cells: int
    masked_area_km2: float
    classes: tuple[LandcoverMaskClassSummary, ...]


def serialize_landcover_mask_config(lc_cfg: LandcoverMaskConfig) -> dict[str, object]:
    """Return a dict-safe representation of LandcoverMaskConfig."""
    return {
        "enabled": lc_cfg.enabled,
        "path": str(lc_cfg.path) if lc_cfg.path else None,
        "classes": tuple(lc_cfg.classes),
        "project_crs_wkt": lc_cfg.project_crs.to_wkt(),
    }


def deserialize_landcover_mask_config(data: Any) -> LandcoverMaskConfig | None:
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


def _format_resolution(resolution: object) -> str:
    if isinstance(resolution, (int, float)):
        if float(resolution).is_integer():
            return str(int(resolution))
    return str(resolution).strip()


def _derive_landcover_path(grids_dir: Path, domain: str, resolution: object) -> Path:
    """Find land-cover file matching lc_<domain>_<resolution>*.asc under grids/.

    Prefers an exact name lc_<domain>_<resolution>.asc. If not present, falls
    back to a single unique match with a suffix (e.g., lc_domain_res_large.asc).
    Raises when zero or multiple matches are found to avoid ambiguous selection.
    """
    grids_dir = Path(grids_dir)
    # Accept either the grids directory itself or a parent path that contains
    # the conventional "grids/" subdirectory to preserve the older helper API.
    if grids_dir.name != "grids" and (grids_dir / "grids").is_dir():
        grids_dir = grids_dir / "grids"

    base = f"lc_{domain}_{_format_resolution(resolution)}"
    exact = grids_dir / f"{base}.asc"
    if exact.is_file():
        return exact

    candidates = sorted(grids_dir.glob(f"{base}_*.asc"))
    if not candidates:
        raise FileNotFoundError(f"No land-cover file matching {base}*.asc under {grids_dir}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise FileExistsError(
            f"Multiple land-cover files match {base}_*.asc under {grids_dir}: {names}. "
            "Keep exactly one matching file or rename to lc_<domain>_<resolution>.asc."
        )
    return candidates[0]


def resolve_setup_landcover_grid(setup_dir: Path) -> Path:
    """Resolve the setup land-cover grid using setup YAML grid-dir conventions."""
    setup_yaml = find_setup_yaml(setup_dir)
    setup_cfg = _read_yaml_file(setup_yaml) or {}
    domain = setup_cfg.get("domain")
    resolution = setup_cfg.get("resolution")
    if domain is None:
        raise ValueError(f"{setup_yaml} missing 'domain' required for land-cover filename derivation")
    if resolution is None:
        raise ValueError(f"{setup_yaml} missing 'resolution' required for land-cover filename derivation")

    grids_rel = (((setup_cfg.get("input_data") or {}).get("grids") or {}).get("dir")) or "grids"
    grids_dir = Path(abspath_relative_to(setup_dir, Path(grids_rel)))
    if not grids_dir.is_dir():
        raise FileNotFoundError(f"Grids directory not found: {grids_dir}")
    return _derive_landcover_path(grids_dir, str(domain), resolution)


def resolve_landcover_mask(setup_dir: Path, project_dir: Path) -> LandcoverMaskConfig:
    """Return land-cover mask configuration from setup YAML + project YAML."""
    setup_yaml = find_setup_yaml(setup_dir)
    project_yaml = find_project_yaml(project_dir)
    setup_cfg = _read_yaml_file(setup_yaml) or {}
    project_cfg = _read_yaml_file(project_yaml) or {}
    da_cfg = project_cfg.get("data_assimilation") or {}
    lc_cfg = da_cfg.get(LC_MASK_BLOCK) or {}
    enabled = bool(lc_cfg.get(LC_MASK_ENABLED, False))
    classes_raw = lc_cfg.get(LC_MASK_CLASSES) or []
    classes: tuple[int, ...] = tuple(int(c) for c in classes_raw) if enabled else tuple()

    crs_val = setup_cfg.get("crs")
    if not crs_val:
        raise ValueError(f"{setup_yaml} missing required 'crs' for land-cover masking")
    project_crs = CRS.from_user_input(crs_val)

    if not enabled:
        return LandcoverMaskConfig(enabled=False, path=None, classes=tuple(), project_crs=project_crs)

    if not classes:
        raise ValueError(f"Land-cover mask enabled in {project_yaml} but 'classes_to_exclude' is empty")

    lc_path = resolve_setup_landcover_grid(Path(setup_dir))
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


def summarize_landcover_mask(roi_path: Path, lc_cfg: LandcoverMaskConfig) -> LandcoverMaskSummary:
    """Summarize excluded land-cover classes within the ROI.

    Returns counts, areas, and percentages for each masked class. Raises on
    missing CRS or mask data; otherwise attempts best-effort reporting.
    """
    if not lc_cfg.enabled:
        raise ValueError("Land-cover mask is disabled; no summary available")
    if lc_cfg.path is None:
        raise ValueError("Land-cover mask is enabled but no mask path is configured")

    roi_gdf, _ = read_single_roi(Path(roi_path), required_field=None, to_crs=None)

    with rasterio.open(lc_cfg.path) as src:
        src_crs = src.crs if src.crs is not None else lc_cfg.project_crs
        if roi_gdf.crs is None:
            raise ValueError("ROI has no CRS; unable to align with land-cover mask")
        roi_aligned = roi_gdf.to_crs(src_crs)
        shapes = [roi_aligned.geometry.iloc[0]]

        data, transform = rio_mask(src, shapes, crop=True, filled=False)
        arr = np.ma.array(data[0])
        valid_mask = ~np.ma.getmaskarray(arr)
        pixel_area_m2 = abs(transform.a * transform.e)
        roi_cells = int(valid_mask.sum())

        roi_area_m2 = float(roi_aligned.geometry.area.iloc[0]) if roi_aligned.geometry.iloc[0] is not None else None

    values = np.ma.compressed(arr)
    class_summaries: list[LandcoverMaskClassSummary] = []
    masked_cells = 0
    for cls in lc_cfg.classes:
        count = int(np.count_nonzero(values == cls))
        masked_cells += count
        area_km2 = (count * pixel_area_m2) / 1_000_000.0
        pct = None
        if roi_area_m2 and roi_area_m2 > 0:
            pct = (count * pixel_area_m2 / roi_area_m2) * 100.0
        code = str(int(cls))
        name = _LC_CLASS_NAMES.get(int(cls), f"class {code}")
        class_summaries.append(
            LandcoverMaskClassSummary(
                code=code,
                name=name,
                cells=count,
                area_km2=area_km2,
                percent_of_roi=pct,
            )
        )

    # Combined forest row (classes 8-12) if any are part of the mask
    forest_classes = {8, 9, 10, 11, 12}
    forest_masked = forest_classes.intersection(set(int(c) for c in lc_cfg.classes))
    if forest_masked:
        forest_count = sum(cs.cells for cs in class_summaries if int(cs.code) in forest_classes)
        forest_area_km2 = (forest_count * pixel_area_m2) / 1_000_000.0
        forest_pct = None
        if roi_area_m2 and roi_area_m2 > 0:
            forest_pct = (forest_count * pixel_area_m2 / roi_area_m2) * 100.0
        class_summaries.append(
            LandcoverMaskClassSummary(
                code="8,9,10,11,12",
                name="forest",
                cells=forest_count,
                area_km2=forest_area_km2,
                percent_of_roi=forest_pct,
            )
        )

    masked_area_km2 = (masked_cells * pixel_area_m2) / 1_000_000.0
    roi_area_km2 = roi_area_m2 / 1_000_000.0 if roi_area_m2 is not None else None

    # Total row across all masked classes
    total_pct = None
    if roi_area_m2 and roi_area_m2 > 0:
        total_pct = (masked_area_km2 * 1_000_000.0 / roi_area_m2) * 100.0
    class_summaries.append(
        LandcoverMaskClassSummary(
            code="total",
            name="total",
            cells=masked_cells,
            area_km2=masked_area_km2,
            percent_of_roi=total_pct,
        )
    )

    return LandcoverMaskSummary(
        roi_area_km2=roi_area_km2,
        total_roi_cells=roi_cells,
        pixel_area_m2=pixel_area_m2,
        masked_cells=masked_cells,
        masked_area_km2=masked_area_km2,
        classes=tuple(class_summaries),
    )


def write_landcover_mask_report(summary: LandcoverMaskSummary, output_path: Path) -> None:
    """Write a per-class mask summary to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_code", "class_name", "cells", "area_km2", "percent_of_roi"])
        for cls in summary.classes:
            pct = "" if cls.percent_of_roi is None else f"{cls.percent_of_roi:.4f}"
            writer.writerow([cls.code, cls.name, cls.cells, f"{cls.area_km2:.6f}", pct])


__all__ = [
    "LandcoverMaskConfig",
    "serialize_landcover_mask_config",
    "deserialize_landcover_mask_config",
    "resolve_landcover_mask",
    "apply_landcover_mask",
    "summarize_landcover_mask",
    "write_landcover_mask_report",
]
