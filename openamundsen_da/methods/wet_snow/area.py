"""Wet-snow area fractions for model and observation rasters.

This module mirrors the SCF operator structure (openamundsen_da.methods.h_of_x)
but works on categorical wet-snow masks:

* Model side: use :func:`compute_model_wet_snow_fraction` on the binary masks
  produced by ``wet_snow.classify`` (1 = wet, 0 = dry, 255 = nodata).
* Observation side: :func:`compute_wet_snow_fraction_from_raster` can ingest
  arbitrary categorical rasters such as the Sentinel-1 WSM product where
  110 = wet, 125 = dry, 200 = radar shadow, 210 = water.

Both paths clip the raster to an AOI polygon (multiple features are unioned) and report the fraction of
wet pixels among all valid pixels (area-weighted under the equal-area pixel
assumption). ``point_wet_snow_roi.csv`` mirrors ``point_scf_roi.csv`` and can
be generated per member or per setup via the provided CLIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import re
import numpy as np
import pandas as pd
from pyproj import CRS
import rasterio
from loguru import logger
from rasterio import features
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, reproject

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    find_project_yaml,
    member_id_from_results_dir,
    list_step_dirs,
    infer_project_dir,
    infer_setup_dir_from_project,
)
from openamundsen_da.methods.daily_aoi_series import (
    compute_step_daily_series_for_all_members,
    step_start_end,
)
from openamundsen_da.util.config_validators import require_mapping, require_nonempty_str
from openamundsen_da.util.landcover_mask import (
    LandcoverMaskConfig,
    apply_landcover_mask,
    deserialize_landcover_mask_config,
    resolve_landcover_mask,
    serialize_landcover_mask_config,
)
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector, load_setup_roi_mask
from openamundsen_da.observer.class_config import load_wetsnow_classes
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.project_dates import resolve_project_dates
from openamundsen_da.util.uncertainty_common import (
    assert_same_grid as assert_same_grid_shared,
    normalize_netcdf_times as normalize_netcdf_times_shared,
)
from openamundsen_da.methods.wet_snow.wsl import compute_wet_snow_line_from_masks


_MODEL_WET = (1,)
_MODEL_VALID = (0, 1)


@dataclass(frozen=True)
class WetSnowStats:
    """Summary of wet-snow coverage inside an AOI."""

    wet_fraction: float
    wet_pixels: int
    valid_pixels: int
    wet_area_m2: float | None
    valid_area_m2: float | None
    region_id: str


@dataclass(frozen=True)
class WetSnowUncertaintyIngestConfig:
    enabled: bool
    wet_snow_variable: str | None
    uncertainty_variable: str | None
    time_variable: str | None


def _load_wet_snow_uncertainty_ingest_config(project_dir: Path) -> WetSnowUncertaintyIngestConfig:
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    da_cfg = require_mapping(cfg.get("data_assimilation"), path="project.data_assimilation")
    unc_root = da_cfg.get("uncertainty")
    if unc_root is None:
        return WetSnowUncertaintyIngestConfig(
            enabled=False,
            wet_snow_variable=None,
            uncertainty_variable=None,
            time_variable=None,
        )
    unc_cfg = require_mapping(unc_root, path="project.data_assimilation.uncertainty")
    wet_unc_raw = unc_cfg.get("wet_snow")
    if wet_unc_raw is None:
        return WetSnowUncertaintyIngestConfig(
            enabled=False,
            wet_snow_variable=None,
            uncertainty_variable=None,
            time_variable=None,
        )
    wet_unc = require_mapping(wet_unc_raw, path="project.data_assimilation.uncertainty.wet_snow")
    enabled = bool(wet_unc.get("enabled", False))
    if not enabled:
        return WetSnowUncertaintyIngestConfig(
            enabled=False,
            wet_snow_variable=None,
            uncertainty_variable=None,
            time_variable=None,
        )

    ingest = require_mapping(
        wet_unc.get("ingest"),
        path="project.data_assimilation.uncertainty.wet_snow.ingest",
    )
    ingest_path = "project.data_assimilation.uncertainty.wet_snow.ingest"
    wet_snow_variable = require_nonempty_str(ingest, "wet_snow_variable", path=ingest_path)
    uncertainty_variable = require_nonempty_str(ingest, "uncertainty_variable", path=ingest_path)
    time_variable = require_nonempty_str(ingest, "time_variable", path=ingest_path)
    return WetSnowUncertaintyIngestConfig(
        enabled=True,
        wet_snow_variable=wet_snow_variable,
        uncertainty_variable=uncertainty_variable,
        time_variable=time_variable,
    )


def _disabled_wet_snow_uncertainty_ingest_config() -> WetSnowUncertaintyIngestConfig:
    return WetSnowUncertaintyIngestConfig(
        enabled=False,
        wet_snow_variable=None,
        uncertainty_variable=None,
        time_variable=None,
    )


def _assert_same_grid(src: rasterio.DatasetReader, other: rasterio.DatasetReader, *, left: Path, right: Path) -> None:
    assert_same_grid_shared(src, other, left=left, right=right)


def _normalize_netcdf_times(time_values: object, *, source_name: str) -> pd.DatetimeIndex:
    return normalize_netcdf_times_shared(time_values, source_name=source_name)


def _missing_uncertainty_companion_error(raster_path: Path) -> FileNotFoundError:
    unc_path = raster_path.parent / f"{raster_path.stem}_uncertainty.tif"
    return FileNotFoundError(
        "Missing required uncertainty companion raster: "
        f"{unc_path}. Generate it first with "
        "'python -m openamundsen_da.observer.wetsnow_uncertainty --setup-dir <SETUP_DIR> "
        "--project-label <PROJECT_LABEL> --overwrite' or provide uncertainty in NetCDF."
    )


def _find_mask_raster(
    results_dir: Path,
    date: datetime,
    *,
    subdir: str = "wet_snow",
    prefix: str = "wet_snow_mask",
) -> Path:
    """Find wet-snow mask inside a member results directory for a date."""

    date_str = date.strftime("%Y-%m-%d")
    pattern = f"{prefix}_{date_str}T*.tif"
    base = Path(results_dir) / subdir
    matches = sorted(base.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No mask matching {pattern} in {base}")
    return matches[0]


def _read_mask_by_aoi(
    raster_path: Path | str,
    aoi_path: Path,
    *,
    lc_cfg: LandcoverMaskConfig,
    band_index: int = 1,
) -> tuple[np.ma.MaskedArray, np.ndarray, rasterio.Affine, float | None, str, object, float | int | None]:
    """Read raster values cropped to the AOI; return masked array, ROI mask, and metadata."""

    with rasterio.open(str(raster_path)) as src:
        src_crs = src.crs
        src_nodata = src.nodata
        if src_crs is None:
            raise ValueError(f"Raster {raster_path} lacks a CRS")
        gdf, region_id = read_single_roi(
            aoi_path,
            required_field=None,
            to_crs=src_crs,
        )
        data, transform = rio_mask(
            src,
            gdf.geometry,
            crop=True,
            nodata=src.nodata,
            filled=False,
            indexes=band_index,
        )
        roi_mask = features.geometry_mask(
            gdf.geometry,
            out_shape=(data.shape if data.ndim == 2 else data.shape[1:]),
            transform=transform,
            invert=True,
        )
    if data.ndim == 2:
        band = data
    elif data.ndim == 3:
        band = data[0]
    else:
        raise ValueError(f"Unexpected masked raster dimensions: {data.ndim}")
    arr = np.ma.array(band, copy=False)
    pixel_area = None
    if transform is not None:
        try:
            pixel_area = abs(float(transform.a) * float(transform.e))
        except AttributeError:
            pass
    arr, _ = apply_landcover_mask(
        arr,
        transform=transform,
        target_crs=src_crs,
        roi_mask=roi_mask,
        lc_cfg=lc_cfg,
    )
    return arr, roi_mask, transform, pixel_area, region_id, src_crs, src_nodata


def _read_mask_full_grid(
    raster_path: Path | str,
    aoi_path: Path,
    *,
    lc_cfg: LandcoverMaskConfig | None,
    band_index: int = 1,
) -> tuple[np.ma.MaskedArray, np.ndarray, rasterio.Affine, float | None, str, object, float | int | None]:
    """Read raster values on the full model grid and mask outside the ROI."""

    with rasterio.open(str(raster_path)) as src:
        src_crs = src.crs
        src_nodata = src.nodata
        if src_crs is None:
            raise ValueError(f"Raster {raster_path} lacks a CRS")
        gdf, region_id = read_single_roi(
            aoi_path,
            required_field=None,
            to_crs=src_crs,
        )
        roi_mask = features.geometry_mask(
            gdf.geometry,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
        )
        data = src.read(band_index, masked=False)
        arr = np.ma.array(data, mask=~roi_mask, copy=False)
        if src_nodata is not None:
            arr.mask = np.ma.getmaskarray(arr) | (data == src_nodata)
        pixel_area = None
        try:
            pixel_area = abs(float(src.transform.a) * float(src.transform.e))
        except AttributeError:
            pass
        if lc_cfg is not None and lc_cfg.enabled:
            arr, _ = apply_landcover_mask(
                arr,
                transform=src.transform,
                target_crs=src_crs,
                roi_mask=roi_mask,
                lc_cfg=lc_cfg,
            )
        return arr, roi_mask, src.transform, pixel_area, region_id, src_crs, src_nodata


def _compute_valid_and_wet_masks(
    arr: np.ma.MaskedArray,
    *,
    wet_values: Sequence[int],
    valid_values: Sequence[int] | None = None,
    exclude_values: Sequence[int] | None = None,
    support_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.ma.getdata(arr)
    mask = np.ma.getmaskarray(arr)
    valid = (~mask) & np.isfinite(data)
    if support_mask is not None:
        support = np.asarray(support_mask, dtype=bool)
        if support.shape != arr.shape:
            raise ValueError(f"Support mask shape {support.shape} does not match raster shape {arr.shape}")
        valid &= support
    if valid_values:
        valid &= np.isin(data, valid_values)
    if exclude_values:
        valid &= ~np.isin(data, exclude_values)
    wet = valid & np.isin(data, wet_values)
    return valid, wet


def _compute_fraction(
    arr: np.ma.MaskedArray,
    wet_values: Sequence[int],
    valid_values: Sequence[int] | None = None,
    exclude_values: Sequence[int] | None = None,
    pixel_area: float | None = None,
    region_id: str = "",
    support_mask: np.ndarray | None = None,
) -> WetSnowStats:
    """Return wet/valid counts and their ratio for the provided array."""

    valid, wet = _compute_valid_and_wet_masks(
        arr,
        wet_values=wet_values,
        valid_values=valid_values,
        exclude_values=exclude_values,
        support_mask=support_mask,
    )
    valid_pixels = int(valid.sum())
    if valid_pixels == 0:
        raise ValueError("AOI contains no valid wet-snow classification pixels")

    wet_pixels = int(wet.sum())
    wet_fraction = wet_pixels / valid_pixels

    if pixel_area and pixel_area > 0:
        valid_area = valid_pixels * pixel_area
        wet_area = wet_pixels * pixel_area
    else:
        valid_area = None
        wet_area = None

    return WetSnowStats(
        wet_fraction=float(wet_fraction),
        wet_pixels=wet_pixels,
        valid_pixels=valid_pixels,
        wet_area_m2=wet_area,
        valid_area_m2=valid_area,
        region_id=region_id,
    )


def _eligible_setup_mask(
    *,
    setup_dir: Path,
    landcover_cfg: LandcoverMaskConfig | None,
) -> tuple[np.ndarray, object]:
    roi_mask, spec, _ = load_setup_roi_mask(setup_dir, ensure_grid=True)
    arr = np.ma.array(np.ones(roi_mask.shape, dtype=float), mask=~roi_mask, copy=False)
    if landcover_cfg is not None and landcover_cfg.enabled:
        target_crs = CRS.from_user_input(spec.crs) if spec.crs is not None else None
        if target_crs is None:
            raise ValueError(f"Setup grid CRS is missing for {setup_dir}")
        arr, _ = apply_landcover_mask(
            arr,
            transform=spec.transform,
            target_crs=target_crs,
            roi_mask=roi_mask,
            lc_cfg=landcover_cfg,
        )
    eligible = (~np.ma.getmaskarray(arr)) & np.isfinite(np.ma.getdata(arr))
    return eligible, spec


def _project_observation_masks_to_setup_grid(
    *,
    setup_dir: Path,
    project_dir: Path,
    source_path: Path | str,
    band_index: int,
    landcover_cfg: LandcoverMaskConfig | None,
    wet_values: Sequence[int],
    valid_values: Sequence[int] | None,
    exclude_values: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    eligible_mask, spec = _eligible_setup_mask(
        setup_dir=setup_dir,
        landcover_cfg=landcover_cfg,
    )
    with rasterio.open(str(source_path)) as src:
        data = src.read(band_index).astype(float)
        mask = ~np.isfinite(data)
        if src.nodata is not None:
            mask |= data == src.nodata
        arr = np.ma.array(data, mask=mask, copy=False)
        valid_src, wet_src = _compute_valid_and_wet_masks(
            arr,
            wet_values=wet_values,
            valid_values=valid_values,
            exclude_values=exclude_values,
        )
        dst_valid = np.zeros(eligible_mask.shape, dtype=np.uint8)
        dst_wet = np.zeros(eligible_mask.shape, dtype=np.uint8)
        reproject(
            source=valid_src.astype(np.uint8),
            destination=dst_valid,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=spec.transform,
            dst_crs=spec.crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
        reproject(
            source=wet_src.astype(np.uint8),
            destination=dst_wet,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=spec.transform,
            dst_crs=spec.crs,
            resampling=Resampling.nearest,
            src_nodata=0,
            dst_nodata=0,
        )
    valid_mask = dst_valid.astype(bool) & eligible_mask
    wet_mask = dst_wet.astype(bool) & valid_mask
    return valid_mask, wet_mask


def compute_wet_snow_fraction_from_raster(
    raster_path: Path,
    aoi_path: Path,
    *,
    wet_values: Sequence[int],
    valid_values: Sequence[int] | None = None,
    exclude_values: Sequence[int] | None = None,
    landcover_cfg: LandcoverMaskConfig,
) -> WetSnowStats:
    """Compute wet-snow coverage from an arbitrary categorical raster."""

    arr, _, _, pixel_area, region_id, _, _ = _read_mask_by_aoi(
        Path(raster_path),
        Path(aoi_path),
        lc_cfg=landcover_cfg,
    )
    return _compute_fraction(
        arr,
        wet_values=wet_values,
        valid_values=valid_values,
        exclude_values=exclude_values,
        pixel_area=pixel_area,
        region_id=region_id,
    )


def _build_wetsnow_summary_row(
    *,
    date_key: str,
    region_id: str,
    tile: str,
    source_name: str,
    arr: np.ma.MaskedArray,
    wet_values: Sequence[int],
    valid_values: Sequence[int] | None = None,
    exclude_values: Sequence[int] | None = None,
    unc_arr: np.ma.MaskedArray | None = None,
    unc_nodata: float | int | None = None,
    require_uncertainty: bool = False,
) -> dict[str, object] | None:
    valid, wet = _compute_valid_and_wet_masks(
        arr,
        wet_values=wet_values,
        valid_values=valid_values,
        exclude_values=exclude_values,
    )
    n_valid = int(np.count_nonzero(valid))
    if n_valid <= 0:
        return None
    n_wet = int(np.count_nonzero(wet))
    frac = float(n_wet / n_valid)
    row: dict[str, object] = {
        "date": date_key,
        "region_id": region_id,
        "tile": tile,
        "wet_snow_fraction": round(frac, 4),
        "n_valid": n_valid,
        "n_wet": n_wet,
        "source": source_name,
    }
    if require_uncertainty:
        if unc_arr is None:
            raise ValueError(f"Missing uncertainty values for source {source_name}")
        unc_data = np.ma.getdata(unc_arr)
        unc_mask = np.ma.getmaskarray(unc_arr)
        if unc_data.shape != valid.shape:
            raise ValueError(f"Uncertainty shape mismatch for source {source_name}")
        unc_roi = (~unc_mask) & np.isfinite(unc_data)
        if unc_nodata is not None:
            unc_roi &= unc_data != unc_nodata
        if np.any(unc_roi):
            unc_vals_roi = unc_data[unc_roi]
            if np.any(unc_vals_roi < 0.0) or np.any(unc_vals_roi > 100.0):
                raise ValueError(f"Uncertainty values out of [0,100] range in {source_name}")
        unc_valid = valid & unc_roi
        unc_n_valid = int(np.count_nonzero(unc_valid))
        if unc_n_valid <= 0:
            raise ValueError(f"No valid uncertainty support for source {source_name}")
        unc_vals = unc_data[unc_valid].astype(float)
        row["unc_mean"] = float(np.mean(unc_vals))
        row["unc_min"] = float(np.min(unc_vals))
        row["unc_max"] = float(np.max(unc_vals))
        row["unc_n_valid"] = unc_n_valid
    return row


def compute_model_wet_snow_fraction(
    *,
    setup_dir: Path,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    date: datetime,
    mask_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
    support_mask: np.ndarray | None = None,
) -> dict:
    """Compute AOI wet-snow fraction for one member/date."""

    if landcover_cfg is not None:
        lc_cfg = landcover_cfg
    else:
        lc_cfg = resolve_landcover_mask(Path(setup_dir), Path(project_dir))
    raster = _find_mask_raster(Path(results_dir), date, subdir=mask_subdir, prefix=mask_prefix)
    arr, _roi_mask, _transform, pixel_area, region_id, _src_crs, _src_nodata = _read_mask_full_grid(
        Path(raster),
        Path(aoi_path),
        lc_cfg=lc_cfg,
    )
    stats_full = _compute_fraction(
        arr,
        wet_values=_MODEL_WET,
        valid_values=_MODEL_VALID,
        pixel_area=pixel_area,
        region_id=region_id,
    )
    stats = _compute_fraction(
        arr,
        wet_values=_MODEL_WET,
        valid_values=_MODEL_VALID,
        pixel_area=pixel_area,
        region_id=region_id,
        support_mask=support_mask,
    )
    member_id = member_id_from_results_dir(Path(results_dir))
    return {
        "date": date.strftime("%Y-%m-%d"),
        "member_id": member_id,
        "region_id": stats.region_id,
        "wet_fraction": stats.wet_fraction,
        "n_valid": stats.valid_pixels,
        "n_wet": stats.wet_pixels,
        "valid_area_m2": stats.valid_area_m2,
        "wet_area_m2": stats.wet_area_m2,
        "wet_fraction_full_roi": stats_full.wet_fraction,
        "n_valid_full_roi": stats_full.valid_pixels,
        "n_wet_full_roi": stats_full.wet_pixels,
        "raster": Path(raster).name,
    }


def compute_model_wet_snow_line(
    *,
    setup_dir: Path,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    date: datetime,
    mask_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
    support_mask: np.ndarray | None = None,
) -> dict:
    """Compute basin-wide wet-snow line diagnostics for one member/date."""

    if landcover_cfg is not None:
        lc_cfg = landcover_cfg
    else:
        lc_cfg = resolve_landcover_mask(Path(setup_dir), Path(project_dir))
    raster = _find_mask_raster(Path(results_dir), date, subdir=mask_subdir, prefix=mask_prefix)
    arr_full, _roi_mask, _transform, _pixel_area, region_id, _src_crs, _src_nodata = _read_mask_full_grid(
        Path(raster),
        Path(aoi_path),
        lc_cfg=None,
    )
    arr_support, _roi_mask, _transform, _pixel_area, region_id, _src_crs, _src_nodata = _read_mask_full_grid(
        Path(raster),
        Path(aoi_path),
        lc_cfg=lc_cfg,
    )
    valid_full, wet_full = _compute_valid_and_wet_masks(
        arr_full,
        wet_values=_MODEL_WET,
        valid_values=_MODEL_VALID,
    )
    support = np.asarray(support_mask, dtype=bool) if support_mask is not None else valid_full.copy()
    valid_support, wet_support = _compute_valid_and_wet_masks(
        arr_support,
        wet_values=_MODEL_WET,
        valid_values=_MODEL_VALID,
        support_mask=support,
    )
    full_eval = compute_wet_snow_line_from_masks(
        setup_dir=Path(setup_dir),
        project_dir=Path(project_dir),
        valid_mask=valid_full,
        wet_mask=wet_full,
    )
    support_eval = compute_wet_snow_line_from_masks(
        setup_dir=Path(setup_dir),
        project_dir=Path(project_dir),
        valid_mask=valid_support,
        wet_mask=wet_support,
    )
    member_id = member_id_from_results_dir(Path(results_dir))
    out = {
        "date": date.strftime("%Y-%m-%d"),
        "member_id": member_id,
        "region_id": region_id,
        "wet_snow_line": support_eval.wet_snow_line,
        "wet_snow_line_full_roi": full_eval.wet_snow_line,
        "wet_snow_line_p95": support_eval.wet_elevation_percentile,
        "wet_snow_line_p95_full_roi": full_eval.wet_elevation_percentile,
        "n_valid": support_eval.n_valid,
        "n_wet": support_eval.n_wet,
        "wet_bands": support_eval.wet_bands,
        "wet_snow_line_gate_reason": support_eval.gate_reason,
        "n_valid_full_roi": full_eval.n_valid,
        "n_wet_full_roi": full_eval.n_wet,
        "wet_bands_full_roi": full_eval.wet_bands,
        "wet_snow_line_gate_reason_full_roi": full_eval.gate_reason,
        "raster": Path(raster).name,
        "profile": support_eval.profile,
        "profile_full_roi": full_eval.profile,
        "sector_relative_profiles": support_eval.sector_relative_profiles,
        "sector_relative_profiles_full_roi": full_eval.sector_relative_profiles,
    }
    for sector, value in support_eval.sector_relative_lines.items():
        out[f"wet_snow_line_sector_rel_{sector.lower()}"] = value
    for sector, value in full_eval.sector_relative_lines.items():
        out[f"wet_snow_line_sector_rel_{sector.lower()}_full_roi"] = value
    return out


def compute_member_wet_snow_daily(
    *,
    setup_dir: Path,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    start: datetime,
    end: datetime,
    mask_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
) -> pd.DataFrame:
    """Return daily wet-snow fraction inside the AOI for a member."""

    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(setup_dir), Path(project_dir))
    start_day = datetime(start.year, start.month, start.day)
    end_day = datetime(end.year, end.month, end.day)
    if end_day < start_day:
        return pd.DataFrame(columns=["time", "wet_snow_fraction"])

    dates = pd.date_range(start_day, end_day, freq="D").to_pydatetime()
    rows: list[dict[str, object]] = []
    for dt in dates:
        try:
            stats = compute_model_wet_snow_fraction(
                setup_dir=Path(setup_dir),
                project_dir=Path(project_dir),
                results_dir=Path(results_dir),
                aoi_path=Path(aoi_path),
                landcover_cfg=lc_cfg,
                date=dt,
                mask_subdir=mask_subdir,
                mask_prefix=mask_prefix,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wet-snow fraction failed for {} {}: {}", results_dir, dt.date(), exc)
            continue
        rows.append({"time": dt, "wet_snow_fraction": float(stats["wet_fraction"])})

    if not rows:
        return pd.DataFrame(columns=["time", "wet_snow_fraction"])
    df = pd.DataFrame(rows)
    return df.sort_values("time")


def compute_member_wet_snow_line_daily(
    *,
    setup_dir: Path,
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    start: datetime,
    end: datetime,
    mask_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
) -> pd.DataFrame:
    """Return daily full-ROI wet-snow-line diagnostics for a member."""

    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(setup_dir), Path(project_dir))
    start_day = datetime(start.year, start.month, start.day)
    end_day = datetime(end.year, end.month, end.day)
    if end_day < start_day:
        return pd.DataFrame(
            columns=[
                "time",
                "wet_snow_line",
                "wet_snow_line_p95",
                "n_valid",
                "n_wet",
                "wet_bands",
                "wet_snow_line_gate_reason",
            ]
        )

    dates = pd.date_range(start_day, end_day, freq="D").to_pydatetime()
    rows: list[dict[str, object]] = []
    for dt in dates:
        try:
            wsl = compute_model_wet_snow_line(
                setup_dir=Path(setup_dir),
                project_dir=Path(project_dir),
                results_dir=Path(results_dir),
                aoi_path=Path(aoi_path),
                landcover_cfg=lc_cfg,
                date=dt,
                mask_subdir=mask_subdir,
                mask_prefix=mask_prefix,
                support_mask=None,
            )
        except FileNotFoundError:
            continue
        except Exception as exc:  # noqa: BLE001
            logger.warning("Wet-snow line daily computation failed for {} {}: {}", results_dir, dt.date(), exc)
            continue
        rows.append(
            {
                "time": dt,
                "wet_snow_line": wsl["wet_snow_line_full_roi"],
                "wet_snow_line_p95": wsl["wet_snow_line_p95_full_roi"],
                "n_valid": int(wsl["n_valid_full_roi"]),
                "n_wet": int(wsl["n_wet_full_roi"]),
                "wet_bands": int(wsl["wet_bands_full_roi"]),
                "wet_snow_line_gate_reason": wsl["wet_snow_line_gate_reason_full_roi"] or "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "time",
                "wet_snow_line",
                "wet_snow_line_p95",
                "n_valid",
                "n_wet",
                "wet_bands",
                "wet_snow_line_gate_reason",
            ]
        )
    df = pd.DataFrame(rows)
    return df.sort_values("time")


def _compute_member_daily_worker(
    results_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
    out_csv: Path,
    overwrite: bool,
    extra: Dict[str, Any],
) -> bool:
    """Worker: compute wet-snow daily series for a single member results dir."""
    mask_subdir = str(extra.get("mask_subdir", "wet_snow"))
    mask_prefix = str(extra.get("mask_prefix", "wet_snow_mask"))
    lc_cfg = deserialize_landcover_mask_config(extra.get("landcover_cfg"))
    setup_dir = Path(extra["setup_dir"])
    project_dir = Path(extra["project_dir"])
    df = compute_member_wet_snow_daily(
        setup_dir=setup_dir,
        project_dir=project_dir,
        results_dir=results_dir,
        aoi_path=aoi_path,
        landcover_cfg=lc_cfg,
        start=start,
        end=end,
        mask_subdir=mask_subdir,
        mask_prefix=mask_prefix,
    )
    if df.empty:
        return False
    if out_csv.exists() and not overwrite:
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return True


def _compute_member_wet_snow_line_daily_worker(
    results_dir: Path,
    aoi_path: Path,
    start: datetime,
    end: datetime,
    out_csv: Path,
    overwrite: bool,
    extra: Dict[str, Any],
) -> bool:
    """Worker: compute daily full-ROI wet-snow-line series for one member."""

    mask_subdir = str(extra.get("mask_subdir", "wet_snow"))
    mask_prefix = str(extra.get("mask_prefix", "wet_snow_mask"))
    lc_cfg = deserialize_landcover_mask_config(extra.get("landcover_cfg"))
    setup_dir = Path(extra["setup_dir"])
    project_dir = Path(extra["project_dir"])
    df = compute_member_wet_snow_line_daily(
        setup_dir=setup_dir,
        project_dir=project_dir,
        results_dir=results_dir,
        aoi_path=aoi_path,
        landcover_cfg=lc_cfg,
        start=start,
        end=end,
        mask_subdir=mask_subdir,
        mask_prefix=mask_prefix,
    )
    if df.empty:
        return False
    if out_csv.exists() and not overwrite:
        return False
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return True


def compute_step_wet_snow_daily_for_all_members(
    *,
    setup_dir: Path,
    project_dir: Path,
    step_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    max_workers: int = 4,
    overwrite: bool = False,
    mask_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
) -> None:
    """Compute daily wet-snow fractions for all prior members in a step."""

    step_dir = Path(step_dir)
    aoi_path = Path(aoi_path)
    resolved_project = infer_project_dir(step_dir)
    setup_dir = Path(setup_dir)
    project_dir = Path(project_dir)
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

    logger.info("Computing wet-snow daily fractions for {}", step_dir.name)

    start, end = step_start_end(step_dir)

    compute_step_daily_series_for_all_members(
        step_dir=step_dir,
        aoi_path=aoi_path,
        start=start,
        end=end,
        csv_name="point_wet_snow_roi.csv",
        worker=_compute_member_daily_worker,
        ensemble="prior",
        include_open_loop=True,
        max_workers=max_workers,
        overwrite=overwrite,
        worker_kwargs={
            "mask_subdir": mask_subdir,
            "mask_prefix": mask_prefix,
            "landcover_cfg": serialize_landcover_mask_config(lc_cfg),
            "setup_dir": str(setup_dir),
            "project_dir": str(project_dir),
        },
    )
    compute_step_daily_series_for_all_members(
        step_dir=step_dir,
        aoi_path=aoi_path,
        start=start,
        end=end,
        csv_name="point_wet_snow_line_roi.csv",
        worker=_compute_member_wet_snow_line_daily_worker,
        ensemble="prior",
        include_open_loop=True,
        max_workers=max_workers,
        overwrite=overwrite,
        worker_kwargs={
            "mask_subdir": mask_subdir,
            "mask_prefix": mask_prefix,
            "landcover_cfg": serialize_landcover_mask_config(lc_cfg),
            "setup_dir": str(setup_dir),
            "project_dir": str(project_dir),
        },
    )


def summarize_s1_directory(
    *,
    setup_dir: Path,
    project_dir: Path,
    raster_dir: Path,
    aoi_path: Path,
    output_csv: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    overwrite: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
    wet_values: Sequence[int] | None = None,
    valid_values: Sequence[int] | None = None,
    exclude_values: Sequence[int] | None = None,
    recursive: bool = False,
    wsl_diagnostics_csv: Path | None = None,
    wsl_profile_dir: Path | None = None,
) -> Path:
    """Summarize Sentinel-1 wet-snow maps into one CSV (date, fraction)."""

    if landcover_cfg is not None:
        lc_cfg = landcover_cfg
    else:
        lc_cfg = resolve_landcover_mask(Path(setup_dir), Path(project_dir))
    if wet_values is None or valid_values is None or exclude_values is None:
        raise ValueError(
            "wet_values, valid_values, and exclude_values must be provided explicitly from setup configuration"
        )
    if output_csv.exists() and not overwrite:
        return output_csv

    files = []
    for patt in ("*.tif", "*.tiff", "*.nc"):
        globber = Path(raster_dir).rglob if recursive else Path(raster_dir).glob
        files.extend(sorted(globber(patt)))
    files = [p for p in files if not p.stem.lower().endswith("_uncertainty")]
    uncertainty_cfg = _load_wet_snow_uncertainty_ingest_config(Path(project_dir))
    disabled_uncertainty_cfg = _disabled_wet_snow_uncertainty_ingest_config()
    rows: list[dict[str, object]] = []
    for tif in files:
        suffix = tif.suffix.lower()
        try:
            stats_rows: list[dict[str, object]] = []
            effective_unc_cfg = uncertainty_cfg if uncertainty_cfg.enabled else disabled_uncertainty_cfg
            if effective_unc_cfg.enabled and suffix == ".nc":
                if (
                    effective_unc_cfg.wet_snow_variable is None
                    or effective_unc_cfg.uncertainty_variable is None
                    or effective_unc_cfg.time_variable is None
                ):
                    raise ValueError("Missing required NetCDF ingest variable names for wet-snow uncertainty")
                try:
                    import xarray as xr  # lazy dependency
                except Exception as exc:  # pragma: no cover
                    raise RuntimeError("xarray is required to process NetCDF wet-snow uncertainty ingestion") from exc

                with xr.open_dataset(tif) as ds:
                    if effective_unc_cfg.wet_snow_variable not in ds:
                        raise ValueError(f"Variable '{effective_unc_cfg.wet_snow_variable}' not found in {tif.name}")
                    if effective_unc_cfg.uncertainty_variable not in ds:
                        raise ValueError(f"Variable '{effective_unc_cfg.uncertainty_variable}' not found in {tif.name}")
                    if effective_unc_cfg.time_variable not in ds:
                        raise ValueError(f"Time variable '{effective_unc_cfg.time_variable}' not found in {tif.name}")
                    times = _normalize_netcdf_times(ds[effective_unc_cfg.time_variable].values, source_name=tif.name)

                ws_uri = f"NETCDF:{tif}:{effective_unc_cfg.wet_snow_variable}"
                unc_uri = f"NETCDF:{tif}:{effective_unc_cfg.uncertainty_variable}"
                with rasterio.open(ws_uri) as src, rasterio.open(unc_uri) as src_unc:
                    _assert_same_grid(src, src_unc, left=tif, right=tif)
                    if src.count != len(times):
                        raise ValueError(
                            f"Band/time mismatch in {tif.name}: wet-snow bands={src.count} but time steps={len(times)}"
                        )
                    if src_unc.count != len(times):
                        raise ValueError(
                            f"Band/time mismatch in {tif.name}: uncertainty bands={src_unc.count} but time steps={len(times)}"
                        )
                for i, ts in enumerate(times, start=1):
                    arr, _, _, _, region_id, _, _ = _read_mask_by_aoi(
                        ws_uri,
                        aoi_path,
                        lc_cfg=lc_cfg,
                        band_index=i,
                    )
                    unc_arr, _, _, _, _, _, unc_nodata = _read_mask_by_aoi(
                        unc_uri,
                        aoi_path,
                        lc_cfg=lc_cfg,
                        band_index=i,
                    )
                    row = _build_wetsnow_summary_row(
                        date_key=ts.date().isoformat(),
                        region_id=region_id,
                        tile=_extract_tile(tif.name),
                        source_name=f"{tif.name}@{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}",
                        arr=arr,
                        wet_values=wet_values,
                        valid_values=valid_values,
                        exclude_values=exclude_values,
                        unc_arr=unc_arr,
                        unc_nodata=unc_nodata,
                        require_uncertainty=True,
                    )
                    if row is not None:
                        valid_mask, wet_mask = _project_observation_masks_to_setup_grid(
                            setup_dir=Path(setup_dir),
                            project_dir=Path(project_dir),
                            source_path=ws_uri,
                            band_index=i,
                            landcover_cfg=lc_cfg,
                            wet_values=wet_values,
                            valid_values=valid_values,
                            exclude_values=exclude_values,
                        )
                        row["__wsl_valid_mask"] = valid_mask
                        row["__wsl_wet_mask"] = wet_mask
                        stats_rows.append(row)
            else:
                if suffix not in {".tif", ".tiff", ".nc"}:
                    continue
                try:
                    date = _parse_s1_timestamp(tif.name)
                except ValueError:
                    continue
                if start and date < start:
                    continue
                if end and date > end:
                    continue

                arr, _, _, _, region_id, _, _ = _read_mask_by_aoi(
                    tif,
                    aoi_path,
                    lc_cfg=lc_cfg,
                    band_index=1,
                )
                unc_arr: np.ma.MaskedArray | None = None
                unc_nodata: float | int | None = None
                if effective_unc_cfg.enabled:
                    if suffix not in {".tif", ".tiff"}:
                        raise ValueError(
                            f"Uncertainty-enabled wet-snow ingestion expects either NetCDF with uncertainty "
                            f"or GeoTIFF with sidecar uncertainty raster, got {tif.name}"
                        )
                    unc_path = tif.parent / f"{tif.stem}_uncertainty.tif"
                    if not unc_path.is_file():
                        raise _missing_uncertainty_companion_error(tif)
                    with rasterio.open(tif) as src, rasterio.open(unc_path) as src_unc:
                        _assert_same_grid(src, src_unc, left=tif, right=unc_path)
                    unc_arr, _, _, _, _, _, unc_nodata = _read_mask_by_aoi(
                        unc_path,
                        aoi_path,
                        lc_cfg=lc_cfg,
                        band_index=1,
                    )
                row = _build_wetsnow_summary_row(
                    date_key=date.strftime("%Y-%m-%d"),
                    region_id=region_id,
                    tile=_extract_tile(tif.name),
                    source_name=tif.name,
                    arr=arr,
                    wet_values=wet_values,
                    valid_values=valid_values,
                    exclude_values=exclude_values,
                    unc_arr=unc_arr,
                    unc_nodata=unc_nodata,
                    require_uncertainty=bool(effective_unc_cfg.enabled),
                )
                if row is not None:
                    valid_mask, wet_mask = _project_observation_masks_to_setup_grid(
                        setup_dir=Path(setup_dir),
                        project_dir=Path(project_dir),
                        source_path=tif,
                        band_index=1,
                        landcover_cfg=lc_cfg,
                        wet_values=wet_values,
                        valid_values=valid_values,
                        exclude_values=exclude_values,
                    )
                    row["__wsl_valid_mask"] = valid_mask
                    row["__wsl_wet_mask"] = wet_mask
                    stats_rows.append(row)

            accepted = 0
            for row in stats_rows:
                stats_date = pd.to_datetime(str(row["date"]), errors="coerce")
                if pd.isna(stats_date):
                    if effective_unc_cfg.enabled:
                        raise ValueError(
                            f"Could not parse derived observation date '{row['date']}' for source {row.get('source', tif.name)}"
                        )
                    continue
                if start and stats_date.to_pydatetime() < start:
                    continue
                if end and stats_date.to_pydatetime() > end:
                    continue
                rows.append(row)
                accepted += 1
                logger.info(
                    "Wet-snow {} -> wet_fraction={:.3f} n_valid={} n_wet={}",
                    str(row.get("source", tif.name)),
                    float(row["wet_snow_fraction"]),
                    int(row["n_valid"]),
                    int(row["n_wet"]),
                )
            if accepted == 0 and stats_rows:
                logger.warning("Discarded {} because date filter removed all matching records", tif.name)
            elif accepted == 0:
                logger.warning("Discarded {} because AOI contained no valid pixels", tif.name)
        except Exception as exc:
            if uncertainty_cfg.enabled:
                raise RuntimeError(
                    f"Wet-snow preprocessing failed for {tif.name} with uncertainty enabled: {exc}"
                ) from exc
            logger.warning("Skipping {}: {}", tif.name, exc)
            continue

    if not rows:
        raise RuntimeError(f"No valid Sentinel-1 rasters processed in {raster_dir}")

    # Keep one best raster per date/tile (highest valid pixel count), then aggregate across tiles.
    best_per_date_tile: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["date"]), str(row.get("tile", "UNKNOWN")))
        prev = best_per_date_tile.get(key)
        if prev is None or int(row.get("n_valid", 0)) > int(prev.get("n_valid", 0)):
            best_per_date_tile[key] = row

    agg: dict[tuple[str, str], dict[str, object]] = {}
    for row in best_per_date_tile.values():
        key = (str(row["date"]), str(row["region_id"]))
        slot = agg.get(key)
        if slot is None:
            slot = {
                "date": row["date"],
                "region_id": row["region_id"],
                "n_valid": 0,
                "n_wet": 0,
                "source_set": set(),
                "tile_set": set(),
                "wsl_valid_mask": None,
                "wsl_wet_mask": None,
            }
            if uncertainty_cfg.enabled:
                slot["unc_mean_num"] = 0.0
                slot["unc_n_valid"] = 0
                slot["unc_min"] = float("inf")
                slot["unc_max"] = float("-inf")
            agg[key] = slot
        slot["n_valid"] = int(slot["n_valid"]) + int(row.get("n_valid", 0))
        slot["n_wet"] = int(slot["n_wet"]) + int(row.get("n_wet", 0))
        slot["source_set"].add(str(row.get("source", "")))
        slot["tile_set"].add(str(row.get("tile", "UNKNOWN")))
        row_valid_mask = row.get("__wsl_valid_mask")
        row_wet_mask = row.get("__wsl_wet_mask")
        if isinstance(row_valid_mask, np.ndarray) and isinstance(row_wet_mask, np.ndarray):
            if slot["wsl_valid_mask"] is None:
                slot["wsl_valid_mask"] = row_valid_mask.copy()
                slot["wsl_wet_mask"] = row_wet_mask.copy()
            else:
                slot["wsl_valid_mask"] = np.asarray(slot["wsl_valid_mask"], dtype=bool) | row_valid_mask
                slot["wsl_wet_mask"] = np.asarray(slot["wsl_wet_mask"], dtype=bool) | row_wet_mask
        if uncertainty_cfg.enabled:
            unc_n_valid = int(row.get("unc_n_valid", 0))
            if unc_n_valid > 0:
                unc_mean = float(row["unc_mean"])
                slot["unc_mean_num"] = float(slot["unc_mean_num"]) + (unc_mean * unc_n_valid)
                slot["unc_n_valid"] = int(slot["unc_n_valid"]) + unc_n_valid
                slot["unc_min"] = min(float(slot["unc_min"]), float(row["unc_min"]))
                slot["unc_max"] = max(float(slot["unc_max"]), float(row["unc_max"]))

    diagnostics_rows: list[dict[str, object]] = []
    out_rows: list[dict[str, object]] = []
    resolved_wsl_diagnostics_csv = (
        Path(wsl_diagnostics_csv)
        if wsl_diagnostics_csv is not None
        else output_csv.parent / "wet_snow_line_diagnostics.csv"
    )
    resolved_wsl_profile_dir = (
        Path(wsl_profile_dir)
        if wsl_profile_dir is not None
        else output_csv.parent / "wet_snow_line_profiles"
    )
    for entry in agg.values():
        n_valid = int(entry["n_valid"])
        n_wet = int(entry["n_wet"])
        frac = (n_wet / n_valid) if n_valid > 0 else 0.0
        valid_mask = np.asarray(entry["wsl_valid_mask"], dtype=bool)
        wet_mask = np.asarray(entry["wsl_wet_mask"], dtype=bool)
        wsl_eval = compute_wet_snow_line_from_masks(
            setup_dir=Path(setup_dir),
            project_dir=Path(project_dir),
            valid_mask=valid_mask,
            wet_mask=wet_mask,
        )
        row_out = {
            "date": entry["date"],
            "region_id": entry["region_id"],
            "wet_snow_fraction": round(frac, 4),
            "n_valid": n_valid,
            "n_wet": n_wet,
            "tiles_used": ";".join(sorted(x for x in entry["tile_set"] if x)),
            "source": ";".join(sorted(x for x in entry["source_set"] if x)),
            "wet_snow_line": wsl_eval.wet_snow_line,
            "wet_snow_line_p95": wsl_eval.wet_elevation_percentile,
            "wet_snow_line_n_valid": wsl_eval.n_valid,
            "wet_snow_line_n_wet": wsl_eval.n_wet,
            "wet_snow_line_wet_bands": wsl_eval.wet_bands,
            "wet_snow_line_support_coverage_ratio": float(n_valid / max(1, wsl_eval.n_valid)),
            "wet_snow_line_method": wsl_eval.method,
            "wet_snow_line_gate_reason": wsl_eval.gate_reason or "",
        }
        out_rows.append(row_out)
        diag_row = {
            "date": entry["date"],
            "region_id": entry["region_id"],
            "wet_snow_line": wsl_eval.wet_snow_line,
            "wet_snow_line_p95": wsl_eval.wet_elevation_percentile,
            "wet_snow_line_n_valid": wsl_eval.n_valid,
            "wet_snow_line_n_wet": wsl_eval.n_wet,
            "wet_snow_line_wet_bands": wsl_eval.wet_bands,
            "wet_snow_line_method": wsl_eval.method,
            "wet_snow_line_gate_reason": wsl_eval.gate_reason or "",
        }
        for sector, value in wsl_eval.sector_relative_lines.items():
            diag_row[f"wet_snow_line_sector_rel_{sector.lower()}"] = value
        diagnostics_rows.append(diag_row)
        profile_frames: list[pd.DataFrame] = []
        if not wsl_eval.profile.empty:
            profile_df = wsl_eval.profile.copy()
            profile_df.insert(0, "sector", "")
            profile_df.insert(0, "scope", "basin")
            profile_frames.append(profile_df)
        for sector, sector_profile in wsl_eval.sector_relative_profiles.items():
            if sector_profile.empty:
                continue
            working = sector_profile.copy()
            working.insert(0, "sector", sector)
            working.insert(0, "scope", "sector_rel")
            profile_frames.append(working)
        if profile_frames:
            profile_df = pd.concat(profile_frames, ignore_index=True)
            profile_df.insert(0, "date", entry["date"])
            profile_df.insert(1, "region_id", entry["region_id"])
            resolved_wsl_profile_dir.mkdir(parents=True, exist_ok=True)
            profile_df.to_csv(
                resolved_wsl_profile_dir / f"wet_snow_line_profile_{str(entry['date']).replace('-', '')}.csv",
                index=False,
            )
        if uncertainty_cfg.enabled:
            row_ref = out_rows[-1]
            row_unc_mean_num = float(entry.get("unc_mean_num", 0.0))
            row_unc_n_valid = int(entry.get("unc_n_valid", 0))
            if row_unc_n_valid <= 0:
                raise ValueError(
                    f"Uncertainty is enabled but no uncertainty-valid support exists for date {entry['date']}"
                )
            row_ref["unc_mean"] = row_unc_mean_num / row_unc_n_valid
            row_ref["unc_n_valid"] = row_unc_n_valid
            row_ref["unc_min"] = float(entry["unc_min"])
            row_ref["unc_max"] = float(entry["unc_max"])

    df = pd.DataFrame(out_rows).sort_values("date")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    if diagnostics_rows:
        pd.DataFrame(diagnostics_rows).sort_values("date").to_csv(resolved_wsl_diagnostics_csv, index=False)
    logger.info(
        "Sentinel-1 wet-snow summary written: {} ({} day(s), {} raster(s))",
        output_csv,
        len(df),
        len(best_per_date_tile),
    )
    return output_csv


def _extract_tile(name: str) -> str:
    m = re.search(r"T(\d{2}[A-Z]{3})", name.upper())
    return m.group(1) if m else "UNKNOWN"


def _parse_s1_timestamp(name: str) -> datetime:
    parts = name.split("_")
    # Legacy S1 pattern: *_YYYY_MM_DD_*
    for idx in range(len(parts) - 2):
        try:
            year, month, day = map(int, parts[idx : idx + 3])
            return datetime(year, month, day)
        except Exception:
            continue
    # Fallback: look for an 8-digit token YYYYMMDD anywhere in the name.
    for token in parts:
        if len(token) == 8 and token.isdigit():
            try:
                return datetime.strptime(token, "%Y%m%d")
            except Exception:
                continue
    # CLMS naming often embeds YYYYMMDD in longer tokens (e.g. 20250317T052706).
    for token in parts:
        m = re.search(r"(20\d{2})(\d{2})(\d{2})", token)
        if not m:
            continue
        y, mth, d = m.groups()
        try:
            return datetime.strptime(f"{y}{mth}{d}", "%Y%m%d")
        except Exception:
            continue
    raise ValueError(f"Cannot parse date from {name}")


def _load_obs_wetsnow_classes(project_dir: Path) -> tuple[list[int], list[int], list[int]]:
    # Kept as thin wrapper for backward-compatible imports in tests/callers.
    return load_wetsnow_classes(project_dir)


def cli_model(argv: list[str] | None = None) -> int:
    """CLI entry point mirroring oa-da-model-scf but for wet-snow area."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-wet-snow",
        description="Compute AOI wet-snow fraction for one member/date.",
    )
    parser.add_argument("--setup-dir", required=True, type=Path, help="Setup root containing setup YAML")
    parser.add_argument("--project-dir", type=Path, help="Project directory under setup/projects (auto-inferred from --member-results when omitted)")
    parser.add_argument("--member-results", required=True, type=Path, help="Member results directory (contains wet_snow/)")
    parser.add_argument("--aoi", "--roi", dest="aoi", required=True, type=Path, help="Single-feature ROI vector file")
    parser.add_argument("--date", required=True, type=str, help="Date YYYY-MM-DD")
    parser.add_argument("--mask-subdir", default="wet_snow", help="Subdirectory under results/ holding masks")
    parser.add_argument("--mask-prefix", default="wet_snow_mask", help="Filename prefix of masks")
    parser.add_argument("--output", type=Path, help="Optional CSV path (default: <member-results>/wet_snow_fraction_YYYYMMDD.csv)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)

    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
    except Exception as exc:
        logger.error("Invalid --date {}: {}", args.date, exc)
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
        lc_cfg = resolve_landcover_mask(setup_dir, project_dir)
        stats = compute_model_wet_snow_fraction(
            setup_dir=setup_dir,
            project_dir=project_dir,
            results_dir=Path(args.member_results),
            aoi_path=Path(args.aoi),
            landcover_cfg=lc_cfg,
            date=dt,
            mask_subdir=args.mask_subdir,
            mask_prefix=args.mask_prefix,
        )
    except Exception as exc:
        logger.error("Wet-snow computation failed: {}", exc)
        return 1

    out_csv = (
        args.output
        if args.output
        else Path(args.member_results) / f"wet_snow_fraction_{dt.strftime('%Y%m%d')}.csv"
    )
    df = pd.DataFrame(
        {
            "date": [stats["date"]],
            "member_id": [stats["member_id"]],
            "region_id": [stats["region_id"]],
            "wet_snow_fraction": [round(stats["wet_fraction"], 4)],
            "n_valid": [stats["n_valid"]],
            "n_wet": [stats["n_wet"]],
            "raster": [stats["raster"]],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    logger.info(
        "WET_SNOW | raster={} member={} wet_fraction={:.3f} n_valid={} -> {}",
        stats["raster"],
        stats["member_id"],
        stats["wet_fraction"],
        stats["n_valid"],
        out_csv.name,
    )
    return 0


def cli_model_project(argv: list[str] | None = None) -> int:
    """CLI: compute point_wet_snow_roi.csv for every member in each project step."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-wet-snow-project-daily",
        description="Compute daily AOI wet-snow fractions for all prior members in a project.",
    )
    parser.add_argument("--setup-dir", required=True, type=Path, help="Setup root containing setup YAML")
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory containing step_* folders (under steps/)")
    parser.add_argument("--aoi", "--roi", dest="aoi", required=True, type=Path, help="Single-feature ROI vector")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask-subdir", default="wet_snow")
    parser.add_argument("--mask-prefix", default="wet_snow_mask")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)

    setup_dir = Path(args.setup_dir)
    project_dir = Path(args.project_dir)
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
        logger.error("No steps found under {}", project_dir)
        return 1
    lc_cfg = resolve_landcover_mask(setup_dir, project_dir)

    for step in steps:
        try:
            compute_step_wet_snow_daily_for_all_members(
                setup_dir=setup_dir,
                project_dir=project_dir,
                step_dir=step,
                aoi_path=Path(args.aoi),
                landcover_cfg=lc_cfg,
                max_workers=int(args.max_workers or 1),
                overwrite=bool(args.overwrite),
                mask_subdir=args.mask_subdir,
                mask_prefix=args.mask_prefix,
            )
        except Exception as exc:
            logger.error("Wet-snow computation failed for {}: {}", step.name, exc)
            return 2
    return 0


def cli_s1_summary(argv: list[str] | None = None) -> int:
    """CLI: summarize Sentinel-1 WSM rasters into wet_snow_summary.csv."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-wetsnow-raster",
        description="Aggregate categorical wet-snow rasters into a CSV summary.",
    )
    parser.add_argument("--setup-dir", required=True, type=Path, help="Setup root with setup YAML and grids/lc_*.asc")
    parser.add_argument("--project-dir", type=Path, help="Project directory under setup/projects (overrides --project-label lookup)")
    parser.add_argument("--raster-dir", type=Path, help="Directory with WSM_S1*_*.tif rasters (default: <project>/obs/WSM_S1_SAR)")
    parser.add_argument("--aoi", "--roi", dest="aoi", type=Path, help="ROI vector (default: <project>/env/roi.gpkg)")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV (e.g., wet_snow_summary.csv)")
    parser.add_argument("--project-label", type=str, help="Project label to bound dates (default: inferred from output path parent name project_YYYY-YYYY when possible)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)

    setup_root: Path = Path(args.setup_dir)
    raster_dir: Optional[Path] = Path(args.raster_dir) if args.raster_dir else None
    aoi_path: Optional[Path] = Path(args.aoi) if args.aoi else None
    project_label: Optional[str] = args.project_label

    if raster_dir is None:
        cand = setup_root / "obs" / "WSM_S1_SAR"
        if not cand.is_dir():
            logger.error("Default raster dir not found: {}", cand)
            return 1
        raster_dir = cand
    if aoi_path is None:
        try:
            aoi_path = ensure_setup_roi_vector(setup_root)
        except Exception as exc:
            logger.error("No AOI could be resolved/generated under {}: {}", setup_root / "env", exc)
            return 1
    if project_label is None and args.output:
        parent = Path(args.output).parent.name
        if parent.startswith("project_"):
            project_label = parent

    project_dir_for_lc: Path | None = Path(args.project_dir) if args.project_dir is not None else None
    if project_dir_for_lc is None and project_label:
        cand = setup_root / "projects" / project_label
        if cand.is_dir():
            project_dir_for_lc = cand
    if project_dir_for_lc is None:
        logger.error("Cannot resolve project for land-cover config. Provide --project-dir or --project-label.")
        return 1
    lc_cfg = resolve_landcover_mask(setup_root, project_dir_for_lc)
    wet_values, valid_values, exclude_values = _load_obs_wetsnow_classes(project_dir_for_lc)

    project_dates = resolve_project_dates(setup_root, project_label) if project_label else None

    try:
        out_csv = summarize_s1_directory(
            setup_dir=setup_root,
            project_dir=project_dir_for_lc,
            raster_dir=raster_dir,
            aoi_path=aoi_path,
            output_csv=Path(args.output),
            landcover_cfg=lc_cfg,
            overwrite=bool(args.overwrite),
            start=project_dates["start"] if project_dates else None,
            end=project_dates["end"] if project_dates else None,
            wet_values=wet_values,
            valid_values=valid_values,
            exclude_values=exclude_values,
        )
    except Exception as exc:
        logger.error("Sentinel-1 wet-snow summary failed: {}", exc)
        return 1

    logger.info("Sentinel-1 wet-snow summary complete -> {}", out_csv)
    return 0


__all__ = [
    "WetSnowStats",
    "compute_wet_snow_fraction_from_raster",
    "compute_model_wet_snow_fraction",
    "compute_model_wet_snow_line",
    "compute_member_wet_snow_daily",
    "compute_member_wet_snow_line_daily",
    "compute_step_wet_snow_daily_for_all_members",
    "summarize_s1_directory",
    "cli_model",
    "cli_model_project",
    "cli_s1_summary",
]


# Backward-compatible alias for transitional references.
cli_model_setup = cli_model_project


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_s1_summary())
