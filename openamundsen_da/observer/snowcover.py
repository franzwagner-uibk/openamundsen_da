"""Generic snow-cover summarization with configurable classes."""

from __future__ import annotations

import re
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio import features
from rasterio.mask import mask as rio_mask
from rasterio.warp import Resampling, reproject

from openamundsen_da.core.constants import OBS_DIR_NAME
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.observer.scf_uncertainty import (
    ScfUncertaintyConfig,
    _build_uncertainty as build_internal_scf_uncertainty,
    _load_project_config as load_internal_scf_uncertainty_config,
)
from openamundsen_da.util.config_validators import require_mapping, require_nonempty_str
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig, apply_landcover_mask, resolve_landcover_mask
from openamundsen_da.util.observation_raster import (
    netcdf_variable_slice_count,
    open_netcdf_variable_raster,
)
from openamundsen_da.util.observation_time import resolve_acquisition_time
from openamundsen_da.util.project_dates import resolve_project_dates
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector
from openamundsen_da.util.ts import parse_datetime_opt
from openamundsen_da.util.uncertainty_common import (
    assert_same_grid as assert_same_grid_shared,
    normalize_netcdf_times as normalize_netcdf_times_shared,
)


@dataclass(frozen=True)
class SnowcoverClasses:
    valid: list[int]
    cloud: list[int]
    water: list[int]
    nodata: list[int]


@dataclass(frozen=True)
class ScfUncertaintyIngestConfig:
    enabled: bool
    scf_variable: str | None
    uncertainty_variable: str | None
    time_variable: str | None
    uncertainty_source: str | None
    internal_config: ScfUncertaintyConfig | None


def _require_int_list(
    mapping: dict[str, object],
    key: str,
    *,
    path: str,
    allow_empty: bool = False,
) -> list[int]:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {path}.{key}")
    vals = mapping.get(key)
    if not isinstance(vals, Sequence) or isinstance(vals, (str, bytes)):
        raise ValueError(f"{path}.{key} must be a list of integers")
    out: list[int] = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception as exc:
            raise ValueError(f"{path}.{key} contains non-integer value: {v!r}") from exc
    if not out and not allow_empty:
        raise ValueError(f"{path}.{key} must contain at least one integer")
    return out


def _load_classes(project_dir: Path) -> SnowcoverClasses:
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = require_mapping(cfg.get("obs"), path="project.obs")
    snow_cfg = require_mapping(obs_cfg.get("snowcover"), path="project.obs.snowcover")
    classes = require_mapping(snow_cfg.get("classes"), path="project.obs.snowcover.classes")
    return SnowcoverClasses(
        valid=_require_int_list(classes, "valid", path="project.obs.snowcover.classes"),
        cloud=_require_int_list(classes, "cloud", path="project.obs.snowcover.classes", allow_empty=True),
        water=_require_int_list(classes, "water", path="project.obs.snowcover.classes", allow_empty=True),
        nodata=_require_int_list(classes, "nodata", path="project.obs.snowcover.classes", allow_empty=True),
    )


def _load_uncertainty_ingest_config(project_dir: Path) -> ScfUncertaintyIngestConfig:
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    da_cfg = require_mapping(cfg.get("data_assimilation"), path="project.data_assimilation")
    unc_root = da_cfg.get("uncertainty")
    if unc_root is None:
        return ScfUncertaintyIngestConfig(
            enabled=False,
            scf_variable=None,
            uncertainty_variable=None,
            time_variable=None,
            uncertainty_source=None,
            internal_config=None,
        )
    unc_cfg = require_mapping(unc_root, path="project.data_assimilation.uncertainty")
    scf_unc_raw = unc_cfg.get("scf")
    if scf_unc_raw is None:
        return ScfUncertaintyIngestConfig(
            enabled=False,
            scf_variable=None,
            uncertainty_variable=None,
            time_variable=None,
            uncertainty_source=None,
            internal_config=None,
        )
    scf_unc = require_mapping(scf_unc_raw, path="project.data_assimilation.uncertainty.scf")
    enabled = bool(scf_unc.get("enabled", False))
    if not enabled:
        return ScfUncertaintyIngestConfig(
            enabled=False,
            scf_variable=None,
            uncertainty_variable=None,
            time_variable=None,
            uncertainty_source=None,
            internal_config=None,
        )

    ingest = require_mapping(
        scf_unc.get("ingest"),
        path="project.data_assimilation.uncertainty.scf.ingest",
    )
    ingest_path = "project.data_assimilation.uncertainty.scf.ingest"
    scf_variable = require_nonempty_str(ingest, "scf_variable", path=ingest_path)
    time_variable = require_nonempty_str(ingest, "time_variable", path=ingest_path)
    source_raw = ingest.get("uncertainty_source")
    if source_raw is None:
        uncertainty_source = "product"
    else:
        uncertainty_source = str(source_raw).strip().lower()
    if uncertainty_source not in {"product", "internal"}:
        raise ValueError(f"{ingest_path}.uncertainty_source must be one of: product, internal")

    uncertainty_variable: str | None = None
    internal_config: ScfUncertaintyConfig | None = None
    if uncertainty_source == "product":
        uncertainty_variable = require_nonempty_str(ingest, "uncertainty_variable", path=ingest_path)
    else:
        if "uncertainty_variable" in ingest:
            raise ValueError(
                f"{ingest_path}.uncertainty_variable must not be set when uncertainty_source is 'internal'"
            )
        for key in ("u_min", "u_max"):
            if key not in scf_unc:
                raise ValueError(f"Missing required configuration key: project.data_assimilation.uncertainty.scf.{key}")
        internal_config, _ = load_internal_scf_uncertainty_config(project_dir)

    return ScfUncertaintyIngestConfig(
        enabled=True,
        scf_variable=scf_variable,
        uncertainty_variable=uncertainty_variable,
        time_variable=time_variable,
        uncertainty_source=uncertainty_source,
        internal_config=internal_config,
    )


def _disabled_uncertainty_ingest_config() -> ScfUncertaintyIngestConfig:
    return ScfUncertaintyIngestConfig(
        enabled=False,
        scf_variable=None,
        uncertainty_variable=None,
        time_variable=None,
        uncertainty_source=None,
        internal_config=None,
    )


def _extract_date(path: Path) -> datetime:
    stem = path.stem
    parts = stem.split("_")
    # Try contiguous YYYY_MM_DD tokens
    for i in range(len(parts) - 2):
        y, m, d = parts[i : i + 3]
        if all(p.isdigit() for p in (y, m, d)) and len(y) == 4 and len(m) == 2 and len(d) == 2:
            try:
                return datetime.strptime(f"{y}{m}{d}", "%Y%m%d")
            except Exception:
                pass
    # Fallback: any 8-digit token
    for token in parts:
        if len(token) == 8 and token.isdigit():
            try:
                return datetime.strptime(token, "%Y%m%d")
            except Exception:
                continue
    # CLMS naming often embeds YYYYMMDD in longer tokens (e.g. 20250317T101756).
    for token in parts:
        m = re.search(r"(20\d{2})(\d{2})(\d{2})", token)
        if not m:
            continue
        y, mth, d = m.groups()
        try:
            return datetime.strptime(f"{y}{mth}{d}", "%Y%m%d")
        except Exception:
            continue
    raise ValueError(f"Could not infer date from {path.name}")


def _extract_tile(path: Path) -> str:
    m = re.search(r"T(\d{2}[A-Z]{3})", path.name.upper())
    return m.group(1) if m else "UNKNOWN"


def _mask_band(
    src: rasterio.DatasetReader,
    *,
    band_index: int,
    gdf,
    lc_cfg: LandcoverMaskConfig,
) -> tuple[np.ndarray, np.ndarray, float | int | None, np.ndarray, np.ndarray, object]:
    data, transform = rio_mask(
        src,
        gdf.geometry,
        crop=True,
        nodata=src.nodata,
        filled=False,
        indexes=band_index,
    )
    if data.ndim == 2:
        band = data
        out_shape = data.shape
    elif data.ndim == 3:
        band = data[0]
        out_shape = data.shape[1:]
    else:
        raise ValueError(f"Unexpected masked raster dimensions: {data.ndim}")

    roi_mask = features.geometry_mask(
        gdf.geometry,
        out_shape=out_shape,
        transform=transform,
        invert=True,
    )
    arr = np.ma.array(band, copy=False)
    source_mask = np.ma.getmaskarray(arr)
    arr, _ = apply_landcover_mask(arr, transform=transform, target_crs=src.crs, roi_mask=roi_mask, lc_cfg=lc_cfg)
    return np.ma.getdata(arr), np.ma.getmaskarray(arr), src.nodata, roi_mask, source_mask, transform


def _resample_landcover_to_masked_grid(
    *,
    lc_cfg: LandcoverMaskConfig,
    template: rasterio.DatasetReader,
    transform,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if not lc_cfg.enabled or lc_cfg.path is None:
        return None
    if template.crs is None:
        raise ValueError("Raster has no CRS; cannot align land-cover uncertainty penalties")
    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(lc_cfg.path) as src:
        src_crs = src.crs if src.crs is not None else lc_cfg.project_crs
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=transform,
            dst_crs=template.crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    return dst


def _compute_internal_uncertainty(
    *,
    data: np.ndarray,
    lc_cfg: LandcoverMaskConfig,
    template: rasterio.DatasetReader,
    transform,
    uncertainty_cfg: ScfUncertaintyIngestConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    if uncertainty_cfg.internal_config is None:
        raise ValueError("Internal SCF uncertainty configuration is missing")
    landcover = _resample_landcover_to_masked_grid(
        lc_cfg=lc_cfg,
        template=template,
        transform=transform,
        shape=data.shape,
    )
    unc, _fractions = build_internal_scf_uncertainty(
        fsc=data.astype(np.float32, copy=False),
        landcover_resampled=landcover,
        shadow_by_rule={},
        cfg=uncertainty_cfg.internal_config,
    )
    unc_mask = unc == float(uncertainty_cfg.internal_config.nodata_value)
    return unc, unc_mask, float(uncertainty_cfg.internal_config.nodata_value)


def _assert_same_grid(src: rasterio.DatasetReader, other: rasterio.DatasetReader, *, left: Path, right: Path) -> None:
    assert_same_grid_shared(src, other, left=left, right=right)


def _normalize_netcdf_times(time_values: object, *, source_name: str) -> pd.DatetimeIndex:
    return normalize_netcdf_times_shared(time_values, source_name=source_name)


def _missing_uncertainty_companion_error(raster_path: Path) -> FileNotFoundError:
    unc_path = raster_path.parent / f"{raster_path.stem}_uncertainty.tif"
    return FileNotFoundError(
        "Missing required uncertainty companion raster: "
        f"{unc_path}. Generate it first with "
        "'python -m openamundsen_da.observer.scf_uncertainty --setup-dir <SETUP_DIR> "
        "--project-label <PROJECT_LABEL> --overwrite' or provide uncertainty in NetCDF."
    )


def _build_stats_row(
    *,
    date_key: str,
    region_id: str,
    tile: str,
    source_name: str,
    data: np.ndarray,
    mask: np.ndarray,
    nodata: float | int | None,
    roi_mask: np.ndarray,
    source_mask: np.ndarray,
    classes: SnowcoverClasses,
    unc_data: np.ndarray | None = None,
    unc_mask: np.ndarray | None = None,
    unc_nodata: float | int | None = None,
    require_uncertainty: bool = False,
) -> dict[str, object] | None:
    valid = (~mask) & np.isfinite(data)
    if nodata is not None:
        valid &= data != nodata
    if classes.valid:
        valid &= np.isin(data, classes.valid)
    if classes.cloud:
        valid &= ~np.isin(data, classes.cloud)
    if classes.water:
        valid &= ~np.isin(data, classes.water)
    if classes.nodata:
        valid &= ~np.isin(data, classes.nodata)

    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return None

    vals = data[valid].astype(float) / 100.0
    scf = float(vals.mean())
    n_snow = int(round(scf * n_valid))

    scene_valid = roi_mask & (~source_mask) & np.isfinite(data)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        scene_valid &= data != nodata
    if classes.valid:
        scene_valid &= np.isin(data, classes.valid)
    if classes.cloud:
        scene_valid &= ~np.isin(data, classes.cloud)
    if classes.water:
        scene_valid &= ~np.isin(data, classes.water)
    if classes.nodata:
        scene_valid &= ~np.isin(data, classes.nodata)

    n_scene_valid = int(np.count_nonzero(scene_valid))
    invalid = roi_mask & (~scene_valid)
    n_invalid = int(np.count_nonzero(invalid))
    clouds = (~mask) & np.isin(data, classes.cloud) if classes.cloud else np.zeros_like(valid, dtype=bool)
    n_cloud = int(np.count_nonzero(clouds))
    denom = n_valid + n_cloud
    cloud_fraction = (n_cloud / denom) if denom > 0 else 0.0
    invalid_fraction = (n_invalid / (n_scene_valid + n_invalid)) if (n_scene_valid + n_invalid) > 0 else 0.0

    row: dict[str, object] = {
        "date": date_key,
        "region_id": region_id,
        "tile": tile,
        "n_valid": n_valid,
        "n_snow": n_snow,
        "n_cloud": n_cloud,
        "n_invalid": n_invalid,
        "scf": scf,
        "cloud_fraction": cloud_fraction,
        "invalid_fraction": invalid_fraction,
        "source": source_name,
        "_n_scene_valid": n_scene_valid,
    }
    if require_uncertainty:
        if unc_data is None or unc_mask is None:
            raise ValueError(f"Missing uncertainty values for source {source_name}")
        if unc_data.shape != data.shape:
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
        if unc_n_valid != n_valid:
            raise ValueError(
                f"Incomplete uncertainty coverage for source {source_name}: "
                f"unc_n_valid={unc_n_valid}, n_valid={n_valid}"
            )
        unc_vals = unc_data[unc_valid].astype(float)
        row["unc_mean"] = float(np.mean(unc_vals))
        row["unc_min"] = float(np.min(unc_vals))
        row["unc_max"] = float(np.max(unc_vals))
        row["unc_n_valid"] = unc_n_valid
    return row


def _compute_tif_stats(
    *,
    raster_path: Path,
    aoi_path: Path,
    region_field: str | None,
    lc_cfg: LandcoverMaskConfig,
    classes: SnowcoverClasses,
    uncertainty_cfg: ScfUncertaintyIngestConfig,
) -> dict[str, object] | None:
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} has no CRS; cannot align AOI/land cover")
        gdf, region_id = read_single_roi(aoi_path, required_field=region_field, to_crs=src.crs)
        data, mask, nodata, roi_mask, source_mask, transform = _mask_band(src, band_index=1, gdf=gdf, lc_cfg=lc_cfg)

        unc_data: np.ndarray | None = None
        unc_mask: np.ndarray | None = None
        unc_nodata: float | int | None = None
        require_uncertainty = bool(uncertainty_cfg.enabled)
        if uncertainty_cfg.enabled:
            if uncertainty_cfg.uncertainty_source == "internal":
                unc_data, unc_mask, unc_nodata = _compute_internal_uncertainty(
                    data=data,
                    lc_cfg=lc_cfg,
                    template=src,
                    transform=transform,
                    uncertainty_cfg=uncertainty_cfg,
                )
            else:
                unc_path = raster_path.parent / f"{raster_path.stem}_uncertainty.tif"
                if not unc_path.is_file():
                    raise _missing_uncertainty_companion_error(raster_path)
                with rasterio.open(unc_path) as src_unc:
                    _assert_same_grid(src, src_unc, left=raster_path, right=unc_path)
                    unc_data, unc_mask, unc_nodata, _unc_roi_mask, _unc_source_mask, _unc_transform = _mask_band(
                        src_unc,
                        band_index=1,
                        gdf=gdf,
                        lc_cfg=lc_cfg,
                    )

    return _build_stats_row(
        date_key=_extract_date(raster_path).strftime("%Y-%m-%d"),
        region_id=region_id,
        tile=_extract_tile(raster_path),
        source_name=raster_path.name,
        data=data,
        mask=mask,
        nodata=nodata,
        roi_mask=roi_mask,
        source_mask=source_mask,
        classes=classes,
        unc_data=unc_data,
        unc_mask=unc_mask,
        unc_nodata=unc_nodata,
        require_uncertainty=require_uncertainty,
    )


def _compute_netcdf_product_stats(
    *,
    raster_path: Path,
    aoi_path: Path,
    region_field: str | None,
    lc_cfg: LandcoverMaskConfig,
    classes: SnowcoverClasses,
    uncertainty_cfg: ScfUncertaintyIngestConfig,
) -> list[dict[str, object]]:
    if uncertainty_cfg.scf_variable is None or uncertainty_cfg.time_variable is None:
        raise ValueError("Missing required NetCDF ingest variable names for snow-cover uncertainty")

    try:
        import xarray as xr  # lazy dependency
    except Exception as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("xarray is required to process NetCDF snow-cover uncertainty ingestion") from exc

    with xr.open_dataset(raster_path) as ds:
        if uncertainty_cfg.scf_variable not in ds:
            raise ValueError(
                f"Variable '{uncertainty_cfg.scf_variable}' not found in {raster_path.name}"
            )
        if uncertainty_cfg.uncertainty_source == "product":
            if uncertainty_cfg.uncertainty_variable is None:
                raise ValueError("Missing required NetCDF uncertainty variable name")
            if uncertainty_cfg.uncertainty_variable not in ds:
                raise ValueError(
                    f"Variable '{uncertainty_cfg.uncertainty_variable}' not found in {raster_path.name}"
                )
        if uncertainty_cfg.time_variable not in ds:
            raise ValueError(
                f"Time variable '{uncertainty_cfg.time_variable}' not found in {raster_path.name}"
            )
        times = _normalize_netcdf_times(
            ds[uncertainty_cfg.time_variable].values,
            source_name=raster_path.name,
        )

    scf_count = netcdf_variable_slice_count(raster_path, uncertainty_cfg.scf_variable)
    unc_count = (
        netcdf_variable_slice_count(raster_path, uncertainty_cfg.uncertainty_variable)
        if uncertainty_cfg.uncertainty_source == "product" and uncertainty_cfg.uncertainty_variable is not None
        else None
    )
    if unc_count is not None and unc_count != len(times):
        raise ValueError(
            f"Band/time mismatch in {raster_path.name}: uncertainty bands={unc_count} but time steps={len(times)}"
        )
    if scf_count != len(times):
        raise ValueError(
            f"Band/time mismatch in {raster_path.name}: SCF bands={scf_count} but time steps={len(times)}"
        )

    rows: list[dict[str, object]] = []
    tile = _extract_tile(raster_path)
    for i, ts in enumerate(times, start=1):
        with ExitStack() as stack:
            src = stack.enter_context(
                open_netcdf_variable_raster(
                    raster_path,
                    variable=uncertainty_cfg.scf_variable,
                    band_index=i,
                )
            )
            if src.crs is None:
                raise ValueError(f"Raster {raster_path} has no CRS; cannot align AOI/land cover")
            gdf, region_id = read_single_roi(aoi_path, required_field=region_field, to_crs=src.crs)
            data, mask, nodata, roi_mask, source_mask, transform = _mask_band(
                src,
                band_index=1,
                gdf=gdf,
                lc_cfg=lc_cfg,
            )
            if uncertainty_cfg.uncertainty_source == "product":
                assert uncertainty_cfg.uncertainty_variable is not None
                src_unc = stack.enter_context(
                    open_netcdf_variable_raster(
                        raster_path,
                        variable=uncertainty_cfg.uncertainty_variable,
                        band_index=i,
                    )
                )
                _assert_same_grid(src, src_unc, left=raster_path, right=raster_path)
                unc_data, unc_mask, unc_nodata, _unc_roi_mask, _unc_source_mask, _unc_transform = _mask_band(
                    src_unc,
                    band_index=1,
                    gdf=gdf,
                    lc_cfg=lc_cfg,
                )
            else:
                unc_data, unc_mask, unc_nodata = _compute_internal_uncertainty(
                    data=data,
                    lc_cfg=lc_cfg,
                    template=src,
                    transform=transform,
                    uncertainty_cfg=uncertainty_cfg,
                )
            source_name = f"{raster_path.name}@{ts.strftime('%Y-%m-%dT%H:%M:%SZ')}"
            row = _build_stats_row(
                date_key=ts.date().isoformat(),
                region_id=region_id,
                tile=tile,
                source_name=source_name,
                data=data,
                mask=mask,
                nodata=nodata,
                roi_mask=roi_mask,
                source_mask=source_mask,
                classes=classes,
                unc_data=unc_data,
                unc_mask=unc_mask,
                unc_nodata=unc_nodata,
                require_uncertainty=True,
            )
            if row is not None:
                rows.append(row)
    return rows


def summarize_snowcover_directory(
    *,
    setup_dir: Path,
    input_dir: Path,
    aoi: Path,
    project_label: str,
    output_root: Path,
    region_field: str | None = None,
    landcover_cfg: LandcoverMaskConfig | None = None,
    classes: SnowcoverClasses | None = None,
    recursive: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[Path]:
    """Summarize snow-cover rasters into scf_summary.csv."""
    patterns = ["*.tif", "*.tiff", "*.nc"]
    rasters_all: list[Path] = []
    for patt in patterns:
        rasters_all.extend(sorted(input_dir.rglob(patt) if recursive else input_dir.glob(patt)))
    rasters = [p for p in rasters_all if not p.stem.lower().endswith("_uncertainty")]

    if not rasters:
        logger.warning("No snow-cover rasters found in {}", input_dir)
        return []

    project_dir = Path(setup_dir) / "projects" / str(project_label)
    project_cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = require_mapping(project_cfg.get("obs"), path="project.obs")
    product_cfg = require_mapping(obs_cfg.get("snowcover"), path="project.obs.snowcover")
    product_tag = require_nonempty_str(product_cfg, "product_tag", path="project.obs.snowcover")
    parser_raw = product_cfg.get("filename_time_parser")
    filename_parser = str(parser_raw).strip() if parser_raw is not None else None
    manifest_raw = product_cfg.get("acquisition_manifest")
    manifest_path = Path(setup_dir) / str(manifest_raw) if manifest_raw is not None else None
    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(setup_dir), project_dir)
    cls = classes or _load_classes(project_dir)
    uncertainty_cfg = _load_uncertainty_ingest_config(project_dir)
    output_dir = output_root / project_label
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "scf_summary.csv"
    audit_path = output_dir / "scf_source_audit.csv"

    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    written: list[Path] = []
    disabled_uncertainty_cfg = _disabled_uncertainty_ingest_config()
    for rast in rasters:
        try:
            suffix = rast.suffix.lower()
            if not uncertainty_cfg.enabled:
                if suffix not in {".tif", ".tiff", ".nc"}:
                    continue
                stats = _compute_tif_stats(
                    raster_path=rast,
                    aoi_path=aoi,
                    region_field=region_field,
                    lc_cfg=lc_cfg,
                    classes=cls,
                    uncertainty_cfg=disabled_uncertainty_cfg,
                )
                stats_rows = [stats] if stats is not None else []
            else:
                if suffix == ".nc":
                    stats_rows = _compute_netcdf_product_stats(
                        raster_path=rast,
                        aoi_path=aoi,
                        region_field=region_field,
                        lc_cfg=lc_cfg,
                        classes=cls,
                        uncertainty_cfg=uncertainty_cfg,
                    )
                elif suffix in {".tif", ".tiff"}:
                    stats = _compute_tif_stats(
                        raster_path=rast,
                        aoi_path=aoi,
                        region_field=region_field,
                        lc_cfg=lc_cfg,
                        classes=cls,
                        uncertainty_cfg=uncertainty_cfg,
                    )
                    stats_rows = [stats] if stats is not None else []
                else:
                    continue
        except Exception as exc:
            if uncertainty_cfg.enabled:
                raise RuntimeError(
                    f"Snow-cover preprocessing failed for {rast.name} with uncertainty enabled: {exc}"
                ) from exc
            audit_rows.append(
                {"source": rast.name, "status": "failed", "reason": str(exc)}
            )
            logger.debug("Skipping snow-cover source {}: {}", rast.name, exc)
            continue
        accepted = 0
        for stats in stats_rows:
            if stats is None:
                continue
            stats_date = parse_datetime_opt(str(stats["date"]))
            if stats_date is None:
                if uncertainty_cfg.enabled:
                    raise ValueError(
                        f"Could not parse derived observation date '{stats['date']}' for source {stats.get('source', rast.name)}"
                    )
                continue
            if start and stats_date < start:
                continue
            if end and stats_date > end:
                continue
            source_text = str(stats.get("source", rast.name))
            source_name, separator, cf_time = source_text.partition("@")
            acquisition = resolve_acquisition_time(
                source_path=Path(input_dir) / Path(source_name).name,
                product=product_tag,
                observation_date=stats["date"],
                cf_time=(cf_time if separator else None),
                filename_parser=filename_parser,
                manifest_path=manifest_path,
            )
            stats["acquisition_time"] = acquisition.value.isoformat().replace("+00:00", "Z")
            stats["time_source"] = acquisition.source
            stats["time_quality"] = acquisition.quality
            rows.append(stats)
            accepted += 1
            audit_rows.append(
                {
                    "source": str(stats.get("source", rast.name)),
                    "status": "accepted",
                    "reason": (
                        "fallback_midnight"
                        if acquisition.quality == "fallback_midnight"
                        else ""
                    ),
                    "date": stats["date"],
                    "acquisition_time": stats["acquisition_time"],
                    "n_valid": int(stats["n_valid"]),
                }
            )
            logger.debug(
                "Snowcover {} -> scf={:.3f} n_valid={} n_snow={}",
                str(stats.get("source", rast.name)),
                float(stats["scf"]),
                int(stats["n_valid"]),
                int(stats["n_snow"]),
            )
        if accepted > 0:
            written.append(rast)
        elif stats_rows:
            audit_rows.append(
                {
                    "source": rast.name,
                    "status": "discarded",
                    "reason": "date filter removed all matching records",
                }
            )
        else:
            audit_rows.append(
                {
                    "source": rast.name,
                    "status": "discarded",
                    "reason": "AOI contained no valid pixels",
                }
            )

    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "source",
            "status",
            "reason",
            "date",
            "acquisition_time",
            "n_valid",
        ],
    )
    audit.to_csv(audit_path, index=False)
    status_counts = audit.get("status", pd.Series(dtype="object")).value_counts()
    logger.info(
        "Snow-cover source audit | accepted={} discarded={} failed={} details={}",
        int(status_counts.get("accepted", 0)),
        int(status_counts.get("discarded", 0)),
        int(status_counts.get("failed", 0)),
        audit_path,
    )

    if not rows:
        logger.warning("No valid snow-cover rasters processed.")
        return []

    # Keep one best raster per date/tile and aggregate tile contributions to one row per date.
    # "Best" means highest valid pixel count; tie-breaker is lower cloud fraction.
    best_per_date_tile: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["date"]),
            str(row["acquisition_time"]),
            str(row.get("tile", "UNKNOWN")),
        )
        prev = best_per_date_tile.get(key)
        if prev is None:
            best_per_date_tile[key] = row
            continue
        prev_valid = int(prev.get("n_valid", 0))
        curr_valid = int(row.get("n_valid", 0))
        if curr_valid > prev_valid:
            best_per_date_tile[key] = row
            continue
        if curr_valid == prev_valid:
            prev_cloud = float(prev.get("cloud_fraction", 1.0))
            curr_cloud = float(row.get("cloud_fraction", 1.0))
            if curr_cloud < prev_cloud:
                best_per_date_tile[key] = row

    agg: dict[tuple[str, str, str], dict[str, object]] = {}
    for row in best_per_date_tile.values():
        key = (str(row["date"]), str(row["acquisition_time"]), str(row["region_id"]))
        slot = agg.get(key)
        if slot is None:
            slot = {
                "date": row["date"],
                "region_id": row["region_id"],
                "acquisition_time": row["acquisition_time"],
                "time_source": row["time_source"],
                "time_quality": row["time_quality"],
                "n_valid": 0,
                "n_snow": 0,
                "n_cloud": 0,
                "n_invalid": 0,
                "_n_scene_valid": 0,
                "source_set": set(),
                "tile_set": set(),
            }
            if uncertainty_cfg.enabled:
                slot["unc_mean_num"] = 0.0
                slot["unc_n_valid"] = 0
                slot["unc_min"] = float("inf")
                slot["unc_max"] = float("-inf")
            agg[key] = slot
        slot["n_valid"] = int(slot["n_valid"]) + int(row.get("n_valid", 0))
        slot["n_snow"] = int(slot["n_snow"]) + int(row.get("n_snow", 0))
        slot["n_cloud"] = int(slot["n_cloud"]) + int(row.get("n_cloud", 0))
        slot["n_invalid"] = int(slot["n_invalid"]) + int(row.get("n_invalid", 0))
        slot["_n_scene_valid"] = int(slot["_n_scene_valid"]) + int(row.get("_n_scene_valid", 0))
        slot["source_set"].add(str(row.get("source", "")))
        slot["tile_set"].add(str(row.get("tile", "UNKNOWN")))
        if uncertainty_cfg.enabled:
            unc_n_valid = int(row.get("unc_n_valid", 0))
            if unc_n_valid > 0:
                unc_mean = float(row["unc_mean"])
                slot["unc_mean_num"] = float(slot["unc_mean_num"]) + (unc_mean * unc_n_valid)
                slot["unc_n_valid"] = int(slot["unc_n_valid"]) + unc_n_valid
                slot["unc_min"] = min(float(slot["unc_min"]), float(row["unc_min"]))
                slot["unc_max"] = max(float(slot["unc_max"]), float(row["unc_max"]))

    out_rows: list[dict[str, object]] = []
    for entry in agg.values():
        n_valid = int(entry["n_valid"])
        n_snow = int(entry["n_snow"])
        n_cloud = int(entry["n_cloud"])
        n_invalid = int(entry["n_invalid"])
        n_scene_valid = int(entry["_n_scene_valid"])
        scf = (n_snow / n_valid) if n_valid > 0 else 0.0
        denom = n_valid + n_cloud
        cloud_fraction = (n_cloud / denom) if denom > 0 else 0.0
        invalid_fraction = (n_invalid / (n_scene_valid + n_invalid)) if (n_scene_valid + n_invalid) > 0 else 0.0
        row = {
            "date": entry["date"],
            "region_id": entry["region_id"],
            "acquisition_time": entry["acquisition_time"],
            "time_source": entry["time_source"],
            "time_quality": entry["time_quality"],
            "n_valid": n_valid,
            "n_snow": n_snow,
            "n_cloud": n_cloud,
            "n_invalid": n_invalid,
            "scf": scf,
            "cloud_fraction": cloud_fraction,
            "invalid_fraction": invalid_fraction,
            "tiles_used": ";".join(sorted(x for x in entry["tile_set"] if x)),
            "source": ";".join(sorted(x for x in entry["source_set"] if x)),
        }
        if uncertainty_cfg.enabled:
            # Weighted by contributing uncertainty-valid pixels.
            row_unc_mean_num = float(entry.get("unc_mean_num", 0.0))
            row_unc_n_valid = int(entry.get("unc_n_valid", 0))
            if row_unc_n_valid <= 0:
                raise ValueError(
                    f"Uncertainty is enabled but no uncertainty-valid support exists for date {entry['date']}"
                )
            row["unc_mean"] = row_unc_mean_num / row_unc_n_valid
            row["unc_n_valid"] = row_unc_n_valid
            row["unc_min"] = float(entry["unc_min"])
            row["unc_max"] = float(entry["unc_max"])
        out_rows.append(row)

    df = pd.DataFrame(out_rows).sort_values("date")
    for column in ("unc_min", "unc_max"):
        if column in df.columns:
            df[column] = df[column].round(3)
    df.to_csv(summary_path, index=False)
    logger.info("Snow-cover summary written: {} ({} day(s), {} raster(s))", summary_path, len(df), len(best_per_date_tile))
    return written


def cli_main(argv: List[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-snowcover",
        description="Summarize snow-cover rasters (GeoTIFF/NetCDF) into scf_summary.csv with configurable classes.",
    )
    p.add_argument("--input-dir", required=True, type=Path, help="Directory containing snow-cover rasters (.tif/.tiff/.nc)")
    p.add_argument("--roi", dest="roi", type=Path, help="Single-feature ROI vector (default: <project>/env/roi.gpkg)")
    p.add_argument("--project-label", required=True, help="Project folder name under obs/")
    p.add_argument("--setup-dir", type=Path, help="Setup directory (default: CWD)")
    p.add_argument("--output-root", type=Path, help="Override output root (default: <project>/obs/summaries)")
    p.add_argument("--roi-field", dest="roi_field", default=None, help="Field name in ROI with the region identifier (optional)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for rasters")
    p.add_argument("--start-date", type=str, help="Optional ISO start date filter")
    p.add_argument("--end-date", type=str, help="Optional ISO end date filter")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing scf_summary.csv")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_cli_logger(args.log_level)

    setup_dir = Path(args.setup_dir) if args.setup_dir else Path.cwd()
    default_root = setup_dir / OBS_DIR_NAME / "summaries"
    output_root = Path(args.output_root) if args.output_root else default_root
    project_cfg_dir = setup_dir / "projects" / str(args.project_label)
    lc_cfg = resolve_landcover_mask(setup_dir, project_cfg_dir)
    cls = _load_classes(project_cfg_dir)
    project_dates = resolve_project_dates(setup_dir, args.project_label)

    start = parse_datetime_opt(args.start_date) if args.start_date else None
    end = parse_datetime_opt(args.end_date) if args.end_date else None
    if project_dates:
        start = start or project_dates.get("start")
        end = end or project_dates.get("end")

    try:
        aoi_path = Path(args.roi) if args.roi else ensure_setup_roi_vector(setup_dir)
        summarize_snowcover_directory(
            setup_dir=setup_dir,
            input_dir=Path(args.input_dir),
            aoi=aoi_path,
            project_label=str(args.project_label),
            output_root=output_root,
            region_field=str(args.roi_field) if args.roi_field else None,
            landcover_cfg=lc_cfg,
            classes=cls,
            recursive=bool(args.recursive),
            start=start,
            end=end,
        )
        return 0
    except Exception as exc:
        logger.error("Snow-cover summarization failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
