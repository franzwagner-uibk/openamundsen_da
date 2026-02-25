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
import rasterio
from loguru import logger
from rasterio import features
from rasterio.mask import mask as rio_mask

from openamundsen_da.io.paths import (
    member_id_from_results_dir,
    list_step_dirs,
    infer_project_dir,
    infer_setup_dir_from_project,
)
from openamundsen_da.methods.daily_aoi_series import (
    compute_step_daily_series_for_all_members,
    step_start_end,
)
from openamundsen_da.util.landcover_mask import (
    LandcoverMaskConfig,
    apply_landcover_mask,
    deserialize_landcover_mask_config,
    resolve_landcover_mask,
    serialize_landcover_mask_config,
)
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector
from openamundsen_da.observer.class_config import load_wetsnow_classes
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.project_dates import resolve_project_dates


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
    raster_path: Path,
    aoi_path: Path,
    *,
    lc_cfg: LandcoverMaskConfig,
) -> tuple[np.ma.MaskedArray, np.ndarray, rasterio.Affine, float | None, str, object]:
    """Read raster values cropped to the AOI; return masked array, ROI mask, and metadata."""

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} lacks a CRS")
        gdf, region_id = read_single_roi(
            aoi_path,
            required_field=None,
            to_crs=src.crs,
        )
        data, transform = rio_mask(
            src,
            gdf.geometry,
            crop=True,
            nodata=src.nodata,
            filled=False,
        )
        roi_mask = features.geometry_mask(
            gdf.geometry,
            out_shape=data.shape[1:],
            transform=transform,
            invert=True,
        )
    arr = np.ma.array(data[0], copy=False)
    pixel_area = None
    if transform is not None:
        try:
            pixel_area = abs(float(transform.a) * float(transform.e))
        except AttributeError:
            pass
    arr, _ = apply_landcover_mask(
        arr,
        transform=transform,
        target_crs=src.crs,
        roi_mask=roi_mask,
        lc_cfg=lc_cfg,
    )
    return arr, roi_mask, transform, pixel_area, region_id, src.crs


def _compute_fraction(
    arr: np.ma.MaskedArray,
    wet_values: Sequence[int],
    valid_values: Sequence[int] | None = None,
    exclude_values: Sequence[int] | None = None,
    pixel_area: float | None = None,
    region_id: str = "",
) -> WetSnowStats:
    """Return wet/valid counts and their ratio for the provided array."""

    data = np.ma.getdata(arr)
    mask = np.ma.getmaskarray(arr)
    valid = (~mask) & np.isfinite(data)
    if valid_values:
        valid &= np.isin(data, valid_values)
    if exclude_values:
        valid &= ~np.isin(data, exclude_values)

    wet = valid & np.isin(data, wet_values)
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

    arr, roi_mask, transform, pixel_area, region_id, crs = _read_mask_by_aoi(
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
) -> dict:
    """Compute AOI wet-snow fraction for one member/date."""

    if landcover_cfg is not None:
        lc_cfg = landcover_cfg
    else:
        lc_cfg = resolve_landcover_mask(Path(setup_dir), Path(project_dir))
    raster = _find_mask_raster(Path(results_dir), date, subdir=mask_subdir, prefix=mask_prefix)
    stats = compute_wet_snow_fraction_from_raster(
        raster,
        aoi_path,
        wet_values=_MODEL_WET,
        valid_values=_MODEL_VALID,
        landcover_cfg=lc_cfg,
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
        "raster": Path(raster).name,
    }


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
    rows: list[dict[str, object]] = []
    for tif in files:
        try:
            date = _parse_s1_timestamp(tif.name)
        except ValueError:
            continue
        if start and date < start:
            continue
        if end and date > end:
            continue
        try:
            stats = compute_wet_snow_fraction_from_raster(
                tif,
                aoi_path,
                wet_values=wet_values,
                valid_values=valid_values,
                exclude_values=exclude_values,
                landcover_cfg=lc_cfg,
            )
        except Exception as exc:
            logger.warning("Skipping {}: {}", tif.name, exc)
            continue
        logger.info(
            "Wet-snow {} -> wet_fraction={:.3f} n_valid={} n_wet={}",
            tif.name,
            stats.wet_fraction,
            stats.valid_pixels,
            stats.wet_pixels,
        )
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "region_id": stats.region_id,
                "tile": _extract_tile(tif.name),
                "wet_snow_fraction": round(stats.wet_fraction, 4),
                "n_valid": stats.valid_pixels,
                "n_wet": stats.wet_pixels,
                "source": tif.name,
            }
        )

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
            }
            agg[key] = slot
        slot["n_valid"] = int(slot["n_valid"]) + int(row.get("n_valid", 0))
        slot["n_wet"] = int(slot["n_wet"]) + int(row.get("n_wet", 0))
        slot["source_set"].add(str(row.get("source", "")))
        slot["tile_set"].add(str(row.get("tile", "UNKNOWN")))

    out_rows: list[dict[str, object]] = []
    for entry in agg.values():
        n_valid = int(entry["n_valid"])
        n_wet = int(entry["n_wet"])
        frac = (n_wet / n_valid) if n_valid > 0 else 0.0
        out_rows.append(
            {
                "date": entry["date"],
                "region_id": entry["region_id"],
                "wet_snow_fraction": round(frac, 4),
                "n_valid": n_valid,
                "n_wet": n_wet,
                "tiles_used": ";".join(sorted(x for x in entry["tile_set"] if x)),
                "source": ";".join(sorted(x for x in entry["source_set"] if x)),
            }
        )

    df = pd.DataFrame(out_rows).sort_values("date")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
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
    "compute_member_wet_snow_daily",
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



