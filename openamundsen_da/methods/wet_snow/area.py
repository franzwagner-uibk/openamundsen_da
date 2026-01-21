"""Wet-snow area fractions for model and observation rasters.

This module mirrors the SCF operator structure (openamundsen_da.methods.h_of_x)
but works on categorical wet-snow masks:

* Model side: use :func:`compute_model_wet_snow_fraction` on the binary masks
  produced by ``wet_snow.classify`` (1 = wet, 0 = dry, 255 = nodata).
* Observation side: :func:`compute_wet_snow_fraction_from_raster` can ingest
  arbitrary categorical rasters such as the Sentinel-1 WSM product where
  110 = wet, 125 = dry, 200 = radar shadow, 210 = water.

Both paths clip the raster to a single-feature AOI and report the fraction of
wet pixels among all valid pixels (area-weighted under the equal-area pixel
assumption). ``point_wet_snow_roi.csv`` mirrors ``point_scf_roi.csv`` and can
be generated per member or per season via the provided CLIs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import numpy as np
import pandas as pd
import rasterio
import yaml
from loguru import logger
from rasterio import features
from rasterio.mask import mask as rio_mask
from pyproj import CRS

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.io.paths import member_id_from_results_dir, list_step_dirs
from openamundsen_da.methods.daily_aoi_series import (
    compute_step_daily_series_for_all_members,
    step_start_end,
)
from openamundsen_da.util.landcover_mask import (
    LandcoverMaskConfig,
    apply_landcover_mask,
    resolve_landcover_mask,
)
from openamundsen_da.util.roi import read_single_roi


_MODEL_WET = (1,)
_MODEL_VALID = (0, 1)
_S1_WET = (110,)
_S1_VALID = (110, 125)
_S1_EXCLUDE = (200, 210)


@dataclass(frozen=True)
class WetSnowStats:
    """Summary of wet-snow coverage inside an AOI."""

    wet_fraction: float
    wet_pixels: int
    valid_pixels: int
    wet_area_m2: float | None
    valid_area_m2: float | None
    region_id: str


def _serialize_lc_cfg(lc_cfg: LandcoverMaskConfig) -> dict[str, object]:
    """Return a dict-friendly form of LandcoverMaskConfig."""
    return {
        "enabled": lc_cfg.enabled,
        "path": str(lc_cfg.path) if lc_cfg.path else None,
        "classes": tuple(lc_cfg.classes),
        "project_crs_wkt": lc_cfg.project_crs.to_wkt(),
    }


def _deserialize_lc_cfg(data: Any) -> LandcoverMaskConfig | None:
    """Rehydrate LandcoverMaskConfig from serialized data."""
    if isinstance(data, LandcoverMaskConfig):
        return data
    if not isinstance(data, dict):
        return None
    path_val = data.get("path")
    crs_wkt = data.get("project_crs_wkt")
    if crs_wkt is None:
        return None
    return LandcoverMaskConfig(
        enabled=bool(data.get("enabled", False)),
        path=Path(path_val) if path_val else None,
        classes=tuple(data.get("classes") or ()),
        project_crs=CRS.from_wkt(str(crs_wkt)),
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
    project_dir: Path,
    results_dir: Path,
    aoi_path: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    date: datetime,
    mask_subdir: str = "wet_snow",
    mask_prefix: str = "wet_snow_mask",
) -> dict:
    """Compute AOI wet-snow fraction for one member/date."""

    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(project_dir))
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

    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(project_dir))
    start_day = datetime(start.year, start.month, start.day)
    end_day = datetime(end.year, end.month, end.day)
    if end_day < start_day:
        return pd.DataFrame(columns=["time", "wet_snow_fraction"])

    dates = pd.date_range(start_day, end_day, freq="D").to_pydatetime()
    rows: list[dict[str, object]] = []
    for dt in dates:
        try:
            stats = compute_model_wet_snow_fraction(
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
    lc_cfg = _deserialize_lc_cfg(extra.get("landcover_cfg"))
    df = compute_member_wet_snow_daily(
        project_dir=Path(extra["project_dir"]),
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
    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(project_dir))

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
            "landcover_cfg": _serialize_lc_cfg(lc_cfg),
            "project_dir": str(project_dir),
        },
    )


def summarize_s1_directory(
    *,
    project_dir: Path,
    raster_dir: Path,
    aoi_path: Path,
    output_csv: Path,
    landcover_cfg: LandcoverMaskConfig | None = None,
    overwrite: bool = False,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Path:
    """Summarize Sentinel-1 wet-snow maps into one CSV (date, fraction)."""

    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(project_dir))
    if output_csv.exists() and not overwrite:
        return output_csv

    files = sorted(Path(raster_dir).glob("*.tif"))
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
                wet_values=_S1_WET,
                valid_values=_S1_VALID,
                exclude_values=_S1_EXCLUDE,
                landcover_cfg=lc_cfg,
            )
        except Exception as exc:
            logger.warning("Skipping {}: {}", tif.name, exc)
            continue
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "region_id": stats.region_id,
                "wet_snow_fraction": round(stats.wet_fraction, 4),
                "n_valid": stats.valid_pixels,
                "n_wet": stats.wet_pixels,
                "source": tif.name,
            }
        )

    if not rows:
        raise RuntimeError(f"No valid Sentinel-1 rasters processed in {raster_dir}")

    df = pd.DataFrame(rows).sort_values("date")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(
        "Sentinel-1 wet-snow summary written: {} ({} day(s))",
        output_csv,
        len(df),
    )
    return output_csv


def _parse_s1_timestamp(name: str) -> datetime:
    parts = name.split("_")
    if len(parts) < 7:
        raise ValueError(f"Cannot parse timestamp from {name}")
    try:
        year, month, day = map(int, parts[4:7])
    except Exception as exc:
        raise ValueError(f"Cannot parse date from {name}") from exc
    return datetime(year, month, day)


def _resolve_season_dates(project_dir: Optional[Path], season_label: Optional[str]) -> dict[str, datetime] | None:
    """Load start/end dates from propagation/<season_label>/season.yml if present."""

    if project_dir is None or season_label is None:
        return None
    season_yml = project_dir / "propagation" / season_label / "season.yml"
    if not season_yml.exists():
        return None
    try:
        data = yaml.safe_load(season_yml.read_text())
        start = datetime.fromisoformat(str(data.get("start_date")))
        end = datetime.fromisoformat(str(data.get("end_date")))
        return {"start": start, "end": end}
    except Exception as exc:
        logger.warning("Could not parse season dates from {}: {}", season_yml, exc)
        return None


def cli_model(argv: list[str] | None = None) -> int:
    """CLI entry point mirroring oa-da-model-scf but for wet-snow area."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-wet-snow",
        description="Compute AOI wet-snow fraction for one member/date.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project root containing project.yml")
    parser.add_argument("--member-results", required=True, type=Path, help="Member results directory (contains wet_snow/)")
    parser.add_argument("--aoi", "--roi", dest="aoi", required=True, type=Path, help="Single-feature ROI vector file")
    parser.add_argument("--date", required=True, type=str, help="Date YYYY-MM-DD")
    parser.add_argument("--mask-subdir", default="wet_snow", help="Subdirectory under results/ holding masks")
    parser.add_argument("--mask-prefix", default="wet_snow_mask", help="Filename prefix of masks")
    parser.add_argument("--output", type=Path, help="Optional CSV path (default: <member-results>/wet_snow_fraction_YYYYMMDD.csv)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    try:
        dt = datetime.strptime(args.date, "%Y-%m-%d")
    except Exception as exc:
        logger.error("Invalid --date {}: {}", args.date, exc)
        return 2

    try:
        lc_cfg = resolve_landcover_mask(Path(args.project_dir))
        stats = compute_model_wet_snow_fraction(
            project_dir=Path(args.project_dir),
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


def cli_model_season(argv: list[str] | None = None) -> int:
    """CLI: compute point_wet_snow_roi.csv for every member in each step."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-model-wet-snow-season-daily",
        description="Compute daily AOI wet-snow fractions for all prior members in a season.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project root containing project.yml")
    parser.add_argument("--season-dir", required=True, type=Path, help="Season directory containing step_* folders (under steps/)")
    parser.add_argument("--aoi", "--roi", dest="aoi", required=True, type=Path, help="Single-feature ROI vector")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--mask-subdir", default="wet_snow")
    parser.add_argument("--mask-prefix", default="wet_snow_mask")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    steps = list_step_dirs(args.season_dir)
    if not steps:
        logger.error("No steps found under {}", args.season_dir)
        return 1
    lc_cfg = resolve_landcover_mask(Path(args.project_dir))

    for step in steps:
        try:
            compute_step_wet_snow_daily_for_all_members(
                project_dir=Path(args.project_dir),
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
        prog="oa-da-wet-snow-s1",
        description="Aggregate Sentinel-1 WSM wet-snow rasters into a CSV summary.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project root with project.yml and grids/lc_*.asc")
    parser.add_argument("--raster-dir", type=Path, help="Directory with WSM_S1*_*.tif rasters (default: <project>/obs/WSM_S1_SAR)")
    parser.add_argument("--aoi", "--roi", dest="aoi", type=Path, help="Single-feature ROI vector (default: <project>/env/roi.gpkg)")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV (e.g., wet_snow_summary.csv)")
    parser.add_argument("--season-label", type=str, help="Season label to bound dates (default: inferred from output path name season_YYYY-YYYY when possible)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    proj_dir: Path = Path(args.project_dir)
    raster_dir: Optional[Path] = Path(args.raster_dir) if args.raster_dir else None
    aoi_path: Optional[Path] = Path(args.aoi) if args.aoi else None
    season_label: Optional[str] = args.season_label

    if raster_dir is None:
        cand = proj_dir / "obs" / "WSM_S1_SAR"
        if not cand.is_dir():
            logger.error("Default raster dir not found: {}", cand)
            return 1
        raster_dir = cand
    if aoi_path is None:
        env_dir = proj_dir / "env"
        roi = env_dir / "roi.gpkg"
        if roi.is_file():
            aoi_path = roi
        else:
            candidates = sorted(list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp")))
            if not candidates:
                logger.error("No AOI found under {}", env_dir)
                return 1
            if len(candidates) > 1:
                logger.error("Expected roi.gpkg under {}; found multiple candidates. Please pass --aoi.", env_dir)
                return 1
            aoi_path = candidates[0]
    if season_label is None and args.output:
        parent = Path(args.output).parent.name
        if parent.startswith("season_"):
            season_label = parent

    lc_cfg = resolve_landcover_mask(proj_dir)

    season_dates = _resolve_season_dates(proj_dir, season_label) if proj_dir and season_label else None

    try:
        out_csv = summarize_s1_directory(
            project_dir=proj_dir,
            raster_dir=raster_dir,
            aoi_path=aoi_path,
            output_csv=Path(args.output),
            landcover_cfg=lc_cfg,
            overwrite=bool(args.overwrite),
            start=season_dates["start"] if season_dates else None,
            end=season_dates["end"] if season_dates else None,
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
    "cli_model_season",
    "cli_s1_summary",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_s1_summary())
