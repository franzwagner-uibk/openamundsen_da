"""Generic snow-cover summarization with configurable classes."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
import rasterio
import yaml
from loguru import logger
from rasterio import features
from rasterio.mask import mask as rio_mask

from openamundsen_da.core.constants import LOGURU_FORMAT, OBS_DIR_NAME
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, find_season_yaml
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig, apply_landcover_mask, resolve_landcover_mask
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.ts import parse_datetime_opt


@dataclass(frozen=True)
class SnowcoverClasses:
    valid: list[int]
    cloud: list[int]
    water: list[int]
    nodata: list[int]


def _parse_int_list(vals: Sequence[object] | None, default: Sequence[int]) -> list[int]:
    out: list[int] = []
    for v in vals if vals is not None else default:
        try:
            out.append(int(v))
        except Exception:
            continue
    return out


def _load_classes(project_dir: Path) -> SnowcoverClasses:
    cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    obs_cfg = (cfg.get("obs") or {}).get("snowcover") or {}
    classes = obs_cfg.get("classes") or {}
    return SnowcoverClasses(
        valid=_parse_int_list(classes.get("valid"), default=list(range(0, 101))),
        cloud=_parse_int_list(classes.get("cloud"), default=[205]),
        water=_parse_int_list(classes.get("water"), default=[210]),
        nodata=_parse_int_list(classes.get("nodata"), default=[255]),
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
    raise ValueError(f"Could not infer date from {path.name}")


def _compute_stats(raster_path: Path, aoi_path: Path, region_field: str | None, lc_cfg: LandcoverMaskConfig, classes: SnowcoverClasses):
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster {raster_path} has no CRS; cannot align AOI/land cover")
        gdf, region_id = read_single_roi(aoi_path, required_field=region_field, to_crs=src.crs)
        data, transform = rio_mask(src, gdf.geometry, crop=True, nodata=src.nodata, filled=False)
        roi_mask = features.geometry_mask(gdf.geometry, out_shape=data.shape[1:], transform=transform, invert=True)
        arr = np.ma.array(data[0], copy=False)
        arr, _ = apply_landcover_mask(arr, transform=transform, target_crs=src.crs, roi_mask=roi_mask, lc_cfg=lc_cfg)
        nodata = src.nodata

    data = np.ma.getdata(arr)
    mask = np.ma.getmaskarray(arr)
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

    clouds = (~mask) & np.isin(data, classes.cloud) if classes.cloud else np.zeros_like(valid, dtype=bool)
    n_cloud = int(np.count_nonzero(clouds))
    denom = n_valid + n_cloud
    cloud_fraction = (n_cloud / denom) if denom > 0 else 0.0

    return {
        "date": _extract_date(raster_path).strftime("%Y-%m-%d"),
        "region_id": region_id,
        "n_valid": n_valid,
        "n_snow": n_snow,
        "scf": scf,
        "cloud_fraction": cloud_fraction,
        "source": raster_path.name,
    }


def summarize_snowcover_directory(
    *,
    project_dir: Path,
    input_dir: Path,
    aoi: Path,
    season_label: str,
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
    rasters: list[Path] = []
    for patt in patterns:
        rasters.extend(sorted(input_dir.rglob(patt) if recursive else input_dir.glob(patt)))
    if start or end:
        filtered: list[Path] = []
        for rast in rasters:
            try:
                d = _extract_date(rast)
            except Exception:
                continue
            if start and d < start:
                continue
            if end and d > end:
                continue
            filtered.append(rast)
        rasters = filtered

    if not rasters:
        logger.warning("No snow-cover rasters found in {}", input_dir)
        return []

    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(project_dir))
    cls = classes or _load_classes(project_dir)
    season_dir = output_root / season_label
    season_dir.mkdir(parents=True, exist_ok=True)
    summary_path = season_dir / "scf_summary.csv"

    rows: list[dict[str, object]] = []
    written: list[Path] = []
    for rast in rasters:
        try:
            stats = _compute_stats(rast, aoi, region_field, lc_cfg, cls)
        except Exception as exc:
            logger.error("Skipping {}: {}", rast.name, exc)
            continue
        if stats is None:
            logger.warning("Discarded {} because AOI contained no valid pixels", rast.name)
            continue
        rows.append(stats)
        written.append(rast)
        logger.info("Snowcover {} -> scf={:.3f} n_valid={} n_snow={}", rast.name, stats["scf"], stats["n_valid"], stats["n_snow"])

    if not rows:
        logger.warning("No valid snow-cover rasters processed.")
        return []

    df = pd.DataFrame(rows).sort_values("date")
    df.to_csv(summary_path, index=False)
    logger.info("Snow-cover summary written: {} ({} raster(s))", summary_path, len(df))
    return written


def _resolve_season_dates(project_dir: Path, season_label: str) -> dict[str, datetime] | None:
    season_yml = project_dir / "propagation" / season_label / "season.yml"
    if not season_yml.exists():
        return None
    try:
        data = yaml.safe_load(season_yml.read_text())
        start = datetime.fromisoformat(str(data.get("start_date")))
        end = datetime.fromisoformat(str(data.get("end_date")))
        return {"start": start, "end": end}
    except Exception:
        return None


def cli_main(argv: List[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-snowcover",
        description="Summarize snow-cover rasters (GeoTIFF/NetCDF) into scf_summary.csv with configurable classes.",
    )
    p.add_argument("--input-dir", required=True, type=Path, help="Directory containing snow-cover rasters (.tif/.tiff/.nc)")
    p.add_argument("--roi", dest="roi", type=Path, help="Single-feature ROI vector (default: <project>/env/roi.gpkg)")
    p.add_argument("--season-label", required=True, help="Season folder name under obs/")
    p.add_argument("--project-dir", type=Path, help="Project directory (default: CWD)")
    p.add_argument("--output-root", type=Path, help="Override output root (default: <project>/obs)")
    p.add_argument("--roi-field", dest="roi_field", default=None, help="Field name in ROI with the region identifier (optional)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for rasters")
    p.add_argument("--start-date", type=str, help="Optional ISO start date filter")
    p.add_argument("--end-date", type=str, help="Optional ISO end date filter")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing scf_summary.csv")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
    output_root = Path(args.output_root) if args.output_root else project_dir / OBS_DIR_NAME
    lc_cfg = resolve_landcover_mask(project_dir)
    cls = _load_classes(project_dir)
    season_dates = _resolve_season_dates(project_dir, args.season_label)

    start = parse_datetime_opt(args.start_date) if args.start_date else None
    end = parse_datetime_opt(args.end_date) if args.end_date else None
    if season_dates:
        start = start or season_dates.get("start")
        end = end or season_dates.get("end")

    try:
        aoi_path = Path(args.roi) if args.roi else project_dir / "env" / "roi.gpkg"
        summarize_snowcover_directory(
            project_dir=project_dir,
            input_dir=Path(args.input_dir),
            aoi=aoi_path,
            season_label=str(args.season_label),
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
