"""Generic snow-cover summarization with configurable classes."""

from __future__ import annotations

import re
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

from openamundsen_da.core.constants import OBS_DIR_NAME
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.landcover_mask import LandcoverMaskConfig, apply_landcover_mask, resolve_landcover_mask
from openamundsen_da.util.project_dates import resolve_project_dates
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector
from openamundsen_da.util.ts import parse_datetime_opt


@dataclass(frozen=True)
class SnowcoverClasses:
    valid: list[int]
    cloud: list[int]
    water: list[int]
    nodata: list[int]


def _require_mapping(raw: object, *, path: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


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
    cfg = _require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = _require_mapping(cfg.get("obs"), path="project.obs")
    snow_cfg = _require_mapping(obs_cfg.get("snowcover"), path="project.obs.snowcover")
    classes = _require_mapping(snow_cfg.get("classes"), path="project.obs.snowcover.classes")
    return SnowcoverClasses(
        valid=_require_int_list(classes, "valid", path="project.obs.snowcover.classes"),
        cloud=_require_int_list(classes, "cloud", path="project.obs.snowcover.classes", allow_empty=True),
        water=_require_int_list(classes, "water", path="project.obs.snowcover.classes", allow_empty=True),
        nodata=_require_int_list(classes, "nodata", path="project.obs.snowcover.classes", allow_empty=True),
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
        "tile": _extract_tile(raster_path),
        "n_valid": n_valid,
        "n_snow": n_snow,
        "n_cloud": n_cloud,
        "scf": scf,
        "cloud_fraction": cloud_fraction,
        "source": raster_path.name,
    }


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

    project_dir = Path(setup_dir) / "projects" / str(project_label)
    lc_cfg = landcover_cfg or resolve_landcover_mask(Path(setup_dir), project_dir)
    cls = classes or _load_classes(project_dir)
    output_dir = output_root / project_label
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "scf_summary.csv"

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

    # Keep one best raster per date/tile and aggregate tile contributions to one row per date.
    # "Best" means highest valid pixel count; tie-breaker is lower cloud fraction.
    best_per_date_tile: dict[tuple[str, str], dict[str, object]] = {}
    for row in rows:
        key = (str(row["date"]), str(row.get("tile", "UNKNOWN")))
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

    agg: dict[tuple[str, str], dict[str, object]] = {}
    for row in best_per_date_tile.values():
        key = (str(row["date"]), str(row["region_id"]))
        slot = agg.get(key)
        if slot is None:
            slot = {
                "date": row["date"],
                "region_id": row["region_id"],
                "n_valid": 0,
                "n_snow": 0,
                "n_cloud": 0,
                "source_set": set(),
                "tile_set": set(),
            }
            agg[key] = slot
        slot["n_valid"] = int(slot["n_valid"]) + int(row.get("n_valid", 0))
        slot["n_snow"] = int(slot["n_snow"]) + int(row.get("n_snow", 0))
        slot["n_cloud"] = int(slot["n_cloud"]) + int(row.get("n_cloud", 0))
        slot["source_set"].add(str(row.get("source", "")))
        slot["tile_set"].add(str(row.get("tile", "UNKNOWN")))

    out_rows: list[dict[str, object]] = []
    for entry in agg.values():
        n_valid = int(entry["n_valid"])
        n_snow = int(entry["n_snow"])
        n_cloud = int(entry["n_cloud"])
        scf = (n_snow / n_valid) if n_valid > 0 else 0.0
        denom = n_valid + n_cloud
        cloud_fraction = (n_cloud / denom) if denom > 0 else 0.0
        out_rows.append(
            {
                "date": entry["date"],
                "region_id": entry["region_id"],
                "n_valid": n_valid,
                "n_snow": n_snow,
                "n_cloud": n_cloud,
                "scf": scf,
                "cloud_fraction": cloud_fraction,
                "tiles_used": ";".join(sorted(x for x in entry["tile_set"] if x)),
                "source": ";".join(sorted(x for x in entry["source_set"] if x)),
            }
        )

    df = pd.DataFrame(out_rows).sort_values("date")
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


