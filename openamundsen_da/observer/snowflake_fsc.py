"""Snowflake FSC (fractional snow cover) summarization.

Purpose
-------
- Read Snowflake FSC GeoTIFFs (0..100 %) for a season, clip to a single-feature
  AOI, and compute season-level SCF statistics.
- Write/update ``scf_summary.csv`` under ``obs/<season_label>/`` with one row
  per date: date, region_id, n_valid, n_snow, scf, source.

Assumptions
-----------
- Raster values are 0..100 percent FSC; clouds/invalid pixels are marked as
  nodata. AOI CRS is reprojected to the raster CRS.
- Filenames contain the acquisition date as YYYY_MM_DD (e.g.,
  ``s2_fsc_snowflake_rofental_2019_10_01.tif``).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from rasterio.mask import mask as rio_mask

from openamundsen_da.core.constants import LOGURU_FORMAT, OBS_DIR_NAME
from openamundsen_da.util.aoi import read_single_aoi


_DATE_RE = re.compile(r"(\d{4})_(\d{2})_(\d{2})")


@dataclass(frozen=True)
class FscStats:
    date: datetime
    region_id: str
    n_valid: int
    n_snow: int
    scf: float
    source: str


def _extract_date(path: Path) -> datetime:
    """Parse YYYY_MM_DD from filename."""
    m = _DATE_RE.search(path.name)
    if not m:
        raise ValueError(f"Filename missing date (YYYY_MM_DD): {path.name}")
    y, mo, d = (int(m.group(i)) for i in range(1, 4))
    return datetime(year=y, month=mo, day=d)


def _compute_stats_for_raster(raster_path: Path, aoi_path: Path, region_field: str | None) -> Optional[FscStats]:
    """Mask raster to AOI and return SCF stats."""
    with rasterio.open(raster_path) as src:
        try:
            gdf, region_id = read_single_aoi(aoi_path, required_field=region_field, to_crs=src.crs)
        except KeyError:
            # AOI has no region_id field -> accept empty region_id
            gdf, region_id = read_single_aoi(aoi_path, required_field=None, to_crs=src.crs)
            region_id = ""
        data, _ = rio_mask(src, gdf.geometry, crop=True, nodata=src.nodata, filled=False)
        arr = np.ma.array(data[0], copy=False)
        nodata = src.nodata

    valid = (~arr.mask) & np.isfinite(arr)
    if nodata is not None:
        valid &= arr.data != nodata
    valid &= (arr.data >= 0) & (arr.data <= 100)

    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return None

    vals = arr.data[valid].astype(float) / 100.0
    scf = float(vals.mean())
    n_snow = int(round(scf * n_valid))

    return FscStats(
        date=_extract_date(raster_path),
        region_id=region_id,
        n_valid=n_valid,
        n_snow=n_snow,
        scf=scf,
        source=raster_path.name,
    )


def _list_rasters(root: Path, recursive: bool) -> list[Path]:
    files: Iterable[Path] = root.rglob("*.tif") if recursive else root.glob("*.tif")
    return sorted(p for p in files if p.is_file())


def _auto_aoi(project_dir: Path) -> Path:
    """Pick a single AOI from <project>/env if present."""
    env_dir = project_dir / "env"
    cands = list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp"))
    if not cands:
        raise FileNotFoundError(f"No AOI found under {env_dir}")
    if len(cands) > 1:
        raise ValueError(f"Multiple AOI candidates under {env_dir}; specify --aoi")
    return cands[0]


def _update_summary(summary_path: Path, stats: FscStats) -> None:
    """Append or replace one row in scf_summary.csv."""
    row = pd.DataFrame(
        {
            "date": [stats.date.strftime("%Y-%m-%d")],
            "region_id": [stats.region_id],
            "n_valid": [int(stats.n_valid)],
            "n_snow": [int(stats.n_snow)],
            "scf": [round(stats.scf, 3)],
            "source": [stats.source],
        }
    )

    if summary_path.exists():
        existing = pd.read_csv(summary_path)
        existing = existing[existing["date"] != row.loc[0, "date"]]
        row = pd.concat([existing, row], ignore_index=True)
        row = row.sort_values("date")

    row.to_csv(summary_path, index=False)


def summarize_snowflake_directory(
    *,
    input_dir: Path,
    aoi: Path,
    season_label: str,
    output_root: Path,
    region_field: str | None = "region_id",
    recursive: bool = False,
) -> list[Path]:
    """Summarize all FSC rasters under input_dir into scf_summary.csv."""
    rasters = _list_rasters(input_dir, recursive)
    if not rasters:
        logger.warning("No FSC rasters found in {}", input_dir)
        return []

    season_dir = output_root / season_label
    season_dir.mkdir(parents=True, exist_ok=True)
    summary_path = season_dir / "scf_summary.csv"

    written: list[Path] = []
    for rast in rasters:
        try:
            stats = _compute_stats_for_raster(rast, aoi, region_field)
        except Exception as exc:
            logger.error("Skipping {}: {}", rast.name, exc)
            continue
        if stats is None:
            logger.warning("Discarded {} because AOI contained no valid pixels", rast.name)
            continue
        try:
            _update_summary(summary_path, stats)
        except Exception as exc:
            logger.error("Failed updating summary for {}: {}", rast.name, exc)
            continue
        written.append(rast)
        logger.info(
            "FSC {} -> {} (scf={:.3f}, n_valid={}, n_snow={})",
            rast.name,
            summary_path.relative_to(output_root.parent) if output_root.parent in summary_path.parents else summary_path,
            stats.scf,
            stats.n_valid,
            stats.n_snow,
        )
    logger.info("Snowflake FSC summary complete: {} raster(s) processed", len(written))
    return written


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry: summarize Snowflake FSC rasters into scf_summary.csv."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-snowflake-fsc",
        description="Summarize Snowflake FSC GeoTIFFs (0..100 %) into scf_summary.csv for a season.",
    )
    p.add_argument("--input-dir", required=True, type=Path, help="Directory containing FSC GeoTIFFs (0..100 %)")
    p.add_argument("--aoi", type=Path, help="Single-feature AOI vector (auto-detected from <project>/env if omitted)")
    p.add_argument("--season-label", required=True, help="Season folder name under the obs root")
    p.add_argument("--project-dir", type=Path, help="Project directory (default: current working directory)")
    p.add_argument(
        "--output-root",
        type=Path,
        help="Override output root (default: <project-dir>/obs)",
    )
    p.add_argument("--aoi-field", default="region_id", help="Field name in AOI with the region identifier (ignored if missing)")
    p.add_argument("--recursive", action="store_true", help="Recurse into subdirectories for FSC rasters")
    p.add_argument("--log-level", default="INFO", help="Loguru log level (default: INFO)")

    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    project_dir = Path(args.project_dir) if args.project_dir else Path.cwd()
    output_root = Path(args.output_root) if args.output_root else project_dir / OBS_DIR_NAME

    try:
        aoi_path = Path(args.aoi) if args.aoi else _auto_aoi(project_dir)
        summarize_snowflake_directory(
            input_dir=Path(args.input_dir),
            aoi=aoi_path,
            season_label=str(args.season_label),
            output_root=output_root,
            region_field=str(args.aoi_field) if args.aoi_field else None,
            recursive=bool(args.recursive),
        )
        return 0
    except Exception as exc:
        logger.error("Snowflake FSC summarization failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
