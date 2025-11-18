"""openamundsen_da.observer.satellite_scf

Purpose
- Compute a single snow‑cover fraction (SCF) from a preprocessed MODIS/Terra
  MOD10A1 (C6/6.1) NDSI raster and a single‑feature AOI.

Behavior
- Reads the `NDSI_Snow_Cover` GeoTIFF (0..100 scale), masks by AOI polygon,
  validates CRS match, thresholds NDSI (default 40) to classify snow, and
  writes one CSV row: `date,region_id,scf`.

Notes
- Inputs are assumed pre‑screened (projection/QA/mosaics) for this minimal
  experimental implementation.
- One image + one region per run; batch/multi‑region can be added later.
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rio_mask
from loguru import logger

from openamundsen_da.core.constants import (
    OBS_DIR_NAME,
    SCF_BLOCK,
    SCF_NDSI_THRESHOLD,
    SCF_REGION_ID_FIELD,
)
from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import ensure_gdal_proj_from_conda
from openamundsen_da.io.paths import read_step_config
from openamundsen_da.util.aoi import read_single_aoi


def _extract_yyyymmdd(p: Path) -> datetime:
    """Extract first YYYYMMDD occurrence from filename and return a date.

    Raises ValueError if no valid date can be found or parsed.
    """
    m = re.search(r"(\d{8})", p.stem)
    if not m:
        raise ValueError(f"No YYYYMMDD found in raster name: {p.name}")
    s = m.group(1)
    try:
        return datetime.strptime(s, "%Y%m%d")
    except Exception as e:
        raise ValueError(f"Invalid YYYYMMDD in raster name: {p.name}") from e


def _read_step_config(step_dir: Path) -> dict:
    """Read step YAML using shared helper and return dict (or {})."""
    return read_step_config(step_dir)


def _parse_dt_opt(text: str | None) -> datetime | None:
    if not text:
        return None
    t = str(text).strip().replace("_", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def _find_step_for_date(season_dir: Path, dt: datetime) -> Path | None:
    """Return the step directory under season_dir whose window contains dt.

    Preference: start <= dt <= end. If none, prefer a step whose end == dt.
    Returns None if no suitable step windows could be parsed.
    """
    candidates: list[tuple[Path, datetime | None, datetime | None]] = []
    for p in sorted(season_dir.glob("step_*")):
        if not p.is_dir():
            continue
        cfg = _read_step_config(p) or {}
        sd = _parse_dt_opt(str(cfg.get("start_date")))
        ed = _parse_dt_opt(str(cfg.get("end_date")))
        candidates.append((p, sd, ed))
    # First pass: containment
    for p, sd, ed in candidates:
        if sd is not None and ed is not None and sd <= dt <= ed:
            return p
    # Second pass: exact end match
    for p, _sd, ed in candidates:
        if ed is not None and ed.date() == dt.date():
            return p
    return None


def _list_steps_sorted(season_dir: Path) -> List[Path]:
    items: List[Tuple[datetime, Path]] = []
    for p in sorted(season_dir.glob("step_*")):
        if not p.is_dir():
            continue
        cfg = _read_step_config(p) or {}
        start = _parse_dt_opt(str(cfg.get("start_date")))
        items.append((start or datetime.min, p))
    items.sort(key=lambda t: (t[0], t[1].name))
    return [p for _, p in items]


def _list_rasters(obs_dir: Path, glob_pattern: str) -> List[Tuple[datetime, Path]]:
    collected: List[Tuple[datetime, Path]] = []
    for raster in sorted(obs_dir.glob(glob_pattern)):
        if not raster.is_file():
            continue
        try:
            collected.append((_extract_yyyymmdd(raster), raster))
        except ValueError as exc:
            logger.warning("Skipping raster {}: {}", raster.name, exc)
    collected.sort(key=lambda pair: (pair[0], pair[1].name))
    return collected


def _step_config_overrides(step_dir: Path, default_ndsi: float, default_region: str) -> tuple[float, str]:
    cfg = _read_step_config(step_dir) or {}
    scf_cfg = (cfg or {}).get(SCF_BLOCK) or {}
    ndsi = float(scf_cfg.get(SCF_NDSI_THRESHOLD, default_ndsi))
    region_field = str(scf_cfg.get(SCF_REGION_ID_FIELD, default_region))
    return ndsi, region_field


def _compute_scf(arr: np.ma.MaskedArray, threshold: float) -> tuple[int, int, float]:
    """Compute N_valid, N_snow, SCF given a masked NDSI array and threshold.

    Only 0..100 values are considered valid; masked/NaN/out-of-range are invalid.
    Returns (N_valid, N_snow, scf).
    """
    data = np.ma.array(arr, copy=False)
    # valid: not masked, finite, and within expected NDSI range
    valid = (~data.mask) & np.isfinite(data) & (data >= 0) & (data <= 100)
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("AOI contains no valid NDSI pixels (N_valid=0)")
    snow = valid & (data > threshold)
    n_snow = int(snow.sum())
    scf = float(n_snow / n_valid)
    return n_valid, n_snow, scf


def run_observation_processing(
    input_raster: Path,
    region_path: Path,
    output_csv: Path,
    *,
    ndsi_threshold: float = 40.0,
    region_id_field: str = "region_id",
) -> None:
    """Compute single-region SCF from one MOD10A1 NDSI raster and write CSV.

    Parameters
    ----------
    input_raster : Path
        Path to a MODIS/Terra MOD10A1 Collection 6/6.1 raster exported to
        GeoTIFF containing band `NDSI_Snow_Cover`, scaled 0..100 with nodata.
    region_path : Path
        Path to a vector file containing a single AOI polygon with field
        `region_id` (or `region_id_field`).
    output_csv : Path
        Path to output CSV file; parent directories are created if needed.
    ndsi_threshold : float, optional
        NDSI threshold for snow classification (default 40.0).
    region_id_field : str, optional
        Name of the region identifier field in AOI (default 'region_id').
    """
    # Step 1: Ensure GDAL/PROJ lookup from active conda env (Windows safety)
    ensure_gdal_proj_from_conda()

    input_raster = Path(input_raster)
    region_path = Path(region_path)
    output_csv = Path(output_csv)

    # Step 3: Load AOI and enforce single feature + required field
    # AOI and region id
    # Need raster CRS for reprojecting AOI; open raster first to get src.crs

    # Step 4: Open raster and validate CRS alignment
    with rasterio.open(input_raster) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; unable to align with AOI")
        gdf, region_id = read_single_aoi(region_path, required_field=region_id_field, to_crs=src.crs)
        # Rasterize-mask by AOI geometry, preserve mask (masked array)
        shapes: Iterable = gdf.geometry
        data, _ = rio_mask(src, shapes, crop=True, nodata=src.nodata, filled=False)
        band = np.ma.array(data[0], copy=False)  # first band

    # Step 5: Compute SCF and date
    n_valid, n_snow, scf = _compute_scf(band, float(ndsi_threshold))
    dt = _extract_yyyymmdd(input_raster)

    # Step 6: Prepare output directory and write CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date": [dt.strftime("%Y-%m-%d")],
        "region_id": [region_id],
        "n_valid": [n_valid],
        "n_snow": [n_snow],
        "scf": [round(scf, 3)],
    })
    df.to_csv(output_csv, index=False)

    # Step 7: Single concise log line (ready for DA ingestion)
    logger.info(
        f"SCF | raster={input_raster.name} region={region_id} valid={n_valid} snow={n_snow} scf={scf:.2f} -> {output_csv.name}"
    )


def generate_season_observations(
    season_dir: Path,
    obs_dir: Path,
    aoi_path: Path,
    *,
    raster_glob: str,
    default_ndsi: float,
    default_region_field: str,
    overwrite: bool,
) -> None:
    """Process every raster under `obs_dir` and drop CSVs into matching steps."""

    if not season_dir.is_dir():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    if not obs_dir.is_dir():
        raise FileNotFoundError(f"Observation directory not found: {obs_dir}")

    rasters = _list_rasters(obs_dir, raster_glob)
    if not rasters:
        raise FileNotFoundError(f"No rasters matching '{raster_glob}' found under {obs_dir}")

    processed = skipped_existing = skipped_unmatched = skipped_invalid = 0
    for dt, raster in rasters:
        target_step = _find_step_for_date(season_dir, dt)
        if target_step is None:
            logger.warning("No step window covers {}; skipping {}", dt.strftime("%Y-%m-%d"), raster.name)
            skipped_unmatched += 1
            continue

        ndsi, region_field = _step_config_overrides(target_step, default_ndsi, default_region_field)
        out_csv = target_step / OBS_DIR_NAME / f"obs_scf_MOD10A1_{dt.strftime('%Y%m%d')}.csv"
        if out_csv.exists() and not overwrite:
            logger.info("Skipping existing obs CSV for {} (step {})", dt.strftime("%Y-%m-%d"), target_step.name)
            skipped_existing += 1
            continue

        logger.info("Processing {} into {} ({})", raster.name, target_step.name, out_csv.name)
        try:
            run_observation_processing(
                input_raster=raster,
                region_path=aoi_path,
                output_csv=out_csv,
                ndsi_threshold=ndsi,
                region_id_field=region_field,
            )
        except ValueError as exc:
            if "N_valid=0" in str(exc):
                logger.warning("Skipping {}: {}", raster.name, exc)
                skipped_invalid += 1
                continue
            raise
        processed += 1

    logger.info(
        "Season SCF prep complete: processed={} skipped_existing={} skipped_unmatched={} skipped_invalid={}",
        processed,
        skipped_existing,
        skipped_unmatched,
        skipped_invalid,
    )


def _sanitize_summary_value(val: object) -> Union[str, float, int, None]:
    if val is None:
        return None
    if pd.isna(val):
        return None
    return val


def generate_season_from_summary(
    season_dir: Path,
    summary_csv: Path,
    *,
    overwrite: bool,
) -> None:
    """Extract per-step obs CSVs from a season-wide `scf_summary.csv`."""

    if not season_dir.is_dir():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv, parse_dates=["date"])
    by_date = {}
    for _, row in df.iterrows():
        datum = row["date"]
        if not pd.notna(datum):
            continue
        by_date[datum.date()] = row

    steps = _list_steps_sorted(season_dir)
    if len(steps) < 2:
        raise FileNotFoundError(f"Not enough steps to derive assimilation dates under {season_dir}")

    written = skipped_missing = skipped_existing = 0
    for i in range(len(steps) - 1):
        next_cfg = read_step_config(steps[i + 1]) or {}
        start = _parse_dt_opt(str(next_cfg.get("start_date")))
        if start is None:
            logger.warning("Skipping step {} (missing start_date)", steps[i + 1].name)
            continue

        row = by_date.get(start.date())
        if row is None:
            logger.warning("No summary entry for assimilation date {}; skipping {}", start.date(), steps[i].name)
            skipped_missing += 1
            continue

        out_dir = steps[i] / OBS_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"obs_scf_MOD10A1_{start.strftime('%Y%m%d')}.csv"
        if out_csv.exists() and not overwrite:
            logger.info("Skipping existing obs CSV for {} (step {})", start.strftime("%Y-%m-%d"), steps[i].name)
            skipped_existing += 1
            continue

        payload = {col: _sanitize_summary_value(row[col]) for col in row.index}
        payload["date"] = start.strftime("%Y-%m-%d")
        out_df = pd.DataFrame({k: [v] for k, v in payload.items()})
        out_df.to_csv(out_csv, index=False)
        written += 1
        logger.info("Wrote summary obs {} -> {} ({})", start.strftime("%Y-%m-%d"), steps[i].name, out_csv.name)

    logger.info(
        "Season summary prep complete: written={} skipped_missing={} skipped_existing={}",
        written,
        skipped_missing,
        skipped_existing,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry for single MODIS/Terra MOD10A1 SCF extraction or season batch mode."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-scf",
        description=(
            "Compute SCF from one MODIS/Terra MOD10A1 Collection 6/6.1 NDSI_Snow_Cover "
            "GeoTIFF and one AOI polygon, or run season mode to process every raster."
        ),
    )
    parser.add_argument("--season-dir", type=Path, help="Season directory (propagation/season_YYYY-YYYY) for batch mode")
    parser.add_argument("--obs-dir", type=Path, help="Preprocessed MOD10A1 season folder (default: <project>/obs/<season>)")
    parser.add_argument("--raster-glob", default="*.tif", help="Glob for season rasters (default: *.tif)")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing obs CSVs when running season mode")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to scf_summary.csv (default: <project>/obs/<season>/scf_summary.csv). Enables the summary-only season mode.",
    )
    parser.add_argument(
        "--raster",
        type=Path,
        help="Path to MOD10A1 C6/6.1 NDSI_Snow_Cover GeoTIFF (0..100) for single-run mode",
    )
    parser.add_argument(
        "--region",
        type=Path,
        help="Path to AOI vector (single feature) with field 'region_id'",
    )
    parser.add_argument("--step-dir", type=Path, help="Step directory to place CSV under 'obs' and read config")
    parser.add_argument("--output", type=Path, help="Explicit output CSV path (overrides --step-dir)")
    parser.add_argument("--ndsi-threshold", type=float, help="Override NDSI threshold (default 40)")
    parser.add_argument("--region-field", default="region_id", help="AOI field containing the region identifier")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    if args.season_dir:
        if args.summary_csv:
            if args.raster:
                parser.error("--raster cannot be combined with --season-dir and --summary-csv")
            if args.step_dir:
                parser.error("--step-dir cannot be combined with --season-dir and --summary-csv")
            if args.output:
                parser.error("--output cannot be combined with --season-dir and --summary-csv")

            season_dir = args.season_dir
            project_root = season_dir.parent.parent
            summary_path = args.summary_csv or (project_root / "obs" / season_dir.name / "scf_summary.csv")

            try:
                generate_season_from_summary(
                    season_dir=season_dir,
                    summary_csv=summary_path,
                    overwrite=args.overwrite,
                )
                return 0
            except Exception as exc:
                logger.error("Season summary prep failed: {}", exc)
                return 1

        if not args.region:
            parser.error("--region is required when running season mode without --summary-csv")

        if args.raster:
            parser.error("--raster cannot be combined with --season-dir")
        if args.step_dir:
            parser.error("--step-dir cannot be combined with --season-dir")
        if args.output:
            parser.error("--output cannot be combined with --season-dir")

        season_dir = args.season_dir
        project_root = season_dir.parent.parent
        obs_dir = args.obs_dir or (project_root / "obs" / season_dir.name)
        default_ndsi = args.ndsi_threshold if args.ndsi_threshold is not None else 40.0

        try:
            generate_season_observations(
                season_dir=season_dir,
                obs_dir=obs_dir,
                aoi_path=args.region,
                raster_glob=args.raster_glob,
                default_ndsi=default_ndsi,
                default_region_field=args.region_field,
                overwrite=args.overwrite,
            )
            return 0
        except Exception as exc:
            logger.error("Season SCF prep failed: {}", exc)
            return 1

    if args.raster is None:
        parser.error("--raster is required when --season-dir is not provided")

    if not args.region:
        parser.error("--region is required when --season-dir is not provided")

    ndsi_thr = args.ndsi_threshold if args.ndsi_threshold is not None else 40.0
    region_field = args.region_field
    out_csv: Path | None = args.output

    if args.step_dir and out_csv is None:
        dt = _extract_yyyymmdd(Path(args.raster))
        step_dir = Path(args.step_dir)
        target_step = None
        if any(step_dir.glob("step_*")):
            target_step = _find_step_for_date(step_dir, dt)
        else:
            if step_dir.name.endswith("_init") and step_dir.parent.is_dir():
                target_step = _find_step_for_date(step_dir.parent, dt)
        if target_step is None:
            target_step = step_dir
        if target_step != step_dir:
            logger.info("Routing SCF output to '{}' for date {} (was given '{}')", target_step.name, dt.strftime("%Y-%m-%d"), step_dir.name)

        out_dir = target_step / OBS_DIR_NAME
        out_csv = out_dir / f"obs_scf_MOD10A1_{dt.strftime('%Y%m%d')}.csv"
        ndsi_thr, region_field = _step_config_overrides(target_step, ndsi_thr, region_field)

    if out_csv is None:
        parser.error("Either --step-dir or --output must be provided")

    try:
        run_observation_processing(
            input_raster=Path(args.raster),
            region_path=Path(args.region),
            output_csv=out_csv,
            ndsi_threshold=ndsi_thr,
            region_id_field=region_field,
        )
        return 0
    except Exception as exc:
        logger.error("SCF processing failed: {}", exc)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())


