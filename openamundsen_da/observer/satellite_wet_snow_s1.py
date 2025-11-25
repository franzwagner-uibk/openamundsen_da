"""Season helper for wet-snow observations from Sentinel-1 WSM summaries.

Purpose
-------
- Take the season-level ``wet_snow_summary.csv`` produced by
  ``openamundsen_da.methods.wet_snow.area.summarize_s1_directory`` and
  write one-row ``obs_wet_snow_S1_YYYYMMDD.csv`` files into each step's
  ``obs/`` folder for the dates that are actually assimilated by the
  season pipeline.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.observer.fraction_obs import (
    list_steps_sorted,
    read_fraction_summary,
    write_obs_from_summary_row,
)
from openamundsen_da.io.paths import read_step_config


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


def generate_season_from_summary(
    season_dir: Path,
    summary_csv: Path,
    *,
    overwrite: bool,
) -> None:
    """Extract per-step obs CSVs from a season-wide ``wet_snow_summary.csv``."""

    if not season_dir.is_dir():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    summary = read_fraction_summary(summary_csv, date_col="date")

    steps = list_steps_sorted(season_dir)
    if len(steps) < 2:
        raise FileNotFoundError(f"Not enough steps to derive assimilation dates under {season_dir}")

    written = skipped_missing = skipped_existing = 0
    for i in range(len(steps) - 1):
        curr_cfg = read_step_config(steps[i]) or {}
        end_dt = _parse_dt_opt(str(curr_cfg.get("end_date")))
        if end_dt is None:
            logger.warning("Skipping step {} (missing end_date)", steps[i].name)
            continue

        row = summary.by_date.get(end_dt.date())
        if row is None:
            logger.warning("No wet-snow summary entry for assimilation date {}; skipping {}", end_dt.date(), steps[i].name)
            skipped_missing += 1
            continue

        out_csv = write_obs_from_summary_row(
            step_dir=steps[i],
            date=end_dt,
            row=row,
            value_col="wet_snow_fraction",
            product="S1",
            variable="wet_snow",
            overwrite=overwrite,
        )
        written += 1
        logger.info("Wrote wet-snow obs {} -> {} ({})", end_dt.strftime("%Y-%m-%d"), steps[i].name, out_csv.name)

    logger.info(
        "Wet-snow season summary prep complete: written={} skipped_missing={} skipped_existing={}",
        written,
        skipped_missing,
        skipped_existing,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: fill per-step obs CSVs from wet_snow_summary.csv for a season."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-wet-snow-s1-season",
        description=(
            "Copy wet-snow rows from wet_snow_summary.csv into per-step "
            "obs_wet_snow_S1_YYYYMMDD.csv files for a season."
        ),
    )
    parser.add_argument("--season-dir", required=True, type=Path, help="Season directory (propagation/season_YYYY-YYYY)")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to wet_snow_summary.csv (default: <project>/obs/<season>/wet_snow_summary.csv)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing obs_wet_snow_*.csv files")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    season_dir = args.season_dir
    if args.summary_csv is not None:
        summary_path = args.summary_csv
    else:
        project_root = season_dir.parent.parent
        summary_path = project_root / "obs" / season_dir.name / "wet_snow_summary.csv"

    try:
        generate_season_from_summary(
            season_dir=season_dir,
            summary_csv=summary_path,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        logger.error("Wet-snow season summary prep failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

