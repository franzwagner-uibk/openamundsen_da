"""openamundsen_da.observer.satellite_scf

Season helper for SCF observations backed by ``scf_summary.csv``.

Purpose
-------
- Take the season-level ``scf_summary.csv`` produced by
  ``openamundsen_da.observer.mod10a1_preprocess`` and write one-row
  ``obs_scf_MOD10A1_YYYYMMDD.csv`` files into each step's ``obs/`` folder
  for the dates that are actually assimilated by the season pipeline.

Behavior
--------
- Reads all steps under ``<season_dir>/step_*`` and their YAML configs.
- For every step i that has a following step i+1, uses this step's
  ``end_date`` as the assimilation datetime (mirroring the season
  orchestrator) and its calendar date as the assimilation date.
- Looks up that date in ``scf_summary.csv`` and writes a single-row CSV
  into ``step_i/obs/obs_scf_MOD10A1_YYYYMMDD.csv``.
- Does not touch rasters; all SCF statistics come from the summary file.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import OBS_DIR_NAME, LOGURU_FORMAT
from openamundsen_da.io.paths import read_step_config
from openamundsen_da.observer.fraction_obs import (
    list_steps_sorted,
    read_fraction_summary,
    write_obs_from_summary_row,
)
from openamundsen_da.util.da_events import load_assimilation_events


def generate_season_from_summary(
    season_dir: Path,
    summary_csv: Path,
    *,
    product: str = "MOD10A1",
    overwrite: bool,
) -> None:
    """Extract per-step obs CSVs from a season-wide ``scf_summary.csv``."""

    if not season_dir.is_dir():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    summary = read_fraction_summary(summary_csv, date_col="date")

    steps = list_steps_sorted(season_dir)
    if len(steps) < 2:
        raise FileNotFoundError(f"Not enough steps to derive assimilation dates under {season_dir}")

    events = load_assimilation_events(season_dir)
    n = min(len(events), len(steps) - 1)
    if n < len(events):
        logger.warning("Only {} steps (excluding final) available for {} assimilation events; extra events will be ignored.", n, len(events))
    if n < len(steps) - 1:
        logger.warning("Only {} assimilation events available for {} steps; later steps will not receive obs CSVs.", len(events), len(steps) - 1)

    written = skipped_missing = skipped_existing = 0
    for i in range(n):
        ev = events[i]
        curr_cfg = read_step_config(steps[i]) or {}
        start_dt = _parse_dt_opt(str(curr_cfg.get("start_date")))
        end_dt = _parse_dt_opt(str(curr_cfg.get("end_date")))
        if start_dt and end_dt:
            if not (start_dt.date() <= ev.date <= end_dt.date()):
                logger.warning(
                    "Assimilation date {} is outside step {} window ({} .. {})",
                    ev.date,
                    steps[i].name,
                    start_dt.date(),
                    end_dt.date(),
                )

        row = summary.by_date.get(ev.date)
        if row is None:
            logger.warning("No summary entry for assimilation date {}; skipping {}", ev.date, steps[i].name)
            skipped_missing += 1
            continue

        prod_tag = str(product).upper()
        out_csv = steps[i] / OBS_DIR_NAME / f"obs_scf_{prod_tag}_{ev.date.strftime('%Y%m%d')}.csv"
        if out_csv.exists() and not overwrite:
            logger.info("Skipping existing obs CSV for {} (step {})", ev.date.strftime("%Y-%m-%d"), steps[i].name)
            skipped_existing += 1
            continue

        write_obs_from_summary_row(
            step_dir=steps[i],
            date=datetime.combine(ev.date, datetime.min.time()),
            row=row,
            value_col="scf",
            product=prod_tag,
            variable="scf",
            overwrite=overwrite,
        )
        written += 1

    logger.info(
        "Season summary prep complete: written={} skipped_missing={} skipped_existing={}",
        written,
        skipped_missing,
        skipped_existing,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: fill per-step obs CSVs from scf_summary.csv for a season."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-scf",
        description=(
            "Copy SCF rows from scf_summary.csv into per-step "
            "obs_scf_MOD10A1_YYYYMMDD.csv files for a season."
        ),
    )
    parser.add_argument("--season-dir", required=True, type=Path, help="Season directory (propagation/season_YYYY-YYYY)")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to scf_summary.csv (default: <project>/obs/<season>/scf_summary.csv)",
    )
    parser.add_argument("--product", default="MOD10A1", help="Product code to use in obs filename (default: MOD10A1)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing obs_scf_*.csv files")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    season_dir = args.season_dir
    if args.summary_csv is not None:
        summary_path = args.summary_csv
    else:
        project_root = season_dir.parent.parent
        summary_path = project_root / "obs" / season_dir.name / "scf_summary.csv"

    try:
        generate_season_from_summary(
            season_dir=season_dir,
            summary_csv=summary_path,
            product=str(args.product or "MOD10A1"),
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        logger.error("Season summary prep failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
