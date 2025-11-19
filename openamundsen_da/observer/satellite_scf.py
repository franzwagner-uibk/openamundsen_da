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


def _list_steps_sorted(season_dir: Path) -> List[Path]:
    items: List[Tuple[datetime, Path]] = []
    for p in sorted(season_dir.glob("step_*")):
        if not p.is_dir():
            continue
        cfg = read_step_config(p) or {}
        start = _parse_dt_opt(str(cfg.get("start_date")))
        items.append((start or datetime.min, p))
    items.sort(key=lambda t: (t[0], t[1].name))
    return [p for _, p in items]


def _sanitize_summary_value(val: object) -> object | None:
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
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
    """Extract per-step obs CSVs from a season-wide ``scf_summary.csv``."""

    if not season_dir.is_dir():
        raise FileNotFoundError(f"Season directory not found: {season_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv, parse_dates=["date"])
    by_date: dict[datetime.date, pd.Series] = {}
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
        # Assimilation datetime = current step end_date (aligned with
        # season_skeleton and season orchestrator). We intentionally skip
        # the final step, which has no following assimilation.
        curr_cfg = read_step_config(steps[i]) or {}
        end_dt = _parse_dt_opt(str(curr_cfg.get("end_date")))
        if end_dt is None:
            logger.warning("Skipping step {} (missing end_date)", steps[i].name)
            continue

        row = by_date.get(end_dt.date())
        if row is None:
            logger.warning("No summary entry for assimilation date {}; skipping {}", end_dt.date(), steps[i].name)
            skipped_missing += 1
            continue

        out_dir = steps[i] / OBS_DIR_NAME
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"obs_scf_MOD10A1_{end_dt.strftime('%Y%m%d')}.csv"
        if out_csv.exists() and not overwrite:
            logger.info("Skipping existing obs CSV for {} (step {})", end_dt.strftime("%Y-%m-%d"), steps[i].name)
            skipped_existing += 1
            continue

        payload = {col: _sanitize_summary_value(row[col]) for col in row.index}
        payload["date"] = end_dt.strftime("%Y-%m-%d")
        out_df = pd.DataFrame({k: [v] for k, v in payload.items()})
        out_df.to_csv(out_csv, index=False)
        written += 1
        logger.info("Wrote summary obs {} -> {} ({})", end_dt.strftime("%Y-%m-%d"), steps[i].name, out_csv.name)

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
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        logger.error("Season summary prep failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
