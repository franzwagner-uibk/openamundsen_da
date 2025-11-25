"""Helpers for converting season-wide fraction summaries into per-step obs CSVs.

Both MODIS SCF and Sentinel-1 wet-snow summaries share the same pattern:

- A season-level summary CSV in ``obs/<season>/`` with one row per date.
- Per-step observation CSVs in ``step_XX_*/obs`` that contain one row for the
  assimilation date of that step.

This module provides small utilities that satellite-specific observers can use
to avoid duplicating CSV handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import OBS_DIR_NAME
from openamundsen_da.io.paths import read_step_config


@dataclass(frozen=True)
class SummaryIndex:
    by_date: Dict[datetime, pd.Series]


def read_fraction_summary(summary_csv: Path, *, date_col: str = "date") -> SummaryIndex:
    """Read a season-level summary CSV and index rows by date."""

    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv, parse_dates=[date_col])
    by_date: Dict[datetime, pd.Series] = {}
    for _, row in df.iterrows():
        datum = row[date_col]
        if not pd.notna(datum):
            continue
        by_date[datum.to_pydatetime().date()] = row
    return SummaryIndex(by_date=by_date)


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


def list_steps_sorted(season_dir: Path) -> List[Path]:
    """Return step_* directories sorted by their start_date."""
    items: List[tuple[datetime, Path]] = []
    for p in sorted(season_dir.glob("step_*")):
        if not p.is_dir():
            continue
        cfg = read_step_config(p) or {}
        start = _parse_dt_opt(str(cfg.get("start_date")))
        items.append((start or datetime.min, p))
    items.sort(key=lambda t: (t[0], t[1].name))
    return [p for _, p in items]


def write_obs_from_summary_row(
    *,
    step_dir: Path,
    date: datetime,
    row: Mapping[str, object],
    value_col: str,
    product: str,
    variable: str,
    overwrite: bool,
) -> Path:
    """Write a one-row obs CSV for a given date and summary row."""

    out_dir = step_dir / OBS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"obs_{variable}_{product}_{date.strftime('%Y%m%d')}.csv"
    if out_csv.exists() and not overwrite:
        logger.info("Skipping existing obs CSV for {} (step {})", date.strftime("%Y-%m-%d"), step_dir.name)
        return out_csv

    payload: Dict[str, object] = {}
    for col, val in row.items():
        if pd.isna(val):
            continue
        payload[col] = val
    payload["date"] = date.strftime("%Y-%m-%d")

    # Ensure the primary value column is present under its variable-specific name.
    if value_col in row:
        payload[value_col] = row[value_col]

    df = pd.DataFrame({k: [v] for k, v in payload.items()})
    df.to_csv(out_csv, index=False)
    logger.info("Wrote obs {} -> {} ({})", date.strftime("%Y-%m-%d"), step_dir.name, out_csv.name)
    return out_csv


