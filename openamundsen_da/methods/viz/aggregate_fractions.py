"""
aggregate_fractions.py
Author: openamundsen_da
Date: 2025-12-02
Description:
    Aggregate member-level fraction time series into a season-wide envelope.

    Looks for member CSVs (e.g., point_scf_aoi.csv or point_wet_snow_aoi.csv)
    under <season>/step_*/ensembles/prior/*/results and writes an envelope CSV
    with date, value_mean, value_min, value_max, n.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT


def _find_series_files(season_dir: Path, filename: str) -> list[Path]:
    """Return all matching member result CSVs for a season."""
    pattern = f"step_*/ensembles/prior/*/results/{filename}"
    return sorted(p for p in season_dir.glob(pattern) if p.is_file())


def _load_series(path: Path, value_col: str) -> pd.DataFrame | None:
    """Read one member CSV and return normalized date/value columns."""
    df = pd.read_csv(path)
    if df.empty or value_col not in df.columns:
        return None
    for col in ("date", "time", "datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.rename(columns={col: "date"})
            break
    else:
        return None
    out = df[["date", value_col]].dropna()
    if out.empty:
        return None
    out["date"] = out["date"].dt.normalize()
    return out


def aggregate_fraction_envelope(
    *,
    season_dir: Path,
    filename: str,
    value_col: str,
    output_name: str,
) -> Path | None:
    """
    Aggregate per-member fraction CSVs into a season-level envelope CSV.

    Parameters
    ----------
    season_dir : Path
        Season directory containing step_* folders.
    filename : str
        Member CSV filename to collect (e.g., point_scf_aoi.csv).
    value_col : str
        Value column to aggregate (e.g., scf or wet_snow_fraction).
    output_name : str
        Output CSV name written into the season directory.

    Returns
    -------
    Path or None
        Path to the written envelope CSV, or None if no inputs found.
    """
    files = _find_series_files(season_dir, filename)
    if not files:
        logger.warning("No member series found for {} under {}", filename, season_dir)
        return None

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            df = _load_series(f, value_col)
        except Exception as exc:
            logger.warning("Skipping {}: {}", f, exc)
            continue
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        logger.warning("No usable rows found for {} under {}", filename, season_dir)
        return None

    all_df = pd.concat(frames, ignore_index=True)
    grp = all_df.groupby("date")[value_col]
    out = pd.DataFrame(
        {
            "date": grp.mean().index,
            "value_mean": grp.mean().values,
            "value_min": grp.min().values,
            "value_max": grp.max().values,
            "n": grp.count().values,
        }
    ).sort_values("date")

    out_path = Path(season_dir) / output_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Wrote envelope CSV ({} rows) -> {}", len(out), out_path)
    return out_path


def cli_main(argv: list[str] | None = None) -> int:
    """CLI to aggregate SCF or wet-snow envelopes."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-aggregate-fractions",
        description="Aggregate member fraction time series into a season-level envelope CSV.",
    )
    p.add_argument("--season-dir", required=True, type=Path, help="Season directory (propagation/season_YYYY-YYYY)")
    p.add_argument("--filename", required=True, help="Member CSV filename to collect (e.g., point_scf_aoi.csv)")
    p.add_argument("--value-col", required=True, help="Value column to aggregate (e.g., scf)")
    p.add_argument("--output-name", required=True, help="Output CSV name (e.g., point_scf_aoi_envelope.csv)")
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    try:
        aggregate_fraction_envelope(
            season_dir=Path(args.season_dir),
            filename=str(args.filename),
            value_col=str(args.value_col),
            output_name=str(args.output_name),
        )
    except Exception as exc:
        logger.error("Envelope aggregation failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
