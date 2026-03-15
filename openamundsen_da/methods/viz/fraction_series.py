"""Helpers for loading fraction time series and default fraction plot paths."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from openamundsen_da.io.paths import list_step_dirs


def parse_fraction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a date-like column to a canonical ``date`` column."""
    for col in ("date", "time", "datetime"):
        if col in df.columns:
            df = df.copy()
            df[col] = pd.to_datetime(df[col])
            return df.rename(columns={col: "date"})
    raise KeyError("No date/time column found")


def load_fraction_series(path: Path | None, value_col: str) -> pd.DataFrame | None:
    """Load a fraction time series if the file exists and contains the value column."""
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or value_col not in df.columns:
        return None
    df = parse_fraction_dates(df)
    cols = ["date", value_col]
    for extra in ("value_min", "value_max", "n"):
        if extra in df.columns:
            cols.append(extra)
    return df[cols].dropna(subset=[value_col]).sort_values("date")


def default_fraction_obs_path(setup_dir: Path, setup_name: str, filename: str) -> Path:
    """Return the default obs summary path for one fraction summary CSV."""
    candidates = [
        setup_dir / "obs" / setup_name / filename,
        setup_dir / "obs" / "summaries" / setup_name / filename,
    ]
    if "-" in setup_name:
        candidates.append(setup_dir / "obs" / setup_name.replace("-", "_") / filename)
    elif "_" in setup_name:
        candidates.append(setup_dir / "obs" / setup_name.replace("_", "-") / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def default_fraction_plot_output(project_dir: Path, output: Path | None) -> Path:
    """Return the output path for the combined fraction time-series plot."""
    if output is not None:
        return output
    out_dir = project_dir / "plots" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fraction_timeseries.png"


def load_open_loop_fraction_series(project_dir: Path, filename: str, value_col: str) -> pd.DataFrame | None:
    """Stitch open-loop point series across project steps into one DataFrame."""
    frames: list[pd.DataFrame] = []
    for step in list_step_dirs(project_dir):
        df = load_fraction_series(step / "ensembles" / "prior" / "open_loop" / "results" / filename, value_col)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True).dropna(subset=[value_col])
    if out.empty:
        return None
    return out.sort_values("date")
