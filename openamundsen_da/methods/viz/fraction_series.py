"""Helpers for loading setup-level time series and result overview defaults."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from openamundsen_da.io.paths import list_member_dirs, list_step_dirs
from openamundsen_da.util.ts import concat_series


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


def _load_fraction_value_series(path: Path, value_col: str) -> pd.Series | None:
    """Load one stitched value column as a datetime-indexed series."""
    df = load_fraction_series(path, value_col)
    if df is None or df.empty:
        return None
    series = df.set_index("date")[value_col].dropna().sort_index()
    if series.empty:
        return None
    return series


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


def default_result_overview_output(project_dir: Path, output: Path | None) -> Path:
    """Return the output path for the default setup-level result overview plot."""
    if output is not None:
        return output
    out_dir = project_dir / "plots" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "result_overview.png"


def load_open_loop_fraction_series(project_dir: Path, filename: str, value_col: str) -> pd.DataFrame | None:
    """Stitch open-loop point series across project steps into one DataFrame."""
    segments: list[pd.Series] = []
    for step in list_step_dirs(project_dir):
        series = _load_fraction_value_series(
            step / "ensembles" / "prior" / "open_loop" / "results" / filename,
            value_col,
        )
        if series is not None:
            segments.append(series)
    if not segments:
        return None
    stitched = concat_series(segments).dropna().sort_index()
    if stitched.empty:
        return None
    return pd.DataFrame({"date": stitched.index, value_col: stitched.values})


def load_member_series(project_dir: Path, filename: str, value_col: str) -> list[pd.Series]:
    """Return per-member setup-wide series stitched across all project steps."""
    member_segments: dict[str, list[pd.Series]] = defaultdict(list)
    for step in list_step_dirs(project_dir):
        for member_dir in list_member_dirs(step / "ensembles", "prior"):
            series = _load_fraction_value_series(member_dir / "results" / filename, value_col)
            if series is not None:
                member_segments[member_dir.name].append(series)

    stitched_members: list[pd.Series] = []
    for member_name in sorted(member_segments):
        stitched = concat_series(member_segments[member_name]).dropna().sort_index()
        if not stitched.empty:
            stitched_members.append(stitched)
    return stitched_members
