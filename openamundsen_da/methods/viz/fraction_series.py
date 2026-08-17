"""Helpers for loading setup-level time series and result overview defaults."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from openamundsen_da.io.paths import (
    default_results_dir,
    list_member_dirs,
    list_step_dirs,
    project_result_overview_output_path,
    read_step_config,
)
from openamundsen_da.methods.pf.weights import load_prior_weights
from openamundsen_da.util.stats import (
    effective_sample_size,
    normalize_weights,
    weighted_mean,
    weighted_quantile,
    weighted_std,
)
from openamundsen_da.util.ts import concat_series, parse_time_column
from openamundsen_da.util.point_output import compact_point_members, load_compact_point_series


def parse_fraction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a date-like column to a canonical ``date`` column."""
    for col in ("date", "time", "datetime"):
        if col in df.columns:
            df = df.copy()
            df[col] = parse_time_column(df[col])
            return df.rename(columns={col: "date"})
    raise KeyError("No date/time column found")


def load_fraction_series(
    path: Path | None,
    value_col: str,
    *,
    preserve_missing_values: bool = False,
) -> pd.DataFrame | None:
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
    out = df[cols].copy()
    if not preserve_missing_values:
        out = out.dropna(subset=[value_col])
    return out.sort_values("date")


def _load_fraction_value_series(
    path: Path,
    value_col: str,
    *,
    preserve_missing_values: bool = False,
) -> pd.Series | None:
    """Load one stitched value column as a datetime-indexed series."""
    df = load_fraction_series(path, value_col, preserve_missing_values=preserve_missing_values)
    if df is None or df.empty:
        return None
    series = df.set_index("date")[value_col].sort_index()
    if not preserve_missing_values:
        series = series.dropna()
    if series.empty:
        return None
    return series


def default_result_overview_output(project_dir: Path, output: Path | None) -> Path:
    """Return the output path for the default setup-level result overview plot."""
    if output is not None:
        return output
    out_path = project_result_overview_output_path(project_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def load_open_loop_fraction_series(
    project_dir: Path,
    filename: str,
    value_col: str,
    *,
    preserve_missing_values: bool = False,
) -> pd.DataFrame | None:
    """Stitch open-loop point series across project steps into one DataFrame."""
    segments: list[pd.Series] = []
    for step in list_step_dirs(project_dir):
        series = _load_fraction_value_series(
            default_results_dir(step / "ensembles" / "prior" / "open_loop")
            / filename,
            value_col,
            preserve_missing_values=preserve_missing_values,
        )
        if series is not None:
            segments.append(series)
    if not segments:
        compact = load_compact_point_series(
            project_dir,
            point_filename=filename,
            member="open_loop",
            variable=value_col,
        )
        if compact is None:
            return None
        segments = [compact]
    stitched = concat_series(segments).sort_index()
    if not preserve_missing_values:
        stitched = stitched.dropna()
    if stitched.empty:
        return None
    return pd.DataFrame({"date": stitched.index, value_col: stitched.values})


def load_member_series(
    project_dir: Path,
    filename: str,
    value_col: str,
    *,
    preserve_missing_values: bool = False,
) -> list[pd.Series]:
    """Return per-member setup-wide series stitched across all project steps."""
    member_segments: dict[str, list[pd.Series]] = defaultdict(list)
    for step in list_step_dirs(project_dir):
        for member_dir in list_member_dirs(step / "ensembles", "prior"):
            series = _load_fraction_value_series(
                default_results_dir(member_dir) / filename,
                value_col,
                preserve_missing_values=preserve_missing_values,
            )
            if series is not None:
                member_segments[member_dir.name].append(series)

    stitched_members: list[pd.Series] = []
    if not member_segments:
        for member_name in compact_point_members(project_dir):
            series = load_compact_point_series(
                project_dir,
                point_filename=filename,
                member=member_name,
                variable=value_col,
            )
            if series is not None:
                member_segments[member_name].append(series)
    for member_name in sorted(member_segments):
        stitched = concat_series(member_segments[member_name]).sort_index()
        if not preserve_missing_values:
            stitched = stitched.dropna()
        if not stitched.empty:
            stitched_members.append(stitched)
    return stitched_members


def load_named_member_series(
    project_dir: Path,
    filename: str,
    value_col: str,
    *,
    preserve_missing_values: bool = False,
) -> dict[str, pd.Series]:
    """Return per-member setup-wide series stitched across all project steps."""
    member_segments: dict[str, list[pd.Series]] = defaultdict(list)
    for step in list_step_dirs(project_dir):
        for member_dir in list_member_dirs(step / "ensembles", "prior"):
            series = _load_fraction_value_series(
                default_results_dir(member_dir) / filename,
                value_col,
                preserve_missing_values=preserve_missing_values,
            )
            if series is not None:
                member_segments[member_dir.name].append(series)

    stitched_members: dict[str, pd.Series] = {}
    if not member_segments:
        for member_name in compact_point_members(project_dir):
            series = load_compact_point_series(
                project_dir,
                point_filename=filename,
                member=member_name,
                variable=value_col,
            )
            if series is not None:
                member_segments[member_name].append(series)
    for member_name in sorted(member_segments):
        stitched = concat_series(member_segments[member_name]).sort_index()
        if not preserve_missing_values:
            stitched = stitched.dropna()
        if not stitched.empty:
            stitched_members[member_name] = stitched
    return stitched_members


def load_weighted_member_envelope(
    project_dir: Path,
    filename: str,
    value_col: str,
    *,
    q_low: float = 0.0,
    q_high: float = 1.0,
    preserve_missing_values: bool = False,
    daily_mean: bool = False,
    series_transform: Callable[[pd.Series], pd.Series] | None = None,
) -> pd.DataFrame | None:
    """Build a stepwise envelope using each step's persistent PF prior ledger."""
    if not 0.0 <= q_low <= q_high <= 1.0:
        raise ValueError("Envelope quantiles must satisfy 0 <= q_low <= q_high <= 1")
    if not (Path(project_dir) / "steps").is_dir():
        return None
    segments: list[pd.DataFrame] = []
    for step in list_step_dirs(project_dir):
        members = list_member_dirs(step / "ensembles", "prior")
        if not members:
            continue
        member_ids = [member.name for member in members]
        ledger = load_prior_weights(step, member_ids).set_index("member_id")
        series_by_member: dict[str, pd.Series] = {}
        for member in members:
            series = _load_fraction_value_series(
                default_results_dir(member) / filename,
                value_col,
                preserve_missing_values=preserve_missing_values,
            )
            if series is None:
                series = load_compact_point_series(
                    project_dir,
                    point_filename=filename,
                    member=member.name,
                    variable=value_col,
                )
                if series is not None:
                    step_cfg = read_step_config(step) or {}
                    start = pd.to_datetime(step_cfg.get("start_date"), errors="coerce")
                    end = pd.to_datetime(step_cfg.get("end_date"), errors="coerce")
                    if pd.notna(start):
                        series = series.loc[series.index >= start]
                    if pd.notna(end):
                        series = series.loc[series.index <= end]
            if series is None:
                continue
            if daily_mean:
                series = series.resample("D").mean()
            if series_transform is not None:
                series = series_transform(series)
            series_by_member[member.name] = series
        if not series_by_member:
            continue
        aligned = pd.concat(series_by_member, axis=1, join="outer").sort_index()
        rows: list[dict[str, object]] = []
        for timestamp, row in aligned.iterrows():
            values = pd.to_numeric(row, errors="coerce")
            valid = values.notna()
            if not valid.any():
                if preserve_missing_values:
                    rows.append(
                        {
                            "date": pd.Timestamp(timestamp),
                            "value_mean": np.nan,
                            "value_std": np.nan,
                            "value_min": np.nan,
                            "value_max": np.nan,
                            "value_q_low": np.nan,
                            "value_q_high": np.nan,
                            "n": 0,
                            "ess": np.nan,
                            "weighting": "pf_prior_ledger",
                            "bounds_semantics": "materialized_member_range",
                        }
                    )
                continue
            ids = values.index[valid].astype(str).tolist()
            arr = values.loc[valid].to_numpy(dtype=float)
            weights = normalize_weights(ledger.loc[ids, "weight"].to_numpy(dtype=float))
            rows.append(
                {
                    "date": pd.Timestamp(timestamp),
                    "value_mean": weighted_mean(arr, weights=weights),
                    "value_std": weighted_std(arr, weights=weights),
                    "value_min": float(np.min(arr)),
                    "value_max": float(np.max(arr)),
                    "value_q_low": weighted_quantile(arr, q_low, weights=weights),
                    "value_q_high": weighted_quantile(arr, q_high, weights=weights),
                    "n": int(arr.size),
                    "ess": effective_sample_size(weights),
                    "weighting": "pf_prior_ledger",
                    "bounds_semantics": "materialized_member_range",
                }
            )
        if rows:
            segments.append(pd.DataFrame(rows))
    if not segments:
        return None
    result = pd.concat(segments, ignore_index=True).sort_values("date")
    result = result.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)
    return result


__all__ = [
    "default_result_overview_output",
    "load_fraction_series",
    "load_member_series",
    "load_named_member_series",
    "load_open_loop_fraction_series",
    "load_weighted_member_envelope",
    "parse_fraction_dates",
]
