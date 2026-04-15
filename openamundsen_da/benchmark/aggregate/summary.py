"""Benchmark aggregation, reliability diagnostics, and derived fields."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from openamundsen_da.io.paths import list_steps_sorted
from openamundsen_da.methods.viz._ensemble_meta import load_stations_table_from_steps
from openamundsen_da.util.station_da import (
    load_station_assimilation_config,
    read_station_metadata,
    resolve_station_uncertainty_pct,
)


_COVERAGE_LEVELS = (50, 80, 90)
_PIT_BIN_COUNT = 10


def _finite_mean(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.nan
    return float(valid.mean())


def _load_station_metadata(project_dir: Path, setup_dir: Path) -> pd.DataFrame | None:
    candidates = [setup_dir / "meteo" / "stations.csv"]
    for candidate in candidates:
        if candidate.is_file():
            try:
                return pd.read_csv(candidate)
            except Exception:
                pass
    try:
        return load_stations_table_from_steps(list_steps_sorted(project_dir), "prior")
    except Exception:
        return None


def _normalize_station_metadata(stations_df: pd.DataFrame | None) -> pd.DataFrame:
    if stations_df is None or stations_df.empty:
        return pd.DataFrame(columns=["obs_id", "station_name", "station_elevation_m", "station_x", "station_y"])
    df = stations_df.copy()
    cols_lower = {c.lower().strip(): c for c in df.columns}
    id_candidates = [c for c in ("id", "station_id", "station", "code") if c in cols_lower]
    name_candidates = [c for c in ("name", "station_name") if c in cols_lower]
    elev_candidates = [c for c in ("alt", "altitude", "elev", "elevation", "z", "height", "height_m") if c in cols_lower]
    x_candidates = [c for c in ("x", "lon", "longitude", "east", "easting") if c in cols_lower]
    y_candidates = [c for c in ("y", "lat", "latitude", "north", "northing") if c in cols_lower]
    id_col = cols_lower[id_candidates[0]] if id_candidates else None
    if id_col is None:
        return pd.DataFrame(columns=["obs_id", "station_name", "station_elevation_m", "station_x", "station_y"])
    name_col = cols_lower[name_candidates[0]] if name_candidates else None
    elev_col = cols_lower[elev_candidates[0]] if elev_candidates else None
    x_col = cols_lower[x_candidates[0]] if x_candidates else None
    y_col = cols_lower[y_candidates[0]] if y_candidates else None
    normalized = pd.DataFrame(
        {
            "obs_id": df[id_col].astype(str).str.strip().str.lower(),
            "station_name": df[name_col].astype(str).str.strip() if name_col is not None else pd.Series(index=df.index, dtype=object),
            "station_elevation_m": pd.to_numeric(df[elev_col], errors="coerce") if elev_col is not None else np.nan,
            "station_x": pd.to_numeric(df[x_col], errors="coerce") if x_col is not None else np.nan,
            "station_y": pd.to_numeric(df[y_col], errors="coerce") if y_col is not None else np.nan,
        }
    )
    return normalized.drop_duplicates(subset=["obs_id"]).reset_index(drop=True)


def _station_uncertainty_lookup(
    *,
    case_scores: pd.DataFrame,
    project_dir: Path,
    setup_dir: Path,
    strict: bool,
) -> dict[str, float]:
    station_ids = sorted(
        {
            str(obs_id).strip().lower()
            for obs_id in case_scores.loc[case_scores["obs_kind"] == "station", "obs_id"].tolist()
            if str(obs_id).strip()
        }
    )
    if not station_ids:
        return {}

    try:
        config = load_station_assimilation_config(setup_dir, project_dir)
        metadata_df = read_station_metadata(config.metadata_path)
    except Exception:
        if strict:
            raise
        return {}

    lookup: dict[str, float] = {}
    for station_id in station_ids:
        try:
            station_uncertainty_pct, _source = resolve_station_uncertainty_pct(station_id, metadata_df, config)
        except Exception:
            if strict:
                raise
            continue
        lookup[station_id] = float(station_uncertainty_pct)
    return lookup


def enrich_case_scores(
    case_scores: pd.DataFrame,
    *,
    project_dir: Path,
    setup_dir: Path,
    score_station_sigma_threshold: float | None = None,
) -> pd.DataFrame:
    if case_scores.empty:
        return case_scores.copy()

    out = case_scores.copy()
    baseline_cols = ["score_set", "variable", "stream", "timestamp", "obs_id"]
    baseline = out[out["representation"] == "open_loop"][
        baseline_cols + ["error", "abs_error", "sq_error", "crps"]
    ].rename(
        columns={
            "error": "open_loop_error",
            "abs_error": "open_loop_abs_error",
            "sq_error": "open_loop_sq_error",
            "crps": "open_loop_case_crps",
        }
    )
    out = out.merge(baseline, on=baseline_cols, how="left")
    out["delta_error"] = out["open_loop_error"] - out["error"]
    out["delta_abs_error"] = out["open_loop_abs_error"] - out["abs_error"]
    out["delta_sq_error"] = out["open_loop_sq_error"] - out["sq_error"]
    out["delta_crps"] = out["open_loop_case_crps"] - out["crps"]
    out["obs_kind"] = np.where(out["obs_id"].astype(str).str.lower() == "roi", "roi", "station")
    out["station_uncertainty_pct"] = np.nan
    out["exclude_from_non_sigma_scores"] = False

    station_meta = _normalize_station_metadata(_load_station_metadata(project_dir, setup_dir))
    if not station_meta.empty:
        out = out.merge(station_meta, on="obs_id", how="left")
    else:
        out["station_name"] = np.nan
        out["station_elevation_m"] = np.nan
        out["station_x"] = np.nan
        out["station_y"] = np.nan

    out["obs_name"] = np.where(
        out["obs_kind"] == "roi",
        "roi",
        out["station_name"].fillna(out["obs_id"].astype(str)),
    )
    out["obs_label"] = np.where(
        out["obs_kind"] == "roi",
        out["variable"].astype(str).str.replace("_", " ", regex=False).str.upper() + " ROI",
        np.where(
            pd.notna(out["station_elevation_m"]),
            out["obs_name"].astype(str) + " (" + out["station_elevation_m"].round(0).astype("Int64").astype(str) + " m)",
            out["obs_name"].astype(str),
        ),
    )

    if score_station_sigma_threshold is not None:
        uncertainty_lookup = _station_uncertainty_lookup(
            case_scores=out,
            project_dir=project_dir,
            setup_dir=setup_dir,
            strict=True,
        )
        if uncertainty_lookup:
            out["station_uncertainty_pct"] = np.where(
                out["obs_kind"] == "station",
                out["obs_id"].astype(str).str.strip().str.lower().map(uncertainty_lookup),
                np.nan,
            )
        uncertainty_pct = pd.to_numeric(out["station_uncertainty_pct"], errors="coerce")
        out["exclude_from_non_sigma_scores"] = (
            (out["obs_kind"] == "station") & uncertainty_pct.ge(float(score_station_sigma_threshold))
        ).fillna(False)
    return out


def _non_sigma_score_group(group: pd.DataFrame) -> pd.DataFrame:
    if "exclude_from_non_sigma_scores" not in group.columns:
        return group
    exclude = group["exclude_from_non_sigma_scores"].fillna(False).astype(bool)
    return group.loc[~exclude].copy()


def _aggregate_group_rows(group: pd.DataFrame) -> pd.Series:
    non_sigma_group = _non_sigma_score_group(group)
    errors = pd.to_numeric(group["error"], errors="coerce").to_numpy(dtype=float)
    non_sigma_sq_errors = pd.to_numeric(non_sigma_group["sq_error"], errors="coerce").to_numpy(dtype=float)
    z_sq_errors = pd.to_numeric(group["z_sq_error"], errors="coerce").to_numpy(dtype=float)
    abs_errors = pd.to_numeric(group["abs_error"], errors="coerce").to_numpy(dtype=float)
    non_sigma_crps = pd.to_numeric(non_sigma_group["crps"], errors="coerce").to_numpy(dtype=float)
    non_sigma_spread = pd.to_numeric(non_sigma_group["spread"], errors="coerce").to_numpy(dtype=float)
    row = {
        "n_cases": int(len(group)),
        "rmse": float(np.sqrt(np.nanmean(non_sigma_sq_errors))) if len(non_sigma_sq_errors) else np.nan,
        "z_rmse": float(np.sqrt(_finite_mean(z_sq_errors))) if len(z_sq_errors) else np.nan,
        "mae": float(np.nanmean(abs_errors)) if len(abs_errors) else np.nan,
        "bias": float(np.nanmean(errors)) if len(errors) else np.nan,
        "ubrmse": float(np.sqrt(np.nanmean((errors - np.nanmean(errors)) ** 2))) if len(errors) else np.nan,
        "crps": float(np.nanmean(non_sigma_crps)) if len(non_sigma_crps) else np.nan,
        "spread_mean": float(np.nanmean(non_sigma_spread)) if len(non_sigma_spread) else np.nan,
    }
    for nominal in _COVERAGE_LEVELS:
        values = pd.to_numeric(group[f"coverage_{nominal}"], errors="coerce").to_numpy(dtype=float)
        row[f"coverage_{nominal}"] = float(np.nanmean(values)) if len(values) else np.nan
    return pd.Series(row)


def aggregate_scores(case_scores: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if case_scores.empty:
        return pd.DataFrame()

    rep_group_cols = list(group_cols) + ["representation"]
    grouped = (
        case_scores.groupby(rep_group_cols, dropna=False, sort=True)
        .apply(_aggregate_group_rows)
        .reset_index()
    )
    grouped["spread_skill_ratio"] = np.where(
        grouped["rmse"] > 0.0,
        grouped["spread_mean"] / grouped["rmse"],
        np.nan,
    )

    baseline_cols = list(group_cols)
    baseline = grouped[grouped["representation"] == "open_loop"][
        baseline_cols + ["rmse", "z_rmse", "crps", "bias"]
    ].rename(
        columns={
            "rmse": "open_loop_rmse",
            "z_rmse": "open_loop_z_rmse",
            "crps": "open_loop_crps",
            "bias": "open_loop_bias",
        }
    )
    merged = grouped.merge(baseline, on=baseline_cols, how="left")
    merged["ner"] = np.where(
        merged["open_loop_rmse"] > 0.0,
        1.0 - (merged["rmse"] / merged["open_loop_rmse"]),
        np.nan,
    )
    merged["zskill"] = np.where(
        merged["open_loop_z_rmse"] > 0.0,
        1.0 - (merged["z_rmse"] / merged["open_loop_z_rmse"]),
        np.nan,
    )
    merged["crpss"] = np.where(
        merged["open_loop_crps"] > 0.0,
        1.0 - (merged["crps"] / merged["open_loop_crps"]),
        np.nan,
    )
    merged["delta_bias"] = merged["open_loop_bias"] - merged["bias"]
    merged["delta_rmse"] = merged["open_loop_rmse"] - merged["rmse"]
    merged["delta_z_rmse"] = merged["open_loop_z_rmse"] - merged["z_rmse"]
    merged["delta_crps"] = merged["open_loop_crps"] - merged["crps"]
    return merged.sort_values(rep_group_cols).reset_index(drop=True)


def reliability_rows(case_scores: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if case_scores.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    rep_group_cols = list(group_cols) + ["representation"]
    for keys, group in case_scores.groupby(rep_group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: value for col, value in zip(rep_group_cols, keys, strict=True)}
        ensemble_kind = str(group["ensemble_kind"].iloc[0])

        for nominal in _COVERAGE_LEVELS:
            coverage = pd.to_numeric(group[f"coverage_{nominal}"], errors="coerce").to_numpy(dtype=float)
            rows.append(
                {
                    **base,
                    "diagnostic": "interval_coverage",
                    "nominal_level": nominal / 100.0,
                    "value": float(np.nanmean(coverage)) if len(coverage) else np.nan,
                    "count": int(len(group)),
                    "bin_index": np.nan,
                    "bin_count": np.nan,
                }
            )

        if ensemble_kind == "unweighted_ensemble":
            valid_rank = group.dropna(subset=["rank_bin", "rank_bin_count"])
            if not valid_rank.empty:
                bin_counts = sorted({int(v) for v in valid_rank["rank_bin_count"].tolist()})
                if len(bin_counts) == 1:
                    n_bins = int(bin_counts[0])
                    bins = np.bincount(valid_rank["rank_bin"].astype(int), minlength=n_bins)
                    total = int(valid_rank.shape[0])
                    for idx, count in enumerate(bins):
                        rows.append(
                            {
                                **base,
                                "diagnostic": "rank_histogram",
                                "nominal_level": np.nan,
                                "value": float(count / total) if total else np.nan,
                                "count": int(count),
                                "bin_index": int(idx),
                                "bin_count": int(n_bins),
                            }
                        )
        elif ensemble_kind == "weighted_ensemble":
            pits = pd.to_numeric(group["pit"], errors="coerce").dropna().to_numpy(dtype=float)
            if pits.size:
                edges = np.linspace(0.0, 1.0, _PIT_BIN_COUNT + 1)
                counts, _ = np.histogram(pits, bins=edges)
                total = int(pits.size)
                for idx, count in enumerate(counts):
                    rows.append(
                        {
                            **base,
                            "diagnostic": "pit_histogram",
                            "nominal_level": np.nan,
                            "value": float(count / total) if total else np.nan,
                            "count": int(count),
                            "bin_index": int(idx),
                            "bin_count": int(_PIT_BIN_COUNT),
                        }
                    )
    return pd.DataFrame(rows)


__all__ = ["aggregate_scores", "enrich_case_scores", "reliability_rows"]
