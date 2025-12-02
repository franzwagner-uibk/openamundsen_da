"""
plot_fractions.py
Author: openamundsen_da
Date: 2025-12-02
Description:
    Plot SCF and wet-snow fractions (obs + model) in one figure.

    Sources (all optional; at least one required):
    - SCF observations: obs/<season>/scf_summary.csv
    - Wet-snow observations: obs/<season>/wet_snow_summary.csv
    - Model SCF: CSV with date/time column and scf
    - Model wet snow: CSV with date/time column and wet_snow_fraction

    Defaults resolve obs paths from season/project and write
    plots/obs/fraction_timeseries.png under the season directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.observer.plot_scf_summary import _load_summary as _load_scf_obs
from openamundsen_da.methods.viz.aggregate_fractions import aggregate_fraction_envelope


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date/time column to 'date'."""
    for col in ("date", "time", "datetime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            return df.rename(columns={col: "date"})
    raise KeyError("No date/time column found")


def _load_fraction(path: Path, value_col: str) -> Optional[pd.DataFrame]:
    if path is None or not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or value_col not in df.columns:
        return None
    try:
        df = _parse_dates(df)
    except Exception:
        return None
    cols = ["date", value_col]
    for extra in ("value_min", "value_max", "n"):
        if extra in df.columns:
            cols.append(extra)
    return df[cols].dropna(subset=[value_col]).sort_values("date")


def _default_obs_path(project_dir: Path, season_name: str, filename: str) -> Path:
    return project_dir / "obs" / season_name / filename


def _default_output(season_dir: Path, output: Optional[Path]) -> Path:
    if output is not None:
        return output
    out_dir = season_dir / "plots" / "obs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "fraction_timeseries.png"


def plot_fractions(
    *,
    scf_obs: Optional[pd.DataFrame],
    scf_model: Optional[pd.DataFrame],
    wet_obs: Optional[pd.DataFrame],
    wet_model: Optional[pd.DataFrame],
    scf_env: Optional[pd.DataFrame],
    wet_env: Optional[pd.DataFrame],
    output: Path,
) -> None:
    """Render SCF and wet-snow series into one PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_scf = scf_obs is not None or scf_model is not None
    has_wet = wet_obs is not None or wet_model is not None
    if scf_env is not None:
        has_scf = True
    if wet_env is not None:
        has_wet = True
    n_axes = int(has_scf) + int(has_wet)
    if n_axes == 0:
        raise ValueError("No data available to plot.")

    fig, axes = plt.subplots(n_axes, 1, figsize=(10, 4 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]

    idx = 0
    if has_scf:
        ax = axes[idx]
        if scf_env is not None and not scf_env.empty:
            ax.fill_between(
                scf_env["date"],
                scf_env["value_min"],
                scf_env["value_max"],
                color="tab:blue",
                alpha=0.1,
                label="SCF model band",
            )
            ax.plot(scf_env["date"], scf_env["value_mean"], "-", color="tab:blue", alpha=0.7, label="SCF model mean")
        if scf_model is not None and not scf_model.empty:
            ax.plot(scf_model["date"], scf_model["scf"], "-", color="tab:blue", label="SCF model (single)")
        if scf_obs is not None and not scf_obs.empty:
            ax.plot(scf_obs["date"], scf_obs["scf"], "o", color="tab:orange", label="SCF obs")
        ax.set_ylabel("SCF (0..1)")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.2)
        idx += 1

    if has_wet:
        ax = axes[idx]
        if wet_env is not None and not wet_env.empty:
            ax.fill_between(
                wet_env["date"],
                wet_env["value_min"],
                wet_env["value_max"],
                color="tab:green",
                alpha=0.1,
                label="Wet-snow model band",
            )
            ax.plot(wet_env["date"], wet_env["value_mean"], "-", color="tab:green", alpha=0.7, label="Wet-snow model mean")
        if wet_model is not None and not wet_model.empty:
            ax.plot(wet_model["date"], wet_model["wet_snow_fraction"], "-", color="tab:green", label="Wet-snow model (single)")
        if wet_obs is not None and not wet_obs.empty:
            ax.plot(wet_obs["date"], wet_obs["wet_snow_fraction"], "o", color="tab:red", label="Wet-snow obs")
        ax.set_ylabel("Wet snow (0..1)")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-fractions",
        description="Plot SCF and wet-snow fractions (obs and model) in one figure.",
    )
    parser.add_argument("--season-dir", required=True, type=Path, help="Season directory (propagation/season_YYYY-YYYY)")
    parser.add_argument("--project-dir", type=Path, help="Project directory (default: season_dir/../..)")
    parser.add_argument("--scf-obs-csv", type=Path, help="Path to scf_summary.csv (obs)")
    parser.add_argument("--wet-obs-csv", type=Path, help="Path to wet_snow_summary.csv (obs)")
    parser.add_argument("--scf-model-csv", type=Path, help="Model SCF CSV (date/time + scf)")
    parser.add_argument("--wet-model-csv", type=Path, help="Model wet-snow CSV (date/time + wet_snow_fraction)")
    parser.add_argument("--scf-env-csv", type=Path, help="SCF envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--wet-env-csv", type=Path, help="Wet-snow envelope CSV (value_min/value_max/value_mean)")
    parser.add_argument("--output", type=Path, help="Output PNG path (default: <season>/plots/obs/fraction_timeseries.png)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    season_dir = Path(args.season_dir)
    season_name = season_dir.name
    project_dir = Path(args.project_dir) if args.project_dir else season_dir.parent.parent

    scf_obs_path = Path(args.scf_obs_csv) if args.scf_obs_csv else _default_obs_path(project_dir, season_name, "scf_summary.csv")
    wet_obs_path = Path(args.wet_obs_csv) if args.wet_obs_csv else _default_obs_path(project_dir, season_name, "wet_snow_summary.csv")
    scf_env_path = Path(args.scf_env_csv) if args.scf_env_csv else (season_dir / "point_scf_aoi_envelope.csv")
    wet_env_path = Path(args.wet_env_csv) if args.wet_env_csv else (season_dir / "point_wet_snow_aoi_envelope.csv")

    scf_obs = None
    try:
        scf_obs = _load_scf_obs(scf_obs_path)
    except Exception:
        scf_obs = _load_fraction(scf_obs_path, "scf")

    wet_obs = _load_fraction(wet_obs_path, "wet_snow_fraction")
    scf_model = _load_fraction(Path(args.scf_model_csv), "scf") if args.scf_model_csv else None
    wet_model = _load_fraction(Path(args.wet_model_csv), "wet_snow_fraction") if args.wet_model_csv else None
    scf_env = _load_fraction(scf_env_path, "value_mean")
    if scf_env is not None and not scf_env.empty and {"value_min", "value_max"}.issubset(scf_env.columns) is False:
        scf_env = None
    wet_env = _load_fraction(wet_env_path, "value_mean")
    if wet_env is not None and not wet_env.empty and {"value_min", "value_max"}.issubset(wet_env.columns) is False:
        wet_env = None

    if all(x is None or x.empty for x in (scf_obs, wet_obs, scf_model, wet_model, scf_env, wet_env)):
        logger.error("No data available to plot. Provide at least one obs/model series.")
        return 1

    out_path = _default_output(season_dir, args.output)
    try:
        plot_fractions(
            scf_obs=scf_obs,
            scf_model=scf_model,
            wet_obs=wet_obs,
            wet_model=wet_model,
            scf_env=scf_env,
            wet_env=wet_env,
            output=out_path,
        )
    except ModuleNotFoundError as exc:
        logger.error("matplotlib is required to plot: {}", exc)
        return 1
    except Exception as exc:
        logger.error("Plotting failed: {}", exc)
        return 1

    logger.info("Wrote plot: {}", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
