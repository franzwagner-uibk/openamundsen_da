"""Plot station results versus observations for merged batch outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402
import sys

from openamundsen_da.batch.manifest import BatchManifest
from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.methods.viz._utils import format_station_label
from openamundsen_da.methods.viz.plot_results_ensemble import _pretty_var_title
from openamundsen_da.methods.viz._style import (
    COLOR_DA_OBS,
    COLOR_OPEN_LOOP,
    GRID_ALPHA,
    GRID_LS,
    GRID_LW,
    FIGSIZE_RESULTS,
    FS_TITLE,
    LW_OPEN,
    LW_MEMBER,
)
from openamundsen_da.util.ts import read_timeseries_csv, parse_time_column


def _load_stations_df(points_dir: Path) -> Optional[pd.DataFrame]:
    """Load station metadata table if present."""
    meta_path = points_dir / "obs" / "stations" / "stations_snow_depth.csv"
    if not meta_path.is_file():
        return None
    try:
        return pd.read_csv(meta_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read station metadata {}: {}", meta_path.name, exc)
        return None


def _parse_time(series: pd.Series) -> pd.Series:
    """Use shared parser to keep behavior aligned with setup plotting."""
    return parse_time_column(series)


def _read_obs(path: Path, obs_column: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, nrows=1)
        time_col = "date" if "date" in df.columns else df.columns[0]
        if obs_column not in df.columns:
            raise ValueError(f"Missing obs column '{obs_column}'")
        obs_df = read_timeseries_csv(path, time_col, [obs_column])
        return obs_df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read obs {}: {}", path.name, exc)
        return None


def _read_point(path: Path, var: str) -> Optional[pd.Series]:
    try:
        series_df = read_timeseries_csv(path, "time", [var])
        return series_df[var]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read point {}: {}", path.name, exc)
        return None


def plot_station_comparisons(
    *,
    manifest_path: Path,
    points_dir: Optional[Path] = None,
    obs_dir: Optional[Path] = None,
    variable: str = "snow_depth",
    obs_column: str = "snow_height",
    obs_scale: float = 1.0,
    station_ids: Optional[Iterable[str]] = None,
) -> list[Path]:
    """Generate plots comparing model point output to station observations."""
    logger.remove()
    logger.add(sys.stdout, level="INFO", colorize=True, format=LOGURU_FORMAT)
    manifest = BatchManifest.load(manifest_path)
    pts_root = points_dir or (manifest_path.parent / "merged" / "points")
    obs_root = obs_dir or (pts_root / "obs" / "stations")
    # Store plots under batch_root/plots/points (sibling to perf)
    plot_dir = manifest_path.parent / "plots" / "points"
    plot_dir.mkdir(parents=True, exist_ok=True)

    obs_files = list(obs_root.glob("*.csv"))
    if station_ids:
        obs_files = [f for f in obs_files if f.stem in station_ids]

    stations_df = _load_stations_df(pts_root)

    written: list[Path] = []
    for obs_path in obs_files:
        sid = obs_path.stem
        point_path = pts_root / f"point_{sid}.csv"
        if not point_path.is_file():
            logger.debug("No point output for station {}; skipping plot", sid)
            continue

        obs_df = _read_obs(obs_path, obs_column)
        series = _read_point(point_path, variable)
        if obs_df is None or series is None:
            continue
        if obs_column not in obs_df.columns:
            logger.debug("Obs {} missing column {}; skipping", sid, obs_column)
            continue

        obs_series = obs_df[obs_column] * obs_scale

        # Daily aggregation to harmonize timestep differences and avoid time parsing quirks
        model_daily = series.resample("D").mean()
        obs_daily = obs_series.resample("D").mean()

        merged = pd.DataFrame({"model": model_daily, "obs": obs_daily}).dropna()
        if merged.empty:
            logger.debug("No overlapping data for {}", sid)
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_RESULTS)
        ax.plot(merged.index, merged["model"], label="model", color=COLOR_OPEN_LOOP, lw=LW_OPEN)
        ax.plot(merged.index, merged["obs"], label="obs", color=COLOR_DA_OBS, lw=LW_MEMBER, linestyle="--")
        ax.set_ylabel(_pretty_var_title(variable))
        ax.set_xlabel("Time")
        ax.grid(True, linestyle=GRID_LS, linewidth=GRID_LW, alpha=GRID_ALPHA)
        ax.legend()
        # Title with station name and altitude if available, similar to setup pipeline
        title_name, alt, _label = format_station_label(sid, stations_df, fallback=sid)
        alt_txt = f" ({int(alt)} m)" if alt is not None else ""
        fig.text(
            0.5,
            0.95,
            f"{title_name}{alt_txt} | {_pretty_var_title(variable)}",
            ha="center",
            va="top",
            fontsize=FS_TITLE,
        )
        fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))

        out_path = plot_dir / f"{sid}_{variable}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        written.append(out_path)
        logger.info("Wrote {}", out_path)

    if not written:
        logger.warning("No station plots were generated; check variable/obs column names.")
    else:
        logger.info("Finished station plots: wrote {} of {} candidates", len(written), len(obs_files))
    return written
