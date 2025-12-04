"""openamundsen_da.methods.viz.plot_season_ensemble

Season-wide ensemble plots that stitch together all step segments into a single
figure per station, with vertical dashed lines marking assimilation instants.

Provides two plot types in the same style as the per-step modules:
- Forcing: two-panel layout (top: air temperature timeseries; bottom: cumulative
  precipitation by hydrological year)
- Results: single-panel (e.g., SWE or snow_depth)

Behavior and conventions
- Discovers steps under a season root (e.g., ``.../propagation/season_2017-2018``)
  by reading each ``step_XX.yml`` for ``start_date`` and ``end_date``.
- Uses the prior ensemble only and optionally draws open-loop segments when
  present in steps.
- Draws vertical dashed lines at the start of each step i >= 1 (assimilation
  times), excluding the first step (typically October 1st).
- Output figures are written under ``<season_dir>/plots/{forcing,results}/`` and
  include the season identifier in the filename.

CLI usage examples
- Forcing (two panels):
  ``python -m openamundsen_da.methods.viz.plot_season_ensemble forcing --season-dir <path/to/season> --hydro-month 10 --hydro-day 1``
- Results (SWE):
  ``python -m openamundsen_da.methods.viz.plot_season_ensemble results --season-dir <path/to/season> --var-col swe``

Notes
- End date accepts both ``YYYY-MM-DD`` and compact forms like ``YYYY-06_01``; the
  latter is normalized to ``YYYY-06-01``.
- Results autostop: if plotting a snow variable (SWE/HS/snow_depth), the plot is
  automatically truncated one month after the last date when any member remains
  positive, unless an explicit ``--end-date`` is earlier.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.io.paths import (
    list_member_dirs,
    read_step_config,
    list_station_files_forcing as io_list_station_files_forcing,
    list_point_files_results as io_list_point_files_results,
)
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.methods.viz._style import (
    BAND_ALPHA,
    COLOR_MEAN,
    COLOR_OPEN_LOOP,
    LEGEND_NCOL,
    LEGEND_NCOL_SEASON,
    LW_MEMBER,
    LW_MEAN,
    LW_OPEN,
    COLOR_DA_OBS,
    SIZE_DA_OBS,
    LW_DA_OBS,
    COLOR_OBS_SCF,
    SIZE_OBS_SCF,
    GRID_LS,
    GRID_LW,
    GRID_ALPHA,
    FS_TITLE,
    FS_SUBTITLE,
    COLOR_SUBTITLE,
    FS_ASSIM_LABEL,
    ASSIM_LABEL_ROT,
    FIGSIZE_FORCING,
    FIGSIZE_RESULTS,
)
from openamundsen_da.util.stats import envelope
from openamundsen_da.util.ts import (
    apply_window,
    resample_and_smooth,
    cumulative_hydro,
    read_timeseries_csv,
    concat_series,
)
from openamundsen_da.methods.viz._utils import (
    draw_assimilation_vlines,
    dedupe_legend,
    draw_assimilation_markers,
    draw_assim_labels,
    format_station_label,
)


# ---- Data structures --------------------------------------------------------


@dataclass
class StepInfo:
    path: Path
    start: Optional[datetime]
    end: Optional[datetime]


# ---- Utilities --------------------------------------------------------------


def _parse_date_opt(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    # allow YYYY-06_01 like inputs by normalizing
    t = text.replace("_", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def _list_steps_sorted(season_dir: Path) -> List[StepInfo]:
    steps: List[StepInfo] = []
    for p in sorted(season_dir.glob("step_*")):
        if not p.is_dir():
            continue
        cfg = read_step_config(p)
        steps.append(
            StepInfo(
                path=p,
                start=_parse_date_opt(str(cfg.get("start_date", ""))),
                end=_parse_date_opt(str(cfg.get("end_date", ""))),
            )
        )
    # sort by start date if present, else by name
    steps.sort(key=lambda s: (s.start or datetime.min, s.path.name))
    return steps


def _assimilation_dates(season_dir: Path) -> List[datetime]:
    """Return assimilation datetimes (midnight) from season.yml events."""
    events = load_assimilation_events(season_dir)
    return [datetime.combine(ev.date, datetime.min.time()) for ev in events]


def _season_id_from_dir(season_dir: Path) -> str:
    # Expect name like season_2017-2018
    name = season_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def _read_member_perturbations(step_dir: Path) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return mapping member_name -> (delta_T, f_p) parsed from INFO.txt if present (prior)."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    members = list_member_dirs(step_dir / "ensembles", "prior")
    import re as _re
    for member in members:
        info = member / "INFO.txt"
        dT: Optional[float] = None
        fp: Optional[float] = None
        if info.is_file():
            try:
                text = info.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    lower = line.lower()
                    if "delta" in lower or "dt" in lower:
                        m = _re.search(r"([-+]?\d+\.?\d*)", line)
                        if m:
                            dT = float(m.group(1))
                    if "f_p" in lower or "precip factor" in lower:
                        m = _re.search(r"([-+]?\d+\.?\d*)", line)
                        if m:
                            fp = float(m.group(1))
            except Exception:
                pass
        out[member.name] = (dT, fp)
    return out


def _format_member_label(member_name: str, pert: Tuple[Optional[float], Optional[float]]) -> str:
    dT, fp = pert
    if dT is None and fp is None:
        return member_name
    parts: List[str] = []
    if dT is not None:
        parts.append(f"dT={dT:+.2f}")
    if fp is not None:
        parts.append(f"f_p={fp:.2f}")
    return f"{member_name} ({', '.join(parts)})"


def _build_member_label_map(steps: Sequence[StepInfo]) -> Dict[str, str]:
    """Return empty map to avoid ambiguous labels across steps.

    Season plots span multiple steps; rejuvenation can change perturbations per
    step, so embedding (dT, f_p) in labels becomes misleading. We therefore use
    plain member names in legends for season plots.
    """
    return {}



def _auto_end_from_swe_zero(member_series: List[pd.Series]) -> Optional[datetime]:
    """Return autostop = last date where any member SWE>0 + 30 days.

    If no member data or all NaN, returns None.
    """
    if not member_series:
        return None
    # Build a union index
    all_idx = pd.DatetimeIndex(sorted({ts for s in member_series for ts in s.index}))
    if all_idx.empty:
        return None
    any_positive = pd.Series(False, index=all_idx)
    for s in member_series:
        # Align to union, NaNs treated as not positive
        ss = s.reindex(all_idx).fillna(0.0)
        any_positive = any_positive | (ss > 0)
    if not any_positive.any():
        # never positive -> return None so caller can decide
        return None
    last_pos_idx = any_positive[any_positive].index.max()
    return (last_pos_idx + timedelta(days=30)).to_pydatetime()


def _draw_assim(ax, dates: Sequence[datetime]) -> None:
    """Draw assimilation vlines only; figure-level legend is composed later."""
    draw_assimilation_vlines(ax, dates)


def _draw_assim_labels(ax, dates: Sequence[datetime]) -> None:
    """Draw per-assimilation labels centered on each vline above the axes."""
    draw_assim_labels(ax, dates, labels=None, max_labels=12, y_offset_pts=3.0, fontsize=FS_ASSIM_LABEL, color="black")


def _format_assim_summary(dates: Sequence[datetime]) -> str:
    """Return multi-line summary DAi: YYYY-MM-DD for assimilation dates."""
    return "\n".join(f"DA{i}: {d.strftime('%Y-%m-%d')}" for i, d in enumerate(dates, start=1))


def _draw_assim_summary_box(fig, ax, dates: Sequence[datetime], base_y: Optional[float] = None) -> None:
    """Disabled: assimilation date summary box removed to reduce clutter."""
    return


def _load_stations_table_from_steps(steps: Sequence["StepInfo"]) -> Optional[pd.DataFrame]:
    """Load stations.csv from the first step that provides it (open_loop or member)."""
    for st in steps:
        base = st.path / "ensembles" / "prior"
        candidates = [base / "open_loop" / "meteo" / "stations.csv"]
        members = list_member_dirs(st.path / "ensembles", "prior")
        if members:
            candidates.append(members[0] / "meteo" / "stations.csv")
        for p in candidates:
            if p.is_file():
                try:
                    return pd.read_csv(p)
                except Exception:
                    continue
    return None


# ---- Plotting: Forcing (two-panel) -----------------------------------------


def plot_season_forcing(
    *,
    season_dir: Path,
    date_col: str = "date",
    temp_col: str = "temp",
    precip_col: str = "precip",
    hydro_month: int = 10,
    hydro_day: int = 1,
    stations: Optional[List[str]] = None,
    max_stations: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    resample: Optional[str] = None,
    rolling: Optional[int] = None,
    backend: str = "Agg",
    log_level: str = "INFO",
) -> Path:
    """Create season-wide forcing plots for one or more stations.

    Parameters
    - season_dir: Season root directory (contains ``step_*`` subfolders).
    - date_col: Timestamp column in station CSVs (default: ``date``).
    - temp_col: Temperature column (default: ``temp``).
    - precip_col: Precipitation column (default: ``precip``).
    - hydro_month, hydro_day: Hydrological year start (default: 10/1).
    - stations: Optional list of station filenames to include (e.g., ``102376.csv``).
    - max_stations: Optional cap on the number of stations.
    - start_date, end_date: Optional window for the x-axis.
    - resample: Optional pandas resample rule (e.g., ``D``).
    - rolling: Optional rolling window (samples) applied after resampling.
    - backend: Matplotlib backend (default: ``Agg`` for headless).
    - log_level: Loguru level string (e.g., ``INFO``).

    Returns
    - Path to the output directory ``<season_dir>/plots/forcing``.
    """
    import matplotlib

    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    logger.remove()
    logger.add(sys.stdout, level=(log_level or "INFO").upper(), enqueue=False, colorize=True, format=LOGURU_FORMAT)

    season_dir = Path(season_dir)
    steps = _list_steps_sorted(season_dir)
    if not steps:
        raise FileNotFoundError(f"No step_* directories found under {season_dir}")

    # Determine station files from first step with meteo
    station_files: List[str] = []
    for s in steps:
        _ol, station_files = io_list_station_files_forcing(s.path, "prior")
        if station_files:
            break
    if not station_files:
        raise FileNotFoundError("No station CSV files found in any step's meteo directories")

    if stations:
        keep = set(stations)
        station_files = [f for f in station_files if f in keep]
    if max_stations is not None:
        station_files = station_files[: max(0, int(max_stations))]

    out_root = season_dir / "plots" / "forcing"
    out_root.mkdir(parents=True, exist_ok=True)
    season_id = _season_id_from_dir(season_dir)
    stations_df = _load_stations_table_from_steps(steps)
    member_label_map = _build_member_label_map(steps)
    assim_dates = _assimilation_dates(season_dir)
    assim_date_set = {d.date() for d in assim_dates}

    for fname in station_files:
        # Collect series per member across all steps
        member_series_temp: List[pd.Series] = []
        member_series_prec: List[pd.Series] = []
        member_labels_temp: List[str] = []
        member_labels_prec: List[str] = []
        open_loop_temp: List[pd.Series] = []
        open_loop_prec: List[pd.Series] = []

        for st in steps:
            # Open loop
            ol_dir = st.path / "ensembles" / "prior" / "open_loop" / "meteo"
            if ol_dir.is_dir():
                csv_path = ol_dir / fname
                if csv_path.is_file():
                    try:
                        df = read_timeseries_csv(csv_path, date_col, [temp_col, precip_col])
                        df = resample_and_smooth(df, resample, None, rolling)
                        df = apply_window(df, start_date, end_date)
                        if temp_col in df.columns:
                            s = df[temp_col].dropna()
                            if not s.empty:
                                open_loop_temp.append(s)
                        if precip_col in df.columns:
                            s = df[precip_col].dropna()
                            if not s.empty:
                                open_loop_prec.append(s)
                    except Exception as exc:
                        # Missing precip is expected for many stations -> skip quietly
                        if isinstance(exc, ValueError) and f"Missing column '{precip_col}'" in str(exc):
                            continue
                        logger.warning("Failed reading open_loop forcing {} in {}: {}", fname, st.path.name, exc)

            # Members
            members = list_member_dirs(st.path / "ensembles", "prior")
            for m in members:
                met_dir = m / "meteo"
                if not met_dir.is_dir():
                    continue
                csv_path = met_dir / fname
                if not csv_path.is_file():
                    continue
                try:
                    df = read_timeseries_csv(csv_path, date_col, [temp_col, precip_col])
                    df = resample_and_smooth(df, resample, None, rolling)
                    df = apply_window(df, start_date, end_date)
                    if temp_col in df.columns:
                        s = df[temp_col].dropna()
                        if not s.empty:
                            member_series_temp.append(s)
                            member_labels_temp.append(member_label_map.get(m.name, m.name))
                    if precip_col in df.columns:
                        s = df[precip_col].dropna()
                        if not s.empty:
                            member_series_prec.append(s)
                            member_labels_prec.append(member_label_map.get(m.name, m.name))
                except Exception as exc:
                    # Missing precip is expected for many stations -> skip quietly
                    if isinstance(exc, ValueError) and f"Missing column '{precip_col}'" in str(exc):
                        continue
                    logger.warning("Failed reading member forcing {} in {}: {}", fname, m.name, exc)

        if not member_series_temp and not member_series_prec:
            logger.warning("No member data for station {} across season; skipping.", fname)
            continue

        # Prepare figure
        fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_FORCING, sharex=True)

        # Panel A: Temperature (degC)
        ax = axes[0]
        for s, lbl in zip(member_series_temp, member_labels_temp):
            ax.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.9, label=lbl)
        mean, lo, hi = envelope(member_series_temp, q_low=0.05, q_high=0.95)
        # Removed ensemble mean line
        if open_loop_temp:
            ol = concat_series(open_loop_temp)
            if not ol.empty:
                ax.plot(ol.index, ol.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)
        ax.set_ylabel("Temperature (degC)")
        ax.grid(True, ls=GRID_LS, lw=GRID_LW, alpha=GRID_ALPHA)

        # Panel B: Cumulative precipitation (hydrological year)
        ax = axes[1]
        mem_cum: List[pd.Series] = []
        for s in member_series_prec:
            try:
                mem_cum.append(cumulative_hydro(s, hydro_month, hydro_day))
            except Exception:
                mem_cum.append(s)
        for s in mem_cum:
            ax.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.9)
        mean, lo, hi = envelope(mem_cum, q_low=0.05, q_high=0.95)
        # Removed ensemble mean line
        if open_loop_prec:
            olp = concat_series(open_loop_prec)
            if not olp.empty:
                try:
                    olp = cumulative_hydro(olp, hydro_month, hydro_day)
                except Exception:
                    pass
                ax.plot(olp.index, olp.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)
        ax.set_ylabel("Cum. precipitation (mm)")
        ax.grid(True, ls=GRID_LS, lw=GRID_LW, alpha=GRID_ALPHA)

        # Assimilation markers on both panels (step starts i >= 1)
        for ax in axes:
            _draw_assim(ax, assim_dates)

        # Titles, assimilation date line, and figure-level legend (de-duplicated)
        token = Path(fname).stem
        title = f"Season Forcing | {season_dir.name}"
        _base, _alt, station_label = format_station_label(token, stations_df, fallback=token)
        subtitle = station_label
        # Move title and subtitle slightly up to create more clearance
        fig.text(0.5, 0.985, title, ha="center", va="top", fontsize=FS_TITLE)
        fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=FS_SUBTITLE, color=COLOR_SUBTITLE)
        # Per-assimilation labels centered above the vlines on the top panel
        _draw_assim_labels(axes[0], assim_dates)
        # Provide extra vertical space between subtitle and axes for labels
        # Increase space further when many assimilation dates exist
        top_margin = 0.84 if len(assim_dates) <= 4 else (0.82 if len(assim_dates) <= 8 else 0.80)
        bottom_margin = 0.24
        fig.subplots_adjust(top=top_margin, bottom=bottom_margin)

        # Build a clean figure-level legend (avoid per-member clutter)
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles:
            new_h, new_l = dedupe_legend(handles, labels)
            # Align legend directly under the left part of the plot area
            pos = axes[0].get_position()
            legend_x = pos.x0
            legend_y = max(0.02, pos.y0 - 0.06)
            # Fixed 6 columns in the legend
            fig.legend(
                new_h,
                new_l,
                loc="upper left",
                bbox_to_anchor=(legend_x, legend_y),
                ncol=LEGEND_NCOL_SEASON,
                frameon=False,
                fontsize=8,
            )
            # DA date summary box under the right part of the plot area,
            # sharing the same vertical baseline as the legend.
            _draw_assim_summary_box(fig, axes[0], assim_dates, base_y=legend_y)
        else:
            _draw_assim_summary_box(fig, axes[0], assim_dates)

        out_path = out_root / f"season_forcing_{token}_{season_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        logger.info("Wrote {}", out_path)

    return out_root


# ---- Plotting: Results (single-panel) --------------------------------------


def plot_season_results(
    *,
    season_dir: Path,
    time_col: str = "time",
    var_col: str = "swe",
    var_label: str = "",
    var_units: str = "",
    stations: Optional[List[str]] = None,
    max_stations: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    resample: Optional[str] = None,
    resample_agg: str = "mean",
    rolling: Optional[int] = None,
    band_low: float = 0.05,
    band_high: float = 0.95,
    show_members: bool = False,
    backend: str = "Agg",
    log_level: str = "INFO",
    mode: str = "members",
) -> Path:
    """Create season-wide results plots (e.g., SWE or snow_depth) for one or more stations.

    mode:
      - "members": draw member traces (no band); member traces are hidden from the legend. (default)
      - "band": draw only the ensemble band/mean (no member traces).
    """
    import matplotlib

    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    logger.remove()
    logger.add(sys.stdout, level=(log_level or "INFO").upper(), enqueue=False, colorize=True, format=LOGURU_FORMAT)

    season_dir = Path(season_dir)
    steps = _list_steps_sorted(season_dir)
    if not steps:
        raise FileNotFoundError(f"No step_* directories found under {season_dir}")

    mode = (mode or "members").lower()
    if mode not in {"members", "band"}:
        mode = "members"

    assim_dates = _assimilation_dates(season_dir)

    # Determine available stations from first step that has results
    point_files: List[str] = []
    for s in steps:
        _ol, point_files = io_list_point_files_results(s.path, "prior")
        if point_files:
            break
    if not point_files:
        raise FileNotFoundError("No point_*.csv files found in any step's results directories")

    if stations:
        keep = set(stations)
        point_files = [f for f in point_files if f in keep]
    if max_stations is not None:
        point_files = point_files[: max(0, int(max_stations))]

    out_root = season_dir / "plots" / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    season_id = _season_id_from_dir(season_dir)
    stations_df = _load_stations_table_from_steps(steps)

    vv = (var_col or "").strip().lower()
    if not var_label and not var_units:
        if vv == "swe":
            var_title = "snow water equivalent [mm]"
        elif vv in ("snow_depth", "snowdepth", "hs"):
            var_title = "snow depth [m]"
        else:
            var_title = vv.replace("_", " ")
    else:
        var_title = var_label or var_col
        if var_units:
            var_title = f"{var_title} [{var_units}]"

    member_label_map = _build_member_label_map(steps)

    for fname in point_files:
        member_series: List[pd.Series] = []
        member_labels: List[str] = []
        open_loop: List[pd.Series] = []

        for st in steps:
            # Open loop
            ol_dir = st.path / "ensembles" / "prior" / "open_loop" / "results"
            if ol_dir.is_dir():
                csv_path = ol_dir / fname
                if csv_path.is_file():
                    try:
                        df = read_timeseries_csv(csv_path, time_col, [var_col])
                        df = resample_and_smooth(df, resample, {var_col: resample_agg} if resample else None, rolling)
                        df = apply_window(df, start_date, end_date)
                        if var_col in df.columns:
                            s = df[var_col].dropna()
                            if not s.empty:
                                open_loop.append(s)
                    except Exception as exc:
                        if isinstance(exc, ValueError) and f"Missing column '{var_col}'" in str(exc):
                            continue
                        logger.warning("Failed reading open_loop results {} in {}: {}", fname, st.path.name, exc)

            # Members
            members = list_member_dirs(st.path / "ensembles", "prior")
            for m in members:
                res_dir = m / "results"
                if not res_dir.is_dir():
                    continue
                csv_path = res_dir / fname
                if not csv_path.is_file():
                    continue
                try:
                    df = read_timeseries_csv(csv_path, time_col, [var_col])
                    df = resample_and_smooth(df, resample, {var_col: resample_agg} if resample else None, rolling)
                    df = apply_window(df, start_date, end_date)
                    if var_col not in df.columns:
                        continue
                    s = df[var_col].dropna()
                    if s.empty:
                        continue
                    member_series.append(s)
                    member_labels.append(member_label_map.get(m.name, m.name))
                except Exception as exc:
                    if isinstance(exc, ValueError) and f"Missing column '{var_col}'" in str(exc):
                        continue
                    logger.warning("Failed reading member results {} in {}: {}", fname, m.name, exc)

        if not member_series and not open_loop:
            logger.warning("No data for station {} across season; skipping.", fname)
            continue

        # Build figure
        fig, ax = plt.subplots(figsize=FIGSIZE_RESULTS)

        # Members or band
        if mode == "members":
            for s in member_series:
                ax.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.9, label="_nolegend_")
        else:
            mem_for_env: List[pd.Series] = []
            for s in member_series:
                mem_for_env.append(s)
            mean, lo, hi = envelope(mem_for_env, q_low=band_low, q_high=band_high)
            if not mean.empty:
                label_band = f"{int(band_low*100)}-{int(band_high*100)}% band"
                ax.fill_between(
                    mean.index,
                    lo,
                    hi,
                    color=COLOR_MEAN,
                    alpha=BAND_ALPHA,
                    label=label_band,
                    zorder=2,
                )
                ax.plot(
                    mean.index,
                    mean.values,
                    color=COLOR_MEAN,
                    lw=LW_OPEN,
                    label="ensemble mean",
                    zorder=4,
                )

        if open_loop:
            ol = concat_series(open_loop)
            if not ol.empty:
                ax.plot(ol.index, ol.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)

        ax.set_ylabel(var_title)
        ax.grid(True, ls=GRID_LS, lw=GRID_LW, alpha=GRID_ALPHA)

        # Assimilation markers and labels
        _draw_assim(ax, assim_dates)
        _draw_assim_labels(ax, assim_dates)

        # Titles/legend
        token = Path(fname).stem
        display_token = token.replace("point_", "", 1)
        _base, _alt, station_label = format_station_label(display_token, stations_df, fallback=display_token)
        title = f"Season Results | {season_dir.name}"
        subtitle = f"{station_label} - {var_title}"
        # Reduce vertical gap between title/subtitle and the plot area.
        fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=FS_TITLE)
        fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=FS_SUBTITLE, color=COLOR_SUBTITLE)
        top_margin = 0.90 if len(assim_dates) <= 4 else (0.88 if len(assim_dates) <= 8 else 0.86)
        bottom_margin = 0.30
        fig.subplots_adjust(top=top_margin, bottom=bottom_margin)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # Keep only one open loop and one assimilation entry; drop members/_nolegend_.
            want = {"open loop", "assimilation"}
            filtered = []
            seen = set()
            for h, l in zip(handles, labels):
                if l == "_nolegend_":
                    continue
                if l in want and l not in seen:
                    filtered.append((h, l))
                    seen.add(l)
            if filtered:
                handles, labels = zip(*filtered)
                ax.legend(
                    handles,
                    labels,
                    loc="upper right",
                    fontsize=8.5,
                    labelspacing=0.3,
                    borderpad=0.3,
                    handlelength=1.2,
                    handletextpad=0.4,
                )

        out_path = out_root / f"season_results_{token}_{var_col}_{season_id}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        logger.info("Wrote {}", out_path)

    logger.info("Finished season results plots -> {}", out_root)
    return out_root


def plot_season_both(
    *,
    season_dir: Path,
    stations: Optional[List[str]] = None,
    max_stations: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    show_members: bool = False,
    backend: str = "Agg",
    log_level: str = "INFO",
) -> Tuple[Path, Path]:
    """Convenience wrapper: generate both forcing and results season plots."""
    forcing_dir = plot_season_forcing(
        season_dir=season_dir,
        stations=stations,
        max_stations=max_stations,
        start_date=start_date,
        end_date=end_date,
        backend=backend,
        log_level=log_level,
    )
    results_dir = plot_season_results(
        season_dir=season_dir,
        stations=stations,
        max_stations=max_stations,
        start_date=start_date,
        end_date=end_date,
        show_members=show_members,
        backend=backend,
        log_level=log_level,
    )
    return forcing_dir, results_dir


# ---- CLI --------------------------------------------------------------------


def _cli(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="oa-da-plot-season",
        description="Season-wide ensemble plots (forcing/results) with assimilation markers.",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    def _common(sp):
        sp.add_argument("--season-dir", required=True, type=Path)
        sp.add_argument("--station", action="append", help="Specific station file name (e.g., 102376.csv or point_station_001.csv)")
        sp.add_argument("--max-stations", type=int)
        sp.add_argument("--start-date", type=str, help="YYYY-MM-DD")
        sp.add_argument("--end-date", type=str, help="YYYY-MM-DD or YYYY-06_01")
        sp.add_argument("--backend", default="Agg")
        sp.add_argument("--log-level", default="INFO")

    sp_f = sub.add_parser("forcing", help="Two-panel forcing season plot")
    _common(sp_f)
    sp_f.add_argument("--date-col", default="date")
    sp_f.add_argument("--temp-col", default="temp")
    sp_f.add_argument("--precip-col", default="precip")
    sp_f.add_argument("--resample", type=str, help="Pandas resample rule (e.g., D)")
    sp_f.add_argument("--rolling", type=int, help="Rolling window length (samples) after resample")
    sp_f.add_argument("--hydro-month", type=int, default=10, help="Hydrological year start month (default: 10)")
    sp_f.add_argument("--hydro-day", type=int, default=1, help="Hydrological year start day (default: 1)")

    sp_r = sub.add_parser("results", help="Results season plot (e.g., SWE or snow_depth)")
    _common(sp_r)
    sp_r.add_argument("--time-col", default="time")
    sp_r.add_argument("--var-col", default="swe")
    sp_r.add_argument("--var-label", default="")
    sp_r.add_argument("--var-units", default="")
    sp_r.add_argument("--resample", type=str, help="Pandas resample rule (e.g., D)")
    sp_r.add_argument("--resample-agg", type=str, default="mean")
    sp_r.add_argument("--rolling", type=int, help="Rolling window length (samples) after resample")
    sp_r.add_argument("--band-low", type=float, default=0.05)
    sp_r.add_argument("--band-high", type=float, default=0.95)
    sp_r.add_argument("--show-members", action="store_true", help="Draw individual ensemble members (default: hidden)")
    sp_r.add_argument("--mode", choices=["band", "members"], default="members", help="Plot mode: members (default) or band")

    args = p.parse_args(list(argv) if argv is not None else None)

    start = _parse_date_opt(args.start_date)
    end = _parse_date_opt(args.end_date)

    if args.mode == "forcing":
        plot_season_forcing(
            season_dir=args.season_dir,
            date_col=args.date_col,
            temp_col=args.temp_col,
            precip_col=args.precip_col,
            hydro_month=int(args.hydro_month),
            hydro_day=int(args.hydro_day),
            stations=args.station,
            max_stations=args.max_stations,
            start_date=start,
            end_date=end,
            resample=args.resample,
            rolling=args.rolling,
            backend=args.backend,
            log_level=args.log_level,
        )
    elif args.mode == "results":
        if args.band_low >= args.band_high:
            logger.error("--band-low ({}) must be smaller than --band-high ({})", args.band_low, args.band_high)
            return 2
        plot_season_results(
            season_dir=args.season_dir,
            time_col=args.time_col,
            var_col=args.var_col,
            var_label=args.var_label,
            var_units=args.var_units,
            stations=args.station,
            max_stations=args.max_stations,
            start_date=start,
            end_date=end,
            resample=args.resample,
            resample_agg=args.resample_agg,
            rolling=args.rolling,
            band_low=float(args.band_low),
            band_high=float(args.band_high),
            show_members=bool(getattr(args, "show_members", False)),
            backend=args.backend,
            log_level=args.log_level,
            mode=str(args.mode or "members"),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())


