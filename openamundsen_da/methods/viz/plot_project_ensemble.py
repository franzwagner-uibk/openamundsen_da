"""openamundsen_da.methods.viz.plot_project_ensemble

Setup-wide ensemble plots that stitch together all step segments into a single
figure per station, with vertical dashed lines marking assimilation instants.

Provides two plot types in the same style as the per-step modules:
- Forcing: two-panel layout (top: air temperature timeseries; bottom: cumulative
  precipitation by hydrological year)
- Results: single-panel (e.g., SWE or snow_depth)

Behavior and conventions
- Discovers steps under a setup root (e.g., ``.../projects/setup_2017-2018``)
  by reading each ``step_XX.yml`` for ``start_date`` and ``end_date``.
- Uses the prior ensemble only and optionally draws open-loop segments when
  present in steps.
- Draws vertical dashed lines at the start of each step i >= 1 (assimilation
  times), excluding the first step (typically October 1st).
- Output figures are written under ``<setup_dir>/plots/{forcing,results}/`` and
  include the setup identifier in the filename.

CLI usage examples
- Forcing (two panels):
  ``python -m openamundsen_da.methods.viz.plot_project_ensemble forcing --setup-dir <path/to/setup> --hydro-month 10 --hydro-day 1``
- Results (SWE):
  ``python -m openamundsen_da.methods.viz.plot_project_ensemble results --setup-dir <path/to/setup> --var-col swe``

Notes
- End date accepts both ``YYYY-MM-DD`` and compact forms like ``YYYY-06_01``; the
  latter is normalized to ``YYYY-06-01``.
- Results autostop: if plotting a snow variable (SWE/HS/snow_depth), the plot is
  automatically truncated one month after the last date when any member remains
  positive, unless an explicit ``--end-date`` is earlier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger

from openamundsen_da.io.paths import (
    list_member_dirs,
    list_steps_sorted,
    read_step_config,
    list_station_files_forcing as io_list_station_files_forcing,
    list_point_files_results as io_list_point_files_results,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.methods.viz._style import (
    BAND_ALPHA,
    COLOR_MEMBER,
    COLOR_MEAN,
    COLOR_OPEN_LOOP,
    LEGEND_NCOL,
    LEGEND_NCOL_SETUP,
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
    apply_fraction_grid,
    draw_assimilation_vlines,
    dedupe_legend,
    draw_assimilation_markers,
    draw_assim_labels,
    force_figure_text_black,
    format_station_label,
    save_figure_png,
    set_matplotlib_text_black,
)
from openamundsen_da.methods.viz._ensemble_meta import load_stations_table_from_steps
from openamundsen_da.util.station_da import station_observation_csvs


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


def _list_steps_sorted(setup_dir: Path) -> List[StepInfo]:
    steps: List[StepInfo] = []
    for p in list_steps_sorted(setup_dir):
        cfg = read_step_config(p)
        steps.append(
            StepInfo(
                path=p,
                start=_parse_date_opt(str(cfg.get("start_date", ""))),
                end=_parse_date_opt(str(cfg.get("end_date", ""))),
            )
        )
    return steps


def _assimilation_event_dates(setup_dir: Path) -> List[datetime]:
    """Return assimilation datetimes (midnight) from setup.yml events."""
    events = load_assimilation_events(setup_dir)
    return [datetime.combine(ev.date, datetime.min.time()) for ev in events]


def _setup_id_from_dir(setup_dir: Path) -> str:
    # Expect name like setup_2017-2018
    name = setup_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def _build_member_label_map(steps: Sequence[StepInfo]) -> Dict[str, str]:
    """Return empty map to avoid ambiguous labels across steps.

    Setup plots span multiple steps; rejuvenation can change perturbations per
    step, so embedding (dT, f_p) in labels becomes misleading. We therefore use
    plain member names in legends for setup plots.
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


def _station_obs_color(var_col: str) -> str:
    return COLOR_DA_OBS


def _station_model_color(var_col: str) -> str:
    key = str(var_col or "").strip().lower()
    if key in {"snow_depth", "hs"}:
        return "#cf7a20"
    if key == "swe":
        return "#7a58b5"
    return COLOR_MEAN


def _apply_result_axis_ticks(ax, var_col: str) -> None:
    import math
    from matplotlib.ticker import MultipleLocator

    key = str(var_col or "").strip().lower()
    data_max = max(0.0, float(getattr(ax.dataLim, "ymax", 0.0) or 0.0))
    if key == "swe":
        step_options = [50.0, 100.0]
    elif key in {"snow_depth", "hs"}:
        step_options = [0.25, 0.5, 1.0]
    else:
        return

    step = next((candidate for candidate in step_options if data_max <= candidate * 4.0), step_options[-1])
    upper = step * 4.0 if data_max <= step * 4.0 else math.ceil(data_max / step) * step
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def _format_assim_summary(dates: Sequence[datetime]) -> str:
    """Return multi-line summary DAi: YYYY-MM-DD for assimilation dates."""
    return "\n".join(f"DA{i}: {d.strftime('%Y-%m-%d')}" for i, d in enumerate(dates, start=1))


def _draw_assim_summary_box(fig, ax, dates: Sequence[datetime], base_y: Optional[float] = None) -> None:
    """Disabled: assimilation date summary box removed to reduce clutter."""
    return


def _plot_stepwise_mean(
    ax,
    mean: pd.Series,
    steps: Sequence[StepInfo],
    *,
    label: str,
    lw: float,
    color: str,
    zorder: int = 4,
) -> None:
    """Plot ensemble mean as separate line segments per step to avoid jumps.

    Draws one labeled segment for the first step that has data and hides
    subsequent segments from the legend (``_nolegend_``).
    """
    plotted = False
    for st in steps:
        if st.start is None or st.end is None:
            continue
        seg = mean[(mean.index >= st.start) & (mean.index <= st.end)]
        if seg.empty:
            continue
        this_label = label if not plotted else "_nolegend_"
        ax.plot(seg.index, seg.values, color=color, lw=lw, label=this_label, zorder=zorder)
        plotted = True


def _project_dir_from_setup(setup_dir: Path) -> Optional[Path]:
    """Best-effort project directory inference from a setup directory.

    Assumes layout <project>/projects/setup_xx. Returns None if the
    expected parents are not present.
    """
    setup_dir = Path(setup_dir)
    try:
        return setup_dir.parent.parent
    except Exception:
        return None


def _load_station_obs_for_setup(
    *,
    setup_dir: Path,
    time_col: str,
    var_col: str,
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    resample: Optional[str],
    resample_agg: str,
    rolling: Optional[int],
) -> Dict[str, pd.Series]:
    """Load station observations for the setup window, if available.

    Expects station CSVs under ``<project>/obs/stations`` with filenames
    matching station tokens (e.g., ``latschbloder.csv``). Station DA metadata
    files are ignored automatically. Each file should contain a time column
    (``time_col``) and the requested variable (``var_col``, e.g., ``swe`` or
    ``snow_depth``). Data are clipped to the provided start/end dates
    (typically the full setup).

    Returns
    -------
    dict
        Mapping ``station_token.lower()`` -> non-empty pandas Series.
    """
    project_dir = _project_dir_from_setup(setup_dir)
    if project_dir is None:
        return {}

    obs_dir = project_dir / "obs" / "stations"
    if not obs_dir.is_dir():
        return {}

    out: Dict[str, pd.Series] = {}
    for csv_path in station_observation_csvs(obs_dir):
        token = csv_path.stem.strip()
        if not token:
            continue
        try:
            df = read_timeseries_csv(csv_path, time_col, [var_col])
        except Exception as exc:
            # Missing column or parse error -> skip this station silently.
            if isinstance(exc, ValueError) and f"Missing column '{var_col}'" in str(exc):
                continue
            logger.debug("Skipping station obs {}: {}", csv_path.name, exc)
            continue
        try:
            df = resample_and_smooth(
                df,
                resample,
                {var_col: resample_agg} if resample else None,
                rolling,
            )
            df = apply_window(df, start_date, end_date)
        except Exception as exc:
            logger.debug("Failed to resample/window station obs {}: {}", csv_path.name, exc)
            continue
        if var_col not in df.columns:
            continue
        s = df[var_col].dropna()
        if s.empty:
            continue
        out[token.lower()] = s
    return out


# ---- Plotting: Forcing (two-panel) -----------------------------------------


def plot_setup_forcing(
    *,
    setup_dir: Path,
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
    configure_logger: bool = True,
) -> Path:
    """Create setup-wide forcing plots for one or more stations.

    Parameters
    - setup_dir: Setup root directory (contains ``steps/step_*`` subfolders).
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
    - Path to the output directory ``<setup_dir>/plots/forcing``.
    """
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt

    if configure_logger:
        configure_cli_logger(log_level or "INFO", enqueue=False)

    setup_dir = Path(setup_dir)
    steps = _list_steps_sorted(setup_dir)
    if not steps:
        raise FileNotFoundError(f"No step directories found under {setup_dir / 'steps'}")

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

    out_root = setup_dir / "plots" / "forcing"
    out_root.mkdir(parents=True, exist_ok=True)
    setup_id = _setup_id_from_dir(setup_dir)
    stations_df = load_stations_table_from_steps([s.path for s in steps], "prior")
    member_label_map = _build_member_label_map(steps)
    assim_dates = _assimilation_event_dates(setup_dir)
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
            logger.warning("No member data for station {} across setup; skipping.", fname)
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
        title = f"Setup Forcing | {setup_dir.name}"
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
                ncol=LEGEND_NCOL_SETUP,
                frameon=False,
                fontsize=8,
            )
            # DA date summary box under the right part of the plot area,
            # sharing the same vertical baseline as the legend.
            _draw_assim_summary_box(fig, axes[0], assim_dates, base_y=legend_y)
        else:
            _draw_assim_summary_box(fig, axes[0], assim_dates)

        out_path = out_root / f"setup_forcing_{token}_{setup_id}.png"
        force_figure_text_black(fig, axes)
        save_figure_png(fig, out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        logger.info("Wrote {}", out_path)

    return out_root


# ---- Plotting: Results (single-panel) --------------------------------------


def plot_setup_results(
    *,
    setup_dir: Path,
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
    configure_logger: bool = True,
) -> Path:
    """Create setup-wide results plots (e.g., SWE or snow_depth) for one or more stations.

    mode:
      - "members": draw member traces (no band); member traces are hidden from the legend. (default)
      - "band": draw only the ensemble band/mean (no member traces).
    """
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt

    if configure_logger:
        configure_cli_logger(log_level or "INFO", enqueue=False)

    setup_dir = Path(setup_dir)
    steps = _list_steps_sorted(setup_dir)
    if not steps:
        raise FileNotFoundError(f"No step directories found under {setup_dir / 'steps'}")

    mode = (mode or "members").lower()
    if mode not in {"members", "band"}:
        mode = "members"

    assim_dates = _assimilation_event_dates(setup_dir)

    # Effective setup window from step configs if not explicitly provided
    setup_start: Optional[datetime] = None
    setup_end: Optional[datetime] = None
    if steps:
        starts = [s.start for s in steps if s.start is not None]
        ends = [s.end for s in steps if s.end is not None]
        if starts:
            setup_start = min(starts)
        if ends:
            setup_end = max(ends)
    effective_start = start_date or setup_start
    effective_end = end_date or setup_end

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

    out_root = setup_dir / "plots" / "results"
    out_root.mkdir(parents=True, exist_ok=True)
    setup_id = _setup_id_from_dir(setup_dir)
    stations_df = load_stations_table_from_steps([s.path for s in steps], "prior")

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
    # Best-effort station observations (e.g., SWE/HS) over the full setup window
    station_obs = _load_station_obs_for_setup(
        setup_dir=setup_dir,
        time_col=time_col,
        var_col=var_col,
        start_date=effective_start,
        end_date=effective_end,
        resample=resample,
        resample_agg=resample_agg,
        rolling=rolling,
    )

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
                        df = apply_window(df, effective_start, effective_end)
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
                    df = apply_window(df, effective_start, effective_end)
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
            logger.warning("No data for station {} across setup; skipping.", fname)
            continue

        # Build figure
        fig, ax = plt.subplots(figsize=FIGSIZE_RESULTS)

        # Station token (e.g., point_latschbloder -> latschbloder)
        token = Path(fname).stem
        display_token = token.replace("point_", "", 1)

        # Optional station observations (if available for this station/variable)
        obs_series: Optional[pd.Series] = None
        if station_obs:
            obs_series = station_obs.get(display_token.lower())

        # Members or band + ensemble mean (stepwise, to avoid jumps after resampling)
        station_obs_color = _station_obs_color(var_col)
        station_model_color = _station_model_color(var_col)
        mean, lo, hi = envelope(member_series, q_low=0.0, q_high=1.0)
        if not mean.empty:
            ax.fill_between(
                mean.index,
                lo,
                hi,
                color=station_model_color,
                alpha=BAND_ALPHA,
                label="_nolegend_",
                zorder=2,
            )
            _plot_stepwise_mean(
                ax,
                mean,
                steps,
                label="ensemble mean (variable color)",
                lw=LW_MEAN,
                color=station_model_color,
                zorder=4,
            )
        if open_loop:
            ol = concat_series(open_loop)
            if not ol.empty:
                ax.plot(ol.index, ol.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop (model)", zorder=5)

        # Station observations stay visually dominant over all model traces.
        if obs_series is not None and not obs_series.empty:
            ax.plot(
                obs_series.index,
                obs_series.values,
                "-",
                color=station_obs_color,
                lw=LW_DA_OBS,
                label="station observation",
                zorder=6,
            )

        ax.set_ylabel(var_title)
        apply_fraction_grid(ax, y_step=None)
        _apply_result_axis_ticks(ax, var_col)

        # Assimilation markers and labels
        _draw_assim(ax, assim_dates)
        _draw_assim_labels(ax, assim_dates)

        # Always show the full setup window on the x-axis when available,
        # regardless of station/model data coverage.
        if effective_start is not None and effective_end is not None:
            try:
                ax.set_xlim(effective_start, effective_end)
            except Exception:
                pass

        # Titles/legend
        _base, _alt, station_label = format_station_label(display_token, stations_df, fallback=display_token)
        title = f"Setup Results | {setup_dir.name}"
        subtitle = f"{station_label} - {var_title}"
        # Reduce vertical gap between title/subtitle and the plot area.
        fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=FS_TITLE)
        fig.text(0.5, 0.94, subtitle, ha="center", va="top", fontsize=FS_SUBTITLE, color=COLOR_SUBTITLE)
        top_margin = 0.90 if len(assim_dates) <= 4 else (0.88 if len(assim_dates) <= 8 else 0.86)
        bottom_margin = 0.30
        fig.subplots_adjust(top=top_margin, bottom=bottom_margin)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            filtered = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
            if filtered:
                handles, labels = zip(*filtered)
                handles, labels = dedupe_legend(list(handles), list(labels))
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

        out_path = out_root / f"setup_results_{token}_{var_col}_{setup_id}.png"
        force_figure_text_black(fig, [ax])
        save_figure_png(fig, out_path, bbox_inches="tight", pad_inches=0.08)
        plt.close(fig)
        logger.info("Wrote {}", out_path)

    logger.info("Finished setup results plots -> {}", out_root)
    return out_root


def plot_setup_both(
    *,
    setup_dir: Path,
    stations: Optional[List[str]] = None,
    max_stations: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    show_members: bool = False,
    backend: str = "Agg",
    log_level: str = "INFO",
    configure_logger: bool = True,
) -> Tuple[Path, Path]:
    """Convenience wrapper: generate both forcing and results setup plots."""
    forcing_dir = plot_setup_forcing(
        setup_dir=setup_dir,
        stations=stations,
        max_stations=max_stations,
        start_date=start_date,
        end_date=end_date,
        backend=backend,
        log_level=log_level,
        configure_logger=configure_logger,
    )
    results_dir = plot_setup_results(
        setup_dir=setup_dir,
        stations=stations,
        max_stations=max_stations,
        start_date=start_date,
        end_date=end_date,
        show_members=show_members,
        backend=backend,
        log_level=log_level,
        configure_logger=configure_logger,
    )
    return forcing_dir, results_dir


# ---- CLI --------------------------------------------------------------------


def _cli(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="oa-da-plot-setup",
        description="Setup-wide ensemble plots (forcing/results) with assimilation markers.",
    )
    # Use a separate dest name for the subcommand to avoid clashing with the
    # '--mode' option used by the results plot (band vs members).
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--setup-dir", required=True, type=Path)
        sp.add_argument("--station", action="append", help="Specific station file name (e.g., 102376.csv or point_station_001.csv)")
        sp.add_argument("--max-stations", type=int)
        sp.add_argument("--start-date", type=str, help="YYYY-MM-DD")
        sp.add_argument("--end-date", type=str, help="YYYY-MM-DD or YYYY-06_01")
        sp.add_argument("--backend", default="Agg")
        sp.add_argument("--log-level", default="INFO")

    sp_f = sub.add_parser("forcing", help="Two-panel forcing setup plot")
    _common(sp_f)
    sp_f.add_argument("--date-col", default="date")
    sp_f.add_argument("--temp-col", default="temp")
    sp_f.add_argument("--precip-col", default="precip")
    sp_f.add_argument("--resample", type=str, help="Pandas resample rule (e.g., D)")
    sp_f.add_argument("--rolling", type=int, help="Rolling window length (samples) after resample")
    sp_f.add_argument("--hydro-month", type=int, default=10, help="Hydrological year start month (default: 10)")
    sp_f.add_argument("--hydro-day", type=int, default=1, help="Hydrological year start day (default: 1)")

    sp_r = sub.add_parser("results", help="Results setup plot (e.g., SWE or snow_depth)")
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

    if args.command == "forcing":
        plot_setup_forcing(
            setup_dir=args.setup_dir,
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
    elif args.command == "results":
        if args.band_low >= args.band_high:
            logger.error("--band-low ({}) must be smaller than --band-high ({})", args.band_low, args.band_high)
            return 2
        plot_setup_results(
            setup_dir=args.setup_dir,
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
