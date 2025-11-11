"""openamundsen_da.methods.viz.plot_season_ensemble

Season-wide plots that stitch together all steps into a single figure per
station, with vertical markers for assimilation dates. Two plot types are
supported:

- Forcing: two-panel layout (temperature, cumulative precipitation)
- Results: single-panel (e.g., SWE or snow_depth)

Layout and style mirror the existing per-step modules while aggregating data
across all steps found in a season directory. Only the prior ensemble is
supported as requested.
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
from openamundsen_da.methods.viz._style import (
    BAND_ALPHA,
    COLOR_MEAN,
    COLOR_OPEN_LOOP,
    LEGEND_NCOL,
    LW_MEMBER,
    LW_MEAN,
    LW_OPEN,
)
from openamundsen_da.util.stats import envelope
from openamundsen_da.util.ts import (
    apply_window,
    resample_and_smooth,
    cumulative_hydro,
    read_timeseries_csv,
    concat_series,
)
from openamundsen_da.methods.viz._utils import draw_assimilation_vlines, dedupe_legend


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


def _assimilation_dates(steps: Sequence[StepInfo]) -> List[datetime]:
    # Use each step's start_date for i >= 1 as assimilation instant
    dates: List[datetime] = []
    for i, s in enumerate(steps):
        if i == 0:
            continue
        if s.start is not None:
            dates.append(s.start)
    return dates


def _season_id_from_dir(season_dir: Path) -> str:
    # Expect name like season_2017-2018
    name = season_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name




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


def _draw_assim_and_legend(ax, dates: Sequence[datetime]) -> None:
    draw_assimilation_vlines(ax, dates)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        h, l = dedupe_legend(handles, labels)
        ax.legend_.remove() if getattr(ax, "legend_", None) else None
        if h:
            ax.legend(h, l, loc="best", frameon=False, fontsize=8)


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

    for fname in station_files:
        # Collect series per member across all steps
        member_series_temp: List[pd.Series] = []
        member_series_prec: List[pd.Series] = []
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
                    if precip_col in df.columns:
                        s = df[precip_col].dropna()
                        if not s.empty:
                            member_series_prec.append(s)
                except Exception as exc:
                    logger.warning("Failed reading member forcing {} in {}: {}", fname, m.name, exc)

        if not member_series_temp and not member_series_prec:
            logger.warning("No member data for station {} across season; skipping.", fname)
            continue

        # Prepare figure
        fig, axes = plt.subplots(2, 1, figsize=(12.0, 6.8), sharex=True)

        # Panel A: Temperature
        ax = axes[0]
        for s in member_series_temp:
            ax.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.9)
        mean, lo, hi = envelope(member_series_temp, q_low=0.05, q_high=0.95)
        if not mean.empty:
            ax.fill_between(mean.index, lo.values, hi.values, color=COLOR_MEAN, alpha=BAND_ALPHA, linewidth=0, label="ensemble band")
            ax.plot(mean.index, mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")
        if open_loop_temp:
            ol = concat_series(open_loop_temp)
            if not ol.empty:
                ax.plot(ol.index, ol.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, ls=":", lw=0.6, alpha=0.7)

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
        if not mean.empty:
            ax.fill_between(mean.index, lo.values, hi.values, color=COLOR_MEAN, alpha=BAND_ALPHA, linewidth=0, label="ensemble band")
            ax.plot(mean.index, mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")
        if open_loop_prec:
            olp = concat_series(open_loop_prec)
            if not olp.empty:
                try:
                    olp = cumulative_hydro(olp, hydro_month, hydro_day)
                except Exception:
                    pass
                ax.plot(olp.index, olp.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)
        ax.set_ylabel("Cum. precipitation (mm)")
        ax.grid(True, ls=":", lw=0.6, alpha=0.7)

        # Assimilation markers on both panels
        assim_dates = _assimilation_dates(steps)
        for ax in axes:
            _draw_assim_and_legend(ax, assim_dates)

        # Titles and legend
        token = Path(fname).stem
        title = f"Season Forcing | {season_dir.name}"
        subtitle = f"Station {token}"
        fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=12)
        fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=10, color="#555555")

        # Build a clean figure-level legend (avoid per-member clutter)
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        if handles:
            new_h, new_l = dedupe_legend(handles, labels)
            fig.legend(new_h, new_l, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=LEGEND_NCOL, frameon=False, fontsize=8)
            fig.subplots_adjust(bottom=0.17)

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
    backend: str = "Agg",
    log_level: str = "INFO",
) -> Path:
    import matplotlib

    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    logger.remove()
    logger.add(sys.stdout, level=(log_level or "INFO").upper(), enqueue=False, colorize=True, format=LOGURU_FORMAT)

    season_dir = Path(season_dir)
    steps = _list_steps_sorted(season_dir)
    if not steps:
        raise FileNotFoundError(f"No step_* directories found under {season_dir}")

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
    var_title = var_label or var_col
    if var_units:
        var_title = f"{var_title} ({var_units})"

    for fname in point_files:
        member_series: List[pd.Series] = []
        open_loop: List[pd.Series] = []

        for st in steps:
            # Open loop
            ol_dir = st.path / "ensembles" / "prior" / "open_loop" / "results"
            if ol_dir.is_dir():
                csv_path = ol_dir / fname
                if csv_path.is_file():
                    try:
                        df = _read_point_series(csv_path, time_col, var_col)
                        df = resample_and_smooth(df, resample, {var_col: resample_agg} if resample else None, rolling)
                        df = apply_window(df, start_date, end_date)
                        if var_col in df.columns:
                            s = df[var_col].dropna()
                            if not s.empty:
                                open_loop.append(s)
                    except Exception as exc:
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
                    df = _read_point_series(csv_path, time_col, var_col)
                    df = resample_and_smooth(df, resample, {var_col: resample_agg} if resample else None, rolling)
                    df = apply_window(df, start_date, end_date)
                    if var_col not in df.columns:
                        continue
                    s = df[var_col].dropna()
                    if s.empty:
                        continue
                    member_series.append(s)
                except Exception as exc:
                    logger.warning("Failed reading member results {} in {}: {}", fname, m.name, exc)

        if not member_series and not open_loop:
            logger.warning("No data for station {} across season; skipping.", fname)
            continue

        # Determine autostop if requested: one month after all members reach zero
        auto_end = _auto_end_from_swe_zero(member_series) if var_col.lower() in ("swe", "hs", "snow_depth", "snowdepth") else None
        effective_end = end_date
        if auto_end is not None:
            effective_end = min(effective_end, auto_end) if effective_end else auto_end

        # Build figure
        import matplotlib.pyplot as plt  # ensure pyplot is loaded
        fig, ax = plt.subplots(figsize=(12.0, 5.2))

        for s in member_series:
            if effective_end or start_date:
                df = apply_window(s.to_frame(var_col), start_date, effective_end)
                s_use = df[var_col].dropna()
            else:
                s_use = s
            ax.plot(s_use.index, s_use.values, lw=LW_MEMBER, alpha=0.9)

        # Envelope
        mem_for_env: List[pd.Series] = []
        for s in member_series:
            if effective_end or start_date:
                df = apply_window(s.to_frame(var_col), start_date, effective_end)
                ss = df[var_col].dropna()
            else:
                ss = s
            if not ss.empty:
                mem_for_env.append(ss)
        mean, lo, hi = envelope(mem_for_env, q_low=band_low, q_high=band_high)
        if not mean.empty:
            ax.fill_between(mean.index, lo.values, hi.values, color=COLOR_MEAN, alpha=BAND_ALPHA, linewidth=0, label="ensemble band")
            ax.plot(mean.index, mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")

        if open_loop:
            ol = concat_series(open_loop)
            if effective_end or start_date:
                df = apply_window(ol.to_frame(var_col), start_date, effective_end)
                ol = df[var_col].dropna()
            if not ol.empty:
                ax.plot(ol.index, ol.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)

        ax.set_xlabel("Time")
        ax.set_ylabel(var_title)
        ax.grid(True, ls=":", lw=0.6, alpha=0.7)

        # Assimilation markers
        assim_dates = _assimilation_dates(steps)
        _draw_assim_and_legend(ax, assim_dates)

        # Titles/legend
        token = Path(fname).stem
        title = f"Season Results | {season_dir.name}"
        subtitle = f"Station {token} - {var_title}"
        fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=12)
        fig.text(0.5, 0.92, subtitle, ha="center", va="top", fontsize=10, color="#555555")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # de-duplicate
            seen: Dict[str, int] = {}
            new_h, new_l = [], []
            for h, l in zip(handles, labels):
                if l not in seen:
                    seen[l] = 1
                    new_h.append(h)
                    new_l.append(l)
            fig.legend(
                new_h,
                new_l,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                ncol=LEGEND_NCOL,
                frameon=False,
                fontsize=8,
            )
            fig.subplots_adjust(bottom=0.32)

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
    backend: str = "Agg",
    log_level: str = "INFO",
) -> Tuple[Path, Path]:
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
            backend=args.backend,
            log_level=args.log_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
