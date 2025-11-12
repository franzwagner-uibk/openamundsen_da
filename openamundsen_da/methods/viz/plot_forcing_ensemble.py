"""openamundsen_da.methods.viz.plot_forcing_ensemble

Per-station ensemble plots for forcing time series (temperature and cumulative
precipitation) across all members. Designed to visualize prior/posterior
forcing after propagation/rejuvenation.

Behavior
- Reads station CSVs from `<step>/ensembles/<ensemble>/{open_loop,member_XXX}/meteo`.
- Requires explicit column names (no autodetection):
  date column (default 'date'), temperature 'temp', precipitation 'precip'.
- Produces two-panel figures per station:
  A) Temperature time series (members, mean, 5–95% band, open-loop)
  B) Cumulative precipitation by hydrological year (members, mean, band, open-loop)

Notes
- If open_loop is unavailable (e.g., posterior ensemble), the plot omits it.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.io.paths import list_member_dirs, read_step_config, list_station_files_forcing
from openamundsen_da.util.ts import apply_window, resample_and_smooth, cumulative_hydro, read_timeseries_csv
from openamundsen_da.util.stats import envelope
from openamundsen_da.methods.viz._style import (
    COLOR_MEAN,
    COLOR_OPEN_LOOP,
    BAND_ALPHA,
    LW_MEMBER,
    LW_MEAN,
    LW_OPEN,
    LEGEND_NCOL,
)


def _list_station_files(step_dir: Path, ensemble: str) -> Tuple[Optional[Path], List[str]]:
    """Delegate to io.paths.list_station_files_forcing for discovery."""
    return list_station_files_forcing(step_dir, ensemble)


def _load_stations_table(step_dir: Path, ensemble: str) -> Optional[pd.DataFrame]:
    """Load stations.csv from open_loop or first member meteo dir if available."""
    base = step_dir / "ensembles" / ensemble
    cand = [base / "open_loop" / "meteo" / "stations.csv"]
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if members:
        cand.append(members[0] / "meteo" / "stations.csv")
    for p in cand:
        if p.is_file():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
    return None


def _find_station_meta(st_df: Optional[pd.DataFrame], token: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (name, altitude_m) for a station token using a stations table.

    Recognizes common column names for id/name and altitude in a case-insensitive
    manner. Returns (None, None) if not found.
    """
    if st_df is None or st_df.empty:
        return None, None
    df = st_df.copy()
    cols_lower = {c.lower().strip(): c for c in df.columns}
    id_candidates = [c for c in ("id", "station_id", "station", "code") if c in cols_lower]
    name_candidates = [c for c in ("name", "station_name") if c in cols_lower]
    alt_candidates = [c for c in ("alt", "altitude", "elev", "elevation", "z", "height", "height_m") if c in cols_lower]
    alt_col = cols_lower[alt_candidates[0]] if alt_candidates else None
    def _match(col_key: str) -> Optional[pd.Series]:
        col = cols_lower[col_key]
        try:
            s = df[col].astype(str).str.strip().str.lower()
            hit = df.loc[s == token.lower()]
            if not hit.empty:
                return hit.iloc[0]
        except Exception:
            return None
        return None
    row = None
    for k in id_candidates:
        row = _match(k)
        if row is not None:
            break
    if row is None:
        for k in name_candidates:
            row = _match(k)
            if row is not None:
                break
    if row is None:
        return None, None
    name_val = None
    for k in name_candidates:
        try:
            name_val = str(row[cols_lower[k]]).strip()
            break
        except Exception:
            continue
    alt_val = None
    if alt_col is not None:
        try:
            alt_val = float(row[alt_col])
        except Exception:
            alt_val = None
    return name_val, alt_val


def _parse_time_index(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    # Kept for backward compatibility; prefer util.ts.read_timeseries_csv
    idx = pd.to_datetime(df[time_col], errors="coerce")
    out = df.copy()
    out.index = idx
    return out


def _read_station_series(csv_path: Path, time_col: str, temp_col: str, precip_col: str) -> pd.DataFrame:
    """Thin wrapper around util.ts.read_timeseries_csv."""
    cols: List[str] = []
    # We request both, then drop missing after
    cols = [c for c in (temp_col, precip_col) if c]
    df = read_timeseries_csv(csv_path, time_col, cols)
    return df


    # removed: use util.ts and util.stats helpers


def _read_member_perturbations(step_dir: Path, ensemble: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return mapping member_name -> (delta_T, f_p) parsed from INFO.txt if present."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    for m in members:
        info = m / "INFO.txt"
        dT: Optional[float] = None
        fp: Optional[float] = None
        if info.is_file():
            try:
                text = info.read_text(encoding="utf-8", errors="ignore")
                m1 = re.search(r"delta_T\s*\(additive\)\s*:\s*([+-]?\d+\.?\d*)", text, re.IGNORECASE)
                m2 = re.search(r"precip factor\s*f_p\s*:\s*([+-]?\d+\.?\d*)", text, re.IGNORECASE)
                if m1:
                    dT = float(m1.group(1))
                if m2:
                    fp = float(m2.group(1))
            except Exception:
                pass
        out[m.name] = (dT, fp)
    return out


def _plot_station(
    *,
    station_name: str,
    ol_df: Optional[pd.DataFrame],
    mem_dfs: List[pd.DataFrame],
    temp_col: str,
    precip_col: str,
    hydro_m: int,
    hydro_d: int,
    title: str,
    subtitle: Optional[str],
    backend: str,
    out_path: Path,
    member_labels: Optional[List[str]] = None,
) -> None:
    import matplotlib
    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    has_precip = bool(
        (ol_df is not None and precip_col in ol_df.columns)
        or any(precip_col in d.columns for d in mem_dfs)
    )
    if has_precip:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.5, 7.2), sharex=True)
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(11.5, 4.2))

    # Temperature panel
    temp_series = [d[temp_col].copy() for d in mem_dfs if temp_col in d.columns]
    t_mean, t_lo, t_hi = envelope(temp_series)
    for i, s in enumerate(temp_series):
        lbl = member_labels[i] if member_labels and i < len(member_labels) else None
        ax0.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.85, label=lbl)
    if not t_mean.empty:
        # Removed ensemble band fill; keep mean line only
        ax0.plot(t_mean.index, t_mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")
    if ol_df is not None and temp_col in ol_df.columns:
        ax0.plot(ol_df.index, ol_df[temp_col], color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open_loop")
    ax0.set_ylabel(temp_col)
    ax0.grid(True, ls=":", lw=0.6, alpha=0.7)

    # Precip cumulative panel
    if has_precip:
        mem_prec_cum = []
        for d in mem_dfs:
            if precip_col in d.columns:
                mem_prec_cum.append(cumulative_hydro(d[precip_col], hydro_m, hydro_d))
        p_mean, p_lo, p_hi = envelope(mem_prec_cum)
        for i, d in enumerate(mem_dfs):
            if precip_col in d.columns:
                s = cumulative_hydro(d[precip_col], hydro_m, hydro_d)
                lbl = member_labels[i] if member_labels and i < len(member_labels) else None
                ax1.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.85, label=lbl)
        if not p_mean.empty:
            # Removed ensemble band fill; keep mean line only
            ax1.plot(p_mean.index, p_mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")
        if ol_df is not None and precip_col in ol_df.columns:
            s = cumulative_hydro(ol_df[precip_col], hydro_m, hydro_d)
            ax1.plot(s.index, s.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open_loop")
        ax1.set_xlabel("date")
        ax1.set_ylabel(f"{precip_col} (cumulative)")
        ax1.grid(True, ls=":", lw=0.6, alpha=0.7)

    # Layout and legend below
    # Provide headroom above axes; temp-only plots keep extra space at top
    top_rect = (0.88 if (title or subtitle) else 0.94) if not has_precip else (0.90 if (title or subtitle) else 0.94)
    fig.tight_layout(rect=[0.02, 0.08, 0.98, top_rect])
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12)
    if subtitle:
        # Position subtitle slightly lower (closer to axes); use different offsets
        # for temp-only vs two-panel layouts.
        sub_y = 0.920 if not has_precip else 0.910
        fig.text(0.5, sub_y, subtitle, ha="center", va="top", fontsize=10, color="#555555")

    # Compose a shared legend from ax0 handles (they include member labels if provided)
    handles, labels = ax0.get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=LEGEND_NCOL, frameon=False, fontsize=8)
        # Increase bottom margin so the legend sits clearly below x tick labels.
        # Provide extra space for temperature-only plots.
        fig.subplots_adjust(bottom=0.30 if has_precip else 0.40)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-forcing", description="Per-station ensemble forcing plots (temp + cumulative precip)")
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--time-col", default="date")
    p.add_argument("--temp-col", default="temp")
    p.add_argument("--precip-col", default="precip")
    p.add_argument("--station", action="append", help="Specific station CSV basename (e.g., station_001.csv); repeat for multiple")
    p.add_argument("--max-stations", type=int)
    p.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    p.add_argument("--resample", type=str, help="Pandas resample rule (e.g., D)")
    p.add_argument("--rolling", type=int, help="Rolling window length (samples)")
    p.add_argument("--hydro-month", type=int, default=10)
    p.add_argument("--hydro-day", type=int, default=1)
    p.add_argument("--title", default="Forcing Ensemble")
    p.add_argument("--subtitle", default="")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <step>/plots/forcing)")
    p.add_argument("--backend", default="Agg")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), enqueue=False, colorize=True, format=LOGURU_FORMAT)

    start = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    # If no explicit dates are provided, try to read the step YAML for defaults
    if start is None or end is None:
        try:
            cfg = read_step_config(Path(args.step_dir))
        except Exception:
            cfg = {}
        def _parse_dt(val):
            try:
                return pd.to_datetime(val).to_pydatetime()
            except Exception:
                return None
        if start is None and cfg.get("start_date") is not None:
            start = _parse_dt(cfg.get("start_date"))
        if end is None and cfg.get("end_date") is not None:
            end = _parse_dt(cfg.get("end_date"))
        if (args.start_date is None and args.end_date is None) and (start is not None or end is not None):
            logger.info("Using step YAML window: start_date={} end_date={}",
                        (start.strftime("%Y-%m-%d") if start else "None"),
                        (end.strftime("%Y-%m-%d") if end else "None"))

    ol_dir, station_files = _list_station_files(Path(args.step_dir), args.ensemble)
    if not station_files:
        logger.error("No station CSVs found under {}/ensembles/{}/.../meteo", args.step_dir, args.ensemble)
        return 2
    if args.station:
        keep = set(args.station)
        station_files = [f for f in station_files if f in keep]
    if args.max_stations is not None:
        station_files = station_files[: max(0, int(args.max_stations))]

    member_dirs = list_member_dirs(Path(args.step_dir) / "ensembles", args.ensemble)
    if not member_dirs:
        logger.error("No member directories found for ensemble={}", args.ensemble)
        return 3

    # Member perturbation labels for legend (if INFO.txt present)
    pert_labels: Dict[str, Tuple[Optional[float], Optional[float]]] = _read_member_perturbations(Path(args.step_dir), args.ensemble)

    out_root = args.output_dir if args.output_dir else (Path(args.step_dir) / "plots" / "forcing")
    stations_df = _load_stations_table(Path(args.step_dir), args.ensemble)
    step_name = Path(args.step_dir).name
    effective_title = f"{args.title} | {step_name}" if args.title else step_name

    for fname in station_files:
        # Read open_loop (if present)
        ol_df: Optional[pd.DataFrame] = None
        if ol_dir is not None:
            try:
                ol_df = _read_station_series(ol_dir / fname, args.time_col, args.temp_col, args.precip_col)
                agg_map = {}
                if args.temp_col in ol_df.columns:
                    agg_map[args.temp_col] = "mean"
                if args.precip_col in ol_df.columns:
                    agg_map[args.precip_col] = "sum"
                ol_df = resample_and_smooth(ol_df, args.resample, agg_map if agg_map else None, args.rolling)
                ol_df = apply_window(ol_df, start, end)
            except Exception as e:
                logger.warning("open_loop read failed for {}: {}", fname, e)
                ol_df = None

        # Read members
        mem_dfs: List[pd.DataFrame] = []
        mem_labels: List[str] = []
        for m in member_dirs:
            p = m / "meteo" / fname
            if not p.is_file():
                continue
            try:
                d = _read_station_series(p, args.time_col, args.temp_col, args.precip_col)
                agg_map = {}
                if args.temp_col in d.columns:
                    agg_map[args.temp_col] = "mean"
                if args.precip_col in d.columns:
                    agg_map[args.precip_col] = "sum"
                d = resample_and_smooth(d, args.resample, agg_map if agg_map else None, args.rolling)
                d = apply_window(d, start, end)
                mem_dfs.append(d)
                dt, fp = pert_labels.get(m.name, (None, None))
                if dt is not None or fp is not None:
                    mem_labels.append(f"{m.name} (dT={dt:+.2f} fp={fp:.2f})" if dt is not None and fp is not None else m.name)
                else:
                    mem_labels.append(m.name)
            except Exception as e:
                logger.warning("member read failed for {} in {}: {}", fname, m.name, e)

        if not mem_dfs:
            logger.warning("No member data for station {} -> skipping", fname)
            continue

        token = Path(fname).stem
        st_name, st_alt = _find_station_meta(stations_df, token)
        # If no custom subtitle, build one from station metadata
        auto_sub = None
        if not args.subtitle:
            if st_name and st_alt is not None:
                auto_sub = f"{st_name} ({st_alt:.0f} m)"
            elif st_name:
                auto_sub = st_name
            elif st_alt is not None:
                auto_sub = f"({st_alt:.0f} m)"
        out_path = out_root / f"{token}.png"
        _plot_station(
            station_name=token,
            ol_df=ol_df,
            mem_dfs=mem_dfs,
            temp_col=args.temp_col,
            precip_col=args.precip_col,
            hydro_m=int(args.hydro_month),
            hydro_d=int(args.hydro_day),
            title=effective_title,
            subtitle=(args.subtitle or auto_sub),
            backend=args.backend,
            out_path=out_path,
            member_labels=mem_labels,
        )

    logger.info("Finished forcing plots -> {}", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
