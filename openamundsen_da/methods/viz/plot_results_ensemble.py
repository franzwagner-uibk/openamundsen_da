"""openamundsen_da.methods.viz.plot_results_ensemble

Per-station ensemble plots for point-model CSV outputs (e.g., SWE or snow
depth) across all members. Mirrors the prior plotting script used in the
project but adapts to the packaged directory layout via `step_dir /
ensembles / <ensemble> / {open_loop, member_xxx} / results`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import re

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.io.paths import list_member_dirs, list_point_files_results
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
from openamundsen_da.util.ts import apply_window, resample_and_smooth, read_timeseries_csv


FIGSIZE = (12.0, 5.2)
DEFAULT_BAND_LOW = 0.05
DEFAULT_BAND_HIGH = 0.95


def _list_point_files(step_dir: Path, ensemble: str) -> Tuple[Optional[Path], List[str]]:
    """Delegate to io.paths.list_point_files_results for discovery."""
    return list_point_files_results(step_dir, ensemble)


def _load_stations_table(step_dir: Path, ensemble: str) -> Optional[pd.DataFrame]:
    """Load stations.csv from open_loop or first member meteo dir if available."""
    base = step_dir / "ensembles" / ensemble
    candidates = [base / "open_loop" / "meteo" / "stations.csv"]
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if members:
        candidates.append(members[0] / "meteo" / "stations.csv")
    for path in candidates:
        if path.is_file():
            try:
                return pd.read_csv(path)
            except Exception:
                return None
    return None


def _find_station_meta(st_df: Optional[pd.DataFrame], token: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (name, altitude_m) for a station token using a stations table."""
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
            normalized = df[col].astype(str).str.strip().str.lower()
            hit = df.loc[normalized == token.lower()]
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


def _read_point_series(csv_path: Path, time_col: str, value_col: str) -> pd.DataFrame:
    # Thin wrapper around util.ts.read_timeseries_csv
    df = read_timeseries_csv(csv_path, time_col, [value_col])
    return df


def _read_member_perturbations(step_dir: Path, ensemble: str) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """Return mapping member_name -> (delta_T, f_p) parsed from INFO.txt if present."""
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    members = list_member_dirs(step_dir / "ensembles", ensemble)
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
                        m = re.search(r"([-+]?\d+\.?\d*)", line)
                        if m:
                            dT = float(m.group(1))
                    if "f_p" in lower or "precip factor" in lower:
                        m = re.search(r"([-+]?\d+\.?\d*)", line)
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
    pieces = []
    if dT is not None:
        pieces.append(f"dT={dT:+.2f}")
    if fp is not None:
        pieces.append(f"f_p={fp:.2f}")
    return f"{member_name} ({', '.join(pieces)})"


def _series_has_data(series: pd.Series) -> bool:
    return pd.to_numeric(series, errors="coerce").notna().any()


def _plot_point_station(
    *,
    station_token: str,
    station_name: str,
    altitude_m: Optional[float],
    mem_series: List[pd.Series],
    mem_labels: List[str],
    open_loop: Optional[pd.Series],
    var_title: str,
    title: str,
    subtitle: str,
    backend: str,
    out_path: Path,
    band_low: float,
    band_high: float,
) -> None:
    import matplotlib

    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIGSIZE)

    for series, label in zip(mem_series, mem_labels):
        ax.plot(series.index, series.values, lw=LW_MEMBER, alpha=0.9, label=label)

    mean, lo, hi = envelope(mem_series, q_low=band_low, q_high=band_high)
    if not mean.empty:
        ax.plot(mean.index, mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")

    if open_loop is not None and _series_has_data(open_loop):
        ax.plot(open_loop.index, open_loop.values, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open loop", zorder=5)

    alt_suffix = f" ({altitude_m:.0f} m)" if altitude_m is not None else ""
    long_subtitle = subtitle or f"{station_name}{alt_suffix} - {var_title}"

    ax.set_xlabel("Time")
    ax.set_ylabel(var_title)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=LEGEND_NCOL,
            frameon=False,
            fontsize=8,
        )
        fig.subplots_adjust(bottom=0.34)

    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.92, long_subtitle, ha="center", va="top", fontsize=10, color="#555555")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="oa-da-plot-results",
        description="Per-station ensemble plots for point-model CSV results (e.g., swe, snow_depth).",
    )
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--time-col", default="time")
    p.add_argument("--var-col", default="swe", help="Column inside the CSV to plot (default: swe)")
    p.add_argument("--var-label", default="", help="Pretty variable label for titles (defaults to var name)")
    p.add_argument("--var-units", default="", help="Units appended to axis label")
    p.add_argument("--station", action="append", help="Specific point CSV basename (e.g., point_station_001.csv)")
    p.add_argument("--max-stations", type=int)
    p.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    p.add_argument("--resample", type=str, help="Pandas resample rule (e.g., D)")
    p.add_argument("--resample-agg", type=str, default="mean", help="Aggregation to use when resampling (default: mean)")
    p.add_argument("--rolling", type=int, help="Rolling window length (samples) applied after resampling")
    p.add_argument("--band-low", type=float, default=DEFAULT_BAND_LOW)
    p.add_argument("--band-high", type=float, default=DEFAULT_BAND_HIGH)
    p.add_argument("--title", default="Model Results Ensemble")
    p.add_argument("--subtitle", default="")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <step>/plots/results)")
    p.add_argument("--backend", default="Agg")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), enqueue=False, colorize=True, format=LOGURU_FORMAT)

    if args.band_low >= args.band_high:
        logger.error("--band-low ({}) must be smaller than --band-high ({})", args.band_low, args.band_high)
        return 2

    start = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    step_dir = Path(args.step_dir)
    ol_results_dir, point_files = _list_point_files(step_dir, args.ensemble)
    if not point_files:
        logger.error("No point_*.csv files found under {}/ensembles/{}/.../results", args.step_dir, args.ensemble)
        return 3

    if args.station:
        keep = set(args.station)
        point_files = [f for f in point_files if f in keep]
    if args.max_stations is not None:
        point_files = point_files[: max(0, int(args.max_stations))]

    member_dirs = list_member_dirs(step_dir / "ensembles", args.ensemble)
    if not member_dirs:
        logger.error("No member directories found for ensemble={}", args.ensemble)
        return 4

    member_results = [(m.name, m / "results") for m in member_dirs if (m / "results").is_dir()]
    if not member_results:
        logger.error("No member results directories found under {}/ensembles/{}", args.step_dir, args.ensemble)
        return 5

    pert_map = _read_member_perturbations(step_dir, args.ensemble)
    stations_df = _load_stations_table(step_dir, args.ensemble)
    out_root = args.output_dir if args.output_dir else (step_dir / "plots" / "results")
    step_name = step_dir.name
    effective_title = f"{args.title} | {step_name}" if args.title else step_name
    var_label = args.var_label or args.var_col
    var_title = f"{var_label} ({args.var_units})" if args.var_units else var_label

    for fname in point_files:
        ol_series: Optional[pd.Series] = None
        if ol_results_dir is not None:
            csv_path = ol_results_dir / fname
            if csv_path.is_file():
                try:
                    df = _read_point_series(csv_path, args.time_col, args.var_col)
                    df = resample_and_smooth(df, args.resample, {args.var_col: args.resample_agg} if args.resample else None, args.rolling)
                    df = apply_window(df, start, end)
                    if args.var_col in df.columns:
                        series = df[args.var_col].dropna()
                        if not series.empty:
                            ol_series = series
                except Exception as exc:
                    logger.warning("Failed to read open_loop results for {}: {}", fname, exc)

        mem_series: List[pd.Series] = []
        mem_labels: List[str] = []
        for member_name, res_dir in member_results:
            csv_path = res_dir / fname
            if not csv_path.is_file():
                continue
            try:
                df = _read_point_series(csv_path, args.time_col, args.var_col)
                df = resample_and_smooth(df, args.resample, {args.var_col: args.resample_agg} if args.resample else None, args.rolling)
                df = apply_window(df, start, end)
                if args.var_col not in df.columns:
                    continue
                series = df[args.var_col].dropna()
                if series.empty:
                    continue
                mem_series.append(series)
                mem_labels.append(_format_member_label(member_name, pert_map.get(member_name, (None, None))))
            except Exception as exc:
                logger.warning("Failed to read results for {} in {}: {}", fname, member_name, exc)

        if not mem_series:
            logger.warning("Skipping {}: no member data found in the requested window.", fname)
            continue

        token = Path(fname).stem
        display_token = token.replace("point_", "", 1)
        st_name, st_alt = _find_station_meta(stations_df, display_token)
        station_name = st_name or display_token

        out_path = out_root / f"{token}_{args.var_col}.png"
        _plot_point_station(
            station_token=token,
            station_name=station_name,
            altitude_m=st_alt,
            mem_series=mem_series,
            mem_labels=mem_labels,
            open_loop=ol_series,
            var_title=var_title,
            title=effective_title,
            subtitle=args.subtitle,
            backend=args.backend,
            out_path=out_path,
            band_low=float(args.band_low),
            band_high=float(args.band_high),
        )
        logger.info("Wrote {}", out_path)

    logger.info("Finished results plots -> {}", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
