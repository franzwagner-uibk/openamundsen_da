"""openamundsen_da.methods.viz.plot_results_ensemble

Per-point ensemble plots for model results (SWE or snow depth) across all
members using point CSV outputs (point_*.csv) from openAMUNDSEN.

Behavior
- Reads point CSVs from `<step>/ensembles/<ensemble>/{open_loop,member_XXX}/results`.
- Requires explicit columns: time column (default 'time') and variable column
  (e.g., 'swe' or 'snow_depth'). No autodetection.
- Produces a single-panel figure per point file with member lines, ensemble mean,
  5–95% band, and open-loop if available.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.io.paths import list_member_dirs
from openamundsen_da.util.ts import apply_window, resample_and_smooth
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


def _list_point_files(step_dir: Path, ensemble: str) -> Tuple[Optional[Path], List[str]]:
    base = step_dir / "ensembles" / ensemble
    ol_res = base / "open_loop" / "results"
    if ol_res.is_dir():
        files = [f.name for f in sorted(ol_res.glob("point_*.csv"))]
        if files:
            return ol_res, files
    # fallback: scan first member
    members = list_member_dirs(step_dir / "ensembles", ensemble)
    if not members:
        return None, []
    first_res = members[0] / "results"
    files = [f.name for f in sorted(first_res.glob("point_*.csv"))]
    return (ol_res if ol_res.is_dir() else None), files


def _read_point_csv(csv_path: Path, time_col: str, var_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if time_col not in df.columns:
        raise ValueError(f"{csv_path.name}: missing time column: {time_col}")
    if var_col not in df.columns:
        raise ValueError(f"{csv_path.name}: missing variable column: {var_col}")
    t_raw = df[time_col].astype(str)
    # Try common formats, then fallback to generic parsing
    try:
        t = pd.to_datetime(t_raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
        if t.isna().all():
            t = pd.to_datetime(t_raw, format="%Y-%m-%d", errors="coerce")
    except Exception:
        t = pd.to_datetime(t_raw, errors="coerce")
    if t.isna().all():
        raise ValueError(f"{csv_path.name}: could not parse time values in column: {time_col}")
    s = pd.Series(pd.to_numeric(df[var_col], errors="coerce"), index=t, dtype=float)
    s = s[~s.index.isna()]
    if not s.index.is_unique:
        s = s.groupby(level=0).mean()
    return pd.DataFrame({var_col: s}).sort_index()


    # removed local helpers; use util.ts and util.stats


def _plot_point(
    *,
    token: str,
    var_col: str,
    ol_df: Optional[pd.DataFrame],
    mem_dfs: List[pd.DataFrame],
    title: str,
    subtitle: Optional[str],
    backend: str,
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.5, 4.8))

    mem_series = [d[var_col].copy() for d in mem_dfs if var_col in d.columns]
    m_mean, m_lo, m_hi = _envelope(mem_series)
    for s in mem_series:
        ax.plot(s.index, s.values, lw=LW_MEMBER, alpha=0.85)
    if not m_mean.empty:
        ax.fill_between(m_mean.index, m_lo, m_hi, color=COLOR_MEAN, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(m_mean.index, m_mean.values, color=COLOR_MEAN, lw=LW_MEAN, label="ensemble mean")
    if ol_df is not None and var_col in ol_df.columns:
        ax.plot(ol_df.index, ol_df[var_col], color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="open_loop")

    ax.set_xlabel("date")
    ax.set_ylabel(var_col)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)

    top_rect = 0.90 if (title or subtitle) else 0.94
    fig.tight_layout(rect=[0.02, 0.06, 0.98, top_rect])
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10, color="#555555")

    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        fig.legend(handles, labels, loc="lower center", ncol=LEGEND_NCOL, frameon=False, fontsize=8)
        fig.subplots_adjust(bottom=0.16)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def cli_main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-results", description="Per-point ensemble plots for SWE or snow depth")
    p.add_argument("--step-dir", required=True, type=Path)
    p.add_argument("--ensemble", required=True, choices=("prior", "posterior"))
    p.add_argument("--time-col", default="time")
    p.add_argument("--var-col", required=True, help="Variable column in point CSVs, e.g., swe or snow_depth")
    p.add_argument("--point", action="append", help="Specific point CSV name (e.g., point_A.csv); repeat to plot a subset")
    p.add_argument("--max-points", type=int)
    p.add_argument("--start-date", type=str, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, help="YYYY-MM-DD")
    p.add_argument("--resample", type=str, help="Pandas resample rule (e.g., D)")
    p.add_argument("--rolling", type=int, help="Rolling window length (samples)")
    p.add_argument("--title", default="Results Ensemble")
    p.add_argument("--subtitle", default="")
    p.add_argument("--output-dir", type=Path, help="Output directory (default: <step>/assim/plots/results)")
    p.add_argument("--backend", default="Agg")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), enqueue=False, colorize=True, format=LOGURU_FORMAT)

    start = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None
    end = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else None

    ol_dir, point_files = _list_point_files(Path(args.step_dir), args.ensemble)
    if not point_files:
        logger.error("No point_*.csv files found under {}/ensembles/{}/.../results", args.step_dir, args.ensemble)
        return 2
    if args.point:
        keep = set(args.point)
        point_files = [f for f in point_files if f in keep]
    if args.max_points is not None:
        point_files = point_files[: max(0, int(args.max_points))]

    member_dirs = list_member_dirs(Path(args.step_dir) / "ensembles", args.ensemble)
    if not member_dirs:
        logger.error("No member directories found for ensemble={}", args.ensemble)
        return 3

    out_root = args.output_dir if args.output_dir else (Path(args.step_dir) / "assim" / "plots" / "results")

    for fname in point_files:
        # open_loop
        ol_df: Optional[pd.DataFrame] = None
        if ol_dir is not None:
            p = ol_dir / fname
            if p.is_file():
                try:
                    ol_df = _read_point_csv(p, args.time_col, args.var_col)
                    ol_df = resample_and_smooth(ol_df, args.resample, {args.var_col: "mean"}, args.rolling)
                    ol_df = apply_window(ol_df, start, end)
                except Exception as e:
                    logger.warning("open_loop read failed for {}: {}", fname, e)
                    ol_df = None

        mem_dfs: List[pd.DataFrame] = []
        for m in member_dirs:
            p = m / "results" / fname
            if not p.is_file():
                continue
            try:
                d = _read_point_csv(p, args.time_col, args.var_col)
                d = resample_and_smooth(d, args.resample, {args.var_col: "mean"}, args.rolling)
                d = apply_window(d, start, end)
                mem_dfs.append(d)
            except Exception as e:
                logger.warning("member read failed for {} in {}: {}", fname, m.name, e)

        if not mem_dfs:
            logger.warning("No member data for point {} -> skipping", fname)
            continue

        token = Path(fname).stem.replace("point_", "")
        out_path = out_root / f"{token}_{args.var_col}.png"
        _plot_point(
            token=token,
            var_col=args.var_col,
            ol_df=ol_df,
            mem_dfs=mem_dfs,
            title=args.title,
            subtitle=(args.subtitle or None),
            backend=args.backend,
            out_path=out_path,
        )

    logger.info("Finished results plots -> {}", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
