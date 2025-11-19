"""openamundsen_da.methods.pf.plot_ess_timeline

Plot ESS (and optionally normalized ESS/N) vs time by scanning a step's
assim directory for weights_scf_YYYYMMDD.csv files.

Outputs a PNG line plot, saved next to the inputs unless --output is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from openamundsen_da.util.stats import effective_sample_size


_RE_DATE = re.compile(r"weights_scf_(\d{8})\.csv$", re.IGNORECASE)


def _scan_weights(assim_dir: Path) -> list[tuple[datetime, Path]]:
    files: list[tuple[datetime, Path]] = []
    for p in sorted(assim_dir.glob("weights_scf_*.csv")):
        m = _RE_DATE.search(p.name)
        if not m:
            continue
        dt = datetime.strptime(m.group(1), "%Y%m%d")
        files.append((dt, p))
    return files


def _compute_series(files: list[tuple[datetime, Path]]) -> pd.DataFrame:
    rows: list[dict] = []
    for dt, p in files:
        df = pd.read_csv(p)
        if "weight" not in df:
            continue
        w = np.asarray(df["weight"], dtype=float)
        ess = effective_sample_size(w)
        rows.append({"date": dt, "ess": ess, "n": w.size, "ess_norm": ess / w.size if w.size > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("date")


def _plot(df: pd.DataFrame, normalized: bool, threshold: float | None, title: str, subtitle: str | None, *, backend: str = "Agg"):
    import matplotlib
    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    ycol = "ess_norm" if normalized else "ess"
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.plot(df["date"], df[ycol], marker="o", lw=1.8, color="#1f77b4")
    ax.set_xlabel("date")
    ax.set_ylabel("ESS/N" if normalized else "ESS")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    if threshold is not None:
        ax.axhline(threshold, color="#d62728", lw=1.2, ls="--")

    top_rect = 0.90 if (title or subtitle) else 0.94
    fig.tight_layout(rect=[0.02, 0.04, 0.98, top_rect])
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10, color="#555555")
    return fig


def _season_id_from_dir(season_dir: Path) -> str:
    """Derive a compact season identifier from a season directory name.

    Mirrors the behavior used in plot_season_ensemble: if the directory
    name contains an underscore (e.g., 'season_2017-2018'), the portion
    after the first underscore is used; otherwise the directory name is
    returned as-is.
    """
    name = season_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def plot_season_ess_timeline(
    season_dir: Path,
    *,
    normalized: bool = False,
    threshold: float | None = None,
    backend: str = "Agg",
) -> Path:
    """Season-wide ESS timeline across all steps.

    Scans step_*/assim/weights_scf_*.csv under season_dir, computes ESS per
    assimilation date, and writes a single PNG under
    <season_dir>/plots/assim/ess/season_ess_timeline_<season_id>.png.
    """
    season_dir = Path(season_dir)
    files: list[tuple[datetime, Path]] = []
    for step in sorted(season_dir.glob("step_*")):
        assim_dir = step / "assim"
        if not assim_dir.is_dir():
            continue
        files.extend(_scan_weights(assim_dir))
    if not files:
        raise FileNotFoundError(f"No weights_scf_*.csv found under steps in {season_dir}")

    df = _compute_series(files)
    fig = _plot(
        df,
        normalized=normalized,
        threshold=threshold,
        title="ESS over time",
        subtitle=None,
        backend=backend,
    )
    season_id = _season_id_from_dir(season_dir)
    out_dir = season_dir / "plots" / "assim" / "ess"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"season_ess_timeline_{season_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    return out


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-ess", description="Plot ESS over time from weights_scf_*.csv files")
    p.add_argument("--step-dir", type=Path, help="Step directory containing 'assim' folder")
    p.add_argument("--assim-dir", type=Path, help="Assimilation directory (default: <step-dir>/assim)")
    p.add_argument("--normalized", action="store_true", help="Plot ESS/N instead of ESS")
    p.add_argument("--threshold", type=float, help="Draw horizontal reference line (ESS/N if --normalized else ESS)")
    p.add_argument("--output", type=Path, help="Output PNG path (default: <assim-dir>/ess_timeline.png)")
    p.add_argument("--title", default="ESS over time", help="Plot title")
    p.add_argument("--subtitle", default="", help="Plot subtitle")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--backend", default="Agg", help="Matplotlib backend (Agg, SVG, module://mplcairo.Agg)")
    args = p.parse_args(argv)

    from openamundsen_da.core.constants import LOGURU_FORMAT
    logger.remove()
    # Avoid enqueue for short-lived CLIs so messages flush before exit
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=False, format=LOGURU_FORMAT)

    assim = Path(args.assim_dir) if args.assim_dir else (Path(args.step_dir) / "assim" if args.step_dir else None)
    if assim is None:
        logger.error("Provide --step-dir or --assim-dir")
        return 2
    logger.info("Scanning for weights under: {}", assim)
    files = _scan_weights(assim)
    logger.info("Found {} file(s)", len(files))
    if not files:
        logger.error("No weights_scf_*.csv found under {}", assim)
        return 1

    df = _compute_series(files)
    logger.info("Computed ESS for {} date(s) (normalized={}): {}", len(df), bool(args.normalized), 
                ", ".join(d.strftime("%Y-%m-%d") for d in df["date"]))
    try:
        fig = _plot(
            df,
            normalized=bool(args.normalized),
            threshold=args.threshold,
            title=args.title,
            subtitle=(args.subtitle or None),
            backend=args.backend,
        )
    except ModuleNotFoundError:
        logger.error("matplotlib is required to plot. Install it in your environment.")
        return 3
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return 4

    out = Path(args.output) if args.output else (assim / "ess_timeline.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plot to: {}", out)
    try:
        fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    except Exception as e:
        logger.error(f"Saving PNG failed: {e}")
        return 5
    logger.info("Wrote plot: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
