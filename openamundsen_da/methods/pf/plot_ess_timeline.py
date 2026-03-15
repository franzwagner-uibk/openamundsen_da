"""openamundsen_da.methods.pf.plot_ess_timeline

Plot ESS (and optionally normalized ESS/N) vs time by scanning a step's
assim directory for weights_*_YYYYMMDD.csv files.

Outputs a PNG line plot, saved next to the inputs unless --output is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from openamundsen_da.util.stats import effective_sample_size
from openamundsen_da.io.paths import list_step_dirs
from openamundsen_da.util.loguru_utils import configure_cli_logger


_RE_DATE = re.compile(r"weights_.+_(\d{8})\.csv$", re.IGNORECASE)


def _scan_weights(assim_dir: Path) -> list[tuple[datetime, Path]]:
    files: list[tuple[datetime, Path]] = []
    for p in sorted(assim_dir.glob("weights_*_*.csv")):
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


def _setup_id_from_dir(setup_dir: Path) -> str:
    """Derive a compact setup identifier from a setup directory name.

    Mirrors the behavior used in plot_setup_ensemble: if the directory
    name contains an underscore (e.g., 'setup_2017-2018'), the portion
    after the first underscore is used; otherwise the directory name is
    returned as-is.
    """
    name = setup_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def plot_setup_ess_timeline(
    setup_dir: Path,
    *,
    normalized: bool = False,
    threshold: float | None = None,
    backend: str = "Agg",
) -> Path:
    """Setup-wide ESS timeline across all steps.

    Scans steps/step_*/assim/weights_*_*.csv under setup_dir, computes ESS per
    assimilation date, and writes a single PNG under
    <setup_dir>/plots/assim/ess/setup_ess_timeline_<setup_id>.png.
    """
    setup_dir = Path(setup_dir)
    files: list[tuple[datetime, Path]] = []
    for step in list_step_dirs(setup_dir):
        assim_dir = step / "assim"
        if not assim_dir.is_dir():
            continue
        files.extend(_scan_weights(assim_dir))
    if not files:
        raise FileNotFoundError(f"No weights_*_*.csv found under steps in {setup_dir}")

    df = _compute_series(files)
    fig = _plot(
        df,
        normalized=normalized,
        threshold=threshold,
        title="ESS over time",
        subtitle=None,
        backend=backend,
    )
    setup_id = _setup_id_from_dir(setup_dir)
    out_dir = setup_dir / "plots" / "assim" / "ess"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"setup_ess_timeline_{setup_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    return out


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-ess", description="Plot ESS over time from weights_*_*.csv files")
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

    # Avoid enqueue for short-lived CLIs so messages flush before exit
    configure_cli_logger(args.log_level, enqueue=False)

    assim = Path(args.assim_dir) if args.assim_dir else (Path(args.step_dir) / "assim" if args.step_dir else None)
    if assim is None:
        logger.error("Provide --step-dir or --assim-dir")
        return 2
    logger.info("Scanning for weights under: {}", assim)
    files = _scan_weights(assim)
    logger.info("Found {} file(s)", len(files))
    if not files:
        logger.error("No weights_*_*.csv found under {}", assim)
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
