"""
openamundsen_da.observer.plot_scf_summary

Purpose
- Plot a simple, publication-ready SCF time series from a season-level
  `scf_summary.csv` produced by the MOD10A1 preprocessing workflow.

Behavior
- Reads CSV, parses/validates the `date` and `scf` columns, sorts by date,
  then renders a compact line + scatter figure with SCF on [0..1].
- Saves a PNG next to the CSV unless `--output` is given.
- Logs concise context (rows, date range, output path) with a green-timestamp
  format consistent with the rest of the toolkit.

Notes
- Designed for single-region summaries (one row per day).
- Uses a headless Matplotlib backend; no GUI required.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger
import sys


def _load_summary(csv_path: Path) -> pd.DataFrame:
    """Load and normalize SCF summary CSV.

    Parameters
    ----------
    csv_path : Path
        Path to `scf_summary.csv` containing at least `date` and `scf`.

    Returns
    -------
    pandas.DataFrame
        Data sorted by `date` with parsed datetime column.
    """
    # Read and validate minimal schema
    df = pd.read_csv(csv_path)
    if "date" not in df or "scf" not in df:
        raise ValueError("CSV must contain columns 'date' and 'scf'")
    # Parse dates and sort chronologically
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def _plot(df: pd.DataFrame, title: str | None = None, subtitle: str | None = None):
    """Render a compact SCF time series plot.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with `date` and `scf` columns.
    title : str, optional
        Main title placed above the axes.
    subtitle : str, optional
        Secondary title placed just below the main title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (not shown).
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Use manual layout reservation for title/subtitle to avoid clipping
    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(df["date"], df["scf"], color="#1f77b4", linewidth=1.8)
    ax.scatter(df["date"], df["scf"], s=10, color="#1f77b4", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SCF")
    ax.set_xlabel("Date")
    # One major tick per month
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    # Reserve a modest top margin for headers
    top_rect = 0.90 if (title or subtitle) else 0.94
    fig.tight_layout(rect=[0.02, 0.02, 0.98, top_rect])
    if title:
        fig.text(0.5, 0.965, title, ha="center", va="top", fontsize=12)
    if subtitle:
        fig.text(0.5, 0.925, subtitle, ha="center", va="top", fontsize=10, color="#555555")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    return fig


def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point to plot SCF from `scf_summary.csv`.

    Examples
    --------
    oa-da-plot-scf path\to\scf_summary.csv \
      --output path\to\scf_summary.png \
      --title "Snow Cover Fraction for observation period" \
      --subtitle "derived from MODIS 10A1 v6 NDSI"
    """
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-scf",
        description="Plot SCF vs time from scf_summary.csv",
    )
    parser.add_argument("csv", type=Path, help="Path to scf_summary.csv")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: <csv_dir>/scf_summary.png)",
    )
    parser.add_argument(
        "--title",
        default="Snow Cover Fraction (SCF) for observation period",
        help="Plot title (default: 'Snow Cover Fraction (SCF) for observation period')",
    )
    parser.add_argument(
        "--subtitle",
        default="derived from MODIS 10A1 v6 NDSI (threshold = 0.4)",
        help="Plot subtitle (default: 'derived from MODIS 10A1 v6 NDSI (threshold = 0.4)')",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args(argv)

    # Configure logging: green timestamp | level | message
    from openamundsen_da.core.constants import LOGURU_FORMAT
    logger.remove()
    logger.add(
        sys.stdout,
        level=args.log_level.upper(),
        colorize=True,
        enqueue=True,
        format=LOGURU_FORMAT,
    )

    csv_path = Path(args.csv)
    # Read and validate CSV
    logger.info("Reading SCF summary: {}", csv_path)
    df = _load_summary(csv_path)
    if df.empty:
        logger.warning("No rows found in {}", csv_path)
    else:
        logger.info(
            "Loaded {} row(s) | date range: {} .. {}",
            len(df),
            df["date"].min().date(),
            df["date"].max().date(),
        )
    try:
        fig = _plot(df, title=args.title, subtitle=args.subtitle)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required to plot. Install it in your environment."
        ) from e

    out = Path(args.output) if args.output else csv_path.parent / "scf_summary.png"
    # Save image
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.1)
    logger.info("Wrote plot: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
