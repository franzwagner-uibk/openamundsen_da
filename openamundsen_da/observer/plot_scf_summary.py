"""
openamundsen_da.observer.plot_scf_summary

Small helper to plot SCF time series from a season-level `scf_summary.csv`.

Focus: Visualize `scf` over time (0..1). Optionally write PNG to disk.
Logs basic information (rows, date range, output path).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from loguru import logger


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df or "scf" not in df:
        raise ValueError("CSV must contain columns 'date' and 'scf'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def _plot(df: pd.DataFrame, title: str | None = None, subtitle: str | None = None):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(df["date"], df["scf"], color="#1f77b4", linewidth=1.8)
    ax.scatter(df["date"], df["scf"], s=10, color="#1f77b4", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SCF")
    ax.set_xlabel("Date")
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    if subtitle:
        ax.set_title(subtitle, fontsize=10)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    return fig


def cli_main(argv: list[str] | None = None) -> int:
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
        default="Snow Cover Fraction for observation period",
        help="Plot title (default: 'Snow Cover Fraction for observation period')",
    )
    parser.add_argument(
        "--subtitle",
        default="derived from MODIS 10A1 v6 NDSI",
        help="Plot subtitle (default: 'derived from MODIS 10A1 v6 NDSI')",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (default: INFO)",
    )

    args = parser.parse_args(argv)

    # configure logging to console
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=args.log_level.upper())

    csv_path = Path(args.csv)
    logger.info("Reading SCF summary: {}\n", csv_path)
    df = _load_summary(csv_path)
    if df.empty:
        logger.warning("No rows found in {}\n", csv_path)
    else:
        logger.info(
            "Loaded {} row(s) | date range: {} .. {}\n",
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
    fig.savefig(out, dpi=150)
    logger.info("Wrote plot: {}\n", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
