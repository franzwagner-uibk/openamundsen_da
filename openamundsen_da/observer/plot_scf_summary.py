"""
openamundsen_da.observer.plot_scf_summary

Small helper to plot SCF time series from a season-level `scf_summary.csv`.

Focus: Visualize `scf` over time (0..1). Optionally write PNG to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "date" not in df or "scf" not in df:
        raise ValueError("CSV must contain columns 'date' and 'scf'")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    return df


def _plot(df: pd.DataFrame, title: str | None = None):
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    ax.plot(df["date"], df["scf"], color="#1f77b4", linewidth=1.8)
    ax.scatter(df["date"], df["scf"], s=10, color="#1f77b4", alpha=0.8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("SCF")
    ax.set_xlabel("Date")
    if title:
        ax.set_title(title)
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
    parser.add_argument("--title", help="Optional plot title")

    args = parser.parse_args(argv)

    df = _load_summary(Path(args.csv))
    try:
        fig = _plot(df, title=args.title)
    except ModuleNotFoundError as e:
        raise SystemExit(
            "matplotlib is required to plot. Install it in your environment."
        ) from e

    out = Path(args.output) if args.output else Path(args.csv).parent / "scf_summary.png"
    fig.savefig(out, dpi=150)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

