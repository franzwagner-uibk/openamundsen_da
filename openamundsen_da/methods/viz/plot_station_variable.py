"""openamundsen_da.methods.viz.plot_station_variable

Small, manual CLI to plot a single variable from a single station
results CSV (e.g. point_latschbloder.csv) as a time series.

Typical usage
-------------

    python -m openamundsen_da.methods.viz.plot_station_variable \\
        path/to/point_latschbloder.csv \\
        --var swe \\
        --start-date 2020-03-01 \\
        --end-date 2020-07-31

This is intentionally simpler than the ensemble/season plotting tools:
you point it to one CSV, choose the column, and get one figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.util.ts import apply_window, read_timeseries_csv


def _pretty_var_title(var: str, label: str = "", units: str = "") -> str:
    """Return a friendly variable title with units for subtitles and y-labels."""
    v = (var or "").strip()
    if not label and not units:
        # A few common shortcuts
        lv = v.lower()
        if lv == "swe":
            return "snow water equivalent [mm]"
        if lv in ("snow_depth", "snowdepth", "hs"):
            return "snow depth [m]"
    base = label.strip() if label else v.replace("_", " ")
    if units:
        return f"{base} [{units}]"
    return base


def _load_series(csv_path: Path, time_col: str, var_col: str) -> pd.Series:
    """Load a single variable time series from a station CSV."""
    df = read_timeseries_csv(csv_path, time_col, [var_col])
    if var_col not in df.columns:
        raise KeyError(f"Column '{var_col}' not found in {csv_path.name}")
    return df[var_col].dropna()


def plot_station_variable(
    *,
    csv_path: Path,
    time_col: str = "time",
    var_col: str,
    var_label: str = "",
    var_units: str = "",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    backend: str = "Agg",
) -> Path:
    """Plot a single variable from one station CSV.

    Parameters
    ----------
    csv_path : Path
        Path to the station results CSV (e.g. point_latschbloder.csv).
    time_col : str, optional
        Timestamp column name (default: "time").
    var_col : str
        Column to plot (e.g. "swe", "snow_depth", "temp").
    var_label : str, optional
        Pretty label for y-axis/title (defaults to var_col).
    var_units : str, optional
        Units appended in brackets to the label.
    start_date, end_date : str, optional
        Optional ISO dates ("YYYY-MM-DD") to clip the series.
    backend : str, optional
        Matplotlib backend, default "Agg" (headless).
    """
    import matplotlib

    matplotlib.use(backend or "Agg")
    import matplotlib.pyplot as plt

    csv_path = Path(csv_path)
    series = _load_series(csv_path, time_col, var_col)

    # Optional windowing
    if start_date or end_date:
        df = series.to_frame(var_col)
        df = apply_window(df, start_date, end_date)
        series = df[var_col].dropna()

    if series.empty:
        raise ValueError(f"No data for '{var_col}' in selected time window.")

    var_title = _pretty_var_title(var_col, var_label, var_units)

    fig, ax = plt.subplots(figsize=(10.0, 4.2))
    ax.plot(series.index, series.values, color="#1f77b4", lw=1.8)
    ax.set_xlabel("Time")
    ax.set_ylabel(var_title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    token = csv_path.stem
    title = f"Station variable | {token}"
    subtitle = f"{var_title}"
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=12)
    fig.text(0.5, 0.93, subtitle, ha="center", va="top", fontsize=10, color="#555555")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.90])

    out_path = csv_path.with_suffix(f".{var_col}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out_path


def cli_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-station-var",
        description="Plot a single variable from a station results CSV.",
    )
    parser.add_argument("csv", type=Path, help="Path to station results CSV (e.g. point_latschbloder.csv)")
    parser.add_argument("--time-col", default="time", help="Timestamp column name in CSV (default: time)")
    parser.add_argument("--var", required=True, help="Column to plot (e.g. swe, snow_depth, temp)")
    parser.add_argument("--var-label", default="", help="Pretty label for y-axis/title (defaults to var name)")
    parser.add_argument("--var-units", default="", help="Units appended to label in brackets")
    parser.add_argument("--start-date", type=str, help="Optional start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="Optional end date (YYYY-MM-DD)")
    parser.add_argument("--backend", default="Agg", help="Matplotlib backend (default: Agg)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(list(argv) if argv is not None else None)

    logger.remove()
    logger.add(
        lambda msg: print(msg, end=""),
        level=args.log_level.upper(),
        colorize=True,
        enqueue=False,
        format=LOGURU_FORMAT,
    )

    try:
        out = plot_station_variable(
            csv_path=args.csv,
            time_col=args.time_col,
            var_col=args.var,
            var_label=args.var_label,
            var_units=args.var_units,
            start_date=args.start_date,
            end_date=args.end_date,
            backend=args.backend,
        )
        logger.info("Wrote {}", out)
        return 0
    except Exception as exc:  # keep CLI failure concise
        logger.error("Failed to plot {}: {}", args.csv, exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())

