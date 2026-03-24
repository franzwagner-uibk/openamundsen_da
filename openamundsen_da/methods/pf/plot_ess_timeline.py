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
from openamundsen_da.io.paths import list_step_dirs, list_steps_sorted, read_step_config
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.methods.viz._utils import (
    apply_fraction_grid,
    draw_assim_labels,
    draw_assimilation_vlines,
    force_figure_text_black,
    save_figure_png,
    set_matplotlib_text_black,
)


_RE_DATE = re.compile(r"weights_.+_(\d{8})\.csv$", re.IGNORECASE)
_ESS_PANEL_FIGSIZE = (7.2876875, 2.28)
_ASSIM_LABEL_ROW_OFFSETS_PTS = [2.0, 8.0]
_ASSIM_LABEL_MIN_SPACING_DAYS = 18.0


def ess_title(*, ensemble_size: int | None, normalized: bool = False) -> str:
    base = "effective sample size"
    if normalized:
        base = "effective sample size ratio"
    if ensemble_size is None or ensemble_size <= 0:
        return base
    return f"{base} (ensemble size = {ensemble_size})"


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


def load_setup_ess_series(setup_dir: Path) -> pd.DataFrame:
    setup_dir = Path(setup_dir)
    files: list[tuple[datetime, Path]] = []
    for step in list_step_dirs(setup_dir):
        assim_dir = step / "assim"
        if not assim_dir.is_dir():
            continue
        files.extend(_scan_weights(assim_dir))
    if not files:
        raise FileNotFoundError(f"No weights_*_*.csv found under steps in {setup_dir}")
    return _compute_series(files)


def _parse_date_opt(text: str | None) -> datetime | None:
    if not text:
        return None
    t = str(text).replace("_", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def _setup_time_bounds(setup_dir: Path) -> tuple[datetime | None, datetime | None]:
    starts: list[datetime] = []
    ends: list[datetime] = []
    for step_dir in list_steps_sorted(setup_dir):
        cfg = read_step_config(step_dir) or {}
        start = _parse_date_opt(cfg.get("start_date"))
        end = _parse_date_opt(cfg.get("end_date"))
        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
    return (min(starts) if starts else None, max(ends) if ends else None)


def _assimilation_event_dates(setup_dir: Path) -> list[pd.Timestamp]:
    return [pd.Timestamp(ev.date) for ev in load_assimilation_events(setup_dir)]


def _add_assim_label_axis(ax, dates: list[pd.Timestamp]) -> None:
    import matplotlib.dates as mdates

    if not dates:
        return
    x_min, x_max = sorted(ax.get_xlim())
    visible_start = pd.Timestamp(mdates.num2date(x_min)).tz_localize(None)
    visible_end = pd.Timestamp(mdates.num2date(x_max)).tz_localize(None)
    visible_items = [
        (date, str(i))
        for i, date in enumerate(dates, start=1)
        if visible_start <= date <= visible_end
    ]
    if not visible_items:
        return

    label_axis = ax.twiny()
    label_axis.patch.set_alpha(0.0)
    if hasattr(label_axis, "set_in_layout"):
        label_axis.set_in_layout(False)
    label_axis.set_xlim(ax.get_xlim())
    label_axis.set_xticks([])
    label_axis.set_xlabel("")
    label_axis.yaxis.set_visible(False)
    label_axis.xaxis.set_visible(False)
    for spine in label_axis.spines.values():
        spine.set_visible(False)

    draw_assim_labels(
        label_axis,
        [item[0] for item in visible_items],
        labels=[item[1] for item in visible_items],
        max_labels=max(1, len(visible_items)),
        y_offset_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS[0],
        fontsize=6.0,
        color="#000000",
        rotation=0.0,
        va="bottom",
        row_y_offsets_pts=_ASSIM_LABEL_ROW_OFFSETS_PTS,
        min_row_spacing_days=_ASSIM_LABEL_MIN_SPACING_DAYS,
        axes_y=1.0,
        ha="center",
        x_offset_pts=0.0,
    )


def _apply_result_like_time_axis_labels(ax) -> None:
    import matplotlib.dates as mdates

    locator = mdates.MonthLocator()
    formatter = mdates.DateFormatter("%b")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    x_min, x_max = sorted(ax.get_xlim())
    tick_values = locator.tick_values(mdates.num2date(x_min), mdates.num2date(x_max))
    tick_dates = [pd.Timestamp(mdates.num2date(val)).tz_localize(None) for val in tick_values]
    if not tick_dates:
        return

    labels: list[str] = []
    prev_year: int | None = None
    for idx, tick_dt in enumerate(tick_dates):
        if idx == 0 or tick_dt.year != prev_year:
            labels.append(tick_dt.strftime("%b\n%Y"))
        else:
            labels.append(tick_dt.strftime("%b"))
        prev_year = tick_dt.year
    ax.set_xticks(tick_values)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", labelsize=8.4)


def _plot(
    df: pd.DataFrame,
    normalized: bool,
    threshold: float | None,
    title: str,
    subtitle: str | None,
    *,
    ensemble_size: int | None = None,
    x_bounds: tuple[datetime | None, datetime | None] | None = None,
    assim_dates: list[pd.Timestamp] | None = None,
    backend: str = "Agg",
):
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt

    ycol = "ess_norm" if normalized else "ess"
    fig, ax = plt.subplots(figsize=_ESS_PANEL_FIGSIZE)
    ax.plot(df["date"], df[ycol], marker="o", lw=1.8, color="#1f77b4")
    ax.set_ylabel("ESS/N" if normalized else "ESS", fontsize=8.6)
    ax.set_xlabel("")
    ax.set_title(title, loc="left", fontsize=9.4, pad=16.0 if assim_dates else 9.0)
    apply_fraction_grid(ax, y_step=None)
    if x_bounds is not None:
        x_start, x_end = x_bounds
        if x_start is not None and x_end is not None:
            ax.set_xlim(x_start, x_end)
    if ensemble_size is not None and ensemble_size > 0:
        ax.set_ylim(0.0, float(ensemble_size))
    _apply_result_like_time_axis_labels(ax)
    ax.tick_params(axis="y", labelsize=8.4)
    if threshold is not None:
        ax.axhline(threshold, color="#d62728", lw=1.2, ls="--")
    if assim_dates:
        draw_assimilation_vlines(ax, assim_dates, color="#777777", ls="--", lw=1.0, alpha=0.9, label="_nolegend_", zorder=20)
        _add_assim_label_axis(ax, assim_dates)

    fig.subplots_adjust(left=0.11, right=0.965, top=0.82 if assim_dates else 0.86, bottom=0.30)
    force_figure_text_black(fig, [ax])
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
    df = load_setup_ess_series(setup_dir)
    x_bounds = _setup_time_bounds(setup_dir)
    assim_dates = _assimilation_event_dates(setup_dir)
    ensemble_size = int(df["n"].iloc[0]) if "n" in df.columns and not df.empty else None
    fig = _plot(
        df,
        normalized=normalized,
        threshold=threshold,
        title=ess_title(ensemble_size=ensemble_size, normalized=normalized),
        subtitle=None,
        ensemble_size=ensemble_size,
        x_bounds=x_bounds,
        assim_dates=assim_dates,
        backend=backend,
    )
    setup_id = _setup_id_from_dir(setup_dir)
    out_dir = setup_dir / "plots" / "assim" / "ess"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"setup_ess_timeline_{setup_id}.png"
    save_figure_png(fig, out, bbox_inches="tight", pad_inches=0.08)
    return out


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-ess", description="Plot ESS over time from weights_*_*.csv files")
    p.add_argument("--step-dir", type=Path, help="Step directory containing 'assim' folder")
    p.add_argument("--assim-dir", type=Path, help="Assimilation directory (default: <step-dir>/assim)")
    p.add_argument("--normalized", action="store_true", help="Plot ESS/N instead of ESS")
    p.add_argument("--threshold", type=float, help="Draw horizontal reference line (ESS/N if --normalized else ESS)")
    p.add_argument("--output", type=Path, help="Output PNG path (default: <assim-dir>/ess_timeline.png)")
    p.add_argument("--title", default="", help="Plot title")
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
    ensemble_size = int(df["n"].iloc[0]) if "n" in df.columns and not df.empty else None
    logger.info("Computed ESS for {} date(s) (normalized={}): {}", len(df), bool(args.normalized), 
                ", ".join(d.strftime("%Y-%m-%d") for d in df["date"]))
    try:
        fig = _plot(
            df,
            normalized=bool(args.normalized),
            threshold=args.threshold,
            title=(args.title or ess_title(ensemble_size=ensemble_size, normalized=bool(args.normalized))),
            subtitle=(args.subtitle or None),
            ensemble_size=ensemble_size,
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
        save_figure_png(fig, out, bbox_inches="tight", pad_inches=0.08)
    except Exception as e:
        logger.error(f"Saving PNG failed: {e}")
        return 5
    logger.info("Wrote plot: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
