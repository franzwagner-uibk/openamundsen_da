"""ESS timeline plot for setup-level assimilation diagnostics.

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
from openamundsen_da.io.paths import (
    find_project_yaml,
    list_step_dirs,
    list_steps_sorted,
    project_plot_assim_ess_dir,
    read_step_config,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.yaml_utils import read_yaml_mapping
from openamundsen_da.methods.viz.plots.common import (
    add_assim_label_axis,
    apply_fraction_grid,
    apply_month_interval_axis_labels,
    draw_assimilation_vlines,
    result_title_pad,
)
from openamundsen_da.methods.viz.common import (
    force_figure_text_black,
    save_figure_png,
    set_matplotlib_text_black,
)


_RE_DATE = re.compile(r"weights_.+_(\d{8})\.csv$", re.IGNORECASE)
_ESS_PANEL_FIGSIZE = (7.2876875, 2.28)
def ess_title(*, ensemble_size: int | None, normalized: bool = False) -> str:
    if normalized:
        return "Effective sample size ratio"
    return "Effective sample size"


def ess_axis_ticks(ensemble_size: int | None, *, threshold: float | None = None) -> list[float]:
    if ensemble_size is None or ensemble_size <= 0:
        return []
    upper = float(ensemble_size)
    step = next((candidate for candidate in (1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0) if np.ceil(upper / candidate) <= 5.0), 100.0)
    ticks = [0.0]
    current = step
    while current < upper:
        ticks.append(float(current))
        current += step
    ticks.append(upper)
    return sorted(set(ticks))


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
    add_assim_label_axis(ax, dates)


def _apply_result_like_time_axis_labels(ax) -> None:
    apply_month_interval_axis_labels(ax)


def _apply_ess_ticks(ax, ensemble_size: int | None, *, threshold: float | None = None) -> None:
    ticks = ess_axis_ticks(ensemble_size, threshold=threshold)
    if ticks:
        ax.set_yticks(ticks)


def _add_ess_threshold_legend(ax) -> None:
    from matplotlib.lines import Line2D

    legend = ax.legend(
        handles=[Line2D([0], [0], color="black", lw=0.9, ls="--", label="ESS threshold")],
        loc="upper right",
        frameon=False,
        fontsize=6.2,
        handlelength=1.8,
        handletextpad=0.35,
        labelspacing=0.2,
        borderpad=0.0,
        borderaxespad=0.35,
    )
    legend.set_zorder(40)


def load_setup_ess_threshold(setup_dir: Path, *, ensemble_size: int | None) -> float | None:
    if ensemble_size is None or ensemble_size <= 0:
        return None
    try:
        cfg = read_yaml_mapping(find_project_yaml(setup_dir), error_cls=RuntimeError, context="Project YAML root")
    except Exception:
        return None

    da_cfg = cfg.get("data_assimilation") or {}
    resampling_cfg = da_cfg.get("resampling") or cfg.get("resampling") or {}
    ratio_raw = resampling_cfg.get("ess_threshold_ratio")
    abs_raw = resampling_cfg.get("ess_threshold")

    if ratio_raw is not None:
        try:
            ratio = float(ratio_raw)
        except (TypeError, ValueError):
            ratio = None
        if ratio is not None and ratio > 0:
            return ratio * float(ensemble_size)

    if abs_raw is None:
        return None
    try:
        absolute = float(abs_raw)
    except (TypeError, ValueError):
        return None
    if absolute <= 0:
        return None
    if absolute <= 1.0:
        return absolute * float(ensemble_size)
    return absolute


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
    ax.plot(df["date"], df[ycol], marker="o", ms=4.0, lw=0.0, ls="none", color="#000000", zorder=25)
    ax.set_ylabel("ESS/N" if normalized else "ESS", fontsize=8.6)
    ax.set_xlabel("")
    ax.set_title(title, loc="left", fontsize=9.4, pad=result_title_pad(bool(assim_dates)))
    apply_fraction_grid(ax, y_step=None)
    if x_bounds is not None:
        x_start, x_end = x_bounds
        if x_start is not None and x_end is not None:
            ax.set_xlim(x_start, x_end)
    if ensemble_size is not None and ensemble_size > 0 and not normalized:
        ax.set_ylim(0.0, float(ensemble_size))
        _apply_ess_ticks(ax, ensemble_size, threshold=threshold)
    _apply_result_like_time_axis_labels(ax)
    ax.tick_params(axis="y", labelsize=8.4)
    if threshold is not None:
        ax.axhline(threshold, color="black", lw=0.9, ls="--")
        _add_ess_threshold_legend(ax)
    if assim_dates:
        draw_assimilation_vlines(ax, assim_dates, color="#777777", ls="--", lw=1.0, alpha=0.9, label="_nolegend_", zorder=20)
        _add_assim_label_axis(ax, assim_dates)

    if subtitle:
        fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=8.8, color="#000000")
    top_margin = 0.82 if assim_dates else 0.86
    if subtitle:
        top_margin -= 0.06
    fig.subplots_adjust(left=0.11, right=0.965, top=top_margin, bottom=0.30)
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
    <project_dir>/results/plots/assim/ess/setup_ess_timeline_<setup_id>.png.
    """
    setup_dir = Path(setup_dir)
    df = load_setup_ess_series(setup_dir)
    x_bounds = _setup_time_bounds(setup_dir)
    assim_dates = _assimilation_event_dates(setup_dir)
    ensemble_size = int(df["n"].iloc[0]) if "n" in df.columns and not df.empty else None
    threshold = load_setup_ess_threshold(setup_dir, ensemble_size=ensemble_size) if threshold is None else threshold
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
    out_dir = project_plot_assim_ess_dir(setup_dir)
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
