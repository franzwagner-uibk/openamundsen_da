"""Minimal performance monitor for project runs (CPU / RAM).

When enabled, a background thread samples:
- System CPU percent
- System RAM percent/GB

Outputs under `<project_dir>/results/plots/perf/`:
- `project_perf_metrics.csv`
- `project_perf.png` (CPU+RAM)

Dependencies: psutil (required), matplotlib (optional for plotting).
If psutil is missing, the monitor logs a warning and no files are written.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import List

from loguru import logger
from openamundsen_da.io.paths import project_plot_perf_dir
from openamundsen_da.methods.viz.common import force_figure_text_black, save_figure_png

try:
    import psutil  # type: ignore[import]
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PerfMonitorConfig:
    project_dir: Path
    sample_interval_sec: float = 5.0
    plot_interval_sec: float = 30.0
    run_start: datetime | None = None


def start_perf_monitor(cfg: PerfMonitorConfig) -> Event:
    """Start a background performance monitor thread."""
    stop_event = Event()
    if psutil is None:
        logger.warning("psutil is not available; performance monitoring is disabled.")
        return stop_event

    out_dir = project_plot_perf_dir(cfg.project_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover
        logger.warning("Could not create perf monitor output directory {}: {}", out_dir, exc)
        return stop_event

    thread = Thread(
        target=_monitor_loop,
        args=(cfg, out_dir, stop_event),
        name="oa-da-perf-monitor",
        daemon=True,
    )
    thread.start()
    logger.info("Performance monitor started -> {}", out_dir)
    return stop_event


def _monitor_loop(cfg: PerfMonitorConfig, out_dir: Path, stop_event: Event) -> None:
    csv_path = out_dir / "project_perf_metrics.csv"
    png_path = out_dir / "project_perf.png"

    timestamps: List[datetime] = []
    cpu_pct: List[float] = []
    mem_pct: List[float] = []
    mem_used_gb: List[float] = []
    mem_total_gb: List[float] = []

    last_plot_ts: float | None = None
    run_start = cfg.run_start or datetime.utcnow()

    while not stop_event.is_set():
        now = datetime.utcnow()
        try:
            vm = psutil.virtual_memory() if psutil is not None else None  # type: ignore[assignment]
            cpu_val = psutil.cpu_percent(interval=None) if psutil is not None else 0.0  # type: ignore[assignment]
        except Exception as exc:  # pragma: no cover
            logger.warning("Performance monitor sampling failed: {}", exc)
            break

        timestamps.append(now)
        cpu_pct.append(float(cpu_val))
        if vm is not None:
            mem_pct.append(float(vm.percent))
            mem_used_gb.append(vm.used / (1024.0 * 1024.0 * 1024.0))
            mem_total_gb.append(vm.total / (1024.0 * 1024.0 * 1024.0))
        else:
            mem_pct.append(0.0)
            mem_used_gb.append(0.0)
            mem_total_gb.append(0.0)

        try:
            _append_csv_row(
                csv_path,
                now,
                cpu_pct[-1],
                mem_pct[-1],
                mem_used_gb[-1],
                mem_total_gb[-1],
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Performance monitor failed to update CSV {}: {}", csv_path, exc)

        if plt is not None:
            ts = now.timestamp()
            if last_plot_ts is None or (ts - last_plot_ts) >= cfg.plot_interval_sec:
                try:
                    _render_plot(
                        png_path,
                        timestamps,
                        cpu_pct,
                        mem_pct,
                        mem_used_gb,
                        mem_total_gb,
                        run_start,
                    )
                    last_plot_ts = ts
                except Exception as exc:  # pragma: no cover
                    logger.warning("Performance monitor failed to update plot: {}", exc)

        stop_event.wait(cfg.sample_interval_sec)


def _append_csv_row(
    csv_path: Path,
    t: datetime,
    cpu_total_pct: float,
    mem_used_pct: float,
    mem_used_gb: float,
    mem_total_gb: float,
) -> None:
    is_new = not csv_path.exists()
    line = (
        f"{t.isoformat(timespec='seconds')},{cpu_total_pct:.3f},{mem_used_pct:.3f},"
        f"{mem_used_gb:.3f},{mem_total_gb:.3f}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp,cpu_total_pct,mem_used_pct,mem_used_gb,mem_total_gb\n")
        f.write(line)


def _render_plot(
    out_path: Path,
    timestamps: List[datetime],
    cpu_pct: List[float],
    mem_pct: List[float],
    mem_used_gb: List[float],
    mem_total_gb: List[float],
    run_start: datetime,
) -> None:
    if not timestamps or plt is None:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(timestamps, cpu_pct, label="CPU [%]", color="tab:blue")
    ax1.plot(timestamps, mem_pct, label="RAM [%]", color="tab:orange")
    ax1.set_ylabel("CPU / RAM [%]")
    ax1.grid(True, alpha=0.3)

    ax1.legend(loc="upper left", fontsize=8)
    ax1.set_xlabel("Time")

    elapsed_sec = max(0, int((timestamps[-1] - run_start).total_seconds()))
    hh, rem = divmod(elapsed_sec, 3600)
    mm = rem // 60
    elapsed_hhmm = f"{hh:02d}:{mm:02d}"
    summary = (
        f"Elapsed: {elapsed_hhmm}   "
        f"Peak RAM: {max(mem_used_gb or [0]):.2f} / {max(mem_total_gb or [0]):.2f} GB"
    )
    fig.text(0.5, 0.94, summary, ha="center", va="top", fontsize=9)

    fig.tight_layout(rect=(0.04, 0.06, 0.96, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    force_figure_text_black(fig, [ax1])
    save_figure_png(fig, out_path)
    plt.close(fig)


def cli_main(argv: List[str] | None = None) -> int:
    """Foreground performance monitor for a project directory."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-perf-monitor",
        description="Monitor CPU/RAM usage for a project directory.",
    )
    p.add_argument("--project-dir", required=True, type=Path, help="Project directory (contains steps/)")
    p.add_argument("--sample-interval", type=float, default=5.0, help="Sampling interval in seconds (default: 5)")
    p.add_argument("--plot-interval", type=float, default=30.0, help="Plot refresh interval in seconds (default: 30)")
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True)

    if psutil is None:
        logger.error("psutil is required for performance monitoring but is not installed.")
        return 1

    project_dir = Path(args.project_dir)
    out_dir = project_plot_perf_dir(project_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PerfMonitorConfig(
        project_dir=project_dir,
        sample_interval_sec=float(args.sample_interval or 5.0),
        plot_interval_sec=float(args.plot_interval or 30.0),
        run_start=datetime.utcnow(),
    )
    stop_event = Event()
    try:
        _monitor_loop(cfg, out_dir, stop_event)
        return 0
    except KeyboardInterrupt:
        logger.info("Performance monitor interrupted by user.")
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
