"""Minimal performance monitor for project runs (CPU / RAM / disk).

When enabled, a background thread samples:
- System CPU percent
- System RAM percent/GB
- Filesystem disk usage percent/GB
- Project directory disk usage GB, throttled to avoid expensive scans

Outputs under `<project_dir>/results/plots/perf/`:
- `project_perf_metrics.csv`
- `project_perf.png` (CPU+RAM+disk)

Dependencies: psutil (required), matplotlib (optional for plotting).
If psutil is missing, the monitor logs a warning and no files are written.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import List

from loguru import logger
from openamundsen_da.io.paths import project_plot_perf_dir
from openamundsen_da.methods.viz.common import force_figure_text_black, save_figure_png
from openamundsen_da.methods.viz.theme import FIGHEIGHT_OVERVIEW_ROW, FIGWIDTH_OVERVIEW_PAPER

try:
    import psutil  # type: ignore[import]
except Exception:  # pragma: no cover
    psutil = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]

PROJECT_PERF_FIGSIZE = (FIGWIDTH_OVERVIEW_PAPER, FIGHEIGHT_OVERVIEW_ROW * 1.4)


@dataclass(frozen=True)
class PerfMonitorConfig:
    project_dir: Path
    sample_interval_sec: float = 5.0
    plot_interval_sec: float = 30.0
    disk_scan_interval_sec: float = 300.0
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
    disk_fs_used_pct: List[float] = []
    disk_fs_used_gb: List[float] = []
    disk_fs_free_gb: List[float] = []
    disk_fs_total_gb: List[float] = []
    disk_project_used_gb: List[float] = []

    last_plot_ts: float | None = None
    last_disk_scan_ts: float | None = None
    last_project_used_gb = 0.0
    run_start = cfg.run_start or datetime.utcnow()

    while not stop_event.is_set():
        now = datetime.utcnow()
        now_ts = now.timestamp()
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

        fs_used_pct, fs_used_gb, fs_free_gb, fs_total_gb = _filesystem_disk_usage_gb(cfg.project_dir)
        disk_fs_used_pct.append(fs_used_pct)
        disk_fs_used_gb.append(fs_used_gb)
        disk_fs_free_gb.append(fs_free_gb)
        disk_fs_total_gb.append(fs_total_gb)

        disk_interval = max(0.0, float(cfg.disk_scan_interval_sec or 0.0))
        if last_disk_scan_ts is None or (now_ts - last_disk_scan_ts) >= disk_interval:
            last_project_used_gb = _directory_size_gb(cfg.project_dir)
            last_disk_scan_ts = now_ts
        disk_project_used_gb.append(last_project_used_gb)

        try:
            _append_csv_row(
                csv_path,
                now,
                cpu_pct[-1],
                mem_pct[-1],
                mem_used_gb[-1],
                mem_total_gb[-1],
                disk_fs_used_pct[-1],
                disk_fs_used_gb[-1],
                disk_fs_free_gb[-1],
                disk_fs_total_gb[-1],
                disk_project_used_gb[-1],
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
                        disk_fs_used_pct=disk_fs_used_pct,
                        disk_fs_free_gb=disk_fs_free_gb,
                        disk_project_used_gb=disk_project_used_gb,
                    )
                    last_plot_ts = ts
                except Exception as exc:  # pragma: no cover
                    logger.warning("Performance monitor failed to update plot: {}", exc)

        stop_event.wait(cfg.sample_interval_sec)


def _bytes_to_gb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0 * 1024.0)


def _filesystem_disk_usage_gb(path: Path) -> tuple[float, float, float, float]:
    try:
        usage = shutil.disk_usage(path)
    except Exception as exc:  # pragma: no cover
        logger.warning("Performance monitor disk usage sampling failed for {}: {}", path, exc)
        return 0.0, 0.0, 0.0, 0.0
    total_gb = _bytes_to_gb(usage.total)
    used_gb = _bytes_to_gb(usage.used)
    free_gb = _bytes_to_gb(usage.free)
    used_pct = (used_gb / total_gb * 100.0) if total_gb > 0 else 0.0
    return used_pct, used_gb, free_gb, total_gb


def _directory_size_gb(path: Path) -> float:
    total = 0
    try:
        for entry in os.scandir(path):
            total += _directory_entry_size_bytes(entry)
    except FileNotFoundError:
        return 0.0
    except Exception as exc:  # pragma: no cover
        logger.warning("Performance monitor project disk scan failed for {}: {}", path, exc)
        return 0.0
    return _bytes_to_gb(total)


def _directory_entry_size_bytes(entry: os.DirEntry[str]) -> int:
    try:
        if entry.is_symlink():
            return 0
        if entry.is_file(follow_symlinks=False):
            return int(entry.stat(follow_symlinks=False).st_size)
        if entry.is_dir(follow_symlinks=False):
            total = 0
            with os.scandir(entry.path) as it:
                for child in it:
                    total += _directory_entry_size_bytes(child)
            return total
    except (FileNotFoundError, PermissionError):
        return 0
    return 0


def _append_csv_row(
    csv_path: Path,
    t: datetime,
    cpu_total_pct: float,
    mem_used_pct: float,
    mem_used_gb: float,
    mem_total_gb: float,
    disk_fs_used_pct: float = 0.0,
    disk_fs_used_gb: float = 0.0,
    disk_fs_free_gb: float = 0.0,
    disk_fs_total_gb: float = 0.0,
    disk_project_used_gb: float = 0.0,
) -> None:
    is_new = not csv_path.exists()
    line = (
        f"{t.isoformat(timespec='seconds')},{cpu_total_pct:.3f},{mem_used_pct:.3f},"
        f"{mem_used_gb:.3f},{mem_total_gb:.3f},"
        f"{disk_fs_used_pct:.3f},{disk_fs_used_gb:.3f},{disk_fs_free_gb:.3f},"
        f"{disk_fs_total_gb:.3f},{disk_project_used_gb:.3f}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "timestamp,cpu_total_pct,mem_used_pct,mem_used_gb,mem_total_gb,"
                "disk_fs_used_pct,disk_fs_used_gb,disk_fs_free_gb,"
                "disk_fs_total_gb,disk_project_used_gb\n"
            )
        f.write(line)


def _render_plot(
    out_path: Path,
    timestamps: List[datetime],
    cpu_pct: List[float],
    mem_pct: List[float],
    mem_used_gb: List[float],
    mem_total_gb: List[float],
    run_start: datetime,
    disk_fs_used_pct: List[float] | None = None,
    disk_fs_free_gb: List[float] | None = None,
    disk_project_used_gb: List[float] | None = None,
) -> None:
    if not timestamps or plt is None:
        return

    fig, ax1 = plt.subplots(figsize=PROJECT_PERF_FIGSIZE)

    ax1.plot(timestamps, cpu_pct, label="CPU [%]", color="#46307e")
    ax1.plot(timestamps, mem_pct, label="RAM [%]", color="#355f8d")
    if disk_fs_used_pct:
        ax1.plot(timestamps, disk_fs_used_pct, label="Disk used [%]", color="#24868e")
    ax1.set_ylabel("CPU / RAM / disk used [%]")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    if disk_project_used_gb:
        ax2.plot(timestamps, disk_project_used_gb, label="Project size [GB]", color="#26ad81", linestyle="-")
    if disk_fs_free_gb:
        ax2.plot(timestamps, disk_fs_free_gb, label="Disk free [GB]", color="#6ece58", linestyle="--")
    ax2.set_ylabel("Disk [GB]")
    ax2.set_ylim(bottom=0)

    lines = [*ax1.get_lines(), *ax2.get_lines()]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=8, ncol=2)
    ax1.set_xlabel("Time")

    elapsed_sec = max(0, int((timestamps[-1] - run_start).total_seconds()))
    hh, rem = divmod(elapsed_sec, 3600)
    mm = rem // 60
    elapsed_hhmm = f"{hh:02d}:{mm:02d}"
    min_disk_free = min(disk_fs_free_gb) if disk_fs_free_gb else 0.0
    peak_project_size = max(disk_project_used_gb) if disk_project_used_gb else 0.0
    summary = (
        f"Elapsed: {elapsed_hhmm}   "
        f"Peak RAM: {max(mem_used_gb or [0]):.2f} / {max(mem_total_gb or [0]):.2f} GB   "
        f"Peak project: {peak_project_size:.2f} GB   Min free disk: {min_disk_free:.2f} GB"
    )
    fig.text(0.5, 0.985, summary, ha="center", va="top", fontsize=9)

    fig.tight_layout(rect=(0.005, 0.03, 0.995, 0.91))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    force_figure_text_black(fig, [ax1, ax2])
    _save_perf_plot_atomic(fig, out_path)
    plt.close(fig)


def _save_perf_plot_atomic(fig, out_path: Path) -> None:
    """Replace the performance PNG only after a complete image was written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=out_path.parent,
            prefix=f".{out_path.name}.",
            suffix=".tmp.png",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        save_figure_png(fig, tmp_path)
        os.replace(tmp_path, out_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


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
    p.add_argument(
        "--disk-scan-interval",
        type=float,
        default=300.0,
        help="Recursive project directory disk scan interval in seconds (default: 300)",
    )
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
        disk_scan_interval_sec=float(args.disk_scan_interval or 300.0),
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
