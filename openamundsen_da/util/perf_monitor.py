"""Minimal performance monitor for season runs (CPU / RAM / disk).

When enabled, a background thread samples:
- System CPU percent
- System RAM percent/GB
- Season directory size (GB)

Outputs under `<season_dir>/plots/perf/`:
- `season_perf_metrics.csv`
- `season_perf.png` (CPU+RAM on left axis, disk on right axis)

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
    season_dir: Path
    sample_interval_sec: float = 5.0
    plot_interval_sec: float = 30.0
    run_start: datetime | None = None


def start_perf_monitor(cfg: PerfMonitorConfig) -> Event:
    """Start a background performance monitor thread."""
    stop_event = Event()
    if psutil is None:
        logger.warning("psutil is not available; performance monitoring is disabled.")
        return stop_event

    out_dir = Path(cfg.season_dir) / "plots" / "perf"
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
    season_dir = Path(cfg.season_dir).resolve()
    csv_path = out_dir / "season_perf_metrics.csv"
    png_path = out_dir / "season_perf.png"

    timestamps: List[datetime] = []
    cpu_pct: List[float] = []
    mem_pct: List[float] = []
    mem_used_gb: List[float] = []
    mem_total_gb: List[float] = []
    disk_gb: List[float] = []

    last_plot_ts: float | None = None
    run_start = cfg.run_start or datetime.utcnow()

    while not stop_event.is_set():
        now = datetime.utcnow()
        try:
            vm = psutil.virtual_memory() if psutil is not None else None  # type: ignore[assignment]
            cpu_val = psutil.cpu_percent(interval=None) if psutil is not None else 0.0  # type: ignore[assignment]
            season_bytes = _season_dir_size(season_dir)
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
        disk_gb.append(season_bytes / (1024.0 * 1024.0 * 1024.0))

        try:
            _append_csv_row(
                csv_path,
                now,
                cpu_pct[-1],
                mem_pct[-1],
                mem_used_gb[-1],
                mem_total_gb[-1],
                disk_gb[-1],
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
                        disk_gb,
                        run_start,
                    )
                    last_plot_ts = ts
                except Exception as exc:  # pragma: no cover
                    logger.warning("Performance monitor failed to update plot: {}", exc)

        stop_event.wait(cfg.sample_interval_sec)


def _season_dir_size(season_dir: Path) -> int:
    """Return total size in bytes of all files under season_dir."""
    total = 0
    for path in season_dir.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except Exception:
            continue
    return total


def _append_csv_row(
    csv_path: Path,
    t: datetime,
    cpu_total_pct: float,
    mem_used_pct: float,
    mem_used_gb: float,
    mem_total_gb: float,
    season_size_gb: float,
) -> None:
    is_new = not csv_path.exists()
    line = (
        f"{t.isoformat(timespec='seconds')},{cpu_total_pct:.3f},{mem_used_pct:.3f},"
        f"{mem_used_gb:.3f},{mem_total_gb:.3f},{season_size_gb:.3f}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("timestamp,cpu_total_pct,mem_used_pct,mem_used_gb,mem_total_gb,season_size_gb\n")
        f.write(line)


def _render_plot(
    out_path: Path,
    timestamps: List[datetime],
    cpu_pct: List[float],
    mem_pct: List[float],
    mem_used_gb: List[float],
    mem_total_gb: List[float],
    disk_gb: List[float],
    run_start: datetime,
) -> None:
    if not timestamps or plt is None:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(timestamps, cpu_pct, label="CPU [%]", color="tab:blue")
    ax1.plot(timestamps, mem_pct, label="RAM [%]", color="tab:orange")
    ax1.set_ylabel("CPU / RAM [%]")
    ax1.grid(True, alpha=0.3)

    ax2.plot(timestamps, disk_gb, label="Season disk [GB]", color="tab:green")
    ax2.set_ylabel("Disk [GB]")

    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.set_xlabel("Time")

    elapsed_hours = (timestamps[-1] - run_start).total_seconds() / 3600.0
    summary = f"Elapsed: {elapsed_hours:.2f} h   Peak RAM: {max(mem_used_gb or [0]):.2f} / {max(mem_total_gb or [0]):.2f} GB"
    fig.text(0.5, 0.94, summary, ha="center", va="top", fontsize=9)

    fig.tight_layout(rect=(0.04, 0.06, 0.96, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def cli_main(argv: List[str] | None = None) -> int:
    """Foreground performance monitor for a season directory."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-perf-monitor",
        description="Monitor CPU/RAM/disk usage for a season directory.",
    )
    p.add_argument("--season-dir", required=True, type=Path, help="Season directory (contains season.yml)")
    p.add_argument("--sample-interval", type=float, default=5.0, help="Sampling interval in seconds (default: 5)")
    p.add_argument("--plot-interval", type=float, default=30.0, help="Plot refresh interval in seconds (default: 30)")
    p.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True)

    if psutil is None:
        logger.error("psutil is required for performance monitoring but is not installed.")
        return 1

    season_dir = Path(args.season_dir)
    out_dir = season_dir / "plots" / "perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = PerfMonitorConfig(
        season_dir=season_dir,
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
