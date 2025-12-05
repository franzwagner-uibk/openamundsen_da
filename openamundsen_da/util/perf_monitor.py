"""Performance monitoring utilities for season runs (CPU / RAM / disk).

This module provides:

- A background monitor used by the season pipeline when enabled via a flag.
- A simple CLI (`oa-da-perf-monitor`) for manual monitoring.

The monitor:
- Tracks CPU and RSS memory of processes related to a given season directory.
- Tracks system memory usage (used vs. total).
- Tracks total size of the season directory on disk.
- Writes a CSV log and periodically refreshes a PNG plot under
  ``<season_dir>/plots/perf/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from threading import Event, Thread
from typing import Iterable, List, Sequence

from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, find_season_yaml
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.ts import parse_datetime_opt

try:  # psutil is required for process metrics
    import psutil  # type: ignore[import]
except Exception:  # pragma: no cover - defensive
    psutil = None  # type: ignore[assignment]

try:  # matplotlib is optional for plotting
    import matplotlib.pyplot as plt  # type: ignore[import]
except Exception:  # pragma: no cover - defensive
    plt = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PerfMonitorConfig:
    season_dir: Path
    sample_interval_sec: float = 5.0
    plot_interval_sec: float = 30.0
    process_name_filters: Sequence[str] = (
        "python",
        "openamundsen",
        "openamundsen_da",
    )
    roi_area_km2: float | None = None
    resolution_m: float | None = None
    timestep: str | None = None
    season_days: int | None = None
    num_da_dates: int | None = None
    num_workers: int | None = None
    ensemble_size: int | None = None
    run_start: datetime | None = None
    tz_offset_hours: float | None = None


def start_perf_monitor(cfg: PerfMonitorConfig) -> Event:
    """Start a background performance monitor thread.

    Returns an Event that can be set to request shutdown.
    If psutil is not available, logs a warning and returns an unused Event.
    """
    if psutil is None:
        logger.warning("psutil is not available; performance monitoring is disabled.")
        return Event()

    season_dir = Path(cfg.season_dir)
    out_dir = season_dir / "plots" / "perf"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not create perf monitor output directory {}: {}", out_dir, exc)

    stop_event = Event()
    thread = Thread(
        target=_monitor_loop,
        args=(cfg, out_dir, stop_event),
        name="oa-da-perf-monitor",
        daemon=True,
    )
    thread.start()
    logger.info("Performance monitor started for season {} -> {}", season_dir.name, out_dir)
    return stop_event


def _monitor_loop(cfg: PerfMonitorConfig, out_dir: Path, stop_event: Event) -> None:
    season_dir = Path(cfg.season_dir).resolve()
    season_dir_str = str(season_dir)

    timestamps: List[datetime] = []
    cpu_tracked: List[float] = []
    cpu_total_pct: List[float] = []
    mem_tracked_mb: List[float] = []
    mem_used_gb: List[float] = []
    mem_total_gb: List[float] = []
    mem_used_pct: List[float] = []
    disk_gb: List[float] = []

    csv_path = out_dir / "season_perf_metrics.csv"
    last_plot_ts: float | None = None
    run_start = cfg.run_start

    while not stop_event.is_set():
        now = datetime.utcnow()
        if run_start is None:
            run_start = now

        try:
            cpu_total = psutil.cpu_percent(interval=None) if psutil is not None else 0.0  # type: ignore[assignment]
            cpu_sum, mem_sum_bytes = _sample_tracked_processes(season_dir_str, cfg.process_name_filters)
            vm = psutil.virtual_memory() if psutil is not None else None  # type: ignore[assignment]
            season_bytes = _season_dir_size(season_dir)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Performance monitor sampling failed: {}", exc)
            break

        timestamps.append(now)
        cpu_tracked.append(cpu_sum)
        cpu_total_pct.append(float(cpu_total))
        mem_tracked_mb.append(mem_sum_bytes / (1024.0 * 1024.0))
        if vm is not None:
            used_gb = vm.used / (1024.0 * 1024.0 * 1024.0)
            total_gb = vm.total / (1024.0 * 1024.0 * 1024.0)
            mem_used_gb.append(used_gb)
            mem_total_gb.append(total_gb)
            mem_used_pct.append(float(vm.percent))
        else:  # pragma: no cover - defensive
            mem_used_gb.append(0.0)
            mem_total_gb.append(0.0)
            mem_used_pct.append(0.0)
        disk_gb.append(season_bytes / (1024.0 * 1024.0 * 1024.0))

        elapsed_sec = (now - run_start).total_seconds()

        # Progress / ETA based on completed steps vs total steps (linear in time)
        eta_utc: datetime | None = None
        eta_local: datetime | None = None
        progress_steps = 0.0
        done_steps = 0
        total_steps = 0
        try:
            total_steps, done_steps = _compute_step_progress(season_dir)
            if total_steps > 0:
                progress_steps = done_steps / float(total_steps)
                if 0.0 < progress_steps < 1.0:
                    expected_total_sec = elapsed_sec / progress_steps
                    eta_utc = run_start + timedelta(seconds=expected_total_sec)
                    if cfg.tz_offset_hours is not None:
                        eta_local = eta_utc + timedelta(hours=cfg.tz_offset_hours)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Performance monitor: failed to compute progress/ETA: {}", exc)

        try:
            _append_csv_row(
                csv_path,
                now,
                cpu_tracked[-1],
                cpu_total_pct[-1],
                mem_tracked_mb[-1],
                mem_used_gb[-1],
                mem_total_gb[-1],
                mem_used_pct[-1],
                disk_gb[-1],
                elapsed_sec,
                cfg,
                progress_steps,
                done_steps,
                total_steps,
                eta_utc,
                eta_local,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Performance monitor failed to update CSV {}: {}", csv_path, exc)

        if plt is not None:
            ts = now.timestamp()
            if last_plot_ts is None or (ts - last_plot_ts) >= cfg.plot_interval_sec:
                try:
                    _render_plot(
                        out_dir / "season_perf.png",
                        timestamps,
                        cpu_tracked,
                        cpu_total_pct,
                        mem_tracked_mb,
                        mem_used_gb,
                        mem_total_gb,
                        mem_used_pct,
                        disk_gb,
                        cfg,
                        run_start,
                    )
                    last_plot_ts = ts
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Performance monitor failed to update plot: {}", exc)

        stop_event.wait(cfg.sample_interval_sec)


def _sample_tracked_processes(
    season_dir_str: str,
    name_filters: Sequence[str],
) -> tuple[float, int]:
    """Return (cpu_percent_sum, rss_bytes_sum) for relevant processes."""
    assert psutil is not None  # for type checkers

    cpu_sum = 0.0
    mem_sum = 0
    filters_lower = [s.lower() for s in name_filters]

    for proc in psutil.process_iter(["pid", "name", "cmdline", "cwd"]):
        try:
            if not _is_relevant_process(proc, season_dir_str, filters_lower):
                continue
            cpu_sum += float(proc.cpu_percent(interval=None))
            info = proc.memory_info()
            mem_sum += int(info.rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):  # type: ignore[attr-defined]
            continue
        except Exception:
            continue
    return cpu_sum, mem_sum


def _is_relevant_process(proc: "psutil.Process", season_dir_str: str, filters_lower: Sequence[str]) -> bool:
    """Heuristic to decide whether a process belongs to the season run."""
    try:
        name = (proc.info.get("name") or "").lower()
        if any(f in name for f in filters_lower):
            return True

        try:
            cwd = proc.info.get("cwd") or ""
        except Exception:
            cwd = ""
        if cwd and season_dir_str and season_dir_str in str(cwd):
            return True

        cmd_parts = proc.info.get("cmdline") or []
        cmdline = " ".join(str(p) for p in cmd_parts)
        if season_dir_str and season_dir_str in cmdline:
            return True
        if any(f in cmdline.lower() for f in filters_lower):
            return True
    except Exception:
        return False
    return False


def _season_dir_size(season_dir: Path) -> int:
    """Return total size in bytes of all files under season_dir."""
    total = 0
    for p in _iter_files(season_dir):
        try:
            total += p.stat().st_size
        except Exception:
            continue
    return total


def _compute_step_progress(season_dir: Path) -> tuple[int, int]:
    """Return (total_steps, done_steps) under a season directory.

    A step is considered 'done' when it has either:
    - a posterior ensemble with at least one member, or
    - at least one weights_* CSV under step/assim/.
    """
    total = 0
    done = 0
    for step_dir in sorted(season_dir.glob("step_*")):
        if not step_dir.is_dir():
            continue
        total += 1
        posterior_root = step_dir / "ensembles" / "posterior"
        has_posterior = posterior_root.is_dir() and any(posterior_root.glob("member_*"))
        assim_dir = step_dir / "assim"
        has_weights = assim_dir.is_dir() and any(assim_dir.glob("weights_*_*.csv"))
        if has_posterior or has_weights:
            done += 1
    return total, done


def _iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _append_csv_row(
    csv_path: Path,
    t: datetime,
    cpu_tracked_pct: float,
    cpu_total_pct: float,
    mem_tracked_mb: float,
    mem_used_gb: float,
    mem_total_gb: float,
    mem_used_pct: float,
    season_size_gb: float,
    elapsed_run_sec: float,
    cfg: PerfMonitorConfig,
    progress_steps: float,
    done_steps: int,
    total_steps: int,
    eta_utc: datetime | None,
    eta_local: datetime | None,
) -> None:
    is_new = not csv_path.exists()
    timestep = cfg.timestep or ""
    roi_txt = f"{cfg.roi_area_km2:.3f}" if cfg.roi_area_km2 is not None else ""
    res_txt = f"{cfg.resolution_m:.3f}" if cfg.resolution_m is not None else ""
    days_txt = str(cfg.season_days) if cfg.season_days is not None else ""
    da_txt = str(cfg.num_da_dates) if cfg.num_da_dates is not None else ""
    workers_txt = str(cfg.num_workers) if cfg.num_workers is not None else ""
    prog_txt = f"{progress_steps:.4f}" if total_steps > 0 else ""
    done_txt = str(done_steps) if total_steps > 0 else ""
    total_txt = str(total_steps) if total_steps > 0 else ""
    eta_utc_txt = eta_utc.isoformat(timespec="seconds") if eta_utc is not None else ""
    eta_local_txt = eta_local.isoformat(timespec="seconds") if eta_local is not None else ""

    line = (
        f"{t.isoformat(timespec='seconds')},{cpu_tracked_pct:.3f},{cpu_total_pct:.3f},"
        f"{mem_tracked_mb:.3f},{mem_used_gb:.3f},{mem_total_gb:.3f},{mem_used_pct:.3f},"
        f"{season_size_gb:.3f},{elapsed_run_sec:.1f},"
        f"{roi_txt},{res_txt},{timestep},{days_txt},{da_txt},{workers_txt},"
        f"{prog_txt},{done_txt},{total_txt},{eta_utc_txt},{eta_local_txt}\n"
    )
    with csv_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "timestamp,cpu_tracked_pct,cpu_total_pct,mem_tracked_mb,mem_used_gb,mem_total_gb,mem_used_pct,"
                "season_size_gb,elapsed_run_sec,roi_km2,resolution_m,timestep,"
                "season_days,num_da_dates,num_workers,progress_steps,done_steps,"
                "total_steps,eta_utc,eta_local\n"
            )
        f.write(line)


def _render_plot(
    out_path: Path,
    timestamps: List[datetime],
    cpu_tracked: List[float],
    cpu_total_pct: List[float],
    mem_tracked_mb: List[float],
    mem_used_gb: List[float],
    mem_total_gb: List[float],
    mem_used_pct: List[float],
    disk_gb: List[float],
    cfg: PerfMonitorConfig,
    run_start: datetime,
) -> None:
    if not timestamps or plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

    # Panel 1: system CPU and RAM utilization (Task Manager style)
    ax = axes[0]
    ax.plot(timestamps, cpu_total_pct, label="CPU total [%]", color="tab:blue")
    if cpu_tracked:
        ax.plot(timestamps, cpu_tracked, label="CPU tracked sum [%]", color="tab:cyan", alpha=0.7)
    ax.set_ylabel("CPU [%]")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(timestamps, mem_used_pct, label="RAM total [%]", color="tab:orange")
    ax2.set_ylabel("RAM [%]")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)

    # Panel 2: absolute memory (tracked vs system)
    ax = axes[1]
    ax.plot(timestamps, mem_tracked_mb, label="tracked RSS [MB]")
    ax.set_ylabel("Tracked [MB]")
    ax.grid(True, alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(timestamps, mem_used_gb, color="tab:red", label="system used [GB]")
    ax2.plot(timestamps, mem_total_gb, color="tab:gray", linestyle="--", label="system total [GB]")
    ax2.set_ylabel("System [GB]")
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=8)

    # Panel 3: disk usage
    ax = axes[2]
    ax.plot(timestamps, disk_gb, label="season dir size [GB]")
    ax.set_ylabel("Season [GB]")
    ax.set_xlabel("Time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    # Summary text overlay
    elapsed_hours = (timestamps[-1] - run_start).total_seconds() / 3600.0
    roi_txt = f"{cfg.roi_area_km2:.2f} km²" if cfg.roi_area_km2 is not None else "n/a"
    res_txt = f"{cfg.resolution_m:.0f} m" if cfg.resolution_m is not None else "n/a"
    timestep = cfg.timestep or "n/a"
    days_txt = str(cfg.season_days) if cfg.season_days is not None else "n/a"
    da_txt = str(cfg.num_da_dates) if cfg.num_da_dates is not None else "n/a"
    workers_txt = str(cfg.num_workers) if cfg.num_workers is not None else "n/a"
    ens_txt = str(cfg.ensemble_size) if cfg.ensemble_size is not None else "n/a"

    summary_1 = f"ROI: {roi_txt} | res: {res_txt} | dt: {timestep}"
    summary_2 = (
        f"season days: {days_txt} | DA dates: {da_txt} | workers: {workers_txt} | "
        f"ensemble size: {ens_txt} | elapsed: {elapsed_hours:.2f} h"
    )

    # ETA summary
    eta_utc: datetime | None = None
    eta_local: datetime | None = None
    try:
        total_steps, done_steps = _compute_step_progress(Path(cfg.season_dir))
        if total_steps > 0:
            progress_steps = done_steps / float(total_steps)
            if 0.0 < progress_steps < 1.0:
                expected_total_sec = (timestamps[-1] - run_start).total_seconds() / progress_steps
                eta_utc = run_start + timedelta(seconds=expected_total_sec)
                if cfg.tz_offset_hours is not None:
                    eta_local = eta_utc + timedelta(hours=cfg.tz_offset_hours)
    except Exception:
        eta_utc = None
        eta_local = None

    if eta_local is not None:
        tz_label = f"UTC{cfg.tz_offset_hours:+.1f}"
        eta_txt = eta_local.strftime("%Y-%m-%d %H:%M")
    elif eta_utc is not None:
        tz_label = "UTC"
        eta_txt = eta_utc.strftime("%Y-%m-%d %H:%M")
    else:
        tz_label = "UTC"
        eta_txt = "n/a"

    summary_3 = f"ETA ({tz_label}): {eta_txt}"

    fig.text(0.5, 0.98, summary_1, ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.955, summary_2, ha="center", va="top", fontsize=9)
    fig.text(0.5, 0.93, summary_3, ha="center", va="top", fontsize=9)

    fig.tight_layout(rect=(0.04, 0.04, 0.96, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def cli_main(argv: List[str] | None = None) -> int:
    """CLI entry: run a foreground performance monitor for a season."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-perf-monitor",
        description="Monitor CPU/RAM/disk usage for a season directory.",
    )
    p.add_argument("--season-dir", required=True, type=Path, help="Season directory (contains season.yml)")
    p.add_argument("--sample-interval", type=float, default=5.0, help="Sampling interval in seconds (default: 5)")
    p.add_argument("--plot-interval", type=float, default=30.0, help="Plot refresh interval in seconds (default: 30)")
    p.add_argument(
        "--tz-offset-hours",
        type=float,
        default=0.0,
        help="Optional timezone offset hours for ETA label (e.g. 1.0 for CET = UTC+1).",
    )
    p.add_argument("--num-workers", type=int, default=None, help="Optional number of workers used by the season run.")
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

    # Derive project directory from season layout: project/propagation/season_YYYY-YYYY
    project_dir = season_dir.parent.parent

    roi_area_km2 = None
    proj_resolution = None
    proj_timestep = None
    proj_crs = None
    season_days = None
    num_da_dates = None
    ensemble_size = None

    # Project metadata: resolution, timestep, CRS, AOI area
    try:
        proj_yaml = find_project_yaml(project_dir)
        proj_cfg = _read_yaml_file(proj_yaml) or {}
        if "resolution" in proj_cfg:
            try:
                proj_resolution = float(proj_cfg.get("resolution"))
            except Exception:
                proj_resolution = None
        if "timestep" in proj_cfg:
            proj_timestep = str(proj_cfg.get("timestep"))
        proj_crs = proj_cfg.get("crs")
        da_cfg = proj_cfg.get("data_assimilation") or {}
        pf_cfg = da_cfg.get("prior_forcing") or {}
        if "ensemble_size" in pf_cfg:
            try:
                ensemble_size = int(pf_cfg.get("ensemble_size"))
            except Exception:
                ensemble_size = None

        # AOI area
        env_dir = Path(project_dir) / "env"
        aoi_candidates = list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp"))
        if aoi_candidates:
            gdf, _ = read_single_roi(aoi_candidates[0], required_field=None, to_crs=proj_crs if proj_crs is not None else None)
            roi_area_km2 = float(gdf.geometry.area.iloc[0]) / 1_000_000.0
    except Exception as exc:
        logger.warning("Perf monitor CLI: failed to read project/AOI metadata: {}", exc)

    # Season dates and DA events
    try:
        seas_yaml = find_season_yaml(season_dir)
        seas_cfg = _read_yaml_file(seas_yaml) or {}
        start_val = seas_cfg.get("start_date")
        end_val = seas_cfg.get("end_date")
        start_dt = parse_datetime_opt(str(start_val)) if start_val is not None else None
        end_dt = parse_datetime_opt(str(end_val)) if end_val is not None else None
        if start_dt is not None and end_dt is not None:
            season_days = (end_dt.date() - start_dt.date()).days + 1
        events = load_assimilation_events(season_dir)
        num_da_dates = len(events)
    except Exception as exc:
        logger.warning("Perf monitor CLI: failed to read season/assimilation metadata: {}", exc)

    cfg = PerfMonitorConfig(
        season_dir=season_dir,
        sample_interval_sec=float(args.sample_interval or 5.0),
        plot_interval_sec=float(args.plot_interval or 30.0),
        roi_area_km2=roi_area_km2,
        resolution_m=proj_resolution,
        timestep=proj_timestep,
        season_days=season_days,
        num_da_dates=num_da_dates,
        num_workers=(int(args.num_workers) if args.num_workers is not None else None),
        ensemble_size=ensemble_size,
        tz_offset_hours=float(args.tz_offset_hours),
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
