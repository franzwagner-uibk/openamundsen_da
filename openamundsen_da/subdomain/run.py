"""Parallel execution of independent sub-domain DA workflows."""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.methods.wet_snow.area import summarize_s1_directory
from openamundsen_da.observer.class_config import load_wetsnow_classes
from openamundsen_da.observer.satellite_scf import generate_project_from_summary as scf_project_obs
from openamundsen_da.observer.satellite_wet_snow_s1 import generate_project_from_summary as wet_project_obs
from openamundsen_da.observer.snowcover import summarize_snowcover_directory
from openamundsen_da.pipeline.project import OrchestratorConfig, run_project
from openamundsen_da.pipeline.project_skeleton import create_project_skeleton
from openamundsen_da.subdomain.event_filter import filter_project_events_for_subdomain
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.util.run_mode import ensure_run_mode
from openamundsen_da.util.ts import parse_datetime_opt


@dataclass
class RunResult:
    subdomain_id: str
    status: str  # success | failed | skipped
    duration_seconds: float
    setup_dir: Path
    log_path: Path
    error: Optional[str] = None
    run_manifest: Optional[Path] = None
    dropped_events: list[dict] | None = None


def _project_window(project_yaml: Path) -> tuple[datetime | None, datetime | None]:
    cfg = _read_yaml_file(project_yaml) or {}
    start = parse_datetime_opt(str(cfg.get("start_date"))) if cfg.get("start_date") is not None else None
    end = parse_datetime_opt(str(cfg.get("end_date"))) if cfg.get("end_date") is not None else None
    return start, end


def _read_summary_dates(csv_path: Path) -> set[date]:
    if not csv_path.is_file():
        return set()
    try:
        df = pd.read_csv(csv_path, usecols=["date"])
    except Exception:
        return set()
    out: set[date] = set()
    for raw in df["date"].dropna().astype(str):
        dt = parse_datetime_opt(raw)
        if dt is not None:
            out.add(dt.date())
    return out


def _normalize_event_variable(value: object) -> str:
    v = str(value or "").strip().lower()
    if v in {"wet_snow", "wet_snow_fraction", "wet_snow_line"}:
        return "wet_snow"
    return v


def _validate_project_events_have_obs(project_yaml: Path, *, available_by_var: dict[str, set[date]]) -> None:
    cfg = _read_yaml_file(project_yaml) or {}
    da_cfg = cfg.get("data_assimilation") or {}
    raw_events = list(da_cfg.get("assimilation_events") or [])
    missing: list[str] = []

    for idx, ev in enumerate(raw_events, start=1):
        if not isinstance(ev, dict):
            raise ValueError(
                f"Expected mapping at data_assimilation.assimilation_events[{idx}], got {type(ev).__name__}"
            )
        var = _normalize_event_variable(ev.get("variable"))
        keep_dates = available_by_var.get(var)
        if keep_dates is None:
            continue
        dt = parse_datetime_opt(str(ev.get("date")))
        if dt is None:
            raise ValueError(
                f"Invalid or missing date at data_assimilation.assimilation_events[{idx}].date"
            )
        if dt.date() not in keep_dates:
            missing.append(f"{dt.date()} ({var})")

    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(
            f"Configured assimilation events have no local observations in {project_yaml}: {joined}"
        )


def _write_run_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _dropped_events_csv(sub_setup_dir: Path) -> Path:
    return Path(sub_setup_dir) / "subdomain_dropped_events.csv"


def _read_dropped_events(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return df.fillna("").to_dict(orient="records")


def _configure_worker_logger(log_path: Path, log_level: str, root_log_path: Path | None) -> None:
    logger.remove()
    logger.add(
        log_path.open("a", encoding="utf-8"),
        level=log_level.upper(),
        colorize=False,
        enqueue=False,
        format=LOGURU_FORMAT,
    )
    if root_log_path is not None:
        root_log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            root_log_path,
            level=log_level.upper(),
            colorize=False,
            enqueue=True,
            format=LOGURU_FORMAT,
        )


def _prepare_obs_for_subdomain(sub, manifest: SubdomainManifest, *, overwrite: bool) -> None:
    events = load_assimilation_events(sub.project_dir)
    variables = {ev.variable for ev in events}
    if not variables:
        return

    start, end = _project_window(sub.project_yaml)
    setup_obs_root = sub.setup_dir / "obs"
    setup_obs_root.mkdir(parents=True, exist_ok=True)
    scf_summary = setup_obs_root / sub.project_name / "scf_summary.csv"
    wet_summary = setup_obs_root / sub.project_name / "wet_snow_summary.csv"

    if "scf" in variables:
        if not manifest.raw_snowcover_dir.is_dir():
            raise FileNotFoundError(f"Raw SCF directory not found: {manifest.raw_snowcover_dir}")
        summarize_snowcover_directory(
            setup_dir=sub.setup_dir,
            input_dir=manifest.raw_snowcover_dir,
            aoi=sub.roi_vector_path,
            project_label=sub.project_name,
            output_root=setup_obs_root,
            recursive=True,
            start=start,
            end=end,
        )

    if {"wet_snow", "wet_snow_line"} & variables:
        if not manifest.raw_wetsnow_dir.is_dir():
            raise FileNotFoundError(f"Raw wet-snow directory not found: {manifest.raw_wetsnow_dir}")
        wet, valid, exclude = load_wetsnow_classes(sub.project_dir)
        try:
            summarize_s1_directory(
                setup_dir=sub.setup_dir,
                project_dir=sub.project_dir,
                raster_dir=manifest.raw_wetsnow_dir,
                aoi_path=sub.roi_vector_path,
                output_csv=wet_summary,
                overwrite=overwrite,
                start=start,
                end=end,
                wet_values=wet,
                valid_values=valid,
                exclude_values=exclude,
                recursive=True,
            )
        except RuntimeError as exc:
            logger.warning("No valid wet-snow observations for {}: {}", sub.id, exc)

    filter_project_events_for_subdomain(
        project_yaml=sub.project_yaml,
        setup_dir=sub.setup_dir,
        project_name=sub.project_name,
        subdomain_id=sub.id,
        dropped_events_csv=_dropped_events_csv(sub.setup_dir),
    )
    events = load_assimilation_events(sub.project_dir)
    variables = {ev.variable for ev in events}

    _validate_project_events_have_obs(
        sub.project_yaml,
        available_by_var={
            "scf": _read_summary_dates(scf_summary),
            "wet_snow": _read_summary_dates(wet_summary),
        },
    )

    # Build step folders first, then distribute per-step obs CSVs from summaries.
    create_project_skeleton(
        setup_dir=sub.setup_dir,
        project_dir=sub.project_dir,
        overwrite=overwrite,
    )

    if scf_summary.is_file() and "scf" in variables:
        scf_project_obs(
            project_dir=sub.project_dir,
            summary_csv=scf_summary,
            product=None,
            overwrite=overwrite,
        )

    if wet_summary.is_file() and ({"wet_snow", "wet_snow_line"} & variables):
        wet_project_obs(
            project_dir=sub.project_dir,
            summary_csv=wet_summary,
            product=None,
            overwrite=overwrite,
        )


def _run_one(
    subdomain_id: str,
    manifest_path: Path,
    inner_max_workers: int,
    overwrite: bool,
    retries: int,
    log_level: str,
    root_log_path: Path | None,
) -> RunResult:
    """Worker: run one fully independent sub-domain DA setup."""
    manifest = SubdomainManifest.load(manifest_path)
    sub = manifest.subdomains[subdomain_id]

    run_manifest_path = sub.setup_dir / "run_manifest.json"
    log_path = sub.setup_dir / "run.log"

    if run_manifest_path.is_file() and not overwrite:
        try:
            data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if str(data.get("status", "")).lower() == "success":
                return RunResult(
                    subdomain_id=sub.id,
                    status="skipped",
                    duration_seconds=0.0,
                    setup_dir=sub.setup_dir,
                    log_path=log_path,
                    run_manifest=run_manifest_path,
                    dropped_events=list(
                        data.get("dropped_events")
                        or _read_dropped_events(_dropped_events_csv(sub.setup_dir))
                    ),
                )
        except Exception:
            pass

    _configure_worker_logger(log_path, log_level, root_log_path)

    attempt = 0
    while attempt <= retries:
        attempt += 1
        started = time.time()
        run_meta = {
            "id": sub.id,
            "setup_dir": str(sub.setup_dir),
            "project_dir": str(sub.project_dir),
            "attempt": attempt,
            "status": "running",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_run_manifest(run_manifest_path, run_meta)
        logger.info("START sub-domain={} attempt={}", sub.id, attempt)
        try:
            _prepare_obs_for_subdomain(sub, manifest, overwrite=overwrite)
            dropped_events = _read_dropped_events(_dropped_events_csv(sub.setup_dir))
            run_project(
                OrchestratorConfig(
                    project_dir=sub.project_dir,
                    setup_dir=sub.setup_dir,
                    max_workers=int(inner_max_workers),
                    overwrite=overwrite,
                    log_level=log_level,
                    live_plots=False,
                    plot_workers=int(inner_max_workers),
                    monitor_perf=False,
                )
            )
            duration = time.time() - started
            run_meta.update(
                {
                    "status": "success",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                    "dropped_events": dropped_events,
                }
            )
            _write_run_manifest(run_manifest_path, run_meta)
            logger.info("OK sub-domain={} duration_s={:.1f}", sub.id, duration)
            return RunResult(
                subdomain_id=sub.id,
                status="success",
                duration_seconds=duration,
                setup_dir=sub.setup_dir,
                log_path=log_path,
                run_manifest=run_manifest_path,
                dropped_events=dropped_events,
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - started
            dropped_events = _read_dropped_events(_dropped_events_csv(sub.setup_dir))
            run_meta.update(
                {
                    "status": "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                    "error": repr(exc),
                    "dropped_events": dropped_events,
                }
            )
            _write_run_manifest(run_manifest_path, run_meta)
            logger.exception("Sub-domain {} failed on attempt {}: {}", sub.id, attempt, exc)
            if attempt > retries:
                return RunResult(
                    subdomain_id=sub.id,
                    status="failed",
                    duration_seconds=duration,
                    setup_dir=sub.setup_dir,
                    log_path=log_path,
                    error=repr(exc),
                    run_manifest=run_manifest_path,
                    dropped_events=dropped_events,
                )
    return RunResult(
        subdomain_id=sub.id,
        status="failed",
        duration_seconds=0.0,
        setup_dir=sub.setup_dir,
        log_path=log_path,
        error="unknown",
        run_manifest=run_manifest_path,
        dropped_events=_read_dropped_events(_dropped_events_csv(sub.setup_dir)),
    )


def _write_project_dropped_events(manifest: SubdomainManifest) -> None:
    rows: list[dict] = []
    for meta in manifest.subdomains.values():
        rows.extend(list(meta.dropped_events or []))
    columns = [
        "subdomain_id",
        "date",
        "variable",
        "product",
        "reason",
        "metric",
        "value",
        "threshold",
        "active_station_ids",
        "project_yaml",
    ]
    out = manifest.project_dir / "results" / "subdomain_dropped_events.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)


def run_subdomains(
    *,
    manifest_path: Path,
    subdomains: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
    inner_max_workers: Optional[int] = None,
    retries: int = 0,
    overwrite: bool = False,
    log_level: str = "INFO",
    perf_monitor: bool = True,
    log_to_file: bool = True,
) -> List[RunResult]:
    """Run sub-domain DA workflows in parallel and stop on first failure."""
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "subdomain":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='subdomain'.")
    ensure_run_mode(manifest.project_dir, expected="subdomain", write_if_missing=False)
    selected_ids = list(subdomains) if subdomains else list(manifest.subdomains.keys())
    unknown = [sid for sid in selected_ids if sid not in manifest.subdomains]
    if unknown:
        raise ValueError(f"Sub-domains not in manifest: {', '.join(unknown)}")

    root_log = manifest.project_dir / "subdomain_run.log"
    sink_id = None
    if log_to_file:
        root_log.parent.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(
            root_log,
            level=log_level.upper(),
            colorize=False,
            enqueue=True,
            format=LOGURU_FORMAT,
            mode="w" if overwrite else "a",
        )

    outer_workers = pick_max_workers(max_workers, fallback=len(selected_ids), limit=len(selected_ids))
    cpu = os.cpu_count() or 1
    auto_inner = max(1, cpu // max(1, outer_workers))
    inner_workers = int(inner_max_workers) if inner_max_workers is not None else auto_inner
    inner_workers = max(1, inner_workers)

    logger.info(
        "START sub-domain run count={} outer_workers={} inner_workers={} fail_fast=true",
        len(selected_ids),
        outer_workers,
        inner_workers,
    )

    perf_stop = None
    if perf_monitor:
        perf_stop = start_perf_monitor(
            PerfMonitorConfig(project_dir=manifest.project_dir, sample_interval_sec=5.0, plot_interval_sec=30.0)
        )

    results: List[RunResult] = []
    failed_id: str | None = None

    ctx = mp.get_context("spawn")
    executor = cf.ProcessPoolExecutor(max_workers=outer_workers, mp_context=ctx)
    future_map: dict[cf.Future, str] = {}
    try:
        for sid in selected_ids:
            fut = executor.submit(
                _run_one,
                sid,
                manifest_path,
                inner_workers,
                overwrite,
                int(max(0, retries)),
                log_level,
                root_log if log_to_file else None,
            )
            future_map[fut] = sid

        for fut in cf.as_completed(future_map):
            sid = future_map[fut]
            res = fut.result()
            results.append(res)
            meta = manifest.subdomains[sid]
            meta.status = res.status
            if res.run_manifest:
                meta.run_manifest = res.run_manifest
            meta.dropped_events = list(res.dropped_events or [])
            manifest.save(manifest_path)
            logger.info(
                "STATUS sub-domain={} status={} duration_s={:.1f}",
                sid,
                res.status,
                res.duration_seconds,
            )
            if res.status == "failed":
                failed_id = sid
                logger.error("Fail-fast triggered by sub-domain {}", sid)
                for other in future_map:
                    if not other.done():
                        other.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                break
    finally:
        if perf_stop:
            perf_stop.set()
        # Ensure process cleanup.
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass

    completed_ids = {r.subdomain_id for r in results}
    for sid in selected_ids:
        if sid in completed_ids:
            continue
        meta = manifest.subdomains[sid]
        if failed_id is not None:
            meta.status = "skipped"
    manifest.save(manifest_path)
    _write_project_dropped_events(manifest)

    ok = sum(1 for r in results if r.status == "success")
    fail = sum(1 for r in results if r.status == "failed")
    skip = sum(1 for r in results if r.status == "skipped")
    logger.info(
        "SUMMARY total_selected={} completed={} success={} failed={} skipped={}",
        len(selected_ids),
        len(results),
        ok,
        fail,
        skip,
    )
    if sink_id is not None:
        logger.remove(sink_id)
    if failed_id is not None:
        raise RuntimeError(f"Sub-domain run failed in {failed_id}; fail-fast stopped remaining tasks.")
    return results
