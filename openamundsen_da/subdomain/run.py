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
from openamundsen_da.io.paths import find_setup_yaml
from openamundsen_da.methods.wet_snow.area import summarize_s1_directory
from openamundsen_da.observer.satellite_scf import generate_project_from_summary as scf_project_obs
from openamundsen_da.observer.satellite_wet_snow_s1 import generate_project_from_summary as wet_project_obs
from openamundsen_da.observer.snowcover import summarize_snowcover_directory
from openamundsen_da.pipeline.project import OrchestratorConfig, run_project
from openamundsen_da.pipeline.project_skeleton import create_project_skeleton
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
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


def _load_wetsnow_classes(setup_dir: Path) -> tuple[list[int], list[int], list[int]]:
    cfg = _read_yaml_file(find_setup_yaml(setup_dir)) or {}
    obs_cfg = (cfg.get("obs") or {}).get("wetsnow") or {}
    classes = obs_cfg.get("classes") or {}

    def _ints(vals, default):
        out = []
        for v in vals if vals is not None else default:
            try:
                out.append(int(v))
            except Exception:
                continue
        return out

    wet = _ints(classes.get("wet"), [1, 2])
    valid = _ints(classes.get("valid"), [1, 2, 3, 4, 255])
    exclude = _ints(classes.get("exclude"), [5, 6])
    return wet, valid, exclude


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
    if v in {"wet_snow", "wet_snow_fraction"}:
        return "wet_snow"
    return v


def _filter_project_events_to_available_obs(project_yaml: Path, *, available_by_var: dict[str, set[date]]) -> int:
    cfg = _read_yaml_file(project_yaml) or {}
    da_cfg = cfg.get("data_assimilation") or {}
    raw_events = list(da_cfg.get("assimilation_events") or [])
    kept: list[dict] = []
    removed = 0

    for ev in raw_events:
        if not isinstance(ev, dict):
            kept.append(ev)
            continue
        var = _normalize_event_variable(ev.get("variable"))
        keep_dates = available_by_var.get(var)
        if keep_dates is None:
            kept.append(ev)
            continue
        dt = parse_datetime_opt(str(ev.get("date")))
        if dt is None:
            kept.append(ev)
            continue
        if dt.date() in keep_dates:
            kept.append(ev)
        else:
            removed += 1

    if removed == 0:
        return 0

    da_cfg["assimilation_events"] = kept
    cfg["data_assimilation"] = da_cfg
    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML()
        with project_yaml.open("w", encoding="utf-8") as f:
            y.dump(cfg, f)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to write filtered assimilation_events to {project_yaml}: {exc}") from exc
    return removed


def _write_run_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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

    if "wet_snow" in variables:
        if not manifest.raw_wetsnow_dir.is_dir():
            raise FileNotFoundError(f"Raw wet-snow directory not found: {manifest.raw_wetsnow_dir}")
        wet, valid, exclude = _load_wetsnow_classes(sub.setup_dir)
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

    removed = _filter_project_events_to_available_obs(
        sub.project_yaml,
        available_by_var={
            "scf": _read_summary_dates(scf_summary),
            "wet_snow": _read_summary_dates(wet_summary),
        },
    )
    if removed:
        logger.info("Filtered {} assimilation event(s) without local observations in {}", removed, sub.project_yaml)

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

    if wet_summary.is_file() and "wet_snow" in variables:
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
            run_project(
                OrchestratorConfig(
                    project_dir=sub.project_dir,
                    setup_dir=sub.setup_dir,
                    max_workers=int(inner_max_workers),
                    overwrite=overwrite,
                    log_level=log_level,
                    live_plots=False,
                    plot_workers=None,
                    monitor_perf=False,
                )
            )
            duration = time.time() - started
            run_meta.update(
                {
                    "status": "success",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
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
            )
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - started
            run_meta.update(
                {
                    "status": "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                    "error": repr(exc),
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
                )
    return RunResult(
        subdomain_id=sub.id,
        status="failed",
        duration_seconds=0.0,
        setup_dir=sub.setup_dir,
        log_path=log_path,
        error="unknown",
        run_manifest=run_manifest_path,
    )


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
    selected_ids = list(subdomains) if subdomains else list(manifest.subdomains.keys())
    unknown = [sid for sid in selected_ids if sid not in manifest.subdomains]
    if unknown:
        raise ValueError(f"Sub-domains not in manifest: {', '.join(unknown)}")

    root_log = manifest.subdomain_root / "subdomain_run.log"
    sink_id = None
    if log_to_file:
        root_log.parent.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(
            root_log,
            level=log_level.upper(),
            colorize=False,
            enqueue=True,
            format=LOGURU_FORMAT,
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
            PerfMonitorConfig(project_dir=manifest.subdomain_root, sample_interval_sec=5.0, plot_interval_sec=30.0)
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
