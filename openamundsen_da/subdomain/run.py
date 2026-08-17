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
from ruamel.yaml.error import YAMLError
from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.exceptions import LowDiskSpaceError
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    recursive_files,
    write_manifest_atomic,
)
from openamundsen_da.methods.wet_snow.area import summarize_s1_directory
from openamundsen_da.observer.class_config import load_wetsnow_classes
from openamundsen_da.observer.satellite_scf import generate_project_from_summary as scf_project_obs
from openamundsen_da.observer.satellite_wet_snow_s1 import generate_project_from_summary as wet_project_obs
from openamundsen_da.observer.snowcover import summarize_snowcover_directory
from openamundsen_da.pipeline.project import OrchestratorConfig, run_project
from openamundsen_da.pipeline.project_skeleton import (
    create_project_skeleton,
    plan_project_steps,
)
from openamundsen_da.subdomain.event_support import resolve_subdomain_event_plan
from openamundsen_da.subdomain.leaf_finalization import (
    finalize_leaf as _finalize_leaf,
    leaf_finalization_manifest_path as _leaf_finalization_manifest_path,
    measured_retained_leaf_bytes as _measured_retained_leaf_bytes,
)
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.subdomain.status import save_stage, terminal_status
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.da_output import (
    output_retention_mode,
    validate_compact_output_file,
)
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.util.run_mode import ensure_run_mode
from openamundsen_da.util.storage_budget import (
    DiskBudgetSnapshot,
    StorageReservationProject,
    check_step_admission,
    estimate_coordinated_storage_reserve,
    estimate_parent_compact_merge_bytes,
    estimate_parent_render_bytes,
    estimate_project_storage_components,
)
from openamundsen_da.util.storage_admission import (
    StorageAdmissionClient,
    StorageAdmissionCoordinator,
    StorageAdmissionServer,
    build_storage_plan,
)
from openamundsen_da.util.ts import parse_datetime_opt
from openamundsen_da.io.paths import list_steps_sorted


@dataclass
class SubdomainRunResult:
    subdomain_id: str
    status: str  # success | failed | paused_low_disk | skipped
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
    write_manifest_atomic(path, data)


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


def _prepare_obs_for_subdomain(
    sub,
    manifest: SubdomainManifest,
    *,
    overwrite: bool,
    scientific_identity: str | None = None,
) -> None:
    def preparation_inventory() -> list[dict]:
        paths = [
            path
            for root in (sub.project_dir / "steps", sub.setup_dir / "obs")
            for path in recursive_files(root)
            if "/ensembles/" not in path.as_posix()
            and "/assim/" not in path.as_posix()
            and "/results/" not in path.as_posix()
        ]
        return file_inventory(root=sub.setup_dir, files=paths)

    preparation_manifest_path = (
        sub.setup_dir / ".openamundsen-da" / "manifests" / "leaf_preparation.json"
    )
    existing_preparation = load_manifest(preparation_manifest_path)
    if (
        existing_preparation is not None
        and existing_preparation.get("status") == "success"
        and not overwrite
    ):
        if existing_preparation.get("scientific_identity") != scientific_identity:
            raise RuntimeError(
                f"Leaf preparation scientific identity changed: {preparation_manifest_path}"
            )
        recorded = existing_preparation.get("outputs")
        if not isinstance(recorded, list):
            raise RuntimeError(f"Leaf preparation outputs are invalid: {preparation_manifest_path}")
        current = preparation_inventory()
        if current != recorded:
            raise RuntimeError(f"Leaf preparation inventory changed: {preparation_manifest_path}")
        if inventory_digest(current) != existing_preparation.get("output_digest"):
            raise RuntimeError(f"Leaf preparation outputs changed: {preparation_manifest_path}")
        return
    if _project_has_started(sub.project_dir):
        raise RuntimeError(
            f"Incomplete leaf preparation has runtime evidence: {sub.project_dir}"
        )
    # Without completed authority, every prep-owned output is potentially a
    # truncated crash artifact. Rebuild deterministically before propagation.
    preparation_overwrite = True
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
                overwrite=preparation_overwrite,
                start=start,
                end=end,
                wet_values=wet,
                valid_values=valid,
                exclude_values=exclude,
                recursive=True,
            )
        except RuntimeError as exc:
            logger.warning("No valid wet-snow observations for {}: {}", sub.id, exc)

    _validate_project_events_have_obs(
        sub.project_yaml,
        available_by_var={
            "scf": _read_summary_dates(scf_summary),
            "wet_snow": _read_summary_dates(wet_summary),
        },
    )

    # Build or validate the immutable virtual step plan. Partial preparation may
    # be completed only before any model/runtime evidence exists.
    planned_names = tuple(
        plan.name for plan in plan_project_steps(sub.setup_dir, sub.project_dir)
    )
    try:
        existing_names = tuple(
            path.name for path in list_steps_sorted(sub.project_dir)
        )
    except (FileNotFoundError, ValueError, YAMLError):
        if _project_has_started(sub.project_dir):
            raise RuntimeError(
                f"Incomplete leaf preparation has runtime evidence: {sub.project_dir}"
            )
        existing_names = ()
    unexpected = sorted(set(existing_names) - set(planned_names))
    if unexpected:
        raise RuntimeError(
            f"Prepared leaf steps differ from virtual plan: {unexpected}"
        )
    if existing_names != planned_names:
        if _project_has_started(sub.project_dir):
            raise RuntimeError(
                f"Incomplete leaf preparation has runtime evidence: {sub.project_dir}"
            )
        create_project_skeleton(
            setup_dir=sub.setup_dir,
            project_dir=sub.project_dir,
            overwrite=True,
        )

    if scf_summary.is_file() and "scf" in variables:
        scf_project_obs(
            project_dir=sub.project_dir,
            summary_csv=scf_summary,
            product=None,
            overwrite=preparation_overwrite,
        )

    if wet_summary.is_file() and ({"wet_snow", "wet_snow_line"} & variables):
        wet_project_obs(
            project_dir=sub.project_dir,
            summary_csv=wet_summary,
            product=None,
            overwrite=preparation_overwrite,
        )
    prepared_inventory = preparation_inventory()
    write_manifest_atomic(
        preparation_manifest_path,
        {
            "status": "success",
            "scientific_identity": scientific_identity,
            "outputs": prepared_inventory,
            "output_digest": inventory_digest(prepared_inventory),
        },
    )


def _run_one(
    subdomain_id: str,
    manifest_path: Path,
    inner_max_workers: int,
    overwrite: bool,
    retries: int,
    log_level: str,
    root_log_path: Path | None,
    storage_reservation_projects: tuple[StorageReservationProject, ...] = (),
    storage_outer_workers: int = 1,
    shared_storage_reserve_bytes: int = 0,
    storage_admission_client: StorageAdmissionClient | None = None,
    prepared_before_storage_plan: bool = False,
) -> SubdomainRunResult:
    """Worker: run one fully independent sub-domain DA setup."""
    manifest = SubdomainManifest.load(manifest_path)
    sub = manifest.subdomains[subdomain_id]

    run_manifest_path = sub.setup_dir / "run_manifest.json"
    log_path = sub.setup_dir / "run.log"
    finalization_path = _leaf_finalization_manifest_path(sub.setup_dir)

    if finalization_path.is_file() and not overwrite:
        finalized = _finalize_leaf(
            sub,
            resume=True,
            scientific_identity=(
                storage_admission_client.leaf_identity
                if storage_admission_client is not None
                else None
            ),
        )
        if finalized.get("status") == "success":
            if storage_admission_client is not None:
                storage_admission_client.transition(
                    "leaf_finalized",
                    removed_bytes=int(finalized.get("cleanup_freed_bytes", 0)),
                    request_id=f"{sub.id}:leaf_finalized",
                )
            recovered = {
                "id": sub.id,
                "setup_dir": str(sub.setup_dir),
                "project_dir": str(sub.project_dir),
                "status": "success",
                "scientific_identity": (
                    storage_admission_client.leaf_identity
                    if storage_admission_client is not None
                    else None
                ),
                "recovered_from_finalization": True,
                "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                "dropped_events": _read_dropped_events(
                    _dropped_events_csv(sub.setup_dir)
                ),
                "leaf_finalization_manifest": str(finalization_path),
                "retained_leaf_bytes": int(finalized.get("retained_leaf_bytes", 0)),
            }
            _write_run_manifest(run_manifest_path, recovered)
            return SubdomainRunResult(
                subdomain_id=sub.id,
                status="skipped",
                duration_seconds=0.0,
                setup_dir=sub.setup_dir,
                log_path=log_path,
                run_manifest=run_manifest_path,
                dropped_events=list(recovered["dropped_events"]),
            )

    previous_status: str | None = None
    if run_manifest_path.is_file() and not overwrite:
        try:
            data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        previous_status = str(data.get("status", "")).lower()
        if previous_status == "success":
            current_identity = (
                storage_admission_client.leaf_identity
                if storage_admission_client is not None
                else None
            )
            if data.get("scientific_identity") != current_identity:
                raise RuntimeError(
                    f"Completed leaf run scientific identity changed: {run_manifest_path}"
                )
            finalized = _finalize_leaf(
                sub,
                resume=False,
                scientific_identity=(
                    storage_admission_client.leaf_identity
                    if storage_admission_client is not None
                    else None
                ),
            )
            if storage_admission_client is not None:
                storage_admission_client.transition(
                    "leaf_finalized",
                    removed_bytes=int(finalized.get("cleanup_freed_bytes", 0)),
                    request_id=f"{sub.id}:leaf_finalized",
                )
            data["retained_leaf_bytes"] = int(finalized.get("retained_leaf_bytes", 0))
            if finalization_path.is_file():
                data["leaf_finalization_manifest"] = str(finalization_path)
            _write_run_manifest(run_manifest_path, data)
            return SubdomainRunResult(
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

    # A failed/interrupted leaf is a resumable project. Never turn an ordinary
    # resume into destructive overwrite implicitly; callers must request
    # ``--overwrite`` explicitly after deciding that completed work may be
    # discarded.
    effective_overwrite = bool(overwrite)

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
            "scientific_identity": (
                storage_admission_client.leaf_identity
                if storage_admission_client is not None
                else None
            ),
        }
        logger.info("START sub-domain={} attempt={}", sub.id, attempt)
        try:
            if not prepared_before_storage_plan:
                _prepare_obs_for_subdomain(
                    sub,
                    manifest,
                    overwrite=effective_overwrite,
                    scientific_identity=(
                        storage_admission_client.leaf_identity
                        if storage_admission_client is not None
                        else None
                    ),
                )
            elif not list_steps_sorted(sub.project_dir):
                raise RuntimeError(
                    f"Prepared leaf steps disappeared after storage preflight: {sub.project_dir}"
                )
            if storage_admission_client is not None:
                storage_admission_client.transition(
                    "leaf_prepared",
                    request_id=f"{sub.id}:leaf_prepared",
                )
            _write_run_manifest(run_manifest_path, run_meta)
            dropped_events = _read_dropped_events(_dropped_events_csv(sub.setup_dir))
            run_project(
                OrchestratorConfig(
                    project_dir=sub.project_dir,
                    setup_dir=sub.setup_dir,
                    max_workers=int(inner_max_workers),
                    overwrite=effective_overwrite,
                    log_level=log_level,
                    live_plots=False,
                    plot_workers=int(inner_max_workers),
                    monitor_perf=False,
                    storage_reservation_projects=storage_reservation_projects,
                    storage_outer_workers=int(storage_outer_workers),
                    shared_storage_reserve_bytes=int(shared_storage_reserve_bytes),
                    storage_admission_client=storage_admission_client,
                    initial_step_preadmitted=(
                        prepared_before_storage_plan
                        and storage_admission_client is not None
                    ),
                )
            )
            finalized = _finalize_leaf(
                sub,
                resume=False,
                scientific_identity=(
                    storage_admission_client.leaf_identity
                    if storage_admission_client is not None
                    else None
                ),
            )
            if storage_admission_client is not None:
                storage_admission_client.transition(
                    "leaf_finalized",
                    release_bytes=0,
                    removed_bytes=int(finalized.get("cleanup_freed_bytes", 0)),
                    request_id=f"{sub.id}:leaf_finalized",
                )
            duration = time.time() - started
            run_meta.update(
                {
                    "status": "success",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                    "dropped_events": dropped_events,
                    "retained_leaf_bytes": int(finalized.get("retained_leaf_bytes", 0)),
                    "cleanup_freed_bytes": int(finalized.get("cleanup_freed_bytes", 0)),
                }
            )
            if finalization_path.is_file():
                run_meta["leaf_finalization_manifest"] = str(finalization_path)
            _write_run_manifest(run_manifest_path, run_meta)
            logger.info("OK sub-domain={} duration_s={:.1f}", sub.id, duration)
            return SubdomainRunResult(
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
                    "status": "paused_low_disk" if isinstance(exc, LowDiskSpaceError) else "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                    "error": repr(exc),
                    "dropped_events": dropped_events,
                }
            )
            _write_run_manifest(run_manifest_path, run_meta)
            logger.exception("Sub-domain {} failed on attempt {}: {}", sub.id, attempt, exc)
            if isinstance(exc, LowDiskSpaceError):
                return SubdomainRunResult(
                    subdomain_id=sub.id,
                    status="paused_low_disk",
                    duration_seconds=duration,
                    setup_dir=sub.setup_dir,
                    log_path=log_path,
                    error=repr(exc),
                    run_manifest=run_manifest_path,
                    dropped_events=dropped_events,
                )
            if attempt > retries:
                return SubdomainRunResult(
                    subdomain_id=sub.id,
                    status="failed",
                    duration_seconds=duration,
                    setup_dir=sub.setup_dir,
                    log_path=log_path,
                    error=repr(exc),
                    run_manifest=run_manifest_path,
                    dropped_events=dropped_events,
                )
    return SubdomainRunResult(
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
    complete = bool(manifest.subdomains) and all(
        str(meta.status).lower() == "success" for meta in manifest.subdomains.values()
    )
    if complete:
        event_plan_rows = resolve_subdomain_event_plan(manifest, require_artifacts=True)
        rows = [row for row in event_plan_rows if row["status"] == "dropped"]
    else:
        rows = []
        event_plan_rows = []
        for meta in manifest.subdomains.values():
            dropped = list(meta.dropped_events or [])
            rows.extend(dropped)
            for row in dropped:
                event_plan_rows.append({**row, "status": "dropped"})
            try:
                for event in load_assimilation_events(meta.project_dir):
                    event_plan_rows.append(
                        {
                            "subdomain_id": meta.id,
                            "date": event.date.isoformat(),
                            "assimilation_time": "",
                            "variable": event.variable,
                            "product": event.product or "",
                            "reason": "",
                            "metric": "",
                            "value": "",
                            "threshold": "",
                            "active_station_ids": "",
                            "project_yaml": str(meta.project_yaml),
                            "status": "kept",
                        }
                    )
            except Exception as exc:
                logger.warning("Could not read partial event plan for sub-domain {}: {}", meta.id, exc)
    columns = [
        "subdomain_id",
        "date",
        "assimilation_time",
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
    plan_columns = [*columns, "status"]
    plan_out = manifest.project_dir / "results" / "event_plan_by_subdomain.csv"
    if event_plan_rows:
        plan_df = pd.DataFrame(event_plan_rows, columns=plan_columns).sort_values(
            ["subdomain_id", "date", "variable", "status"]
        )
    else:
        plan_df = pd.DataFrame(columns=plan_columns)
    plan_df.to_csv(plan_out, index=False)


def _project_has_started(project_dir: Path) -> bool:
    return any(
        path.is_file()
        for pattern in (
            "steps/step_*/assim/prior_forcing_manifest.json",
            "steps/step_*/assim/rejuvenate_manifest.json",
            "steps/step_*/ensembles/*/*/results/member_run.json",
        )
        for path in project_dir.glob(pattern)
    )


def _deterministic_preparation_reserve(
    manifest: SubdomainManifest,
    selected_ids: list[str],
) -> dict[str, int]:
    """Conservatively bound preparation outputs plus atomic coexistence."""
    by_leaf: dict[str, int] = {}
    for sid in selected_ids:
        source_bytes = sum(
            path.stat().st_size
            for root in _leaf_scientific_input_paths(manifest, sid)
            for path in recursive_files(root)
        )
        existing_bytes = sum(
            path.stat().st_size
            for path in recursive_files(manifest.subdomains[sid].setup_dir / "obs")
        )
        try:
            event_count = len(
                load_assimilation_events(manifest.subdomains[sid].project_dir)
            )
        except (ValueError, FileNotFoundError):
            event_count = 1
        metadata_bound = (event_count + 2) * 1024 * 1024
        by_leaf[sid] = 2 * (source_bytes + existing_bytes + metadata_bound)
    return by_leaf


def _leaf_scientific_input_paths(
    manifest: SubdomainManifest,
    sid: str,
) -> tuple[Path, ...]:
    subdomain = manifest.subdomains[sid]
    variables = {
        event.variable
        for event in load_assimilation_events(subdomain.project_dir)
    }
    paths: list[Path] = []
    if "scf" in variables:
        paths.append(manifest.raw_snowcover_dir)
    if {"wet_snow", "wet_snow_line"} & variables:
        paths.append(manifest.raw_wetsnow_dir)
    if {"scf", "wet_snow", "wet_snow_line"} & variables:
        paths.append(
            getattr(subdomain, "roi_vector_path", subdomain.setup_dir / "env")
        )
    project_cfg = _read_yaml_file(subdomain.project_yaml) or {}
    obs_cfg = project_cfg.get("obs")
    if isinstance(obs_cfg, dict):
        used_sections: list[str] = []
        if "scf" in variables:
            used_sections.append("snowcover")
        if {"wet_snow", "wet_snow_line"} & variables:
            used_sections.append("wetsnow")
        for key in used_sections:
            section = obs_cfg.get(key)
            if not isinstance(section, dict):
                continue
            manifest_path = section.get("acquisition_manifest")
            if manifest_path is not None:
                relative = Path(str(manifest_path))
                if relative.is_absolute():
                    raise ValueError(
                        f"Sub-domain acquisition manifest must be setup-relative: {relative}"
                    )
                parent_support = manifest.setup_dir / relative
                leaf_support = subdomain.setup_dir / relative
                try:
                    parent_support.resolve().relative_to(manifest.setup_dir.resolve())
                    leaf_support.resolve().relative_to(subdomain.setup_dir.resolve())
                except ValueError as exc:
                    raise ValueError(
                        f"Sub-domain acquisition manifest escapes its setup: {relative}"
                    ) from exc
                if not parent_support.is_file() or parent_support.is_symlink():
                    raise FileNotFoundError(
                        f"Authoritative acquisition manifest is missing: {parent_support}"
                    )
                if not leaf_support.is_file() or leaf_support.is_symlink():
                    raise FileNotFoundError(
                        f"Leaf acquisition manifest is missing: {leaf_support}"
                    )
                if parent_support.read_bytes() != leaf_support.read_bytes():
                    raise RuntimeError(
                        "Leaf acquisition manifest differs from its authoritative "
                        f"parent source: {leaf_support}"
                    )
                # Bind both the canonical source and the file actually consumed
                # by leaf preparation. The coordinator rehashes both after
                # preparation, so mutations on either side fail the wave gate.
                paths.extend((parent_support, leaf_support))
    return tuple(paths)


def _projected_retained_compact_bytes(
    manifest: SubdomainManifest,
    *,
    selected_ids: list[str],
    overwrite: bool,
) -> dict[str, int]:
    """Project queued post-finalization products from prepared leaf inventories."""
    projected: dict[str, int] = {}
    for sid in selected_ids:
        subdomain = manifest.subdomains[sid]
        estimate = estimate_project_storage_components(
            setup_dir=subdomain.setup_dir,
            project_dir=subdomain.project_dir,
            overwrite=overwrite,
            grid_cell_count=int(subdomain.window.height) * int(subdomain.window.width),
        )
        projected[sid] = (
            estimate.retained_compact_bytes
            if output_retention_mode(subdomain.project_dir) == "compact"
            else estimate.total_bytes
        )
    return projected


def _parent_finalization_reserve(
    manifest: SubdomainManifest,
    *,
    overwrite: bool,
) -> int:
    """Return parent merge/render growth without estimating any leaf again."""
    merge_stage = manifest.stages.get("merge") or {}
    merged_output = manifest.project_dir / "results" / "grids" / "da_output_grids.nc"
    merge_is_accepted = (
        not overwrite
        and str(merge_stage.get("status", "")).lower() == "completed"
        and merged_output.is_file()
    )
    if merge_is_accepted:
        try:
            validate_compact_output_file(
                project_dir=manifest.project_dir,
                output_nc=merged_output,
            )
        except Exception:  # noqa: BLE001 - invalid accepted output must be rebuilt
            merge_is_accepted = False
    parent_merge_reserve = (
        0
        if merge_is_accepted
        else estimate_parent_compact_merge_bytes(
            setup_dir=manifest.setup_dir,
            project_dir=manifest.project_dir,
            grid_cell_count=int(manifest.grid_rows) * int(manifest.grid_cols),
        )
    )
    render_stage = manifest.stages.get("render") or {}
    render_outputs = render_stage.get("outputs")
    render_is_accepted = (
        not overwrite
        and str(render_stage.get("status", "")).lower() == "completed"
        and isinstance(render_outputs, list)
        and bool(render_outputs)
        and all(Path(str(path)).is_file() for path in render_outputs)
    )
    parent_render_reserve = (
        0
        if render_is_accepted
        else estimate_parent_render_bytes(
            project_dir=manifest.project_dir,
            grid_cell_count=int(manifest.grid_rows) * int(manifest.grid_cols),
            overwrite=overwrite,
        )
    )
    return parent_merge_reserve + parent_render_reserve


def _coordinator_storage_reserve(
    manifest: SubdomainManifest,
    *,
    selected_ids: list[str],
    queued_ids: list[str] | None = None,
    queued_retained_by_id: dict[str, int] | None = None,
    outer_workers: int,
    overwrite: bool,
    reservation_started_ns: int = 0,
) -> tuple[
    int,
    dict[str, int],
    tuple[StorageReservationProject, ...],
    int,
    int,
]:
    """Reserve active-leaf growth and unfinished parent finalization."""
    parent_device = manifest.project_dir.resolve().stat().st_dev
    projects: list[StorageReservationProject] = []
    for sid in selected_ids:
        subdomain = manifest.subdomains[sid]
        leaf_device = subdomain.project_dir.resolve().stat().st_dev
        if leaf_device != parent_device:
            raise ValueError(
                "Bounded subdomain storage admission requires the parent and all selected "
                f"leaf projects to share one filesystem; {sid} is on another device"
            )
        projects.append(
            StorageReservationProject(
                setup_dir=subdomain.setup_dir.resolve(),
                project_dir=subdomain.project_dir.resolve(),
                grid_cell_count=int(subdomain.window.height) * int(subdomain.window.width),
                run_manifest=(subdomain.setup_dir / "run_manifest.json").resolve(),
                completion_not_before_ns=(reservation_started_ns if overwrite else 0),
                scientific_input_paths=_leaf_scientific_input_paths(manifest, sid),
                scientific_root=manifest.setup_dir,
            )
        )
    project_specs = tuple(projects)
    queued_ids = list(queued_ids or [])
    for sid in queued_ids:
        subdomain = manifest.subdomains[sid]
        leaf_device = subdomain.project_dir.resolve().stat().st_dev
        if leaf_device != parent_device:
            raise ValueError(
                "Bounded subdomain storage admission requires the parent and all queued "
                f"leaf projects to share one filesystem; {sid} is on another device"
            )
    if queued_retained_by_id is None:
        queued_retained_by_id = _projected_retained_compact_bytes(
            manifest,
            selected_ids=queued_ids,
            overwrite=overwrite,
        )
    missing_projection = sorted(set(queued_ids) - set(queued_retained_by_id))
    if missing_projection:
        raise ValueError(
            "Queued compact-retention projection is missing subdomain(s): "
            + ", ".join(missing_projection)
        )
    queued_retained_reserve = sum(
        int(queued_retained_by_id[sid]) for sid in queued_ids
    )
    parent_finalization_reserve = _parent_finalization_reserve(
        manifest,
        overwrite=overwrite,
    )
    concurrent, estimates = estimate_coordinated_storage_reserve(
        project_specs,
        outer_workers=outer_workers,
        parent_finalization_reserve_bytes=(
            parent_finalization_reserve + queued_retained_reserve
        ),
        overwrite=overwrite,
    )
    reserves = {
        sid: estimates.get(str(manifest.subdomains[sid].project_dir.resolve())).total_bytes
        if str(manifest.subdomains[sid].project_dir.resolve()) in estimates
        else 0
        for sid in selected_ids
    }
    return (
        concurrent,
        reserves,
        project_specs,
        parent_finalization_reserve,
        queued_retained_reserve,
    )


def _leaf_waves(selected_ids: list[str], outer_workers: int) -> list[list[str]]:
    """Partition leaves into deterministic storage-admission cohorts."""
    if outer_workers < 1:
        raise ValueError("outer_workers must be positive")
    return [
        selected_ids[index : index + outer_workers]
        for index in range(0, len(selected_ids), outer_workers)
    ]


def _all_storage_reservation_projects(
    manifest: SubdomainManifest,
    *,
    selected_ids: list[str],
    overwrite: bool,
    reservation_started_ns: int,
    preparation_by_id: dict[str, int] | None = None,
) -> tuple[StorageReservationProject, ...]:
    parent_device = manifest.project_dir.resolve().stat().st_dev
    projects: list[StorageReservationProject] = []
    preparation_by_id = preparation_by_id or {}
    for sid in selected_ids:
        subdomain = manifest.subdomains[sid]
        if subdomain.project_dir.resolve().stat().st_dev != parent_device:
            raise ValueError(
                "Bounded subdomain storage admission requires all leaves and the "
                f"parent to share one filesystem; {sid} is on another device"
            )
        projects.append(
            StorageReservationProject(
                setup_dir=subdomain.setup_dir.resolve(),
                project_dir=subdomain.project_dir.resolve(),
                grid_cell_count=int(subdomain.window.height)
                * int(subdomain.window.width),
                run_manifest=(subdomain.setup_dir / "run_manifest.json").resolve(),
                completion_not_before_ns=(reservation_started_ns if overwrite else 0),
                scientific_input_paths=_leaf_scientific_input_paths(manifest, sid),
                scientific_root=manifest.setup_dir,
                preparation_bytes=int(preparation_by_id.get(sid, 0)),
                requires_preparation=True,
            )
        )
    return tuple(projects)


def _start_wave_storage_coordinator(
    *,
    root_project_dir: Path,
    projects: tuple[StorageReservationProject, ...],
    leaf_ids: list[str],
    waves: list[list[str]],
    queued_retained_by_id: dict[str, int],
    outer_workers: int,
    shared_reserve_bytes: int,
    overwrite: bool,
    allow_existing_step_drain: bool,
) -> tuple[StorageAdmissionServer, DiskBudgetSnapshot]:
    """Build, persist and preflight one spawn-safe wave coordinator."""
    plan = build_storage_plan(
        root_project_dir=root_project_dir,
        projects=projects,
        outer_workers=outer_workers,
        parent_finalization_reserve_bytes=shared_reserve_bytes,
        overwrite=overwrite,
        leaf_ids=tuple(leaf_ids),
        waves=tuple(tuple(wave) for wave in waves),
        queued_retained_by_id=queued_retained_by_id,
    )
    budget = check_step_admission(
        root_project_dir,
        estimated_growth_bytes=plan.estimated_growth_bytes,
        allow_existing_step_drain=allow_existing_step_drain,
    )
    coordinator = StorageAdmissionCoordinator(plan)
    server = StorageAdmissionServer(coordinator)
    try:
        coordinator.record_preflight(budget)
        budget = coordinator.admit_wave(
            0,
            request_id="wave:0",
            allow_existing_step_drain=allow_existing_step_drain,
        )
    except Exception:
        server.close()
        raise
    return server, budget


def _prepare_and_preadmit_wave(
    *,
    manifest: SubdomainManifest,
    wave_server: StorageAdmissionServer,
    wave_index: int,
    wave_ids: list[str],
    overwrite: bool,
) -> DiskBudgetSnapshot:
    """Prepare unfinished leaves and pre-admit their first propagation step."""
    leaf_states = wave_server.coordinator.snapshot()["leaves"]
    for sid in wave_ids:
        if leaf_states[sid]["phase"] == "finalized":
            continue
        _prepare_obs_for_subdomain(
            manifest.subdomains[sid],
            manifest,
            overwrite=overwrite,
            scientific_identity=wave_server.client(leaf_id=sid).leaf_identity,
        )
    storage_budget = wave_server.coordinator.prepare_wave(
        wave_index,
        request_id=f"wave_prepared:{wave_index}",
    )
    leaf_states = wave_server.coordinator.snapshot()["leaves"]
    for sid in wave_ids:
        if leaf_states[sid]["phase"] == "finalized":
            continue
        first_step = wave_server.coordinator.plan.leaves[sid].step_names[0]
        wave_server.client(leaf_id=sid).admit_step(
            first_step,
            request_id=f"{sid}:admit:{first_step}",
            allow_existing_step_drain=_project_has_started(
                manifest.subdomains[sid].project_dir
            ),
        )
    return storage_budget


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
) -> List[SubdomainRunResult]:
    """Run sub-domain DA workflows in parallel and stop on first failure."""
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "subdomain":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='subdomain'.")
    ensure_run_mode(manifest.project_dir, expected="subdomain", write_if_missing=False)
    selected_ids = list(subdomains) if subdomains else list(manifest.subdomains.keys())
    if not selected_ids:
        raise ValueError("At least one sub-domain must be selected")
    unknown = [sid for sid in selected_ids if sid not in manifest.subdomains]
    if unknown:
        raise ValueError(f"Sub-domains not in manifest: {', '.join(unknown)}")
    root_log = manifest.project_dir / "subdomain_run.log"
    sink_id = None

    outer_workers = pick_max_workers(max_workers, fallback=len(selected_ids), limit=len(selected_ids))
    cpu = os.cpu_count() or 1
    auto_inner = max(1, cpu // max(1, outer_workers))
    inner_workers = int(inner_max_workers) if inner_max_workers is not None else auto_inner
    inner_workers = max(1, inner_workers)

    waves = _leaf_waves(selected_ids, outer_workers)
    reservation_started_ns = time.time_ns()
    try:
        preparation_by_id = _deterministic_preparation_reserve(
            manifest,
            selected_ids,
        )
        parent_finalization_reserve = _parent_finalization_reserve(
            manifest,
            overwrite=overwrite,
        )
        resuming_batch = not overwrite and all(
            _project_has_started(manifest.subdomains[sid].project_dir)
            for sid in waves[0]
        )
        all_storage_projects = _all_storage_reservation_projects(
            manifest,
            selected_ids=selected_ids,
            overwrite=overwrite,
            reservation_started_ns=reservation_started_ns,
            preparation_by_id=preparation_by_id,
        )
        wave_server, storage_budget = _start_wave_storage_coordinator(
            root_project_dir=manifest.project_dir,
            projects=all_storage_projects,
            leaf_ids=selected_ids,
            waves=waves,
            queued_retained_by_id={},
            outer_workers=outer_workers,
            shared_reserve_bytes=parent_finalization_reserve,
            overwrite=overwrite,
            allow_existing_step_drain=resuming_batch,
        )
        plan = wave_server.coordinator.plan
        storage_reservation_projects = all_storage_projects
        concurrent_storage_reserve = plan.estimated_growth_bytes
        leaf_storage_reserves = {
            leaf_id: leaf.total_bytes for leaf_id, leaf in plan.leaves.items()
        }
        queued_retained_by_id = {
            leaf_id: int(leaf.queued_retained_bytes)
            for leaf_id, leaf in plan.leaves.items()
        }
        queued_retained_reserve = sum(
            queued_retained_by_id[sid]
            for wave in waves[1:]
            for sid in wave
        )
    except LowDiskSpaceError as exc:
        save_stage(manifest, manifest_path, "run", "paused_low_disk", error=str(exc))
        raise
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
    save_stage(manifest, manifest_path, "run", "running")

    logger.info(
        "START sub-domain run count={} outer_workers={} inner_workers={} "
        "first_wave_storage_reserve_gib={:.1f} parent_finalization_reserve_gib={:.1f} "
        "queued_retained_reserve_gib={:.1f} retained_leaf_gib={:.1f} "
        "used={:.1%} fail_fast=true",
        len(selected_ids),
        outer_workers,
        inner_workers,
        concurrent_storage_reserve / (1024**3),
        parent_finalization_reserve / (1024**3),
        queued_retained_reserve / (1024**3),
        _measured_retained_leaf_bytes(manifest) / (1024**3),
        storage_budget.used_fraction,
    )
    for sid in sorted(leaf_storage_reserves):
        logger.info(
            "Storage reserve sub-domain={} reserve_gib={:.1f}",
            sid,
            leaf_storage_reserves[sid] / (1024**3),
        )

    perf_stop = None
    if perf_monitor:
        perf_stop = start_perf_monitor(
            PerfMonitorConfig(project_dir=manifest.project_dir, sample_interval_sec=5.0, plot_interval_sec=30.0)
        )

    results: List[SubdomainRunResult] = []
    failed_id: str | None = None

    ctx = mp.get_context("spawn")
    executor = cf.ProcessPoolExecutor(max_workers=outer_workers, mp_context=ctx)
    try:
        for wave_index, wave_ids in enumerate(waves):
            if wave_index > 0:
                manifest = SubdomainManifest.load(manifest_path)
                resuming_batch = not overwrite and all(
                    _project_has_started(manifest.subdomains[sid].project_dir)
                    for sid in wave_ids
                )
                storage_budget = wave_server.client(
                    leaf_id=wave_ids[0]
                ).admit_wave(
                    wave_index,
                    request_id=f"wave:{wave_index}",
                    allow_existing_step_drain=resuming_batch,
                )
                ledger = wave_server.coordinator.snapshot()
                concurrent_storage_reserve = storage_budget.estimated_growth_bytes
                queued_retained_reserve = int(
                    ledger["queued_retained_reserve_bytes"]
                )
                parent_finalization_reserve = int(
                    ledger["parent_finalization_reserve_bytes"]
                )
            logger.info(
                "ADMIT wave={}/{} subdomains={} growth_gib={:.1f} "
                "retained_leaf_gib={:.1f} queued_retained_gib={:.1f} "
                "parent_finalization_gib={:.1f} used={:.1%}",
                wave_index + 1,
                len(waves),
                ",".join(wave_ids),
                concurrent_storage_reserve / (1024**3),
                _measured_retained_leaf_bytes(manifest) / (1024**3),
                queued_retained_reserve / (1024**3),
                parent_finalization_reserve / (1024**3),
                storage_budget.used_fraction,
            )
            # Preparation happens only after the coordinated preflight.  Once
            # every active leaf has produced or validated its deterministic
            # preparation manifest, the coordinator rehashes shared raw inputs
            # once and atomically admits the whole prepared wave.  Workers then
            # consume the frozen preparation without writing it again.
            storage_budget = _prepare_and_preadmit_wave(
                manifest=manifest,
                wave_server=wave_server,
                wave_index=wave_index,
                wave_ids=wave_ids,
                overwrite=overwrite,
            )
            future_map: dict[cf.Future, str] = {}
            for sid in wave_ids:
                fut = executor.submit(
                    _run_one,
                    sid,
                    manifest_path,
                    inner_workers,
                    overwrite,
                    int(max(0, retries)),
                    log_level,
                    root_log if log_to_file else None,
                    storage_reservation_projects,
                    len(wave_ids),
                    parent_finalization_reserve + queued_retained_reserve,
                    wave_server.client(leaf_id=sid),
                    True,
                )
                future_map[fut] = sid

            for fut in cf.as_completed(future_map):
                sid = future_map[fut]
                res = fut.result()
                results.append(res)
                meta = manifest.subdomains[sid]
                meta.status = "success" if res.status == "skipped" else res.status
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
                if res.status in {"failed", "paused_low_disk"}:
                    failed_id = sid
                    logger.error("Fail-fast triggered by sub-domain {}", sid)
                    for other in future_map:
                        if not other.done():
                            other.cancel()
                    break
            if failed_id is not None:
                executor.shutdown(wait=False, cancel_futures=True)
                break
    except BaseException as exc:
        current = SubdomainManifest.load(manifest_path)
        save_stage(
            current,
            manifest_path,
            "run",
            terminal_status(exc),
            error=str(exc),
        )
        raise
    finally:
        if perf_stop:
            perf_stop.stop_and_join()
            perf_stop.capture_now()
        # Ensure process cleanup.
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception:
            pass
        try:
            wave_server.close()
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
    fail = sum(1 for r in results if r.status in {"failed", "paused_low_disk"})
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
        error = f"Sub-domain run failed in {failed_id}; fail-fast stopped remaining tasks."
        final_status = "paused_low_disk" if any(
            result.status == "paused_low_disk" for result in results
        ) else "failed"
        save_stage(manifest, manifest_path, "run", final_status, error=error)
        raise RuntimeError(error)
    save_stage(
        manifest,
        manifest_path,
        "run",
        "completed",
        outputs=(result.run_manifest for result in results if result.run_manifest is not None),
    )
    return results
