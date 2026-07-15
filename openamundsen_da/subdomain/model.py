"""Plain openAMUNDSEN sub-domain execution helpers."""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.subdomain.status import save_stage, terminal_status
from openamundsen_da.util.parallel import pick_max_workers


@dataclass
class ModelRunResult:
    subdomain_id: str
    status: str  # success | failed | skipped
    duration_seconds: float
    setup_dir: Path
    log_path: Path
    error: Optional[str] = None
    run_manifest: Optional[Path] = None


def _write_run_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _openamundsen_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(key, "1")
    return env


def _stage_model_grid_outputs(results_dir: Path) -> list[Path]:
    """Move openAMUNDSEN grid files into the model sub-domain grids folder."""
    if not results_dir.is_dir():
        return []
    grids_dir = results_dir / "grids"
    staged: list[Path] = []
    for path in sorted(results_dir.iterdir()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix == ".nc" and not path.name.startswith("output_grids"):
            continue
        if suffix not in {".nc", ".tif", ".tiff", ".asc"}:
            continue
        grids_dir.mkdir(parents=True, exist_ok=True)
        target = grids_dir / path.name
        if target.exists():
            target.unlink()
        shutil.move(str(path), target)
        staged.append(target)
    return staged


def _run_one_model(
    subdomain_id: str,
    manifest_path: Path,
    overwrite: bool,
    retries: int,
    log_level: str,
) -> ModelRunResult:
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "model":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='model'.")
    sub = manifest.subdomains[subdomain_id]

    run_manifest_path = sub.setup_dir / "run_manifest.json"
    log_path = sub.setup_dir / "run.log"

    if run_manifest_path.is_file() and not overwrite:
        try:
            data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if str(data.get("status", "")).lower() == "success":
                return ModelRunResult(
                    subdomain_id=sub.id,
                    status="skipped",
                    duration_seconds=0.0,
                    setup_dir=sub.setup_dir,
                    log_path=log_path,
                    run_manifest=run_manifest_path,
                )
        except Exception:
            pass

    executable = shutil.which("openamundsen") or "openamundsen"

    attempt = 0
    while attempt <= retries:
        attempt += 1
        started = time.time()
        command = [executable, str(sub.setup_yaml)]
        run_meta = {
            "id": sub.id,
            "mode": "model",
            "setup_dir": str(sub.setup_dir),
            "setup_yaml": str(sub.setup_yaml),
            "command": command,
            "attempt": attempt,
            "status": "running",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_run_manifest(run_manifest_path, run_meta)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w" if overwrite and attempt == 1 else "a", encoding="utf-8") as log_file:
            log_file.write(
                f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] START sub-domain={sub.id} "
                f"attempt={attempt} log_level={log_level}\n"
            )
            try:
                subprocess.run(
                    command,
                    cwd=sub.setup_dir,
                    env=_openamundsen_env(),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
                staged_grids = _stage_model_grid_outputs(sub.setup_dir / "results")
                duration = time.time() - started
                run_meta.update(
                    {
                        "status": "success",
                        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_seconds": duration,
                        "returncode": 0,
                        "log_path": str(log_path),
                        "staged_grid_outputs": [str(path) for path in staged_grids],
                    }
                )
                _write_run_manifest(run_manifest_path, run_meta)
                return ModelRunResult(
                    subdomain_id=sub.id,
                    status="success",
                    duration_seconds=duration,
                    setup_dir=sub.setup_dir,
                    log_path=log_path,
                    run_manifest=run_manifest_path,
                )
            except Exception as exc:  # noqa: BLE001
                duration = time.time() - started
                returncode = getattr(exc, "returncode", None)
                run_meta.update(
                    {
                        "status": "failed",
                        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "duration_seconds": duration,
                        "returncode": returncode,
                        "error": repr(exc),
                        "log_path": str(log_path),
                    }
                )
                _write_run_manifest(run_manifest_path, run_meta)
                log_file.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] FAILED {repr(exc)}\n")
                if attempt > retries:
                    return ModelRunResult(
                        subdomain_id=sub.id,
                        status="failed",
                        duration_seconds=duration,
                        setup_dir=sub.setup_dir,
                        log_path=log_path,
                        error=repr(exc),
                        run_manifest=run_manifest_path,
                    )

    return ModelRunResult(
        subdomain_id=sub.id,
        status="failed",
        duration_seconds=0.0,
        setup_dir=sub.setup_dir,
        log_path=log_path,
        error="unknown",
        run_manifest=run_manifest_path,
    )


def _record_model_result(
    *,
    manifest: SubdomainManifest,
    manifest_path: Path,
    result: ModelRunResult,
) -> None:
    meta = manifest.subdomains[result.subdomain_id]
    meta.status = "success" if result.status == "skipped" else result.status
    if result.run_manifest:
        meta.run_manifest = result.run_manifest
    manifest.save(manifest_path)


def _mark_unfinished_skipped(
    *,
    manifest: SubdomainManifest,
    manifest_path: Path,
    selected_ids: Iterable[str],
    completed_ids: set[str],
) -> None:
    for sid in selected_ids:
        if sid not in completed_ids:
            manifest.subdomains[sid].status = "skipped"
    manifest.save(manifest_path)


def run_model_subdomains(
    *,
    manifest_path: Path,
    subdomains: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
    retries: int = 0,
    overwrite: bool = False,
    log_level: str = "INFO",
    log_to_file: bool = True,
) -> List[ModelRunResult]:
    """Run plain openAMUNDSEN sub-domain setups in parallel and stop on first failure."""
    manifest_path = Path(manifest_path).resolve()
    manifest = SubdomainManifest.load(manifest_path)
    if str(getattr(manifest, "run_mode", "")).lower() != "model":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='model'.")

    selected_ids = list(subdomains) if subdomains is not None else list(manifest.subdomains.keys())
    unknown = [sid for sid in selected_ids if sid not in manifest.subdomains]
    if unknown:
        raise ValueError(f"Sub-domains not in manifest: {', '.join(unknown)}")
    if not selected_ids:
        return []
    save_stage(manifest, manifest_path, "run", "running")

    root_log = manifest.subdomain_root / "model_run.log"
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
    logger.info(
        "START model sub-domain run count={} workers={} fail_fast=true",
        len(selected_ids),
        outer_workers,
    )

    results: List[ModelRunResult] = []
    failed_id: str | None = None
    completed_ids: set[str] = set()
    try:
        if outer_workers <= 1:
            for sid in selected_ids:
                res = _run_one_model(
                    sid,
                    manifest_path,
                    overwrite,
                    int(max(0, retries)),
                    log_level,
                )
                results.append(res)
                completed_ids.add(res.subdomain_id)
                _record_model_result(manifest=manifest, manifest_path=manifest_path, result=res)
                logger.info("STATUS model sub-domain={} status={}", sid, res.status)
                if res.status == "failed":
                    failed_id = sid
                    break
        else:
            ctx = mp.get_context("spawn")
            executor = cf.ProcessPoolExecutor(max_workers=outer_workers, mp_context=ctx)
            future_map: dict[cf.Future, str] = {}
            try:
                for sid in selected_ids:
                    fut = executor.submit(
                        _run_one_model,
                        sid,
                        manifest_path,
                        overwrite,
                        int(max(0, retries)),
                        log_level,
                    )
                    future_map[fut] = sid

                for fut in cf.as_completed(future_map):
                    sid = future_map[fut]
                    res = fut.result()
                    results.append(res)
                    completed_ids.add(res.subdomain_id)
                    _record_model_result(manifest=manifest, manifest_path=manifest_path, result=res)
                    logger.info("STATUS model sub-domain={} status={}", sid, res.status)
                    if res.status == "failed":
                        failed_id = sid
                        for other in future_map:
                            if not other.done():
                                other.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
            finally:
                try:
                    executor.shutdown(wait=True, cancel_futures=True)
                except Exception:
                    pass

        if failed_id is not None:
            _mark_unfinished_skipped(
                manifest=manifest,
                manifest_path=manifest_path,
                selected_ids=selected_ids,
                completed_ids=completed_ids,
            )
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
        if sink_id is not None:
            logger.remove(sink_id)

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
    if failed_id is not None:
        error = f"Model sub-domain run failed in {failed_id}; fail-fast stopped remaining tasks."
        save_stage(manifest, manifest_path, "run", "failed", error=error)
        raise RuntimeError(error)
    save_stage(
        manifest,
        manifest_path,
        "run",
        "completed",
        outputs=(result.run_manifest for result in results if result.run_manifest is not None),
    )
    return results
