"""Parallel execution of per-subregion openAMUNDSEN runs."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from contextlib import redirect_stdout, redirect_stderr

from loguru import logger

from openamundsen import read_config
from openamundsen.model import OpenAmundsen
from openamundsen_da.batch.manifest import BatchManifest
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.util.parallel import pick_max_workers, run_tasks_with_pool
from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import apply_env_from_project, apply_numeric_thread_defaults, ensure_gdal_proj_from_conda
from openamundsen_da.core.runner import (  # reuse runtime patches
    _patch_linear_fit,
    _patch_pandas_inferred_freq,
    _patch_rasterio_transform,
)


def _patch_meteo_resample_allow_downsample() -> None:
    """Relax OA meteo resampling to allow downsampling with aggregation.

    The upstream openamundsen rejects resampling when the target index is not a
    strict subset of the source index (e.g., hourly -> 3‑hourly when timestamps
    are offset from 00:00). For downsampling with aggregation this is harmless,
    so we bypass the strict subset check while still blocking upsampling.
    """
    try:
        import pandas as pd
        from openamundsen import util, errors
        from openamundsen.fileio import meteo as m
    except Exception:
        return

    if getattr(m._resample_dataset, "__oa_da_patched__", False):  # type: ignore[attr-defined]
        return

    orig = m._resample_dataset

    def _patched(ds, start_date, end_date, freq, aggregate=False):  # noqa: N802
        td = util.offset_to_timedelta(freq)
        td_1d = pd.Timedelta("1d")
        if td < td_1d:
            resample_kwargs = {
                "label": "right",
                "closed": "right",
                "origin": pd.Timestamp(start_date),
            }
        elif td == td_1d:
            resample_kwargs = {
                "label": "left",
                "closed": "right",
                "origin": "start",
            }
        else:
            raise errors.MeteoDataError("Resampling to frequencies > 1 day is not supported")

        if ds.sizes["time"] == 0:
            return ds

        df = ds.to_dataframe().drop(columns=["lon", "lat", "alt"])

        if aggregate:
            df_res = df.resample(freq, **resample_kwargs).mean()
            if "wind_dir" in df.columns:
                df_res["wind_dir"] = m._aggregate_wind_dir(df, freq, resample_kwargs)  # type: ignore[attr-defined]
            df_res = df_res.loc[start_date:end_date]
            if df_res.index[0] < df.index[0]:
                df_res = df_res.iloc[1:]
        else:
            df_res = df.reindex(
                pd.date_range(
                    start=start_date,
                    end=end_date,
                    freq=freq,
                    name="time",
                )
            ).loc[df.index[0] : df.index[-1]]

        if "precip" in df:
            df_res["precip"] = (
                df["precip"]
                .resample(freq, **resample_kwargs)
                .agg(pd.Series.sum, skipna=False)
                .reindex(df_res.index)
            )

        if "wind_speed_gust" in df.columns:
            df_res["wind_speed_gust"] = (
                df["wind_speed_gust"]
                .resample(freq, **resample_kwargs)
                .max()
                .reindex(df_res.index)
            )

        inferred = df.index.inferred_freq
        try:
            src_td = pd.Timedelta(inferred) if inferred else None
        except Exception:
            src_td = None

        # Allow downsampling when aggregate=True and target period is >= source period.
        if not aggregate:
            if not df.index.intersection(df_res.index).equals(df_res.index):
                raise errors.MeteoDataError(
                    f'Resampling from freq "{inferred}" to "{freq}" not supported'
                )
        else:
            if src_td is not None and td < src_td:
                raise errors.MeteoDataError(
                    f'Resampling from freq "{inferred}" to "{freq}" not supported'
                )

        ds_res = ds[["lon", "lat", "alt", "time"]]
        ds_res["time"] = df_res.index
        ds_res.attrs = ds.attrs
        for param in df_res.columns:
            ds_res[param] = df_res[param]
        return ds_res

    _patched.__oa_da_patched__ = True
    m._resample_dataset = _patched



@dataclass
class RunResult:
    subregion_id: str
    status: str  # success | failed | skipped
    duration_seconds: float
    results_dir: Path
    log_path: Path
    error: Optional[str] = None
    run_manifest: Optional[Path] = None


def _write_run_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def _run_one(
    sub_id: str,
    manifest: BatchManifest,
    log_level: str,
    overwrite: bool,
    retries: int,
    batch_log: Path | None = None,
) -> RunResult:
    """Worker: run one subregion open loop."""
    sub = manifest.subregions[sub_id]
    # Use setup root for log/manifest (no extra log subdir)
    run_dir = sub.results_dir.parent  # = setup_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    run_manifest_path = run_dir / "run_manifest.json"

    cfg = read_config(sub.config_path)

    results_dir = Path(cfg.get("results_dir", sub.results_dir))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already successful and not overwriting
    if run_manifest_path.is_file() and not overwrite:
        try:
            data = json.loads(run_manifest_path.read_text(encoding="utf-8"))
            if str(data.get("status", "")).lower() == "success":
                return RunResult(sub.id, "skipped", 0.0, results_dir, log_path, run_manifest=run_manifest_path)
        except Exception:
            pass

    # Configure logging for worker: local run.log + optional batch log
    logger.remove()
    logger.add(log_path.open("a", encoding="utf-8"), level=log_level.upper(), colorize=False, enqueue=False, format=LOGURU_FORMAT)
    if batch_log:
        batch_log.parent.mkdir(parents=True, exist_ok=True)
        logger.add(batch_log, level=log_level.upper(), colorize=False, enqueue=True, format=LOGURU_FORMAT)

    attempt = 0
    while attempt <= retries:
        attempt += 1
        apply_numeric_thread_defaults()
        ensure_gdal_proj_from_conda()
        apply_env_from_project(manifest.base_config)
        _patch_pandas_inferred_freq()
        _patch_rasterio_transform()
        _patch_linear_fit()
        _patch_meteo_resample_allow_downsample()

        logger.info("START subregion={} attempt={}", sub.id, attempt)
        manifest_data = {
            "id": sub.id,
            "config": str(sub.config_path),
            "results_dir": str(results_dir),
            "attempt": attempt,
            "status": "starting",
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_run_manifest(run_manifest_path, manifest_data)

        start = time.time()
        try:
            os.chdir(sub.setup_dir)
            # Capture stdout/stderr from upstream libraries into the run log for parity with season pipeline.
            with log_path.open("a", encoding="utf-8") as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
                model = OpenAmundsen(cfg)
                model.initialize()
                model.run()
            duration = time.time() - start
            manifest_data.update(
                {
                    "status": "success",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                }
            )
            _write_run_manifest(run_manifest_path, manifest_data)
            logger.info("OK subregion={} duration_s={:.1f}", sub.id, duration)
            return RunResult(sub.id, "success", duration, results_dir, log_path, run_manifest=run_manifest_path)
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start
            manifest_data.update(
                {
                    "status": "failed",
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_seconds": duration,
                    "error": repr(exc),
                }
            )
            _write_run_manifest(run_manifest_path, manifest_data)
            logger.exception("Subregion {} failed on attempt {}: {}", sub.id, attempt, exc)
            logger.error("FAIL subregion={} attempt={} err={}", sub.id, attempt, exc)
            if attempt > retries:
                return RunResult(sub.id, "failed", duration, results_dir, log_path, error=repr(exc), run_manifest=run_manifest_path)
    # should not reach
    return RunResult(sub.id, "failed", 0.0, results_dir, log_path, error="unknown", run_manifest=run_manifest_path)


def run_batch(
    *,
    manifest_path: Path,
    subregions: Optional[Iterable[str]] = None,
    max_workers: Optional[int] = None,
    retries: int = 1,
    overwrite: bool = False,
    log_level: str = "INFO",
    perf_monitor: bool = True,
    log_to_file: bool = True,
) -> List[RunResult]:
    """Run open-loop simulations for all (or selected) subregions in parallel."""
    # Make parent processes safe to fork with numeric libs that spin OpenMP threads
    # (avoids "fork() called from a process already using GNU OpenMP" crashes).
    apply_numeric_thread_defaults()

    manifest = BatchManifest.load(manifest_path)
    batch_root = manifest_path.parent
    selected_ids = list(subregions) if subregions else list(manifest.subregions.keys())
    unknown = [s for s in selected_ids if s not in manifest.subregions]
    if unknown:
        raise ValueError(f"Subregions not in manifest: {', '.join(unknown)}")

    batch_log = batch_root / "batch_run.log"
    tasks = [(sid, manifest, log_level, overwrite, retries, batch_log if log_to_file else None) for sid in selected_ids]

    workers = pick_max_workers(max_workers, fallback=len(tasks), limit=len(tasks))
    # Add batch-level file sink when requested
    sink_id = None
    if log_to_file:
        batch_log.parent.mkdir(parents=True, exist_ok=True)
        sink_id = logger.add(batch_log, level=log_level.upper(), colorize=False, enqueue=True, format=LOGURU_FORMAT)

    logger.info("START batch subregions={} workers={}", len(tasks), workers)

    perf_stop = None
    if perf_monitor:
        perf_stop = start_perf_monitor(
            PerfMonitorConfig(season_dir=batch_root, sample_interval_sec=5.0, plot_interval_sec=30.0)
        )

    results: List[RunResult] = []
    try:
        run_results = run_tasks_with_pool(
            _run_one,
            tasks,
            max_workers=workers,
            fallback_workers=len(tasks),
            unpack=True,
        )
        results.extend(run_results)
    finally:
        if perf_stop:
            perf_stop.set()

    # Update manifest statuses
    for res in results:
        meta = manifest.subregions[res.subregion_id]
        meta.status = res.status
        if res.run_manifest:
            meta.run_manifest = res.run_manifest
        logger.info("STATUS subregion={} status={} duration_s={:.1f} log={}", res.subregion_id, res.status, res.duration_seconds, res.log_path)
    manifest.save(manifest_path)
    logger.info("Updated manifest statuses -> {}", manifest_path)
    ok = sum(1 for r in results if r.status == "success")
    fail = sum(1 for r in results if r.status == "failed")
    skip = sum(1 for r in results if r.status == "skipped")
    logger.info("Summary: total={} success={} failed={} skipped={}", len(results), ok, fail, skip)
    if fail:
        failed_ids = [r.subregion_id for r in results if r.status == "failed"]
        logger.warning("Failed subregions: {}", ", ".join(failed_ids))
    logger.info("SUMMARY total={} success={} failed={} skipped={} at {}", len(results), ok, fail, skip, time.strftime('%Y-%m-%d %H:%M:%S'))
    if sink_id is not None:
        logger.remove(sink_id)
    return results
