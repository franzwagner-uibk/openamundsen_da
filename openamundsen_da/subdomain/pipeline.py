"""End-to-end sub-domain pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from openamundsen_da.subdomain.merge import (
    CompactCleanupSafetyError,
    cleanup_deferred_compact_grid_artifacts,
    mark_compact_cleanup_artifacts_ready,
    merge_grids,
)
from openamundsen_da.subdomain.prepare import prepare_subdomains
from openamundsen_da.subdomain.report import write_subdomain_reports
from openamundsen_da.subdomain.run import run_subdomains
from openamundsen_da.methods.viz.maps import project_maps_enabled, render_project_maps
from openamundsen_da.pipeline.plot_tasks import render_project_report_best_effort
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.util.run_mode import ensure_run_mode


def run_pipeline(
    *,
    setup_dir: Path,
    project_dir: Path,
    regions_path: Path,
    subdomain_root: Path,
    id_field: str = "id",
    clip_mode: str = "window",
    station_buffer_m: float = 50_000.0,
    roi_buffer_m: float = 0.0,
    grid_buffer_m: float | None = None,
    obs_stations_dir: Optional[Path] = None,
    overlap_area_tol_m2: float = 100.0,
    sliver_fix_m: float = 0.0,
    max_workers: Optional[int] = None,
    inner_max_workers: Optional[int] = None,
    retries: int = 1,
    coverage_sliver_tol_px: int = 4,
    plot_var: str = "snow_depth",
    plot_obs_col: str = "snow_height",
    plot_obs_scale: float = 1.0,
    plot_stations: Optional[list[str]] = None,
    skip_merge: bool = False,
    skip_plot: bool = False,
    overwrite: bool = False,
    log_level: str = "INFO",
    perf_monitor: bool = True,
) -> None:
    """Run prepare -> run -> merge -> plot for sub-domain mode."""
    setup_dir = Path(setup_dir).resolve()
    project_dir = Path(project_dir).resolve()
    ensure_run_mode(project_dir, expected="subdomain", write_if_missing=True)
    subdomain_root = Path(subdomain_root).resolve()
    manifest_path = subdomain_root / "subdomain_manifest.json"
    results_root = project_dir / "results"
    pipeline_log = project_dir / "subdomain_run.log"

    pipeline_log.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(
        pipeline_log,
        level=log_level.upper(),
        colorize=False,
        enqueue=True,
        mode="w" if overwrite else "a",
    )
    logger.info(
        "PIPELINE START setup_dir={} project_dir={} regions={} subdomain_root={}",
        setup_dir,
        project_dir,
        regions_path,
        subdomain_root,
    )
    try:
        prepare_subdomains(
            setup_dir=setup_dir,
            project_dir=project_dir,
            regions_path=regions_path,
            subdomain_root=subdomain_root,
            id_field=id_field,
            clip_mode=clip_mode,
            station_buffer_m=station_buffer_m,
            roi_buffer_m=roi_buffer_m,
            grid_buffer_m=grid_buffer_m,
            obs_stations_dir=obs_stations_dir,
            overlap_area_tol_m2=overlap_area_tol_m2,
            sliver_fix_m=sliver_fix_m,
            overwrite=overwrite,
        )
        logger.info("PREP OK manifest={}", manifest_path)

        perf_stop = None
        if perf_monitor:
            perf_stop = start_perf_monitor(
                PerfMonitorConfig(project_dir=project_dir, sample_interval_sec=5.0, plot_interval_sec=30.0)
            )
        try:
            run_subdomains(
                manifest_path=manifest_path,
                max_workers=max_workers,
                inner_max_workers=inner_max_workers,
                retries=retries,
                overwrite=overwrite,
                log_level=log_level,
                perf_monitor=False,
                log_to_file=False,
            )
            logger.info("RUN OK")
        finally:
            if perf_stop is not None:
                perf_stop.set()

        write_subdomain_reports(
            manifest_path=manifest_path,
            out_dir=results_root,
        )
        logger.info("REPORT OK")

        cleanup_deferred = False
        maps_complete = False
        if skip_merge:
            logger.info("MERGE skipped")
        else:
            merge_grids(
                manifest_path=manifest_path,
                out_dir=results_root / "grids",
                coverage_sliver_tol_px=int(coverage_sliver_tol_px),
                defer_compact_cleanup=True,
            )
            cleanup_deferred = True
            logger.info("MERGE OK (grids)")

        if skip_plot:
            logger.info("PLOT skipped")
        else:
            if skip_merge:
                logger.info("PLOT skipped in subdomain pipeline: merged project maps require merge_grids output.")
            elif not project_maps_enabled(project_dir):
                logger.info("PLOT skipped in subdomain pipeline: no maps.yml found under {}", project_dir)
                maps_complete = True
            else:
                try:
                    outputs = render_project_maps(project_dir=project_dir)
                    logger.info("PLOT OK (project maps, {} output(s))", len(outputs))
                    maps_complete = True
                except Exception as exc:
                    logger.warning(
                        "PLOT failed in subdomain pipeline after merge (variable={}, obs_col={}, obs_scale={}, stations={}): {}",
                        plot_var,
                        plot_obs_col,
                        plot_obs_scale,
                        plot_stations,
                        exc,
                    )
                    logger.warning(
                        "Deferred compact grid cleanup skipped so top-level maps can be rerendered after fixing the plot issue."
                    )

        render_project_report_best_effort(project_dir)

        if cleanup_deferred and maps_complete:
            try:
                lock_path = mark_compact_cleanup_artifacts_ready(
                    project_dir=project_dir,
                    out_dir=results_root / "grids",
                )
                logger.info("Compact cleanup readiness lock written: {}", lock_path)
                archived, bytes_staged = cleanup_deferred_compact_grid_artifacts(
                    manifest_path=manifest_path,
                    out_dir=results_root / "grids",
                )
                logger.info(
                    "Deferred compact grid retention cleanup: archived {} file(s), staged {:.1f} MB",
                    archived,
                    bytes_staged / 1_000_000.0,
                )
            except CompactCleanupSafetyError as exc:
                logger.warning("Deferred compact grid cleanup skipped: {}", exc)
        elif cleanup_deferred:
            logger.info("Deferred compact grid cleanup skipped because top-level maps were not completed.")

        logger.info("PIPELINE DONE subdomain_root={}", subdomain_root)
    finally:
        logger.remove(sink_id)
