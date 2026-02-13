"""End-to-end sub-domain pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from openamundsen_da.subdomain.merge import merge_grids, merge_points
from openamundsen_da.subdomain.plot import plot_station_comparisons
from openamundsen_da.subdomain.prepare import prepare_subdomains
from openamundsen_da.subdomain.run import run_subdomains
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor


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
    subdomain_root = Path(subdomain_root).resolve()
    manifest_path = subdomain_root / "subdomain_manifest.json"
    pipeline_log = subdomain_root / "subdomain_run.log"

    pipeline_log.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(pipeline_log, level=log_level.upper(), colorize=False, enqueue=True)
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
                PerfMonitorConfig(project_dir=subdomain_root, sample_interval_sec=5.0, plot_interval_sec=30.0)
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

        if skip_merge:
            logger.info("MERGE skipped")
        else:
            merged_root = subdomain_root / "merged"
            merge_grids(
                manifest_path=manifest_path,
                out_dir=merged_root / "grids",
                coverage_sliver_tol_px=int(coverage_sliver_tol_px),
            )
            merge_points(
                manifest_path=manifest_path,
                out_dir=merged_root / "points",
            )
            logger.info("MERGE OK")

        if skip_plot:
            logger.info("PLOT skipped")
        else:
            plot_station_comparisons(
                manifest_path=manifest_path,
                points_dir=None,
                obs_dir=None,
                variable=plot_var,
                obs_column=plot_obs_col,
                obs_scale=plot_obs_scale,
                station_ids=plot_stations,
            )
            logger.info("PLOT OK")

        logger.info("PIPELINE DONE subdomain_root={}", subdomain_root)
    finally:
        logger.remove(sink_id)
