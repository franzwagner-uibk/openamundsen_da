from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

from openamundsen_da.batch.prepare import prepare_batch
from openamundsen_da.batch.run import run_batch
from openamundsen_da.batch.merge import merge_grids, merge_points
from openamundsen_da.batch.plot import plot_station_comparisons
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor


def run_pipeline(
    *,
    base_config: Path,
    regions_path: Path,
    batch_root: Path,
    id_field: str = "id",
    clip_mode: str = "window",
    station_buffer_m: float = 50_000.0,
    roi_buffer_m: float = 0.0,
    grid_buffer_m: float | None = None,
    obs_stations_dir: Optional[Path] = None,
    overlap_area_tol_m2: float = 100.0,
    sliver_fix_m: float = 0.0,
    max_workers: Optional[int] = None,
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
    # Single batch-wide log sink for all stages
    batch_log = batch_root / "batch_run.log"
    batch_log.parent.mkdir(parents=True, exist_ok=True)
    sink_id = logger.add(batch_log, level=log_level.upper(), colorize=False, enqueue=True)
    logger.info("PIPELINE START batch_root={} regions={}", batch_root, regions_path)

    manifest = prepare_batch(
        base_config=base_config,
        regions_path=regions_path,
        batch_root=batch_root,
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
    logger.info("PREP OK manifest={}", batch_root / "batch_manifest.json")

    perf_stop = None
    if perf_monitor:
        perf_stop = start_perf_monitor(
            PerfMonitorConfig(project_dir=batch_root, sample_interval_sec=5.0, plot_interval_sec=30.0)
        )

    try:
        run_batch(
            manifest_path=batch_root / "batch_manifest.json",
            max_workers=max_workers,
            retries=retries,
            overwrite=overwrite,
            log_level=log_level,
            perf_monitor=False,  # pipeline-level monitor already running
            log_to_file=True,
        )
        logger.info("RUN OK")
    finally:
        if perf_stop:
            perf_stop.set()

    if not skip_merge:
        merge_grids(
            manifest_path=batch_root / "batch_manifest.json",
            coverage_sliver_tol_px=int(coverage_sliver_tol_px),
        )
        merge_points(manifest_path=batch_root / "batch_manifest.json")
        logger.info("MERGE OK")
    else:
        logger.info("MERGE skipped")

    if not skip_plot:
        plot_station_comparisons(
            manifest_path=batch_root / "batch_manifest.json",
            points_dir=None,
            obs_dir=None,
            variable=plot_var,
            obs_column=plot_obs_col,
            obs_scale=plot_obs_scale,
            station_ids=plot_stations,
        )
        logger.info("PLOT OK")
    else:
        logger.info("PLOT skipped")

    logger.info("PIPELINE DONE batch_root={}", batch_root)
    logger.remove(sink_id)
