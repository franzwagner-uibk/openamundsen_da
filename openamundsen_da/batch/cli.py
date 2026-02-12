"""CLI entrypoint for the sub-domain toolkit."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT


def _default_batch_root(regions_path: Path) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return Path("batch_runs") / f"{regions_path.stem}_{ts}"


def cli(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="oa-da-batch", description="Sub-domain runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare per-sub-domain setups from a regions file")
    p_prep.add_argument("--config", required=True, type=Path, help="Base openAMUNDSEN config (domain.yml)")
    p_prep.add_argument("--regions", required=True, type=Path, help="Regions vector (e.g., GPKG) with polygons")
    p_prep.add_argument("--batch-root", type=Path, help="Root dir for batch (default: batch_runs/<regions>_<ts>)")
    p_prep.add_argument("--id-field", default="id", help="Field name containing sub-domain ID (default: id)")
    p_prep.add_argument("--clip-mode", choices=("window", "roi-symlink"), default="window", help="Grid handling mode")
    p_prep.add_argument("--station-buffer-km", type=float, default=50.0, help="Buffer (km) to include neighboring stations")
    p_prep.add_argument("--roi-buffer-m", type=float, default=0.0, help="Optional ROI buffer (m)")
    p_prep.add_argument("--grid-buffer-m", type=float, help="Grid window buffer (m); defaults to station buffer")
    p_prep.add_argument("--obs-stations-dir", type=Path, help="Override obs stations directory")
    p_prep.add_argument("--overlap-area-tol-m2", type=float, default=100.0, help="Allow tiny overlaps up to this area (m^2) (default: 100)")
    p_prep.add_argument("--sliver-fix-m", type=float, default=0.0, help="Optional shrink/expand buffer (m) to snap sliver overlaps (default: 0)")
    p_prep.add_argument("--overwrite", action="store_true", help="Overwrite existing setups if present")
    p_prep.add_argument("--log-level", default="INFO")

    p_run = sub.add_parser("run", help="Run open-loop simulations for prepared sub-domains")
    p_run.add_argument("--manifest", type=Path, help="Path to batch_manifest.json")
    p_run.add_argument("--batch-root", type=Path, help="Batch root (used when manifest not given)")
    p_run.add_argument("--subregions", nargs="+", help="Optional list of sub-domain IDs to run")
    p_run.add_argument("--max-workers", type=int, help="Max parallel workers")
    p_run.add_argument("--retries", type=int, default=1, help="Retries per sub-domain on failure (default: 1)")
    p_run.add_argument("--overwrite", action="store_true", help="Overwrite even if run manifest reports success")
    p_run.add_argument("--log-level", default="INFO")
    p_run.add_argument("--no-perf-monitor", action="store_true", help="Disable performance monitor")

    p_merge = sub.add_parser("merge", help="Merge outputs across sub-domains")
    p_merge.add_argument("--manifest", type=Path, help="Path to batch_manifest.json")
    p_merge.add_argument("--batch-root", type=Path, help="Batch root (used when manifest not given)")
    p_merge.add_argument("--subregions", nargs="+", help="Optional list of sub-domain IDs to merge")
    p_merge.add_argument("--coverage-sliver-tol-px", type=int, default=4, help="Allowed uncovered expected pixels before merge fails (default: 4)")
    p_merge.add_argument("--out-dir", type=Path, help="Override output directory for merged grids/points")
    p_merge.add_argument("--log-level", default="INFO")

    p_plot = sub.add_parser("plot", help="Plot station obs vs model outputs from merged points")
    p_plot.add_argument("--manifest", type=Path, help="Path to batch_manifest.json")
    p_plot.add_argument("--batch-root", type=Path, help="Batch root (used when manifest not given)")
    p_plot.add_argument("--points-dir", type=Path, help="Merged points directory (default: <batch>/merged/points)")
    p_plot.add_argument("--obs-dir", type=Path, help="Obs directory (default: <points>/obs/stations)")
    p_plot.add_argument("--var", default="snow_depth", help="Model variable column to plot (default: snow_depth)")
    p_plot.add_argument("--obs-col", default="snow_height", help="Observation column name (default: snow_height)")
    p_plot.add_argument("--obs-scale", type=float, default=1.0, help="Scale factor applied to obs values (default: 1)")
    p_plot.add_argument("--stations", nargs="+", help="Optional station IDs to plot")
    p_plot.add_argument("--log-level", default="INFO")

    p_pipe = sub.add_parser("pipeline", help="Run prepare -> run -> merge -> plot in one go")
    p_pipe.add_argument("--config", required=True, type=Path, help="Base openAMUNDSEN config (domain.yml)")
    p_pipe.add_argument("--regions", required=True, type=Path, help="Regions vector (e.g., GPKG) with polygons")
    p_pipe.add_argument("--batch-root", type=Path, help="Root dir for batch (default: batch_runs/<regions>_<ts>)")
    p_pipe.add_argument("--id-field", default="id", help="Field name containing sub-domain ID (default: id)")
    p_pipe.add_argument("--clip-mode", choices=("window", "roi-symlink"), default="window", help="Grid handling mode")
    p_pipe.add_argument("--station-buffer-km", type=float, default=50.0, help="Buffer (km) to include neighboring stations")
    p_pipe.add_argument("--roi-buffer-m", type=float, default=0.0, help="Optional ROI buffer (m)")
    p_pipe.add_argument("--grid-buffer-m", type=float, help="Grid window buffer (m); defaults to station buffer")
    p_pipe.add_argument("--obs-stations-dir", type=Path, help="Override obs stations directory")
    p_pipe.add_argument("--overlap-area-tol-m2", type=float, default=100.0, help="Allow tiny overlaps up to this area (m^2) (default: 100)")
    p_pipe.add_argument("--sliver-fix-m", type=float, default=0.0, help="Optional shrink/expand buffer (m) to snap sliver overlaps (default: 0)")
    p_pipe.add_argument("--max-workers", type=int, help="Max parallel workers for run stage")
    p_pipe.add_argument("--retries", type=int, default=1, help="Retries per sub-domain on failure (default: 1)")
    p_pipe.add_argument("--coverage-sliver-tol-px", type=int, default=4, help="Allowed uncovered expected pixels before merge fails (default: 4)")
    p_pipe.add_argument("--var", default="snow_depth", help="Model variable column to plot (default: snow_depth)")
    p_pipe.add_argument("--obs-col", default="snow_height", help="Observation column name (default: snow_height)")
    p_pipe.add_argument("--obs-scale", type=float, default=1.0, help="Scale factor applied to obs values (default: 1)")
    p_pipe.add_argument("--stations", nargs="+", help="Optional station IDs to plot")
    p_pipe.add_argument("--no-merge", action="store_true", help="Skip merge stage")
    p_pipe.add_argument("--no-plot", action="store_true", help="Skip plot stage")
    p_pipe.add_argument("--overwrite", action="store_true", help="Overwrite setups and rerun sub-domains")
    p_pipe.add_argument("--log-level", default="INFO")
    p_pipe.add_argument("--no-perf-monitor", action="store_true", help="Disable performance monitor during run stage")

    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stderr, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    if args.command == "prepare":
        from openamundsen_da.batch.prepare import prepare_batch

        batch_root = args.batch_root or _default_batch_root(args.regions)
        prepare_batch(
            base_config=args.config,
            regions_path=args.regions,
            batch_root=batch_root,
            id_field=args.id_field,
            clip_mode=args.clip_mode,
            station_buffer_m=args.station_buffer_km * 1000.0,
            roi_buffer_m=args.roi_buffer_m,
            grid_buffer_m=args.grid_buffer_m,
            obs_stations_dir=args.obs_stations_dir,
            overlap_area_tol_m2=args.overlap_area_tol_m2,
            sliver_fix_m=args.sliver_fix_m,
            overwrite=args.overwrite,
        )
        return 0

    if args.command == "run":
        from openamundsen_da.batch.run import run_batch

        manifest = _resolve_manifest(args.manifest, args.batch_root)
        run_batch(
            manifest_path=manifest,
            subregions=args.subregions,
            max_workers=args.max_workers,
            retries=max(0, int(args.retries)),
            overwrite=args.overwrite,
            log_level=args.log_level,
            perf_monitor=not args.no_perf_monitor,
        )
        return 0

    if args.command == "merge":
        from openamundsen_da.batch.merge import merge_grids, merge_points

        manifest = _resolve_manifest(args.manifest, args.batch_root)
        merge_grids(
            manifest_path=manifest,
            subregions=args.subregions,
            out_dir=args.out_dir,
            coverage_sliver_tol_px=int(args.coverage_sliver_tol_px),
        )
        merge_points(
            manifest_path=manifest,
            subregions=args.subregions,
            out_dir=(args.out_dir / "points" if args.out_dir else None),
        )
        return 0

    if args.command == "plot":
        from openamundsen_da.batch.plot import plot_station_comparisons

        manifest = _resolve_manifest(args.manifest, args.batch_root)
        plot_station_comparisons(
            manifest_path=manifest,
            points_dir=args.points_dir,
            obs_dir=args.obs_dir,
            variable=args.var,
            obs_column=args.obs_col,
            obs_scale=args.obs_scale,
            station_ids=args.stations,
        )
        return 0
    if args.command == "pipeline":
        from openamundsen_da.batch.pipeline import run_pipeline

        batch_root = args.batch_root or _default_batch_root(args.regions)
        run_pipeline(
            base_config=args.config,
            regions_path=args.regions,
            batch_root=batch_root,
            id_field=args.id_field,
            clip_mode=args.clip_mode,
            station_buffer_m=args.station_buffer_km * 1000.0,
            roi_buffer_m=args.roi_buffer_m,
            grid_buffer_m=args.grid_buffer_m,
            obs_stations_dir=args.obs_stations_dir,
            overlap_area_tol_m2=args.overlap_area_tol_m2,
            sliver_fix_m=args.sliver_fix_m,
            max_workers=args.max_workers,
            retries=max(0, int(args.retries)),
            coverage_sliver_tol_px=int(args.coverage_sliver_tol_px),
            plot_var=args.var,
            plot_obs_col=args.obs_col,
            plot_obs_scale=args.obs_scale,
            plot_stations=args.stations,
            skip_merge=args.no_merge,
            skip_plot=args.no_plot,
            overwrite=args.overwrite,
            log_level=args.log_level,
            perf_monitor=not args.no_perf_monitor,
        )
        return 0

    return 1


def _resolve_manifest(manifest_arg: Optional[Path], batch_root: Optional[Path]) -> Path:
    if manifest_arg:
        return manifest_arg
    root = batch_root or Path(".")
    cand = root / "batch_manifest.json"
    if not cand.is_file():
        hint = "Run 'oa-da-batch prepare' first or pass --manifest."
        raise FileNotFoundError(f"Manifest not found at {cand}. {hint}")
    return cand


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
