"""CLI entrypoint for the sub-domain toolkit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.util.run_mode import ensure_run_mode


def _default_subdomain_root(project_dir: Path) -> Path:
    return Path(project_dir) / "subdomains"


def _default_regions_path(setup_dir: Path) -> Path:
    env_dir = Path(setup_dir) / "env"
    preferred = [env_dir / "subdomains.gpkg", env_dir / "roi.gpkg"]
    for cand in preferred:
        if cand.is_file():
            return cand
    return preferred[0]


def _resolve_manifest(
    *,
    manifest_arg: Optional[Path],
    project_dir: Optional[Path],
    subdomain_root: Optional[Path],
) -> Path:
    if manifest_arg is not None:
        return Path(manifest_arg)
    root = Path(subdomain_root) if subdomain_root is not None else None
    if root is None and project_dir is not None:
        root = _default_subdomain_root(Path(project_dir))
    if root is None:
        raise FileNotFoundError("Manifest not provided. Pass --manifest or --project-dir (or --subdomain-root).")
    cand = root / "subdomain_manifest.json"
    if not cand.is_file():
        raise FileNotFoundError(f"Manifest not found at {cand}. Run 'oa-da-subdomain prepare' first.")
    return cand


def _configure_logger(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)


def cli(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="oa-da-subdomain", description="Sub-domain mode for openAMUNDSEN-DA")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Prepare per-sub-domain setups from a regions file")
    p_prep.add_argument("--setup-dir", required=True, type=Path, help="Setup root directory")
    p_prep.add_argument("--project-dir", required=True, type=Path, help="Project directory under setup/projects")
    p_prep.add_argument(
        "--roi",
        "--regions",
        dest="regions",
        type=Path,
        help="Sub-domain polygons (default: <setup>/env/subdomains.gpkg, fallback: <setup>/env/roi.gpkg)",
    )
    p_prep.add_argument("--subdomain-root", type=Path, help="Output root (default: <project>/subdomains)")
    p_prep.add_argument("--id-field", default="id", help="Field name containing sub-domain id (default: id)")
    p_prep.add_argument("--clip-mode", choices=("window", "roi-symlink"), default="window", help="Grid handling mode")
    p_prep.add_argument("--station-buffer-km", type=float, default=50.0, help="Station search buffer in km (default: 50)")
    p_prep.add_argument("--roi-buffer-m", type=float, default=0.0, help="Optional ROI buffer in meters")
    p_prep.add_argument("--grid-buffer-m", type=float, help="Grid window buffer in meters (default: station buffer)")
    p_prep.add_argument("--obs-stations-dir", type=Path, help="Override setup obs stations directory")
    p_prep.add_argument("--overlap-area-tol-m2", type=float, default=100.0, help="Allowed overlap area in m^2 (default: 100)")
    p_prep.add_argument("--sliver-fix-m", type=float, default=0.0, help="Optional shrink-expand tolerance fix in meters")
    p_prep.add_argument("--overwrite", action="store_true", help="Overwrite existing sub-domain setups")
    p_prep.add_argument("--log-level", default="INFO")

    p_run = sub.add_parser("run", help="Run prepared sub-domains (DA pipeline per sub-domain)")
    p_run.add_argument("--manifest", type=Path, help="Path to subdomain_manifest.json")
    p_run.add_argument("--project-dir", type=Path, help="Project directory (used to resolve manifest)")
    p_run.add_argument("--subdomain-root", type=Path, help="Sub-domain root (used to resolve manifest)")
    p_run.add_argument("--subdomains", nargs="+", help="Optional list of sub-domain ids to run")
    p_run.add_argument("--max-workers", type=int, help="Parallel sub-domain workers")
    p_run.add_argument("--inner-max-workers", type=int, help="Parallel member workers per sub-domain")
    p_run.add_argument("--retries", type=int, default=1, help="Retries per sub-domain on failure (default: 1)")
    p_run.add_argument("--overwrite", action="store_true", help="Overwrite existing successful run manifests")
    p_run.add_argument("--log-level", default="INFO")
    p_run.add_argument("--no-perf-monitor", action="store_true", help="Disable performance monitor")

    p_merge = sub.add_parser("merge", help="Merge compact DA grids across sub-domains")
    p_merge.add_argument("--manifest", type=Path, help="Path to subdomain_manifest.json")
    p_merge.add_argument("--project-dir", type=Path, help="Project directory (used to resolve manifest)")
    p_merge.add_argument("--subdomain-root", type=Path, help="Sub-domain root (used to resolve manifest)")
    p_merge.add_argument("--subdomains", nargs="+", help="Optional list of sub-domain ids to merge")
    p_merge.add_argument("--coverage-sliver-tol-px", type=int, default=4, help="Allowed uncovered expected pixels (default: 4)")
    p_merge.add_argument("--out-dir", type=Path, help="Override results output directory (default: <project>/results)")
    p_merge.add_argument("--log-level", default="INFO")

    p_plot = sub.add_parser("plot", help="Plot station obs vs consolidated model point outputs")
    p_plot.add_argument("--manifest", type=Path, help="Path to subdomain_manifest.json")
    p_plot.add_argument("--project-dir", type=Path, help="Project directory (used to resolve manifest)")
    p_plot.add_argument("--subdomain-root", type=Path, help="Sub-domain root (used to resolve manifest)")
    p_plot.add_argument("--points-dir", type=Path, help="Points directory (default: <project>/results/points)")
    p_plot.add_argument("--obs-dir", type=Path, help="Obs directory (default: <points_dir>/obs/stations)")
    p_plot.add_argument("--var", default="snow_depth", help="Model variable column to plot")
    p_plot.add_argument("--obs-col", default="snow_depth", help="Observation column name")
    p_plot.add_argument("--obs-scale", type=float, default=1.0, help="Observation scaling factor")
    p_plot.add_argument("--stations", nargs="+", help="Optional list of station ids")
    p_plot.add_argument("--log-level", default="INFO")

    p_pipe = sub.add_parser("pipeline", help="Run prepare -> run -> report -> merge")
    p_pipe.add_argument("--setup-dir", required=True, type=Path, help="Setup root directory")
    p_pipe.add_argument("--project-dir", required=True, type=Path, help="Project directory under setup/projects")
    p_pipe.add_argument(
        "--roi",
        "--regions",
        dest="regions",
        type=Path,
        help="Sub-domain polygons (default: <setup>/env/subdomains.gpkg, fallback: <setup>/env/roi.gpkg)",
    )
    p_pipe.add_argument("--subdomain-root", type=Path, help="Output root (default: <project>/subdomains)")
    p_pipe.add_argument("--id-field", default="id", help="Field name containing sub-domain id (default: id)")
    p_pipe.add_argument("--clip-mode", choices=("window", "roi-symlink"), default="window", help="Grid handling mode")
    p_pipe.add_argument("--station-buffer-km", type=float, default=50.0, help="Station search buffer in km (default: 50)")
    p_pipe.add_argument("--roi-buffer-m", type=float, default=0.0, help="Optional ROI buffer in meters")
    p_pipe.add_argument("--grid-buffer-m", type=float, help="Grid window buffer in meters (default: station buffer)")
    p_pipe.add_argument("--obs-stations-dir", type=Path, help="Override setup obs stations directory")
    p_pipe.add_argument("--overlap-area-tol-m2", type=float, default=100.0, help="Allowed overlap area in m^2 (default: 100)")
    p_pipe.add_argument("--sliver-fix-m", type=float, default=0.0, help="Optional shrink-expand tolerance fix in meters")
    p_pipe.add_argument("--max-workers", type=int, help="Parallel sub-domain workers")
    p_pipe.add_argument("--inner-max-workers", type=int, help="Parallel member workers per sub-domain")
    p_pipe.add_argument("--retries", type=int, default=1, help="Retries per sub-domain on failure (default: 1)")
    p_pipe.add_argument("--coverage-sliver-tol-px", type=int, default=4, help="Allowed uncovered expected pixels (default: 4)")
    p_pipe.add_argument("--var", default="snow_depth", help="Model variable column to plot")
    p_pipe.add_argument("--obs-col", default="snow_depth", help="Observation column name")
    p_pipe.add_argument("--obs-scale", type=float, default=1.0, help="Observation scaling factor")
    p_pipe.add_argument("--stations", nargs="+", help="Optional list of station ids")
    p_pipe.add_argument("--no-merge", action="store_true", help="Skip merge stage")
    p_pipe.add_argument("--no-plot", action="store_true", help="Skip plot stage")
    p_pipe.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    p_pipe.add_argument("--log-level", default="INFO")
    p_pipe.add_argument("--no-perf-monitor", action="store_true", help="Disable performance monitor")

    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logger(args.log_level)

    if args.command == "prepare":
        from openamundsen_da.subdomain.prepare import prepare_subdomains

        ensure_run_mode(args.project_dir, expected="subdomain", write_if_missing=True)
        regions_path = args.regions or _default_regions_path(Path(args.setup_dir))
        prepare_subdomains(
            setup_dir=args.setup_dir,
            project_dir=args.project_dir,
            regions_path=regions_path,
            subdomain_root=args.subdomain_root or _default_subdomain_root(args.project_dir),
            id_field=args.id_field,
            clip_mode=args.clip_mode,
            station_buffer_m=float(args.station_buffer_km) * 1000.0,
            roi_buffer_m=float(args.roi_buffer_m),
            grid_buffer_m=args.grid_buffer_m,
            obs_stations_dir=args.obs_stations_dir,
            overlap_area_tol_m2=float(args.overlap_area_tol_m2),
            sliver_fix_m=float(args.sliver_fix_m),
            overwrite=bool(args.overwrite),
        )
        return 0

    if args.command == "run":
        from openamundsen_da.subdomain.run import run_subdomains

        manifest = _resolve_manifest(
            manifest_arg=args.manifest,
            project_dir=args.project_dir,
            subdomain_root=args.subdomain_root,
        )
        run_subdomains(
            manifest_path=manifest,
            subdomains=args.subdomains,
            max_workers=args.max_workers,
            inner_max_workers=args.inner_max_workers,
            retries=max(0, int(args.retries)),
            overwrite=bool(args.overwrite),
            log_level=args.log_level,
            perf_monitor=not args.no_perf_monitor,
        )
        return 0

    if args.command == "merge":
        from openamundsen_da.subdomain.merge import merge_grids
        from openamundsen_da.subdomain.manifest import SubdomainManifest

        manifest = _resolve_manifest(
            manifest_arg=args.manifest,
            project_dir=args.project_dir,
            subdomain_root=args.subdomain_root,
        )
        results_root = args.out_dir or (SubdomainManifest.load(manifest).project_dir / "results")
        merge_grids(
            manifest_path=manifest,
            subdomains=args.subdomains,
            out_dir=results_root / "grids",
            coverage_sliver_tol_px=int(args.coverage_sliver_tol_px),
        )
        return 0

    if args.command == "plot":
        from openamundsen_da.subdomain.plot import plot_station_comparisons

        manifest = _resolve_manifest(
            manifest_arg=args.manifest,
            project_dir=args.project_dir,
            subdomain_root=args.subdomain_root,
        )
        plot_station_comparisons(
            manifest_path=manifest,
            points_dir=args.points_dir,
            obs_dir=args.obs_dir,
            variable=args.var,
            obs_column=args.obs_col,
            obs_scale=float(args.obs_scale),
            station_ids=args.stations,
        )
        return 0

    if args.command == "pipeline":
        from openamundsen_da.subdomain.pipeline import run_pipeline

        ensure_run_mode(args.project_dir, expected="subdomain", write_if_missing=True)
        regions_path = args.regions or _default_regions_path(Path(args.setup_dir))
        run_pipeline(
            setup_dir=args.setup_dir,
            project_dir=args.project_dir,
            regions_path=regions_path,
            subdomain_root=args.subdomain_root or _default_subdomain_root(args.project_dir),
            id_field=args.id_field,
            clip_mode=args.clip_mode,
            station_buffer_m=float(args.station_buffer_km) * 1000.0,
            roi_buffer_m=float(args.roi_buffer_m),
            grid_buffer_m=args.grid_buffer_m,
            obs_stations_dir=args.obs_stations_dir,
            overlap_area_tol_m2=float(args.overlap_area_tol_m2),
            sliver_fix_m=float(args.sliver_fix_m),
            max_workers=args.max_workers,
            inner_max_workers=args.inner_max_workers,
            retries=max(0, int(args.retries)),
            coverage_sliver_tol_px=int(args.coverage_sliver_tol_px),
            plot_var=args.var,
            plot_obs_col=args.obs_col,
            plot_obs_scale=float(args.obs_scale),
            plot_stations=args.stations,
            skip_merge=bool(args.no_merge),
            skip_plot=bool(args.no_plot),
            overwrite=bool(args.overwrite),
            log_level=args.log_level,
            perf_monitor=not args.no_perf_monitor,
        )
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())

