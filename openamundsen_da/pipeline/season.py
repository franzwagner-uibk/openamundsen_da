"""openamundsen_da.pipeline.season

End-to-end season orchestrator with strict, opinionated behavior:

- Discovers step_* under a season directory (preferring season_dir/steps) and processes them in order.
- Step 00: cold start (no restart), dumps states at the end.
- Steps >= 01: strict warm start from member-root pointer; aborts on failure.
- For each step except the last:
  - Assimilate SCF on the next step start_date.
  - Resample to posterior using project.yml resampling defaults.
  - Rejuvenate posterior -> next-step prior (writes only member-root pointers).
- At the end: generates season plots (forcing + fraction overlay).

Minimal CLI; defaults handle all formats/columns/behavior without user choices.
"""

from __future__ import annotations

import os
import shutil
import threading
import sys
import concurrent.futures as cf
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.core.launch import launch_members
from openamundsen_da.core.prior_forcing import build_prior_ensemble
from openamundsen_da.io.paths import (
    read_step_config,
    find_project_yaml,
    find_season_yaml,
    list_steps_sorted,
)
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.landcover_mask import (
    resolve_landcover_mask,
    summarize_landcover_mask,
    write_landcover_mask_report,
)
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.da_events import load_assimilation_events, AssimilationEvent
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.util.ts import parse_datetime_opt
from openamundsen_da.methods.pf.assimilate_scf import (
    assimilate_scf_for_date,
    assimilate_wet_snow_for_date,
)
from openamundsen_da.pipeline.cleanup import cleanup_season_dir, is_cleanup_enabled, state_patterns_from_project
from openamundsen_da.methods.h_of_x.model_scf import compute_step_scf_daily_for_all_members
from openamundsen_da.methods.wet_snow.classify import classify_step_wet_snow
from openamundsen_da.methods.wet_snow.area import compute_step_wet_snow_daily_for_all_members
from openamundsen_da.methods.pf.rejuvenate import rejuvenate
from openamundsen_da.methods.pf.resample import resample_from_weights, _read_resampling_from_project
from openamundsen_da.methods.pf.plot_weights import plot_weights_for_csv
from openamundsen_da.methods.pf.plot_ess_timeline import plot_season_ess_timeline
from openamundsen_da.methods.viz.aggregate_fractions import aggregate_fraction_envelope
from openamundsen_da.observer.plot_fractions import cli_main as plot_fractions_cli
from openamundsen_da.methods.viz.plot_season_ensemble import plot_season_results
from openamundsen_da.methods.viz.plot_forcing_ensemble import cli_main as plot_forcing_cli
from openamundsen_da.util.da_events import AssimilationEvent

# Map assimilation variables to the diagnostics/plots we should run.
# Extend this mapping when new observables are added.
DA_DIAGNOSTICS = {
    "scf": {
        "model_daily": True,
        "plots": True,
    },
    "wet_snow": {
        "wet_classify": True,
        "wet_daily": True,
        "wet_plots": True,
    },
}


@dataclass(frozen=True)
class PlotTask:
    """Picklable plot task for process-based fan-out."""
    name: str
    func: object
    args: tuple
    kwargs: dict


def _run_plot_task(task: PlotTask) -> tuple[str, str | None]:
    """Execute a PlotTask in a worker process and return (name, error)."""
    try:
        task.func(*task.args, **task.kwargs)
        return task.name, None
    except Exception as exc:  # pragma: no cover
        return task.name, str(exc)


def _validate_assimilation_prereqs(
    project_dir: Path,
    season_dir: Path,
    steps: list[Path],
    events: list[AssimilationEvent],
) -> None:
    """Ensure required outputs/config/obs exist before running a season."""
    proj_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    grid_vars = ((proj_cfg.get("output_data") or {}).get("grids") or {}).get("variables") or []
    names: set[str] = set()
    vars_: set[str] = set()
    for entry in grid_vars:
        if not isinstance(entry, dict):
            continue
        if entry.get("name"):
            names.add(str(entry["name"]))
        if entry.get("var"):
            vars_.add(str(entry["var"]))

    errors: list[str] = []

    needs_scf = any(ev.variable == "scf" for ev in events)
    if needs_scf and not ({"snowdepth_daily"} & names or {"snow.depth"} & vars_):
        errors.append("Configure snow depth daily output (var: snow.depth, name: snowdepth_daily) in output_data.grids for SCF assimilation.")

    needs_wet = any(ev.variable == "wet_snow" for ev in events)
    if needs_wet and not ({"liquid_water_content"} & names or {"snow.liquid_water_content"} & vars_):
        errors.append("Configure liquid water content output (var: snow.liquid_water_content, name: liquid_water_content) in output_data.grids for wet-snow assimilation.")

    max_idx = min(len(events), len(steps) - 1)
    for idx in range(max_idx):
        ev = events[idx]
        step_dir = Path(steps[idx])
        obs_name = f"obs_{ev.variable}_{ev.product}_{ev.date.strftime('%Y%m%d')}.csv"
        obs_path = step_dir / "obs" / obs_name
        if not obs_path.is_file():
            errors.append(f"Missing obs CSV for {ev.variable} ({ev.product}) on {ev.date}: expected {obs_path}")

    if errors:
        raise ValueError("Config/obs validation failed:\n- " + "\n- ".join(errors))


def _list_steps_sorted(season_dir: Path) -> List[Path]:
    return list_steps_sorted(season_dir)


def _next_step_start(steps: List[Path], idx: int) -> Optional[datetime]:
    if idx + 1 >= len(steps):
        return None
    cfg = read_step_config(steps[idx + 1]) or {}
    val = cfg.get("start_date")
    try:
        return datetime.fromisoformat(str(val)) if val else None
    except Exception:
        return None


def _find_roi(project_dir: Path) -> Path:
    """Return the conventional ROI path env/roi.gpkg if present."""
    env_dir = Path(project_dir) / "env"
    roi = env_dir / "roi.gpkg"
    if roi.is_file():
        return roi
    cands = list(env_dir.glob("*.gpkg")) + list(env_dir.glob("*.shp"))
    if not cands:
        raise FileNotFoundError(f"No ROI vector found under {env_dir}")
    return sorted(cands)[0]


def _load_wet_snow_threshold_percent(project_dir: Path) -> float:
    """Read wet-snow classification threshold (percent) from project.yml."""
    try:
        proj_yaml = find_project_yaml(project_dir)
        cfg = _read_yaml_file(proj_yaml) or {}
        da_cfg = cfg.get("data_assimilation") or {}
        wet_cfg = da_cfg.get("wet_snow") or {}
        if "classification_threshold_percent" in wet_cfg:
            return float(wet_cfg["classification_threshold_percent"])
        if "classification_threshold" in wet_cfg:
            return float(wet_cfg["classification_threshold"])
    except Exception:
        pass
    return 0.1


def _aggregate_and_copy_fraction(
    season_dir: Path,
    filename: str,
    value_col: str,
    output_name: str,
) -> tuple[Path | None, Path | None]:
    """Aggregate fraction envelopes and mirror them into plots/results."""
    env_path = aggregate_fraction_envelope(
        season_dir=season_dir,
        filename=filename,
        value_col=value_col,
        output_name=output_name,
    )
    copy_path: Path | None = None
    if env_path is not None:
        try:
            copy_path = Path(season_dir) / "plots" / "results" / Path(output_name).name
            copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(env_path, copy_path)
        except Exception as exc:
            logger.warning("Failed to copy {} -> {}: {}", env_path, copy_path, exc)
            copy_path = None
    return env_path, copy_path


def _aggregate_fraction_envelopes(season_dir: Path) -> None:
    """Aggregate SCF and wet-snow envelopes and mirror them into plots/results."""
    try:
        _aggregate_and_copy_fraction(
            season_dir=season_dir,
            filename="point_scf_roi.csv",
            value_col="scf",
            output_name="point_scf_roi_envelope.csv",
        )
    except Exception as exc:
        logger.warning("SCF envelope aggregation failed: {}", exc)
    try:
        _aggregate_and_copy_fraction(
            season_dir=season_dir,
            filename="point_wet_snow_roi.csv",
            value_col="wet_snow_fraction",
            output_name="point_wet_snow_roi_envelope.csv",
        )
    except Exception as exc:
        logger.warning("Wet-snow envelope aggregation failed: {}", exc)


def _run_plot_tasks_parallel(
    tasks: List[PlotTask],
    max_workers: int | None,
    season_max_workers: int | None,
) -> None:
    """Execute plot tasks concurrently using process-based workers."""
    if not tasks:
        return
    cpu_cap = os.cpu_count() or len(tasks)
    candidates = [len(tasks), cpu_cap]
    if max_workers is not None:
        candidates.append(max_workers)
    if season_max_workers is not None:
        candidates.append(season_max_workers)
    workers = max(1, min(candidates))
    logger.info("Running {} plot task(s) with {} worker(s) ...", len(tasks), workers)
    with cf.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_plot_task, t): t for t in tasks}
        for fut in cf.as_completed(futures):
            task = futures[fut]
            try:
                name, err = fut.result()
            except Exception as exc:  # pragma: no cover
                logger.warning("Plot task {} crashed: {}", task.name, exc)
                continue
            if err:
                logger.warning("Plot task {} failed: {}", name, err)
            else:
                logger.info("Plot task {} completed", name)


def _run_live_plots(
    cfg: OrchestratorConfig,
    step_dir: Path,
    step_name: str,
    wcsv: Path,
    *,
    reset_logger: bool = True,
    wet_snow_enabled: bool = True,
) -> None:
    """Run per-step plotting suite."""
    try:
        logger.info("Updating season plots after assimilation step {} ...", step_name)
        _aggregate_fraction_envelopes(cfg.season_dir)
        try:
            plot_forcing_cli([
                "--step-dir", str(step_dir),
                "--ensemble", "prior",
                "--log-level", cfg.log_level,
            ], configure_logger=False)
        except Exception as exc:
            logger.warning("Forcing plot failed for {}: {}", step_name, exc)
        try:
            plot_season_results(
                season_dir=cfg.season_dir,
                var_col="swe",
                mode="members",
                resample="D",
                resample_agg="mean",
                configure_logger=False,
            )
            plot_season_results(
                season_dir=cfg.season_dir,
                var_col="snow_depth",
                mode="members",
                resample="D",
                resample_agg="mean",
                configure_logger=False,
            )
        except Exception as exc:
            logger.warning("Season point results plot failed after step {}: {}", step_name, exc)
        try:
            plot_fractions_cli([
                "--season-dir", str(cfg.season_dir),
                "--project-dir", str(cfg.project_dir),
                "--log-level", cfg.log_level,
                "--mode", "band",
            ], configure_logger=False)
        except Exception as exc:
            logger.warning("Fraction overlay plot skipped after step {}: {}", step_name, exc)
        plot_weights_for_csv(wcsv)
        try:
            plot_season_ess_timeline(cfg.season_dir)
        except FileNotFoundError:
            pass
    except Exception as exc:
        logger.warning("Season plotting failed after step {}: {}", step_name, exc)
    finally:
        if reset_logger:
            _setup_logger(cfg.season_dir, cfg.log_level)


@dataclass
class OrchestratorConfig:
    project_dir: Path
    season_dir: Path
    max_workers: int = 4
    overwrite: bool = False
    log_level: str = "INFO"
    live_plots: bool = False
    plot_workers: int | None = None
    monitor_perf: bool = False
    perf_sample_interval: float = 5.0
    perf_plot_interval: float = 30.0


def _setup_logger(season_dir: Path, log_level: str) -> None:
    """Configure Loguru sinks for console and season file log."""
    logger.remove()
    logger.add(sys.stdout, level=log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)
    log_file = Path(season_dir) / f"{Path(season_dir).name}.log"
    logger.add(log_file, level=log_level.upper(), colorize=False, enqueue=True, format=LOGURU_FORMAT)


def run_season(cfg: OrchestratorConfig) -> None:
    run_start = datetime.utcnow()
    # Console + file log under season root (e.g. season_2017-2018/season_2017-2018.log)
    _setup_logger(cfg.season_dir, cfg.log_level)
    live_plot_threads: list[threading.Thread] = []

    steps = _list_steps_sorted(cfg.season_dir)
    if not steps:
        raise FileNotFoundError(f"No steps found under {cfg.season_dir}")
    logger.info("Discovered {} step(s)", len(steps))
    workers = pick_max_workers(cfg.max_workers)

    # Ensure first step has its prior ensemble (project/meteo is required)
    if steps:
        meteo_dir = cfg.project_dir / "meteo"
        if not meteo_dir.is_dir():
            raise FileNotFoundError(f"Required meteo directory not found: {meteo_dir}")
        logger.info("Initializing prior ensemble for step {} …", steps[0].name)

    # Validate required outputs and obs inputs before running assimilation
    _validate_assimilation_prereqs(cfg.project_dir, cfg.season_dir, steps, events)
    build_prior_ensemble(
        input_meteo_dir=meteo_dir,
        project_dir=cfg.project_dir,
        step_dir=steps[0],
        max_workers=int(workers),
        overwrite=bool(cfg.overwrite),
    )

    roi = _find_roi(cfg.project_dir)
    logger.info("Using ROI: {}", roi)
    lc_cfg = resolve_landcover_mask(cfg.project_dir)
    if lc_cfg.enabled:
        logger.info("Land-cover mask enabled -> {} (classes {})", lc_cfg.path, list(lc_cfg.classes))
    else:
        logger.info("Land-cover mask disabled; no land-cover exclusions applied")

    # Project/season metadata for DA and performance monitoring
    wet_snow_threshold = _load_wet_snow_threshold_percent(cfg.project_dir)
    logger.info("Wet-snow classification threshold set to {:.3f} % (project.yml or default)", wet_snow_threshold)

    proj_resolution = None
    proj_timestep = None
    proj_crs = None
    season_days = None
    ensemble_size = None
    try:
        proj_yaml = find_project_yaml(cfg.project_dir)
        proj_cfg = _read_yaml_file(proj_yaml) or {}
        if "resolution" in proj_cfg:
            try:
                proj_resolution = float(proj_cfg.get("resolution"))
            except Exception:
                proj_resolution = None
        if "timestep" in proj_cfg:
            proj_timestep = str(proj_cfg.get("timestep"))
        proj_crs = proj_cfg.get("crs")
        da_cfg = proj_cfg.get("data_assimilation") or {}
        pf_cfg = da_cfg.get("prior_forcing") or {}
        if "ensemble_size" in pf_cfg:
            try:
                ensemble_size = int(pf_cfg.get("ensemble_size"))
            except Exception:
                ensemble_size = None
    except Exception as exc:
        logger.warning("Perf monitor: failed to read project.yml metadata: {}", exc)

    # Season length (days) from season.yml
    try:
        seas_yaml = find_season_yaml(cfg.season_dir)
        seas_cfg = _read_yaml_file(seas_yaml) or {}
        start_val = seas_cfg.get("start_date")
        end_val = seas_cfg.get("end_date")
        start_dt = parse_datetime_opt(str(start_val)) if start_val is not None else None
        end_dt = parse_datetime_opt(str(end_val)) if end_val is not None else None
        if start_dt is not None and end_dt is not None:
            season_days = (end_dt.date() - start_dt.date()).days + 1
    except Exception as exc:
        logger.warning("Perf monitor: failed to read season.yml dates: {}", exc)

    # Assimilation configuration (variable/product per date)
    events = load_assimilation_events(cfg.season_dir)
    n_expected = max(0, len(steps) - 1)
    if len(events) < n_expected:
        raise ValueError(
            f"Configured {len(events)} assimilation event(s) but the season needs {n_expected}. "
            "Add events in season.yml (data_assimilation.assimilation_events) or adjust steps."
        )
    if len(events) > n_expected:
        logger.warning("More assimilation events ({}) than steps needing DA ({}); extra events will be ignored.", len(events), n_expected)
    vars_used = {getattr(ev, "variable", None) for ev in events if getattr(ev, "variable", None)}
    scf_enabled = "scf" in vars_used
    wet_snow_enabled = "wet_snow" in vars_used
    if not vars_used:
        logger.info("No assimilation events found; skipping SCF/wet-snow diagnostics (explicit variables only).")
        scf_enabled = False
        wet_snow_enabled = False
    else:
        logger.info("Assimilation variables detected: {}", ", ".join(sorted(vars_used)))
        if wet_snow_enabled:
            logger.info("Wet-snow diagnostics enabled (wet_snow present in assimilation_events).")
        else:
            logger.info("Wet-snow diagnostics disabled (wet_snow not in assimilation_events).")
        if scf_enabled:
            logger.info("SCF diagnostics enabled (scf present in assimilation_events).")
        else:
            logger.info("SCF diagnostics disabled (scf not in assimilation_events).")

    # Approximate AOI area in km2 for performance summary
    roi_area_km2 = None
    try:
        gdf, _ = read_single_roi(Path(roi), required_field=None, to_crs=proj_crs if proj_crs is not None else None)
        roi_area_km2 = float(gdf.geometry.area.iloc[0]) / 1_000_000.0
    except Exception as exc:
        logger.warning("Perf monitor: failed to compute AOI area: {}", exc)

    # Report land-cover mask coverage within the ROI
    if lc_cfg.enabled:
        try:
            lc_report = summarize_landcover_mask(Path(roi), lc_cfg)
            lc_report_path = Path(cfg.season_dir) / "plots" / "results" / "lc_mask_report.csv"
            write_landcover_mask_report(lc_report, lc_report_path)
            for cls in lc_report.classes:
                label = cls.name
                if cls.code != cls.name:
                    label = f"{cls.name} ({cls.code})"
                if cls.percent_of_roi is None:
                    logger.info("LC mask class {}: {} cell(s), {:.3f} km^2", label, cls.cells, cls.area_km2)
                else:
                    logger.info(
                        "LC mask class {}: {} cell(s), {:.3f} km^2 ({:.2f}% of ROI)",
                        label,
                        cls.cells,
                        cls.area_km2,
                        cls.percent_of_roi,
                    )
            if lc_report.roi_area_km2 is not None and lc_report.roi_area_km2 > 0:
                masked_pct = (lc_report.masked_area_km2 / lc_report.roi_area_km2) * 100.0
                logger.info(
                    "LC mask total: masked {} cell(s), {:.3f} km^2 ({:.2f}% of ROI); report -> {}",
                    lc_report.masked_cells,
                    lc_report.masked_area_km2,
                    masked_pct,
                    lc_report_path,
                )
            else:
                logger.info(
                    "LC mask total: masked {} cell(s), {:.3f} km^2 (ROI area unknown); report -> {}",
                    lc_report.masked_cells,
                    lc_report.masked_area_km2,
                    lc_report_path,
                )
            if roi_area_km2 is None and lc_report.roi_area_km2 is not None:
                roi_area_km2 = lc_report.roi_area_km2
        except Exception as exc:
            logger.warning("Land-cover mask report failed: {}", exc)

    perf_stop_event = None
    if cfg.monitor_perf:
        pm_cfg = PerfMonitorConfig(
            season_dir=cfg.season_dir,
            sample_interval_sec=float(cfg.perf_sample_interval or 5.0),
            plot_interval_sec=float(cfg.perf_plot_interval or 30.0),
            run_start=run_start,
        )
        perf_stop_event = start_perf_monitor(pm_cfg)

    # Process each step
    cleanup_enabled = is_cleanup_enabled(cfg.project_dir)
    member_failures = False
    for i, step_dir in enumerate(steps):
        step_name = Path(step_dir).name
        logger.info("== Step {} ==", step_name)

        # Launch ensemble (runner enforces strict cold/warm semantics by step)
        logger.info("Launching ensemble (prior) with max_workers={} overwrite={} ...", workers, cfg.overwrite)
        launch_summary = launch_members(
            project_dir=cfg.project_dir,
            season_dir=cfg.season_dir,
            step_dir=step_dir,
            ensemble="prior",
            max_workers=int(workers),
            overwrite=bool(cfg.overwrite),
            results_root=None,
            log_level=cfg.log_level,
            state_pattern=None,
        )
        if launch_summary.get("summary", {}).get("failed", 0) > 0:
            member_failures = True

        # After propagation: compute daily model SCF for all prior members in
        # this step so that season-level plots can use var_col='scf' via the
        # generated point_scf_roi.csv files.
        try:
            if scf_enabled:
                compute_step_scf_daily_for_all_members(
                    project_dir=cfg.project_dir,
                    step_dir=step_dir,
                    aoi_path=roi,
                    landcover_cfg=lc_cfg,
                    max_workers=int(workers),
                    overwrite=bool(cfg.overwrite),
                )
        except Exception as exc:
            logger.warning("Model SCF daily computation failed for {}: {}", step_name, exc)

        # After propagation: also compute model wet-snow diagnostics (masks +
        # daily AOI fractions) for all prior members in this step so that
        # wet-snow plots are always available regardless of which observable
        # is assimilated.
        try:
            if wet_snow_enabled:
                classify_step_wet_snow(
                    step_dir=step_dir,
                    members=None,
                    threshold_percent=wet_snow_threshold,
                    output_subdir="wet_snow",
                    mask_prefix="wet_snow_mask",
                    fraction_prefix="lwc_fraction",
                    write_fraction=False,
                    overwrite=bool(cfg.overwrite),
                    max_workers=int(workers),
                )
                compute_step_wet_snow_daily_for_all_members(
                    project_dir=cfg.project_dir,
                    step_dir=step_dir,
                    aoi_path=roi,
                    landcover_cfg=lc_cfg,
                    max_workers=int(workers),
                    overwrite=bool(cfg.overwrite),
                    mask_subdir="wet_snow",
                    mask_prefix="wet_snow_mask",
                )
        except Exception as exc:
            logger.warning("Model wet-snow diagnostics failed for {}: {}", step_name, exc)

        # If not the last step: Assimilation -> Resample -> Rejuvenate
        next_start = _next_step_start(steps, i)
        if next_start is None:
            logger.info("Final step reached; skipping assimilation/resample/rejuvenate.")
            continue

        # Quick warm-start boundary check (best effort)
        try:
            curr_cfg = read_step_config(step_dir) or {}
            end_val = curr_cfg.get("end_date")
            if end_val is not None and next_start is not None:
                curr_end = datetime.fromisoformat(str(end_val))
                gap = (next_start - curr_end).total_seconds()
                if gap <= 0:
                    logger.warning(
                        "Next step start ({}) is not after current step end ({}). Warm start expects start = end + one model timestep.",
                        next_start,
                        curr_end,
                    )
                else:
                    logger.info("Step boundary gap: {} seconds. Ensure it equals exactly one model timestep.", int(gap))
        except Exception:
            # Best-effort; do not fail if step YAMLs are incomplete or unparsable
            pass

        # Assimilation date: map step i -> event i (skip last step)
        assim_dt = None
        ev: AssimilationEvent | None = None
        try:
            curr_cfg = read_step_config(step_dir) or {}
            start_val = curr_cfg.get("start_date")
            end_val = curr_cfg.get("end_date")
            start_dt = datetime.fromisoformat(str(start_val)) if start_val is not None else None
            end_dt = datetime.fromisoformat(str(end_val)) if end_val is not None else None
        except Exception:
            start_dt = None
            end_dt = None

        if i < len(events):
            ev = events[i]
            assim_dt = datetime.combine(ev.date, (start_dt or datetime.min).time())
            if start_dt is not None and end_dt is not None:
                if not (start_dt.date() <= ev.date <= end_dt.date()):
                    logger.warning(
                        "Configured DA date {} is outside step {} window ({} .. {})",
                        ev.date,
                        step_name,
                        start_dt.date(),
                        end_dt.date(),
                    )
        else:
            assim_dt = next_start

        if ev is None:
            logger.warning(
                "No assimilation event configured for {} -> skipping assimilation for {}",
                assim_dt.date(),
                step_name,
            )
            continue

        logger.info(
            "Assimilating {} (product {}) for date {}",
            ev.variable,
            ev.product,
            assim_dt.strftime("%Y-%m-%d"),
        )

        # Reuse existing weights if present and overwrite=False so that
        # re-running oa-da-season can skip already-assimilated steps.
        assim_dir = Path(step_dir) / "assim"
        assim_dir.mkdir(parents=True, exist_ok=True)
        if ev.variable == "wet_snow":
            weights_name = f"weights_wet_snow_{assim_dt.strftime('%Y%m%d')}.csv"
        else:
            weights_name = f"weights_scf_{assim_dt.strftime('%Y%m%d')}.csv"
        wcsv = assim_dir / weights_name
        if wcsv.is_file() and not cfg.overwrite:
            logger.info(
                "Weights CSV already exists for {}; overwrite=False -> reusing existing weights: {}",
                step_name,
                wcsv,
            )
            # Downstream resampling/rejuvenation will read this file; no need
            # to recompute or touch assimilation for this step.
        else:
            try:
                if ev.variable == "wet_snow":
                    weights = assimilate_wet_snow_for_date(
                        project_dir=cfg.project_dir,
                        step_dir=step_dir,
                        ensemble="prior",
                        date=assim_dt,
                        aoi=roi,
                        landcover_cfg=lc_cfg,
                        obs_csv=None,
                    )
                else:
                    weights = assimilate_scf_for_date(
                        project_dir=cfg.project_dir,
                        step_dir=step_dir,
                        ensemble="prior",
                        date=assim_dt,
                        aoi=roi,
                        landcover_cfg=lc_cfg,
                        obs_csv=None,
                        product=ev.product,
                    )
            except FileNotFoundError as exc:
                logger.error(
                    "Assimilation failed for step {} at date {}: {}. "
                    "Ensure the appropriate obs CSV exists under {}/obs for this date "
                    "or generate it via the corresponding observer CLI.",
                    step_name,
                    assim_dt.strftime("%Y-%m-%d"),
                    exc,
                    step_dir,
                )
                raise
            weights.to_csv(wcsv, index=False)
            logger.info("Wrote weights -> {}", wcsv)

        # Resample to posterior
        posterior_root = Path(step_dir) / "ensembles" / "posterior"
        has_posterior = posterior_root.is_dir() and any(posterior_root.glob("member_*"))
        if has_posterior and not cfg.overwrite:
            logger.info("Posterior ensemble already exists and overwrite=False; skipping resampling.")
        else:
            rs_cfg = _read_resampling_from_project(cfg.project_dir)
            algo = rs_cfg.algorithm or "systematic"
            ess_thr_abs = float(rs_cfg.ess_threshold or 0.0)
            ess_thr_ratio = rs_cfg.ess_threshold_ratio
            ratio_text = f"{ess_thr_ratio:.3f}" if ess_thr_ratio is not None else "None"
            logger.info(
                "Resampling to posterior ... (algorithm={} seed={} ess_thr_abs={} ess_thr_ratio={})",
                algo,
                rs_cfg.seed if rs_cfg.seed is not None else "auto",
                ess_thr_abs,
                ratio_text,
            )
            resample_from_weights(
                step_dir=step_dir,
                source_ensemble="prior",
                weights_csv=wcsv,
                target_ensemble="posterior",
                seed=rs_cfg.seed,
                algorithm=algo,
                ess_threshold=ess_thr_abs,
                ess_threshold_ratio=ess_thr_ratio,
                overwrite=bool(cfg.overwrite),
            )

        # Rejuvenate posterior -> next prior
        rejuvenate_manifest = Path(steps[i + 1]) / "assim" / "rejuvenate_manifest.json"
        if rejuvenate_manifest.is_file() and not cfg.overwrite:
            logger.info("Rejuvenation manifest already exists for {}; overwrite=False -> skipping rejuvenation.", steps[i + 1].name)
        else:
            logger.info("Rejuvenating posterior -> {} (prior) ...", steps[i + 1].name)
            rejuvenate(
                project_dir=cfg.project_dir,
                prev_step_dir=step_dir,
                next_step_dir=steps[i + 1],
                source_ensemble="posterior",
                target_ensemble="prior",
                source_meteo_dir=None,
            )

        # Update season-wide plots after each assimilation/rejuvenation cycle so
        # users can monitor progress while the pipeline continues running. Plots
        # are written with deterministic filenames and therefore overwritten on
        # each update.
        if cfg.live_plots:
            logger.info("Dispatching live plots in background for {} ...", step_name)
            t = threading.Thread(
                target=_run_live_plots,
                args=(cfg, step_dir, step_name, wcsv),
                kwargs={"reset_logger": False, "wet_snow_enabled": wet_snow_enabled},
                daemon=False,
            )
            t.start()
            live_plot_threads.append(t)

    # Final assimilation-level plots (weights per step + ESS timeline),
    # regardless of live_plots. Best-effort: failures do not abort.
    for t in live_plot_threads:
        try:
            t.join()
        except Exception:
            pass
    # Aggregate fraction envelopes before plotting overlays
    _aggregate_fraction_envelopes(cfg.season_dir)

    # Build post-run plot tasks (per-step forcing, season results, fraction overlay, weights, ESS)
    plot_tasks: List[PlotTask] = []
    for step_dir in steps:
        plot_tasks.append(
            PlotTask(
                name=f"forcing:{Path(step_dir).name}",
                func=plot_forcing_cli,
                args=(
                    [
                        "--step-dir",
                        str(step_dir),
                        "--ensemble",
                        "prior",
                        "--log-level",
                        cfg.log_level,
                    ],
                ),
                kwargs={"configure_logger": False},
            )
        )
    plot_tasks.append(
        PlotTask(
            name="season_results_swe",
            func=plot_season_results,
            args=(),
            kwargs={
                "season_dir": cfg.season_dir,
                "var_col": "swe",
                "mode": "members",
                "resample": "D",
                "resample_agg": "mean",
                "configure_logger": False,
            },
        )
    )
    plot_tasks.append(
        PlotTask(
            name="season_results_snow_depth",
            func=plot_season_results,
            args=(),
            kwargs={
                "season_dir": cfg.season_dir,
                "var_col": "snow_depth",
                "mode": "members",
                "resample": "D",
                "resample_agg": "mean",
                "configure_logger": False,
            },
        )
    )
    plot_tasks.append(
        PlotTask(
            name="fraction_overlay",
            func=plot_fractions_cli,
            args=(
                [
                    "--season-dir",
                    str(cfg.season_dir),
                    "--project-dir",
                    str(cfg.project_dir),
                ],
            ),
            kwargs={"configure_logger": False},
        )
    )
    weights_csvs: List[Path] = []
    for step_dir in steps:
        assim_dir = Path(step_dir) / "assim"
        if not assim_dir.is_dir():
            continue
        candidates = sorted(assim_dir.glob("weights_*_*.csv"))
        if candidates:
            weights_csvs.append(candidates[-1])
    for wcsv in weights_csvs:
        plot_tasks.append(
            PlotTask(
                name=f"weights:{wcsv.parent.parent.name}",
                func=plot_weights_for_csv,
                args=(wcsv,),
                kwargs={},
            )
        )
    plot_tasks.append(
        PlotTask(
            name="season_ess_timeline",
            func=plot_season_ess_timeline,
            args=(cfg.season_dir,),
            kwargs={},
        )
    )
    try:
        _run_plot_tasks_parallel(plot_tasks, cfg.plot_workers, cfg.max_workers)
    except Exception as exc:
        logger.warning("Post-run plotting failed: {}", exc)

    # Cleanup state files if configured and no member failures occurred
    try:
        if cleanup_enabled:
            if member_failures:
                logger.info("Skipping season cleanup because some member runs failed.")
            else:
                patterns = state_patterns_from_project(cfg.project_dir)
                summary = cleanup_season_dir(project_dir=cfg.project_dir, season_dir=cfg.season_dir, patterns=patterns)
                patt = ",".join(summary.patterns)
                if summary.attempted == 0:
                    logger.info("Season cleanup: no matching state files found (patterns={})", patt)
                elif summary.failures:
                    logger.warning(
                        "Season cleanup completed with {} failure(s): deleted {}/{} file(s), freed {:.1f} MB (patterns={})",
                        summary.failures,
                        summary.files_deleted,
                        summary.attempted,
                        summary.bytes_freed / 1_000_000.0,
                        patt,
                    )
                else:
                    logger.info(
                        "Season cleanup succeeded: deleted {}/{} file(s), freed {:.1f} MB (patterns={})",
                        summary.files_deleted,
                        summary.attempted,
                        summary.bytes_freed / 1_000_000.0,
                        patt,
                    )
        else:
            logger.info("Season cleanup disabled via project.yml (data_assimilation.restart.cleanup_after_season=false).")
    except Exception as exc:
        logger.warning("Season cleanup failed: {}", exc)

    _setup_logger(cfg.season_dir, cfg.log_level)
    run_end = datetime.utcnow()
    duration = (run_end - run_start).total_seconds()
    logger.info("Season processing complete: {} (wall-clock {:.1f} s, ~{:.2f} h)", cfg.season_dir, duration, duration / 3600.0)

    if perf_stop_event is not None:
        perf_stop_event.set()


def cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-season", description="Process a full season: run steps, assimilate, resample, rejuvenate, plot.")
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--season-dir", required=True, type=Path)
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max workers (overrides MAX_WORKERS env). Defaults to min(CPU, #members).",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--live-plots",
        dest="live_plots",
        action="store_true",
        help="Enable plotting during the season; default is off (plots run after completion).",
    )
    p.add_argument(
        "--no-live-plots",
        dest="live_plots",
        action="store_false",
        help="Skip plotting during the season (default).",
    )
    p.add_argument(
        "--monitor-perf",
        action="store_true",
        help="Enable background performance monitor (CPU/RAM/disk) during the season run.",
    )
    p.add_argument(
        "--perf-sample-interval",
        type=float,
        default=5.0,
        help="Performance monitor sampling interval in seconds (default: 5).",
    )
    p.add_argument(
        "--perf-plot-interval",
        type=float,
        default=30.0,
        help="Performance monitor plot refresh interval in seconds (default: 30).",
    )
    p.add_argument(
        "--plot-workers",
        type=int,
        default=None,
        help="Parallel plot workers for post-run plotting (default: min(CPU, tasks)).",
    )
    p.add_argument("--log-level", default="INFO")
    p.set_defaults(live_plots=False)
    args = p.parse_args(argv)

    resolved_workers = pick_max_workers(args.max_workers, fallback=4)

    run_season(
        OrchestratorConfig(
            project_dir=Path(args.project_dir),
            season_dir=Path(args.season_dir),
            max_workers=int(resolved_workers),
            overwrite=bool(args.overwrite),
            log_level=str(args.log_level or "INFO"),
            live_plots=bool(getattr(args, "live_plots", False)),
            plot_workers=(int(args.plot_workers) if args.plot_workers is not None else None),
            monitor_perf=bool(getattr(args, "monitor_perf", False)),
            perf_sample_interval=float(getattr(args, "perf_sample_interval", 5.0)),
            perf_plot_interval=float(getattr(args, "perf_plot_interval", 30.0)),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
