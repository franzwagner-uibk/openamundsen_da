"""openamundsen_da.pipeline.project

End-to-end project orchestrator with strict, opinionated behavior:

- Discovers step_* under a project directory and processes them in order.
- Step 00: cold start (no restart), dumps states at the end.
- Steps >= 01: strict warm start from member-root pointer; aborts on failure.
- For each step except the last:
  - Assimilate the configured observable on the next step start_date.
  - Resample to posterior using project YAML resampling defaults.
  - Rejuvenate posterior -> next-step prior (writes only member-root pointers).
- At the end: generates project plots (forcing + fraction overlay).

Minimal CLI; defaults handle all formats/columns/behavior without user choices.
"""

from __future__ import annotations

import threading
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from loguru import logger
import numpy as np
import pandas as pd

from openamundsen_da.configuration import load_project_configuration
from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.core.launch import launch_members
from openamundsen_da.core.prior_forcing import build_prior_ensemble
from openamundsen_da.benchmark.pipeline import run_project_benchmark
from openamundsen_da.io.paths import (
    read_step_config,
    find_setup_yaml,
    find_project_yaml,
    list_member_dirs,
    list_steps_sorted,
    project_da_output_grids_path,
    project_ensemble_points_path,
    project_ensemble_forcing_path,
    project_fraction_envelope_path,
    project_landcover_mask_report_path,
)
from openamundsen_da.util.roi import read_single_roi
from openamundsen_da.util.roi_grid import (
    ensure_setup_roi_grid,
    ensure_setup_roi_vector,
    resolve_setup_grid_spec,
)
from openamundsen_da.util.landcover_mask import (
    LandcoverMaskConfig,
    resolve_landcover_mask,
    summarize_landcover_mask,
    write_landcover_mask_report,
)
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.da_events import load_assimilation_events, AssimilationEvent
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, start_perf_monitor
from openamundsen_da.util.storage_budget import (
    StorageReservationProject,
    check_step_admission,
)
from openamundsen_da.util.storage_admission import (
    StorageAccountingSummary,
    StorageAdmissionClient,
    StorageAdmissionCoordinator,
    accounting_summary_from_inventory,
    accounting_summary_from_paths,
    build_storage_plan,
    reused_accounting_summary,
)
from openamundsen_da.util.point_output import (
    validate_project_ensemble_points,
    write_project_ensemble_points,
)
from openamundsen_da.util.forcing_output import (
    validate_project_ensemble_forcing,
    write_project_ensemble_forcing,
)
from openamundsen_da.methods.pf.assimilate_fraction import (
    assimilate_scf_for_date,
    assimilate_wet_snow_line_for_date,
    assimilate_wet_snow_for_date,
)
from openamundsen_da.methods.pf.assimilate_station import (
    assimilate_station_hs_for_date,
    assimilate_station_swe_for_date,
)
from openamundsen_da.methods.h_of_x.model_scf import compute_step_scf_daily_for_all_members
from openamundsen_da.methods.roi_mean import compute_step_roi_mean_daily_for_all_members
from openamundsen_da.io.model_grids import configured_model_grid_format
from openamundsen_da.methods.wet_snow.classify import (
    CLASSIFICATION_METHOD_AMOUNT,
    WetSnowClassificationConfig,
    classify_step_wet_snow,
    load_wet_snow_classification_config,
)
from openamundsen_da.methods.wet_snow.area import compute_step_wet_snow_daily_for_all_members
from openamundsen_da.methods.pf.rejuvenate import rejuvenate, validate_rejuvenation_manifest
from openamundsen_da.methods.pf.resample import resample_from_weights, _read_resampling_from_project
from openamundsen_da.methods.pf.weights import (
    carry_weights_to_next_step,
    combine_event_weights,
    initialize_prior_weights,
    load_event_weights,
    load_prior_weights,
    prior_weight_paths,
    write_event_weights,
    event_weights_manifest_path,
)
from openamundsen_da.pipeline.plot_tasks import (
    run_live_plots,
)
from openamundsen_da.results import RenderResult
from openamundsen_da.pipeline.rendering import render_required_project_outputs
from openamundsen_da.pipeline.cleanup import clean_predecessor_checkpoint
from openamundsen_da.util.validation import validate_assimilation_requirements
from openamundsen_da.util.run_mode import ensure_run_mode
from openamundsen_da.util.station_da import is_station_variable
from openamundsen_da.util.da_output import (
    output_retention_mode,
    write_project_da_output_grids,
)
from openamundsen_da.util.da_observables import (
    station_diagnostics_csv_name,
    weights_csv_name,
)

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
    "wet_snow_line": {
        "wet_classify": True,
        "wet_daily": True,
        "wet_plots": True,
    },
}


def _list_steps_sorted(project_dir: Path) -> List[Path]:
    return list_steps_sorted(project_dir)


def _next_step_start(steps: List[Path], idx: int) -> Optional[datetime]:
    if idx + 1 >= len(steps):
        return None
    cfg = read_step_config(steps[idx + 1]) or {}
    val = cfg.get("start_date")
    try:
        return datetime.fromisoformat(str(val)) if val else None
    except Exception:
        return None


def _find_roi(setup_dir: Path) -> Path:
    """Return an ROI vector path, generating one from ROI raster if needed."""
    return ensure_setup_roi_vector(Path(setup_dir))


def _load_wet_snow_classification_config(project_dir: Path) -> WetSnowClassificationConfig:
    """Read model wet-snow classification config from project YAML."""
    return load_wet_snow_classification_config(project_dir)


def _load_wet_snow_threshold_percent(project_dir: Path) -> float:
    """Read wet-snow ratio classification threshold (percent) from project YAML."""
    return _load_wet_snow_classification_config(project_dir).threshold_percent


def _resolve_wet_snow_classification_config(
    project_dir: Path,
    *,
    wet_snow_enabled: bool,
) -> WetSnowClassificationConfig | None:
    """Read wet-snow config only when wet-snow diagnostics are active."""
    if not wet_snow_enabled:
        return None
    return _load_wet_snow_classification_config(project_dir)


def _compute_prior_step_diagnostics(
    *,
    cfg: "OrchestratorConfig",
    step_dir: Path,
    roi: Path,
    lc_cfg: LandcoverMaskConfig,
    workers: int,
    scf_enabled: bool,
    wet_snow_enabled: bool,
    wet_snow_classification: WetSnowClassificationConfig | None,
) -> StorageAccountingSummary:
    """Compute setup-level prior diagnostics that depend on propagated member outputs."""
    step_name = Path(step_dir).name
    outputs: list[Path] = []

    def record(produced: object) -> None:
        if produced is None:
            return
        outputs.extend(Path(path) for path in produced)

    try:
        if scf_enabled:
            record(compute_step_scf_daily_for_all_members(
                setup_dir=cfg.setup_dir,
                project_dir=cfg.project_dir,
                step_dir=step_dir,
                aoi_path=roi,
                landcover_cfg=lc_cfg,
                max_workers=int(workers),
                overwrite=bool(cfg.overwrite),
            ))
    except Exception as exc:
        logger.warning("Model SCF daily computation failed for {}: {}", step_name, exc)

    for variable in ("swe", "hs"):
        try:
            record(compute_step_roi_mean_daily_for_all_members(
                step_dir=step_dir,
                aoi_path=roi,
                variable=variable,
                model_grid_format=configured_model_grid_format(cfg.setup_dir).value,
                max_workers=int(workers),
                overwrite=bool(cfg.overwrite),
            ))
        except Exception as exc:
            logger.warning("ROI mean {} daily computation failed for {}: {}", variable, step_name, exc)

    try:
        if wet_snow_enabled:
            if wet_snow_classification is None:
                raise ValueError("Wet-snow diagnostics are enabled but no classification config was loaded")
            record(classify_step_wet_snow(
                step_dir=step_dir,
                members=None,
                threshold_percent=wet_snow_classification.threshold_percent,
                classification_method=wet_snow_classification.method,
                liquid_water_amount_threshold_mm=wet_snow_classification.liquid_water_amount_threshold_mm,
                output_subdir="wet_snow",
                mask_prefix="wet_snow_mask",
                fraction_prefix="lwc_fraction",
                write_fraction=False,
                overwrite=bool(cfg.overwrite),
                max_workers=int(workers),
            ))
            record(compute_step_wet_snow_daily_for_all_members(
                setup_dir=cfg.setup_dir,
                project_dir=cfg.project_dir,
                step_dir=step_dir,
                aoi_path=roi,
                landcover_cfg=lc_cfg,
                max_workers=int(workers),
                overwrite=bool(cfg.overwrite),
                mask_subdir="wet_snow",
                mask_prefix="wet_snow_mask",
            ))
    except Exception as exc:
        logger.warning("Model wet-snow diagnostics failed for {}: {}", step_name, exc)
    return accounting_summary_from_paths(
        completed_step=step_name,
        root=step_dir,
        paths=outputs,
        source="prior_step_diagnostics",
    )


def _write_station_diagnostics(
    *,
    assim_dir: Path,
    variable: str,
    dt: datetime,
    diagnostics: pd.DataFrame,
) -> Path:
    out = assim_dir / station_diagnostics_csv_name(variable, dt)
    diagnostics.to_csv(out, index=False)
    logger.info("Wrote station diagnostics -> {}", out)
    return out


def _update_station_diagnostics_weights(path: Path, weights: pd.DataFrame) -> None:
    """Replace provisional likelihood-only station weights with PF posteriors."""
    diagnostics = pd.read_csv(path)
    stale = [
        column
        for column in (
            "final_weight",
            "final_log_weight",
            "prior_weight",
            "prior_log_weight",
            "member_log_likelihood",
        )
        if column in diagnostics.columns
    ]
    diagnostics = diagnostics.drop(columns=stale)
    aligned = weights[
        ["member_id", "prior_weight", "prior_log_weight", "log_likelihood", "weight", "log_weight"]
    ].rename(
        columns={
            "log_likelihood": "member_log_likelihood",
            "weight": "final_weight",
            "log_weight": "final_log_weight",
        }
    )
    diagnostics = diagnostics.merge(aligned, on="member_id", how="left", validate="many_to_one")
    diagnostics.to_csv(path, index=False)


def _run_assimilation_for_event(
    *,
    cfg: "OrchestratorConfig",
    step_dir: Path,
    roi: Path,
    lc_cfg: LandcoverMaskConfig,
    assim_dir: Path,
    ev: AssimilationEvent,
    assim_dt: datetime,
) -> tuple[pd.DataFrame, Path | None]:
    if ev.variable == "wet_snow_line":
        weights = assimilate_wet_snow_line_for_date(
            setup_dir=cfg.setup_dir,
            step_dir=step_dir,
            ensemble="prior",
            date=assim_dt,
            aoi=roi,
            landcover_cfg=lc_cfg,
            obs_csv=None,
            product=ev.product,
        )
        return weights, None
    if ev.variable == "wet_snow":
        weights = assimilate_wet_snow_for_date(
            setup_dir=cfg.setup_dir,
            step_dir=step_dir,
            ensemble="prior",
            date=assim_dt,
            aoi=roi,
            landcover_cfg=lc_cfg,
            obs_csv=None,
            product=ev.product,
        )
        return weights, None
    if ev.variable == "station_hs":
        station_result = assimilate_station_hs_for_date(
            setup_dir=cfg.setup_dir,
            step_dir=step_dir,
            ensemble="prior",
            date=assim_dt,
        )
        diag_csv = _write_station_diagnostics(
            assim_dir=assim_dir,
            variable=ev.variable,
            dt=assim_dt,
            diagnostics=station_result.diagnostics,
        )
        return station_result.weights, diag_csv
    if ev.variable == "station_swe":
        station_result = assimilate_station_swe_for_date(
            setup_dir=cfg.setup_dir,
            step_dir=step_dir,
            ensemble="prior",
            date=assim_dt,
        )
        diag_csv = _write_station_diagnostics(
            assim_dir=assim_dir,
            variable=ev.variable,
            dt=assim_dt,
            diagnostics=station_result.diagnostics,
        )
        return station_result.weights, diag_csv
    weights = assimilate_scf_for_date(
        setup_dir=cfg.setup_dir,
        step_dir=step_dir,
        ensemble="prior",
        date=assim_dt,
        aoi=roi,
        landcover_cfg=lc_cfg,
        obs_csv=None,
        product=ev.product,
    )
    return weights, None


@dataclass
class OrchestratorConfig:
    project_dir: Path
    setup_dir: Path
    max_workers: int = 4
    overwrite: bool = False
    log_level: str = "INFO"
    live_plots: bool = False
    plot_workers: int | None = None
    monitor_perf: bool = False
    perf_sample_interval: float = 5.0
    perf_plot_interval: float = 30.0
    storage_reservation_projects: tuple[StorageReservationProject, ...] = ()
    storage_outer_workers: int = 1
    shared_storage_reserve_bytes: int = 0
    storage_admission_client: StorageAdmissionClient | None = None


def preadmit_project_storage(cfg: OrchestratorConfig) -> OrchestratorConfig:
    """Build and admit a single-domain plan before any runtime mutation."""
    if cfg.storage_admission_client is not None:
        return cfg
    project_config = load_project_configuration(cfg.project_dir)
    grid = resolve_setup_grid_spec(cfg.setup_dir)
    reservation_project = StorageReservationProject(
        setup_dir=Path(cfg.setup_dir).resolve(),
        project_dir=Path(cfg.project_dir).resolve(),
        grid_cell_count=int(grid.rows) * int(grid.cols),
    )
    plan = build_storage_plan(
        root_project_dir=cfg.project_dir,
        projects=(reservation_project,),
        outer_workers=1,
        overwrite=cfg.overwrite,
        leaf_ids=("project",),
    )
    steps = _list_steps_sorted(project_config.project_dir)
    if not steps:
        raise FileNotFoundError(f"No steps found under {project_config.project_dir}")
    initial_resume = (
        steps[0] / "assim" / "prior_forcing_manifest.json"
    ).is_file() and not cfg.overwrite
    # This read-only gate happens before the coordinator ledger, run manifest,
    # file log or performance monitor is created.
    check_step_admission(
        cfg.project_dir,
        estimated_growth_bytes=plan.estimated_growth_bytes,
        allow_existing_step_drain=initial_resume,
    )
    coordinator = StorageAdmissionCoordinator(plan)
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="project")
    client.admit_step(
        steps[0].name,
        request_id=f"project:admit:{steps[0].name}",
        allow_existing_step_drain=initial_resume,
    )
    return replace(cfg, storage_admission_client=client)


def _storage_summary(value: object) -> StorageAccountingSummary:
    if not isinstance(value, dict):
        raise RuntimeError("Producer omitted its required storage accounting summary")
    return StorageAccountingSummary.from_dict(value)


def _setup_logger(project_dir: Path, log_level: str) -> None:
    """Configure Loguru sinks for console and project file log."""
    logger.remove()
    logger.add(sys.stdout, level=log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)
    log_file = _setup_log_path(project_dir)
    logger.add(log_file, level=log_level.upper(), colorize=False, enqueue=True, format=LOGURU_FORMAT)


def _auto_project_dir(setup_dir: Path) -> Path:
    """Best-effort discovery of a single project under `<setup_dir>/projects`."""
    setup_dir = Path(setup_dir).resolve()
    projects_root = setup_dir / "projects"
    if not projects_root.is_dir():
        raise FileNotFoundError(
            f"Could not find projects directory under setup dir {setup_dir}. "
            "Pass --project-dir explicitly."
        )
    candidates = []
    for cand in sorted(projects_root.iterdir()):
        if not cand.is_dir():
            continue
        try:
            _ = find_project_yaml(cand)
            candidates.append(cand)
        except FileNotFoundError:
            continue
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No project directories with YAML found under {projects_root}. "
            "Pass --project-dir explicitly."
        )
    raise FileNotFoundError(
        f"Multiple project directories found under {projects_root}. "
        "Pass --project-dir explicitly."
    )


def _setup_log_path(project_dir: Path) -> Path:
    """Return a project log path named by project years when available."""
    project_dir = Path(project_dir)
    label = project_dir.name
    try:
        cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
        start_val = cfg.get("start_date")
        end_val = cfg.get("end_date")

        def _year(val: object | None) -> int | None:
            if val is None:
                return None
            try:
                return datetime.fromisoformat(str(val)).year
            except Exception:
                return None

        sy = _year(start_val)
        ey = _year(end_val)
        if sy and ey:
            label = f"project_{sy}_{ey}"
        elif sy:
            label = f"project_{sy}"
    except Exception:
        # fall back to directory name
        pass
    return project_dir / f"{label}.log"


def _run_project_impl(cfg: OrchestratorConfig, *, run_start: datetime) -> RenderResult:
    project_config = load_project_configuration(cfg.project_dir)
    if cfg.storage_admission_client is None:
        grid = resolve_setup_grid_spec(cfg.setup_dir)
        reservation_project = StorageReservationProject(
            setup_dir=Path(cfg.setup_dir).resolve(),
            project_dir=Path(cfg.project_dir).resolve(),
            grid_cell_count=int(grid.rows) * int(grid.cols),
        )
        plan = build_storage_plan(
            root_project_dir=cfg.project_dir,
            projects=(reservation_project,),
            outer_workers=1,
            overwrite=cfg.overwrite,
            leaf_ids=("project",),
        )
        coordinator = StorageAdmissionCoordinator(plan)
        cfg = replace(
            cfg,
            storage_admission_client=StorageAdmissionClient.in_process(
                coordinator,
                leaf_id="project",
            ),
        )
    if project_config.setup_dir != Path(cfg.setup_dir).resolve():
        raise ValueError(
            f"Project {project_config.project_dir} belongs to setup {project_config.setup_dir}, "
            f"not configured setup {Path(cfg.setup_dir).resolve()}"
        )

    live_plot_threads: list[threading.Thread] = []

    steps = _list_steps_sorted(cfg.project_dir)
    if not steps:
        raise FileNotFoundError(f"No steps found under {cfg.project_dir}")
    logger.info("Discovered {} step(s) for project {}", len(steps), cfg.project_dir.name)
    workers = pick_max_workers(cfg.max_workers)

    # Ensure first step has its prior ensemble (project/meteo is required)
    if steps:
        meteo_dir = cfg.setup_dir / "meteo"
        if not meteo_dir.is_dir():
            raise FileNotFoundError(f"Required meteo directory not found: {meteo_dir}")
        logger.info("Initializing prior ensemble for step {} ...", steps[0].name)

    # Assimilation configuration (variable/product per date)
    events = load_assimilation_events(cfg.project_dir)
    n_expected = max(0, len(steps) - 1)
    if len(events) < n_expected:
        raise ValueError(
            f"Configured {len(events)} assimilation event(s) but the project needs {n_expected}. "
            "Add events in project YAML (data_assimilation.assimilation_events) or adjust steps."
        )
    if len(events) > n_expected:
        logger.warning("More assimilation events ({}) than steps needing DA ({}); extra events will be ignored.", len(events), n_expected)
    vars_used = {getattr(ev, "variable", None) for ev in events if getattr(ev, "variable", None)}
    retention_mode = output_retention_mode(cfg.project_dir)
    logger.info("Output retention policy: {}", retention_mode)
    scf_enabled = "scf" in vars_used
    wet_snow_enabled = bool({"wet_snow", "wet_snow_line"} & vars_used)
    if not vars_used:
        logger.info("No assimilation events found; skipping SCF/wet-snow diagnostics (explicit variables only).")
        scf_enabled = False
        wet_snow_enabled = False
    else:
        logger.info("Assimilation variables detected: {}", ", ".join(sorted(vars_used)))
        if wet_snow_enabled:
            logger.info("Wet-snow diagnostics enabled (wet_snow / wet_snow_line present in assimilation_events).")
        else:
            logger.info("Wet-snow diagnostics disabled (wet_snow / wet_snow_line not in assimilation_events).")
        if scf_enabled:
            logger.info("SCF diagnostics enabled (scf present in assimilation_events).")
        else:
            logger.info("SCF diagnostics disabled (scf not in assimilation_events).")

    # Validate required outputs and obs inputs before running assimilation
    validate_assimilation_requirements(setup_dir=cfg.setup_dir, project_dir=cfg.project_dir, steps=steps, events=events)
    initial_forcing_manifest = steps[0] / "assim" / "prior_forcing_manifest.json"
    initial_resume = initial_forcing_manifest.is_file() and not cfg.overwrite
    if cfg.storage_admission_client is None:
        raise RuntimeError("Project execution requires a storage admission client")
    initial_budget = cfg.storage_admission_client.admit_step(
        steps[0].name,
        request_id=f"{cfg.storage_admission_client.leaf_id}:admit:{steps[0].name}",
        allow_existing_step_drain=initial_resume,
    )
    # Do not create/truncate the mutable project log until new work has passed
    # the hard 80% admission gate.
    _setup_logger(cfg.project_dir, cfg.log_level)
    logger.info(
        "Disk admission {}: used={:.1%}, incremental reserve={:.1f} GiB, "
        "operational reserve={:.1f} GiB",
        steps[0].name,
        initial_budget.used_fraction,
        initial_budget.estimated_growth_bytes / (1024**3),
        initial_budget.operational_reserve_bytes / (1024**3),
    )
    initial_forcing_accounting = build_prior_ensemble(
        input_meteo_dir=meteo_dir,
        project_dir=cfg.project_dir,
        step_dir=steps[0],
        max_workers=int(workers),
        overwrite=bool(cfg.overwrite),
    )
    pending_storage_accounting: dict[str, StorageAccountingSummary] = {
        steps[0].name: _storage_summary(initial_forcing_accounting)
    }
    initial_members = list_member_dirs(steps[0] / "ensembles", "prior")
    if not initial_members:
        raise RuntimeError(f"No prior members available for PF initialization in {steps[0]}")
    initial_weight_csv, initial_weight_manifest = prior_weight_paths(steps[0])
    if initial_weight_csv.is_file() and initial_weight_manifest.is_file():
        load_prior_weights(steps[0], [member.name for member in initial_members])
    elif initial_weight_csv.exists() or initial_weight_manifest.exists():
        raise RuntimeError(f"Incomplete PF prior-weight ledger in {steps[0] / 'assim'}")
    else:
        existing_event_weights = [
            path
            for step in steps
            for path in (step / "assim").glob("weights_*.csv")
        ]
        if existing_event_weights:
            raise RuntimeError(
                "Existing assimilation artifacts do not have the required sequential PF weight ledger; "
                "v0.9.3 chains are view-only and must be rerun from a fresh project"
            )
        initialize_prior_weights(steps[0], [member.name for member in initial_members])

    roi_grid = ensure_setup_roi_grid(cfg.setup_dir)
    logger.info("Using ROI grid: {}", roi_grid)
    roi = _find_roi(cfg.setup_dir)
    logger.info("Using ROI: {}", roi)
    lc_cfg = resolve_landcover_mask(cfg.setup_dir, cfg.project_dir)
    if lc_cfg.enabled:
        logger.info("Land-cover mask enabled -> {} (classes {})", lc_cfg.path, list(lc_cfg.classes))
    else:
        logger.info("Land-cover mask disabled; no land-cover exclusions applied")

    # Project/setup metadata for DA and performance monitoring
    wet_snow_classification = _resolve_wet_snow_classification_config(
        cfg.project_dir,
        wet_snow_enabled=wet_snow_enabled,
    )
    if wet_snow_classification is not None:
        if wet_snow_classification.method == CLASSIFICATION_METHOD_AMOUNT:
            logger.info(
                "Wet-snow classification method={} threshold={:.3f} mm (project YAML)",
                wet_snow_classification.method,
                wet_snow_classification.liquid_water_amount_threshold_mm,
            )
        else:
            logger.info(
                "Wet-snow classification method={} threshold={:.3f} % (project YAML)",
                wet_snow_classification.method,
                wet_snow_classification.threshold_percent,
            )

    proj_crs = None
    try:
        proj_yaml = find_setup_yaml(cfg.setup_dir)
        proj_cfg = _read_yaml_file(proj_yaml) or {}
        proj_crs = proj_cfg.get("crs")
    except Exception as exc:
        logger.warning("Perf monitor: failed to read config metadata: {}", exc)

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
            lc_report_path = project_landcover_mask_report_path(cfg.project_dir)
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

    # Process each step
    last_step_storage_accounting: StorageAccountingSummary | None = None
    for i, step_dir in enumerate(steps):
        step_name = Path(step_dir).name
        logger.info("== Step {} ==", step_name)

        # Launch ensemble (runner enforces strict cold/warm semantics by step)
        logger.info("Launching ensemble (prior) with max_workers={} overwrite={} ...", workers, cfg.overwrite)
        launch_summary = launch_members(
            project_dir=cfg.project_dir,
            setup_dir=cfg.setup_dir,
            step_dir=step_dir,
            ensemble="prior",
            max_workers=int(workers),
            overwrite=bool(cfg.overwrite),
            results_root=None,
            log_level=cfg.log_level,
            state_pattern=None,
        )
        if launch_summary.get("summary", {}).get("failed", 0) > 0:
            raise RuntimeError(f"Member propagation failed in {step_name}; restart states were retained")
        forcing_accounting = pending_storage_accounting.pop(step_name, None)
        if forcing_accounting is None:
            raise RuntimeError(
                f"Storage accounting for generated forcing is missing in {step_name}"
            )
        step_storage_accounting = forcing_accounting.merged(
            _storage_summary(launch_summary.get("storage_accounting"))
        )

        if i > 0:
            try:
                removed_checkpoint = clean_predecessor_checkpoint(
                    cfg.project_dir,
                    steps[i - 1],
                    successor_step=step_dir,
                    apply=True,
                )
            except RuntimeError:
                if i < len(steps) - 1:
                    raise
                removed_checkpoint = ()
                logger.warning(
                    "Final-step checkpoint is unavailable; retaining predecessor {} "
                    "until final project cleanup",
                    steps[i - 1].name,
                )
            if removed_checkpoint:
                logger.info(
                    "Compact retention removed validated predecessor checkpoint {} ({} files)",
                    steps[i - 1].name,
                    len(removed_checkpoint),
                )

        diagnostics_accounting = _compute_prior_step_diagnostics(
            cfg=cfg,
            step_dir=step_dir,
            roi=roi,
            lc_cfg=lc_cfg,
            workers=int(workers),
            scf_enabled=scf_enabled,
            wet_snow_enabled=wet_snow_enabled,
            wet_snow_classification=wet_snow_classification,
        )
        step_storage_accounting = step_storage_accounting.merged(
            diagnostics_accounting
        )
        last_step_storage_accounting = step_storage_accounting

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
        # re-running oa-da-project can skip already-assimilated steps.
        assim_dir = Path(step_dir) / "assim"
        assim_dir.mkdir(parents=True, exist_ok=True)
        weights_name = weights_csv_name(ev.variable, assim_dt)
        wcsv = assim_dir / weights_name
        rs_cfg = _read_resampling_from_project(cfg.project_dir)
        algo = rs_cfg.algorithm
        ess_thr_abs = float(rs_cfg.ess_threshold or 0.0)
        ess_thr_ratio = rs_cfg.ess_threshold_ratio
        station_diag_csv: Path | None = None
        if is_station_variable(ev.variable):
            station_diag_csv = assim_dir / station_diagnostics_csv_name(ev.variable, assim_dt)
        if wcsv.is_file() and not cfg.overwrite:
            logger.info(
                "Weights CSV already exists for {}; overwrite=False -> reusing existing weights: {}",
                step_name,
                wcsv,
            )
            weights = load_event_weights(
                wcsv,
                project_dir=cfg.project_dir,
                step_dir=step_dir,
            )
            required_weight_columns = {
                "member_id",
                "prior_log_weight",
                "prior_weight",
                "log_likelihood",
                "log_weight",
                "weight",
            }
            if not required_weight_columns.issubset(weights.columns):
                raise RuntimeError(
                    f"Existing weights use an incompatible pre-ledger contract: {wcsv}; "
                    "rerun the project from a fresh copy"
                )
            prior = load_prior_weights(step_dir, weights["member_id"].astype(str).tolist())
            if not np.allclose(
                pd.to_numeric(weights["prior_weight"], errors="raise").to_numpy(dtype=float),
                prior["weight"].to_numpy(dtype=float),
            ):
                raise RuntimeError(f"Existing event weights do not match the prior-weight ledger: {wcsv}")
        else:
            try:
                weights, station_diag_csv = _run_assimilation_for_event(
                    cfg=cfg,
                    step_dir=step_dir,
                    roi=roi,
                    lc_cfg=lc_cfg,
                    assim_dir=assim_dir,
                    ev=ev,
                    assim_dt=assim_dt,
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
            weights = combine_event_weights(weights, step_dir=step_dir)
            n_members = len(weights)
            threshold = (
                float(ess_thr_ratio) * float(n_members)
                if ess_thr_ratio is not None
                else float(ess_thr_abs)
            )
            posterior_ess = float(weights["posterior_ess"].iloc[0])
            resampled = bool(threshold > 0.0 and posterior_ess < threshold)
            weights["resampling_threshold"] = threshold
            weights["resampling_threshold_ratio"] = ess_thr_ratio
            weights["resampled"] = resampled
            weights["ess_below_threshold"] = resampled
            write_event_weights(
                wcsv,
                weights,
                project_dir=cfg.project_dir,
                step_dir=step_dir,
            )
            if station_diag_csv is not None:
                _update_station_diagnostics_weights(station_diag_csv, weights)
            logger.info(
                "Wrote recursive PF weights -> {} (prior ESS={:.2f}, posterior ESS={:.2f}, threshold={:.2f}, resampled={})",
                wcsv,
                float(weights["prior_ess"].iloc[0]),
                posterior_ess,
                threshold,
                resampled,
            )

        # Resample to posterior
        posterior_root = Path(step_dir) / "ensembles" / "posterior"
        has_posterior = posterior_root.is_dir() and any(posterior_root.glob("member_*"))
        if has_posterior and not cfg.overwrite:
            logger.info("Posterior ensemble already exists and overwrite=False; skipping resampling.")
        else:
            ratio_text = f"{ess_thr_ratio:.3f}" if ess_thr_ratio is not None else "None"
            logger.info(
                "Resampling to posterior ... (algorithm={} seed={} ess_thr_abs={} ess_thr_ratio={})",
                algo,
                rs_cfg.seed if rs_cfg.seed is not None else "auto",
                ess_thr_abs,
                ratio_text,
            )
            resampling_summary = resample_from_weights(
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
        if has_posterior and not cfg.overwrite:
            # Re-evaluate the deterministic resampling decision and validate
            # the existing materialization against its mapping.
            resampling_summary = resample_from_weights(
                step_dir=step_dir,
                source_ensemble="prior",
                weights_csv=wcsv,
                target_ensemble="posterior",
                seed=rs_cfg.seed,
                algorithm=algo,
                ess_threshold=ess_thr_abs,
                ess_threshold_ratio=ess_thr_ratio,
                overwrite=False,
            )
        expected_resampled = bool(weights["resampled"].iloc[0])
        if bool(resampling_summary["resampled"]) != expected_resampled:
            raise RuntimeError(
                f"Resampling result disagrees with recorded event decision in {wcsv}"
            )

        mapping = pd.read_csv(Path(resampling_summary["indices_csv"]))
        next_weight_csv, next_weight_manifest = prior_weight_paths(steps[i + 1])
        if next_weight_csv.exists() != next_weight_manifest.exists():
            raise RuntimeError(f"Incomplete next-step PF ledger in {steps[i + 1] / 'assim'}")
        if next_weight_csv.is_file() and next_weight_manifest.is_file() and not cfg.overwrite:
            expected_ids = mapping["posterior_member_id"].astype(str).tolist()
            existing_next = load_prior_weights(steps[i + 1], expected_ids)
            if bool(resampling_summary["resampled"]):
                expected_weights = pd.Series([1.0 / len(expected_ids)] * len(expected_ids), dtype=float)
            else:
                event_by_member_frame = weights.copy()
                event_by_member_frame["member_id"] = event_by_member_frame["member_id"].astype(str)
                event_by_member = event_by_member_frame.set_index("member_id")["weight"]
                expected_weights = pd.Series(
                    [float(event_by_member.loc[str(source)]) for source in mapping["source_member_id"]],
                    dtype=float,
                )
                expected_weights /= expected_weights.sum()
            if not np.allclose(
                existing_next["weight"].to_numpy(dtype=float),
                expected_weights.to_numpy(dtype=float),
            ):
                raise RuntimeError(f"Existing next-step PF ledger does not match event ancestry: {next_weight_csv}")
        else:
            carry_weights_to_next_step(
                current_step_dir=step_dir,
                next_step_dir=steps[i + 1],
                event_weights=weights,
                mapping=mapping,
                resampled=bool(resampling_summary["resampled"]),
                source_weights=wcsv,
                overwrite=bool(cfg.overwrite),
            )

        # Rejuvenate posterior -> next prior
        rejuvenate_manifest = Path(steps[i + 1]) / "assim" / "rejuvenate_manifest.json"
        next_resume = rejuvenate_manifest.is_file() and not cfg.overwrite
        boundary_outputs = [
            wcsv,
            event_weights_manifest_path(wcsv),
            Path(resampling_summary["indices_csv"]),
            Path(resampling_summary["manifest_json"]),
            next_weight_csv,
            next_weight_manifest,
            *(Path(path) for path in resampling_summary["storage_output_paths"]),
        ]
        if station_diag_csv is not None:
            boundary_outputs.append(station_diag_csv)
        boundary_accounting = accounting_summary_from_paths(
            completed_step=step_name,
            root=cfg.project_dir,
            paths=boundary_outputs,
            source="step_lifecycle_outputs",
        )
        step_storage_accounting = step_storage_accounting.merged(boundary_accounting)
        next_budget = cfg.storage_admission_client.admit_step(
            steps[i + 1].name,
            summary=step_storage_accounting,
            request_id=(
                f"{cfg.storage_admission_client.leaf_id}:admit:{steps[i + 1].name}"
            ),
            allow_existing_step_drain=next_resume,
        )
        logger.info(
            "Disk admission {}: used={:.1%}, incremental reserve={:.1f} GiB",
            steps[i + 1].name,
            next_budget.used_fraction,
            next_budget.estimated_growth_bytes / (1024**3),
        )
        if rejuvenate_manifest.is_file() and not cfg.overwrite:
            rejuvenation_data = validate_rejuvenation_manifest(
                setup_dir=cfg.setup_dir,
                prev_step_dir=step_dir,
                next_step_dir=steps[i + 1],
            )
            storage_accounting = rejuvenation_data.get("storage_accounting")
            if not isinstance(storage_accounting, dict):
                from openamundsen_da.methods.pf.rejuvenate import _rejuvenation_output_inventory

                storage_accounting = accounting_summary_from_inventory(
                    completed_step=steps[i + 1].name,
                    inventory=_rejuvenation_output_inventory(
                        setup_dir=cfg.setup_dir,
                        next_step_dir=steps[i + 1],
                        target_ensemble="prior",
                    ),
                    source="rejuvenation_reconciliation",
                ).as_dict()
            pending_storage_accounting[steps[i + 1].name] = reused_accounting_summary(
                _storage_summary(storage_accounting),
                source="rejuvenation_reused",
            )
            logger.info(
                "Validated rejuvenation manifest for {}; overwrite=False -> skipping process-noise rebuild.",
                steps[i + 1].name,
            )
        else:
            logger.info("Rejuvenating posterior -> {} (prior) ...", steps[i + 1].name)
            rejuvenation_summary = rejuvenate(
                setup_dir=cfg.setup_dir,
                prev_step_dir=step_dir,
                next_step_dir=steps[i + 1],
                source_ensemble="posterior",
                target_ensemble="prior",
                source_meteo_dir=None,
                max_workers=int(workers),
            )
            pending_storage_accounting[steps[i + 1].name] = _storage_summary(
                rejuvenation_summary.get("storage_accounting")
            )

        # Update project-wide plots after each assimilation/rejuvenation cycle so
        # users can monitor progress while the pipeline continues running. Plots
        # are written with deterministic filenames and therefore overwritten on
        # each update.
        if cfg.live_plots:
            logger.info("Dispatching live plots in background for {} ...", step_name)
            t = threading.Thread(
                target=run_live_plots,
                args=(cfg, step_dir, step_name, wcsv),
                kwargs={
                    "project_fraction_envelope_path": project_fraction_envelope_path,
                    "variable": ev.variable,
                    "station_diagnostics_csv": station_diag_csv,
                    "reset_logger": False,
                    "reset_logger_func": _setup_logger,
                    "wet_snow_enabled": wet_snow_enabled,
                },
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
    if last_step_storage_accounting is None:
        raise RuntimeError("Final step omitted its required storage accounting summary")
    cfg.storage_admission_client.reconcile_finalization(
        request_id=(
            f"{cfg.storage_admission_client.leaf_id}:reconcile_finalization"
        ),
    )
    finalization_budget = cfg.storage_admission_client.transition(
        "project_finalizing",
        summary=last_step_storage_accounting,
        request_id=f"{cfg.storage_admission_client.leaf_id}:project_finalizing",
    )
    logger.info(
        "Project-finalization storage admission: used={:.1%}, incremental reserve={:.1f} GiB",
        finalization_budget.used_fraction,
        finalization_budget.estimated_growth_bytes / (1024**3),
    )
    da_summary_path = project_da_output_grids_path(cfg.project_dir)
    if da_summary_path.is_file() and not cfg.overwrite:
        logger.info("Using existing DA output summary {}", da_summary_path)
    else:
        da_path = write_project_da_output_grids(
            step_dirs=steps,
            output_nc=da_summary_path,
        )
        if da_path is None or not da_summary_path.is_file():
            raise RuntimeError(f"Required compact DA output was not written: {da_summary_path}")

    if retention_mode == "compact":
        point_output_path = project_ensemble_points_path(cfg.project_dir)
        forcing_output_path = project_ensemble_forcing_path(cfg.project_dir)
        if point_output_path.is_file() and not cfg.overwrite:
            validate_project_ensemble_points(cfg.project_dir, output_nc=point_output_path)
            logger.info("Using existing compact point output {}", point_output_path)
        else:
            written_points = write_project_ensemble_points(cfg.project_dir, output_nc=point_output_path)
            if not written_points.is_file():
                raise RuntimeError(f"Required compact point output was not written: {point_output_path}")

        if forcing_output_path.is_file() and not cfg.overwrite:
            validate_project_ensemble_forcing(cfg.project_dir, output_nc=forcing_output_path)
            logger.info("Using existing compact forcing output {}", forcing_output_path)
        else:
            written_forcing = write_project_ensemble_forcing(
                cfg.project_dir,
                output_nc=forcing_output_path,
            )
            if not written_forcing.is_file():
                raise RuntimeError(f"Required compact forcing output was not written: {forcing_output_path}")

        if bool({"scf", "wet_snow", "wet_snow_line"} & vars_used):
            from openamundsen_da.methods.viz.maps.panel_renderers import write_project_da_map_support
            from openamundsen_da.util.map_support import validate_map_support

            map_support_path = write_project_da_map_support(cfg.project_dir)
            if map_support_path is None or not map_support_path.is_file():
                raise RuntimeError("Required retained DA-event map support was not written")
            required_fields: set[str] = set()
            if "scf" in vars_used:
                required_fields.update(
                    {
                        "scf_open_loop_binary",
                        "scf_prior_probability",
                        "scf_posterior_probability",
                    }
                )
            if vars_used & {"wet_snow", "wet_snow_line"}:
                required_fields.update(
                    {
                        "wet_snow_open_loop",
                        "wet_snow_prior_probability",
                        "wet_snow_posterior_probability",
                    }
                )
            validate_map_support(
                cfg.project_dir,
                dates=[event.date for event in events],
                fields=required_fields,
            )

    try:
        benchmark_outputs = run_project_benchmark(
            project_dir=cfg.project_dir,
            setup_dir=cfg.setup_dir,
            max_workers=cfg.max_workers,
            overwrite=cfg.overwrite,
            reuse_existing_prerequisites=True,
        )
        logger.info(
            "Benchmark stage complete -> {}",
            benchmark_outputs.get("manifest", cfg.project_dir / "results" / "benchmark" / "manifest.json"),
        )
    except Exception:
        logger.exception("Project benchmarking failed")
        raise

    render_result = render_required_project_outputs(
        cfg.project_dir,
        max_workers=cfg.plot_workers or cfg.max_workers,
    )

    _setup_logger(cfg.project_dir, cfg.log_level)
    run_end = datetime.utcnow()
    duration = (run_end - run_start).total_seconds()
    logger.info("Project processing complete: {} (wall-clock {:.1f} s, ~{:.2f} h)", cfg.project_dir, duration, duration / 3600.0)

    completion_phase = (
        "leaf_finalized"
        if cfg.storage_admission_client.leaf_id == "project"
        else "leaf_project_complete"
    )
    cfg.storage_admission_client.transition(
        completion_phase,
        request_id=f"{cfg.storage_admission_client.leaf_id}:{completion_phase}",
    )

    return render_result


def run_project(cfg: OrchestratorConfig) -> RenderResult:
    """Run the orchestrator while owning performance-monitor shutdown."""

    cfg = preadmit_project_storage(cfg)
    run_start = datetime.utcnow()
    perf_handle = None
    if cfg.monitor_perf:
        perf_handle = start_perf_monitor(
            PerfMonitorConfig(
                project_dir=cfg.project_dir,
                sample_interval_sec=float(cfg.perf_sample_interval or 5.0),
                plot_interval_sec=float(cfg.perf_plot_interval or 30.0),
                run_start=run_start,
            )
        )
    try:
        return _run_project_impl(
            replace(cfg, monitor_perf=False),
            run_start=run_start,
        )
    finally:
        if perf_handle is not None:
            perf_handle.stop_and_join()
            perf_handle.capture_now()


def cli(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="oa-da-project", description="Process a full project: run steps, assimilate, resample, rejuvenate, plot.")
    p.add_argument("--project-dir", type=Path, help="Project directory (auto-detected by walking up from --setup-dir when omitted).")
    p.add_argument("--setup-dir", required=True, type=Path)
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
        help="Enable plotting during the project run; default is off (plots run after completion).",
    )
    p.add_argument(
        "--no-live-plots",
        dest="live_plots",
        action="store_false",
        help="Skip plotting during the project run (default).",
    )
    p.add_argument(
        "--monitor-perf",
        dest="monitor_perf",
        action="store_true",
        help="Enable background performance monitor (CPU/RAM/disk and optional CPU temperature) during the project run (default).",
    )
    p.add_argument(
        "--no-monitor-perf",
        dest="monitor_perf",
        action="store_false",
        help="Disable background performance monitor (CPU/RAM/disk and optional CPU temperature) during the project run.",
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
    p.set_defaults(live_plots=False, monitor_perf=True)
    args = p.parse_args(argv)

    setup_dir = Path(args.setup_dir)
    project_dir = Path(args.project_dir) if args.project_dir is not None else _auto_project_dir(setup_dir)
    if args.project_dir is None:
        print(f"[oa-da-project] Auto-detected project dir: {project_dir}", file=sys.stderr)
    ensure_run_mode(project_dir, expected="single", write_if_missing=True)

    resolved_workers = pick_max_workers(args.max_workers, fallback=4)

    run_project(
        OrchestratorConfig(
            project_dir=project_dir,
            setup_dir=setup_dir,
            max_workers=int(resolved_workers),
            overwrite=bool(args.overwrite),
            log_level=str(args.log_level or "INFO"),
            live_plots=bool(getattr(args, "live_plots", False)),
            plot_workers=(int(args.plot_workers) if args.plot_workers is not None else None),
            monitor_perf=bool(getattr(args, "monitor_perf", True)),
            perf_sample_interval=float(getattr(args, "perf_sample_interval", 5.0)),
            perf_plot_interval=float(getattr(args, "perf_plot_interval", 30.0)),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
