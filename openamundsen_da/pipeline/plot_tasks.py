"""Project-level plotting helpers used by the main project orchestrator."""

from __future__ import annotations

import concurrent.futures as cf
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.methods.viz.maps import project_maps_enabled, render_project_maps
from openamundsen_da.methods.viz.maps.generated import default_project_maps_rerun_command
from openamundsen_da.methods.viz.maps.runner import ProjectMapRenderError
from openamundsen_da.methods.viz.poster import (
    default_project_poster_rerun_command,
    poster_profile_enabled,
    render_poster_profile,
)
from openamundsen_da.methods.viz.reports import build_project_collection_pdf
from openamundsen_da.methods.viz.reports.project_collection_pdf import MissingProjectPdfArtifactsError
from openamundsen_da.methods.viz.plots.assimilation import (
    plot_setup_ess_timeline,
    plot_setup_weights_overview,
    plot_station_diagnostics_for_csv,
    plot_weights_for_csv,
)
from openamundsen_da.methods.viz.plots.forcing_ensemble import cli_main as plot_forcing_cli
from openamundsen_da.methods.viz.plots.project_ensemble import plot_setup_results
from openamundsen_da.methods.viz.plots.result_overview import cli_main as plot_result_overview_cli
from openamundsen_da.util.fraction_envelope import aggregate_fraction_envelope
from openamundsen_da.util.da_events import load_assimilation_events


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
        result = task.func(*task.args, **task.kwargs)
        if isinstance(result, int) and result != 0:
            return task.name, f"exit code {result}"
        return task.name, None
    except Exception as exc:  # pragma: no cover
        return task.name, str(exc)


def _aggregate_fraction(
    project_dir: Path,
    filename: str,
    value_col: str,
    output_path: Path,
) -> Path | None:
    return aggregate_fraction_envelope(
        setup_dir=project_dir,
        filename=filename,
        value_col=value_col,
        output_name=str(output_path.relative_to(project_dir)),
    )


def aggregate_fraction_envelopes(
    *,
    project_dir: Path,
    project_fraction_envelope_path: Callable[[Path, str], Path],
) -> None:
    """Aggregate configured fraction envelopes into results/misc."""
    specs = {
        "scf": ("SCF", "point_scf_roi.csv", "scf"),
        "wet_snow": ("WSF", "point_wet_snow_roi.csv", "wet_snow_fraction"),
        "wet_snow_line": ("WSLA", "point_wet_snow_line_roi.csv", "wet_snow_line"),
    }
    try:
        event_variables = {event.variable for event in load_assimilation_events(project_dir)}
    except Exception as exc:
        logger.debug("Could not load assimilation_events for envelope filtering in {}: {}", project_dir, exc)
        event_variables = set(specs)

    variables = set()
    if "scf" in event_variables:
        variables.add("scf")
    if "wet_snow" in event_variables or "wet_snow_line" in event_variables:
        variables.add("wet_snow")
    if "wet_snow_line" in event_variables:
        variables.add("wet_snow_line")

    for variable in ("scf", "wet_snow", "wet_snow_line"):
        if variable not in variables:
            continue
        label, filename, value_col = specs[variable]
        try:
            _aggregate_fraction(
                project_dir=project_dir,
                filename=filename,
                value_col=value_col,
                output_path=project_fraction_envelope_path(project_dir, variable),
            )
        except Exception as exc:
            logger.warning("{} envelope aggregation failed: {}", label, exc)


def run_plot_tasks_parallel(
    tasks: List[PlotTask],
    max_workers: int | None,
    setup_max_workers: int | None,
) -> None:
    """Execute plot tasks concurrently using process-based workers."""
    if not tasks:
        return
    cpu_cap = os.cpu_count() or len(tasks)
    candidates = [len(tasks), cpu_cap]
    if max_workers is not None:
        candidates.append(max_workers)
    if setup_max_workers is not None:
        candidates.append(setup_max_workers)
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


def run_live_plots(
    cfg,
    step_dir: Path,
    step_name: str,
    wcsv: Path,
    *,
    project_fraction_envelope_path: Callable[[Path, str], Path],
    variable: str,
    station_diagnostics_csv: Path | None = None,
    reset_logger: bool = True,
    reset_logger_func: Callable[[Path, str], None] | None = None,
    wet_snow_enabled: bool = True,
) -> None:
    """Run the per-step plotting suite."""
    try:
        logger.info("Updating project plots after assimilation step {} ...", step_name)
        aggregate_fraction_envelopes(
            project_dir=cfg.project_dir,
            project_fraction_envelope_path=project_fraction_envelope_path,
        )
        try:
            rc = plot_forcing_cli([
                "--step-dir", str(step_dir),
                "--ensemble", "prior",
                "--log-level", cfg.log_level,
            ], configure_logger=False)
            if isinstance(rc, int) and rc != 0:
                raise RuntimeError(f"plot_forcing_cli returned {rc}")
        except Exception as exc:
            logger.warning("Forcing plot failed for {}: {}", step_name, exc)
        try:
            plot_setup_results(
                setup_dir=cfg.project_dir,
                var_col="swe",
                mode="band",
                resample="D",
                resample_agg="mean",
                configure_logger=False,
            )
            plot_setup_results(
                setup_dir=cfg.project_dir,
                var_col="snow_depth",
                mode="band",
                resample="D",
                resample_agg="mean",
                configure_logger=False,
            )
        except Exception as exc:
            logger.warning("Setup point results plot failed after step {}: {}", step_name, exc)
        try:
            rc = plot_result_overview_cli([
                "--project-dir", str(cfg.project_dir),
                "--setup-dir", str(cfg.setup_dir),
                "--log-level", cfg.log_level,
                "--mode", "band",
            ], configure_logger=False)
            if isinstance(rc, int) and rc != 0:
                raise RuntimeError(f"plot_result_overview_cli returned {rc}")
        except Exception as exc:
            logger.warning("Result overview plot skipped after step {}: {}", step_name, exc)
        plot_weights_for_csv(wcsv)
        if station_diagnostics_csv is not None and station_diagnostics_csv.is_file():
            try:
                plot_station_diagnostics_for_csv(station_diagnostics_csv)
            except Exception as exc:
                logger.warning("Station diagnostics plot failed after step {}: {}", step_name, exc)
        try:
            plot_setup_ess_timeline(cfg.project_dir)
        except FileNotFoundError:
            pass
    except Exception as exc:
        logger.warning("Setup plotting failed after step {}: {}", step_name, exc)
    finally:
        if reset_logger and reset_logger_func is not None:
            reset_logger_func(cfg.project_dir, cfg.log_level)


def build_fraction_overlay_task(cfg) -> PlotTask:
    return PlotTask(
        name="fraction_overlay",
        func=plot_result_overview_cli,
        args=(
            [
                "--project-dir",
                str(cfg.project_dir),
                "--setup-dir",
                str(cfg.setup_dir),
            ],
        ),
        kwargs={"configure_logger": False},
    )


def custom_overview_needs_benchmark_scores(project_dir: Path) -> bool:
    custom_cfg = Path(project_dir) / "plots.yml"
    if not custom_cfg.is_file():
        return False
    try:
        data = _read_yaml_file(custom_cfg) or {}
    except Exception:
        return False
    panels = data.get("panels") or []
    for entry in panels:
        if isinstance(entry, str):
            panel = entry
        elif isinstance(entry, dict):
            panel = entry.get("panel")
        else:
            continue
        if str(panel or "").strip().lower() in {"scores-crpss", "scores-ner", "scores-zskill"}:
            return True
    return False


def render_project_maps_best_effort(project_dir: Path) -> None:
    if not project_maps_enabled(project_dir):
        logger.info("Project maps skipped: no generated events or custom maps found under {}", project_dir)
        return
    try:
        outputs = render_project_maps(project_dir=project_dir)
        logger.info("Project maps complete -> {} output(s)", len(outputs))
    except ProjectMapRenderError as exc:
        logger.warning("Project maps failed on {} map {}: {}", exc.output_class, exc.recipe_name, exc)
        logger.warning(
            "Rerun all project maps with: {}",
            default_project_maps_rerun_command(project_dir),
        )
        logger.warning(
            "Rerun only this map with: {}",
            default_project_maps_rerun_command(project_dir, recipe_name=exc.recipe_name),
        )
    except Exception as exc:
        logger.warning("Project maps failed: {}", exc)
        logger.warning(
            "Rerun project maps with: {}",
            default_project_maps_rerun_command(project_dir),
        )


def render_project_poster_best_effort(project_dir: Path, *, max_workers: int | None = None) -> None:
    if not poster_profile_enabled(project_dir):
        logger.info("Poster rendering skipped: no poster.yml found under {}", project_dir)
        return
    try:
        outputs = render_poster_profile(project_dir=project_dir, max_workers=max_workers)
        logger.info("Poster rendering complete -> {} output(s)", len(outputs))
    except Exception as exc:
        logger.warning("Poster rendering failed: {}", exc)
        logger.warning("Rerun poster rendering with: {}", default_project_poster_rerun_command(project_dir))


def default_project_report_rerun_command(project_dir: Path) -> str:
    return f"python -m openamundsen_da.methods.viz.reports --project-dir {Path(project_dir)}"


def render_project_report_best_effort(project_dir: Path) -> None:
    try:
        output = build_project_collection_pdf(project_dir=Path(project_dir))
        logger.info("Project report complete -> {}", output)
    except MissingProjectPdfArtifactsError as exc:
        logger.warning("Project report skipped: {}", exc)
        logger.warning(
            "Rerun project report with: {}",
            default_project_report_rerun_command(project_dir),
        )
    except Exception as exc:
        logger.warning("Project report failed: {}", exc)
        logger.warning(
            "Rerun project report with: {}",
            default_project_report_rerun_command(project_dir),
        )


def build_post_run_plot_tasks(
    cfg,
    steps: List[Path],
    *,
    include_fraction_overlay: bool = True,
) -> List[PlotTask]:
    """Build final project-level plot tasks executed after the run completes."""
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
            name="setup_results_swe",
            func=plot_setup_results,
            args=(),
            kwargs={
                "setup_dir": cfg.project_dir,
                "var_col": "swe",
                "mode": "band",
                "resample": "D",
                "resample_agg": "mean",
                "configure_logger": False,
            },
        )
    )
    plot_tasks.append(
        PlotTask(
            name="setup_results_snow_depth",
            func=plot_setup_results,
            args=(),
            kwargs={
                "setup_dir": cfg.project_dir,
                "var_col": "snow_depth",
                "mode": "band",
                "resample": "D",
                "resample_agg": "mean",
                "configure_logger": False,
            },
        )
    )
    if include_fraction_overlay:
        plot_tasks.append(build_fraction_overlay_task(cfg))
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
    for step_dir in steps:
        assim_dir = Path(step_dir) / "assim"
        if not assim_dir.is_dir():
            continue
        for diag_csv in sorted(assim_dir.glob("station_diagnostics_*_*.csv")):
            plot_tasks.append(
                PlotTask(
                    name=f"station_diag:{diag_csv.parent.parent.name}:{diag_csv.stem}",
                    func=plot_station_diagnostics_for_csv,
                    args=(diag_csv,),
                    kwargs={},
                )
            )
    plot_tasks.append(
        PlotTask(
            name="setup_ess_timeline",
            func=plot_setup_ess_timeline,
            args=(cfg.project_dir,),
            kwargs={},
        )
    )
    plot_tasks.append(
        PlotTask(
            name="setup_weights_overview",
            func=plot_setup_weights_overview,
            args=(cfg.project_dir,),
            kwargs={},
        )
    )
    return plot_tasks


__all__ = [
    "PlotTask",
    "aggregate_fraction_envelopes",
    "build_post_run_plot_tasks",
    "custom_overview_needs_benchmark_scores",
    "default_project_report_rerun_command",
    "render_project_maps_best_effort",
    "render_project_poster_best_effort",
    "render_project_report_best_effort",
    "run_live_plots",
    "run_plot_tasks_parallel",
]
