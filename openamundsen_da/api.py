"""Small supported Python interface for single-domain workflows."""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from openamundsen_da.configuration import ProjectConfiguration, load_project_configuration
from openamundsen_da.exceptions import (
    ProjectCleanupError,
    ProjectPreparationError,
    ProjectRenderError,
    ProjectRunError,
)
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    project_run_manifest_path,
    recursive_files,
    workflow_manifest_path,
    write_manifest_atomic,
)
from openamundsen_da.methods.viz.reports import build_project_collection_pdf
from openamundsen_da.observer.satellite_scf import generate_project_from_summary as prepare_scf_observations
from openamundsen_da.observer.satellite_wet_snow_s1 import (
    generate_project_from_summary as prepare_wet_snow_observations,
)
from openamundsen_da.pipeline.cleanup import clean_project_artifacts
from openamundsen_da.pipeline.project_skeleton import create_project_skeleton
from openamundsen_da.pipeline.rendering import render_required_project_outputs
from openamundsen_da.results import CleanupResult, PreparationResult, RenderResult, RunResult, WorkflowStatus
from openamundsen_da.util.perf_monitor import PerfMonitorConfig, capture_perf_snapshot


PREPARATION_SCHEMA_VERSION = 2
RUN_INPUT_SCHEMA_VERSION = 2


def _setup_path(config: ProjectConfiguration, raw: object) -> Path:
    return (config.setup_dir / str(raw)).resolve()


def _preparation_inputs(config: ProjectConfiguration) -> tuple[list[dict], str]:
    files = [config.setup_yaml, config.project_yaml]
    obs = config.project.get("obs")
    if isinstance(obs, dict):
        for section_name, section in obs.items():
            if not isinstance(section, dict):
                continue
            for key in ("summary_csv", "wet_snow_line_diagnostics_csv", "acquisition_manifest"):
                if section.get(key):
                    files.extend(recursive_files(_setup_path(config, section[key])))
            if section.get("dir") and section_name == "stations":
                files.extend(recursive_files(_setup_path(config, section["dir"])))
    inventory = file_inventory(root=config.setup_dir, files=files)
    return inventory, inventory_digest(inventory)


def _configured_roi_grid_paths(config: ProjectConfiguration) -> set[Path]:
    domain = str(config.setup.get("domain", "")).strip()
    resolution_raw = config.setup.get("resolution")
    input_data = config.setup.get("input_data")
    grids = input_data.get("grids") if isinstance(input_data, dict) else None
    if not domain or resolution_raw is None or not isinstance(grids, dict) or not grids.get("dir"):
        return set()
    try:
        resolution_float = float(resolution_raw)
    except (TypeError, ValueError):
        resolution = str(resolution_raw).strip()
    else:
        resolution = str(int(resolution_float)) if resolution_float.is_integer() else str(resolution_raw).strip()
    base = _setup_path(config, grids["dir"]) / f"roi_{domain}_{resolution}"
    return {base.with_suffix(".asc"), base.with_suffix(".prj")}


def _preparation_output_files(config: ProjectConfiguration) -> list[Path]:
    steps_dir = config.project_dir / "steps"
    files = [*sorted(steps_dir.glob("step_*/*.yml")), *sorted(steps_dir.glob("step_*/obs/*.csv"))]
    files.extend(path for path in sorted(_configured_roi_grid_paths(config)) if path.is_file())
    return files


def _validate_preparation_outputs(config: ProjectConfiguration, manifest: dict) -> list[Path]:
    if manifest.get("preparation_schema_version") != PREPARATION_SCHEMA_VERSION:
        raise ProjectRunError(
            "Unsupported preparation manifest contract; rerun preparation with overwrite enabled"
        )
    recorded = manifest.get("outputs")
    if not isinstance(recorded, list) or not isinstance(manifest.get("output_digest"), str):
        raise ProjectRunError("Preparation manifest is missing its output inventory")
    paths = [config.setup_dir / str(entry.get("path", "")) for entry in recorded if isinstance(entry, dict)]
    current = file_inventory(root=config.setup_dir, files=paths)
    if inventory_digest(current) != manifest["output_digest"]:
        raise ProjectRunError("Preparation outputs differ from the completed preparation manifest")
    return paths


def prepare_project(project_dir: str | Path, *, overwrite: bool = False) -> PreparationResult:
    """Create deterministic steps and map configured observation summaries."""
    config = load_project_configuration(project_dir)
    if str(config.project["run_mode"]).lower() != "single":
        raise ProjectPreparationError(
            "prepare_project supports run_mode: single; use the subdomains command tree for subdomain projects"
        )
    manifest_path = workflow_manifest_path(config.project_dir, "preparation")
    inventory, digest = _preparation_inputs(config)
    existing = load_manifest(manifest_path)
    existing_steps = sorted((config.project_dir / "steps").glob("step_*"))
    if existing is not None and existing.get("status") == "success" and not overwrite:
        if existing.get("preparation_schema_version") != PREPARATION_SCHEMA_VERSION:
            raise ProjectPreparationError(
                "Unsupported preparation manifest contract; rerun with overwrite=True"
            )
        if existing.get("input_digest") != digest:
            raise ProjectPreparationError(
                f"Preparation inputs differ from completed manifest {manifest_path}; rerun with overwrite=True"
            )
        recorded_outputs = existing.get("outputs")
        output_paths = [
            config.setup_dir / str(entry.get("path", ""))
            for entry in recorded_outputs
            if isinstance(entry, dict)
        ] if isinstance(recorded_outputs, list) else []
        current_outputs = file_inventory(root=config.setup_dir, files=output_paths)
        if inventory_digest(current_outputs) != existing.get("output_digest"):
            raise ProjectPreparationError(
                f"Preparation outputs differ from completed manifest {manifest_path}; rerun with overwrite=True"
            )
        obs_paths = tuple(
            (config.project_dir / path).resolve()
            for path in existing.get("observation_paths", [])
        )
        return PreparationResult(
            setup_dir=config.setup_dir,
            project_dir=config.project_dir,
            status=WorkflowStatus.REUSED,
            step_dirs=tuple(path.resolve() for path in existing_steps),
            observation_paths=obs_paths,
            manifest_path=manifest_path.resolve(),
        )
    if existing_steps and not overwrite:
        raise ProjectPreparationError(
            f"Project steps exist without a matching completed preparation manifest: {config.project_dir / 'steps'}"
        )
    run_manifest = load_manifest(project_run_manifest_path(config.project_dir))
    if overwrite and run_manifest is not None and run_manifest.get("status") == "success":
        raise ProjectPreparationError(
            f"Completed project is immutable and cannot be prepared again: {project_run_manifest_path(config.project_dir)}"
        )

    manifest = {
        "operation": "prepare-project",
        "preparation_schema_version": PREPARATION_SCHEMA_VERSION,
        "status": "running",
        "input_digest": digest,
        "inputs": inventory,
    }
    write_manifest_atomic(manifest_path, manifest)
    try:
        if overwrite and (config.project_dir / "steps").is_dir():
            shutil.rmtree(config.project_dir / "steps")
        create_project_skeleton(config.setup_dir, config.project_dir, overwrite=overwrite)
        da = config.project["data_assimilation"]
        events = da["assimilation_events"]
        variables = {str(event["variable"]).lower() for event in events}
        obs = config.project["obs"]
        if "scf" in variables:
            section = obs["snowcover"]
            prepare_scf_observations(
                config.project_dir,
                _setup_path(config, section["summary_csv"]),
                product=str(section["product_tag"]),
                overwrite=overwrite,
            )
        if variables & {"wet_snow", "wet_snow_line"}:
            section = obs["wetsnow"]
            prepare_wet_snow_observations(
                config.project_dir,
                _setup_path(config, section["summary_csv"]),
                product=str(section["product_tag"]),
                overwrite=overwrite,
            )
        step_dirs = tuple(path.resolve() for path in sorted((config.project_dir / "steps").glob("step_*")))
        observation_paths = tuple(
            path.resolve()
            for step in step_dirs
            for path in sorted((step / "obs").glob("*.csv"))
        )
        output_inventory = file_inventory(root=config.setup_dir, files=_preparation_output_files(config))
        manifest.update(
            {
                "status": "success",
                "step_dirs": [path.relative_to(config.project_dir).as_posix() for path in step_dirs],
                "observation_paths": [
                    path.relative_to(config.project_dir).as_posix() for path in observation_paths
                ],
                "outputs": output_inventory,
                "output_digest": inventory_digest(output_inventory),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            {
                "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                "error": str(exc),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, ProjectPreparationError):
            raise
        raise ProjectPreparationError(f"Project preparation failed: {exc}") from exc
    return PreparationResult(
        setup_dir=config.setup_dir,
        project_dir=config.project_dir,
        status=WorkflowStatus.COMPLETED,
        step_dirs=step_dirs,
        observation_paths=observation_paths,
        manifest_path=manifest_path.resolve(),
    )


def render_project(project_dir: str | Path, *, max_workers: int | None = None) -> RenderResult:
    """Strictly regenerate configured plots, maps and project report."""
    config = load_project_configuration(project_dir)
    try:
        return render_required_project_outputs(config.project_dir, max_workers=max_workers)
    except Exception as exc:
        if isinstance(exc, ProjectRenderError):
            raise
        raise ProjectRenderError(f"Project rendering failed: {exc}") from exc


def clean_project(project_dir: str | Path, *, apply: bool = False) -> CleanupResult:
    """Preview or apply safe cleanup of package-owned restart artifacts."""
    config = load_project_configuration(project_dir)
    if str(config.project["run_mode"]).lower() != "single":
        raise ProjectCleanupError(
            "clean_project supports run_mode: single; subdomain cleanup is finalized after merge and rendering"
        )
    try:
        return clean_project_artifacts(config.project_dir, apply=apply)
    except Exception as exc:
        if isinstance(exc, ProjectCleanupError):
            raise
        raise ProjectCleanupError(f"Project cleanup failed: {exc}") from exc


def _run_input_inventory(config: ProjectConfiguration, preparation: dict) -> tuple[list[dict], str]:
    files = [config.setup_yaml, config.project_yaml, *sorted(config.project_dir.glob("*.yml"))]
    generated_roi_paths = {path.resolve() for path in _configured_roi_grid_paths(config)}
    input_data = config.setup.get("input_data")
    if isinstance(input_data, dict):
        for name in ("grids", "meteo"):
            section = input_data.get(name)
            if isinstance(section, dict) and section.get("dir"):
                discovered = recursive_files(_setup_path(config, section["dir"]))
                files.extend(path for path in discovered if path.resolve() not in generated_roi_paths)
    obs = config.project.get("obs")
    if isinstance(obs, dict):
        for section in obs.values():
            if isinstance(section, dict) and section.get("dir"):
                files.extend(recursive_files(_setup_path(config, section["dir"])))
            if isinstance(section, dict):
                for key in ("summary_csv", "wet_snow_line_diagnostics_csv", "acquisition_manifest"):
                    if section.get(key):
                        files.extend(recursive_files(_setup_path(config, section[key])))
    files.extend(recursive_files(config.setup_dir / "env"))
    files.append(workflow_manifest_path(config.project_dir, "preparation"))
    files.extend(_validate_preparation_outputs(config, preparation))
    inventory = file_inventory(root=config.setup_dir, files=files)
    return inventory, inventory_digest(inventory)


def _software_version() -> str:
    try:
        return version("openamundsen-da")
    except PackageNotFoundError:
        from openamundsen_da import __version__

        return __version__


def _required_run_outputs(config: ProjectConfiguration) -> tuple[Path, Path]:
    compact = (config.project_dir / "results" / "grids" / "da_output_grids.nc").resolve()
    benchmark = (config.project_dir / "results" / "benchmark" / "manifest.json").resolve()
    perf_csv = (config.project_dir / "results" / "plots" / "perf" / "project_perf_metrics.csv").resolve()
    perf_plot = (config.project_dir / "results" / "plots" / "perf" / "project_perf.png").resolve()
    missing = [path for path in (compact, benchmark, perf_csv, perf_plot) if not path.is_file()]
    if missing:
        raise ProjectRunError("Required run output(s) missing: " + ", ".join(str(path) for path in missing))
    try:
        benchmark_data = json.loads(benchmark.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectRunError(f"Invalid benchmark manifest {benchmark}: {exc}") from exc
    if not isinstance(benchmark_data, dict):
        raise ProjectRunError(f"Benchmark manifest root must be an object: {benchmark}")
    return compact, benchmark


def _member_counts(project_dir: Path) -> tuple[int, int, list[dict]]:
    completed = 0
    skipped = 0
    members: list[dict] = []
    for path in sorted((project_dir / "steps").glob("step_*/ensembles/prior/*/results/member_run.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectRunError(f"Invalid member manifest {path}: {exc}") from exc
        status = str(data.get("status", ""))
        if status == "success":
            completed += 1
        elif status == "skipped":
            skipped += 1
        else:
            raise ProjectRunError(f"Member manifest is not successful: {path} (status={status!r})")
        members.append(
            {
                "path": path.relative_to(project_dir).as_posix(),
                "status": status,
            }
        )
    if not members:
        raise ProjectRunError(f"No member run manifests found under {project_dir / 'steps'}")
    return completed, skipped, members


def _render_data(result: RenderResult, project_dir: Path) -> dict[str, list[str]]:
    return {
        "plots": [path.relative_to(project_dir).as_posix() for path in result.plot_paths],
        "maps": [path.relative_to(project_dir).as_posix() for path in result.map_paths],
        "reports": [path.relative_to(project_dir).as_posix() for path in result.report_paths],
    }


def _cleanup_data(result: CleanupResult, project_dir: Path) -> dict:
    return {
        "deleted_paths": [path.relative_to(project_dir).as_posix() for path in result.deleted_paths],
        "deleted_count": len(result.deleted_paths),
        "eligible_bytes": result.eligible_bytes,
        "freed_bytes": result.freed_bytes,
        "failures": [
            {"path": failure.path.relative_to(project_dir).as_posix(), "error": failure.error}
            for failure in result.failures
        ],
    }


def _result_from_manifest(
    *,
    config: ProjectConfiguration,
    manifest_path: Path,
    manifest: dict,
    status: WorkflowStatus,
) -> RunResult:
    compact, benchmark = _required_run_outputs(config)
    render = manifest.get("render") if isinstance(manifest.get("render"), dict) else {}
    cleanup = manifest.get("cleanup") if isinstance(manifest.get("cleanup"), dict) else {}
    render_result = RenderResult(
        project_dir=config.project_dir,
        status=status,
        plot_paths=tuple((config.project_dir / path).resolve() for path in render.get("plots", [])),
        map_paths=tuple((config.project_dir / path).resolve() for path in render.get("maps", [])),
        report_paths=tuple((config.project_dir / path).resolve() for path in render.get("reports", [])),
    )
    deleted = tuple((config.project_dir / path).resolve() for path in cleanup.get("deleted_paths", []))
    cleanup_result = CleanupResult(
        project_dir=config.project_dir,
        status=WorkflowStatus.APPLIED,
        applied=True,
        eligible_paths=deleted,
        deleted_paths=deleted,
        failures=(),
        eligible_bytes=int(cleanup.get("eligible_bytes", 0)),
        freed_bytes=int(cleanup.get("freed_bytes", 0)),
    )
    return RunResult(
        setup_dir=config.setup_dir,
        project_dir=config.project_dir,
        results_dir=(config.project_dir / "results").resolve(),
        status=status,
        manifest_path=manifest_path.resolve(),
        step_dirs=tuple(path.resolve() for path in sorted((config.project_dir / "steps").glob("step_*"))),
        completed_members=int(manifest.get("completed_members", 0)),
        skipped_members=int(manifest.get("skipped_members", 0)),
        compact_output_path=compact,
        benchmark_manifest_path=benchmark,
        render_result=render_result,
        cleanup_result=cleanup_result,
        duration_seconds=float(manifest.get("duration_seconds", 0.0)),
    )


def run_project(project_dir: str | Path, *, max_workers: int | None = None) -> RunResult:
    """Execute one prepared project with hash-safe resume and atomic finalization."""
    config = load_project_configuration(project_dir)
    if str(config.project["run_mode"]).lower() != "single":
        raise ProjectRunError(
            "run_project supports run_mode: single; use the subdomains command tree for subdomain projects"
        )
    preparation_path = workflow_manifest_path(config.project_dir, "preparation")
    preparation = load_manifest(preparation_path)
    if preparation is None or preparation.get("status") != "success":
        raise ProjectRunError(f"A completed preparation manifest is required: {preparation_path}")

    inputs, digest = _run_input_inventory(config, preparation)
    manifest_path = project_run_manifest_path(config.project_dir)
    existing = load_manifest(manifest_path)
    if existing is not None:
        if existing.get("run_input_schema_version") != RUN_INPUT_SCHEMA_VERSION:
            raise ProjectRunError(f"Unsupported run input contract in {manifest_path}")
        if existing.get("input_digest") != digest:
            raise ProjectRunError(
                f"Run inputs differ from manifest {manifest_path}; completed projects are immutable and "
                "mismatched runs cannot resume"
            )
        if existing.get("status") == "success":
            output_paths = [config.setup_dir / entry["path"] for entry in existing.get("outputs", [])]
            current_outputs = file_inventory(root=config.setup_dir, files=output_paths)
            if inventory_digest(current_outputs) != existing.get("output_digest"):
                raise ProjectRunError(f"Completed run outputs no longer match {manifest_path}")
            return _result_from_manifest(
                config=config,
                manifest_path=manifest_path,
                manifest=existing,
                status=WorkflowStatus.REUSED,
            )
    else:
        results_dir = config.project_dir / "results"
        unmanaged = [path for path in recursive_files(results_dir) if path.name != "run_manifest.json"]
        if unmanaged:
            raise ProjectRunError(
                f"Results exist without a project run manifest: {unmanaged[0]}; move or clean unmanaged outputs first"
            )

    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    manifest = {
        "operation": "run-project",
        "run_input_schema_version": RUN_INPUT_SCHEMA_VERSION,
        "status": "running",
        "started_at": started_at.isoformat(),
        "input_digest": digest,
        "inputs": inputs,
        "provenance": {
            "software_version": _software_version(),
            "image": os.environ.get("OPENAMUNDSEN_DA_IMAGE"),
            "image_digest": os.environ.get("OPENAMUNDSEN_DA_IMAGE_DIGEST"),
        },
        "stages": {
            "execution": "running",
            "render": "pending",
            "cleanup": "pending",
        },
    }
    if existing is not None:
        manifest["resumed_from_status"] = existing.get("status")
    write_manifest_atomic(manifest_path, manifest)
    try:
        from openamundsen_da.pipeline.project import OrchestratorConfig, run_project as execute_project

        render_result = execute_project(
            OrchestratorConfig(
                project_dir=config.project_dir,
                setup_dir=config.setup_dir,
                max_workers=max_workers,
                plot_workers=max_workers,
                overwrite=False,
                monitor_perf=True,
            )
        )
        compact, benchmark = _required_run_outputs(config)
        completed, skipped, members = _member_counts(config.project_dir)
        if not isinstance(render_result, RenderResult):
            raise ProjectRunError("Internal project execution did not return a RenderResult")
        if not render_result.report_paths or any(not path.is_file() for path in render_result.report_paths):
            raise ProjectRunError("Configured project report validation failed")
        manifest["stages"].update({"execution": "success", "render": "success", "cleanup": "running"})
        manifest["members"] = members
        write_manifest_atomic(manifest_path, manifest)

        cleanup_result = clean_project_artifacts(config.project_dir, apply=True)
        if cleanup_result.failures:
            raise ProjectRunError(
                f"Restart-state cleanup failed for {len(cleanup_result.failures)} artifact(s)"
            )
        if capture_perf_snapshot(
            PerfMonitorConfig(project_dir=config.project_dir, run_start=started_at)
        ):
            for report_path in render_result.report_paths:
                build_project_collection_pdf(project_dir=config.project_dir, output=report_path)
        output_files = [
            path
            for path in recursive_files(config.project_dir / "results")
            if path.resolve() != manifest_path.resolve()
            and (config.project_dir / "results" / "plots" / "perf").resolve() not in path.resolve().parents
        ]
        outputs = file_inventory(root=config.setup_dir, files=output_files)
        duration = time.monotonic() - started
        manifest.update(
            {
                "status": "success",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": duration,
                "completed_members": completed,
                "skipped_members": skipped,
                "compact_output": compact.relative_to(config.project_dir).as_posix(),
                "benchmark_manifest": benchmark.relative_to(config.project_dir).as_posix(),
                "render": _render_data(render_result, config.project_dir),
                "cleanup": _cleanup_data(cleanup_result, config.project_dir),
                "outputs": outputs,
                "output_digest": inventory_digest(outputs),
                "performance_outputs": [
                    "results/plots/perf/project_perf_metrics.csv",
                    "results/plots/perf/project_perf.png",
                ],
            }
        )
        manifest["stages"]["cleanup"] = "success"
        write_manifest_atomic(manifest_path, manifest)
    except BaseException as exc:
        terminal_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        for stage, stage_status in manifest["stages"].items():
            if stage_status == "running":
                manifest["stages"][stage] = terminal_status
        manifest.update(
            {
                "status": terminal_status,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started,
                "error": str(exc),
            }
        )
        write_manifest_atomic(manifest_path, manifest)
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, ProjectRunError):
            raise
        raise ProjectRunError(f"Project run failed: {exc}") from exc
    return _result_from_manifest(
        config=config,
        manifest_path=manifest_path,
        manifest=manifest,
        status=WorkflowStatus.COMPLETED,
    )


__all__ = ["clean_project", "prepare_project", "render_project", "run_project"]
