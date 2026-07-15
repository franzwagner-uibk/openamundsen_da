"""Strict top-level rendering for the DA subdomain workflow."""

from __future__ import annotations

from pathlib import Path

from openamundsen_da.exceptions import ProjectRenderError
from openamundsen_da.methods.viz.maps import project_maps_enabled, render_project_maps
from openamundsen_da.methods.viz.reports import build_project_collection_pdf
from openamundsen_da.results import RenderResult, WorkflowStatus
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.subdomain.merge import cleanup_compact_grid_artifacts
from openamundsen_da.subdomain.report import write_subdomain_reports
from openamundsen_da.subdomain.status import save_stage, terminal_status
from openamundsen_da.util.run_mode import ensure_run_mode


def render_subdomain_outputs(
    project_dir: str | Path,
    *,
    max_workers: int | None = None,
) -> RenderResult:
    """Render validated parent-level tables, maps and report after a DA merge."""
    project_dir = Path(project_dir).resolve()
    ensure_run_mode(project_dir, expected="subdomain", write_if_missing=False)
    manifest_path = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest = SubdomainManifest.load(manifest_path)
    if manifest.project_dir.resolve() != project_dir:
        raise ProjectRenderError(
            f"Subdomain manifest project does not match requested project: {manifest.project_dir}"
        )

    failed = [
        sid
        for sid, subdomain in sorted(manifest.subdomains.items())
        if str(subdomain.status).lower() != "success"
    ]
    if failed:
        raise ProjectRenderError(
            "Cannot render an incomplete subdomain workflow; unsuccessful subdomains: "
            + ", ".join(failed)
        )

    merged_grid = project_dir / "results" / "grids" / "da_output_grids.nc"
    if not merged_grid.is_file():
        raise ProjectRenderError(
            f"Merged compact DA output is required before rendering: {merged_grid}"
        )

    save_stage(manifest, manifest_path, "render", "running")
    try:
        write_subdomain_reports(manifest_path=manifest_path, out_dir=project_dir / "results")
        if project_maps_enabled(project_dir):
            render_project_maps(project_dir=project_dir, max_workers=max_workers)
        report = build_project_collection_pdf(project_dir=project_dir).resolve()
    except BaseException as exc:
        current = SubdomainManifest.load(manifest_path)
        save_stage(
            current,
            manifest_path,
            "render",
            terminal_status(exc),
            error=str(exc),
        )
        if isinstance(exc, KeyboardInterrupt):
            raise
        if isinstance(exc, ProjectRenderError):
            raise
        raise ProjectRenderError(f"Subdomain rendering failed: {exc}") from exc

    results_dir = project_dir / "results"
    result = RenderResult(
        project_dir=project_dir,
        status=WorkflowStatus.COMPLETED,
        plot_paths=tuple(path.resolve() for path in sorted((results_dir / "plots").rglob("*.png"))),
        map_paths=tuple(path.resolve() for path in sorted((results_dir / "maps").rglob("*.png"))),
        report_paths=(report,),
    )
    current = SubdomainManifest.load(manifest_path)
    render_outputs = (*result.plot_paths, *result.map_paths, *result.report_paths)
    save_stage(current, manifest_path, "render", "completed", outputs=render_outputs)
    try:
        cleanup_compact_grid_artifacts(manifest_path=manifest_path)
    except Exception as exc:
        raise ProjectRenderError(f"Subdomain compact cleanup failed: {exc}") from exc
    return result


__all__ = ["render_subdomain_outputs"]
