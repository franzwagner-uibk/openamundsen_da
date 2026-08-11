"""Strict shared rendering stage for single-domain and subdomain runs."""

from __future__ import annotations

from pathlib import Path

from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    load_manifest,
    recursive_files,
    write_manifest_atomic,
)
from openamundsen_da.methods.viz.maps import project_maps_enabled, render_project_maps
from openamundsen_da.methods.viz.plots.runner import render_project_plots
from openamundsen_da.methods.viz.reports import build_project_collection_pdf
from openamundsen_da.results import RenderResult, WorkflowStatus


RENDER_COMPLETION_MANIFEST = "render_completion_manifest.json"


def render_completion_manifest_path(project_dir: str | Path) -> Path:
    """Return stable evidence for the last successful configured render."""
    return Path(project_dir).resolve() / "results" / RENDER_COMPLETION_MANIFEST


def validate_render_completion(project_dir: str | Path) -> Path:
    """Validate durable evidence that this exact project completed rendering."""
    project_dir = Path(project_dir).resolve()
    path = render_completion_manifest_path(project_dir)
    try:
        manifest = load_manifest(path)
    except ValueError as exc:
        raise ValueError(f"Invalid render completion evidence: {path}") from exc
    if manifest is None:
        raise FileNotFoundError(f"Render completion evidence is missing: {path}")
    if manifest.get("contract") != "project-render-v1":
        raise ValueError(f"Unsupported render completion contract: {path}")
    if manifest.get("status") != "success":
        raise ValueError(f"Render completion is not successful: {path}")
    if Path(str(manifest.get("project_dir", ""))).resolve() != project_dir:
        raise ValueError(f"Render completion project identity changed: {path}")
    return path


def _record_render_completion(project_dir: Path, *, report: Path) -> Path:
    """Durably attest a completed render without binding mutable report bytes."""
    if not report.is_file() or report.is_symlink():
        raise FileNotFoundError(f"Rendered project report is missing: {report}")
    render_files = [
        path
        for root in (
            project_dir / "results" / "plots",
            project_dir / "results" / "maps",
        )
        for path in recursive_files(root)
        if (project_dir / "results" / "plots" / "perf") not in path.parents
    ]
    render_files.append(report)
    inventory = file_inventory(root=project_dir, files=render_files)
    return write_manifest_atomic(
        render_completion_manifest_path(project_dir),
        {
            "contract": "project-render-v1",
            "status": "success",
            "project_dir": str(project_dir),
            "output_count": len(inventory),
            "output_inventory_sha256": inventory_digest(inventory),
        },
    )


def render_required_project_outputs(
    project_dir: Path,
    *,
    max_workers: int | None,
) -> RenderResult:
    """Render and validate every configured public project output."""
    project_dir = Path(project_dir).resolve()
    render_project_plots(
        project_dir=project_dir,
        plot_workers=max_workers,
        max_workers=max_workers,
    )
    if project_maps_enabled(project_dir):
        render_project_maps(project_dir=project_dir, max_workers=max_workers)
    report = build_project_collection_pdf(project_dir=project_dir).resolve()
    _record_render_completion(project_dir, report=report)
    results_dir = project_dir / "results"
    return RenderResult(
        project_dir=project_dir,
        status=WorkflowStatus.COMPLETED,
        plot_paths=tuple(path.resolve() for path in sorted((results_dir / "plots").rglob("*.png"))),
        map_paths=tuple(path.resolve() for path in sorted((results_dir / "maps").rglob("*.png"))),
        report_paths=(report,),
    )


__all__ = [
    "RENDER_COMPLETION_MANIFEST",
    "render_completion_manifest_path",
    "render_required_project_outputs",
    "validate_render_completion",
]
