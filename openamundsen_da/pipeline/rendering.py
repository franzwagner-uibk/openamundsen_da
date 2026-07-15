"""Strict shared rendering stage for single-domain and subdomain runs."""

from __future__ import annotations

from pathlib import Path

from openamundsen_da.methods.viz.maps import project_maps_enabled, render_project_maps
from openamundsen_da.methods.viz.plots.runner import render_project_plots
from openamundsen_da.methods.viz.reports import build_project_collection_pdf
from openamundsen_da.results import RenderResult, WorkflowStatus


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
    results_dir = project_dir / "results"
    return RenderResult(
        project_dir=project_dir,
        status=WorkflowStatus.COMPLETED,
        plot_paths=tuple(path.resolve() for path in sorted((results_dir / "plots").rglob("*.png"))),
        map_paths=tuple(path.resolve() for path in sorted((results_dir / "maps").rglob("*.png"))),
        report_paths=(report,),
    )


__all__ = ["render_required_project_outputs"]
