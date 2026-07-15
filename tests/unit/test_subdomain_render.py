from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.exceptions import ProjectRenderError
from openamundsen_da.results import WorkflowStatus
from openamundsen_da.subdomain import render as render_mod


def _manifest(project_dir: Path, *, status: str = "success") -> SimpleNamespace:
    return SimpleNamespace(
        project_dir=project_dir,
        subdomains={"sd_01": SimpleNamespace(status=status)},
    )


def test_render_subdomain_outputs_runs_parent_level_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "winter"
    merged = project_dir / "results" / "grids" / "da_output_grids.nc"
    merged.parent.mkdir(parents=True)
    merged.write_bytes(b"netcdf")
    calls: list[str] = []

    monkeypatch.setattr(render_mod, "ensure_run_mode", lambda *args, **kwargs: calls.append("mode"))
    monkeypatch.setattr(
        render_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: _manifest(project_dir.resolve())),
    )
    monkeypatch.setattr(
        render_mod,
        "write_subdomain_reports",
        lambda **kwargs: calls.append("tables"),
    )
    monkeypatch.setattr(render_mod, "project_maps_enabled", lambda _project: True)

    def fake_maps(**_kwargs) -> list[Path]:
        calls.append("maps")
        path = project_dir / "results" / "maps" / "overview.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"map")
        return [path]

    def fake_report(**_kwargs) -> Path:
        calls.append("report")
        path = project_dir / "results" / "reports" / "project_report.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"pdf")
        return path

    monkeypatch.setattr(render_mod, "render_project_maps", fake_maps)
    monkeypatch.setattr(render_mod, "build_project_collection_pdf", fake_report)

    result = render_mod.render_subdomain_outputs(project_dir, max_workers=2)

    assert calls == ["mode", "tables", "maps", "report"]
    assert result.status is WorkflowStatus.COMPLETED
    assert [path.name for path in result.map_paths] == ["overview.png"]
    assert [path.name for path in result.report_paths] == ["project_report.pdf"]


def test_render_subdomain_outputs_rejects_incomplete_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_dir = tmp_path / "setup" / "projects" / "winter"
    project_dir.mkdir(parents=True)
    monkeypatch.setattr(render_mod, "ensure_run_mode", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        render_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: _manifest(project_dir.resolve(), status="failed")),
    )

    with pytest.raises(ProjectRenderError, match="unsuccessful subdomains: sd_01"):
        render_mod.render_subdomain_outputs(project_dir)
