from __future__ import annotations

import json
from pathlib import Path

import pytest

from openamundsen_da.manifests import write_manifest_atomic
from openamundsen_da.pipeline import rendering as rendering_mod


def test_required_render_writes_stable_completion_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"

    def render_plots(**_kwargs):
        plot = project / "results" / "plots" / "results" / "overview.png"
        plot.parent.mkdir(parents=True)
        plot.write_bytes(b"plot")

    def build_report(**_kwargs):
        report = project / "results" / "reports" / "project_report.pdf"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"%PDF-rendered")
        return report

    monkeypatch.setattr(rendering_mod, "render_project_plots", render_plots)
    monkeypatch.setattr(rendering_mod, "project_maps_enabled", lambda _project: False)
    monkeypatch.setattr(rendering_mod, "build_project_collection_pdf", build_report)

    result = rendering_mod.render_required_project_outputs(project, max_workers=1)

    evidence = rendering_mod.render_completion_manifest_path(project)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["output_count"] == 2
    assert rendering_mod.validate_render_completion(project) == evidence
    assert result.report_paths[0].is_file()


def test_render_evidence_excludes_mutable_performance_output_and_remains_stable(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    report = project / "results" / "reports" / "project_report.pdf"
    plot = project / "results" / "plots" / "results" / "overview.png"
    perf = project / "results" / "plots" / "perf" / "project_perf.png"
    for path, payload in (
        (report, b"%PDF-initial"),
        (plot, b"plot"),
        (perf, b"live performance"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    evidence = rendering_mod._record_render_completion(project, report=report)
    before = evidence.read_bytes()
    payload = json.loads(before)
    report.write_bytes(b"%PDF-final performance refresh")
    perf.write_bytes(b"new live performance")

    assert payload["output_count"] == 2
    assert evidence.read_bytes() == before


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"contract": "wrong", "status": "success"}, "Unsupported"),
        ({"contract": "project-render-v1", "status": "failed"}, "not successful"),
    ],
)
def test_render_completion_rejects_invalid_evidence(
    tmp_path: Path,
    payload: dict,
    message: str,
) -> None:
    project = tmp_path / "project"
    write_manifest_atomic(
        rendering_mod.render_completion_manifest_path(project),
        {**payload, "project_dir": str(project.resolve())},
    )

    with pytest.raises(ValueError, match=message):
        rendering_mod.validate_render_completion(project)
