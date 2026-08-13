from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from openamundsen_da.manifests import file_inventory, inventory_digest, write_manifest_atomic
from openamundsen_da.results import WorkflowStatus
from openamundsen_da.subdomain import leaf_finalization as finalization_mod
from openamundsen_da.subdomain import render as render_mod


def _subdomain(tmp_path, *, retention: str = "compact"):
    setup = tmp_path / "leaf"
    project = setup / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "demo.yml").write_text(
        "data_assimilation:\n"
        "  output:\n"
        f"    retention: {retention}\n",
        encoding="utf-8",
    )
    return SimpleNamespace(id="S1", setup_dir=setup, project_dir=project)


def test_leaf_finalization_is_durable_before_cleanup_and_recovers_planned_state(
    tmp_path,
    monkeypatch,
):
    subdomain = _subdomain(tmp_path)
    support = subdomain.project_dir / "results" / "grids" / "accepted.nc"
    support.parent.mkdir(parents=True)
    support.write_bytes(b"accepted")
    calls: list[str] = []
    monkeypatch.setattr(
        finalization_mod,
        "_leaf_parent_support_files",
        lambda _subdomain: (support,),
    )
    monkeypatch.setattr(
        finalization_mod,
        "clean_project_artifacts",
        lambda *_args, **_kwargs: (
            calls.append("cleanup")
            or SimpleNamespace(failures=(), deleted_paths=(), freed_bytes=0)
        ),
    )
    monkeypatch.setattr(
        finalization_mod,
        "validate_retained_consumers",
        lambda *_args, **_kwargs: (),
    )
    inventory = file_inventory(root=subdomain.setup_dir, files=(support,))
    write_manifest_atomic(
        finalization_mod.leaf_finalization_manifest_path(subdomain.setup_dir),
        {
            "contract": finalization_mod.LEAF_FINALIZATION_CONTRACT,
            "status": "planned",
            "project_dir": str(subdomain.project_dir.resolve()),
            "retention": "compact",
            "retained_support": inventory,
            "retained_support_sha256": inventory_digest(inventory),
        },
    )

    completed = finalization_mod.finalize_leaf(subdomain, resume=True)

    assert calls == ["cleanup"]
    assert completed["status"] == "success"
    assert completed["retained_support"] == inventory


def test_leaf_resume_rejects_changed_retained_support(tmp_path, monkeypatch):
    subdomain = _subdomain(tmp_path)
    support = subdomain.project_dir / "results" / "accepted.nc"
    support.parent.mkdir(parents=True)
    support.write_bytes(b"accepted")
    inventory = file_inventory(root=subdomain.setup_dir, files=(support,))
    write_manifest_atomic(
        finalization_mod.leaf_finalization_manifest_path(subdomain.setup_dir),
        {
            "contract": finalization_mod.LEAF_FINALIZATION_CONTRACT,
            "status": "success",
            "project_dir": str(subdomain.project_dir.resolve()),
            "retention": "compact",
            "retained_support": inventory,
            "retained_support_sha256": inventory_digest(inventory),
        },
    )
    support.write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="changed after finalization"):
        finalization_mod.finalize_leaf(subdomain, resume=True)


def test_full_retention_leaf_finalization_does_not_remove_raw_artifacts(
    tmp_path,
    monkeypatch,
):
    subdomain = _subdomain(tmp_path, retention="full")
    support = subdomain.project_dir / "results" / "accepted.nc"
    raw = subdomain.project_dir / "steps" / "step_00" / "member.bin"
    support.parent.mkdir(parents=True)
    raw.parent.mkdir(parents=True)
    support.write_bytes(b"accepted")
    raw.write_bytes(b"raw")
    monkeypatch.setattr(
        finalization_mod,
        "_leaf_parent_support_files",
        lambda _subdomain: (support,),
    )

    completed = finalization_mod.finalize_leaf(subdomain)

    assert completed["status"] == "success"
    assert completed["retention"] == "full"
    assert raw.read_bytes() == b"raw"
    manifest = finalization_mod.load_manifest(
        finalization_mod.leaf_finalization_manifest_path(subdomain.setup_dir)
    )
    assert manifest is not None
    assert manifest["status"] == "success"
    assert manifest["cleanup_deleted_files"] == 0


def test_full_retention_scf_leaf_uses_raw_render_support_without_map_archive(
    tmp_path,
    monkeypatch,
):
    subdomain = _subdomain(tmp_path, retention="full")
    (subdomain.project_dir / "demo.yml").write_text(
        "data_assimilation:\n"
        "  assimilation_events:\n"
        "    - {date: 2023-01-07, variable: scf, product: EURAC}\n"
        "  output:\n"
        "    retention: full\n",
        encoding="utf-8",
    )
    compact_grid = subdomain.project_dir / "results" / "grids" / "da_output_grids.nc"
    compact_grid.parent.mkdir(parents=True)
    compact_grid.write_bytes(b"validated summary")
    report = subdomain.project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"%PDF-accepted leaf render")
    render_evidence = (
        subdomain.project_dir / "results" / "render_completion_manifest.json"
    )
    write_manifest_atomic(
        render_evidence,
        {
            "contract": "project-render-v1",
            "status": "success",
            "project_dir": str(subdomain.project_dir.resolve()),
        },
    )
    weights = (
        subdomain.project_dir
        / "steps"
        / "step_01"
        / "assim"
        / "weights_scf_20230107.csv"
    )
    weights.parent.mkdir(parents=True)
    weights.write_text("member,weight\n1,1\n", encoding="utf-8")
    raw_grid = (
        subdomain.project_dir
        / "steps"
        / "step_01"
        / "ensembles"
        / "prior"
        / "member_001"
        / "results"
        / "output_grids.nc"
    )
    raw_grid.parent.mkdir(parents=True)
    raw_grid.write_bytes(b"raw full-retention grid")
    validated: list[bool] = []

    def validate_summary(**_kwargs):
        validated.append(raw_grid.is_file())
        return compact_grid

    monkeypatch.setattr(
        finalization_mod,
        "validate_compact_output_file",
        validate_summary,
    )
    raw_map_validations: list[bool] = []

    def rebuild_raw_map_support(_project_dir):
        raw_map_validations.append(raw_grid.is_file())
        return (
            [datetime(2023, 1, 7)],
            {
                "scf_open_loop_binary": [object()],
                "scf_prior_probability": [object()],
                "scf_posterior_probability": [object()],
            },
            object(),
        )

    monkeypatch.setattr(
        finalization_mod,
        "project_da_map_support_fields",
        rebuild_raw_map_support,
    )

    completed = finalization_mod.finalize_leaf(subdomain)

    assert completed["status"] == "success"
    assert completed["retention"] == "full"
    assert validated == [True]
    assert raw_map_validations == [True]
    assert raw_grid.is_file()
    assert report.is_file()
    assert not (subdomain.project_dir / "results" / "grids" / "da_map_support.nc").exists()

    parent_project = tmp_path / "parent" / "projects" / "demo"
    parent_grid = parent_project / "results" / "grids" / "da_output_grids.nc"
    parent_grid.parent.mkdir(parents=True)
    parent_grid.write_bytes(b"merged compact grid")
    (parent_project / "demo.yml").write_text("data_assimilation: {}\n", encoding="utf-8")
    parent_manifest = SimpleNamespace(
        project_dir=parent_project.resolve(),
        subdomains={subdomain.id: SimpleNamespace(status="success")},
        grid_rows=1,
        grid_cols=1,
    )
    calls: list[str] = []
    monkeypatch.setattr(render_mod, "ensure_run_mode", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        render_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: parent_manifest),
    )
    monkeypatch.setattr(render_mod, "resolve_subdomain_event_plan", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(render_mod, "save_stage", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        render_mod,
        "write_subdomain_reports",
        lambda **_kwargs: calls.append("tables"),
    )
    monkeypatch.setattr(render_mod, "project_maps_enabled", lambda _project: True)
    monkeypatch.setattr(
        render_mod,
        "render_project_maps",
        lambda **_kwargs: calls.append("maps") or [],
    )

    def build_parent_report(**_kwargs):
        calls.append("report")
        path = parent_project / "results" / "reports" / "project_report.pdf"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"%PDF-parent")
        return path

    monkeypatch.setattr(render_mod, "build_project_collection_pdf", build_parent_report)
    monkeypatch.setattr(
        render_mod,
        "cleanup_compact_grid_artifacts",
        lambda **_kwargs: calls.append("cleanup") or ([], 0),
    )

    rendered = render_mod.render_subdomain_outputs(parent_project)

    assert rendered.status is WorkflowStatus.COMPLETED
    assert calls == ["tables", "maps", "report", "cleanup"]
    assert raw_grid.is_file()
