from __future__ import annotations

from types import SimpleNamespace

import pytest

from openamundsen_da.manifests import file_inventory, inventory_digest, write_manifest_atomic
from openamundsen_da.subdomain import leaf_finalization as finalization_mod


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
    assert not finalization_mod.leaf_finalization_manifest_path(
        subdomain.setup_dir
    ).exists()
