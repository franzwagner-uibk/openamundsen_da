from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import write_manifest_atomic
from openamundsen_da.util.runtime_generation import (
    LEGACY_LAYOUT,
    RUNTIME_LAYOUT,
    ensure_runtime_generation,
    load_runtime_generation,
    mapped_runtime_path,
    record_runtime_step_accounting,
    runtime_accounted_totals,
    runtime_generation_manifest_path,
    runtime_generation_root,
)
from openamundsen_da.util import runtime_generation as runtime_generation_mod


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "setup" / "projects" / "project"
    project.mkdir(parents=True)
    (project / "project.yml").write_text("run_mode: single\n", encoding="utf-8")
    return project


def test_fresh_generation_maps_member_artifacts_below_one_root(tmp_path: Path) -> None:
    project = _project(tmp_path)

    manifest = ensure_runtime_generation(project)
    member = project / "steps/step_00/ensembles/prior/member_001"
    member.mkdir(parents=True)

    assert manifest["layout"] == RUNTIME_LAYOUT
    root = runtime_generation_root(project)
    assert root is not None
    assert mapped_runtime_path(member, "meteo") == root / member.relative_to(project) / "meteo"
    assert mapped_runtime_path(member, "results") == root / member.relative_to(project) / "results"


def test_runtime_layout_mapping_reuses_command_local_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project(tmp_path)
    ensure_runtime_generation(project)
    member = project / "steps/step_00/ensembles/prior/member_001"
    member.mkdir(parents=True)
    monkeypatch.setattr(
        runtime_generation_mod,
        "load_runtime_generation",
        lambda _project: (_ for _ in ()).throw(
            AssertionError("mapped paths must not reparse the generation manifest")
        ),
    )

    first = mapped_runtime_path(member, "meteo")
    second = mapped_runtime_path(member, "results")

    assert first is not None
    assert second is not None


def test_existing_member_layout_is_bound_to_legacy_resume(tmp_path: Path) -> None:
    project = _project(tmp_path)
    legacy = project / "steps/step_00/ensembles/prior/member_001/results"
    legacy.mkdir(parents=True)
    (legacy / "output_grids.nc").write_bytes(b"existing")

    manifest = ensure_runtime_generation(project)

    assert manifest["layout"] == LEGACY_LAYOUT
    assert runtime_generation_root(project) is None
    assert mapped_runtime_path(legacy.parent, "results") is None


def test_completed_generation_overwrite_starts_a_new_root(tmp_path: Path) -> None:
    project = _project(tmp_path)
    first = ensure_runtime_generation(project)
    completed = dict(first)
    completed["status"] = "complete"
    write_manifest_atomic(runtime_generation_manifest_path(project), completed)

    reused = ensure_runtime_generation(project, overwrite=False)
    replacement = ensure_runtime_generation(project, overwrite=True)

    assert reused["generation_id"] == first["generation_id"]
    assert replacement["generation"] == 2
    assert replacement["generation_id"] != first["generation_id"]


def test_generation_creation_recovers_only_empty_unowned_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = _project(tmp_path)
    real_write = runtime_generation_mod.write_manifest_atomic

    def interrupted_write(_path: Path, _payload: dict) -> Path:
        raise OSError("injected crash before authority publication")

    monkeypatch.setattr(
        runtime_generation_mod,
        "write_manifest_atomic",
        interrupted_write,
    )
    with pytest.raises(OSError, match="before authority publication"):
        ensure_runtime_generation(project)
    orphans = list((project / ".openamundsen-da/runtime").iterdir())
    assert len(orphans) == 1
    assert not any(orphans[0].iterdir())

    monkeypatch.setattr(runtime_generation_mod, "write_manifest_atomic", real_write)
    manifest = ensure_runtime_generation(project)

    assert manifest["layout"] == RUNTIME_LAYOUT
    assert runtime_generation_root(project) is not None


def test_generation_creation_refuses_nonempty_unowned_root(tmp_path: Path) -> None:
    project = _project(tmp_path)
    orphan = project / ".openamundsen-da/runtime/generation-orphan"
    orphan.mkdir(parents=True)
    (orphan / "unknown.bin").write_bytes(b"unknown")

    with pytest.raises(CleanupSafetyError, match="requires manual inspection"):
        ensure_runtime_generation(project)

    assert (orphan / "unknown.bin").read_bytes() == b"unknown"


def test_runtime_accounting_is_monotonic_and_excludes_durable_components(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    ensure_runtime_generation(project)

    record_runtime_step_accounting(
        project,
        step_name="step_00",
        component_bytes={
            "forcing_bytes": 100,
            "member_grid_bytes": 200,
            "retained_diagnostics_bytes": 900,
        },
        file_counts={
            "forcing_bytes": 2,
            "member_grid_bytes": 3,
            "retained_diagnostics_bytes": 8,
        },
    )
    record_runtime_step_accounting(
        project,
        step_name="step_00",
        component_bytes={"forcing_bytes": 50},
        file_counts={"forcing_bytes": 1},
    )

    assert runtime_accounted_totals(project) == (300, 5)


def test_runtime_manifest_refuses_an_escaping_root(tmp_path: Path) -> None:
    project = _project(tmp_path)
    manifest = ensure_runtime_generation(project)
    manifest["runtime_root"] = "../../outside"
    write_manifest_atomic(runtime_generation_manifest_path(project), manifest)

    with pytest.raises(CleanupSafetyError, match="escapes"):
        load_runtime_generation(project)
