from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.util import retention as retention_mod
from openamundsen_da.util.retention import apply_retention_batch, completed_retention_paths


def test_retention_batch_is_contained_recorded_and_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    artifact = project / "steps" / "step_00" / "member.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"payload")

    batch = apply_retention_batch(
        project,
        artifact_class="member_forcing",
        paths=[artifact],
        final_consumer="propagation",
        regeneration_recipe="rebuild from setup forcing and keyed RNG ledger",
    )

    assert batch["status"] == "complete"
    assert batch["bytes"] == 7
    assert batch["producer_digest"] == batch["inventory_sha256"]
    assert not artifact.exists()
    assert completed_retention_paths(project) == {"steps/step_00/member.bin"}
    assert apply_retention_batch(
        project,
        artifact_class="member_forcing",
        paths=[artifact],
        final_consumer="propagation",
        regeneration_recipe="rebuild from setup forcing and keyed RNG ledger",
    )["status"] == "complete"


def test_retention_planned_batch_recovers_after_interrupted_delete(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    first = project / "a.bin"
    second = project / "b.bin"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    calls = 0
    real_unlink = retention_mod._unlink_path

    def interrupted(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected interruption")
        real_unlink(path)

    monkeypatch.setattr(retention_mod, "_unlink_path", interrupted)
    with pytest.raises(CleanupSafetyError, match="Retention cleanup failed"):
        apply_retention_batch(
            project,
            artifact_class="state",
            paths=[first, second],
            final_consumer="successor checkpoint",
            regeneration_recipe="rerun predecessor propagation",
        )
    monkeypatch.setattr(retention_mod, "_unlink_path", real_unlink)
    batch = apply_retention_batch(
        project,
        artifact_class="state",
        paths=[first, second],
        final_consumer="successor checkpoint",
        regeneration_recipe="rerun predecessor propagation",
    )
    assert batch["status"] == "complete"
    assert not second.exists()


def test_retention_planned_batch_matches_only_remaining_paths(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    first = project / "a.bin"
    second = project / "b.bin"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    real_unlink = retention_mod._unlink_path
    calls = 0

    def interrupted(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected interruption")
        real_unlink(path)

    monkeypatch.setattr(retention_mod, "_unlink_path", interrupted)
    with pytest.raises(CleanupSafetyError):
        apply_retention_batch(
            project,
            artifact_class="state",
            paths=[first, second],
            final_consumer="successor checkpoint",
            regeneration_recipe="rerun predecessor propagation",
        )
    monkeypatch.setattr(retention_mod, "_unlink_path", real_unlink)

    batch = apply_retention_batch(
        project,
        artifact_class="state",
        paths=[second],
        final_consumer="successor checkpoint",
        regeneration_recipe="rerun predecessor propagation",
    )

    assert batch["batch_id"] == "0001"
    assert batch["status"] == "complete"


def test_retention_refuses_paths_outside_project(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"x")
    with pytest.raises(CleanupSafetyError, match="escapes project root"):
        apply_retention_batch(
            project,
            artifact_class="state",
            paths=[outside],
            final_consumer="test",
            regeneration_recipe="none",
        )
