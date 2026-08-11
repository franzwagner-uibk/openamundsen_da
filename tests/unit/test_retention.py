from __future__ import annotations

from pathlib import Path

import pytest

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import hash_json
from openamundsen_da.util import retention as retention_mod
from openamundsen_da.util.retention import (
    apply_retention_batch,
    completed_retention_paths,
    validate_retained_consumers,
)


def _apply_retention(project: Path, **kwargs):
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    consumer.parent.mkdir(parents=True, exist_ok=True)
    consumer.write_bytes(b"accepted")
    producer.write_text('{"status": "success"}\n', encoding="utf-8")
    return apply_retention_batch(
        project,
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
        **kwargs,
    )


def test_retention_batch_is_contained_recorded_and_idempotent(tmp_path: Path) -> None:
    project = tmp_path / "setup" / "projects" / "demo"
    artifact = project / "steps" / "step_00" / "member.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"payload")

    batch = _apply_retention(
        project,
        artifact_class="member_forcing",
        paths=[artifact],
        final_consumer="propagation",
        regeneration_recipe="rebuild from setup forcing and keyed RNG ledger",
    )

    assert batch["status"] == "complete"
    assert batch["bytes"] == 7
    assert batch["producer_digest"] != batch["inventory_sha256"]
    assert batch["producer_digest"] == hash_json(
        {"file_inventory": batch["producer_manifest_inventory"]}
    )
    assert batch["producer_manifest_inventory"][0]["path"] == "results/run_manifest.json"
    assert not artifact.exists()
    assert completed_retention_paths(project) == {"steps/step_00/member.bin"}
    assert _apply_retention(
        project,
        artifact_class="member_forcing",
        paths=[artifact],
        final_consumer="propagation",
        regeneration_recipe="rebuild from setup forcing and keyed RNG ledger",
    )["status"] == "complete"


def test_completed_retention_consumer_is_revalidated_on_leaf_resume(tmp_path: Path) -> None:
    project = tmp_path / "project"
    artifact = project / "step" / "forcing.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"raw")
    _apply_retention(
        project,
        artifact_class="forcing",
        paths=[artifact],
        final_consumer="compact forcing",
        regeneration_recipe="regenerate",
    )

    assert validate_retained_consumers(project, require_complete=True) == ("0001",)
    (project / "results" / "accepted.nc").write_bytes(b"corrupt")
    with pytest.raises(CleanupSafetyError, match="retained consumer changed"):
        validate_retained_consumers(project, require_complete=True)


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
        _apply_retention(
            project,
            artifact_class="state",
            paths=[first, second],
            final_consumer="successor checkpoint",
            regeneration_recipe="rerun predecessor propagation",
        )
    monkeypatch.setattr(retention_mod, "_unlink_path", real_unlink)
    batch = _apply_retention(
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
        _apply_retention(
            project,
            artifact_class="state",
            paths=[first, second],
            final_consumer="successor checkpoint",
            regeneration_recipe="rerun predecessor propagation",
        )
    monkeypatch.setattr(retention_mod, "_unlink_path", real_unlink)

    batch = _apply_retention(
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
        _apply_retention(
            project,
            artifact_class="state",
            paths=[outside],
            final_consumer="test",
            regeneration_recipe="none",
        )


def test_completed_batch_treats_recreated_path_as_new_generation(tmp_path: Path) -> None:
    project = tmp_path / "project"
    artifact = project / "step" / "forcing.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"generation-one")
    first = _apply_retention(
        project,
        artifact_class="forcing",
        paths=[artifact],
        final_consumer="propagation",
        regeneration_recipe="regenerate",
    )
    artifact.write_bytes(b"generation-two")

    second = _apply_retention(
        project,
        artifact_class="forcing",
        paths=[artifact],
        final_consumer="propagation",
        regeneration_recipe="regenerate",
    )

    assert first["batch_id"] == "0001"
    assert second["batch_id"] == "0002"
    assert not artifact.exists()


def test_planned_retry_refuses_modified_generation(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    artifact = project / "state.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"original")

    monkeypatch.setattr(retention_mod, "_unlink_path", lambda _path: (_ for _ in ()).throw(OSError("stop")))
    with pytest.raises(CleanupSafetyError, match="Retention cleanup failed"):
        _apply_retention(
            project,
            artifact_class="state",
            paths=[artifact],
            final_consumer="successor",
            regeneration_recipe="rerun",
        )
    artifact.write_bytes(b"modified")

    with pytest.raises(CleanupSafetyError, match="changed after planning"):
        _apply_retention(
            project,
            artifact_class="state",
            paths=[artifact],
            final_consumer="successor",
            regeneration_recipe="rerun",
        )


def test_interrupted_cleanup_revalidates_consumer_before_resumed_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    first = project / "a.bin"
    second = project / "b.bin"
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    consumer.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    consumer.write_bytes(b"valid")
    producer.write_text('{"status": "success"}\n', encoding="utf-8")
    real_unlink = retention_mod._unlink_path
    calls = 0

    def interrupted(path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("stop")
        real_unlink(path)

    monkeypatch.setattr(retention_mod, "_unlink_path", interrupted)
    with pytest.raises(CleanupSafetyError, match="Retention cleanup failed"):
        apply_retention_batch(
            project,
            artifact_class="member_grid",
            paths=(first, second),
            final_consumer="compact grid",
            regeneration_recipe="rerun",
            retained_consumers=(consumer,),
            producer_manifests=(producer,),
        )
    consumer.write_bytes(b"corrupt")
    monkeypatch.setattr(retention_mod, "_unlink_path", real_unlink)

    with pytest.raises(CleanupSafetyError, match="retained consumer changed"):
        apply_retention_batch(
            project,
            artifact_class="member_grid",
            paths=(second,),
            final_consumer="compact grid",
            regeneration_recipe="rerun",
            retained_consumers=(consumer,),
            producer_manifests=(producer,),
        )
    assert second.is_file()
