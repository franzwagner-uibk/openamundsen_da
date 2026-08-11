from __future__ import annotations

import json
from pathlib import Path

import pytest

from openamundsen_da.exceptions import CleanupSafetyError
from openamundsen_da.manifests import hash_json
from openamundsen_da.util import retention as retention_mod
from openamundsen_da.util.retention import (
    apply_retention_batch,
    complete_retention_generation,
    completed_retention_paths,
    start_retention_generation,
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
    assert first["generation"] == 1
    assert second["generation"] == 2
    assert not artifact.exists()


def test_overwrite_supersedes_old_consumer_generation_but_preserves_audit(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    artifact = project / "step" / "forcing.csv"
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    artifact.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    artifact.write_bytes(b"generation-one-raw")
    consumer.write_bytes(b"generation-one-compact")
    producer.write_text('{"run": 1, "status": "success"}\n', encoding="utf-8")
    first = apply_retention_batch(
        project,
        artifact_class="forcing",
        paths=(artifact,),
        final_consumer="compact forcing",
        regeneration_recipe="rerun",
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
    )

    artifact.write_bytes(b"generation-two-raw")
    consumer.write_bytes(b"generation-two-compact")
    producer.write_text('{"run": 2, "status": "success"}\n', encoding="utf-8")
    second = apply_retention_batch(
        project,
        artifact_class="forcing",
        paths=(artifact,),
        final_consumer="compact forcing",
        regeneration_recipe="rerun",
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
    )

    assert validate_retained_consumers(project, require_complete=True) == ("0002",)
    ledger = json.loads(retention_mod.retention_manifest_path(project).read_text())
    assert ledger["active_generation"] == 2
    assert [item["status"] for item in ledger["generations"]] == [
        "superseded",
        "complete",
    ]
    assert ledger["batches"][0]["superseded_by_generation"] == 2
    assert first["consumer_inventory"][0]["sha256"] != second["consumer_inventory"][0]["sha256"]


def test_interrupted_overwrite_never_rolls_validation_back_to_old_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    artifact = project / "state.bin"
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    artifact.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    artifact.write_bytes(b"old raw")
    consumer.write_bytes(b"old accepted")
    producer.write_text('{"run": 1, "status": "success"}\n', encoding="utf-8")
    apply_retention_batch(
        project,
        artifact_class="state",
        paths=(artifact,),
        final_consumer="checkpoint",
        regeneration_recipe="rerun",
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
    )

    artifact.write_bytes(b"new raw")
    consumer.write_bytes(b"new accepted")
    producer.write_text('{"run": 2, "status": "success"}\n', encoding="utf-8")
    monkeypatch.setattr(
        retention_mod,
        "_unlink_path",
        lambda _path: (_ for _ in ()).throw(OSError("power loss")),
    )
    with pytest.raises(CleanupSafetyError, match="Retention cleanup failed"):
        apply_retention_batch(
            project,
            artifact_class="state",
            paths=(artifact,),
            final_consumer="checkpoint",
            regeneration_recipe="rerun",
            retained_consumers=(consumer,),
            producer_manifests=(producer,),
        )
    consumer.write_bytes(b"old accepted")

    with pytest.raises(CleanupSafetyError, match="retained consumer changed"):
        validate_retained_consumers(project)
    assert artifact.read_bytes() == b"new raw"
    ledger = json.loads(retention_mod.retention_manifest_path(project).read_text())
    assert ledger["generations"][0]["status"] == "superseded"
    assert ledger["generations"][1]["status"] == "planned"


def test_power_failure_between_classes_refuses_overwrite_generation_mix(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    first = project / "step" / "point.csv"
    second = project / "step" / "forcing.csv"
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    first.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    first.write_bytes(b"old point")
    second.write_bytes(b"old forcing")
    consumer.write_bytes(b"old compact")
    producer.write_text('{"run": 1, "status": "success"}\n', encoding="utf-8")
    generation = start_retention_generation(
        project,
        source_paths=(first, second),
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
    )
    apply_retention_batch(
        project,
        artifact_class="point",
        paths=(first,),
        final_consumer="compact point output",
        regeneration_recipe="rerun",
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
        generation=generation,
    )

    # A process dies before planning the forcing batch, then overwrite creates
    # a new point file and new scientific generation at the same paths.
    first.write_bytes(b"new point")
    second.write_bytes(b"new forcing")
    consumer.write_bytes(b"new compact")
    producer.write_text('{"run": 2, "status": "success"}\n', encoding="utf-8")

    with pytest.raises(CleanupSafetyError, match="changed after planning|identity"):
        start_retention_generation(
            project,
            source_paths=(first, second),
            retained_consumers=(consumer,),
            producer_manifests=(producer,),
        )

    assert first.read_bytes() == b"new point"
    assert second.read_bytes() == b"new forcing"
    ledger = json.loads(retention_mod.retention_manifest_path(project).read_text())
    assert len(ledger["generations"]) == 1
    assert ledger["generations"][0]["status"] == "planned"
    assert [batch["artifact_class"] for batch in ledger["batches"]] == ["point"]


def test_correct_cross_class_generation_resumes_and_completes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    first = project / "a.bin"
    second = project / "b.bin"
    first_consumer = project / "results" / "a.nc"
    second_consumer = project / "results" / "b.nc"
    producer = project / "results" / "run_manifest.json"
    first.parent.mkdir(parents=True)
    first_consumer.parent.mkdir(parents=True)
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    first_consumer.write_bytes(b"accepted-a")
    second_consumer.write_bytes(b"accepted-b")
    producer.write_text('{"run": 1, "status": "success"}\n', encoding="utf-8")
    generation = start_retention_generation(
        project,
        source_paths=(first, second),
        retained_consumers=(first_consumer, second_consumer),
        producer_manifests=(producer,),
    )
    apply_retention_batch(
        project,
        artifact_class="first",
        paths=(first,),
        final_consumer="first compact output",
        regeneration_recipe="rerun",
        retained_consumers=(first_consumer,),
        producer_manifests=(producer,),
        generation=generation,
    )

    resumed = start_retention_generation(
        project,
        source_paths=(second,),
        retained_consumers=(first_consumer, second_consumer),
        producer_manifests=(producer,),
    )
    assert resumed == generation
    apply_retention_batch(
        project,
        artifact_class="second",
        paths=(second,),
        final_consumer="second compact output",
        regeneration_recipe="rerun",
        retained_consumers=(second_consumer,),
        producer_manifests=(producer,),
        generation=generation,
    )
    complete_retention_generation(project, generation=generation)

    assert validate_retained_consumers(project, require_complete=True) == (
        "0001",
        "0002",
    )


@pytest.mark.parametrize("changed", ["source", "consumer", "producer"])
def test_generation_completion_revalidates_every_batch_identity(
    tmp_path: Path,
    changed: str,
) -> None:
    project = tmp_path / "project"
    source = project / "source.bin"
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    source.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    consumer.write_bytes(b"accepted")
    producer.write_text('{"run": 1, "status": "success"}\n', encoding="utf-8")
    generation = start_retention_generation(
        project,
        source_paths=(source,),
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
    )
    apply_retention_batch(
        project,
        artifact_class="source",
        paths=(source,),
        final_consumer="accepted output",
        regeneration_recipe="rerun",
        retained_consumers=(consumer,),
        producer_manifests=(producer,),
        generation=generation,
    )
    if changed == "source":
        source.write_bytes(b"recreated")
    elif changed == "consumer":
        consumer.write_bytes(b"changed")
    else:
        producer.write_text('{"run": 2, "status": "success"}\n', encoding="utf-8")

    with pytest.raises(CleanupSafetyError, match="changed|recreated"):
        complete_retention_generation(project, generation=generation)


def test_invalid_active_generation_cannot_validate_as_an_empty_ledger(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = project / "source.bin"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    _apply_retention(
        project,
        artifact_class="source",
        paths=(source,),
        final_consumer="accepted output",
        regeneration_recipe="rerun",
    )
    ledger_path = retention_mod.retention_manifest_path(project)
    ledger = json.loads(ledger_path.read_text())
    ledger["active_generation"] = None
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(CleanupSafetyError, match="no valid active generation"):
        validate_retained_consumers(project, require_complete=True)


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

    with pytest.raises(CleanupSafetyError, match="changed after planning|identity does not match"):
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


def test_planned_retry_refuses_changed_producer_before_delete(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    source = project / "source.bin"
    consumer = project / "results" / "accepted.nc"
    producer = project / "results" / "run_manifest.json"
    source.parent.mkdir(parents=True)
    consumer.parent.mkdir(parents=True)
    source.write_bytes(b"source")
    consumer.write_bytes(b"accepted")
    producer.write_text('{"run": 1, "status": "success"}\n', encoding="utf-8")
    real_unlink = retention_mod._unlink_path
    monkeypatch.setattr(
        retention_mod,
        "_unlink_path",
        lambda _path: (_ for _ in ()).throw(OSError("stop")),
    )
    with pytest.raises(CleanupSafetyError, match="Retention cleanup failed"):
        apply_retention_batch(
            project,
            artifact_class="source",
            paths=(source,),
            final_consumer="accepted output",
            regeneration_recipe="rerun",
            retained_consumers=(consumer,),
            producer_manifests=(producer,),
        )
    producer.write_text('{"run": 2, "status": "success"}\n', encoding="utf-8")
    monkeypatch.setattr(retention_mod, "_unlink_path", real_unlink)

    with pytest.raises(CleanupSafetyError, match="producer manifest changed"):
        apply_retention_batch(
            project,
            artifact_class="source",
            paths=(source,),
            final_consumer="accepted output",
            regeneration_recipe="rerun",
            retained_consumers=(consumer,),
            producer_manifests=(producer,),
        )

    assert source.is_file()
