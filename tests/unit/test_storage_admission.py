import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.exceptions import LowDiskPauseError
from openamundsen_da.util.storage_admission import (
    StorageAccountingSummary,
    StorageAdmissionClient,
    StorageAdmissionCoordinator,
    StorageAdmissionServer,
    StorageLeafPlan,
    StoragePlan,
)


def _plan(tmp_path: Path, *, steps: tuple[str, ...] = ("step_00", "step_01")) -> StoragePlan:
    project = tmp_path / "project"
    project.mkdir()
    leaf = StorageLeafPlan(
        leaf_id="leaf",
        setup_dir=tmp_path,
        project_dir=project,
        step_names=steps,
        obligations={
            "forcing_bytes": 200,
            "member_grid_bytes": 200,
            "point_bytes": 200,
            "restart_baseline_bytes": 200,
            "restart_transition_bytes": 100,
            "compact_timeseries_bytes": 100,
            "compact_grid_bytes": 100,
            "map_support_bytes": 0,
            "derived_forcing_plot_bytes": 100,
            "retained_diagnostics_bytes": 100,
        },
        identity="leaf-identity",
    )
    return StoragePlan(
        root_project_dir=project,
        leaves={"leaf": leaf},
        outer_workers=1,
        parent_finalization_reserve_bytes=0,
        estimated_growth_bytes=1300,
        overwrite=False,
        filesystem_device=project.stat().st_dev,
        filesystem_capacity_bytes=10_000,
        identity="plan-identity",
        estimate_duration_seconds=0.25,
    )


def _usage(*, used: int = 1000):
    return SimpleNamespace(total=10_000, used=used, free=10_000 - used)


def _summary(*, forcing: int = 100) -> StorageAccountingSummary:
    return StorageAccountingSummary(
        completed_step="step_00",
        materialized_bytes={
            "forcing_bytes": forcing,
            "member_grid_bytes": 50,
            "point_bytes": 25,
            "restart_baseline_bytes": 75,
            "derived_forcing_plot_bytes": 10,
            "retained_diagnostics_bytes": 20,
        },
        file_counts={"forcing_bytes": 2},
        cleanup_freed_bytes=15,
    )


def test_step_boundary_uses_one_disk_check_and_no_estimator_or_source_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path),
        disk_usage=lambda path: calls.append(path) or _usage(),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step("step_00", request_id="initial")

    monkeypatch.setattr(
        "openamundsen_da.util.storage_budget.estimate_project_storage_components",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("estimator hot path")),
    )
    snapshot = client.admit_step(
        "step_01",
        summary=_summary(),
        request_id="next",
    )

    assert len(calls) == 2
    assert snapshot.estimated_growth_bytes < 1300
    ledger = json.loads(coordinator.ledger_path.read_text(encoding="utf-8"))
    assert ledger["full_estimate_count"] == 1
    assert ledger["lightweight_check_count"] == 2
    assert ledger["leaves"]["leaf"]["last_completed_step_index"] == 0


def test_observed_size_calibration_only_raises_future_component_bound(tmp_path: Path) -> None:
    plan = _plan(tmp_path, steps=("step_00", "step_01", "step_02"))
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step("step_00", request_id="initial")
    client.admit_step(
        "step_01",
        summary=_summary(forcing=250),
        request_id="second",
    )

    leaf = coordinator.snapshot()["leaves"]["leaf"]
    assert leaf["observed_step_high_water_bytes"]["forcing_bytes"] == 250
    assert leaf["remaining_by_component"]["forcing_bytes"] >= 250


def test_missing_stale_and_duplicate_requests_fail_closed(tmp_path: Path) -> None:
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step("step_00", request_id="same")
    duplicate = client.admit_step("step_00", request_id="same")
    assert duplicate.estimated_growth_bytes == 1300

    with pytest.raises(ValueError, match="accounting summary is required"):
        client.admit_step("step_01", request_id="missing")
    with pytest.raises(ValueError, match="stale or out of order"):
        client.admit_step(
            "step_01",
            summary=StorageAccountingSummary(
                completed_step="wrong",
                materialized_bytes={},
            ),
            request_id="stale",
        )


def test_low_disk_is_durable_and_resume_is_non_destructive(tmp_path: Path) -> None:
    used = [8500]
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path),
        disk_usage=lambda _path: _usage(used=used[0]),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    with pytest.raises(LowDiskPauseError):
        client.admit_step("step_00", request_id="low")
    assert coordinator.snapshot()["status"] == "paused_low_disk"

    used[0] = 1000
    admitted = client.admit_step("step_00", request_id="low")
    assert admitted.used_bytes == 1000
    assert coordinator.snapshot()["status"] == "admitted"


def test_atomic_ledger_failure_does_not_publish_mutated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    before = coordinator.snapshot()
    original = coordinator.ledger_path.read_text(encoding="utf-8")

    def _fail(*_args, **_kwargs):
        raise RuntimeError("atomic write failed")

    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.write_manifest_atomic",
        _fail,
    )
    with pytest.raises(RuntimeError, match="atomic write failed"):
        coordinator.admit_step(leaf_id="leaf", step_name="step_00")
    assert coordinator.snapshot() == before
    assert coordinator.ledger_path.read_text(encoding="utf-8") == original


def test_spawn_safe_server_serializes_client_requests(tmp_path: Path) -> None:
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    with StorageAdmissionServer(coordinator) as server:
        client = server.client(leaf_id="leaf")
        snapshot = client.admit_step("step_00", request_id="ipc")

    assert snapshot.estimated_growth_bytes == 1300
    assert coordinator.snapshot()["requests"]["ipc"]["status"] == "admitted"


def test_overwrite_supersedes_previous_generation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    first = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    first_generation = first.generation
    overwrite_plan = StoragePlan(**{**plan.__dict__, "overwrite": True})
    second = StorageAdmissionCoordinator(overwrite_plan, disk_usage=lambda _path: _usage())

    assert second.generation != first_generation
    archive = second.ledger_path.with_name(
        f"storage_reservation.{first_generation}.json"
    )
    assert archive.is_file()
