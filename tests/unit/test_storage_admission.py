import json
import socket
import threading
import time
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
    project.mkdir(parents=True)
    obligations = {
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
    }
    step_obligations = {
        step: {
            component: (
                value // len(steps)
                if component in {
                    "forcing_bytes",
                    "member_grid_bytes",
                    "point_bytes",
                    "restart_baseline_bytes",
                    "derived_forcing_plot_bytes",
                    "retained_diagnostics_bytes",
                }
                else 0
            )
            for component, value in obligations.items()
        }
        for step in steps
    }
    leaf = StorageLeafPlan(
        leaf_id="leaf",
        setup_dir=tmp_path,
        project_dir=project,
        step_names=steps,
        obligations=obligations,
        step_obligations=step_obligations,
        queued_retained_bytes=300,
        identity="leaf-identity",
    )
    return StoragePlan(
        root_project_dir=project,
        leaves={"leaf": leaf},
        waves=(("leaf",),),
        wave_growth_bytes=(1300,),
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
    monkeypatch.setattr(
        Path,
        "glob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("glob hot path")),
    )
    monkeypatch.setattr(
        Path,
        "rglob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rglob hot path")),
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


def test_resume_does_not_double_credit_skipped_existing_step(tmp_path: Path) -> None:
    """A net plan already credited existing step_00 bytes at preflight."""
    plan = _plan(tmp_path)
    leaf = plan.leaves["leaf"]
    net_obligations = {name: 0 for name in leaf.obligations}
    net_obligations["forcing_bytes"] = 200
    resumed_leaf = type(leaf)(
        **{
            **leaf.__dict__,
            "obligations": net_obligations,
            "step_obligations": {
                "step_00": {**leaf.step_obligations["step_00"], "forcing_bytes": 0},
                "step_01": {**leaf.step_obligations["step_01"], "forcing_bytes": 200},
            },
        }
    )
    resumed_plan = StoragePlan(
        **{
            **plan.__dict__,
            "leaves": {"leaf": resumed_leaf},
            "estimated_growth_bytes": 200,
            "wave_growth_bytes": (200,),
        }
    )
    coordinator = StorageAdmissionCoordinator(
        resumed_plan,
        disk_usage=lambda _path: _usage(),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step("step_00", request_id="resume-step-00")
    skipped_existing = StorageAccountingSummary(
        completed_step="step_00",
        materialized_bytes={"forcing_bytes": 0},
        observed_bytes={"forcing_bytes": 100},
        source="reused",
    )

    snapshot = client.admit_step(
        "step_01",
        summary=skipped_existing,
        request_id="resume-step-01",
    )

    assert snapshot.estimated_growth_bytes == 200
    assert (
        coordinator.snapshot()["leaves"]["leaf"]["remaining_by_component"][
            "forcing_bytes"
        ]
        == 200
    )


def test_unequal_step_obligations_release_only_completed_step(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    leaf = plan.leaves["leaf"]
    obligations = {name: 0 for name in leaf.obligations}
    obligations["forcing_bytes"] = 300
    step_obligations = {
        "step_00": {name: 0 for name in leaf.step_obligations["step_00"]},
        "step_01": {name: 0 for name in leaf.step_obligations["step_01"]},
    }
    step_obligations["step_00"]["forcing_bytes"] = 100
    step_obligations["step_01"]["forcing_bytes"] = 200
    unequal_leaf = type(leaf)(
        **{
            **leaf.__dict__,
            "obligations": obligations,
            "step_obligations": step_obligations,
        }
    )
    unequal_plan = StoragePlan(
        **{
            **plan.__dict__,
            "leaves": {"leaf": unequal_leaf},
            "estimated_growth_bytes": 300,
            "wave_growth_bytes": (300,),
            "identity": "unequal",
        }
    )
    coordinator = StorageAdmissionCoordinator(
        unequal_plan,
        disk_usage=lambda _path: _usage(),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step("step_00", request_id="unequal-0")
    snapshot = client.admit_step(
        "step_01",
        summary=StorageAccountingSummary(
            completed_step="step_00",
            materialized_bytes={"forcing_bytes": 150},
            observed_bytes={"forcing_bytes": 150},
        ),
        request_id="unequal-1",
    )

    assert snapshot.estimated_growth_bytes == 200
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
    with pytest.raises(ValueError, match="different payload"):
        client.admit_step("step_01", request_id="same")


def test_inactive_wave_and_premature_finalization_fail_closed(tmp_path: Path) -> None:
    base = _plan(tmp_path, steps=("step_00",))
    leaf = base.leaves["leaf"]
    queued = type(leaf)(**{**leaf.__dict__, "leaf_id": "queued", "identity": "queued"})
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": {"leaf": leaf, "queued": queued},
            "waves": (("leaf",), ("queued",)),
            "wave_growth_bytes": (1600, 1600),
            "estimated_growth_bytes": 1600,
            "identity": "lifecycle",
        }
    )
    coordinator = StorageAdmissionCoordinator(
        plan,
        disk_usage=lambda _path: SimpleNamespace(
            total=1_000_000,
            used=1_000,
            free=999_000,
        ),
    )
    with pytest.raises(ValueError, match="active wave"):
        coordinator.admit_step(leaf_id="queued", step_name="step_00")
    with pytest.raises(ValueError, match="unfinished"):
        coordinator.transition(phase="leaf_finalized", leaf_id="leaf")


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


@pytest.mark.parametrize("client_count", [8, 24])
def test_spawn_safe_server_handles_simultaneous_clients_with_bounded_latency(
    tmp_path: Path,
    client_count: int,
) -> None:
    base = _plan(tmp_path, steps=("step_00",))
    template = base.leaves["leaf"]
    leaves = {
        f"leaf-{index}": type(template)(
            **{
                **template.__dict__,
                "leaf_id": f"leaf-{index}",
                "identity": f"leaf-{index}",
            }
        )
        for index in range(client_count)
    }
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": leaves,
            "waves": (tuple(leaves),),
            "outer_workers": client_count,
            "wave_growth_bytes": (1300 * client_count,),
            "estimated_growth_bytes": 1300 * client_count,
            "identity": f"stress-{client_count}",
        }
    )
    coordinator = StorageAdmissionCoordinator(
        plan,
        disk_usage=lambda _path: SimpleNamespace(
            total=1_000_000,
            used=1_000,
            free=999_000,
        ),
    )
    barrier = threading.Barrier(client_count)
    latencies: list[float] = []
    errors: list[BaseException] = []
    with StorageAdmissionServer(coordinator) as server:
        def admit(leaf_id: str) -> None:
            try:
                barrier.wait()
                started = time.perf_counter()
                server.client(leaf_id=leaf_id).admit_step(
                    "step_00", request_id=f"stress:{leaf_id}"
                )
                latencies.append(time.perf_counter() - started)
            except BaseException as exc:  # pragma: no cover - assertion captures it
                errors.append(exc)

        threads = [threading.Thread(target=admit, args=(leaf_id,)) for leaf_id in leaves]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=8)

    assert errors == []
    assert len(latencies) == client_count
    p95 = sorted(latencies)[max(0, int(client_count * 0.95) - 1)]
    assert p95 < 2.0


def test_failed_handshake_does_not_kill_server(tmp_path: Path) -> None:
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    with StorageAdmissionServer(coordinator) as server:
        address = server.client(leaf_id="leaf").address
        assert address is not None
        raw = socket.create_connection(address, timeout=1)
        raw.close()
        server.client(leaf_id="leaf").admit_step("step_00", request_id="after-bad-auth")


def test_whole_ipc_request_timeout_is_bounded(tmp_path: Path, monkeypatch) -> None:
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    with StorageAdmissionServer(coordinator) as server:
        monkeypatch.setattr(
            "openamundsen_da.util.storage_admission._timed_connection",
            lambda *_args, **_kwargs: time.sleep(5),
        )
        monkeypatch.setattr(
            "openamundsen_da.util.storage_admission.STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS",
            0.05,
        )
        started = time.perf_counter()
        with pytest.raises(RuntimeError, match="unavailable or timed out"):
            server.client(leaf_id="leaf").admit_step("step_00", request_id="timeout")
        assert time.perf_counter() - started < 0.5


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


def test_missing_ledger_with_partial_project_records_reconciliation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    manifest = (
        plan.leaves["leaf"].project_dir
        / "steps"
        / "step_00"
        / "assim"
        / "prior_forcing_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")

    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())

    assert coordinator.snapshot()["targeted_reconciliation_count"] == 1
    assert coordinator.snapshot()["status"] == "reconciled_legacy_partial"


def test_resume_reconciles_crash_after_leaf_finalization_manifest(tmp_path: Path) -> None:
    plan = _plan(tmp_path, steps=("step_00",))
    leaf = plan.leaves["leaf"]
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    coordinator.admit_step(leaf_id="leaf", step_name="step_00", request_id="crash-admit")
    coordinator.transition(
        phase="leaf_project_complete",
        leaf_id="leaf",
        summary=StorageAccountingSummary(
            completed_step="step_00",
            materialized_bytes={},
        ),
        request_id="crash-complete",
    )
    (leaf.setup_dir / "leaf_finalization_manifest.json").write_text(
        json.dumps({"status": "success", "project_dir": str(leaf.project_dir)}),
        encoding="utf-8",
    )

    resumed = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())

    state = resumed.snapshot()
    assert state["generation"] == coordinator.generation
    assert state["leaves"]["leaf"]["phase"] == "finalized"
    assert state["leaves"]["leaf"]["remaining_by_component"] == {
        name: 0 for name in leaf.obligations
    }
    assert state["targeted_reconciliation_count"] == 1


def test_compact_idempotence_survives_audit_eviction_and_resume(tmp_path: Path) -> None:
    steps = tuple(f"step_{index:02d}" for index in range(70))
    plan = _plan(tmp_path, steps=steps)
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step(steps[0], request_id=f"leaf:admit:{steps[0]}")
    for previous, current in zip(steps, steps[1:], strict=False):
        client.admit_step(
            current,
            summary=StorageAccountingSummary(
                completed_step=previous,
                materialized_bytes={},
            ),
            request_id=f"leaf:admit:{current}",
        )
    assert len(coordinator.snapshot()["requests"]) == 64

    resumed = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    replay = StorageAdmissionClient.in_process(resumed, leaf_id="leaf").admit_step(
        steps[0], request_id=f"leaf:admit:{steps[0]}"
    )

    assert replay.estimated_growth_bytes == resumed.snapshot()["remaining_peak_growth_bytes"]
    assert resumed.snapshot()["leaves"]["leaf"]["last_admitted_step_index"] == 69


def test_maximum_wave_and_concurrent_finalization_arithmetic(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    component_names = tuple(_plan(tmp_path / "components").leaves["leaf"].obligations)

    def leaf(leaf_id: str, non_transition: int, transition: int, retained: int):
        obligations = {name: 0 for name in component_names}
        obligations["forcing_bytes"] = non_transition
        obligations["restart_transition_bytes"] = transition
        setup_dir = tmp_path / leaf_id
        setup_dir.mkdir()
        return StorageLeafPlan(
            leaf_id=leaf_id,
            setup_dir=setup_dir,
            project_dir=project,
            step_names=("step_00",),
            obligations=obligations,
            step_obligations={
                "step_00": {
                    component: int(obligations.get(component, 0))
                    for component in (
                        "forcing_bytes",
                        "member_grid_bytes",
                        "point_bytes",
                        "restart_baseline_bytes",
                        "derived_forcing_plot_bytes",
                        "retained_diagnostics_bytes",
                    )
                }
            },
            queued_retained_bytes=retained,
            identity=leaf_id,
        )

    leaves = {
        "A": leaf("A", 100, 50, 20),
        "B": leaf("B", 200, 25, 30),
        "C": leaf("C", 500, 100, 40),
    }
    # Wave 1 cumulative peak: A+B active (375), C retained (40), parent (20).
    # Wave 2 cumulative peak: A+B retained (50), C active (600), parent (20).
    plan = StoragePlan(
        root_project_dir=project,
        leaves=leaves,
        waves=(("A", "B"), ("C",)),
        wave_growth_bytes=(435, 670),
        outer_workers=2,
        parent_finalization_reserve_bytes=20,
        estimated_growth_bytes=670,
        overwrite=False,
        filesystem_device=project.stat().st_dev,
        filesystem_capacity_bytes=10_000,
        identity="waves",
        estimate_duration_seconds=0.1,
    )
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    coordinator.record_preflight(
        coordinator.admit_wave(0, request_id="wave-0")
    )
    assert coordinator.snapshot()["remaining_peak_growth_bytes"] == 670

    def complete_leaf(leaf_id: str) -> None:
        coordinator.admit_step(
            leaf_id=leaf_id,
            step_name="step_00",
            request_id=f"admit-{leaf_id}",
        )
        coordinator.transition(
            phase="leaf_project_complete",
            leaf_id=leaf_id,
            summary=StorageAccountingSummary(
                completed_step="step_00",
                materialized_bytes={
                    "forcing_bytes": leaves[leaf_id].step_obligations["step_00"][
                        "forcing_bytes"
                    ]
                },
            ),
            request_id=f"complete-{leaf_id}",
        )
        (leaves[leaf_id].setup_dir / "leaf_finalization_manifest.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "project_dir": str(leaves[leaf_id].project_dir),
                }
            ),
            encoding="utf-8",
        )
        coordinator.transition(
            phase="leaf_finalized",
            leaf_id=leaf_id,
            request_id=f"final-{leaf_id}",
        )

    complete_leaf("A")
    after_a = coordinator.snapshot()["remaining_peak_growth_bytes"]
    complete_leaf("B")
    after_b = coordinator.snapshot()["remaining_peak_growth_bytes"]
    assert after_a < 670
    assert after_b < after_a

    coordinator.admit_wave(1, request_id="wave-1")
    assert coordinator.snapshot()["remaining_peak_growth_bytes"] == 620


def test_two_leaf_server_finalization_has_no_lost_release(tmp_path: Path) -> None:
    base = _plan(tmp_path, steps=("step_00",))
    leaf_a = base.leaves["leaf"]
    setup_a = tmp_path / "A"
    setup_b = tmp_path / "B"
    setup_a.mkdir()
    setup_b.mkdir()
    leaf_b = StorageLeafPlan(
        **{
            **leaf_a.__dict__,
            "leaf_id": "B",
            "setup_dir": setup_b,
            "identity": "B",
        }
    )
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": {
                "A": type(leaf_a)(
                    **{**leaf_a.__dict__, "leaf_id": "A", "setup_dir": setup_a}
                ),
                "B": leaf_b,
            },
            "waves": (("A", "B"),),
            "wave_growth_bytes": (2600,),
            "estimated_growth_bytes": 2600,
            "identity": "two-leaf",
        }
    )
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    with StorageAdmissionServer(coordinator) as server:
        for leaf_id in ("A", "B"):
            server.client(leaf_id=leaf_id).admit_step(
                "step_00", request_id=f"admit-{leaf_id}"
            )
            server.client(leaf_id=leaf_id).transition(
                "leaf_project_complete",
                summary=StorageAccountingSummary(
                    completed_step="step_00",
                    materialized_bytes={},
                ),
                request_id=f"complete-{leaf_id}",
            )
            (plan.leaves[leaf_id].setup_dir / "leaf_finalization_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "success",
                        "project_dir": str(plan.leaves[leaf_id].project_dir),
                    }
                ),
                encoding="utf-8",
            )
        errors: list[Exception] = []

        def finalize(leaf_id: str) -> None:
            try:
                server.client(leaf_id=leaf_id).transition(
                    "leaf_finalized", request_id=f"final-{leaf_id}"
                )
            except Exception as exc:  # pragma: no cover - assertion captures it
                errors.append(exc)

        threads = [threading.Thread(target=finalize, args=(leaf_id,)) for leaf_id in ("A", "B")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert errors == []
    assert all(
        state["phase"] == "finalized"
        for state in coordinator.snapshot()["leaves"].values()
    )
