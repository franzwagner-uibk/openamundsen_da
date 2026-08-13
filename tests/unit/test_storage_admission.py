import json
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
    build_storage_plan,
    accounting_summary_from_paths,
)
from openamundsen_da.util import storage_admission as storage_admission_mod
from openamundsen_da.manifests import (
    file_inventory,
    inventory_digest,
    project_scientific_input_inventory,
    write_manifest_atomic,
)
from openamundsen_da.util.storage_budget import (
    ProjectStorageEstimate,
    StorageReservationProject,
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


def test_explicit_boundary_accounting_uses_stat_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "step_00" / "plots" / "forcing" / "station.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"plot")
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.sha256_file",
        lambda _path: (_ for _ in ()).throw(AssertionError("boundary content hash")),
    )

    summary = accounting_summary_from_paths(
        completed_step="step_00",
        root=tmp_path,
        paths=(output, output),
        source="producer",
    )

    assert summary.materialized_bytes["derived_forcing_plot_bytes"] == 4
    assert summary.file_counts["derived_forcing_plot_bytes"] == 1


@pytest.mark.parametrize("kind", ("missing", "symlink", "escape"))
def test_explicit_boundary_accounting_rejects_untrusted_paths(
    tmp_path: Path,
    kind: str,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    if kind == "missing":
        claimed = root / "missing.bin"
    elif kind == "symlink":
        claimed = root / "link.bin"
        claimed.symlink_to(outside)
    else:
        claimed = outside
    with pytest.raises(ValueError, match="missing|symlinked|outside"):
        accounting_summary_from_paths(
            completed_step="step_00",
            root=root,
            paths=(claimed,),
            source="producer",
        )


def _scientific_config(tmp_path: Path) -> SimpleNamespace:
    setup = tmp_path / "setup"
    project = setup / "projects" / "project"
    obs = setup / "obs" / "stations"
    obs.mkdir(parents=True)
    project.mkdir(parents=True)
    setup_yaml = setup / "setup.yml"
    project_yaml = project / "project.yml"
    setup_yaml.write_text("domain: test\n", encoding="utf-8")
    project_yaml.write_text("start_date: 2023-01-01\n", encoding="utf-8")
    return SimpleNamespace(
        setup_dir=setup,
        project_dir=project,
        setup_yaml=setup_yaml,
        project_yaml=project_yaml,
        setup={"domain": "test", "input_data": {}},
        project={"obs": {"stations": {"dir": "obs/stations"}}},
    )


def test_scientific_identity_hashes_contained_file_symlink_target(tmp_path: Path) -> None:
    config = _scientific_config(tmp_path)
    shared = tmp_path / "shared.csv"
    shared.write_text("a\n1\n", encoding="utf-8")
    link = config.setup_dir / "obs" / "stations" / "station.csv"
    link.symlink_to(shared)

    _inventory, first = project_scientific_input_inventory(
        config,
        identity_root=tmp_path,
    )
    shared.write_text("a\n2\n", encoding="utf-8")
    _inventory, second = project_scientific_input_inventory(
        config,
        identity_root=tmp_path,
    )

    assert first != second


def test_scientific_identity_rejects_escaping_file_and_directory_symlinks(
    tmp_path: Path,
) -> None:
    config = _scientific_config(tmp_path)
    external_root = tmp_path.parent / f"{tmp_path.name}-external"
    external_root.mkdir()
    external_file = external_root / "station.csv"
    external_file.write_text("a\n1\n", encoding="utf-8")
    link = config.setup_dir / "obs" / "stations" / "station.csv"
    link.symlink_to(external_file)
    with pytest.raises(Exception, match="escapes"):
        project_scientific_input_inventory(config, identity_root=tmp_path)
    link.unlink()
    linked_dir = config.setup_dir / "obs" / "linked"
    linked_dir.symlink_to(config.setup_dir / "obs" / "stations", target_is_directory=True)
    config.project["obs"]["stations"]["dir"] = "obs/linked"
    with pytest.raises(Exception, match="directory symlinks"):
        project_scientific_input_inventory(config, identity_root=tmp_path)


def test_cross_wave_calibration_uses_immutable_base_without_compounding(
    tmp_path: Path,
) -> None:
    steps = tuple(f"step_{index:02d}" for index in range(10))
    base = _plan(tmp_path, steps=steps)
    template = base.leaves["leaf"]
    obligations_a = {**template.obligations, "forcing_bytes": 1_000}
    obligations_b = {**template.obligations, "forcing_bytes": 10_000}
    leaves = {
        "A": type(template)(
            **{**template.__dict__, "leaf_id": "A", "obligations": obligations_a}
        ),
        "B": type(template)(
            **{**template.__dict__, "leaf_id": "B", "obligations": obligations_b}
        ),
    }
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": leaves,
            "waves": (("A",), ("B",)),
            "wave_growth_bytes": (2_100, 11_100),
            "estimated_growth_bytes": 11_100,
            "identity": "calibration",
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
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="A")
    client.admit_step("step_00", request_id="A:0")
    summary = StorageAccountingSummary(
        completed_step="step_00",
        materialized_bytes={"forcing_bytes": 200},
        observed_bytes={"forcing_bytes": 200},
    )
    client.admit_step("step_01", summary=summary, request_id="A:1")
    first = coordinator.snapshot()["leaves"]["B"]["planned_by_component"]["forcing_bytes"]
    client.admit_step("step_01", summary=summary, request_id="A:1")
    second = coordinator.snapshot()["leaves"]["B"]["planned_by_component"]["forcing_bytes"]

    assert first == 20_000
    assert second == first


@pytest.mark.parametrize(
    ("reporting_mode", "future_mode", "expected"),
    (("compact", "full", 2_000), ("full", "full", 20_000), ("full", "compact", 1_000)),
)
def test_restart_calibration_uses_reporting_and_future_retention_units(
    tmp_path: Path,
    reporting_mode: str,
    future_mode: str,
    expected: int,
) -> None:
    steps = tuple(f"step_{index:02d}" for index in range(10))
    base = _plan(tmp_path, steps=steps)
    template = base.leaves["leaf"]
    leaves = {
        "A": type(template)(
            **{
                **template.__dict__,
                "leaf_id": "A",
                "retention_mode": reporting_mode,
                "obligations": {**template.obligations, "restart_baseline_bytes": 100},
            }
        ),
        "B": type(template)(
            **{
                **template.__dict__,
                "leaf_id": "B",
                "retention_mode": future_mode,
                "obligations": {**template.obligations, "restart_baseline_bytes": 1_000},
            }
        ),
    }
    coordinator = StorageAdmissionCoordinator(
        StoragePlan(
            **{
                **base.__dict__,
                "leaves": leaves,
                "waves": (("A",), ("B",)),
                "wave_growth_bytes": (2_000, 3_000),
                "estimated_growth_bytes": 3_000,
                "identity": f"retention-{reporting_mode}-{future_mode}",
            }
        ),
        disk_usage=lambda _path: SimpleNamespace(
            total=1_000_000,
            used=1_000,
            free=999_000,
        ),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="A")
    client.admit_step("step_00", request_id="A:0")
    client.admit_step(
        "step_01",
        summary=StorageAccountingSummary(
            completed_step="step_00",
            materialized_bytes={"restart_baseline_bytes": 200},
            observed_bytes={"restart_baseline_bytes": 200},
        ),
        request_id="A:1",
    )
    assert (
        coordinator.snapshot()["leaves"]["B"]["planned_by_component"][
            "restart_baseline_bytes"
        ]
        == expected
    )


def _prepared_project(tmp_path: Path, *, malformed: bool = False):
    setup = tmp_path / "setup"
    project = setup / "projects" / "project_2023"
    setup.mkdir(parents=True)
    project.mkdir(parents=True)
    (setup / "test.yml").write_text("domain: test\n", encoding="utf-8")
    (project / "project_2023.yml").write_text("start_date: 2023-01-01\n", encoding="utf-8")
    for index in range(2):
        step = project / "steps" / f"step_{index:02d}"
        member = step / "ensembles" / "prior" / "member_000" / "results"
        member.mkdir(parents=True)
        (step / "step.yml").write_text(
            f"start_date: 2023-01-0{index + 1}\nend_date: 2023-01-0{index + 2}\n",
            encoding="utf-8",
        )
        if index == 0:
            (member / "member_run.json").write_text(
                "{}" if malformed else json.dumps({"status": "success"}),
                encoding="utf-8",
            )
    if not malformed:
        next_assim = project / "steps" / "step_01" / "assim"
        next_assim.mkdir()
        for name in ("prior_weights_manifest.json", "rejuvenate_manifest.json"):
            (next_assim / name).write_text(
                json.dumps({"status": "complete"}),
                encoding="utf-8",
            )
    reservation = StorageReservationProject(
        setup_dir=setup,
        project_dir=project,
        grid_cell_count=1,
    )
    return setup, project, reservation


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
    assert snapshot.estimated_growth_bytes == 1300
    ledger = json.loads(coordinator.ledger_path.read_text(encoding="utf-8"))
    assert ledger["full_estimate_count"] == 1
    assert ledger["lightweight_check_count"] == 2
    assert ledger["leaves"]["leaf"]["last_completed_step_index"] == 0


def test_single_domain_first_step_does_not_require_leaf_preparation(tmp_path: Path) -> None:
    plan = _plan(tmp_path, steps=("step_00",))
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())

    snapshot = coordinator.admit_step(
        leaf_id="leaf", step_name="step_00", request_id="single-first"
    )

    assert snapshot.estimated_growth_bytes == plan.estimated_growth_bytes


def test_subdomain_preparation_releases_each_leaf_obligation_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _plan(tmp_path, steps=("step_00",))
    leaf = base.leaves["leaf"]
    step_dir = leaf.project_dir / "steps" / "step_00"
    step_dir.mkdir(parents=True)
    (step_dir / "step.yml").write_text(
        "start_date: 2023-01-01\nend_date: 2023-01-02\n",
        encoding="utf-8",
    )
    from datetime import datetime

    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.plan_project_steps",
        lambda *_args: [
            SimpleNamespace(
                name="step_00",
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
            )
        ],
    )
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission._project_identity",
        lambda *_args, **_kwargs: leaf.identity,
    )
    manifest_path = (
        leaf.setup_dir
        / ".openamundsen-da"
        / "manifests"
        / "leaf_preparation.json"
    )
    manifest_path.parent.mkdir(parents=True)
    prepared_inventory = file_inventory(root=leaf.setup_dir, files=[step_dir / "step.yml"])
    write_manifest_atomic(
        manifest_path,
        {
            "status": "success",
            "scientific_identity": leaf.identity,
            "outputs": prepared_inventory,
            "output_digest": inventory_digest(prepared_inventory),
        },
    )
    prepared_leaf = type(leaf)(
        **{
            **leaf.__dict__,
            "preparation_bytes": 100,
            "requires_preparation": True,
        }
    )
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": {"leaf": prepared_leaf},
            "wave_growth_bytes": (base.wave_growth_bytes[0] + 100,),
            "estimated_growth_bytes": base.estimated_growth_bytes + 100,
            "identity": "prepared-leaf",
        }
    )
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    coordinator.admit_wave(0, request_id="wave-0")
    with pytest.raises(ValueError, match="authoritative preparation"):
        coordinator.admit_step(
            leaf_id="leaf", step_name="step_00", request_id="before-prep"
        )

    before = coordinator.snapshot()["remaining_peak_growth_bytes"]
    coordinator.prepare_wave(0, request_id="prepared-wave")
    after = coordinator.snapshot()["remaining_peak_growth_bytes"]
    coordinator.prepare_wave(0, request_id="prepared-wave")

    assert before - after == 100
    assert coordinator.snapshot()["remaining_peak_growth_bytes"] == after
    coordinator.admit_step(
        leaf_id="leaf", step_name="step_00", request_id="after-prep"
    )

    (step_dir / "step.yml").write_text(
        "start_date: 2023-01-01\nend_date: 2023-01-03\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outputs changed|windows changed"):
        coordinator._validate_leaf_preparation("leaf")


def test_prepare_wave_rejects_raw_source_mutation_and_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, steps=("step_00",))
    leaf = plan.leaves["leaf"]
    raw = tmp_path / "raw.tif"
    raw.write_bytes(b"original")
    digest = storage_admission_mod._scientific_paths_identity(
        (raw,), identity_root=tmp_path
    )
    guarded = type(leaf)(
        **{
            **leaf.__dict__,
            "scientific_input_paths": (raw,),
            "scientific_root": tmp_path,
            "preparation_inputs_identity": digest,
        }
    )
    guarded_plan = StoragePlan(
        **{**plan.__dict__, "leaves": {"leaf": guarded}, "identity": "guarded"}
    )
    coordinator = StorageAdmissionCoordinator(
        guarded_plan, disk_usage=lambda _path: _usage()
    )
    coordinator.admit_wave(0, request_id="wave-raw")
    raw.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed during wave preparation"):
        coordinator.prepare_wave(0, request_id="prepared-raw")

    linked = tmp_path / "linked.tif"
    linked.symlink_to(raw)
    with pytest.raises(RuntimeError, match="symlink is unsupported"):
        storage_admission_mod._scientific_paths_identity(
            (linked,), identity_root=tmp_path
        )


def test_prepare_wave_rechecks_both_canonical_and_consumed_support_files(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, steps=("step_00",))
    leaf = plan.leaves["leaf"]
    parent_support = tmp_path / "obs/acquisition.csv"
    consumed_support = tmp_path / "leaf/obs/acquisition.csv"
    parent_support.parent.mkdir(parents=True)
    consumed_support.parent.mkdir(parents=True)
    parent_support.write_bytes(b"same")
    consumed_support.write_bytes(b"same")
    inputs = (parent_support, consumed_support)
    digest = storage_admission_mod._scientific_paths_identity(
        inputs, identity_root=tmp_path
    )
    guarded = type(leaf)(
        **{
            **leaf.__dict__,
            "scientific_input_paths": inputs,
            "scientific_root": tmp_path,
            "preparation_inputs_identity": digest,
        }
    )
    coordinator = StorageAdmissionCoordinator(
        StoragePlan(
            **{**plan.__dict__, "leaves": {"leaf": guarded}, "identity": "support"}
        ),
        disk_usage=lambda _path: _usage(),
    )
    coordinator.admit_wave(0, request_id="support-wave")
    parent_support.write_bytes(b"mutated")

    with pytest.raises(RuntimeError, match="changed during wave preparation"):
        coordinator.prepare_wave(0, request_id="support-prepared")


def test_prepare_wave_terminal_commit_at_85_percent_then_step_zero_pauses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Reuse the authoritative preparation fixture above while making the
    # filesystem large enough that the compact reserve remains below 90%.
    plan = _plan(tmp_path, steps=("step_00",))
    leaf = type(plan.leaves["leaf"])(
        **{
            **plan.leaves["leaf"].__dict__,
            "requires_preparation": True,
            "preparation_bytes": 100,
        }
    )
    step_dir = leaf.project_dir / "steps" / "step_00"
    step_dir.mkdir(parents=True)
    step_yaml = step_dir / "step.yml"
    step_yaml.write_text(
        "start_date: 2023-01-01\nend_date: 2023-01-02\n", encoding="utf-8"
    )
    inventory = file_inventory(root=leaf.setup_dir, files=[step_yaml])
    prep = leaf.setup_dir / ".openamundsen-da/manifests/leaf_preparation.json"
    write_manifest_atomic(
        prep,
        {
            "status": "success",
            "scientific_identity": leaf.identity,
            "outputs": inventory,
            "output_digest": inventory_digest(inventory),
        },
    )
    from datetime import datetime

    monkeypatch.setattr(
        storage_admission_mod,
        "plan_project_steps",
        lambda *_args: [
            SimpleNamespace(
                name="step_00",
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
            )
        ],
    )
    monkeypatch.setattr(storage_admission_mod, "_project_identity", lambda *_a, **_k: leaf.identity)
    guarded_plan = StoragePlan(
        **{
            **plan.__dict__,
            "leaves": {"leaf": leaf},
            "wave_growth_bytes": (plan.wave_growth_bytes[0] + 100,),
            "estimated_growth_bytes": plan.estimated_growth_bytes + 100,
            "identity": "soft-prep",
        }
    )
    coordinator = StorageAdmissionCoordinator(
        guarded_plan,
        disk_usage=lambda _path: SimpleNamespace(
            total=100_000_000, used=84_000_000, free=16_000_000
        ),
    )
    coordinator.admit_wave(0, request_id="soft-wave", allow_existing_step_drain=True)
    coordinator.prepare_wave(0, request_id="soft-prepared")
    with pytest.raises(LowDiskPauseError):
        coordinator.admit_step(
            leaf_id="leaf", step_name="step_00", request_id="soft-step"
        )


def test_prepare_wave_resume_preserves_progressed_leaves_and_allows_next_wave(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _plan(tmp_path, steps=("step_00",))
    template = base.leaves["leaf"]
    leaves = {
        name: type(template)(
            **{
                **template.__dict__,
                "leaf_id": name,
                "identity": "mixed-identity",
                "requires_preparation": True,
            }
        )
        for name in ("done", "interrupted", "queued")
    }
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": leaves,
            "waves": (("done", "interrupted"), ("queued",)),
            "wave_growth_bytes": (2_600, 1_900),
            "estimated_growth_bytes": 2_600,
            "identity": "mixed-resume",
        }
    )
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    coordinator.admit_wave(0, request_id="mixed-wave-0")
    coordinator._ledger["leaves"]["done"]["phase"] = "finalized"
    coordinator._ledger["leaves"]["done"]["last_admitted_step_index"] = 0
    coordinator._ledger["leaves"]["done"]["last_completed_step_index"] = 0
    coordinator._ledger["leaves"]["interrupted"]["phase"] = "running"
    coordinator._ledger["leaves"]["interrupted"]["last_admitted_step_index"] = 0
    coordinator._ledger["phase"] = "running"
    monkeypatch.setattr(coordinator, "_validate_leaf_preparation", lambda _leaf: None)
    monkeypatch.setattr(
        storage_admission_mod,
        "_project_identity",
        lambda *_a, **_k: "mixed-identity",
    )

    coordinator.prepare_wave(0, request_id="mixed-prepared")
    state = coordinator.snapshot()
    assert state["leaves"]["done"]["phase"] == "finalized"
    assert state["leaves"]["interrupted"]["phase"] == "running"
    coordinator._ledger["leaves"]["interrupted"]["phase"] = "finalized"
    coordinator._ledger["leaves"]["interrupted"]["last_completed_step_index"] = 0

    admitted = coordinator.admit_wave(1, request_id="mixed-wave-1")
    assert admitted.estimated_growth_bytes >= 0
    assert coordinator.snapshot()["active_leaf_ids"] == ["queued"]


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


def test_unequal_step_obligations_remain_reserved_until_finalization(tmp_path: Path) -> None:
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

    assert snapshot.estimated_growth_bytes == 300


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


def test_new_lifecycle_phase_pauses_at_soft_limit_but_admitted_step_can_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = StorageAdmissionCoordinator(
        StoragePlan(
            **{
                **_plan(tmp_path, steps=("step_00",)).__dict__,
                "estimated_growth_bytes": 100,
                "wave_growth_bytes": (100,),
            }
        ),
        disk_usage=lambda _path: SimpleNamespace(
            total=100_000,
            used=81_000,
            free=19_000,
        ),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    with pytest.raises(LowDiskPauseError, match="80%"):
        client.admit_wave(0, request_id="soft-wave")
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_project_storage_components",
        lambda **_kwargs: ProjectStorageEstimate(
            forcing_bytes=1,
            member_grid_bytes=1,
            point_bytes=1,
            restart_baseline_bytes=1,
            restart_transition_bytes=1,
            compact_timeseries_bytes=1,
            compact_grid_bytes=1,
        ),
    )
    with pytest.raises(LowDiskPauseError, match="80%"):
        client.reconcile_finalization(request_id="soft-reconcile")
    with pytest.raises(LowDiskPauseError, match="80%"):
        client.transition("parent_render", request_id="soft-render")

    admitted = client.admit_step(
        "step_00",
        request_id="draining-step",
        allow_existing_step_drain=True,
    )
    assert admitted.used_fraction == pytest.approx(0.81)


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


def test_file_ipc_accepts_back_to_back_lifecycle_requests(tmp_path: Path) -> None:
    steps = tuple(f"step_{index:02d}" for index in range(20))
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path, steps=steps),
        disk_usage=lambda _path: _usage(),
    )
    with StorageAdmissionServer(coordinator) as server:
        client = server.client(leaf_id="leaf")
        client.admit_step(steps[0], request_id="back-to-back-0")
        for index, step in enumerate(steps[1:], start=1):
            client.admit_step(
                step,
                summary=StorageAccountingSummary(
                    completed_step=steps[index - 1],
                    materialized_bytes={},
                ),
                request_id=f"back-to-back-{index}",
            )

    assert coordinator.snapshot()["leaves"]["leaf"]["last_admitted_step_index"] == 19


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


def test_file_ipc_polls_only_active_wave_and_control_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
        for index in range(90)
    }
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": leaves,
            "waves": tuple((leaf_id,) for leaf_id in leaves),
            "wave_growth_bytes": tuple(1300 for _ in leaves),
            "estimated_growth_bytes": 1300,
            "identity": "ninety-leaf-polling",
        }
    )
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    checked: set[Path] = set()
    original = Path.is_file

    def counted(path: Path) -> bool:
        if "ipc" in path.parts:
            checked.add(path)
        return original(path)

    monkeypatch.setattr(Path, "is_file", counted)
    with StorageAdmissionServer(coordinator) as server:
        time.sleep(0.06)
        expected = {
            server._control_request_path,
            server._request_paths["leaf-0"],
        }
        serving_checked = set(checked)

    assert serving_checked <= expected


def test_large_prepared_plan_boundary_latency_is_below_one_second(tmp_path: Path) -> None:
    steps = tuple(f"step_{index:02d}" for index in range(51))
    base = _plan(tmp_path, steps=steps)
    template = base.leaves["leaf"]
    leaves = {
        f"leaf-{index:02d}": type(template)(
            **{
                **template.__dict__,
                "leaf_id": f"leaf-{index:02d}",
                "identity": f"leaf-{index:02d}",
            }
        )
        for index in range(90)
    }
    plan = StoragePlan(
        **{
            **base.__dict__,
            "leaves": leaves,
            "waves": tuple(
                tuple(list(leaves)[index : index + 8])
                for index in range(0, len(leaves), 8)
            ),
            "outer_workers": 24,
            "wave_growth_bytes": tuple(1300 * 8 for _ in range(12)),
            "estimated_growth_bytes": 1300 * len(leaves),
            "identity": "large-performance-contract",
        }
    )
    coordinator = StorageAdmissionCoordinator(
        plan,
        disk_usage=lambda _path: SimpleNamespace(
            total=10_000_000,
            used=1_000,
            free=9_999_000,
        ),
    )
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf-00")
    latencies: list[float] = []
    started = time.perf_counter()
    client.admit_step(steps[0], request_id="large:0")
    latencies.append(time.perf_counter() - started)
    for index, step in enumerate(steps[1:], start=1):
        started = time.perf_counter()
        client.admit_step(
            step,
            summary=StorageAccountingSummary(
                completed_step=steps[index - 1],
                materialized_bytes={},
            ),
            request_id=f"large:{index}",
        )
        latencies.append(time.perf_counter() - started)

    p95 = sorted(latencies)[int(len(latencies) * 0.95) - 1]
    ledger = json.loads(coordinator.ledger_path.read_text(encoding="utf-8"))
    assert p95 < 1.0
    assert ledger["precommit_request_count"] == len(steps)
    assert ledger["cumulative_precommit_latency_seconds"] > 0
    assert ledger["max_precommit_latency_seconds"] > 0


def test_close_waits_for_inflight_reconciliation_without_late_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path, steps=("step_00",)),
        disk_usage=lambda _path: _usage(),
    )
    estimator_started = threading.Event()

    def delayed_estimate(**_kwargs):
        estimator_started.set()
        time.sleep(1.1)
        return ProjectStorageEstimate(
            forcing_bytes=1,
            member_grid_bytes=1,
            point_bytes=1,
            restart_baseline_bytes=1,
            restart_transition_bytes=1,
            compact_timeseries_bytes=1,
            compact_grid_bytes=1,
        )

    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_project_storage_components",
        delayed_estimate,
    )
    server = StorageAdmissionServer(coordinator)
    errors: list[BaseException] = []

    def reconcile() -> None:
        try:
            server.client(leaf_id="leaf").reconcile_finalization(
                request_id="slow-reconcile"
            )
        except BaseException as exc:  # pragma: no cover - assertion captures it
            errors.append(exc)

    client_thread = threading.Thread(target=reconcile)
    client_thread.start()
    assert estimator_started.wait(timeout=1)
    server.close()
    client_thread.join(timeout=1)

    assert errors == []
    assert not server._thread.is_alive()
    assert not client_thread.is_alive()
    closed_snapshot = coordinator.snapshot()
    time.sleep(0.1)
    assert coordinator.snapshot() == closed_snapshot


def test_heartbeat_keeps_long_reconciliation_alive_beyond_base_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS",
        0.2,
    )
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path, steps=("step_00",)),
        disk_usage=lambda _path: _usage(),
    )

    def delayed_estimate(**_kwargs):
        time.sleep(0.9)
        return ProjectStorageEstimate(
            forcing_bytes=1,
            member_grid_bytes=1,
            point_bytes=1,
            restart_baseline_bytes=1,
            restart_transition_bytes=1,
            compact_timeseries_bytes=1,
            compact_grid_bytes=1,
        )

    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_project_storage_components",
        delayed_estimate,
    )
    with StorageAdmissionServer(coordinator) as server:
        snapshot = server.client(leaf_id="leaf").reconcile_finalization(
            request_id="long-reconcile"
        )
    assert snapshot.estimated_growth_bytes > 0


def test_unexpected_serve_death_stops_heartbeat_and_close_terminalizes_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS",
        0.2,
    )
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission._dispatch_coordinator_request",
        lambda *_args: (_ for _ in ()).throw(SystemExit("dispatch died")),
    )
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    server = StorageAdmissionServer(coordinator)
    errors: list[BaseException] = []
    client = server.client(leaf_id="leaf")

    def request() -> None:
        try:
            client.admit_step("step_00", request_id="fatal")
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=request)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert errors
    assert not server._heartbeat_thread.is_alive()
    server.close()
    assert not server._thread.is_alive()
    generation_dir = server._generation_dir
    assert not list(generation_dir.rglob("request.json"))
    assert not list(generation_dir.rglob("progress.*.json"))


def test_server_context_exit_waits_after_body_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path, steps=("step_00",)),
        disk_usage=lambda _path: _usage(),
    )
    estimator_started = threading.Event()
    release_estimator = threading.Event()

    def blocked_estimate(**_kwargs):
        estimator_started.set()
        assert release_estimator.wait(timeout=2)
        raise RuntimeError("estimator failed")

    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_project_storage_components",
        blocked_estimate,
    )
    server = StorageAdmissionServer(coordinator)
    client_errors: list[BaseException] = []

    def reconcile() -> None:
        try:
            server.client(leaf_id="leaf").reconcile_finalization(
                request_id="failed-reconcile"
            )
        except BaseException as exc:  # pragma: no cover - assertion captures it
            client_errors.append(exc)

    client_thread = threading.Thread(target=reconcile)
    try:
        with server:
            client_thread.start()
            assert estimator_started.wait(timeout=1)
            release_estimator.set()
            raise ValueError("body failed")
    except ValueError as exc:
        assert str(exc) == "body failed"
    client_thread.join(timeout=1)

    assert not server._thread.is_alive()
    assert not client_thread.is_alive()
    assert len(client_errors) == 1
    assert "estimator failed" in str(client_errors[0])


def test_stale_file_response_is_ignored(tmp_path: Path) -> None:
    coordinator = StorageAdmissionCoordinator(_plan(tmp_path), disk_usage=lambda _path: _usage())
    with StorageAdmissionServer(coordinator) as server:
        client = server.client(leaf_id="leaf")
        leaf_token = __import__("hashlib").sha256(b"leaf").hexdigest()[:16]
        response_token = __import__("hashlib").sha256(b"fresh").hexdigest()[:16]
        stale = client.ipc_dir / client.generation / leaf_token / f"response.{response_token}.json"
        stale.write_text(
            json.dumps(
                {
                    "generation": "stale",
                    "request_id": "fresh",
                    "payload_sha256": "stale",
                    "response": {"ok": True},
                }
            ),
            encoding="utf-8",
        )
        client.admit_step("step_00", request_id="fresh")
    assert not list((tmp_path / "project/results/storage/ipc").rglob("response.*.json"))


def test_server_close_removes_published_response_abandoned_by_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        storage_admission_mod,
        "STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS",
        0.05,
    )
    coordinator = StorageAdmissionCoordinator(
        _plan(tmp_path), disk_usage=lambda _path: _usage()
    )
    server = StorageAdmissionServer(coordinator)
    payload = {
        "kind": "step",
        "leaf_id": "leaf",
        "step_name": "step_00",
        "summary": None,
        "request_id": "abandoned",
        "allow_existing_step_drain": False,
    }
    nonce = "a" * 32
    request_path = server._request_paths["leaf"]
    write_manifest_atomic(
        request_path,
        {
            "generation": coordinator.generation,
            "leaf_id": "leaf",
            "route_id": "leaf",
            "request_id": "abandoned",
            "transport_nonce": nonce,
            "payload_sha256": storage_admission_mod.hash_json(payload),
            "payload": payload,
        },
    )
    response = request_path.parent / f"response.{nonce}.json"
    deadline = time.monotonic() + 1.0
    while not response.is_file() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert response.is_file()

    server.close()

    assert not response.exists()
    assert not list((tmp_path / "project/results/storage/ipc").rglob("*.json"))


def test_whole_ipc_request_timeout_is_bounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.STORAGE_ADMISSION_REQUEST_TIMEOUT_SECONDS",
        0.05,
    )
    before_threads = threading.active_count()
    client = StorageAdmissionClient(
        ipc_dir=tmp_path / "ipc",
        generation="dead-generation",
        leaf_id="leaf",
    )
    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="unavailable or timed out"):
        client.admit_step("step_00", request_id="timeout")
    assert time.perf_counter() - started < 0.5
    assert threading.active_count() == before_threads


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


def test_missing_ledger_keeps_full_reserve_and_replays_from_step_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup, project, reservation = _prepared_project(tmp_path)
    estimate = ProjectStorageEstimate(
        forcing_bytes=200,
        member_grid_bytes=0,
        point_bytes=0,
        restart_baseline_bytes=0,
        restart_transition_bytes=0,
        compact_timeseries_bytes=0,
        compact_grid_bytes=0,
    )
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_coordinated_storage_reserve",
        lambda projects, **_kwargs: (200, {str(projects[0].project_dir.resolve()): estimate}),
    )
    plan = build_storage_plan(
        root_project_dir=project,
        projects=(reservation,),
        outer_workers=1,
        leaf_ids=("project",),
    )

    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())

    leaf = coordinator.snapshot()["leaves"]["project"]
    assert leaf["last_admitted_step_index"] == -1
    assert leaf["last_completed_step_index"] == -1
    assert leaf["remaining_by_component"]["forcing_bytes"] == 200
    assert coordinator.snapshot()["status"] == "reconciled_legacy_partial"


def test_malformed_authoritative_member_manifest_refuses_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup, project, reservation = _prepared_project(tmp_path, malformed=True)
    estimate = ProjectStorageEstimate(
        forcing_bytes=200,
        member_grid_bytes=0,
        point_bytes=0,
        restart_baseline_bytes=0,
        restart_transition_bytes=0,
        compact_timeseries_bytes=0,
        compact_grid_bytes=0,
    )
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_coordinated_storage_reserve",
        lambda projects, **_kwargs: (200, {str(projects[0].project_dir.resolve()): estimate}),
    )

    with pytest.raises(RuntimeError, match="Malformed authoritative member manifest"):
        build_storage_plan(
            root_project_dir=project,
            projects=(reservation,),
            outer_workers=1,
            leaf_ids=("project",),
        )


def test_finalization_reconciliation_never_lowers_aggregate_obligation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path, steps=("step_00",))
    coordinator = StorageAdmissionCoordinator(plan, disk_usage=lambda _path: _usage())
    client = StorageAdmissionClient.in_process(coordinator, leaf_id="leaf")
    client.admit_step("step_00", request_id="final-reconcile-admit")
    before = dict(
        coordinator.snapshot()["leaves"]["leaf"]["remaining_by_component"]
    )
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission.estimate_project_storage_components",
        lambda **_kwargs: ProjectStorageEstimate(
            forcing_bytes=1,
            member_grid_bytes=1,
            point_bytes=1,
            restart_baseline_bytes=1,
            restart_transition_bytes=1,
            compact_timeseries_bytes=1,
            compact_grid_bytes=1,
        ),
    )

    client.reconcile_finalization(request_id="final-reconcile")

    assert coordinator.snapshot()["leaves"]["leaf"]["remaining_by_component"] == before


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
        json.dumps(
            {
                "status": "success",
                "project_dir": str(leaf.project_dir),
                "scientific_identity": leaf.identity,
            }
        ),
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
                    "scientific_identity": leaves[leaf_id].identity,
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
                        "scientific_identity": plan.leaves[leaf_id].identity,
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
