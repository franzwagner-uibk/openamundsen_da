import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.exceptions import LowDiskPauseError
from openamundsen_da.io.paths import list_steps_sorted, read_step_config
from openamundsen_da.manifests import inventory_digest, write_manifest_atomic
from openamundsen_da.pipeline.project_skeleton import plan_project_steps
from openamundsen_da.subdomain import manifest as manifest_mod
from openamundsen_da.subdomain import run as run_mod
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec
from openamundsen_da.util.storage_budget import ProjectStorageEstimate
from openamundsen_da.util.storage_admission import (
    StorageAdmissionClient,
    StorageAdmissionCoordinator,
    StorageLeafPlan,
    StoragePlan,
)


def _single_subdomain_manifest(tmp_path: Path) -> tuple[SubdomainManifest, Path]:
    setup_dir = tmp_path / "subdomains" / "S1"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    setup_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    setup_yaml = setup_dir / "S1.yml"
    project_yaml = project_dir / "project_2022_2023.yml"
    setup_yaml.write_text("domain: test\ntimestep: 1D\n", encoding="utf-8")
    project_yaml.write_text(
        "start_date: 2023-01-01\nend_date: 2023-01-03\n"
        "data_assimilation:\n  assimilation_events:\n"
        "    - {date: '2023-01-02', variable: station_hs}\n",
        encoding="utf-8",
    )

    sub = SubdomainMeta(
        id="S1",
        label="Subdomain 1",
        setup_dir=setup_dir,
        setup_yaml=setup_yaml,
        project_dir=project_dir,
        project_yaml=project_yaml,
        project_name="project_2022_2023",
        grids_dir=setup_dir / "grids",
        meteo_dir=setup_dir / "meteo",
        obs_stations_dir=setup_dir / "obs" / "stations",
        roi_raster_path=setup_dir / "grids" / "roi.asc",
        roi_vector_path=setup_dir / "env" / "roi.gpkg",
        window=WindowSpec(row_off=0, col_off=0, height=1, width=1),
        transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:25832",
    )
    manifest = SubdomainManifest(
        run_mode="subdomain",
        setup_dir=tmp_path / "subdomains",
        project_dir=tmp_path / "subdomains" / "projects" / "project_2022_2023",
        project_name="project_2022_2023",
        setup_yaml=tmp_path / "subdomains" / "subdomains.yml",
        project_yaml=tmp_path / "subdomains" / "projects" / "project_2022_2023" / "project_2022_2023.yml",
        subdomain_root=tmp_path / "subdomains" / "projects" / "project_2022_2023" / "subdomains",
        regions_path=tmp_path / "subdomains" / "env" / "subdomains.gpkg",
        id_field="id",
        crs="EPSG:25832",
        grid_rows=1,
        grid_cols=1,
        grid_transform=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
        grid_resolution=100.0,
        grid_domain="test",
        clip_mode="window",
        station_buffer_m=10_000.0,
        roi_buffer_m=0.0,
        grid_buffer_m=10_000.0,
        raw_snowcover_dir=tmp_path / "subdomains" / "obs" / "snowcover",
        raw_wetsnow_dir=tmp_path / "subdomains" / "obs" / "wetsnow",
        subdomains={"S1": sub},
    )
    manifest.project_dir.mkdir(parents=True)
    manifest.setup_yaml.write_text("domain: parent\n", encoding="utf-8")
    manifest.project_yaml.write_text(
        "run_mode: subdomain\nstart_date: 2023-01-01\nend_date: 2023-01-02\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    return manifest, manifest_path


@pytest.mark.parametrize("partial", ["empty", "yaml_less", "malformed"])
def test_prepare_obs_rebuilds_partial_step_skeleton_without_runtime_evidence(
    tmp_path: Path,
    partial: str,
) -> None:
    manifest, _ = _single_subdomain_manifest(tmp_path)
    sub = manifest.subdomains["S1"]
    expected = plan_project_steps(sub.setup_dir, sub.project_dir)
    steps_root = sub.project_dir / "steps"
    steps_root.mkdir()
    if partial != "empty":
        partial_step = steps_root / expected[0].name
        partial_step.mkdir()
        if partial == "malformed":
            (partial_step / "00.yml").write_text(
                "start_date: [unterminated\n",
                encoding="utf-8",
            )

    run_mod._prepare_obs_for_subdomain(
        sub,
        manifest,
        overwrite=False,
        scientific_identity="scientific-v1",
    )

    actual_steps = list_steps_sorted(sub.project_dir)
    assert [path.name for path in actual_steps] == [plan.name for plan in expected]
    for path, plan in zip(actual_steps, expected, strict=True):
        config = read_step_config(path)
        assert run_mod.parse_datetime_opt(str(config["start_date"])) == plan.start
        assert run_mod.parse_datetime_opt(str(config["end_date"])) == plan.end
    preparation = run_mod.load_manifest(
        sub.setup_dir / ".openamundsen-da/manifests/leaf_preparation.json"
    )
    assert preparation is not None
    assert preparation["status"] == "success"
    assert preparation["scientific_identity"] == "scientific-v1"


def test_prepare_obs_rejects_partial_step_skeleton_with_runtime_evidence(
    tmp_path: Path,
) -> None:
    manifest, _ = _single_subdomain_manifest(tmp_path)
    sub = manifest.subdomains["S1"]
    partial_step = sub.project_dir / "steps/step_00_init"
    partial_step.mkdir(parents=True)
    (partial_step / "00.yml").write_text(
        "start_date: [unterminated\n",
        encoding="utf-8",
    )
    member_run = partial_step / "ensembles/prior/member_001/results/member_run.json"
    member_run.parent.mkdir(parents=True)
    member_run.write_text('{"status": "success"}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="runtime evidence"):
        run_mod._prepare_obs_for_subdomain(
            sub,
            manifest,
            overwrite=False,
            scientific_identity="scientific-v1",
        )


def test_prepare_obs_rejects_malformed_steps_with_completed_prep_authority(
    tmp_path: Path,
) -> None:
    manifest, _ = _single_subdomain_manifest(tmp_path)
    sub = manifest.subdomains["S1"]
    partial_step = sub.project_dir / "steps/step_00_init"
    partial_step.mkdir(parents=True)
    (partial_step / "00.yml").write_text(
        "start_date: [unterminated\n",
        encoding="utf-8",
    )
    preparation_path = (
        sub.setup_dir / ".openamundsen-da/manifests/leaf_preparation.json"
    )
    write_manifest_atomic(
        preparation_path,
        {
            "status": "success",
            "scientific_identity": "scientific-v1",
            "outputs": [],
            "output_digest": inventory_digest([]),
        },
    )

    with pytest.raises(RuntimeError, match="inventory changed"):
        run_mod._prepare_obs_for_subdomain(
            sub,
            manifest,
            overwrite=False,
            scientific_identity="scientific-v1",
        )


def test_missing_ledger_finalized_leaf_skips_prep_and_step_zero_preadmission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, _ = _single_subdomain_manifest(tmp_path)
    done = manifest.subdomains["S1"]
    interrupted_setup = tmp_path / "subdomains/S2"
    interrupted_project = interrupted_setup / "projects/project_2022_2023"
    interrupted_project.mkdir(parents=True)
    interrupted = SimpleNamespace(
        id="S2",
        setup_dir=interrupted_setup,
        project_dir=interrupted_project,
    )
    manifest.subdomains["S2"] = interrupted
    obligations = {
        "forcing_bytes": 100,
        "member_grid_bytes": 0,
        "point_bytes": 0,
        "restart_baseline_bytes": 0,
        "restart_transition_bytes": 0,
        "compact_timeseries_bytes": 0,
        "compact_grid_bytes": 0,
        "map_support_bytes": 0,
        "derived_forcing_plot_bytes": 0,
        "retained_diagnostics_bytes": 0,
    }
    leaves = {
        sid: StorageLeafPlan(
            leaf_id=sid,
            setup_dir=sub.setup_dir,
            project_dir=sub.project_dir,
            step_names=("step_00",),
            obligations=obligations,
            step_obligations={"step_00": obligations},
            queued_retained_bytes=0,
            identity=f"identity-{sid}",
        )
        for sid, sub in (("S1", done), ("S2", interrupted))
    }
    plan = StoragePlan(
        root_project_dir=manifest.project_dir,
        leaves=leaves,
        waves=(("S1", "S2"),),
        wave_growth_bytes=(200,),
        outer_workers=2,
        parent_finalization_reserve_bytes=0,
        estimated_growth_bytes=200,
        overwrite=False,
        filesystem_device=manifest.project_dir.stat().st_dev,
        filesystem_capacity_bytes=10_000,
        identity="mixed-missing-ledger",
        estimate_duration_seconds=0.0,
    )
    (done.setup_dir / "leaf_finalization_manifest.json").write_text(
        json.dumps(
            {
                "status": "success",
                "project_dir": str(done.project_dir),
                "scientific_identity": "identity-S1",
            }
        ),
        encoding="utf-8",
    )
    coordinator = StorageAdmissionCoordinator(
        plan,
        disk_usage=lambda _path: SimpleNamespace(total=10_000, used=1_000, free=9_000),
    )
    assert coordinator.snapshot()["leaves"]["S1"]["phase"] == "finalized"
    coordinator.admit_wave(0, request_id="wave:0")
    monkeypatch.setattr(
        run_mod,
        "_prepare_obs_for_subdomain",
        lambda sub, *_args, **_kwargs: prepared.append(sub.id),
    )
    monkeypatch.setattr(coordinator, "_validate_leaf_preparation", lambda _sid: None)
    monkeypatch.setattr(
        run_mod,
        "_project_has_started",
        lambda _project: False,
    )
    monkeypatch.setattr(
        "openamundsen_da.util.storage_admission._project_identity",
        lambda project, *_args, **_kwargs: f"identity-{project.setup_dir.name}",
    )
    prepared: list[str] = []
    admitted: list[tuple[str, str]] = []

    class _WaveServer:
        def __init__(self) -> None:
            self.coordinator = coordinator

        def client(self, *, leaf_id: str):
            client = StorageAdmissionClient.in_process(coordinator, leaf_id=leaf_id)

            class _Client:
                leaf_identity = client.leaf_identity

                @staticmethod
                def admit_step(step_name: str, **kwargs):
                    admitted.append((leaf_id, step_name))
                    return client.admit_step(step_name, **kwargs)

            return _Client()

    run_mod._prepare_and_preadmit_wave(
        manifest=manifest,
        wave_server=_WaveServer(),
        wave_index=0,
        wave_ids=["S1", "S2"],
        overwrite=False,
    )

    assert prepared == ["S2"]
    assert admitted == [("S2", "step_00")]
    state = coordinator.snapshot()["leaves"]
    assert state["S1"]["phase"] == "finalized"
    assert state["S2"]["last_admitted_step_index"] == 0


@pytest.mark.parametrize("mutated", ["parent", "leaf"])
def test_leaf_acquisition_identity_rejects_post_copy_mutation(
    tmp_path: Path,
    mutated: str,
) -> None:
    manifest, _ = _single_subdomain_manifest(tmp_path)
    sub = manifest.subdomains["S1"]
    sub.project_yaml.write_text(
        sub.project_yaml.read_text(encoding="utf-8").replace(
            "    - {date: '2023-01-02', variable: station_hs}\n",
            "    - {date: '2023-01-02', variable: scf}\n",
        )
        + "obs:\n  snowcover:\n"
        "    product_tag: MODIS\n"
        "    acquisition_manifest: obs/satellite_acquisition_times.csv\n",
        encoding="utf-8",
    )
    parent_support = manifest.setup_dir / "obs/satellite_acquisition_times.csv"
    leaf_support = sub.setup_dir / "obs/satellite_acquisition_times.csv"
    parent_support.parent.mkdir(parents=True)
    leaf_support.parent.mkdir(parents=True)
    payload = "source,acquisition_time\nscene.tif,2023-01-01T10:00:00Z\n"
    parent_support.write_text(payload, encoding="utf-8")
    leaf_support.write_text(payload, encoding="utf-8")
    manifest.raw_snowcover_dir.mkdir(parents=True)

    bound_paths = run_mod._leaf_scientific_input_paths(manifest, "S1")
    assert parent_support in bound_paths
    assert leaf_support in bound_paths
    target = parent_support if mutated == "parent" else leaf_support
    target.write_text(payload + "changed.tif,2023-01-02T10:00:00Z\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="differs from.*parent"):
        run_mod._leaf_scientific_input_paths(manifest, "S1")


def test_manifest_save_preserves_existing_file_when_new_write_fails(tmp_path, monkeypatch):
    manifest, manifest_path = _single_subdomain_manifest(tmp_path)
    original = manifest_path.read_text(encoding="utf-8")

    manifest.subdomains["S1"].status = "success"

    def _fail_dump(*_args, **_kwargs):
        raise RuntimeError("interrupted manifest write")

    monkeypatch.setattr(manifest_mod.json, "dump", _fail_dump)

    with pytest.raises(RuntimeError, match="interrupted manifest write"):
        manifest.save(manifest_path)

    assert manifest_path.read_text(encoding="utf-8") == original
    loaded = SubdomainManifest.load(manifest_path)
    assert loaded.subdomains["S1"].status == "pending"
    assert not list(manifest_path.parent.glob(f".{manifest_path.name}.*.tmp"))


def test_manifest_round_trip_preserves_stage_state(tmp_path):
    manifest, manifest_path = _single_subdomain_manifest(tmp_path)
    manifest.stages["merge"] = {
        "status": "interrupted",
        "updated_at": "2026-07-15T12:00:00+00:00",
        "error": "worker stopped",
    }
    manifest.save(manifest_path)

    loaded = SubdomainManifest.load(manifest_path)

    assert loaded.stages["merge"] == manifest.stages["merge"]


def test_run_one_caps_project_plot_workers_to_inner_workers(tmp_path, monkeypatch):
    _, manifest_path = _single_subdomain_manifest(tmp_path)

    captured = {}

    monkeypatch.setattr(run_mod, "_prepare_obs_for_subdomain", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "run_project", lambda cfg: captured.setdefault("cfg", cfg))
    monkeypatch.setattr(
        run_mod,
        "_finalize_leaf",
        lambda *_args, **_kwargs: {"retained_leaf_bytes": 10, "cleanup_freed_bytes": 20},
    )

    result = run_mod._run_one(
        "S1",
        manifest_path,
        inner_max_workers=6,
        overwrite=True,
        retries=0,
        log_level="INFO",
        root_log_path=None,
        storage_reservation_projects=(),
        storage_outer_workers=2,
        shared_storage_reserve_bytes=123456,
    )

    assert result.status == "success"
    assert captured["cfg"].max_workers == 6
    assert captured["cfg"].plot_workers == 6
    assert captured["cfg"].storage_outer_workers == 2
    assert captured["cfg"].shared_storage_reserve_bytes == 123456


def test_coordinator_reserves_largest_active_leaf_transitions(monkeypatch, tmp_path):
    manifest, _ = _single_subdomain_manifest(tmp_path)
    template = manifest.subdomains["S1"]
    manifest.subdomains = {
        "S1": template,
        "S2": SimpleNamespace(
            setup_dir=tmp_path / "S2",
            project_dir=tmp_path / "P2",
            window=WindowSpec(row_off=0, col_off=0, height=2, width=2),
        ),
        "S3": SimpleNamespace(
            setup_dir=tmp_path / "S3",
            project_dir=tmp_path / "P3",
            window=WindowSpec(row_off=0, col_off=0, height=3, width=3),
        ),
    }
    for path in (tmp_path / "S2", tmp_path / "S3", tmp_path / "P2", tmp_path / "P3"):
        path.mkdir()
    estimates_by_project = {
        str(template.project_dir.resolve()): ProjectStorageEstimate(1, 2, 3, 4, 5, 6, 7),
        str((tmp_path / "P2").resolve()): ProjectStorageEstimate(2, 3, 4, 5, 6, 7, 8),
        str((tmp_path / "P3").resolve()): ProjectStorageEstimate(3, 4, 5, 6, 7, 8, 9),
    }
    monkeypatch.setattr(run_mod, "estimate_parent_compact_merge_bytes", lambda **_kwargs: 100)
    monkeypatch.setattr(run_mod, "estimate_parent_render_bytes", lambda **_kwargs: 0)
    monkeypatch.setattr(run_mod, "_leaf_scientific_input_paths", lambda *_args: ())
    monkeypatch.setattr(
        run_mod,
        "estimate_coordinated_storage_reserve",
        lambda projects, **_kwargs: (
            500,
            {
                str(project.project_dir): estimates_by_project[str(project.project_dir)]
                for project in projects
                if not (
                    project.run_manifest
                    and project.run_manifest.is_file()
                    and json.loads(project.run_manifest.read_text(encoding="utf-8"))["status"]
                    == "success"
                )
            },
        ),
    )

    concurrent, leaves, projects, merge, queued = run_mod._coordinator_storage_reserve(
        manifest,
        selected_ids=["S1", "S2", "S3"],
        outer_workers=2,
        overwrite=False,
    )

    assert leaves == {
        "S1": estimates_by_project[str(template.project_dir.resolve())].total_bytes,
        "S2": estimates_by_project[str((tmp_path / "P2").resolve())].total_bytes,
        "S3": estimates_by_project[str((tmp_path / "P3").resolve())].total_bytes,
    }
    assert concurrent == 500
    assert len(projects) == 3
    assert merge == 100
    assert queued == 0

    (tmp_path / "S3" / "run_manifest.json").write_text(
        json.dumps({"status": "success"}),
        encoding="utf-8",
    )
    concurrent, leaves, _projects, _merge, _queued = run_mod._coordinator_storage_reserve(
        manifest,
        selected_ids=["S1", "S2", "S3"],
        outer_workers=2,
        overwrite=False,
    )
    assert leaves["S3"] == 0
    assert concurrent == 500


def test_leaf_queue_is_admitted_in_bounded_deterministic_waves():
    assert run_mod._leaf_waves(["S3", "S1", "S2", "S5", "S4"], 2) == [
        ["S3", "S1"],
        ["S2", "S5"],
        ["S4"],
    ]


def test_coordinator_reserves_compact_outputs_of_queued_leaves(monkeypatch, tmp_path):
    manifest, _ = _single_subdomain_manifest(tmp_path)
    template = manifest.subdomains["S1"]
    for sid in ("S2", "S3"):
        setup_dir = tmp_path / sid
        project_dir = setup_dir / "projects" / "demo"
        project_dir.mkdir(parents=True)
        manifest.subdomains[sid] = SimpleNamespace(
            setup_dir=setup_dir,
            project_dir=project_dir,
            window=WindowSpec(row_off=0, col_off=0, height=1, width=1),
        )
    retained = {
        str(manifest.subdomains["S2"].project_dir.resolve()): ProjectStorageEstimate(
            1, 2, 3, 4, 5, 50, 60, 7,
            retained_diagnostics_bytes=11,
        ),
        str(manifest.subdomains["S3"].project_dir.resolve()): ProjectStorageEstimate(
            1, 2, 3, 4, 5, 70, 80, 9,
            retained_diagnostics_bytes=13,
        ),
    }
    monkeypatch.setattr(run_mod, "estimate_parent_compact_merge_bytes", lambda **_kwargs: 100)
    monkeypatch.setattr(run_mod, "estimate_parent_render_bytes", lambda **_kwargs: 10)
    monkeypatch.setattr(
        run_mod,
        "estimate_project_storage_components",
        lambda *, project_dir, **_kwargs: retained[str(project_dir)],
    )
    monkeypatch.setattr(run_mod, "output_retention_mode", lambda _project: "compact")
    captured = {}
    monkeypatch.setattr(
        run_mod,
        "estimate_coordinated_storage_reserve",
        lambda projects, **kwargs: (
            captured.setdefault(
                "shared", kwargs["parent_finalization_reserve_bytes"]
            ),
            {
                str(template.project_dir.resolve()): ProjectStorageEstimate(
                    1, 2, 3, 4, 5, 6, 7
                )
            },
        ),
    )

    _total, _leaves, _projects, parent, queued = run_mod._coordinator_storage_reserve(
        manifest,
        selected_ids=["S1"],
        queued_ids=["S2", "S3"],
        outer_workers=1,
        overwrite=False,
    )

    expected_queued = sum(estimate.retained_compact_bytes for estimate in retained.values())
    assert parent == 110
    assert queued == expected_queued
    assert captured["shared"] == parent + expected_queued


def test_queued_full_retention_reserves_all_future_growth(monkeypatch, tmp_path):
    manifest, _ = _single_subdomain_manifest(tmp_path)
    queued_setup = tmp_path / "S2"
    queued_project = queued_setup / "projects" / "demo"
    queued_project.mkdir(parents=True)
    manifest.subdomains["S2"] = SimpleNamespace(
        setup_dir=queued_setup,
        project_dir=queued_project,
        window=WindowSpec(row_off=0, col_off=0, height=1, width=1),
    )
    estimate = ProjectStorageEstimate(1, 2, 3, 4, 5, 6, 7, 8, 9)
    monkeypatch.setattr(
        run_mod,
        "estimate_project_storage_components",
        lambda **_kwargs: estimate,
    )
    monkeypatch.setattr(run_mod, "output_retention_mode", lambda _project: "full")

    projected = run_mod._projected_retained_compact_bytes(
        manifest,
        selected_ids=["S2"],
        overwrite=False,
    )

    assert projected == {"S2": estimate.total_bytes}


def test_coordinator_drops_parent_finalization_reserve_only_after_accepted_merge(
    monkeypatch,
    tmp_path,
):
    manifest, _ = _single_subdomain_manifest(tmp_path)
    merged = manifest.project_dir / "results" / "grids" / "da_output_grids.nc"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_bytes(b"accepted")
    manifest.stages["merge"] = {"status": "completed"}
    captured = {}
    monkeypatch.setattr(
        run_mod,
        "validate_compact_output_file",
        lambda **_kwargs: merged,
    )
    monkeypatch.setattr(run_mod, "estimate_parent_compact_merge_bytes", lambda **_kwargs: 999)
    monkeypatch.setattr(run_mod, "estimate_parent_render_bytes", lambda **_kwargs: 0)
    monkeypatch.setattr(
        run_mod,
        "estimate_coordinated_storage_reserve",
        lambda _projects, **kwargs: (
            captured.setdefault("merge", kwargs["parent_finalization_reserve_bytes"]),
            {},
        ),
    )

    _total, _leaves, _projects, reserve, _queued = run_mod._coordinator_storage_reserve(
        manifest,
        selected_ids=["S1"],
        outer_workers=1,
        overwrite=False,
    )
    assert reserve == 0
    assert captured["merge"] == 0

    _total, _leaves, _projects, reserve, _queued = run_mod._coordinator_storage_reserve(
        manifest,
        selected_ids=["S1"],
        outer_workers=1,
        overwrite=True,
    )
    assert reserve == 999


def test_coordinator_keeps_merge_reserve_for_invalid_completed_output(
    monkeypatch,
    tmp_path,
):
    manifest, _ = _single_subdomain_manifest(tmp_path)
    merged = manifest.project_dir / "results" / "grids" / "da_output_grids.nc"
    merged.parent.mkdir(parents=True, exist_ok=True)
    merged.write_bytes(b"corrupt")
    manifest.stages["merge"] = {"status": "completed"}
    monkeypatch.setattr(
        run_mod,
        "validate_compact_output_file",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("corrupt")),
    )
    monkeypatch.setattr(
        run_mod,
        "estimate_parent_compact_merge_bytes",
        lambda **_kwargs: 999,
    )
    monkeypatch.setattr(run_mod, "estimate_parent_render_bytes", lambda **_kwargs: 0)
    monkeypatch.setattr(
        run_mod,
        "estimate_coordinated_storage_reserve",
        lambda _projects, **kwargs: (
            kwargs["parent_finalization_reserve_bytes"],
            {},
        ),
    )

    total, _leaves, _projects, reserve, _queued = run_mod._coordinator_storage_reserve(
        manifest,
        selected_ids=["S1"],
        outer_workers=1,
        overwrite=False,
    )

    assert reserve == 999
    assert total == 999


def test_run_subdomains_refuses_coordinated_growth_before_workers(tmp_path, monkeypatch):
    manifest, manifest_path = _single_subdomain_manifest(tmp_path)
    project_spec = run_mod.StorageReservationProject(
        setup_dir=manifest.subdomains["S1"].setup_dir,
        project_dir=manifest.subdomains["S1"].project_dir,
        grid_cell_count=1,
    )
    monkeypatch.setattr(run_mod, "ensure_run_mode", lambda *_args, **_kwargs: "subdomain")
    monkeypatch.setattr(
        run_mod,
        "_projected_retained_compact_bytes",
        lambda *_args, **_kwargs: {"S1": 0},
    )
    monkeypatch.setattr(
        run_mod,
        "_parent_finalization_reserve",
        lambda *_args, **_kwargs: 100,
    )
    monkeypatch.setattr(
        run_mod,
        "build_storage_plan",
        lambda *_args, **_kwargs: SimpleNamespace(estimated_growth_bytes=600),
    )
    monkeypatch.setattr(
        run_mod,
        "_coordinator_storage_reserve",
        lambda *_args, **_kwargs: (600, {"S1": 500}, (project_spec,), 100, 0),
    )

    def _refuse(*_args, **kwargs):
        assert kwargs["estimated_growth_bytes"] == 600
        raise LowDiskPauseError("projected use would exceed 90%")

    monkeypatch.setattr(run_mod, "check_step_admission", _refuse)
    monkeypatch.setattr(
        run_mod.cf,
        "ProcessPoolExecutor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("workers must not start after failed admission")
        ),
    )

    with pytest.raises(LowDiskPauseError, match="exceed 90%"):
        run_mod.run_subdomains(
            manifest_path=manifest_path,
            max_workers=1,
            perf_monitor=False,
            log_to_file=False,
        )

    reloaded = SubdomainManifest.load(manifest_path)
    assert reloaded.stages["run"]["status"] == "paused_low_disk"


def test_run_one_resumes_failed_partial_subdomain_without_implicit_overwrite(tmp_path, monkeypatch):
    _, manifest_path = _single_subdomain_manifest(tmp_path)
    run_manifest = tmp_path / "subdomains" / "S1" / "run_manifest.json"
    run_manifest.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    captured = {}

    def _prepare(*args, **kwargs):
        captured["prepare_overwrite"] = kwargs["overwrite"]

    def _run_project(cfg):
        captured["project_overwrite"] = cfg.overwrite

    monkeypatch.setattr(run_mod, "_prepare_obs_for_subdomain", _prepare)
    monkeypatch.setattr(run_mod, "run_project", _run_project)
    monkeypatch.setattr(
        run_mod,
        "_finalize_leaf",
        lambda *_args, **_kwargs: {"retained_leaf_bytes": 10, "cleanup_freed_bytes": 20},
    )

    result = run_mod._run_one(
        "S1",
        manifest_path,
        inner_max_workers=6,
        overwrite=False,
        retries=0,
        log_level="INFO",
        root_log_path=None,
    )

    assert result.status == "success"
    assert captured["prepare_overwrite"] is False
    assert captured["project_overwrite"] is False


def test_run_one_finalizes_leaf_before_success_manifest(tmp_path, monkeypatch):
    _, manifest_path = _single_subdomain_manifest(tmp_path)
    order: list[str] = []

    monkeypatch.setattr(run_mod, "_prepare_obs_for_subdomain", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_mod, "run_project", lambda _cfg: order.append("run"))
    monkeypatch.setattr(
        run_mod,
        "_finalize_leaf",
        lambda *_args, **_kwargs: (
            order.append("finalize")
            or {"retained_leaf_bytes": 10, "cleanup_freed_bytes": 20}
        ),
    )
    real_write = run_mod._write_run_manifest

    def _record_manifest(path, data):
        if data.get("status") == "success":
            order.append("success")
        real_write(path, data)

    monkeypatch.setattr(run_mod, "_write_run_manifest", _record_manifest)

    result = run_mod._run_one(
        "S1",
        manifest_path,
        inner_max_workers=1,
        overwrite=True,
        retries=0,
        log_level="INFO",
        root_log_path=None,
    )

    assert result.status == "success"
    assert order == ["run", "finalize", "success"]


def test_failed_leaf_never_runs_final_cleanup(tmp_path, monkeypatch):
    _, manifest_path = _single_subdomain_manifest(tmp_path)
    monkeypatch.setattr(run_mod, "_prepare_obs_for_subdomain", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_mod,
        "run_project",
        lambda _cfg: (_ for _ in ()).throw(RuntimeError("propagation failed")),
    )
    monkeypatch.setattr(
        run_mod,
        "_finalize_leaf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("failed leaves must retain restart artifacts")
        ),
    )

    result = run_mod._run_one(
        "S1",
        manifest_path,
        inner_max_workers=1,
        overwrite=False,
        retries=0,
        log_level="INFO",
        root_log_path=None,
    )

    assert result.status == "failed"
    assert "propagation failed" in str(result.error)
