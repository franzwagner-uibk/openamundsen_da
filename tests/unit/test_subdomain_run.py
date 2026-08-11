import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from openamundsen_da.exceptions import LowDiskPauseError
from openamundsen_da.subdomain import manifest as manifest_mod
from openamundsen_da.subdomain import run as run_mod
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec
from openamundsen_da.util.storage_budget import ProjectStorageEstimate


def _single_subdomain_manifest(tmp_path: Path) -> tuple[SubdomainManifest, Path]:
    setup_dir = tmp_path / "subdomains" / "S1"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    setup_dir.mkdir(parents=True)
    project_dir.mkdir(parents=True)
    setup_yaml = setup_dir / "S1.yml"
    project_yaml = project_dir / "project_2022_2023.yml"
    setup_yaml.write_text("domain: test\n", encoding="utf-8")
    project_yaml.write_text("start_date: 2023-01-01\nend_date: 2023-01-02\n", encoding="utf-8")

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
