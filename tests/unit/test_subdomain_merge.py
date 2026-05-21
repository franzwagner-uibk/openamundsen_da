from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openamundsen_da.subdomain import merge as merge_mod


def test_merge_grids_uses_compact_da_summary(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_project_dir = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    compact_da = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
    compact_da.parent.mkdir(parents=True, exist_ok=True)
    compact_da.write_bytes(b"compact")

    sub = SimpleNamespace(id="sd_01", project_dir=sub_project_dir)
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "full")
    write_calls: list[dict] = []
    monkeypatch.setattr(merge_mod, "write_da_output_grids", lambda **kwargs: write_calls.append(kwargs))

    calls: list[tuple[str, list[Path]]] = []

    def _fake_merge_netcdf(*, output_name: str, nc_paths, out_dir: Path, **kwargs):
        source_paths = [Path(path) for _, path in nc_paths]
        calls.append((output_name, source_paths))
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"merged")
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    written = merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    merged_da = project_dir / "results" / "grids" / "da_output_grids.nc"
    assert merged_da.is_file()
    assert written == [merged_da]
    assert calls == [("da_output_grids.nc", [compact_da])]
    assert write_calls == []


def test_merge_grids_skips_latest_member_outputs_when_compact_summary_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_project_dir = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    compact_da = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
    compact_da.parent.mkdir(parents=True, exist_ok=True)
    compact_da.write_bytes(b"compact")

    latest_open_loop = tmp_path / "open_loop" / "results"
    latest_member = tmp_path / "member_001" / "results"
    latest_open_loop.mkdir(parents=True)
    latest_member.mkdir(parents=True)
    (latest_open_loop / "output_grids.nc").write_bytes(b"open-loop")
    (latest_member / "output_grids.nc").write_bytes(b"member")

    sub = SimpleNamespace(id="sd_01", project_dir=sub_project_dir)
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "full")
    monkeypatch.setattr(
        merge_mod,
        "_result_sources",
        lambda _sub: [("open_loop", latest_open_loop), ("member_001", latest_member)],
    )

    write_calls: list[dict] = []
    monkeypatch.setattr(merge_mod, "write_da_output_grids", lambda **kwargs: write_calls.append(kwargs))
    merge_calls: list[str] = []

    def _fake_merge_netcdf(*, output_name: str, out_dir: Path, **kwargs):
        merge_calls.append(output_name)
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"merged {output_name}".encode("utf-8"))
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    assert (project_dir / "results" / "grids" / "da_output_grids.nc").read_bytes() == b"merged da_output_grids.nc"
    assert not (project_dir / "results" / "grids" / "output_grids.nc").exists()
    assert not (project_dir / "results" / "grids" / "member_001_output_grids.nc").exists()
    assert merge_calls == ["da_output_grids.nc"]
    assert write_calls == []


def test_merge_grids_does_not_inspect_raw_latest_steps_when_compact_summaries_exist(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    subdomains = {}
    compact_paths = []
    for sid in ("sd_01", "sd_02"):
        sub_project_dir = project_dir / "subdomains" / sid / "projects" / "project_2022_2023"
        compact_da = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
        compact_da.parent.mkdir(parents=True, exist_ok=True)
        compact_da.write_bytes(f"compact {sid}".encode("utf-8"))
        compact_paths.append(compact_da)
        subdomains[sid] = SimpleNamespace(id=sid, project_dir=sub_project_dir)

    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains=subdomains,
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "full")
    monkeypatch.setattr(
        merge_mod,
        "_result_sources",
        lambda _sub: (_ for _ in ()).throw(AssertionError("raw latest steps should not be inspected")),
    )
    monkeypatch.setattr(merge_mod, "write_da_output_grids", lambda **_kwargs: None)

    calls: list[tuple[str, list[Path]]] = []

    def _fake_merge_netcdf(*, output_name: str, nc_paths, out_dir: Path, **kwargs):
        del kwargs
        source_paths = [Path(path) for _, path in nc_paths]
        calls.append((output_name, source_paths))
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"merged")
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    written = merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    assert written == [project_dir / "results" / "grids" / "da_output_grids.nc"]
    assert calls == [("da_output_grids.nc", compact_paths)]


def test_merge_grids_rejects_partial_compact_da_summary_availability(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_01_project = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    sub_02_project = project_dir / "subdomains" / "sd_02" / "projects" / "project_2022_2023"
    compact_da = sub_01_project / "results" / "grids" / "da_output_grids.nc"
    compact_da.parent.mkdir(parents=True, exist_ok=True)
    compact_da.write_bytes(b"compact")

    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={
            "sd_01": SimpleNamespace(id="sd_01", project_dir=sub_01_project),
            "sd_02": SimpleNamespace(id="sd_02", project_dir=sub_02_project),
        },
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )

    with pytest.raises(FileNotFoundError, match="sd_02"):
        merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")


def test_merge_grids_recomputes_da_summary_when_no_compact_da_summary_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_project_dir = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    latest_open_loop = tmp_path / "open_loop" / "results"
    latest_member = tmp_path / "member_001" / "results"
    latest_open_loop.mkdir(parents=True)
    latest_member.mkdir(parents=True)
    (latest_open_loop / "output_grids.nc").write_bytes(b"open-loop")
    (latest_member / "output_grids.nc").write_bytes(b"member")

    sub = SimpleNamespace(id="sd_01", project_dir=sub_project_dir)
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "full")
    monkeypatch.setattr(
        merge_mod,
        "_result_sources",
        lambda _sub: [("open_loop", latest_open_loop), ("member_001", latest_member)],
    )

    def _fake_merge_netcdf(*, output_name: str, out_dir: Path, **kwargs):
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"merged {output_name}".encode("utf-8"))
        return out_path

    write_calls: list[dict] = []

    def _fake_write_da_output_grids(**kwargs):
        write_calls.append(kwargs)
        Path(kwargs["output_nc"]).write_bytes(b"summary")
        return Path(kwargs["output_nc"])

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)
    monkeypatch.setattr(merge_mod, "write_da_output_grids", _fake_write_da_output_grids)

    merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    assert len(write_calls) == 1
    assert write_calls[0]["open_loop_nc"] == project_dir / "results" / "grids" / "output_grids.nc"
    assert write_calls[0]["member_ncs"] == [project_dir / "results" / "grids" / "member_001_output_grids.nc"]
    assert write_calls[0]["output_nc"] == project_dir / "results" / "grids" / "da_output_grids.nc"


def test_merge_grids_can_defer_compact_cleanup(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_project_dir = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    compact_da = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
    compact_da.parent.mkdir(parents=True, exist_ok=True)
    compact_da.write_bytes(b"compact")

    sub = SimpleNamespace(id="sd_01", project_dir=sub_project_dir)
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "compact")
    monkeypatch.setattr(
        merge_mod,
        "delete_files",
        lambda paths: (_ for _ in ()).throw(AssertionError("compact cleanup should be deferred")),
    )

    def _fake_merge_netcdf(*, output_name: str, out_dir: Path, **kwargs):
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"merged {output_name}".encode("utf-8"))
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json", defer_compact_cleanup=True)

    assert (project_dir / "results" / "grids" / "da_output_grids.nc").is_file()


def test_merge_grids_does_not_cleanup_compact_artifacts_by_default(monkeypatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    sub_project_dir = project_dir / "subdomains" / "sd_01" / "projects" / "project_2022_2023"
    compact_da = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
    compact_da.parent.mkdir(parents=True, exist_ok=True)
    compact_da.write_bytes(b"compact")

    sub = SimpleNamespace(id="sd_01", project_dir=sub_project_dir)
    manifest = SimpleNamespace(
        run_mode="subdomain",
        project_dir=project_dir,
        subdomains={"sd_01": sub},
        grid_rows=1,
        grid_cols=1,
        grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0),
        crs="EPSG:31254",
    )

    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "ensure_run_mode", lambda *args, **kwargs: "subdomain")
    monkeypatch.setattr(
        merge_mod,
        "_expected_coverage_mask",
        lambda *_args, **_kwargs: np.zeros((1, 1), dtype=bool),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "compact")
    monkeypatch.setattr(
        merge_mod,
        "cleanup_deferred_compact_grid_artifacts",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("cleanup must be explicit")),
    )

    def _fake_merge_netcdf(*, output_name: str, out_dir: Path, **kwargs):
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"merged {output_name}".encode("utf-8"))
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    assert (project_dir / "results" / "grids" / "da_output_grids.nc").is_file()


def test_cleanup_deferred_compact_grid_artifacts_archives_raw_support_files(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    grids_dir = project_dir / "results" / "grids"
    grids_dir.mkdir(parents=True)
    keep = grids_dir / "da_output_grids.nc"
    merged_open_loop = grids_dir / "output_grids.nc"
    merged_member = grids_dir / "member_001_output_grids.nc"
    merged_tif = grids_dir / "snow_depth.tif"
    subdomain_artifact = project_dir / "subdomains" / "sd_01" / "steps" / "step_01" / "output_grids.nc"
    subdomain_artifact.parent.mkdir(parents=True)
    for path in (keep, merged_open_loop, merged_member, merged_tif, subdomain_artifact):
        path.write_bytes(b"data")
    report = project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"pdf")
    da_map = project_dir / "results" / "maps" / "da_events" / "da_1.png"
    da_map.parent.mkdir(parents=True)
    da_map.write_bytes(b"map")
    lock = merge_mod.compact_cleanup_ready_lock_path(project_dir)
    lock.parent.mkdir(parents=True)
    lock.write_text("ready\n", encoding="utf-8")

    manifest = SimpleNamespace(project_dir=project_dir)
    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "compact")
    monkeypatch.setattr(
        merge_mod,
        "load_assimilation_events",
        lambda _project_dir: [SimpleNamespace(date="2022-10-01", variable="scf", product="test")],
    )
    monkeypatch.setattr(
        merge_mod,
        "collect_subdomain_grid_artifacts",
        lambda _project_dir: [subdomain_artifact],
    )
    archive_dir = tmp_path / "archive"

    archived, _bytes_staged = merge_mod.cleanup_deferred_compact_grid_artifacts(
        manifest_path=tmp_path / "manifest.json",
        out_dir=grids_dir,
        archive_dir=archive_dir,
    )

    assert archived == 4
    assert keep.is_file()
    assert not merged_open_loop.exists()
    assert not merged_member.exists()
    assert not merged_tif.exists()
    assert not subdomain_artifact.exists()
    assert (archive_dir / "results" / "grids" / "output_grids.nc").is_file()
    assert (archive_dir / "results" / "grids" / "member_001_output_grids.nc").is_file()
    assert (archive_dir / "results" / "grids" / "snow_depth.tif").is_file()
    assert (archive_dir / "subdomains" / "sd_01" / "steps" / "step_01" / "output_grids.nc").is_file()


def test_cleanup_deferred_compact_grid_artifacts_refuses_without_ready_lock(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    grids_dir = project_dir / "results" / "grids"
    grids_dir.mkdir(parents=True)
    (grids_dir / "da_output_grids.nc").write_bytes(b"data")
    report = project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"pdf")
    da_map = project_dir / "results" / "maps" / "da_events" / "da_1.png"
    da_map.parent.mkdir(parents=True)
    da_map.write_bytes(b"map")

    manifest = SimpleNamespace(project_dir=project_dir)
    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(merge_mod, "output_retention_mode", lambda *_args, **_kwargs: "compact")
    monkeypatch.setattr(
        merge_mod,
        "load_assimilation_events",
        lambda _project_dir: [SimpleNamespace(date="2022-10-01", variable="scf", product="test")],
    )

    with pytest.raises(merge_mod.CompactCleanupSafetyError, match="artifact_cleanup_allowed"):
        merge_mod.cleanup_deferred_compact_grid_artifacts(
            manifest_path=tmp_path / "manifest.json",
            out_dir=grids_dir,
        )
