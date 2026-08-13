from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from openamundsen_da.subdomain import merge as merge_mod
from openamundsen_da.subdomain.manifest import WindowSpec
from openamundsen_da.util.storage_policy import da_summary_netcdf_encoding


@pytest.fixture(autouse=True)
def _bounded_merge_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        merge_mod,
        "estimate_parent_compact_merge_bytes",
        lambda **_kwargs: 1024,
    )
    monkeypatch.setattr(
        merge_mod,
        "estimate_parent_render_bytes",
        lambda **_kwargs: 512,
    )
    monkeypatch.setattr(
        merge_mod,
        "admit_storage_transition",
        lambda *_args, **_kwargs: SimpleNamespace(used_fraction=0.1),
    )


def _write_roi(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        transform=from_origin(0.0, float(data.shape[0]), 1.0, 1.0),
    ) as dst:
        dst.write(data.astype(np.uint8), 1)


@pytest.fixture(autouse=True)
def _ignore_stage_persistence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(merge_mod, "save_stage", lambda *args, **kwargs: None)
    monkeypatch.setattr(merge_mod, "resolve_subdomain_event_plan", lambda *args, **kwargs: [])


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
    validation_calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        merge_mod,
        "validate_compact_output_file",
        lambda *, project_dir, output_nc: validation_calls.append((Path(project_dir), Path(output_nc))),
    )

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
    assert validation_calls == [
        (project_dir, compact_da),
        (project_dir, merged_da),
    ]


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


def test_merge_netcdf_reapplies_compact_da_summary_encoding(tmp_path: Path) -> None:
    roi_path = tmp_path / "roi.tif"
    _write_roi(roi_path, np.ones((1, 2), dtype=np.uint8))
    source_nc = tmp_path / "sub" / "da_output_grids.nc"
    source_nc.parent.mkdir(parents=True)
    source = xr.Dataset(
        data_vars={
            "ens_mean_snowdepth_daily": xr.DataArray(
                np.array([[[1.234, 2.345]]], dtype=np.float32),
                dims=("time", "y", "x"),
                coords={"time": [np.datetime64("2023-01-01")], "y": [0.0], "x": [0.0, 1.0]},
            )
        },
        coords={"y": [0.0], "x": [0.0, 1.0]},
    )
    source.to_netcdf(source_nc, encoding=da_summary_netcdf_encoding(source))
    sub = SimpleNamespace(
        id="sd_01",
        roi_raster_path=roi_path,
        window=WindowSpec(row_off=0, col_off=0, height=1, width=2),
    )
    manifest = SimpleNamespace(grid_transform=(1.0, 0.0, 0.0, 0.0, -1.0, 1.0))
    out_dir = tmp_path / "merged"
    out_dir.mkdir()

    out = merge_mod._merge_netcdf(
        output_name="da_output_grids.nc",
        nc_paths=[(sub, source_nc)],
        global_shape=(1, 2),
        manifest=manifest,
        out_dir=out_dir,
        expected_mask=np.ones((1, 2), dtype=bool),
        sliver_tol_px=0,
    )

    with xr.open_dataset(out) as ds:
        np.testing.assert_allclose(ds["ens_mean_snowdepth_daily"].values, [[[1.234, 2.345]]], atol=0.001)
    with xr.open_dataset(out, decode_cf=False) as raw:
        var = raw["ens_mean_snowdepth_daily"]
        assert var.dtype == np.dtype("int16")
        assert var.attrs["scale_factor"] == np.float32(0.001)
        assert var.attrs["_FillValue"] == np.int16(-32768)
        assert var.encoding.get("zlib") is True
        assert var.encoding.get("shuffle") is True


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


def test_merge_grids_defers_compact_cleanup_until_render(monkeypatch, tmp_path: Path) -> None:
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
    def _fake_merge_netcdf(*, output_name: str, out_dir: Path, **kwargs):
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"merged {output_name}".encode("utf-8"))
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    assert (project_dir / "results" / "grids" / "da_output_grids.nc").is_file()
    assert compact_da.is_file()


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
        "_manifest_owned_compact_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cleanup must wait for render")),
    )

    def _fake_merge_netcdf(*, output_name: str, out_dir: Path, **kwargs):
        out_path = Path(out_dir) / output_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(f"merged {output_name}".encode("utf-8"))
        return out_path

    monkeypatch.setattr(merge_mod, "_merge_netcdf", _fake_merge_netcdf)

    merge_mod.merge_grids(manifest_path=tmp_path / "manifest.json")

    assert (project_dir / "results" / "grids" / "da_output_grids.nc").is_file()


def test_cleanup_deletes_only_manifest_owned_files_after_render(
    monkeypatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "projects" / "project_2022_2023"
    subdomain_root = project_dir / "subdomains"
    sub_project_dir = subdomain_root / "sd_01" / "projects" / "project_2022_2023"
    grids_dir = project_dir / "results" / "grids"
    grids_dir.mkdir(parents=True)
    keep = grids_dir / "da_output_grids.nc"
    merged_open_loop = grids_dir / "output_grids.nc"
    merged_member = grids_dir / "member_001_output_grids.nc"
    merged_tif = grids_dir / "snow_depth.tif"
    unlisted_merged_tif = grids_dir / "unlisted_reference.tif"
    compact_subdomain = sub_project_dir / "results" / "grids" / "da_output_grids.nc"
    leaf_project_yaml = sub_project_dir / "project_2022_2023.yml"
    leaf_project_yaml.parent.mkdir(parents=True, exist_ok=True)
    leaf_project_yaml.write_text(
        "data_assimilation:\n  output:\n    retention: compact\n",
        encoding="utf-8",
    )
    map_support = sub_project_dir / "results" / "grids" / "da_map_support.nc"
    subdomain_artifact = (
        sub_project_dir
        / "steps"
        / "step_01"
        / "ensembles"
        / "prior"
        / "member_000"
        / "results"
        / "output_grids.nc"
    )
    subdomain_artifact.parent.mkdir(parents=True)
    (subdomain_artifact.parent / "member_run.json").write_text(
        '{"member": "member_000", "status": "success"}\n', encoding="utf-8"
    )
    compact_subdomain.parent.mkdir(parents=True)
    unowned = subdomain_root / "not_in_manifest" / "output_grids.nc"
    unowned.parent.mkdir(parents=True)
    for path in (
        keep,
        merged_open_loop,
        merged_member,
        merged_tif,
        unlisted_merged_tif,
        compact_subdomain,
        map_support,
        subdomain_artifact,
        unowned,
    ):
        path.write_bytes(b"data")
    report = project_dir / "results" / "reports" / "project_report.pdf"
    report.parent.mkdir(parents=True)
    report.write_bytes(b"pdf")
    da_map = project_dir / "results" / "maps" / "da_events" / "da_1.png"
    da_map.parent.mkdir(parents=True)
    da_map.write_bytes(b"map")

    manifest = SimpleNamespace(
        project_dir=project_dir,
        subdomain_root=subdomain_root,
        subdomains={"sd_01": SimpleNamespace(id="sd_01", project_dir=sub_project_dir)},
        stages={
            "merge": {
                "status": "completed",
                "outputs": [
                    str(keep),
                    str(merged_open_loop),
                    str(merged_member),
                    str(merged_tif),
                ],
            },
            "render": {"status": "completed"},
        },
    )
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
        "openamundsen_da.pipeline.cleanup.load_assimilation_events",
        lambda _project_dir: [SimpleNamespace(date="2022-10-01", variable="scf")],
    )
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_map_support",
        lambda *_args, **_kwargs: map_support,
    )
    monkeypatch.setattr(
        "openamundsen_da.pipeline.cleanup.validate_project_da_output_grids",
        lambda *_args, **_kwargs: compact_subdomain,
    )
    monkeypatch.setattr(
        "openamundsen_da.methods.viz.maps.panel_renderers.project_da_map_support_fields",
        lambda *_args, **_kwargs: (
            ["2022-10-01"],
            {"scf_prior_probability": [np.zeros((1, 1))]},
            np.ones((1, 1), dtype=bool),
        ),
    )

    manifest_path = project_dir / "subdomain_manifest.json"
    manifest_path.write_text('{"stage": "merge"}\n', encoding="utf-8")
    deleted, bytes_freed = merge_mod.cleanup_compact_grid_artifacts(
        manifest_path=manifest_path,
        out_dir=grids_dir,
    )

    assert set(deleted) == {
        merged_open_loop.resolve(),
        merged_member.resolve(),
        merged_tif.resolve(),
        subdomain_artifact.resolve(),
    }
    assert bytes_freed == 4 * len(b"data")
    assert keep.is_file()
    assert compact_subdomain.is_file()
    assert unlisted_merged_tif.is_file()
    assert unowned.is_file()
    assert all(not path.exists() for path in deleted)


def test_cleanup_refuses_before_render_stage_completes(
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

    manifest = SimpleNamespace(
        project_dir=project_dir,
        subdomain_root=project_dir / "subdomains",
        subdomains={},
        stages={
            "merge": {"status": "completed"},
            "render": {"status": "interrupted"},
        },
    )
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

    with pytest.raises(merge_mod.CompactCleanupSafetyError, match="render.*interrupted"):
        merge_mod.cleanup_compact_grid_artifacts(
            manifest_path=tmp_path / "manifest.json",
            out_dir=grids_dir,
        )


def test_atomic_output_preserves_previous_merge_after_interruption(tmp_path: Path) -> None:
    output = tmp_path / "merged.nc"
    output.write_bytes(b"previous complete output")

    with pytest.raises(KeyboardInterrupt):
        with merge_mod._atomic_output(output) as partial:
            partial.write_bytes(b"partial")
            raise KeyboardInterrupt

    assert output.read_bytes() == b"previous complete output"
    assert not list(tmp_path.glob(".merged.*.nc"))


def test_tracked_merge_records_interruption(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = SimpleNamespace(stages={})
    transitions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: manifest),
    )
    monkeypatch.setattr(
        merge_mod,
        "save_stage",
        lambda _manifest, _path, stage, status, **_kwargs: transitions.append((stage, status)),
    )

    @merge_mod._tracked_merge
    def interrupted_merge(*, manifest_path: Path) -> list[Path]:
        del manifest_path
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted_merge(manifest_path=tmp_path / "manifest.json")

    assert transitions == [("merge", "interrupted")]


def test_tracked_merge_low_disk_does_not_mutate_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transitions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: SimpleNamespace(stages={})),
    )
    monkeypatch.setattr(
        merge_mod,
        "save_stage",
        lambda _manifest, _path, stage, status, **_kwargs: transitions.append(
            (stage, status)
        ),
    )

    @merge_mod._tracked_merge
    def refused_merge(*, manifest_path: Path) -> list[Path]:
        del manifest_path
        from openamundsen_da.exceptions import LowDiskPauseError

        raise LowDiskPauseError("80%")

    from openamundsen_da.exceptions import LowDiskPauseError

    with pytest.raises(LowDiskPauseError):
        refused_merge(manifest_path=tmp_path / "manifest.json")

    assert transitions == []


def test_tracked_model_merge_preserves_running_and_completed_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transitions: list[tuple[str, str]] = []
    monkeypatch.setattr(
        merge_mod.SubdomainManifest,
        "load",
        classmethod(lambda cls, path: SimpleNamespace(stages={})),
    )
    monkeypatch.setattr(
        merge_mod,
        "save_stage",
        lambda _manifest, _path, stage, status, **_kwargs: transitions.append(
            (stage, status)
        ),
    )

    def model_merge(*, manifest_path: Path) -> list[Path]:
        del manifest_path
        return [tmp_path / "model.nc"]

    model_merge.__name__ = "merge_model_grids"
    tracked = merge_mod._tracked_merge(model_merge)

    assert tracked(manifest_path=tmp_path / "manifest.json") == [
        tmp_path / "model.nc"
    ]
    assert transitions == [("merge", "running"), ("merge", "completed")]
