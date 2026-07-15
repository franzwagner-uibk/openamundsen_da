from __future__ import annotations

import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from affine import Affine
from rasterio.transform import from_origin
from shapely.geometry import box

from openamundsen_da.subdomain import model as model_mod
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec
from openamundsen_da.subdomain.merge import merge_model_grids
from openamundsen_da.subdomain.prepare import prepare_model_subdomains
from openamundsen_da.util.yaml_utils import read_yaml_mapping


def _write_setup_yaml(setup_dir: Path, *, include_dates: bool = True) -> None:
    lines = [
        'domain: "demo"',
        "resolution: 100",
        'timestep: "D"',
        'crs: "EPSG:25832"',
        "timezone: 1",
    ]
    if include_dates:
        lines.extend(["start_date: '2022-10-01'", "end_date: '2022-10-02'"])
    lines.extend(
        [
            "input_data:",
            "  grids:",
            "    dir: grids",
            "  meteo:",
            "    dir: meteo",
            "",
        ]
    )
    (setup_dir / "demo.yml").write_text("\n".join(lines), encoding="utf-8")


def _write_dem(path: Path, *, shape: tuple[int, int] = (4, 4)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ones(shape, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="AAIGrid",
        dtype="float32",
        nodata=-9999.0,
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        crs="EPSG:25832",
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
    ) as ds:
        ds.write(arr, 1)


def _write_meteo(meteo_dir: Path) -> None:
    meteo_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ["s1", "s2"], "x": [50.0, 250.0], "y": [50.0, 50.0]}).to_csv(
        meteo_dir / "stations.csv",
        index=False,
    )
    for sid in ("s1", "s2"):
        (meteo_dir / f"{sid}.csv").write_text("date,temp\n2022-10-01,273.15\n", encoding="utf-8")


def _write_regions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"id": pd.Series(["sd_01", "sd_02"], dtype="object")},
        geometry=[box(0, 0, 200, 400), box(200, 0, 400, 400)],
        crs="EPSG:25832",
    ).to_file(path, driver="GPKG")


def _write_center_regions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"id": pd.Series(["sd_01", "sd_02"], dtype="object")},
        geometry=[box(100, 100, 200, 300), box(200, 100, 300, 300)],
        crs="EPSG:25832",
    ).to_file(path, driver="GPKG")


def test_prepare_model_subdomains_writes_plain_setup_folders(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    _write_setup_yaml(setup_dir)
    _write_dem(setup_dir / "grids" / "dem_demo_100.asc")
    _write_meteo(setup_dir / "meteo")
    _write_regions(setup_dir / "env" / "subdomains.gpkg")

    manifest = prepare_model_subdomains(
        setup_dir=setup_dir,
        regions_path=setup_dir / "env" / "subdomains.gpkg",
        station_buffer_m=0.0,
        overwrite=True,
    )

    root = setup_dir.resolve() / "subdomains" / "model"
    sub = manifest.subdomains["sd_01"]
    cfg = read_yaml_mapping(sub.setup_yaml)

    assert manifest.run_mode == "model"
    assert manifest.project_dir == root
    assert sub.project_dir == sub.setup_dir
    assert sub.project_yaml == sub.setup_yaml
    assert sub.setup_yaml == root / "sd_01" / "sd_01.yml"
    assert not (root / "sd_01" / "projects").exists()
    assert not (root / "sd_01" / "obs").exists()
    assert cfg["domain"] == "demo_sd_01"
    assert cfg["start_date"] == "2022-10-01"
    assert cfg["end_date"] == "2022-10-02"
    assert cfg["input_data"]["grids"]["dir"] == str((root / "sd_01" / "grids").resolve())
    assert cfg["input_data"]["meteo"]["dir"] == str((root / "sd_01" / "meteo").resolve())
    assert cfg["results_dir"] == str((root / "sd_01" / "results").resolve())


def test_prepare_model_subdomains_keeps_buffered_grid_context_outside_roi(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    _write_setup_yaml(setup_dir)
    _write_dem(setup_dir / "grids" / "dem_demo_100.asc")
    _write_meteo(setup_dir / "meteo")
    _write_center_regions(setup_dir / "env" / "subdomains.gpkg")

    manifest = prepare_model_subdomains(
        setup_dir=setup_dir,
        regions_path=setup_dir / "env" / "subdomains.gpkg",
        station_buffer_m=200.0,
        grid_buffer_m=100.0,
        overwrite=True,
    )

    sub = manifest.subdomains["sd_01"]
    dem_path = sub.grids_dir / "dem_demo_sd_01_100.asc"
    roi_path = sub.roi_raster_path
    with rasterio.open(dem_path) as dem_ds, rasterio.open(roi_path) as roi_ds:
        dem = dem_ds.read(1)
        roi = roi_ds.read(1).astype(bool)
        nodata = dem_ds.nodata

    assert dem.shape == roi.shape
    assert np.count_nonzero(~roi) > 0
    assert np.isfinite(dem[~roi]).all()
    assert not np.any(np.isclose(dem[~roi], nodata))


def test_prepare_model_subdomains_requires_setup_dates(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    setup_dir.mkdir()
    _write_setup_yaml(setup_dir, include_dates=False)
    _write_dem(setup_dir / "grids" / "dem_demo_100.asc")
    _write_meteo(setup_dir / "meteo")
    _write_regions(setup_dir / "env" / "subdomains.gpkg")

    with pytest.raises(ValueError, match="requires the source setup YAML"):
        prepare_model_subdomains(
            setup_dir=setup_dir,
            regions_path=setup_dir / "env" / "subdomains.gpkg",
            overwrite=True,
        )


def _model_manifest(
    tmp_path: Path,
    ids: tuple[str, ...] = ("sd_01", "sd_02"),
    *,
    grid_format: str = "netcdf",
) -> Path:
    setup_dir = tmp_path / "setup"
    root = setup_dir / "subdomains" / "model"
    setup_dir.mkdir(parents=True, exist_ok=True)
    setup_yaml = setup_dir / "demo.yml"
    setup_yaml.write_text(
        "\n".join(
            [
                "start_date: '2022-10-01'",
                "end_date: '2022-10-02'",
                "output_data:",
                "  grids:",
                f"    format: {grid_format}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = SubdomainManifest(
        run_mode="model",
        setup_dir=setup_dir,
        project_dir=root,
        project_name="model",
        setup_yaml=setup_yaml,
        project_yaml=setup_yaml,
        subdomain_root=root,
        regions_path=setup_dir / "env" / "subdomains.gpkg",
        id_field="id",
        crs="EPSG:25832",
        grid_rows=2,
        grid_cols=2,
        grid_transform=tuple(from_origin(0.0, 2.0, 1.0, 1.0)),
        grid_resolution=1.0,
        grid_domain="demo",
        clip_mode="window",
        station_buffer_m=0.0,
        roi_buffer_m=0.0,
        grid_buffer_m=0.0,
        raw_snowcover_dir=setup_dir / "obs" / "snowcover",
        raw_wetsnow_dir=setup_dir / "obs" / "wetsnow",
    )
    for idx, sid in enumerate(ids):
        sub_dir = root / sid
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_yaml = sub_dir / f"{sid}.yml"
        sub_yaml.write_text("domain: demo\n", encoding="utf-8")
        manifest.subdomains[sid] = SubdomainMeta(
            id=sid,
            label=sid,
            setup_dir=sub_dir,
            setup_yaml=sub_yaml,
            project_dir=sub_dir,
            project_yaml=sub_yaml,
            project_name="model",
            grids_dir=sub_dir / "grids",
            meteo_dir=sub_dir / "meteo",
            obs_stations_dir=sub_dir / "obs" / "stations",
            roi_raster_path=sub_dir / "grids" / "roi.asc",
            roi_vector_path=sub_dir / "env" / "roi.gpkg",
            window=WindowSpec(row_off=0, col_off=idx, height=2, width=1),
            transform=tuple(from_origin(float(idx), 2.0, 1.0, 1.0)),
            bounds=(float(idx), 0.0, float(idx + 1), 2.0),
            crs="EPSG:25832",
        )
    manifest_path = root / "subdomain_manifest.json"
    manifest.save(manifest_path)
    return manifest_path


def test_run_model_subdomains_runs_selected_subdomains(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _model_manifest(tmp_path)
    calls: list[tuple[list[str], Path]] = []

    monkeypatch.setattr(model_mod.shutil, "which", lambda _name: "/usr/bin/openamundsen")

    def _fake_run(command, *, cwd, env, stdout, stderr, check):
        calls.append((command, cwd))
        assert env["OMP_NUM_THREADS"] == "1"
        assert stderr is subprocess.STDOUT
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(model_mod.subprocess, "run", _fake_run)

    results = model_mod.run_model_subdomains(
        manifest_path=manifest_path,
        subdomains=["sd_02"],
        max_workers=1,
        overwrite=True,
        log_to_file=False,
    )

    assert [r.subdomain_id for r in results] == ["sd_02"]
    assert results[0].status == "success"
    sd_02_dir = tmp_path / "setup" / "subdomains" / "model" / "sd_02"
    assert calls == [(["/usr/bin/openamundsen", str(sd_02_dir / "sd_02.yml")], sd_02_dir)]
    loaded = SubdomainManifest.load(manifest_path)
    assert loaded.subdomains["sd_01"].status == "pending"
    assert loaded.subdomains["sd_02"].status == "success"


def test_run_model_subdomains_fail_fast_marks_unfinished_skipped(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _model_manifest(tmp_path)

    monkeypatch.setattr(model_mod.shutil, "which", lambda _name: "/usr/bin/openamundsen")

    def _fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(model_mod.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="sd_01"):
        model_mod.run_model_subdomains(
            manifest_path=manifest_path,
            subdomains=["sd_01", "sd_02"],
            max_workers=1,
            overwrite=True,
            log_to_file=False,
        )

    loaded = SubdomainManifest.load(manifest_path)
    assert loaded.subdomains["sd_01"].status == "failed"
    assert loaded.subdomains["sd_02"].status == "skipped"


def test_run_model_subdomains_reuses_success_without_downgrading_manifest(tmp_path: Path) -> None:
    manifest_path = _model_manifest(tmp_path, ids=("sd_01",))
    manifest = SubdomainManifest.load(manifest_path)
    subdomain = manifest.subdomains["sd_01"]
    run_manifest = subdomain.setup_dir / "run_manifest.json"
    run_manifest.write_text('{"status": "success"}\n', encoding="utf-8")
    subdomain.status = "success"
    subdomain.run_manifest = run_manifest
    manifest.save(manifest_path)

    results = model_mod.run_model_subdomains(
        manifest_path=manifest_path,
        max_workers=1,
        overwrite=False,
        log_to_file=False,
    )

    loaded = SubdomainManifest.load(manifest_path)
    assert [result.status for result in results] == ["skipped"]
    assert loaded.subdomains["sd_01"].status == "success"
    assert loaded.stages["run"]["status"] == "completed"


def test_stage_model_grid_outputs_leaves_timeseries_outputs_in_place(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "output_grids.nc").write_bytes(b"grid")
    (results_dir / "snowdepth_2023-03-17.tif").write_bytes(b"grid")
    (results_dir / "output_timeseries.nc").write_bytes(b"points")

    staged = model_mod._stage_model_grid_outputs(results_dir)

    assert sorted(path.name for path in staged) == ["output_grids.nc", "snowdepth_2023-03-17.tif"]
    assert (results_dir / "grids" / "output_grids.nc").is_file()
    assert (results_dir / "grids" / "snowdepth_2023-03-17.tif").is_file()
    assert (results_dir / "output_timeseries.nc").is_file()


def _write_roi(path: Path, arr: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="uint8",
        nodata=0,
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        crs="EPSG:25832",
        transform=transform,
    ) as ds:
        ds.write(arr.astype("uint8"), 1)


def _write_tif(path: Path, arr: np.ndarray, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype="float32",
        nodata=-9999.0,
        width=arr.shape[1],
        height=arr.shape[0],
        count=1,
        crs="EPSG:25832",
        transform=transform,
    ) as ds:
        ds.write(arr.astype("float32"), 1)


def _write_nc(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {"swe": (("y", "x"), arr.astype("float32"))},
        coords={"y": np.arange(arr.shape[0], dtype=np.float32), "x": np.arange(arr.shape[1], dtype=np.float32)},
    )
    ds.to_netcdf(path)


def _write_paired_model_grid_inputs(manifest: SubdomainManifest, *, grid_format: str) -> None:
    values = {
        "sd_01": np.array([[1.0], [2.0]], dtype=np.float32),
        "sd_02": np.array([[3.0], [4.0]], dtype=np.float32),
    }
    transforms = {
        "sd_01": from_origin(0.0, 2.0, 1.0, 1.0),
        "sd_02": from_origin(1.0, 2.0, 1.0, 1.0),
    }
    for sid, subdomain in manifest.subdomains.items():
        _write_roi(
            subdomain.roi_raster_path,
            np.ones((2, 1), dtype=np.uint8),
            transforms[sid],
        )
        output_dir = subdomain.setup_dir / "results" / "grids"
        if grid_format == "netcdf":
            _write_nc(output_dir / "output_grids.nc", values[sid])
        else:
            _write_tif(output_dir / "swe.tif", values[sid], transforms[sid])


def test_model_netcdf_and_geotiff_mosaics_are_scientifically_equivalent(tmp_path: Path) -> None:
    nc_manifest_path = _model_manifest(tmp_path / "netcdf", grid_format="netcdf")
    tif_manifest_path = _model_manifest(tmp_path / "geotiff", grid_format="geotiff")
    nc_manifest = SubdomainManifest.load(nc_manifest_path)
    tif_manifest = SubdomainManifest.load(tif_manifest_path)
    _write_paired_model_grid_inputs(nc_manifest, grid_format="netcdf")
    _write_paired_model_grid_inputs(tif_manifest, grid_format="geotiff")

    nc_written = merge_model_grids(manifest_path=nc_manifest_path, coverage_sliver_tol_px=0)
    tif_written = merge_model_grids(manifest_path=tif_manifest_path, coverage_sliver_tol_px=0)

    assert [path.name for path in nc_written] == ["output_grids.nc"]
    assert [path.name for path in tif_written] == ["swe.tif"]
    with xr.open_dataset(nc_written[0]) as nc_ds, rasterio.open(tif_written[0]) as tif_ds:
        np.testing.assert_allclose(nc_ds["swe"].values, tif_ds.read(1))


def test_model_merge_rejects_mixed_configured_and_stale_formats(tmp_path: Path) -> None:
    manifest_path = _model_manifest(tmp_path, grid_format="netcdf")
    manifest = SubdomainManifest.load(manifest_path)
    _write_paired_model_grid_inputs(manifest, grid_format="netcdf")
    stale = manifest.subdomains["sd_01"].setup_dir / "results" / "grids" / "stale.tif"
    _write_tif(stale, np.ones((2, 1), dtype=np.float32), from_origin(0.0, 2.0, 1.0, 1.0))

    with pytest.raises(ValueError, match="Mixed model grid artifacts"):
        merge_model_grids(manifest_path=manifest_path, coverage_sliver_tol_px=0)
