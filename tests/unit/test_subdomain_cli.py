from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from shapely.geometry import box

from openamundsen_da.subdomain.prepare import prepare_model_subdomains
from openamundsen_da.subdomain import cli as subdomain_cli


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / f"{project_dir.name}.yml").write_text(
        "start_date: '2022-10-01'\nend_date: '2022-10-02'\ndata_assimilation: {}\n",
        encoding="utf-8",
    )


def test_prepare_uses_setup_default_subdomain_root(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    regions = tmp_path / "regions.gpkg"
    setup_dir.mkdir(parents=True, exist_ok=True)
    _write_project_yaml(project_dir)
    regions.write_text("", encoding="utf-8")

    called: dict = {}

    def _fake_prepare_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.prepare.prepare_subdomains", _fake_prepare_subdomains)

    rc = subdomain_cli.cli(
        [
            "prepare",
            "--setup-dir",
            str(setup_dir),
            "--project-dir",
            str(project_dir),
            "--regions",
            str(regions),
        ]
    )

    assert rc == 0
    assert called["setup_dir"] == setup_dir
    assert called["project_dir"] == project_dir
    assert called["subdomain_root"] == project_dir / "subdomains"


def test_model_prepare_uses_setup_default_subdomain_root(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    regions = tmp_path / "regions.gpkg"
    setup_dir.mkdir(parents=True, exist_ok=True)
    regions.write_text("", encoding="utf-8")

    called: dict = {}

    def _fake_prepare_model_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        "openamundsen_da.subdomain.prepare.prepare_model_subdomains",
        _fake_prepare_model_subdomains,
    )

    rc = subdomain_cli.cli(
        [
            "model-prepare",
            "--setup-dir",
            str(setup_dir),
            "--regions",
            str(regions),
        ]
    )

    assert rc == 0
    assert called["setup_dir"] == setup_dir
    assert called["setup_yaml"] is None
    assert called["subdomain_root"] == setup_dir / "subdomains" / "model"


def test_model_prepare_passes_explicit_setup_yaml(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "adige"
    setup_yaml = setup_dir / "setup_100m.yml"
    regions = tmp_path / "regions.gpkg"
    setup_dir.mkdir(parents=True, exist_ok=True)
    setup_yaml.write_text("domain: demo\nresolution: 100\n", encoding="utf-8")
    regions.write_text("", encoding="utf-8")

    called: dict = {}

    def _fake_prepare_model_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(
        "openamundsen_da.subdomain.prepare.prepare_model_subdomains",
        _fake_prepare_model_subdomains,
    )

    rc = subdomain_cli.cli(
        [
            "model-prepare",
            "--setup-dir",
            str(setup_dir),
            "--setup-yaml",
            str(setup_yaml),
            "--regions",
            str(regions),
        ]
    )

    assert rc == 0
    assert called["setup_yaml"] == setup_yaml


def test_model_prepare_records_explicit_setup_yaml_in_manifest(tmp_path: Path) -> None:
    setup_dir = tmp_path / "adige"
    grids_dir = setup_dir / "grids"
    meteo_dir = setup_dir / "meteo"
    env_dir = setup_dir / "env"
    setup_dir.mkdir()
    grids_dir.mkdir()
    meteo_dir.mkdir()
    env_dir.mkdir()
    setup_yaml = setup_dir / "setup_100m.yml"
    setup_yaml.write_text(
        "domain: demo\n"
        "resolution: 100\n"
        "crs: EPSG:25832\n"
        "start_date: '2010-09-01'\n"
        "end_date: '2010-09-02'\n"
        "input_data:\n"
        "  grids:\n"
        "    dir: grids\n"
        "  meteo:\n"
        "    dir: meteo\n"
        "    format: csv\n"
        "output_data:\n"
        "  timeseries:\n"
        "    points: []\n",
        encoding="utf-8",
    )
    transform = Affine.translation(0.0, 100.0) * Affine.scale(100.0, -100.0)
    with rasterio.open(
        grids_dir / "dem_demo_100.asc",
        "w",
        driver="AAIGrid",
        dtype="float32",
        nodata=-9999.0,
        width=2,
        height=1,
        count=1,
        crs="EPSG:25832",
        transform=transform,
    ) as ds:
        ds.write(np.array([[1000.0, 1100.0]], dtype="float32"), 1)
    (meteo_dir / "stations.csv").write_text(
        "id,name,x,y,alt\ns1,S1,50,50,1000\ns2,S2,150,50,1100\n",
        encoding="utf-8",
    )
    for station_id in ("s1", "s2"):
        (meteo_dir / f"{station_id}.csv").write_text("time,temp\n2010-09-01 00:00:00,0\n", encoding="utf-8")
    regions = env_dir / "subdomains.gpkg"
    gpd.GeoDataFrame(
        {"id": pd.Series(["left", "right"], dtype=object)},
        geometry=[box(0.0, 0.0, 100.0, 100.0), box(100.0, 0.0, 200.0, 100.0)],
        crs="EPSG:25832",
    ).to_file(regions, driver="GPKG")

    manifest = prepare_model_subdomains(
        setup_dir=setup_dir,
        setup_yaml=setup_yaml,
        regions_path=regions,
        subdomain_root=setup_dir / "subdomains" / "model_r100",
        station_buffer_m=0.0,
        overwrite=True,
    )

    assert manifest.setup_yaml == setup_yaml.resolve()
    assert manifest.project_yaml == setup_yaml.resolve()
    assert manifest.subdomain_root == setup_dir / "subdomains" / "model_r100"


def test_model_prepare_missing_explicit_setup_yaml_fails(tmp_path: Path) -> None:
    setup_dir = tmp_path / "adige"
    regions = setup_dir / "env" / "subdomains.gpkg"
    regions.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"id": pd.Series(["left", "right"], dtype=object)},
        geometry=[box(0.0, 0.0, 100.0, 100.0), box(100.0, 0.0, 200.0, 100.0)],
        crs="EPSG:25832",
    ).to_file(regions, driver="GPKG")

    with pytest.raises(FileNotFoundError, match="Setup YAML not found"):
        prepare_model_subdomains(
            setup_dir=setup_dir,
            setup_yaml=setup_dir / "missing.yml",
            regions_path=regions,
        )


def test_prepare_defaults_to_subdomains_regions_file(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    env_dir = setup_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "subdomains.gpkg").write_text("", encoding="utf-8")
    _write_project_yaml(project_dir)

    called: dict = {}

    def _fake_prepare_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.prepare.prepare_subdomains", _fake_prepare_subdomains)

    rc = subdomain_cli.cli(
        [
            "prepare",
            "--setup-dir",
            str(setup_dir),
            "--project-dir",
            str(project_dir),
        ]
    )

    assert rc == 0
    assert called["regions_path"] == setup_dir / "env" / "subdomains.gpkg"


def test_prepare_defaults_to_roi_regions_file_when_subdomains_missing(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    env_dir = setup_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    (env_dir / "roi.gpkg").write_text("", encoding="utf-8")
    _write_project_yaml(project_dir)

    called: dict = {}

    def _fake_prepare_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.prepare.prepare_subdomains", _fake_prepare_subdomains)

    rc = subdomain_cli.cli(
        [
            "prepare",
            "--setup-dir",
            str(setup_dir),
            "--project-dir",
            str(project_dir),
        ]
    )

    assert rc == 0
    assert called["regions_path"] == setup_dir / "env" / "roi.gpkg"


def test_run_resolves_manifest_from_project_dir(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_run_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.run.run_subdomains", _fake_run_subdomains)

    rc = subdomain_cli.cli(["run", "--project-dir", str(project_dir), "--no-perf-monitor"])

    assert rc == 0
    assert called["manifest_path"] == manifest


def test_model_run_resolves_manifest_from_setup_dir_and_selected_subdomains(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    manifest = setup_dir / "subdomains" / "model" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_run_model_subdomains(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.model.run_model_subdomains", _fake_run_model_subdomains)

    rc = subdomain_cli.cli(
        [
            "model-run",
            "--setup-dir",
            str(setup_dir),
            "--subdomains",
            "sd_02",
            "--max-workers",
            "3",
        ]
    )

    assert rc == 0
    assert called["manifest_path"] == manifest
    assert called["subdomains"] == ["sd_02"]
    assert called["max_workers"] == 3


def test_model_pipeline_passes_explicit_setup_yaml(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "adige"
    setup_yaml = setup_dir / "setup_100m.yml"
    regions = tmp_path / "regions.gpkg"
    setup_dir.mkdir(parents=True, exist_ok=True)
    setup_yaml.write_text("domain: demo\nresolution: 100\n", encoding="utf-8")
    regions.write_text("", encoding="utf-8")

    called: dict = {}

    def _fake_run_model_pipeline(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr("openamundsen_da.subdomain.model.run_model_pipeline", _fake_run_model_pipeline)

    rc = subdomain_cli.cli(
        [
            "model-pipeline",
            "--setup-dir",
            str(setup_dir),
            "--setup-yaml",
            str(setup_yaml),
            "--regions",
            str(regions),
            "--subdomain-root",
            str(setup_dir / "subdomains" / "model_r100"),
        ]
    )

    assert rc == 0
    assert called["setup_yaml"] == setup_yaml
    assert called["subdomain_root"] == setup_dir / "subdomains" / "model_r100"


def test_merge_uses_default_output_layout(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        "{}",
        encoding="utf-8",
    )

    called_grids: dict = {}
    def _fake_merge_grids(**kwargs):
        called_grids.update(kwargs)
        return []

    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_grids", _fake_merge_grids)
    monkeypatch.setattr(
        "openamundsen_da.subdomain.manifest.SubdomainManifest.load",
        lambda _path: type("M", (), {"project_dir": project_dir})(),
    )

    rc = subdomain_cli.cli(["merge", "--project-dir", str(project_dir)])

    assert rc == 0
    assert called_grids["out_dir"] == project_dir / "results" / "grids"
    assert called_grids["cleanup_compact_artifacts"] is False


def test_merge_cleanup_requires_confirmation(tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)

    with pytest.raises(SystemExit) as excinfo:
        subdomain_cli.cli(
            [
                "merge",
                "--project-dir",
                str(project_dir),
                "--cleanup-compact-artifacts",
            ]
        )

    assert excinfo.value.code == 2


def test_merge_passes_explicit_cleanup_flag(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called_grids: dict = {}

    def _fake_merge_grids(**kwargs):
        called_grids.update(kwargs)
        return []

    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_grids", _fake_merge_grids)
    monkeypatch.setattr(
        "openamundsen_da.subdomain.manifest.SubdomainManifest.load",
        lambda _path: type("M", (), {"project_dir": project_dir})(),
    )

    rc = subdomain_cli.cli(
        [
            "merge",
            "--project-dir",
            str(project_dir),
            "--cleanup-compact-artifacts",
            "--confirm-delete-raw-grid-support",
        ]
    )

    assert rc == 0
    assert called_grids["cleanup_compact_artifacts"] is True


def test_model_merge_resolves_manifest_from_setup_dir(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    manifest = setup_dir / "subdomains" / "model" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_merge_model_grids(**kwargs):
        called.update(kwargs)
        return []

    monkeypatch.setattr("openamundsen_da.subdomain.merge.merge_model_grids", _fake_merge_model_grids)

    rc = subdomain_cli.cli(["model-merge", "--setup-dir", str(setup_dir)])

    assert rc == 0
    assert called["manifest_path"] == manifest
    assert called["out_dir"] is None


def test_model_plot_resolves_manifest_from_setup_dir(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    manifest = setup_dir / "subdomains" / "model" / "subdomain_manifest.json"
    config = tmp_path / "model_plots.yml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    config.write_text("monthly_maps: {}\n", encoding="utf-8")

    called: dict = {}

    def _fake_plot_model_subdomains(**kwargs):
        called.update(kwargs)
        return []

    monkeypatch.setattr("openamundsen_da.subdomain.model_plot.plot_model_subdomains", _fake_plot_model_subdomains)

    rc = subdomain_cli.cli(
        [
            "model-plot",
            "--setup-dir",
            str(setup_dir),
            "--subdomains",
            "sd_01",
            "--config",
            str(config),
        ]
    )

    assert rc == 0
    assert called["manifest_path"] == manifest
    assert called["subdomains"] == ["sd_01"]
    assert called["config_path"] == config


def test_resolve_manifest_from_subdomain_root(tmp_path: Path) -> None:
    subdomain_root = tmp_path / "custom_subdomains"
    manifest = subdomain_root / "subdomain_manifest.json"
    subdomain_root.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    resolved = subdomain_cli._resolve_manifest(
        manifest_arg=None,
        project_dir=None,
        subdomain_root=subdomain_root,
    )
    assert resolved == manifest


def test_resolve_model_manifest_from_subdomain_root(tmp_path: Path) -> None:
    subdomain_root = tmp_path / "model_subdomains"
    manifest = subdomain_root / "subdomain_manifest.json"
    subdomain_root.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    resolved = subdomain_cli._resolve_model_manifest(
        manifest_arg=None,
        setup_dir=None,
        subdomain_root=subdomain_root,
    )
    assert resolved == manifest


def test_plot_defaults_to_snow_depth_obs_column(monkeypatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "rofental"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    _write_project_yaml(project_dir)
    manifest = project_dir / "subdomains" / "subdomain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")

    called: dict = {}

    def _fake_plot_station_comparisons(**kwargs):
        called.update(kwargs)
        return []

    monkeypatch.setattr(
        "openamundsen_da.methods.viz.plots.subdomain.station_comparisons.plot_station_comparisons",
        _fake_plot_station_comparisons,
    )

    rc = subdomain_cli.cli(["plot", "--project-dir", str(project_dir)])

    assert rc == 0
    assert called["manifest_path"] == manifest
    assert called["obs_column"] == "snow_depth"
