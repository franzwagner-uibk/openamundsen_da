from pathlib import Path

from openamundsen_da.subdomain import run as run_mod
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec


def test_run_one_caps_project_plot_workers_to_inner_workers(tmp_path, monkeypatch):
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
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)

    captured = {}

    monkeypatch.setattr(run_mod, "_prepare_obs_for_subdomain", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "run_project", lambda cfg: captured.setdefault("cfg", cfg))

    result = run_mod._run_one(
        "S1",
        manifest_path,
        inner_max_workers=6,
        overwrite=True,
        retries=0,
        log_level="INFO",
        root_log_path=None,
    )

    assert result.status == "success"
    assert captured["cfg"].max_workers == 6
    assert captured["cfg"].plot_workers == 6
