from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from affine import Affine
from shapely.geometry import box

from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta, WindowSpec
from openamundsen_da.subdomain.model_plot import (
    build_monthly_model_map_recipe,
    first_day_of_month_dates,
    plot_model_subdomains,
    render_station_swe_comparison,
    station_swe_comparison_frame,
)


def _model_manifest(tmp_path: Path, *, status: str = "success") -> tuple[SubdomainManifest, Path]:
    setup_dir = tmp_path / "setup"
    root = setup_dir / "subdomains" / "model"
    setup_dir.mkdir(parents=True)
    setup_yaml = setup_dir / "setup.yml"
    setup_yaml.write_text(
        "domain: demo\n"
        "resolution: 1\n"
        "crs: EPSG:25832\n"
        "start_date: '2010-09-01'\n"
        "end_date: '2011-08-31'\n"
        "input_data:\n"
        "  meteo:\n"
        "    dir: meteo\n"
        "    format: csv\n",
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
        grid_transform=tuple(Affine.translation(0.0, 2.0) * Affine.scale(1.0, -1.0)),
        grid_resolution=1.0,
        grid_domain="demo",
        clip_mode="window",
        station_buffer_m=0.0,
        roi_buffer_m=0.0,
        grid_buffer_m=0.0,
        raw_snowcover_dir=setup_dir / "obs" / "snowcover",
        raw_wetsnow_dir=setup_dir / "obs" / "wetsnow",
    )
    sub_dir = root / "sd_01"
    sub_dir.mkdir(parents=True)
    sub_yaml = sub_dir / "sd_01.yml"
    sub_yaml.write_text(
        "domain: demo\n"
        "resolution: 1\n"
        "crs: EPSG:25832\n"
        "input_data:\n"
        "  grids:\n"
        "    dir: grids\n"
        "  meteo:\n"
        "    dir: meteo\n"
        "    format: csv\n",
        encoding="utf-8",
    )
    manifest.subdomains["sd_01"] = SubdomainMeta(
        id="sd_01",
        label="sd_01",
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
        window=WindowSpec(row_off=0, col_off=0, height=2, width=2),
        transform=tuple(Affine.translation(0.0, 2.0) * Affine.scale(1.0, -1.0)),
        bounds=(0.0, 0.0, 2.0, 2.0),
        crs="EPSG:25832",
        status=status,
    )
    manifest_path = root / "subdomain_manifest.json"
    manifest.save(manifest_path)
    return manifest, manifest_path


def _write_roi(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="AAIGrid",
        dtype="uint8",
        nodata=0,
        width=2,
        height=2,
        count=1,
        crs="EPSG:25832",
        transform=Affine.translation(0.0, 2.0) * Affine.scale(1.0, -1.0),
    ) as ds:
        ds.write(np.ones((2, 2), dtype="uint8"), 1)


def _write_grid(path: Path, values: np.ndarray, *, transform=None) -> None:
    transform = transform or Affine.translation(0.0, 2.0) * Affine.scale(1.0, -1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape
    with rasterio.open(
        path,
        "w",
        driver="AAIGrid",
        dtype="float32",
        nodata=-9999.0,
        width=width,
        height=height,
        count=1,
        crs="EPSG:25832",
        transform=transform,
    ) as ds:
        ds.write(values.astype("float32"), 1)


def _write_static_inputs(sub: SubdomainMeta) -> None:
    _write_roi(sub.roi_raster_path)
    _write_grid(sub.setup_dir / "grids" / "dem_demo_1.asc", np.array([[2000.0, 2100.0], [2200.0, 2300.0]]))
    _write_grid(sub.setup_dir / "grids" / "svf_demo_1.asc", np.ones((2, 2), dtype=float))
    _write_grid(sub.setup_dir / "grids" / "srf_demo_1.asc", np.ones((2, 2), dtype=float))
    sub.roi_vector_path.parent.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(
        {"id": pd.Series(["sd_01"], dtype=object)},
        geometry=[box(0.0, 0.0, 2.0, 2.0)],
        crs="EPSG:25832",
    ).to_file(
        sub.roi_vector_path,
        driver="GPKG",
    )


def _write_model_nc(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.date_range("2010-09-01", "2011-08-01", freq="MS")
    arr = np.ones((len(times), 2, 2), dtype="float32")
    ds = xr.Dataset(
        {
            "snowdepth_daily": (("time1", "y", "x"), arr),
            "swe_daily": (("time1", "y", "x"), arr * 100.0),
        },
        coords={"time1": times, "y": [1.5, 0.5], "x": [0.5, 1.5]},
    )
    ds.to_netcdf(path)


def test_first_day_of_month_dates_for_adige_window() -> None:
    dates = first_day_of_month_dates("2010-09-01", "2011-08-31")

    assert len(dates) == 12
    assert dates[0] == pd.Timestamp("2010-09-01")
    assert dates[-1] == pd.Timestamp("2011-08-01")


def test_build_monthly_model_map_recipe_uses_da_map_panel_contract() -> None:
    dates = first_day_of_month_dates("2010-09-01", "2011-08-31")

    snow = build_monthly_model_map_recipe(subdomain_id="sd_01", variable="snowdepth_daily", dates=dates)
    swe = build_monthly_model_map_recipe(subdomain_id="sd_01", variable="swe_daily", dates=dates)

    assert snow.layout.nrows == 4
    assert snow.layout.ncols == 3
    assert len(snow.panels) == 12
    assert {panel.kind for panel in snow.panels} == {"snow_depth"}
    assert {panel.source for panel in snow.panels} == {"open_loop"}
    assert all(panel.show_hillshade for panel in snow.panels)
    assert all(panel.show_station_marker for panel in snow.panels)
    assert all(panel.show_stations_name for panel in snow.panels)
    assert all(panel.show_stations_elev for panel in snow.panels)
    assert swe.layout.nrows == 4
    assert swe.layout.ncols == 3
    assert {panel.kind for panel in swe.panels} == {"swe"}
    assert all(panel.show_station_marker for panel in swe.panels)
    assert all(panel.show_stations_name for panel in swe.panels)
    assert all(panel.show_stations_elev for panel in swe.panels)


def test_station_comparison_uses_first_model_timestamp_and_keeps_qc_rows(tmp_path: Path) -> None:
    model = tmp_path / "point_NHSWEID_1.csv"
    obs = tmp_path / "NHSWEID_1.csv"
    model.write_text(
        "time,swe\n"
        "2010-09-01 00:00:00,10\n"
        "2010-09-01 03:00:00,99\n"
        "2010-09-02 00:00:00,20\n",
        encoding="utf-8",
    )
    obs.write_text(
        "time,swe,QC_flag\n"
        "2010-09-01 00:00:00,7,1\n"
        "2010-09-02 00:00:00,25,0\n",
        encoding="utf-8",
    )

    frame = station_swe_comparison_frame(
        model_point_csv=model,
        obs_csv=obs,
        start=pd.Timestamp("2010-09-01"),
        end=pd.Timestamp("2010-09-02"),
    )

    assert list(frame["model"]) == [10, 20]
    assert list(frame["obs"]) == [7, 25]


def test_plot_model_subdomains_fails_when_subdomain_netcdf_missing(tmp_path: Path) -> None:
    _manifest, manifest_path = _model_manifest(tmp_path)

    with pytest.raises(FileNotFoundError, match="Missing model grid output"):
        plot_model_subdomains(manifest_path=manifest_path)


def test_station_plot_skips_missing_observation_file(tmp_path: Path) -> None:
    manifest, _manifest_path = _model_manifest(tmp_path)
    sub = manifest.subdomains["sd_01"]
    results = sub.setup_dir / "results"
    results.mkdir(parents=True)
    (results / "point_NHSWEID_1.csv").write_text("time,swe\n2010-09-01,10\n", encoding="utf-8")

    written = render_station_swe_comparison(
        subdomain=sub,
        setup_dir=manifest.setup_dir,
        start=pd.Timestamp("2010-09-01"),
        end=pd.Timestamp("2011-08-31"),
        output_dir=manifest.subdomain_root / "results" / "plots" / "stations",
    )

    assert written == []


def test_plot_model_subdomains_writes_monthly_and_station_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _model_manifest(tmp_path)
    sub = manifest.subdomains["sd_01"]
    _write_static_inputs(sub)
    _write_grid(
        manifest.setup_dir / "grids" / "dem_demo_1.asc",
        np.arange(9, dtype=float).reshape(3, 3),
        transform=Affine.translation(-1.0, 3.0) * Affine.scale(1.0, -1.0),
    )
    _write_model_nc(sub.setup_dir / "results" / "grids" / "output_grids.nc")
    (sub.setup_dir / "results").mkdir(parents=True, exist_ok=True)
    (sub.setup_dir / "results" / "point_NHSWEID_1.csv").write_text(
        "time,swe\n2010-09-01,10\n2010-10-01,20\n",
        encoding="utf-8",
    )
    sub.meteo_dir.mkdir(parents=True, exist_ok=True)
    (sub.meteo_dir / "stations.csv").write_text(
        "id,name,x,y,alt\nMETEO_1,Meteo station,1.0,1.0,1000\n",
        encoding="utf-8",
    )
    obs_dir = manifest.setup_dir / "obs" / "stations"
    obs_dir.mkdir(parents=True)
    (obs_dir / "stations_snow_depth.csv").write_text(
        "id,name,x,y,alt\nNHSWEID_1,Snow station,1.5,1.5,2000\n",
        encoding="utf-8",
    )
    (obs_dir / "NHSWEID_1.csv").write_text(
        "time,swe,QC_flag\n2010-09-01,11,1\n2010-10-01,21,0\n",
        encoding="utf-8",
    )
    legacy_map = manifest.subdomain_root / "results" / "maps" / "monthly" / "sd_01_monthly_snow.png"
    legacy_map.parent.mkdir(parents=True, exist_ok=True)
    legacy_map.write_text("bad legacy map", encoding="utf-8")
    rendered_recipes = []
    rendered_station_ids = []
    rendered_hillshade_shapes = []

    def _fake_render_map_recipe(**kwargs):
        rendered_recipes.append(kwargs["recipe"])
        rendered_station_ids.append(tuple(kwargs["context"].stations["id"]))
        rendered_hillshade_shapes.append(kwargs["context"].hillshade_dem.shape)
        out_path = kwargs["output_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return out_path

    monkeypatch.setattr("openamundsen_da.subdomain.model_plot.render_map_recipe", _fake_render_map_recipe)

    written = plot_model_subdomains(manifest_path=manifest_path)

    assert manifest.subdomain_root / "results" / "maps" / "monthly" / "sd_01_snowdepth_monthly.png" in written
    assert manifest.subdomain_root / "results" / "maps" / "monthly" / "sd_01_swe_monthly.png" in written
    assert manifest.subdomain_root / "results" / "plots" / "stations" / "NHSWEID_1_swe_comparison.png" in written
    assert not legacy_map.exists()
    assert [recipe.layout.nrows for recipe in rendered_recipes] == [4, 4]
    assert [recipe.layout.ncols for recipe in rendered_recipes] == [3, 3]
    assert rendered_station_ids == [("NHSWEID_1",), ("NHSWEID_1",)]
    assert rendered_hillshade_shapes == [(3, 3), (3, 3)]


def test_plot_model_subdomains_uses_maps_yaml_recipes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _model_manifest(tmp_path)
    sub = manifest.subdomains["sd_01"]
    _write_static_inputs(sub)
    _write_model_nc(sub.setup_dir / "results" / "grids" / "output_grids.nc")
    maps_config = manifest.setup_dir / "maps.yaml"
    maps_config.write_text(
        """
maps:
  custom_snow:
    title: "{subdomain_label} custom snow"
    output_name: "{subdomain_id}_custom_snow"
    layout:
      nrows: 1
      ncols: 1
    defaults:
      show_scalebar: false
      show_grid: false
      show_colorbar: false
    panels:
      - row: 0
        col: 0
        kind: snow_depth
        title: Custom panel
        date: "2010-09-01"
        source: open_loop
        show_hillshade: false
        show_station_marker: false
        show_stations_name: false
        show_stations_elev: false
""",
        encoding="utf-8",
    )
    rendered_recipes = []

    def _fake_render_map_recipe(**kwargs):
        rendered_recipes.append(kwargs["recipe"])
        out_path = kwargs["output_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return out_path

    monkeypatch.setattr("openamundsen_da.subdomain.model_plot.render_map_recipe", _fake_render_map_recipe)

    written = plot_model_subdomains(manifest_path=manifest_path, config_path=maps_config)

    assert manifest.subdomain_root / "results" / "maps" / "monthly" / "sd_01_custom_snow.png" in written
    assert len(rendered_recipes) == 1
    recipe = rendered_recipes[0]
    assert recipe.layout.nrows == 1
    assert recipe.layout.ncols == 1
    assert recipe.title == "sd_01 custom snow"
    assert recipe.panels[0].show_station_marker is False
    assert recipe.panels[0].show_stations_name is False
    assert recipe.panels[0].show_stations_elev is False


def test_plot_model_subdomains_expands_generic_model_maps_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest, manifest_path = _model_manifest(tmp_path)
    sub = manifest.subdomains["sd_01"]
    _write_static_inputs(sub)
    _write_model_nc(sub.setup_dir / "results" / "grids" / "output_grids.nc")
    maps_config = manifest.setup_dir / "maps.yaml"
    maps_config.write_text(
        """
model_maps:
  monthly:
    kind: monthly
    variables: [snowdepth_daily, swe_daily]
    date_rule: first_day_of_month
    output_name: "{subdomain_id}_{variable_token}_monthly"
    title: "{subdomain_id} monthly {variable_title}"
    layout:
      ncols: 3
    defaults:
      show_scalebar: true
      show_grid: true
      show_colorbar: true
    panel:
      title: "{date}"
      source: open_loop
      show_hillshade: true
      show_station_marker: true
      show_stations_name: false
      show_stations_elev: false
""",
        encoding="utf-8",
    )
    rendered_recipes = []

    def _fake_render_map_recipe(**kwargs):
        rendered_recipes.append(kwargs["recipe"])
        out_path = kwargs["output_path"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"png")
        return out_path

    monkeypatch.setattr("openamundsen_da.subdomain.model_plot.render_map_recipe", _fake_render_map_recipe)

    written = plot_model_subdomains(manifest_path=manifest_path)

    assert manifest.subdomain_root / "results" / "maps" / "monthly" / "sd_01_snowdepth_monthly.png" in written
    assert manifest.subdomain_root / "results" / "maps" / "monthly" / "sd_01_swe_monthly.png" in written
    assert [recipe.layout.nrows for recipe in rendered_recipes] == [4, 4]
    assert [recipe.layout.ncols for recipe in rendered_recipes] == [3, 3]
    assert [len(recipe.panels) for recipe in rendered_recipes] == [12, 12]
    assert rendered_recipes[0].panels[0].date == "2010-09-01"
    assert rendered_recipes[0].panels[-1].date == "2011-08-01"
    assert all(panel.show_station_marker is True for panel in rendered_recipes[0].panels)
    assert all(panel.show_stations_name is False for panel in rendered_recipes[0].panels)
