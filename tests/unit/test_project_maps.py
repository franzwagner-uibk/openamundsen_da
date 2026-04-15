from __future__ import annotations

from pathlib import Path
import textwrap

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from openamundsen_da.methods.viz.project_maps.config import DateSelector, load_project_maps_config
from openamundsen_da.methods.viz.project_maps.data import (
    ModelFields,
    load_observation_scene,
    load_static_context,
    resolve_comparison_dates,
    resolve_observation_context_dates,
)
from openamundsen_da.methods.viz.project_maps.render import (
    _comparison_scales,
    _masked_model,
    buffered_extent,
    figure_height_for_extent,
)
from openamundsen_da.methods.viz.project_maps.runner import render_project_maps
from openamundsen_da.methods.viz.project_maps.styles import (
    INCREMENT_CMAP,
    SNOW_DEPTH_REFERENCE_TICKS_M,
    SNOW_DEPTH_REFERENCE_TICKLABELS_CM,
    model_colorbar_style,
    model_map_cmap,
    require_variable_preset,
)


def _write_yaml(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).strip() + "\n", encoding="utf-8")


def _write_grid(path: Path, array: np.ndarray, *, transform, crs: str = "EPSG:25832", nodata: float = -9999.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = "AAIGrid" if path.suffix.lower() == ".asc" else "GTiff"
    with rasterio.open(
        path,
        "w",
        driver=driver,
        dtype=str(array.dtype),
        nodata=nodata,
        width=int(array.shape[1]),
        height=int(array.shape[0]),
        count=1,
        crs=crs,
        transform=transform,
    ) as ds:
        ds.write(array, 1)


def _write_da_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(["2023-01-01", "2023-01-02"])
    base = np.array(
        [
            [
                [0.10, 0.15, 0.20, 0.25],
                [0.12, 0.17, 0.22, 0.27],
                [0.14, 0.19, 0.24, 0.29],
                [0.16, 0.21, 0.26, 0.31],
            ],
            [
                [0.20, 0.25, 0.30, 0.35],
                [0.22, 0.27, 0.32, 0.37],
                [0.24, 0.29, 0.34, 0.39],
                [0.26, 0.31, 0.36, 0.41],
            ],
        ],
        dtype=np.float32,
    )
    ds = xr.Dataset(
        {
            "open_loop_snowdepth_daily": (("time", "y", "x"), base),
            "ens_mean_snowdepth_daily": (("time", "y", "x"), base + 0.05),
            "increment_snowdepth_daily": (("time", "y", "x"), np.full(base.shape, 0.05, dtype=np.float32)),
            "open_loop_swe_daily": (("time", "y", "x"), base * 100.0),
            "ens_mean_swe_daily": (("time", "y", "x"), base * 100.0 + 10.0),
            "increment_swe_daily": (("time", "y", "x"), np.full(base.shape, 10.0, dtype=np.float32)),
            "open_loop_liquid_water_content": (
                ("time", "snow_layer", "y", "x"),
                np.stack([base / 30.0, base / 40.0, base / 50.0], axis=1),
            ),
            "ens_mean_liquid_water_content": (
                ("time", "snow_layer", "y", "x"),
                np.stack([base / 30.0, base / 40.0, base / 50.0], axis=1) + 0.01,
            ),
            "increment_liquid_water_content": (
                ("time", "snow_layer", "y", "x"),
                np.full((base.shape[0], 3, base.shape[1], base.shape[2]), 0.01, dtype=np.float32),
            ),
        },
        coords={"time": times, "snow_layer": np.arange(3), "y": np.arange(4), "x": np.arange(4)},
    )
    ds.to_netcdf(path)


def _write_station_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"id": "station_a", "name": "Station A", "x": 100.0, "y": 300.0, "alt": 2600.0},
            {"id": "station_b", "name": "Station B", "x": 300.0, "y": 100.0, "alt": 2700.0},
        ]
    ).to_csv(path, index=False)


def _write_station_netcdf(path: Path, *, lon: float, lat: float, alt: float, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "time": ("time", pd.to_datetime(["2023-01-01"])),
            "lon": xr.DataArray(lon),
            "lat": xr.DataArray(lat),
            "alt": xr.DataArray(alt),
            "temp": ("time", np.array([273.15], dtype=np.float32)),
        },
        attrs={"station_name": name},
    )
    ds.to_netcdf(path)


def _write_roi_vector(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(
        {"id": pd.Series(["roi"], dtype=object)},
        geometry=[box(0.0, 0.0, 400.0, 400.0)],
        crs="EPSG:25832",
    )
    gdf.to_file(path, driver="GPKG")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_project_fixture(tmp_path: Path, *, meteo_format: str = "csv") -> tuple[Path, Path]:
    setup_dir = tmp_path / f"setup_{meteo_format}"
    project_dir = setup_dir / "projects" / "project_demo"
    grids_dir = setup_dir / "custom_grids"
    transform = from_origin(0.0, 400.0, 100.0, 100.0)

    _write_yaml(
        setup_dir / "setup.yml",
        f"""
        domain: demo
        resolution: 100
        crs: "EPSG:25832"
        input_data:
          grids:
            dir: custom_grids
          meteo:
            dir: {"meteo_nc" if meteo_format == "netcdf" else "meteo"}
            format: {meteo_format}
            crs: "EPSG:25832"
        """,
    )
    _write_yaml(
        project_dir / "project_demo.yml",
        """
        start_date: "2023-01-01"
        end_date: "2023-01-03"
        obs:
          snowcover:
            dir: obs/snowcover
            product_tag: SNOWCOVER
            classes:
              valid: [0, 50, 90, 100]
              cloud: [205]
              water: [210]
              nodata: [255]
          wetsnow:
            dir: obs/wetsnow
            product_tag: WETSNOW
            classes:
              wet: [110]
              valid: [110, 125, 200, 210]
              exclude: [200, 210]
        data_assimilation:
          assimilation_events:
            - date: "2023-01-02"
              variable: scf
              product: SNOWCOVER
            - date: "2023-01-02"
              variable: wet_snow
              product: WETSNOW
        """,
    )

    dem = np.array(
        [
            [2800, 2820, 2840, 2860],
            [2780, 2800, 2820, 2840],
            [2760, 2780, 2800, 2820],
            [2740, 2760, 2780, 2800],
        ],
        dtype=np.float32,
    )
    landcover = np.array(
        [
            [1, 4, 8, 10],
            [1, 4, 8, 10],
            [2, 5, 9, 11],
            [3, 6, 12, 13],
        ],
        dtype=np.int16,
    )
    roi = np.ones((4, 4), dtype=np.uint8)
    _write_grid(grids_dir / "dem_demo_100.asc", dem, transform=transform)
    _write_grid(grids_dir / "lc_demo_100.asc", landcover, transform=transform, nodata=0)
    _write_grid(grids_dir / "roi_demo_100.asc", roi, transform=transform, nodata=0)
    _write_roi_vector(setup_dir / "env" / "roi.gpkg")
    _write_da_output(project_dir / "results" / "grids" / "da_output_grids.nc")

    if meteo_format == "csv":
        _write_station_csv(setup_dir / "meteo" / "stations.csv")
    else:
        _write_station_netcdf(
            setup_dir / "meteo_nc" / "station_alpha.nc",
            lon=11.0,
            lat=46.8,
            alt=2650.0,
            name="Station Alpha",
        )

    _write_summary(
        setup_dir / "obs" / "summaries" / project_dir.name / "scf_summary.csv",
        [
            {"date": "2023-01-01", "source": "scf_partial.tif"},
            {"date": "2023-01-02", "source": "scf_left.tif; scf_right.tif"},
        ],
    )
    _write_summary(
        setup_dir / "obs" / "summaries" / project_dir.name / "wet_snow_summary.csv",
        [
            {"date": "2023-01-02", "source": "wet_partial.tif"},
        ],
    )

    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_partial.tif",
        np.array(
            [
                [20, 20],
                [40, 40],
                [60, 60],
                [80, 80],
            ],
            dtype=np.float32,
        ),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_left.tif",
        np.array(
            [
                [10, 20],
                [10, 20],
                [10, 20],
                [10, 20],
            ],
            dtype=np.float32,
        ),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_right.tif",
        np.array(
            [
                [90, 100],
                [90, 100],
                [90, 100],
                [90, 100],
            ],
            dtype=np.float32,
        ),
        transform=from_origin(200.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "wetsnow" / "wet_partial.tif",
        np.array(
            [
                [110, 110],
                [110, 110],
                [200, 200],
                [200, 200],
            ],
            dtype=np.float32,
        ),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    return setup_dir, project_dir


def test_load_static_context_reads_csv_station_metadata_and_landcover_from_setup_grid_dir(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path, meteo_format="csv")

    context = load_static_context(project_dir)

    assert context.dem.shape == (4, 4)
    assert context.landcover.shape == (4, 4)
    assert set(context.stations["id"]) == {"station_a", "station_b"}


def test_load_static_context_reads_netcdf_station_metadata(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path, meteo_format="netcdf")

    context = load_static_context(project_dir)

    assert context.stations is not None
    assert list(context.stations["id"]) == ["station_alpha"]
    assert list(context.stations["name"]) == ["Station Alpha"]


def test_project_maps_config_and_date_resolution_follow_recipe_selectors(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "project_maps.yml"
    _write_yaml(
        config_path,
        """
        overview_maps:
          - name: overview
        comparison_maps:
          - name: snowdepth_window
            variable: snowdepth_daily
            dates:
              assimilation_variables: [scf]
              include_first: true
        observation_context_maps:
          - name: snowdepth_scf
            model_variable: snowdepth_daily
            observation: scf
            dates:
              explicit: ["2023-01-02"]
        """,
    )

    cfg = load_project_maps_config(config_path)
    comparison_dates = resolve_comparison_dates(
        project_dir,
        "snowdepth_daily",
        cfg.comparison_maps[0].dates,
    )
    observation_dates = resolve_observation_context_dates(
        project_dir,
        model_variable="snowdepth_daily",
        observation="scf",
        selector=cfg.observation_context_maps[0].dates,
    )

    assert [date.strftime("%Y-%m-%d") for date in comparison_dates] == ["2023-01-01", "2023-01-02"]
    assert [date.strftime("%Y-%m-%d") for date in observation_dates] == ["2023-01-02"]


def test_load_observation_scene_uses_setup_relative_obs_dir_and_reports_partial_coverage(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    scene = load_observation_scene(project_dir, context, observation="scf", date=pd.Timestamp("2023-01-01"))

    assert scene.coverage_fraction == 0.5
    assert np.isfinite(scene.array).sum() == 8


def test_load_observation_scene_mosaics_multiple_sources_and_masks_excluded_wet_snow_classes(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    scf_scene = load_observation_scene(project_dir, context, observation="scf", date=pd.Timestamp("2023-01-02"))
    wet_scene = load_observation_scene(project_dir, context, observation="wet_snow", date=pd.Timestamp("2023-01-02"))

    assert scf_scene.coverage_fraction == 1.0
    assert np.isfinite(scf_scene.array).sum() == 16
    assert wet_scene.coverage_fraction == 0.25
    assert set(np.unique(wet_scene.array[np.isfinite(wet_scene.array)])) == {110.0}


def test_buffered_extent_and_figure_height_follow_bounds(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    extent = buffered_extent(context)
    height = figure_height_for_extent(extent)

    assert extent == (-200.0, 600.0, -200.0, 600.0)
    assert 2.8 <= height <= 5.2


def test_comparison_scales_use_zero_centered_diverging_increment_norm() -> None:
    preset = require_variable_preset("snowdepth_daily")
    fields = [
        ModelFields(
            date=pd.Timestamp("2023-01-01"),
            open_loop=np.array([[0.1, 0.2]], dtype=float),
            ens_mean=np.array([[0.3, 0.4]], dtype=float),
            increment=np.array([[-0.05, 0.10]], dtype=float),
        ),
        ModelFields(
            date=pd.Timestamp("2023-01-02"),
            open_loop=np.array([[0.2, 0.3]], dtype=float),
            ens_mean=np.array([[0.4, 0.5]], dtype=float),
            increment=np.array([[-0.20, 0.05]], dtype=float),
        ),
    ]

    _model_norm, increment_norm = _comparison_scales(fields, preset)

    assert increment_norm.vcenter == 0.0
    assert increment_norm.vmin == -increment_norm.vmax


def test_comparison_scales_fix_snowdepth_model_range_to_reference_palette() -> None:
    preset = require_variable_preset("snowdepth_daily")
    fields = [
        ModelFields(
            date=pd.Timestamp("2023-01-01"),
            open_loop=np.array([[0.005, 0.20]], dtype=float),
            ens_mean=np.array([[0.30, 5.00]], dtype=float),
            increment=np.array([[-0.05, 0.10]], dtype=float),
        ),
    ]

    model_norm, _increment_norm = _comparison_scales(fields, preset)

    assert model_norm.vmin == SNOW_DEPTH_REFERENCE_TICKS_M[0]
    assert model_norm.vmax == SNOW_DEPTH_REFERENCE_TICKS_M[-1]


def test_snowdepth_model_mask_hides_values_below_one_centimeter() -> None:
    preset = require_variable_preset("snowdepth_daily")

    masked = _masked_model(
        np.array([[0.005, 0.010, 0.020]], dtype=float),
        np.ones((1, 3), dtype=bool),
        preset=preset,
    )

    assert masked.mask.tolist() == [[True, False, False]]


def test_snowdepth_model_palette_uses_reference_ticks_and_transparent_under_range() -> None:
    preset = require_variable_preset("snowdepth_daily")

    colorbar_style = model_colorbar_style(preset)
    cmap = model_map_cmap(preset)

    assert colorbar_style.label == "snow depth [cm]"
    assert colorbar_style.ticks == SNOW_DEPTH_REFERENCE_TICKS_M
    assert colorbar_style.ticklabels == SNOW_DEPTH_REFERENCE_TICKLABELS_CM
    assert cmap(-0.1)[3] == 0.0


def test_increment_cmap_runs_from_negative_red_to_positive_blue() -> None:
    low = INCREMENT_CMAP(0.0)
    high = INCREMENT_CMAP(1.0)

    assert low[0] > low[2]
    assert high[2] > high[0]


def test_render_project_maps_writes_overview_comparison_and_observation_outputs(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    _write_yaml(
        project_dir / "project_maps.yml",
        """
        overview_maps:
          - name: domain_overview
            title: Demo overview
        comparison_maps:
          - name: snowdepth_series
            variable: snowdepth_daily
            dates:
              include_first: true
              include_last: true
        observation_context_maps:
          - name: snowdepth_scf
            model_variable: snowdepth_daily
            observation: scf
            dates:
              explicit: ["2023-01-02"]
        """,
    )

    outputs = render_project_maps(project_dir=project_dir)

    expected = {
        project_dir / "results" / "maps" / "overview" / "domain_overview.png",
        project_dir / "results" / "maps" / "comparison" / "snowdepth_series_2023-01-01.png",
        project_dir / "results" / "maps" / "comparison" / "snowdepth_series_2023-01-02.png",
        project_dir / "results" / "maps" / "observation_context" / "snowdepth_scf.png",
    }

    assert set(outputs) == expected
    for path in outputs:
        assert path.is_file()
        assert path.stat().st_size > 0
