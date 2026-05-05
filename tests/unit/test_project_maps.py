from __future__ import annotations

from concurrent.futures import Future
import json
from pathlib import Path
import shutil
import textwrap

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import Point, box

import openamundsen_da.methods.viz.maps.render as render_module
import openamundsen_da.methods.viz.maps.runner as runner_module
import openamundsen_da.methods.viz.maps.data as data_module
import openamundsen_da.methods.viz.maps.generated as generated_module
import openamundsen_da.methods.viz.maps.overview as overview_module
import openamundsen_da.methods.viz.maps.panel_renderers as panel_renderers_module
from openamundsen_da.methods.viz.reports.project_collection_pdf import MissingProjectPdfArtifactsError
from openamundsen_da.methods.viz.maps.config import (
    DateSelector,
    LayoutSpec,
    MapDefaults,
    MapPanelSpec,
    MapRecipe,
    MapRowViewSpec,
    load_project_maps_config,
)
from openamundsen_da.methods.viz.maps.data import (
    ModelFields,
    ObservationScene,
    load_observation_scene,
    load_observation_uncertainty_scene,
    load_static_context,
    resolve_comparison_dates,
    resolve_observation_context_dates,
)
from openamundsen_da.methods.viz.maps.render import (
    _apply_map_axis_style,
    _comparison_scales,
    _draw_map_grid_overlay,
    _draw_scale_bar,
    _draw_stations_overlay,
    _horizontal_legend_row_layout,
    _horizontal_legend_total_extra,
    _masked_model,
    _overview_extent,
    _pack_horizontal_legend_rows,
    buffered_extent,
    figure_height_for_extent,
)
import openamundsen_da.pipeline.plot_tasks as plot_tasks_module
from openamundsen_da.methods.viz.maps.runner import project_maps_enabled, render_project_maps
from openamundsen_da.methods.viz.maps.styles import (
    FSC_OBS_CMAP,
    FSC_INVALID_COLOR,
    INCREMENT_CMAP,
    SNOW_DEPTH_REFERENCE_TICKS_M,
    WET_SNOW_COLORS,
    WET_SNOW_LABELS,
    model_colorbar_style,
    model_map_cmap,
    require_static_field_preset,
    require_variable_preset,
    snow_depth_colorbar_labels_cm,
    snow_depth_colorbar_ticklabels,
    snow_depth_colorbar_ticks,
    snow_depth_scale_ticks,
    static_field_cmap,
    static_field_colorbar_style,
)


PROJECT_MAPS_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "project_maps" / "rofental"


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
            "analysis_mean_snowdepth_daily": (("time", "y", "x"), base + 0.08),
            "analysis_increment_snowdepth_daily": (("time", "y", "x"), np.full(base.shape, 0.03, dtype=np.float32)),
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


def _mean_abs_png_diff(expected_path: Path, actual_path: Path) -> float:
    expected = np.asarray(plt.imread(expected_path), dtype=float)
    actual = np.asarray(plt.imread(actual_path), dtype=float)
    assert expected.shape == actual.shape
    return float(np.mean(np.abs(expected - actual)))


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


def _write_roi_vector(path: Path, *, bounds: tuple[float, float, float, float] = (0.0, 0.0, 400.0, 400.0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    xmin, ymin, xmax, ymax = bounds
    gdf = gpd.GeoDataFrame(
        {"id": pd.Series(["roi"], dtype=object)},
        geometry=[box(xmin, ymin, xmax, ymax)],
        crs="EPSG:25832",
    )
    gdf.to_file(path, driver="GPKG")


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_project_fixture(
    tmp_path: Path,
    *,
    meteo_format: str = "csv",
    roi_mask: np.ndarray | None = None,
    roi_bounds: tuple[float, float, float, float] | None = None,
) -> tuple[Path, Path]:
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
          h_of_x:
            method: depth_threshold
            variable: hs
            params:
              h0: 0.25
          wet_snow_line:
            elevation_band_size_m: 100.0
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
    svf = np.array(
        [
            [0.75, 0.78, 0.80, 0.82],
            [0.70, 0.74, 0.77, 0.81],
            [0.68, 0.71, 0.74, 0.78],
            [0.65, 0.68, 0.71, 0.75],
        ],
        dtype=np.float32,
    )
    srf = np.array(
        [
            [1.15, 1.20, 1.18, 1.12],
            [1.10, 1.14, 1.16, 1.08],
            [1.05, 1.10, 1.12, 1.04],
            [1.00, 1.04, 1.07, 1.02],
        ],
        dtype=np.float32,
    )
    roi = np.asarray(roi_mask, dtype=np.uint8) if roi_mask is not None else np.ones((4, 4), dtype=np.uint8)
    roi_vector_bounds = roi_bounds or (0.0, 0.0, 400.0, 400.0)
    _write_grid(grids_dir / "dem_demo_100.asc", dem, transform=transform)
    _write_grid(grids_dir / "lc_demo_100.asc", landcover, transform=transform, nodata=0)
    _write_grid(grids_dir / "svf_demo_100.asc", svf, transform=transform)
    _write_grid(grids_dir / "srf_demo_100.asc", srf, transform=transform)
    _write_grid(grids_dir / "roi_demo_100.asc", roi, transform=transform, nodata=0)
    _write_roi_vector(setup_dir / "env" / "roi.gpkg", bounds=roi_vector_bounds)
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
        np.array([[20, 20], [40, 40], [60, 60], [80, 80]], dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_partial_uncertainty.tif",
        np.array([[10, 10], [15, 15], [20, 20], [25, 25]], dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_left.tif",
        np.array([[10, 20], [10, 20], [10, 20], [10, 20]], dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_left_uncertainty.tif",
        np.array([[10, 20], [10, 20], [10, 20], [10, 20]], dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_right.tif",
        np.array([[90, 100], [90, 100], [90, 100], [90, 100]], dtype=np.float32),
        transform=from_origin(200.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_right_uncertainty.tif",
        np.array([[30, 40], [30, 40], [30, 40], [30, 40]], dtype=np.float32),
        transform=from_origin(200.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "wetsnow" / "wet_partial.tif",
        np.array([[110, 110], [110, 110], [200, 200], [200, 200]], dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "wetsnow" / "wet_partial_uncertainty.tif",
        np.array([[20, 20], [30, 30], [40, 40], [50, 50]], dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    return setup_dir, project_dir


def test_load_static_context_reads_csv_station_metadata_and_landcover_from_setup_grid_dir(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path, meteo_format="csv")

    context = load_static_context(project_dir)

    assert context.dem.shape == (4, 4)
    assert context.landcover.shape == (4, 4)
    assert context.svf is not None and context.svf.shape == (4, 4)
    assert context.srf is not None and context.srf.shape == (4, 4)
    assert set(context.stations["id"]) == {"station_a", "station_b"}


def test_load_static_context_reuses_in_process_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path, meteo_format="csv")
    data_module._load_static_context_cached.cache_clear()

    read_calls = 0
    original_read_dataset_array = data_module._read_dataset_array

    def counting_read_dataset_array(*args, **kwargs):
        nonlocal read_calls
        read_calls += 1
        return original_read_dataset_array(*args, **kwargs)

    monkeypatch.setattr(data_module, "_read_dataset_array", counting_read_dataset_array)

    first = load_static_context(project_dir)
    second = load_static_context(project_dir)

    assert first is second
    assert read_calls == 4
    data_module._load_static_context_cached.cache_clear()


def test_load_static_context_reads_netcdf_station_metadata(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path, meteo_format="netcdf")

    context = load_static_context(project_dir)

    assert context.stations is not None
    assert list(context.stations["id"]) == ["station_alpha"]
    assert list(context.stations["name"]) == ["Station Alpha"]


def test_overview_geojson_loaders_reuse_cached_reads_and_return_copies(monkeypatch: pytest.MonkeyPatch) -> None:
    overview_module._load_overview_geojson_cached.cache_clear()
    read_calls: list[str] = []
    sample = gpd.GeoDataFrame(
        {"name": pd.Series(["demo"], dtype=object)},
        geometry=[box(0.0, 0.0, 1.0, 1.0)],
        crs="EPSG:3857",
    )

    def fake_ensure(*, setup_dir=None, cache_dir=None, filename: str):
        del setup_dir, cache_dir
        return Path("/tmp") / filename

    def fake_read_file(path: Path):
        read_calls.append(Path(path).name)
        return sample.copy()

    monkeypatch.setattr(overview_module, "ensure_overview_countries_geojson", fake_ensure)
    monkeypatch.setattr(overview_module.gpd, "read_file", fake_read_file)

    first = overview_module.load_overview_boundaries()
    second = overview_module.load_overview_boundaries()
    labels = overview_module.load_overview_labels()
    regions = overview_module.load_overview_regions()

    assert first is not second
    assert list(first["name"]) == ["demo"]
    assert read_calls == [
        overview_module.GISCO_BOUNDARIES_GEOJSON_NAME,
        overview_module.GISCO_LABELS_GEOJSON_NAME,
        overview_module.GISCO_REGIONS_GEOJSON_NAME,
    ]
    assert list(labels["name"]) == ["demo"]
    assert list(regions["name"]) == ["demo"]
    overview_module._load_overview_geojson_cached.cache_clear()


def test_overview_geojson_path_resolves_under_setup_env(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup_demo"
    expected = setup_dir / "env" / overview_module.GISCO_BOUNDARIES_GEOJSON_NAME

    assert overview_module.overview_geojson_path(
        setup_dir=setup_dir,
        filename=overview_module.GISCO_BOUNDARIES_GEOJSON_NAME,
    ) == expected


def test_overview_cli_fetches_into_setup_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup_demo"
    calls: list[Path] = []

    def fake_ensure(*, setup_dir=None, cache_dir=None):
        del cache_dir
        calls.append(Path(setup_dir))
        return {name: Path(setup_dir) / "env" / name for name in (
            overview_module.GISCO_BOUNDARIES_GEOJSON_NAME,
            overview_module.GISCO_REGIONS_GEOJSON_NAME,
            overview_module.GISCO_LABELS_GEOJSON_NAME,
        )}

    monkeypatch.setattr(overview_module, "ensure_overview_geojsons", fake_ensure)

    exit_code = overview_module.cli_main(["--setup-dir", str(setup_dir)])

    assert exit_code == 0
    assert calls == [setup_dir.resolve()]


def test_project_maps_enabled_accepts_generated_events_without_maps_yml(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    assert project_maps_enabled(project_dir) is True

    (project_dir / "maps.yml").write_text("maps: {}\n", encoding="utf-8")

    assert project_maps_enabled(project_dir) is True


def test_project_maps_config_loads_keyed_grid_recipes(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          snowdepth_reference:
            title: Snow depth 2023/01/02
            output_name: snowdepth_2023-01-02
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 3
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
                title: Open loop
              - row: 0
                col: 1
                kind: snow_depth
                source: ensemble_mean
                title: Ensemble mean
              - row: 0
                col: 2
                kind: fsc
                title: Sentinel-2
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert [item.name for item in cfg.maps] == ["snowdepth_reference"]
    assert cfg.maps[0].layout.ncols == 3
    assert cfg.maps[0].defaults.date == "2023-01-02"
    assert cfg.maps[0].output_stem == "snowdepth_2023-01-02"


def test_project_maps_config_accepts_uncertainty_panel_observation(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          uncertainty_demo:
            title: uncertainty_demo
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: uncertainty
                observation: scf
                title: uncertainty
        """,
    )

    cfg = load_project_maps_config(config_path)

    panel = cfg.maps[0].panels[0]
    assert panel.kind == "uncertainty"
    assert panel.observation == "scf"


def test_project_maps_config_accepts_row_views(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          zoom_demo:
            title: zoom_demo
            layout:
              nrows: 2
              ncols: 1
            row_views:
              - row: 1
                center: [643767, 5191680]
                zoom: 13
                center_crs: EPSG:25832
                viewport_px: [800, 600]
            panels:
              - row: 0
                col: 0
                kind: roi
              - row: 1
                col: 0
                kind: roi
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert cfg.maps[0].row_views == (
        MapRowViewSpec(
            row=1,
            center=(643767.0, 5191680.0),
            zoom=13.0,
            center_crs="EPSG:25832",
            viewport_px=(800, 600),
        ),
    )


@pytest.mark.parametrize(
    ("row_view_yaml", "match"),
    [
        ("row: 2\n                center: [100, 200]\n                zoom: 13", "row must be < 2"),
        ("row: 1\n                center: [100, 200]\n                zoom: 0", "zoom must be > 0"),
        ("row: 1\n                center: [100]\n                zoom: 13", "center must contain exactly two numbers"),
        ("row: 1\n                center: [100, 200]\n                zoom: 13\n                viewport_px: [1024, 0]", "viewport_px\\[1\\] must be >= 1"),
    ],
)
def test_project_maps_config_rejects_invalid_row_views(
    tmp_path: Path,
    row_view_yaml: str,
    match: str,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        f"""
        maps:
          zoom_demo:
            title: zoom_demo
            layout:
              nrows: 2
              ncols: 1
            row_views:
              - {row_view_yaml}
            panels:
              - row: 0
                col: 0
                kind: roi
              - row: 1
                col: 0
                kind: roi
        """,
    )

    with pytest.raises(ValueError, match=match):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_duplicate_row_views(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          zoom_demo:
            title: zoom_demo
            layout:
              nrows: 2
              ncols: 1
            row_views:
              - row: 1
                center: [100, 200]
                zoom: 13
              - row: 1
                center: [120, 220]
                zoom: 12
            panels:
              - row: 0
                col: 0
                kind: roi
              - row: 1
                col: 0
                kind: roi
        """,
    )

    with pytest.raises(ValueError, match="duplicate row view"):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_panel_spanning_different_row_views(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          zoom_demo:
            title: zoom_demo
            layout:
              nrows: 2
              ncols: 1
            row_views:
              - row: 1
                center: [100, 200]
                zoom: 13
            panels:
              - row: 0
                col: 0
                rowspan: 2
                kind: roi
        """,
    )

    with pytest.raises(ValueError, match="spans rows with different row_views"):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_uncertainty_panel_without_observation(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          uncertainty_demo:
            title: uncertainty_demo
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: uncertainty
        """,
    )

    with pytest.raises(ValueError, match="observation is required for uncertainty panels"):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_invalid_uncertainty_observation(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          uncertainty_demo:
            title: uncertainty_demo
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: uncertainty
                observation: station_hs
        """,
    )

    with pytest.raises(ValueError, match="observation must be one of"):
        load_project_maps_config(config_path)


def test_project_maps_config_keeps_observation_key_removed_for_other_panels(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          fsc_demo:
            title: fsc_demo
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: fsc
                observation: scf
        """,
    )

    with pytest.raises(ValueError, match="removed panel keys: observation"):
        load_project_maps_config(config_path)


def test_project_maps_config_accepts_static_panels_and_legend_items(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_map:
            title: Setup map
            layout:
              nrows: 1
              ncols: 3
            panels:
              - row: 0
                col: 0
                kind: overview
                title: Overview
                scale: 1000000
                label_fit_margin: 0.04
                roi_label: Demo ROI
              - row: 0
                col: 1
                kind: svf
                title: Sky view factor
              - row: 0
                col: 2
                kind: legend
                items:
                  - kind: station_symbol
                    label: Demo stations
                  - kind: heading
                    label: Static setup context
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert cfg.maps[0].panels[0].kind == "overview"
    assert cfg.maps[0].panels[0].scale == 1000000
    assert cfg.maps[0].panels[0].label_fit_margin == 0.04
    assert cfg.maps[0].panels[1].kind == "svf"
    assert [item.kind for item in cfg.maps[0].panels[2].items] == ["station_symbol", "heading"]


def test_project_maps_config_accepts_below_panel_legend_items(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_overview:
            title: setup_overview
            output_name: setup_overview
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: roi
                title: region of interest
                below_items:
                  - kind: station_symbol
                    label: Meteorological stations
              - row: 0
                col: 1
                kind: dem
                title: digital elevation model
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert cfg.maps[0].output_stem == "setup_overview"
    assert [item.kind for item in cfg.maps[0].panels[0].below_items] == ["station_symbol"]
    assert cfg.maps[0].panels[0].below_items[0].label == "Meteorological stations"


def test_project_maps_config_accepts_hillshade_extent_on_defaults_and_panels(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          demo_map:
            title: demo_map
            layout:
              nrows: 1
              ncols: 2
            defaults:
              show_hillshade: true
              hillshade_extent: roi
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
                date: "2023-01-02"
              - row: 0
                col: 1
                kind: liquid_water_content
                source: increment
                date: "2023-01-02"
                hillshade_extent: full
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert cfg.maps[0].defaults.show_hillshade is True
    assert cfg.maps[0].defaults.hillshade_extent == "roi"
    assert cfg.maps[0].panels[1].hillshade_extent == "full"


def test_project_maps_config_rejects_invalid_hillshade_extent(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          demo_map:
            title: demo_map
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
                date: "2023-01-02"
                hillshade_extent: outside
        """,
    )

    with pytest.raises(ValueError, match="hillshade_extent"):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_below_items_on_legend_panel(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          bad_legend:
            title: bad_legend
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: legend
                below_items:
                  - kind: station_symbol
                    label: Demo stations
                items:
                  - kind: heading
                    label: Existing legend
        """,
    )

    with pytest.raises(ValueError, match="below_items is only supported for non-legend panels"):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_text_panel_kind(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          bad_recipe:
            title: Bad
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: text
                lines:
                  - "not supported anymore"
        """,
    )

    with pytest.raises(ValueError, match="no longer supported"):
        load_project_maps_config(config_path)


def test_project_maps_config_accepts_explicit_station_overlay_flags(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_map:
            title: Setup map
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: roi
                title: Region of interest
                show_station_marker: true
                show_stations_name: true
                show_stations_elev: false
        """,
    )

    cfg = load_project_maps_config(config_path)

    panel = cfg.maps[0].panels[0]
    assert panel.show_station_marker is True
    assert panel.show_stations_name is True
    assert panel.show_stations_elev is False


def test_project_maps_config_rejects_nonpositive_overview_label_fit_margin(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_map:
            title: Setup map
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: overview
                scale: 1000000
                label_fit_margin: 0
        """,
    )

    with pytest.raises(ValueError, match="label_fit_margin"):
        load_project_maps_config(config_path)


def test_project_maps_config_accepts_classified_legend_layout(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_map:
            title: Setup map
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: landcover
                legend: horizontal
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert cfg.maps[0].panels[0].legend == "horizontal"


def test_project_maps_config_rejects_removed_station_overlay_keys(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_map:
            title: Setup map
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: roi
                show_stations: true
        """,
    )

    with pytest.raises(ValueError, match="removed panel keys: show_stations"):
        load_project_maps_config(config_path)


def test_project_maps_config_rejects_unknown_legend_layout(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          bad_recipe:
            title: Bad
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: wet_snow
                legend: diagonal
        """,
    )

    with pytest.raises(ValueError, match="legend must be one of"):
        load_project_maps_config(config_path)


def test_project_maps_config_accepts_wet_snow_line_panel_kind(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          wsl_demo:
            title: WSLA demo
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 4
            panels:
              - row: 0
                col: 0
                kind: wet_snow_line
                source: open_loop
              - row: 0
                col: 1
                kind: wet_snow_line
                source: prior_probability
              - row: 0
                col: 2
                kind: wet_snow_line
                source: posterior_probability
              - row: 0
                col: 3
                kind: wet_snow_line
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert [panel.kind for panel in cfg.maps[0].panels] == [
        "wet_snow_line",
        "wet_snow_line",
        "wet_snow_line",
        "wet_snow_line",
    ]
    assert [panel.source for panel in cfg.maps[0].panels] == [
        "open_loop",
        "prior_probability",
        "posterior_probability",
        None,
    ]


def test_project_maps_config_accepts_four_column_probability_sources(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          probability_sources:
            title: Probability sources
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 2
              ncols: 4
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: analysis_mean
              - row: 0
                col: 1
                kind: snow_depth
                source: analysis_increment
              - row: 0
                col: 2
                kind: fsc
                source: prior_probability
              - row: 0
                col: 3
                kind: fsc
                source: posterior_probability
              - row: 1
                col: 0
                kind: wet_snow
                source: prior_probability
              - row: 1
                col: 1
                kind: wet_snow
                source: posterior_probability
              - row: 1
                col: 2
                kind: wet_snow_elevation_fraction
                source: prior_probability
              - row: 1
                col: 3
                kind: wet_snow_elevation_fraction
                source: posterior_probability
        """,
    )

    cfg = load_project_maps_config(config_path)

    assert [panel.source for panel in cfg.maps[0].panels] == [
        "analysis_mean",
        "analysis_increment",
        "prior_probability",
        "posterior_probability",
        "prior_probability",
        "posterior_probability",
        "prior_probability",
        "posterior_probability",
    ]


def test_shipped_rofental_project_maps_config_matches_curated_recipe_set() -> None:
    config_path = Path(__file__).resolve().parents[2] / "examples/rofental/projects/project_2022_2023/maps.yml"

    cfg = load_project_maps_config(config_path)

    assert [recipe.name for recipe in cfg.maps] == ["setup_overview"]
    assert [recipe.title for recipe in cfg.maps] == ["setup_overview"]
    assert [recipe.output_stem for recipe in cfg.maps] == ["setup_overview"]
    assert cfg.maps[0].layout.nrows == 2
    assert cfg.maps[0].layout.ncols == 3
    assert [panel.title for panel in cfg.maps[0].panels] == [
        "overview",
        "region of interest",
        "digital elevation model",
        "landcover",
        "hillshade",
        "snow redistribution factor",
    ]
    assert [(panel.row, panel.col) for panel in cfg.maps[0].panels] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    assert cfg.maps[0].panels[3].name is None
    assert cfg.maps[0].panels[1].below_items[0].label == "Meteorological stations"
    assert cfg.maps[0].panels[5].show_hillshade is True


def test_generated_da_map_recipes_build_stable_da_event_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(generated_module, "_fraction_model_support_available", lambda *_args, **_kwargs: False)

    recipes = generated_module.generated_da_map_recipes(project_dir)

    assert [recipe.name for recipe in recipes] == ["da_1", "da_2"]
    assert all(recipe.output_subdir == generated_module.GENERATED_DA_MAPS_SUBDIR for recipe in recipes)
    assert recipes[0].figure_title == "DA 1 - 2023-01-02 (snow cover fraction)"
    assert recipes[1].figure_title == "DA 2 - 2023-01-02 (wet snow fraction (WSF))"
    assert recipes[0].row_labels == ("station snow depth",)
    assert recipes[0].layout.ncols == 4
    assert [panel.title for panel in recipes[0].panels] == ["open loop", "prior mean", "posterior mean", "posterior - prior"]
    assert [panel.source for panel in recipes[0].panels] == [
        "open_loop",
        "ensemble_mean",
        "analysis_mean",
        "analysis_increment",
    ]


def test_generated_da_map_title_marks_skipped_resampling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(generated_module, "_fraction_model_support_available", lambda *_args, **_kwargs: False)
    manifest_path = project_dir / "steps" / "step_01_20230101-20230102" / "assim" / "resample_manifest_20230102.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"skipped": True}), encoding="utf-8")

    recipes = generated_module.generated_da_map_recipes(project_dir)

    assert recipes[0].figure_title == "DA 1 - 2023-01-02 (snow cover fraction) - resampling skipped"


def test_generated_da_map_recipes_use_probabilistic_scf_panels_when_fraction_support_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(
        generated_module,
        "_fraction_model_support_available",
        lambda _project_dir, variable: variable == "scf",
    )

    recipes = generated_module.generated_da_map_recipes(project_dir)

    assert recipes[0].row_labels[0] == "snow cover"
    scf_panels = recipes[0].panels[:8]
    assert recipes[0].layout.ncols == 4
    assert [panel.source for panel in scf_panels[:4]] == [
        "open_loop_binary",
        "prior_probability",
        "posterior_probability",
        None,
    ]
    assert [panel.title for panel in scf_panels[:4]] == [
        "open-loop snow cover",
        "prior snow cover",
        "posterior snow cover",
        "satellite FSC observation",
    ]


def test_generated_da_map_recipes_use_true_wsl_panels_for_wet_snow_line_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(
        generated_module,
        "load_assimilation_events",
        lambda _project_dir: (
            generated_module.AssimilationEvent(
                date=pd.Timestamp("2023-01-02").date(),
                variable="wet_snow_line",
                product="WETSNOW",
            ),
        ),
    )
    monkeypatch.setattr(
        generated_module,
        "_fraction_model_support_available",
        lambda _project_dir, variable: variable == "wet_snow",
    )

    recipes = generated_module.generated_da_map_recipes(project_dir)

    assert recipes[0].figure_title == "DA 1 - 2023-01-02 (wet snow line altitude - WSLA)"
    assert recipes[0].row_labels[0] == "wet snow line altitude (WSLA)"
    assert recipes[0].row_labels[1] == "elevation-band WSF"
    assert recipes[0].layout.ncols == 4
    assert [panel.kind for panel in recipes[0].panels[:4]] == [
        "wet_snow_line",
        "wet_snow_line",
        "wet_snow_line",
        "wet_snow_line",
    ]
    assert [panel.source for panel in recipes[0].panels[:4]] == [
        "open_loop",
        "prior_probability",
        "posterior_probability",
        None,
    ]
    assert [panel.title for panel in recipes[0].panels[:4]] == ["open loop", "prior", "posterior", "observation"]
    assert [panel.kind for panel in recipes[0].panels[4:8]] == [
        "wet_snow_elevation_fraction",
        "wet_snow_elevation_fraction",
        "wet_snow_elevation_fraction",
        "wet_snow_elevation_fraction",
    ]
    assert [panel.source for panel in recipes[0].panels[4:8]] == [
        "open_loop",
        "prior_probability",
        "posterior_probability",
        None,
    ]
    assert [panel.variable for panel in recipes[0].panels[4:8]] == [
        "wet_snow_line",
        "wet_snow_line",
        "wet_snow_line",
        "wet_snow_line",
    ]


def test_generated_da_map_recipes_add_elevation_band_wsf_row_for_wet_snow_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(
        generated_module,
        "_fraction_model_support_available",
        lambda _project_dir, variable: variable == "wet_snow",
    )

    recipes = generated_module.generated_da_map_recipes(project_dir)

    wet_recipe = recipes[1]
    assert wet_recipe.row_labels[:2] == ("wet snow fraction (WSF)", "elevation-band WSF")
    assert wet_recipe.layout.ncols == 4
    assert [panel.kind for panel in wet_recipe.panels[:4]] == ["wet_snow", "wet_snow", "wet_snow", "wet_snow"]
    assert [panel.source for panel in wet_recipe.panels[:4]] == [
        "open_loop",
        "prior_probability",
        "posterior_probability",
        None,
    ]
    assert [panel.kind for panel in wet_recipe.panels[4:8]] == [
        "wet_snow_elevation_fraction",
        "wet_snow_elevation_fraction",
        "wet_snow_elevation_fraction",
        "wet_snow_elevation_fraction",
    ]
    assert [panel.source for panel in wet_recipe.panels[4:8]] == [
        "open_loop",
        "prior_probability",
        "posterior_probability",
        None,
    ]
    assert [panel.variable for panel in wet_recipe.panels[4:8]] == ["wet_snow", "wet_snow", "wet_snow", "wet_snow"]


def test_scf_probability_model_resolves_posterior_member_source_pointers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    step_dir = project_dir / "steps" / "step_01_20230101-20230102"
    source_member_dir = step_dir / "ensembles" / "prior" / "member_002"
    (source_member_dir / "results").mkdir(parents=True, exist_ok=True)
    posterior_member_dir = step_dir / "ensembles" / "posterior" / "member_001"
    posterior_member_dir.mkdir(parents=True, exist_ok=True)
    (posterior_member_dir / "source_pointer.json").write_text(
        json.dumps({"member_dir": str(source_member_dir)}, indent=2),
        encoding="utf-8",
    )

    seen_results_dirs: list[Path] = []

    monkeypatch.setattr(panel_renderers_module, "_step_dir_for_date", lambda *_args, **_kwargs: step_dir)
    monkeypatch.setattr(panel_renderers_module, "list_member_dirs", lambda *_args, **_kwargs: [posterior_member_dir])

    def fake_binary_grid(**kwargs):
        seen_results_dirs.append(Path(kwargs["results_dir"]))
        return np.ones((4, 4), dtype=float)

    monkeypatch.setattr(panel_renderers_module, "_scf_binary_grid_from_results", fake_binary_grid)

    probability = panel_renderers_module._scf_model_probability_array(
        context=context,
        source="posterior_probability",
        date=pd.Timestamp("2023-01-02"),
        derived_cache={},
    )

    assert seen_results_dirs == [source_member_dir / "results"]
    assert np.allclose(probability[context.roi_mask], 1.0)


def test_scf_binary_map_uses_full_roi_without_landcover_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    captured: dict[str, object] = {}

    def fake_compute_model_scf_binary_grid(**kwargs):
        captured.update(kwargs)
        return np.ones((4, 4), dtype=float)

    monkeypatch.setattr(panel_renderers_module, "compute_model_scf_binary_grid", fake_compute_model_scf_binary_grid)

    panel_renderers_module._scf_binary_grid_from_results(
        context=context,
        results_dir=project_dir / "steps" / "step_01_20230101-20230102" / "ensembles" / "prior" / "open_loop" / "results",
        date=pd.Timestamp("2023-01-02"),
    )

    assert captured["apply_landcover_mask"] is False


def test_reference_stream_uses_variable_name_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    target_date = pd.Timestamp("2023-01-02")

    monkeypatch.setattr(generated_module, "_fraction_summary_dates", lambda *_args, **_kwargs: {target_date})
    monkeypatch.setattr(generated_module, "_event_dates_by_variable", lambda *_args, **_kwargs: {"scf": {pd.Timestamp("2023-02-01")}})

    assert generated_module._reference_stream(project_dir, variable="scf", date=target_date) == "snow cover (independent)"


def test_render_project_maps_generates_da_event_maps_under_subdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(generated_module, "_fraction_model_support_available", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", generated_module.generated_da_map_recipes)

    outputs = render_project_maps(project_dir=project_dir, max_workers=1)

    assert outputs == [
        project_dir / "results" / "maps" / "da_events" / "da_1.png",
        project_dir / "results" / "maps" / "da_events" / "da_2.png",
    ]
    for output in outputs:
        assert output.is_file()


def test_project_maps_config_rejects_overlapping_panels(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          bad_recipe:
            title: Bad
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: hillshade
              - row: 0
                col: 0
                kind: landcover
        """,
    )

    try:
        load_project_maps_config(config_path)
    except ValueError as exc:
        assert "Overlapping panel placement" in str(exc)
    else:
        raise AssertionError("Expected overlap validation failure")


def test_date_resolution_helpers_follow_selectors(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)

    comparison_dates = resolve_comparison_dates(
        project_dir,
        "snowdepth_daily",
        DateSelector(assimilation_variables=("scf",), include_first=True),
    )
    observation_dates = resolve_observation_context_dates(
        project_dir,
        model_variable="snowdepth_daily",
        observation="scf",
        selector=DateSelector(explicit=("2023-01-02",)),
    )

    assert [date.strftime("%Y-%m-%d") for date in comparison_dates] == ["2023-01-01", "2023-01-02"]
    assert [date.strftime("%Y-%m-%d") for date in observation_dates] == ["2023-01-02"]


def test_load_observation_scene_uses_setup_relative_obs_dir_and_reports_partial_coverage(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    scene = load_observation_scene(project_dir, context, observation="scf", date=pd.Timestamp("2023-01-01"))

    assert scene.coverage_fraction == 0.5
    assert np.isfinite(scene.array).sum() == 8


def test_load_observation_scene_ignores_source_timestamp_token(tmp_path: Path) -> None:
    setup_dir, project_dir = _build_project_fixture(tmp_path)
    summary_path = setup_dir / "obs" / "summaries" / project_dir.name / "scf_summary.csv"
    pd.DataFrame(
        [
            {"date": "2023-01-02", "source": "scf_left.tif@2023-01-02T00:00:00Z; scf_right.tif@2023-01-02T00:00:00Z"},
        ]
    ).to_csv(summary_path, index=False)
    context = load_static_context(project_dir)

    scene = load_observation_scene(project_dir, context, observation="scf", date=pd.Timestamp("2023-01-02"))

    assert scene.coverage_fraction == 1.0
    assert np.isfinite(scene.array).sum() == 16


def test_load_observation_scene_keeps_visual_wet_snow_classes(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    scf_scene = load_observation_scene(project_dir, context, observation="scf", date=pd.Timestamp("2023-01-02"))
    wet_scene = load_observation_scene(project_dir, context, observation="wet_snow", date=pd.Timestamp("2023-01-02"))

    assert scf_scene.coverage_fraction == 1.0
    assert np.isfinite(scf_scene.array).sum() == 16
    assert wet_scene.coverage_fraction == 0.5
    assert set(np.unique(wet_scene.array[np.isfinite(wet_scene.array)])) == {110.0, 200.0}


def test_load_observation_uncertainty_scene_resolves_sidecar_mosaic(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    scene = load_observation_uncertainty_scene(
        project_dir,
        context,
        observation="scf",
        date=pd.Timestamp("2023-01-02"),
    )

    assert scene.coverage_fraction == 1.0
    assert np.isfinite(scene.array).sum() == 16
    assert scene.array[0, 0] == 10.0
    assert scene.array[0, 1] == 20.0
    assert scene.array[0, 2] == 30.0
    assert scene.array[0, 3] == 40.0


def test_load_observation_uncertainty_scene_masks_invalid_observation_pixels(tmp_path: Path) -> None:
    setup_dir, project_dir = _build_project_fixture(tmp_path)
    _write_summary(
        setup_dir / "obs" / "summaries" / project_dir.name / "scf_summary.csv",
        [{"date": "2023-01-03", "source": "scf_cloud.tif"}],
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_cloud.tif",
        np.array(
            [
                [20, 205, 40, 50],
                [20, 255, 40, 50],
                [20, 20, 40, 50],
                [20, 20, 40, 50],
            ],
            dtype=np.float32,
        ),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    _write_grid(
        setup_dir / "obs" / "snowcover" / "scf_cloud_uncertainty.tif",
        np.full((4, 4), 35.0, dtype=np.float32),
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
        nodata=255.0,
    )
    context = load_static_context(project_dir)

    scene = load_observation_uncertainty_scene(
        project_dir,
        context,
        observation="scf",
        date=pd.Timestamp("2023-01-03"),
    )

    assert np.isnan(scene.array[0, 1])
    assert np.isnan(scene.array[1, 1])
    assert scene.invalid_mask is not None
    assert scene.invalid_mask[0, 1]
    assert scene.invalid_mask[1, 1]
    assert scene.array[0, 0] == 35.0


def test_load_observation_uncertainty_scene_requires_sidecars(tmp_path: Path) -> None:
    setup_dir, project_dir = _build_project_fixture(tmp_path)
    (setup_dir / "obs" / "snowcover" / "scf_right_uncertainty.tif").unlink()
    context = load_static_context(project_dir)

    with pytest.raises(FileNotFoundError, match="Observation uncertainty raster not found"):
        load_observation_uncertainty_scene(
            project_dir,
            context,
            observation="scf",
            date=pd.Timestamp("2023-01-02"),
        )


def test_buffered_extent_and_figure_height_follow_bounds(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    extent = buffered_extent(context)
    height = figure_height_for_extent(extent)

    assert extent == (-200.0, 600.0, -200.0, 600.0)
    assert 2.9 <= height <= 4.8


def test_google_zoom_meters_per_pixel_uses_slippy_map_scale() -> None:
    equator_zoom_1 = render_module._google_zoom_meters_per_pixel(0.0, 1.0)
    latitude_60_zoom_1 = render_module._google_zoom_meters_per_pixel(60.0, 1.0)

    assert equator_zoom_1 == pytest.approx(2.0 * np.pi * 6378137.0 / (256.0 * 2.0))
    assert latitude_60_zoom_1 == pytest.approx(0.5 * equator_zoom_1)


def test_row_view_extent_uses_zoom_center_and_viewport(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    view = MapRowViewSpec(
        row=1,
        center=(9.0, 60.0),
        zoom=10.0,
        center_crs="EPSG:4326",
        viewport_px=(512, 256),
    )

    extent = render_module._row_view_extent(view, context)

    meters_per_pixel = render_module._google_zoom_meters_per_pixel(60.0, 10.0)
    assert extent[1] - extent[0] == pytest.approx(512.0 * meters_per_pixel)
    assert extent[3] - extent[2] == pytest.approx(256.0 * meters_per_pixel)


def test_row_extents_use_full_extent_unless_row_view_is_configured(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    recipe = MapRecipe(
        name="zoom_demo",
        title="zoom_demo",
        layout=LayoutSpec(nrows=2, ncols=1),
        panels=(
            MapPanelSpec(kind="roi", row=0, col=0),
            MapPanelSpec(kind="roi", row=1, col=0),
        ),
        row_views=(MapRowViewSpec(row=1, center=(200.0, 200.0), zoom=19.0, viewport_px=(1024, 512)),),
    )

    row_extents = render_module._row_extents_for_recipe(recipe, context)
    row_ratios = render_module._effective_row_height_ratios(recipe, row_extents=row_extents)

    assert row_extents[0] == buffered_extent(context)
    assert row_extents[1] != row_extents[0]
    assert row_ratios[0] == pytest.approx(1.0)
    assert row_ratios[1] == pytest.approx(0.5)


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
            analysis_increment=np.array([[0.35, -0.10]], dtype=float),
        ),
    ]

    _model_norm, increment_norm = _comparison_scales(fields, preset)

    assert increment_norm.vcenter == 0.0
    assert increment_norm.vmin == -increment_norm.vmax
    assert increment_norm.vmax == pytest.approx(0.4)


def test_comparison_scales_stretch_snowdepth_model_range_to_data_ceiling() -> None:
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
    assert model_norm.vmax == 5.0


def test_comparison_scales_allow_shared_snowdepth_upper_bound() -> None:
    preset = require_variable_preset("snowdepth_daily")
    fields = [
        ModelFields(
            date=pd.Timestamp("2023-01-01"),
            open_loop=np.array([[0.05, 0.20]], dtype=float),
            ens_mean=np.array([[0.30, 0.40]], dtype=float),
            increment=np.array([[-0.05, 0.10]], dtype=float),
        ),
    ]

    model_norm, increment_norm = _comparison_scales(fields, preset, model_vmax=1.5)

    assert model_norm.vmin == SNOW_DEPTH_REFERENCE_TICKS_M[0]
    assert model_norm.vmax == 1.5
    assert increment_norm.vmin == -0.25
    assert increment_norm.vmax == 0.25


def test_snowdepth_model_mask_hides_values_below_one_centimeter() -> None:
    preset = require_variable_preset("snowdepth_daily")

    masked = _masked_model(
        np.array([[0.005, 0.010, 0.020]], dtype=float),
        np.ones((1, 3), dtype=bool),
        preset=preset,
    )

    assert masked.mask.tolist() == [[True, False, False]]


def test_static_panels_keep_context_outside_roi(tmp_path: Path) -> None:
    roi_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _setup_dir, project_dir = _build_project_fixture(
        tmp_path,
        roi_mask=roi_mask,
    )
    context = load_static_context(project_dir)
    extent = buffered_extent(context)
    grid_extent = render_module._grid_extent(context)

    hillshade = render_module._hillshade(context)
    assert not context.roi_mask[0, 0]
    assert np.isfinite(hillshade[0, 0])

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    try:
        dem_artifact = render_module._render_static_panel(
            axes[0],
            panel=MapPanelSpec(kind="dem", row=0, col=0, show_colorbar=False),
            context=context,
            extent=extent,
            grid_extent=grid_extent,
            label=None,
            defaults=MapDefaults(),
            figure_horizontal_default=True,
        )
        landcover_artifact = render_module._render_static_panel(
            axes[1],
            panel=MapPanelSpec(kind="landcover", row=0, col=0),
            context=context,
            extent=extent,
            grid_extent=grid_extent,
            label=None,
            defaults=MapDefaults(),
            figure_horizontal_default=True,
        )

        dem_array = np.ma.asarray(dem_artifact["mappable"].get_array())
        landcover_array = np.ma.asarray(landcover_artifact["mappable"].get_array())
        assert not np.ma.getmaskarray(dem_array)[0, 0]
        assert np.isfinite(dem_array[0, 0])
        assert not np.ma.getmaskarray(landcover_array)[0, 0]
        assert np.isfinite(landcover_array[0, 0])
    finally:
        plt.close(fig)


def test_model_panels_remain_masked_outside_roi(tmp_path: Path) -> None:
    roi_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _setup_dir, project_dir = _build_project_fixture(
        tmp_path,
        roi_mask=roi_mask,
        roi_bounds=(100.0, 100.0, 300.0, 300.0),
    )
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        artifact = render_module._render_model_panel(
            ax,
            panel=MapPanelSpec(
                kind="snow_depth",
                row=0,
                col=0,
                source="open_loop",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(),
            model_cache={},
            scale_cache={},
            figure_horizontal_default=True,
        )
        model_array = np.ma.asarray(artifact["mappable"].get_array())
        assert np.ma.getmaskarray(model_array)[0, 0]
        assert not np.ma.getmaskarray(model_array)[1, 1]
    finally:
        plt.close(fig)


def test_fsc_probability_model_panel_renders_probability_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        monkeypatch.setattr(
            panel_renderers_module,
            "_scf_model_probability_array",
            lambda **_kwargs: np.array(
                [
                    [0.0, 0.5, 1.0, np.nan],
                    [0.0, 0.5, 1.0, np.nan],
                    [0.0, 0.5, 1.0, np.nan],
                    [0.0, 0.5, 1.0, np.nan],
                ],
                dtype=float,
            ),
        )
        artifact = render_module._render_observation_panel(
            ax,
            panel=MapPanelSpec(
                kind="fsc",
                row=0,
                col=0,
                source="posterior_probability",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )
        model_array = np.ma.asarray(artifact["mappable"].get_array())
        assert float(model_array[0, 0]) == 0.0
        assert float(model_array[0, 1]) == 50.0
        assert float(model_array[0, 2]) == 100.0
        assert isinstance(artifact["mappable"].norm, matplotlib.colors.Normalize)
    finally:
        plt.close(fig)


def test_fsc_prior_probability_model_panel_renders_probability_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    captured_sources: list[str] = []
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        def fake_scf_probability(**kwargs):
            captured_sources.append(str(kwargs["source"]))
            return np.array(
                [
                    [0.25, 0.5, 0.75, np.nan],
                    [0.25, 0.5, 0.75, np.nan],
                    [0.25, 0.5, 0.75, np.nan],
                    [0.25, 0.5, 0.75, np.nan],
                ],
                dtype=float,
            )

        monkeypatch.setattr(panel_renderers_module, "_scf_model_probability_array", fake_scf_probability)
        artifact = render_module._render_observation_panel(
            ax,
            panel=MapPanelSpec(
                kind="fsc",
                row=0,
                col=0,
                source="prior_probability",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )
        model_array = np.ma.asarray(artifact["mappable"].get_array())
        assert captured_sources == ["prior_probability"]
        assert float(model_array[0, 0]) == 25.0
        assert float(model_array[0, 1]) == 50.0
        assert float(model_array[0, 2]) == 75.0
        assert isinstance(artifact["mappable"].norm, matplotlib.colors.Normalize)
    finally:
        plt.close(fig)


def test_fsc_open_loop_model_panel_renders_binary_field_discretely(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        monkeypatch.setattr(
            panel_renderers_module,
            "_scf_model_probability_array",
            lambda **_kwargs: np.array(
                [
                    [0.0, 1.0, np.nan, 0.0],
                    [1.0, 0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0],
                ],
                dtype=float,
            ),
        )
        artifact = render_module._render_observation_panel(
            ax,
            panel=MapPanelSpec(
                kind="fsc",
                row=0,
                col=0,
                source="open_loop_binary",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )
        model_array = np.ma.asarray(artifact["mappable"].get_array())
        assert float(model_array[0, 0]) == 0.0
        assert float(model_array[0, 1]) == 1.0
        assert isinstance(artifact["mappable"].norm, matplotlib.colors.BoundaryNorm)
    finally:
        plt.close(fig)


def test_wet_snow_probability_model_panel_renders_spatial_wsf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    wet_fraction = np.array(
        [
            [0.0, 0.4, 1.0, np.nan],
            [0.0, 0.4, 1.0, np.nan],
            [0.0, 0.4, 1.0, np.nan],
            [0.0, 0.4, 1.0, np.nan],
        ],
        dtype=float,
    )
    monkeypatch.setattr(panel_renderers_module, "_prior_wet_fraction_array", lambda **_kwargs: wet_fraction)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        artifact = render_module._render_observation_panel(
            ax,
            panel=MapPanelSpec(
                kind="wet_snow",
                row=0,
                col=0,
                source="prior_probability",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )
        model_array = np.ma.asarray(artifact["mappable"].get_array())
        assert float(model_array[0, 0]) == 0.0
        assert float(model_array[0, 1]) == 40.0
        assert float(model_array[0, 2]) == 100.0
        assert isinstance(artifact["mappable"].norm, matplotlib.colors.Normalize)
    finally:
        plt.close(fig)


def test_invalid_pixels_inside_roi_use_fsc_invalid_overlay_color(tmp_path: Path) -> None:
    roi_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _setup_dir, project_dir = _build_project_fixture(
        tmp_path,
        roi_mask=roi_mask,
        roi_bounds=(100.0, 100.0, 300.0, 300.0),
    )
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        context.dem[1, 1] = np.nan
        render_module._render_static_panel(
            ax,
            panel=MapPanelSpec(kind="dem", row=0, col=0, show_colorbar=False),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(),
            figure_horizontal_default=True,
        )
        assert len(ax.images) >= 2
        overlay = np.asarray(ax.images[-1].get_array(), dtype=float)
        pink = matplotlib.colors.to_rgba(FSC_INVALID_COLOR)
        assert np.allclose(overlay[1, 1], pink)
        assert overlay[0, 0, 3] == 0.0
    finally:
        plt.close(fig)


def test_model_panel_hillshade_defaults_to_off_for_snow_depth(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        render_module._render_model_panel(
            ax,
            panel=MapPanelSpec(
                kind="snow_depth",
                row=0,
                col=0,
                source="open_loop",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(),
            model_cache={},
            scale_cache={},
            figure_horizontal_default=True,
        )
        assert len(ax.images) == 1
    finally:
        plt.close(fig)


def test_model_panel_hillshade_respects_recipe_default_and_panel_override(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig_default, ax_default = plt.subplots(figsize=(3, 3))
    fig_override, ax_override = plt.subplots(figsize=(3, 3))
    try:
        render_module._render_model_panel(
            ax_default,
            panel=MapPanelSpec(
                kind="snow_depth",
                row=0,
                col=0,
                source="open_loop",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(show_hillshade=True),
            model_cache={},
            scale_cache={},
            figure_horizontal_default=True,
        )
        render_module._render_model_panel(
            ax_override,
            panel=MapPanelSpec(
                kind="snow_depth",
                row=0,
                col=0,
                source="open_loop",
                date="2023-01-02",
                show_colorbar=False,
                show_hillshade=False,
            ),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(show_hillshade=True),
            model_cache={},
            scale_cache={},
            figure_horizontal_default=True,
        )
        assert len(ax_default.images) == 2
        assert len(ax_override.images) == 1
    finally:
        plt.close(fig_default)
        plt.close(fig_override)


def test_model_panel_hillshade_extent_can_switch_between_full_and_roi(tmp_path: Path) -> None:
    roi_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _setup_dir, project_dir = _build_project_fixture(
        tmp_path,
        roi_mask=roi_mask,
        roi_bounds=(100.0, 100.0, 300.0, 300.0),
    )
    context = load_static_context(project_dir)
    fig_full, ax_full = plt.subplots(figsize=(3, 3))
    fig_roi, ax_roi = plt.subplots(figsize=(3, 3))
    try:
        render_module._render_model_panel(
            ax_full,
            panel=MapPanelSpec(
                kind="liquid_water_content",
                row=0,
                col=0,
                source="increment",
                date="2023-01-02",
                show_colorbar=False,
            ),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(show_hillshade=True, hillshade_extent="full"),
            model_cache={},
            scale_cache={},
            figure_horizontal_default=True,
        )
        render_module._render_model_panel(
            ax_roi,
            panel=MapPanelSpec(
                kind="liquid_water_content",
                row=0,
                col=0,
                source="increment",
                date="2023-01-02",
                show_colorbar=False,
                hillshade_extent="roi",
            ),
            context=context,
            extent=buffered_extent(context),
            grid_extent=render_module._grid_extent(context),
            label=None,
            defaults=MapDefaults(show_hillshade=True, hillshade_extent="full"),
            model_cache={},
            scale_cache={},
            figure_horizontal_default=True,
        )
        full_underlay = np.ma.asarray(ax_full.images[0].get_array())
        roi_underlay = np.ma.asarray(ax_roi.images[0].get_array())
        assert not np.ma.getmaskarray(full_underlay)[0, 0]
        assert np.ma.getmaskarray(roi_underlay)[0, 0]
        assert not np.ma.getmaskarray(roi_underlay)[1, 1]
    finally:
        plt.close(fig_full)
        plt.close(fig_roi)


def test_observation_panel_hillshade_can_be_disabled_or_limited_to_roi(tmp_path: Path) -> None:
    roi_mask = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    _setup_dir, project_dir = _build_project_fixture(
        tmp_path,
        roi_mask=roi_mask,
        roi_bounds=(100.0, 100.0, 300.0, 300.0),
    )
    context = load_static_context(project_dir)
    fig_off, ax_off = plt.subplots(figsize=(3, 3))
    fig_roi, ax_roi = plt.subplots(figsize=(3, 3))
    try:
        render_module._render_observation_panel(
            ax_off,
            panel=MapPanelSpec(kind="wet_snow", row=0, col=0, date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(show_hillshade=False),
            obs_cache={},
            figure_horizontal_default=True,
        )
        render_module._render_observation_panel(
            ax_roi,
            panel=MapPanelSpec(kind="wet_snow", row=0, col=0, date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(show_hillshade=True, hillshade_extent="roi"),
            obs_cache={},
            figure_horizontal_default=True,
        )
        assert len(ax_off.images) == 2
        assert len(ax_roi.images) == 3
        off_overlay = np.asarray(ax_off.images[-1].get_array(), dtype=float)
        assert off_overlay.shape[-1] == 4
        roi_underlay = np.ma.asarray(ax_roi.images[0].get_array())
        assert np.ma.getmaskarray(roi_underlay)[0, 0]
        assert not np.ma.getmaskarray(roi_underlay)[1, 1]
    finally:
        plt.close(fig_off)
        plt.close(fig_roi)


def test_snowdepth_model_palette_uses_dynamic_ticks_and_transparent_under_range() -> None:
    preset = require_variable_preset("snowdepth_daily")
    vmax = 2.5

    colorbar_style = model_colorbar_style(preset, vmax=vmax)
    cmap = model_map_cmap(preset)

    assert colorbar_style.label == "snow depth [cm]"
    assert colorbar_style.ticks == snow_depth_colorbar_ticks(vmax)
    assert colorbar_style.ticklabels == snow_depth_colorbar_ticklabels(vmax)
    assert colorbar_style.ticks[0] == SNOW_DEPTH_REFERENCE_TICKS_M[0]
    assert colorbar_style.ticks[-1] == vmax
    assert cmap(-0.1)[3] == 0.0


def test_snowdepth_colorbar_ticks_keep_reference_style_with_dynamic_top_bin() -> None:
    vmax = 2.25

    assert snow_depth_scale_ticks(vmax)[-1] == vmax
    assert snow_depth_colorbar_labels_cm(vmax) == (1.0, 50.0, 100.0, 150.0, 200.0, 225.0)
    assert len(snow_depth_colorbar_ticks(vmax)) == 6
    assert snow_depth_colorbar_ticklabels(vmax) == ("1", "50", "100", "150", "200", "225")


def test_increment_cmap_runs_from_negative_red_to_positive_blue() -> None:
    low = INCREMENT_CMAP(0.0)
    high = INCREMENT_CMAP(1.0)

    assert low[0] > low[2]
    assert high[2] > high[0]


def test_scf_observation_palette_uses_greys_for_example_map_style() -> None:
    low = FSC_OBS_CMAP(0.0)
    high = FSC_OBS_CMAP(1.0)

    assert sum(high[:3]) < sum(low[:3])


def test_static_field_palettes_follow_reference_style() -> None:
    dem_cmap = static_field_cmap(require_static_field_preset("dem"))
    svf_cmap = static_field_cmap(require_static_field_preset("svf"))
    srf_cmap = static_field_cmap(require_static_field_preset("srf"))
    srf_style = static_field_colorbar_style(require_static_field_preset("srf"))

    dem_low = dem_cmap(0.0)
    dem_high = dem_cmap(1.0)
    svf_low = svf_cmap(0.0)
    svf_high = svf_cmap(1.0)
    srf_low = srf_cmap(0.0)
    srf_high = srf_cmap(1.0)

    assert sum(dem_high[:3]) > sum(dem_low[:3])
    assert sum(svf_high[:3]) > sum(svf_low[:3])
    assert srf_low[0] > srf_low[2]
    assert srf_high[2] > srf_high[0]
    assert srf_style.ticks == (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8)


def test_wet_snow_reference_colors_follow_example_palette() -> None:
    assert WET_SNOW_COLORS[110] == "#000000"
    assert WET_SNOW_COLORS[125] == "#d8d8d8"
    assert WET_SNOW_COLORS[200] == "#ddb9ba"
    assert WET_SNOW_COLORS[210] == "#4b79c6"
    assert FSC_INVALID_COLOR == "#d8b3b7"


def test_map_axis_style_places_title_above_axes_and_can_hide_nonfirstcolumn_ylabels() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        _apply_map_axis_style(
            ax,
            (632000.0, 645000.0, 5180000.0, 5195000.0),
            title="Demo",
            show_grid=True,
            show_y_ticklabels=False,
        )
        _draw_map_grid_overlay(ax, show_grid=True)
        fig.canvas.draw()
        xlabels = [label for label in ax.get_xticklabels()]
        ylabels = [label for label in ax.get_yticklabels()]
        grid_lines = [line for line in ax.lines if line.get_gid() == "oa_da_grid"]
        title_bbox = ax._left_title.get_window_extent(fig.canvas.get_renderer())
        axes_bbox = ax.get_window_extent(fig.canvas.get_renderer())
        assert any(label.get_text() == "" for label in xlabels)
        assert all(label.get_text() == "" for label in ylabels)
        assert {label.get_rotation() for label in xlabels if label.get_text()} == {0.0}
        assert ax.get_title(loc="left") == "Demo"
        assert title_bbox.y0 >= axes_bbox.y1
        assert all(not tick.label2.get_visible() for tick in ax.xaxis.get_major_ticks())
        assert all(not tick.label2.get_visible() for tick in ax.yaxis.get_major_ticks())
        assert grid_lines
        assert all(line.get_zorder() == 120 for line in grid_lines)
    finally:
        plt.close(fig)


def test_panel_date_callout_stays_inside_axes_with_white_background() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        _apply_map_axis_style(ax, (632000.0, 645000.0, 5180000.0, 5195000.0), title="Demo", show_grid=False)
        render_module._draw_panel_date(ax, pd.Timestamp("2023-05-03"))
        fig.canvas.draw()
        axes_bbox = ax.get_window_extent(fig.canvas.get_renderer())
        text = ax.texts[-1]
        text_bbox = text.get_window_extent(fig.canvas.get_renderer())
        facecolor = text.get_bbox_patch().get_facecolor()
        anchor = text.get_position()
        assert text_bbox.x0 >= axes_bbox.x0
        assert text_bbox.x1 <= axes_bbox.x1
        assert text_bbox.y0 >= axes_bbox.y0
        assert text_bbox.y1 <= axes_bbox.y1
        assert facecolor[0:3] == pytest.approx((1.0, 1.0, 1.0))
        assert anchor[1] > 0.92
    finally:
        plt.close(fig)


def test_draw_scale_bar_adds_reference_style_annotations() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        extent = (632000.0, 645000.0, 5180000.0, 5195000.0)
        _draw_scale_bar(ax, extent)
        labels = {text.get_text() for text in ax.texts}
        assert {"0", "2.5", "5", "km"} <= labels
        bar = ax.lines[0]
        km_text = next(text for text in ax.texts if text.get_text() == "km")
        zero_text = next(text for text in ax.texts if text.get_text() == "0")
        assert any(effect.__class__.__name__ == "Stroke" for effect in bar.get_path_effects())
        assert any(effect.__class__.__name__ == "Stroke" for effect in km_text.get_path_effects())
        halo = next(effect for effect in km_text.get_path_effects() if isinstance(effect, pe.Stroke))
        zero_halo = next(effect for effect in zero_text.get_path_effects() if isinstance(effect, pe.Stroke))
        assert halo._gc["foreground"] == "white"
        assert halo._gc["linewidth"] == pytest.approx(2.0)
        assert zero_halo._gc["linewidth"] == pytest.approx(2.0)
        assert km_text.get_position()[1] < min(line.get_ydata()[0] for line in ax.lines)
    finally:
        plt.close(fig)


def test_draw_overview_label_specs_use_bbox_sensitive_halo_widths() -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        render_module._draw_overview_label_specs(
            ax,
            [
                panel_renderers_module.OverviewLabelSpec("Country", 0.4, 0.6, "center", "center", 6.2, True, 10),
                panel_renderers_module.OverviewLabelSpec("ROI", 0.6, 0.4, "left", "center", 6.2, False, 10),
            ],
        )
        country_halo = next(effect for effect in ax.texts[0].get_path_effects() if isinstance(effect, pe.Stroke))
        roi_halo = next(effect for effect in ax.texts[1].get_path_effects() if isinstance(effect, pe.Stroke))
        assert country_halo._gc["foreground"] == "white"
        assert country_halo._gc["linewidth"] == pytest.approx(2.4)
        assert roi_halo._gc["foreground"] == "white"
        assert roi_halo._gc["linewidth"] == pytest.approx(2.0)
    finally:
        plt.close(fig)


def test_overview_extent_follows_axes_aspect_ratio(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    try:
        extent = _overview_extent(ax, context, scale=1_000_000)
        extent_aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
        axes_aspect = (fig.get_size_inches()[1] * ax.get_position().height) / (fig.get_size_inches()[0] * ax.get_position().width)
        assert np.isclose(extent_aspect, axes_aspect, rtol=1e-6)
    finally:
        plt.close(fig)


def test_pack_horizontal_legend_rows_keep_single_line_labels() -> None:
    labels = [
        "rock",
        "ice",
        "water",
        "grassland",
        "shrubland",
        "farmland",
        "transitional",
        "deciduous 30-60",
        "deciduous 60-100",
        "mixed forest",
        "coniferous 30-60",
        "coniferous 60-100",
        "built-up",
    ]

    rows = _pack_horizontal_legend_rows(labels, panel_width_in=2.2)

    assert len(rows) > 1
    assert sum(len(row) for row in rows) == len(labels)
    assert len({len(row) for row in rows}) > 1
    assert all("\n" not in label for row in rows for label in row)


def test_horizontal_legend_row_layout_caps_extra_item_spacing() -> None:
    item_widths, start_x_in, item_gap_in = _horizontal_legend_row_layout(
        ["rock", "ice", "water"],
        panel_width_in=3.2,
    )

    assert len(item_widths) == 3
    assert start_x_in >= 0.0
    assert item_gap_in <= (
        render_module._HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN + render_module._HORIZONTAL_LEGEND_ITEM_GAP_IN
    )


def test_horizontal_legend_row_layout_left_aligns_single_item_rows() -> None:
    row_labels = ["coniferous 60-100"]

    item_widths, start_x_in, item_gap_in = _horizontal_legend_row_layout(
        row_labels,
        panel_width_in=2.2,
    )

    assert len(item_widths) == 1
    assert np.isclose(start_x_in, render_module._horizontal_legend_side_pad_in(2.2))
    assert item_gap_in == 0.0


def test_horizontal_legend_total_extra_is_tighter_for_multirow_classified_legends() -> None:
    rows = _pack_horizontal_legend_rows(
        [
            "rock",
            "ice",
            "water",
            "grassland",
            "shrubland",
            "farmland",
            "transitional",
            "deciduous 30-60",
            "deciduous 60-100",
            "mixed forest",
            "coniferous 30-60",
            "coniferous 60-100",
            "built-up",
        ],
        panel_width_in=2.2,
    )

    extra = _horizontal_legend_total_extra(rows, panel_width_in=2.2)

    assert len(rows) > 1
    assert extra < 0.55


def test_horizontal_colorbar_gap_is_tighter_than_legacy_spacing() -> None:
    assert render_module._HORIZONTAL_COLORBAR_GAP_AXES < 0.22


def test_station_entry_is_more_compact() -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    try:
        y_next = render_module._draw_station_entry(ax, y=0.8, label="Meteorological stations")
        scatter = ax.collections[-1]
        text = ax.texts[-1]
        assert scatter.get_sizes()[0] < 110
        assert text.get_fontsize() < 6.4
        assert y_next > 0.8 - 0.068
    finally:
        plt.close(fig)


def test_wet_snow_legend_handles_only_include_present_classes(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    scene = ObservationScene(
        date=pd.Timestamp("2023-05-03"),
        observation="wet_snow",
        array=np.array([[110.0, 110.0], [np.nan, 125.0]], dtype=float),
        transform=rasterio.Affine.identity(),
        roi_mask=np.array([[True, True], [False, False]], dtype=bool),
        invalid_mask=np.zeros((2, 2), dtype=bool),
        bounds=(0.0, 2.0, 0.0, 2.0),
        coverage_fraction=1.0,
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        artifacts = render_module._render_observation_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow", row=0, col=0, date="2023-05-03"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={("wet_snow", pd.Timestamp("2023-05-03")): scene},
            figure_horizontal_default=True,
        )
        assert [handle.get_label() for handle in artifacts["legend_handles"]] == [WET_SNOW_LABELS[110]]
    finally:
        plt.close(fig)


def test_uncertainty_panel_uses_fixed_percent_colorbar(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    date = pd.Timestamp("2023-05-03")
    scene = ObservationScene(
        date=date,
        observation="scf",
        array=np.array([[10.0, np.nan], [50.0, 90.0]], dtype=float),
        transform=rasterio.Affine.identity(),
        roi_mask=np.array([[True, True], [True, True]], dtype=bool),
        invalid_mask=np.array([[False, True], [False, False]], dtype=bool),
        bounds=(0.0, 2.0, 0.0, 2.0),
        coverage_fraction=0.75,
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        artifacts = render_module._render_uncertainty_panel(
            ax,
            panel=MapPanelSpec(kind="uncertainty", row=0, col=0, date="2023-05-03", observation="scf"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            uncertainty_cache={("scf", date): scene},
            figure_horizontal_default=True,
        )

        mappable = artifacts["mappable"]
        assert mappable.norm.vmin == 0.0
        assert mappable.norm.vmax == 100.0
        assert artifacts["colorbar_style"] == {
            "label": "uncertainty [%]",
            "ticks": (0, 20, 40, 60, 80, 100),
        }
    finally:
        plt.close(fig)


def test_wet_snow_line_model_panel_draws_wsl_contour(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    classified = np.full(context.roi_mask.shape, np.nan, dtype=float)
    classified[context.roi_mask] = 125.0
    classified[:2, :] = 110.0
    recorded_levels: list[float | None] = []
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        monkeypatch.setattr(panel_renderers_module, "_wet_snow_model_classified_array", lambda **kwargs: classified)
        monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_classified", lambda **kwargs: 2450.0)
        monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: 2550.0)

        def _record_contour(_ax, *, context, level, color, linestyle, linewidth=1.6, zorder=9):
            recorded_levels.append((level, color, linestyle))
            return True

        monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", _record_contour)

        artifacts = panel_renderers_module.render_wet_snow_line_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_line", row=0, col=0, source="open_loop", date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )

        assert recorded_levels == [(2450.0, panel_renderers_module._WSL_MODEL_COLOR, "-")]
        assert artifacts["wsl"] == 2450.0
        assert artifacts["obs_wsl"] == 2550.0
        legend_labels = [
            handle.get_label()
            for handle in panel_renderers_module._wet_snow_line_legend_handles(
                list(artifacts["legend_handles"]),
                include_model_wsl=True,
                include_obs_wsl=False,
            )
        ]
        assert legend_labels[-1:] == ["model WSLA"]
        assert "WSLA unavailable" not in {text.get_text() for text in ax.texts}
        callout = next(text for text in ax.texts if text.get_text() == "WSLA 2450 m")
        assert callout.get_color() == panel_renderers_module._WSL_MODEL_COLOR
    finally:
        plt.close(fig)


def test_posterior_weighted_wet_fraction_array_uses_da_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    step_dir = project_dir / "steps" / "step_01_20230101-20230102"
    member_a = step_dir / "ensembles" / "prior" / "member_001"
    member_b = step_dir / "ensembles" / "prior" / "member_002"

    monkeypatch.setattr(
        panel_renderers_module,
        "_posterior_da_weights",
        lambda *_args, **_kwargs: pd.Series([0.25, 0.75], index=["member_001", "member_002"], dtype=float),
    )
    monkeypatch.setattr(panel_renderers_module, "_step_dir_for_date", lambda *_args, **_kwargs: step_dir)
    monkeypatch.setattr(panel_renderers_module, "list_member_dirs", lambda *_args, **_kwargs: [member_a, member_b])
    monkeypatch.setattr(panel_renderers_module, "_wet_snow_mask_path", lambda results_dir, _date: Path(results_dir) / "dummy.tif")

    def fake_load_mask(path: Path) -> np.ndarray:
        mask = np.zeros(context.roi_mask.shape, dtype=np.uint8)
        if "member_001" in str(path):
            mask[:2, :2] = np.array([[1, 0], [1, 0]], dtype=np.uint8)
            return mask
        mask[:2, :2] = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        return mask

    monkeypatch.setattr(panel_renderers_module, "_load_wet_snow_mask", fake_load_mask)

    weighted = panel_renderers_module._posterior_weighted_wet_fraction_array(
        context=context,
        date=pd.Timestamp("2023-01-02"),
        derived_cache={},
    )

    assert weighted[0, 0] == pytest.approx(0.25)
    assert weighted[0, 1] == pytest.approx(0.75)
    assert weighted[1, 0] == pytest.approx(1.0)
    assert weighted[1, 1] == pytest.approx(0.0)


def test_wet_snow_elevation_fraction_open_loop_aggregates_raw_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    dem = np.asarray(context.dem, dtype=float)
    low_band = context.roi_mask & (dem < 2800.0)
    high_band = context.roi_mask & (dem >= 2800.0)
    classified = np.full(context.roi_mask.shape, float(panel_renderers_module._WET_SNOW_MODEL_CODES[1]), dtype=float)
    for row, col in [(1, 0), (2, 0), (3, 0), (0, 0), (0, 1)]:
        classified[row, col] = float(panel_renderers_module._WET_SNOW_MODEL_CODES[0])

    monkeypatch.setattr(panel_renderers_module, "_wet_snow_model_classified_array", lambda **_kwargs: classified)

    values = panel_renderers_module._wet_snow_elevation_fraction_array(
        context=context,
        source="open_loop",
        date=pd.Timestamp("2023-01-02"),
        weights_variable="wet_snow",
        obs_cache={},
        derived_cache={},
        observation_loader=load_observation_scene,
    )

    assert np.allclose(values[low_band], 0.5)
    assert np.allclose(values[high_band], 0.2)


def test_wet_snow_elevation_fraction_prior_probability_aggregates_raw_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    dem = np.asarray(context.dem, dtype=float)
    low_band = context.roi_mask & (dem < 2800.0)
    high_band = context.roi_mask & (dem >= 2800.0)
    wet_fraction = np.full(context.roi_mask.shape, np.nan, dtype=float)
    wet_fraction[low_band] = 0.5
    wet_fraction[high_band] = 0.5

    monkeypatch.setattr(panel_renderers_module, "_prior_wet_fraction_array", lambda **_kwargs: wet_fraction)

    values = panel_renderers_module._wet_snow_elevation_fraction_array(
        context=context,
        source="prior_probability",
        date=pd.Timestamp("2023-01-02"),
        weights_variable="wet_snow",
        obs_cache={},
        derived_cache={},
        observation_loader=load_observation_scene,
    )

    assert np.allclose(values[low_band], 0.5)
    assert np.allclose(values[high_band], 0.5)


def test_elevation_band_wsf_cmap_accents_fifty_percent() -> None:
    low = panel_renderers_module._ELEVATION_BAND_WSF_CMAP(0.25)
    lower_accent = panel_renderers_module._ELEVATION_BAND_WSF_CMAP(0.46)
    midpoint = panel_renderers_module._ELEVATION_BAND_WSF_CMAP(0.5)
    upper_accent = panel_renderers_module._ELEVATION_BAND_WSF_CMAP(0.54)
    high = panel_renderers_module._ELEVATION_BAND_WSF_CMAP(0.75)

    assert midpoint[0] > midpoint[1]
    assert midpoint[0] > midpoint[2]
    assert np.allclose(lower_accent[:3], midpoint[:3], atol=0.02)
    assert np.allclose(upper_accent[:3], midpoint[:3], atol=0.02)
    assert midpoint[3] == pytest.approx(1.0)
    assert not np.allclose(midpoint[:3], low[:3])
    assert not np.allclose(midpoint[:3], high[:3])


def test_wet_snow_elevation_fraction_panel_draws_source_local_wsl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    values = np.full(context.roi_mask.shape, np.nan, dtype=float)
    values[context.roi_mask] = 0.5
    recorded: list[tuple[float | None, str, str]] = []

    monkeypatch.setattr(panel_renderers_module, "_wet_snow_elevation_fraction_array", lambda **_kwargs: values)
    monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_fraction", lambda **_kwargs: 2450.0)

    def _record_contour(_ax, *, context, level, color, linestyle, **_kwargs):
        recorded.append((level, color, linestyle))
        return True

    monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", _record_contour)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        artifacts = panel_renderers_module.render_wet_snow_elevation_fraction_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_elevation_fraction", row=0, col=0, source="posterior", date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(show_colorbar=False),
            obs_cache={},
            figure_horizontal_default=True,
        )

        assert recorded == [(2450.0, panel_renderers_module._WSL_MODEL_COLOR, "-")]
        assert artifacts["wsl"] == 2450.0
        assert artifacts["wsl_drawn"] is True
        callout = next(text for text in ax.texts if text.get_text() == "WSLA 2450 m")
        assert callout.get_color() == panel_renderers_module._WSL_MODEL_COLOR
    finally:
        plt.close(fig)


def test_wet_snow_elevation_fraction_observation_reuses_observed_wsl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    values = np.full(context.roi_mask.shape, np.nan, dtype=float)
    values[context.roi_mask] = 0.2
    recorded: list[tuple[float | None, str, str]] = []

    monkeypatch.setattr(panel_renderers_module, "_wet_snow_elevation_fraction_array", lambda **_kwargs: values)
    monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: 3320.0)

    def _unexpected_fraction_wsl(**_kwargs):
        raise AssertionError("observation elevation-band WSF panels must reuse diagnostics WSLA")

    monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_fraction", _unexpected_fraction_wsl)

    def _record_contour(_ax, *, context, level, color, linestyle, **_kwargs):
        recorded.append((level, color, linestyle))
        return True

    monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", _record_contour)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        artifacts = panel_renderers_module.render_wet_snow_elevation_fraction_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_elevation_fraction", row=0, col=0, source=None, date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(show_colorbar=False),
            obs_cache={},
            figure_horizontal_default=True,
        )

        assert recorded == [(3320.0, panel_renderers_module._WSL_OBS_COLOR, "-")]
        assert artifacts["wsl"] == 3320.0
        assert artifacts["wsl_drawn"] is True
        callout = next(text for text in ax.texts if text.get_text() == "WSLA 3320 m")
        assert callout.get_color() == panel_renderers_module._WSL_OBS_COLOR
    finally:
        plt.close(fig)


def test_wet_snow_elevation_fraction_panel_annotates_unavailable_wsl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    values = np.full(context.roi_mask.shape, np.nan, dtype=float)
    values[context.roi_mask] = 0.2

    monkeypatch.setattr(panel_renderers_module, "_wet_snow_elevation_fraction_array", lambda **_kwargs: values)
    monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: None)

    def _unexpected_fraction_wsl(**_kwargs):
        raise AssertionError("missing observation diagnostics WSLA must not fall back to local recomputation")

    monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_fraction", _unexpected_fraction_wsl)
    monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", lambda *_args, **_kwargs: False)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        artifacts = panel_renderers_module.render_wet_snow_elevation_fraction_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_elevation_fraction", row=0, col=0, source=None, date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(show_colorbar=False),
            obs_cache={},
            figure_horizontal_default=True,
        )

        assert artifacts["wsl"] is None
        assert artifacts["wsl_drawn"] is False
        callout = next(text for text in ax.texts if text.get_text() == "WSLA unavailable")
        assert callout.get_color() == "black"
    finally:
        plt.close(fig)


def test_posterior_da_weights_reads_requested_wet_snow_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    step_dir = project_dir / "steps" / "step_01_20230101-20230102"
    assim_dir = step_dir / "assim"
    assim_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"member_id": ["member_001", "member_002"], "weight": [0.2, 0.8]},
    ).to_csv(assim_dir / "weights_wet_snow_20230102.csv", index=False)
    pd.DataFrame(
        {"member_id": ["member_001", "member_002"], "weight": [0.5, 0.5]},
    ).to_csv(assim_dir / "weights_wet_snow_line_20230102.csv", index=False)
    monkeypatch.setattr(panel_renderers_module, "_step_dir_for_date", lambda *_args, **_kwargs: step_dir)

    wsf_weights = panel_renderers_module._posterior_da_weights(context, pd.Timestamp("2023-01-02"), variable="wet_snow")
    wsla_weights = panel_renderers_module._posterior_da_weights(context, pd.Timestamp("2023-01-02"), variable="wet_snow_line")

    assert wsf_weights is not None
    assert wsla_weights is not None
    assert wsf_weights["member_001"] == pytest.approx(0.2)
    assert wsla_weights["member_001"] == pytest.approx(0.5)


def test_wet_snow_elevation_fraction_observation_excludes_non_contributors(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    dem = np.asarray(context.dem, dtype=float)
    low_band = context.roi_mask & (dem < 2800.0)
    arr = np.full(context.roi_mask.shape, 125.0, dtype=float)
    arr[1, 0] = 110.0
    arr[2, 0] = 110.0
    arr[3, 0] = 200.0
    arr[3, 1] = 210.0
    arr[3, 2] = np.nan
    scene = ObservationScene(
        date=pd.Timestamp("2023-01-02"),
        observation="wet_snow",
        array=arr,
        transform=context.spec.transform,
        bounds=(0.0, 400.0, 0.0, 400.0),
        coverage_fraction=1.0,
        roi_mask=context.roi_mask,
        invalid_mask=np.zeros(context.roi_mask.shape, dtype=bool),
    )

    values = panel_renderers_module._wet_snow_elevation_fraction_array(
        context=context,
        source=None,
        date=pd.Timestamp("2023-01-02"),
        weights_variable="wet_snow",
        obs_cache={},
        derived_cache={},
        observation_loader=lambda *_args, **_kwargs: scene,
    )

    assert values[1, 0] == pytest.approx(2.0 / 3.0)
    assert values[2, 1] == pytest.approx(2.0 / 3.0)
    assert np.isnan(values[3, 0])
    assert np.isnan(values[3, 1])
    assert np.isnan(values[3, 2])
    assert np.count_nonzero(np.isfinite(values[low_band])) == 3


def test_wet_snow_line_observation_panel_uses_obs_color_for_callout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    observation = ObservationScene(
        date=pd.Timestamp("2023-01-02"),
        observation="wet_snow",
        array=np.full(context.roi_mask.shape, 125.0, dtype=float),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        bounds=(0.0, float(context.roi_mask.shape[1]), 0.0, float(context.roi_mask.shape[0])),
        coverage_fraction=1.0,
        roi_mask=context.roi_mask,
        invalid_mask=np.zeros(context.roi_mask.shape, dtype=bool),
    )
    recorded_levels: list[tuple[float | None, str, str]] = []
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: 2550.0)

        def _record_contour(_ax, *, context, level, color, linestyle, linewidth=1.6, zorder=9):
            recorded_levels.append((level, color, linestyle))
            return True

        monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", _record_contour)

        panel_renderers_module.render_wet_snow_line_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_line", row=0, col=0, source=None, date="2023-01-02", legend="vertical"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
            observation_loader=lambda *_args, **_kwargs: observation,
        )

        assert recorded_levels == [(2550.0, panel_renderers_module._WSL_OBS_COLOR, "-")]
        callout = next(text for text in ax.texts if text.get_text() == "WSLA 2550 m")
        assert callout.get_color() == panel_renderers_module._WSL_OBS_COLOR
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()][-1] == "observation WSLA"
    finally:
        plt.close(fig)


def test_wet_snow_line_observation_panel_annotates_unavailable_without_wsl_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    observation = ObservationScene(
        date=pd.Timestamp("2023-01-02"),
        observation="wet_snow",
        array=np.full(context.roi_mask.shape, 125.0, dtype=float),
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        bounds=(0.0, float(context.roi_mask.shape[1]), 0.0, float(context.roi_mask.shape[0])),
        coverage_fraction=1.0,
        roi_mask=context.roi_mask,
        invalid_mask=np.zeros(context.roi_mask.shape, dtype=bool),
    )
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", lambda *_args, **_kwargs: False)

        panel_renderers_module.render_wet_snow_line_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_line", row=0, col=0, source=None, date="2023-01-02", legend="vertical"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
            observation_loader=lambda *_args, **_kwargs: observation,
        )

        assert "WSLA unavailable" in {text.get_text() for text in ax.texts}
        legend = ax.get_legend()
        assert legend is not None
        assert "observation WSLA" not in {text.get_text() for text in legend.get_texts()}
    finally:
        plt.close(fig)


def test_wet_snow_line_posterior_panel_uses_weighted_posterior_field(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    wet_fraction = np.full(context.roi_mask.shape, np.nan, dtype=float)
    wet_fraction[context.roi_mask] = 0.2
    recorded_levels: list[float | None] = []
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        monkeypatch.setattr(
            panel_renderers_module,
            "_posterior_weighted_wet_fraction_array",
            lambda **kwargs: wet_fraction,
        )
        monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_fraction", lambda **_kwargs: 2525.0)
        monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: 2625.0)

        def _unexpected_raster_wsl(**_kwargs):
            raise AssertionError("posterior panel should not derive its WSLA from the classified raster helper")

        monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_classified", _unexpected_raster_wsl)

        def _record_contour(_ax, *, context, level, color, linestyle, linewidth=1.6, zorder=9):
            recorded_levels.append((level, color, linestyle, zorder))
            return True

        monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", _record_contour)

        artifacts = panel_renderers_module.render_wet_snow_line_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_line", row=0, col=0, source="posterior", date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )

        assert recorded_levels == [(2525.0, panel_renderers_module._WSL_MODEL_COLOR, "-", 9.5)]
        assert artifacts["wsl"] == 2525.0
        assert artifacts["obs_wsl"] == 2625.0
        assert [handle.get_label() for handle in artifacts["posterior_overlay_handles"]] == ["posterior WSLA"]
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == ["posterior WSLA"]
        assert "WSLA unavailable" not in {text.get_text() for text in ax.texts}
        callout = next(text for text in ax.texts if text.get_text() == "WSLA 2530 m")
        assert callout.get_color() == panel_renderers_module._WSL_MODEL_COLOR
    finally:
        plt.close(fig)

def test_wet_snow_line_panel_annotates_unavailable_wsl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    wet_fraction = np.full(context.roi_mask.shape, np.nan, dtype=float)
    wet_fraction[context.roi_mask] = 0.2
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        monkeypatch.setattr(
            panel_renderers_module,
            "_posterior_weighted_wet_fraction_array",
            lambda **kwargs: wet_fraction,
        )
        monkeypatch.setattr(panel_renderers_module, "_observed_wet_snow_line_value", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(panel_renderers_module, "_prior_wet_fraction_array", lambda **kwargs: wet_fraction)
        monkeypatch.setattr(panel_renderers_module, "_wet_snow_line_from_fraction", lambda **kwargs: None)
        monkeypatch.setattr(panel_renderers_module, "_draw_wsl_contour", lambda *_args, **_kwargs: False)

        panel_renderers_module.render_wet_snow_line_panel(
            ax,
            panel=MapPanelSpec(kind="wet_snow_line", row=0, col=0, source="posterior", date="2023-01-02"),
            context=context,
            extent=buffered_extent(context),
            label=None,
            defaults=MapDefaults(),
            obs_cache={},
            figure_horizontal_default=True,
        )

        assert "WSLA unavailable" in {text.get_text() for text in ax.texts}
        callout = next(text for text in ax.texts if text.get_text() == "WSLA unavailable")
        assert callout.get_position() == pytest.approx((0.98, 0.955))
        assert callout.get_color() == "black"
    finally:
        plt.close(fig)


def test_panel_empty_below_units_only_counts_directly_empty_cells() -> None:
    recipe_empty = MapRecipe(
        name="setup_overview",
        title="setup_overview",
        layout=LayoutSpec(nrows=2, ncols=3),
        panels=(
            MapPanelSpec(kind="landcover", row=0, col=2),
            MapPanelSpec(kind="dem", row=1, col=0),
            MapPanelSpec(kind="srf", row=1, col=1),
        ),
    )
    recipe_occupied = MapRecipe(
        name="setup_overview",
        title="setup_overview",
        layout=LayoutSpec(nrows=2, ncols=3),
        panels=(
            MapPanelSpec(kind="landcover", row=0, col=2),
            MapPanelSpec(kind="dem", row=1, col=0),
            MapPanelSpec(kind="srf", row=1, col=1),
            MapPanelSpec(kind="roi", row=1, col=2),
        ),
    )

    assert render_module._panel_empty_below_units(recipe_empty, recipe_empty.panels[0]) == pytest.approx(1.0)
    assert render_module._panel_empty_below_units(recipe_occupied, recipe_occupied.panels[0]) == pytest.approx(0.0)


def test_row_bottom_extras_include_below_panel_legend_items(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    config_path = project_dir / "maps.yml"
    _write_yaml(
        config_path,
        """
        maps:
          setup_overview:
            title: setup_overview
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: roi
                title: region of interest
                below_items:
                  - kind: station_symbol
                    label: Meteorological stations
              - row: 0
                col: 1
                kind: dem
                title: digital elevation model
        """,
    )

    cfg = load_project_maps_config(config_path)
    extra = render_module._row_bottom_extras(
        cfg.maps[0],
        context=load_static_context(project_dir),
        panel_width_in=2.2,
        figure_horizontal_default=True,
    )

    assert extra[0] >= render_module._panel_below_items_extra(cfg.maps[0].panels[0].below_items)


def test_pack_horizontal_legend_rows_wrap_wet_snow_labels_in_narrow_panel() -> None:
    labels = [WET_SNOW_LABELS[code] for code in sorted(WET_SNOW_LABELS)]

    rows = _pack_horizontal_legend_rows(labels, panel_width_in=1.15)
    extra = _horizontal_legend_total_extra(rows, panel_width_in=1.15)
    legacy_extra = 0.10 + len(rows) * 0.155 + 1.0

    assert len(rows) > 1
    assert sum(len(row) for row in rows) == len(labels)
    assert extra < legacy_extra


def test_render_overview_panel_adds_country_labels(tmp_path: Path, monkeypatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    boundaries = gpd.GeoDataFrame(
        {"CNTR_ID": ["AT", "IT"]},
        geometry=[
            box(-90000.0, -90000.0, -5000.0, 90000.0),
            box(5000.0, -90000.0, 90000.0, 90000.0),
        ],
        crs="EPSG:25832",
    )
    regions = gpd.GeoDataFrame(
        {"CNTR_ID": ["AT", "IT"]},
        geometry=[
            box(-90000.0, -90000.0, -5000.0, 90000.0),
            box(5000.0, -90000.0, 90000.0, 90000.0),
        ],
        crs="EPSG:25832",
    )
    labels = gpd.GeoDataFrame(
        {"CNTR_ID": ["AT", "IT"], "NAME_ENGL": ["Austria", "Italy"]},
        geometry=[Point(-40000.0, 30000.0), Point(45000.0, -5000.0)],
        crs="EPSG:25832",
    )
    monkeypatch.setattr(render_module, "load_overview_boundaries", lambda **_kwargs: boundaries)
    monkeypatch.setattr(render_module, "load_overview_regions", lambda **_kwargs: regions)
    monkeypatch.setattr(render_module, "load_overview_labels", lambda **_kwargs: labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        render_module._render_overview_panel(
            ax,
            panel=MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000, roi_label="Demo ROI"),
            context=context,
            label="a",
            defaults=MapDefaults(),
        )
        texts = {text.get_text() for text in ax.texts}
        assert "Austria" in texts
        assert "Italy" in texts
        assert "Demo ROI" in texts
    finally:
        plt.close(fig)


def test_render_overview_panel_falls_back_to_roi_only_when_country_layers_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)

    def unavailable(**_kwargs):
        raise OSError("offline")

    monkeypatch.setattr(render_module, "load_overview_boundaries", unavailable)
    monkeypatch.setattr(render_module, "load_overview_regions", unavailable)
    monkeypatch.setattr(render_module, "load_overview_labels", unavailable)

    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        artifact = render_module._render_overview_panel(
            ax,
            panel=MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000, roi_label="Demo ROI"),
            context=context,
            label="a",
            defaults=MapDefaults(),
        )

        assert "extent" in artifact
        assert any(text.get_text() == "Demo ROI" for text in ax.texts)
    finally:
        plt.close(fig)


def test_overview_extent_with_label_fit_expands_for_border_hugging_labels(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        panel = MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000)
        base_extent = render_module._overview_extent(ax, context, scale=1_800_000)
        regions = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"]},
            geometry=[box(base_extent[1] - 500.0, base_extent[2] + 500.0, base_extent[1] - 10.0, base_extent[3] - 500.0)],
            crs="EPSG:25832",
        )
        labels = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"], "NAME_ENGL": ["A very long country label near the right border"]},
            geometry=[Point(base_extent[1] - 100.0, 0.5 * (base_extent[2] + base_extent[3]))],
            crs="EPSG:25832",
        )

        fitted_extent = render_module._overview_extent_with_label_fit(
            ax,
            panel=panel,
            context=context,
            labels=labels,
            visible_regions_getter=lambda _extent: regions,
        )

        assert fitted_extent[1] > base_extent[1]
    finally:
        plt.close(fig)


def test_overview_extent_with_label_fit_keeps_base_extent_when_labels_fit(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        panel = MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000)
        base_extent = render_module._overview_extent(ax, context, scale=1_800_000)
        center_x = 0.5 * (base_extent[0] + base_extent[1])
        center_y = 0.5 * (base_extent[2] + base_extent[3])
        regions = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"]},
            geometry=[box(center_x - 5000.0, center_y - 5000.0, center_x + 5000.0, center_y + 5000.0)],
            crs="EPSG:25832",
        )
        labels = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"], "NAME_ENGL": ["Demo"]},
            geometry=[Point(center_x, center_y)],
            crs="EPSG:25832",
        )

        fitted_extent = render_module._overview_extent_with_label_fit(
            ax,
            panel=panel,
            context=context,
            labels=labels,
            visible_regions_getter=lambda _extent: regions,
        )

        assert np.allclose(fitted_extent, base_extent)
    finally:
        plt.close(fig)


def test_overview_label_fit_margin_scales_out_beyond_automatic_fit(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        base_extent = render_module._overview_extent(ax, context, scale=1_800_000)
        regions = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"]},
            geometry=[box(base_extent[1] - 500.0, base_extent[2] + 500.0, base_extent[1] - 10.0, base_extent[3] - 500.0)],
            crs="EPSG:25832",
        )
        labels = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"], "NAME_ENGL": ["Long border label"]},
            geometry=[Point(base_extent[1] - 100.0, 0.5 * (base_extent[2] + base_extent[3]))],
            crs="EPSG:25832",
        )

        auto_extent = render_module._overview_extent_with_label_fit(
            ax,
            panel=MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000),
            context=context,
            labels=labels,
            visible_regions_getter=lambda _extent: regions,
        )
        expanded_extent = render_module._overview_extent_with_label_fit(
            ax,
            panel=MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000, label_fit_margin=0.03),
            context=context,
            labels=labels,
            visible_regions_getter=lambda _extent: regions,
        )

        auto_width = auto_extent[1] - auto_extent[0]
        expanded_width = expanded_extent[1] - expanded_extent[0]
        auto_height = auto_extent[3] - auto_extent[2]
        expanded_height = expanded_extent[3] - expanded_extent[2]
        assert expanded_width > auto_width
        assert expanded_height > auto_height
    finally:
        plt.close(fig)


def test_render_overview_panel_keeps_country_labels_inside_axes_bounds(tmp_path: Path, monkeypatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    fig, ax = plt.subplots(figsize=(4, 4))
    try:
        base_extent = render_module._overview_extent(ax, context, scale=1_800_000)
        boundaries = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"]},
            geometry=[box(base_extent[1] - 500.0, base_extent[2] + 500.0, base_extent[1] - 10.0, base_extent[3] - 500.0)],
            crs="EPSG:25832",
        )
        regions = boundaries.copy()
        label_text = "A very long country label near the right border"
        labels = gpd.GeoDataFrame(
            {"CNTR_ID": ["DEMO"], "NAME_ENGL": [label_text]},
            geometry=[Point(base_extent[1] - 100.0, 0.5 * (base_extent[2] + base_extent[3]))],
            crs="EPSG:25832",
        )
        monkeypatch.setattr(render_module, "load_overview_boundaries", lambda **_kwargs: boundaries)
        monkeypatch.setattr(render_module, "load_overview_regions", lambda **_kwargs: regions)
        monkeypatch.setattr(render_module, "load_overview_labels", lambda **_kwargs: labels)

        render_module._render_overview_panel(
            ax,
            panel=MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000),
            context=context,
            label="a",
            defaults=MapDefaults(),
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax.get_window_extent(renderer)
        country_text = next(text for text in ax.texts if text.get_text() == label_text)
        text_bbox = country_text.get_window_extent(renderer)

        assert text_bbox.x0 >= ax_bbox.x0 - 1.0
        assert text_bbox.x1 <= ax_bbox.x1 + 1.0
        assert text_bbox.y0 >= ax_bbox.y0 - 1.0
        assert text_bbox.y1 <= ax_bbox.y1 + 1.0
    finally:
        plt.close(fig)


def test_render_map_recipe_applies_row_view_extent_to_all_panels_in_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    recipe = MapRecipe(
        name="zoom_demo",
        title="zoom_demo",
        layout=LayoutSpec(nrows=2, ncols=2),
        panels=(
            MapPanelSpec(kind="roi", row=0, col=0),
            MapPanelSpec(kind="roi", row=0, col=1),
            MapPanelSpec(kind="roi", row=1, col=0),
            MapPanelSpec(kind="roi", row=1, col=1),
        ),
        row_views=(MapRowViewSpec(row=1, center=(200.0, 200.0), zoom=19.0, viewport_px=(1024, 512)),),
    )
    recorded_extents: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    def record_panel(*, ax, panel, extent, **_kwargs):
        recorded_extents[(int(panel.row), int(panel.col))] = extent
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        return {}

    monkeypatch.setattr(render_module, "_render_panel", record_panel)

    output_path = render_module.render_map_recipe(
        project_dir=project_dir,
        context=context,
        recipe=recipe,
        output_path=tmp_path / "row_view_demo.png",
    )

    expected_extents = render_module._row_extents_for_recipe(recipe, context)
    assert output_path.is_file()
    assert recorded_extents[(0, 0)] == expected_extents[0]
    assert recorded_extents[(0, 1)] == expected_extents[0]
    assert recorded_extents[(1, 0)] == expected_extents[1]
    assert recorded_extents[(1, 1)] == expected_extents[1]


def test_render_overview_panel_keeps_axes_box_aligned_with_sibling_panels(tmp_path: Path, monkeypatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    boundaries = gpd.GeoDataFrame(
        {"CNTR_ID": ["DEMO"]},
        geometry=[box(-90000.0, -90000.0, 90000.0, 90000.0)],
        crs="EPSG:25832",
    )
    regions = boundaries.copy()
    labels = gpd.GeoDataFrame(
        {"CNTR_ID": ["DEMO"], "NAME_ENGL": ["Long country label at the border"]},
        geometry=[Point(85000.0, 0.0)],
        crs="EPSG:25832",
    )
    monkeypatch.setattr(render_module, "load_overview_boundaries", lambda **_kwargs: boundaries)
    monkeypatch.setattr(render_module, "load_overview_regions", lambda **_kwargs: regions)
    monkeypatch.setattr(render_module, "load_overview_labels", lambda **_kwargs: labels)

    fig = plt.figure(figsize=(6, 3))
    gs = fig.add_gridspec(1, 2, wspace=0.0, hspace=0.0)
    ax_overview = fig.add_subplot(gs[0, 0])
    ax_roi = fig.add_subplot(gs[0, 1])
    try:
        render_module._render_overview_panel(
            ax_overview,
            panel=MapPanelSpec(kind="overview", row=0, col=0, scale=1_800_000),
            context=context,
            label="a",
            defaults=MapDefaults(),
        )
        render_module._render_roi_panel(
            ax_roi,
            panel=MapPanelSpec(kind="roi", row=0, col=1),
            context=context,
            extent=render_module.buffered_extent(context),
            label="b",
            defaults=MapDefaults(),
        )
        overview_box = ax_overview.get_position()
        roi_box = ax_roi.get_position()
        assert np.isclose(overview_box.width, roi_box.width, rtol=0.02)
        assert np.isclose(overview_box.height, roi_box.height, rtol=0.02)
    finally:
        plt.close(fig)


def test_draw_stations_overlay_supports_marker_only_and_explicit_labels(tmp_path: Path) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    context = load_static_context(project_dir)
    extent = buffered_extent(context)
    ordered = context.stations.sort_values("id").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(8, 3))
    try:
        _draw_stations_overlay(
            axes[0],
            context,
            extent,
            show_station_marker=True,
            show_stations_name=False,
            show_stations_elev=False,
        )
        _draw_stations_overlay(
            axes[1],
            context,
            extent,
            show_station_marker=True,
            show_stations_name=True,
            show_stations_elev=False,
        )
        _draw_stations_overlay(
            axes[2],
            context,
            extent,
            show_station_marker=False,
            show_stations_name=True,
            show_stations_elev=True,
        )

        assert len(axes[0].collections) == 1
        assert len(axes[0].texts) == 0
        assert len(axes[1].collections) == 1
        assert all("Station" in text.get_text() for text in axes[1].texts)
        expected_dx = 0.026 * (extent[1] - extent[0])
        expected_dy = 0.013 * (extent[3] - extent[2])
        for text, (_, row) in zip(axes[1].texts, ordered.iterrows()):
            xpos, ypos = text.get_position()
            assert np.isclose(xpos, float(row["x"]) + expected_dx)
            assert np.isclose(ypos, float(row["y"]) + expected_dy)
            halo = next(effect for effect in text.get_path_effects() if isinstance(effect, pe.Stroke))
            assert halo._gc["foreground"] == "white"
            assert halo._gc["linewidth"] == pytest.approx(2.0)
        assert len(axes[2].collections) == 0
        assert len(axes[2].texts) == 0
    finally:
        plt.close(fig)


def test_render_project_maps_writes_flat_recipe_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          setup_map:
            title: Demo setup
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: hillshade
                title: Hillshade
                show_station_marker: true
                show_stations_name: false
                show_stations_elev: false
              - row: 0
                col: 1
                kind: landcover
                title: Landcover
                legend: horizontal
          snowdepth_reference:
            title: Snow depth 2023/01/02
            output_name: snowdepth_2023-01-02
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 3
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
                title: Open loop
              - row: 0
                col: 1
                kind: snow_depth
                source: ensemble_mean
                title: Ensemble mean
              - row: 0
                col: 2
                kind: fsc
                title: Sentinel-2
        """,
    )

    outputs = render_project_maps(project_dir=project_dir)

    expected = {
        project_dir / "results" / "maps" / "setup_map.png",
        project_dir / "results" / "maps" / "snowdepth_2023-01-02.png",
    }

    assert set(outputs) == expected
    for path in outputs:
        assert path.is_file()
        assert path.stat().st_size > 0


def test_shipped_rofental_render_regression_against_tuned_baseline(tmp_path: Path) -> None:
    benchmark_inputs_dir = PROJECT_MAPS_FIXTURE_DIR / "inputs"
    da_output_fixture = benchmark_inputs_dir / "da_output_grids.nc"
    overview_assets = (
        benchmark_inputs_dir / "CNTR_BN_01M_2020_3857.geojson",
        benchmark_inputs_dir / "CNTR_RG_01M_2020_3857.geojson",
        benchmark_inputs_dir / "CNTR_LB_2020_3857.geojson",
    )
    if not da_output_fixture.is_file() or not all(path.is_file() for path in overview_assets):
        pytest.skip("project-map image regression inputs are not available")

    shipped_setup_dir = Path(__file__).resolve().parents[2] / "examples" / "rofental"
    work_setup_dir = tmp_path / "rofental"
    shutil.copytree(shipped_setup_dir, work_setup_dir)
    project_dir = work_setup_dir / "projects" / "project_2022_2023"
    results_dir = project_dir / "results"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    results_grids_dir = project_dir / "results" / "grids"
    results_grids_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(da_output_fixture, results_grids_dir / "da_output_grids.nc")
    env_dir = work_setup_dir / "env"
    env_dir.mkdir(parents=True, exist_ok=True)
    for path in overview_assets:
        shutil.copy2(path, env_dir / path.name)

    outputs = render_project_maps(project_dir=project_dir, max_workers=1)

    expected_names = sorted(path.name for path in (PROJECT_MAPS_FIXTURE_DIR / "expected").glob("*.png"))
    output_names = [path.name for path in outputs]
    assert set(expected_names).issubset(output_names)
    for output_path in outputs:
        expected_path = PROJECT_MAPS_FIXTURE_DIR / "expected" / output_path.name
        if not expected_path.is_file():
            continue
        diff = _mean_abs_png_diff(expected_path, output_path)
        if diff >= 0.01:
            failure_dir = tmp_path / "project_map_regression_failure"
            failure_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_path, failure_dir / output_path.name)
        assert diff < 0.01, f"{output_path.name} mean abs diff too high: {diff:.5f}"


def test_resolve_effective_max_workers_clamps_to_recipe_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module.os, "cpu_count", lambda: 8)

    assert runner_module._resolve_effective_max_workers(None, recipe_count=3) == 3
    assert runner_module._resolve_effective_max_workers(1, recipe_count=3) == 1
    assert runner_module._resolve_effective_max_workers(10, recipe_count=3) == 3


def test_render_project_maps_parallel_matches_sequential_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          setup_map:
            title: Demo setup
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: hillshade
          snowdepth_reference:
            title: Snow depth 2023/01/02
            output_name: snowdepth_2023-01-02
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
              - row: 0
                col: 1
                kind: fsc
        """,
    )

    sequential = render_project_maps(project_dir=project_dir, max_workers=1)
    parallel = render_project_maps(project_dir=project_dir, max_workers=2)

    expected = [
        project_dir / "results" / "maps" / "setup_map.png",
        project_dir / "results" / "maps" / "snowdepth_2023-01-02.png",
    ]
    assert sequential == expected
    assert parallel == expected


def test_collect_shared_model_vmax_uses_selected_nonincrement_snowdepth_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    recipes = (
        MapRecipe(
            name="a",
            title="A",
            layout=LayoutSpec(nrows=1, ncols=1),
            defaults=MapDefaults(date="2023-01-01"),
            panels=(MapPanelSpec(kind="snow_depth", row=0, col=0, source="open_loop"),),
        ),
        MapRecipe(
            name="b",
            title="B",
            layout=LayoutSpec(nrows=1, ncols=3),
            defaults=MapDefaults(date="2023-01-02"),
            panels=(
                MapPanelSpec(kind="snow_depth", row=0, col=0, source="ensemble_mean"),
                MapPanelSpec(kind="snow_depth", row=0, col=1, source="increment"),
                MapPanelSpec(kind="snow_depth", row=0, col=2, source="analysis_increment"),
            ),
        ),
    )

    calls: list[tuple[str, tuple[pd.Timestamp, ...]]] = []

    def fake_load_model_fields(project_dir_arg: Path, variable: str, dates: tuple[pd.Timestamp, ...]) -> list[ModelFields]:
        assert project_dir_arg == project_dir
        calls.append((variable, dates))
        return [
            ModelFields(
                date=dates[0],
                open_loop=np.array([[0.2, 0.4]], dtype=float),
                ens_mean=np.array([[0.6, 0.9]], dtype=float),
                increment=np.array([[0.1, -0.1]], dtype=float),
                analysis_mean=np.array([[1.3, 1.4]], dtype=float),
                analysis_increment=np.array([[0.7, 0.5]], dtype=float),
            ),
            ModelFields(
                date=dates[1],
                open_loop=np.array([[0.3, 0.5]], dtype=float),
                ens_mean=np.array([[0.8, 1.2]], dtype=float),
                increment=np.array([[0.2, -0.2]], dtype=float),
                analysis_mean=np.array([[1.4, 1.6]], dtype=float),
                analysis_increment=np.array([[0.6, 0.4]], dtype=float),
            ),
        ]

    monkeypatch.setattr(runner_module, "load_model_fields", fake_load_model_fields)

    shared_vmax = runner_module._collect_shared_model_vmax(project_dir, recipes)

    assert calls == [("snowdepth_daily", (pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")))]
    assert shared_vmax == {"snowdepth_daily": 1.75}


def test_render_project_maps_name_filter_limits_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          setup_map:
            title: Demo setup
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: hillshade
          snowdepth_reference:
            title: Snow depth 2023/01/02
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
        """,
    )

    outputs = render_project_maps(project_dir=project_dir, names={"setup_map"}, max_workers=8)

    assert outputs == [project_dir / "results" / "maps" / "setup_map.png"]


def test_render_project_maps_logs_batch_and_per_map_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          setup_map:
            title: Demo setup
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: hillshade
          wet_scene:
            title: Wet snow scene
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: wet_snow
        """,
    )

    info_messages: list[str] = []
    error_messages: list[str] = []

    class StubLogger:
        def info(self, message: str, *args) -> None:
            info_messages.append(message.format(*args))

        def error(self, message: str, *args) -> None:
            error_messages.append(message.format(*args))

    monkeypatch.setattr(runner_module, "logger", StubLogger())

    outputs = render_project_maps(project_dir=project_dir, max_workers=1)

    assert len(outputs) == 2
    assert any("Rendering 2 project map(s)" in msg for msg in info_messages)
    assert "Starting custom map setup_map" in info_messages
    assert "Starting custom map wet_scene" in info_messages
    assert any(msg.startswith("Finished map setup_map -> ") for msg in info_messages)
    assert any(msg.startswith("Finished map wet_scene -> ") for msg in info_messages)
    assert error_messages == []


def test_render_project_maps_reuses_model_and_observation_caches_across_recipes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          first:
            title: First
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
              - row: 0
                col: 1
                kind: fsc
          second:
            title: Second
            defaults:
              date: "2023-01-02"
            layout:
              nrows: 1
              ncols: 2
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: ensemble_mean
              - row: 0
                col: 1
                kind: fsc
        """,
    )

    model_calls = 0
    obs_calls = 0
    original_load_model_fields = render_module.load_model_fields
    original_load_observation_scene = render_module.load_observation_scene

    def counting_load_model_fields(project_dir_arg: Path, variable: str, dates: tuple[pd.Timestamp, ...]) -> list[ModelFields]:
        nonlocal model_calls
        model_calls += 1
        return original_load_model_fields(project_dir_arg, variable, dates)

    def counting_load_observation_scene(project_dir_arg: Path, context, *, observation: str, date: pd.Timestamp):
        nonlocal obs_calls
        obs_calls += 1
        return original_load_observation_scene(project_dir_arg, context, observation=observation, date=date)

    monkeypatch.setattr(render_module, "load_model_fields", counting_load_model_fields)
    monkeypatch.setattr(render_module, "load_observation_scene", counting_load_observation_scene)

    outputs = render_project_maps(project_dir=project_dir, max_workers=1)

    assert len(outputs) == 2
    assert model_calls == 1
    assert obs_calls == 1


def test_render_project_maps_parallel_failure_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          ok_map:
            title: OK
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: hillshade
          broken_map:
            title: Broken
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: snow_depth
                source: open_loop
        """,
    )

    with pytest.raises(runner_module.ProjectMapRenderError, match="broken_map"):
        render_project_maps(project_dir=project_dir, max_workers=2)


def test_render_project_maps_parallel_logs_recipe_attributed_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _setup_dir, project_dir = _build_project_fixture(tmp_path)
    monkeypatch.setattr(runner_module, "generated_da_map_recipes", lambda *_args, **_kwargs: ())
    _write_yaml(
        project_dir / "maps.yml",
        """
        maps:
          ok_map:
            title: OK
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: hillshade
          broken_map:
            title: Broken
            layout:
              nrows: 1
              ncols: 1
            panels:
              - row: 0
                col: 0
                kind: hillshade
        """,
    )

    info_messages: list[str] = []
    error_messages: list[str] = []

    class StubLogger:
        def info(self, message: str, *args) -> None:
            info_messages.append(message.format(*args))

        def error(self, message: str, *args) -> None:
            error_messages.append(message.format(*args))

    class FakeExecutor:
        def __init__(self, *, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, _fn, project_dir_arg, recipe, shared_model_vmax):
            del project_dir_arg, shared_model_vmax
            future: Future = Future()
            if recipe.name == "broken_map":
                future.set_exception(ValueError("boom"))
            else:
                future.set_result(
                    runner_module.RecipeRenderResult(
                        recipe_name=recipe.name,
                        output_path=Path("/tmp") / f"{recipe.name}.png",
                    )
                )
            return future

    monkeypatch.setattr(runner_module, "logger", StubLogger())
    monkeypatch.setattr(runner_module, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        runner_module,
        "as_completed",
        lambda futures: iter(sorted(futures, key=lambda future: 0 if futures[future].name == "broken_map" else 1)),
    )

    with pytest.raises(runner_module.ProjectMapRenderError, match="broken_map"):
        render_project_maps(project_dir=project_dir, max_workers=2)

    assert "Starting custom map ok_map" in info_messages
    assert "Starting custom map broken_map" in info_messages
    assert any(msg == "Failed custom map broken_map: boom" for msg in error_messages)


def test_project_pipeline_best_effort_map_render_logs_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project_demo"
    project_dir.mkdir(parents=True)

    warnings: list[str] = []

    class StubLogger:
        def info(self, *_args, **_kwargs) -> None:
            return None

        def warning(self, message: str, *args) -> None:
            warnings.append(message.format(*args))

    monkeypatch.setattr(plot_tasks_module, "logger", StubLogger())
    monkeypatch.setattr(plot_tasks_module, "project_maps_enabled", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        plot_tasks_module,
        "render_project_maps",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    plot_tasks_module.render_project_maps_best_effort(project_dir)

    assert warnings == [
        "Project maps failed: boom",
        f"Rerun project maps with: python -m openamundsen_da.methods.viz.maps.runner --project-dir {project_dir}",
    ]


def test_project_pipeline_best_effort_report_render_logs_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project_demo"
    output = project_dir / "results" / "reports" / "project_report.pdf"
    infos: list[str] = []

    class StubLogger:
        def info(self, message: str, *args) -> None:
            infos.append(message.format(*args))

        def warning(self, *_args, **_kwargs) -> None:
            raise AssertionError("report success should not log warnings")

    def fake_build_project_collection_pdf(**kwargs):
        assert kwargs == {"project_dir": project_dir}
        return output

    monkeypatch.setattr(plot_tasks_module, "logger", StubLogger())
    monkeypatch.setattr(plot_tasks_module, "build_project_collection_pdf", fake_build_project_collection_pdf)

    plot_tasks_module.render_project_report_best_effort(project_dir)

    assert infos == [f"Project report complete -> {output}"]


def test_project_pipeline_best_effort_report_render_logs_missing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project_demo"
    missing = project_dir / "results" / "plots" / "results" / "result_overview.png"
    warnings: list[str] = []

    class StubLogger:
        def info(self, *_args, **_kwargs) -> None:
            return None

        def warning(self, message: str, *args) -> None:
            warnings.append(message.format(*args))

    def fake_build_project_collection_pdf(**_kwargs):
        raise MissingProjectPdfArtifactsError(project_dir, [missing])

    monkeypatch.setattr(plot_tasks_module, "logger", StubLogger())
    monkeypatch.setattr(plot_tasks_module, "build_project_collection_pdf", fake_build_project_collection_pdf)

    plot_tasks_module.render_project_report_best_effort(project_dir)

    assert warnings[0].startswith("Project report skipped: Missing required project PDF artifact(s).")
    assert str(missing) in warnings[0]
    assert warnings[1] == (
        "Rerun project report with: "
        f"python -m openamundsen_da.methods.viz.reports --project-dir {project_dir}"
    )


def test_project_pipeline_best_effort_report_render_logs_unexpected_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_dir = tmp_path / "project_demo"
    warnings: list[str] = []

    class StubLogger:
        def info(self, *_args, **_kwargs) -> None:
            return None

        def warning(self, message: str, *args) -> None:
            warnings.append(message.format(*args))

    monkeypatch.setattr(plot_tasks_module, "logger", StubLogger())
    monkeypatch.setattr(
        plot_tasks_module,
        "build_project_collection_pdf",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    plot_tasks_module.render_project_report_best_effort(project_dir)

    assert warnings == [
        "Project report failed: boom",
        f"Rerun project report with: python -m openamundsen_da.methods.viz.reports --project-dir {project_dir}",
    ]


def test_project_maps_cli_passes_max_workers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project_demo"
    calls: list[dict[str, object]] = []

    def fake_render_project_maps(**kwargs):
        calls.append(kwargs)
        return [Path(kwargs["project_dir"]) / "results" / "maps" / "demo.png"]

    monkeypatch.setattr(runner_module, "render_project_maps", fake_render_project_maps)

    exit_code = runner_module.cli_main(
        [
            "--project-dir",
            str(project_dir),
            "--max-workers",
            "4",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "project_dir": project_dir,
            "config_path": None,
            "names": set(),
            "max_workers": 4,
        }
    ]


def test_project_maps_cli_logs_recipe_attributed_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_dir = tmp_path / "project_demo"
    error_messages: list[str] = []

    class StubLogger:
        def error(self, message: str, *args) -> None:
            error_messages.append(message.format(*args))

    def fake_render_project_maps(**_kwargs):
        raise runner_module.ProjectMapRenderError("broken_map", "boom")

    monkeypatch.setattr(runner_module, "logger", StubLogger())
    monkeypatch.setattr(runner_module, "render_project_maps", fake_render_project_maps)

    exit_code = runner_module.cli_main(
        [
            "--project-dir",
            str(project_dir),
        ]
    )

    assert exit_code == 1
    assert error_messages == [
        "Project maps rendering failed: custom map 'broken_map' failed: boom",
        f"Rerun project maps with: python -m openamundsen_da.methods.viz.maps.runner --project-dir {project_dir}",
    ]
