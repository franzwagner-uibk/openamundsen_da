from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from shapely.geometry import box

from openamundsen_da.methods.viz.maps.annotations import draw_station_categories
from openamundsen_da.methods.viz.maps.config import load_project_maps_config
from openamundsen_da.methods.viz.maps.panel_renderers import (
    draw_stations_overlay,
    overview_label_data_box,
    overview_subdomain_label_specs,
)
from openamundsen_da.methods.viz.maps.station_markers import (
    FORCING_STATION_COLOR,
    HOLDOUT_STATION_COLOR,
    LEFT_HALF_TRIANGLE,
    RIGHT_HALF_TRIANGLE,
    SNOW_STATION_COLOR,
    classify_station_markers,
)


def _station_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    forcing = pd.DataFrame(
        [
            {"id": "forcing_a", "name": "Forcing A", "x": 0.0, "y": 0.0, "alt": 1000.0},
            {
                "id": "forcing_b",
                "name": "Forcing B",
                "x": 100.0,
                "y": 100.0,
                "alt": 2000.0,
            },
        ]
    )
    snow = pd.DataFrame(
        [
            {
                "station_id": "snow_a",
                "name": "Snow A",
                "x": 0.0,
                "y": 0.0,
                "alt": 1001.0,
                "use_for_da": True,
                "use_for_benchmark": False,
            },
            {
                "station_id": "snow_b",
                "name": "Snow B",
                "x": 1.0,
                "y": 0.0,
                "alt": 1002.0,
                "use_for_da": False,
                "use_for_benchmark": True,
            },
            {
                "station_id": "snow_c",
                "name": "Snow C",
                "x": -1.0,
                "y": 0.0,
                "alt": 1003.0,
                "use_for_da": True,
                "use_for_benchmark": False,
            },
            {
                "station_id": "snow_d",
                "name": "Snow D",
                "x": 50.0,
                "y": 50.0,
                "alt": 1500.0,
                "use_for_da": True,
                "use_for_benchmark": False,
            },
        ]
    )
    return forcing, snow


def test_classified_station_markers_retain_collision_records_and_roles() -> None:
    forcing, snow = _station_tables()

    markers = classify_station_markers(forcing, snow, tolerance_m=10.0)

    assert [marker.kind for marker in markers] == [
        "both",
        "holdout",
        "both",
        "forcing",
        "snow",
    ]
    collision = markers[:3]
    assert [marker.station_id for marker in collision] == ["snow_a", "snow_b", "snow_c"]
    np.testing.assert_allclose(
        [(marker.offset_x_points, marker.offset_y_points) for marker in collision],
        [(0.0, 5.0), (4.330127, -2.5), (-4.330127, -2.5)],
        atol=1e-6,
    )
    assert all(marker.forcing_id == "forcing_a" for marker in collision)


def test_classified_station_rendering_uses_split_and_role_colors() -> None:
    forcing, snow = _station_tables()
    context = SimpleNamespace(stations=forcing, snow_stations=snow)
    fig, ax = plt.subplots(figsize=(3, 3))
    try:
        draw_stations_overlay(
            ax,
            context,
            (-20.0, 120.0, -20.0, 120.0),
            show_station_marker=True,
            show_stations_name=False,
            show_stations_elev=False,
            station_marker_mode="sources_and_roles",
            station_match_tolerance_m=10.0,
        )
        colors = [
            tuple(collection.get_facecolors()[0]) for collection in ax.collections
        ]
        assert colors.count(to_rgba(FORCING_STATION_COLOR)) == 3
        assert colors.count(to_rgba(SNOW_STATION_COLOR)) == 3
        assert colors.count(to_rgba(HOLDOUT_STATION_COLOR)) == 1
        paths = [collection.get_paths()[0].vertices for collection in ax.collections]
        assert any(np.array_equal(path, LEFT_HALF_TRIANGLE.vertices) for path in paths)
        assert any(np.array_equal(path, RIGHT_HALF_TRIANGLE.vertices) for path in paths)
    finally:
        plt.close(fig)

    missing_context = SimpleNamespace(stations=forcing, snow_stations=None)
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        with pytest.raises(ValueError, match="stations_da_metadata.csv"):
            draw_stations_overlay(
                ax,
                missing_context,
                (-20.0, 120.0, -20.0, 120.0),
                show_station_marker=True,
                show_stations_name=False,
                show_stations_elev=False,
                station_marker_mode="sources_and_roles",
            )
    finally:
        plt.close(fig)


def test_station_map_config_legend_and_subdomain_labels(tmp_path) -> None:
    maps_path = tmp_path / "maps.yml"
    maps_path.write_text(
        """maps:
  setup:
    title: Setup
    layout: {nrows: 1, ncols: 2}
    panels:
      - {row: 0, col: 0, kind: overview, scale: 2500000, show_subdomain_labels: true}
      - row: 0
        col: 1
        kind: roi
        show_station_marker: true
        station_marker_mode: sources_and_roles
        station_match_tolerance_m: 10
        legend_items:
          - {kind: station_categories, placement: inside, anchor: top_left}
""",
        encoding="utf-8",
    )
    config = load_project_maps_config(maps_path)
    overview, roi = config.maps[0].panels
    assert overview.show_subdomain_labels is True
    assert roi.station_marker_mode == "sources_and_roles"
    assert roi.station_match_tolerance_m == 10.0
    assert roi.inside_legend_items[0].kind == "station_categories"

    maps_path.write_text(
        maps_path.read_text(encoding="utf-8").replace(
            "station_match_tolerance_m: 10", "station_match_tolerance_m: 0"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must be > 0"):
        load_project_maps_config(maps_path)

    subdomains = gpd.GeoDataFrame(
        {"subdomain_id": ["AT-07-13", "AT-07-14-01"]},
        geometry=[
            box(500_000, 5_200_000, 510_000, 5_210_000),
            box(510_000, 5_200_000, 520_000, 5_210_000),
        ],
        crs="EPSG:25832",
    )
    fig, axes = plt.subplots(1, 2, figsize=(5, 2))
    try:
        extent = (300_000.0, 700_000.0, 5_000_000.0, 5_400_000.0)
        label_specs = overview_subdomain_label_specs(
            axes[0],
            SimpleNamespace(subdomain_gdf=subdomains),
            extent=extent,
        )
        label_boxes = [
            overview_label_data_box(axes[0], spec, extent=extent)
            for spec in label_specs
        ]
        assert not label_boxes[0].intersects(label_boxes[1])
        for spec in label_specs:
            axes[0].text(spec.x, spec.y, spec.text)
        draw_station_categories(axes[1], y=0.86)
        assert [text.get_text() for text in axes[0].texts] == [
            "AT-07-13",
            "AT-07-14-01",
        ]
        assert [text.get_text() for text in axes[1].texts] == [
            "Forcing station",
            "Snow observation station",
            "Forcing + snow station",
            "Holdout snow station",
        ]
    finally:
        plt.close(fig)
