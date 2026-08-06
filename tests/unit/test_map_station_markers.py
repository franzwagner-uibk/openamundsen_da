from __future__ import annotations

from types import SimpleNamespace

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from shapely.geometry import LineString, Point, box

from openamundsen_da.methods.viz.maps import panel_renderers as panel_renderers_module
from openamundsen_da.methods.viz.maps.annotations import (
    draw_station_categories,
    draw_station_categories_below,
)
from openamundsen_da.methods.viz.maps.config import load_project_maps_config
from openamundsen_da.methods.viz.maps.panel_renderers import (
    OverviewLabelSpec,
    _overview_subdomain_candidate_score,
    draw_overview_label_leaders,
    draw_stations_overlay,
    overview_label_data_box,
    overview_label_leader_segment,
    overview_subdomain_label_specs,
)
from openamundsen_da.methods.viz.maps.station_markers import (
    FORCING_STATION_COLOR,
    HOLDOUT_STATION_COLOR,
    HOLDOUT_STATION_LINEWIDTH,
    HOLDOUT_STATION_MARKER,
    LEFT_HALF_TRIANGLE,
    RIGHT_HALF_TRIANGLE,
    SNOW_STATION_COLOR,
    classify_station_markers,
)

_HOLDOUT_PATH = MarkerStyle(HOLDOUT_STATION_MARKER).get_path().transformed(
    MarkerStyle(HOLDOUT_STATION_MARKER).get_transform()
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
        holdout_collection = next(
            collection
            for collection in ax.collections
            if tuple(collection.get_facecolors()[0]) == to_rgba(HOLDOUT_STATION_COLOR)
        )
        np.testing.assert_allclose(
            holdout_collection.get_paths()[0].vertices,
            _HOLDOUT_PATH.vertices,
        )
        assert holdout_collection.get_linewidths()[0] == pytest.approx(
            HOLDOUT_STATION_LINEWIDTH
        )
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


def test_subdomain_label_scoring_and_dense_fallback_are_deterministic(
    monkeypatch,
) -> None:
    base = OverviewLabelSpec("A", 0.0, 0.0, "center", "center", 6.0, True, 10)
    candidate = OverviewLabelSpec("A", 2.0, 2.0, "center", "center", 6.0, True, 10)
    existing = [LineString([(0.0, -2.0), (0.0, 2.0)])]
    clean_score = _overview_subdomain_candidate_score(
        candidate=candidate,
        candidate_box=box(2.0, 2.0, 3.0, 3.0),
        leader=LineString([(1.0, 1.0), (2.0, 1.0)]),
        occupied_boxes=[],
        placed_leaders=existing,
        base=base,
    )
    crossing_score = _overview_subdomain_candidate_score(
        candidate=candidate,
        candidate_box=box(2.0, 2.0, 3.0, 3.0),
        leader=LineString([(-1.0, 0.0), (1.0, 0.0)]),
        occupied_boxes=[],
        placed_leaders=existing,
        base=base,
    )
    assert clean_score[4] == 0
    assert crossing_score[4] == 1
    assert clean_score < crossing_score

    warnings = []
    monkeypatch.setattr(
        panel_renderers_module.logger,
        "warning",
        lambda message, label: warnings.append((message, label)),
    )
    subdomains = gpd.GeoDataFrame(
        {"subdomain_id": ["dense"]},
        geometry=[box(45.0, 45.0, 55.0, 55.0)],
        crs="EPSG:25832",
    )
    occupied = OverviewLabelSpec(
        "occupied",
        50.0,
        50.0,
        "center",
        "center",
        1000.0,
        True,
        10,
    )
    fig, ax = plt.subplots(figsize=(2, 2))
    try:
        first = overview_subdomain_label_specs(
            ax,
            SimpleNamespace(subdomain_gdf=subdomains),
            extent=(0.0, 100.0, 0.0, 100.0),
            occupied_specs=[occupied],
        )
        second = overview_subdomain_label_specs(
            ax,
            SimpleNamespace(subdomain_gdf=subdomains),
            extent=(0.0, 100.0, 0.0, 100.0),
            occupied_specs=[occupied],
        )
        assert first == second
        assert warnings
        assert warnings[0][1] == "dense"
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
    fig, axes = plt.subplots(1, 3, figsize=(7, 2))
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
        assert [(spec.anchor_x, spec.anchor_y) for spec in label_specs] == [
            (505_000.0, 5_205_000.0),
            (515_000.0, 5_205_000.0),
        ]
        displaced = [
            spec
            for spec in label_specs
            if not np.isclose(spec.x, spec.anchor_x, rtol=0.0, atol=1e-9)
            or not np.isclose(spec.y, spec.anchor_y, rtol=0.0, atol=1e-9)
        ]
        assert len(displaced) == 1
        leader = overview_label_leader_segment(axes[0], displaced[0], extent=extent)
        assert leader is not None
        assert Point(leader.coords[0]).distance(
            overview_label_data_box(axes[0], displaced[0], extent=extent).boundary
        ) == pytest.approx(0.0)
        assert leader.coords[-1] == pytest.approx(
            (displaced[0].anchor_x, displaced[0].anchor_y)
        )
        draw_overview_label_leaders(axes[0], label_specs, extent=extent)
        assert len(axes[0].lines) == 1
        assert axes[0].lines[0].get_color() == "#4d4d4d"
        assert axes[0].lines[0].get_linewidth() == pytest.approx(0.6)
        assert axes[0].lines[0].get_marker() in {None, "None"}
        halo = next(
            effect
            for effect in axes[0].lines[0].get_path_effects()
            if isinstance(effect, pe.Stroke)
        )
        assert halo._gc["foreground"] == "white"
        assert halo._gc["linewidth"] == pytest.approx(1.8)
        for spec in label_specs:
            axes[0].text(spec.x, spec.y, spec.text)
        draw_station_categories(axes[1], y=0.86)
        draw_station_categories_below(axes[2], y=0.80)
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
        assert [text.get_text() for text in axes[2].texts] == [
            "Forcing station",
            "Snow obs. station",
            "Forcing + snow station",
            "Holdout snow station",
        ]
        for legend_ax in axes[1:]:
            holdout_collection = legend_ax.collections[-1]
            assert tuple(holdout_collection.get_facecolors()[0]) == to_rgba(
                HOLDOUT_STATION_COLOR
            )
            np.testing.assert_allclose(
                holdout_collection.get_paths()[0].vertices,
                _HOLDOUT_PATH.vertices,
            )
        np.testing.assert_allclose(
            sorted({text.get_position()[1] for text in axes[2].texts}),
            [0.25, 0.80],
        )
    finally:
        plt.close(fig)
