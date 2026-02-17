from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon

from openamundsen_da.util.roi import read_single_roi


def test_read_single_roi_merges_multiple_polygons(monkeypatch, tmp_path: Path) -> None:
    roi_path = tmp_path / "roi.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        ],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("openamundsen_da.util.roi.gpd.read_file", lambda *_args, **_kwargs: gdf.copy())

    merged, region_id = read_single_roi(roi_path, required_field=None)

    assert len(merged) == 1
    assert region_id == ""
    assert float(merged.geometry.iloc[0].area) == 2.0


def test_read_single_roi_keeps_first_region_id_for_multi_feature(monkeypatch, tmp_path: Path) -> None:
    roi_path = tmp_path / "roi.gpkg"
    gdf = gpd.GeoDataFrame(
        {"region_id": ["main", "tile_b"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(3, 0), (4, 0), (4, 1), (3, 1)]),
        ],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("openamundsen_da.util.roi.gpd.read_file", lambda *_args, **_kwargs: gdf.copy())

    merged, region_id = read_single_roi(roi_path, required_field="region_id")

    assert len(merged) == 1
    assert region_id == "main"


def test_read_single_roi_falls_back_to_id_column(monkeypatch, tmp_path: Path) -> None:
    roi_path = tmp_path / "roi.gpkg"
    gdf = gpd.GeoDataFrame(
        {"id": ["tile_a", "tile_b"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:4326",
    )
    monkeypatch.setattr("openamundsen_da.util.roi.gpd.read_file", lambda *_args, **_kwargs: gdf.copy())

    merged, region_id = read_single_roi(roi_path, required_field="region_id")

    assert len(merged) == 1
    assert region_id == "tile_a"
