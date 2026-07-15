from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import windows
from rasterio.transform import from_origin
from shapely.geometry import box

from openamundsen_da.subdomain.prepare import (
    _union_windows,
    _window_for_mask,
    prepare_subdomains,
)


def _write_setup_yaml(setup_dir: Path) -> None:
    (setup_dir / "demo.yml").write_text(
        "\n".join(
            [
                'domain: "demo"',
                "resolution: 100",
                'crs: "EPSG:25832"',
                "input_data:",
                "  grids:",
                "    dir: grids",
                "  meteo:",
                "    dir: meteo",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_project_yaml(project_dir: Path) -> None:
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "project_2022_2023.yml").write_text(
        "\n".join(
            [
                "run_mode: subdomain",
                "start_date: '2022-10-01'",
                "end_date: '2022-10-02'",
                "data_assimilation: {}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_dem(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ones((4, 4), dtype=np.float32)
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


def test_prepare_subdomains_fails_when_buffered_regions_overlap_on_grid(tmp_path: Path) -> None:
    setup_dir = tmp_path / "setup"
    project_dir = setup_dir / "projects" / "project_2022_2023"
    env_dir = setup_dir / "env"
    meteo_dir = setup_dir / "meteo"
    env_dir.mkdir(parents=True, exist_ok=True)
    meteo_dir.mkdir(parents=True, exist_ok=True)
    _write_setup_yaml(setup_dir)
    _write_project_yaml(project_dir)
    _write_dem(setup_dir / "grids" / "dem_demo_100.asc")

    gpd.GeoDataFrame(
        {"id": pd.Series(["roi"], dtype="object")},
        geometry=[box(0, 0, 400, 400)],
        crs="EPSG:25832",
    ).to_file(env_dir / "roi.gpkg", driver="GPKG")

    gpd.GeoDataFrame(
        {"id": pd.Series(["a", "b"], dtype="object")},
        geometry=[box(0, 0, 200, 400), box(200, 0, 400, 400)],
        crs="EPSG:25832",
    ).to_file(env_dir / "subdomains.gpkg", driver="GPKG")

    try:
        prepare_subdomains(
            setup_dir=setup_dir,
            project_dir=project_dir,
            regions_path=env_dir / "subdomains.gpkg",
            roi_buffer_m=80.0,
            overwrite=True,
        )
    except ValueError as exc:
        assert "Rasterized sub-domain overlap detected" in str(exc)
    else:
        raise AssertionError("Expected ValueError for buffered overlap pixels")


def test_window_union_covers_owner_pixels_outside_geometry_window() -> None:
    owner_mask = np.zeros((8, 8), dtype=bool)
    owner_mask[2:7, 2:4] = True
    owner_mask[7, 3] = True

    geom_window = windows.Window(col_off=2, row_off=2, width=2, height=5)
    owner_window = _window_for_mask(owner_mask)
    merged_window = _union_windows(geom_window, owner_window, owner_mask.shape)

    covered = np.zeros_like(owner_mask, dtype=bool)
    r0 = int(merged_window.row_off)
    c0 = int(merged_window.col_off)
    r1 = r0 + int(merged_window.height)
    c1 = c0 + int(merged_window.width)
    covered[r0:r1, c0:c1] = True

    assert int(np.count_nonzero(owner_mask & (~covered))) == 0
