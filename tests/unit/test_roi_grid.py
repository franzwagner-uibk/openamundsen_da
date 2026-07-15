from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from openamundsen_da.util.roi_grid import (
    ensure_setup_roi_grid,
    ensure_setup_roi_vector,
    load_setup_roi_mask,
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


def _build_setup(tmp_path: Path) -> Path:
    setup_dir = tmp_path / "setup"
    (setup_dir / "projects").mkdir(parents=True, exist_ok=True)
    (setup_dir / "meteo").mkdir(parents=True, exist_ok=True)
    (setup_dir / "env").mkdir(parents=True, exist_ok=True)
    _write_setup_yaml(setup_dir)
    _write_dem(setup_dir / "grids" / "dem_demo_100.asc")
    return setup_dir


def test_ensure_setup_roi_grid_generates_from_vector(tmp_path: Path) -> None:
    setup_dir = _build_setup(tmp_path)
    gdf = gpd.GeoDataFrame(
        {"id": pd.Series(["a", "b"], dtype="object")},
        geometry=[box(0, 0, 200, 400), box(200, 0, 400, 400)],
        crs="EPSG:25832",
    )
    regions = setup_dir / "env" / "subdomains.gpkg"
    gdf.to_file(regions, driver="GPKG")

    roi_grid = ensure_setup_roi_grid(setup_dir)

    assert roi_grid.is_file()
    mask, _, _ = load_setup_roi_mask(setup_dir)
    assert int(mask.sum()) == 16


def test_ensure_setup_roi_vector_generates_from_roi_grid(tmp_path: Path) -> None:
    setup_dir = _build_setup(tmp_path)
    roi_grid = setup_dir / "grids" / "roi_demo_100.asc"
    mask = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    )
    with rasterio.open(
        roi_grid,
        "w",
        driver="AAIGrid",
        dtype="uint8",
        nodata=0,
        width=mask.shape[1],
        height=mask.shape[0],
        count=1,
        crs="EPSG:25832",
        transform=from_origin(0.0, 400.0, 100.0, 100.0),
    ) as ds:
        ds.write(mask.astype("uint8"), 1)

    roi_vector = ensure_setup_roi_vector(setup_dir)

    assert roi_vector.is_file()
    gdf = gpd.read_file(roi_vector)
    assert len(gdf) == 1


def test_ensure_setup_roi_grid_fails_without_grid_or_vector(tmp_path: Path) -> None:
    setup_dir = _build_setup(tmp_path)

    try:
        ensure_setup_roi_grid(setup_dir)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError when ROI grid and ROI vectors are missing")
