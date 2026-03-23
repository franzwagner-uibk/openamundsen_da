from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from openamundsen_da.methods.roi_mean import compute_member_roi_mean_daily


def _mock_roi(*_args, **_kwargs):
    gdf = gpd.GeoDataFrame(
        {"id": ["roi"]},
        geometry=[box(0.0, 0.0, 2.0, 2.0)],
        crs="EPSG:4326",
    )
    return gdf, ""


def test_compute_member_roi_mean_daily_reads_geotiff(monkeypatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tif_path = results_dir / "swe_daily_2023-01-01T0000.tif"
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0.0, 2.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    monkeypatch.setattr("openamundsen_da.util.grid_roi.read_single_roi", _mock_roi)

    df = compute_member_roi_mean_daily(
        results_dir=results_dir,
        aoi_path=tmp_path / "roi.gpkg",
        start=datetime(2023, 1, 1),
        end=datetime(2023, 1, 1),
        variable="swe",
    )

    assert list(df.columns) == ["time", "swe"]
    assert len(df) == 1
    assert float(df["swe"].iloc[0]) == 2.5


def test_compute_member_roi_mean_daily_reads_netcdf(monkeypatch, tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    nc_path = results_dir / "output_grids.nc"
    ds = xr.Dataset(
        {
            "snowdepth_daily": (
                ("time1", "y", "x"),
                np.array([[[0.2, 0.4], [0.6, 0.8]]], dtype=np.float32),
            ),
        },
        coords={
            "time1": [np.datetime64("2023-01-02T00:00:00")],
            "x": np.array([0.5, 1.5], dtype=np.float32),
            "y": np.array([1.5, 0.5], dtype=np.float32),
        },
    )
    ds["crs"] = xr.DataArray(0)
    ds["crs"].attrs["spatial_ref"] = "EPSG:4326"
    ds.to_netcdf(nc_path)

    monkeypatch.setattr("openamundsen_da.util.grid_roi.read_single_roi", _mock_roi)

    df = compute_member_roi_mean_daily(
        results_dir=results_dir,
        aoi_path=tmp_path / "roi.gpkg",
        start=datetime(2023, 1, 2),
        end=datetime(2023, 1, 2),
        variable="hs",
    )

    assert list(df.columns) == ["time", "snow_depth"]
    assert len(df) == 1
    assert float(df["snow_depth"].iloc[0]) == 0.5
