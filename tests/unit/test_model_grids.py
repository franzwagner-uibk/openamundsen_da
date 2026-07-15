from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from openamundsen_da.io.model_grids import resolve_model_grid_slice


def _write_geotiff(path: Path) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="float32",
        crs="EPSG:25832",
        transform=from_origin(0, 2, 1, 1),
    ) as dataset:
        dataset.write(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), 1)


def _write_netcdf(path: Path, *, dims: tuple[str, ...] = ("time", "y", "x")) -> None:
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    if dims == ("time", "roi_pixel"):
        data = data.reshape(1, 4)
    dataset = xr.Dataset(
        {"snowdepth_daily": (dims, data)},
        coords={
            "time": [np.datetime64("2023-04-26")],
            **({"x": [0.5, 1.5], "y": [1.5, 0.5]} if "x" in dims else {"roi_pixel": [0, 1, 2, 3]}),
        },
    )
    dataset.to_netcdf(path)


def test_explicit_adapters_resolve_equivalent_daily_grid(tmp_path: Path) -> None:
    geotiff_dir = tmp_path / "geotiff"
    netcdf_dir = tmp_path / "netcdf"
    geotiff_dir.mkdir()
    netcdf_dir.mkdir()
    geotiff = geotiff_dir / "snowdepth_daily_2023-04-26T0000.tif"
    netcdf = netcdf_dir / "output_grids.nc"
    _write_geotiff(geotiff)
    _write_netcdf(netcdf)

    tif_slice = resolve_model_grid_slice(
        results_dir=geotiff_dir,
        variable="hs",
        date=datetime(2023, 4, 26),
        grid_format="geotiff",
    )
    nc_slice = resolve_model_grid_slice(
        results_dir=netcdf_dir,
        variable="hs",
        date=datetime(2023, 4, 26),
        grid_format="netcdf",
    )

    assert tif_slice.kind == "geotiff"
    assert nc_slice.kind == "netcdf"
    assert nc_slice.nc_var == "snowdepth_daily"
    assert nc_slice.band == 1


def test_netcdf_adapter_rejects_roi_pixel_layout(tmp_path: Path) -> None:
    _write_netcdf(tmp_path / "output_grids.nc", dims=("time", "roi_pixel"))

    with pytest.raises(ValueError, match="grid dimensions x and y"):
        resolve_model_grid_slice(
            results_dir=tmp_path,
            variable="hs",
            date=datetime(2023, 4, 26),
            grid_format="netcdf",
        )


@pytest.mark.parametrize("grid_format", ["netcdf", "geotiff"])
def test_adapters_reject_mixed_model_grid_artifacts(tmp_path: Path, grid_format: str) -> None:
    _write_netcdf(tmp_path / "output_grids.nc")
    _write_geotiff(tmp_path / "snowdepth_daily_2023-04-26T0000.tif")

    with pytest.raises(ValueError, match="Mixed model-grid artifacts"):
        resolve_model_grid_slice(
            results_dir=tmp_path,
            variable="hs",
            date=datetime(2023, 4, 26),
            grid_format=grid_format,
        )
