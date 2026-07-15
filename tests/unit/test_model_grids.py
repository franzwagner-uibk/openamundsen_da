from __future__ import annotations

from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import xarray as xr
from rasterio.transform import from_origin
from shapely.geometry import box

from openamundsen_da.io.model_grids import model_grid_reader, resolve_model_grid_slice
from openamundsen_da.util.grid_roi import read_grid_slice_roi_masked_array


def _write_geotiff(
    path: Path,
    data: np.ndarray | None = None,
    *,
    nodata: float | None = None,
) -> None:
    if data is None:
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
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
        nodata=nodata,
    ) as dataset:
        dataset.write(data.astype(np.float32), 1)


def _write_netcdf(path: Path, *, dims: tuple[str, ...] = ("time", "y", "x")) -> None:
    data = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    if dims == ("time", "roi_pixel"):
        data = data.reshape(1, 4)
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {
        "snowdepth_daily": (dims, data),
    }
    if dims == ("time", "y", "x"):
        variables["liquid_water_content"] = (
            ("time", "snow_layer", "y", "x"),
            np.array(
                [[[[0.1, 0.2], [0.3, 0.4]], [[1.0, 2.0], [3.0, 4.0]]]],
                dtype=np.float32,
            ),
        )
    dataset = xr.Dataset(
        variables,
        coords={
            "time": [np.datetime64("2023-04-26")],
            **({"x": [0.5, 1.5], "y": [1.5, 0.5]} if "x" in dims else {"roi_pixel": [0, 1, 2, 3]}),
        },
    )
    dataset["crs"] = xr.DataArray(0)
    dataset["crs"].attrs["spatial_ref"] = "EPSG:25832"
    if "x" in dims:
        dataset["x"].attrs.update(
            {"standard_name": "projection_x_coordinate", "units": "m"}
        )
        dataset["y"].attrs.update(
            {"standard_name": "projection_y_coordinate", "units": "m"}
        )
        dataset["snowdepth_daily"].attrs["grid_mapping"] = "crs"
        dataset["liquid_water_content"].attrs["grid_mapping"] = "crs"
    dataset.to_netcdf(path)


def test_explicit_adapters_return_equivalent_depth_and_liquid_water_series(
    tmp_path: Path,
) -> None:
    geotiff_dir = tmp_path / "geotiff"
    netcdf_dir = tmp_path / "netcdf"
    geotiff_dir.mkdir()
    netcdf_dir.mkdir()
    _write_geotiff(geotiff_dir / "snowdepth_daily_2023-04-26T0000.tif")
    _write_geotiff(
        geotiff_dir / "liquid_water_content_0_2023-04-26T0000_2023-04-27T0000.tif",
        np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
    )
    _write_geotiff(
        geotiff_dir / "liquid_water_content_1_2023-04-26T0000_2023-04-27T0000.tif",
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    _write_netcdf(netcdf_dir / "output_grids.nc")

    geotiff_reader = model_grid_reader("geotiff")
    netcdf_reader = model_grid_reader("netcdf")
    geotiff_depth = geotiff_reader.depth_series(geotiff_dir)
    netcdf_depth = netcdf_reader.depth_series(netcdf_dir)
    geotiff_lwc = geotiff_reader.liquid_water_series(geotiff_dir)
    netcdf_lwc = netcdf_reader.liquid_water_series(netcdf_dir)

    assert [frame.stamp for frame in geotiff_depth] == ["2023-04-26T0000"]
    assert [frame.stamp for frame in netcdf_depth] == ["2023-04-26T0000"]
    np.testing.assert_array_equal(geotiff_depth[0].data, netcdf_depth[0].data)
    assert geotiff_lwc.keys() == netcdf_lwc.keys() == {"2023-04-26T0000"}
    for geotiff_layer, netcdf_layer in zip(
        geotiff_lwc["2023-04-26T0000"],
        netcdf_lwc["2023-04-26T0000"],
        strict=True,
    ):
        np.testing.assert_allclose(geotiff_layer, netcdf_layer)


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

    roi = tmp_path / "roi.gpkg"
    gpd.GeoDataFrame(
        geometry=[box(0.0, 0.0, 2.0, 2.0)],
        crs="EPSG:25832",
    ).to_file(roi, driver="GPKG")
    tif_values = read_grid_slice_roi_masked_array(tif_slice, roi)
    nc_values = read_grid_slice_roi_masked_array(nc_slice, roi)
    np.testing.assert_array_equal(tif_values.mask, nc_values.mask)
    np.testing.assert_array_equal(tif_values.data, nc_values.data)


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
