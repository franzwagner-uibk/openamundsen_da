"""Deterministic model-grid adapters selected by setup configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
import rasterio
import xarray as xr

from openamundsen_da.io.paths import GridSlice, find_setup_yaml
from openamundsen_da.util.yaml_utils import read_yaml_mapping

_VARIABLE_NAMES = {
    "hs": "snowdepth_daily",
    "swe": "swe_daily",
}


class ModelGridFormat(str, Enum):
    """Supported openAMUNDSEN gridded-output formats for DA operators."""

    GEOTIFF = "geotiff"
    NETCDF = "netcdf"


def configured_model_grid_format(setup_dir: str | Path) -> ModelGridFormat:
    """Read the required model-grid format from canonical setup YAML."""
    setup_yaml = find_setup_yaml(Path(setup_dir))
    config = read_yaml_mapping(setup_yaml, context="Setup YAML root")
    output_data = config.get("output_data")
    grids = output_data.get("grids") if isinstance(output_data, dict) else None
    raw = grids.get("format") if isinstance(grids, dict) else None
    try:
        return ModelGridFormat(str(raw).strip().lower())
    except ValueError as exc:
        supported = ", ".join(item.value for item in ModelGridFormat)
        raise ValueError(
            f"setup.output_data.grids.format must be explicitly configured as one of: {supported}; got {raw!r}"
        ) from exc


class ModelGridReader(ABC):
    """Resolve one logical daily model grid without cross-format discovery."""

    format: ModelGridFormat

    @abstractmethod
    def resolve(self, results_dir: Path, variable: str, date: datetime) -> GridSlice:
        """Return one validated grid slice for ``variable`` and ``date``."""

    @staticmethod
    def _variable_name(variable: str) -> str:
        try:
            return _VARIABLE_NAMES[variable]
        except KeyError as exc:
            raise ValueError(f"Unsupported model-grid variable: {variable!r}") from exc


class GeoTiffGridReader(ModelGridReader):
    """Resolve deterministic georeferenced openAMUNDSEN GeoTIFF outputs."""

    format = ModelGridFormat.GEOTIFF

    def resolve(self, results_dir: Path, variable: str, date: datetime) -> GridSlice:
        results_dir = Path(results_dir)
        name = self._variable_name(variable)
        matches = sorted(results_dir.glob(f"{name}_{date:%Y-%m-%d}T*.tif"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one GeoTIFF for {variable} on {date:%Y-%m-%d} in {results_dir}; "
                f"found {len(matches)}"
            )
        if (results_dir / "output_grids.nc").is_file():
            raise ValueError(f"Mixed model-grid artifacts in {results_dir}: GeoTIFF and output_grids.nc")
        path = matches[0]
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"GeoTIFF model grid has no CRS: {path}")
            if dataset.count != 1:
                raise ValueError(f"GeoTIFF daily model grid must have exactly one band: {path}")
        return GridSlice(kind="geotiff", path=path, variable=variable, date=date, band=1)


class NetCdfGridReader(ModelGridReader):
    """Resolve a 2-D grid slice from canonical ``output_grids.nc``."""

    format = ModelGridFormat.NETCDF

    def resolve(self, results_dir: Path, variable: str, date: datetime) -> GridSlice:
        results_dir = Path(results_dir)
        name = self._variable_name(variable)
        geotiffs = sorted(results_dir.glob(f"{name}_{date:%Y-%m-%d}T*.tif"))
        path = results_dir / "output_grids.nc"
        if geotiffs:
            raise ValueError(f"Mixed model-grid artifacts in {results_dir}: output_grids.nc and GeoTIFF")
        if not path.is_file():
            raise FileNotFoundError(f"Canonical NetCDF model grid not found: {path}")

        with xr.open_dataset(path) as dataset:
            if name not in dataset:
                raise FileNotFoundError(f"Variable {name!r} not found in {path}")
            data = dataset[name]
            if "x" not in data.dims or "y" not in data.dims:
                raise ValueError(
                    f"NetCDF model variable {name!r} must use grid dimensions x and y; got {data.dims}"
                )
            time_dims = [dim for dim in data.dims if dim.startswith("time")]
            if len(time_dims) != 1:
                raise ValueError(f"NetCDF model variable {name!r} must have exactly one time dimension")
            time_dim = time_dims[0]
            times = pd.to_datetime(dataset[time_dim].values)
            matches = [
                index
                for index, timestamp in enumerate(times)
                if pd.to_datetime(timestamp).date() == date.date()
            ]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one NetCDF grid for {variable} on {date:%Y-%m-%d} in {path}; "
                    f"found {len(matches)}"
                )
        return GridSlice(
            kind="netcdf",
            path=path,
            variable=variable,
            date=date,
            band=matches[0] + 1,
            nc_var=name,
        )


_READERS: dict[ModelGridFormat, ModelGridReader] = {
    ModelGridFormat.GEOTIFF: GeoTiffGridReader(),
    ModelGridFormat.NETCDF: NetCdfGridReader(),
}


def model_grid_reader(grid_format: str | ModelGridFormat) -> ModelGridReader:
    """Return the adapter for an explicit supported format."""
    try:
        resolved = grid_format if isinstance(grid_format, ModelGridFormat) else ModelGridFormat(str(grid_format).lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported model-grid format: {grid_format!r}") from exc
    return _READERS[resolved]


def resolve_model_grid_slice(
    *,
    results_dir: str | Path,
    variable: str,
    date: datetime,
    grid_format: str | ModelGridFormat,
) -> GridSlice:
    """Resolve a daily slice through the explicitly selected adapter."""
    return model_grid_reader(grid_format).resolve(Path(results_dir), variable, date)


__all__ = [
    "GeoTiffGridReader",
    "ModelGridFormat",
    "ModelGridReader",
    "NetCdfGridReader",
    "configured_model_grid_format",
    "model_grid_reader",
    "resolve_model_grid_slice",
]
