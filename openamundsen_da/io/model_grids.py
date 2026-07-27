"""Deterministic model-grid adapters selected by setup configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from openamundsen_da.io.paths import GridSlice, find_setup_yaml
from openamundsen_da.util.yaml_utils import read_yaml_mapping

_VARIABLE_NAMES = {
    "hs": "snowdepth_daily",
    "swe": "swe_daily",
    "hs_instantaneous": "snowdepth_instantaneous",
    "swe_instantaneous": "swe_instantaneous",
}

_DEPTH_NAME = "snowdepth_instantaneous"
_LIQUID_WATER_NAME = "liquid_water_content_instantaneous"


@dataclass(frozen=True)
class ModelGridFrame:
    """One timestamped model-grid frame returned by a configured adapter."""

    stamp: str
    data: np.ndarray
    profile: dict


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
    """Resolve instantaneous DA model grids without cross-format discovery."""

    format: ModelGridFormat

    @abstractmethod
    def resolve(self, results_dir: Path, variable: str, date: datetime) -> GridSlice:
        """Return one validated grid slice for ``variable`` and ``date``."""

    @abstractmethod
    def depth_series(self, results_dir: Path) -> list[ModelGridFrame]:
        """Return ordered instantaneous snow-depth frames."""

    @abstractmethod
    def liquid_water_series(self, results_dir: Path) -> dict[str, list[np.ndarray]]:
        """Return instantaneous liquid-water layers grouped by timestamp."""

    @staticmethod
    def _variable_name(variable: str) -> str:
        try:
            return _VARIABLE_NAMES[variable]
        except KeyError as exc:
            raise ValueError(f"Unsupported model-grid variable: {variable!r}") from exc


class GeoTiffGridReader(ModelGridReader):
    """Resolve deterministic georeferenced openAMUNDSEN GeoTIFF outputs."""

    format = ModelGridFormat.GEOTIFF

    @staticmethod
    def _validate_artifacts(results_dir: Path) -> None:
        if (results_dir / "output_grids.nc").is_file():
            raise ValueError(f"Mixed model-grid artifacts in {results_dir}: GeoTIFF and output_grids.nc")

    def resolve(self, results_dir: Path, variable: str, date: datetime) -> GridSlice:
        results_dir = Path(results_dir)
        self._validate_artifacts(results_dir)
        name = self._variable_name(variable)
        matches = sorted(results_dir.glob(f"{name}_{date:%Y-%m-%dT%H%M}.tif"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one GeoTIFF for {variable} at {date:%Y-%m-%dT%H:%M} in {results_dir}; "
                f"found {len(matches)}"
            )
        path = matches[0]
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"GeoTIFF model grid has no CRS: {path}")
            if dataset.count != 1:
                raise ValueError(f"GeoTIFF instantaneous model grid must have exactly one band: {path}")
        return GridSlice(kind="geotiff", path=path, variable=variable, date=date, band=1)

    def depth_series(self, results_dir: Path) -> list[ModelGridFrame]:
        import re

        results_dir = Path(results_dir)
        self._validate_artifacts(results_dir)
        pattern = re.compile(rf"^{_DEPTH_NAME}_(?P<stamp>[^.]+)\.tif$")
        entries: list[ModelGridFrame] = []
        for path in sorted(results_dir.glob(f"{_DEPTH_NAME}_*.tif")):
            match = pattern.match(path.name)
            if match is None:
                continue
            with rasterio.open(path) as dataset:
                entries.append(
                    ModelGridFrame(
                        stamp=match.group("stamp"),
                        data=dataset.read(1).astype("float32"),
                        profile=dict(dataset.profile),
                    )
                )
        return entries

    def liquid_water_series(self, results_dir: Path) -> dict[str, list[np.ndarray]]:
        import re

        results_dir = Path(results_dir)
        self._validate_artifacts(results_dir)
        pattern = re.compile(
            rf"^{_LIQUID_WATER_NAME}_(?P<layer>\d+)_(?P<stamp>\d{{4}}-\d{{2}}-\d{{2}}T\d{{4}})\.tif$"
        )
        grouped: dict[str, list[np.ndarray]] = {}
        for path in sorted(results_dir.glob("liquid_water_content_*.tif")):
            match = pattern.match(path.name)
            if match is None:
                continue
            with rasterio.open(path) as dataset:
                data = dataset.read(1).astype("float32")
                if dataset.nodata is not None:
                    data[data == dataset.nodata] = np.nan
                grouped.setdefault(match.group("stamp"), []).append(data)
        return grouped


class NetCdfGridReader(ModelGridReader):
    """Resolve a 2-D grid slice from canonical ``output_grids.nc``."""

    format = ModelGridFormat.NETCDF

    @staticmethod
    def _path(results_dir: Path) -> Path:
        geotiffs = sorted(results_dir.glob("*.tif"))
        path = results_dir / "output_grids.nc"
        if geotiffs:
            raise ValueError(f"Mixed model-grid artifacts in {results_dir}: output_grids.nc and GeoTIFF")
        if not path.is_file():
            raise FileNotFoundError(f"Canonical NetCDF model grid not found: {path}")
        return path

    def resolve(self, results_dir: Path, variable: str, date: datetime) -> GridSlice:
        results_dir = Path(results_dir)
        name = self._variable_name(variable)
        path = self._path(results_dir)

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
            target = pd.Timestamp(date)
            matches = [index for index, timestamp in enumerate(times) if pd.Timestamp(timestamp) == target]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one NetCDF grid for {variable} at {date:%Y-%m-%dT%H:%M} in {path}; "
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

    def depth_series(self, results_dir: Path) -> list[ModelGridFrame]:
        path = self._path(Path(results_dir))
        with xr.open_dataset(path) as dataset:
            if _DEPTH_NAME not in dataset:
                raise FileNotFoundError(f"Variable {_DEPTH_NAME!r} not found in {path}")
            data = dataset[_DEPTH_NAME]
            if "x" not in data.dims or "y" not in data.dims:
                raise ValueError(f"NetCDF {_DEPTH_NAME} must use grid dimensions x and y; got {data.dims}")
            time_dims = [dim for dim in data.dims if dim.startswith("time")]
            if len(time_dims) != 1:
                raise ValueError(f"NetCDF {_DEPTH_NAME} must have exactly one time dimension")
            times = pd.to_datetime(dataset[time_dims[0]].values)
        url = f"NETCDF:{path}:{_DEPTH_NAME}"
        with rasterio.open(url) as source:
            profile = dict(source.profile)
            return [
                ModelGridFrame(
                    stamp=timestamp.strftime("%Y-%m-%dT%H%M"),
                    data=source.read(index + 1).astype("float32"),
                    profile=profile,
                )
                for index, timestamp in enumerate(times)
            ]

    def liquid_water_series(self, results_dir: Path) -> dict[str, list[np.ndarray]]:
        path = self._path(Path(results_dir))
        with xr.open_dataset(path) as dataset:
            if _LIQUID_WATER_NAME not in dataset:
                raise FileNotFoundError(f"Variable {_LIQUID_WATER_NAME!r} not found in {path}")
            data = dataset[_LIQUID_WATER_NAME]
            if "snow_layer" not in data.dims or "x" not in data.dims or "y" not in data.dims:
                raise ValueError(
                    f"NetCDF {_LIQUID_WATER_NAME} must use snow_layer, x and y grid dimensions; "
                    f"got {data.dims}"
                )
            time_dims = [dim for dim in data.dims if dim.startswith("time")]
            if len(time_dims) != 1:
                raise ValueError(f"NetCDF {_LIQUID_WATER_NAME} must have exactly one time dimension")
            time_dim = time_dims[0]
            times = pd.to_datetime(dataset[time_dim].values)
            return {
                timestamp.strftime("%Y-%m-%dT%H%M"): [
                    layer.astype("float32")
                    for layer in data.isel({time_dim: index}).values
                ]
                for index, timestamp in enumerate(times)
            }


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
    "ModelGridFrame",
    "ModelGridFormat",
    "ModelGridReader",
    "NetCdfGridReader",
    "configured_model_grid_format",
    "model_grid_reader",
    "resolve_model_grid_slice",
]
