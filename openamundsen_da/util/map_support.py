"""NetCDF I/O for retained DA-event spatial map support."""

from __future__ import annotations

import os
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from openamundsen_da.io.paths import project_map_support_path


def write_map_support(
    project_dir: str | Path,
    *,
    dates: list[pd.Timestamp],
    fields: dict[str, list[np.ndarray]],
    output_nc: str | Path | None = None,
) -> Path:
    """Atomically write compressed event fields with common grid geometry."""
    project_dir = Path(project_dir).resolve()
    if not dates or not fields:
        raise ValueError("Map support requires at least one date and field")
    normalized_dates = [pd.Timestamp(date).normalize() for date in dates]
    if len(set(normalized_dates)) != len(normalized_dates):
        raise ValueError("Map-support dates must be unique")
    shape: tuple[int, int] | None = None
    for name, arrays in fields.items():
        if len(arrays) != len(dates):
            raise ValueError(f"Map-support field {name!r} does not align with dates")
        for array in arrays:
            candidate = tuple(np.asarray(array).shape)
            if len(candidate) != 2:
                raise ValueError(f"Map-support field {name!r} must be two-dimensional")
            if shape is None:
                shape = candidate
            elif candidate != shape:
                raise ValueError(f"Map-support grid mismatch: {candidate} != {shape}")
    assert shape is not None
    output = Path(output_nc) if output_nc is not None else project_map_support_path(project_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        with netCDF4.Dataset(tmp, "w", format="NETCDF4") as dataset:
            dataset.setncattr("Conventions", "CF-1.10")
            dataset.setncattr("title", "openAMUNDSEN-DA retained DA-event map support")
            dataset.createDimension("event", len(dates))
            dataset.createDimension("y", shape[0])
            dataset.createDimension("x", shape[1])
            event = dataset.createVariable("event", "i8", ("event",))
            event.units = "days since 1970-01-01 00:00:00"
            event.calendar = "proleptic_gregorian"
            event[:] = netCDF4.date2num(
                [date.to_pydatetime() for date in normalized_dates],
                event.units,
                event.calendar,
            )
            for name, arrays in sorted(fields.items()):
                variable = dataset.createVariable(
                    name,
                    "f4",
                    ("event", "y", "x"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=(1, min(256, shape[0]), min(256, shape[1])),
                    fill_value=np.nan,
                )
                variable.units = "1"
                variable[:] = np.stack([np.asarray(array, dtype=np.float32) for array in arrays])
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return output


def load_map_support_field(
    project_dir: str | Path,
    *,
    date: pd.Timestamp,
    field: str,
) -> np.ndarray | None:
    """Load one retained event field, or return ``None`` when unavailable."""
    path = project_map_support_path(project_dir)
    if not path.is_file():
        return None
    target = pd.Timestamp(date).normalize()
    with netCDF4.Dataset(path) as dataset:
        if field not in dataset.variables:
            return None
        event = dataset.variables["event"]
        dates = pd.DatetimeIndex(
            netCDF4.num2date(
                event[:],
                units=event.units,
                calendar=getattr(event, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
        ).normalize()
        matches = np.flatnonzero(dates == target)
        if len(matches) != 1:
            return None
        return np.asarray(dataset.variables[field][int(matches[0])], dtype=float)


def validate_map_support(
    project_dir: str | Path,
    *,
    dates: list[pd.Timestamp],
    fields: set[str],
) -> Path:
    """Validate required event dates and fields before raw-grid cleanup."""
    path = project_map_support_path(project_dir)
    expected_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(date).normalize() for date in dates})
    )
    with netCDF4.Dataset(path) as dataset:
        if not {"event", "y", "x"}.issubset(dataset.dimensions):
            raise ValueError(f"Invalid map-support dimensions in {path}")
        event = dataset.variables["event"]
        retained_dates = pd.DatetimeIndex(
            netCDF4.num2date(
                event[:],
                units=event.units,
                calendar=getattr(event, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
        ).normalize()
        if not retained_dates.is_unique or not retained_dates.equals(expected_dates):
            raise ValueError(f"Map-support event dates do not match configured events: {path}")
        missing = sorted(field for field in fields if field not in dataset.variables)
        if missing:
            raise ValueError(f"Map-support fields missing in {path}: {', '.join(missing)}")
    return path


__all__ = ["load_map_support_field", "validate_map_support", "write_map_support"]
