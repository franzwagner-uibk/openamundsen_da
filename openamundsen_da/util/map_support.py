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
    roi_mask: np.ndarray | None = None,
    source_fields: dict[str, list[np.ndarray]] | None = None,
) -> Path:
    """Validate geometry, domain and optional source-value equivalence."""
    path = project_map_support_path(project_dir)
    expected_dates = pd.DatetimeIndex(
        sorted({pd.Timestamp(date).normalize() for date in dates})
    )
    with netCDF4.Dataset(path) as dataset:
        if set(dataset.dimensions) != {"event", "y", "x"}:
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
        expected_shape = None if roi_mask is None else tuple(np.asarray(roi_mask, dtype=bool).shape)
        for field in sorted(fields):
            variable = dataset.variables[field]
            if tuple(variable.dimensions) != ("event", "y", "x"):
                raise ValueError(f"Invalid map-support dimensions for {field} in {path}")
            values = np.ma.filled(variable[:], np.nan).astype(float)
            if expected_shape is not None and tuple(values.shape[1:]) != expected_shape:
                raise ValueError(f"Map-support ROI shape differs for {field} in {path}")
            finite = np.isfinite(values)
            if not np.any(finite):
                raise ValueError(f"Map-support field contains no finite values: {field} in {path}")
            if np.any(values[finite] < 0.0) or np.any(values[finite] > 1.0):
                raise ValueError(f"Map-support values outside [0, 1] for {field} in {path}")
            if roi_mask is not None and np.any(finite[:, ~np.asarray(roi_mask, dtype=bool)]):
                raise ValueError(f"Map-support contains finite values outside the ROI for {field} in {path}")
            if source_fields is not None:
                expected_arrays = source_fields.get(field)
                if expected_arrays is None:
                    raise ValueError(f"Map-support source field is unavailable for {field}")
                expected = np.stack(
                    [np.asarray(array, dtype=np.float32) for array in expected_arrays]
                ).astype(float)
                if not np.allclose(values, expected, rtol=0.0, atol=1e-6, equal_nan=True):
                    raise ValueError(f"Map-support values differ from raw sources for {field} in {path}")
    return path


__all__ = ["load_map_support_field", "validate_map_support", "write_map_support"]
