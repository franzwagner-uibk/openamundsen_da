"""Compressed, streaming all-member point time-series output."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np
import pandas as pd

from openamundsen_da.io.paths import (
    list_member_dirs,
    list_steps_sorted,
    project_ensemble_points_path,
)


_TIME_COLUMNS = ("date", "time", "datetime")
_KNOWN_UNITS = {
    "snow_depth": "m",
    "swe": "kg m-2",
    "scf": "1",
    "wet_snow_fraction": "1",
    "wet_snow_line": "m",
    "temp": "K",
    "precip": "kg m-2",
}


def _read_point_csv(path: Path) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    frame = pd.read_csv(path)
    time_col = next((column for column in _TIME_COLUMNS if column in frame.columns), None)
    if time_col is None:
        raise ValueError(f"Point output has no date/time column: {path}")
    times = pd.DatetimeIndex(pd.to_datetime(frame.pop(time_col), errors="raise"))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return times, numeric


def _reference_results_dir(step_dir: Path) -> Path:
    open_loop = step_dir / "ensembles" / "prior" / "open_loop" / "results"
    if list(open_loop.glob("point_*.csv")):
        return open_loop
    members = list_member_dirs(step_dir / "ensembles", "prior")
    for member in members:
        results = member / "results"
        if list(results.glob("point_*.csv")):
            return results
    raise FileNotFoundError(f"No point outputs found in {step_dir}")


def _schema(steps: Iterable[Path]) -> tuple[list[str], list[str], pd.DatetimeIndex]:
    point_names: set[str] = set()
    variable_names: set[str] = set()
    times: set[pd.Timestamp] = set()
    for step in steps:
        reference = _reference_results_dir(step)
        for path in sorted(reference.glob("point_*.csv")):
            point_names.add(path.stem.removeprefix("point_"))
            index, frame = _read_point_csv(path)
            times.update(pd.Timestamp(value) for value in index)
            variable_names.update(str(column) for column in frame.columns)
    if not point_names or not variable_names or not times:
        raise ValueError("Point output schema is empty")
    return sorted(point_names), sorted(variable_names), pd.DatetimeIndex(sorted(times))


def _member_names(steps: Iterable[Path]) -> list[str]:
    expected: list[str] | None = None
    for step in steps:
        names = [member.name for member in list_member_dirs(step / "ensembles", "prior")]
        if expected is None:
            expected = names
        elif names != expected:
            raise ValueError(f"Point member identities differ in {step}: {names} != {expected}")
    if not expected:
        raise ValueError("No prior members found for point output")
    return ["open_loop", *expected]


def _result_roots(step: Path) -> list[tuple[str, Path]]:
    roots = [("open_loop", step / "ensembles" / "prior" / "open_loop" / "results")]
    roots.extend(
        (member.name, member / "results")
        for member in list_member_dirs(step / "ensembles", "prior")
    )
    return roots


def _validate_source_completeness(steps: Iterable[Path]) -> None:
    """Require every member to contain the reference point files and schema."""
    for step in steps:
        reference = _reference_results_dir(step)
        expected_paths = sorted(reference.glob("point_*.csv"))
        expected_names = [path.name for path in expected_paths]
        expected = {
            path.name: _read_point_csv(path)
            for path in expected_paths
        }
        for member, results in _result_roots(step):
            if not results.is_dir():
                raise FileNotFoundError(f"Missing point results directory for {member}: {results}")
            actual_names = [path.name for path in sorted(results.glob("point_*.csv"))]
            if actual_names != expected_names:
                raise ValueError(
                    f"Point files differ for {member} in {step}: "
                    f"{actual_names} != {expected_names}"
                )
            for name in expected_names:
                expected_index, expected_frame = expected[name]
                index, frame = _read_point_csv(results / name)
                if not index.is_unique:
                    raise ValueError(f"Duplicate point timestamps in {results / name}")
                if not index.equals(expected_index) or list(frame.columns) != list(expected_frame.columns):
                    raise ValueError(f"Point time/variable schema differs in {results / name}")


def validate_project_ensemble_points(
    project_dir: str | Path,
    *,
    output_nc: str | Path | None = None,
) -> Path:
    """Validate the retained all-member point NetCDF contract."""
    project_dir = Path(project_dir).resolve()
    steps = list_steps_sorted(project_dir)
    point_names, variable_names, times = _schema(steps)
    member_names = _member_names(steps)
    path = Path(output_nc) if output_nc is not None else project_ensemble_points_path(project_dir)
    with netCDF4.Dataset(path) as dataset:
        if set(dataset.dimensions) != {"time", "member", "point"}:
            raise ValueError(f"Invalid compact point dimensions in {path}")
        retained_members = [str(value) for value in dataset.variables["member"][:]]
        retained_points = [str(value) for value in dataset.variables["point"][:]]
        if retained_members != member_names or retained_points != point_names:
            raise ValueError(f"Compact point identities do not match member outputs: {path}")
        time_var = dataset.variables["time"]
        retained_times = pd.DatetimeIndex(
            netCDF4.num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
        )
        if not retained_times.equals(times):
            raise ValueError(f"Compact point time coverage does not match member outputs: {path}")
        missing = [name for name in variable_names if name not in dataset.variables]
        if missing:
            raise ValueError(f"Compact point variables missing in {path}: {', '.join(missing)}")
    return path


def write_project_ensemble_points(
    project_dir: str | Path,
    *,
    output_nc: str | Path | None = None,
) -> Path:
    """Stream all project point CSVs into one compressed NetCDF4 file."""
    project_dir = Path(project_dir).resolve()
    steps = list_steps_sorted(project_dir)
    if not steps:
        raise FileNotFoundError(f"No steps found under {project_dir}")
    point_names, variable_names, times = _schema(steps)
    member_names = _member_names(steps)
    _validate_source_completeness(steps)
    output = Path(output_nc) if output_nc is not None else project_ensemble_points_path(project_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    time_lookup = {pd.Timestamp(value): idx for idx, value in enumerate(times)}
    member_lookup = {name: idx for idx, name in enumerate(member_names)}
    point_lookup = {name: idx for idx, name in enumerate(point_names)}

    try:
        with netCDF4.Dataset(tmp, "w", format="NETCDF4") as dataset:
            dataset.setncattr("Conventions", "CF-1.10")
            dataset.setncattr("title", "openAMUNDSEN-DA all-member point time series")
            dataset.setncattr("source", "native openAMUNDSEN point CSV output")
            dataset.setncattr(
                "time_semantics",
                "timezone-aware source timestamps converted to UTC; naive source timestamps preserved",
            )
            dataset.createDimension("time", len(times))
            dataset.createDimension("member", len(member_names))
            dataset.createDimension("point", len(point_names))
            time_var = dataset.createVariable("time", "f8", ("time",))
            time_var.units = "seconds since 1970-01-01 00:00:00"
            time_var.calendar = "proleptic_gregorian"
            time_var[:] = netCDF4.date2num(times.to_pydatetime(), time_var.units, time_var.calendar)
            dataset.createVariable("member", str, ("member",))[:] = np.asarray(member_names, dtype=object)
            dataset.createVariable("point", str, ("point",))[:] = np.asarray(point_names, dtype=object)
            variables = {}
            chunks = (min(256, len(times)), 1, min(64, len(point_names)))
            for name in variable_names:
                var = dataset.createVariable(
                    name,
                    "f8",
                    ("time", "member", "point"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=chunks,
                    fill_value=np.nan,
                )
                var.units = _KNOWN_UNITS.get(name, "1")
                var.long_name = name.replace("_", " ")
                variables[name] = var

            for step in steps:
                for member_name, results_dir in _result_roots(step):
                    for path in sorted(results_dir.glob("point_*.csv")):
                        point_name = path.stem.removeprefix("point_")
                        if point_name not in point_lookup:
                            raise ValueError(f"Unexpected point output {point_name!r} in {path}")
                        index, frame = _read_point_csv(path)
                        time_indices = np.asarray([time_lookup[pd.Timestamp(value)] for value in index], dtype=int)
                        member_idx = member_lookup[member_name]
                        point_idx = point_lookup[point_name]
                        for name in frame.columns:
                            if name not in variables:
                                raise ValueError(f"Unexpected point variable {name!r} in {path}")
                            variables[name][time_indices, member_idx, point_idx] = frame[name].to_numpy(dtype=float)
        os.replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return validate_project_ensemble_points(project_dir, output_nc=output)


def load_compact_point_series(
    project_dir: str | Path,
    *,
    point_filename: str,
    member: str,
    variable: str,
) -> pd.Series | None:
    """Read one member/point series from retained compact output."""
    path = project_ensemble_points_path(project_dir)
    if not path.is_file():
        return None
    point = Path(point_filename).stem.removeprefix("point_")
    with netCDF4.Dataset(path) as dataset:
        members = [str(value) for value in dataset.variables["member"][:]]
        points = [str(value) for value in dataset.variables["point"][:]]
        if member not in members or point not in points or variable not in dataset.variables:
            return None
        time_var = dataset.variables["time"]
        dates = pd.DatetimeIndex(
            netCDF4.num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
        )
        values = np.asarray(
            dataset.variables[variable][:, members.index(member), points.index(point)],
            dtype=float,
        )
    series = pd.Series(values, index=dates, name=variable).dropna()
    return None if series.empty else series


def compact_point_filenames(project_dir: str | Path) -> list[str]:
    """Return point CSV basenames represented in the compact store."""
    path = project_ensemble_points_path(project_dir)
    if not path.is_file():
        return []
    with netCDF4.Dataset(path) as dataset:
        return [f"point_{str(value)}.csv" for value in dataset.variables["point"][:]]


def compact_point_members(project_dir: str | Path) -> list[str]:
    """Return retained ensemble member identities, excluding open loop."""
    path = project_ensemble_points_path(project_dir)
    if not path.is_file():
        return []
    with netCDF4.Dataset(path) as dataset:
        return [
            str(value)
            for value in dataset.variables["member"][:]
            if str(value) != "open_loop"
        ]


__all__ = [
    "compact_point_filenames",
    "compact_point_members",
    "load_compact_point_series",
    "validate_project_ensemble_points",
    "write_project_ensemble_points",
]
