"""Compressed, streaming all-member forcing time-series output."""

from __future__ import annotations

import os
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd

from openamundsen_da.io.paths import (
    meteo_dir_for_member,
    list_member_dirs,
    list_steps_sorted,
    project_ensemble_forcing_path,
)
from openamundsen_da.util.atomic import durable_replace
from openamundsen_da.util.ts import collapse_duplicates


_TIME_COLUMNS = ("date", "time", "datetime")
_KNOWN_UNITS = {
    "temp": "K",
    "precip": "kg m-2",
    "rel_hum": "%",
    "sw_in": "W m-2",
    "wind_speed": "m s-1",
    "wind_dir": "degree",
}


def _read_forcing_csv(path: Path) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    frame = pd.read_csv(path)
    time_col = next((column for column in _TIME_COLUMNS if column in frame.columns), None)
    if time_col is None:
        raise ValueError(f"Forcing output has no date/time column: {path}")
    times = pd.DatetimeIndex(pd.to_datetime(frame.pop(time_col), errors="raise"))
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Forcing output contains an unrecognized nonnumeric value: {path}") from exc
    return times, numeric


def _reference_meteo_dir(step_dir: Path) -> Path:
    open_loop = meteo_dir_for_member(
        step_dir / "ensembles" / "prior" / "open_loop"
    )
    if any(path.name != "stations.csv" for path in open_loop.glob("*.csv")):
        return open_loop
    for member in list_member_dirs(step_dir / "ensembles", "prior"):
        meteo = meteo_dir_for_member(member)
        if any(path.name != "stations.csv" for path in meteo.glob("*.csv")):
            return meteo
    raise FileNotFoundError(f"No member forcing found in {step_dir}")


def _schema(steps: list[Path]) -> tuple[list[str], list[str], pd.DatetimeIndex]:
    stations: set[str] = set()
    variables: set[str] = set()
    times: set[pd.Timestamp] = set()
    for step in steps:
        reference = _reference_meteo_dir(step)
        for path in sorted(reference.glob("*.csv")):
            if path.name == "stations.csv":
                continue
            stations.add(path.stem)
            index, frame = _read_forcing_csv(path)
            times.update(pd.Timestamp(value) for value in index)
            variables.update(str(column) for column in frame.columns)
    if not stations or not variables or not times:
        raise ValueError("Forcing output schema is empty")
    return sorted(stations), sorted(variables), pd.DatetimeIndex(sorted(times))


def _member_names(steps: list[Path]) -> list[str]:
    expected: list[str] | None = None
    for step in steps:
        names = [member.name for member in list_member_dirs(step / "ensembles", "prior")]
        if expected is None:
            expected = names
        elif names != expected:
            raise ValueError(f"Forcing member identities differ in {step}: {names} != {expected}")
    if not expected:
        raise ValueError("No prior members found for forcing output")
    return ["open_loop", *expected]


def _meteo_roots(step: Path) -> list[tuple[str, Path]]:
    roots = [
        (
            "open_loop",
            meteo_dir_for_member(step / "ensembles" / "prior" / "open_loop"),
        )
    ]
    roots.extend(
        (member.name, meteo_dir_for_member(member))
        for member in list_member_dirs(step / "ensembles", "prior")
    )
    return roots


def _station_paths(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.csv") if path.name != "stations.csv")


def _validate_source_completeness(steps: list[Path]) -> None:
    """Require every member to contain the reference forcing files and schema."""
    for step in steps:
        reference = _reference_meteo_dir(step)
        expected_paths = _station_paths(reference)
        expected_names = [path.name for path in expected_paths]
        expected = {path.name: _read_forcing_csv(path) for path in expected_paths}
        for member, meteo in _meteo_roots(step):
            if not meteo.is_dir():
                raise FileNotFoundError(f"Missing forcing directory for {member}: {meteo}")
            actual_names = [path.name for path in _station_paths(meteo)]
            if actual_names != expected_names:
                raise ValueError(
                    f"Forcing station files differ for {member} in {step}: "
                    f"{actual_names} != {expected_names}"
                )
            for name in expected_names:
                expected_index, expected_frame = expected[name]
                index, frame = _read_forcing_csv(meteo / name)
                if not index.is_unique:
                    raise ValueError(f"Duplicate forcing timestamps in {meteo / name}")
                if not index.equals(expected_index) or list(frame.columns) != list(expected_frame.columns):
                    raise ValueError(f"Forcing time/variable schema differs in {meteo / name}")


def _collapsed_forcing_frame(
    steps: list[Path],
    *,
    member: str,
    station: str,
) -> pd.DataFrame:
    """Load one logical raw series with the established mean-overlap rule."""
    frames: list[pd.DataFrame] = []
    filename = f"{station}.csv"
    for step in steps:
        roots = dict(_meteo_roots(step))
        meteo = roots.get(member)
        path = meteo / filename if meteo is not None else None
        if path is None or not path.is_file():
            continue
        index, frame = _read_forcing_csv(path)
        nonempty = ~frame.isna().all(axis=1)
        if not bool(nonempty.any()):
            continue
        index = index[nonempty.to_numpy()]
        frame = frame.loc[nonempty].copy()
        frame.index = index
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return collapse_duplicates(pd.concat(frames, axis=0))


def validate_project_ensemble_forcing(
    project_dir: str | Path,
    *,
    output_nc: str | Path | None = None,
) -> Path:
    """Validate the retained all-member forcing NetCDF contract."""
    project_dir = Path(project_dir).resolve()
    steps = list_steps_sorted(project_dir)
    station_names, variable_names, times = _schema(steps)
    member_names = _member_names(steps)
    path = Path(output_nc) if output_nc is not None else project_ensemble_forcing_path(project_dir)
    with netCDF4.Dataset(path) as dataset:
        if set(dataset.dimensions) != {"time", "member", "station"}:
            raise ValueError(f"Invalid compact forcing dimensions in {path}")
        retained_members = [str(value) for value in dataset.variables["member"][:]]
        retained_stations = [str(value) for value in dataset.variables["station"][:]]
        if retained_members != member_names or retained_stations != station_names:
            raise ValueError(f"Compact forcing identities do not match member forcing: {path}")
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
            raise ValueError(f"Compact forcing time coverage does not match member forcing: {path}")
        missing = [name for name in variable_names if name not in dataset.variables]
        if missing:
            raise ValueError(f"Compact forcing variables missing in {path}: {', '.join(missing)}")
        for member_idx, member in enumerate(member_names):
            for station_idx, station in enumerate(station_names):
                expected = _collapsed_forcing_frame(
                    steps,
                    member=member,
                    station=station,
                ).reindex(retained_times)
                for name in variable_names:
                    variable = dataset.variables[name]
                    if tuple(variable.dimensions) != ("time", "member", "station"):
                        raise ValueError(f"Invalid compact forcing variable dimensions for {name} in {path}")
                    expected_unit = _KNOWN_UNITS.get(name)
                    if expected_unit is not None and getattr(variable, "units", None) != expected_unit:
                        raise ValueError(f"Invalid compact forcing units for {name} in {path}")
                    expected_values = (
                        expected[name].to_numpy(dtype=float)
                        if name in expected.columns
                        else np.full(len(retained_times), np.nan)
                    )
                    retained = variable[:, member_idx, station_idx]
                    retained_values = np.ma.filled(retained, np.nan).astype(float)
                    if not np.allclose(
                        retained_values,
                        expected_values,
                        rtol=0.0,
                        atol=0.0,
                        equal_nan=True,
                    ):
                        mismatch = next(
                            (
                                stamp
                                for stamp, actual, wanted in zip(
                                    retained_times,
                                    retained_values,
                                    expected_values,
                                )
                                if not (
                                    (np.isnan(actual) and np.isnan(wanted))
                                    or actual == wanted
                                )
                            ),
                            None,
                        )
                        raise ValueError(
                            "Compact forcing values do not match mean-collapsed raw output "
                            f"for member={member}, station={station}, variable={name}, time={mismatch}: {path}"
                        )
    return path


def write_project_ensemble_forcing(
    project_dir: str | Path,
    *,
    output_nc: str | Path | None = None,
) -> Path:
    """Stream consumed step forcing into one compressed NetCDF4 file."""
    project_dir = Path(project_dir).resolve()
    steps = list_steps_sorted(project_dir)
    if not steps:
        raise FileNotFoundError(f"No steps found under {project_dir}")
    station_names, variable_names, times = _schema(steps)
    member_names = _member_names(steps)
    _validate_source_completeness(steps)
    output = Path(output_nc) if output_nc is not None else project_ensemble_forcing_path(project_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    time_lookup = {pd.Timestamp(value): idx for idx, value in enumerate(times)}
    member_lookup = {name: idx for idx, name in enumerate(member_names)}
    station_lookup = {name: idx for idx, name in enumerate(station_names)}
    try:
        with netCDF4.Dataset(tmp, "w", format="NETCDF4") as dataset:
            dataset.setncattr("Conventions", "CF-1.10")
            dataset.setncattr("title", "openAMUNDSEN-DA consumed all-member forcing")
            dataset.setncattr("source", "native perturbed forcing CSV output")
            dataset.setncattr(
                "time_semantics",
                "timezone-aware source timestamps converted to UTC; naive source timestamps preserved",
            )
            dataset.createDimension("time", len(times))
            dataset.createDimension("member", len(member_names))
            dataset.createDimension("station", len(station_names))
            time_var = dataset.createVariable("time", "f8", ("time",))
            time_var.units = "seconds since 1970-01-01 00:00:00"
            time_var.calendar = "proleptic_gregorian"
            time_var[:] = netCDF4.date2num(times.to_pydatetime(), time_var.units, time_var.calendar)
            dataset.createVariable("member", str, ("member",))[:] = np.asarray(member_names, dtype=object)
            dataset.createVariable("station", str, ("station",))[:] = np.asarray(station_names, dtype=object)
            chunks = (min(256, len(times)), 1, min(32, len(station_names)))
            variables = {}
            for name in variable_names:
                var = dataset.createVariable(
                    name,
                    "f8",
                    ("time", "member", "station"),
                    zlib=True,
                    complevel=4,
                    shuffle=True,
                    chunksizes=chunks,
                    fill_value=np.nan,
                )
                if name in _KNOWN_UNITS:
                    var.units = _KNOWN_UNITS[name]
                var.long_name = name.replace("_", " ")
                variables[name] = var
            for member_name, member_idx in member_lookup.items():
                for station, station_idx in station_lookup.items():
                    frame = _collapsed_forcing_frame(
                        steps,
                        member=member_name,
                        station=station,
                    )
                    if frame.empty:
                        continue
                    time_indices = np.asarray(
                        [time_lookup[pd.Timestamp(value)] for value in frame.index],
                        dtype=int,
                    )
                    for name in frame.columns:
                        if name not in variables:
                            raise ValueError(f"Unexpected forcing variable {name!r} for {station}")
                        variables[name][time_indices, member_idx, station_idx] = frame[name].to_numpy(dtype=float)
        # Validate the temporary while all raw sources still exist. A failed
        # validation must not replace an already accepted compact store.
        validate_project_ensemble_forcing(project_dir, output_nc=tmp)
        durable_replace(tmp, output)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return output


def compact_forcing_stations(project_dir: str | Path) -> list[str]:
    path = project_ensemble_forcing_path(project_dir)
    if not path.is_file():
        return []
    with netCDF4.Dataset(path) as dataset:
        return [f"{str(value)}.csv" for value in dataset.variables["station"][:]]


def compact_forcing_members(project_dir: str | Path) -> list[str]:
    path = project_ensemble_forcing_path(project_dir)
    if not path.is_file():
        return []
    with netCDF4.Dataset(path) as dataset:
        return [str(value) for value in dataset.variables["member"][:] if str(value) != "open_loop"]


def load_compact_forcing_series(
    project_dir: str | Path,
    *,
    station_filename: str,
    member: str,
    variables: list[str],
) -> pd.DataFrame | None:
    path = project_ensemble_forcing_path(project_dir)
    if not path.is_file():
        return None
    station = Path(station_filename).stem
    with netCDF4.Dataset(path) as dataset:
        members = [str(value) for value in dataset.variables["member"][:]]
        stations = [str(value) for value in dataset.variables["station"][:]]
        if member not in members or station not in stations:
            return None
        missing = [name for name in variables if name not in dataset.variables]
        if missing:
            raise ValueError(f"Missing compact forcing variable(s): {', '.join(missing)}")
        time_var = dataset.variables["time"]
        dates = pd.DatetimeIndex(
            netCDF4.num2date(
                time_var[:],
                units=time_var.units,
                calendar=getattr(time_var, "calendar", "standard"),
                only_use_cftime_datetimes=False,
            )
        )
        data = {
            name: np.asarray(
                dataset.variables[name][:, members.index(member), stations.index(station)],
                dtype=float,
            )
            for name in variables
        }
    frame = pd.DataFrame(data, index=dates).dropna(how="all")
    return None if frame.empty else frame


__all__ = [
    "compact_forcing_members",
    "compact_forcing_stations",
    "load_compact_forcing_series",
    "validate_project_ensemble_forcing",
    "write_project_ensemble_forcing",
]
