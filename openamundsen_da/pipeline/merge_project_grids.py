"""Merge completed project DA summary grid NetCDFs along time axes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import xarray as xr
from loguru import logger

from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.storage_policy import preserved_netcdf_encoding


DA_SUMMARY_NAME = "da_output_grids.nc"
MERGE_ATTR_PREFIX = "project_merge_"
_DROPPED_SOURCE_ATTRS = {
    "source_step_count",
    "source_steps",
}


def _project_grid_path(project_dir: Path) -> Path:
    return Path(project_dir) / "results" / "grids" / DA_SUMMARY_NAME


def _is_time_dim(dim: str) -> bool:
    return "time" in str(dim).lower()


def _time_dims(dims: Iterable[str]) -> list[str]:
    return [str(dim) for dim in dims if _is_time_dim(str(dim))]


def _format_value(value: object) -> str:
    text = np.datetime_as_string(value, unit="D") if np.issubdtype(np.asarray(value).dtype, np.datetime64) else str(value)
    return text


def _normalize_attr_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize_attrs(attrs: dict) -> dict:
    return {str(key): _normalize_attr_value(value) for key, value in dict(attrs).items()}


def _attrs_equal(left: dict, right: dict) -> bool:
    return _normalize_attrs(left) == _normalize_attrs(right)


def _encoding_for_compare(encoding: dict) -> dict:
    out: dict[str, object] = {}
    for key, value in preserved_netcdf_encoding(encoding).items():
        if key == "chunksizes":
            continue
        if key == "dtype":
            out[key] = str(np.dtype(value))
        else:
            out[key] = _normalize_attr_value(value)
    return out


def _encoding_for_write(encoding: dict) -> dict:
    return dict(preserved_netcdf_encoding(encoding))


def _coord_values_equal(left: xr.DataArray, right: xr.DataArray) -> bool:
    if left.dims != right.dims:
        return False
    if left.shape != right.shape:
        return False
    return bool(np.array_equal(left.values, right.values, equal_nan=True))


def _require_unique_merged_coord(
    *,
    dim: str,
    datasets: Sequence[xr.Dataset],
    sources: Sequence[Path],
) -> None:
    values: list[np.ndarray] = []
    for ds, source in zip(datasets, sources, strict=True):
        if dim not in ds.coords:
            raise ValueError(f"Time dimension {dim!r} is missing as a coordinate in {source}")
        arr = np.asarray(ds[dim].values).reshape(-1)
        if arr.size != np.unique(arr).size:
            _, counts = np.unique(arr, return_counts=True)
            duplicate_values = np.unique(arr)[counts > 1]
            formatted = ", ".join(_format_value(value) for value in duplicate_values[:10])
            raise ValueError(f"Duplicate {dim!r} timestamps inside {source}: {formatted}")
        values.append(arr)
    merged = np.concatenate(values) if values else np.asarray([])
    unique, counts = np.unique(merged, return_counts=True)
    duplicates = unique[counts > 1]
    if duplicates.size:
        formatted = ", ".join(_format_value(value) for value in duplicates[:10])
        raise ValueError(f"Duplicate {dim!r} timestamps across project DA grid files: {formatted}")


def _sort_data_array_along_time(da: xr.DataArray, time_dim: str) -> xr.DataArray:
    if time_dim in da.coords:
        return da.sortby(time_dim)
    return da


def _validate_dataset_contract(
    *,
    datasets: Sequence[xr.Dataset],
    sources: Sequence[Path],
) -> None:
    if len(datasets) < 2:
        raise ValueError("At least two project DA grid files are required")

    first = datasets[0]
    first_source = sources[0]
    first_vars = set(first.data_vars)
    first_coords = set(first.coords)

    for ds, source in zip(datasets[1:], sources[1:], strict=True):
        vars_here = set(ds.data_vars)
        if vars_here != first_vars:
            missing = sorted(first_vars - vars_here)
            extra = sorted(vars_here - first_vars)
            raise ValueError(
                f"Data variables in {source} do not match {first_source}; "
                f"missing={missing}, extra={extra}"
            )

        coords_here = set(ds.coords)
        missing_coords = sorted(first_coords - coords_here)
        if missing_coords:
            raise ValueError(f"Coordinates in {source} do not match {first_source}; missing={missing_coords}")

    static_coords = [name for name, coord in first.coords.items() if not _time_dims(coord.dims)]
    for coord_name in static_coords:
        coord_first = first.coords[coord_name]
        for ds, source in zip(datasets[1:], sources[1:], strict=True):
            coord = ds.coords[coord_name]
            if not _coord_values_equal(coord_first, coord) or not _attrs_equal(coord_first.attrs, coord.attrs):
                raise ValueError(f"Static coordinate {coord_name!r} in {source} does not match {first_source}")

    for var_name in sorted(first_vars):
        da_first = first[var_name]
        var_time_dims = _time_dims(da_first.dims)
        if len(var_time_dims) > 1:
            raise ValueError(f"Variable {var_name!r} has multiple time-like dimensions: {var_time_dims}")

        first_compare_encoding = _encoding_for_compare(da_first.encoding)
        first_attrs = _normalize_attrs(da_first.attrs)
        for ds, source in zip(datasets[1:], sources[1:], strict=True):
            da = ds[var_name]
            if da.dims != da_first.dims:
                raise ValueError(f"Variable {var_name!r} dimensions in {source} do not match {first_source}")
            if _time_dims(da.dims) != var_time_dims:
                raise ValueError(f"Variable {var_name!r} time dimensions in {source} do not match {first_source}")
            if _normalize_attrs(da.attrs) != first_attrs:
                raise ValueError(f"Variable {var_name!r} attributes in {source} do not match {first_source}")
            if _encoding_for_compare(da.encoding) != first_compare_encoding:
                raise ValueError(f"Variable {var_name!r} NetCDF encoding in {source} does not match {first_source}")

    checked_time_dims = sorted(
        {
            dim
            for ds in datasets
            for var_name in ds.data_vars
            for dim in _time_dims(ds[var_name].dims)
        }
    )
    for dim in checked_time_dims:
        _require_unique_merged_coord(dim=dim, datasets=datasets, sources=sources)


def _build_merged_dataset(
    *,
    datasets: Sequence[xr.Dataset],
    project_dirs: Sequence[Path],
    sources: Sequence[Path],
    merge_command: str | None,
) -> tuple[xr.Dataset, dict[str, dict]]:
    first = datasets[0]
    data_vars: dict[str, xr.DataArray] = {}
    encoding: dict[str, dict] = {}

    for var_name in sorted(first.data_vars):
        arrays = [ds[var_name] for ds in datasets]
        time_dims = _time_dims(arrays[0].dims)
        if not time_dims:
            data_vars[var_name] = arrays[0]
        else:
            time_dim = time_dims[0]
            merged = xr.concat(arrays, dim=time_dim, join="exact", compat="equals", coords="minimal")
            data_vars[var_name] = _sort_data_array_along_time(merged, time_dim)
        encoding[var_name] = _encoding_for_write(arrays[0].encoding)

    merged_ds = xr.Dataset(data_vars=data_vars, attrs=_merged_attrs(first, project_dirs, sources, datasets, merge_command))
    merged_ds = _attach_missing_coords(merged_ds, datasets)
    return merged_ds, encoding


def _attach_missing_coords(merged_ds: xr.Dataset, datasets: Sequence[xr.Dataset]) -> xr.Dataset:
    coord_names = sorted({name for ds in datasets for name in ds.coords})
    for coord_name in coord_names:
        if coord_name in merged_ds.coords:
            continue
        coord_arrays = [ds.coords[coord_name] for ds in datasets if coord_name in ds.coords]
        if not coord_arrays:
            continue
        time_dims = _time_dims(coord_arrays[0].dims)
        if len(time_dims) == 1:
            time_dim = time_dims[0]
            coord = xr.concat(coord_arrays, dim=time_dim, join="exact", compat="equals", coords="minimal")
            merged_ds = merged_ds.assign_coords({coord_name: _sort_data_array_along_time(coord, time_dim)})
        else:
            merged_ds = merged_ds.assign_coords({coord_name: coord_arrays[0]})
    return merged_ds


def _time_range_for_dataset(ds: xr.Dataset) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    time_dims = sorted({dim for name in ds.data_vars for dim in _time_dims(ds[name].dims)})
    for dim in time_dims:
        values = np.asarray(ds[dim].values).reshape(-1)
        if values.size:
            out[dim] = [_format_value(values[0]), _format_value(values[-1])]
    return out


def _merged_attrs(
    template: xr.Dataset,
    project_dirs: Sequence[Path],
    sources: Sequence[Path],
    datasets: Sequence[xr.Dataset],
    merge_command: str | None,
) -> dict[str, str]:
    attrs = {
        str(key): str(value)
        for key, value in dict(template.attrs).items()
        if key not in _DROPPED_SOURCE_ATTRS and not str(key).startswith(MERGE_ATTR_PREFIX)
    }
    project_names = [path.name for path in project_dirs]
    ranges = [
        {
            "project": project_dir.name,
            "project_dir": str(project_dir),
            "source_file": str(source),
            "time_ranges": _time_range_for_dataset(ds),
        }
        for project_dir, source, ds in zip(project_dirs, sources, datasets, strict=True)
    ]
    attrs.update(
        {
            "project_merge": "true",
            "project_merge_version": "1",
            "project_merge_created_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_merge_source_project_count": str(len(project_dirs)),
            "project_merge_source_projects": ",".join(project_names),
            "project_merge_source_project_dirs": "|".join(str(path) for path in project_dirs),
            "project_merge_source_files": "|".join(str(path) for path in sources),
            "project_merge_source_time_ranges": json.dumps(ranges, separators=(",", ":")),
        }
    )
    if merge_command:
        attrs["project_merge_command"] = merge_command
    return attrs


def merge_project_da_output_grids(
    project_dirs: Sequence[Path | str],
    output_nc: Path | str,
    *,
    overwrite: bool = False,
    merge_command: str | None = None,
) -> Path:
    """Merge completed project ``da_output_grids.nc`` files along their time axes."""
    resolved_project_dirs = [Path(path) for path in project_dirs]
    if len(resolved_project_dirs) < 2:
        raise ValueError("At least two project directories are required")

    output_path = Path(output_nc)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output NetCDF already exists: {output_path}. Pass --overwrite to replace it.")

    sources = [_project_grid_path(project_dir) for project_dir in resolved_project_dirs]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Project DA grid NetCDF missing:\n{formatted}")

    datasets: list[xr.Dataset] = []
    try:
        for source in sources:
            datasets.append(xr.open_dataset(source))
        _validate_dataset_contract(datasets=datasets, sources=sources)
        merged, encoding = _build_merged_dataset(
            datasets=datasets,
            project_dirs=resolved_project_dirs,
            sources=sources,
            merge_command=merge_command,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_netcdf(output_path, encoding=encoding)
        merged.close()
    finally:
        for ds in datasets:
            ds.close()

    logger.info("Wrote merged project DA grid NetCDF {}", output_path)
    return output_path


def _resolve_project_dirs(
    *,
    setup: Path | None,
    projects: Sequence[str] | None,
    project_dirs: Sequence[Path] | None,
) -> list[Path]:
    projects = list(projects or [])
    project_dirs = list(project_dirs or [])
    if setup is not None:
        if project_dirs:
            raise ValueError("Use either --setup with --project names or --project-dir paths, not both.")
        if not projects:
            raise ValueError("--setup requires at least one --project name")
        return [Path(setup) / "projects" / name for name in projects]
    if projects:
        raise ValueError("--project requires --setup so names can be resolved below <setup>/projects")
    if not project_dirs:
        raise ValueError("Pass --setup with --project names or repeated --project-dir paths")
    return [Path(path) for path in project_dirs]


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-merge-project-grids",
        description="Merge completed project DA summary grid NetCDFs along time axes.",
    )
    parser.add_argument("--setup", type=Path, help="Setup root; --project names are resolved below <setup>/projects")
    parser.add_argument("--project", action="append", help="Project directory name below <setup>/projects; repeatable")
    parser.add_argument("--project-dir", action="append", type=Path, help="Completed project directory path; repeatable")
    parser.add_argument("--output-nc", required=True, type=Path, help="Merged output NetCDF path")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output NetCDF if it exists")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args(list(argv) if argv is not None else None)
    configure_cli_logger(args.log_level, stream=sys.stderr)

    try:
        project_dirs = _resolve_project_dirs(
            setup=args.setup,
            projects=args.project,
            project_dirs=args.project_dir,
        )
        merge_project_da_output_grids(
            project_dirs,
            args.output_nc,
            overwrite=args.overwrite,
            merge_command=" ".join(["oa-da-merge-project-grids", *(argv or sys.argv[1:])]),
        )
    except Exception as exc:
        logger.error("Project DA grid merge failed: {}", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
