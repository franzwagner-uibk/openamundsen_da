"""DA output-grid summaries and compact retention helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml, infer_project_dir, read_step_config
from openamundsen_da.util.da_events import AssimilationEvent, load_assimilation_events
from openamundsen_da.util.da_observables import weights_csv_name


def output_retention_mode(project_dir: Path) -> str:
    """Return output retention mode from project YAML, defaulting to compact."""
    try:
        cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
        da_cfg = cfg.get("data_assimilation") or {}
        out_cfg = da_cfg.get("output") or {}
        mode = str(out_cfg.get("retention", "compact")).strip().lower()
        if mode in {"compact", "full"}:
            return mode
        logger.warning(
            "Unknown data_assimilation.output.retention='{}' in {}; using 'compact'",
            mode,
            project_dir,
        )
        return "compact"
    except Exception:
        return "compact"


def _as_nan_array(da: xr.DataArray) -> np.ndarray:
    arr = np.asarray(da.values, dtype=np.float64)
    fill = da.attrs.get("_FillValue", np.nan)
    if not (isinstance(fill, float) and np.isnan(fill)):
        arr = np.where(arr == fill, np.nan, arr)
    return arr


def _member_id_from_output_nc(path: Path) -> str:
    """Return member directory name for a member results/output_grids.nc path."""
    return Path(path).parent.parent.name


def _normalize_time_index(values: object) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(value).normalize() for value in pd.to_datetime(values))


def _time_index_by_date(ds: xr.Dataset, da: xr.DataArray) -> tuple[int | None, dict[pd.Timestamp, int]]:
    """Return the single time axis and normalized-date index for a grid variable."""
    time_dims = [dim for dim in da.dims if "time" in str(dim).lower()]
    if len(time_dims) != 1:
        return None, {}
    time_dim = time_dims[0]
    if time_dim not in ds.coords:
        return None, {}
    try:
        dates = _normalize_time_index(ds[time_dim].values)
    except Exception:
        return None, {}
    return da.get_axis_num(time_dim), {date: idx for idx, date in enumerate(dates)}


def _analysis_mean_for_time(
    *,
    var_name: str,
    member_files: Sequence[Path],
    weights: pd.Series,
    shape: tuple[int, ...],
    time_axis: int,
    time_idx: int,
) -> np.ndarray:
    """Compute weighted member mean for one variable/time slice."""
    slicer = [slice(None)] * len(shape)
    slicer[time_axis] = time_idx
    target = tuple(slicer)
    slice_shape = tuple(size for axis, size in enumerate(shape) if axis != time_axis)
    weighted_sum = np.zeros(slice_shape, dtype=np.float64)
    weight_sum = np.zeros(slice_shape, dtype=np.float64)
    matched = 0

    for nc_path in member_files:
        member_id = _member_id_from_output_nc(nc_path)
        if member_id not in weights.index:
            continue
        weight = float(weights.loc[member_id])
        if not np.isfinite(weight) or weight <= 0.0:
            continue
        with xr.open_dataset(nc_path) as ds_m:
            if var_name not in ds_m.data_vars:
                continue
            da_m = ds_m[var_name]
            if tuple(int(s) for s in da_m.shape) != shape:
                continue
            arr = _as_nan_array(da_m)[target]
        valid = np.isfinite(arr)
        if not np.any(valid):
            continue
        weighted_sum[valid] += arr[valid] * weight
        weight_sum[valid] += weight
        matched += 1

    if matched == 0:
        logger.warning("Analysis mean skipped for {}: no weighted member grids matched", var_name)
    out = np.full(slice_shape, np.nan, dtype=np.float64)
    np.divide(weighted_sum, weight_sum, out=out, where=weight_sum > 0.0)
    return out


def _analysis_arrays(
    *,
    var_name: str,
    ds_ol: xr.Dataset,
    da_ol: xr.DataArray,
    member_files: Sequence[Path],
    arr_mean: np.ndarray,
    analysis_weights: dict[pd.Timestamp, pd.Series],
    write_analysis_fields: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not write_analysis_fields:
        return None
    shape = tuple(int(s) for s in da_ol.shape)
    time_axis, index_by_date = _time_index_by_date(ds_ol, da_ol)
    if time_axis is None:
        return None

    analysis_mean = np.full(shape, np.nan, dtype=np.float64)
    analysis_increment = np.full(shape, np.nan, dtype=np.float64)
    for date, weights in sorted(analysis_weights.items()):
        idx = index_by_date.get(pd.Timestamp(date).normalize())
        if idx is None:
            continue
        slicer = [slice(None)] * len(shape)
        slicer[time_axis] = idx
        target = tuple(slicer)
        weighted_mean = _analysis_mean_for_time(
            var_name=var_name,
            member_files=member_files,
            weights=weights,
            shape=shape,
            time_axis=time_axis,
            time_idx=idx,
        )
        analysis_mean[target] = weighted_mean
        analysis_increment[target] = weighted_mean - arr_mean[target]

    return analysis_mean, analysis_increment


def _build_da_output_dataset(
    *,
    open_loop_nc: Path,
    member_ncs: Sequence[Path],
    analysis_weights: dict[pd.Timestamp, pd.Series] | None = None,
    write_analysis_fields: bool = False,
) -> xr.Dataset | None:
    """Build compact DA summary grids for one step from open-loop + members."""
    if not open_loop_nc.is_file():
        logger.warning("DA output summary skipped: open_loop NetCDF not found at {}", open_loop_nc)
        return None
    member_files = [Path(p) for p in member_ncs if Path(p).is_file()]
    if not member_files:
        logger.warning("DA output summary skipped: no member NetCDF files provided")
        return None

    with xr.open_dataset(open_loop_nc) as ds_ol:
        out_vars: dict[str, xr.DataArray] = {}
        encoding: dict[str, dict] = {}
        n_members = int(len(member_files))
        grid_var_names = []
        for var_name, da_ol in ds_ol.data_vars.items():
            if "y" not in da_ol.dims or "x" not in da_ol.dims:
                continue
            grid_var_names.append(var_name)
            shape = tuple(int(s) for s in da_ol.shape)
            arr_sum = np.zeros(shape, dtype=np.float64)
            arr_sum_sq = np.zeros(shape, dtype=np.float64)
            arr_count = np.zeros(shape, dtype=np.int32)
            arr_min = np.full(shape, np.nan, dtype=np.float64)
            arr_max = np.full(shape, np.nan, dtype=np.float64)

            for nc_path in member_files:
                with xr.open_dataset(nc_path) as ds_m:
                    if var_name not in ds_m.data_vars:
                        logger.warning("Variable {} missing in {}", var_name, nc_path)
                        continue
                    da_m = ds_m[var_name]
                    if tuple(int(s) for s in da_m.shape) != shape:
                        logger.warning(
                            "Variable {} shape mismatch for {} (expected {}, got {})",
                            var_name,
                            nc_path,
                            shape,
                            tuple(int(s) for s in da_m.shape),
                        )
                        continue
                    arr = _as_nan_array(da_m)
                    valid = np.isfinite(arr)
                    if not np.any(valid):
                        continue
                    arr_sum[valid] += arr[valid]
                    arr_sum_sq[valid] += arr[valid] * arr[valid]
                    arr_count[valid] += 1
                    arr_min = np.where(np.isnan(arr_min), arr, np.fmin(arr_min, arr))
                    arr_max = np.where(np.isnan(arr_max), arr, np.fmax(arr_max, arr))

            with np.errstate(invalid="ignore", divide="ignore"):
                arr_mean = np.full(shape, np.nan, dtype=np.float64)
                np.divide(arr_sum, arr_count, out=arr_mean, where=arr_count > 0)
                arr_second_moment = np.full(shape, np.nan, dtype=np.float64)
                np.divide(arr_sum_sq, arr_count, out=arr_second_moment, where=arr_count > 0)
                arr_var = arr_second_moment - (arr_mean * arr_mean)
            arr_mean = np.where(arr_count > 0, arr_mean, np.nan)
            arr_var = np.where(arr_count > 0, arr_var, np.nan)
            arr_var = np.where(arr_var < 0, 0.0, arr_var)
            arr_std = np.sqrt(arr_var)
            arr_ol = _as_nan_array(da_ol)
            # DA increment is posterior ensemble mean minus open-loop baseline.
            arr_inc = arr_mean - arr_ol
            analysis = _analysis_arrays(
                var_name=var_name,
                ds_ol=ds_ol,
                da_ol=da_ol,
                member_files=member_files,
                arr_mean=arr_mean,
                analysis_weights=analysis_weights or {},
                write_analysis_fields=write_analysis_fields,
            )

            dims = da_ol.dims
            coords = {d: ds_ol.coords[d] for d in dims if d in ds_ol.coords}
            base_attrs = dict(da_ol.attrs) if da_ol.attrs is not None else {}
            out_vars[f"open_loop_{var_name}"] = xr.DataArray(
                arr_ol.astype(np.float32),
                dims=dims,
                coords=coords,
                attrs={
                    **base_attrs,
                    "summary_metric": "open_loop",
                    "description": "Open-loop baseline output (no assimilation)",
                },
            )
            out_vars[f"ens_mean_{var_name}"] = xr.DataArray(
                arr_mean.astype(np.float32),
                dims=dims,
                coords=coords,
                attrs={
                    **base_attrs,
                    "summary_metric": "ens_mean",
                    "description": "Posterior ensemble mean",
                },
            )
            out_vars[f"ens_std_{var_name}"] = xr.DataArray(
                arr_std.astype(np.float32),
                dims=dims,
                coords=coords,
                attrs={
                    **base_attrs,
                    "summary_metric": "ens_std",
                    "description": "Posterior ensemble standard deviation",
                },
            )
            out_vars[f"ens_min_{var_name}"] = xr.DataArray(
                arr_min.astype(np.float32),
                dims=dims,
                coords=coords,
                attrs={
                    **base_attrs,
                    "summary_metric": "ens_min",
                    "description": "Posterior ensemble minimum",
                },
            )
            out_vars[f"ens_max_{var_name}"] = xr.DataArray(
                arr_max.astype(np.float32),
                dims=dims,
                coords=coords,
                attrs={
                    **base_attrs,
                    "summary_metric": "ens_max",
                    "description": "Posterior ensemble maximum",
                },
            )
            out_vars[f"increment_{var_name}"] = xr.DataArray(
                arr_inc.astype(np.float32),
                dims=dims,
                coords=coords,
                attrs={
                    **base_attrs,
                    "summary_metric": "increment",
                    "description": "Posterior ensemble mean minus open-loop baseline",
                },
            )
            if analysis is not None:
                arr_analysis_mean, arr_analysis_increment = analysis
                out_vars[f"analysis_mean_{var_name}"] = xr.DataArray(
                    arr_analysis_mean.astype(np.float32),
                    dims=dims,
                    coords=coords,
                    attrs={
                        **base_attrs,
                        "summary_metric": "analysis_mean",
                        "description": "Event-weighted posterior ensemble mean at assimilation date",
                    },
                )
                out_vars[f"analysis_increment_{var_name}"] = xr.DataArray(
                    arr_analysis_increment.astype(np.float32),
                    dims=dims,
                    coords=coords,
                    attrs={
                        **base_attrs,
                        "summary_metric": "analysis_increment",
                        "description": "Event-weighted posterior ensemble mean minus prior ensemble mean",
                    },
                )
            for out_name in (
                f"open_loop_{var_name}",
                f"ens_mean_{var_name}",
                f"ens_std_{var_name}",
                f"ens_min_{var_name}",
                f"ens_max_{var_name}",
                f"increment_{var_name}",
            ):
                encoding[out_name] = {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": -9999.0}
            if analysis is not None:
                for out_name in (
                    f"analysis_mean_{var_name}",
                    f"analysis_increment_{var_name}",
                ):
                    encoding[out_name] = {"zlib": True, "complevel": 4, "shuffle": True, "_FillValue": -9999.0}

        if not out_vars:
            logger.warning("DA output summary skipped: no grid variables with x/y dims in {}", open_loop_nc)
            return None

        summary_variables = ["open_loop", "ens_mean", "ens_std", "ens_min", "ens_max", "increment"]
        has_analysis_fields = any(name.startswith("analysis_mean_") for name in out_vars)
        if has_analysis_fields:
            summary_variables.extend(["analysis_mean", "analysis_increment"])
        attrs = {
            **(dict(ds_ol.attrs) if ds_ol.attrs is not None else {}),
            "da_output_version": "2",
            "source_open_loop_nc": str(open_loop_nc),
            "source_member_count": str(n_members),
            "source_member_weighting": "uniform",
            "source_grid_variables": ",".join(grid_var_names),
            "summary_variables": ",".join(summary_variables),
            "increment_definition": "increment_<var> = ens_mean_<var> - open_loop_<var>",
        }
        if has_analysis_fields:
            attrs["analysis_increment_definition"] = "analysis_increment_<var> = analysis_mean_<var> - ens_mean_<var>"

        out_ds = xr.Dataset(
            data_vars=out_vars,
            coords=ds_ol.coords,
            attrs=attrs,
        )
        for out_name, enc in encoding.items():
            out_ds[out_name].encoding.update(enc)
        return out_ds


def _sort_and_unique_along_dim(da: xr.DataArray, dim: str) -> xr.DataArray:
    """Sort DataArray by dim and drop duplicate coordinate labels along dim."""
    if dim not in da.dims or dim not in da.coords:
        return da
    out = da.sortby(dim)
    coord_vals = out[dim].values
    _, first_idx = np.unique(coord_vals, return_index=True)
    if len(first_idx) == len(coord_vals):
        return out
    keep = np.sort(first_idx)
    return out.isel({dim: keep})


def _combine_step_summaries(step_summaries: Sequence[xr.Dataset]) -> xr.Dataset:
    """Combine per-step DA summaries into one full-project summary dataset."""
    if not step_summaries:
        raise ValueError("No step summaries provided")

    first = step_summaries[0]
    combined_vars: dict[str, xr.DataArray] = {}
    var_names = sorted({name for ds in step_summaries for name in ds.data_vars})

    for var_name in var_names:
        arrays = [ds[var_name] for ds in step_summaries if var_name in ds.data_vars]
        if not arrays:
            continue
        if len(arrays) == 1:
            combined_vars[var_name] = arrays[0]
            continue

        time_dims = [dim for dim in arrays[0].dims if "time" in dim.lower()]
        if not time_dims:
            combined_vars[var_name] = arrays[-1]
            continue

        time_dim = time_dims[0]
        merged = xr.concat(
            arrays,
            dim=time_dim,
            join="outer",
            compat="override",
            coords="minimal",
        )
        combined_vars[var_name] = _sort_and_unique_along_dim(merged, time_dim)

    merged_ds = xr.Dataset(
        data_vars=combined_vars,
        attrs={**(dict(first.attrs) if first.attrs is not None else {})},
    )

    # Re-attach scalar/non-time coordinates (e.g. crs) and concatenate time-dependent
    # auxiliary coordinates (e.g. time bounds) where possible.
    coord_names = sorted({name for ds in step_summaries for name in ds.coords})
    for coord_name in coord_names:
        if coord_name in merged_ds.coords:
            continue
        coord_arrays = [ds.coords[coord_name] for ds in step_summaries if coord_name in ds.coords]
        if not coord_arrays:
            continue
        if len(coord_arrays) == 1:
            merged_ds = merged_ds.assign_coords({coord_name: coord_arrays[0]})
            continue
        time_dims = [dim for dim in coord_arrays[0].dims if "time" in dim.lower()]
        if len(time_dims) == 1:
            time_dim = time_dims[0]
            merged_coord = xr.concat(
                coord_arrays,
                dim=time_dim,
                join="outer",
                compat="override",
                coords="minimal",
            )
            merged_ds = merged_ds.assign_coords({coord_name: _sort_and_unique_along_dim(merged_coord, time_dim)})
        else:
            merged_ds = merged_ds.assign_coords({coord_name: coord_arrays[0]})

    return merged_ds


def write_da_output_grids(
    *,
    open_loop_nc: Path,
    member_ncs: Sequence[Path],
    output_nc: Path,
) -> Path | None:
    """Write compact DA summary grids into a single NetCDF file."""
    out_ds = _build_da_output_dataset(
        open_loop_nc=open_loop_nc,
        member_ncs=member_ncs,
    )
    if out_ds is None:
        return None
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    out_ds.to_netcdf(output_nc)
    logger.info("Wrote DA output summary NetCDF {}", output_nc)
    return output_nc


def _step_event_weights(step_dir: Path, events: Sequence[AssimilationEvent]) -> dict[pd.Timestamp, pd.Series]:
    """Load normalized member weights for assimilation events inside one step."""
    try:
        cfg = read_step_config(step_dir) or {}
        start = pd.Timestamp(cfg["start_date"]).normalize()
        end = pd.Timestamp(cfg["end_date"]).normalize()
    except Exception as exc:
        logger.warning("Analysis increment skipped for {}: could not read step dates ({})", step_dir, exc)
        return {}

    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    for event in events:
        event_date = pd.Timestamp(event.date).normalize()
        if not (start <= event_date <= end):
            continue
        weights_path = Path(step_dir) / "assim" / weights_csv_name(event.variable, event_date.to_pydatetime())
        if not weights_path.is_file():
            logger.warning("Analysis increment skipped for {}: weights CSV not found at {}", event_date.date(), weights_path)
            continue
        try:
            df = pd.read_csv(weights_path)
        except Exception as exc:
            logger.warning("Analysis increment skipped for {}: could not read {} ({})", event_date.date(), weights_path, exc)
            continue
        if "member_id" not in df.columns or "weight" not in df.columns:
            logger.warning("Analysis increment skipped for {}: {} must contain member_id and weight columns", event_date.date(), weights_path)
            continue
        working = df[["member_id", "weight"]].copy()
        working["member_id"] = working["member_id"].astype(str).str.strip()
        working["weight"] = pd.to_numeric(working["weight"], errors="coerce")
        working = working.loc[(working["member_id"] != "") & np.isfinite(working["weight"]) & (working["weight"] >= 0.0)]
        if working.empty:
            logger.warning("Analysis increment skipped for {}: no finite non-negative weights in {}", event_date.date(), weights_path)
            continue
        weights = working.groupby("member_id")["weight"].sum()
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            logger.warning("Analysis increment skipped for {}: weights sum is not positive in {}", event_date.date(), weights_path)
            continue
        if event_date in weights_by_date:
            logger.warning(
                "Multiple assimilation events found for {}; using weights from {} for compact analysis fields",
                event_date.date(),
                weights_path,
            )
        weights_by_date[event_date] = weights.astype(float) / total
    return weights_by_date


def _load_project_assimilation_events(project_dir: Path) -> list[AssimilationEvent]:
    try:
        return load_assimilation_events(project_dir)
    except Exception as exc:
        logger.warning("Analysis increment fields skipped: could not load assimilation events for {} ({})", project_dir, exc)
        return []


def write_project_da_output_grids(
    *,
    step_dirs: Sequence[Path],
    output_nc: Path,
) -> Path | None:
    """Write one compact DA summary NetCDF spanning all available project steps."""
    step_summaries: list[xr.Dataset] = []
    used_steps: list[str] = []
    project_dir: Path | None = None
    if step_dirs:
        try:
            project_dir = infer_project_dir(step_dirs[0])
        except Exception as exc:
            logger.warning("Analysis increment fields skipped: could not infer project directory from {} ({})", step_dirs[0], exc)
    events = _load_project_assimilation_events(project_dir) if project_dir is not None else []

    for step_dir in step_dirs:
        prior_root = Path(step_dir) / "ensembles" / "prior"
        open_loop_nc = prior_root / "open_loop" / "results" / "output_grids.nc"
        member_ncs = [
            p / "results" / "output_grids.nc"
            for p in sorted(prior_root.glob("member_*"))
            if p.is_dir()
        ]
        analysis_weights = _step_event_weights(Path(step_dir), events) if events else {}
        ds = _build_da_output_dataset(
            open_loop_nc=open_loop_nc,
            member_ncs=member_ncs,
            analysis_weights=analysis_weights,
            write_analysis_fields=bool(events),
        )
        if ds is None:
            continue
        step_summaries.append(ds)
        used_steps.append(Path(step_dir).name)

    if not step_summaries:
        logger.warning("DA output summary skipped: no valid step summaries found")
        return None

    combined = _combine_step_summaries(step_summaries)
    combined.attrs.update(
        {
            "source_step_count": str(len(step_summaries)),
            "source_steps": ",".join(used_steps),
        }
    )
    output_nc.parent.mkdir(parents=True, exist_ok=True)
    combined.to_netcdf(output_nc)
    logger.info("Wrote DA output summary NetCDF {} ({} step(s))", output_nc, len(step_summaries))
    return output_nc


def delete_files(paths: Iterable[Path]) -> tuple[int, int]:
    """Delete files best-effort, returning (count, bytes_freed)."""
    deleted = 0
    bytes_freed = 0
    seen: set[Path] = set()
    for p in paths:
        path = Path(p)
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        try:
            path.unlink()
            deleted += 1
            bytes_freed += size
        except Exception as exc:
            logger.warning("Could not delete {}: {}", path, exc)
    return deleted, bytes_freed


def collect_project_grid_artifacts(project_dir: Path) -> list[Path]:
    """Collect step-level grid artifacts for compact retention cleanup."""
    project_dir = Path(project_dir)
    patterns = (
        "steps/step_*/ensembles/prior/member_*/results/output_grids*.nc",
        "steps/step_*/ensembles/prior/open_loop/results/output_grids*.nc",
        "steps/step_*/ensembles/prior/member_*/results/**/*.tif",
        "steps/step_*/ensembles/prior/open_loop/results/**/*.tif",
        "steps/step_*/ensembles/posterior/member_*/results/output_grids*.nc",
        "steps/step_*/ensembles/posterior/member_*/results/**/*.tif",
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for patt in patterns:
        for path in project_dir.glob(patt):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return files


def collect_subdomain_grid_artifacts(project_dir: Path) -> list[Path]:
    """Collect sub-domain step-level grid artifacts for compact retention."""
    project_dir = Path(project_dir)
    patterns = (
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/member_*/results/output_grids*.nc",
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/open_loop/results/output_grids*.nc",
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/member_*/results/**/*.tif",
        "subdomains/*/projects/*/steps/step_*/ensembles/prior/open_loop/results/**/*.tif",
        "subdomains/*/projects/*/steps/step_*/ensembles/posterior/member_*/results/output_grids*.nc",
        "subdomains/*/projects/*/steps/step_*/ensembles/posterior/member_*/results/**/*.tif",
    )
    files: list[Path] = []
    seen: set[Path] = set()
    for patt in patterns:
        for path in project_dir.glob(patt):
            if path.is_file() and path not in seen:
                files.append(path)
                seen.add(path)
    return files
