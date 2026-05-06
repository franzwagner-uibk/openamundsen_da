from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio import features
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape as shapely_shape

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    abspath_relative_to,
    find_project_yaml,
    infer_setup_dir_from_project,
    project_da_output_grids_path,
)
from openamundsen_da.methods.pf.fraction_support import (
    _fallback_observation_dir_for_project,
    _source_dataset_ref,
    _source_path_from_token,
)
from openamundsen_da.methods.viz.maps.config import DateSelector
from openamundsen_da.methods.viz.station_meta import load_setup_station_table
from openamundsen_da.observer.class_config import load_observation_classes, load_wetsnow_classes
from openamundsen_da.subdomain.manifest import SubdomainManifest
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.landcover_mask import resolve_setup_landcover_grid
from openamundsen_da.util.roi_grid import _find_grid_file, load_setup_roi_mask, resolve_setup_grid_spec


@dataclass(frozen=True)
class StaticContext:
    project_dir: Path
    setup_dir: Path
    spec: object
    roi_mask: np.ndarray
    roi_gdf: gpd.GeoDataFrame
    dem: np.ndarray
    landcover: np.ndarray
    svf: np.ndarray | None
    srf: np.ndarray | None
    stations: pd.DataFrame | None
    hillshade_dem: np.ndarray | None = None
    hillshade_transform: object | None = None
    subdomain_gdf: gpd.GeoDataFrame | None = None
    subdomain_dropped_events: pd.DataFrame | None = None


@dataclass(frozen=True)
class ModelFields:
    date: pd.Timestamp
    open_loop: np.ndarray
    ens_mean: np.ndarray
    increment: np.ndarray
    analysis_mean: np.ndarray | None = None
    analysis_increment: np.ndarray | None = None


@dataclass(frozen=True)
class ObservationScene:
    date: pd.Timestamp
    observation: str
    array: np.ndarray
    transform: rasterio.Affine
    bounds: tuple[float, float, float, float]
    coverage_fraction: float
    roi_mask: np.ndarray
    invalid_mask: np.ndarray | None = None


def _normalize_dates(values: object) -> tuple[pd.Timestamp, ...]:
    idx = pd.to_datetime(values)
    return tuple(pd.Timestamp(value).normalize() for value in idx)


def _metric_var(metric: str, variable: str) -> str:
    return f"{metric}_{variable}"


def _read_dataset_array(path: Path, *, shape: tuple[int, int], transform, crs: str | None) -> np.ndarray:
    with rasterio.open(path) as src:
        if (
            src.crs is not None
            and crs is not None
            and str(src.crs).lower() == str(crs).lower()
            and int(src.height) == int(shape[0])
            and int(src.width) == int(shape[1])
            and tuple(src.transform) == tuple(transform)
        ):
            arr = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None:
                arr[arr == nodata] = np.nan
            return arr

        dst = np.full(shape, np.nan, dtype=float)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
            dst_nodata=np.nan,
        )
        return dst


def _read_native_dataset_array(path: Path) -> tuple[np.ndarray, object, str | None]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        crs = src.crs.to_string() if src.crs else None
        return arr, src.transform, crs


def _highest_resolution_dem_path(spec) -> Path:
    candidates = {spec.dem_path}
    prefix = f"dem_{spec.domain}_"
    for pattern in ("*.asc", "*.tif", "*.tiff"):
        for path in spec.grids_dir.glob(pattern):
            if path.is_file() and path.stem.startswith(prefix):
                candidates.add(path)

    def _resolution_key(path: Path) -> tuple[int, int | str]:
        stem = path.stem
        if not stem.startswith(prefix):
            return (1, path.name)
        suffix = stem[len(prefix) :]
        match = re.match(r"(?P<res>\d+)(?:_.*)?$", suffix)
        if match is None:
            return (1, path.name)
        return (0, int(match.group("res")))

    return sorted(candidates, key=_resolution_key)[0]


def _load_optional_setup_grid(
    setup_dir: Path,
    *,
    prefix: str,
    shape: tuple[int, int],
    transform,
    crs: str | None,
) -> np.ndarray | None:
    spec = resolve_setup_grid_spec(setup_dir)
    try:
        grid_path = _find_grid_file(spec.grids_dir, prefix, spec.domain, spec.resolution)
    except FileNotFoundError:
        return None
    return _read_dataset_array(grid_path, shape=shape, transform=transform, crs=crs)


def _load_subdomain_regions(project_dir: Path, setup_dir: Path, crs: str | None) -> gpd.GeoDataFrame | None:
    manifest_path = project_dir / "subdomains" / "subdomain_manifest.json"
    if not manifest_path.is_file():
        return None

    manifest = SubdomainManifest.load(manifest_path)
    if manifest.run_mode != "subdomain":
        return None

    regions_path = Path(manifest.regions_path)
    if not regions_path.is_file():
        setup_relative = setup_dir / "env" / regions_path.name
        if setup_relative.is_file():
            regions_path = setup_relative
    if not regions_path.is_file():
        return None

    regions = gpd.read_file(regions_path)
    if regions.empty:
        return None
    regions = regions.loc[regions.geometry.notna()].copy()
    regions = regions.loc[~regions.geometry.is_empty].copy()
    if regions.empty:
        return None
    if crs and regions.crs is not None:
        regions = regions.to_crs(crs)
    if "subdomain_id" not in regions.columns:
        if manifest.id_field in regions.columns:
            regions["subdomain_id"] = regions[manifest.id_field].astype(str)
        elif "id" in regions.columns:
            regions["subdomain_id"] = regions["id"].astype(str)
    return regions


def _load_subdomain_dropped_events(project_dir: Path) -> pd.DataFrame | None:
    manifest_path = project_dir / "subdomains" / "subdomain_manifest.json"
    if not manifest_path.is_file():
        return None

    candidates = [project_dir / "results" / "subdomain_dropped_events.csv"]
    candidates.extend(sorted((project_dir / "subdomains").glob("*/subdomain_dropped_events.csv")))
    frames = []
    for path in candidates:
        if not path.is_file():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    required = {"subdomain_id", "date", "variable"}
    if not required.issubset(out.columns):
        return None
    out = out.copy()
    out["subdomain_id"] = out["subdomain_id"].astype(str)
    out["variable"] = out["variable"].astype(str).str.strip().str.lower()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "variable", "subdomain_id"])
    if out.empty:
        return None
    subset = ["subdomain_id", "date", "variable"]
    if "reason" in out.columns:
        subset.append("reason")
    return out.drop_duplicates(subset=subset)


@lru_cache(maxsize=16)
def _load_static_context_cached(project_dir_str: str) -> StaticContext:
    project_dir = Path(project_dir_str)
    setup_dir = infer_setup_dir_from_project(project_dir)
    roi_mask, spec, _ = load_setup_roi_mask(setup_dir, ensure_grid=True)
    roi_vector_path = None
    from openamundsen_da.util.roi_grid import discover_setup_roi_vector

    roi_vector_path = discover_setup_roi_vector(setup_dir)
    if roi_vector_path is not None and roi_vector_path.is_file():
        roi_gdf = gpd.read_file(roi_vector_path)
        if roi_gdf.empty:
            raise ValueError(f"ROI vector has no features: {roi_vector_path}")
        if spec.crs and roi_gdf.crs is not None:
            roi_gdf = roi_gdf.to_crs(spec.crs)
    else:
        shapes = []
        for geom, value in features.shapes(
            roi_mask.astype("uint8"),
            mask=roi_mask.astype(bool),
            transform=spec.transform,
        ):
            if int(value) == 1:
                shapes.append(geom)
        if not shapes:
            raise ValueError(f"Could not derive ROI geometry from mask for {setup_dir}")
        roi_gdf = gpd.GeoDataFrame(geometry=[shapely_shape(geom) for geom in shapes], crs=spec.crs)

    dem = _read_dataset_array(spec.dem_path, shape=roi_mask.shape, transform=spec.transform, crs=spec.crs)
    hillshade_dem = None
    hillshade_transform = None
    highest_dem_path = _highest_resolution_dem_path(spec)
    try:
        native_dem, native_transform, native_crs = _read_native_dataset_array(highest_dem_path)
        if spec.crs is None or native_crs is None or str(native_crs).lower() == str(spec.crs).lower():
            hillshade_dem = native_dem
            hillshade_transform = native_transform
    except Exception:
        hillshade_dem = None
        hillshade_transform = None
    landcover_path = resolve_setup_landcover_grid(setup_dir)
    landcover = _read_dataset_array(landcover_path, shape=roi_mask.shape, transform=spec.transform, crs=spec.crs)
    svf = _load_optional_setup_grid(
        setup_dir,
        prefix="svf",
        shape=roi_mask.shape,
        transform=spec.transform,
        crs=spec.crs,
    )
    srf = _load_optional_setup_grid(
        setup_dir,
        prefix="srf",
        shape=roi_mask.shape,
        transform=spec.transform,
        crs=spec.crs,
    )
    stations = load_setup_station_table(setup_dir)
    subdomain_gdf = _load_subdomain_regions(project_dir, setup_dir, spec.crs)
    subdomain_dropped_events = _load_subdomain_dropped_events(project_dir)
    return StaticContext(
        project_dir=project_dir,
        setup_dir=setup_dir,
        spec=spec,
        roi_mask=roi_mask,
        roi_gdf=roi_gdf,
        dem=dem,
        landcover=landcover,
        svf=svf,
        srf=srf,
        stations=stations,
        hillshade_dem=hillshade_dem,
        hillshade_transform=hillshade_transform,
        subdomain_gdf=subdomain_gdf,
        subdomain_dropped_events=subdomain_dropped_events,
    )


def load_static_context(project_dir: Path) -> StaticContext:
    return _load_static_context_cached(str(Path(project_dir).resolve()))


@lru_cache(maxsize=8)
def _load_da_dataset(project_dir: Path) -> xr.Dataset:
    ds_path = project_da_output_grids_path(project_dir)
    if not ds_path.is_file():
        raise FileNotFoundError(f"Compact DA output grid not found: {ds_path}")
    return xr.load_dataset(ds_path)


def available_model_dates(project_dir: Path, variable: str) -> tuple[pd.Timestamp, ...]:
    ds = _load_da_dataset(project_dir)
    try:
        var_name = _metric_var("open_loop", variable)
        if var_name not in ds:
            raise KeyError(f"Missing variable '{var_name}' in {project_da_output_grids_path(project_dir)}")
        da = ds[var_name]
        time_dims = [dim for dim in da.dims if str(dim).startswith("time")]
        if len(time_dims) != 1:
            raise ValueError(f"Expected exactly one time dimension for '{var_name}', got {da.dims}")
        return _normalize_dates(ds[time_dims[0]].values)
    finally:
        ds.close()


def _resolve_candidate_dates(
    selector: DateSelector,
    *,
    candidate_dates: tuple[pd.Timestamp, ...],
    project_dir: Path,
) -> tuple[pd.Timestamp, ...]:
    candidate_set = {date.normalize() for date in candidate_dates}
    selected: set[pd.Timestamp] = set()

    for text in selector.explicit:
        parsed = pd.Timestamp(text).normalize()
        if parsed not in candidate_set:
            raise KeyError(f"Requested date {parsed.date()} is not available in candidate pool")
        selected.add(parsed)

    if selector.assimilation_variables:
        allowed = {str(item).strip().lower() for item in selector.assimilation_variables}
        for event in load_assimilation_events(project_dir):
            if str(event.variable).strip().lower() in allowed:
                parsed = pd.Timestamp(event.date).normalize()
                if parsed in candidate_set:
                    selected.add(parsed)

    if selector.include_first and candidate_dates:
        selected.add(candidate_dates[0].normalize())
    if selector.include_last and candidate_dates:
        selected.add(candidate_dates[-1].normalize())

    if not selected:
        raise ValueError("Date selector resolved to an empty set")
    return tuple(sorted(selected))


def resolve_comparison_dates(project_dir: Path, variable: str, selector: DateSelector) -> tuple[pd.Timestamp, ...]:
    candidate_dates = available_model_dates(project_dir, variable)
    return _resolve_candidate_dates(selector, candidate_dates=candidate_dates, project_dir=project_dir)


@lru_cache(maxsize=16)
def _load_summary(project_dir: Path, observation: str) -> pd.DataFrame:
    filename = "scf_summary.csv" if observation == "scf" else "wet_snow_summary.csv"
    from openamundsen_da.observer.summary_paths import resolve_fraction_summary_path

    summary_path = resolve_fraction_summary_path(infer_setup_dir_from_project(project_dir), Path(project_dir), filename)
    if not summary_path.is_file():
        raise FileNotFoundError(f"Observation summary not found: {summary_path}")
    df = pd.read_csv(summary_path)
    if "date" not in df.columns:
        raise ValueError(f"Observation summary missing 'date' column: {summary_path}")
    if "source" not in df.columns:
        raise ValueError(f"Observation summary missing 'source' column: {summary_path}")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


def resolve_observation_context_dates(
    project_dir: Path,
    *,
    model_variable: str,
    observation: str,
    selector: DateSelector,
) -> tuple[pd.Timestamp, ...]:
    model_dates = set(available_model_dates(project_dir, model_variable))
    summary = _load_summary(project_dir, observation)
    obs_dates = {pd.Timestamp(value).normalize() for value in summary["date"].tolist()}
    candidate = tuple(sorted(model_dates & obs_dates))
    if not candidate:
        raise ValueError(
            f"No common dates between compact grids for '{model_variable}' and observation summary '{observation}'"
        )
    return _resolve_candidate_dates(selector, candidate_dates=candidate, project_dir=project_dir)


def _extract_spatial_field(da: xr.DataArray, *, time_dim: str, idx: int, variable_name: str, ds_path: Path) -> np.ndarray:
    field = da.isel({time_dim: idx})
    extra_dims = [dim for dim in field.dims if dim not in {"y", "x"}]
    if extra_dims:
        if extra_dims == ["snow_layer"]:
            field = field.sum(dim="snow_layer", skipna=True)
        else:
            raise ValueError(
                f"Unsupported non-spatial dimensions for '{variable_name}' in {ds_path}: {field.dims}"
            )
    if set(field.dims) != {"y", "x"}:
        raise ValueError(
            f"Expected '{variable_name}' to resolve to y/x grid after selection in {ds_path}, got {field.dims}"
        )
    field = field.transpose("y", "x")
    return np.asarray(field.values, dtype=float)


def load_model_fields(project_dir: Path, variable: str, dates: tuple[pd.Timestamp, ...]) -> list[ModelFields]:
    ds_path = project_da_output_grids_path(project_dir)
    with xr.open_dataset(ds_path) as ds:
        open_name = _metric_var("open_loop", variable)
        mean_name = _metric_var("ens_mean", variable)
        inc_name = _metric_var("increment", variable)
        analysis_mean_name = _metric_var("analysis_mean", variable)
        analysis_inc_name = _metric_var("analysis_increment", variable)
        missing = [name for name in (open_name, mean_name, inc_name) if name not in ds]
        if missing:
            raise KeyError(f"Missing required variables in {ds_path}: {', '.join(missing)}")
        time_dims = [dim for dim in ds[open_name].dims if str(dim).startswith("time")]
        if len(time_dims) != 1:
            raise ValueError(f"Expected exactly one time dimension for '{open_name}', got {ds[open_name].dims}")
        time_dim = time_dims[0]
        available = _normalize_dates(ds[time_dim].values)
        index_by_date = {date: idx for idx, date in enumerate(available)}
        items: list[ModelFields] = []
        for date in dates:
            if date.normalize() not in index_by_date:
                raise KeyError(f"Date {date.date()} not found in {ds_path}")
            idx = index_by_date[date.normalize()]
            items.append(
                ModelFields(
                    date=date.normalize(),
                    open_loop=_extract_spatial_field(
                        ds[open_name], time_dim=time_dim, idx=idx, variable_name=open_name, ds_path=ds_path
                    ),
                    ens_mean=_extract_spatial_field(
                        ds[mean_name], time_dim=time_dim, idx=idx, variable_name=mean_name, ds_path=ds_path
                    ),
                    increment=_extract_spatial_field(
                        ds[inc_name], time_dim=time_dim, idx=idx, variable_name=inc_name, ds_path=ds_path
                    ),
                    analysis_mean=(
                        _extract_spatial_field(
                            ds[analysis_mean_name],
                            time_dim=time_dim,
                            idx=idx,
                            variable_name=analysis_mean_name,
                            ds_path=ds_path,
                        )
                        if analysis_mean_name in ds
                        else None
                    ),
                    analysis_increment=(
                        _extract_spatial_field(
                            ds[analysis_inc_name],
                            time_dim=time_dim,
                            idx=idx,
                            variable_name=analysis_inc_name,
                            ds_path=ds_path,
                        )
                        if analysis_inc_name in ds
                        else None
                    ),
                )
            )
        return items


def _split_source_tokens(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    tokens = [token.strip() for token in re.split(r"[;,]", text) if token.strip()]
    return tuple(sorted(set(tokens)))


def _observation_dir(project_dir: Path, observation: str) -> Path:
    setup_dir = infer_setup_dir_from_project(project_dir)
    project_cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    if not isinstance(project_cfg, dict):
        raise ValueError(f"Expected project YAML mapping in {project_dir}")
    obs_cfg = project_cfg.get("obs")
    if not isinstance(obs_cfg, dict):
        raise ValueError(f"Missing 'obs' block in {find_project_yaml(project_dir)}")
    key = "snowcover" if observation == "scf" else "wetsnow"
    entry = obs_cfg.get(key)
    if not isinstance(entry, dict):
        raise ValueError(f"Missing 'obs.{key}.dir' in {find_project_yaml(project_dir)}")
    raw_dir = entry.get("dir")
    if raw_dir is None:
        raise ValueError(f"Missing 'obs.{key}.dir' in {find_project_yaml(project_dir)}")
    return Path(abspath_relative_to(setup_dir, Path(str(raw_dir))))


def _observation_source_paths(project_dir: Path, observation: str, date: pd.Timestamp) -> tuple[Path, ...]:
    summary = _load_summary(project_dir, observation)
    rows = summary[summary["date"] == pd.Timestamp(date).normalize()]
    if rows.empty:
        raise KeyError(f"No {observation} observation summary row for {pd.Timestamp(date).date()}")
    source_tokens: list[str] = []
    for _, row in rows.iterrows():
        source_tokens.extend(_split_source_tokens(row.get("source")))
    source_tokens = sorted(set(source_tokens))
    if not source_tokens:
        raise FileNotFoundError(f"No source raster(s) listed for {observation} on {pd.Timestamp(date).date()}")

    obs_dir = _observation_dir(project_dir, observation)
    fallback_dir = _fallback_observation_dir_for_project(project_dir, observable=observation)
    source_paths = []
    for token in source_tokens:
        source_path = _source_path_from_token(obs_dir, token, fallback_dir=fallback_dir)
        source_paths.append(source_path)
    return tuple(source_paths)


def _roi_geometry_for_context(context: StaticContext) -> gpd.GeoDataFrame:
    roi_geom = context.roi_gdf
    if context.spec.crs and roi_geom.crs is not None:
        roi_geom = roi_geom.to_crs(context.spec.crs)
    return roi_geom


def _merge_observation_rasters(
    source_paths: tuple[Path, ...],
    *,
    context: StaticContext,
    roi_geom: gpd.GeoDataFrame,
    observation: str,
) -> tuple[np.ndarray, rasterio.Affine, tuple[float, float, float, float], np.ndarray]:
    merge_bounds = tuple(float(value) for value in roi_geom.total_bounds)

    with ExitStack() as stack:
        datasets = []
        for source_path in source_paths:
            dataset_ref = _source_dataset_ref(source_path, token=source_path.name, observable=observation)
            src = stack.enter_context(rasterio.open(dataset_ref))
            if src.crs is not None and context.spec.crs is not None and str(src.crs).lower() != str(context.spec.crs).lower():
                src = stack.enter_context(
                    WarpedVRT(src, crs=context.spec.crs, resampling=Resampling.nearest)
                )
            datasets.append(src)
        mosaic, transform = merge(
            datasets,
            bounds=merge_bounds,
            nodata=np.nan,
            method="first",
        )
    arr = np.asarray(mosaic[0], dtype=float)
    roi_mask = features.geometry_mask(
        roi_geom.geometry,
        out_shape=arr.shape,
        transform=transform,
        invert=True,
    )
    left, bottom, right, top = rasterio.transform.array_bounds(arr.shape[0], arr.shape[1], transform)
    return arr, transform, (left, right, bottom, top), roi_mask


def _mask_observation_array(
    arr: np.ndarray,
    *,
    project_dir: Path,
    observation: str,
    roi_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    invalid_mask = np.zeros(arr.shape, dtype=bool)
    if observation == "scf":
        classes = load_observation_classes(project_dir, obs_key="snowcover")
        invalid_classes = set(classes.get("cloud", [])) | set(classes.get("water", [])) | set(classes.get("nodata", []))
        if invalid_classes:
            invalid_mask = np.isin(arr, list(invalid_classes))
            arr[invalid_mask] = np.nan
    else:
        _wet, valid, _exclude = load_wetsnow_classes(project_dir)
        if valid:
            invalid_mask = ~np.isin(arr, list(valid))
            arr[invalid_mask] = np.nan

    arr[~roi_mask] = np.nan
    invalid_mask &= roi_mask
    return arr, invalid_mask


def _observation_uncertainty_path(source_path: Path) -> Path:
    if source_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(
            f"Uncertainty map panels currently require GeoTIFF observation sources, got: {source_path}"
        )
    uncertainty_path = source_path.parent / f"{source_path.stem}_uncertainty.tif"
    if not uncertainty_path.is_file():
        raise FileNotFoundError(f"Observation uncertainty raster not found: {uncertainty_path}")
    return uncertainty_path


def _align_uncertainty_array(
    arr: np.ndarray,
    transform: rasterio.Affine,
    *,
    scene: ObservationScene,
    context: StaticContext,
) -> np.ndarray:
    if arr.shape == scene.array.shape and tuple(transform) == tuple(scene.transform):
        return arr
    if context.spec.crs is None:
        raise ValueError("Cannot align uncertainty raster to observation grid because the setup CRS is undefined")
    dst = np.full(scene.array.shape, np.nan, dtype=float)
    reproject(
        source=arr,
        destination=dst,
        src_transform=transform,
        src_crs=context.spec.crs,
        dst_transform=scene.transform,
        dst_crs=context.spec.crs,
        resampling=Resampling.nearest,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    return dst


def load_observation_scene(project_dir: Path, context: StaticContext, *, observation: str, date: pd.Timestamp) -> ObservationScene:
    source_paths = _observation_source_paths(project_dir, observation, date)
    roi_geom = _roi_geometry_for_context(context)
    arr, transform, bounds, roi_mask = _merge_observation_rasters(
        source_paths,
        context=context,
        roi_geom=roi_geom,
        observation=observation,
    )
    arr, invalid_mask = _mask_observation_array(
        arr,
        project_dir=project_dir,
        observation=observation,
        roi_mask=roi_mask,
    )
    coverage_fraction = float(np.isfinite(arr).sum()) / float(max(1, roi_mask.sum()))
    return ObservationScene(
        date=pd.Timestamp(date).normalize(),
        observation=observation,
        array=arr,
        transform=transform,
        bounds=bounds,
        coverage_fraction=coverage_fraction,
        roi_mask=roi_mask,
        invalid_mask=invalid_mask,
    )


def load_observation_uncertainty_scene(
    project_dir: Path,
    context: StaticContext,
    *,
    observation: str,
    date: pd.Timestamp,
) -> ObservationScene:
    observation_scene = load_observation_scene(project_dir, context, observation=observation, date=date)
    uncertainty_paths = tuple(
        _observation_uncertainty_path(source_path)
        for source_path in _observation_source_paths(project_dir, observation, date)
    )
    roi_geom = _roi_geometry_for_context(context)
    arr, transform, _bounds, _roi_mask = _merge_observation_rasters(
        uncertainty_paths,
        context=context,
        roi_geom=roi_geom,
        observation=observation,
    )
    arr = _align_uncertainty_array(arr, transform, scene=observation_scene, context=context).astype(float, copy=True)
    valid_observation = observation_scene.roi_mask & np.isfinite(observation_scene.array)
    missing_on_valid = valid_observation & ~np.isfinite(arr)
    if np.any(missing_on_valid):
        raise ValueError(
            f"Uncertainty raster has missing values on valid {observation} pixels for {pd.Timestamp(date).date()}"
        )
    finite_on_valid = valid_observation & np.isfinite(arr)
    outside_range = finite_on_valid & ((arr < 0.0) | (arr > 100.0))
    if np.any(outside_range):
        raise ValueError(
            f"Uncertainty raster values must be within 0..100 for {observation} on {pd.Timestamp(date).date()}"
        )
    arr[~valid_observation] = np.nan
    invalid_mask = observation_scene.roi_mask & ~valid_observation
    coverage_fraction = float(np.isfinite(arr).sum()) / float(max(1, observation_scene.roi_mask.sum()))
    return ObservationScene(
        date=pd.Timestamp(date).normalize(),
        observation=observation,
        array=arr,
        transform=observation_scene.transform,
        bounds=observation_scene.bounds,
        coverage_fraction=coverage_fraction,
        roi_mask=observation_scene.roi_mask,
        invalid_mask=invalid_mask,
    )
