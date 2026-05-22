"""Model-only plots for plain openAMUNDSEN sub-domain runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.methods.viz.common import force_figure_text_black, save_figure_png, set_matplotlib_text_black
from openamundsen_da.methods.viz.maps.annotations import panel_date
from openamundsen_da.methods.viz.maps.config import (
    LayoutSpec,
    MapDefaults,
    MapPanelSpec,
    MapRecipe,
    load_project_maps_config,
)
from openamundsen_da.methods.viz.maps.data import (
    ModelFields,
    StaticContext,
    _highest_resolution_dem_path,
    _load_optional_setup_grid,
    _read_dataset_array,
    _read_native_dataset_array,
)
from openamundsen_da.methods.viz.maps.render import RenderRuntimeCache, render_map_recipe
from openamundsen_da.methods.viz.maps.styles import nice_ceiling, require_variable_preset
from openamundsen_da.methods.viz.maps.theme import _MODEL_KIND_TO_VARIABLE
from openamundsen_da.methods.viz.plots.common import format_station_label, pretty_var_title, result_axis_scale
from openamundsen_da.methods.viz.plots.theme import (
    COLOR_DA_OBS,
    COLOR_OPEN_LOOP,
    FIGSIZE_RESULTS,
    FS_TITLE,
    GRID_ALPHA,
    GRID_LS,
    GRID_LW,
    LW_MEMBER,
    LW_OPEN,
)
from openamundsen_da.methods.viz.station_meta import load_setup_snow_depth_station_table, load_setup_station_table
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta
from openamundsen_da.util.landcover_mask import resolve_setup_landcover_grid
from openamundsen_da.util.roi_grid import _read_mask_from_grid, resolve_setup_grid_spec
from openamundsen_da.util.ts import read_timeseries_csv


DEFAULT_MONTHLY_MAP_VARIABLES = ("snowdepth_daily", "swe_daily")
DEFAULT_STATION_VARIABLE = "swe"

_MODEL_VARIABLE_TO_PANEL_KIND = {
    "snowdepth_daily": "snow_depth",
    "swe_daily": "swe",
}
_MODEL_VARIABLE_TO_OUTPUT_TOKEN = {
    "snowdepth_daily": "snowdepth",
    "swe_daily": "swe",
}
_MODEL_MAP_CONFIG_FILENAMES = ("maps.yml", "maps.yaml")
_MODEL_ONLY_UNSUPPORTED_PANEL_KINDS = {
    "fsc",
    "wet_snow",
    "uncertainty",
    "wet_snow_line",
    "wet_snow_elevation_fraction",
}
_TEMPLATE_FIELDS = {
    "subdomain_id",
    "subdomain_label",
}


class _PartialTemplateContext(dict):
    def __missing__(self, key):
        return "{" + str(key) + "}"


@dataclass(frozen=True)
class ModelPlotConfig:
    monthly_map_variables: tuple[str, ...] = DEFAULT_MONTHLY_MAP_VARIABLES
    monthly_date_rule: str = "first_day_of_month"
    station_variable: str = DEFAULT_STATION_VARIABLE
    use_all_qc_flags: bool = True


def load_model_plot_config(path: Path | None) -> ModelPlotConfig:
    """Load optional model-plot config, falling back to defaults."""
    if path is None:
        return ModelPlotConfig()
    cfg = _read_yaml_file(path) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Model plot config must be a mapping: {path}")
    monthly = cfg.get("monthly_maps") or {}
    station = cfg.get("station_comparison") or {}
    if not isinstance(monthly, dict):
        raise ValueError("model plot config monthly_maps must be a mapping")
    if not isinstance(station, dict):
        raise ValueError("model plot config station_comparison must be a mapping")

    raw_vars = monthly.get("variables", DEFAULT_MONTHLY_MAP_VARIABLES)
    if isinstance(raw_vars, str):
        variables = (raw_vars,)
    else:
        variables = tuple(str(item) for item in raw_vars)
    if not variables:
        raise ValueError("monthly_maps.variables must contain at least one variable")
    for variable in variables:
        require_variable_preset(variable)
        if variable not in _MODEL_VARIABLE_TO_PANEL_KIND:
            raise ValueError(f"Model monthly maps do not support variable '{variable}'")
    date_rule = str(monthly.get("date_rule", "first_day_of_month"))
    if date_rule != "first_day_of_month":
        raise ValueError(f"Unsupported monthly_maps.date_rule: {date_rule}")

    return ModelPlotConfig(
        monthly_map_variables=variables,
        monthly_date_rule=date_rule,
        station_variable=str(station.get("variable", DEFAULT_STATION_VARIABLE)),
        use_all_qc_flags=bool(station.get("use_all_qc_flags", True)),
    )


def _config_has_maps(path: Path | None) -> bool:
    if path is None or not Path(path).is_file():
        return False
    cfg = _read_yaml_file(Path(path)) or {}
    return isinstance(cfg, dict) and (isinstance(cfg.get("maps"), dict) or isinstance(cfg.get("model_maps"), dict))


def _default_model_maps_config_path(setup_dir: Path) -> Path | None:
    for filename in _MODEL_MAP_CONFIG_FILENAMES:
        path = Path(setup_dir) / filename
        if path.is_file():
            return path
    return None


def _resolve_model_maps_config_path(setup_dir: Path, config_path: Path | None) -> Path | None:
    if _config_has_maps(config_path):
        return Path(config_path)
    return _default_model_maps_config_path(Path(setup_dir))


def _as_mapping(value: object, *, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _as_bool(value: object, *, default: bool | None = None) -> bool | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value, got {value!r}")


def _as_variables(value: object, *, context: str) -> tuple[str, ...]:
    if isinstance(value, str):
        variables = (value,)
    elif isinstance(value, list):
        variables = tuple(str(item) for item in value)
    else:
        raise ValueError(f"{context} must be a string or list")
    if not variables:
        raise ValueError(f"{context} must contain at least one variable")
    for variable in variables:
        require_variable_preset(variable)
        if variable not in _MODEL_VARIABLE_TO_PANEL_KIND:
            raise ValueError(f"Model maps do not support variable '{variable}'")
    return variables


def _variable_context(variable: str) -> dict[str, str]:
    preset = require_variable_preset(variable)
    return {
        "variable": variable,
        "variable_kind": _MODEL_VARIABLE_TO_PANEL_KIND[variable],
        "variable_token": _MODEL_VARIABLE_TO_OUTPUT_TOKEN[variable],
        "variable_title": preset.title,
    }


def _format_model_map_template(value: str, *, variable: str, **extra: str) -> str:
    context = _PartialTemplateContext(_variable_context(variable))
    context.update(extra)
    return str(value).format_map(context)


def _model_map_dates(raw: dict[str, object], *, start: pd.Timestamp, end: pd.Timestamp, context: str) -> tuple[pd.Timestamp, ...]:
    if "dates" in raw:
        dates = tuple(pd.Timestamp(item).normalize() for item in raw["dates"])
    else:
        date_rule = str(raw.get("date_rule", "first_day_of_month")).strip()
        if date_rule != "first_day_of_month":
            raise ValueError(f"Unsupported {context}.date_rule: {date_rule}")
        dates = first_day_of_month_dates(start, end)
    if not dates:
        raise ValueError(f"{context} resolved no dates")
    return dates


def _template_defaults(raw: dict[str, object]) -> MapDefaults:
    defaults = _as_mapping(raw.get("defaults"), context="model_maps[].defaults")
    return MapDefaults(
        show_scalebar=_as_bool(defaults.get("show_scalebar"), default=True),
        show_grid=_as_bool(defaults.get("show_grid"), default=True),
        show_colorbar=_as_bool(defaults.get("show_colorbar"), default=True),
        show_hillshade=_as_bool(defaults.get("show_hillshade"), default=None),
        hillshade_extent=str(defaults["hillshade_extent"]) if defaults.get("hillshade_extent") is not None else None,
    )


def _expand_model_map_template(
    name: str,
    raw_value: object,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[MapRecipe, ...]:
    raw = _as_mapping(raw_value, context=f"model_maps.{name}")
    kind = str(raw.get("kind", "monthly")).strip()
    if kind != "monthly":
        raise ValueError(f"Unsupported model_maps.{name}.kind: {kind}")
    variables = _as_variables(raw.get("variables", raw.get("variable")), context=f"model_maps.{name}.variables")
    dates = _model_map_dates(raw, start=start, end=end, context=f"model_maps.{name}")
    layout_cfg = _as_mapping(raw.get("layout"), context=f"model_maps.{name}.layout")
    ncols = int(layout_cfg.get("ncols", 3))
    if ncols < 1:
        raise ValueError(f"model_maps.{name}.layout.ncols must be >= 1")
    nrows = int(layout_cfg.get("nrows", int(np.ceil(len(dates) / ncols))))
    if nrows * ncols < len(dates):
        raise ValueError(f"model_maps.{name}.layout has fewer cells than resolved dates")
    panel_cfg = _as_mapping(raw.get("panel"), context=f"model_maps.{name}.panel")
    defaults = _template_defaults(raw)
    output_name_template = str(raw.get("output_name", "{subdomain_id}_{variable_token}_monthly"))
    title_template = str(raw.get("title", "{subdomain_id} monthly {variable_title}"))
    figure_title_template = raw.get("figure_title", title_template)

    recipes: list[MapRecipe] = []
    for variable in variables:
        panel_kind = _MODEL_VARIABLE_TO_PANEL_KIND[variable]
        panels = tuple(
            MapPanelSpec(
                kind=panel_kind,
                row=idx // ncols,
                col=idx % ncols,
                title=_format_model_map_template(
                    str(panel_cfg.get("title", "{date}")),
                    variable=variable,
                    date=date.date().isoformat(),
                ),
                date=date.date().isoformat(),
                source=str(panel_cfg.get("source", "open_loop")),
                show_hillshade=_as_bool(panel_cfg.get("show_hillshade"), default=True),
                show_station_marker=_as_bool(panel_cfg.get("show_station_marker"), default=True),
                show_stations_name=_as_bool(panel_cfg.get("show_stations_name"), default=True),
                show_stations_elev=_as_bool(panel_cfg.get("show_stations_elev"), default=True),
                show_scalebar=_as_bool(panel_cfg.get("show_scalebar"), default=None),
                show_grid=_as_bool(panel_cfg.get("show_grid"), default=None),
                show_colorbar=_as_bool(panel_cfg.get("show_colorbar"), default=None),
                show_roi=_as_bool(panel_cfg.get("show_roi"), default=None),
                hillshade_extent=str(panel_cfg["hillshade_extent"]) if panel_cfg.get("hillshade_extent") is not None else None,
            )
            for idx, date in enumerate(dates)
        )
        variable_name = _MODEL_VARIABLE_TO_OUTPUT_TOKEN[variable]
        recipes.append(
            MapRecipe(
                name=f"{name}_{variable_name}",
                title=_format_model_map_template(title_template, variable=variable),
                output_name=_format_model_map_template(output_name_template, variable=variable),
                figure_title=_format_model_map_template(str(figure_title_template), variable=variable)
                if figure_title_template is not None
                else None,
                output_subdir=_format_model_map_template(str(raw["output_subdir"]), variable=variable)
                if raw.get("output_subdir") is not None
                else None,
                layout=LayoutSpec(nrows=nrows, ncols=ncols),
                defaults=defaults,
                panels=panels,
            )
        )
    return tuple(recipes)


def _load_model_map_recipes(config_path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> tuple[MapRecipe, ...]:
    cfg = _read_yaml_file(config_path) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Model maps config must be a mapping: {config_path}")
    recipes: list[MapRecipe] = []
    if isinstance(cfg.get("model_maps"), dict):
        for name, raw in cfg["model_maps"].items():
            recipes.extend(_expand_model_map_template(str(name), raw, start=start, end=end))
    if isinstance(cfg.get("maps"), dict):
        recipes.extend(load_project_maps_config(config_path).maps)
    if not recipes:
        raise ValueError(f"Missing non-empty maps or model_maps mapping in {config_path}")
    return tuple(recipes)


def first_day_of_month_dates(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    """Return first-of-month dates inside a closed date window."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"end_date {end.date()} is before start_date {start.date()}")
    current = pd.Timestamp(year=start.year, month=start.month, day=1)
    if current < start:
        current = current + pd.offsets.MonthBegin(1)
    dates: list[pd.Timestamp] = []
    while current <= end:
        dates.append(current.normalize())
        current = current + pd.offsets.MonthBegin(1)
    return tuple(dates)


def _selected_subdomains(manifest: SubdomainManifest, subdomains: Iterable[str] | None) -> list[SubdomainMeta]:
    if subdomains is None:
        ids = sorted(manifest.subdomains)
    else:
        ids = [str(item) for item in subdomains]
    missing = [sid for sid in ids if sid not in manifest.subdomains]
    if missing:
        raise KeyError(f"Unknown model subdomain id(s): {', '.join(missing)}")
    selected = [manifest.subdomains[sid] for sid in ids]
    not_success = [sub.id for sub in selected if str(sub.status).lower() != "success"]
    if not_success:
        raise RuntimeError(
            "Model plotting requires successful sub-domain runs; not successful: " + ", ".join(not_success)
        )
    return selected


def _setup_date_window(manifest: SubdomainManifest) -> tuple[pd.Timestamp, pd.Timestamp]:
    setup = _read_yaml_file(manifest.setup_yaml) or {}
    if not isinstance(setup, dict):
        raise ValueError(f"Setup YAML root must be a mapping: {manifest.setup_yaml}")
    try:
        start = pd.Timestamp(setup["start_date"]).normalize()
        end = pd.Timestamp(setup["end_date"]).normalize()
    except KeyError as exc:
        raise ValueError(f"Model plotting requires setup start_date/end_date in {manifest.setup_yaml}") from exc
    return start, end


def _grid_nc_path(sub: SubdomainMeta | Path) -> Path:
    setup_dir = sub.setup_dir if isinstance(sub, SubdomainMeta) else Path(sub)
    return setup_dir / "results" / "grids" / "output_grids.nc"


def _time_dim_for_variable(da: xr.DataArray, variable: str) -> str:
    time_dims = [dim for dim in da.dims if str(dim).startswith("time")]
    if len(time_dims) != 1:
        raise ValueError(f"Expected exactly one time dimension for {variable}, got {da.dims}")
    return time_dims[0]


def _extract_daily_field(ds: xr.Dataset, variable: str, date: pd.Timestamp) -> np.ndarray:
    if variable not in ds:
        raise KeyError(f"Missing variable '{variable}' in model output")
    da = ds[variable]
    time_dim = _time_dim_for_variable(da, variable)
    dates = pd.to_datetime(ds[time_dim].values).normalize()
    matches = np.flatnonzero(dates == date.normalize())
    if matches.size == 0:
        raise KeyError(f"Date {date.date()} not found for {variable}")
    field = da.isel({time_dim: int(matches[0])})
    extra_dims = [dim for dim in field.dims if dim not in {"y", "x"}]
    if extra_dims:
        if extra_dims == ["snow_layer"]:
            field = field.sum(dim="snow_layer", skipna=True)
        else:
            raise ValueError(f"Unsupported non-spatial dimensions for {variable}: {field.dims}")
    return np.asarray(field.transpose("y", "x").values, dtype=float)


def _plain_model_fields_loader(project_dir: Path, variable: str, dates: tuple[pd.Timestamp, ...]) -> list[ModelFields]:
    ds_path = _grid_nc_path(Path(project_dir))
    if not ds_path.is_file():
        raise FileNotFoundError(f"Missing model grid output for {Path(project_dir).name}: {ds_path}")
    with xr.open_dataset(ds_path) as ds:
        fields: list[ModelFields] = []
        for date in dates:
            arr = _extract_daily_field(ds, variable, date)
            fields.append(
                ModelFields(
                    date=pd.Timestamp(date).normalize(),
                    open_loop=arr,
                    ens_mean=arr.copy(),
                    increment=np.zeros_like(arr, dtype=float),
                )
            )
        return fields


def _infer_setup_dir_from_subdomain(subdomain: SubdomainMeta) -> Path | None:
    setup_dir = Path(subdomain.setup_dir)
    if setup_dir.parent.name == "model" and setup_dir.parent.parent.name == "subdomains":
        return setup_dir.parent.parent.parent
    return None


def _load_hillshade_dem_from_specs(
    *,
    model_spec,
    root_setup_dir: Path | None,
    root_setup_yaml: Path | None = None,
) -> tuple[np.ndarray | None, object | None]:
    candidate_specs = []
    if root_setup_dir is not None:
        try:
            candidate_specs.append(resolve_setup_grid_spec(root_setup_dir, setup_yaml=root_setup_yaml))
        except Exception as exc:
            logger.debug("Could not resolve root setup grid spec for hillshade DEM {}: {}", root_setup_dir, exc)
    candidate_specs.append(model_spec)

    seen: set[Path] = set()
    for spec in candidate_specs:
        try:
            dem_path = _highest_resolution_dem_path(spec)
        except Exception as exc:
            logger.debug("Could not resolve hillshade DEM for {}: {}", spec.setup_dir, exc)
            continue
        if dem_path in seen:
            continue
        seen.add(dem_path)
        try:
            native_dem, native_transform, native_crs = _read_native_dataset_array(dem_path)
        except Exception as exc:
            logger.debug("Could not read hillshade DEM {}: {}", dem_path, exc)
            continue
        if model_spec.crs is None or native_crs is None or str(native_crs).lower() == str(model_spec.crs).lower():
            return native_dem, native_transform
        logger.debug(
            "Skipping hillshade DEM {} because CRS {} does not match model CRS {}",
            dem_path,
            native_crs,
            model_spec.crs,
        )
    return None, None


def _load_model_static_context(
    subdomain: SubdomainMeta,
    *,
    setup_dir: Path | None = None,
    setup_yaml: Path | None = None,
) -> StaticContext:
    spec = resolve_setup_grid_spec(subdomain.setup_dir)
    roi_mask = _read_mask_from_grid(subdomain.roi_raster_path, spec)
    roi_gdf = gpd.read_file(subdomain.roi_vector_path)
    if roi_gdf.empty:
        raise ValueError(f"ROI vector has no features: {subdomain.roi_vector_path}")
    if spec.crs and roi_gdf.crs is not None:
        roi_gdf = roi_gdf.to_crs(spec.crs)

    dem = _read_dataset_array(spec.dem_path, shape=roi_mask.shape, transform=spec.transform, crs=spec.crs)
    try:
        landcover_path = resolve_setup_landcover_grid(subdomain.setup_dir)
        landcover = _read_dataset_array(landcover_path, shape=roi_mask.shape, transform=spec.transform, crs=spec.crs)
    except Exception:
        landcover = np.full(roi_mask.shape, np.nan, dtype=float)
    svf = _load_optional_setup_grid(
        subdomain.setup_dir,
        prefix="svf",
        shape=roi_mask.shape,
        transform=spec.transform,
        crs=spec.crs,
    )
    srf = _load_optional_setup_grid(
        subdomain.setup_dir,
        prefix="srf",
        shape=roi_mask.shape,
        transform=spec.transform,
        crs=spec.crs,
    )
    obs_station_dirs = [subdomain.obs_stations_dir] if subdomain.obs_stations_dir is not None else []
    root_setup_dir = Path(setup_dir) if setup_dir is not None else _infer_setup_dir_from_subdomain(subdomain)
    stations = load_setup_snow_depth_station_table(
        root_setup_dir or subdomain.setup_dir,
        obs_stations_dirs=obs_station_dirs,
        grid_crs=spec.crs,
    )

    hillshade_dem, hillshade_transform = _load_hillshade_dem_from_specs(
        model_spec=spec,
        root_setup_dir=root_setup_dir,
        root_setup_yaml=setup_yaml,
    )

    return StaticContext(
        project_dir=subdomain.setup_dir,
        setup_dir=subdomain.setup_dir,
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
        subdomain_gdf=None,
        subdomain_dropped_events=None,
    )


def _shared_model_vmax(subdomain: SubdomainMeta, variable: str, dates: tuple[pd.Timestamp, ...], roi_mask: np.ndarray) -> float:
    preset = require_variable_preset(variable)
    ds_path = _grid_nc_path(subdomain)
    with xr.open_dataset(ds_path) as ds:
        max_value = 0.0
        for date in dates:
            arr = _extract_daily_field(ds, variable, date)
            values = np.where(roi_mask, arr, np.nan)
            finite = values[np.isfinite(values)]
            if finite.size:
                max_value = max(max_value, float(finite.max()))
    return nice_ceiling(max_value, step=preset.max_step, minimum=preset.max_floor)


def _shared_model_vmax_for_recipes(
    subdomain: SubdomainMeta,
    recipes: tuple[MapRecipe, ...],
    roi_mask: np.ndarray,
) -> dict[str, float]:
    dates_by_variable: dict[str, set[pd.Timestamp]] = {}
    for recipe in recipes:
        for panel in recipe.panels:
            if panel.kind not in _MODEL_KIND_TO_VARIABLE:
                continue
            if panel.source in {"increment", "analysis_increment"}:
                continue
            date = panel_date(panel, recipe.defaults)
            if date is None:
                raise ValueError(f"Model map panel '{panel.title or panel.kind}' requires a date")
            variable = _MODEL_KIND_TO_VARIABLE[panel.kind]
            dates_by_variable.setdefault(variable, set()).add(pd.Timestamp(date).normalize())
    return {
        variable: _shared_model_vmax(subdomain, variable, tuple(sorted(dates)), roi_mask)
        for variable, dates in dates_by_variable.items()
    }


def build_monthly_model_map_recipe(
    *,
    subdomain_id: str,
    variable: str,
    dates: tuple[pd.Timestamp, ...],
) -> MapRecipe:
    """Build a model-only monthly map recipe rendered by the existing map stack."""
    if len(dates) != 12:
        raise ValueError(f"Monthly model maps require exactly 12 dates, got {len(dates)}")
    if variable not in _MODEL_VARIABLE_TO_PANEL_KIND:
        raise ValueError(f"Model monthly maps do not support variable '{variable}'")
    preset = require_variable_preset(variable)
    kind = _MODEL_VARIABLE_TO_PANEL_KIND[variable]
    token = _MODEL_VARIABLE_TO_OUTPUT_TOKEN[variable]
    panels = tuple(
        MapPanelSpec(
            kind=kind,
            row=idx // 3,
            col=idx % 3,
            title=pd.Timestamp(date).strftime("%Y-%m-%d"),
            date=pd.Timestamp(date).date().isoformat(),
            source="open_loop",
            show_hillshade=True,
            show_station_marker=True,
            show_stations_name=True,
            show_stations_elev=True,
        )
        for idx, date in enumerate(dates)
    )
    return MapRecipe(
        name=f"{subdomain_id}_{token}_monthly",
        title=f"{subdomain_id} monthly {preset.title}",
        output_name=f"{subdomain_id}_{token}_monthly",
        figure_title=f"{subdomain_id} monthly {preset.title}",
        layout=LayoutSpec(nrows=4, ncols=3),
        defaults=MapDefaults(show_scalebar=True, show_grid=True, show_colorbar=True),
        panels=panels,
    )


def default_monthly_model_map_recipes(
    *,
    subdomain_id: str,
    variables: tuple[str, ...],
    dates: tuple[pd.Timestamp, ...],
) -> tuple[MapRecipe, ...]:
    return tuple(
        build_monthly_model_map_recipe(subdomain_id=subdomain_id, variable=variable, dates=dates)
        for variable in variables
    )


def _format_template(value: str | None, *, subdomain: SubdomainMeta) -> str | None:
    if value is None:
        return None
    try:
        return value.format(
            subdomain_id=subdomain.id,
            subdomain_label=subdomain.label,
        )
    except KeyError as exc:
        supported = ", ".join(sorted(_TEMPLATE_FIELDS))
        raise ValueError(f"Unsupported model map template field {{{exc.args[0]}}}; supported fields: {supported}") from exc


def _format_templates(values: tuple[str, ...], *, subdomain: SubdomainMeta) -> tuple[str, ...]:
    return tuple(str(_format_template(value, subdomain=subdomain) or "") for value in values)


def _format_panel_templates(panel: MapPanelSpec, *, subdomain: SubdomainMeta) -> MapPanelSpec:
    return replace(
        panel,
        title=_format_template(panel.title, subdomain=subdomain),
        name=_format_template(panel.name, subdomain=subdomain),
        roi_label=_format_template(panel.roi_label, subdomain=subdomain),
        lines=_format_templates(panel.lines, subdomain=subdomain),
    )


def _recipe_for_subdomain(recipe: MapRecipe, *, subdomain: SubdomainMeta) -> MapRecipe:
    name = _format_template(recipe.name, subdomain=subdomain)
    output_name = _format_template(recipe.output_name, subdomain=subdomain)
    output_stem_source = recipe.output_name or recipe.name
    if "{subdomain_id}" not in output_stem_source and "{subdomain_label}" not in output_stem_source:
        output_name = f"{subdomain.id}_{output_name or name}"
    return replace(
        recipe,
        name=str(name),
        title=str(_format_template(recipe.title, subdomain=subdomain)),
        output_name=output_name,
        output_subdir=_format_template(recipe.output_subdir, subdomain=subdomain),
        figure_title=_format_template(recipe.figure_title, subdomain=subdomain),
        row_labels=_format_templates(recipe.row_labels, subdomain=subdomain),
        panels=tuple(_format_panel_templates(panel, subdomain=subdomain) for panel in recipe.panels),
    )


def _validate_model_only_recipe(recipe: MapRecipe) -> None:
    for panel in recipe.panels:
        if panel.kind in _MODEL_ONLY_UNSUPPORTED_PANEL_KINDS:
            raise ValueError(
                f"Model-only map recipe '{recipe.name}' uses DA/observation panel kind '{panel.kind}'"
            )
        if panel.kind in _MODEL_KIND_TO_VARIABLE and panel.source != "open_loop":
            raise ValueError(
                f"Model-only map recipe '{recipe.name}' panel '{panel.title or panel.kind}' must use source: open_loop"
            )


def _recipe_output_path(output_dir: Path, recipe: MapRecipe) -> Path:
    out_dir = Path(output_dir)
    if recipe.output_subdir:
        out_dir = out_dir / recipe.output_subdir
    return out_dir / f"{recipe.output_stem}.png"


def _remove_legacy_monthly_map(output_dir: Path, subdomain_id: str) -> None:
    legacy_path = output_dir / f"{subdomain_id}_monthly_snow.png"
    if legacy_path.is_file():
        legacy_path.unlink()
        logger.info("Removed legacy monthly model map {}", legacy_path)


def render_monthly_model_maps(
    *,
    subdomain: SubdomainMeta,
    dates: tuple[pd.Timestamp, ...],
    setup_dir: Path | None = None,
    setup_yaml: Path | None = None,
    variables: tuple[str, ...] = DEFAULT_MONTHLY_MAP_VARIABLES,
    recipes: tuple[MapRecipe, ...] | None = None,
    output_dir: Path,
) -> list[Path]:
    """Render model maps through the existing openAMUNDSEN-DA map renderer."""
    nc_path = _grid_nc_path(subdomain)
    if not nc_path.is_file():
        raise FileNotFoundError(f"Missing model grid output for {subdomain.id}: {nc_path}")

    _remove_legacy_monthly_map(output_dir, subdomain.id)
    context = _load_model_static_context(subdomain, setup_dir=setup_dir, setup_yaml=setup_yaml)
    effective_recipes = (
        tuple(_recipe_for_subdomain(recipe, subdomain=subdomain) for recipe in recipes)
        if recipes is not None
        else default_monthly_model_map_recipes(subdomain_id=subdomain.id, variables=variables, dates=dates)
    )
    for recipe in effective_recipes:
        _validate_model_only_recipe(recipe)
    shared_model_vmax = _shared_model_vmax_for_recipes(subdomain, effective_recipes, context.roi_mask)
    written: list[Path] = []
    for recipe in effective_recipes:
        runtime_cache = RenderRuntimeCache(
            model_loader=_plain_model_fields_loader,
            shared_model_vmax=shared_model_vmax,
        )
        out_path = _recipe_output_path(output_dir, recipe)
        written.append(
            render_map_recipe(
                project_dir=subdomain.setup_dir,
                context=context,
                recipe=recipe,
                output_path=out_path,
                runtime_cache=runtime_cache,
            )
        )
        logger.info("Wrote model map {}", out_path)
    return written


def _read_model_point_daily_first(point_csv: Path, variable: str) -> pd.Series:
    df = read_timeseries_csv(point_csv, "time", [variable])
    if variable not in df:
        raise KeyError(f"Missing model column '{variable}' in {point_csv}")
    work = df[[variable]].dropna().copy()
    work["date"] = work.index.normalize()
    return work.groupby("date", sort=True)[variable].first()


def _clip_daily_series(series: pd.Series, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    dates = pd.to_datetime(series.index).normalize()
    return series[(dates >= start) & (dates <= end)]


def _read_obs_daily(obs_csv: Path, variable: str, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    df = read_timeseries_csv(obs_csv, "time", [variable])
    if variable not in df:
        raise KeyError(f"Missing observation column '{variable}' in {obs_csv}")
    series = df[variable].dropna()
    series = series[(series.index.normalize() >= start) & (series.index.normalize() <= end)]
    work = series.to_frame(variable)
    work["date"] = work.index.normalize()
    return work.groupby("date", sort=True)[variable].first()


def station_swe_comparison_frame(
    *,
    model_point_csv: Path,
    obs_csv: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    variable: str = DEFAULT_STATION_VARIABLE,
) -> pd.DataFrame:
    """Return daily model/observation comparison using first model timestamp per date."""
    model = _clip_daily_series(_read_model_point_daily_first(model_point_csv, variable), start=start, end=end)
    obs = _read_obs_daily(obs_csv, variable, start=start, end=end)
    return pd.DataFrame({"model": model, "obs": obs}).dropna()


def _comparison_stats(frame: pd.DataFrame) -> tuple[float, float, int]:
    diff = frame["model"] - frame["obs"]
    return float(diff.mean()), float(np.sqrt(np.mean(np.square(diff)))), int(len(frame))


def _apply_station_axis_scale(ax, variable: str, model: pd.Series, obs: pd.Series) -> None:
    from matplotlib.ticker import MultipleLocator

    scale = result_axis_scale("station-swe" if variable == "swe" else variable, max(float(model.max()), float(obs.max())))
    if scale is None:
        return
    step, upper = scale
    ax.set_ylim(0.0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.yaxis.set_minor_locator(MultipleLocator(step / 2.0))


def _render_station_swe_comparison_for_point(
    *,
    subdomain: SubdomainMeta,
    setup_dir: Path,
    point_csv: Path,
    stations_df: pd.DataFrame | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    variable: str = DEFAULT_STATION_VARIABLE,
    backend: str = "Agg",
) -> Path | None:
    """Render model-vs-observed station SWE using existing plot styling helpers."""
    del subdomain
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt

    station_id = point_csv.stem.removeprefix("point_")
    obs_csv = setup_dir / "obs" / "stations" / f"{station_id}.csv"
    if not obs_csv.is_file():
        logger.warning("Missing station observation file {}; skipping {}", obs_csv, station_id)
        return None

    model_series = _clip_daily_series(_read_model_point_daily_first(point_csv, variable), start=start, end=end)
    obs_series = _read_obs_daily(obs_csv, variable, start=start, end=end)
    frame = station_swe_comparison_frame(
        model_point_csv=point_csv,
        obs_csv=obs_csv,
        start=start,
        end=end,
        variable=variable,
    )
    if frame.empty:
        logger.warning("No overlapping model/observation SWE data for {}; skipping", station_id)
        return None
    bias, rmse, n = _comparison_stats(frame)

    fig, ax = plt.subplots(figsize=FIGSIZE_RESULTS)
    ax.plot(model_series.index, model_series, color=COLOR_OPEN_LOOP, lw=LW_OPEN, label="model")
    ax.plot(obs_series.index, obs_series, color=COLOR_DA_OBS, lw=LW_MEMBER, label="station observation")
    ax.set_xlabel("Time")
    ax.set_ylabel(pretty_var_title(variable))
    ax.grid(True, linestyle=GRID_LS, linewidth=GRID_LW, alpha=GRID_ALPHA)
    _apply_station_axis_scale(ax, variable, model_series, obs_series)
    ax.legend()

    title_name, alt, _label = format_station_label(station_id, stations_df, fallback=station_id)
    alt_txt = f" ({int(alt)} m)" if alt is not None else ""
    fig.text(
        0.5,
        0.95,
        f"{title_name}{alt_txt} | {pretty_var_title(variable)} | bias={bias:.1f}, RMSE={rmse:.1f}, N={n}",
        ha="center",
        va="top",
        fontsize=FS_TITLE,
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.92))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{station_id}_swe_comparison.png"
    force_figure_text_black(fig, [ax])
    save_figure_png(fig, out_path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    logger.info("Wrote station comparison {}", out_path)
    return out_path


def render_station_swe_comparison(
    *,
    subdomain: SubdomainMeta,
    setup_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output_dir: Path,
    variable: str = DEFAULT_STATION_VARIABLE,
    backend: str = "Agg",
) -> list[Path]:
    """Render model-vs-observed station SWE for all point outputs in one sub-domain."""
    point_files = sorted((subdomain.setup_dir / "results").glob("point_*.csv"))
    if not point_files:
        logger.warning("No point outputs found for {}; skipping station comparison", subdomain.id)
        return []
    stations_df = load_setup_station_table(setup_dir)
    written: list[Path] = []
    for point_csv in point_files:
        station_plot = _render_station_swe_comparison_for_point(
            subdomain=subdomain,
            setup_dir=setup_dir,
            point_csv=point_csv,
            stations_df=stations_df,
            start=start,
            end=end,
            output_dir=output_dir,
            variable=variable,
            backend=backend,
        )
        if station_plot is not None:
            written.append(station_plot)
    return written


def plot_model_subdomains(
    *,
    manifest_path: Path,
    subdomains: Iterable[str] | None = None,
    config_path: Path | None = None,
    backend: str = "Agg",
) -> list[Path]:
    """Render monthly model maps and station SWE comparisons for model sub-domain runs."""
    manifest = SubdomainManifest.load(manifest_path)
    if str(manifest.run_mode).lower() != "model":
        raise ValueError(f"Manifest at {manifest_path} is not marked as run_mode='model'.")
    selected = _selected_subdomains(manifest, subdomains)
    start, end = _setup_date_window(manifest)
    dates = first_day_of_month_dates(start, end)
    if not dates:
        raise ValueError(f"No first-of-month dates in setup window {start.date()} to {end.date()}")

    config = load_model_plot_config(config_path)
    maps_config_path = _resolve_model_maps_config_path(manifest.setup_dir, config_path)
    map_recipes = (
        _load_model_map_recipes(maps_config_path, start=start, end=end)
        if maps_config_path is not None
        else None
    )
    if maps_config_path is not None:
        logger.info("Using model maps config {}", maps_config_path)
    out_root = manifest.subdomain_root / "results"
    monthly_dir = out_root / "maps" / "monthly"
    stations_dir = out_root / "plots" / "stations"

    written: list[Path] = []
    for sub in selected:
        written.extend(
            render_monthly_model_maps(
                subdomain=sub,
                setup_dir=manifest.setup_dir,
                setup_yaml=manifest.setup_yaml,
                dates=dates,
                variables=config.monthly_map_variables,
                recipes=map_recipes,
                output_dir=monthly_dir,
            )
        )
        written.extend(
            render_station_swe_comparison(
                subdomain=sub,
                setup_dir=manifest.setup_dir,
                start=start,
                end=end,
                output_dir=stations_dir,
                variable=config.station_variable,
                backend=backend,
            )
        )
    logger.info("Model plotting complete -> {} output(s)", len(written))
    return written
