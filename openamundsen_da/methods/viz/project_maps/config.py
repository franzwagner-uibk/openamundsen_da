from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openamundsen_da.core.env import _read_yaml_file


SUPPORTED_PANEL_KINDS = {
    "overview",
    "roi",
    "hillshade",
    "dem",
    "svf",
    "srf",
    "landcover",
    "snow_depth",
    "swe",
    "liquid_water_content",
    "fsc",
    "wet_snow",
    "legend",
    "colorbar",
}
SUPPORTED_MODEL_SOURCES = {"open_loop", "ensemble_mean", "increment"}
SUPPORTED_LEGEND_ITEM_KINDS = {"heading", "station_symbol", "source_legend", "scale_bar"}
SUPPORTED_PANEL_LEGEND_LAYOUTS = {"horizontal", "vertical"}
_REMOVED_PANEL_KEYS = {"variable", "metric", "observation", "field", "style", "legend_inside"}
_REMOVED_LAYOUT_KEYS = {"wspace", "hspace"}
_REMOVED_PANEL_KINDS = {"stations", "static_field", "model_field", "increment_field", "observation_field", "roi_overview", "text"}


@dataclass(frozen=True)
class DateSelector:
    explicit: tuple[str, ...] = ()
    assimilation_variables: tuple[str, ...] = ()
    include_first: bool = False
    include_last: bool = False


@dataclass(frozen=True)
class MapDefaults:
    date: str | None = None
    show_colorbar: bool | None = None
    show_scalebar: bool | None = None
    show_grid: bool | None = None
    show_hillshade: bool | None = None


@dataclass(frozen=True)
class LayoutSpec:
    nrows: int
    ncols: int
    width_ratios: tuple[float, ...] = ()
    height_ratios: tuple[float, ...] = ()


@dataclass(frozen=True)
class LegendItemSpec:
    kind: str
    label: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class MapPanelSpec:
    kind: str
    row: int
    col: int
    title: str | None = None
    name: str | None = None
    rowspan: int = 1
    colspan: int = 1
    date: str | None = None
    source: str | None = None
    scale: int | None = None
    label_fit_margin: float | None = None
    roi_label: str | None = None
    lines: tuple[str, ...] = ()
    items: tuple[LegendItemSpec, ...] = ()
    show_colorbar: bool | None = None
    show_scalebar: bool | None = None
    show_grid: bool | None = None
    show_hillshade: bool | None = None
    show_roi: bool | None = None
    show_station_marker: bool | None = None
    show_stations_name: bool | None = None
    show_stations_elev: bool | None = None
    legend: str | None = None


@dataclass(frozen=True)
class MapRecipe:
    name: str
    title: str
    layout: LayoutSpec
    panels: tuple[MapPanelSpec, ...]
    output_name: str | None = None
    defaults: MapDefaults = MapDefaults()

    @property
    def output_stem(self) -> str:
        return str(self.output_name or self.name)


@dataclass(frozen=True)
class ProjectMapsConfig:
    path: Path
    maps: tuple[MapRecipe, ...]

    def all_names(self) -> set[str]:
        return {item.name for item in self.maps}


def default_project_maps_config_path(project_dir: Path) -> Path:
    return Path(project_dir) / "maps.yml"


def _require_mapping(value: object, *, context: str) -> dict[str, object]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a mapping")
    return value


def _require_str(value: object, *, context: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{context} must be a non-empty string")
    return text


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_bool(value: object, *, default: bool | None = None) -> bool | None:
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


def _coerce_float_tuple(value: object, *, context: str) -> tuple[float, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of numbers")
    items: list[float] = []
    for idx, raw in enumerate(value):
        try:
            items.append(float(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}[{idx}] must be numeric") from exc
    return tuple(items)


def _coerce_int(value: object, *, context: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return parsed


def _optional_positive_int(value: object, *, context: str) -> int | None:
    if value is None:
        return None
    return _coerce_int(value, context=context, minimum=1)


def _optional_positive_float(value: object, *, context: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if parsed <= 0.0:
        raise ValueError(f"{context} must be > 0")
    return parsed


def _coerce_str_list(value: object, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    return tuple(_require_str(item, context=f"{context}[]") for item in value)


def _parse_legend_items(value: object, *, context: str) -> tuple[LegendItemSpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    items: list[LegendItemSpec] = []
    for idx, raw in enumerate(value):
        mapping = _require_mapping(raw, context=f"{context}[{idx}]")
        kind = _require_str(mapping.get("kind"), context=f"{context}[{idx}].kind")
        if kind not in SUPPORTED_LEGEND_ITEM_KINDS:
            supported = ", ".join(sorted(SUPPORTED_LEGEND_ITEM_KINDS))
            raise ValueError(f"{context}[{idx}].kind must be one of: {supported}")
        label = _optional_str(mapping.get("label"))
        source = _optional_str(mapping.get("source"))
        if kind == "source_legend" and source is None:
            raise ValueError(f"{context}[{idx}].source is required for source_legend items")
        if kind in {"heading", "station_symbol"} and label is None:
            raise ValueError(f"{context}[{idx}].label is required for {kind}")
        items.append(LegendItemSpec(kind=kind, label=label, source=source))
    return tuple(items)


def _parse_date_selector(value: object, *, context: str) -> DateSelector:
    mapping = _require_mapping(value, context=context)
    return DateSelector(
        explicit=_coerce_str_list(mapping.get("explicit"), context=f"{context}.explicit"),
        assimilation_variables=_coerce_str_list(
            mapping.get("assimilation_variables"),
            context=f"{context}.assimilation_variables",
        ),
        include_first=bool(_coerce_bool(mapping.get("include_first"), default=False)),
        include_last=bool(_coerce_bool(mapping.get("include_last"), default=False)),
    )


def _parse_defaults(value: object, *, context: str) -> MapDefaults:
    mapping = _require_mapping(value, context=context)
    if _REMOVED_LAYOUT_KEYS & set(mapping):
        removed = ", ".join(sorted(_REMOVED_LAYOUT_KEYS & set(mapping)))
        raise ValueError(f"{context} uses removed layout keys: {removed}")
    return MapDefaults(
        date=_optional_str(mapping.get("date")),
        show_colorbar=_coerce_bool(mapping.get("show_colorbar"), default=None),
        show_scalebar=_coerce_bool(mapping.get("show_scalebar"), default=None),
        show_grid=_coerce_bool(mapping.get("show_grid"), default=None),
        show_hillshade=_coerce_bool(mapping.get("show_hillshade"), default=None),
    )


def _parse_layout(value: object, *, context: str) -> LayoutSpec:
    mapping = _require_mapping(value, context=context)
    removed = _REMOVED_LAYOUT_KEYS & set(mapping)
    if removed:
        raise ValueError(f"{context} uses removed layout keys: {', '.join(sorted(removed))}")
    nrows = _coerce_int(mapping.get("nrows"), context=f"{context}.nrows", minimum=1)
    ncols = _coerce_int(mapping.get("ncols"), context=f"{context}.ncols", minimum=1)
    width_ratios = _coerce_float_tuple(mapping.get("width_ratios"), context=f"{context}.width_ratios")
    height_ratios = _coerce_float_tuple(mapping.get("height_ratios"), context=f"{context}.height_ratios")
    if width_ratios and len(width_ratios) != ncols:
        raise ValueError(f"{context}.width_ratios must have length {ncols}")
    if height_ratios and len(height_ratios) != nrows:
        raise ValueError(f"{context}.height_ratios must have length {nrows}")
    return LayoutSpec(nrows=nrows, ncols=ncols, width_ratios=width_ratios, height_ratios=height_ratios)


def _parse_panel(value: object, *, context: str) -> MapPanelSpec:
    mapping = _require_mapping(value, context=context)
    removed_keys = (_REMOVED_PANEL_KEYS | {"show_stations", "annotate_stations"}) & set(mapping)
    if removed_keys:
        raise ValueError(f"{context} uses removed panel keys: {', '.join(sorted(removed_keys))}")

    kind = _require_str(mapping.get("kind"), context=f"{context}.kind")
    if kind in _REMOVED_PANEL_KINDS:
        raise ValueError(f"{context}.kind '{kind}' is no longer supported; use the simplified public panel kinds")
    if kind not in SUPPORTED_PANEL_KINDS:
        supported = ", ".join(sorted(SUPPORTED_PANEL_KINDS))
        raise ValueError(f"{context}.kind must be one of: {supported}")

    panel = MapPanelSpec(
        kind=kind,
        row=_coerce_int(mapping.get("row"), context=f"{context}.row", minimum=0),
        col=_coerce_int(mapping.get("col"), context=f"{context}.col", minimum=0),
        title=_optional_str(mapping.get("title")),
        name=_optional_str(mapping.get("name")),
        rowspan=_coerce_int(mapping.get("rowspan", 1), context=f"{context}.rowspan", minimum=1),
        colspan=_coerce_int(mapping.get("colspan", 1), context=f"{context}.colspan", minimum=1),
        date=_optional_str(mapping.get("date")),
        source=_optional_str(mapping.get("source")),
        scale=_optional_positive_int(mapping.get("scale"), context=f"{context}.scale"),
        label_fit_margin=_optional_positive_float(mapping.get("label_fit_margin"), context=f"{context}.label_fit_margin"),
        roi_label=_optional_str(mapping.get("roi_label")),
        lines=_coerce_str_list(mapping.get("lines"), context=f"{context}.lines"),
        items=_parse_legend_items(mapping.get("items"), context=f"{context}.items"),
        show_colorbar=_coerce_bool(mapping.get("show_colorbar"), default=None),
        show_scalebar=_coerce_bool(mapping.get("show_scalebar"), default=None),
        show_grid=_coerce_bool(mapping.get("show_grid"), default=None),
        show_hillshade=_coerce_bool(mapping.get("show_hillshade"), default=None),
        show_roi=_coerce_bool(mapping.get("show_roi"), default=None),
        show_station_marker=_coerce_bool(mapping.get("show_station_marker"), default=None),
        show_stations_name=_coerce_bool(mapping.get("show_stations_name"), default=None),
        show_stations_elev=_coerce_bool(mapping.get("show_stations_elev"), default=None),
        legend=_optional_str(mapping.get("legend")),
    )

    if panel.legend is not None and panel.legend not in SUPPORTED_PANEL_LEGEND_LAYOUTS:
        supported = ", ".join(sorted(SUPPORTED_PANEL_LEGEND_LAYOUTS))
        raise ValueError(f"{context}.legend must be one of: {supported}")

    if panel.kind in {"snow_depth", "swe", "liquid_water_content"}:
        if panel.source is None:
            raise ValueError(f"{context}.source is required for {panel.kind}")
        if panel.source not in SUPPORTED_MODEL_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_SOURCES))
            raise ValueError(f"{context}.source must be one of: {supported}")
    elif panel.kind == "overview":
        if panel.scale is None:
            raise ValueError(f"{context}.scale is required for overview panels")
    elif panel.kind == "colorbar":
        if panel.source is None:
            raise ValueError(f"{context}.source is required for colorbar panels")
    elif panel.kind == "legend":
        if not (panel.items or panel.source or panel.lines):
            raise ValueError(f"{context} must define items, source, or lines for legend panels")
    elif panel.source is not None and panel.kind not in {"fsc", "wet_snow"}:
        raise ValueError(f"{context}.source is only valid for model panels, legend panels, and colorbar panels")

    return panel


def _validate_panel_layout(recipe: MapRecipe, *, config_path: Path) -> None:
    occupied: set[tuple[int, int]] = set()
    for panel in recipe.panels:
        if panel.row + panel.rowspan > recipe.layout.nrows:
            raise ValueError(f"Panel '{panel.kind}' exceeds layout rows in {config_path}")
        if panel.col + panel.colspan > recipe.layout.ncols:
            raise ValueError(f"Panel '{panel.kind}' exceeds layout columns in {config_path}")
        for row in range(panel.row, panel.row + panel.rowspan):
            for col in range(panel.col, panel.col + panel.colspan):
                cell = (row, col)
                if cell in occupied:
                    raise ValueError(f"Overlapping panel placement at row={row}, col={col} in {config_path}")
                occupied.add(cell)


def _parse_recipe(recipe_name: str, value: object, *, context: str, config_path: Path) -> MapRecipe:
    mapping = _require_mapping(value, context=context)
    title = _require_str(mapping.get("title"), context=f"{context}.title")
    layout = _parse_layout(mapping.get("layout"), context=f"{context}.layout")
    raw_panels = mapping.get("panels")
    if not isinstance(raw_panels, list) or not raw_panels:
        raise ValueError(f"{context}.panels must be a non-empty list")
    recipe = MapRecipe(
        name=recipe_name,
        title=title,
        output_name=_optional_str(mapping.get("output_name")),
        layout=layout,
        defaults=_parse_defaults(mapping.get("defaults"), context=f"{context}.defaults"),
        panels=tuple(_parse_panel(item, context=f"{context}.panels[{idx}]") for idx, item in enumerate(raw_panels)),
    )
    _validate_panel_layout(recipe, config_path=config_path)
    return recipe


def load_project_maps_config(config_path: Path) -> ProjectMapsConfig:
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Project maps config not found: {config_path}")
    cfg = _read_yaml_file(config_path) or {}
    root = _require_mapping(cfg, context=str(config_path))
    raw_maps = root.get("maps")
    if not isinstance(raw_maps, dict) or not raw_maps:
        raise ValueError(f"Missing non-empty 'maps' mapping in {config_path}")
    recipes = tuple(
        _parse_recipe(_require_str(name, context="maps key"), value, context=f"maps.{name}", config_path=config_path)
        for name, value in raw_maps.items()
    )
    stems = [recipe.output_stem for recipe in recipes]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Map recipe output names must be unique in {config_path}")
    return ProjectMapsConfig(path=config_path, maps=recipes)


__all__ = [
    "DateSelector",
    "LayoutSpec",
    "LegendItemSpec",
    "MapDefaults",
    "MapPanelSpec",
    "MapRecipe",
    "ProjectMapsConfig",
    "SUPPORTED_MODEL_SOURCES",
    "SUPPORTED_PANEL_KINDS",
    "default_project_maps_config_path",
    "load_project_maps_config",
]
