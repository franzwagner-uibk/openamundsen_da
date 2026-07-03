from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from openamundsen_da.core.env import _read_yaml_file


SUPPORTED_PANEL_KINDS = {
    "overview",
    "roi",
    "hillshade",
    "dem",
    "aspect",
    "svf",
    "srf",
    "landcover",
    "snow_depth",
    "swe",
    "liquid_water_content",
    "fsc",
    "wet_snow",
    "uncertainty",
    "wet_snow_line",
    "wet_snow_elevation_fraction",
    "legend",
    "colorbar",
}
SUPPORTED_MODEL_SOURCES = {"open_loop", "ensemble_mean", "analysis_mean", "increment", "analysis_increment"}
SUPPORTED_FSC_SOURCES = {
    "open_loop",
    "ensemble_mean",
    "open_loop_binary",
    "prior_probability",
    "posterior_probability",
}
SUPPORTED_WET_SNOW_SOURCES = {
    "open_loop",
    "ensemble_mean",
    "prior_probability",
    "posterior_probability",
}
SUPPORTED_WET_SNOW_LINE_SOURCES = {
    "open_loop",
    "prior_probability",
    "posterior",
    "posterior_probability",
}
SUPPORTED_WET_SNOW_ELEVATION_FRACTION_SOURCES = {
    "open_loop",
    "prior_probability",
    "posterior",
    "posterior_probability",
}
SUPPORTED_UNCERTAINTY_OBSERVATIONS = {"scf", "wet_snow"}
SUPPORTED_LEGEND_ITEM_KINDS = {"heading", "station_symbol", "source_legend", "scale_bar"}
SUPPORTED_LEGEND_ITEM_PLACEMENTS = {"below", "inside"}
SUPPORTED_LEGEND_ITEM_ANCHORS = {"top_left", "top_right", "bottom_left", "bottom_right"}
SUPPORTED_PANEL_LEGEND_LAYOUTS = {"horizontal", "vertical"}
SUPPORTED_HILLSHADE_EXTENTS = {"full", "roi"}
SUPPORTED_LANDCOVER_GROUPINGS = {"native", "broad", "rofental_manuscript"}
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
    hillshade_extent: str | None = None


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
    placement: str | None = None
    anchor: str | None = None


@dataclass(frozen=True)
class MapRowViewSpec:
    row: int
    center: tuple[float, float]
    zoom: float
    center_crs: str | None = None
    viewport_px: tuple[int, int] = (1024, 1024)


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
    observation: str | None = None
    scale: int | None = None
    label_fit_margin: float | None = None
    roi_label: str | None = None
    lines: tuple[str, ...] = ()
    items: tuple[LegendItemSpec, ...] = ()
    below_items: tuple[LegendItemSpec, ...] = ()
    legend_items: tuple[LegendItemSpec, ...] = ()
    show_colorbar: bool | None = None
    show_scalebar: bool | None = None
    show_grid: bool | None = None
    show_hillshade: bool | None = None
    hillshade_extent: str | None = None
    show_roi: bool | None = None
    show_station_marker: bool | None = None
    show_stations_name: bool | None = None
    show_stations_elev: bool | None = None
    legend: str | None = None
    landcover_grouping: str | None = None
    variable: str | None = None

    @property
    def bottom_legend_items(self) -> tuple[LegendItemSpec, ...]:
        return self.below_items + tuple(
            item for item in self.legend_items if str(item.placement or "below") == "below"
        )

    @property
    def inside_legend_items(self) -> tuple[LegendItemSpec, ...]:
        return tuple(item for item in self.legend_items if str(item.placement or "below") == "inside")


@dataclass(frozen=True)
class MapRecipe:
    name: str
    title: str
    layout: LayoutSpec
    panels: tuple[MapPanelSpec, ...]
    output_name: str | None = None
    output_subdir: str | None = None
    figure_title: str | None = None
    row_labels: tuple[str, ...] = ()
    row_views: tuple[MapRowViewSpec, ...] = ()
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
            parsed = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{context}[{idx}] must be numeric") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"{context}[{idx}] must be finite")
        items.append(parsed)
    return tuple(items)


def _coerce_float_pair(value: object, *, context: str) -> tuple[float, float]:
    items = _coerce_float_tuple(value, context=context)
    if len(items) != 2:
        raise ValueError(f"{context} must contain exactly two numbers")
    return (items[0], items[1])


def _coerce_positive_float(value: object, *, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{context} must be finite")
    if parsed <= 0.0:
        raise ValueError(f"{context} must be > 0")
    return parsed


def _coerce_int(value: object, *, context: str, minimum: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return parsed


def _coerce_exact_int(value: object, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{context} must be an integer")
    try:
        parsed_float = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} must be an integer") from exc
    if not parsed_float.is_integer():
        raise ValueError(f"{context} must be an integer")
    parsed = int(parsed_float)
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
    return _coerce_positive_float(value, context=context)


def _coerce_viewport_px(value: object, *, context: str) -> tuple[int, int]:
    if value is None:
        return (1024, 1024)
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{context} must contain exactly two positive integers")
    width = _coerce_exact_int(value[0], context=f"{context}[0]", minimum=1)
    height = _coerce_exact_int(value[1], context=f"{context}[1]", minimum=1)
    return (width, height)


def _optional_hillshade_extent(value: object, *, context: str) -> str | None:
    token = _optional_str(value)
    if token is None:
        return None
    if token not in SUPPORTED_HILLSHADE_EXTENTS:
        supported = ", ".join(sorted(SUPPORTED_HILLSHADE_EXTENTS))
        raise ValueError(f"{context} must be one of: {supported}")
    return token


def _coerce_str_list(value: object, *, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings")
    return tuple(_require_str(item, context=f"{context}[]") for item in value)


def _parse_legend_items(
    value: object,
    *,
    context: str,
    allow_placement: bool = False,
    default_placement: str | None = None,
) -> tuple[LegendItemSpec, ...]:
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
        placement = default_placement
        anchor = None
        if allow_placement:
            placement = _optional_str(mapping.get("placement")) or default_placement
            if placement is not None:
                placement = placement.lower()
                if placement not in SUPPORTED_LEGEND_ITEM_PLACEMENTS:
                    supported = ", ".join(sorted(SUPPORTED_LEGEND_ITEM_PLACEMENTS))
                    raise ValueError(f"{context}[{idx}].placement must be one of: {supported}")
            anchor = _optional_str(mapping.get("anchor"))
            if anchor is not None:
                anchor = anchor.lower()
                if anchor not in SUPPORTED_LEGEND_ITEM_ANCHORS:
                    supported = ", ".join(sorted(SUPPORTED_LEGEND_ITEM_ANCHORS))
                    raise ValueError(f"{context}[{idx}].anchor must be one of: {supported}")
            if anchor is not None and placement != "inside":
                raise ValueError(f"{context}[{idx}].anchor is only supported when placement is inside")
        elif "placement" in mapping or "anchor" in mapping:
            raise ValueError(f"{context}[{idx}].placement and anchor are only supported for legend_items")
        items.append(LegendItemSpec(kind=kind, label=label, source=source, placement=placement, anchor=anchor))
    return tuple(items)


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
        hillshade_extent=_optional_hillshade_extent(mapping.get("hillshade_extent"), context=f"{context}.hillshade_extent"),
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


def _parse_row_views(value: object, *, context: str, nrows: int) -> tuple[MapRowViewSpec, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    views: list[MapRowViewSpec] = []
    seen_rows: set[int] = set()
    for idx, raw in enumerate(value):
        mapping = _require_mapping(raw, context=f"{context}[{idx}]")
        row = _coerce_int(mapping.get("row"), context=f"{context}[{idx}].row", minimum=0)
        if row >= nrows:
            raise ValueError(f"{context}[{idx}].row must be < {nrows}")
        if row in seen_rows:
            raise ValueError(f"{context}[{idx}].row defines a duplicate row view for row {row}")
        seen_rows.add(row)
        views.append(
            MapRowViewSpec(
                row=row,
                center=_coerce_float_pair(mapping.get("center"), context=f"{context}[{idx}].center"),
                zoom=_coerce_positive_float(mapping.get("zoom"), context=f"{context}[{idx}].zoom"),
                center_crs=_optional_str(mapping.get("center_crs")),
                viewport_px=_coerce_viewport_px(mapping.get("viewport_px"), context=f"{context}[{idx}].viewport_px"),
            )
        )
    return tuple(views)


def _parse_panel(value: object, *, context: str) -> MapPanelSpec:
    mapping = _require_mapping(value, context=context)
    kind = _require_str(mapping.get("kind"), context=f"{context}.kind")
    landcover_grouping = _optional_str(mapping.get("landcover_grouping"))
    if landcover_grouping is not None:
        landcover_grouping = landcover_grouping.lower()
    removed_panel_keys = _REMOVED_PANEL_KEYS
    if kind == "uncertainty":
        removed_panel_keys = removed_panel_keys - {"observation"}
    removed_keys = (removed_panel_keys | {"show_stations", "annotate_stations"}) & set(mapping)
    if removed_keys:
        raise ValueError(f"{context} uses removed panel keys: {', '.join(sorted(removed_keys))}")

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
        observation=_optional_str(mapping.get("observation")),
        scale=_optional_positive_int(mapping.get("scale"), context=f"{context}.scale"),
        label_fit_margin=_optional_positive_float(mapping.get("label_fit_margin"), context=f"{context}.label_fit_margin"),
        roi_label=_optional_str(mapping.get("roi_label")),
        lines=_coerce_str_list(mapping.get("lines"), context=f"{context}.lines"),
        items=_parse_legend_items(mapping.get("items"), context=f"{context}.items"),
        below_items=_parse_legend_items(
            mapping.get("below_items"),
            context=f"{context}.below_items",
            default_placement="below",
        ),
        legend_items=_parse_legend_items(
            mapping.get("legend_items"),
            context=f"{context}.legend_items",
            allow_placement=True,
            default_placement="below",
        ),
        show_colorbar=_coerce_bool(mapping.get("show_colorbar"), default=None),
        show_scalebar=_coerce_bool(mapping.get("show_scalebar"), default=None),
        show_grid=_coerce_bool(mapping.get("show_grid"), default=None),
        show_hillshade=_coerce_bool(mapping.get("show_hillshade"), default=None),
        hillshade_extent=_optional_hillshade_extent(mapping.get("hillshade_extent"), context=f"{context}.hillshade_extent"),
        show_roi=_coerce_bool(mapping.get("show_roi"), default=None),
        show_station_marker=_coerce_bool(mapping.get("show_station_marker"), default=None),
        show_stations_name=_coerce_bool(mapping.get("show_stations_name"), default=None),
        show_stations_elev=_coerce_bool(mapping.get("show_stations_elev"), default=None),
        legend=_optional_str(mapping.get("legend")),
        landcover_grouping=landcover_grouping,
    )

    if panel.legend is not None and panel.legend not in SUPPORTED_PANEL_LEGEND_LAYOUTS:
        supported = ", ".join(sorted(SUPPORTED_PANEL_LEGEND_LAYOUTS))
        raise ValueError(f"{context}.legend must be one of: {supported}")
    if panel.landcover_grouping is not None:
        if panel.kind != "landcover":
            raise ValueError(f"{context}.landcover_grouping is only supported for landcover panels")
        if panel.landcover_grouping not in SUPPORTED_LANDCOVER_GROUPINGS:
            supported = ", ".join(sorted(SUPPORTED_LANDCOVER_GROUPINGS))
            raise ValueError(f"{context}.landcover_grouping must be one of: {supported}")

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
        if panel.below_items:
            raise ValueError(f"{context}.below_items is only supported for non-legend panels")
        if panel.legend_items:
            raise ValueError(f"{context}.legend_items is only supported for non-legend panels")
    elif panel.kind == "fsc":
        if panel.source is not None and panel.source not in SUPPORTED_FSC_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_FSC_SOURCES))
            raise ValueError(f"{context}.source must be one of: {supported}")
    elif panel.kind == "wet_snow":
        if panel.source is not None and panel.source not in SUPPORTED_WET_SNOW_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_WET_SNOW_SOURCES))
            raise ValueError(f"{context}.source must be one of: {supported}")
    elif panel.kind == "uncertainty":
        if panel.observation is None:
            raise ValueError(f"{context}.observation is required for uncertainty panels")
        if panel.observation not in SUPPORTED_UNCERTAINTY_OBSERVATIONS:
            supported = ", ".join(sorted(SUPPORTED_UNCERTAINTY_OBSERVATIONS))
            raise ValueError(f"{context}.observation must be one of: {supported}")
        if panel.source is not None:
            raise ValueError(f"{context}.source is not supported for uncertainty panels")
    elif panel.kind == "wet_snow_line":
        if panel.source is not None and panel.source not in SUPPORTED_WET_SNOW_LINE_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_WET_SNOW_LINE_SOURCES))
            raise ValueError(f"{context}.source must be one of: {supported}")
    elif panel.kind == "wet_snow_elevation_fraction":
        if panel.source is not None and panel.source not in SUPPORTED_WET_SNOW_ELEVATION_FRACTION_SOURCES:
            supported = ", ".join(sorted(SUPPORTED_WET_SNOW_ELEVATION_FRACTION_SOURCES))
            raise ValueError(f"{context}.source must be one of: {supported}")
    elif panel.source is not None:
        raise ValueError(f"{context}.source is only valid for model panels, legend panels, and colorbar panels")

    return panel


def _validate_panel_layout(recipe: MapRecipe, *, config_path: Path) -> None:
    occupied: set[tuple[int, int]] = set()
    row_view_by_row = {view.row: view for view in recipe.row_views}
    for panel in recipe.panels:
        if panel.row + panel.rowspan > recipe.layout.nrows:
            raise ValueError(f"Panel '{panel.kind}' exceeds layout rows in {config_path}")
        if panel.col + panel.colspan > recipe.layout.ncols:
            raise ValueError(f"Panel '{panel.kind}' exceeds layout columns in {config_path}")
        spanned_view_keys = {
            row_view_by_row[row] if row in row_view_by_row else None
            for row in range(panel.row, panel.row + panel.rowspan)
        }
        if len(spanned_view_keys) > 1:
            raise ValueError(
                f"Panel '{panel.kind}' spans rows with different row_views in {config_path}"
            )
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
    row_views = _parse_row_views(mapping.get("row_views"), context=f"{context}.row_views", nrows=layout.nrows)
    recipe = MapRecipe(
        name=recipe_name,
        title=title,
        output_name=_optional_str(mapping.get("output_name")),
        output_subdir=_optional_str(mapping.get("output_subdir")),
        figure_title=_optional_str(mapping.get("figure_title")),
        row_labels=_coerce_str_list(mapping.get("row_labels"), context=f"{context}.row_labels"),
        row_views=row_views,
        layout=layout,
        defaults=_parse_defaults(mapping.get("defaults"), context=f"{context}.defaults"),
        panels=tuple(_parse_panel(item, context=f"{context}.panels[{idx}]") for idx, item in enumerate(raw_panels)),
    )
    if recipe.row_labels and len(recipe.row_labels) != recipe.layout.nrows:
        raise ValueError(f"{context}.row_labels must have length {recipe.layout.nrows}")
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
    "MapRowViewSpec",
    "ProjectMapsConfig",
    "SUPPORTED_MODEL_SOURCES",
    "SUPPORTED_UNCERTAINTY_OBSERVATIONS",
    "SUPPORTED_PANEL_KINDS",
    "default_project_maps_config_path",
    "load_project_maps_config",
]
