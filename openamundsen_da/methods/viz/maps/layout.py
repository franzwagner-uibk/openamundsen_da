from __future__ import annotations

import math
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from rasterio.crs import CRS
from rasterio.warp import transform as transform_coords

from openamundsen_da.methods.viz.theme import FIGWIDTH_OVERVIEW_PAPER
from openamundsen_da.methods.viz.maps.config import MapDefaults, MapPanelSpec, MapRecipe, MapRowViewSpec
from openamundsen_da.methods.viz.maps.data import ObservationScene, StaticContext
from openamundsen_da.methods.viz.maps.styles import LANDCOVER_LABELS, WET_SNOW_LABELS
from openamundsen_da.methods.viz.maps.theme import (
    _ANNOTATION_ZORDER,
    _BOTTOM_MARGIN,
    _BUFFER_RATIO,
    _CLASSIFIED_PANEL_KINDS,
    _COLORBAR_TICK_SIZE,
    _COLORBAR_TITLE_SIZE,
    _CONTINUOUS_COLORBAR_PANEL_KINDS,
    _FIGURE_HEIGHT_MAX,
    _FIGURE_HEIGHT_MIN,
    _GRID_COLOR,
    _GRID_ZORDER,
    _GRID_STYLE,
    _GRID_WIDTH,
    _HORIZONTAL_ANNOTATION_MIN_GAP_MAX_ASPECT,
    _HORIZONTAL_COLORBAR_GAP_AXES,
    _HORIZONTAL_COLORBAR_HEIGHT_AXES,
    _HORIZONTAL_COLORBAR_BOTTOM_PAD_AXES,
    _HORIZONTAL_COLORBAR_MIN_GAP_IN,
    _HORIZONTAL_LEGEND_BOTTOM_PAD_AXES,
    _HORIZONTAL_LEGEND_GAP_AXES,
    _HORIZONTAL_LEGEND_HANDLE_TEXT_PAD_IN,
    _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN,
    _HORIZONTAL_LEGEND_ITEM_GAP_IN,
    _HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN,
    _HORIZONTAL_LEGEND_MIN_GAP_IN,
    _HORIZONTAL_LEGEND_MIN_TEXT_WIDTH_IN,
    _HORIZONTAL_LEGEND_ROW_HEIGHT_AXES,
    _HORIZONTAL_LEGEND_SIDE_PAD_IN,
    _HORIZONTAL_LEGEND_TEXT_SIZE,
    _LAYOUT_COL_GAP,
    _LAYOUT_ROW_GAP,
    _LEFT_MARGIN,
    _RIGHT_MARGIN,
    _SPINE_WIDTH,
    _TICK_LABEL_MIN_GAP_IN,
    _TICK_SIZE,
    _TOP_MARGIN,
    _VERTICAL_COLORBAR_BOTTOM_AXES,
    _VERTICAL_COLORBAR_GAP_EXTRA,
    _VERTICAL_COLORBAR_HEIGHT_AXES,
    _VERTICAL_COLORBAR_OUTER_EXTRA,
    _VERTICAL_COLORBAR_WIDTH_AXES,
    _VERTICAL_COLORBAR_XOFFSET_AXES,
)

_GOOGLE_TILE_SIZE_PX = 256.0
_WEB_MERCATOR_RADIUS_M = 6378137.0
_WGS84_CRS = "EPSG:4326"


def buffered_extent(context: StaticContext) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = context.roi_gdf.total_bounds
    dx = float(maxx - minx)
    dy = float(maxy - miny)
    cell = abs(float(context.spec.transform.a))
    pad_x = max(dx * _BUFFER_RATIO, 2.0 * cell)
    pad_y = max(dy * _BUFFER_RATIO, 2.0 * cell)
    return (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)


def google_zoom_meters_per_pixel(latitude_deg: float, zoom: float) -> float:
    cos_lat = float(np.cos(np.deg2rad(float(latitude_deg))))
    if cos_lat <= 0.0:
        raise ValueError(f"Google zoom scale is undefined at latitude {latitude_deg:g}")
    return cos_lat * 2.0 * np.pi * _WEB_MERCATOR_RADIUS_M / (_GOOGLE_TILE_SIZE_PX * (2.0 ** float(zoom)))


def _coerce_crs(value: object, *, context: str) -> CRS:
    if value is None:
        raise ValueError(f"{context} is required for row zoom views")
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f"Invalid CRS for {context}: {value!r}") from exc


def _require_meter_projected_setup_crs(context: StaticContext) -> CRS:
    crs = _coerce_crs(context.spec.crs, context="setup CRS")
    units = str(getattr(crs, "linear_units", "") or "").lower()
    if not crs.is_projected or ("metre" not in units and "meter" not in units):
        raise ValueError(f"Row zoom views require a projected meter-based setup CRS, got {context.spec.crs!r}")
    return crs


def _transform_point(x: float, y: float, *, src_crs: CRS | str, dst_crs: CRS | str) -> tuple[float, float]:
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    if src == dst:
        return float(x), float(y)
    xs, ys = transform_coords(src, dst, [float(x)], [float(y)])
    return float(xs[0]), float(ys[0])


def row_view_extent(view: MapRowViewSpec, context: StaticContext) -> tuple[float, float, float, float]:
    setup_crs = _require_meter_projected_setup_crs(context)
    center_crs = _coerce_crs(view.center_crs or setup_crs, context=f"row_views[{view.row}].center_crs")
    center_x, center_y = view.center
    map_x, map_y = _transform_point(center_x, center_y, src_crs=center_crs, dst_crs=setup_crs)
    _lon, lat = _transform_point(center_x, center_y, src_crs=center_crs, dst_crs=_WGS84_CRS)
    meters_per_pixel = google_zoom_meters_per_pixel(lat, view.zoom)
    width_m = meters_per_pixel * float(view.viewport_px[0])
    height_m = meters_per_pixel * float(view.viewport_px[1])
    return (
        map_x - 0.5 * width_m,
        map_x + 0.5 * width_m,
        map_y - 0.5 * height_m,
        map_y + 0.5 * height_m,
    )


def row_extents_for_recipe(recipe: MapRecipe, context: StaticContext) -> dict[int, tuple[float, float, float, float]]:
    full_extent = buffered_extent(context)
    extents = {row: full_extent for row in range(recipe.layout.nrows)}
    for view in recipe.row_views:
        extents[int(view.row)] = row_view_extent(view, context)
    return extents


def figure_height_for_extent(extent: tuple[float, float, float, float]) -> float:
    width = max(1.0, float(extent[1] - extent[0]))
    height = max(1.0, float(extent[3] - extent[2]))
    aspect = height / width
    panel_width = FIGWIDTH_OVERVIEW_PAPER / 3.0
    computed = panel_width * aspect * 1.04
    return float(np.clip(computed, _FIGURE_HEIGHT_MIN, _FIGURE_HEIGHT_MAX))


def nice_tick_step(span: float) -> float:
    target = max(span / 4.0, 1.0)
    exponent = int(np.floor(np.log10(target)))
    base = 10**exponent
    candidates = [1.0 * base, 2.0 * base, 2.5 * base, 5.0 * base, 10.0 * base]
    return min(candidates, key=lambda value: abs(value - target))


def ticks_for_extent(start: float, stop: float) -> np.ndarray:
    span = max(1.0, stop - start)
    step = nice_tick_step(span)
    first = np.ceil(start / step) * step
    last = np.floor(stop / step) * step
    return np.arange(first, last + 0.5 * step, step)


def coord_label(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def tick_label_stride(ax, ticks: np.ndarray, *, axis: str) -> int:
    base_stride = 2
    if len(ticks) <= 1:
        return 1
    axis_length_in = axis_width_inches(ax) if axis == "x" else axis_height_inches(ax)
    max_labels = max(1, int(math.floor(axis_length_in / _TICK_LABEL_MIN_GAP_IN)))
    physical_stride = int(math.ceil(len(ticks) / max_labels))
    return max(base_stride, physical_stride)


def axis_width_inches(ax) -> float:
    bbox = ax.get_position()
    return float(ax.figure.get_size_inches()[0] * bbox.width)


def axis_height_inches(ax) -> float:
    bbox = ax.get_position()
    return float(ax.figure.get_size_inches()[1] * bbox.height)


@lru_cache(maxsize=256)
def text_width_in(text: str, *, size: float) -> float:
    if not text:
        return 0.0
    path = TextPath((0.0, 0.0), text, prop=FontProperties(size=size))
    return float(path.get_extents().width) / 72.0


@lru_cache(maxsize=256)
def text_size_in(text: str, *, size: float) -> tuple[float, float]:
    if not text:
        return (0.0, 0.0)
    path = TextPath((0.0, 0.0), text, prop=FontProperties(size=size))
    bounds = path.get_extents()
    return float(bounds.width) / 72.0, float(bounds.height) / 72.0


def axes_title_fontsize(ax) -> float:
    width_in = axis_width_inches(ax)
    if width_in < 1.45:
        return 6.4
    if width_in < 1.8:
        return 7.0
    if width_in < 2.35:
        return 7.8
    if width_in < 2.8:
        return 8.6
    return 9.4


def axes_date_fontsize(ax) -> float:
    width_in = axis_width_inches(ax)
    if width_in < 1.45:
        return 5.8
    if width_in < 1.8:
        return 6.2
    if width_in < 2.35:
        return 6.8
    if width_in < 2.8:
        return 7.4
    return 8.0


def draw_axes_title(ax, title: str) -> None:
    ax.set_title(
        title,
        loc="left",
        pad=2.8,
        fontsize=axes_title_fontsize(ax),
        color="black",
    )


def apply_map_axis_style(
    ax,
    extent: tuple[float, float, float, float],
    *,
    title: str | None,
    show_grid: bool,
    show_y_ticklabels: bool = True,
    aspect_adjustable: str = "box",
) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable=aspect_adjustable)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(_SPINE_WIDTH)
        spine.set_color("black")
        spine.set_visible(True)

    xticks = ticks_for_extent(extent[0], extent[1])
    yticks = ticks_for_extent(extent[2], extent[3])
    x_label_stride = tick_label_stride(ax, xticks, axis="x")
    y_label_stride = tick_label_stride(ax, yticks, axis="y")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([coord_label(value) if idx % x_label_stride == 0 else "" for idx, value in enumerate(xticks)])
    ax.set_yticklabels(
        [coord_label(value) if show_y_ticklabels and idx % y_label_stride == 0 else "" for idx, value in enumerate(yticks)]
    )
    ax.tick_params(
        axis="x",
        direction="out",
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        labelsize=_TICK_SIZE,
        length=2.2,
        width=0.75,
        pad=1.6,
    )
    ax.tick_params(
        axis="y",
        direction="out",
        left=True,
        right=False,
        labelleft=show_y_ticklabels,
        labelright=False,
        labelsize=_TICK_SIZE,
        length=2.2,
        width=0.75,
        pad=3.2 if show_y_ticklabels else 1.0,
    )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_rotation(0)
        tick.label1.set_ha("center")
        tick.label1.set_va("top")
        tick.label2.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_rotation(90)
        tick.label1.set_rotation_mode("anchor")
        tick.label1.set_ha("center")
        tick.label1.set_va("center")
        tick.label2.set_visible(False)
    ax.grid(False)
    if title:
        draw_axes_title(ax, title)


def draw_map_grid_overlay(ax, *, show_grid: bool) -> None:
    ax.grid(False)
    if not show_grid:
        return
    for line in list(ax.lines):
        if line.get_gid() == "oa_da_grid":
            line.remove()

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xticks = [float(value) for value in ax.get_xticks() if xmin <= float(value) <= xmax]
    yticks = [float(value) for value in ax.get_yticks() if ymin <= float(value) <= ymax]

    for value in xticks:
        line = ax.axvline(
            value,
            color=_GRID_COLOR,
            linestyle=_GRID_STYLE,
            linewidth=_GRID_WIDTH,
            alpha=0.92,
            zorder=_GRID_ZORDER,
            clip_on=True,
        )
        line.set_gid("oa_da_grid")
    for value in yticks:
        line = ax.axhline(
            value,
            color=_GRID_COLOR,
            linestyle=_GRID_STYLE,
            linewidth=_GRID_WIDTH,
            alpha=0.92,
            zorder=_GRID_ZORDER,
            clip_on=True,
        )
        line.set_gid("oa_da_grid")

    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_zorder(_ANNOTATION_ZORDER)


def resolve_flag(panel_value: bool | None, defaults: MapDefaults, attr_name: str, builtin_default: bool) -> bool:
    recipe_value = getattr(defaults, attr_name)
    if panel_value is not None:
        return bool(panel_value)
    if recipe_value is not None:
        return bool(recipe_value)
    return builtin_default


def resolve_panel_toggle(value: bool | None, builtin_default: bool) -> bool:
    if value is None:
        return builtin_default
    return bool(value)


def figure_prefers_horizontal_legends(recipe: MapRecipe) -> bool:
    return recipe.layout.ncols >= 4 or any(panel.kind in _CLASSIFIED_PANEL_KINDS for panel in recipe.panels)


def panel_legend_layout(panel: MapPanelSpec, *, figure_horizontal_default: bool, is_colorbar: bool = False) -> str:
    if panel.legend is not None:
        return panel.legend
    if figure_horizontal_default:
        return "horizontal"
    if panel.kind in _CLASSIFIED_PANEL_KINDS:
        return "horizontal"
    return "vertical" if is_colorbar else "horizontal"


def effective_width_ratios(recipe: MapRecipe) -> tuple[float, ...]:
    return recipe.layout.width_ratios or tuple(1.0 for _ in range(recipe.layout.ncols))


def effective_height_ratios(recipe: MapRecipe) -> tuple[float, ...]:
    return recipe.layout.height_ratios or tuple(1.0 for _ in range(recipe.layout.nrows))


def _extent_aspect(extent: tuple[float, float, float, float]) -> float:
    width = max(1.0, float(extent[1] - extent[0]))
    height = max(1.0, float(extent[3] - extent[2]))
    return height / width


def effective_row_height_ratios(
    recipe: MapRecipe,
    *,
    row_extents: dict[int, tuple[float, float, float, float]],
) -> tuple[float, ...]:
    height_factors = effective_height_ratios(recipe)
    return tuple(float(height_factors[row]) * _extent_aspect(row_extents[row]) for row in range(recipe.layout.nrows))


def horizontal_legend_item_width_in(label: str) -> float:
    text_width = max(_HORIZONTAL_LEGEND_MIN_TEXT_WIDTH_IN, text_width_in(str(label), size=_HORIZONTAL_LEGEND_TEXT_SIZE))
    return _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN + _HORIZONTAL_LEGEND_HANDLE_TEXT_PAD_IN + text_width


def horizontal_legend_side_pad_in(panel_width_in: float) -> float:
    return min(_HORIZONTAL_LEGEND_SIDE_PAD_IN, max(panel_width_in / 2.0 - 0.05, 0.0))


def horizontal_legend_available_width_in(panel_width_in: float) -> float:
    side_pad_in = horizontal_legend_side_pad_in(panel_width_in)
    return max(panel_width_in - 2.0 * side_pad_in, 0.25)


def pack_horizontal_legend_rows(labels: list[str], *, panel_width_in: float) -> list[list[str]]:
    if not labels:
        return []
    available_width_in = horizontal_legend_available_width_in(panel_width_in)
    item_widths = [horizontal_legend_item_width_in(label) for label in labels]
    rows: list[list[str]] = []
    current_row: list[str] = []
    current_width = 0.0

    for label, item_width in zip(labels, item_widths, strict=False):
        proposed_width = item_width if not current_row else current_width + _HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN + item_width
        if current_row and proposed_width > available_width_in:
            rows.append(current_row)
            current_row = [label]
            current_width = item_width
        else:
            current_row.append(label)
            current_width = proposed_width

    if current_row:
        rows.append(current_row)
    return rows


def horizontal_legend_row_height_factors(rows: list[list[str]]) -> list[float]:
    return [1.0] * len(rows)


def horizontal_legend_row_layout(
    row_labels: list[str],
    *,
    panel_width_in: float,
) -> tuple[list[float], float, float]:
    item_widths = [horizontal_legend_item_width_in(label) for label in row_labels]
    side_pad_in = horizontal_legend_side_pad_in(panel_width_in)
    available_width_in = horizontal_legend_available_width_in(panel_width_in)

    if len(item_widths) <= 1:
        return item_widths, side_pad_in, 0.0

    min_gaps_total = _HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN * (len(item_widths) - 1)
    content_width_in = float(sum(item_widths) + min_gaps_total)
    extra_gap_in = max(available_width_in - content_width_in, 0.0) / (len(item_widths) - 1)
    return item_widths, side_pad_in, _HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN + min(extra_gap_in, _HORIZONTAL_LEGEND_ITEM_GAP_IN)


def classified_legend_labels(panel_kind: str) -> list[str]:
    if panel_kind == "landcover":
        return list(LANDCOVER_LABELS.values())
    if panel_kind == "wet_snow":
        return [WET_SNOW_LABELS[code] for code in sorted(WET_SNOW_LABELS)]
    raise ValueError(f"Unsupported classified panel kind '{panel_kind}'")


def horizontal_legend_bottom_pad(panel_width_in: float) -> float:
    if panel_width_in < 1.7:
        return max(_HORIZONTAL_LEGEND_BOTTOM_PAD_AXES, 0.56)
    if panel_width_in < 2.0:
        return max(_HORIZONTAL_LEGEND_BOTTOM_PAD_AXES, 0.28)
    return _HORIZONTAL_LEGEND_BOTTOM_PAD_AXES


def horizontal_annotation_gap_axes(
    *,
    panel_height_in: float | None,
    panel_aspect: float | None = None,
    base_gap_axes: float,
    min_gap_in: float,
) -> float:
    if panel_height_in is None or panel_height_in <= 0.0:
        return float(base_gap_axes)
    if panel_aspect is None or panel_aspect >= _HORIZONTAL_ANNOTATION_MIN_GAP_MAX_ASPECT:
        return float(base_gap_axes)
    return max(float(base_gap_axes), float(min_gap_in) / float(panel_height_in))


def horizontal_annotation_total_extra(
    *,
    panel_height_in: float | None,
    panel_aspect: float | None = None,
    base_gap_axes: float,
    min_gap_in: float,
    content_height_axes: float,
    bottom_pad_axes: float = 0.0,
) -> float:
    gap_axes = horizontal_annotation_gap_axes(
        panel_height_in=panel_height_in,
        panel_aspect=panel_aspect,
        base_gap_axes=base_gap_axes,
        min_gap_in=min_gap_in,
    )
    return gap_axes + float(content_height_axes) + float(bottom_pad_axes)


def horizontal_colorbar_gap_axes(ax) -> float:
    return horizontal_annotation_gap_axes(
        panel_height_in=axis_height_inches(ax),
        panel_aspect=axis_height_inches(ax) / max(axis_width_inches(ax), 1e-9),
        base_gap_axes=_HORIZONTAL_COLORBAR_GAP_AXES,
        min_gap_in=_HORIZONTAL_COLORBAR_MIN_GAP_IN,
    )


def horizontal_colorbar_total_extra(*, panel_height_in: float | None, panel_aspect: float | None = None) -> float:
    return horizontal_annotation_total_extra(
        panel_height_in=panel_height_in,
        panel_aspect=panel_aspect,
        base_gap_axes=_HORIZONTAL_COLORBAR_GAP_AXES,
        min_gap_in=_HORIZONTAL_COLORBAR_MIN_GAP_IN,
        content_height_axes=_HORIZONTAL_COLORBAR_HEIGHT_AXES,
        bottom_pad_axes=_HORIZONTAL_COLORBAR_BOTTOM_PAD_AXES,
    )


def horizontal_legend_gap_axes(ax) -> float:
    return horizontal_annotation_gap_axes(
        panel_height_in=axis_height_inches(ax),
        panel_aspect=axis_height_inches(ax) / max(axis_width_inches(ax), 1e-9),
        base_gap_axes=_HORIZONTAL_LEGEND_GAP_AXES,
        min_gap_in=_HORIZONTAL_LEGEND_MIN_GAP_IN,
    )


def horizontal_legend_total_extra(
    rows: list[list[str]],
    *,
    panel_width_in: float,
    panel_height_in: float | None = None,
    panel_aspect: float | None = None,
) -> float:
    if not rows:
        return 0.0
    total_row_units = float(sum(horizontal_legend_row_height_factors(rows)))
    return horizontal_annotation_total_extra(
        panel_height_in=panel_height_in,
        panel_aspect=panel_aspect,
        base_gap_axes=_HORIZONTAL_LEGEND_GAP_AXES,
        min_gap_in=_HORIZONTAL_LEGEND_MIN_GAP_IN,
        content_height_axes=total_row_units * _HORIZONTAL_LEGEND_ROW_HEIGHT_AXES,
        bottom_pad_axes=horizontal_legend_bottom_pad(panel_width_in),
    )


def panel_has_vertical_colorbar(
    panel: MapPanelSpec,
    *,
    defaults: MapDefaults,
    figure_horizontal_default: bool,
) -> bool:
    if panel.kind not in _CONTINUOUS_COLORBAR_PANEL_KINDS:
        return False
    if not resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        return False
    return panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True) == "vertical"


def column_gap_factors(
    recipe: MapRecipe,
    *,
    figure_horizontal_default: bool,
) -> tuple[float, ...]:
    gap_factors: list[float] = []
    for col in range(recipe.layout.ncols - 1):
        extra = 0.0
        for panel in recipe.panels:
            end_col = int(panel.col + panel.colspan - 1)
            if end_col != col:
                continue
            if panel_has_vertical_colorbar(panel, defaults=recipe.defaults, figure_horizontal_default=figure_horizontal_default):
                extra = max(extra, _VERTICAL_COLORBAR_GAP_EXTRA)
        gap_factors.append(_LAYOUT_COL_GAP + extra)
    return tuple(gap_factors)


def outer_right_factor(
    recipe: MapRecipe,
    *,
    figure_horizontal_default: bool,
) -> float:
    outer_extra = 0.0
    last_col = recipe.layout.ncols - 1
    for panel in recipe.panels:
        end_col = int(panel.col + panel.colspan - 1)
        if end_col != last_col:
            continue
        if panel_has_vertical_colorbar(panel, defaults=recipe.defaults, figure_horizontal_default=figure_horizontal_default):
            outer_extra = max(outer_extra, _VERTICAL_COLORBAR_OUTER_EXTRA)
    return outer_extra


def panel_empty_below_units(recipe: MapRecipe, panel: MapPanelSpec) -> float:
    end_row = int(panel.row + panel.rowspan - 1)
    start_col = int(panel.col)
    end_col = int(panel.col + panel.colspan - 1)
    height_factors = effective_height_ratios(recipe)
    occupied: set[tuple[int, int]] = set()
    for other in recipe.panels:
        for row in range(other.row, other.row + other.rowspan):
            for col in range(other.col, other.col + other.colspan):
                occupied.add((row, col))

    borrow_units = 0.0
    for row in range(end_row + 1, recipe.layout.nrows):
        if any((row, col) in occupied for col in range(start_col, end_col + 1)):
            break
        borrow_units += float(height_factors[row])
    return borrow_units


def row_bottom_extras(
    recipe: MapRecipe,
    *,
    context: StaticContext,
    panel_width_in: float,
    row_panel_height_in: dict[int, float] | None = None,
    row_panel_aspect: dict[int, float] | None = None,
    figure_horizontal_default: bool,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
    classified_labels_getter,
    below_items_extra_getter,
) -> dict[int, float]:
    row_extras = {row: 0.0 for row in range(recipe.layout.nrows)}
    for panel in recipe.panels:
        row = int(panel.row + panel.rowspan - 1)
        panel_height_in = None if row_panel_height_in is None else row_panel_height_in.get(row)
        panel_aspect = None if row_panel_aspect is None else row_panel_aspect.get(row)
        extra = 0.0
        probability_classified_panel = panel.kind in {"wet_snow", "wet_snow_line"} and panel.source in {
            "prior_probability",
            "posterior",
            "posterior_probability",
        }
        if probability_classified_panel:
            if (
                resolve_flag(panel.show_colorbar, recipe.defaults, "show_colorbar", True)
                and panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True) == "horizontal"
            ):
                extra = horizontal_colorbar_total_extra(panel_height_in=panel_height_in, panel_aspect=panel_aspect)
        elif panel.kind in _CLASSIFIED_PANEL_KINDS:
            layout = panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default)
            if layout == "horizontal":
                labels = classified_labels_getter(panel, context=context, defaults=recipe.defaults, obs_cache=obs_cache)
                rows = pack_horizontal_legend_rows(labels, panel_width_in=panel_width_in)
                extra = horizontal_legend_total_extra(
                    rows,
                    panel_width_in=panel_width_in,
                    panel_height_in=panel_height_in,
                    panel_aspect=panel_aspect,
                )
                extra = max(extra - panel_empty_below_units(recipe, panel), 0.0)
        elif panel.kind in _CONTINUOUS_COLORBAR_PANEL_KINDS:
            if (
                resolve_flag(panel.show_colorbar, recipe.defaults, "show_colorbar", True)
                and panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True) == "horizontal"
            ):
                extra = horizontal_colorbar_total_extra(panel_height_in=panel_height_in, panel_aspect=panel_aspect)
        if panel.bottom_legend_items:
            below_items_extra = below_items_extra_getter(
                panel.bottom_legend_items,
                panel_height_in=panel_height_in,
                panel_aspect=panel_aspect,
            )
            below_items_extra = max(below_items_extra - panel_empty_below_units(recipe, panel), 0.0)
            extra = max(extra, below_items_extra)
        row_extras[row] = max(row_extras.get(row, 0.0), extra)
    return row_extras


def extract_unit_title(label: str | None) -> str | None:
    if label is None:
        return None
    if "[" in label and "]" in label:
        return label[label.index("[") : label.rindex("]") + 1]
    return label


def register_child_axes(parent_ax, child_ax) -> None:
    children = list(getattr(parent_ax, "_oa_child_axes", []))
    children.append(child_ax)
    setattr(parent_ax, "_oa_child_axes", children)


def attach_colorbar(
    ax,
    mappable,
    *,
    label: str | None = None,
    ticks: tuple[float, ...] = (),
    ticklabels: tuple[str, ...] = (),
    layout: str,
) -> None:
    if layout == "horizontal":
        gap_axes = horizontal_colorbar_gap_axes(ax)
        title = extract_unit_title(label)
        panel_width_in = axis_width_inches(ax)
        unit_pad_in = 0.035 if title else 0.0
        reserved_width_in = text_width_in(title, size=_COLORBAR_TITLE_SIZE) + unit_pad_in if title else 0.0
        colorbar_width = 1.0 if not title else max(0.55, 1.0 - reserved_width_in / max(panel_width_in, 1e-9))
        container_ax = ax.inset_axes(
            [
                0.0,
                -(gap_axes + _HORIZONTAL_COLORBAR_HEIGHT_AXES),
                1.0,
                _HORIZONTAL_COLORBAR_HEIGHT_AXES,
            ],
            transform=ax.transAxes,
        )
        container_ax.set_axis_off()
        cax = container_ax.inset_axes([0.0, 0.0, colorbar_width, 1.0], transform=container_ax.transAxes)
        register_child_axes(ax, container_ax)
        register_child_axes(ax, cax)
        cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
        if ticks:
            cbar.set_ticks(ticks)
        if ticklabels:
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE, length=2.0, width=0.65, pad=1.0)
        if title:
            container_ax.text(
                1.0,
                0.5,
                title,
                transform=container_ax.transAxes,
                ha="right",
                va="center",
                fontsize=_COLORBAR_TITLE_SIZE,
            )
        elif label:
            cbar.set_label(label, fontsize=_COLORBAR_TITLE_SIZE)
        return

    cax = ax.inset_axes(
        [
            1.0 + _VERTICAL_COLORBAR_XOFFSET_AXES,
            _VERTICAL_COLORBAR_BOTTOM_AXES,
            _VERTICAL_COLORBAR_WIDTH_AXES,
            _VERTICAL_COLORBAR_HEIGHT_AXES,
        ],
        transform=ax.transAxes,
    )
    cbar = plt.colorbar(mappable, cax=cax, orientation="vertical")
    if ticks:
        cbar.set_ticks(ticks)
    if ticklabels:
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE, length=2.2, width=0.7, pad=1.0)
    for tick in cbar.ax.get_yticklabels():
        tick.set_rotation(90)
        tick.set_va("center")
        tick.set_ha("center")
    title = extract_unit_title(label)
    if title:
        cbar.ax.text(0.5, 1.02, title, transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=_COLORBAR_TITLE_SIZE)
    elif label:
        cbar.set_label(label, fontsize=_COLORBAR_TITLE_SIZE)


def panel_width_in_for_recipe(recipe: MapRecipe, *, figure_horizontal_default: bool) -> float:
    width_factors = effective_width_ratios(recipe)
    col_gap = column_gap_factors(recipe, figure_horizontal_default=figure_horizontal_default)
    outer_extra = outer_right_factor(recipe, figure_horizontal_default=figure_horizontal_default)
    inner_width_in = FIGWIDTH_OVERVIEW_PAPER * (_RIGHT_MARGIN - _LEFT_MARGIN)
    width_units = float(sum(width_factors)) + float(sum(col_gap)) + outer_extra
    return inner_width_in / max(width_units, 1.0)


def figure_size(
    extent: tuple[float, float, float, float],
    recipe: MapRecipe,
    *,
    context: StaticContext,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
    row_extents: dict[int, tuple[float, float, float, float]] | None = None,
    classified_labels_getter,
    below_items_extra_getter,
) -> tuple[float, float]:
    row_extents = row_extents or {row: extent for row in range(recipe.layout.nrows)}
    row_height_ratios = effective_row_height_ratios(recipe, row_extents=row_extents)
    figure_horizontal_default = figure_prefers_horizontal_legends(recipe)
    panel_width = panel_width_in_for_recipe(recipe, figure_horizontal_default=figure_horizontal_default)
    row_heights = tuple(panel_width * float(ratio) * 1.02 for ratio in row_height_ratios)
    row_panel_height_in = {row: row_heights[row] for row in range(recipe.layout.nrows)}
    row_panel_aspect = {row: float(row_height_ratios[row]) for row in range(recipe.layout.nrows)}
    bottom_extras = row_bottom_extras(
        recipe,
        context=context,
        panel_width_in=panel_width,
        row_panel_height_in=row_panel_height_in,
        row_panel_aspect=row_panel_aspect,
        figure_horizontal_default=figure_horizontal_default,
        obs_cache=obs_cache,
        classified_labels_getter=classified_labels_getter,
        below_items_extra_getter=below_items_extra_getter,
    )
    inter_row_gap_heights = tuple(
        (_LAYOUT_ROW_GAP + bottom_extras[row]) * row_heights[row]
        for row in range(recipe.layout.nrows - 1)
    )
    bottom_extra_height = bottom_extras.get(recipe.layout.nrows - 1, 0.0) * row_heights[-1]
    inner_height = float(sum(row_heights) + sum(inter_row_gap_heights) + bottom_extra_height)
    fig_height = inner_height / max(_TOP_MARGIN - _BOTTOM_MARGIN, 1e-9)
    return FIGWIDTH_OVERVIEW_PAPER, float(np.clip(fig_height, _FIGURE_HEIGHT_MIN, _FIGURE_HEIGHT_MAX))


def expanded_grid_ratios(data_ratios: tuple[float, ...], gap_factors: tuple[float, ...]) -> list[float]:
    ratios: list[float] = []
    for idx, ratio in enumerate(data_ratios):
        ratios.append(float(ratio))
        if idx < len(gap_factors):
            ratios.append(float(gap_factors[idx]))
    return ratios


def grid_span(start: int, span: int) -> slice:
    grid_start = start * 2
    grid_stop = grid_start + span * 2 - 1
    return slice(grid_start, grid_stop)


def axes_group_bbox(fig, renderer, axes_group) -> tuple[float, float] | None:
    inverse = fig.transFigure.inverted()
    y0_values: list[float] = []
    y1_values: list[float] = []
    for ax in axes_group:
        bbox = ax.get_tightbbox(renderer)
        if bbox is not None:
            bbox_fig = bbox.transformed(inverse)
            y0_values.append(float(bbox_fig.y0))
            y1_values.append(float(bbox_fig.y1))
        for child_ax in getattr(ax, "_oa_child_axes", []):
            child_bbox = child_ax.get_tightbbox(renderer)
            if child_bbox is None:
                continue
            child_bbox_fig = child_bbox.transformed(inverse)
            y0_values.append(float(child_bbox_fig.y0))
            y1_values.append(float(child_bbox_fig.y1))
    if not y0_values:
        return None
    return min(y0_values), max(y1_values)


def axes_group_panel_bbox(axes_group) -> tuple[float, float] | None:
    y0_values: list[float] = []
    y1_values: list[float] = []
    for ax in axes_group:
        pos = ax.get_position()
        y0_values.append(float(pos.y0))
        y1_values.append(float(pos.y1))
    if not y0_values:
        return None
    return min(y0_values), max(y1_values)


def shift_axes_group(axes_group, delta_y: float) -> None:
    if abs(delta_y) <= 1e-9:
        return
    for ax in axes_group:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + delta_y, pos.width, pos.height])


def tighten_panel_row_gaps(fig, row_axes: dict[int, list], *, target_gap_scale: float = 1.0) -> None:
    if len(row_axes) < 2:
        return
    canvas = fig.canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    ordered_rows = sorted(row_axes)
    for idx in range(len(ordered_rows) - 1):
        upper_row = ordered_rows[idx]
        lower_rows = ordered_rows[idx + 1 :]
        upper_bbox = axes_group_bbox(fig, renderer, row_axes[upper_row])
        lower_group = [ax for row in lower_rows for ax in row_axes[row]]
        lower_bbox = axes_group_bbox(fig, renderer, lower_group)
        next_lower_axes = row_axes[ordered_rows[idx + 1]]
        next_lower_bbox = axes_group_bbox(fig, renderer, next_lower_axes)
        next_lower_panel_bbox = axes_group_panel_bbox(next_lower_axes)
        if upper_bbox is None or lower_bbox is None:
            continue
        current_gap = upper_bbox[0] - lower_bbox[1]
        mean_panel_height = float(np.mean([ax.get_position().height for ax in row_axes[upper_row]]))
        base_gap = max(0.0, _LAYOUT_ROW_GAP * float(target_gap_scale) * mean_panel_height)
        lower_title_overhang = 0.0
        if next_lower_bbox is not None and next_lower_panel_bbox is not None:
            lower_title_overhang = max(0.0, next_lower_bbox[1] - next_lower_panel_bbox[1])
        target_gap = max(base_gap, 0.008 + 0.65 * lower_title_overhang)
        delta_y = current_gap - target_gap
        if abs(delta_y) <= 1e-6:
            continue
        for row in lower_rows:
            shift_axes_group(row_axes[row], delta_y)
        canvas.draw()
        renderer = canvas.get_renderer()
