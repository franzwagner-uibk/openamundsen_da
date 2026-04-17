from __future__ import annotations

from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from openamundsen_da.methods.viz.theme import FIGWIDTH_OVERVIEW_PAPER
from openamundsen_da.methods.viz.project_maps.config import MapDefaults, MapPanelSpec, MapRecipe
from openamundsen_da.methods.viz.project_maps.data import ObservationScene, StaticContext
from openamundsen_da.methods.viz.project_maps.styles import LANDCOVER_LABELS, WET_SNOW_LABELS
from openamundsen_da.methods.viz.project_maps.theme import (
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
    _HORIZONTAL_COLORBAR_EXTRA,
    _HORIZONTAL_COLORBAR_GAP_AXES,
    _HORIZONTAL_COLORBAR_HEIGHT_AXES,
    _HORIZONTAL_LEGEND_BOTTOM_PAD_AXES,
    _HORIZONTAL_LEGEND_GAP_AXES,
    _HORIZONTAL_LEGEND_HANDLE_TEXT_PAD_IN,
    _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN,
    _HORIZONTAL_LEGEND_ITEM_GAP_IN,
    _HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN,
    _HORIZONTAL_LEGEND_MIN_TEXT_WIDTH_IN,
    _HORIZONTAL_LEGEND_PATCH_HEIGHT_IN,
    _HORIZONTAL_LEGEND_ROW_HEIGHT_AXES,
    _HORIZONTAL_LEGEND_SIDE_PAD_IN,
    _HORIZONTAL_LEGEND_TEXT_SIZE,
    _LAYOUT_COL_GAP,
    _LAYOUT_ROW_GAP,
    _LEFT_MARGIN,
    _MODEL_KIND_TO_VARIABLE,
    _RIGHT_MARGIN,
    _SPINE_WIDTH,
    _SUPPORT_PANEL_KINDS,
    _TICK_SIZE,
    _TOP_MARGIN,
    _VERTICAL_COLORBAR_BOTTOM_AXES,
    _VERTICAL_COLORBAR_GAP_EXTRA,
    _VERTICAL_COLORBAR_HEIGHT_AXES,
    _VERTICAL_COLORBAR_OUTER_EXTRA,
    _VERTICAL_COLORBAR_WIDTH_AXES,
    _VERTICAL_COLORBAR_XOFFSET_AXES,
)


def buffered_extent(context: StaticContext) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = context.roi_gdf.total_bounds
    dx = float(maxx - minx)
    dy = float(maxy - miny)
    cell = abs(float(context.spec.transform.a))
    pad_x = max(dx * _BUFFER_RATIO, 2.0 * cell)
    pad_y = max(dy * _BUFFER_RATIO, 2.0 * cell)
    return (minx - pad_x, maxx + pad_x, miny - pad_y, maxy + pad_y)


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
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([coord_label(value) if idx % 2 == 0 else "" for idx, value in enumerate(xticks)])
    ax.set_yticklabels(
        [coord_label(value) if show_y_ticklabels and idx % 2 == 0 else "" for idx, value in enumerate(yticks)]
    )
    ax.tick_params(
        axis="x",
        direction="out",
        top=False,
        bottom=True,
        labeltop=False,
        labelbottom=True,
        labelsize=_TICK_SIZE,
        length=3.0,
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
        length=3.0,
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


def horizontal_legend_total_extra(rows: list[list[str]], *, panel_width_in: float) -> float:
    if not rows:
        return 0.0
    total_row_units = float(sum(horizontal_legend_row_height_factors(rows)))
    return _HORIZONTAL_LEGEND_GAP_AXES + total_row_units * _HORIZONTAL_LEGEND_ROW_HEIGHT_AXES + horizontal_legend_bottom_pad(panel_width_in)


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
    figure_horizontal_default: bool,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
    classified_labels_getter,
    below_items_extra_getter,
) -> dict[int, float]:
    row_extras = {row: 0.0 for row in range(recipe.layout.nrows)}
    for panel in recipe.panels:
        row = int(panel.row + panel.rowspan - 1)
        extra = 0.0
        if panel.kind in _CLASSIFIED_PANEL_KINDS:
            layout = panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default)
            if layout == "horizontal":
                labels = classified_labels_getter(panel, context=context, defaults=recipe.defaults, obs_cache=obs_cache)
                rows = pack_horizontal_legend_rows(labels, panel_width_in=panel_width_in)
                extra = horizontal_legend_total_extra(rows, panel_width_in=panel_width_in)
                extra = max(extra - panel_empty_below_units(recipe, panel), 0.0)
        elif panel.kind in _CONTINUOUS_COLORBAR_PANEL_KINDS:
            if (
                resolve_flag(panel.show_colorbar, recipe.defaults, "show_colorbar", True)
                and panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True) == "horizontal"
            ):
                extra = _HORIZONTAL_COLORBAR_EXTRA
        if panel.below_items:
            below_items_extra = below_items_extra_getter(panel.below_items)
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
        cax = ax.inset_axes(
            [
                0.06,
                -(_HORIZONTAL_COLORBAR_GAP_AXES + _HORIZONTAL_COLORBAR_HEIGHT_AXES),
                0.88,
                _HORIZONTAL_COLORBAR_HEIGHT_AXES,
            ],
            transform=ax.transAxes,
        )
        register_child_axes(ax, cax)
        cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
        if ticks:
            cbar.set_ticks(ticks)
        if ticklabels:
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE, length=2.0, width=0.65)
        title = extract_unit_title(label)
        if title:
            cbar.ax.text(1.01, 0.5, title, transform=cbar.ax.transAxes, ha="left", va="center", fontsize=_COLORBAR_TITLE_SIZE)
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
    cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE, length=2.2, width=0.7)
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
    classified_labels_getter,
    below_items_extra_getter,
) -> tuple[float, float]:
    width = max(1.0, float(extent[1] - extent[0]))
    height = max(1.0, float(extent[3] - extent[2]))
    aspect = height / width
    height_factors = effective_height_ratios(recipe)
    figure_horizontal_default = figure_prefers_horizontal_legends(recipe)
    panel_width = panel_width_in_for_recipe(recipe, figure_horizontal_default=figure_horizontal_default)
    bottom_extras = row_bottom_extras(
        recipe,
        context=context,
        panel_width_in=panel_width,
        figure_horizontal_default=figure_horizontal_default,
        obs_cache=obs_cache,
        classified_labels_getter=classified_labels_getter,
        below_items_extra_getter=below_items_extra_getter,
    )
    inter_row_gap_factors = tuple(_LAYOUT_ROW_GAP + bottom_extras[row] for row in range(recipe.layout.nrows - 1))
    bottom_extra_factor = bottom_extras.get(recipe.layout.nrows - 1, 0.0)
    panel_height = panel_width * aspect * 1.02
    inner_height = (
        panel_height * float(sum(height_factors))
        + panel_height * float(sum(inter_row_gap_factors))
        + panel_height * bottom_extra_factor
    )
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


def shift_axes_group(axes_group, delta_y: float) -> None:
    if delta_y <= 0.0:
        return
    for ax in axes_group:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + delta_y, pos.width, pos.height])


def tighten_panel_row_gaps(fig, row_axes: dict[int, list]) -> None:
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
        if upper_bbox is None or lower_bbox is None:
            continue
        current_gap = upper_bbox[0] - lower_bbox[1]
        mean_panel_height = float(np.mean([ax.get_position().height for ax in row_axes[upper_row]]))
        target_gap = max(0.0, _LAYOUT_ROW_GAP * mean_panel_height)
        delta_y = current_gap - target_gap
        if delta_y <= 1e-6:
            continue
        for row in lower_rows:
            shift_axes_group(row_axes[row], delta_y)
        canvas.draw()
        renderer = canvas.get_renderer()
