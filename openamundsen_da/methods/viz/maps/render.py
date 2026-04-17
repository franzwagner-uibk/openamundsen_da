from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_lowercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

from openamundsen_da.methods.viz.common import force_figure_text_black, save_figure_png
import openamundsen_da.methods.viz.maps.theme as _theme
import openamundsen_da.methods.viz.maps.annotations as _annotations
import openamundsen_da.methods.viz.maps.hillshade as _hillshade_mod
import openamundsen_da.methods.viz.maps.layout as _layout
import openamundsen_da.methods.viz.maps.panel_renderers as _panels
from openamundsen_da.methods.viz.maps.config import MapRecipe
from openamundsen_da.methods.viz.maps.data import ModelFields, ObservationScene, StaticContext


for _name in dir(_theme):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_theme, _name)

_LAYOUT_ROW_GAP = _theme._LAYOUT_ROW_GAP
_LEFT_MARGIN = _theme._LEFT_MARGIN
_RIGHT_MARGIN = _theme._RIGHT_MARGIN
_BOTTOM_MARGIN = _theme._BOTTOM_MARGIN
_TOP_MARGIN = _theme._TOP_MARGIN
_SUPPORT_PANEL_KINDS = _theme._SUPPORT_PANEL_KINDS
_MODEL_KIND_TO_VARIABLE = _theme._MODEL_KIND_TO_VARIABLE
_OBSERVATION_KIND_TO_NAME = _theme._OBSERVATION_KIND_TO_NAME


@dataclass
class RenderRuntimeCache:
    model_fields: dict[tuple[str, pd.Timestamp], ModelFields] = field(default_factory=dict)
    scale_cache: dict[tuple[str, pd.Timestamp], tuple[object, object]] = field(default_factory=dict)
    observations: dict[tuple[str, pd.Timestamp], ObservationScene] = field(default_factory=dict)
    derived_arrays: dict[str, np.ndarray] = field(default_factory=dict)


buffered_extent = _layout.buffered_extent
figure_height_for_extent = _layout.figure_height_for_extent

_grid_extent = _hillshade_mod.grid_extent
_hillshade_extent = _hillshade_mod.hillshade_extent
_hillshade = _hillshade_mod.hillshade
_hillshade_underlay = _hillshade_mod.hillshade_underlay

load_model_fields = _panels.load_model_fields
load_observation_scene = _panels.load_observation_scene
load_overview_boundaries = _panels.load_overview_boundaries
load_overview_regions = _panels.load_overview_regions
load_overview_labels = _panels.load_overview_labels

_masked = _panels.masked
_masked_invalid = _panels.masked_invalid
_masked_model = _panels.masked_model
_field_array = _panels.field_array
_draw_roi = _panels.draw_roi
_draw_stations_overlay = _panels.draw_stations_overlay
_comparison_scales = _panels.comparison_scales
_classified_display_labels = _panels.classified_display_labels
_classified_legend_handles = _panels.classified_legend_handles
_draw_classified_legend = _panels.draw_classified_legend
_overview_extent = _panels.overview_extent
_overview_extent_with_label_fit = _panels.overview_extent_with_label_fit
def _render_overview_panel(ax, *, panel, context, label, defaults):
    _panels.load_overview_boundaries = load_overview_boundaries
    _panels.load_overview_regions = load_overview_regions
    _panels.load_overview_labels = load_overview_labels
    return _panels.render_overview_panel(ax, panel=panel, context=context, label=label, defaults=defaults)


_render_roi_panel = _panels.render_roi_panel


def _render_static_panel(
    ax,
    *,
    panel,
    context,
    extent,
    grid_extent=None,
    label,
    defaults,
    figure_horizontal_default,
    derived_cache=None,
):
    del grid_extent
    return _panels.render_static_panel(
        ax,
        panel=panel,
        context=context,
        extent=extent,
        label=label,
        defaults=defaults,
        figure_horizontal_default=figure_horizontal_default,
        derived_cache=derived_cache,
    )


def _render_model_panel(
    ax,
    *,
    panel,
    context,
    extent,
    grid_extent=None,
    label,
    defaults,
    model_cache,
    scale_cache,
    figure_horizontal_default,
    derived_cache=None,
):
    del grid_extent
    _panels.load_model_fields = load_model_fields
    return _panels.render_model_panel(
        ax,
        panel=panel,
        context=context,
        extent=extent,
        label=label,
        defaults=defaults,
        model_cache=model_cache,
        scale_cache=scale_cache,
        figure_horizontal_default=figure_horizontal_default,
        derived_cache=derived_cache,
    )


def _render_observation_panel(
    ax,
    *,
    panel,
    context,
    extent,
    label,
    defaults,
    obs_cache,
    figure_horizontal_default,
    derived_cache=None,
):
    _panels.load_observation_scene = load_observation_scene
    return _panels.render_observation_panel(
        ax,
        panel=panel,
        context=context,
        extent=extent,
        label=label,
        defaults=defaults,
        obs_cache=obs_cache,
        figure_horizontal_default=figure_horizontal_default,
        derived_cache=derived_cache,
    )
_render_colorbar_panel = _panels.render_colorbar_panel
_render_legend_panel = _panels.render_legend_panel
_legend_source_handles = _panels.legend_source_handles
_draw_patch_entry = _annotations.draw_patch_entry
_draw_station_entry = _annotations.draw_station_entry
_draw_heading = _annotations.draw_heading
_panel_below_item_units = _annotations.panel_below_item_units
_panel_below_items_layout = _annotations.panel_below_items_layout
_panel_below_items_extra = _annotations.panel_below_items_extra
_draw_scale_bar = _annotations.draw_scale_bar
_scale_bar_length_m = _annotations.scale_bar_length_m
_draw_panel_date = _annotations.draw_panel_date
_panel_date_text = _annotations.panel_date_text
_panel_date = _annotations.panel_date
_panel_semantic_title = _annotations.panel_semantic_title
_panel_title = _annotations.panel_title
_draw_overview_label_specs = _annotations.draw_overview_label_specs

_apply_map_axis_style = _layout.apply_map_axis_style
_draw_map_grid_overlay = _layout.draw_map_grid_overlay
_resolve_flag = _layout.resolve_flag
_resolve_panel_toggle = _layout.resolve_panel_toggle
_figure_prefers_horizontal_legends = _layout.figure_prefers_horizontal_legends
_panel_legend_layout = _layout.panel_legend_layout
_effective_width_ratios = _layout.effective_width_ratios
_effective_height_ratios = _layout.effective_height_ratios
_horizontal_legend_item_width_in = _layout.horizontal_legend_item_width_in
_horizontal_legend_side_pad_in = _layout.horizontal_legend_side_pad_in
_horizontal_legend_available_width_in = _layout.horizontal_legend_available_width_in
_pack_horizontal_legend_rows = _layout.pack_horizontal_legend_rows
_horizontal_legend_row_height_factors = _layout.horizontal_legend_row_height_factors
_horizontal_legend_row_layout = _layout.horizontal_legend_row_layout
_classified_legend_labels = _layout.classified_legend_labels
_horizontal_legend_bottom_pad = _layout.horizontal_legend_bottom_pad
_horizontal_legend_total_extra = _layout.horizontal_legend_total_extra
_panel_has_vertical_colorbar = _layout.panel_has_vertical_colorbar
_column_gap_factors = _layout.column_gap_factors
_outer_right_factor = _layout.outer_right_factor
_panel_empty_below_units = _layout.panel_empty_below_units
_attach_colorbar = _layout.attach_colorbar
_register_child_axes = _layout.register_child_axes
_panel_width_in_for_recipe = _layout.panel_width_in_for_recipe
_expanded_grid_ratios = _layout.expanded_grid_ratios
_grid_span = _layout.grid_span
_axes_group_bbox = _layout.axes_group_bbox
_shift_axes_group = _layout.shift_axes_group
_tighten_panel_row_gaps = _layout.tighten_panel_row_gaps
_axis_width_inches = _layout.axis_width_inches
_axis_height_inches = _layout.axis_height_inches
_text_width_in = _layout.text_width_in
_text_size_in = _layout.text_size_in
_axes_title_fontsize = _layout.axes_title_fontsize
_axes_date_fontsize = _layout.axes_date_fontsize


def _row_bottom_extras(
    recipe: MapRecipe,
    *,
    context: StaticContext,
    panel_width_in: float,
    figure_horizontal_default: bool,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
) -> dict[int, float]:
    return _layout.row_bottom_extras(
        recipe,
        context=context,
        panel_width_in=panel_width_in,
        figure_horizontal_default=figure_horizontal_default,
        obs_cache=obs_cache,
        classified_labels_getter=_classified_display_labels,
        below_items_extra_getter=_panel_below_items_extra,
    )


def _figure_size(
    extent: tuple[float, float, float, float],
    recipe: MapRecipe,
    *,
    context: StaticContext,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
) -> tuple[float, float]:
    return _layout.figure_size(
        extent,
        recipe,
        context=context,
        obs_cache=obs_cache,
        classified_labels_getter=_classified_display_labels,
        below_items_extra_getter=_panel_below_items_extra,
    )


def _draw_panel_below_items(ax, *, panel, artifacts: dict[str, dict[str, object]]) -> None:
    _annotations.draw_panel_below_items(
        ax,
        panel=panel,
        artifacts=artifacts,
        legend_source_handles_getter=_legend_source_handles,
    )


def render_map_recipe(
    *,
    project_dir: Path,
    context: StaticContext,
    recipe: MapRecipe,
    output_path: Path,
    runtime_cache: RenderRuntimeCache | None = None,
) -> Path:
    del project_dir
    cache = runtime_cache or RenderRuntimeCache()
    extent = buffered_extent(context)
    figure_horizontal_default = _figure_prefers_horizontal_legends(recipe)
    fig = plt.figure(figsize=_figure_size(extent, recipe, context=context, obs_cache=cache.observations))
    width_ratios = _effective_width_ratios(recipe)
    height_ratios = _effective_height_ratios(recipe)
    col_gap_factors = _column_gap_factors(recipe, figure_horizontal_default=figure_horizontal_default)
    row_bottom_extras = _row_bottom_extras(
        recipe,
        context=context,
        panel_width_in=_panel_width_in_for_recipe(recipe, figure_horizontal_default=figure_horizontal_default),
        figure_horizontal_default=figure_horizontal_default,
        obs_cache=cache.observations,
    )
    row_gap_factors = tuple(_LAYOUT_ROW_GAP + row_bottom_extras[row] for row in range(recipe.layout.nrows - 1))
    gs = fig.add_gridspec(
        max(1, recipe.layout.nrows * 2 - 1),
        max(1, recipe.layout.ncols * 2 - 1),
        width_ratios=_expanded_grid_ratios(width_ratios, col_gap_factors),
        height_ratios=_expanded_grid_ratios(height_ratios, row_gap_factors),
        wspace=0.0,
        hspace=0.0,
    )
    fig.subplots_adjust(left=_LEFT_MARGIN, right=_RIGHT_MARGIN, bottom=_BOTTOM_MARGIN, top=_TOP_MARGIN)

    title_letters = iter(ascii_lowercase)
    panel_labels: list[str | None] = []
    for panel in recipe.panels:
        if panel.kind in _SUPPORT_PANEL_KINDS:
            panel_labels.append(None)
        else:
            panel_labels.append(next(title_letters))

    artifacts: dict[str, dict[str, object]] = {}
    axes: list = []
    row_axes: dict[int, list] = {row: [] for row in range(recipe.layout.nrows)}

    for idx, panel in enumerate(recipe.panels):
        ax = fig.add_subplot(
            gs[
                _grid_span(panel.row, panel.rowspan),
                _grid_span(panel.col, panel.colspan),
            ]
        )
        axes.append(ax)
        row_axes[int(panel.row)].append(ax)
        key = panel.name or f"panel_{idx}"
        if panel.kind == "overview":
            artifacts[key] = _render_overview_panel(ax, panel=panel, context=context, label=panel_labels[idx], defaults=recipe.defaults)
        elif panel.kind == "roi":
            artifacts[key] = _render_roi_panel(ax, panel=panel, context=context, extent=extent, label=panel_labels[idx], defaults=recipe.defaults)
        elif panel.kind in {"hillshade", "dem", "svf", "srf", "landcover"}:
            artifacts[key] = _render_static_panel(
                ax,
                panel=panel,
                context=context,
                extent=extent,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                figure_horizontal_default=figure_horizontal_default,
                derived_cache=cache.derived_arrays,
            )
        elif panel.kind in _MODEL_KIND_TO_VARIABLE:
            artifacts[key] = _render_model_panel(
                ax,
                panel=panel,
                context=context,
                extent=extent,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                model_cache=cache.model_fields,
                scale_cache=cache.scale_cache,
                figure_horizontal_default=figure_horizontal_default,
                derived_cache=cache.derived_arrays,
            )
        elif panel.kind in _OBSERVATION_KIND_TO_NAME:
            artifacts[key] = _render_observation_panel(
                ax,
                panel=panel,
                context=context,
                extent=extent,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                obs_cache=cache.observations,
                figure_horizontal_default=figure_horizontal_default,
                derived_cache=cache.derived_arrays,
            )
        elif panel.kind == "colorbar":
            artifacts[key] = _render_colorbar_panel(ax, panel=panel, artifacts=artifacts)
        elif panel.kind == "legend":
            artifacts[key] = _render_legend_panel(ax, panel=panel, artifacts=artifacts)
        else:
            raise ValueError(f"Unsupported panel kind '{panel.kind}'")
        if panel.below_items:
            _draw_panel_below_items(ax, panel=panel, artifacts=artifacts)

    _tighten_panel_row_gaps(fig, row_axes)
    force_figure_text_black(fig, axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output_path


__all__ = [
    "_apply_map_axis_style",
    "_comparison_scales",
    "_draw_map_grid_overlay",
    "_draw_scale_bar",
    "_draw_stations_overlay",
    "_masked_model",
    "RenderRuntimeCache",
    "buffered_extent",
    "figure_height_for_extent",
    "render_map_recipe",
]
