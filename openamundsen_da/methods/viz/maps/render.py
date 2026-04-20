from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from string import ascii_lowercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openamundsen_da.methods.viz.common import force_figure_text_black, save_figure_png
from openamundsen_da.methods.viz.maps.annotations import (
    draw_heading as _draw_heading,
    draw_overview_label_specs as _draw_overview_label_specs,
    draw_panel_below_items,
    draw_panel_date as _draw_panel_date,
    draw_patch_entry as _draw_patch_entry,
    draw_scale_bar as _draw_scale_bar,
    draw_station_entry as _draw_station_entry,
    panel_below_item_units as _panel_below_item_units,
    panel_below_items_extra as _panel_below_items_extra,
    panel_below_items_layout as _panel_below_items_layout,
    panel_date as _panel_date,
    panel_date_text as _panel_date_text,
    panel_semantic_title as _panel_semantic_title,
    panel_title as _panel_title,
    scale_bar_length_m as _scale_bar_length_m,
)
from openamundsen_da.methods.viz.maps.config import MapRecipe
from openamundsen_da.methods.viz.maps.data import (
    ModelFields,
    ObservationScene,
    StaticContext,
    load_model_fields,
    load_observation_scene,
)
from openamundsen_da.methods.viz.maps.hillshade import (
    grid_extent as _grid_extent,
    hillshade as _hillshade,
    hillshade_extent as _hillshade_extent,
    hillshade_underlay as _hillshade_underlay,
)
from openamundsen_da.methods.viz.maps.layout import (
    apply_map_axis_style as _apply_map_axis_style,
    attach_colorbar as _attach_colorbar,
    axes_date_fontsize as _axes_date_fontsize,
    axes_group_bbox as _axes_group_bbox,
    axes_title_fontsize as _axes_title_fontsize,
    axis_height_inches as _axis_height_inches,
    axis_width_inches as _axis_width_inches,
    buffered_extent,
    classified_legend_labels as _classified_legend_labels,
    column_gap_factors as _column_gap_factors,
    draw_map_grid_overlay as _draw_map_grid_overlay,
    effective_height_ratios as _effective_height_ratios,
    effective_width_ratios as _effective_width_ratios,
    expanded_grid_ratios as _expanded_grid_ratios,
    figure_height_for_extent,
    figure_prefers_horizontal_legends as _figure_prefers_horizontal_legends,
    figure_size,
    grid_span as _grid_span,
    horizontal_legend_available_width_in as _horizontal_legend_available_width_in,
    horizontal_legend_bottom_pad as _horizontal_legend_bottom_pad,
    horizontal_legend_item_width_in as _horizontal_legend_item_width_in,
    horizontal_legend_row_height_factors as _horizontal_legend_row_height_factors,
    horizontal_legend_row_layout as _horizontal_legend_row_layout,
    horizontal_legend_side_pad_in as _horizontal_legend_side_pad_in,
    horizontal_legend_total_extra as _horizontal_legend_total_extra,
    outer_right_factor as _outer_right_factor,
    pack_horizontal_legend_rows as _pack_horizontal_legend_rows,
    panel_empty_below_units as _panel_empty_below_units,
    panel_has_vertical_colorbar as _panel_has_vertical_colorbar,
    panel_legend_layout as _panel_legend_layout,
    panel_width_in_for_recipe as _panel_width_in_for_recipe,
    register_child_axes as _register_child_axes,
    resolve_flag as _resolve_flag,
    resolve_panel_toggle as _resolve_panel_toggle,
    row_bottom_extras,
    shift_axes_group as _shift_axes_group,
    text_size_in as _text_size_in,
    text_width_in as _text_width_in,
    tighten_panel_row_gaps as _tighten_panel_row_gaps,
)
from openamundsen_da.methods.viz.maps.overview import (
    load_overview_boundaries,
    load_overview_labels,
    load_overview_regions,
)
from openamundsen_da.methods.viz.maps.panel_renderers import (
    classified_display_labels,
    classified_legend_handles as _classified_legend_handles,
    comparison_scales as _comparison_scales,
    draw_classified_legend as _draw_classified_legend,
    draw_roi as _draw_roi,
    draw_stations_overlay as _draw_stations_overlay,
    field_array as _field_array,
    legend_source_handles as _legend_source_handles,
    masked as _masked,
    masked_invalid as _masked_invalid,
    masked_model as _masked_model,
    overview_extent as _overview_extent,
    overview_extent_with_label_fit as _overview_extent_with_label_fit,
    render_colorbar_panel as _render_colorbar_panel,
    render_legend_panel as _render_legend_panel,
    render_model_panel,
    render_observation_panel,
    render_overview_panel,
    render_roi_panel as _render_roi_panel,
    render_static_panel,
)
from openamundsen_da.methods.viz.maps.theme import (
    _BOTTOM_MARGIN,
    _HORIZONTAL_COLORBAR_GAP_AXES,
    _HORIZONTAL_LEGEND_ITEM_GAP_IN,
    _HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN,
    _LAYOUT_ROW_GAP,
    _LEFT_MARGIN,
    _MODEL_KIND_TO_VARIABLE,
    _OBSERVATION_KIND_TO_NAME,
    _RIGHT_MARGIN,
    _SUPPORT_PANEL_KINDS,
    _TOP_MARGIN,
)


@dataclass
class RenderRuntimeCache:
    model_fields: dict[tuple[str, pd.Timestamp], ModelFields] = field(default_factory=dict)
    scale_cache: dict[tuple[str, pd.Timestamp], tuple[object, object]] = field(default_factory=dict)
    shared_model_vmax: dict[str, float] = field(default_factory=dict)
    observations: dict[tuple[str, pd.Timestamp], ObservationScene] = field(default_factory=dict)
    derived_arrays: dict[str, np.ndarray] = field(default_factory=dict)


def _classified_display_labels(
    panel,
    *,
    context,
    defaults,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
) -> list[str]:
    return classified_display_labels(
        panel,
        context=context,
        defaults=defaults,
        obs_cache=obs_cache,
        observation_loader=load_observation_scene,
    )


def _row_bottom_extras(
    recipe: MapRecipe,
    *,
    context: StaticContext,
    panel_width_in: float,
    figure_horizontal_default: bool,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
) -> dict[int, float]:
    return row_bottom_extras(
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
    return figure_size(
        extent,
        recipe,
        context=context,
        obs_cache=obs_cache,
        classified_labels_getter=_classified_display_labels,
        below_items_extra_getter=_panel_below_items_extra,
    )


def _draw_panel_below_items(ax, *, panel, artifacts: dict[str, dict[str, object]]) -> None:
    draw_panel_below_items(
        ax,
        panel=panel,
        artifacts=artifacts,
        legend_source_handles_getter=_legend_source_handles,
    )


def _render_overview_panel(ax, *, panel, context, label, defaults):
    return render_overview_panel(
        ax,
        panel=panel,
        context=context,
        label=label,
        defaults=defaults,
        boundaries_loader=load_overview_boundaries,
        regions_loader=load_overview_regions,
        labels_loader=load_overview_labels,
    )


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
    return render_static_panel(
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
    shared_model_vmax=None,
    figure_horizontal_default,
    derived_cache=None,
):
    del grid_extent
    return render_model_panel(
        ax,
        panel=panel,
        context=context,
        extent=extent,
        label=label,
        defaults=defaults,
        model_cache=model_cache,
        scale_cache=scale_cache,
        shared_model_vmax=shared_model_vmax,
        figure_horizontal_default=figure_horizontal_default,
        derived_cache=derived_cache,
        model_loader=load_model_fields,
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
    return render_observation_panel(
        ax,
        panel=panel,
        context=context,
        extent=extent,
        label=label,
        defaults=defaults,
        obs_cache=obs_cache,
        figure_horizontal_default=figure_horizontal_default,
        derived_cache=derived_cache,
        observation_loader=load_observation_scene,
    )


def _render_panel(
    *,
    ax,
    panel,
    context: StaticContext,
    label: str | None,
    defaults,
    extent: tuple[float, float, float, float],
    cache: RenderRuntimeCache,
    figure_horizontal_default: bool,
) -> dict[str, object]:
    if panel.kind == "overview":
        return _render_overview_panel(ax, panel=panel, context=context, label=label, defaults=defaults)
    if panel.kind == "roi":
        return _render_roi_panel(ax, panel=panel, context=context, extent=extent, label=label, defaults=defaults)
    if panel.kind in {"hillshade", "dem", "svf", "srf", "landcover"}:
        return _render_static_panel(
            ax,
            panel=panel,
            context=context,
            extent=extent,
            label=label,
            defaults=defaults,
            figure_horizontal_default=figure_horizontal_default,
            derived_cache=cache.derived_arrays,
        )
    if panel.kind in _MODEL_KIND_TO_VARIABLE:
        return _render_model_panel(
            ax,
            panel=panel,
            context=context,
            extent=extent,
            label=label,
            defaults=defaults,
            model_cache=cache.model_fields,
            scale_cache=cache.scale_cache,
            shared_model_vmax=cache.shared_model_vmax,
            figure_horizontal_default=figure_horizontal_default,
            derived_cache=cache.derived_arrays,
        )
    if panel.kind in _OBSERVATION_KIND_TO_NAME:
        return _render_observation_panel(
            ax,
            panel=panel,
            context=context,
            extent=extent,
            label=label,
            defaults=defaults,
            obs_cache=cache.observations,
            figure_horizontal_default=figure_horizontal_default,
            derived_cache=cache.derived_arrays,
        )
    if panel.kind == "colorbar":
        raise RuntimeError("Support panels require precomputed artifacts")
    if panel.kind == "legend":
        raise RuntimeError("Support panels require precomputed artifacts")
    raise ValueError(f"Unsupported panel kind '{panel.kind}'")


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
    panel_labels: list[str | None] = [
        None if panel.kind in _SUPPORT_PANEL_KINDS else next(title_letters)
        for panel in recipe.panels
    ]

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

        if panel.kind == "colorbar":
            artifacts[key] = _render_colorbar_panel(ax, panel=panel, artifacts=artifacts)
        elif panel.kind == "legend":
            artifacts[key] = _render_legend_panel(ax, panel=panel, artifacts=artifacts)
        else:
            artifacts[key] = _render_panel(
                ax=ax,
                panel=panel,
                context=context,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                extent=extent,
                cache=cache,
                figure_horizontal_default=figure_horizontal_default,
            )

        if panel.below_items:
            _draw_panel_below_items(ax, panel=panel, artifacts=artifacts)

    _tighten_panel_row_gaps(fig, row_axes)
    force_figure_text_black(fig, axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    return output_path


__all__ = [
    "_HORIZONTAL_COLORBAR_GAP_AXES",
    "_HORIZONTAL_LEGEND_ITEM_GAP_IN",
    "_HORIZONTAL_LEGEND_MIN_ITEM_GAP_IN",
    "_apply_map_axis_style",
    "_attach_colorbar",
    "_axes_date_fontsize",
    "_axes_group_bbox",
    "_axes_title_fontsize",
    "_axis_height_inches",
    "_axis_width_inches",
    "_classified_display_labels",
    "_classified_legend_handles",
    "_classified_legend_labels",
    "_column_gap_factors",
    "_comparison_scales",
    "_draw_classified_legend",
    "_draw_heading",
    "_draw_map_grid_overlay",
    "_draw_overview_label_specs",
    "_draw_panel_date",
    "_draw_patch_entry",
    "_draw_roi",
    "_draw_scale_bar",
    "_draw_station_entry",
    "_draw_stations_overlay",
    "_effective_height_ratios",
    "_effective_width_ratios",
    "_expanded_grid_ratios",
    "_figure_prefers_horizontal_legends",
    "_field_array",
    "_grid_extent",
    "_grid_span",
    "_hillshade",
    "_hillshade_extent",
    "_hillshade_underlay",
    "_horizontal_legend_available_width_in",
    "_horizontal_legend_bottom_pad",
    "_horizontal_legend_item_width_in",
    "_horizontal_legend_row_height_factors",
    "_horizontal_legend_row_layout",
    "_horizontal_legend_side_pad_in",
    "_horizontal_legend_total_extra",
    "_legend_source_handles",
    "_masked",
    "_masked_invalid",
    "_masked_model",
    "_outer_right_factor",
    "_overview_extent",
    "_overview_extent_with_label_fit",
    "_panel_below_item_units",
    "_panel_below_items_extra",
    "_panel_below_items_layout",
    "_panel_empty_below_units",
    "_panel_has_vertical_colorbar",
    "_panel_legend_layout",
    "_panel_title",
    "_panel_width_in_for_recipe",
    "_pack_horizontal_legend_rows",
    "_register_child_axes",
    "_render_colorbar_panel",
    "_render_legend_panel",
    "_render_model_panel",
    "_render_observation_panel",
    "_render_overview_panel",
    "_render_roi_panel",
    "_render_static_panel",
    "_resolve_flag",
    "_resolve_panel_toggle",
    "_row_bottom_extras",
    "_scale_bar_length_m",
    "_shift_axes_group",
    "_text_size_in",
    "_text_width_in",
    "RenderRuntimeCache",
    "buffered_extent",
    "figure_height_for_extent",
    "load_model_fields",
    "load_observation_scene",
    "load_overview_boundaries",
    "load_overview_labels",
    "load_overview_regions",
    "render_map_recipe",
]
