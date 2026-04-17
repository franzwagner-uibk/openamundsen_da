from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import box

from openamundsen_da.methods.viz.maps.annotations import (
    draw_heading,
    draw_overview_label_specs,
    draw_panel_below_items,
    draw_panel_extras,
    draw_patch_entry,
    draw_station_entry,
    overview_label_box_size_in,
    panel_date,
    panel_semantic_title,
    panel_title,
)
from openamundsen_da.methods.viz.maps.config import LegendItemSpec, MapDefaults, MapPanelSpec
from openamundsen_da.methods.viz.maps.data import (
    ModelFields,
    ObservationScene,
    StaticContext,
    load_model_fields,
    load_observation_scene,
)
from openamundsen_da.methods.viz.maps.hillshade import grid_extent, hillshade, hillshade_extent, hillshade_underlay
from openamundsen_da.methods.viz.maps.layout import (
    apply_map_axis_style,
    attach_colorbar,
    axis_height_inches,
    axis_width_inches,
    buffered_extent,
    draw_axes_title,
    draw_map_grid_overlay,
    extract_unit_title,
    horizontal_legend_row_height_factors,
    horizontal_legend_row_layout,
    pack_horizontal_legend_rows,
    panel_legend_layout,
    register_child_axes,
    resolve_flag,
    resolve_panel_toggle,
    text_size_in,
)
from openamundsen_da.methods.viz.maps.overview import (
    load_overview_boundaries,
    load_overview_labels,
    load_overview_regions,
)
from openamundsen_da.methods.viz.maps.styles import (
    FSC_OBS_CMAP,
    FSC_INVALID_COLOR,
    INCREMENT_CMAP,
    LANDCOVER_LABELS,
    SNOW_DEPTH_REFERENCE_TICKS_M,
    WET_SNOW_COLORS,
    WET_SNOW_LABELS,
    landcover_cmap_for_codes,
    model_colorbar_style,
    model_map_cmap,
    model_map_norm,
    nice_ceiling,
    require_static_field_preset,
    require_variable_preset,
    static_field_cmap,
    static_field_colorbar_style,
    static_field_norm,
)
from openamundsen_da.methods.viz.maps.theme import (
    _ANNOTATION_ZORDER,
    _CLASSIFIED_PANEL_KINDS,
    _COLORBAR_TICK_SIZE,
    _COLORBAR_TITLE_SIZE,
    _GRID_ZORDER,
    _HILLSHADE_INTERPOLATION,
    _HORIZONTAL_LEGEND_GAP_AXES,
    _HORIZONTAL_LEGEND_HANDLE_TEXT_PAD_IN,
    _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN,
    _HORIZONTAL_LEGEND_PATCH_HEIGHT_IN,
    _HORIZONTAL_LEGEND_ROW_HEIGHT_AXES,
    _HORIZONTAL_LEGEND_TEXT_SIZE,
    _MODEL_KIND_TO_VARIABLE,
    _OBSERVATION_KIND_TO_NAME,
    _OVERVIEW_FRAGMENT_RATIO,
    _OVERVIEW_LABEL_DX_RATIO,
    _OVERVIEW_LABEL_DY_RATIO,
    _OVERVIEW_LABEL_SIZE,
    _OVERVIEW_ROI_COLOR,
    _OVERVIEW_ROI_LABEL_DX_RATIO,
    _OVERVIEW_ROI_LABEL_SIZE,
    _ROI_FILL,
    _SNOW_DEPTH_PANEL_ALPHA,
    _STATIC_FIELD_KIND_TO_FIELD,
    _STATION_COLOR,
    _STATION_LABEL_RATIO,
)


@dataclass(frozen=True)
class OverviewLabelSpec:
    text: str
    x: float
    y: float
    ha: str
    va: str
    fontsize: float
    with_bbox: bool
    zorder: int


def masked(arr: np.ndarray, roi_mask: np.ndarray) -> np.ma.MaskedArray:
    masked_array = np.asarray(arr, dtype=float).copy()
    masked_array[~roi_mask] = np.nan
    return np.ma.masked_invalid(masked_array)


def masked_invalid(arr: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(arr, dtype=float))


def masked_model(arr: np.ndarray, roi_mask: np.ndarray, *, preset) -> np.ma.MaskedArray:
    masked_array = masked(arr, roi_mask)
    if preset.variable == "snowdepth_daily":
        masked_array = np.ma.masked_less(masked_array, SNOW_DEPTH_REFERENCE_TICKS_M[0])
    return masked_array


def field_array(context: StaticContext, field: str) -> np.ndarray:
    token = str(field).strip()
    if token == "dem":
        return context.dem
    if token == "landcover":
        return context.landcover
    if token == "svf":
        if context.svf is None:
            raise FileNotFoundError(f"Static field 'svf' is not available for setup {context.setup_dir}")
        return context.svf
    if token == "srf":
        if context.srf is None:
            raise FileNotFoundError(f"Static field 'srf' is not available for setup {context.setup_dir}")
        return context.srf
    raise ValueError(f"Unsupported static field '{field}'")


def draw_roi(ax, context: StaticContext, *, linewidth: float = 0.8, facecolor=None, alpha: float | None = None) -> None:
    if facecolor is not None:
        context.roi_gdf.plot(ax=ax, facecolor=facecolor, edgecolor=facecolor, alpha=alpha if alpha is not None else 1.0, zorder=40)
    context.roi_gdf.boundary.plot(ax=ax, color="black", linewidth=linewidth, zorder=45)


def suppress_station_labels(stations, extent: tuple[float, float, float, float]) -> list[int]:
    kept: list[int] = []
    min_dx = (extent[1] - extent[0]) * _STATION_LABEL_RATIO
    min_dy = (extent[3] - extent[2]) * _STATION_LABEL_RATIO
    working = stations.sort_values("id").reset_index(drop=True)
    for idx, row in working.iterrows():
        x = float(row["x"])
        y = float(row["y"])
        if any(abs(x - float(working.iloc[prev]["x"])) < min_dx and abs(y - float(working.iloc[prev]["y"])) < min_dy for prev in kept):
            continue
        kept.append(idx)
    return kept


def draw_stations_overlay(
    ax,
    context: StaticContext,
    extent: tuple[float, float, float, float],
    *,
    show_station_marker: bool,
    show_stations_name: bool,
    show_stations_elev: bool,
) -> None:
    stations = context.stations
    if stations is None or stations.empty:
        return

    station_x = stations["x"].astype(float)
    station_y = stations["y"].astype(float)
    inside = (
        station_x.between(extent[0], extent[1], inclusive="both")
        & station_y.between(extent[2], extent[3], inclusive="both")
    )
    visible = stations.loc[inside].copy()
    if visible.empty:
        return

    ordered = visible.sort_values("id").reset_index(drop=True)
    if show_station_marker:
        ax.scatter(
            ordered["x"].astype(float),
            ordered["y"].astype(float),
            marker="v",
            s=26,
            facecolor=_STATION_COLOR,
            edgecolor="white",
            linewidth=0.45,
            zorder=_GRID_ZORDER + 4,
            clip_on=True,
        )

    if not (show_station_marker and show_stations_name):
        return

    keep_indices = suppress_station_labels(ordered, extent)
    dx = 0.026 * (extent[1] - extent[0])
    dy = 0.013 * (extent[3] - extent[2])
    for idx in keep_indices:
        row = ordered.iloc[idx]
        label = str(row.get("name") or row.get("id") or "").strip()
        if show_stations_elev and "alt" in row and np.isfinite(float(row["alt"])):
            label = f"{label} ({int(round(float(row['alt'])))} m)"
        ax.text(
            float(row["x"]) + dx,
            float(row["y"]) + dy,
            label,
            fontsize=5.8,
            ha="left",
            va="bottom",
            color="black",
            zorder=_GRID_ZORDER + 5,
        )


def comparison_scales(fields: list[ModelFields], preset) -> tuple[Normalize, TwoSlopeNorm]:
    comparisons = [field for field in fields if field is not None]
    if not comparisons:
        raise ValueError("comparison_scales requires at least one model field")

    valid_arrays = []
    increment_arrays = []
    for field in comparisons:
        valid_arrays.extend([field.open_loop, field.ens_mean])
        increment_arrays.append(np.abs(field.increment))
    max_value = max(float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0 for arr in valid_arrays)
    max_increment = max(float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0 for arr in increment_arrays)

    vmax = nice_ceiling(max_value, step=preset.max_step, minimum=preset.max_floor)
    inc_abs = nice_ceiling(max_increment, step=preset.increment_step, minimum=preset.increment_floor)
    return model_map_norm(preset, vmax=vmax), TwoSlopeNorm(vcenter=0.0, vmin=-inc_abs, vmax=inc_abs)


def classified_display_labels(
    panel: MapPanelSpec,
    *,
    context: StaticContext,
    defaults: MapDefaults,
    obs_cache: dict[tuple[str, str], ObservationScene] | None = None,
    observation_loader: Callable[..., ObservationScene] = load_observation_scene,
) -> list[str]:
    if panel.kind == "landcover":
        masked_landcover = np.ma.masked_array(context.landcover, mask=~context.roi_mask)
        present_codes = {int(value) for value in masked_landcover.compressed() if np.isfinite(value)}
        active_codes = [code for code in LANDCOVER_LABELS if code in present_codes]
        if not active_codes:
            return list(LANDCOVER_LABELS.values())
        return [LANDCOVER_LABELS[code] for code in active_codes]

    if panel.kind == "wet_snow":
        date = panel_date(panel, defaults)
        if date is None:
            return [WET_SNOW_LABELS[code] for code in sorted(WET_SNOW_LABELS)]
        obs_key = ("wet_snow", pd.Timestamp(date).isoformat())
        scene = None
        if obs_cache is not None:
            scene = obs_cache.get(obs_key)
        if scene is None:
            scene = observation_loader(context.project_dir, context, observation="wet_snow", date=date)
            if obs_cache is not None:
                obs_cache[obs_key] = scene
        canonical_codes = sorted(WET_SNOW_LABELS)
        present_codes = {
            code
            for code in canonical_codes
            if np.any(scene.roi_mask & np.isclose(scene.array, float(code), equal_nan=False))
        }
        active_codes = [code for code in canonical_codes if code in present_codes]
        if not active_codes:
            return [WET_SNOW_LABELS[code] for code in canonical_codes]
        return [WET_SNOW_LABELS[code] for code in active_codes]

    raise ValueError(f"Unsupported classified panel kind '{panel.kind}'")


def resolve_hillshade_extent(panel: MapPanelSpec, defaults: MapDefaults, *, builtin: str) -> str:
    extent = panel.hillshade_extent
    if extent is None:
        extent = defaults.hillshade_extent
    return str(extent or builtin)


def classified_legend_handles(
    *,
    canonical_codes: list[int],
    present_codes: set[int],
    label_lookup: dict[int, str],
    color_lookup,
    fallback_codes: list[int],
) -> list[Patch]:
    active_codes = [code for code in canonical_codes if code in present_codes]
    if not active_codes:
        active_codes = list(fallback_codes)
    return [
        Patch(facecolor=color_lookup(code), edgecolor="none", label=label_lookup.get(code, str(code)))
        for code in active_codes
    ]


def draw_classified_legend(ax, handles: list[Patch], *, layout: str) -> None:
    if not handles:
        return
    if layout == "horizontal":
        labels = [handle.get_label() for handle in handles]

        panel_width_in = axis_width_inches(ax)
        rows = pack_horizontal_legend_rows(labels, panel_width_in=panel_width_in)
        if not rows:
            return
        row_height_factors = horizontal_legend_row_height_factors(rows)
        total_row_units = float(sum(row_height_factors))
        inset_height = total_row_units * _HORIZONTAL_LEGEND_ROW_HEIGHT_AXES
        legend_height_in = max(axis_height_inches(ax) * inset_height, 1e-9)
        legend_ax = ax.inset_axes(
            [0.0, -(_HORIZONTAL_LEGEND_GAP_AXES + inset_height), 1.0, inset_height],
            transform=ax.transAxes,
        )
        register_child_axes(ax, legend_ax)
        legend_ax.set_axis_off()
        handle_lookup = {handle.get_label(): handle for handle in handles}
        row_top = 1.0
        for row_labels, row_units in zip(rows, row_height_factors, strict=False):
            row_height = row_units / max(total_row_units, 1e-9)
            patch_height = min(_HORIZONTAL_LEGEND_PATCH_HEIGHT_IN / legend_height_in, 0.72 * row_height)
            item_widths, start_x_in, item_gap_in = horizontal_legend_row_layout(row_labels, panel_width_in=panel_width_in)
            y_center = row_top - 0.5 * row_height
            x_in = start_x_in
            for label, item_width in zip(row_labels, item_widths, strict=False):
                handle = handle_lookup[label]
                facecolor = handle.get_facecolor()
                edgecolor = handle.get_edgecolor()
                if np.ndim(facecolor) == 2:
                    facecolor = facecolor[0]
                if np.ndim(edgecolor) == 2:
                    edgecolor = edgecolor[0]
                x0 = x_in / panel_width_in
                patch_width = _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN / panel_width_in
                legend_ax.add_patch(
                    Rectangle(
                        (x0, y_center - 0.5 * patch_height),
                        patch_width,
                        patch_height,
                        transform=legend_ax.transAxes,
                        facecolor=facecolor,
                        edgecolor=edgecolor,
                        linewidth=0.8,
                    )
                )
                legend_ax.text(
                    (x_in + _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN + _HORIZONTAL_LEGEND_HANDLE_TEXT_PAD_IN) / panel_width_in,
                    y_center,
                    label,
                    transform=legend_ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=_HORIZONTAL_LEGEND_TEXT_SIZE,
                )
                x_in += item_width + item_gap_in
            row_top -= row_height
        return
    ax.legend(
        handles=handles,
        loc="lower left",
        frameon=False,
        fontsize=5.8,
        handlelength=1.3,
        handletextpad=0.45,
        borderaxespad=0.2,
    )


def apply_common_overlays(
    ax,
    *,
    context: StaticContext,
    extent: tuple[float, float, float, float],
    show_roi: bool,
    show_station_marker: bool,
    show_stations_name: bool,
    show_stations_elev: bool,
) -> None:
    if show_roi:
        draw_roi(ax, context)
    if show_station_marker:
        draw_stations_overlay(
            ax,
            context,
            extent,
            show_station_marker=show_station_marker,
            show_stations_name=show_stations_name,
            show_stations_elev=show_stations_elev,
        )


def overview_label_column(labels) -> str | None:
    for candidate in ("NAME_ENGL", "CNTR_NAME", "NAME_LATN", "COUNTRY", "label", "name", "CNTR_ID"):
        if candidate in labels.columns:
            return candidate
    return None


def overview_code_column(gdf) -> str | None:
    for candidate in ("CNTR_ID", "CNTR_CODE", "ISO3_CODE"):
        if candidate in gdf.columns:
            return candidate
    return None


def overview_name_lookup(labels) -> dict[str, str]:
    name_col = overview_label_column(labels)
    code_col = overview_code_column(labels)
    if name_col is None or code_col is None or labels.empty:
        return {}

    working = labels.dropna(subset=[name_col, code_col]).copy()
    working[name_col] = working[name_col].astype(str).str.strip()
    working[code_col] = working[code_col].astype(str).str.strip()
    working = working.loc[(working[name_col] != "") & (working[code_col] != "")]
    working = working.drop_duplicates(subset=[code_col])
    return dict(zip(working[code_col], working[name_col], strict=False))


def overview_label_point(geometry):
    if geometry is None or geometry.is_empty:
        return None
    if geometry.geom_type == "MultiPolygon":
        geometry = max(geometry.geoms, key=lambda geom: geom.area, default=geometry)
    elif geometry.geom_type == "GeometryCollection":
        polygons = [geom for geom in geometry.geoms if geom.geom_type in {"Polygon", "MultiPolygon"} and not geom.is_empty]
        if polygons:
            geometry = max(polygons, key=lambda geom: geom.area)
        else:
            geometry = geometry.convex_hull
    elif geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        geometry = geometry.convex_hull
    return geometry.representative_point()


def overview_country_label_specs(
    *,
    visible_countries,
    labels,
    extent: tuple[float, float, float, float],
    roi_anchor: tuple[float, float] | None,
) -> list[OverviewLabelSpec]:
    if visible_countries.empty:
        return []
    code_col = overview_code_column(visible_countries)
    if code_col is None:
        return []
    name_lookup = overview_name_lookup(labels)
    span_x = extent[1] - extent[0]
    span_y = extent[3] - extent[2]
    min_dx = _OVERVIEW_LABEL_DX_RATIO * span_x
    min_dy = _OVERVIEW_LABEL_DY_RATIO * span_y
    placed: list[tuple[float, float]] = []
    if roi_anchor is not None:
        placed.append(roi_anchor)

    working = visible_countries.dropna(subset=[code_col]).copy()
    working[code_col] = working[code_col].astype(str).str.strip()
    working = working.loc[working[code_col] != ""]
    if working.empty:
        return []
    working["label_key"] = working[code_col]
    working = working.dissolve(by="label_key", as_index=False)
    working["label_name"] = working["label_key"].map(name_lookup).fillna(working["label_key"])
    working["label_area"] = working.geometry.area
    placements: list[OverviewLabelSpec] = []

    for row in working.sort_values(by="label_area", ascending=False).itertuples():
        point = overview_label_point(row.geometry)
        if point is None:
            continue
        x = float(point.x)
        y = float(point.y)
        if not (extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]):
            continue
        if any(abs(x - px) < min_dx and abs(y - py) < min_dy for px, py in placed):
            continue
        placements.append(
            OverviewLabelSpec(
                text=str(row.label_name),
                x=x,
                y=y,
                ha="center",
                va="center",
                fontsize=_OVERVIEW_LABEL_SIZE,
                with_bbox=True,
                zorder=_ANNOTATION_ZORDER - 2,
            )
        )
        placed.append((x, y))
    return placements


def overview_roi_label_spec(panel: MapPanelSpec, *, extent: tuple[float, float, float, float], context: StaticContext) -> OverviewLabelSpec | None:
    if not panel.roi_label:
        return None
    centroid = context.roi_gdf.geometry.unary_union.centroid
    return OverviewLabelSpec(
        text=panel.roi_label,
        x=float(centroid.x) + _OVERVIEW_ROI_LABEL_DX_RATIO * (extent[1] - extent[0]),
        y=float(centroid.y),
        ha="left",
        va="center",
        fontsize=_OVERVIEW_ROI_LABEL_SIZE,
        with_bbox=False,
        zorder=_ANNOTATION_ZORDER,
    )


def overview_extent_growth_for_labels(
    ax,
    *,
    extent: tuple[float, float, float, float],
    label_specs: list[OverviewLabelSpec],
    margin_ratio: float = 0.0,
) -> tuple[float, float, float, float]:
    if not label_specs:
        return extent
    span_x = max(float(extent[1] - extent[0]), 1e-9)
    span_y = max(float(extent[3] - extent[2]), 1e-9)

    data_per_in_x = span_x / max(axis_width_inches(ax), 1e-9)
    data_per_in_y = span_y / max(axis_height_inches(ax), 1e-9)
    extra_left = extra_right = extra_bottom = extra_top = 0.0

    for spec in label_specs:
        width_in, height_in = overview_label_box_size_in(spec)
        half_width = 0.5 * width_in
        half_height = 0.5 * height_in
        if spec.ha == "left":
            left_in = 0.0
            right_in = width_in
        elif spec.ha == "right":
            left_in = width_in
            right_in = 0.0
        else:
            left_in = half_width
            right_in = half_width
        if spec.va == "top":
            bottom_in = height_in
            top_in = 0.0
        elif spec.va == "bottom":
            bottom_in = 0.0
            top_in = height_in
        else:
            bottom_in = half_height
            top_in = half_height

        extra_left = max(extra_left, max(left_in * data_per_in_x - (spec.x - extent[0]), 0.0))
        extra_right = max(extra_right, max(right_in * data_per_in_x - (extent[1] - spec.x), 0.0))
        extra_bottom = max(extra_bottom, max(bottom_in * data_per_in_y - (spec.y - extent[2]), 0.0))
        extra_top = max(extra_top, max(top_in * data_per_in_y - (extent[3] - spec.y), 0.0))

    if margin_ratio > 0.0:
        margin_x = margin_ratio * span_x
        margin_y = margin_ratio * span_y
        extra_left += margin_x
        extra_right += margin_x
        extra_bottom += margin_y
        extra_top += margin_y

    return (
        extent[0] - extra_left,
        extent[1] + extra_right,
        extent[2] - extra_bottom,
        extent[3] + extra_top,
    )


def expand_extent_to_target_aspect(
    extent: tuple[float, float, float, float],
    *,
    target_aspect: float,
) -> tuple[float, float, float, float]:
    span_x = max(float(extent[1] - extent[0]), 1e-9)
    span_y = max(float(extent[3] - extent[2]), 1e-9)
    current_aspect = span_y / span_x
    center_x = 0.5 * (extent[0] + extent[1])
    center_y = 0.5 * (extent[2] + extent[3])
    if np.isclose(current_aspect, target_aspect):
        return extent
    if current_aspect < target_aspect:
        target_span_y = span_x * target_aspect
        half_span_y = 0.5 * target_span_y
        return (extent[0], extent[1], center_y - half_span_y, center_y + half_span_y)
    target_span_x = span_y / target_aspect
    half_span_x = 0.5 * target_span_x
    return (center_x - half_span_x, center_x + half_span_x, extent[2], extent[3])


def overview_subset_geometries(
    data,
    *,
    context: StaticContext,
    extent: tuple[float, float, float, float],
    geom_types: set[str],
    clip_to_extent: bool,
    filter_fragments: bool,
):
    target_window = box(extent[0], extent[2], extent[1], extent[3])
    target_window_gdf = context.roi_gdf.iloc[:1].copy()
    target_window_gdf.geometry = [target_window]
    source_window_gdf = target_window_gdf.to_crs(data.crs)
    window_geom = source_window_gdf.geometry.iloc[0]
    window_bounds = source_window_gdf.total_bounds
    subset = data.cx[window_bounds[0] : window_bounds[2], window_bounds[1] : window_bounds[3]].copy()
    if subset.empty:
        subset = data.copy()
    subset = subset[subset.geom_type.isin(geom_types)].copy()
    subset = subset.loc[subset.intersects(window_geom)].copy()
    if subset.crs is not None and context.spec.crs is not None and str(subset.crs) != str(context.spec.crs):
        subset = subset.to_crs(context.spec.crs)
    subset = subset[subset.geometry.notna()].copy()
    subset = subset.loc[~subset.geometry.is_empty].copy()
    if clip_to_extent and not subset.empty:
        clip_geom = box(extent[0], extent[2], extent[1], extent[3])
        subset = subset.loc[subset.intersects(clip_geom)].copy()
        if not subset.empty:
            subset = subset.clip(clip_geom)
            subset = subset.loc[~subset.geometry.is_empty].copy()
    if filter_fragments and not subset.empty:
        subset = subset.explode(ignore_index=True)
        bounds = subset.geometry.bounds
        fragment_dim = np.maximum(bounds.maxx - bounds.minx, bounds.maxy - bounds.miny)
        threshold = _OVERVIEW_FRAGMENT_RATIO * max(extent[1] - extent[0], extent[3] - extent[2])
        subset = subset.loc[fragment_dim > threshold].copy()
    return subset


def overview_extent(ax, context: StaticContext, *, scale: int) -> tuple[float, float, float, float]:
    fig = ax.figure
    bbox = ax.get_position()
    width_in = fig.get_size_inches()[0] * bbox.width
    width_m = width_in * 0.0254 * float(scale)
    target_extent = buffered_extent(context)
    target_aspect = max(float(target_extent[3] - target_extent[2]), 1e-9) / max(float(target_extent[1] - target_extent[0]), 1e-9)
    height_m = width_m * target_aspect
    centroid = context.roi_gdf.geometry.unary_union.centroid
    center_x = float(centroid.x)
    center_y = float(centroid.y)
    return (
        center_x - 0.5 * width_m,
        center_x + 0.5 * width_m,
        center_y - 0.5 * height_m,
        center_y + 0.5 * height_m,
    )


def overview_extent_with_label_fit(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    labels,
    visible_regions_getter,
) -> tuple[float, float, float, float]:
    target_extent = buffered_extent(context)
    target_aspect = max(float(target_extent[3] - target_extent[2]), 1e-9) / max(float(target_extent[1] - target_extent[0]), 1e-9)
    extent = expand_extent_to_target_aspect(
        overview_extent(ax, context, scale=int(panel.scale or 1)),
        target_aspect=target_aspect,
    )
    margin_ratio = float(panel.label_fit_margin or 0.0)
    for _ in range(4):
        roi_label = overview_roi_label_spec(panel, extent=extent, context=context)
        visible_regions = visible_regions_getter(extent)
        label_specs = overview_country_label_specs(
            visible_countries=visible_regions,
            labels=labels,
            extent=extent,
            roi_anchor=(roi_label.x, roi_label.y) if roi_label is not None else None,
        )
        if roi_label is not None:
            label_specs.append(roi_label)
        expanded = overview_extent_growth_for_labels(
            ax,
            extent=extent,
            label_specs=label_specs,
            margin_ratio=margin_ratio,
        )
        expanded = expand_extent_to_target_aspect(expanded, target_aspect=target_aspect)
        if all(np.isclose(a, b) for a, b in zip(expanded, extent, strict=False)):
            return extent
        extent = expanded
        margin_ratio = 0.0
    return extent


def render_overview_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    label: str | None,
    defaults: MapDefaults,
    boundaries_loader: Callable[..., object] = load_overview_boundaries,
    regions_loader: Callable[..., object] = load_overview_regions,
    labels_loader: Callable[..., object] = load_overview_labels,
) -> dict[str, object]:
    countries = boundaries_loader(setup_dir=context.setup_dir)
    country_regions = regions_loader(setup_dir=context.setup_dir)
    country_labels = labels_loader(setup_dir=context.setup_dir)
    visible_regions_getter = lambda current_extent: overview_subset_geometries(
        country_regions,
        context=context,
        extent=current_extent,
        geom_types={"Polygon", "MultiPolygon"},
        clip_to_extent=True,
        filter_fragments=False,
    )
    extent = overview_extent_with_label_fit(
        ax,
        panel=panel,
        context=context,
        labels=country_labels,
        visible_regions_getter=visible_regions_getter,
    )
    subset = overview_subset_geometries(
        countries,
        context=context,
        extent=extent,
        geom_types={"LineString", "MultiLineString", "Polygon", "MultiPolygon"},
        clip_to_extent=True,
        filter_fragments=False,
    )
    if not subset.empty:
        subset.plot(ax=ax, color="black", linewidth=0.65, zorder=8)
    visible_regions = visible_regions_getter(extent)
    apply_map_axis_style(
        ax,
        extent,
        title=panel_title(label, panel_semantic_title(panel)),
        show_grid=resolve_flag(panel.show_grid, defaults, "show_grid", True),
        show_y_ticklabels=panel.col == 0,
        aspect_adjustable="box",
    )
    context.roi_gdf.plot(ax=ax, facecolor=_OVERVIEW_ROI_COLOR, edgecolor=_OVERVIEW_ROI_COLOR, linewidth=0.8, zorder=25)
    roi_label = overview_roi_label_spec(panel, extent=extent, context=context)
    label_specs = overview_country_label_specs(
        visible_countries=visible_regions,
        labels=country_labels,
        extent=extent,
        roi_anchor=(roi_label.x, roi_label.y) if roi_label is not None else None,
    )
    if roi_label is not None:
        label_specs.append(roi_label)
    draw_overview_label_specs(ax, label_specs)
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"extent": extent}


def render_roi_panel(ax, *, panel: MapPanelSpec, context: StaticContext, extent, label: str | None, defaults: MapDefaults) -> dict[str, object]:
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    context.roi_gdf.plot(ax=ax, color=_ROI_FILL, edgecolor="none", zorder=0)
    apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=resolve_panel_toggle(panel.show_station_marker, True),
        show_stations_name=resolve_panel_toggle(panel.show_stations_name, True),
        show_stations_elev=resolve_panel_toggle(panel.show_stations_elev, True),
    )
    apply_map_axis_style(
        ax,
        extent,
        title=panel_title(label, panel_semantic_title(panel)),
        show_grid=show_grid,
        show_y_ticklabels=panel.col == 0,
    )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {}


def render_static_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    label: str | None,
    defaults: MapDefaults,
    figure_horizontal_default: bool,
    derived_cache: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    show_roi = resolve_panel_toggle(panel.show_roi, True)
    show_station_marker = resolve_panel_toggle(panel.show_station_marker, False)
    show_stations_name = resolve_panel_toggle(panel.show_stations_name, False)
    show_stations_elev = resolve_panel_toggle(panel.show_stations_elev, False)
    panel_grid_extent = grid_extent(context)

    if panel.kind == "hillshade":
        ax.imshow(
            hillshade(context, derived_cache=derived_cache),
            cmap="Greys_r",
            extent=hillshade_extent(context),
            origin="upper",
            interpolation=_HILLSHADE_INTERPOLATION,
            vmin=0.0,
            vmax=1.0,
            zorder=5,
        )
        apply_common_overlays(
            ax,
            context=context,
            extent=extent,
            show_roi=show_roi,
            show_station_marker=show_station_marker,
            show_stations_name=show_stations_name,
            show_stations_elev=show_stations_elev,
        )
        apply_map_axis_style(
            ax,
            extent,
            title=panel_title(label, panel_semantic_title(panel)),
            show_grid=show_grid,
            show_y_ticklabels=panel.col == 0,
        )
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {}

    if panel.kind == "landcover":
        masked_landcover = masked_invalid(field_array(context, "landcover"))
        present_codes = {int(value) for value in masked_landcover.compressed() if np.isfinite(value)}
        canonical_codes = sorted(present_codes) if present_codes else [0]
        code_to_index = {code: idx for idx, code in enumerate(canonical_codes)}
        categorical = np.full(masked_landcover.shape, np.nan, dtype=float)
        filled = masked_landcover.filled(np.nan)
        for code, idx in code_to_index.items():
            categorical[np.isclose(filled, float(code), equal_nan=False)] = idx
        cmap = landcover_cmap_for_codes(canonical_codes)
        norm = BoundaryNorm(np.arange(-0.5, len(canonical_codes) + 0.5), cmap.N)
        image = ax.imshow(categorical, cmap=cmap, norm=norm, extent=panel_grid_extent, origin="upper", interpolation="nearest", zorder=5)
        apply_common_overlays(
            ax,
            context=context,
            extent=extent,
            show_roi=show_roi,
            show_station_marker=show_station_marker,
            show_stations_name=show_stations_name,
            show_stations_elev=show_stations_elev,
        )
        apply_map_axis_style(
            ax,
            extent,
            title=panel_title(label, panel_semantic_title(panel)),
            show_grid=show_grid,
            show_y_ticklabels=panel.col == 0,
        )
        legend_handles = classified_legend_handles(
            canonical_codes=canonical_codes,
            present_codes=present_codes,
            label_lookup=LANDCOVER_LABELS,
            color_lookup=lambda code: cmap(code_to_index[code]),
            fallback_codes=[0],
        )
        draw_classified_legend(ax, legend_handles, layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default))
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image, "legend_handles": legend_handles}

    field = _STATIC_FIELD_KIND_TO_FIELD[panel.kind]
    preset = require_static_field_preset(field)
    data = masked_invalid(field_array(context, field))
    norm = static_field_norm(preset, data.filled(np.nan))
    image = ax.imshow(data, cmap=static_field_cmap(preset), norm=norm, extent=panel_grid_extent, origin="upper", interpolation="nearest", zorder=5)
    apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=show_roi,
        show_station_marker=show_station_marker,
        show_stations_name=show_stations_name,
        show_stations_elev=show_stations_elev,
    )
    apply_map_axis_style(
        ax,
        extent,
        title=panel_title(label, panel_semantic_title(panel)),
        show_grid=show_grid,
        show_y_ticklabels=panel.col == 0,
    )
    colorbar_style = static_field_colorbar_style(preset)
    if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        attach_colorbar(
            ax,
            image,
            label=colorbar_style.label,
            ticks=colorbar_style.ticks,
            ticklabels=colorbar_style.ticklabels,
            layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
        )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "colorbar_style": colorbar_style}


def render_model_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    label: str | None,
    defaults: MapDefaults,
    model_cache,
    scale_cache,
    figure_horizontal_default: bool,
    derived_cache: dict[str, np.ndarray] | None = None,
    model_loader: Callable[..., list[ModelFields]] = load_model_fields,
) -> dict[str, object]:
    date = panel_date(panel, defaults)
    if date is None:
        raise ValueError(f"Panel '{panel.kind}' requires a date (panel '{panel.title or panel.kind}')")
    variable = _MODEL_KIND_TO_VARIABLE[panel.kind]
    field_key = (variable, date)
    if field_key not in model_cache:
        model_cache[field_key] = model_loader(context.project_dir, variable, (date,))[0]
    preset = require_variable_preset(variable)
    if field_key not in scale_cache:
        scale_cache[field_key] = comparison_scales([model_cache[field_key]], preset)
    model_norm, increment_norm = scale_cache[field_key]
    if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        hillshade_mode = resolve_hillshade_extent(
            panel,
            defaults,
            builtin="roi" if panel.kind == "snow_depth" else "full",
        )
        underlay = (
            hillshade_underlay(context, derived_cache=derived_cache)
            if hillshade_mode == "roi"
            else hillshade(context, derived_cache=derived_cache)
        )
        ax.imshow(
            underlay,
            cmap="Greys_r",
            extent=hillshade_extent(context),
            origin="upper",
            interpolation=_HILLSHADE_INTERPOLATION,
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )

    colorbar_style: dict[str, object] | object
    if panel.source == "increment":
        image = ax.imshow(
            masked(model_cache[field_key].increment, context.roi_mask),
            cmap=INCREMENT_CMAP,
            norm=increment_norm,
            extent=grid_extent(context),
            origin="upper",
            interpolation="nearest",
            alpha=_SNOW_DEPTH_PANEL_ALPHA if panel.kind == "snow_depth" else 0.95,
            zorder=5,
        )
        colorbar_style = {"label": f"increment {preset.unit_label}"}
    else:
        data = model_cache[field_key].open_loop if panel.source == "open_loop" else model_cache[field_key].ens_mean
        image = ax.imshow(
            masked_model(data, context.roi_mask, preset=preset),
            cmap=model_map_cmap(preset),
            norm=model_norm,
            extent=grid_extent(context),
            origin="upper",
            interpolation="nearest",
            alpha=_SNOW_DEPTH_PANEL_ALPHA if panel.kind == "snow_depth" else 0.96,
            zorder=5,
        )
        colorbar_style = model_colorbar_style(preset)

    apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=resolve_panel_toggle(panel.show_station_marker, False),
        show_stations_name=resolve_panel_toggle(panel.show_stations_name, False),
        show_stations_elev=resolve_panel_toggle(panel.show_stations_elev, False),
    )
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    apply_map_axis_style(
        ax,
        extent,
        title=panel_title(label, panel_semantic_title(panel)),
        show_grid=show_grid,
        show_y_ticklabels=panel.col == 0,
    )
    if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        label_text = colorbar_style["label"] if isinstance(colorbar_style, dict) else colorbar_style.label
        ticks = colorbar_style.get("ticks", ()) if isinstance(colorbar_style, dict) else colorbar_style.ticks
        ticklabels = colorbar_style.get("ticklabels", ()) if isinstance(colorbar_style, dict) else colorbar_style.ticklabels
        attach_colorbar(
            ax,
            image,
            label=label_text,
            ticks=ticks,
            ticklabels=ticklabels,
            layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
        )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "colorbar_style": colorbar_style}


def render_observation_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    label: str | None,
    defaults: MapDefaults,
    obs_cache,
    figure_horizontal_default: bool,
    derived_cache: dict[str, np.ndarray] | None = None,
    observation_loader: Callable[..., ObservationScene] = load_observation_scene,
) -> dict[str, object]:
    date = panel_date(panel, defaults)
    if date is None:
        raise ValueError(f"Panel '{panel.kind}' requires a date (panel '{panel.title or panel.kind}')")
    observation = _OBSERVATION_KIND_TO_NAME[panel.kind]
    obs_key = (observation, date)
    if obs_key not in obs_cache:
        obs_cache[obs_key] = observation_loader(context.project_dir, context, observation=observation, date=date)
    scene = obs_cache[obs_key]
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="full")
        ax.imshow(
            hillshade_underlay(context, derived_cache=derived_cache)
            if hillshade_mode == "roi"
            else hillshade(context, derived_cache=derived_cache),
            cmap="Greys_r",
            extent=hillshade_extent(context),
            origin="upper",
            interpolation=_HILLSHADE_INTERPOLATION,
            vmin=0.0,
            vmax=1.0,
            zorder=0,
        )

    if observation == "scf":
        norm = Normalize(vmin=0.0, vmax=100.0)
        rgba = FSC_OBS_CMAP(norm(np.nan_to_num(scene.array, nan=0.0)))
        rgba[~scene.roi_mask] = (1.0, 1.0, 1.0, 0.0)
        invalid_mask = scene.invalid_mask if scene.invalid_mask is not None else np.zeros(scene.array.shape, dtype=bool)
        missing_inside_roi = scene.roi_mask & ~np.isfinite(scene.array)
        rgba[missing_inside_roi | invalid_mask] = matplotlib.colors.to_rgba(FSC_INVALID_COLOR)
        rgba[~scene.roi_mask] = (1.0, 1.0, 1.0, 0.0)
        ax.imshow(rgba, extent=scene.bounds, origin="upper", interpolation="nearest", zorder=5)
        image = ScalarMappable(norm=norm, cmap=FSC_OBS_CMAP)
        image.set_array([])
        legend_handles = None
    else:
        codes = sorted(WET_SNOW_COLORS)
        present_codes = {
            int(code)
            for code in codes
            if np.any(scene.roi_mask & np.isclose(scene.array, float(code), equal_nan=False))
        }
        code_to_index = {code: idx for idx, code in enumerate(codes)}
        categorical = np.full(scene.array.shape, np.nan, dtype=float)
        for code, idx in code_to_index.items():
            categorical[np.isclose(scene.array, float(code), equal_nan=False)] = idx
        cmap = matplotlib.colors.ListedColormap([WET_SNOW_COLORS[code] for code in codes], name="wet_snow_obs")
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        norm = BoundaryNorm(np.arange(-0.5, len(codes) + 0.5), cmap.N)
        image = ax.imshow(np.ma.masked_invalid(categorical), cmap=cmap, norm=norm, extent=scene.bounds, origin="upper", interpolation="nearest", zorder=5)
        legend_handles = classified_legend_handles(
            canonical_codes=codes,
            present_codes=present_codes,
            label_lookup=WET_SNOW_LABELS,
            color_lookup=lambda code: WET_SNOW_COLORS[code],
            fallback_codes=codes,
        )

    apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=resolve_panel_toggle(panel.show_station_marker, False),
        show_stations_name=resolve_panel_toggle(panel.show_stations_name, False),
        show_stations_elev=resolve_panel_toggle(panel.show_stations_elev, False),
    )
    apply_map_axis_style(
        ax,
        extent,
        title=panel_title(label, panel_semantic_title(panel)),
        show_grid=show_grid,
        show_y_ticklabels=panel.col == 0,
    )
    if observation == "scf":
        if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
            attach_colorbar(
                ax,
                image,
                label="fractional snow cover [%]",
                ticks=(0, 20, 40, 60, 80, 100),
                layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
            )
    else:
        draw_classified_legend(ax, legend_handles, layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default))
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "legend_handles": legend_handles}


def legend_source_handles(item: LegendItemSpec, artifacts: dict[str, dict[str, object]]) -> list[Patch]:
    source = artifacts.get(str(item.source or ""))
    if source is None:
        raise ValueError(f"Legend source '{item.source}' is not available")
    handles = source.get("legend_handles")
    if not handles:
        raise ValueError(f"Legend source '{item.source}' has no legend handles")
    return list(handles)


def render_colorbar_panel(ax, *, panel: MapPanelSpec, artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    ax.set_axis_off()
    source = artifacts.get(str(panel.source or ""))
    if source is None or "mappable" not in source:
        raise ValueError(f"Colorbar panel source '{panel.source}' is not available")
    cbar = plt.colorbar(source["mappable"], cax=ax, orientation="vertical")
    style = source.get("colorbar_style") or {}
    label = style.get("label") if isinstance(style, dict) else getattr(style, "label", None)
    ticks = style.get("ticks", ()) if isinstance(style, dict) else getattr(style, "ticks", ())
    ticklabels = style.get("ticklabels", ()) if isinstance(style, dict) else getattr(style, "ticklabels", ())
    if ticks:
        cbar.set_ticks(ticks)
    if ticklabels:
        cbar.set_ticklabels(ticklabels)
    cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE)
    title = extract_unit_title(label)
    if title:
        cbar.ax.text(0.5, 1.02, title, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=_COLORBAR_TITLE_SIZE)
    return {}


def render_legend_panel(ax, *, panel: MapPanelSpec, artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    ax.set_axis_off()
    items = panel.items
    if not items and panel.source is not None:
        items = (LegendItemSpec(kind="source_legend", source=panel.source),)

    y = 0.97
    for item in items:
        if item.kind == "heading":
            y = draw_heading(ax, y=y, text=str(item.label))
        elif item.kind == "station_symbol":
            y = draw_station_entry(ax, y=y, label=str(item.label))
        elif item.kind == "source_legend":
            if item.label:
                y = draw_heading(ax, y=y, text=str(item.label))
            for handle in legend_source_handles(item, artifacts):
                y = draw_patch_entry(ax, y=y, label=handle.get_label(), facecolor=handle.get_facecolor(), edgecolor=handle.get_edgecolor())
        elif item.kind == "scale_bar":
            continue
        else:
            raise ValueError(f"Unsupported legend item kind '{item.kind}'")
    return {}
