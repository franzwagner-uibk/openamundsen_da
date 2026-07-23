from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from matplotlib.cm import ScalarMappable
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from rasterio.warp import Resampling, reproject
from shapely.geometry import Point, box

from openamundsen_da.io.paths import (
    list_member_dirs,
    list_steps_sorted,
    project_fraction_envelope_path,
    read_step_config,
    resolve_member_source_dir,
)
from openamundsen_da.methods.h_of_x.model_scf import compute_model_scf_binary_grid, load_hofx_from_project
from openamundsen_da.methods.viz.fraction_series import load_fraction_series, load_open_loop_fraction_series
from openamundsen_da.methods.viz.theme import da_variable_line_color
from openamundsen_da.observer.summary_paths import resolve_fraction_summary_path
from openamundsen_da.methods.viz.maps.annotations import (
    apply_overlay_label_halo,
    draw_heading,
    draw_overview_label_specs,
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
    load_static_context,
    load_model_fields,
    load_observation_scene,
    load_observation_uncertainty_scene,
)
from openamundsen_da.methods.viz.maps.hillshade import aspect_hillshade, grid_extent, hillshade, hillshade_extent, hillshade_underlay, terrain_aspect
from openamundsen_da.methods.viz.maps.layout import (
    apply_map_axis_style,
    attach_colorbar,
    axis_height_inches,
    axis_width_inches,
    buffered_extent,
    draw_map_grid_overlay,
    extract_unit_title,
    horizontal_legend_gap_axes,
    horizontal_legend_row_height_factors,
    horizontal_legend_row_layout,
    pack_horizontal_legend_rows,
    panel_legend_layout,
    register_child_axes,
    resolve_flag,
    resolve_panel_toggle,
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
    SNOW_DEPTH_REFERENCE_TICKS_M,
    WET_SNOW_COLORS,
    WET_SNOW_LABELS,
    aspect_cmap,
    aspect_colorbar_style,
    aspect_norm,
    landcover_classes_for_present_codes,
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
    _COLORBAR_TICK_SIZE,
    _COLORBAR_TITLE_SIZE,
    _DATE_CALLOUT_ALPHA,
    _GRID_ZORDER,
    _HILLSHADE_INTERPOLATION,
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
    _SUBDOMAIN_BOUNDARY_COLOR,
    _SUBDOMAIN_BOUNDARY_HALO_COLOR,
    _SUBDOMAIN_BOUNDARY_HALO_WIDTH,
    _SUBDOMAIN_BOUNDARY_WIDTH,
)
from openamundsen_da.methods.viz.wet_snow_fields import (
    elevation_band_fraction_map as _elevation_band_fraction_map,
    wet_snow_line_from_fraction as _wet_snow_line_from_fraction,
)
from openamundsen_da.util.landcover_mask import resolve_landcover_mask
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector
from openamundsen_da.methods.wet_snow.classify import CLASSIFICATION_METHOD_AMOUNT, load_wet_snow_classification_config
from openamundsen_da.methods.wet_snow.wsl import compute_wet_snow_line_from_masks
from openamundsen_da.subdomain.manifest import SubdomainManifest, SubdomainMeta
from openamundsen_da.util.da_observables import weights_csv_name


_FRACTION_MODEL_CMAP = colormaps["Greys"]
_SUBDOMAIN_NO_DA_COLOR = "#525252"
_SUBDOMAIN_NO_DA_HATCH = "////"
_SUBDOMAIN_NO_DA_LABEL = "no DA"
_MAP_SUPPORT_TEXT_SIZE = 5.8
_COLORBAR_PANEL_UNIT_HEADER_ENABLED = False
_COLORBAR_PANEL_UNIT_HEADER_HEIGHT_AXES = 0.20
_COLORBAR_PANEL_UNIT_HEADER_GAP_AXES = 0.03
_COLORBAR_PANEL_POSTER_TICKS_ENABLED = False
_ELEVATION_BAND_WSF_CMAP = _FRACTION_MODEL_CMAP
_WET_SNOW_MODEL_CODES = (110, 125)
_SCF_BINARY_CMAP = ListedColormap(["#efefef", "#111111"], name="scf_binary")


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


def _format_poster_colorbar_tick(value: float) -> str:
    value = float(value)
    if abs(value) < 5e-10:
        value = 0.0
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _poster_colorbar_ticks(
    cbar,
    *,
    label: str | None,
    ticks: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    if ticks and extract_unit_title(label) == "[%]" and len(ticks) > 5 and min(ticks) >= 0.0 and max(ticks) <= 100.0:
        values = (0.0, 25.0, 50.0, 75.0, 100.0)
        return values, tuple(_format_poster_colorbar_tick(value) for value in values)
    if ticks and len(ticks) <= 5:
        return (), ()
    norm = getattr(cbar.mappable, "norm", None)
    vmin = getattr(norm, "vmin", None)
    vmax = getattr(norm, "vmax", None)
    if vmin is None or vmax is None:
        return (), ()
    vmin = float(vmin)
    vmax = float(vmax)
    if vmin < 0.0 < vmax and abs(abs(vmin) - abs(vmax)) < 1e-9:
        values = (vmin, 0.5 * vmin, 0.0, 0.5 * vmax, vmax)
        return values, tuple(_format_poster_colorbar_tick(value) for value in values)
    return (), ()


def _apply_poster_colorbar_ticks(cbar, *, label: str | None, ticks: tuple[float, ...]) -> None:
    if not _COLORBAR_PANEL_POSTER_TICKS_ENABLED:
        return
    poster_ticks, poster_labels = _poster_colorbar_ticks(cbar, label=label, ticks=ticks)
    if not poster_ticks:
        return
    cbar.set_ticks(poster_ticks)
    cbar.set_ticklabels(poster_labels)


def masked(arr: np.ndarray, roi_mask: np.ndarray) -> np.ma.MaskedArray:
    masked_array = np.asarray(arr, dtype=float).copy()
    masked_array[~roi_mask] = np.nan
    return np.ma.masked_invalid(masked_array)


def masked_invalid(arr: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(arr, dtype=float))


def inside_roi_invalid_mask(arr: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    return np.asarray(roi_mask, dtype=bool) & (~np.isfinite(data))


def overlay_invalid_inside_roi(ax, invalid_mask: np.ndarray, *, extent) -> None:
    mask = np.asarray(invalid_mask, dtype=bool)
    if not np.any(mask):
        return
    rgba = np.zeros(mask.shape + (4,), dtype=float)
    rgba[mask] = matplotlib.colors.to_rgba(FSC_INVALID_COLOR)
    ax.imshow(
        rgba,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=6,
    )


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


def draw_subdomain_boundaries(ax, context: StaticContext) -> None:
    subdomains = context.subdomain_gdf
    if subdomains is None or subdomains.empty:
        return
    collection_count = len(ax.collections)
    subdomains.boundary.plot(
        ax=ax,
        color=_SUBDOMAIN_BOUNDARY_COLOR,
        linewidth=_SUBDOMAIN_BOUNDARY_WIDTH,
        zorder=46,
    )
    for collection in ax.collections[collection_count:]:
        collection.set_path_effects(
            [
                pe.Stroke(linewidth=_SUBDOMAIN_BOUNDARY_HALO_WIDTH, foreground=_SUBDOMAIN_BOUNDARY_HALO_COLOR),
                pe.Normal(),
            ]
        )


def subdomain_dropped_event_regions(
    context: StaticContext,
    *,
    date: pd.Timestamp,
    variable: str,
) -> pd.DataFrame | None:
    subdomains = context.subdomain_gdf
    dropped = context.subdomain_dropped_events
    if subdomains is None or subdomains.empty or dropped is None or dropped.empty:
        return None
    if "subdomain_id" not in subdomains.columns:
        return None

    date_key = pd.Timestamp(date).normalize()
    variable_key = str(variable).strip().lower()
    rows = dropped[
        (pd.to_datetime(dropped["date"], errors="coerce").dt.normalize() == date_key)
        & (dropped["variable"].astype(str).str.strip().str.lower() == variable_key)
    ]
    if rows.empty:
        return None

    dropped_ids = {str(value) for value in rows["subdomain_id"].dropna().tolist()}
    selected = subdomains[subdomains["subdomain_id"].astype(str).isin(dropped_ids)].copy()
    if selected.empty:
        return None
    return selected


def _subdomain_dropped_event_ids(context: StaticContext, *, date: pd.Timestamp, variable: str) -> set[str]:
    dropped = context.subdomain_dropped_events
    if dropped is None or dropped.empty:
        return set()
    date_key = pd.Timestamp(date).normalize()
    variable_key = str(variable).strip().lower()
    rows = dropped[
        (pd.to_datetime(dropped["date"], errors="coerce").dt.normalize() == date_key)
        & (dropped["variable"].astype(str).str.strip().str.lower() == variable_key)
    ]
    if rows.empty:
        return set()
    return {str(value) for value in rows["subdomain_id"].dropna().tolist()}


def draw_subdomain_dropped_event_overlay(
    ax,
    context: StaticContext,
    *,
    date: pd.Timestamp,
    variable: str,
) -> None:
    selected = subdomain_dropped_event_regions(context, date=date, variable=variable)
    if selected is None or selected.empty:
        return

    selected.plot(
        ax=ax,
        facecolor="none",
        edgecolor=_SUBDOMAIN_NO_DA_COLOR,
        linewidth=0.0,
        hatch=_SUBDOMAIN_NO_DA_HATCH,
        zorder=47,
    )
    selected.boundary.plot(
        ax=ax,
        color=_SUBDOMAIN_NO_DA_COLOR,
        linewidth=1.05,
        linestyle=(0, (2.5, 1.8)),
        zorder=48,
    )
    for geom in selected.geometry:
        if geom is None or geom.is_empty:
            continue
        point = geom.representative_point()
        apply_overlay_label_halo(
            ax.text(
                float(point.x),
                float(point.y),
                _SUBDOMAIN_NO_DA_LABEL,
                ha="center",
                va="center",
                fontsize=_MAP_SUPPORT_TEXT_SIZE,
                color=_SUBDOMAIN_NO_DA_COLOR,
                zorder=_ANNOTATION_ZORDER + 1,
            ),
            with_bbox=True,
        )


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
            marker="^",
            s=26,
            facecolor=_STATION_COLOR,
            edgecolor="none",
            linewidth=0.0,
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
            label = f"{label}\n({int(round(float(row['alt'])))} m)"
        apply_overlay_label_halo(ax.text(
            float(row["x"]) + dx,
            float(row["y"]) + dy,
            label,
            fontsize=_MAP_SUPPORT_TEXT_SIZE,
            ha="left",
            va="bottom",
            color="black",
            zorder=_GRID_ZORDER + 5,
        ))
def comparison_scales(fields: list[ModelFields], preset, *, model_vmax: float | None = None) -> tuple[Normalize, TwoSlopeNorm]:
    comparisons = [field for field in fields if field is not None]
    if not comparisons:
        raise ValueError("comparison_scales requires at least one model field")

    valid_arrays = []
    increment_arrays = []
    for field in comparisons:
        valid_arrays.extend([field.open_loop, field.ens_mean])
        if field.analysis_mean is not None:
            valid_arrays.append(field.analysis_mean)
        increment_arrays.append(np.abs(field.increment))
        if field.analysis_increment is not None:
            increment_arrays.append(np.abs(field.analysis_increment))
    max_value = max(float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0 for arr in valid_arrays)
    max_increment = max(float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0 for arr in increment_arrays)

    vmax = float(model_vmax) if model_vmax is not None else nice_ceiling(max_value, step=preset.max_step, minimum=preset.max_floor)
    inc_abs = nice_ceiling(max_increment, step=preset.increment_step, minimum=preset.increment_floor)
    return model_map_norm(preset, vmax=vmax), TwoSlopeNorm(vcenter=0.0, vmin=-inc_abs, vmax=inc_abs)


def _required_analysis_array(field: ModelFields, source: str, variable: str, date: pd.Timestamp) -> np.ndarray:
    if source == "analysis_mean":
        arr = field.analysis_mean
    elif source == "analysis_increment":
        arr = field.analysis_increment
    else:
        raise ValueError(f"Unsupported analysis source '{source}'")
    if arr is None:
        raise KeyError(
            f"Compact DA output is missing '{source}_{variable}' for {pd.Timestamp(date).date()}; "
            "rerun the project so analysis_mean/analysis_increment fields are written"
        )
    return arr


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
        return [
            item.label
            for item in landcover_classes_for_present_codes(
                present_codes,
                grouping=panel.landcover_grouping,
            )
        ]

    if panel.kind in {"wet_snow", "wet_snow_line"}:
        if panel.source in {"prior_probability", "posterior", "posterior_probability"}:
            return []
        if panel.source is not None:
            date = panel_date(panel, defaults)
            if date is None:
                return [WET_SNOW_LABELS[code] for code in _WET_SNOW_MODEL_CODES]
            scene = _wet_snow_model_classified_array(
                context=context,
                source=str(panel.source),
                date=pd.Timestamp(date).normalize(),
                derived_cache=None,
            )
            present_codes = {
                code for code in _WET_SNOW_MODEL_CODES if np.any(np.isclose(scene, float(code), equal_nan=False))
            }
            active_codes = [code for code in _WET_SNOW_MODEL_CODES if code in present_codes]
            if not active_codes:
                active_codes = list(_WET_SNOW_MODEL_CODES)
            return [WET_SNOW_LABELS[code] for code in active_codes]
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


def draw_classified_legend(ax, handles: list[object], *, layout: str) -> None:
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
        gap_axes = horizontal_legend_gap_axes(ax)
        legend_height_in = max(axis_height_inches(ax) * inset_height, 1e-9)
        legend_ax = ax.inset_axes(
            [0.0, -(gap_axes + inset_height), 1.0, inset_height],
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
                x0 = x_in / panel_width_in
                patch_width = _HORIZONTAL_LEGEND_HANDLE_WIDTH_IN / panel_width_in
                if isinstance(handle, Patch):
                    facecolor = handle.get_facecolor()
                    edgecolor = handle.get_edgecolor()
                    if np.ndim(facecolor) == 2:
                        facecolor = facecolor[0]
                    if np.ndim(edgecolor) == 2:
                        edgecolor = edgecolor[0]
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
                elif isinstance(handle, Line2D):
                    inset = 0.08 * patch_width
                    legend_ax.plot(
                        [x0 + inset, x0 + patch_width - inset],
                        [y_center, y_center],
                        transform=legend_ax.transAxes,
                        color=handle.get_color(),
                        linewidth=handle.get_linewidth(),
                        linestyle=handle.get_linestyle(),
                        solid_capstyle="round",
                    )
                else:
                    raise TypeError(f"Unsupported classified legend handle type: {type(handle)!r}")
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
        fontsize=_MAP_SUPPORT_TEXT_SIZE,
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
        draw_subdomain_boundaries(ax, context)
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
    ax,
    visible_countries,
    labels,
    extent: tuple[float, float, float, float],
    roi_anchor: tuple[float, float] | None,
    avoid_geometry=None,
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
        spec = _overview_country_label_spec_for_geometry(
            ax=ax,
            text=str(row.label_name),
            geometry=row.geometry,
            extent=extent,
            avoid_geometry=avoid_geometry,
            placed=placed,
            min_dx=min_dx,
            min_dy=min_dy,
        )
        if spec is None:
            continue
        placements.append(spec)
        placed.append((spec.x, spec.y))
    return placements


def _overview_country_label_spec_for_geometry(
    *,
    ax,
    text: str,
    geometry,
    extent: tuple[float, float, float, float],
    avoid_geometry,
    placed: list[tuple[float, float]],
    min_dx: float,
    min_dy: float,
) -> OverviewLabelSpec | None:
    visible_extent = box(extent[0], extent[2], extent[1], extent[3])
    visible_geometry = geometry.intersection(visible_extent)
    if visible_geometry.is_empty:
        visible_geometry = geometry
    base_point = overview_label_point(visible_geometry)
    if base_point is None:
        return None
    base_xy = (float(base_point.x), float(base_point.y))

    candidates = [base_xy]
    if avoid_geometry is not None:
        candidates.extend(_overview_country_label_relocation_candidates(visible_geometry, extent=extent, base_xy=base_xy))

    for x, y in candidates:
        if not (extent[0] <= x <= extent[1] and extent[2] <= y <= extent[3]):
            continue
        if any(abs(x - px) < min_dx and abs(y - py) < min_dy for px, py in placed):
            continue
        spec = OverviewLabelSpec(
            text=text,
            x=x,
            y=y,
            ha="center",
            va="center",
            fontsize=_OVERVIEW_LABEL_SIZE,
            with_bbox=True,
            zorder=_ANNOTATION_ZORDER - 2,
        )
        if avoid_geometry is not None and overview_label_data_box(ax, spec, extent=extent).intersects(avoid_geometry):
            continue
        return spec
    return None


def _overview_country_label_relocation_candidates(
    geometry,
    *,
    extent: tuple[float, float, float, float],
    base_xy: tuple[float, float],
) -> list[tuple[float, float]]:
    minx, miny, maxx, maxy = geometry.bounds
    minx = max(float(minx), float(extent[0]))
    maxx = min(float(maxx), float(extent[1]))
    miny = max(float(miny), float(extent[2]))
    maxy = min(float(maxy), float(extent[3]))
    if minx >= maxx or miny >= maxy:
        return []

    fractions = (0.2, 0.35, 0.5, 0.65, 0.8)
    candidates: list[tuple[float, float]] = []
    for fx in fractions:
        x = minx + fx * (maxx - minx)
        for fy in fractions:
            y = miny + fy * (maxy - miny)
            point = Point(x, y)
            if geometry.covers(point):
                candidates.append((float(x), float(y)))

    base_x, base_y = base_xy
    candidates.sort(key=lambda xy: (xy[0] - base_x) ** 2 + (xy[1] - base_y) ** 2)
    deduped: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for x, y in candidates:
        key = (round(x, 6), round(y, 6))
        if key in seen:
            continue
        seen.add(key)
        deduped.append((x, y))
    return deduped


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
        left_in, right_in, bottom_in, top_in = overview_label_overhang_in(spec)

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


def overview_label_overhang_in(spec: OverviewLabelSpec) -> tuple[float, float, float, float]:
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
    return left_in, right_in, bottom_in, top_in


def overview_label_data_box(ax, spec: OverviewLabelSpec, *, extent: tuple[float, float, float, float]):
    span_x = max(float(extent[1] - extent[0]), 1e-9)
    span_y = max(float(extent[3] - extent[2]), 1e-9)
    data_per_in_x = span_x / max(axis_width_inches(ax), 1e-9)
    data_per_in_y = span_y / max(axis_height_inches(ax), 1e-9)
    left_in, right_in, bottom_in, top_in = overview_label_overhang_in(spec)
    return box(
        spec.x - left_in * data_per_in_x,
        spec.y - bottom_in * data_per_in_y,
        spec.x + right_in * data_per_in_x,
        spec.y + top_in * data_per_in_y,
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
            ax=ax,
            visible_countries=visible_regions,
            labels=labels,
            extent=extent,
            roi_anchor=(roi_label.x, roi_label.y) if roi_label is not None else None,
            avoid_geometry=context.roi_gdf.geometry.unary_union,
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
    try:
        countries = boundaries_loader(setup_dir=context.setup_dir)
        country_regions = regions_loader(setup_dir=context.setup_dir)
        country_labels = labels_loader(setup_dir=context.setup_dir)
    except Exception as exc:
        logger.warning("Overview country layers unavailable; rendering ROI-only overview panel")
        logger.debug("Overview country layer loading raised {}", type(exc).__name__)
        target_extent = buffered_extent(context)
        target_aspect = max(float(target_extent[3] - target_extent[2]), 1e-9) / max(
            float(target_extent[1] - target_extent[0]), 1e-9
        )
        extent = expand_extent_to_target_aspect(
            overview_extent(ax, context, scale=int(panel.scale or 1)),
            target_aspect=target_aspect,
        )
        apply_map_axis_style(
            ax,
            extent,
            title=panel_title(label, panel_semantic_title(panel)),
            show_grid=resolve_flag(panel.show_grid, defaults, "show_grid", True),
            show_y_ticklabels=panel.col == 0,
            aspect_adjustable="box",
        )
        context.roi_gdf.plot(
            ax=ax,
            facecolor=_OVERVIEW_ROI_COLOR,
            edgecolor=_OVERVIEW_ROI_COLOR,
            linewidth=0.8,
            zorder=25,
        )
        draw_subdomain_boundaries(ax, context)
        roi_label = overview_roi_label_spec(panel, extent=extent, context=context)
        if roi_label is not None:
            draw_overview_label_specs(ax, [roi_label])
        show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"extent": extent}

    def visible_regions_getter(current_extent):
        return overview_subset_geometries(
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
    draw_subdomain_boundaries(ax, context)
    roi_label = overview_roi_label_spec(panel, extent=extent, context=context)
    label_specs = overview_country_label_specs(
        ax=ax,
        visible_countries=visible_regions,
        labels=country_labels,
        extent=extent,
        roi_anchor=(roi_label.x, roi_label.y) if roi_label is not None else None,
        avoid_geometry=context.roi_gdf.geometry.unary_union,
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
        shade = hillshade(context, derived_cache=derived_cache)
        if shade.shape == context.roi_mask.shape:
            shade = np.ma.masked_array(
                shade,
                mask=(~context.roi_mask) | (~np.isfinite(shade)),
            )
        else:
            shade = hillshade_underlay(context, derived_cache=derived_cache)
        ax.imshow(
            shade,
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
        masked_landcover = masked(field_array(context, "landcover"), context.roi_mask)
        present_source_codes = {int(value) for value in masked_landcover.compressed() if np.isfinite(value)}
        landcover_classes = landcover_classes_for_present_codes(
            present_source_codes,
            grouping=panel.landcover_grouping,
        )
        canonical_codes = [item.code for item in landcover_classes]
        code_to_index = {code: idx for idx, code in enumerate(canonical_codes)}
        class_lookup = {item.code: item for item in landcover_classes}
        categorical = np.full(masked_landcover.shape, np.nan, dtype=float)
        filled = masked_landcover.filled(np.nan)
        for code, idx in code_to_index.items():
            source_values = [float(source_code) for source_code in class_lookup[code].source_codes]
            categorical[np.isin(filled, source_values)] = idx
        categorical = np.ma.masked_invalid(categorical)
        cmap = ListedColormap([item.color for item in landcover_classes], name="oa_da_landcover")
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
            present_codes={item.code for item in landcover_classes},
            label_lookup={item.code: item.label for item in landcover_classes},
            color_lookup=lambda code: class_lookup[code].color,
            fallback_codes=canonical_codes,
        )
        draw_classified_legend(ax, legend_handles, layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default))
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image, "legend_handles": legend_handles}

    if panel.kind == "aspect":
        aspect = np.deg2rad(terrain_aspect(context, derived_cache=derived_cache))
        mask = (~context.roi_mask.astype(bool)) | (~np.isfinite(aspect))
        norm = aspect_norm()
        cmap = aspect_cmap()
        rgba = cmap(norm(np.where(mask, 0.0, aspect)))
        shade = aspect_hillshade(context, derived_cache=derived_cache)
        modulation = 0.58 + 0.42 * np.clip(shade, 0.0, 1.0)
        rgba[..., :3] = np.clip(rgba[..., :3] * modulation[..., None], 0.0, 1.0)
        rgba[..., 3] = np.where(mask, 0.0, 1.0)
        image = ax.imshow(rgba, extent=panel_grid_extent, origin="upper", interpolation="nearest", zorder=5)
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
        colorbar_style = aspect_colorbar_style()
        if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
            colorbar_mappable = ScalarMappable(norm=norm, cmap=cmap)
            colorbar_mappable.set_array([])
            attach_colorbar(
                ax,
                colorbar_mappable,
                label=colorbar_style.label,
                ticks=colorbar_style.ticks,
                ticklabels=colorbar_style.ticklabels,
                layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
            )
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=panel_date(panel, defaults), resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image, "colorbar_style": colorbar_style, "aspect": np.ma.masked_array(aspect, mask=mask)}

    field = _STATIC_FIELD_KIND_TO_FIELD[panel.kind]
    preset = require_static_field_preset(field)
    raw_data = field_array(context, field)
    data = masked(raw_data, context.roi_mask)
    norm = static_field_norm(preset, data.filled(np.nan))
    alpha = 1.0
    if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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
        alpha = 0.86
    image = ax.imshow(data, cmap=static_field_cmap(preset), norm=norm, extent=panel_grid_extent, origin="upper", interpolation="nearest", alpha=alpha, zorder=5)
    overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(raw_data, context.roi_mask), extent=panel_grid_extent)
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
    colorbar_style = static_field_colorbar_style(preset, data.filled(np.nan))
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
    shared_model_vmax: dict[str, float] | None = None,
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
        shared_vmax = None if shared_model_vmax is None else shared_model_vmax.get(variable)
        scale_cache[field_key] = comparison_scales([model_cache[field_key]], preset, model_vmax=shared_vmax)
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
    field = model_cache[field_key]
    if panel.source in {"increment", "analysis_increment"}:
        data = (
            field.increment
            if panel.source == "increment"
            else _required_analysis_array(field, "analysis_increment", variable, date)
        )
        invalid_mask = inside_roi_invalid_mask(data, context.roi_mask)
        image = ax.imshow(
            masked(data, context.roi_mask),
            cmap=INCREMENT_CMAP,
            norm=increment_norm,
            extent=grid_extent(context),
            origin="upper",
            interpolation="nearest",
            alpha=_SNOW_DEPTH_PANEL_ALPHA if panel.kind == "snow_depth" else 0.95,
            zorder=5,
        )
        colorbar_label = (
            f"DA increment (posterior - prior) {preset.unit_label}"
            if panel.source == "analysis_increment"
            else f"increment {preset.unit_label}"
        )
        colorbar_style = {"label": colorbar_label}
    else:
        if panel.source == "analysis_mean":
            data = _required_analysis_array(field, "analysis_mean", variable, date)
        else:
            data = field.open_loop if panel.source == "open_loop" else field.ens_mean
        invalid_mask = inside_roi_invalid_mask(data, context.roi_mask)
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
        colorbar_style = model_colorbar_style(preset, vmax=model_norm.vmax)
    overlay_invalid_inside_roi(ax, invalid_mask, extent=grid_extent(context))

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
    if panel.source is not None:
        return render_fraction_model_panel(
            ax,
            panel=panel,
            context=context,
            extent=extent,
            label=label,
            defaults=defaults,
            figure_horizontal_default=figure_horizontal_default,
            derived_cache=derived_cache,
            observation=observation,
            date=pd.Timestamp(date).normalize(),
        )
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
        invalid_mask = (scene.invalid_mask if scene.invalid_mask is not None else np.zeros(scene.array.shape, dtype=bool)) | (
            scene.roi_mask & ~np.isfinite(scene.array)
        )
        overlay_invalid_inside_roi(ax, invalid_mask, extent=scene.bounds)
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


def render_uncertainty_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    label: str | None,
    defaults: MapDefaults,
    uncertainty_cache,
    figure_horizontal_default: bool,
    derived_cache: dict[str, np.ndarray] | None = None,
    uncertainty_loader: Callable[..., ObservationScene] = load_observation_uncertainty_scene,
) -> dict[str, object]:
    date = panel_date(panel, defaults)
    if date is None:
        raise ValueError(f"Panel '{panel.kind}' requires a date (panel '{panel.title or panel.kind}')")
    observation = str(panel.observation or "").strip()
    if not observation:
        raise ValueError("Uncertainty panels require an observation name")
    cache_key = (observation, date)
    if cache_key not in uncertainty_cache:
        uncertainty_cache[cache_key] = uncertainty_loader(
            context.project_dir,
            context,
            observation=observation,
            date=date,
        )
    scene = uncertainty_cache[cache_key]
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

    norm = Normalize(vmin=0.0, vmax=100.0)
    image = ax.imshow(
        np.ma.masked_invalid(scene.array),
        cmap=colormaps["viridis"],
        norm=norm,
        extent=scene.bounds,
        origin="upper",
        interpolation="nearest",
        zorder=5,
    )
    invalid_mask = scene.invalid_mask if scene.invalid_mask is not None else scene.roi_mask & ~np.isfinite(scene.array)
    overlay_invalid_inside_roi(ax, invalid_mask, extent=scene.bounds)
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
    colorbar_style = {"label": "uncertainty [%]", "ticks": (0, 20, 40, 60, 80, 100)}
    if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        attach_colorbar(
            ax,
            image,
            label=colorbar_style["label"],
            ticks=colorbar_style["ticks"],
            layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
        )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "colorbar_style": colorbar_style}


def _fraction_value_column(observation: str) -> str:
    return "scf" if observation == "scf" else "wet_snow_fraction"


def _fraction_member_filename(observation: str) -> str:
    return "point_scf_roi.csv" if observation == "scf" else "point_wet_snow_roi.csv"


def _wet_snow_threshold_fraction(project_dir: Path) -> float:
    cfg = load_wet_snow_classification_config(project_dir)
    if cfg.method == CLASSIFICATION_METHOD_AMOUNT:
        return 0.5
    return float(cfg.threshold_percent) / 100.0


def _step_dir_for_date(project_dir: Path, date: pd.Timestamp) -> Path:
    target = pd.Timestamp(date).normalize()
    for step_dir in list_steps_sorted(project_dir):
        cfg = read_step_config(step_dir)
        start = pd.Timestamp(cfg["start_date"]).normalize()
        end = pd.Timestamp(cfg["end_date"]).normalize()
        if start <= target <= end:
            return Path(step_dir)
    raise FileNotFoundError(f"No project step found for {target.date()} in {project_dir}")


def _wet_snow_mask_path(results_dir: Path, date: pd.Timestamp) -> Path:
    stamp = pd.Timestamp(date).strftime("%Y-%m-%dT%H%M")
    path = Path(results_dir) / "wet_snow" / f"wet_snow_mask_{stamp}.tif"
    if not path.is_file():
        raise FileNotFoundError(f"Missing wet-snow mask: {path}")
    return path


def _load_wet_snow_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def _wet_snow_model_classified_array(
    *,
    context: StaticContext,
    source: str,
    date: pd.Timestamp,
    derived_cache: dict[str, np.ndarray] | None,
) -> np.ndarray:
    cache_key = f"wet-snow-model-map:{source}:{pd.Timestamp(date).normalize().isoformat()}"
    if derived_cache is not None and cache_key in derived_cache:
        return np.asarray(derived_cache[cache_key], dtype=float)

    step_dir = _step_dir_for_date(context.project_dir, date)
    if source == "open_loop":
        mask = _load_wet_snow_mask(_wet_snow_mask_path(step_dir / "ensembles" / "prior" / "open_loop" / "results", date))
        classified = np.full(mask.shape, np.nan, dtype=float)
        classified[mask == 1] = float(_WET_SNOW_MODEL_CODES[0])
        classified[mask == 0] = float(_WET_SNOW_MODEL_CODES[1])
    elif source in {"ensemble_mean", "posterior"}:
        member_masks: list[np.ndarray] = []
        ensemble = "prior" if source == "ensemble_mean" else "posterior"
        for member_dir in list_member_dirs(step_dir, ensemble):
            source_member_dir = resolve_member_source_dir(member_dir) if source == "posterior" else member_dir
            mask_path = _wet_snow_mask_path(source_member_dir / "results", date)
            member_masks.append(_load_wet_snow_mask(mask_path))
        if not member_masks:
            raise FileNotFoundError(f"Missing {ensemble} members for wet-snow ensemble map in {step_dir}")
        stack = np.stack([np.where(mask == 255, np.nan, mask.astype(float)) for mask in member_masks], axis=0)
        valid_count = np.sum(np.isfinite(stack), axis=0)
        wet_sum = np.nansum(stack, axis=0)
        wet_fraction = np.divide(wet_sum, valid_count, out=np.full(valid_count.shape, np.nan, dtype=float), where=valid_count > 0)
        valid = np.isfinite(wet_fraction)
        threshold = _wet_snow_threshold_fraction(context.project_dir)
        classified = np.full(wet_fraction.shape, np.nan, dtype=float)
        classified[valid & (wet_fraction >= threshold)] = float(_WET_SNOW_MODEL_CODES[0])
        classified[valid & (wet_fraction < threshold)] = float(_WET_SNOW_MODEL_CODES[1])
    else:
        raise ValueError(f"Unsupported wet-snow model source '{source}'")

    classified = np.asarray(classified, dtype=float)
    classified[~context.roi_mask] = np.nan
    if derived_cache is not None:
        derived_cache[cache_key] = classified
    return classified


def _prior_wet_fraction_array(
    *,
    context: StaticContext,
    date: pd.Timestamp,
    derived_cache: dict[str, np.ndarray] | None,
) -> np.ndarray:
    cache_key = f"wet-snow-prior-fraction:{pd.Timestamp(date).normalize().isoformat()}"
    if derived_cache is not None and cache_key in derived_cache:
        return np.asarray(derived_cache[cache_key], dtype=float)

    step_dir = _step_dir_for_date(context.project_dir, date)
    member_masks: list[np.ndarray] = []
    for member_dir in list_member_dirs(step_dir, "prior"):
        mask_path = _wet_snow_mask_path(member_dir / "results", date)
        member_masks.append(_load_wet_snow_mask(mask_path))
    if not member_masks:
        raise FileNotFoundError(f"Missing prior members for wet-snow probability map in {step_dir}")
    stack = np.stack([np.where(mask == 255, np.nan, mask.astype(float)) for mask in member_masks], axis=0)
    valid_count = np.sum(np.isfinite(stack), axis=0)
    wet_sum = np.nansum(stack, axis=0)
    wet_fraction = np.divide(
        wet_sum,
        valid_count,
        out=np.full(valid_count.shape, np.nan, dtype=float),
        where=valid_count > 0,
    )
    wet_fraction = np.asarray(wet_fraction, dtype=float)
    wet_fraction[~context.roi_mask] = np.nan
    if derived_cache is not None:
        derived_cache[cache_key] = wet_fraction
    return wet_fraction


def _wet_snow_line_from_classified(
    *,
    context: StaticContext,
    classified: np.ndarray,
) -> float | None:
    valid_mask = np.isfinite(classified) & np.asarray(context.roi_mask, dtype=bool)
    wet_mask = valid_mask & np.isclose(classified, float(_WET_SNOW_MODEL_CODES[0]), equal_nan=False)
    evaluation = compute_wet_snow_line_from_masks(
        setup_dir=context.setup_dir,
        project_dir=context.project_dir,
        valid_mask=valid_mask,
        wet_mask=wet_mask,
    )
    return evaluation.wet_snow_line


def _weights_path(project_dir: Path, date: pd.Timestamp, variable: str) -> Path:
    step_dir = _step_dir_for_date(project_dir, date)
    return step_dir / "assim" / weights_csv_name(variable, pd.Timestamp(date).to_pydatetime())


def _load_weights_df(project_dir: Path, date: pd.Timestamp, variable: str) -> pd.DataFrame | None:
    try:
        weights_path = _weights_path(project_dir, date, variable)
    except FileNotFoundError:
        return None
    if not weights_path.is_file():
        return None
    df = pd.read_csv(weights_path)
    if df.empty:
        return None
    return df


def _posterior_da_weights(context: StaticContext, date: pd.Timestamp, variable: str = "wet_snow_line") -> pd.Series | None:
    df = _load_weights_df(context.project_dir, date, variable)
    if df is None:
        return None
    if "member_id" not in df.columns or "weight" not in df.columns:
        return None

    weights = pd.to_numeric(df["weight"], errors="coerce")
    member_ids = df["member_id"].astype(str).str.strip()
    valid = member_ids.ne("") & weights.notna()
    if not bool(valid.any()):
        return None
    series = pd.Series(weights.loc[valid].to_numpy(dtype=float), index=member_ids.loc[valid].tolist(), dtype=float)
    if series.empty:
        return None
    total_weight = float(series.sum())
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        return None
    return series / total_weight


def _posterior_weighted_wet_fraction_array(
    *,
    context: StaticContext,
    date: pd.Timestamp,
    derived_cache: dict[str, np.ndarray] | None,
    weights_variable: str = "wet_snow_line",
) -> np.ndarray:
    cache_key = f"wet-snow-posterior-weighted-fraction:{weights_variable}:{pd.Timestamp(date).normalize().isoformat()}"
    if derived_cache is not None and cache_key in derived_cache:
        return np.asarray(derived_cache[cache_key], dtype=float)

    step_dir = _step_dir_for_date(context.project_dir, date)
    weights = _posterior_da_weights(context, date, variable=weights_variable)
    member_masks: list[np.ndarray] = []
    member_weights: list[float] = []

    if weights is not None:
        prior_members = {member_dir.name: member_dir for member_dir in list_member_dirs(step_dir, "prior")}
        for member_id, weight in weights.items():
            member_dir = prior_members.get(str(member_id))
            if member_dir is None:
                continue
            mask_path = _wet_snow_mask_path(member_dir / "results", date)
            mask = _load_wet_snow_mask(mask_path)
            member_masks.append(np.where(mask == 255, np.nan, mask.astype(float)))
            member_weights.append(float(weight))
    else:
        for member_dir in list_member_dirs(step_dir, "posterior"):
            source_member_dir = resolve_member_source_dir(member_dir)
            mask_path = _wet_snow_mask_path(source_member_dir / "results", date)
            mask = _load_wet_snow_mask(mask_path)
            member_masks.append(np.where(mask == 255, np.nan, mask.astype(float)))
            member_weights.append(1.0)

    if not member_masks:
        raise FileNotFoundError(f"Missing weighted posterior members for wet snow line altitude (WSLA) map in {step_dir}")

    stack = np.stack(member_masks, axis=0)
    weight_arr = np.asarray(member_weights, dtype=float)
    valid_weight = np.sum(np.where(np.isfinite(stack), weight_arr[:, None, None], 0.0), axis=0)
    wet_weight = np.nansum(stack * weight_arr[:, None, None], axis=0)
    wet_fraction = np.divide(
        wet_weight,
        valid_weight,
        out=np.full(valid_weight.shape, np.nan, dtype=float),
        where=valid_weight > 0.0,
    )
    wet_fraction = np.asarray(wet_fraction, dtype=float)
    wet_fraction[~context.roi_mask] = np.nan
    if derived_cache is not None:
        derived_cache[cache_key] = wet_fraction
    return wet_fraction


def _observation_array_on_model_grid(context: StaticContext, scene: ObservationScene) -> np.ndarray:
    arr = np.asarray(scene.array, dtype=float)
    if arr.shape == context.roi_mask.shape and tuple(scene.transform) == tuple(context.spec.transform):
        return arr
    out = np.full(context.roi_mask.shape, np.nan, dtype=float)
    reproject(
        source=arr,
        destination=out,
        src_transform=scene.transform,
        src_crs=context.spec.crs,
        dst_transform=context.spec.transform,
        dst_crs=context.spec.crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return out


def _wet_snow_elevation_fraction_array(
    *,
    context: StaticContext,
    source: str | None,
    date: pd.Timestamp,
    weights_variable: str,
    obs_cache,
    derived_cache: dict[str, np.ndarray] | None,
    observation_loader: Callable[..., ObservationScene],
) -> np.ndarray:
    cache_key = f"wet-snow-elevation-fraction:{weights_variable}:{source or 'observation'}:{pd.Timestamp(date).normalize().isoformat()}"
    if derived_cache is not None and cache_key in derived_cache:
        return np.asarray(derived_cache[cache_key], dtype=float)

    if source == "open_loop":
        classified = _wet_snow_model_classified_array(
            context=context,
            source="open_loop",
            date=date,
            derived_cache=derived_cache,
        )
        valid = np.isclose(classified, float(_WET_SNOW_MODEL_CODES[0]), equal_nan=False) | np.isclose(
            classified,
            float(_WET_SNOW_MODEL_CODES[1]),
            equal_nan=False,
        )
        wet_fraction = np.full(classified.shape, np.nan, dtype=float)
        wet_fraction[valid] = 0.0
        wet_fraction[np.isclose(classified, float(_WET_SNOW_MODEL_CODES[0]), equal_nan=False)] = 1.0
    elif source in {"prior_probability", "ensemble_mean"}:
        wet_fraction = _prior_wet_fraction_array(
            context=context,
            date=date,
            derived_cache=derived_cache,
        )
        valid = np.isfinite(wet_fraction)
    elif source in {"posterior", "posterior_probability"}:
        wet_fraction = _posterior_weighted_wet_fraction_array(
            context=context,
            date=date,
            derived_cache=derived_cache,
            weights_variable=weights_variable,
        )
        valid = np.isfinite(wet_fraction)
    elif source is None:
        obs_key = ("wet_snow", date)
        if obs_key not in obs_cache:
            obs_cache[obs_key] = observation_loader(context.project_dir, context, observation="wet_snow", date=date)
        scene = obs_cache[obs_key]
        arr = _observation_array_on_model_grid(context, scene)
        wet = context.roi_mask & np.isclose(arr, float(_WET_SNOW_MODEL_CODES[0]), equal_nan=False)
        dry = context.roi_mask & np.isclose(arr, float(_WET_SNOW_MODEL_CODES[1]), equal_nan=False)
        valid = wet | dry
        wet_fraction = np.full(arr.shape, np.nan, dtype=float)
        wet_fraction[dry] = 0.0
        wet_fraction[wet] = 1.0
    else:
        raise ValueError(f"Unsupported wet snow elevation band source '{source}'")

    out = _elevation_band_fraction_map(context=context, wet_fraction=wet_fraction, valid_mask=valid)
    if derived_cache is not None:
        derived_cache[cache_key] = out
    return out


def _observed_wet_snow_line_value(context: StaticContext, date: pd.Timestamp) -> float | None:
    diag_path = resolve_fraction_summary_path(
        context.setup_dir,
        context.project_dir,
        "wet_snow_line_diagnostics.csv",
    )
    df = load_fraction_series(diag_path, "wet_snow_line")
    if df is None or df.empty:
        return None
    working = df.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    row = working.loc[working["date"] == pd.Timestamp(date).normalize()]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[-1]["wet_snow_line"], errors="coerce")
    return None if pd.isna(value) else float(value)


def _contour_xy(context: StaticContext) -> tuple[np.ndarray, np.ndarray]:
    transform = context.spec.transform
    height, width = context.dem.shape
    xs = transform.c + (np.arange(width, dtype=float) + 0.5) * transform.a
    ys = transform.f + (np.arange(height, dtype=float) + 0.5) * transform.e
    return np.meshgrid(xs, ys)


_WSL_MODEL_COLOR = da_variable_line_color("wet_snow_line")
_WSL_PANEL_EXTENT_PAD_RATIO = 0.02


def _padded_wsl_panel_extent(extent: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    xmin, xmax, ymin, ymax = (float(value) for value in extent)
    dx = (xmax - xmin) * _WSL_PANEL_EXTENT_PAD_RATIO
    dy = (ymax - ymin) * _WSL_PANEL_EXTENT_PAD_RATIO
    return (xmin - dx, xmax + dx, ymin - dy, ymax + dy)


def _set_contour_path_effects(contour, effects: list[matplotlib.patheffects.AbstractPathEffect]) -> None:
    if hasattr(contour, "set_path_effects"):
        contour.set_path_effects(effects)
        return
    for collection in contour.collections:
        collection.set_path_effects(effects)


def _draw_wsl_contour(
    ax,
    *,
    context: StaticContext,
    level: float | None,
    color: str = _WSL_MODEL_COLOR,
    linestyle: str = "-",
    linewidth: float = 1.6,
    zorder: float = 9,
) -> bool:
    if level is None or not np.isfinite(level):
        return False
    dem = np.asarray(context.dem, dtype=float)
    finite = dem[np.isfinite(dem)]
    if finite.size == 0:
        return False
    if float(level) < float(np.nanmin(finite)) or float(level) > float(np.nanmax(finite)):
        return False
    contour_dem = np.array(dem, copy=True)
    contour_dem[~np.asarray(context.roi_mask, dtype=bool)] = np.nan
    if not np.isfinite(contour_dem).any():
        return False
    xx, yy = _contour_xy(context)
    contour = ax.contour(
        xx,
        yy,
        contour_dem,
        levels=[float(level)],
        colors=[color],
        linewidths=linewidth,
        linestyles=linestyle,
        zorder=zorder,
    )
    _set_contour_path_effects(
        contour,
        [
            matplotlib.patheffects.Stroke(linewidth=2.2, foreground="white"),
            matplotlib.patheffects.Normal(),
        ],
    )
    return True


def _wsl_callout_text(level: float | None) -> str:
    if level is None or not np.isfinite(level):
        return "WSLA unavailable"
    rounded = int(10.0 * np.floor((float(level) / 10.0) + 0.5))
    return f"WSLA {rounded} m"


def _annotate_wsl_callout(ax, *, level: float | None, color: str = "black") -> None:
    apply_overlay_label_halo(
        ax.text(
            0.02,
            0.045,
            _wsl_callout_text(level),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=_MAP_SUPPORT_TEXT_SIZE,
            color=color,
            zorder=_ANNOTATION_ZORDER + 2,
            bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": _DATE_CALLOUT_ALPHA},
        ),
        with_bbox=True,
    )


def _wet_snow_line_legend_handles(
    base_handles: list[object],
    *,
    include_model_wsl: bool,
    include_obs_wsl: bool,
    obs_linestyle: str = "-",
) -> list[object]:
    handles = list(base_handles)
    if include_model_wsl:
        handles.append(Line2D([0], [0], color=_WSL_MODEL_COLOR, linewidth=1.6, label="model WSLA"))
    if include_obs_wsl:
        handles.append(Line2D([0], [0], color=_WSL_MODEL_COLOR, linewidth=1.6, linestyle=obs_linestyle, label="observation WSLA"))
    return handles


def _draw_inpanel_wsl_legend(ax, handles: list[object]) -> None:
    if not handles:
        return
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.87),
        frameon=True,
        fancybox=True,
        framealpha=_DATE_CALLOUT_ALPHA,
        facecolor="white",
        edgecolor="none",
        fontsize=_MAP_SUPPORT_TEXT_SIZE,
        handlelength=1.5,
        handletextpad=0.45,
        borderpad=0.25,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    legend.set_zorder(_ANNOTATION_ZORDER + 1)


def _scf_binary_grid_from_results(
    *,
    context: StaticContext,
    results_dir: Path,
    date: pd.Timestamp,
) -> np.ndarray:
    _method, variable, params = load_hofx_from_project(context.project_dir)
    lc_cfg = resolve_landcover_mask(context.setup_dir, context.project_dir)
    roi_path = ensure_setup_roi_vector(context.setup_dir)
    try:
        return compute_model_scf_binary_grid(
            setup_dir=context.setup_dir,
            project_dir=context.project_dir,
            results_dir=results_dir,
            aoi_path=roi_path,
            landcover_cfg=lc_cfg,
            apply_landcover_mask=False,
            date=date.to_pydatetime(),
            variable=variable,  # type: ignore[arg-type]
            params=params,
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Cannot render spatial SCF DA-event map support for "
            f"{pd.Timestamp(date).date()} from {results_dir}: {exc}. "
            "The required step/member grids may have been removed by compact output retention. "
            "For exact DA-event map regeneration in subdomain runs, run with "
            "data_assimilation.output.retention: full."
        ) from exc


def _top_level_subdomain_manifest(context: StaticContext) -> SubdomainManifest | None:
    manifest_path = Path(context.project_dir) / "subdomains" / "subdomain_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = SubdomainManifest.load(manifest_path)
    except Exception:
        return None
    if str(getattr(manifest, "run_mode", "")).lower() != "subdomain":
        return None
    return manifest


def _window_slices_for_subdomain(
    sub: SubdomainMeta,
    data_shape: tuple[int, int],
    global_shape: tuple[int, int],
) -> tuple[slice, slice]:
    if data_shape == global_shape:
        return slice(0, global_shape[0]), slice(0, global_shape[1])
    return (
        slice(sub.window.row_off, sub.window.row_off + data_shape[0]),
        slice(sub.window.col_off, sub.window.col_off + data_shape[1]),
    )


def _subdomain_roi_mask(sub: SubdomainMeta) -> np.ndarray:
    with rasterio.open(sub.roi_raster_path) as src:
        return src.read(1).astype(bool)


def _single_domain_scf_model_probability_array(
    *,
    context: StaticContext,
    source: str,
    date: pd.Timestamp,
    derived_cache: dict[str, np.ndarray] | None,
) -> np.ndarray:
    step_dir = _step_dir_for_date(context.project_dir, date)
    if source == "open_loop_binary":
        classified = _scf_binary_grid_from_results(
            context=context,
            results_dir=step_dir / "ensembles" / "prior" / "open_loop" / "results",
            date=date,
        )
    elif source in {"prior_probability", "posterior_probability"}:
        member_fields: list[np.ndarray] = []
        ensemble = "prior" if source == "prior_probability" else "posterior"
        for member_dir in list_member_dirs(step_dir, ensemble):
            source_member_dir = member_dir if ensemble == "prior" else resolve_member_source_dir(member_dir)
            member_fields.append(
                _scf_binary_grid_from_results(
                    context=context,
                    results_dir=source_member_dir / "results",
                    date=date,
                )
            )
        if not member_fields:
            raise FileNotFoundError(f"Missing {ensemble} members for SCF probability map in {step_dir}")
        stack = np.stack(member_fields, axis=0)
        valid_count = np.sum(np.isfinite(stack), axis=0)
        classified = np.divide(
            np.nansum(stack, axis=0),
            valid_count,
            out=np.full(valid_count.shape, np.nan, dtype=float),
            where=valid_count > 0,
        )
    else:
        raise ValueError(f"Unsupported SCF model source '{source}'")

    classified = np.asarray(classified, dtype=float)
    classified[~context.roi_mask] = np.nan
    return classified


def _top_level_subdomain_scf_model_probability_array(
    *,
    context: StaticContext,
    source: str,
    date: pd.Timestamp,
) -> np.ndarray:
    manifest = _top_level_subdomain_manifest(context)
    if manifest is None or not manifest.subdomains:
        raise FileNotFoundError(
            f"Cannot render top-level subdomain spatial SCF DA-event map support for {pd.Timestamp(date).date()}: "
            "subdomain_manifest.json is missing or contains no subdomains."
        )

    global_shape = tuple(context.roi_mask.shape)
    mosaic = np.full(global_shape, np.nan, dtype=float)
    dropped_ids = _subdomain_dropped_event_ids(context, date=date, variable="scf")
    missing: list[str] = []
    for sid, sub in manifest.subdomains.items():
        if str(sid) in dropped_ids:
            continue
        try:
            sub_context = load_static_context(sub.project_dir)
            local = _single_domain_scf_model_probability_array(
                context=sub_context,
                source=source,
                date=date,
                derived_cache=None,
            )
            roi = _subdomain_roi_mask(sub)
            sl_r, sl_c = _window_slices_for_subdomain(sub, local.shape, global_shape)
            mask = roi
            if mask.shape != local.shape:
                if mask.shape == global_shape:
                    mask = mask[sl_r, sl_c]
                else:
                    mask = np.ones(local.shape, dtype=bool)
            local = np.where(mask, local, np.nan)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
            missing.append(f"{sid}: {exc}")
            continue

        dest = mosaic[sl_r, sl_c]
        replace = np.isnan(dest) & np.isfinite(local)
        dest[replace] = local[replace]
        mosaic[sl_r, sl_c] = dest

    if missing:
        details = "; ".join(missing[:5])
        if len(missing) > 5:
            details += f"; ... {len(missing) - 5} more"
        raise FileNotFoundError(
            f"Cannot render top-level subdomain spatial SCF DA-event map support for {pd.Timestamp(date).date()}. "
            "Retained per-subdomain step/member grids are required for exact snow-cover panels, but some support "
            f"data are missing ({details}). Compact-cleaned runs cannot be rerendered exactly; future subdomain "
            "runs should use data_assimilation.output.retention: full."
        )

    mosaic[~context.roi_mask] = np.nan
    return mosaic


def _scf_model_probability_array(
    *,
    context: StaticContext,
    source: str,
    date: pd.Timestamp,
    derived_cache: dict[str, np.ndarray] | None,
) -> np.ndarray:
    cache_key = f"scf-model-map:{source}:{pd.Timestamp(date).normalize().isoformat()}"
    if derived_cache is not None and cache_key in derived_cache:
        return np.asarray(derived_cache[cache_key], dtype=float)

    if _top_level_subdomain_manifest(context) is not None:
        classified = _top_level_subdomain_scf_model_probability_array(
            context=context,
            source=source,
            date=date,
        )
    else:
        classified = _single_domain_scf_model_probability_array(
            context=context,
            source=source,
            date=date,
            derived_cache=derived_cache,
        )

    if derived_cache is not None:
        derived_cache[cache_key] = classified
    return classified


def _load_fraction_model_value(
    *,
    project_dir,
    observation: str,
    source: str,
    date: pd.Timestamp,
    derived_cache: dict[str, np.ndarray] | None = None,
) -> float:
    cache_key = f"fraction-model:{observation}:{source}:{date.isoformat()}"
    if derived_cache is not None and cache_key in derived_cache:
        return float(derived_cache[cache_key])

    value_col = _fraction_value_column(observation)
    if source == "open_loop":
        df = load_open_loop_fraction_series(project_dir, _fraction_member_filename(observation), value_col)
    elif source == "ensemble_mean":
        df = load_fraction_series(project_fraction_envelope_path(project_dir, observation), "value_mean")
        value_col = "value_mean"
    else:
        raise ValueError(f"Unsupported fraction model source '{source}' for observation panel")

    if df is None or df.empty:
        raise FileNotFoundError(f"Missing model fraction series for {observation} ({source})")

    working = df.copy()
    working["date"] = pd.to_datetime(working["date"]).dt.normalize()
    row = working.loc[working["date"] == date]
    if row.empty:
        raise KeyError(f"No model fraction value for {observation} ({source}) on {date.date()}")
    value = float(row.iloc[-1][value_col])
    if derived_cache is not None:
        derived_cache[cache_key] = np.asarray(value, dtype=float)
    return value


def render_wet_snow_line_panel(
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
    date = pd.Timestamp(date).normalize()
    display_extent = _padded_wsl_panel_extent(extent)
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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

    obs_contour_level = _observed_wet_snow_line_value(context, date)
    if panel.source is None:
        obs_key = ("wet_snow", date)
        if obs_key not in obs_cache:
            obs_cache[obs_key] = observation_loader(context.project_dir, context, observation="wet_snow", date=date)
        scene = obs_cache[obs_key]
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
        image = ax.imshow(
            np.ma.masked_invalid(categorical),
            cmap=cmap,
            norm=norm,
            extent=scene.bounds,
            origin="upper",
            interpolation="nearest",
            zorder=5,
        )
        invalid_mask = (scene.invalid_mask if scene.invalid_mask is not None else np.zeros(scene.array.shape, dtype=bool)) | (
            scene.roi_mask & ~np.isfinite(scene.array)
        )
        overlay_invalid_inside_roi(ax, invalid_mask, extent=scene.bounds)
        legend_handles = classified_legend_handles(
            canonical_codes=codes,
            present_codes=present_codes,
            label_lookup=WET_SNOW_LABELS,
            color_lookup=lambda code: WET_SNOW_COLORS[code],
            fallback_codes=codes,
        )
        contour_level = obs_contour_level
    else:
        source = str(panel.source)
        if source in {"prior_probability", "posterior", "posterior_probability"}:
            wet_fraction = _posterior_weighted_wet_fraction_array(
                context=context,
                date=date,
                derived_cache=derived_cache,
                weights_variable="wet_snow_line",
            ) if source in {"posterior", "posterior_probability"} else _prior_wet_fraction_array(
                context=context,
                date=date,
                derived_cache=derived_cache,
            )
            fill = 100.0 * np.asarray(wet_fraction, dtype=float)
            norm = Normalize(vmin=0.0, vmax=100.0)
            contour_level = _wet_snow_line_from_fraction(context=context, wet_fraction=wet_fraction)
            image = ax.imshow(
                np.ma.masked_invalid(fill),
                cmap=_FRACTION_MODEL_CMAP,
                norm=norm,
                extent=grid_extent(context),
                origin="upper",
                interpolation="nearest",
                alpha=0.92,
                zorder=5,
            )
            overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(fill, context.roi_mask), extent=grid_extent(context))
            legend_handles = None
        else:
            classified = _wet_snow_model_classified_array(
                context=context,
                source=str(panel.source),
                date=date,
                derived_cache=derived_cache,
            )
            contour_level = _wet_snow_line_from_classified(context=context, classified=classified)
            code_to_index = {code: idx for idx, code in enumerate(_WET_SNOW_MODEL_CODES)}
            categorical = np.full(classified.shape, np.nan, dtype=float)
            for code, idx in code_to_index.items():
                categorical[np.isclose(classified, float(code), equal_nan=False)] = idx
            cmap = matplotlib.colors.ListedColormap([WET_SNOW_COLORS[code] for code in _WET_SNOW_MODEL_CODES], name="wet_snow_model")
            cmap.set_bad((1.0, 1.0, 1.0, 0.0))
            norm = BoundaryNorm(np.arange(-0.5, len(_WET_SNOW_MODEL_CODES) + 0.5), cmap.N)
            image = ax.imshow(
                np.ma.masked_invalid(categorical),
                cmap=cmap,
                norm=norm,
                extent=grid_extent(context),
                origin="upper",
                interpolation="nearest",
                zorder=5,
            )
            overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(classified, context.roi_mask), extent=grid_extent(context))
            present_codes = {
                code for code in _WET_SNOW_MODEL_CODES if np.any(np.isclose(classified, float(code), equal_nan=False))
            }
            legend_handles = classified_legend_handles(
                canonical_codes=list(_WET_SNOW_MODEL_CODES),
                present_codes=present_codes,
                label_lookup=WET_SNOW_LABELS,
                color_lookup=lambda code: WET_SNOW_COLORS[code],
                fallback_codes=list(_WET_SNOW_MODEL_CODES),
            )

    probability_panel = str(panel.source) in {"prior_probability", "posterior", "posterior_probability"}
    model_contour_drawn = False
    model_contour_drawn = _draw_wsl_contour(
        ax,
        context=context,
        level=contour_level,
        color=_WSL_MODEL_COLOR,
        linestyle="-",
        zorder=9.5 if panel.source is not None else 9,
    )
    obs_contour_drawn = False
    callout_color = "black"
    if contour_level is not None and np.isfinite(contour_level):
        callout_color = _WSL_MODEL_COLOR
    _annotate_wsl_callout(ax, level=contour_level, color=callout_color)

    apply_common_overlays(
        ax,
        context=context,
        extent=display_extent,
        show_roi=resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=resolve_panel_toggle(panel.show_station_marker, False),
        show_stations_name=resolve_panel_toggle(panel.show_stations_name, False),
        show_stations_elev=resolve_panel_toggle(panel.show_stations_elev, False),
    )
    apply_map_axis_style(
        ax,
        display_extent,
        title=panel_title(label, panel_semantic_title(panel)),
        show_grid=show_grid,
        show_y_ticklabels=panel.col == 0,
    )
    posterior_overlay_handles: list[object] = []
    if probability_panel:
        if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
            probability_label = "posterior" if str(panel.source) in {"posterior", "posterior_probability"} else "prior"
            attach_colorbar(
                ax,
                image,
                label=f"{probability_label} wet snow fraction (WSF) probability [%]",
                ticks=(0, 20, 40, 60, 80, 100),
                layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
            )
        if model_contour_drawn:
            label_text = "posterior WSLA" if str(panel.source) in {"posterior", "posterior_probability"} else "prior WSLA"
            posterior_overlay_handles.append(Line2D([0], [0], color=_WSL_MODEL_COLOR, linewidth=1.6, linestyle="-", label=label_text))
    else:
        draw_classified_legend(
            ax,
            _wet_snow_line_legend_handles(
                list(legend_handles),
                include_model_wsl=model_contour_drawn and panel.source is not None,
                include_obs_wsl=model_contour_drawn if panel.source is None else obs_contour_drawn,
            ),
            layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default),
        )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=display_extent, date=date, resolve_flag=resolve_flag)
    if probability_panel:
        _draw_inpanel_wsl_legend(ax, posterior_overlay_handles)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {
        "mappable": image,
        "legend_handles": legend_handles,
        "posterior_overlay_handles": posterior_overlay_handles,
        "wsl": contour_level,
        "obs_wsl": obs_contour_level,
        "model_wsl_drawn": model_contour_drawn,
        "obs_wsl_drawn": obs_contour_drawn,
    }


def render_wet_snow_elevation_fraction_panel(
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
    date = pd.Timestamp(date).normalize()
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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

    values = _wet_snow_elevation_fraction_array(
        context=context,
        source=panel.source,
        date=date,
        weights_variable=str(panel.variable or "wet_snow_line"),
        obs_cache=obs_cache,
        derived_cache=derived_cache,
        observation_loader=observation_loader,
    )
    image = ax.imshow(
        np.ma.masked_invalid(100.0 * values),
        cmap=_ELEVATION_BAND_WSF_CMAP,
        norm=Normalize(vmin=0.0, vmax=100.0),
        extent=grid_extent(context),
        origin="upper",
        interpolation="nearest",
        alpha=1.0,
        zorder=5,
    )
    wsl_level = (
        _observed_wet_snow_line_value(context, date)
        if panel.source is None
        else _wet_snow_line_from_fraction(context=context, wet_fraction=values)
    )
    wsl_color = _WSL_MODEL_COLOR
    wsl_drawn = _draw_wsl_contour(
        ax,
        context=context,
        level=wsl_level,
        color=wsl_color,
        linestyle="-",
        linewidth=1.5,
        zorder=9.5,
    )
    _annotate_wsl_callout(ax, level=wsl_level, color=wsl_color if wsl_drawn else "black")
    overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(values, context.roi_mask), extent=grid_extent(context))
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
    if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        attach_colorbar(
            ax,
            image,
            label="wet snow fraction by elevation band [%]",
            ticks=(0, 20, 40, 60, 80, 100),
            layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
    )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "values": values, "wsl": wsl_level, "wsl_drawn": wsl_drawn}


def render_fraction_model_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    label: str | None,
    defaults: MapDefaults,
    figure_horizontal_default: bool,
    derived_cache: dict[str, np.ndarray] | None,
    observation: str,
    date: pd.Timestamp,
) -> dict[str, object]:
    if observation == "scf" and str(panel.source) in {"open_loop_binary", "prior_probability", "posterior_probability"}:
        scf_field = _scf_model_probability_array(
            context=context,
            source=str(panel.source),
            date=date,
            derived_cache=derived_cache,
        )
        show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
        if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
            hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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
        invalid_mask = inside_roi_invalid_mask(scf_field, context.roi_mask)
        is_probability = str(panel.source) in {"prior_probability", "posterior_probability"}
        if is_probability:
            norm = Normalize(vmin=0.0, vmax=100.0)
            image = ax.imshow(
                np.ma.masked_invalid(100.0 * scf_field),
                cmap=FSC_OBS_CMAP,
                norm=norm,
                extent=grid_extent(context),
                origin="upper",
                interpolation="nearest",
                alpha=0.95,
                zorder=5,
            )
        else:
            norm = BoundaryNorm(boundaries=(-0.5, 0.5, 1.5), ncolors=_SCF_BINARY_CMAP.N)
            image = ax.imshow(
                np.ma.masked_invalid(scf_field),
                cmap=_SCF_BINARY_CMAP,
                norm=norm,
                extent=grid_extent(context),
                origin="upper",
                interpolation="nearest",
                alpha=0.95,
                zorder=5,
            )
        overlay_invalid_inside_roi(ax, invalid_mask, extent=grid_extent(context))
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
        if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
            if str(panel.source) == "prior_probability":
                colorbar_label = "prior snow cover probability [%]"
            elif str(panel.source) == "posterior_probability":
                colorbar_label = "posterior snow cover probability [%]"
            else:
                colorbar_label = "binary snow cover [%]"
            ticks = (0, 20, 40, 60, 80, 100) if is_probability else (0, 1)
            ticklabels = () if is_probability else ("0", "100")
            attach_colorbar(
                ax,
                image,
                label=colorbar_label,
                ticks=ticks,
                ticklabels=ticklabels,
                layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
            )
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image}

    if observation == "wet_snow" and str(panel.source) in {"prior_probability", "posterior_probability"}:
        wet_fraction = (
            _prior_wet_fraction_array(context=context, date=date, derived_cache=derived_cache)
            if str(panel.source) == "prior_probability"
            else _posterior_weighted_wet_fraction_array(
                context=context,
                date=date,
                derived_cache=derived_cache,
                weights_variable="wet_snow",
            )
        )
        show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
        if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
            hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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
        fill = 100.0 * np.asarray(wet_fraction, dtype=float)
        norm = Normalize(vmin=0.0, vmax=100.0)
        image = ax.imshow(
            np.ma.masked_invalid(fill),
            cmap=_FRACTION_MODEL_CMAP,
            norm=norm,
            extent=grid_extent(context),
            origin="upper",
            interpolation="nearest",
            alpha=0.95,
            zorder=5,
        )
        overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(fill, context.roi_mask), extent=grid_extent(context))
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
        if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
            colorbar_label = (
                "prior wet snow fraction (WSF) probability [%]"
                if str(panel.source) == "prior_probability"
                else "posterior wet snow fraction (WSF) probability [%]"
            )
            attach_colorbar(
                ax,
                image,
                label=colorbar_label,
                ticks=(0, 20, 40, 60, 80, 100),
                layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
            )
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image, "values": wet_fraction}

    if observation == "wet_snow":
        classified = _wet_snow_model_classified_array(
            context=context,
            source=str(panel.source),
            date=date,
            derived_cache=derived_cache,
        )
        show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
        if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
            hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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
        code_to_index = {code: idx for idx, code in enumerate(_WET_SNOW_MODEL_CODES)}
        categorical = np.full(classified.shape, np.nan, dtype=float)
        for code, idx in code_to_index.items():
            categorical[np.isclose(classified, float(code), equal_nan=False)] = idx
        cmap = matplotlib.colors.ListedColormap([WET_SNOW_COLORS[code] for code in _WET_SNOW_MODEL_CODES], name="wet_snow_model")
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        norm = BoundaryNorm(np.arange(-0.5, len(_WET_SNOW_MODEL_CODES) + 0.5), cmap.N)
        image = ax.imshow(
            np.ma.masked_invalid(categorical),
            cmap=cmap,
            norm=norm,
            extent=grid_extent(context),
            origin="upper",
            interpolation="nearest",
            zorder=5,
        )
        overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(classified, context.roi_mask), extent=grid_extent(context))
        present_codes = {
            code for code in _WET_SNOW_MODEL_CODES if np.any(np.isclose(classified, float(code), equal_nan=False))
        }
        legend_handles = classified_legend_handles(
            canonical_codes=list(_WET_SNOW_MODEL_CODES),
            present_codes=present_codes,
            label_lookup=WET_SNOW_LABELS,
            color_lookup=lambda code: WET_SNOW_COLORS[code],
            fallback_codes=list(_WET_SNOW_MODEL_CODES),
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
        draw_classified_legend(ax, legend_handles, layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default))
        draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
        draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image, "legend_handles": legend_handles}

    value = _load_fraction_model_value(
        project_dir=context.project_dir,
        observation=observation,
        source=str(panel.source),
        date=date,
        derived_cache=derived_cache,
    )
    show_grid = resolve_flag(panel.show_grid, defaults, "show_grid", True)
    if resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        hillshade_mode = resolve_hillshade_extent(panel, defaults, builtin="roi")
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

    fill = np.full(context.roi_mask.shape, np.nan, dtype=float)
    fill[context.roi_mask] = 100.0 * value
    if observation == "scf":
        cmap = FSC_OBS_CMAP
        colorbar_label = "fractional snow cover [%]"
    else:
        cmap = _FRACTION_MODEL_CMAP
        colorbar_label = "wet snow fraction (WSF) [%]"
    norm = Normalize(vmin=0.0, vmax=100.0)
    image = ax.imshow(
        np.ma.masked_invalid(fill),
        cmap=cmap,
        norm=norm,
        extent=grid_extent(context),
        origin="upper",
        interpolation="nearest",
        alpha=0.9,
        zorder=5,
    )
    overlay_invalid_inside_roi(ax, inside_roi_invalid_mask(fill, context.roi_mask), extent=grid_extent(context))
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
    if resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        attach_colorbar(
            ax,
            image,
            label=colorbar_label,
            ticks=(0, 20, 40, 60, 80, 100),
            layout=panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
        )
    draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date, resolve_flag=resolve_flag)
    draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image}


def legend_source_handles(item: LegendItemSpec, artifacts: dict[str, dict[str, object]]) -> list[object]:
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
    style = source.get("colorbar_style") or {}
    label = style.get("label") if isinstance(style, dict) else getattr(style, "label", None)
    ticks = style.get("ticks", ()) if isinstance(style, dict) else getattr(style, "ticks", ())
    ticklabels = style.get("ticklabels", ()) if isinstance(style, dict) else getattr(style, "ticklabels", ())
    title = extract_unit_title(label)
    cax = ax
    if title and _COLORBAR_PANEL_UNIT_HEADER_ENABLED:
        header_height = _COLORBAR_PANEL_UNIT_HEADER_HEIGHT_AXES
        gap = _COLORBAR_PANEL_UNIT_HEADER_GAP_AXES
        cax = ax.inset_axes([0.0, 0.0, 1.0, max(1.0 - header_height - gap, 0.1)], transform=ax.transAxes)
        register_child_axes(ax, cax)
        ax.text(
            0.5,
            1.0,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=_COLORBAR_TITLE_SIZE,
        )
    cbar = plt.colorbar(source["mappable"], cax=cax, orientation="vertical")
    if ticks:
        cbar.set_ticks(ticks)
    if ticklabels:
        cbar.set_ticklabels(ticklabels)
    _apply_poster_colorbar_ticks(cbar, label=label, ticks=ticks)
    cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE, pad=1.0)
    if title and not _COLORBAR_PANEL_UNIT_HEADER_ENABLED:
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
                if isinstance(handle, Patch):
                    y = draw_patch_entry(ax, y=y, label=handle.get_label(), facecolor=handle.get_facecolor(), edgecolor=handle.get_edgecolor())
                elif isinstance(handle, Line2D):
                    ax.plot([0.02, 0.12], [y - 0.02, y - 0.02], transform=ax.transAxes, color=handle.get_color(), lw=handle.get_linewidth(), ls=handle.get_linestyle(), solid_capstyle="round")
                    ax.text(
                        0.15,
                        y - 0.02,
                        handle.get_label(),
                        transform=ax.transAxes,
                        ha="left",
                        va="center",
                        fontsize=_MAP_SUPPORT_TEXT_SIZE,
                    )
                    y -= 0.065
                else:
                    raise TypeError(f"Unsupported legend source handle type: {type(handle)!r}")
        elif item.kind == "scale_bar":
            continue
        else:
            raise ValueError(f"Unsupported legend item kind '{item.kind}'")
    return {}
