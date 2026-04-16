from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, LightSource, Normalize, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from rasterio.transform import array_bounds
from shapely.geometry import box

from openamundsen_da.methods.viz._style import FIGWIDTH_OVERVIEW_PAPER
from openamundsen_da.methods.viz._utils import force_figure_text_black, save_figure_png
from openamundsen_da.methods.viz.project_maps.config import LegendItemSpec, MapDefaults, MapPanelSpec, MapRecipe
from openamundsen_da.methods.viz.project_maps.data import (
    ModelFields,
    ObservationScene,
    StaticContext,
    load_model_fields,
    load_observation_scene,
)
from openamundsen_da.methods.viz.project_maps.overview import load_overview_boundaries
from openamundsen_da.methods.viz.project_maps.styles import (
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


_FIGURE_HEIGHT_MIN = 2.9
_FIGURE_HEIGHT_MAX = 11.5
_BUFFER_RATIO = 0.03
_STATION_LABEL_RATIO = 0.04
_GRID_COLOR = "#666666"
_GRID_STYLE = (0, (4, 4))
_GRID_WIDTH = 0.74
_SPINE_WIDTH = 0.85
_TICK_SIZE = 6.6
_COLORBAR_TICK_SIZE = 6.0
_COLORBAR_TITLE_SIZE = 6.6
_SUPPORT_PANEL_KINDS = {"colorbar", "legend"}
_STATION_COLOR = "#d94801"
_ROI_FILL = "#efefef"
_OVERVIEW_ROI_COLOR = "#c21f24"
_GRID_ZORDER = 120
_ANNOTATION_ZORDER = 130
_LAYOUT_COL_GAP = 0.11
_LAYOUT_ROW_GAP = 0.35
_LEFT_MARGIN = 0.03
_RIGHT_MARGIN = 0.992
_BOTTOM_MARGIN = 0.028
_TOP_MARGIN = 0.992
_DATE_CALLOUT_ALPHA = 0.30
_SCALEBAR_X_FRACTION = 0.68
_OVERVIEW_FRAGMENT_RATIO = 0.018
_FIGURE_HORIZONTAL_COLORBAR_EXTRA = 0.18
_FIGURE_HORIZONTAL_LEGEND_BASE_EXTRA = 0.12
_FIGURE_HORIZONTAL_LEGEND_ROW_EXTRA = 0.07

_MODEL_KIND_TO_VARIABLE = {
    "snow_depth": "snowdepth_daily",
    "swe": "swe_daily",
    "liquid_water_content": "liquid_water_content",
}
_OBSERVATION_KIND_TO_NAME = {
    "fsc": "scf",
    "wet_snow": "wet_snow",
}
_STATIC_FIELD_KIND_TO_FIELD = {
    "dem": "dem",
    "svf": "svf",
    "srf": "srf",
    "landcover": "landcover",
}
_CLASSIFIED_PANEL_KINDS = {"landcover", "wet_snow"}
_CONTINUOUS_COLORBAR_PANEL_KINDS = {"dem", "svf", "srf", "snow_depth", "swe", "liquid_water_content", "fsc"}
_AUTO_TITLE_SOURCE = {
    "open_loop": "Open loop",
    "ensemble_mean": "Ensemble mean",
    "increment": "Increment",
}
_AUTO_TITLE_KIND = {
    "overview": "Overview map",
    "roi": "Region of interest",
    "hillshade": "Hillshade",
    "dem": "Digital elevation model",
    "svf": "Sky view factor",
    "srf": "Snow redistribution factor",
    "landcover": "Landcover",
    "snow_depth": "snow depth",
    "swe": "SWE",
    "liquid_water_content": "liquid water content",
    "fsc": "Sentinel-2 FSC",
    "wet_snow": "Sentinel-1 wet snow",
}


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


def _grid_extent(context: StaticContext) -> tuple[float, float, float, float]:
    left, bottom, right, top = array_bounds(
        int(context.roi_mask.shape[0]),
        int(context.roi_mask.shape[1]),
        context.spec.transform,
    )
    return (float(left), float(right), float(bottom), float(top))


def _masked(arr: np.ndarray, roi_mask: np.ndarray) -> np.ma.MaskedArray:
    masked = np.asarray(arr, dtype=float).copy()
    masked[~roi_mask] = np.nan
    return np.ma.masked_invalid(masked)


def _masked_model(arr: np.ndarray, roi_mask: np.ndarray, *, preset) -> np.ma.MaskedArray:
    masked = _masked(arr, roi_mask)
    if preset.variable == "snowdepth_daily":
        masked = np.ma.masked_less(masked, SNOW_DEPTH_REFERENCE_TICKS_M[0])
    return masked


def _field_array(context: StaticContext, field: str) -> np.ndarray:
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


def _hillshade(context: StaticContext) -> np.ndarray:
    dem = np.asarray(context.dem, dtype=float)
    filled = dem.copy()
    if np.isfinite(filled).any():
        filled[~np.isfinite(filled)] = float(np.nanmedian(filled))
    else:
        filled[:] = 0.0
    light = LightSource(azdeg=315, altdeg=45)
    shade = light.hillshade(
        filled,
        vert_exag=1.3,
        dx=abs(float(context.spec.transform.a)),
        dy=abs(float(context.spec.transform.e)),
    )
    shade = np.where(context.roi_mask, shade, np.nan)
    return shade


def _draw_roi(ax, context: StaticContext, *, linewidth: float = 0.8, facecolor=None, alpha: float | None = None) -> None:
    if facecolor is not None:
        context.roi_gdf.plot(ax=ax, facecolor=facecolor, edgecolor=facecolor, alpha=alpha if alpha is not None else 1.0, zorder=40)
    context.roi_gdf.boundary.plot(ax=ax, color="black", linewidth=linewidth, zorder=45)


def _nice_tick_step(span: float) -> float:
    target = max(span / 4.0, 1.0)
    exponent = int(np.floor(np.log10(target)))
    base = 10**exponent
    candidates = [1.0 * base, 2.0 * base, 2.5 * base, 5.0 * base, 10.0 * base]
    return min(candidates, key=lambda value: abs(value - target))


def _ticks_for_extent(start: float, stop: float) -> np.ndarray:
    span = max(1.0, stop - start)
    step = _nice_tick_step(span)
    first = np.ceil(start / step) * step
    last = np.floor(stop / step) * step
    return np.arange(first, last + 0.5 * step, step)


def _coord_label(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}"


def _panel_semantic_title(panel: MapPanelSpec) -> str | None:
    if panel.title is not None:
        return panel.title
    if panel.kind in _MODEL_KIND_TO_VARIABLE:
        source = _AUTO_TITLE_SOURCE.get(str(panel.source or "").strip())
        target = _AUTO_TITLE_KIND.get(panel.kind)
        if source and target:
            return f"{source} {target}"
    return _AUTO_TITLE_KIND.get(panel.kind)


def _panel_title(letter: str | None, title: str | None) -> str | None:
    if letter is None and title is None:
        return None
    if letter is None:
        return title
    if title is None:
        return f"({letter})"
    return f"({letter}) {title}"


def _axis_width_inches(ax) -> float:
    bbox = ax.get_position()
    return float(ax.figure.get_size_inches()[0] * bbox.width)


def _axes_title_fontsize(ax) -> float:
    width_in = _axis_width_inches(ax)
    if width_in < 1.45:
        return 6.4
    if width_in < 1.8:
        return 7.0
    if width_in < 2.35:
        return 7.8
    if width_in < 2.8:
        return 8.6
    return 9.4


def _axes_date_fontsize(ax) -> float:
    width_in = _axis_width_inches(ax)
    if width_in < 1.45:
        return 5.8
    if width_in < 1.8:
        return 6.2
    if width_in < 2.35:
        return 6.8
    if width_in < 2.8:
        return 7.4
    return 8.0


def _legend_columns_for_width(width_in: float, labels: list[str]) -> int:
    if not labels:
        return 1
    max_chars = max(max(len(label), 6) for label in labels)
    estimated_col_width = max(0.72, 0.22 + 0.052 * max_chars)
    columns = int(np.floor(max(width_in * 0.92, estimated_col_width) / estimated_col_width))
    return max(1, min(len(labels), 3, columns))


def _legend_columns_for_axis(ax, labels: list[str]) -> int:
    return _legend_columns_for_width(_axis_width_inches(ax), labels)


def _apply_map_axis_style(ax, extent: tuple[float, float, float, float], *, title: str | None, show_grid: bool) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_linewidth(_SPINE_WIDTH)
        spine.set_color("black")
        spine.set_visible(True)

    xticks = _ticks_for_extent(extent[0], extent[1])
    yticks = _ticks_for_extent(extent[2], extent[3])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([_coord_label(value) if idx % 2 == 0 else "" for idx, value in enumerate(xticks)])
    ax.set_yticklabels([_coord_label(value) if idx % 2 == 0 else "" for idx, value in enumerate(yticks)])
    ax.tick_params(
        axis="x",
        direction="out",
        top=True,
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
        right=True,
        labelleft=True,
        labelright=True,
        labelsize=_TICK_SIZE,
        length=3.0,
        width=0.75,
        pad=4.0,
    )
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_rotation(0)
        tick.label1.set_ha("center")
        tick.label1.set_va("top")
    for tick in ax.yaxis.get_major_ticks():
        for label in (tick.label1, tick.label2):
            label.set_rotation(90)
            label.set_rotation_mode("anchor")
            label.set_ha("center")
            label.set_va("center")
    ax.grid(False)
    if title:
        ax.set_title(title, loc="left", fontsize=_axes_title_fontsize(ax), pad=2.8)


def _draw_map_grid_overlay(ax, *, show_grid: bool) -> None:
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


def _suppress_station_labels(stations, extent: tuple[float, float, float, float]) -> list[int]:
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


def _draw_stations_overlay(
    ax,
    context: StaticContext,
    extent: tuple[float, float, float, float],
    *,
    show_station_marker: bool,
    show_stations_name: bool,
    show_stations_elev: bool,
) -> None:
    stations = context.stations
    if stations is None or stations.empty or not {"id", "x", "y"}.issubset(stations.columns):
        return
    working = stations.copy()
    if "name" not in working.columns:
        working["name"] = working["id"]
    if show_station_marker:
        ax.scatter(
            working["x"],
            working["y"],
            s=44,
            marker="v",
            facecolor=_STATION_COLOR,
            edgecolor="none",
            zorder=30,
        )
    if not (show_station_marker and (show_stations_name or show_stations_elev)):
        return
    ordered = working.sort_values("id").reset_index(drop=True)
    for idx in _suppress_station_labels(working, extent):
        row = ordered.iloc[idx]
        label_parts: list[str] = []
        if show_stations_name:
            label_parts.append(str(row["name"]))
        alt = row.get("alt") if show_stations_elev else None
        if alt is not None and np.isfinite(float(alt)):
            label_parts.append(f"{int(round(float(alt)))} m")
        if not label_parts:
            continue
        ax.text(
            float(row["x"]) + 0.026 * (extent[1] - extent[0]),
            float(row["y"]) + 0.013 * (extent[3] - extent[2]),
            "\n".join(label_parts),
            fontsize=6.6,
            color="black",
            zorder=35,
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.84},
        )


def _comparison_scales(fields: list[ModelFields], preset) -> tuple[Normalize, TwoSlopeNorm]:
    model_peaks: list[float] = []
    for item in fields:
        combined = np.concatenate(
            [
                np.asarray(item.open_loop, dtype=float).ravel(),
                np.asarray(item.ens_mean, dtype=float).ravel(),
            ]
        )
        finite = combined[np.isfinite(combined)]
        if finite.size:
            model_peaks.append(float(finite.max()))
    model_max = max(model_peaks) if model_peaks else preset.model_min
    model_vmax = nice_ceiling(model_max, step=preset.max_step, minimum=preset.max_floor)
    increment_peaks: list[float] = []
    for item in fields:
        finite = np.abs(np.asarray(item.increment, dtype=float).ravel())
        finite = finite[np.isfinite(finite)]
        if finite.size:
            increment_peaks.append(float(finite.max()))
    increment_abs = max(increment_peaks) if increment_peaks else 0.0
    increment_vmax = nice_ceiling(increment_abs, step=preset.increment_step, minimum=preset.increment_floor)
    return (
        model_map_norm(preset, vmax=model_vmax),
        TwoSlopeNorm(vcenter=0.0, vmin=-increment_vmax, vmax=increment_vmax),
    )


def _resolve_flag(panel_value: bool | None, defaults: MapDefaults, attr_name: str, builtin_default: bool) -> bool:
    recipe_value = getattr(defaults, attr_name)
    if panel_value is not None:
        return bool(panel_value)
    if recipe_value is not None:
        return bool(recipe_value)
    return builtin_default


def _resolve_panel_toggle(value: bool | None, builtin_default: bool) -> bool:
    if value is None:
        return builtin_default
    return bool(value)


def _panel_date(panel: MapPanelSpec, defaults: MapDefaults) -> pd.Timestamp | None:
    raw = panel.date or defaults.date
    if raw is None:
        return None
    return pd.Timestamp(raw).normalize()


def _extract_unit_title(label: str | None) -> str | None:
    if label is None:
        return None
    if "[" in label and "]" in label:
        return label[label.index("[") : label.rindex("]") + 1]
    return label


def _figure_prefers_horizontal_legends(recipe: MapRecipe) -> bool:
    return recipe.layout.ncols >= 4 or any(panel.kind in _CLASSIFIED_PANEL_KINDS for panel in recipe.panels)


def _panel_legend_layout(panel: MapPanelSpec, *, figure_horizontal_default: bool, is_colorbar: bool = False) -> str:
    if panel.legend is not None:
        return panel.legend
    if figure_horizontal_default:
        return "horizontal"
    if panel.kind in _CLASSIFIED_PANEL_KINDS:
        return "horizontal"
    return "vertical" if is_colorbar else "horizontal"


def _row_has_below_panel_artists(
    recipe: MapRecipe,
    *,
    panel_width_in: float,
    figure_horizontal_default: bool,
) -> dict[int, float]:
    row_extras = {row: 0.0 for row in range(recipe.layout.nrows)}
    for panel in recipe.panels:
        row = int(panel.row)
        extra = 0.0
        if panel.kind in {"landcover", "wet_snow"}:
            layout = _panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default)
            if layout == "horizontal":
                labels = (
                    list(LANDCOVER_LABELS.values())
                    if panel.kind == "landcover"
                    else [WET_SNOW_LABELS[code] for code in sorted(WET_SNOW_LABELS)]
                )
                ncols = _legend_columns_for_width(panel_width_in, labels)
                nrows = int(np.ceil(len(labels) / max(ncols, 1)))
                extra = _FIGURE_HORIZONTAL_LEGEND_BASE_EXTRA + _FIGURE_HORIZONTAL_LEGEND_ROW_EXTRA * max(nrows - 1, 0)
        elif panel.kind in _CONTINUOUS_COLORBAR_PANEL_KINDS:
            show_colorbar = _resolve_flag(panel.show_colorbar, recipe.defaults, "show_colorbar", True)
            layout = _panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True)
            if show_colorbar and layout == "horizontal":
                extra = _FIGURE_HORIZONTAL_COLORBAR_EXTRA
        row_extras[row] = max(row_extras.get(row, 0.0), extra)
    return row_extras


def _attach_colorbar(
    ax,
    mappable,
    *,
    label: str | None = None,
    ticks: tuple[float, ...] = (),
    ticklabels: tuple[str, ...] = (),
    layout: str,
) -> None:
    if layout == "horizontal":
        cax = ax.inset_axes([0.06, -0.145, 0.88, 0.052], transform=ax.transAxes)
        cbar = plt.colorbar(mappable, cax=cax, orientation="horizontal")
        if ticks:
            cbar.set_ticks(ticks)
        if ticklabels:
            cbar.set_ticklabels(ticklabels)
        cbar.ax.tick_params(labelsize=_COLORBAR_TICK_SIZE, length=2.0, width=0.65)
        title = _extract_unit_title(label)
        if title:
            cbar.ax.text(1.01, 0.5, title, transform=cbar.ax.transAxes, ha="left", va="center", fontsize=_COLORBAR_TITLE_SIZE)
        elif label:
            cbar.set_label(label, fontsize=_COLORBAR_TITLE_SIZE)
        return

    cax = ax.inset_axes([1.06, -0.045, 0.076, 0.94], transform=ax.transAxes)
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
    title = _extract_unit_title(label)
    if title:
        cbar.ax.text(0.5, 1.02, title, transform=cbar.ax.transAxes, ha="center", va="bottom", fontsize=_COLORBAR_TITLE_SIZE)
    elif label:
        cbar.set_label(label, fontsize=_COLORBAR_TITLE_SIZE)


def _panel_date_text(date: pd.Timestamp | None) -> str | None:
    if date is None:
        return None
    return date.strftime("%Y-%m-%d")


def _draw_panel_date(ax, date: pd.Timestamp | None) -> None:
    text = _panel_date_text(date)
    if not text:
        return
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=_axes_date_fontsize(ax),
        color="black",
        zorder=_ANNOTATION_ZORDER,
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": _DATE_CALLOUT_ALPHA},
    )


def _scale_bar_length_m(extent: tuple[float, float, float, float]) -> float:
    span_m = max(1.0, float(extent[1] - extent[0]))
    max_length = span_m * 0.40
    preferred = np.array([5000.0, 2500.0, 2000.0, 1000.0])
    viable = preferred[preferred <= max_length]
    if viable.size:
        return float(viable[0])
    return float(preferred[-1])


def _format_km_label(length_m: float) -> str:
    value = length_m / 1000.0
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _draw_scale_bar(ax, extent: tuple[float, float, float, float]) -> None:
    span_x = float(extent[1] - extent[0])
    span_y = float(extent[3] - extent[2])
    total_length = _scale_bar_length_m(extent)
    half_length = total_length / 2.0
    x0 = extent[0] + _SCALEBAR_X_FRACTION * span_x
    x0 = min(x0, extent[1] - 0.08 * span_x - total_length)
    x0 = max(x0, extent[0] + 0.08 * span_x)
    y0 = extent[2] + 0.045 * span_y
    tick_height = 0.016 * span_y
    label_y = y0 + 1.15 * tick_height

    box = FancyBboxPatch(
        (x0 - 0.03 * total_length, y0 - 0.9 * tick_height),
        total_length * 1.11,
        tick_height * 3.1,
        boxstyle="round,pad=0.0,rounding_size=0.0",
        facecolor=(1.0, 1.0, 1.0, 0.3),
        edgecolor="none",
        zorder=_ANNOTATION_ZORDER - 1,
    )
    ax.add_patch(box)
    ax.plot([x0, x0 + total_length], [y0, y0], color="black", linewidth=0.8, zorder=_ANNOTATION_ZORDER, solid_capstyle="butt")
    for xpos in (x0, x0 + half_length, x0 + total_length):
        ax.plot([xpos, xpos], [y0, y0 + tick_height], color="black", linewidth=0.8, zorder=_ANNOTATION_ZORDER, solid_capstyle="butt")
    for xpos, label in (
        (x0, "0"),
        (x0 + half_length, _format_km_label(half_length)),
        (x0 + total_length, _format_km_label(total_length)),
    ):
        ax.text(xpos, label_y, label, ha="center", va="bottom", fontsize=5.8, color="black", zorder=_ANNOTATION_ZORDER)
    ax.text(
        x0 + 0.72 * total_length,
        y0 - 0.25 * tick_height,
        "km",
        ha="center",
        va="top",
        fontsize=5.8,
        color="black",
        zorder=_ANNOTATION_ZORDER,
    )


def _figure_size(extent: tuple[float, float, float, float], recipe: MapRecipe) -> tuple[float, float]:
    width = max(1.0, float(extent[1] - extent[0]))
    height = max(1.0, float(extent[3] - extent[2]))
    aspect = height / width
    width_factors = recipe.layout.width_ratios or tuple(1.0 for _ in range(recipe.layout.ncols))
    height_factors = recipe.layout.height_ratios or tuple(1.0 for _ in range(recipe.layout.nrows))
    figure_horizontal_default = _figure_prefers_horizontal_legends(recipe)
    count_vertical_colorbars = sum(
        1
        for panel in recipe.panels
        if panel.kind in _CONTINUOUS_COLORBAR_PANEL_KINDS
        and _panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True) == "vertical"
    )
    width_units = float(sum(width_factors)) + _LAYOUT_COL_GAP * max(recipe.layout.ncols - 1, 0) + 0.22 * count_vertical_colorbars
    panel_width = FIGWIDTH_OVERVIEW_PAPER / max(width_units, 1.0)
    row_bottom_extras = _row_has_below_panel_artists(
        recipe,
        panel_width_in=panel_width,
        figure_horizontal_default=figure_horizontal_default,
    )
    panel_height = panel_width * aspect * 1.02
    fig_height = (
        panel_height * float(sum(height_factors))
        + panel_height * _LAYOUT_ROW_GAP * max(recipe.layout.nrows - 1, 0)
        + float(sum(row_bottom_extras.values()))
    )
    return FIGWIDTH_OVERVIEW_PAPER, float(np.clip(fig_height, _FIGURE_HEIGHT_MIN, _FIGURE_HEIGHT_MAX))


def _draw_classified_legend(ax, handles: list[Patch], *, layout: str) -> None:
    if not handles:
        return
    if layout == "horizontal":
        labels = [handle.get_label() for handle in handles]
        ncols = _legend_columns_for_axis(ax, labels)
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.0, -0.14),
            ncol=ncols,
            frameon=False,
            fontsize=5.7,
            handlelength=1.3,
            handletextpad=0.45,
            columnspacing=0.9,
            borderaxespad=0.0,
        )
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


def _draw_panel_extras(ax, *, panel: MapPanelSpec, defaults: MapDefaults, extent: tuple[float, float, float, float], date: pd.Timestamp | None) -> None:
    _draw_panel_date(ax, date)
    if _resolve_flag(panel.show_scalebar, defaults, "show_scalebar", False):
        _draw_scale_bar(ax, extent)


def _apply_common_overlays(
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
        _draw_roi(ax, context)
    if show_station_marker:
        _draw_stations_overlay(
            ax,
            context,
            extent,
            show_station_marker=show_station_marker,
            show_stations_name=show_stations_name,
            show_stations_elev=show_stations_elev,
        )


def _overview_extent(ax, context: StaticContext, *, scale: int) -> tuple[float, float, float, float]:
    fig = ax.figure
    bbox = ax.get_position()
    width_in = fig.get_size_inches()[0] * bbox.width
    height_in = fig.get_size_inches()[1] * bbox.height
    width_m = width_in * 0.0254 * float(scale)
    height_m = height_in * 0.0254 * float(scale)
    local_extent = buffered_extent(context)
    target_aspect = (local_extent[3] - local_extent[2]) / max(local_extent[1] - local_extent[0], 1.0)
    current_aspect = height_m / max(width_m, 1.0)
    if current_aspect < target_aspect:
        height_m = width_m * target_aspect
    else:
        width_m = height_m / max(target_aspect, 1e-9)
    centroid = context.roi_gdf.geometry.unary_union.centroid
    center_x = float(centroid.x)
    center_y = float(centroid.y)
    return (
        center_x - 0.5 * width_m,
        center_x + 0.5 * width_m,
        center_y - 0.5 * height_m,
        center_y + 0.5 * height_m,
    )


def _render_overview_panel(ax, *, panel: MapPanelSpec, context: StaticContext, label: str | None, defaults: MapDefaults) -> dict[str, object]:
    extent = _overview_extent(ax, context, scale=int(panel.scale or 1))
    countries = load_overview_boundaries()
    target_window = box(extent[0], extent[2], extent[1], extent[3])
    target_window_gdf = context.roi_gdf.iloc[:1].copy()
    target_window_gdf.geometry = [target_window]
    source_window_gdf = target_window_gdf.to_crs(countries.crs)
    window_geom = source_window_gdf.geometry.iloc[0]
    window_bounds = source_window_gdf.total_bounds
    subset = countries.cx[window_bounds[0] : window_bounds[2], window_bounds[1] : window_bounds[3]].copy()
    if subset.empty:
        subset = countries.copy()
    subset = subset[subset.geom_type.isin({"LineString", "MultiLineString", "Polygon", "MultiPolygon"})].copy()
    subset = subset.loc[subset.intersects(window_geom)].copy()
    if subset.crs is not None and context.spec.crs is not None and str(subset.crs) != str(context.spec.crs):
        subset = subset.to_crs(context.spec.crs)
    subset = subset[subset.geometry.notna()].copy()
    subset = subset.loc[~subset.geometry.is_empty].copy()
    if not subset.empty:
        subset = subset.explode(ignore_index=True)
        clip_geom = box(extent[0], extent[2], extent[1], extent[3])
        subset = subset.loc[subset.intersects(clip_geom)].copy()
        if not subset.empty:
            subset = subset.clip(clip_geom)
            bounds = subset.geometry.bounds
            fragment_dim = np.maximum(bounds.maxx - bounds.minx, bounds.maxy - bounds.miny)
            threshold = _OVERVIEW_FRAGMENT_RATIO * max(extent[1] - extent[0], extent[3] - extent[2])
            subset = subset.loc[fragment_dim > threshold].copy()
    if not subset.empty:
        subset.plot(ax=ax, color="black", linewidth=0.65, zorder=8)
    context.roi_gdf.plot(ax=ax, facecolor=_OVERVIEW_ROI_COLOR, edgecolor=_OVERVIEW_ROI_COLOR, linewidth=0.8, zorder=25)
    if panel.roi_label:
        centroid = context.roi_gdf.geometry.unary_union.centroid
        ax.text(
            float(centroid.x) + 0.02 * (extent[1] - extent[0]),
            float(centroid.y),
            panel.roi_label,
            color="black",
            fontsize=6.4,
            ha="left",
            va="center",
            zorder=_ANNOTATION_ZORDER,
        )
    show_grid = _resolve_flag(panel.show_grid, defaults, "show_grid", True)
    _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
    _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=_panel_date(panel, defaults))
    _draw_map_grid_overlay(ax, show_grid=show_grid)
    return {}


def _render_roi_panel(ax, *, panel: MapPanelSpec, context: StaticContext, extent, label: str | None, defaults: MapDefaults) -> dict[str, object]:
    show_grid = _resolve_flag(panel.show_grid, defaults, "show_grid", True)
    context.roi_gdf.plot(ax=ax, color=_ROI_FILL, edgecolor="none", zorder=0)
    _apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=_resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=_resolve_panel_toggle(panel.show_station_marker, True),
        show_stations_name=_resolve_panel_toggle(panel.show_stations_name, True),
        show_stations_elev=_resolve_panel_toggle(panel.show_stations_elev, True),
    )
    _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
    _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=_panel_date(panel, defaults))
    _draw_map_grid_overlay(ax, show_grid=show_grid)
    return {}


def _render_static_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    grid_extent,
    label: str | None,
    defaults: MapDefaults,
    figure_horizontal_default: bool,
) -> dict[str, object]:
    show_grid = _resolve_flag(panel.show_grid, defaults, "show_grid", True)
    show_roi = _resolve_panel_toggle(panel.show_roi, True)
    show_station_marker = _resolve_panel_toggle(panel.show_station_marker, False)
    show_stations_name = _resolve_panel_toggle(panel.show_stations_name, False)
    show_stations_elev = _resolve_panel_toggle(panel.show_stations_elev, False)

    if panel.kind == "hillshade":
        ax.imshow(_hillshade(context), cmap="Greys", extent=grid_extent, origin="upper", vmin=0.0, vmax=1.0, zorder=5)
        _apply_common_overlays(
            ax,
            context=context,
            extent=extent,
            show_roi=show_roi,
            show_station_marker=show_station_marker,
            show_stations_name=show_stations_name,
            show_stations_elev=show_stations_elev,
        )
        _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
        _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=_panel_date(panel, defaults))
        _draw_map_grid_overlay(ax, show_grid=show_grid)
        return {}

    if panel.kind == "landcover":
        masked_landcover = _masked(_field_array(context, "landcover"), context.roi_mask)
        present_codes = sorted({int(value) for value in masked_landcover.compressed() if np.isfinite(value)})
        if not present_codes:
            present_codes = [0]
        code_to_index = {code: idx for idx, code in enumerate(present_codes)}
        categorical = np.full(masked_landcover.shape, np.nan, dtype=float)
        filled = masked_landcover.filled(np.nan)
        for code, idx in code_to_index.items():
            categorical[np.isclose(filled, float(code), equal_nan=False)] = idx
        cmap = landcover_cmap_for_codes(present_codes)
        norm = BoundaryNorm(np.arange(-0.5, len(present_codes) + 0.5), cmap.N)
        image = ax.imshow(categorical, cmap=cmap, norm=norm, extent=grid_extent, origin="upper", interpolation="nearest", zorder=5)
        _apply_common_overlays(
            ax,
            context=context,
            extent=extent,
            show_roi=show_roi,
            show_station_marker=show_station_marker,
            show_stations_name=show_stations_name,
            show_stations_elev=show_stations_elev,
        )
        _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
        legend_handles = [
            Patch(facecolor=cmap(code_to_index[code]), edgecolor="none", label=LANDCOVER_LABELS.get(code, str(code)))
            for code in present_codes
        ]
        _draw_classified_legend(
            ax,
            legend_handles,
            layout=_panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default),
        )
        _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=_panel_date(panel, defaults))
        _draw_map_grid_overlay(ax, show_grid=show_grid)
        return {"mappable": image, "legend_handles": legend_handles}

    field = _STATIC_FIELD_KIND_TO_FIELD[panel.kind]
    preset = require_static_field_preset(field)
    data = _masked(_field_array(context, field), context.roi_mask)
    norm = static_field_norm(preset, data.filled(np.nan))
    image = ax.imshow(data, cmap=static_field_cmap(preset), norm=norm, extent=grid_extent, origin="upper", interpolation="nearest", zorder=5)
    _apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=show_roi,
        show_station_marker=show_station_marker,
        show_stations_name=show_stations_name,
        show_stations_elev=show_stations_elev,
    )
    _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
    colorbar_style = static_field_colorbar_style(preset)
    if _resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        _attach_colorbar(
            ax,
            image,
            label=colorbar_style.label,
            ticks=colorbar_style.ticks,
            ticklabels=colorbar_style.ticklabels,
            layout=_panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
        )
    _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=_panel_date(panel, defaults))
    _draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "colorbar_style": colorbar_style}


def _render_model_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    grid_extent,
    label: str | None,
    defaults: MapDefaults,
    model_cache,
    scale_cache,
    figure_horizontal_default: bool,
) -> dict[str, object]:
    date = _panel_date(panel, defaults)
    if date is None:
        raise ValueError(f"Panel '{panel.kind}' requires a date (panel '{panel.title or panel.kind}')")
    variable = _MODEL_KIND_TO_VARIABLE[panel.kind]
    field_key = (variable, date)
    if field_key not in model_cache:
        model_cache[field_key] = load_model_fields(context.project_dir, variable, (date,))[0]
    preset = require_variable_preset(variable)
    if field_key not in scale_cache:
        scale_cache[field_key] = _comparison_scales([model_cache[field_key]], preset)
    model_norm, increment_norm = scale_cache[field_key]
    if _resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        ax.imshow(_hillshade(context), cmap="Greys", extent=grid_extent, origin="upper", vmin=0.0, vmax=1.0, zorder=0)

    colorbar_style: dict[str, object] | object
    if panel.source == "increment":
        image = ax.imshow(
            _masked(model_cache[field_key].increment, context.roi_mask),
            cmap=INCREMENT_CMAP,
            norm=increment_norm,
            extent=grid_extent,
            origin="upper",
            interpolation="nearest",
            alpha=0.95,
            zorder=5,
        )
        colorbar_style = {"label": f"increment {preset.unit_label}"}
    else:
        data = model_cache[field_key].open_loop if panel.source == "open_loop" else model_cache[field_key].ens_mean
        image = ax.imshow(
            _masked_model(data, context.roi_mask, preset=preset),
            cmap=model_map_cmap(preset),
            norm=model_norm,
            extent=grid_extent,
            origin="upper",
            interpolation="nearest",
            alpha=0.96,
            zorder=5,
        )
        colorbar_style = model_colorbar_style(preset)

    _apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=_resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=_resolve_panel_toggle(panel.show_station_marker, False),
        show_stations_name=_resolve_panel_toggle(panel.show_stations_name, False),
        show_stations_elev=_resolve_panel_toggle(panel.show_stations_elev, False),
    )
    show_grid = _resolve_flag(panel.show_grid, defaults, "show_grid", True)
    _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
    if _resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
        label_text = colorbar_style["label"] if isinstance(colorbar_style, dict) else colorbar_style.label
        ticks = colorbar_style.get("ticks", ()) if isinstance(colorbar_style, dict) else colorbar_style.ticks
        ticklabels = colorbar_style.get("ticklabels", ()) if isinstance(colorbar_style, dict) else colorbar_style.ticklabels
        _attach_colorbar(
            ax,
            image,
            label=label_text,
            ticks=ticks,
            ticklabels=ticklabels,
            layout=_panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
        )
    _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date)
    _draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "colorbar_style": colorbar_style}


def _render_observation_panel(
    ax,
    *,
    panel: MapPanelSpec,
    context: StaticContext,
    extent,
    label: str | None,
    defaults: MapDefaults,
    obs_cache,
    figure_horizontal_default: bool,
) -> dict[str, object]:
    date = _panel_date(panel, defaults)
    if date is None:
        raise ValueError(f"Panel '{panel.kind}' requires a date (panel '{panel.title or panel.kind}')")
    observation = _OBSERVATION_KIND_TO_NAME[panel.kind]
    obs_key = (observation, date)
    if obs_key not in obs_cache:
        obs_cache[obs_key] = load_observation_scene(context.project_dir, context, observation=observation, date=date)
    scene = obs_cache[obs_key]
    show_grid = _resolve_flag(panel.show_grid, defaults, "show_grid", True)
    if _resolve_flag(panel.show_hillshade, defaults, "show_hillshade", False):
        ax.imshow(_hillshade(context), cmap="Greys", extent=_grid_extent(context), origin="upper", vmin=0.0, vmax=1.0, zorder=0)

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
        code_to_index = {code: idx for idx, code in enumerate(codes)}
        categorical = np.full(scene.array.shape, np.nan, dtype=float)
        for code, idx in code_to_index.items():
            categorical[np.isclose(scene.array, float(code), equal_nan=False)] = idx
        cmap = matplotlib.colors.ListedColormap([WET_SNOW_COLORS[code] for code in codes], name="wet_snow_obs")
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))
        norm = BoundaryNorm(np.arange(-0.5, len(codes) + 0.5), cmap.N)
        image = ax.imshow(np.ma.masked_invalid(categorical), cmap=cmap, norm=norm, extent=scene.bounds, origin="upper", interpolation="nearest", zorder=5)
        legend_handles = [Patch(facecolor=WET_SNOW_COLORS[code], edgecolor="none", label=WET_SNOW_LABELS[code]) for code in codes]

    _apply_common_overlays(
        ax,
        context=context,
        extent=extent,
        show_roi=_resolve_panel_toggle(panel.show_roi, True),
        show_station_marker=_resolve_panel_toggle(panel.show_station_marker, False),
        show_stations_name=_resolve_panel_toggle(panel.show_stations_name, False),
        show_stations_elev=_resolve_panel_toggle(panel.show_stations_elev, False),
    )
    _apply_map_axis_style(ax, extent, title=_panel_title(label, _panel_semantic_title(panel)), show_grid=show_grid)
    if observation == "scf":
        if _resolve_flag(panel.show_colorbar, defaults, "show_colorbar", True):
            _attach_colorbar(
                ax,
                image,
                label="fractional snow cover [%]",
                ticks=(0, 20, 40, 60, 80, 100),
                layout=_panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default, is_colorbar=True),
            )
    else:
        _draw_classified_legend(
            ax,
            legend_handles,
            layout=_panel_legend_layout(panel, figure_horizontal_default=figure_horizontal_default),
        )
    _draw_panel_extras(ax, panel=panel, defaults=defaults, extent=extent, date=date)
    _draw_map_grid_overlay(ax, show_grid=show_grid)
    return {"mappable": image, "legend_handles": legend_handles}


def _render_text_panel(ax, *, panel: MapPanelSpec) -> dict[str, object]:
    ax.set_axis_off()
    if panel.title:
        ax.set_title(panel.title, loc="left", fontsize=_axes_title_fontsize(ax), pad=2.8)
    y = 0.95
    for line in panel.lines:
        ax.text(0.0, y, line, transform=ax.transAxes, ha="left", va="top", fontsize=7.2)
        y -= 0.09
    return {}


def _render_colorbar_panel(ax, *, panel: MapPanelSpec, artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
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
    title = _extract_unit_title(label)
    if title:
        cbar.ax.text(0.5, 1.02, title, transform=cbar.ax.transAxes, ha="center", va="top", fontsize=_COLORBAR_TITLE_SIZE)
    return {}


def _draw_patch_entry(ax, *, y: float, label: str, facecolor, edgecolor="none") -> float:
    rect = Rectangle((0.02, y - 0.028), 0.12, 0.05, transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(0.18, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.1)
    return y - 0.061


def _draw_station_entry(ax, *, y: float, label: str) -> float:
    ax.scatter([0.08], [y], s=110, marker="v", facecolor=_STATION_COLOR, edgecolor="none", transform=ax.transAxes, clip_on=False)
    ax.text(0.18, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.4)
    return y - 0.068


def _draw_heading(ax, *, y: float, text: str) -> float:
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.8)
    return y - 0.05


def _legend_source_handles(item: LegendItemSpec, artifacts: dict[str, dict[str, object]]) -> list[Patch]:
    source = artifacts.get(str(item.source or ""))
    if source is None:
        raise ValueError(f"Legend source '{item.source}' is not available")
    handles = source.get("legend_handles")
    if not handles:
        raise ValueError(f"Legend source '{item.source}' has no legend handles")
    return list(handles)


def _render_legend_panel(ax, *, panel: MapPanelSpec, artifacts: dict[str, dict[str, object]]) -> dict[str, object]:
    ax.set_axis_off()
    items = panel.items
    if not items and panel.source is not None:
        items = (LegendItemSpec(kind="source_legend", source=panel.source),)

    y = 0.97
    for item in items:
        if item.kind == "heading":
            y = _draw_heading(ax, y=y, text=str(item.label))
        elif item.kind == "station_symbol":
            y = _draw_station_entry(ax, y=y, label=str(item.label))
        elif item.kind == "source_legend":
            if item.label:
                y = _draw_heading(ax, y=y, text=str(item.label))
            for handle in _legend_source_handles(item, artifacts):
                y = _draw_patch_entry(
                    ax,
                    y=y,
                    label=handle.get_label(),
                    facecolor=handle.get_facecolor(),
                    edgecolor=handle.get_edgecolor(),
                )
        elif item.kind == "scale_bar":
            continue
        else:
            raise ValueError(f"Unsupported legend item kind '{item.kind}'")
    return {}


def render_map_recipe(*, project_dir: Path, context: StaticContext, recipe: MapRecipe, output_path: Path) -> Path:
    del project_dir
    extent = buffered_extent(context)
    grid_extent = _grid_extent(context)
    figure_horizontal_default = _figure_prefers_horizontal_legends(recipe)
    fig = plt.figure(figsize=_figure_size(extent, recipe))
    gs = fig.add_gridspec(
        recipe.layout.nrows,
        recipe.layout.ncols,
        width_ratios=recipe.layout.width_ratios or None,
        height_ratios=recipe.layout.height_ratios or None,
        wspace=_LAYOUT_COL_GAP,
        hspace=_LAYOUT_ROW_GAP,
    )
    fig.subplots_adjust(left=_LEFT_MARGIN, right=_RIGHT_MARGIN, bottom=_BOTTOM_MARGIN, top=_TOP_MARGIN)

    title_letters = iter(ascii_lowercase)
    panel_labels: list[str | None] = []
    for panel in recipe.panels:
        if panel.kind in _SUPPORT_PANEL_KINDS:
            panel_labels.append(None)
        else:
            panel_labels.append(next(title_letters))

    model_cache: dict[tuple[str, pd.Timestamp], ModelFields] = {}
    scale_cache: dict[tuple[str, pd.Timestamp], tuple[Normalize, TwoSlopeNorm]] = {}
    obs_cache: dict[tuple[str, pd.Timestamp], ObservationScene] = {}
    artifacts: dict[str, dict[str, object]] = {}
    axes: list = []

    for idx, panel in enumerate(recipe.panels):
        ax = fig.add_subplot(gs[panel.row : panel.row + panel.rowspan, panel.col : panel.col + panel.colspan])
        axes.append(ax)
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
                grid_extent=grid_extent,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                figure_horizontal_default=figure_horizontal_default,
            )
        elif panel.kind in _MODEL_KIND_TO_VARIABLE:
            artifacts[key] = _render_model_panel(
                ax,
                panel=panel,
                context=context,
                extent=extent,
                grid_extent=grid_extent,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                model_cache=model_cache,
                scale_cache=scale_cache,
                figure_horizontal_default=figure_horizontal_default,
            )
        elif panel.kind in _OBSERVATION_KIND_TO_NAME:
            artifacts[key] = _render_observation_panel(
                ax,
                panel=panel,
                context=context,
                extent=extent,
                label=panel_labels[idx],
                defaults=recipe.defaults,
                obs_cache=obs_cache,
                figure_horizontal_default=figure_horizontal_default,
            )
        elif panel.kind == "colorbar":
            artifacts[key] = _render_colorbar_panel(ax, panel=panel, artifacts=artifacts)
        elif panel.kind == "legend":
            artifacts[key] = _render_legend_panel(ax, panel=panel, artifacts=artifacts)
        else:
            raise ValueError(f"Unsupported panel kind '{panel.kind}'")

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
    "buffered_extent",
    "figure_height_for_extent",
    "render_map_recipe",
]
