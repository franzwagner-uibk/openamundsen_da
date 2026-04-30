from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from openamundsen_da.methods.viz.maps.config import LegendItemSpec, MapDefaults, MapPanelSpec
from openamundsen_da.methods.viz.maps.layout import (
    axes_date_fontsize,
    register_child_axes,
    text_size_in,
)
from openamundsen_da.methods.viz.maps.theme import (
    _ANNOTATION_ZORDER,
    _AUTO_TITLE_KIND,
    _AUTO_TITLE_SOURCE,
    _DATE_CALLOUT_ALPHA,
    _OVERLAY_LABEL_BBOX_HALO_WIDTH,
    _OVERLAY_LABEL_HALO_COLOR,
    _OVERLAY_LABEL_HALO_WIDTH,
    _OVERVIEW_LABEL_BOX_PAD_EM,
    _OVERVIEW_LABEL_BOX_SAFETY_IN,
    _PANEL_BELOW_ITEMS_BOTTOM_PAD_AXES,
    _PANEL_BELOW_ITEMS_DRAW_GAP_BASE_AXES,
    _PANEL_BELOW_ITEMS_DRAW_GAP_PER_HEIGHT_AXES,
    _PANEL_BELOW_ITEMS_GAP_AXES,
    _PANEL_BELOW_ITEMS_ROW_HEIGHT_AXES,
    _SCALEBAR_BOTTOM_FRACTION,
    _SCALEBAR_RIGHT_PAD_FRACTION,
    _SCALEBAR_TARGET_FRACTION,
    _STATION_COLOR,
)


def panel_semantic_title(panel: MapPanelSpec) -> str | None:
    if panel.title is not None:
        return panel.title
    if panel.kind in {"snow_depth", "swe", "liquid_water_content"}:
        source = _AUTO_TITLE_SOURCE.get(str(panel.source or "").strip())
        target = _AUTO_TITLE_KIND.get(panel.kind)
        if source and target:
            return f"{source} {target}"
    return _AUTO_TITLE_KIND.get(panel.kind)


def panel_title(letter: str | None, title: str | None) -> str | None:
    if letter is None and title is None:
        return None
    if letter is None:
        return title
    if title is None:
        return f"({letter})"
    return f"({letter}) {title}"


def panel_date(panel: MapPanelSpec, defaults: MapDefaults) -> pd.Timestamp | None:
    raw = panel.date or defaults.date
    if raw is None:
        return None
    return pd.Timestamp(raw).normalize()


def panel_date_text(date: pd.Timestamp | None) -> str | None:
    if date is None:
        return None
    return date.strftime("%Y-%m-%d")


def draw_panel_date(ax, date: pd.Timestamp | None) -> None:
    text = panel_date_text(date)
    if not text:
        return
    ax.text(
        0.02,
        0.955,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=axes_date_fontsize(ax),
        color="black",
        zorder=_ANNOTATION_ZORDER,
        bbox={"boxstyle": "round,pad=0.10", "facecolor": "white", "edgecolor": "none", "alpha": _DATE_CALLOUT_ALPHA},
    )


def apply_overlay_label_halo(text, *, with_bbox: bool = False):
    halo_width = _OVERLAY_LABEL_BBOX_HALO_WIDTH if with_bbox else _OVERLAY_LABEL_HALO_WIDTH
    text.set_path_effects([pe.Stroke(linewidth=halo_width, foreground=_OVERLAY_LABEL_HALO_COLOR), pe.Normal()])
    return text


def scale_bar_length_m(extent: tuple[float, float, float, float]) -> float:
    span_m = max(1.0, float(extent[1] - extent[0]))
    target_length = span_m * _SCALEBAR_TARGET_FRACTION
    preferred = np.array([500.0, 1000.0, 2000.0, 2500.0, 5000.0, 10000.0, 20000.0, 25000.0, 50000.0, 100000.0])
    viable = preferred[preferred >= target_length]
    if viable.size:
        return float(viable[0])
    return float(preferred[-1])


def format_km_label(length_m: float) -> str:
    value = length_m / 1000.0
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def draw_scale_bar(ax, extent: tuple[float, float, float, float]) -> None:
    span_x = float(extent[1] - extent[0])
    span_y = float(extent[3] - extent[2])
    total_length = scale_bar_length_m(extent)
    half_length = total_length / 2.0
    x0 = extent[1] - _SCALEBAR_RIGHT_PAD_FRACTION * span_x - total_length
    x0 = max(x0, extent[0] + 0.08 * span_x)
    y0 = extent[2] + _SCALEBAR_BOTTOM_FRACTION * span_y
    tick_height = 0.016 * span_y
    label_y = y0 + 1.15 * tick_height
    line_halo = [pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()]
    bar = ax.plot([x0, x0 + total_length], [y0, y0], color="black", linewidth=0.8, zorder=_ANNOTATION_ZORDER, solid_capstyle="butt")[0]
    bar.set_path_effects(line_halo)
    for xpos in (x0, x0 + half_length, x0 + total_length):
        tick = ax.plot([xpos, xpos], [y0, y0 + tick_height], color="black", linewidth=0.8, zorder=_ANNOTATION_ZORDER, solid_capstyle="butt")[0]
        tick.set_path_effects(line_halo)
    for xpos, label in (
        (x0, "0"),
        (x0 + half_length, format_km_label(half_length)),
        (x0 + total_length, format_km_label(total_length)),
    ):
        apply_overlay_label_halo(ax.text(
            xpos,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="black",
            zorder=_ANNOTATION_ZORDER,
        ))
    apply_overlay_label_halo(ax.text(
        x0 + 0.72 * total_length,
        y0 - 0.45 * tick_height,
        "km",
        ha="center",
        va="top",
        fontsize=5.8,
        color="black",
        zorder=_ANNOTATION_ZORDER,
    ))


def draw_patch_entry(ax, *, y: float, label: str, facecolor, edgecolor="none") -> float:
    rect = Rectangle((0.02, y - 0.028), 0.12, 0.05, transform=ax.transAxes, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.0)
    ax.add_patch(rect)
    ax.text(0.18, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.1)
    return y - 0.061


def draw_station_entry(ax, *, y: float, label: str) -> float:
    ax.scatter([0.070], [y], s=40, marker="v", facecolor=_STATION_COLOR, edgecolor="none", transform=ax.transAxes, clip_on=False)
    ax.text(0.148, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.1)
    return y - 0.054


def draw_heading(ax, *, y: float, text: str) -> float:
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=7.8)
    return y - 0.05


def panel_below_item_units(item: LegendItemSpec) -> float:
    if item.kind == "heading":
        return 0.65
    if item.kind == "station_symbol":
        return 0.40
    return 1.0


def panel_below_items_layout(items: tuple[LegendItemSpec, ...]) -> tuple[float, float, float, float]:
    row_units = sum(panel_below_item_units(item) for item in items)
    inset_height = max(row_units * _PANEL_BELOW_ITEMS_ROW_HEIGHT_AXES, 1e-9)
    reserve_extra = _PANEL_BELOW_ITEMS_GAP_AXES + inset_height + _PANEL_BELOW_ITEMS_BOTTOM_PAD_AXES
    draw_gap = _PANEL_BELOW_ITEMS_DRAW_GAP_BASE_AXES + _PANEL_BELOW_ITEMS_DRAW_GAP_PER_HEIGHT_AXES * inset_height
    return row_units, inset_height, reserve_extra, draw_gap


def panel_below_items_extra(items: tuple[LegendItemSpec, ...]) -> float:
    if not items:
        return 0.0
    _, _, reserve_extra, _ = panel_below_items_layout(items)
    return reserve_extra


def overview_label_box_size_in(spec) -> tuple[float, float]:
    width_in, height_in = text_size_in(spec.text, size=spec.fontsize)
    if spec.with_bbox:
        pad_in = _OVERVIEW_LABEL_BOX_PAD_EM * spec.fontsize / 72.0
        width_in += 2.0 * pad_in
        height_in += 2.0 * pad_in
    width_in += 2.0 * _OVERVIEW_LABEL_BOX_SAFETY_IN
    height_in += 2.0 * _OVERVIEW_LABEL_BOX_SAFETY_IN
    return width_in, height_in


def draw_overview_label_specs(ax, specs: list) -> None:
    for spec in specs:
        apply_overlay_label_halo(ax.text(
            spec.x,
            spec.y,
            spec.text,
            ha=spec.ha,
            va=spec.va,
            fontsize=spec.fontsize,
            color="black",
            zorder=spec.zorder,
        ), with_bbox=spec.with_bbox)


def draw_panel_below_items(
    ax,
    *,
    panel: MapPanelSpec,
    artifacts: dict[str, dict[str, object]],
    legend_source_handles_getter,
) -> None:
    if not panel.below_items:
        return
    row_units, inset_height, _, draw_gap = panel_below_items_layout(panel.below_items)
    legend_ax = ax.inset_axes(
        [0.0, -(draw_gap + inset_height), 1.0, inset_height],
        transform=ax.transAxes,
    )
    register_child_axes(ax, legend_ax)
    legend_ax.set_axis_off()
    y = 1.0 - (0.4 / max(row_units, 1e-9))
    for item in panel.below_items:
        if item.kind == "heading":
            y = draw_heading(legend_ax, y=y, text=str(item.label))
        elif item.kind == "station_symbol":
            y = draw_station_entry(legend_ax, y=y, label=str(item.label))
        elif item.kind == "source_legend":
            if item.label:
                y = draw_heading(legend_ax, y=y, text=str(item.label))
            for handle in legend_source_handles_getter(item, artifacts):
                y = draw_patch_entry(
                    legend_ax,
                    y=y,
                    label=handle.get_label(),
                    facecolor=handle.get_facecolor(),
                    edgecolor=handle.get_edgecolor(),
                )
        elif item.kind == "scale_bar":
            continue
        else:
            raise ValueError(f"Unsupported below-panel legend item kind '{item.kind}'")


def draw_panel_extras(
    ax,
    *,
    panel: MapPanelSpec,
    defaults: MapDefaults,
    extent: tuple[float, float, float, float],
    date: pd.Timestamp | None,
    resolve_flag,
) -> None:
    draw_panel_date(ax, date)
    if resolve_flag(panel.show_scalebar, defaults, "show_scalebar", False):
        draw_scale_bar(ax, extent)
