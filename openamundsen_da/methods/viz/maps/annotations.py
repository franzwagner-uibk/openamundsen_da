from __future__ import annotations

import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from openamundsen_da.methods.viz.maps.config import LegendItemSpec, MapDefaults, MapPanelSpec
from openamundsen_da.methods.viz.maps.layout import (
    axes_date_fontsize,
    axis_height_inches,
    axis_width_inches,
    horizontal_annotation_gap_axes,
    register_child_axes,
    text_size_in,
)
from openamundsen_da.methods.viz.maps.station_markers import (
    FORCING_STATION_COLOR,
    HOLDOUT_STATION_COLOR,
    HOLDOUT_STATION_LINEWIDTH,
    HOLDOUT_STATION_MARKER,
    LEFT_HALF_TRIANGLE,
    RIGHT_HALF_TRIANGLE,
    SNOW_STATION_COLOR,
    STATION_MARKER_SIZE,
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
    _PANEL_BELOW_ITEMS_MIN_GAP_IN,
    _PANEL_BELOW_ITEMS_ROW_HEIGHT_AXES,
    _SCALEBAR_BOTTOM_FRACTION,
    _SCALEBAR_RIGHT_PAD_FRACTION,
    _SCALEBAR_TARGET_FRACTION,
    _STATION_COLOR,
)

_INSIDE_LEGEND_ANCHOR_PAD = 0.026
_INSIDE_LEGEND_MIN_WIDTH = 0.26
_INSIDE_LEGEND_MIN_HEIGHT = 0.075
_INSIDE_LEGEND_MAX_HEIGHT = 0.36
_INSIDE_LEGEND_UNIT_HEIGHT = 0.095
_INSIDE_LEGEND_FACE = (1.0, 1.0, 1.0, 0.70)
_INSIDE_LEGEND_RIGHT_PAD = 0.020
_INSIDE_LEGEND_TEXT_WIDTH_SAFETY = 1.03
_LEGEND_ENTRY_FONT_SIZE = 6.1
_LEGEND_HEADING_FONT_SIZE = 7.8
_STATION_LEGEND_FONT_SIZE = 5.8
_STATION_LEGEND_MARKER_SIZE = 26
_SCALEBAR_FONT_SIZE = 5.8
_STATION_MARKER_X = 0.095
_STATION_LABEL_X = 0.225
_PATCH_LABEL_X = 0.18
_STYLE_SCALE = 1.0


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
    line_halo = [pe.Stroke(linewidth=2.2 * _STYLE_SCALE, foreground="white"), pe.Normal()]
    bar = ax.plot(
        [x0, x0 + total_length],
        [y0, y0],
        color="black",
        linewidth=0.8 * _STYLE_SCALE,
        zorder=_ANNOTATION_ZORDER,
        solid_capstyle="butt",
    )[0]
    bar.set_path_effects(line_halo)
    for xpos in (x0, x0 + half_length, x0 + total_length):
        tick = ax.plot(
            [xpos, xpos],
            [y0, y0 + tick_height],
            color="black",
            linewidth=0.8 * _STYLE_SCALE,
            zorder=_ANNOTATION_ZORDER,
            solid_capstyle="butt",
        )[0]
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
            fontsize=_SCALEBAR_FONT_SIZE,
            color="black",
            zorder=_ANNOTATION_ZORDER,
        ))
    apply_overlay_label_halo(ax.text(
        x0 + 0.50 * total_length,
        y0 - 0.95 * tick_height,
        "km",
        ha="center",
        va="top",
        fontsize=_SCALEBAR_FONT_SIZE,
        color="black",
        zorder=_ANNOTATION_ZORDER,
    ))


def draw_patch_entry(ax, *, y: float, label: str, facecolor, edgecolor="none") -> float:
    rect = Rectangle(
        (0.02, y - 0.028),
        0.12,
        0.05,
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.0 * _STYLE_SCALE,
    )
    ax.add_patch(rect)
    ax.text(_PATCH_LABEL_X, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=_LEGEND_ENTRY_FONT_SIZE)
    return y - 0.061


def draw_station_entry(ax, *, y: float, label: str) -> float:
    ax.scatter([_STATION_MARKER_X], [y], s=_STATION_LEGEND_MARKER_SIZE, marker="^", facecolor=_STATION_COLOR, edgecolor="none", linewidth=0.0, transform=ax.transAxes, clip_on=False)
    ax.text(_STATION_LABEL_X, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=_STATION_LEGEND_FONT_SIZE)
    return y - 0.054


def _draw_station_category_entry(ax, *, y: float, kind: str, label: str) -> float:
    _draw_station_category_entry_at(
        ax,
        y=y,
        kind=kind,
        label=label,
        marker_x=_STATION_MARKER_X,
        label_x=_STATION_LABEL_X,
    )
    return y - 0.23


def draw_station_categories(ax, *, y: float) -> float:
    """Draw the fixed station-source and role legend."""
    entries = (
        ("forcing", "Forcing station"),
        ("snow", "Snow observation station"),
        ("both", "Forcing + snow station"),
        ("holdout", "Holdout snow station"),
    )
    for kind, label in entries:
        y = _draw_station_category_entry(ax, y=y, kind=kind, label=label)
    return y


def draw_station_categories_below(ax, *, y: float) -> float:
    """Draw the station categories as a compact two-column below-panel key."""
    rows = (
        (("forcing", "Forcing station"), ("snow", "Snow obs. station")),
        (("both", "Forcing + snow station"), ("holdout", "Holdout snow station")),
    )
    for row, entries in enumerate(rows):
        row_y = y - 0.55 * row
        for column, (kind, label) in enumerate(entries):
            marker_x = 0.055 + 0.50 * column
            label_x = 0.105 + 0.50 * column
            _draw_station_category_entry_at(
                ax,
                y=row_y,
                kind=kind,
                label=label,
                marker_x=marker_x,
                label_x=label_x,
            )
    return y - 1.10


def _draw_station_category_entry_at(
    ax,
    *,
    y: float,
    kind: str,
    label: str,
    marker_x: float,
    label_x: float,
) -> None:
    scatter_kwargs = {
        "s": STATION_MARKER_SIZE,
        "edgecolor": "none",
        "linewidth": 0.0,
        "transform": ax.transAxes,
        "clip_on": False,
    }
    if kind == "both":
        ax.scatter([marker_x], [y], marker=LEFT_HALF_TRIANGLE, facecolor=FORCING_STATION_COLOR, **scatter_kwargs)
        ax.scatter([marker_x], [y], marker=RIGHT_HALF_TRIANGLE, facecolor=SNOW_STATION_COLOR, **scatter_kwargs)
    elif kind == "holdout":
        holdout_kwargs = dict(scatter_kwargs)
        holdout_kwargs.pop("edgecolor")
        holdout_kwargs["linewidth"] = HOLDOUT_STATION_LINEWIDTH
        ax.scatter(
            [marker_x],
            [y],
            marker=HOLDOUT_STATION_MARKER,
            color=HOLDOUT_STATION_COLOR,
            **holdout_kwargs,
        )
    else:
        color = {"forcing": FORCING_STATION_COLOR, "snow": SNOW_STATION_COLOR}[kind]
        ax.scatter([marker_x], [y], marker="^", facecolor=color, **scatter_kwargs)
    ax.text(label_x, y, label, transform=ax.transAxes, ha="left", va="center", fontsize=_STATION_LEGEND_FONT_SIZE)


def draw_heading(ax, *, y: float, text: str) -> float:
    ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=_LEGEND_HEADING_FONT_SIZE)
    return y - 0.05


def panel_below_item_units(item: LegendItemSpec) -> float:
    if item.kind == "heading":
        return 0.65
    if item.kind == "station_symbol":
        return 0.40
    if item.kind == "station_categories":
        return 2.60
    return 1.0


def panel_below_items_layout(
    items: tuple[LegendItemSpec, ...],
    *,
    panel_height_in: float | None = None,
    panel_aspect: float | None = None,
) -> tuple[float, float, float, float]:
    row_units = sum(panel_below_item_units(item) for item in items)
    inset_height = max(row_units * _PANEL_BELOW_ITEMS_ROW_HEIGHT_AXES, 1e-9)
    base_gap = max(
        _PANEL_BELOW_ITEMS_GAP_AXES,
        _PANEL_BELOW_ITEMS_DRAW_GAP_BASE_AXES + _PANEL_BELOW_ITEMS_DRAW_GAP_PER_HEIGHT_AXES * inset_height,
    )
    draw_gap = horizontal_annotation_gap_axes(
        panel_height_in=panel_height_in,
        panel_aspect=panel_aspect,
        base_gap_axes=base_gap,
        min_gap_in=_PANEL_BELOW_ITEMS_MIN_GAP_IN,
    )
    reserve_extra = draw_gap + inset_height + _PANEL_BELOW_ITEMS_BOTTOM_PAD_AXES
    return row_units, inset_height, reserve_extra, draw_gap


def panel_below_items_extra(
    items: tuple[LegendItemSpec, ...],
    *,
    panel_height_in: float | None = None,
    panel_aspect: float | None = None,
) -> float:
    if not items:
        return 0.0
    _, _, reserve_extra, _ = panel_below_items_layout(
        items,
        panel_height_in=panel_height_in,
        panel_aspect=panel_aspect,
    )
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
    items = panel.bottom_legend_items
    if not items:
        return
    row_units, inset_height, _, draw_gap = panel_below_items_layout(
        items,
        panel_height_in=axis_height_inches(ax),
        panel_aspect=axis_height_inches(ax) / max(axis_width_inches(ax), 1e-9),
    )
    legend_ax = ax.inset_axes(
        [0.0, -(draw_gap + inset_height), 1.0, inset_height],
        transform=ax.transAxes,
    )
    register_child_axes(ax, legend_ax)
    legend_ax.set_axis_off()
    y = 1.0 - (0.4 / max(row_units, 1e-9))
    _draw_legend_items_on_axis(
        legend_ax,
        items=items,
        y=y,
        artifacts=artifacts,
        legend_source_handles_getter=legend_source_handles_getter,
        context_label="below-panel",
    )


def _draw_legend_items_on_axis(
    ax,
    *,
    items: tuple[LegendItemSpec, ...],
    y: float,
    artifacts: dict[str, dict[str, object]],
    legend_source_handles_getter,
    context_label: str,
) -> float:
    for item in items:
        if item.kind == "heading":
            y = draw_heading(ax, y=y, text=str(item.label))
        elif item.kind == "station_symbol":
            y = draw_station_entry(ax, y=y, label=str(item.label))
        elif item.kind == "station_categories":
            if context_label == "below-panel":
                y = draw_station_categories_below(ax, y=y)
            else:
                y = draw_station_categories(ax, y=y)
        elif item.kind == "source_legend":
            if item.label:
                y = draw_heading(ax, y=y, text=str(item.label))
            for handle in legend_source_handles_getter(item, artifacts):
                y = draw_patch_entry(
                    ax,
                    y=y,
                    label=handle.get_label(),
                    facecolor=handle.get_facecolor(),
                    edgecolor=handle.get_edgecolor(),
                )
        elif item.kind == "scale_bar":
            continue
        else:
            raise ValueError(f"Unsupported {context_label} legend item kind '{item.kind}'")
    return y


def _legend_label_width_in(label: str | None, *, fontsize: float, text_x: float) -> float:
    if not label:
        return 0.0
    label_width_in, _ = text_size_in(str(label), size=fontsize)
    available_fraction = max(1.0 - text_x - _INSIDE_LEGEND_RIGHT_PAD, 0.10)
    return _INSIDE_LEGEND_TEXT_WIDTH_SAFETY * label_width_in / available_fraction


def _inside_legend_width(
    ax,
    *,
    items: tuple[LegendItemSpec, ...],
    artifacts: dict[str, dict[str, object]],
    legend_source_handles_getter,
) -> float:
    required_width_in = 0.0
    for item in items:
        if item.kind == "heading":
            required_width_in = max(
                required_width_in,
                _legend_label_width_in(item.label, fontsize=_LEGEND_HEADING_FONT_SIZE, text_x=0.0),
            )
        elif item.kind == "station_symbol":
            required_width_in = max(
                required_width_in,
                _legend_label_width_in(item.label, fontsize=_STATION_LEGEND_FONT_SIZE, text_x=_STATION_LABEL_X),
            )
        elif item.kind == "station_categories":
            for label in ("Forcing station", "Snow observation station", "Forcing + snow station", "Holdout snow station"):
                required_width_in = max(
                    required_width_in,
                    _legend_label_width_in(label, fontsize=_STATION_LEGEND_FONT_SIZE, text_x=_STATION_LABEL_X),
                )
        elif item.kind == "source_legend":
            required_width_in = max(
                required_width_in,
                _legend_label_width_in(item.label, fontsize=_LEGEND_HEADING_FONT_SIZE, text_x=0.0),
            )
            for handle in legend_source_handles_getter(item, artifacts):
                required_width_in = max(
                    required_width_in,
                    _legend_label_width_in(handle.get_label(), fontsize=_LEGEND_ENTRY_FONT_SIZE, text_x=_PATCH_LABEL_X),
                )
    parent_width_in = max(axis_width_inches(ax), 1e-9)
    measured_width = required_width_in / parent_width_in
    max_width = max(1.0 - 2.0 * _INSIDE_LEGEND_ANCHOR_PAD, _INSIDE_LEGEND_MIN_WIDTH)
    return min(max(_INSIDE_LEGEND_MIN_WIDTH, measured_width), max_width)


def _inside_legend_bounds(
    ax,
    anchor: str | None,
    *,
    row_units: float,
    items: tuple[LegendItemSpec, ...],
    artifacts: dict[str, dict[str, object]],
    legend_source_handles_getter,
) -> tuple[float, float, float, float]:
    width = _inside_legend_width(
        ax,
        items=items,
        artifacts=artifacts,
        legend_source_handles_getter=legend_source_handles_getter,
    )
    height = min(
        _INSIDE_LEGEND_MAX_HEIGHT,
        max(_INSIDE_LEGEND_MIN_HEIGHT, row_units * _INSIDE_LEGEND_UNIT_HEIGHT),
    )
    token = str(anchor or "top_left")
    x0 = _INSIDE_LEGEND_ANCHOR_PAD if token.endswith("left") else 1.0 - width - _INSIDE_LEGEND_ANCHOR_PAD
    y0 = 1.0 - height - _INSIDE_LEGEND_ANCHOR_PAD if token.startswith("top") else _INSIDE_LEGEND_ANCHOR_PAD
    return x0, y0, width, height


def _clip_axis_artists(ax) -> None:
    for artist in [*ax.texts, *ax.collections, *ax.patches]:
        artist.set_clip_on(True)


def draw_panel_inside_items(
    ax,
    *,
    panel: MapPanelSpec,
    artifacts: dict[str, dict[str, object]],
    legend_source_handles_getter,
) -> None:
    grouped: dict[str | None, list[LegendItemSpec]] = {}
    for item in panel.inside_legend_items:
        grouped.setdefault(item.anchor or "top_left", []).append(item)
    for anchor, raw_items in grouped.items():
        items = tuple(raw_items)
        row_units = sum(panel_below_item_units(item) for item in items)
        x0, y0, width, height = _inside_legend_bounds(
            ax,
            anchor,
            row_units=row_units,
            items=items,
            artifacts=artifacts,
            legend_source_handles_getter=legend_source_handles_getter,
        )
        legend_ax = ax.inset_axes(
            [x0, y0, width, height],
            transform=ax.transAxes,
            zorder=_ANNOTATION_ZORDER + 1,
        )
        register_child_axes(ax, legend_ax)
        legend_ax.set_facecolor(_INSIDE_LEGEND_FACE)
        legend_ax.set_xticks([])
        legend_ax.set_yticks([])
        legend_ax.set_xlim(0.0, 1.0)
        legend_ax.set_ylim(0.0, 1.0)
        for spine in legend_ax.spines.values():
            spine.set_visible(False)
        y = 0.86 if any(item.kind == "station_categories" for item in items) else (0.5 if len(items) == 1 else 0.88)
        _draw_legend_items_on_axis(
            legend_ax,
            items=items,
            y=y,
            artifacts=artifacts,
            legend_source_handles_getter=legend_source_handles_getter,
            context_label="inside-panel",
        )
        _clip_axis_artists(legend_ax)


def draw_panel_legend_items(
    ax,
    *,
    panel: MapPanelSpec,
    artifacts: dict[str, dict[str, object]],
    legend_source_handles_getter,
) -> None:
    draw_panel_below_items(
        ax,
        panel=panel,
        artifacts=artifacts,
        legend_source_handles_getter=legend_source_handles_getter,
    )
    draw_panel_inside_items(
        ax,
        panel=panel,
        artifacts=artifacts,
        legend_source_handles_getter=legend_source_handles_getter,
    )


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
