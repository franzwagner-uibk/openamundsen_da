from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LightSource, Normalize, TwoSlopeNorm
from matplotlib.patches import Patch
from rasterio.transform import array_bounds

from openamundsen_da.methods.viz._style import FIGWIDTH_OVERVIEW_PAPER
from openamundsen_da.methods.viz._utils import force_figure_text_black, save_figure_png
from openamundsen_da.methods.viz.project_maps.data import ModelFields, ObservationScene, StaticContext
from openamundsen_da.methods.viz.project_maps.styles import (
    FSC_OBS_CMAP,
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
)


_FIGURE_HEIGHT_MIN = 2.8
_FIGURE_HEIGHT_MAX = 5.2
_BUFFER_RATIO = 0.03
_STATION_LABEL_RATIO = 0.04
_PANEL_LETTERS = ("a", "b", "c")


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
    computed = panel_width * aspect * 1.2
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


def _hillshade(context: StaticContext) -> np.ndarray:
    dem = np.asarray(context.dem, dtype=float)
    filled = dem.copy()
    if np.isfinite(filled).any():
        filled[~np.isfinite(filled)] = float(np.nanmedian(filled))
    else:
        filled[:] = 0.0
    light = LightSource(azdeg=315, altdeg=42)
    shade = light.hillshade(
        filled,
        vert_exag=1.3,
        dx=abs(float(context.spec.transform.a)),
        dy=abs(float(context.spec.transform.e)),
    )
    shade = np.where(context.roi_mask, shade, np.nan)
    return shade


def _apply_common_axis_style(ax, extent: tuple[float, float, float, float], panel_letter: str) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(
        0.02,
        0.98,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )


def _draw_roi(ax, context: StaticContext) -> None:
    context.roi_gdf.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=20)


def _draw_scale_bar(ax, extent: tuple[float, float, float, float]) -> None:
    span = float(extent[1] - extent[0])
    target = span * 0.20
    exponent = int(np.floor(np.log10(max(target, 1.0))))
    base = 10**exponent
    candidates = [1.0 * base, 2.0 * base, 5.0 * base, 10.0 * base]
    length = min(candidates, key=lambda value: abs(value - target))
    x0 = float(extent[0] + 0.08 * span)
    y0 = float(extent[2] + 0.08 * (extent[3] - extent[2]))
    ax.plot([x0, x0 + length], [y0, y0], color="black", lw=2.0, solid_capstyle="butt", zorder=30)
    ax.plot([x0, x0], [y0, y0 + 0.015 * (extent[3] - extent[2])], color="black", lw=1.5, zorder=30)
    ax.plot(
        [x0 + length, x0 + length],
        [y0, y0 + 0.015 * (extent[3] - extent[2])],
        color="black",
        lw=1.5,
        zorder=30,
    )
    label = f"{length/1000:.0f} km" if length >= 1000 else f"{length:.0f} m"
    ax.text(
        x0 + 0.5 * length,
        y0 + 0.018 * (extent[3] - extent[2]),
        label,
        ha="center",
        va="bottom",
        fontsize=8,
        zorder=30,
        bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )


def _suppress_station_labels(stations, extent: tuple[float, float, float, float]) -> list[int]:
    kept: list[int] = []
    min_dx = (extent[1] - extent[0]) * _STATION_LABEL_RATIO
    min_dy = (extent[3] - extent[2]) * _STATION_LABEL_RATIO
    for idx, row in stations.sort_values("id").reset_index(drop=True).iterrows():
        x = float(row["x"])
        y = float(row["y"])
        if any(abs(x - float(stations.iloc[prev]["x"])) < min_dx and abs(y - float(stations.iloc[prev]["y"])) < min_dy for prev in kept):
            continue
        kept.append(idx)
    return kept


def _draw_stations(ax, context: StaticContext, extent: tuple[float, float, float, float]) -> None:
    stations = context.stations
    if stations is None or stations.empty or not {"id", "x", "y"}.issubset(stations.columns):
        return
    working = stations.copy()
    if "name" not in working.columns:
        working["name"] = working["id"]
    ax.scatter(working["x"], working["y"], s=22, facecolor="white", edgecolor="black", linewidth=0.8, zorder=25)
    for idx in _suppress_station_labels(working, extent):
        row = working.sort_values("id").reset_index(drop=True).iloc[idx]
        ax.text(
            float(row["x"]) + 0.01 * (extent[1] - extent[0]),
            float(row["y"]) + 0.01 * (extent[3] - extent[2]),
            str(row["id"]),
            fontsize=7,
            color="black",
            zorder=30,
            bbox={"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
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


def _apply_colorbar_style(colorbar, *, label: str, ticks: tuple[float, ...] = (), ticklabels: tuple[str, ...] = ()) -> None:
    colorbar.set_label(label, fontsize=8)
    if ticks:
        colorbar.set_ticks(ticks)
    if ticklabels:
        colorbar.set_ticklabels(ticklabels)
    colorbar.ax.tick_params(labelsize=7)


def render_overview_map(
    *,
    context: StaticContext,
    title: str,
    output_path: Path,
) -> Path:
    extent = buffered_extent(context)
    grid_extent = _grid_extent(context)
    fig, axes = plt.subplots(1, 3, figsize=(FIGWIDTH_OVERVIEW_PAPER, figure_height_for_extent(extent)))
    hillshade = _hillshade(context)
    masked_landcover = _masked(context.landcover, context.roi_mask)
    present_codes = sorted({int(value) for value in masked_landcover.compressed() if np.isfinite(value)})
    if not present_codes:
        present_codes = [0]
    code_to_index = {code: idx for idx, code in enumerate(present_codes)}
    landcover_indices = np.full(masked_landcover.shape, np.nan, dtype=float)
    for code, idx in code_to_index.items():
        landcover_indices[np.isclose(masked_landcover.filled(np.nan), float(code), equal_nan=False)] = idx
    lc_cmap = landcover_cmap_for_codes(present_codes)
    lc_norm = BoundaryNorm(np.arange(-0.5, len(present_codes) + 0.5), lc_cmap.N)

    axes[0].imshow(hillshade, cmap="Greys", extent=grid_extent, origin="upper", vmin=0.0, vmax=1.0)
    axes[1].imshow(
        landcover_indices,
        cmap=lc_cmap,
        norm=lc_norm,
        extent=grid_extent,
        origin="upper",
        interpolation="nearest",
    )
    axes[2].imshow(hillshade, cmap="Greys", extent=grid_extent, origin="upper", vmin=0.0, vmax=1.0)

    panel_titles = ("hillshade", "landcover", "stations")
    for ax, letter, panel_title in zip(axes, _PANEL_LETTERS, panel_titles):
        _draw_roi(ax, context)
        _apply_common_axis_style(ax, extent, letter)
        ax.set_title(panel_title, fontsize=9)
    _draw_stations(axes[2], context, extent)
    _draw_scale_bar(axes[0], extent)

    legend_handles = [
        Patch(facecolor=lc_cmap(code_to_index[code]), edgecolor="none", label=LANDCOVER_LABELS.get(code, str(code)))
        for code in present_codes
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, max(1, len(legend_handles))),
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(title, fontsize=11, x=0.05, ha="left")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.88, bottom=0.16, wspace=0.02)
    force_figure_text_black(fig, axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, output_path)
    plt.close(fig)
    return output_path


def render_comparison_map(
    *,
    context: StaticContext,
    fields: ModelFields,
    all_fields: list[ModelFields],
    preset,
    title: str,
    output_path: Path,
) -> Path:
    extent = buffered_extent(context)
    grid_extent = _grid_extent(context)
    fig, axes = plt.subplots(1, 3, figsize=(FIGWIDTH_OVERVIEW_PAPER, figure_height_for_extent(extent)))
    hillshade = _hillshade(context)
    model_norm, increment_norm = _comparison_scales(all_fields, preset)
    model_cmap = model_map_cmap(preset)
    model_cbar_style = model_colorbar_style(preset)

    images = []
    for ax, letter, panel_title, arr in (
        (axes[0], _PANEL_LETTERS[0], "open loop", _masked_model(fields.open_loop, context.roi_mask, preset=preset)),
        (axes[1], _PANEL_LETTERS[1], "ensemble mean", _masked_model(fields.ens_mean, context.roi_mask, preset=preset)),
    ):
        ax.imshow(hillshade, cmap="Greys", extent=grid_extent, origin="upper", vmin=0.0, vmax=1.0)
        images.append(
            ax.imshow(
                arr,
                cmap=model_cmap,
                norm=model_norm,
                extent=grid_extent,
                origin="upper",
                interpolation="nearest",
                alpha=0.88,
            )
        )
        _draw_roi(ax, context)
        _apply_common_axis_style(ax, extent, letter)
        ax.set_title(panel_title, fontsize=9)

    inc_image = axes[2].imshow(
        _masked(fields.increment, context.roi_mask),
        cmap=INCREMENT_CMAP,
        norm=increment_norm,
        extent=grid_extent,
        origin="upper",
        interpolation="nearest",
    )
    _draw_roi(axes[2], context)
    _apply_common_axis_style(axes[2], extent, _PANEL_LETTERS[2])
    axes[2].set_title("increment", fontsize=9)
    _draw_scale_bar(axes[0], extent)

    model_cbar = fig.colorbar(images[0], ax=axes[:2], orientation="horizontal", fraction=0.05, pad=0.08)
    _apply_colorbar_style(
        model_cbar,
        label=model_cbar_style.label,
        ticks=model_cbar_style.ticks,
        ticklabels=model_cbar_style.ticklabels,
    )
    inc_cbar = fig.colorbar(inc_image, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.08)
    _apply_colorbar_style(inc_cbar, label=f"increment {preset.unit_label}")

    fig.suptitle(title, fontsize=11, x=0.05, ha="left")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.88, bottom=0.18, wspace=0.02)
    force_figure_text_black(fig, axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, output_path)
    plt.close(fig)
    return output_path


def render_observation_context_map(
    *,
    context: StaticContext,
    fields: ModelFields,
    all_fields: list[ModelFields],
    observation_scene: ObservationScene,
    preset,
    title: str,
    output_path: Path,
) -> Path:
    extent = buffered_extent(context)
    grid_extent = _grid_extent(context)
    fig, axes = plt.subplots(1, 3, figsize=(FIGWIDTH_OVERVIEW_PAPER, figure_height_for_extent(extent)))
    hillshade = _hillshade(context)
    model_norm, _increment_norm = _comparison_scales(all_fields, preset)
    model_cmap = model_map_cmap(preset)
    model_cbar_style = model_colorbar_style(preset)

    for ax, letter, panel_title, arr in (
        (axes[0], _PANEL_LETTERS[0], "open loop", _masked_model(fields.open_loop, context.roi_mask, preset=preset)),
        (axes[1], _PANEL_LETTERS[1], "ensemble mean", _masked_model(fields.ens_mean, context.roi_mask, preset=preset)),
    ):
        ax.imshow(hillshade, cmap="Greys", extent=grid_extent, origin="upper", vmin=0.0, vmax=1.0)
        ax.imshow(
            arr,
            cmap=model_cmap,
            norm=model_norm,
            extent=grid_extent,
            origin="upper",
            interpolation="nearest",
            alpha=0.88,
        )
        _draw_roi(ax, context)
        _apply_common_axis_style(ax, extent, letter)
        ax.set_title(panel_title, fontsize=9)

    obs_extent = observation_scene.bounds
    if observation_scene.observation == "scf":
        obs_image = axes[2].imshow(
            np.ma.masked_invalid(observation_scene.array),
            cmap=FSC_OBS_CMAP,
            norm=Normalize(vmin=0.0, vmax=100.0),
            extent=obs_extent,
            origin="upper",
            interpolation="nearest",
        )
        obs_legend = None
    else:
        codes = sorted(WET_SNOW_COLORS)
        code_to_index = {code: idx for idx, code in enumerate(codes)}
        categorical = np.full(observation_scene.array.shape, np.nan, dtype=float)
        for code, idx in code_to_index.items():
            categorical[np.isclose(observation_scene.array, float(code), equal_nan=False)] = idx
        cmap = matplotlib.colors.ListedColormap([WET_SNOW_COLORS[code] for code in codes], name="wet_snow_obs")
        norm = BoundaryNorm(np.arange(-0.5, len(codes) + 0.5), cmap.N)
        obs_image = axes[2].imshow(
            np.ma.masked_invalid(categorical),
            cmap=cmap,
            norm=norm,
            extent=obs_extent,
            origin="upper",
            interpolation="nearest",
        )
        obs_legend = [
            Patch(facecolor=WET_SNOW_COLORS[code], edgecolor="none", label=WET_SNOW_LABELS[code])
            for code in codes
        ]

    _draw_roi(axes[2], context)
    _apply_common_axis_style(axes[2], extent, _PANEL_LETTERS[2])
    coverage_pct = int(round(observation_scene.coverage_fraction * 100.0))
    obs_title = "observation"
    obs_subtitle = f"coverage {coverage_pct}%"
    axes[2].set_title(f"{obs_title}\n{obs_subtitle}", fontsize=9)
    _draw_scale_bar(axes[0], extent)

    model_cbar = fig.colorbar(axes[0].images[-1], ax=axes[:2], orientation="horizontal", fraction=0.05, pad=0.08)
    _apply_colorbar_style(
        model_cbar,
        label=model_cbar_style.label,
        ticks=model_cbar_style.ticks,
        ticklabels=model_cbar_style.ticklabels,
    )
    if observation_scene.observation == "scf":
        obs_cbar = fig.colorbar(obs_image, ax=axes[2], orientation="horizontal", fraction=0.05, pad=0.08)
        _apply_colorbar_style(obs_cbar, label="fractional snow cover [%]")
    else:
        fig.legend(
            handles=obs_legend,
            loc="lower right",
            bbox_to_anchor=(0.995, 0.02),
            frameon=False,
            fontsize=7,
        )

    fig.suptitle(title, fontsize=11, x=0.05, ha="left")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.88, bottom=0.18, wspace=0.02)
    force_figure_text_black(fig, axes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, output_path)
    plt.close(fig)
    return output_path
