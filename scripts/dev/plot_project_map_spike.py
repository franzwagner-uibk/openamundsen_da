#!/usr/bin/env python3
"""Prototype snow-depth map renderer for completed project runs.

This is an intentionally monolithic spike script. It renders one static,
publication-style figure from the compact project grid summary written by the
openAMUNDSEN-DA project pipeline.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap, Normalize
import numpy as np
import pandas as pd

from openamundsen_da.io.paths import project_map_family_dir


DEFAULT_PROJECT_DIR = Path(
    "/home/franz/workspace/dev_examples/rofental_paper/projects/project_2022_2023"
)
DEFAULT_DATE = "2023-06-02"
DEFAULT_DPI = 220
SNOW_DEPTH_BOUNDS = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0)
SNOW_DEPTH_LABELS = (
    "0 - 0.1",
    "0.1 - 0.2",
    "0.2 - 0.5",
    "0.5 - 1",
    "1 - 2",
    "2 - 3",
    "> 3",
)
SNOW_DEPTH_COLORS = (
    "#e4f2b2",
    "#bfe6c0",
    "#82d0c6",
    "#4ab3c4",
    "#2b82c5",
    "#3247a1",
    "#8d0dbd",
)


def _derive_setup_dir(project_dir: Path) -> Path:
    project_dir = Path(project_dir).resolve()
    if project_dir.parent.name != "projects":
        raise ValueError(
            f"Expected project directory under '<setup>/projects/<project>', got: {project_dir}"
        )
    return project_dir.parent.parent


def _default_output_path(project_dir: Path, date_str: str) -> Path:
    return project_map_family_dir(project_dir, "overview") / f"spike_snow_depth_{date_str}.png"


def _nice_ceiling(value: float, *, step: float, minimum: float) -> float:
    if step <= 0:
        raise ValueError("step must be > 0")
    return max(minimum, math.ceil(value / step) * step)


def _format_date_title(date_str: str) -> str:
    return pd.Timestamp(date_str).strftime("%Y/%m/%d")


def _find_nearest_dates(times: pd.DatetimeIndex, target: pd.Timestamp, *, count: int = 6) -> list[str]:
    normalized = sorted({pd.Timestamp(t).normalize() for t in times})
    ranked = sorted(normalized, key=lambda value: abs(value - target.normalize()))
    return [value.strftime("%Y-%m-%d") for value in ranked[:count]]


def _extract_grid_for_date(ds, variable_name: str, target: pd.Timestamp) -> tuple[np.ndarray, pd.Timestamp]:
    if variable_name not in ds.data_vars:
        raise KeyError(f"Missing required variable '{variable_name}' in {ds.encoding.get('source') or 'dataset'}")

    da = ds[variable_name]
    time_dims = [dim for dim in da.dims if str(dim).startswith("time")]
    if len(time_dims) != 1:
        raise ValueError(f"Expected exactly one time dimension for '{variable_name}', got {da.dims}")
    time_dim = time_dims[0]
    times = pd.to_datetime(ds[time_dim].values)
    matches = [idx for idx, value in enumerate(times) if pd.Timestamp(value).date() == target.date()]
    if not matches:
        nearby = ", ".join(_find_nearest_dates(pd.DatetimeIndex(times), target))
        raise KeyError(
            f"Date {target.strftime('%Y-%m-%d')} not found for '{variable_name}'. Nearby available dates: {nearby}"
        )

    index = matches[0]
    arr = np.asarray(da.isel({time_dim: index}).values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D raster slice for '{variable_name}', got shape {arr.shape}")
    return arr, pd.Timestamp(times[index])


def _masked_to_roi(arr: np.ndarray, roi_mask: np.ndarray) -> np.ma.MaskedArray:
    masked = np.asarray(arr, dtype=float)
    masked = np.where(roi_mask, masked, np.nan)
    return np.ma.masked_invalid(masked)


def _array_extent(bounds) -> tuple[float, float, float, float]:
    return (float(bounds.left), float(bounds.right), float(bounds.bottom), float(bounds.top))


def _snow_depth_cmap_and_norm() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(SNOW_DEPTH_COLORS, name="snow_depth_spike")
    norm = BoundaryNorm(SNOW_DEPTH_BOUNDS, cmap.N, clip=False)
    return cmap, norm


def _increment_cmap_and_norm(masked_increment: np.ma.MaskedArray) -> tuple[object, Normalize, float]:
    finite = np.asarray(masked_increment.compressed(), dtype=float)
    vmax = _nice_ceiling(float(finite.max()) if finite.size else 0.0, step=0.25, minimum=0.5)
    return matplotlib.colormaps["viridis"], Normalize(vmin=0.0, vmax=vmax), vmax


def _build_hillshade(dem: np.ndarray, roi_mask: np.ndarray, resolution_m: float) -> np.ndarray:
    dem_arr = np.asarray(dem, dtype=float)
    finite = np.isfinite(dem_arr)
    if not np.any(finite):
        raise ValueError("DEM contains no finite values")
    filled = dem_arr.copy()
    filled[~finite] = float(np.nanmedian(dem_arr))
    light = LightSource(azdeg=315, altdeg=42)
    hillshade = light.hillshade(filled, vert_exag=1.3, dx=resolution_m, dy=resolution_m)
    return np.where(roi_mask, hillshade, np.nan)


def _draw_roi_boundary(ax, roi_gdf) -> None:
    roi_gdf.boundary.plot(ax=ax, color="black", linewidth=1.2, zorder=20)


def _style_map_axis(ax, extent: tuple[float, float, float, float], panel_label: str) -> None:
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.text(
        0.5,
        -0.07,
        panel_label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=17,
    )


def _draw_scale_bar(ax) -> None:
    x0 = 0.10
    x1 = 0.86
    y = 0.09
    tick_h = 0.035
    ax.plot([x0, x1], [y, y], color="black", lw=1.8, transform=ax.transAxes, clip_on=False)
    for frac in (0.0, 0.5, 1.0):
        x = x0 + frac * (x1 - x0)
        ax.plot([x, x], [y, y + tick_h], color="black", lw=1.8, transform=ax.transAxes, clip_on=False)
    ax.text(x0, y + tick_h + 0.02, "0", transform=ax.transAxes, ha="center", va="bottom", fontsize=12)
    ax.text(
        x0 + 0.5 * (x1 - x0),
        y + tick_h + 0.02,
        "2.5",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    ax.text(x1, y + tick_h + 0.02, "5 km", transform=ax.transAxes, ha="center", va="bottom", fontsize=12)


def _draw_legends(ax, increment_cmap, increment_norm, increment_vmax: float) -> None:
    ax.set_axis_off()
    ax.text(0.00, 0.88, "snow depth [m]", transform=ax.transAxes, ha="left", va="top", fontsize=16)
    start_y = 0.80
    step_y = 0.060
    for idx, (color, label) in enumerate(zip(SNOW_DEPTH_COLORS, SNOW_DEPTH_LABELS)):
        y = start_y - idx * step_y
        ax.add_patch(
            plt.Rectangle(
                (0.00, y - 0.03),
                0.17,
                0.045,
                facecolor=color,
                edgecolor="none",
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(0.22, y - 0.006, label, transform=ax.transAxes, ha="left", va="center", fontsize=12)

    cb_ax = ax.inset_axes([0.00, 0.15, 0.17, 0.14])
    sm = cm.ScalarMappable(norm=increment_norm, cmap=increment_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cb_ax, orientation="vertical")
    cbar.set_ticks([0.0, increment_vmax])
    cbar.set_ticklabels(["0", f"{increment_vmax:.2f}".rstrip("0").rstrip(".")])
    cbar.ax.tick_params(labelsize=11, length=0)
    ax.text(
        0.00,
        0.30,
        "increment snow\ndepth [m]",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
    )
    _draw_scale_bar(ax)


def _load_plot_inputs(project_dir: Path, date_str: str) -> dict[str, object]:
    import geopandas as gpd
    import rasterio
    import xarray as xr

    from openamundsen_da.util.roi_grid import discover_setup_roi_vector, load_setup_roi_mask

    project_dir = Path(project_dir).resolve()
    setup_dir = _derive_setup_dir(project_dir)
    da_output_path = project_dir / "results" / "grids" / "da_output_grids.nc"
    if not da_output_path.is_file():
        raise FileNotFoundError(f"Compact DA output grid not found: {da_output_path}")

    roi_mask, spec, _ = load_setup_roi_mask(setup_dir, ensure_grid=False)
    roi_vector_path = discover_setup_roi_vector(setup_dir)
    if roi_vector_path is None or not roi_vector_path.is_file():
        raise FileNotFoundError(f"ROI vector not found under {setup_dir / 'env'}")
    roi_gdf = gpd.read_file(roi_vector_path)
    if roi_gdf.empty:
        raise ValueError(f"ROI vector has no features: {roi_vector_path}")
    if spec.crs and roi_gdf.crs is not None:
        roi_gdf = roi_gdf.to_crs(spec.crs)

    target = pd.Timestamp(date_str)
    with xr.open_dataset(da_output_path) as ds:
        open_loop, resolved_time = _extract_grid_for_date(ds, "open_loop_snowdepth_daily", target)
        ens_mean, _ = _extract_grid_for_date(ds, "ens_mean_snowdepth_daily", target)
        increment, _ = _extract_grid_for_date(ds, "increment_snowdepth_daily", target)

    with rasterio.open(spec.dem_path) as src:
        dem = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            dem[dem == nodata] = np.nan
        bounds = src.bounds

    resolution_m = abs(float(spec.transform.a))
    return {
        "project_dir": project_dir,
        "setup_dir": setup_dir,
        "resolved_time": resolved_time,
        "roi_mask": roi_mask,
        "roi_gdf": roi_gdf,
        "dem": dem,
        "bounds": bounds,
        "resolution_m": resolution_m,
        "open_loop": _masked_to_roi(open_loop, roi_mask),
        "ens_mean": _masked_to_roi(ens_mean, roi_mask),
        "increment": _masked_to_roi(increment, roi_mask),
    }


def _plot_figure(plot_inputs: dict[str, object], output_path: Path, dpi: int) -> None:
    extent = _array_extent(plot_inputs["bounds"])
    hillshade = _build_hillshade(
        dem=plot_inputs["dem"],
        roi_mask=plot_inputs["roi_mask"],
        resolution_m=float(plot_inputs["resolution_m"]),
    )
    snow_cmap, snow_norm = _snow_depth_cmap_and_norm()
    increment_cmap, increment_norm, increment_vmax = _increment_cmap_and_norm(plot_inputs["increment"])

    fig = plt.figure(figsize=(17, 6.5))
    gs = fig.add_gridspec(1, 4, width_ratios=(1.0, 1.0, 1.0, 0.85), wspace=0.06)
    ax_open = fig.add_subplot(gs[0, 0])
    ax_mean = fig.add_subplot(gs[0, 1])
    ax_increment = fig.add_subplot(gs[0, 2])
    ax_info = fig.add_subplot(gs[0, 3])

    for ax, arr, label in (
        (ax_open, plot_inputs["open_loop"], "open loop"),
        (ax_mean, plot_inputs["ens_mean"], "ensemble mean"),
    ):
        ax.imshow(hillshade, cmap="Greys", extent=extent, origin="upper", zorder=0, vmin=0.0, vmax=1.0)
        ax.imshow(
            arr,
            cmap=snow_cmap,
            norm=snow_norm,
            extent=extent,
            origin="upper",
            interpolation="nearest",
            alpha=0.90,
            zorder=5,
        )
        _draw_roi_boundary(ax, plot_inputs["roi_gdf"])
        _style_map_axis(ax, extent, label)

    ax_increment.imshow(
        plot_inputs["increment"],
        cmap=increment_cmap,
        norm=increment_norm,
        extent=extent,
        origin="upper",
        interpolation="nearest",
        zorder=5,
    )
    _draw_roi_boundary(ax_increment, plot_inputs["roi_gdf"])
    _style_map_axis(ax_increment, extent, "increment")

    _draw_legends(ax_info, increment_cmap, increment_norm, increment_vmax)

    fig.suptitle(
        f"Snow depth increment (ensemble mean - open loop) {_format_date_title(str(plot_inputs['resolved_time'].date()))}",
        fontsize=20,
        y=0.94,
    )
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.03, right=0.99)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a prototype publication-style snow-depth map figure.")
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--date", default=DEFAULT_DATE, help="Target date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path. Defaults to <project>/results/maps/overview/spike_snow_depth_<date>.png",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    project_dir = Path(args.project_dir).resolve()
    target_date = pd.Timestamp(str(args.date)).strftime("%Y-%m-%d")
    output_path = Path(args.output).resolve() if args.output else _default_output_path(project_dir, target_date)

    plot_inputs = _load_plot_inputs(project_dir=project_dir, date_str=target_date)
    _plot_figure(plot_inputs, output_path=output_path, dpi=int(args.dpi))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
