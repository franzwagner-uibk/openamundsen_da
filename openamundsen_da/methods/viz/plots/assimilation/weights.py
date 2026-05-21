"""Plot per-date assimilation weights and residual summaries.

Inputs
- weights CSV produced by one assimilation workflow with columns:
  member_id, residual, sigma, log_weight, weight

Outputs
- A PNG saved next to the CSV (or --output) with two panels:
  A) sorted normalized weights with ESS annotation
  B) one-point-per-member residual view

Logging uses LOGURU_FORMAT from core.constants.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd
from loguru import logger
from openamundsen_da.io.paths import (
    infer_project_dir,
    infer_setup_dir,
    list_steps_sorted,
    project_paper_output_path,
    project_plot_assim_weights_dir,
)
from openamundsen_da.methods.viz.theme import da_variable_line_color
from openamundsen_da.methods.viz.common import force_figure_text_black, save_figure_png, set_matplotlib_text_black
from openamundsen_da.methods.viz.wet_snow_fields import finite_numeric_column
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.da_observables import station_diagnostics_csv_name, weight_plot_title_from_csv_path
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.stats import effective_sample_size
from openamundsen_da.util.yaml_utils import read_yaml_mapping

_WEIGHTS_FIGSIZE = (7.2876875, 2.82)
_WEIGHTS_PANEL_WIDTH_RATIOS = (1.15, 3.85)
_FRACTION_MISMATCH_COLORS = {
    "scf": da_variable_line_color("scf"),
    "wet_snow": da_variable_line_color("wet_snow"),
    "wet_snow_line": da_variable_line_color("wet_snow_line"),
}
_FRACTION_DISPLAY_LABELS = {
    "scf": "SCF",
    "wet_snow": "WSF",
    "wet_snow_line": "WSLA",
}
_STATION_COLOR_CYCLES = {
    "station_hs": [
        da_variable_line_color("station_hs"),
        da_variable_line_color("station_swe"),
        "#482475",
        "#3c4f8a",
        "#1f9a8a",
        "#a2da37",
    ],
    "station_swe": [
        da_variable_line_color("station_swe"),
        da_variable_line_color("station_hs"),
        "#482475",
        "#3c4f8a",
        "#1f9a8a",
        "#a2da37",
    ],
}
_FS_TITLE = 9.4
_FS_AXIS = 8.6
_FS_TICK = 8.4
_FS_NOTE = 7.4
_COMPOSITE_ROW_HEIGHT = 1.62
_A4_PAGE_HEIGHT_INCHES = 11.6929133858
_STANDALONE_PLOT_WIDTH_SCALE = 0.80
_STANDALONE_PLOT_HEIGHT_SCALE = 0.70
_STANDALONE_PLOT_TOP = 0.80
_STANDALONE_TITLE_Y = 0.958
_STANDALONE_LEGEND_Y = 0.21
_STANDALONE_SAVE_PAD_INCHES = 0.02
_OVERVIEW_PAIR_WSPACE = 0.08
_OVERVIEW_PAIR_SPACER_RATIO = 0.20
_OVERVIEW_ROW_HSPACE = 0.60
_OVERVIEW_SHARED_RESIDUAL_PERCENTILE = 90.0
_AXIS_EDGE_PAD_FRACTION = 0.05
_OVERVIEW_MAX_ROWS_PER_PAGE = max(1, int(math.floor(_A4_PAGE_HEIGHT_INCHES / _COMPOSITE_ROW_HEIGHT)))
_WEIGHT_AXIS_TICKS = [0.0, 0.5, 1.0]
_WEIGHT_AXIS_TICK_LABELS = ["0", "0.5", "1"]


def _load_weights(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"weight", "residual"}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _apply_grid(ax) -> None:
    ax.grid(True, axis="both", which="major", alpha=0.5, linestyle="--", linewidth=0.8)


def _member_ticks(n: int) -> list[int]:
    if n <= 0:
        return []
    if n <= 12:
        return list(range(1, n + 1))

    target_tick_count = 5
    candidate_steps = [2, 5, 10, 20, 25, 50, 100]
    step = min(
        candidate_steps,
        key=lambda candidate: (
            abs((1 + math.floor((n - 1) / candidate)) - target_tick_count),
            candidate,
        ),
    )
    ticks = [1]
    ticks.extend(range(step, n + 1, step))
    return sorted(set(ticks))


def _observable_from_csv_path(csv_path: Path) -> str | None:
    stem = Path(csv_path).stem.lower()
    prefixes = [
        ("weights_wet_snow_line_", "wet_snow_line"),
        ("weights_station_hs_", "station_hs"),
        ("weights_station_swe_", "station_swe"),
        ("weights_wet_snow_", "wet_snow"),
        ("weights_scf_", "scf"),
    ]
    for prefix, variable in prefixes:
        if stem.startswith(prefix):
            return variable
    return None


def _weights_date_from_csv_path(csv_path: Path) -> datetime | None:
    stem = Path(csv_path).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    ds = parts[-1]
    if len(ds) != 8 or not ds.isdigit():
        return None
    try:
        return datetime.strptime(ds, "%Y%m%d")
    except Exception:
        return None


def _fraction_axis_label(observable: str | None) -> str:
    if observable in {"scf", "wet_snow"}:
        return "residual [-]"
    if observable == "wet_snow_line":
        return "residual [m]"
    return "residual"


def _draw_wsl_unavailable_overlay(ax, df: pd.DataFrame, *, fontsize: float) -> None:
    gate_columns = ("support_gate_triggered", "wet_information_gate_triggered", "model_gate_triggered")
    gate_triggered = any(
        column in df.columns and df[column].astype(str).str.lower().isin({"true", "1", "yes"}).any()
        for column in gate_columns
    )
    if not gate_triggered and not finite_numeric_column(df, "residual").empty:
        return
    ax.text(
        0.5,
        0.5,
        "Wet snow line unavailable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize + 0.3,
        color="#000000",
        zorder=6,
        bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "none", "alpha": 0.92},
    )


def _station_axis_label(observable: str | None) -> str:
    if observable == "station_hs":
        return "residual [m]"
    if observable == "station_swe":
        return "residual [mm]"
    return "residual"


def _station_diagnostics_path(csv_path: Path, observable: str | None) -> Path | None:
    if observable not in {"station_hs", "station_swe"}:
        return None
    dt = _weights_date_from_csv_path(csv_path)
    if dt is None or csv_path.parent.name != "assim":
        return None
    try:
        return csv_path.parent / station_diagnostics_csv_name(observable, dt)
    except Exception:
        return None


def _resample_artifact_paths(csv_path: Path) -> tuple[Path | None, Path | None]:
    dt = _weights_date_from_csv_path(csv_path)
    if dt is None or csv_path.parent.name != "assim":
        return None, None
    label = dt.strftime("%Y%m%d")
    assim_dir = csv_path.parent
    return assim_dir / f"resample_manifest_{label}.json", assim_dir / f"resample_indices_{label}.csv"


def _read_resample_manifest(csv_path: Path) -> dict:
    manifest_path, _indices_path = _resample_artifact_paths(csv_path)
    if manifest_path is None or not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resample_source_member_counts(csv_path: Path) -> tuple[dict[str, int], bool]:
    manifest_path, indices_path = _resample_artifact_paths(csv_path)
    if manifest_path is None or indices_path is None or not manifest_path.is_file() or not indices_path.is_file():
        return {}, False
    manifest = _read_resample_manifest(csv_path)
    if not manifest:
        return {}, False
    skipped = bool(manifest.get("skipped", False))
    try:
        idx_df = pd.read_csv(indices_path)
    except Exception:
        return {}, skipped
    if "source_member_id" not in idx_df.columns:
        return {}, skipped
    if skipped:
        member_ids = idx_df["source_member_id"].dropna().astype(str).tolist()
        return {member_id: 1 for member_id in member_ids}, True
    counts = idx_df["source_member_id"].dropna().astype(str).value_counts()
    return {str(member_id): int(count) for member_id, count in counts.items()}, False


def _draw_resample_rings(
    ax,
    x,
    y,
    draw_counts: np.ndarray,
    *,
    base_size: float,
    ring_step: float = 14.0,
    line_scale: float = 1.0,
) -> None:
    if len(x) == 0:
        return
    unique_counts = sorted({int(c) for c in draw_counts if int(c) > 0})
    for count in unique_counts:
        mask = np.asarray(draw_counts == count, dtype=bool)
        if not mask.any():
            continue
        ring_total = max(1, count)
        for ring_idx in range(ring_total):
            ax.scatter(
                np.asarray(x)[mask],
                np.asarray(y)[mask],
                s=base_size + ring_step * ring_idx,
                facecolors="none",
                edgecolors="#000000",
                linewidths=(0.9 if ring_idx == 0 else 0.75) * line_scale,
                zorder=5 + ring_idx * 0.01,
            )


def _station_display_names(csv_path: Path, station_ids: list[str]) -> dict[str, str]:
    if not station_ids:
        return {}
    try:
        project_dir = infer_project_dir(csv_path.parent.parent)
        setup_dir = infer_setup_dir(project_dir)
        meta_path = setup_dir / "meteo" / "stations.csv"
        if not meta_path.is_file():
            return {}
        meta = pd.read_csv(meta_path)
    except Exception:
        return {}

    cols_lower = {str(c).lower(): c for c in meta.columns}
    id_col = next((cols_lower[c] for c in ("id", "station_id", "station", "code") if c in cols_lower), None)
    name_col = next((cols_lower[c] for c in ("name", "station_name") if c in cols_lower), None)
    if id_col is None or name_col is None:
        return {}

    mapping: dict[str, str] = {}
    for _, row in meta.iterrows():
        station_id = str(row[id_col]).strip()
        station_name = str(row[name_col]).strip()
        if station_id:
            mapping[station_id] = station_name or station_id
    return {station_id: mapping.get(station_id, station_id) for station_id in station_ids}


def _station_metadata_uncertainty_pct(csv_path: Path, station_ids: list[str]) -> dict[str, float]:
    if not station_ids:
        return {}
    try:
        project_dir = infer_project_dir(csv_path.parent.parent)
        meta_path = project_dir / "obs" / "stations" / "stations_da_metadata.csv"
        if not meta_path.is_file():
            return {}
        meta = pd.read_csv(meta_path)
    except Exception:
        return {}

    cols_lower = {str(c).lower(): c for c in meta.columns}
    id_col = next((cols_lower[c] for c in ("station_id", "id", "station", "code") if c in cols_lower), None)
    pct_col = cols_lower.get("station_uncertainty_pct")
    if id_col is None or pct_col is None:
        return {}

    selected = {station_id.strip().lower() for station_id in station_ids}
    mapping: dict[str, float] = {}
    for _, row in meta.iterrows():
        station_id = str(row[id_col]).strip().lower()
        if not station_id or station_id not in selected:
            continue
        try:
            pct = float(row[pct_col])
        except Exception:
            continue
        if np.isfinite(pct):
            mapping[station_id] = pct
    return mapping


def _project_dir_for_weights_csv(csv_path: Path) -> Path | None:
    if Path(csv_path).parent.name != "assim":
        return None
    try:
        return infer_project_dir(Path(csv_path).parent.parent)
    except Exception:
        return None


def _resolve_station_plot_color(value: object, *, context: str) -> str:
    from matplotlib.colors import is_color_like

    color_value = str(value).strip()
    if not color_value:
        raise ValueError(f"{context} must not be empty")
    color_alias = color_value.lower()
    if color_alias in _FRACTION_MISMATCH_COLORS or color_alias in _STATION_COLOR_CYCLES:
        return da_variable_line_color(color_alias)
    if is_color_like(color_value):
        return color_value
    raise ValueError(f"{context} must be a DA variable alias or matplotlib color, got {value!r}")


def _load_weights_station_color_config(project_dir: Path | None) -> dict[str, dict[str, str]]:
    if project_dir is None:
        return {}
    config_path = Path(project_dir) / "plots.yml"
    if not config_path.is_file():
        return {}
    config = read_yaml_mapping(config_path, error_cls=RuntimeError, context="plots.yml root")
    weights_cfg = config.get("weights") or {}
    if not isinstance(weights_cfg, dict):
        raise ValueError(f"weights config in {config_path} must be a mapping")
    station_colors = weights_cfg.get("station_colors") or {}
    if not isinstance(station_colors, dict):
        raise ValueError(f"weights.station_colors in {config_path} must be a mapping")

    parsed: dict[str, dict[str, str]] = {}
    for observable_raw, station_mapping in station_colors.items():
        observable = str(observable_raw).strip().lower()
        if observable not in {"station_hs", "station_swe"}:
            raise ValueError(f"weights.station_colors only supports station_hs/station_swe, got {observable_raw!r}")
        if not isinstance(station_mapping, dict):
            raise ValueError(f"weights.station_colors.{observable} in {config_path} must be a mapping")
        parsed[observable] = {}
        for station_id_raw, color_raw in station_mapping.items():
            station_id = str(station_id_raw).strip().lower()
            if not station_id:
                raise ValueError(f"weights.station_colors.{observable} station id in {config_path} must not be empty")
            parsed[observable][station_id] = _resolve_station_plot_color(
                color_raw,
                context=f"weights.station_colors.{observable}.{station_id} in {config_path}",
            )
    return parsed


def _station_color_config_for_csv(csv_path: Path) -> dict[str, dict[str, str]]:
    return _load_weights_station_color_config(_project_dir_for_weights_csv(csv_path))


def _station_color_map(
    station_ids: list[str],
    *,
    observable: str | None = None,
    station_color_config: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    observable_key = str(observable or "").strip().lower()
    color_cycle = _STATION_COLOR_CYCLES.get(
        observable_key,
        _STATION_COLOR_CYCLES["station_hs"],
    )
    configured = {
        station_id: color
        for station_id in station_ids
        if (
            color := (station_color_config or {})
            .get(observable_key, {})
            .get(str(station_id).strip().lower())
        )
    }
    reserved_colors = {color.lower() for color in configured.values()}
    fallback_cycle = [color for color in color_cycle if color.lower() not in reserved_colors] or color_cycle
    color_map: dict[str, str] = {}
    fallback_idx = 0
    for station_id in station_ids:
        if station_id in configured:
            color_map[station_id] = configured[station_id]
            continue
        color_map[station_id] = fallback_cycle[fallback_idx % len(fallback_cycle)]
        fallback_idx += 1
    return color_map


def _marker_handle(
    color: str | None,
    *,
    size: float = 6.0,
    markeredgewidth: float = 0.9,
    edgecolor: str | None = None,
):
    from matplotlib.lines import Line2D

    facecolor = "none" if color is None else color
    markeredgecolor = edgecolor if edgecolor is not None else (color if color is not None else "#000000")
    return Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=facecolor,
        markeredgecolor=markeredgecolor,
        markeredgewidth=markeredgewidth,
        markersize=size,
    )


def _marker_legend_entries_for_csv(
    csv_path: Path,
    observable: str | None,
    *,
    station_color_config: dict[str, dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
    if observable in {"station_hs", "station_swe"}:
        diag_path = _station_diagnostics_path(csv_path, observable)
        diag = pd.read_csv(diag_path) if diag_path is not None and diag_path.is_file() else pd.DataFrame()
        if diag.empty or "station_id" not in diag.columns:
            return []
        station_ids = sorted(diag["station_id"].dropna().astype(str).unique())
        station_display_names = _station_display_names(csv_path, station_ids)
        station_sigma_meta = _station_metadata_uncertainty_pct(csv_path, station_ids)
        station_color_map = _station_color_map(
            station_ids,
            observable=observable,
            station_color_config=station_color_config,
        )
        entries: list[tuple[str, str]] = []
        for station_id in station_ids:
            display_name = station_display_names.get(station_id, station_id)
            sigma_meta = station_sigma_meta.get(station_id.strip().lower())
            if sigma_meta is not None:
                label = f"{display_name} (\u03c3={sigma_meta:.0f}%)"
            else:
                label = display_name
            entries.append((label, station_color_map[station_id]))
        return entries

    if observable in _FRACTION_DISPLAY_LABELS:
        return [(_FRACTION_DISPLAY_LABELS[observable], _FRACTION_MISMATCH_COLORS[observable])]

    return []


def _collect_marker_legend_entries(
    csv_paths: list[Path],
    *,
    station_color_config: dict[str, dict[str, str]] | None = None,
) -> list[tuple[str, str]]:
    entries: dict[str, str] = {}
    for csv_path in csv_paths:
        observable = _observable_from_csv_path(csv_path)
        config = station_color_config if station_color_config is not None else _station_color_config_for_csv(csv_path)
        for label, color in _marker_legend_entries_for_csv(csv_path, observable, station_color_config=config):
            entries.setdefault(label, color)
    return list(entries.items())


def _draw_sigma_strip(ax, entries: list[tuple[str, str]], *, fontsize: float) -> None:
    if not entries:
        return
    handles = [_marker_handle(color, size=4.8, markeredgewidth=0.8) for color, _label in entries]
    labels = [label for _color, label in entries]
    inside_panel = len(labels) <= 3
    loc = "upper right" if inside_panel else "lower right"
    bbox_to_anchor = (0.99, 0.93) if inside_panel else (1.0, 1.05)
    ncol = 1 if inside_panel else len(labels)
    legend = ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.8,
        labelspacing=0.2,
        borderpad=0.0,
        borderaxespad=0.0,
    )
    legend._legend_box.align = "right"


def _shared_station_sigma_groups(diag: pd.DataFrame) -> list[tuple[float, list[str]]]:
    if diag.empty or "station_id" not in diag.columns or "sigma" not in diag.columns:
        return []

    sigma_by_station: list[tuple[str, float]] = []
    for station_id in sorted(diag["station_id"].dropna().astype(str).unique()):
        station_mask = diag["station_id"].astype(str) == station_id
        sigma_series = pd.to_numeric(diag.loc[station_mask, "sigma"], errors="coerce").dropna()
        if sigma_series.empty:
            continue
        sigma_by_station.append((station_id, float(sigma_series.iloc[0])))

    groups: list[list[tuple[str, float]]] = []
    for station_id, sigma_val in sigma_by_station:
        placed = False
        for group in groups:
            if np.isclose(sigma_val, group[0][1], rtol=1e-9, atol=1e-12):
                group.append((station_id, sigma_val))
                placed = True
                break
        if not placed:
            groups.append([(station_id, sigma_val)])

    return [
        (float(group[0][1]), [station_id for station_id, _sigma_val in group])
        for group in groups
        if len(group) > 1
    ]


def _draw_alternating_sigma_line(
    ax,
    x: float,
    y_min: float,
    y_max: float,
    colors: list[str],
    *,
    lw: float = 1.0,
    alpha: float = 0.9,
    zorder: int = 2,
) -> None:
    if not colors or not np.isfinite(x) or y_max <= y_min:
        return

    span = float(y_max - y_min)
    n_segments = max(8, min(18, int(round(span * 2.5))))
    cycle = span / max(1, n_segments)
    dash_len = cycle * 0.62
    gap_len = cycle - dash_len

    y = float(y_min)
    color_idx = 0
    while y < y_max:
        y_end = min(y + dash_len, y_max)
        ax.plot(
            [x, x],
            [y, y_end],
            color=colors[color_idx % len(colors)],
            lw=lw,
            alpha=alpha,
            solid_capstyle="butt",
            zorder=zorder,
        )
        y = y_end + gap_len
        color_idx += 1


def _resample_legend_artists():
    from matplotlib.legend_handler import HandlerTuple

    legend_fill = "#bdbdbd"
    redraw_handle = (
        _marker_handle(legend_fill, size=5.2, markeredgewidth=0.9, edgecolor="#000000"),
        _marker_handle(None, size=7.6, markeredgewidth=0.8, edgecolor="#000000"),
        _marker_handle(None, size=10.0, markeredgewidth=0.7, edgecolor="#000000"),
    )
    labels = ["redrawn source member (extra rings = repeated draws)"]
    return [redraw_handle], labels, {tuple: HandlerTuple(ndivide=1)}


def _figure_legend_spec(
    csv_paths: list[Path],
    *,
    station_color_config: dict[str, dict[str, str]] | None = None,
):
    handles = []
    labels = []
    for label, color in _collect_marker_legend_entries(csv_paths, station_color_config=station_color_config):
        handles.append(_marker_handle(color, size=5.8, markeredgewidth=0.9))
        labels.append(label)
    resample_handles, resample_labels, handler_map = _resample_legend_artists()
    handles.extend(resample_handles)
    labels.extend(resample_labels)
    return handles, labels, handler_map


def _best_figure_legend_ncol(
    fig,
    handles: list[object],
    labels: list[str],
    *,
    handler_map: dict | None = None,
    max_width_frac: float = 0.96,
    **legend_kwargs,
) -> int:
    """Prefer a single legend row and wrap only when it would overflow the figure."""
    if not labels:
        return 1

    for ncol in range(len(labels), 0, -1):
        legend = fig.legend(
            handles,
            labels,
            handler_map=handler_map,
            ncol=ncol,
            **legend_kwargs,
        )
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        fig_bbox = fig.get_window_extent(renderer=renderer)
        legend_bbox = legend.get_window_extent(renderer=renderer)
        legend.remove()
        if legend_bbox.width <= fig_bbox.width * max_width_frac:
            return ncol

    return 1


def _scale_axes_group(
    axes: list[object],
    *,
    width_scale: float,
    height_scale: float,
    top_anchor: float | None = None,
) -> None:
    """Scale a group of axes while preserving their relative layout."""
    if not axes:
        return

    positions = [ax.get_position().frozen() for ax in axes]
    left = min(pos.x0 for pos in positions)
    right = max(pos.x1 for pos in positions)
    bottom = min(pos.y0 for pos in positions)
    top = max(pos.y1 for pos in positions)
    group_width = right - left
    group_height = top - bottom
    if group_width <= 0.0 or group_height <= 0.0:
        return

    scaled_group_width = group_width * width_scale
    scaled_group_height = group_height * height_scale
    new_left = left + 0.5 * (group_width - scaled_group_width)
    if top_anchor is None:
        new_bottom = bottom + 0.5 * (group_height - scaled_group_height)
    else:
        new_bottom = top_anchor - scaled_group_height

    for ax, pos in zip(axes, positions):
        rel_x0 = (pos.x0 - left) / group_width
        rel_y0 = (pos.y0 - bottom) / group_height
        new_x0 = new_left + rel_x0 * scaled_group_width
        new_y0 = new_bottom + rel_y0 * scaled_group_height
        ax.set_position(
            [
                new_x0,
                new_y0,
                pos.width * width_scale,
                pos.height * height_scale,
            ]
        )


def _ordered_weights_df(df: pd.DataFrame) -> pd.DataFrame:
    if "member_id" in df.columns:
        ordered_df = df.copy()
        ordered_df["_member_sort"] = ordered_df["member_id"].astype(str)
        ordered_df = ordered_df.sort_values(["weight", "_member_sort"], ascending=[False, True]).reset_index(drop=True)
        return ordered_df.drop(columns="_member_sort")
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


def _finite_abs_values(values) -> list[float]:
    series = pd.to_numeric(values, errors="coerce")
    if isinstance(series, pd.Series):
        data = series.to_numpy(dtype=float)
    else:
        data = np.asarray(series, dtype=float)
    return [float(abs(v)) for v in data if np.isfinite(v)]


def _expand_xlim(xlim: tuple[float, float], *, pad_fraction: float = _AXIS_EDGE_PAD_FRACTION) -> tuple[float, float]:
    left, right = xlim
    if not np.isfinite(left) or not np.isfinite(right):
        return xlim
    span = right - left
    if span <= 0.0:
        scale = max(abs(left), abs(right), 1.0)
        pad = scale * pad_fraction
    else:
        pad = span * pad_fraction
    return left - pad, right + pad


def _residual_extent_components_for_event(
    csv_path: Path,
    df: pd.DataFrame,
    observable: str | None,
) -> tuple[list[float], list[float]]:
    residual_extents: list[float] = []
    sigma_extents: list[float] = []
    if observable in {"station_hs", "station_swe"}:
        diag_path = _station_diagnostics_path(csv_path, observable)
        diag = pd.read_csv(diag_path) if diag_path is not None and diag_path.is_file() else pd.DataFrame()
        if not diag.empty:
            if "residual" in diag.columns:
                residual_extents.extend(_finite_abs_values(diag["residual"]))
            if "sigma" in diag.columns:
                sigma_extents.extend(_finite_abs_values(diag["sigma"]))
    else:
        if "residual" in df.columns:
            residual_extents.extend(_finite_abs_values(df["residual"]))
        if "sigma" in df.columns:
            sigma_extents.extend(_finite_abs_values(df["sigma"]))
    return residual_extents, sigma_extents


def _nice_axis_extent(extent: float) -> float:
    if extent <= 0.0:
        return extent
    if extent < 0.5:
        step = 0.05
    elif extent < 1.0:
        step = 0.1
    elif extent < 2.0:
        step = 0.25
    elif extent < 5.0:
        step = 0.5
    elif extent < 10.0:
        step = 1.0
    elif extent < 50.0:
        step = 5.0
    else:
        order = 10 ** math.floor(math.log10(extent))
        step = order / 2.0
    return math.ceil(extent / step) * step


def _robust_shared_extent(
    residual_extents: list[float],
    sigma_extents: list[float],
    *,
    percentile: float = _OVERVIEW_SHARED_RESIDUAL_PERCENTILE,
) -> float | None:
    if residual_extents:
        combined = np.asarray(residual_extents + sigma_extents, dtype=float)
        extent = max(float(np.percentile(combined, percentile)), max(residual_extents))
    elif sigma_extents:
        extent = max(sigma_extents)
    else:
        return None

    if sigma_extents and not residual_extents:
        extent = max(extent, max(sigma_extents))
    if extent <= 0.0:
        return None
    return _nice_axis_extent(extent)


def _overview_residual_xlims(event_specs: list[dict[str, object]]) -> dict[str | None, tuple[float, float]]:
    residual_by_observable: dict[str | None, list[float]] = {}
    sigma_by_observable: dict[str | None, list[float]] = {}
    for spec in event_specs:
        observable = spec["observable"]  # type: ignore[index]
        residual_extents, sigma_extents = _residual_extent_components_for_event(
            spec["csv_path"],  # type: ignore[arg-type]
            spec["df"],  # type: ignore[arg-type]
            observable,  # type: ignore[arg-type]
        )
        if residual_extents:
            residual_by_observable.setdefault(observable, []).extend(residual_extents)
        if sigma_extents:
            sigma_by_observable.setdefault(observable, []).extend(sigma_extents)

    return {
        observable: (-extent, extent)
        for observable in sorted(set(residual_by_observable) | set(sigma_by_observable), key=str)
        if (
            extent := _robust_shared_extent(
                residual_by_observable.get(observable, []),
                sigma_by_observable.get(observable, []),
            )
        )
        is not None
    }


def _draw_weights_event(
    fig,
    ax0,
    ax1,
    *,
    csv_path: Path,
    df: pd.DataFrame,
    title: str,
    subtitle: str | None,
    observable: str | None,
    title_mode: str = "figure",
    font_scale: float = 1.0,
    show_metrics_label: bool = True,
    show_metrics_threshold: bool = True,
    show_left_ylabel: bool = True,
    show_right_ylabel: bool = True,
    ring_step_scale: float = 1.0,
    ring_line_scale: float = 1.0,
    marker_scale: float = 1.0,
    font_size_bump: float = 0.0,
    axes_title_y: float = 1.18,
    figure_title_y: float = 0.972,
    residual_xlim: tuple[float, float] | None = None,
    y_ticks: list[int] | None = None,
    station_color_config: dict[str, dict[str, str]] | None = None,
) -> None:
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator

    fs_title = _FS_TITLE * font_scale + font_size_bump
    fs_axis = _FS_AXIS * font_scale + font_size_bump
    fs_tick = _FS_TICK * font_scale + font_size_bump
    fs_note = _FS_NOTE * font_scale + font_size_bump

    ordered_df = _ordered_weights_df(df)

    w = np.asarray(ordered_df["weight"], dtype=float)
    n = w.size
    ess = effective_sample_size(w)
    resample_manifest = _read_resample_manifest(csv_path)
    member_ranks = {
        str(member_id): idx + 1
        for idx, member_id in enumerate(ordered_df.get("member_id", pd.Series(range(1, n + 1))))
    }
    resample_counts, _resampling_skipped = _resample_source_member_counts(csv_path)
    selected_counts = np.asarray(
        [int(resample_counts.get(str(member_id), 0)) for member_id in ordered_df.get("member_id", pd.Series(range(1, n + 1)))],
        dtype=int,
    )

    y_rank = np.arange(1, n + 1, dtype=float)
    weight_marker_size = 13.0 * marker_scale
    mismatch_marker_size = 20.0 * marker_scale
    weight_marker_color = "#b8bec7"
    member_ticks = y_ticks if y_ticks is not None else _member_ticks(n)
    ax0.scatter(
        w,
        y_rank,
        s=weight_marker_size,
        facecolors=weight_marker_color,
        edgecolors=weight_marker_color,
        linewidths=0.8,
        zorder=4,
    )
    _draw_resample_rings(
        ax0,
        w,
        y_rank,
        selected_counts,
        base_size=weight_marker_size,
        ring_step=11.0 * ring_step_scale,
        line_scale=ring_line_scale,
    )
    ax0.set_xlabel("weight", fontsize=fs_axis)
    ax0.set_ylabel("sorted member" if show_left_ylabel else "", fontsize=fs_axis)
    _apply_grid(ax0)
    ax0.set_xlim(*_expand_xlim((0.0, 1.0)))
    ax0.set_yticks(member_ticks)
    ax0.set_ylim(n + 0.5, 0.5)
    ax0.xaxis.set_major_locator(MultipleLocator(0.1))
    ax0.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax0.yaxis.set_minor_locator(NullLocator())
    ax0.tick_params(axis="both", labelsize=fs_tick)
    threshold = resample_manifest.get("ess_threshold")
    metrics_label = f"ESS = {ess:.1f}"
    if show_metrics_threshold and threshold is not None:
        metrics_label = f"{metrics_label} (threshold={float(threshold):.1f})"
    if show_metrics_label:
        ax0.text(
            0.97,
            0.04,
            metrics_label,
            transform=ax0.transAxes,
            ha="right",
            va="bottom",
            fontsize=fs_note,
            color="#000000",
            bbox={"boxstyle": "round,pad=0.16", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
            zorder=8,
        )

    sigma_strip_entries: list[tuple[str, str]] = []
    residual_axis_values: list[float] = [0.0]
    ax1.axvline(0.0, color="black", lw=1.0, zorder=3)
    ax1.set_ylabel("sorted member" if show_right_ylabel else "", fontsize=fs_axis)
    ax1.set_yticks(member_ticks)
    ax1.set_ylim(n + 0.5, 0.5)
    ax1.yaxis.set_minor_locator(NullLocator())
    if observable in {"station_hs", "station_swe"}:
        diag_path = _station_diagnostics_path(csv_path, observable)
        diag = pd.read_csv(diag_path) if diag_path is not None and diag_path.is_file() else pd.DataFrame()
        shared_sigma_groups = _shared_station_sigma_groups(diag)
        shared_sigma_station_ids = {
            station_id
            for _sigma_val, station_ids in shared_sigma_groups
            for station_id in station_ids
        }
        station_ids = (
            sorted(diag["station_id"].dropna().astype(str).unique())
            if not diag.empty and "station_id" in diag.columns
            else []
        )
        config = station_color_config if station_color_config is not None else _station_color_config_for_csv(csv_path)
        station_color_map = _station_color_map(
            station_ids,
            observable=observable,
            station_color_config=config,
        )
        for station_id in station_ids:
            station_mask = diag["station_id"].astype(str) == station_id
            sdf = diag.loc[station_mask].copy()
            if sdf.empty or "member_id" not in sdf.columns:
                continue
            sdf["member_rank"] = sdf["member_id"].astype(str).map(member_ranks)
            sdf["residual_num"] = pd.to_numeric(sdf.get("residual"), errors="coerce")
            sdf = sdf.loc[sdf["member_rank"].notna() & sdf["residual_num"].notna()].copy()
            if sdf.empty:
                continue
            sdf["member_rank"] = sdf["member_rank"].astype(float)
            sdf = sdf.sort_values("member_rank")
            color = station_color_map[station_id]
            y = sdf["member_rank"].to_numpy(dtype=float)
            station_draw_counts = sdf["member_id"].astype(str).map(lambda member_id: int(resample_counts.get(member_id, 0))).to_numpy(dtype=int)
            sigma_series = pd.to_numeric(sdf.get("sigma"), errors="coerce").dropna()
            ax1.scatter(
                sdf["residual_num"].to_numpy(dtype=float),
                y,
                facecolors=color,
                edgecolors=color,
                linewidths=0.9,
                s=mismatch_marker_size,
                zorder=4,
            )
            residual_axis_values.extend(sdf["residual_num"].to_numpy(dtype=float).tolist())
            _draw_resample_rings(
                ax1,
                sdf["residual_num"].to_numpy(dtype=float),
                y,
                station_draw_counts,
                base_size=mismatch_marker_size,
                ring_step=11.0 * ring_step_scale,
                line_scale=ring_line_scale,
            )
            sigma_val: float | None = None
            if not sigma_series.empty:
                sigma_val = float(sigma_series.iloc[0])
                sigma_strip_entries.append((color, f"\u03c3={sigma_val:.2f}"))
                residual_axis_values.extend([-sigma_val, sigma_val])
            if not sigma_series.empty and station_id not in shared_sigma_station_ids:
                ax1.axvline(-sigma_val, color=color, lw=1.0, ls="-", alpha=0.9, zorder=2)
                ax1.axvline(sigma_val, color=color, lw=1.0, ls="-", alpha=0.9, zorder=2)
        y_min = 0.5
        y_max = n + 0.5
        for sigma_val, sigma_station_ids in shared_sigma_groups:
            colors = [station_color_map[station_id] for station_id in sigma_station_ids if station_id in station_color_map]
            _draw_alternating_sigma_line(ax1, -sigma_val, y_min, y_max, colors, lw=1.0, alpha=0.9, zorder=2)
            _draw_alternating_sigma_line(ax1, sigma_val, y_min, y_max, colors, lw=1.0, alpha=0.9, zorder=2)
        ax1.set_xlabel(_station_axis_label(observable), fontsize=fs_axis)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    else:
        residual = pd.to_numeric(ordered_df.get("residual"), errors="coerce")
        frac_color = _FRACTION_MISMATCH_COLORS.get(observable, da_variable_line_color("station_hs"))
        valid = residual.notna()
        if valid.any():
            resid_valid = residual.loc[valid]
            y = np.flatnonzero(valid.to_numpy()) + 1
            resid_selected_counts = selected_counts[valid.to_numpy()]
            ax1.scatter(
                resid_valid.to_numpy(dtype=float),
                y.astype(float),
                facecolors=frac_color,
                edgecolors=frac_color,
                linewidths=0.9,
                s=mismatch_marker_size,
                zorder=4,
            )
            residual_axis_values.extend(resid_valid.to_numpy(dtype=float).tolist())
            _draw_resample_rings(
                ax1,
                resid_valid.to_numpy(dtype=float),
                y.astype(float),
                resid_selected_counts,
                base_size=mismatch_marker_size,
                ring_step=11.0 * ring_step_scale,
                line_scale=ring_line_scale,
            )
        sigma = pd.to_numeric(ordered_df.get("sigma"), errors="coerce") if "sigma" in ordered_df.columns else pd.Series(dtype=float)
        if not sigma.empty and pd.notna(sigma.iloc[0]):
            sigma_val = float(sigma.iloc[0])
            ax1.axvline(-sigma_val, color=frac_color, lw=1.0, ls="-", alpha=0.9, zorder=2)
            ax1.axvline(sigma_val, color=frac_color, lw=1.0, ls="-", alpha=0.9, zorder=2)
            sigma_strip_entries.append((frac_color, f"\u03c3={sigma_val:.2f}"))
            residual_axis_values.extend([-sigma_val, sigma_val])
        ax1.set_xlabel(_fraction_axis_label(observable), fontsize=fs_axis)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        if observable == "wet_snow_line":
            _draw_wsl_unavailable_overlay(ax1, ordered_df, fontsize=fs_note)
    _draw_sigma_strip(ax1, sigma_strip_entries, fontsize=fs_note)
    if residual_xlim is not None:
        ax1.set_xlim(*_expand_xlim(residual_xlim))
    elif residual_axis_values:
        ax1.set_xlim(*_expand_xlim((min(residual_axis_values), max(residual_axis_values))))
    _apply_grid(ax1)
    ax1.tick_params(axis="both", labelsize=fs_tick)

    header = title
    if subtitle:
        if title_mode == "axes":
            header = f"{title}\n{subtitle}" if title else subtitle
        else:
            header = f"{title} - {subtitle}" if title else subtitle
    if title_mode == "axes":
        header = _format_axes_subplot_title(header)
    if header:
        if title_mode == "figure":
            fig.text(0.11, figure_title_y, header, ha="left", va="top", fontsize=fs_title, color="#000000")
        else:
            ax0.text(
                0.0,
                axes_title_y,
                header,
                transform=ax0.transAxes,
                ha="left",
                va="bottom",
                fontsize=fs_title,
                color="#000000",
            )


def _plot(
    csv_path: Path,
    df: pd.DataFrame,
    title: str,
    subtitle: str | None,
    *,
    observable: str | None,
    backend: str = "Agg",
):
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig = plt.figure(figsize=_WEIGHTS_FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=_WEIGHTS_PANEL_WIDTH_RATIOS)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    station_color_config = _station_color_config_for_csv(csv_path)
    _draw_weights_event(
        fig,
        ax0,
        ax1,
        csv_path=csv_path,
        df=df,
        title=title,
        subtitle=subtitle,
        observable=observable,
        title_mode="figure",
        font_scale=1.0,
        figure_title_y=_STANDALONE_TITLE_Y,
        station_color_config=station_color_config,
    )
    ax0.set_xticks(_WEIGHT_AXIS_TICKS)
    ax0.set_xticklabels(_WEIGHT_AXIS_TICK_LABELS)
    ax0.xaxis.set_minor_locator(NullLocator())

    bottom_margin = 0.30
    right_margin = 0.965
    top_margin = 0.78
    fig.subplots_adjust(left=0.095, right=right_margin, top=top_margin, bottom=bottom_margin, wspace=0.30)
    _scale_axes_group(
        [ax0, ax1],
        width_scale=_STANDALONE_PLOT_WIDTH_SCALE,
        height_scale=_STANDALONE_PLOT_HEIGHT_SCALE,
        top_anchor=_STANDALONE_PLOT_TOP,
    )
    legend_handles, legend_labels, handler_map = _figure_legend_spec(
        [csv_path],
        station_color_config=station_color_config,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, _STANDALONE_LEGEND_Y),
        ncol=min(3, len(legend_labels)),
        frameon=False,
        fontsize=6.8,
        handletextpad=0.4,
        columnspacing=1.1,
        borderaxespad=0.0,
    )
    force_figure_text_black(fig, [ax0, ax1])
    return fig


def _setup_id_from_dir(setup_dir: Path) -> str:
    name = setup_dir.name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def _setup_weights_csvs(setup_dir: Path) -> list[Path]:
    files: list[tuple[int, str, Path]] = []
    for step_dir in list_steps_sorted(setup_dir):
        assim_dir = step_dir / "assim"
        if not assim_dir.is_dir():
            continue
        for csv_path in sorted(assim_dir.glob("weights_*_*.csv")):
            idx = _step_da_index_from_path(csv_path)
            files.append((idx if idx is not None else 10_000, csv_path.name, csv_path))
    return [csv_path for _idx, _name, csv_path in sorted(files, key=lambda item: (item[0], item[1]))]


def _default_setup_weights_overview_output(setup_dir: Path) -> Path:
    out_dir = project_plot_assim_weights_dir(Path(setup_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"setup_weights_overview_{_setup_id_from_dir(Path(setup_dir))}.png"


def _setup_weights_overview_page_output(output_path: Path, page_index: int) -> Path:
    output_path = Path(output_path)
    if page_index <= 0:
        return output_path
    return output_path.with_name(f"{output_path.stem}_page_{page_index + 1:02d}{output_path.suffix}")


def _remove_stale_setup_weights_overview_pages(output_path: Path, keep_paths: list[Path]) -> None:
    output_path = Path(output_path)
    keep_set = {Path(path) for path in keep_paths}
    pattern = re.compile(rf"^{re.escape(output_path.stem)}_page_(\d+){re.escape(output_path.suffix)}$")
    for candidate in output_path.parent.glob(f"{output_path.stem}_page_*{output_path.suffix}"):
        if candidate in keep_set:
            continue
        if pattern.match(candidate.name):
            candidate.unlink(missing_ok=True)


def _build_setup_weights_overview_page(
    page_specs: list[dict[str, object]],
    *,
    all_csv_paths: list[Path],
    residual_xlims: dict[str | None, tuple[float, float]],
    ensemble_size: int,
    ess_threshold: float | None,
    page_index: int,
    total_pages: int,
    layout_rows: int | None = None,
    show_figure_title: bool = True,
    station_color_config: dict[str, dict[str, str]] | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    n_events = len(page_specs)
    n_cols = 2
    n_rows = int(math.ceil(n_events / n_cols))
    page_rows = max(n_rows, int(layout_rows or n_rows))
    fig = plt.figure(figsize=(7.2876875, _COMPOSITE_ROW_HEIGHT * page_rows))
    outer = fig.add_gridspec(
        page_rows,
        n_cols,
        left=0.06,
        right=0.99,
        top=0.91,
        bottom=0.12,
        wspace=0.0,
        hspace=_OVERVIEW_ROW_HSPACE,
    )

    axes_for_black: list[object] = []
    font_scale = 0.68
    for idx, spec in enumerate(page_specs):
        csv_path = Path(spec["csv_path"])
        observable = spec["observable"]
        df = spec["df"]
        row = idx // n_cols
        col = idx % n_cols
        sub = outer[row, col].subgridspec(
            2,
            3,
            height_ratios=[1.0, 0.045],
            width_ratios=[*_WEIGHTS_PANEL_WIDTH_RATIOS, _OVERVIEW_PAIR_SPACER_RATIO],
            wspace=_OVERVIEW_PAIR_WSPACE,
            hspace=0.0,
        )
        ax0 = fig.add_subplot(sub[0, 0])
        ax1 = fig.add_subplot(sub[0, 1])
        axes_for_black.extend([ax0, ax1])
        subtitle = _step_date_label_from_path(csv_path)
        base_title = _compact_subplot_title(_title_from_path(csv_path))
        if subtitle:
            title = f"{subtitle} - {base_title}"
        else:
            title = base_title
        _draw_weights_event(
            fig,
            ax0,
            ax1,
            csv_path=csv_path,
            df=df,
            title=title,
            subtitle=None,
            observable=observable,
            title_mode="axes",
            font_scale=font_scale,
            show_metrics_label=True,
            show_metrics_threshold=False,
            show_left_ylabel=(col == 0),
            show_right_ylabel=False,
            ring_step_scale=0.72,
            ring_line_scale=0.72,
            marker_scale=0.8,
            font_size_bump=1.0,
            axes_title_y=1.055,
            residual_xlim=residual_xlims.get(observable),
            y_ticks=_member_ticks(len(df.index)),
            station_color_config=station_color_config,
        )
        ax0.set_xticks(_WEIGHT_AXIS_TICKS)
        ax0.set_xticklabels(_WEIGHT_AXIS_TICK_LABELS)
        ax0.xaxis.set_minor_locator(NullLocator())
        if col != 0:
            ax0.tick_params(axis="y", labelleft=False)
        ax1.tick_params(axis="y", labelleft=False)

    summary = f"ensemble size = {ensemble_size}"
    if ess_threshold is not None:
        summary = f"{summary}, ESS threshold = {float(ess_threshold):.1f}"
    if total_pages > 1:
        summary = f"{summary}, page {page_index + 1}/{total_pages}"
    if show_figure_title:
        fig.text(
            0.06,
            0.974,
            f"Data assimilation weights ({summary})",
            va="top",
            ha="left",
            fontsize=8.6,
            color="#000000",
        )
    legend_handles, legend_labels, handler_map = _figure_legend_spec(
        all_csv_paths,
        station_color_config=station_color_config,
    )
    legend_kwargs = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.052),
        frameon=False,
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    fig.legend(
        legend_handles,
        legend_labels,
        handler_map=handler_map,
        ncol=_best_figure_legend_ncol(
            fig,
            legend_handles,
            legend_labels,
            handler_map=handler_map,
            **legend_kwargs,
        ),
        **legend_kwargs,
    )
    force_figure_text_black(fig, axes_for_black)
    return fig


def plot_setup_weights_overview(setup_dir: Path, *, backend: str = "Agg") -> Path:
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)

    setup_dir = Path(setup_dir)
    csv_paths = _setup_weights_csvs(setup_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No weights_*_*.csv found under steps in {setup_dir}")
    station_color_config = _load_weights_station_color_config(setup_dir)

    event_specs = [
        {
            "csv_path": csv_path,
            "observable": _observable_from_csv_path(csv_path),
            "df": _load_weights(csv_path),
        }
        for csv_path in csv_paths
    ]
    residual_xlims = _overview_residual_xlims(event_specs)

    n_cols = 2
    n_rows = int(math.ceil(len(csv_paths) / n_cols))
    first_df = event_specs[0]["df"]
    ensemble_size = len(first_df)
    first_manifest = _read_resample_manifest(csv_paths[0])
    ess_threshold = first_manifest.get("ess_threshold")
    out = _default_setup_weights_overview_output(setup_dir)
    rows_per_page = min(_OVERVIEW_MAX_ROWS_PER_PAGE, max(1, n_rows))
    events_per_page = rows_per_page * n_cols
    page_specs = [event_specs[start : start + events_per_page] for start in range(0, len(event_specs), events_per_page)]
    output_paths: list[Path] = []
    paper_output_paths: list[Path] = []
    for page_index, page in enumerate(page_specs):
        fig = _build_setup_weights_overview_page(
            page,
            all_csv_paths=csv_paths,
            residual_xlims=residual_xlims,
            ensemble_size=ensemble_size,
            ess_threshold=ess_threshold,
            page_index=page_index,
            total_pages=len(page_specs),
            layout_rows=rows_per_page if len(page_specs) > 1 else None,
            station_color_config=station_color_config,
        )
        page_out = _setup_weights_overview_page_output(out, page_index)
        save_figure_png(fig, page_out, dpi=600, bbox_inches="tight", pad_inches=0.04)
        output_paths.append(page_out)
        paper_fig = _build_setup_weights_overview_page(
            page,
            all_csv_paths=csv_paths,
            residual_xlims=residual_xlims,
            ensemble_size=ensemble_size,
            ess_threshold=ess_threshold,
            page_index=page_index,
            total_pages=len(page_specs),
            layout_rows=rows_per_page if len(page_specs) > 1 else None,
            show_figure_title=False,
            station_color_config=station_color_config,
        )
        paper_page_out = project_paper_output_path(setup_dir, page_out)
        paper_page_out.parent.mkdir(parents=True, exist_ok=True)
        save_figure_png(paper_fig, paper_page_out, dpi=600, bbox_inches="tight", pad_inches=0.04)
        paper_output_paths.append(paper_page_out)
    _remove_stale_setup_weights_overview_pages(out, output_paths)
    _remove_stale_setup_weights_overview_pages(project_paper_output_path(setup_dir, out), paper_output_paths)
    return out


def _default_output_path(csv_path: Path) -> Path:
    """Return default output PNG path for a weights CSV.

    If the CSV lives under <project>/step_XX_*/assim/, write to
    <project>/results/plots/assim/weights/DA_XX_weights.png. Otherwise, fall back
    to csv_path.with_suffix('.png').
    """
    csv_path = csv_path.resolve()
    # Expect .../project_YYYY-YYYY/steps/step_XX_*/assim/weights_*.csv
    if csv_path.parent.name == "assim":
        step_dir = csv_path.parent.parent
        try:
            project_dir = infer_project_dir(step_dir)
        except Exception:
            return csv_path.with_suffix(".png")
        if step_dir.name.startswith("step_"):
            out_dir = project_plot_assim_weights_dir(project_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            da_index = _step_da_index_from_path(csv_path)
            if da_index is not None:
                return out_dir / f"DA_{da_index:02d}_weights.png"
            parts = step_dir.name.split("_")
            step_token = "_".join(parts[:2]) if len(parts) >= 2 else step_dir.name
            return out_dir / f"{step_token}_weights.png"
    # Fallback: same dir as CSV
    return csv_path.with_suffix(".png")


def _step_da_index_from_path(csv_path: Path) -> int | None:
    """Return the 1-based DA event index inferred from the CSV path, if available."""
    try:
        csv_path = csv_path.resolve()
    except Exception:
        return None

    stem = csv_path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    ds = parts[-1]
    if len(ds) != 8 or not ds.isdigit():
        return None
    try:
        date_val = pd.to_datetime(f"{ds[0:4]}-{ds[4:6]}-{ds[6:8]}").date()
    except Exception:
        return None

    if csv_path.parent.name != "assim":
        return None
    step_dir = csv_path.parent.parent
    try:
        project_dir = infer_project_dir(step_dir)
        events = load_assimilation_events(project_dir)
        for idx, event in enumerate(events, start=1):
            if event.date == date_val:
                return idx
    except Exception:
        return None
    return None


def _step_date_label_from_path(csv_path: Path) -> str | None:
    """Return "DA# - YYYY-MM-DD" (or fallback step label) inferred from the CSV path."""
    try:
        csv_path = csv_path.resolve()
    except Exception:
        return None

    stem = csv_path.stem
    # Accept any weights_..._YYYYMMDD pattern (SCF or wet_snow)
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    ds = parts[-1]
    if len(ds) != 8 or not ds.isdigit():
        return None
    date_str = f"{ds[0:4]}-{ds[4:6]}-{ds[6:8]}"
    try:
        pd.to_datetime(date_str).date()
    except Exception:
        return None

    if csv_path.parent.name == "assim":
        step_dir = csv_path.parent.parent
        try:
            idx = _step_da_index_from_path(csv_path)
            if idx is not None:
                return f"DA {idx} - {date_str}"
        except Exception:
            pass

        name = step_dir.name
        if name.startswith("step_"):
            tail = name[len("step_") :]
            token = tail.split("_", 1)[0] if tail else ""
            if token:
                label = f"Step {token}"
            else:
                label = name
            return f"{label} - {date_str}"
    return None


def _title_from_path(csv_path: Path) -> str:
    return weight_plot_title_from_csv_path(csv_path)


def _compact_subplot_title(title: str) -> str:
    compact = str(title or "").strip()
    for suffix in (" data assimilation weights", " assimilation weights"):
        if compact.lower().endswith(suffix):
            return compact[: -len(suffix)].rstrip()
    return compact


def _format_axes_subplot_title(title: str) -> str:
    compact = str(title or "").strip()
    if not compact:
        return compact
    match = re.match(r"^(DA \d+)(.*)$", compact)
    if not match:
        return compact
    prefix, suffix = match.groups()
    prefix_math = prefix.replace(" ", r"\ ")
    return rf"$\mathbf{{{prefix_math}}}$" + suffix


def plot_weights_for_csv(
    csv_path: Path,
    *,
    title: str = "Assimilation Weights",
    subtitle: str | None = None,
    backend: str = "Agg",
) -> Path:
    """Library API: plot weights for a single CSV and return PNG path."""
    df = _load_weights(csv_path)
    # If caller uses the default title and no subtitle, derive a compact
    # label from the path: "Step <number> - <YYYY-MM-DD>".
    if title == "Assimilation Weights":
        title = _title_from_path(csv_path)
    if subtitle is None:
        label = _step_date_label_from_path(csv_path)
        if label:
            subtitle = label
    fig = _plot(
        csv_path,
        df,
        title=title,
        subtitle=subtitle,
        observable=_observable_from_csv_path(csv_path),
        backend=backend,
    )
    out = _default_output_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, out, bbox_inches="tight", pad_inches=_STANDALONE_SAVE_PAD_INCHES)
    return out


def cli_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="oa-da-plot-weights", description="Plot assimilation weights and residuals")
    p.add_argument("csv", type=Path, help="Path to weights_<observable>_YYYYMMDD.csv")
    p.add_argument("--output", type=Path, help="Output PNG path (default: same dir as CSV)")
    p.add_argument("--title", default="Assimilation Weights", help="Plot title")
    p.add_argument("--subtitle", default="", help="Plot subtitle")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--backend", default="Agg", help="Matplotlib backend (Agg, SVG, module://mplcairo.Agg)")
    args = p.parse_args(argv)

    # Avoid enqueue for short-lived CLIs so messages flush before exit
    configure_cli_logger(args.log_level, enqueue=False)

    csv_path = Path(args.csv)
    logger.info("Reading weights CSV: {}", csv_path)
    try:
        df = _load_weights(csv_path)
    except Exception as e:
        logger.error(f"Failed reading weights CSV: {e}")
        return 1

    # Basic stats
    try:
        n = len(df)
        w = np.asarray(df["weight"], dtype=float)
        ess = effective_sample_size(w)
        sigma = df.get("sigma", pd.Series([np.nan])).iloc[0]
        logger.info("Rows={}  ESS={:.1f}  N={}  sigma={}", n, ess, w.size, (f"{sigma:.3f}" if pd.notna(sigma) else "NA"))
    except Exception:
        pass

    try:
        # Automatically derive a compact "Step <number> - <YYYY-MM-DD>" label
        # when the caller uses the default title and no explicit subtitle.
        subtitle = (args.subtitle or None)
        title = args.title
        if title == "Assimilation Weights":
            title = _title_from_path(csv_path)
        if not subtitle:
            label = _step_date_label_from_path(csv_path)
            if label:
                subtitle = label
        fig = _plot(
            csv_path,
            df,
            title=title,
            subtitle=subtitle,
            observable=_observable_from_csv_path(csv_path),
            backend=args.backend,
        )
    except ModuleNotFoundError:
        logger.error("matplotlib is required to plot. Install it in your environment.")
        return 2
    except Exception as e:
        logger.error(f"Plotting failed: {e}")
        return 3

    out = Path(args.output) if args.output else _default_output_path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving plot to: {}", out)
    try:
        save_figure_png(fig, out, bbox_inches="tight", pad_inches=_STANDALONE_SAVE_PAD_INCHES)
    except Exception as e:
        logger.error(f"Saving PNG failed: {e}")
        return 4
    logger.info("Wrote plot: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
