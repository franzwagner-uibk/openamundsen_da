"""openamundsen_da.methods.pf.plot_weights

Plot per-date assimilation weights and residual summaries.

Inputs
- weights CSV produced by one assimilation workflow with columns:
  member_id, residual, sigma, log_weight, weight

Outputs
- A PNG saved next to the CSV (or --output) with two panels:
  A) sorted normalized weights with ESS annotation
  B) one-point-per-member normalized mismatch view

Logging uses LOGURU_FORMAT from core.constants.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from openamundsen_da.io.paths import infer_project_dir, infer_setup_dir, list_steps_sorted
from openamundsen_da.methods.viz._utils import force_figure_text_black, save_figure_png, set_matplotlib_text_black
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.da_observables import station_diagnostics_csv_name, weight_plot_title_from_csv_path
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.stats import effective_sample_size

_WEIGHTS_FIGSIZE = (7.2876875, 2.62)
_FRACTION_MISMATCH_COLORS = {
    "scf": "#2f6fb5",
    "wet_snow": "#2c8a64",
}
_FS_TITLE = 9.4
_FS_AXIS = 8.6
_FS_TICK = 8.4
_FS_NOTE = 7.4
_COMPOSITE_ROW_HEIGHT = 1.548


def _load_weights(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"weight", "residual"}
    if not needed.issubset(df.columns):
        missing = ", ".join(sorted(needed - set(df.columns)))
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def _apply_grid(ax) -> None:
    ax.grid(True, axis="both", which="major", alpha=0.5, linestyle="--", linewidth=0.8)
    ax.grid(True, axis="both", which="minor", alpha=0.38, linestyle="--", linewidth=0.65)


def _member_ticks(n: int) -> list[int]:
    if n <= 0:
        return []
    ticks = [1]
    ticks.extend(range(5, n + 1, 5))
    if ticks[-1] != n:
        ticks.append(n)
    return sorted(set(ticks))


def _observable_from_csv_path(csv_path: Path) -> str | None:
    stem = Path(csv_path).stem.lower()
    prefixes = {
        "weights_scf_": "scf",
        "weights_wet_snow_": "wet_snow",
        "weights_station_hs_": "station_hs",
        "weights_station_swe_": "station_swe",
    }
    for prefix, variable in prefixes.items():
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
    if observable == "scf":
        return "snow cover fraction mismatch"
    if observable == "wet_snow":
        return "wet-snow fraction mismatch"
    return "mismatch"


def _station_axis_label(observable: str | None) -> str:
    if observable == "station_hs":
        return "snow depth mismatch [m]"
    if observable == "station_swe":
        return "SWE mismatch [mm]"
    return "station mismatch"


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


def _station_sigma_note(diag: pd.DataFrame) -> str | None:
    if diag.empty or "station_id" not in diag.columns or "sigma" not in diag.columns:
        return None
    lines: list[str] = []
    for station_id in sorted(diag["station_id"].dropna().astype(str).unique()):
        station_mask = diag["station_id"].astype(str) == station_id
        sigma_series = pd.to_numeric(diag.loc[station_mask, "sigma"], errors="coerce").dropna()
        if sigma_series.empty:
            continue
        lines.append(f"{station_id}: sigma = {float(sigma_series.iloc[0]):.2f}")
    return "\n".join(lines) if lines else None


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
    from matplotlib.lines import Line2D

    legend_fill = "#bdbdbd"
    skipped_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor=legend_fill,
        markeredgecolor="#000000",
        markeredgewidth=0.9,
        markersize=5.8,
    )
    redraw_handle = (
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor=legend_fill,
            markeredgecolor="#000000",
            markeredgewidth=0.9,
            markersize=5.2,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            markersize=7.6,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#000000",
            markeredgewidth=0.7,
            markersize=10.0,
        ),
    )
    labels = [
        "resampling skipped (unchanged ensemble)",
        "redrawn source member (extra rings = repeated draws)",
    ]
    return [skipped_handle, redraw_handle], labels, {tuple: HandlerTuple(ndivide=1)}


def _ordered_weights_df(df: pd.DataFrame) -> pd.DataFrame:
    if "member_id" in df.columns:
        ordered_df = df.copy()
        ordered_df["_member_sort"] = ordered_df["member_id"].astype(str)
        ordered_df = ordered_df.sort_values(["weight", "_member_sort"], ascending=[False, True]).reset_index(drop=True)
        return ordered_df.drop(columns="_member_sort")
    return df.sort_values("weight", ascending=False).reset_index(drop=True)


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
    show_left_ylabel: bool = True,
    show_right_ylabel: bool = True,
    ring_step_scale: float = 1.0,
    ring_line_scale: float = 1.0,
    marker_scale: float = 1.0,
    font_size_bump: float = 0.0,
    axes_title_y: float = 1.22,
) -> None:
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

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
    ax0.set_xlim(0.0, 1.0)
    ax0.set_yticks(_member_ticks(n))
    ax0.set_ylim(n + 0.5, 0.5)
    ax0.xaxis.set_major_locator(MultipleLocator(0.1))
    ax0.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax0.yaxis.set_minor_locator(MultipleLocator(1.0))
    ax0.tick_params(axis="both", labelsize=fs_tick)
    threshold = resample_manifest.get("ess_threshold")
    metrics_label = f"ESS={ess:.1f}, N={n}"
    if threshold is not None:
        metrics_label = f"{metrics_label}, ESS threshold={float(threshold):.1f}"
    if show_metrics_label:
        ax0.text(
            0.5,
            1.02,
            metrics_label,
            transform=ax0.transAxes,
            ha="center",
            va="bottom",
            fontsize=fs_note,
            color="#000000",
        )

    sigma_note: str | None = None
    sigma_label_above: str | None = None
    ax1.axvline(0.0, color="black", lw=1.0, zorder=3)
    ax1.set_ylabel("sorted member" if show_right_ylabel else "", fontsize=fs_axis)
    ax1.set_yticks(_member_ticks(n))
    ax1.set_ylim(n + 0.5, 0.5)
    ax1.yaxis.set_minor_locator(MultipleLocator(1.0))
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
        station_display_names = _station_display_names(csv_path, station_ids)
        station_colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b"]
        station_color_map = {
            station_id: station_colors[idx % len(station_colors)]
            for idx, station_id in enumerate(station_ids)
        }
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
            legend_label = station_display_names.get(station_id, station_id)
            sigma_series = pd.to_numeric(sdf.get("sigma"), errors="coerce").dropna()
            if not sigma_series.empty:
                sigma_val = float(sigma_series.iloc[0])
                legend_label = f"{legend_label} (\u03c3={sigma_val:.2f})"
            ax1.scatter(
                sdf["residual_num"].to_numpy(dtype=float),
                y,
                facecolors=color,
                edgecolors=color,
                linewidths=0.9,
                s=mismatch_marker_size,
                zorder=4,
                label=legend_label,
            )
            _draw_resample_rings(
                ax1,
                sdf["residual_num"].to_numpy(dtype=float),
                y,
                station_draw_counts,
                base_size=mismatch_marker_size,
                ring_step=11.0 * ring_step_scale,
                line_scale=ring_line_scale,
            )
            if not sigma_series.empty and station_id not in shared_sigma_station_ids:
                ax1.axvline(-sigma_val, color=color, lw=1.0, ls="-", alpha=0.9, zorder=2)
                ax1.axvline(sigma_val, color=color, lw=1.0, ls="-", alpha=0.9, zorder=2)
        if station_ids:
            ax1.legend(
                loc="lower center",
                bbox_to_anchor=(0.50, 1.02),
                ncol=min(2, len(station_ids)),
                frameon=False,
                fontsize=fs_note,
                handlelength=1.0,
                handletextpad=0.3,
                columnspacing=0.8,
                borderaxespad=0.0,
            )
        y_min = 0.5
        y_max = n + 0.5
        for sigma_val, sigma_station_ids in shared_sigma_groups:
            colors = [station_color_map[station_id] for station_id in sigma_station_ids if station_id in station_color_map]
            _draw_alternating_sigma_line(ax1, -sigma_val, y_min, y_max, colors, lw=1.0, alpha=0.9, zorder=2)
            _draw_alternating_sigma_line(ax1, sigma_val, y_min, y_max, colors, lw=1.0, alpha=0.9, zorder=2)
        ax1.set_xlabel(_station_axis_label(observable), fontsize=fs_axis)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
        sigma_note = None
    else:
        residual = pd.to_numeric(ordered_df.get("residual"), errors="coerce")
        frac_color = _FRACTION_MISMATCH_COLORS.get(observable, "#ff7f0e")
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
            sigma_label_above = f"\u03c3={sigma_val:.2f}"
        ax1.set_xlabel(_fraction_axis_label(observable), fontsize=fs_axis)
        ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    if sigma_label_above:
        ax1.text(
            0.5,
            1.02,
            sigma_label_above,
            transform=ax1.transAxes,
            ha="center",
            va="bottom",
            fontsize=fs_note,
            color="#000000",
        )
    elif sigma_note:
        ax1.text(
            0.02,
            0.95,
            sigma_note,
            transform=ax1.transAxes,
            ha="left",
            va="top",
            fontsize=fs_note,
            color="#000000",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
        )
    _apply_grid(ax1)
    ax1.tick_params(axis="both", labelsize=fs_tick)

    header = title
    if subtitle:
        if title_mode == "axes":
            header = f"{title}\n{subtitle}" if title else subtitle
        else:
            header = f"{title} - {subtitle}" if title else subtitle
    if header:
        if title_mode == "figure":
            fig.text(0.11, 0.985, header, ha="left", va="top", fontsize=fs_title, color="#000000")
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
    from matplotlib.legend_handler import HandlerTuple
    from matplotlib.lines import Line2D
    from matplotlib.ticker import AutoMinorLocator, MultipleLocator

    fig = plt.figure(figsize=_WEIGHTS_FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.25, 2.75])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
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
    )

    bottom_margin = 0.295
    right_margin = 0.965
    top_margin = 0.79 if observable in {"station_hs", "station_swe"} else 0.845
    fig.subplots_adjust(left=0.095, right=right_margin, top=top_margin, bottom=bottom_margin, wspace=0.30)
    legend_handles, legend_labels, handler_map = _resample_legend_artists()
    fig.legend(
        legend_handles,
        legend_labels,
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.085),
        ncol=2,
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
    out_dir = Path(setup_dir) / "plots" / "assim" / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"setup_weights_overview_{_setup_id_from_dir(Path(setup_dir))}.png"


def plot_setup_weights_overview(setup_dir: Path, *, backend: str = "Agg") -> Path:
    import matplotlib

    matplotlib.use(backend or "Agg")
    set_matplotlib_text_black(matplotlib)
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    setup_dir = Path(setup_dir)
    csv_paths = _setup_weights_csvs(setup_dir)
    if not csv_paths:
        raise FileNotFoundError(f"No weights_*_*.csv found under steps in {setup_dir}")

    n_events = len(csv_paths)
    n_cols = 2
    n_rows = int(math.ceil(n_events / n_cols))
    fig = plt.figure(figsize=(7.2876875, _COMPOSITE_ROW_HEIGHT * n_rows))
    outer = fig.add_gridspec(n_rows, n_cols, left=0.06, right=0.99, top=0.92, bottom=0.095, wspace=0.0, hspace=0.78)

    axes_for_black: list[object] = []
    font_scale = 0.68
    first_df = _load_weights(csv_paths[0])
    ensemble_size = len(first_df)
    first_manifest = _read_resample_manifest(csv_paths[0])
    ess_threshold = first_manifest.get("ess_threshold")
    for idx, csv_path in enumerate(csv_paths):
        row = idx // n_cols
        col = idx % n_cols
        left_ratio = 1.15 * 0.8 * 0.8 * 0.6 * 0.8
        right_ratio = 2.16 * 0.75 * 0.7 * 0.6 * 0.8
        spacer_ratio = 0.06 * (left_ratio + right_ratio)
        sub = outer[row, col].subgridspec(
            2,
            3,
            height_ratios=[1.0, 0.045],
            width_ratios=[left_ratio * 0.7, right_ratio * 0.7, spacer_ratio],
            wspace=0.06,
            hspace=0.0,
        )
        ax0 = fig.add_subplot(sub[0, 0])
        ax1 = fig.add_subplot(sub[0, 1])
        axes_for_black.extend([ax0, ax1])
        df = _load_weights(csv_path)
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
            observable=_observable_from_csv_path(csv_path),
            title_mode="axes",
            font_scale=font_scale,
            show_metrics_label=False,
            show_left_ylabel=(col == 0),
            show_right_ylabel=False,
            ring_step_scale=0.72,
            ring_line_scale=0.72,
            marker_scale=0.8,
            font_size_bump=1.0,
            axes_title_y=1.18,
        )
        ax0.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax0.set_xticklabels(["0.2", "0.4", "0.6", "0.8", "1"])
        ax0.xaxis.set_minor_locator(NullLocator())
        ax1.tick_params(axis="y", labelleft=False)

    summary = f"ensemble size = {ensemble_size}"
    if ess_threshold is not None:
        summary = f"{summary}, ESS threshold = {float(ess_threshold):.1f}"
    fig.text(
        0.06,
        0.985,
        f"data assimilation weights ({summary})",
        va="top",
        ha="left",
        fontsize=8.6,
        color="#000000",
    )
    legend_handles, legend_labels, handler_map = _resample_legend_artists()
    fig.legend(
        legend_handles,
        legend_labels,
        handler_map=handler_map,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.028),
        ncol=2,
        frameon=False,
        fontsize=6.2,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )
    force_figure_text_black(fig, axes_for_black)
    out = _default_setup_weights_overview_output(setup_dir)
    save_figure_png(fig, out, dpi=600, bbox_inches="tight", pad_inches=0.04)
    return out


def _default_output_path(csv_path: Path) -> Path:
    """Return default output PNG path for a weights CSV.

    If the CSV lives under <project>/step_XX_*/assim/, write to
    <project>/plots/assim/weights/DA_XX_weights.png. Otherwise, fall back
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
            out_dir = project_dir / "plots" / "assim" / "weights"
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
        date_val = pd.to_datetime(date_str).date()
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
    save_figure_png(fig, out, bbox_inches="tight", pad_inches=0.08)
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
        save_figure_png(fig, out, bbox_inches="tight", pad_inches=0.08)
    except Exception as e:
        logger.error(f"Saving PNG failed: {e}")
        return 4
    logger.info("Wrote plot: {}", out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
