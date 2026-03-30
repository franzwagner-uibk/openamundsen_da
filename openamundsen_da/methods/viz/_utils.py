"""Visualization utilities shared by plot modules."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple
import math
from pathlib import Path

import pandas as pd

from openamundsen_da.methods.viz._style import EXPORT_DPI


def set_matplotlib_text_black(matplotlib) -> None:
    """Force matplotlib text/ticks/legend defaults to pure black."""
    matplotlib.rcParams["text.color"] = "#000000"
    matplotlib.rcParams["axes.labelcolor"] = "#000000"
    matplotlib.rcParams["axes.titlecolor"] = "#000000"
    matplotlib.rcParams["xtick.color"] = "#000000"
    matplotlib.rcParams["ytick.color"] = "#000000"
    matplotlib.rcParams["legend.labelcolor"] = "#000000"


def force_figure_text_black(fig, axes: Iterable | None = None) -> None:
    """Force existing figure/axes text artists to pure black before save."""
    axes_list = [] if axes is None else list(axes)
    for text in getattr(fig, "texts", []):
        text.set_color("#000000")
    for legend in getattr(fig, "legends", []):
        for text in legend.get_texts():
            text.set_color("#000000")
        title = legend.get_title()
        if title is not None:
            title.set_color("#000000")
    for ax in axes_list:
        ax.title.set_color("#000000")
        ax.xaxis.label.set_color("#000000")
        ax.yaxis.label.set_color("#000000")
        ax.tick_params(axis="both", colors="#000000", labelcolor="#000000")
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color("#000000")
            title = legend.get_title()
            if title is not None:
                title.set_color("#000000")


def save_figure_png(
    fig,
    output_png: Path,
    *,
    dpi: int = EXPORT_DPI,
    bbox_inches=None,
    pad_inches=None,
) -> None:
    """Save a figure as PNG using the shared export DPI."""
    output_png = Path(output_png)
    save_kwargs = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kwargs["pad_inches"] = pad_inches
    fig.savefig(output_png, dpi=dpi, **save_kwargs)


def result_axis_scale(
    token: str,
    data_max: float,
    *,
    shared: bool = False,
) -> Tuple[float, float] | None:
    """Return (major_step, upper_ylim) for SWE / snow-depth style result axes."""
    value = max(0.0, float(data_max or 0.0))
    key = str(token or "").strip().lower()
    if key in {"swe", "roi-swe", "station-swe"}:
        step_options = [50.0, 100.0]
    elif key in {"snow_depth", "snowdepth", "hs", "roi-sd", "station-sd"}:
        step_options = [0.25, 0.5, 1.0]
    else:
        return None

    step = next((candidate for candidate in step_options if value <= candidate * 4.0), step_options[-1])
    if shared:
        substep = step / 2.0
        upper = substep if value <= 0.0 else math.ceil(value / substep) * substep
        return step, upper

    upper = step * 4.0 if value <= step * 4.0 else math.ceil(value / step) * step
    return step, upper


def pretty_var_title(var_col: str, var_label: str = "", var_units: str = "") -> str:
    """Return a friendly variable title with optional units."""
    v = (var_col or "").strip()
    if not var_label and not var_units:
        lv = v.lower()
        if lv == "swe":
            return "snow water equivalent [mm]"
        if lv in {"snow_depth", "snowdepth", "hs"}:
            return "snow depth [m]"
    base = var_label.strip() if var_label else v.replace("_", " ")
    if var_units:
        return f"{base} [{var_units}]"
    return base


def find_station_meta(st_df: Optional[pd.DataFrame], token: str) -> Tuple[Optional[str], Optional[float]]:
    """Return (name, altitude_m) for a station token using a stations table."""
    if st_df is None or st_df.empty:
        return None, None
    df = st_df.copy()
    cols_lower = {c.lower().strip(): c for c in df.columns}
    id_candidates = [c for c in ("id", "station_id", "station", "code") if c in cols_lower]
    name_candidates = [c for c in ("name", "station_name") if c in cols_lower]
    alt_candidates = [c for c in ("alt", "altitude", "elev", "elevation", "z", "height", "height_m") if c in cols_lower]
    alt_col = cols_lower[alt_candidates[0]] if alt_candidates else None

    def _match(col_key: str) -> Optional[pd.Series]:
        col = cols_lower[col_key]
        try:
            normalized = df[col].astype(str).str.strip().str.lower()
            hit = df.loc[normalized == token.lower()]
            if not hit.empty:
                return hit.iloc[0]
        except Exception:
            return None
        return None

    row = None
    for k in id_candidates:
        row = _match(k)
        if row is not None:
            break
    if row is None:
        for k in name_candidates:
            row = _match(k)
            if row is not None:
                break
    if row is None:
        return None, None

    name_val = None
    for k in name_candidates:
        try:
            name_val = str(row[cols_lower[k]]).strip()
            break
        except Exception:
            continue
    alt_val = None
    if alt_col is not None:
        try:
            alt_val = float(row[alt_col])
        except Exception:
            alt_val = None
    return name_val, alt_val


def format_station_label(token: str, st_df: Optional[pd.DataFrame], *, fallback: Optional[str] = None) -> Tuple[str, Optional[float], str]:
    """Return (display_name, altitude_m, label_with_alt) for a station token."""
    name, alt = find_station_meta(st_df, token)
    display = name or fallback or token
    label = f"{display} ({alt:.0f} m)" if alt is not None else display
    return display, alt, label


def draw_assimilation_vlines(
    ax,
    dates: Iterable,
    *,
    color: str = "#777777",
    ls: str = "--",
    lw: float = 1.0,
    alpha: float = 0.9,
    label: str = "assimilation",
    zorder: int = 20,
) -> None:
    for d in dates:
        ax.axvline(d, color=color, ls=ls, lw=lw, alpha=alpha, label=label, zorder=zorder)


def dedupe_legend(handles: List, labels: List) -> Tuple[List, List]:
    seen = set()
    new_h, new_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            new_h.append(h)
            new_l.append(l)
    return new_h, new_l


def draw_assim_labels(
    ax,
    dates: Iterable,
    *,
    labels: Iterable[str] | None = None,
    colors: Iterable[str] | None = None,
    max_labels: int = 12,
    y_offset_pts: float = 3.0,
    fontsize: float = 8.0,
    color: str = "black",
    rotation: float = 0.0,
    va: str = "bottom",
    row_y_offsets_pts: Iterable[float] | None = None,
    min_row_spacing_days: float | None = None,
    axes_y: float = 1.0,
    ha: str = "center",
    x_offset_pts: float = 0.0,
) -> None:
    """Draw decimated assimilation labels aligned to dates near the top of the axes."""
    dates = list(pd.to_datetime(dates))
    label_list = list(labels) if labels is not None else None
    if label_list is not None and len(label_list) != len(dates):
        label_list = None
    color_list = list(colors) if colors is not None else None
    if color_list is not None and len(color_list) != len(dates):
        color_list = None
    if not dates:
        return
    step = max(1, math.ceil(len(dates) / max(1, int(max_labels))))
    display_items: list[tuple[pd.Timestamp, str, str]] = []
    for i, d in enumerate(dates, start=1):
        if (i - 1) % step != 0:
            continue
        text = label_list[i - 1] if label_list is not None else f"{i}"
        text_color = color_list[i - 1] if color_list is not None else color
        display_items.append((d, text, text_color))

    row_offsets = list(row_y_offsets_pts) if row_y_offsets_pts is not None else [y_offset_pts]
    if not row_offsets:
        row_offsets = [y_offset_pts]

    if len(display_items) >= 2 and min_row_spacing_days is None:
        span_days = max(1.0, float((display_items[-1][0] - display_items[0][0]).days))
        min_row_spacing_days = max(5.0, span_days / 18.0)
    elif min_row_spacing_days is None:
        min_row_spacing_days = 5.0

    row_last_dates: list[pd.Timestamp | None] = [None] * len(row_offsets)
    for d, text, text_color in display_items:
        row_idx = 0
        if len(row_offsets) > 1:
            chosen_idx = None
            for idx, last_dt in enumerate(row_last_dates):
                if last_dt is None:
                    chosen_idx = idx
                    break
                delta_days = (d - last_dt).total_seconds() / 86400.0
                if delta_days >= float(min_row_spacing_days):
                    chosen_idx = idx
                    break
            if chosen_idx is None:
                chosen_idx = min(
                    range(len(row_last_dates)),
                    key=lambda idx: row_last_dates[idx] or pd.Timestamp.min,
                )
            row_idx = chosen_idx
        row_last_dates[row_idx] = d
        ax.annotate(
            text,
            xy=(d, axes_y),
            xycoords=("data", "axes fraction"),
            xytext=(x_offset_pts, row_offsets[row_idx]),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            color=text_color,
            rotation=rotation,
            rotation_mode="anchor",
            clip_on=False,
        )


def draw_assimilation_markers(
    ax,
    *,
    dates: Iterable,
    obs: pd.DataFrame | None,
    value_col: str,
    color: str,
    label: str,
    marker: str = "x",
    size: float = 120.0,
    linewidth: float = 2.0,
    zorder: int = 30,
    draw_vlines: bool = True,
) -> None:
    """Draw assimilation vlines and overlay obs crosses on the same dates."""
    target = pd.to_datetime(list(dates))
    if draw_vlines and len(target) > 0:
        draw_assimilation_vlines(ax, target)
    if obs is None or obs.empty or len(target) == 0:
        return
    try:
        obs_dt = pd.to_datetime(obs["date"])
        mask = obs_dt.dt.normalize().isin(target.normalize())
    except Exception:
        return
    if mask.any():
        ax.scatter(
            obs_dt.loc[mask],
            obs.loc[mask, value_col],
            color=color,
            marker=marker,
            s=size,
            linewidths=linewidth,
            zorder=zorder,
            clip_on=False,
            label=label,
        )


def plot_haloed_line(
    ax,
    x,
    y,
    *,
    color: str,
    lw: float,
    label: str = "_nolegend_",
    zorder: int = 10,
    halo_color: str = "white",
    halo_lw_add: float = 1.6,
) -> None:
    """Draw a line with a thin white underlay to keep it readable over dense traces."""
    ax.plot(
        x,
        y,
        "-",
        color=halo_color,
        lw=lw + halo_lw_add,
        label="_nolegend_",
        zorder=max(0, zorder - 1),
        solid_capstyle="round",
    )
    ax.plot(
        x,
        y,
        "-",
        color=color,
        lw=lw,
        label=label,
        zorder=zorder,
        solid_capstyle="round",
    )


def apply_fraction_grid(ax, *, y_step: float | None = 0.1) -> None:
    """Apply consistent grid styling for result overview plots."""
    from matplotlib.ticker import MultipleLocator
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.grid(True, axis="both", alpha=0.5, linestyle="--", linewidth=0.8)
