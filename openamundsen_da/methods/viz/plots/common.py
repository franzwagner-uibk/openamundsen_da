"""Visualization utilities shared by plot modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import math

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from openamundsen_da.methods.viz.common import (
    force_figure_text_black as force_figure_text_black,
    save_figure_png as save_figure_png,
    set_matplotlib_text_black as set_matplotlib_text_black,
)
from openamundsen_da.methods.viz.plots.theme import (
    CRPSS_AXIS_STEP_CANDIDATES,
    CRPSS_AXIS_UPPER_CAP,
    FIGURE_TITLE_CLEARANCE_PTS,
    FIGURE_TITLE_TOP_MARGIN_PTS,
    METRIC_AXIS_MAX_INTERVALS,
    METRIC_AXIS_MIN_INTERVALS,
    TITLE_PAD_DEFAULT,
    TITLE_PAD_WITH_ASSIM_LABELS,
)


@dataclass(frozen=True)
class MetricAxisPolicy:
    """Generic axis-scaling policy for bounded or semi-bounded metrics."""

    preferred_steps: tuple[float, ...]
    min_intervals: int = METRIC_AXIS_MIN_INTERVALS
    max_intervals: int = METRIC_AXIS_MAX_INTERVALS
    lower_cap: float | None = None
    upper_cap: float | None = None
    force_include_zero: bool = False


CRPSS_AXIS_POLICY = MetricAxisPolicy(
    preferred_steps=CRPSS_AXIS_STEP_CANDIDATES,
    min_intervals=METRIC_AXIS_MIN_INTERVALS,
    max_intervals=METRIC_AXIS_MAX_INTERVALS,
    upper_cap=CRPSS_AXIS_UPPER_CAP,
    force_include_zero=False,
)


def result_title_pad(has_assim_labels: bool) -> float:
    """Return the shared title pad for panels with/without assimilation labels."""
    return TITLE_PAD_WITH_ASSIM_LABELS if has_assim_labels else TITLE_PAD_DEFAULT


def align_figure_title_to_plot_block(
    fig,
    axes: Iterable,
    *,
    left_pad_pts: float = 6.0,
) -> None:
    """Align a figure-level title to the visible plot block using shared spacing rules."""
    title_artist = getattr(fig, "_suptitle", None)
    if title_artist is None:
        return

    axes_list = list(axes)
    if not axes_list:
        return

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    global_left_disp: float | None = None
    top_disp = max(ax.bbox.y1 for ax in axes_list)
    for ax in axes_list:
        tick_labels = [label for label in ax.get_yticklabels() if label.get_text()]
        if tick_labels:
            left_disp = min(label.get_window_extent(renderer).x0 for label in tick_labels) - left_pad_pts
            if global_left_disp is None or left_disp < global_left_disp:
                global_left_disp = left_disp

        for text in (*ax.texts, *ax.get_xticklabels(), *ax.get_yticklabels()):
            if not text.get_visible() or not text.get_text():
                continue
            top_disp = max(top_disp, text.get_window_extent(renderer).y1)

    if global_left_disp is None:
        global_left_disp = min(ax.bbox.x0 for ax in axes_list)

    current_bbox = title_artist.get_window_extent(renderer)
    title_height_px = current_bbox.height
    clearance_px = FIGURE_TITLE_CLEARANCE_PTS * fig.dpi / 72.0
    top_margin_px = FIGURE_TITLE_TOP_MARGIN_PTS * fig.dpi / 72.0
    title_x = fig.transFigure.inverted().transform((global_left_disp, 0.0))[0]
    target_top_disp = fig.bbox.y1 - top_margin_px
    min_bottom_disp = top_disp + clearance_px
    desired_top_disp = min(fig.bbox.y1 - 1.0, max(target_top_disp, min_bottom_disp + title_height_px))
    desired_bottom_disp = desired_top_disp - title_height_px
    current_y_disp = fig.transFigure.transform((0.0, title_artist.get_position()[1]))[1]
    title_y = fig.transFigure.inverted().transform((0.0, current_y_disp + (desired_bottom_disp - current_bbox.y0)))[1]
    title_artist.set_x(title_x)
    title_artist.set_y(title_y)
    title_artist.set_ha("left")


def _expand_metric_range(
    lower: float,
    upper: float,
    step: float,
    *,
    min_intervals: int,
    lower_cap: float | None,
    upper_cap: float | None,
    positive_only: bool,
    negative_only: bool,
    center: float,
) -> tuple[float, float]:
    """Expand a rounded metric range to a readable minimum span."""
    tol = 1e-12
    while ((upper - lower) / step) + tol < float(min_intervals):
        candidates: list[tuple[float, float, tuple[float, float]]] = []
        if lower_cap is None or lower - step >= lower_cap - tol:
            candidate = (lower - step, upper)
            candidate_center = (candidate[0] + candidate[1]) / 2.0
            penalty = 0.0
            if positive_only and candidate[0] < -tol:
                penalty += 1e6
            if negative_only and candidate[1] > tol:
                penalty += 1e6
            candidates.append((penalty, abs(candidate_center - center), candidate))
        if upper_cap is None or upper + step <= upper_cap + tol:
            candidate = (lower, upper + step)
            candidate_center = (candidate[0] + candidate[1]) / 2.0
            penalty = 0.0
            if positive_only and candidate[0] < -tol:
                penalty += 1e6
            if negative_only and candidate[1] > tol:
                penalty += 1e6
            candidates.append((penalty, abs(candidate_center - center), candidate))
        if not candidates:
            break
        _, _, chosen = min(candidates, key=lambda item: (item[0], item[1], item[2][0], item[2][1]))
        lower, upper = chosen
    return float(lower), float(upper)


def bounded_metric_range(
    values,
    *,
    policy: MetricAxisPolicy,
) -> tuple[float, float, float]:
    """Return a rounded outward metric range using a generic bounded-metric policy."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    preferred_steps = tuple(float(step) for step in policy.preferred_steps)
    if not preferred_steps:
        raise ValueError("MetricAxisPolicy.preferred_steps must not be empty")
    if arr.size == 0:
        step = preferred_steps[0]
        lower = float(policy.lower_cap) if policy.lower_cap is not None else 0.0
        upper = lower + step * max(1, policy.min_intervals)
        if policy.upper_cap is not None:
            upper = min(upper, float(policy.upper_cap))
            if upper - lower < step:
                lower = upper - step * max(1, policy.min_intervals)
        return float(lower), float(upper), float(step)

    data_min = float(arr.min())
    data_max = float(arr.max())
    if policy.lower_cap is not None:
        data_min = max(data_min, float(policy.lower_cap))
    if policy.upper_cap is not None:
        data_max = min(data_max, float(policy.upper_cap))
    if policy.force_include_zero:
        data_min = min(data_min, 0.0)
        data_max = max(data_max, 0.0)

    positive_only = data_min >= 0.0 and not policy.force_include_zero
    negative_only = data_max <= 0.0 and not policy.force_include_zero
    center = (data_min + data_max) / 2.0

    for step in preferred_steps:
        lower = float(math.floor(data_min / step) * step)
        upper = float(math.ceil(data_max / step) * step)
        if policy.lower_cap is not None:
            lower = max(lower, float(policy.lower_cap))
        if policy.upper_cap is not None:
            upper = min(upper, float(policy.upper_cap))
        lower, upper = _expand_metric_range(
            lower,
            upper,
            step,
            min_intervals=policy.min_intervals,
            lower_cap=policy.lower_cap,
            upper_cap=policy.upper_cap,
            positive_only=positive_only,
            negative_only=negative_only,
            center=center,
        )
        interval_count = (upper - lower) / step
        if interval_count <= policy.max_intervals + 1e-12:
            return float(lower), float(upper), float(step)

    step = preferred_steps[-1]
    lower = float(math.floor(data_min / step) * step)
    upper = float(math.ceil(data_max / step) * step)
    if policy.lower_cap is not None:
        lower = max(lower, float(policy.lower_cap))
    if policy.upper_cap is not None:
        upper = min(upper, float(policy.upper_cap))
    lower, upper = _expand_metric_range(
        lower,
        upper,
        step,
        min_intervals=policy.min_intervals,
        lower_cap=policy.lower_cap,
        upper_cap=policy.upper_cap,
        positive_only=positive_only,
        negative_only=negative_only,
        center=center,
    )
    return float(lower), float(upper), float(step)


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


def add_assim_label_axis(
    ax,
    dates: Iterable,
    *,
    idx: int = 0,
    labels: Iterable[str] | None = None,
    y_offset_pts: float = 2.0,
    fontsize: float = 6.0,
    color: str = "#000000",
    row_y_offsets_pts: Iterable[float] = (2.0, 8.0),
    min_row_spacing_days: float = 18.0,
    axes_y: float = 1.0,
    ha: str = "center",
    x_offset_pts: float = 0.0,
) -> object | None:
    """Create a lightweight top axis for visible assimilation labels."""
    date_index = list(pd.to_datetime(list(dates)))
    if not date_index:
        return None
    x_min, x_max = sorted(ax.get_xlim())
    visible_start = pd.Timestamp(mdates.num2date(x_min)).tz_localize(None)
    visible_end = pd.Timestamp(mdates.num2date(x_max)).tz_localize(None)
    label_list = list(labels) if labels is not None else [str(i) for i in range(1, len(date_index) + 1)]
    visible_items = [
        (date, label)
        for date, label in zip(date_index, label_list)
        if visible_start <= pd.Timestamp(date).tz_localize(None) <= visible_end
    ]
    if not visible_items:
        return None

    label_axis = ax.twiny()
    label_axis.set_label(f"assimilation_label_axis_{idx}")
    label_axis.patch.set_alpha(0.0)
    if hasattr(label_axis, "set_in_layout"):
        label_axis.set_in_layout(False)
    label_axis.set_xlim(ax.get_xlim())
    label_axis.set_xticks([])
    label_axis.set_xlabel("")
    label_axis.yaxis.set_visible(False)
    label_axis.xaxis.set_visible(False)
    for spine in label_axis.spines.values():
        spine.set_visible(False)

    draw_assim_labels(
        label_axis,
        [item[0] for item in visible_items],
        labels=[item[1] for item in visible_items],
        max_labels=max(1, len(visible_items)),
        y_offset_pts=y_offset_pts,
        fontsize=fontsize,
        color=color,
        rotation=0.0,
        va="bottom",
        row_y_offsets_pts=row_y_offsets_pts,
        min_row_spacing_days=min_row_spacing_days,
        axes_y=axes_y,
        ha=ha,
        x_offset_pts=x_offset_pts,
    )
    return label_axis


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


def thin_dense_y_tick_labels(ax, *, max_visible_labels: int = 4) -> None:
    """Hide every second in-range y tick label when a compact axis is too dense."""
    if max_visible_labels < 2:
        raise ValueError("max_visible_labels must be at least 2")

    ymin, ymax = sorted(float(value) for value in ax.get_ylim())
    tol = max(abs(ymax - ymin), 1.0) * 1e-9
    in_range_ticks = [
        tick
        for tick in ax.yaxis.get_major_ticks()
        if ymin - tol <= float(tick.get_loc()) <= ymax + tol
    ]
    if len(in_range_ticks) <= max_visible_labels:
        return

    keep_positions = set(range(0, len(in_range_ticks), 2))
    keep_positions.add(len(in_range_ticks) - 1)
    original_label1_visibility = [tick.label1.get_visible() for tick in in_range_ticks]
    original_label2_visibility = [tick.label2.get_visible() for tick in in_range_ticks]
    for pos, tick in enumerate(in_range_ticks):
        visible = pos in keep_positions
        tick.label1.set_visible(visible and original_label1_visibility[pos])
        tick.label2.set_visible(visible and original_label2_visibility[pos])


def apply_fraction_grid(ax, *, y_step: float | None = 0.1) -> None:
    """Apply consistent grid styling for result overview plots."""
    from matplotlib.ticker import MultipleLocator
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.grid(True, axis="both", alpha=0.5, linestyle="--", linewidth=0.8)
