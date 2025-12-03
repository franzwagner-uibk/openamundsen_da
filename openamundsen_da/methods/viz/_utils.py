"""Visualization utilities shared by plot modules."""

from __future__ import annotations

from typing import Iterable, List, Tuple
import math

import pandas as pd


def draw_assimilation_vlines(ax, dates: Iterable) -> None:
    for d in dates:
        ax.axvline(d, color="#777777", ls="--", lw=1.0, alpha=0.9, label="assimilation")


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
    max_labels: int = 12,
    y_offset_pts: float = 3.0,
    fontsize: float = 8.0,
    color: str = "black",
) -> None:
    """Draw decimated, upright assimilation labels near the top of the axes."""
    dates = list(dates)
    label_list = list(labels) if labels is not None else None
    if label_list is not None and len(label_list) != len(dates):
        label_list = None
    if not dates:
        return
    step = max(1, math.ceil(len(dates) / max(1, int(max_labels))))
    for i, d in enumerate(dates, start=1):
        if (i - 1) % step != 0:
            continue
        text = label_list[i - 1] if label_list is not None else f"{i}"
        ax.annotate(
            text,
            xy=(d, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, y_offset_pts),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            rotation=0,
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


def apply_fraction_grid(ax, *, y_step: float = 0.1) -> None:
    """Apply consistent grid styling for fraction plots."""
    from matplotlib.ticker import MultipleLocator
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.grid(True, axis="both", alpha=0.5, linestyle="--", linewidth=0.8)
