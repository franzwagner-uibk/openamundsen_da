"""Visualization utilities shared by plot modules."""

from __future__ import annotations

from typing import Iterable, List, Tuple

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


def draw_assimilation_markers(
    ax,
    *,
    dates: Iterable,
    obs: pd.DataFrame | None,
    value_col: str,
    color: str,
    label: str,
    marker: str = "x",
    size: float = 80.0,
    zorder: int = 5,
    draw_vlines: bool = True,
) -> None:
    """Draw assimilation vlines and overlay obs crosses on the same dates."""
    if draw_vlines:
        draw_assimilation_vlines(ax, dates)
    if obs is None or obs.empty:
        return
    try:
        obs_dates = pd.to_datetime(obs["date"]).normalize()
        target = pd.to_datetime(list(dates)).normalize()
    except Exception:
        return
    mask = obs_dates.isin(target)
    if not mask.any():
        return
    ax.scatter(
        obs.loc[mask, "date"],
        obs.loc[mask, value_col],
        color=color,
        marker=marker,
        s=size,
        zorder=zorder,
        label=label,
    )


def apply_fraction_grid(ax, *, y_step: float = 0.1) -> None:
    """Apply consistent grid styling for fraction plots."""
    from matplotlib.ticker import MultipleLocator
    import matplotlib.dates as mdates

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_locator(MultipleLocator(y_step))
    ax.grid(True, axis="both", alpha=0.5, linestyle="--", linewidth=0.8)
