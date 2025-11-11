"""Visualization utilities shared by plot modules."""

from __future__ import annotations

from typing import Iterable, List, Tuple


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

