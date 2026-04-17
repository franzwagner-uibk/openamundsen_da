"""Generic figure/export helpers shared across visualization subpackages."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from openamundsen_da.methods.viz.theme import EXPORT_DPI, TEXT_COLOR


def set_matplotlib_text_black(matplotlib) -> None:
    """Force matplotlib text/ticks/legend defaults to pure black."""
    matplotlib.rcParams["text.color"] = TEXT_COLOR
    matplotlib.rcParams["axes.labelcolor"] = TEXT_COLOR
    matplotlib.rcParams["axes.titlecolor"] = TEXT_COLOR
    matplotlib.rcParams["xtick.color"] = TEXT_COLOR
    matplotlib.rcParams["ytick.color"] = TEXT_COLOR
    matplotlib.rcParams["legend.labelcolor"] = TEXT_COLOR


def force_figure_text_black(fig, axes: Iterable | None = None) -> None:
    """Force existing figure/axes text artists to pure black before save."""
    axes_list = [] if axes is None else list(axes)
    for text in getattr(fig, "texts", []):
        text.set_color(TEXT_COLOR)
    for legend in getattr(fig, "legends", []):
        for text in legend.get_texts():
            text.set_color(TEXT_COLOR)
        title = legend.get_title()
        if title is not None:
            title.set_color(TEXT_COLOR)
    for ax in axes_list:
        ax.title.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.tick_params(axis="both", colors=TEXT_COLOR, labelcolor=TEXT_COLOR)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
            title = legend.get_title()
            if title is not None:
                title.set_color(TEXT_COLOR)


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

