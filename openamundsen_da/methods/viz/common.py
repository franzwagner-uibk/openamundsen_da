"""Generic figure/export helpers shared across visualization subpackages."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import ModuleType

from PIL import Image

from openamundsen_da.methods.viz.theme import EXPORT_DPI, TEXT_COLOR


@dataclass(frozen=True)
class PosterTypography:
    title_pt: float
    label_pt: float
    support_pt: float


@dataclass(frozen=True)
class PosterLinework:
    panel_box_pt: float


@dataclass(frozen=True)
class PosterRenderStyle:
    scale: float = 1.0
    typography: PosterTypography | None = None
    linework: PosterLinework | None = None


@contextmanager
def scaled_module_attributes(
    module: ModuleType,
    names: Iterable[str],
    scale: float,
):
    """Temporarily multiply numeric module attributes by a scale factor."""
    if scale == 1.0:
        yield
        return
    original = {name: getattr(module, name) for name in names if hasattr(module, name)}
    try:
        for name, value in original.items():
            if isinstance(value, (int, float)):
                setattr(module, name, value * scale)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)


@contextmanager
def temporary_module_attributes(module: ModuleType, values: dict[str, object]):
    """Temporarily assign module attributes."""
    original = {name: getattr(module, name) for name in values if hasattr(module, name)}
    missing = tuple(name for name in values if not hasattr(module, name))
    try:
        for name, value in values.items():
            setattr(module, name, value)
        yield
    finally:
        for name, value in original.items():
            setattr(module, name, value)
        for name in missing:
            try:
                delattr(module, name)
            except AttributeError:
                pass


def _write_png_with_target_canvas(buffer: BytesIO, output_png: Path, target_px: tuple[int, int]) -> None:
    buffer.seek(0)
    with Image.open(buffer) as image:
        if image.size == target_px:
            buffer.seek(0)
            output_png.write_bytes(buffer.getvalue())
            return
        mode = "RGBA" if image.mode == "RGBA" else "RGB"
        background = (255, 255, 255, 0) if mode == "RGBA" else (255, 255, 255)
        canvas = Image.new(mode, target_px, background)
        source = image.convert(mode).crop((0, 0, min(image.width, target_px[0]), min(image.height, target_px[1])))
        canvas.paste(source, (0, 0))
        canvas.save(output_png, format="PNG", dpi=image.info.get("dpi"))


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
    target_size_in: tuple[float, float] | None = None,
) -> None:
    """Save a figure as PNG using the shared export DPI."""
    output_png = Path(output_png)
    save_kwargs = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kwargs["pad_inches"] = pad_inches
    if target_size_in is None:
        fig.savefig(output_png, dpi=dpi, **save_kwargs)
        return

    target_px = (
        max(1, int(round(float(target_size_in[0]) * dpi))),
        max(1, int(round(float(target_size_in[1]) * dpi))),
    )
    for _ in range(4):
        buffer = BytesIO()
        fig.savefig(buffer, format="png", dpi=dpi, **save_kwargs)
        buffer.seek(0)
        with Image.open(buffer) as image:
            current_px = image.size
        if current_px == target_px:
            _write_png_with_target_canvas(buffer, output_png, target_px)
            return
        width_scale = target_px[0] / max(current_px[0], 1)
        height_scale = target_px[1] / max(current_px[1], 1)
        width_in, height_in = fig.get_size_inches()
        fig.set_size_inches(width_in * width_scale, height_in * height_scale, forward=True)
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, **save_kwargs)
    _write_png_with_target_canvas(buffer, output_png, target_px)
