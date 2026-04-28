from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from loguru import logger
from PIL import Image

from openamundsen_da.io.paths import (
    project_maps_root,
    project_plots_maps_collection_pdf_path,
    project_plots_root,
    project_result_overview_custom_output_path,
    project_result_overview_output_path,
)
from openamundsen_da.methods.viz.maps.generated import GENERATED_DA_MAPS_SUBDIR
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.loguru_utils import configure_cli_logger


DEFAULT_IMAGE_DPI = 600.0


@dataclass(frozen=True)
class PdfImageItem:
    path: Path
    label: str


@dataclass(frozen=True)
class PdfDaStepItem:
    index: int
    map_path: Path
    weights_path: Path


@dataclass(frozen=True)
class ProjectPdfPlan:
    front_items: tuple[PdfImageItem, ...]
    da_steps: tuple[PdfDaStepItem, ...]
    appendix_items: tuple[PdfImageItem, ...]
    missing_paths: tuple[Path, ...]

    @property
    def page_count(self) -> int:
        return len(self.front_items) + 2 * len(self.da_steps) + len(self.appendix_items)


class MissingProjectPdfArtifactsError(FileNotFoundError):
    def __init__(self, project_dir: Path, missing_paths: Iterable[Path]):
        self.project_dir = Path(project_dir)
        self.missing_paths = tuple(Path(path) for path in missing_paths)
        formatted = "\n".join(f"- {path}" for path in self.missing_paths)
        super().__init__(
            "Missing required project PDF artifact(s). Regenerate project plots/maps first:\n"
            f"{formatted}\n\n"
            f"Rerun plots: oa-da-plot-project-plots --project-dir {self.project_dir}\n"
            f"Rerun maps:  oa-da-plot-project-maps --project-dir {self.project_dir}"
        )


def _natural_sort_key(path: Path) -> tuple[object, ...]:
    parts = re.split(r"(\d+)", str(path))
    return tuple(int(part) if part.isdigit() else part for part in parts)


def _setup_id_from_project_dir(project_dir: Path) -> str:
    name = Path(project_dir).name
    if "_" in name:
        return name.split("_", 1)[1]
    return name


def _setup_weights_overview_paths(project_dir: Path) -> list[Path]:
    weights_dir = project_plots_root(project_dir) / "assim" / "weights"
    base = weights_dir / f"setup_weights_overview_{_setup_id_from_project_dir(project_dir)}.png"
    candidates = sorted(weights_dir.glob("setup_weights_overview*.png"), key=_natural_sort_key)
    if base in candidates:
        return [base, *(path for path in candidates if path != base)]
    return [base, *candidates]


def _da_map_path(project_dir: Path, index: int) -> Path:
    return project_maps_root(project_dir) / GENERATED_DA_MAPS_SUBDIR / f"da_{index}.png"


def _da_weights_path(project_dir: Path, index: int) -> Path:
    return project_plots_root(project_dir) / "assim" / "weights" / f"DA_{index:02d}_weights.png"


def _all_collection_pngs(project_dir: Path) -> list[Path]:
    roots = (project_plots_root(project_dir), project_maps_root(project_dir))
    paths: list[Path] = []
    for root in roots:
        if root.is_dir():
            paths.extend(path for path in root.rglob("*.png") if path.is_file())
    return sorted(paths, key=lambda path: _natural_sort_key(path.relative_to(project_dir)))


def collect_project_pdf_items(project_dir: Path) -> ProjectPdfPlan:
    project_dir = Path(project_dir)
    missing: list[Path] = []
    used: set[Path] = set()

    def require(path: Path) -> Path:
        if not path.is_file():
            missing.append(path)
        else:
            used.add(path.resolve())
        return path

    def optional(path: Path) -> Path | None:
        if not path.is_file():
            return None
        used.add(path.resolve())
        return path

    front_items: list[PdfImageItem] = []
    result_overview = require(project_result_overview_output_path(project_dir))
    front_items.append(PdfImageItem(result_overview, "result overview"))

    custom_overview = optional(project_result_overview_custom_output_path(project_dir))
    if custom_overview is not None:
        front_items.append(PdfImageItem(custom_overview, "custom result overview"))

    setup_overview = require(project_maps_root(project_dir) / "setup_overview.png")
    front_items.append(PdfImageItem(setup_overview, "setup overview map"))

    for idx, path in enumerate(_setup_weights_overview_paths(project_dir)):
        if idx == 0:
            require(path)
        else:
            optional(path)
        if path.is_file():
            front_items.append(PdfImageItem(path, "setup weights overview"))

    da_steps: list[PdfDaStepItem] = []
    for index, _event in enumerate(load_assimilation_events(project_dir), start=1):
        map_path = require(_da_map_path(project_dir, index))
        weights_path = require(_da_weights_path(project_dir, index))
        da_steps.append(PdfDaStepItem(index=index, map_path=map_path, weights_path=weights_path))

    appendix_items = [
        PdfImageItem(path, str(path.relative_to(project_dir)))
        for path in _all_collection_pngs(project_dir)
        if path.resolve() not in used
    ]

    return ProjectPdfPlan(
        front_items=tuple(front_items),
        da_steps=tuple(da_steps),
        appendix_items=tuple(appendix_items),
        missing_paths=tuple(sorted(set(missing), key=_natural_sort_key)),
    )


def _image_physical_size(path: Path) -> tuple[float, float]:
    with Image.open(path) as image:
        width, height = image.size
        dpi = image.info.get("dpi") or (DEFAULT_IMAGE_DPI, DEFAULT_IMAGE_DPI)
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height} for {path}")
    try:
        dpi_x = float(dpi[0])
        dpi_y = float(dpi[1])
    except Exception as exc:
        raise ValueError(f"Invalid PNG DPI metadata for {path}: {dpi!r}") from exc
    if dpi_x <= 0 or dpi_y <= 0:
        raise ValueError(f"PNG DPI metadata must be positive for {path}: {dpi!r}")
    return width / dpi_x, height / dpi_y


def _fit_rect(width: float, height: float, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, bottom, box_width, box_height = box
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    if width > box_width or height > box_height:
        raise ValueError(f"Image size {width}x{height} does not fit unscaled inside {box_width}x{box_height}")
    return (
        left + (box_width - width) / 2.0,
        bottom + (box_height - height) / 2.0,
        width,
        height,
    )


def _draw_image(fig: Figure, path: Path, rect: tuple[float, float, float, float]) -> None:
    image = mpimg.imread(path)
    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_axis_off()


def _write_single_image_page(pdf: PdfPages, item: PdfImageItem) -> None:
    import matplotlib.pyplot as plt

    page_width, page_height = _image_physical_size(item.path)
    fig = plt.figure(figsize=(page_width, page_height))
    _draw_image(fig, item.path, (0.0, 0.0, 1.0, 1.0))
    pdf.savefig(fig)
    plt.close(fig)


def write_project_pdf_plan(plan: ProjectPdfPlan, output: Path) -> Path:
    if plan.missing_paths:
        raise MissingProjectPdfArtifactsError(output, plan.missing_paths)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg", force=True)
    with PdfPages(output) as pdf:
        for item in plan.front_items:
            _write_single_image_page(pdf, item)
        for item in plan.da_steps:
            _write_single_image_page(pdf, PdfImageItem(item.map_path, f"DA {item.index} map"))
            _write_single_image_page(pdf, PdfImageItem(item.weights_path, f"DA {item.index} weights"))
        for item in plan.appendix_items:
            _write_single_image_page(pdf, item)
    if plan.page_count < 1:
        raise ValueError("No project PNG artifacts found for PDF collection")
    logger.info("Wrote project plots/maps collection PDF {} ({} page(s))", output, plan.page_count)
    return output


def build_project_collection_pdf(*, project_dir: Path, output: Path | None = None) -> Path:
    project_dir = Path(project_dir)
    output_path = Path(output) if output is not None else project_plots_maps_collection_pdf_path(project_dir)
    plan = collect_project_pdf_items(project_dir)
    if plan.missing_paths:
        raise MissingProjectPdfArtifactsError(project_dir, plan.missing_paths)
    return write_project_pdf_plan(plan, output_path)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oa-da-project-pdf",
        description="Assemble a source-size PDF collection from existing project plots and maps.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument("--output", type=Path, help="Output PDF path (default: <project>/results/reports/project_plots_maps_collection.pdf)")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    return parser.parse_args(argv)


def cli_main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    configure_cli_logger(args.log_level)
    try:
        output = build_project_collection_pdf(project_dir=args.project_dir, output=args.output)
    except Exception as exc:
        logger.error("Project PDF collection failed: {}", exc)
        return 1
    logger.info("Project PDF collection complete -> {}", output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
