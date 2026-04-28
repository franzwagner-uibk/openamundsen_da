from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from textwrap import wrap
from typing import Any, Iterable

import matplotlib
import matplotlib.image as mpimg
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from loguru import logger
from PIL import Image

from openamundsen_da.io.paths import (
    find_project_yaml,
    project_maps_root,
    project_plot_perf_dir,
    project_plots_maps_collection_pdf_path,
    project_plots_root,
    project_result_overview_custom_output_path,
    project_result_overview_output_path,
)
from openamundsen_da.methods.viz.maps.generated import GENERATED_DA_MAPS_SUBDIR
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.yaml_utils import read_yaml_mapping


A4_PORTRAIT_IN = (8.27, 11.69)
_NA = "n/a"
_WORKER_PATTERNS = (
    re.compile(r"max_workers=(\d+)"),
    re.compile(r"\bwith\s+(\d+)\s+worker\(s\)"),
    re.compile(r"\busing\s+(\d+)\s+worker\(s\)"),
)
_WALL_CLOCK_RE = re.compile(r"wall-clock\s+([0-9]+(?:\.[0-9]+)?)\s+s")


@dataclass(frozen=True)
class PdfImageItem:
    path: Path
    label: str


@dataclass(frozen=True)
class PdfDaStepItem:
    index: int
    map_path: Path


@dataclass(frozen=True)
class ProjectPdfPlan:
    project_dir: Path
    front_items: tuple[PdfImageItem, ...]
    da_steps: tuple[PdfDaStepItem, ...]
    appendix_items: tuple[PdfImageItem, ...]
    missing_paths: tuple[Path, ...]

    @property
    def page_count(self) -> int:
        return 1 + len(self.front_items) + len(self.da_steps) + len(self.appendix_items)


@dataclass(frozen=True)
class ProjectReportSection:
    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ProjectComputingCostStats:
    max_workers: int | None
    runtime_seconds: float | None
    runtime_source: str | None
    peak_cpu_pct: float | None
    peak_mem_used_gb: float | None
    peak_mem_used_pct: float | None
    mem_total_gb: float | None
    perf_sample_start: str | None
    perf_sample_end: str | None


@dataclass(frozen=True)
class ProjectReportSummary:
    project_dir: Path
    project_yaml: Path
    sections: tuple[ProjectReportSection, ...]


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


def _as_mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_value(value: object, *, max_items: int = 6) -> str:
    if value is None:
        return _NA
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        items = list(value)
        shown = ", ".join(_format_value(item, max_items=max_items) for item in items[:max_items])
        if len(items) > max_items:
            shown = f"{shown}, ... ({len(items)} total)" if shown else f"... ({len(items)} total)"
        return f"[{shown}]"
    return str(value)


def _format_mapping_values(
    mapping: dict[str, Any],
    keys: Iterable[str],
    *,
    labels: dict[str, str] | None = None,
) -> str:
    parts: list[str] = []
    for key in keys:
        if key not in mapping:
            continue
        label = labels.get(key, key) if labels is not None else key
        parts.append(f"{label}={_format_value(mapping[key])}")
    return ", ".join(parts) if parts else _NA


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return _NA
    return ", ".join(f"{name} x{count}" for name, count in counter.most_common())


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return _NA
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _format_float(value: float | None, *, unit: str = "", precision: int = 1) -> str:
    if value is None:
        return _NA
    suffix = unit if unit == "%" else f" {unit}" if unit else ""
    return f"{value:.{precision}f}{suffix}"


def _obs_line(label: str, cfg: dict[str, Any], *, include_product: bool = True) -> str:
    keys = ("product_tag", "dir", "summary_csv") if include_product else ("dir", "summary_csv")
    labels = {"product_tag": "product", "summary_csv": "summary"}
    return f"{label}: {_format_mapping_values(cfg, keys, labels=labels)}"


def _project_log_paths(project_dir: Path) -> tuple[Path, ...]:
    candidates: set[Path] = set(project_dir.glob("*.log"))
    for rel_dir in ("logs", "results/logs"):
        log_dir = project_dir / rel_dir
        if log_dir.is_dir():
            candidates.update(log_dir.glob("*.log"))
    return tuple(sorted(candidates, key=lambda path: (path.stat().st_mtime, str(path))))


def _read_project_log_stats(project_dir: Path) -> tuple[int | None, float | None]:
    max_workers: int | None = None
    wall_clock_seconds: float | None = None
    for log_path in _project_log_paths(project_dir):
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                for pattern in _WORKER_PATTERNS:
                    match = pattern.search(line)
                    if match:
                        value = int(match.group(1))
                        max_workers = value if max_workers is None else max(max_workers, value)
                runtime_match = _WALL_CLOCK_RE.search(line)
                if runtime_match:
                    wall_clock_seconds = float(runtime_match.group(1))
    return max_workers, wall_clock_seconds


def _max_numeric(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.max())


def _read_perf_stats(project_dir: Path) -> ProjectComputingCostStats:
    max_workers, log_runtime = _read_project_log_stats(project_dir)
    csv_path = project_plot_perf_dir(project_dir) / "project_perf_metrics.csv"
    if not csv_path.is_file():
        return ProjectComputingCostStats(
            max_workers=max_workers,
            runtime_seconds=log_runtime,
            runtime_source="project log" if log_runtime is not None else None,
            peak_cpu_pct=None,
            peak_mem_used_gb=None,
            peak_mem_used_pct=None,
            mem_total_gb=None,
            perf_sample_start=None,
            perf_sample_end=None,
        )

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        logger.warning("Could not read performance metrics CSV {}: {}", csv_path, exc)
        return ProjectComputingCostStats(
            max_workers=max_workers,
            runtime_seconds=log_runtime,
            runtime_source="project log" if log_runtime is not None else None,
            peak_cpu_pct=None,
            peak_mem_used_gb=None,
            peak_mem_used_pct=None,
            mem_total_gb=None,
            perf_sample_start=None,
            perf_sample_end=None,
        )

    runtime_seconds = log_runtime
    runtime_source = "project log" if log_runtime is not None else None
    sample_start: str | None = None
    sample_end: str | None = None
    if "timestamp" in df:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce").dropna().sort_values()
        if not timestamps.empty:
            first = timestamps.iloc[0]
            last = timestamps.iloc[-1]
            sample_start = first.strftime("%Y-%m-%d %H:%M:%S")
            sample_end = last.strftime("%Y-%m-%d %H:%M:%S")
            if runtime_seconds is None and len(timestamps) >= 2:
                runtime_seconds = float((last - first).total_seconds())
                runtime_source = "perf CSV span"

    return ProjectComputingCostStats(
        max_workers=max_workers,
        runtime_seconds=runtime_seconds,
        runtime_source=runtime_source,
        peak_cpu_pct=_max_numeric(df, "cpu_total_pct"),
        peak_mem_used_gb=_max_numeric(df, "mem_used_gb"),
        peak_mem_used_pct=_max_numeric(df, "mem_used_pct"),
        mem_total_gb=_max_numeric(df, "mem_total_gb"),
        perf_sample_start=sample_start,
        perf_sample_end=sample_end,
    )


def _computing_cost_lines(stats: ProjectComputingCostStats) -> tuple[str, ...]:
    runtime = _format_duration(stats.runtime_seconds)
    if stats.runtime_source:
        runtime = f"{runtime} ({stats.runtime_source})"
    worker_text = str(stats.max_workers) if stats.max_workers is not None else _NA
    lines = [
        f"Max workers/cores: {worker_text}",
        f"Runtime: {runtime}",
        f"Peak CPU: {_format_float(stats.peak_cpu_pct, unit='%', precision=1)}",
        (
            "Peak RAM: "
            f"{_format_float(stats.peak_mem_used_gb, unit='GB', precision=1)} "
            f"({_format_float(stats.peak_mem_used_pct, unit='%', precision=1)})"
        ),
        f"Total RAM: {_format_float(stats.mem_total_gb, unit='GB', precision=1)}",
    ]
    if stats.perf_sample_start and stats.perf_sample_end:
        lines.append(f"Perf samples: {stats.perf_sample_start} to {stats.perf_sample_end}")
    return tuple(lines)


def collect_project_report_summary(project_dir: Path) -> ProjectReportSummary:
    project_dir = Path(project_dir)
    project_yaml = find_project_yaml(project_dir)
    cfg = read_yaml_mapping(project_yaml, error_cls=RuntimeError, context="Project YAML root")
    da_cfg = _as_mapping(cfg.get("data_assimilation"))
    obs_cfg = _as_mapping(cfg.get("obs"))
    events = load_assimilation_events(project_dir)
    event_counter = Counter(event.variable for event in events)
    product_counter = Counter(event.product for event in events)
    first_event = events[0].date.isoformat() if events else _NA
    last_event = events[-1].date.isoformat() if events else _NA

    prior_cfg = _as_mapping(da_cfg.get("prior_forcing"))
    h_of_x_cfg = _as_mapping(da_cfg.get("h_of_x"))
    wet_snow_cfg = _as_mapping(da_cfg.get("wet_snow"))
    wsl_cfg = _as_mapping(da_cfg.get("wet_snow_line"))
    station_cfg = _as_mapping(da_cfg.get("station"))
    landcover_cfg = _as_mapping(da_cfg.get("landcover_mask"))
    likelihood_cfg = _as_mapping(da_cfg.get("likelihood"))
    uncertainty_cfg = _as_mapping(da_cfg.get("uncertainty"))
    resampling_cfg = _as_mapping(da_cfg.get("resampling"))
    rejuvenation_cfg = _as_mapping(da_cfg.get("rejuvenation"))
    restart_cfg = _as_mapping(da_cfg.get("restart"))
    output_cfg = _as_mapping(da_cfg.get("output"))
    benchmark_cfg = _as_mapping(da_cfg.get("benchmark"))

    likelihood_lines = []
    for key in ("scf", "wet_snow", "wet_snow_line"):
        likelihood_lines.append(
            f"{key}: "
            + _format_mapping_values(
                _as_mapping(likelihood_cfg.get(key)),
                (
                    "obs_sigma",
                    "min_sigma",
                    "sigma_floor",
                    "min_support_coverage_ratio",
                    "use_binomial",
                ),
                labels={"min_support_coverage_ratio": "min_support"},
            )
        )

    uncertainty_lines = []
    for key in ("scf", "wet_snow"):
        var_cfg = _as_mapping(uncertainty_cfg.get(key))
        assim_cfg = _as_mapping(var_cfg.get("assimilation"))
        assim_text = _format_mapping_values(
            assim_cfg,
            ("sigma_mode", "aggregate_metric"),
            labels={"aggregate_metric": "metric"},
        )
        uncertainty_lines.append(
            f"{key}: "
            + _format_mapping_values(var_cfg, ("enabled",), labels={"enabled": "enabled"})
            + f", {assim_text}"
        )

    sections = (
        ProjectReportSection(
            "Project",
            (
                f"Name: {project_dir.name}",
                f"YAML: {project_yaml.name}",
                f"Period: {_format_value(cfg.get('start_date'))} to {_format_value(cfg.get('end_date'))}",
                f"Run mode: {_format_value(cfg.get('run_mode'))}",
            ),
        ),
        ProjectReportSection(
            "DA Events",
            (
                f"Total: {len(events)}",
                f"By variable: {_format_counter(event_counter)}",
                f"Products: {_format_counter(product_counter)}",
                f"First/last: {first_event} to {last_event}",
            ),
        ),
        ProjectReportSection(
            "Observations",
            (
                _obs_line("Stations", _as_mapping(obs_cfg.get("stations")), include_product=False),
                _obs_line("Snow cover", _as_mapping(obs_cfg.get("snowcover"))),
                _obs_line("Wet snow", _as_mapping(obs_cfg.get("wetsnow"))),
            ),
        ),
        ProjectReportSection(
            "Core DA Settings",
            (
                "H(x): "
                + _format_mapping_values(h_of_x_cfg, ("method", "variable"))
                + f", params={_format_mapping_values(_as_mapping(h_of_x_cfg.get('params')), ('h0', 'k'))}",
                "Wet snow: "
                + _format_mapping_values(
                    wet_snow_cfg,
                    ("classification_threshold_percent",),
                    labels={"classification_threshold_percent": "threshold_pct"},
                ),
                "WSL: "
                + _format_mapping_values(
                    wsl_cfg,
                    ("elevation_band_size_m", "smoothing_window_bands", "crossing_fraction"),
                    labels={"elevation_band_size_m": "band_m", "smoothing_window_bands": "smooth_bands"},
                ),
                "WSL diagnostics: "
                + _format_mapping_values(
                    wsl_cfg,
                    ("wet_elevation_percentile", "aspect_diagnostics", "sector_relative_threshold"),
                    labels={"wet_elevation_percentile": "wet_pct", "sector_relative_threshold": "sector_threshold"},
                ),
                "Station: "
                + _format_mapping_values(
                    station_cfg,
                    ("default_station_uncertainty_pct", "min_station_uncertainty_pct", "single_station_factor"),
                    labels={
                        "default_station_uncertainty_pct": "default_unc_pct",
                        "min_station_uncertainty_pct": "min_unc_pct",
                        "single_station_factor": "single_factor",
                    },
                ),
                "Landcover mask: "
                + _format_mapping_values(
                    landcover_cfg,
                    ("enabled", "classes_to_exclude"),
                    labels={"classes_to_exclude": "exclude"},
                ),
            ),
        ),
        ProjectReportSection(
            "Ensemble And Filter",
            (
                "Prior forcing: "
                + _format_mapping_values(prior_cfg, ("ensemble_size", "random_seed"), labels={"random_seed": "seed"}),
                "Prior sigmas: "
                + _format_mapping_values(prior_cfg, ("sigma_t", "mu_p", "sigma_p", "sigma_rh", "sigma_sw")),
                "Resampling: "
                + _format_mapping_values(
                    resampling_cfg,
                    ("algorithm", "ess_threshold_ratio", "seed"),
                    labels={"ess_threshold_ratio": "ess_ratio"},
                ),
                "Rejuvenation: "
                + _format_mapping_values(
                    rejuvenation_cfg,
                    ("sigma_t", "sigma_p", "sigma_rh", "sigma_sw", "seed", "rebase_open_loop"),
                ),
                "Restart: " + _format_mapping_values(restart_cfg, ("use_state", "dump_state", "state_pattern")),
                "Output: " + _format_mapping_values(output_cfg, ("retention",)),
                "Benchmark: "
                + _format_mapping_values(
                    benchmark_cfg,
                    ("independent_variables", "performance_scores_exclude_variables", "score_station_sigma_threshold"),
                    labels={
                        "performance_scores_exclude_variables": "score_exclude",
                        "score_station_sigma_threshold": "station_sigma_threshold",
                    },
                ),
            ),
        ),
        ProjectReportSection("Likelihood", tuple(likelihood_lines)),
        ProjectReportSection("Uncertainty", tuple(uncertainty_lines)),
        ProjectReportSection("Computing Cost", _computing_cost_lines(_read_perf_stats(project_dir))),
    )
    return ProjectReportSummary(project_dir=project_dir, project_yaml=project_yaml, sections=sections)


def _da_map_path(project_dir: Path, index: int) -> Path:
    return project_maps_root(project_dir) / GENERATED_DA_MAPS_SUBDIR / f"da_{index}.png"


def collect_project_pdf_items(project_dir: Path) -> ProjectPdfPlan:
    project_dir = Path(project_dir)
    missing: list[Path] = []

    def require(path: Path) -> Path:
        if not path.is_file():
            missing.append(path)
        return path

    def optional(path: Path) -> Path | None:
        if not path.is_file():
            return None
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
        da_steps.append(PdfDaStepItem(index=index, map_path=map_path))

    return ProjectPdfPlan(
        project_dir=project_dir,
        front_items=tuple(front_items),
        da_steps=tuple(da_steps),
        appendix_items=(),
        missing_paths=tuple(sorted(set(missing), key=_natural_sort_key)),
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height} for {path}")
    return int(width), int(height)


def _fit_rect(width: int, height: int, box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    left, bottom, box_width, box_height = box
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")
    scale = min(box_width / float(width), box_height / float(height))
    display_width = float(width) * scale
    display_height = float(height) * scale
    return (
        left + (box_width - display_width) / 2.0,
        bottom + box_height - display_height,
        display_width,
        display_height,
    )


def _draw_image(fig: Figure, path: Path, rect: tuple[float, float, float, float]) -> None:
    image = mpimg.imread(path)
    ax = fig.add_axes(rect)
    ax.imshow(image)
    ax.set_axis_off()


def _write_single_image_page(pdf: PdfPages, item: PdfImageItem) -> None:
    import matplotlib.pyplot as plt

    page_width, page_height = A4_PORTRAIT_IN
    margin = 0.25
    fig = plt.figure(figsize=(page_width, page_height))
    width, height = _image_size(item.path)
    left, bottom, display_width, display_height = _fit_rect(
        width,
        height,
        (margin, margin, page_width - 2 * margin, page_height - 2 * margin),
    )
    _draw_image(
        fig,
        item.path,
        (left / page_width, bottom / page_height, display_width / page_width, display_height / page_height),
    )
    pdf.savefig(fig)
    plt.close(fig)


def _wrapped_lines(lines: Iterable[str], *, width: int, max_lines: int) -> list[str]:
    rendered: list[str] = []
    for line in lines:
        pieces = wrap(line, width=width, break_long_words=False, break_on_hyphens=False) or [""]
        for piece in pieces:
            if len(rendered) >= max_lines:
                rendered[-1] = f"{rendered[-1].rstrip()} ..."
                return rendered
            rendered.append(piece)
    return rendered


def _draw_section(
    ax: matplotlib.axes.Axes,
    *,
    title: str,
    lines: Iterable[str],
    x: float,
    y: float,
    width: float,
    max_lines: int,
) -> float:
    line_height = 0.0175
    section_gap = 0.013
    title_y = y
    ax.text(x, title_y, title.upper(), fontsize=7.7, fontweight="bold", color="#263238", va="top")
    ax.plot([x, x + width], [title_y - 0.010, title_y - 0.010], color="#90A4AE", linewidth=0.6)
    current_y = title_y - 0.022
    for line in _wrapped_lines(lines, width=62, max_lines=max_lines):
        ax.text(x, current_y, line, fontsize=6.9, color="#263238", va="top", family="DejaVu Sans")
        current_y -= line_height
    return current_y - section_gap


def _write_project_summary_page(pdf: PdfPages, summary: ProjectReportSummary) -> None:
    import matplotlib.pyplot as plt

    page_width, page_height = A4_PORTRAIT_IN
    fig = plt.figure(figsize=(page_width, page_height))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    ax.text(
        0.065,
        0.955,
        "openAMUNDSEN-DA project report",
        fontsize=14,
        fontweight="bold",
        color="#102027",
        va="top",
    )
    ax.text(
        0.065,
        0.928,
        f"{summary.project_dir.name} | first page: YAML highlights and computing cost",
        fontsize=8.3,
        color="#455A64",
        va="top",
    )
    ax.plot([0.065, 0.935], [0.910, 0.910], color="#607D8B", linewidth=1.0)

    left_sections = summary.sections[:4]
    right_sections = summary.sections[4:]
    left_limits = (5, 6, 8, 13)
    right_limits = (12, 6, 5, 7)
    y_left = 0.885
    y_right = 0.885
    for section, max_lines in zip(left_sections, left_limits, strict=True):
        y_left = _draw_section(
            ax,
            title=section.title,
            lines=section.lines,
            x=0.065,
            y=y_left,
            width=0.405,
            max_lines=max_lines,
        )
    for section, max_lines in zip(right_sections, right_limits, strict=True):
        y_right = _draw_section(
            ax,
            title=section.title,
            lines=section.lines,
            x=0.530,
            y=y_right,
            width=0.405,
            max_lines=max_lines,
        )

    ax.text(
        0.065,
        0.038,
        "Generated from project YAML, project logs, and results/plots/perf/project_perf_metrics.csv when available.",
        fontsize=6.8,
        color="#607D8B",
        va="bottom",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_project_pdf_plan(plan: ProjectPdfPlan, output: Path) -> Path:
    if plan.missing_paths:
        raise MissingProjectPdfArtifactsError(plan.project_dir, plan.missing_paths)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg", force=True)
    with PdfPages(output) as pdf:
        _write_project_summary_page(pdf, collect_project_report_summary(plan.project_dir))
        for item in plan.front_items:
            _write_single_image_page(pdf, item)
        for item in plan.da_steps:
            _write_single_image_page(pdf, PdfImageItem(item.map_path, f"DA {item.index} map"))
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
        description="Assemble a DIN A4 PDF collection from curated project plots and maps.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PDF path (default: <project>/results/reports/project_plots_maps_collection.pdf)",
    )
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
