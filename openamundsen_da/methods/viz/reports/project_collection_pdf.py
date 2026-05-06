from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
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
    find_setup_yaml,
    infer_setup_dir_from_project,
    project_maps_root,
    project_plot_assim_scores_dir,
    project_plot_perf_dir,
    project_plot_points_dir,
    project_plots_maps_collection_pdf_path,
    project_plots_root,
    project_result_overview_custom_output_path,
    project_result_overview_output_path,
)
from openamundsen_da.methods.viz.maps.generated import GENERATED_DA_MAPS_SUBDIR
from openamundsen_da.methods.viz.theme import EXPORT_DPI
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.yaml_utils import read_yaml_mapping


A4_PORTRAIT_IN = (8.27, 11.69)
IMAGE_TOP_MARGIN_IN = 0.65
IMAGE_BOTTOM_GAP_IN = 0.55
IMAGE_ROW_GAP_IN = 0.18
_NA = "n/a"
_WORKER_PATTERNS = (
    re.compile(r"max_workers=(\d+)"),
    re.compile(r"\bwith\s+(\d+)\s+worker\(s\)"),
    re.compile(r"\busing\s+(\d+)\s+worker\(s\)"),
)
_WALL_CLOCK_RE = re.compile(r"wall-clock\s+([0-9]+(?:\.[0-9]+)?)\s+s")
_SUMMARY_FULL_BOLD_PREFIXES = ("Run mode:", "Total:", "By variable:")
_SUMMARY_BOLD_VALUE_RE = re.compile(
    r"(resolution=[^,]+|timestep=[^,]+|ensemble_size=[^,]+|ess_ratio=[^,]+)"
)


@dataclass(frozen=True)
class PdfImageItem:
    path: Path
    label: str


@dataclass(frozen=True)
class PdfDaStepItem:
    index: int
    map_path: Path


@dataclass(frozen=True)
class PdfSection:
    title: str
    start_page: int
    end_page: int


@dataclass(frozen=True)
class ProjectPdfPlan:
    project_dir: Path
    front_items: tuple[PdfImageItem, ...]
    station_snow_depth_items: tuple[PdfImageItem, ...]
    performance_scores_item: PdfImageItem | None
    project_perf_item: PdfImageItem | None
    da_steps: tuple[PdfDaStepItem, ...]
    appendix_items: tuple[PdfImageItem, ...]
    missing_paths: tuple[Path, ...]

    @property
    def page_count(self) -> int:
        return _plan_page_count(self)


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


def _wet_snow_classification_summary(wet_snow_cfg: dict[str, Any]) -> str:
    method = wet_snow_cfg.get("classification_method")
    if method is None and "classification_threshold_percent" in wet_snow_cfg:
        method = "liquid_water_fraction"
    parts = [f"method={_format_value(method)}"]
    method_text = str(method).strip() if method is not None else ""
    if method_text == "liquid_water_amount":
        threshold = wet_snow_cfg.get("liquid_water_amount_threshold_mm")
        parts.append(f"threshold_abs_mm={_format_value(threshold)}")
    elif method_text == "liquid_water_fraction":
        threshold = wet_snow_cfg.get("classification_threshold_percent")
        parts.append(f"threshold_pct={_format_value(threshold)}")
    else:
        for key, label in (
            ("liquid_water_amount_threshold_mm", "threshold_abs_mm"),
            ("classification_threshold_percent", "threshold_pct"),
        ):
            if key in wet_snow_cfg:
                parts.append(f"{label}={_format_value(wet_snow_cfg[key])}")
    return ", ".join(parts)


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
    worker_text = str(stats.max_workers) if stats.max_workers is not None else _NA
    return (
        f"Max workers/cores: {worker_text}",
        f"Runtime: {runtime}",
        f"Peak CPU: {_format_float(stats.peak_cpu_pct, unit='%', precision=1)}",
        (
            "Peak RAM: "
            f"{_format_float(stats.peak_mem_used_gb, unit='GB', precision=1)} "
            f"({_format_float(stats.peak_mem_used_pct, unit='%', precision=1)})"
        ),
        f"Total RAM: {_format_float(stats.mem_total_gb, unit='GB', precision=1)}",
    )


def _setup_yaml_summary_section(project_dir: Path) -> ProjectReportSection:
    try:
        setup_dir = infer_setup_dir_from_project(project_dir)
        setup_yaml = find_setup_yaml(setup_dir)
        cfg = read_yaml_mapping(setup_yaml, error_cls=RuntimeError, context="Setup YAML root")
    except Exception as exc:
        logger.warning("Could not read setup YAML for report summary: {}", exc)
        return ProjectReportSection("openAMUNDSEN Setup", ("Setup YAML: n/a",))

    meteo_cfg = _as_mapping(cfg.get("meteo"))
    interp_cfg = _as_mapping(meteo_cfg.get("interpolation"))
    snow_cfg = _as_mapping(cfg.get("snow"))
    lwc_cfg = _as_mapping(snow_cfg.get("liquid_water_content"))
    melt_cfg = _as_mapping(snow_cfg.get("melt"))
    canopy_cfg = _as_mapping(cfg.get("canopy"))

    interpolation_methods = []
    for key, label in (
        ("temperature", "temp"),
        ("precipitation", "precip"),
        ("humidity", "humidity"),
        ("wind_speed", "wind"),
    ):
        method = _as_mapping(interp_cfg.get(key)).get("trend_method")
        if method is not None:
            interpolation_methods.append(f"{label}={_format_value(method)}")
    cloud_cfg = _as_mapping(interp_cfg.get("cloudiness"))
    if cloud_cfg:
        day = cloud_cfg.get("day_method")
        night = cloud_cfg.get("night_method")
        interpolation_methods.append(f"cloud={_format_value(day)}/{_format_value(night)}")

    precip_correction_methods = []
    for item in meteo_cfg.get("precipitation_correction") or []:
        if not isinstance(item, dict):
            continue
        method = item.get("method")
        if method is None:
            continue
        details = []
        if item.get("gauge") is not None:
            details.append(f"gauge={_format_value(item.get('gauge'))}")
        suffix = f" ({', '.join(details)})" if details else ""
        precip_correction_methods.append(f"{_format_value(method)}{suffix}")

    return ProjectReportSection(
        "openAMUNDSEN Setup",
        (
            f"Setup YAML: {setup_yaml.name}",
            (
                "Domain: "
                f"{_format_value(cfg.get('domain'))}, resolution={_format_value(cfg.get('resolution'))} m, "
                f"timestep={_format_value(cfg.get('timestep'))}, CRS={_format_value(cfg.get('crs'))}"
            ),
            f"Meteo interpolation: {', '.join(interpolation_methods) if interpolation_methods else _NA}",
            f"Precip correction: {', '.join(precip_correction_methods) if precip_correction_methods else _NA}",
            f"Snow model: {_format_value(snow_cfg.get('model'))}, melt={_format_value(melt_cfg.get('method'))}",
            "Liquid water content: "
            + _format_mapping_values(lwc_cfg, ("method", "max"), labels={"max": "max"}),
            f"Canopy enabled: {_format_value(canopy_cfg.get('enabled'))}",
        ),
    )


def collect_project_report_summary(project_dir: Path) -> ProjectReportSummary:
    project_dir = Path(project_dir)
    project_yaml = find_project_yaml(project_dir)
    cfg = read_yaml_mapping(project_yaml, error_cls=RuntimeError, context="Project YAML root")
    da_cfg = _as_mapping(cfg.get("data_assimilation"))
    events = load_assimilation_events(project_dir)
    event_counter = Counter(event.variable for event in events)

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
                    "min_model_finite_fraction",
                    "min_wet_pixels_total",
                    "min_wet_bands",
                ),
                labels={
                    "min_support_coverage_ratio": "min_support",
                    "min_model_finite_fraction": "min_model_finite",
                    "min_wet_pixels_total": "min_wet_px",
                },
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

    sections: list[ProjectReportSection] = [
        ProjectReportSection(
            "Project",
            (
                f"Name: {project_dir.name}",
                f"YAML: {project_yaml.name}",
                f"Period: {_format_value(cfg.get('start_date'))} to {_format_value(cfg.get('end_date'))}",
                f"Run mode: {_format_value(cfg.get('run_mode'))}",
            ),
        ),
        _setup_yaml_summary_section(project_dir),
        ProjectReportSection(
            "DA Events",
            (
                f"Total: {len(events)}",
                f"By variable: {_format_counter(event_counter)}",
            ),
        ),
        ProjectReportSection(
            "Core DA Settings",
            (
                "H(x): "
                + _format_mapping_values(h_of_x_cfg, ("method", "variable"))
                + f", params={_format_mapping_values(_as_mapping(h_of_x_cfg.get('params')), ('h0', 'k'))}",
                "Wet snow classification: " + _wet_snow_classification_summary(wet_snow_cfg),
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
    ]
    subdomain_section = _subdomain_summary_section(project_dir)
    if subdomain_section is not None:
        sections.append(subdomain_section)
    sections.append(ProjectReportSection("Computing Cost", _computing_cost_lines(_read_perf_stats(project_dir))))
    return ProjectReportSummary(project_dir=project_dir, project_yaml=project_yaml, sections=tuple(sections))


def _subdomain_summary_section(project_dir: Path) -> ProjectReportSection | None:
    results_dir = Path(project_dir) / "results"
    overview_path = results_dir / "subdomain_overview.csv"
    aggregate_path = results_dir / "subdomain_assimilation_aggregate.csv"
    dropped_path = results_dir / "subdomain_dropped_events.csv"
    if not overview_path.is_file():
        return None
    try:
        overview = pd.read_csv(overview_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read sub-domain overview for report summary: {}", exc)
        return None

    lines: list[str] = []
    if "status" in overview.columns:
        status_counts = Counter(str(value) for value in overview["status"].dropna())
        lines.append(f"Statuses: {_format_counter(status_counts)}")
    lines.append(f"Subdomains: {len(overview)}")
    if "duration_seconds" in overview.columns:
        duration = pd.to_numeric(overview["duration_seconds"], errors="coerce")
        if duration.notna().any():
            lines.append(f"Slowest subdomain: {_format_duration(float(duration.max()))}")
    if aggregate_path.is_file():
        try:
            aggregate = pd.read_csv(aggregate_path)
            if "ess_norm_mean" in aggregate.columns:
                ess_mean = pd.to_numeric(aggregate["ess_norm_mean"], errors="coerce")
                if ess_mean.notna().any():
                    lines.append(f"Mean ESS/n range: {ess_mean.min():.3f} to {ess_mean.max():.3f}")
            if {"subdomain_id", "ess_norm_min"}.issubset(aggregate.columns):
                ess_min = pd.to_numeric(aggregate["ess_norm_min"], errors="coerce")
                if ess_min.notna().any():
                    idx = ess_min.idxmin()
                    lines.append(
                        f"Weakest ESS/n: {aggregate.loc[idx, 'subdomain_id']} = {float(ess_min.loc[idx]):.3f}"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read sub-domain aggregate for report summary: {}", exc)
    if dropped_path.is_file():
        try:
            dropped = pd.read_csv(dropped_path)
            lines.append(f"Dropped subdomain events: {len(dropped)}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read dropped sub-domain events for report summary: {}", exc)
    return ProjectReportSection("Subdomains", tuple(lines))


def _da_map_path(project_dir: Path, index: int) -> Path:
    return project_maps_root(project_dir) / GENERATED_DA_MAPS_SUBDIR / f"da_{index}.png"


def _station_snow_depth_plot_paths(project_dir: Path) -> list[Path]:
    points_dir = project_plot_points_dir(project_dir)
    if not points_dir.is_dir():
        return []
    return [
        path
        for path in sorted(points_dir.glob("*snow_depth*.png"), key=_natural_sort_key)
        if "_roi_" not in path.name
    ]


def collect_project_pdf_items(project_dir: Path) -> ProjectPdfPlan:
    project_dir = Path(project_dir)
    missing: list[Path] = []

    def optional(path: Path) -> Path | None:
        if not path.is_file():
            return None
        return path

    front_items: list[PdfImageItem] = []
    result_overview = optional(project_result_overview_output_path(project_dir))
    if result_overview is not None:
        front_items.append(PdfImageItem(result_overview, "result overview"))

    custom_overview = optional(project_result_overview_custom_output_path(project_dir))
    if custom_overview is not None:
        front_items.append(PdfImageItem(custom_overview, "custom result overview"))

    setup_overview = optional(project_maps_root(project_dir) / "setup_overview.png")
    if setup_overview is not None:
        front_items.append(PdfImageItem(setup_overview, "setup overview map"))

    for idx, path in enumerate(_setup_weights_overview_paths(project_dir)):
        if path.is_file():
            front_items.append(PdfImageItem(path, "setup weights overview"))

    station_snow_depth_items = tuple(
        PdfImageItem(path, "station snow-depth plots")
        for path in _station_snow_depth_plot_paths(project_dir)
    )
    performance_scores_path = optional(project_plot_assim_scores_dir(project_dir) / "performance_scores.png")
    performance_scores_item = (
        PdfImageItem(performance_scores_path, "performance scores")
        if performance_scores_path is not None
        else None
    )
    project_perf_path = optional(project_plot_perf_dir(project_dir) / "project_perf.png")
    project_perf_item = PdfImageItem(project_perf_path, "project performance") if project_perf_path is not None else None

    da_steps: list[PdfDaStepItem] = []
    for index, _event in enumerate(load_assimilation_events(project_dir), start=1):
        map_path = _da_map_path(project_dir, index)
        if map_path.is_file():
            da_steps.append(PdfDaStepItem(index=index, map_path=map_path))
        else:
            missing.append(map_path)

    return ProjectPdfPlan(
        project_dir=project_dir,
        front_items=tuple(front_items),
        station_snow_depth_items=station_snow_depth_items,
        performance_scores_item=performance_scores_item,
        project_perf_item=project_perf_item,
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


def _image_size_inches(path: Path) -> tuple[float, float]:
    width, height = _image_size(path)
    return width / float(EXPORT_DPI), height / float(EXPORT_DPI)


def _image_height_inches(path: Path) -> float:
    if not path.is_file():
        return 0.0
    _width, height = _image_size_inches(path)
    return height


def _da_step_page_groups(da_steps: Iterable[PdfDaStepItem]) -> tuple[tuple[PdfDaStepItem, ...], ...]:
    steps = list(da_steps)
    groups: list[tuple[PdfDaStepItem, ...]] = []
    current: list[PdfDaStepItem] = []
    current_height = 0.0
    available_height = A4_PORTRAIT_IN[1] - IMAGE_TOP_MARGIN_IN - IMAGE_BOTTOM_GAP_IN
    for item in steps:
        item_height = _image_height_inches(item.map_path)
        candidate_height = item_height if not current else current_height + IMAGE_ROW_GAP_IN + item_height
        if current and candidate_height > available_height:
            groups.append(tuple(current))
            current = [item]
            current_height = item_height
        else:
            current.append(item)
            current_height = candidate_height
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def _plan_page_count(plan: ProjectPdfPlan) -> int:
    return (
        1
        + len(plan.front_items)
        + (1 if plan.station_snow_depth_items else 0)
        + (1 if plan.performance_scores_item is not None else 0)
        + (1 if plan.project_perf_item is not None else 0)
        + len(_da_step_page_groups(plan.da_steps))
        + len(plan.appendix_items)
    )


def _format_page_range(start_page: int, end_page: int) -> str:
    if start_page == end_page:
        return str(start_page)
    return f"{start_page}-{end_page}"


def _project_pdf_sections(plan: ProjectPdfPlan) -> tuple[PdfSection, ...]:
    sections: list[PdfSection] = []
    page = 1

    def add(title: str, count: int) -> None:
        nonlocal page
        if count < 1:
            return
        start = page
        end = page + count - 1
        sections.append(PdfSection(title=title, start_page=start, end_page=end))
        page = end + 1

    add("Project summary and setup", 1)
    idx = 0
    while idx < len(plan.front_items):
        label = plan.front_items[idx].label
        count = 1
        idx += 1
        while idx < len(plan.front_items) and plan.front_items[idx].label == label:
            count += 1
            idx += 1
        add(label, count)
    add("station snow-depth plots", 1 if plan.station_snow_depth_items else 0)
    add("performance scores", 1 if plan.performance_scores_item is not None else 0)
    add("project performance", 1 if plan.project_perf_item is not None else 0)
    add("DA-event maps", len(_da_step_page_groups(plan.da_steps)))
    for item in plan.appendix_items:
        add(item.label, 1)
    return tuple(sections)


def _draw_image(fig: Figure, path: Path, rect: tuple[float, float, float, float]) -> None:
    image = mpimg.imread(path)
    ax = fig.add_axes(rect)
    ax.imshow(image, interpolation="none", resample=False)
    ax.set_axis_off()


def _save_pdf_page(pdf: PdfPages, fig: Figure) -> None:
    pdf.savefig(fig, dpi=EXPORT_DPI)


def _draw_page_number(fig: Figure, *, page_number: int, total_pages: int) -> None:
    fig.text(
        0.5,
        0.018,
        f"{page_number} / {total_pages}",
        ha="center",
        va="bottom",
        fontsize=7.0,
        color="#607D8B",
    )


def _draw_image_at_original_size(fig: Figure, path: Path, *, left: float, top: float) -> None:
    page_width, page_height = fig.get_size_inches()
    display_width, display_height = _image_size_inches(path)
    bottom = top - display_height
    _draw_image(
        fig,
        path,
        (left / page_width, bottom / page_height, display_width / page_width, display_height / page_height),
    )


def _write_single_image_page(pdf: PdfPages, item: PdfImageItem, *, page_number: int, total_pages: int) -> None:
    import matplotlib.pyplot as plt

    page_width, page_height = A4_PORTRAIT_IN
    fig = plt.figure(figsize=(page_width, page_height))
    display_width, _display_height = _image_size_inches(item.path)
    _draw_image_at_original_size(
        fig,
        item.path,
        left=(page_width - display_width) / 2.0,
        top=page_height - IMAGE_TOP_MARGIN_IN,
    )
    _draw_page_number(fig, page_number=page_number, total_pages=total_pages)
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _write_image_group_page(
    pdf: PdfPages,
    items: Iterable[PdfImageItem],
    *,
    page_number: int,
    total_pages: int,
) -> None:
    import matplotlib.pyplot as plt

    page_width, page_height = A4_PORTRAIT_IN
    fig = plt.figure(figsize=(page_width, page_height))
    row_top = page_height - IMAGE_TOP_MARGIN_IN
    for item in items:
        display_width, display_height = _image_size_inches(item.path)
        _draw_image_at_original_size(
            fig,
            item.path,
            left=(page_width - display_width) / 2.0,
            top=row_top,
        )
        row_top -= display_height + IMAGE_ROW_GAP_IN
    _draw_page_number(fig, page_number=page_number, total_pages=total_pages)
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def _write_da_steps_pages(
    pdf: PdfPages,
    da_steps: Iterable[PdfDaStepItem],
    *,
    start_page_number: int,
    total_pages: int,
) -> int:
    import matplotlib.pyplot as plt

    page_width, page_height = A4_PORTRAIT_IN

    page_number = start_page_number
    for group in _da_step_page_groups(da_steps):
        fig = plt.figure(figsize=(page_width, page_height))
        row_top = page_height - IMAGE_TOP_MARGIN_IN
        for item in group:
            display_width, display_height = _image_size_inches(item.map_path)
            _draw_image_at_original_size(
                fig,
                item.map_path,
                left=(page_width - display_width) / 2.0,
                top=row_top,
            )
            row_top -= display_height + IMAGE_ROW_GAP_IN
        _draw_page_number(fig, page_number=page_number, total_pages=total_pages)
        _save_pdf_page(pdf, fig)
        plt.close(fig)
        page_number += 1
    return page_number


def _wrapped_line_sources(
    lines: Iterable[str],
    *,
    width: int,
    max_lines: int | None,
) -> list[tuple[str, str]]:
    rendered: list[tuple[str, str]] = []
    for line in lines:
        pieces = wrap(line, width=width, break_long_words=False, break_on_hyphens=False) or [""]
        for piece in pieces:
            if max_lines is not None and len(rendered) >= max_lines:
                prev_piece, prev_source = rendered[-1]
                rendered[-1] = (f"{prev_piece.rstrip()} ...", prev_source)
                return rendered
            rendered.append((piece, line))
    return rendered


def _summary_line_segments(text: str, *, source: str) -> tuple[tuple[str, bool], ...]:
    if source.startswith(_SUMMARY_FULL_BOLD_PREFIXES):
        return ((text, True),)

    segments: list[tuple[str, bool]] = []
    pos = 0
    for match in _SUMMARY_BOLD_VALUE_RE.finditer(text):
        if match.start() > pos:
            segments.append((text[pos : match.start()], False))
        segments.append((match.group(0), True))
        pos = match.end()
    if pos < len(text):
        segments.append((text[pos:], False))
    return tuple(segments) if segments else ((text, False),)


def _data_dx_from_pixel_width(ax: matplotlib.axes.Axes, width_px: float) -> float:
    origin_px = ax.transData.transform((0.0, 0.0))
    shifted_data = ax.transData.inverted().transform((origin_px[0] + width_px, origin_px[1]))
    return float(shifted_data[0])


def _draw_summary_line(
    ax: matplotlib.axes.Axes,
    *,
    x: float,
    y: float,
    text: str,
    source: str,
    fontsize: float,
    color: str,
) -> None:
    current_x = x
    for segment, is_bold in _summary_line_segments(text, source=source):
        if not segment:
            continue
        artist = ax.text(
            current_x,
            y,
            segment,
            fontsize=fontsize,
            color=color,
            va="top",
            family="DejaVu Sans",
            fontweight="bold" if is_bold else "normal",
        )
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        current_x += _data_dx_from_pixel_width(ax, artist.get_window_extent(renderer=renderer).width)


def _draw_section(
    ax: matplotlib.axes.Axes,
    *,
    title: str,
    lines: Iterable[str],
    x: float,
    y: float,
    width: float,
    max_lines: int | None = None,
) -> float:
    line_height = 0.0175
    section_gap = 0.013
    title_y = y
    ax.text(x, title_y, title.upper(), fontsize=7.7, fontweight="bold", color="#263238", va="top")
    ax.plot([x, x + width], [title_y - 0.010, title_y - 0.010], color="#90A4AE", linewidth=0.6)
    current_y = title_y - 0.022
    for line, source in _wrapped_line_sources(lines, width=62, max_lines=max_lines):
        _draw_summary_line(
            ax,
            x=x,
            y=current_y,
            text=line,
            source=source,
            fontsize=6.9,
            color="#263238",
        )
        current_y -= line_height
    return current_y - section_gap


def _content_rows(toc_sections: Iterable[PdfSection]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (_format_page_range(section.start_page, section.end_page), section.title)
        for section in toc_sections
    )


def _draw_content_section(
    ax: matplotlib.axes.Axes,
    *,
    rows: Iterable[tuple[str, str]],
    x: float,
    y: float,
    width: float,
) -> None:
    line_height = 0.0175
    title_y = y
    ax.text(x, title_y, "CONTENT", fontsize=7.7, fontweight="bold", color="#263238", va="top")
    ax.plot([x, x + width], [title_y - 0.010, title_y - 0.010], color="#90A4AE", linewidth=0.6)

    page_x = x
    name_x = x + 0.085
    current_y = title_y - 0.025
    ax.text(page_x, current_y, "Page", fontsize=6.7, fontweight="bold", color="#455A64", va="top")
    ax.text(name_x, current_y, "Section", fontsize=6.7, fontweight="bold", color="#455A64", va="top")
    current_y -= line_height
    for page_range, title in rows:
        ax.text(page_x, current_y, page_range, fontsize=6.9, color="#263238", va="top", family="DejaVu Sans")
        ax.text(name_x, current_y, title, fontsize=6.9, color="#263238", va="top", family="DejaVu Sans")
        current_y -= line_height


def _write_project_summary_page(
    pdf: PdfPages,
    summary: ProjectReportSummary,
    *,
    toc_sections: tuple[PdfSection, ...],
    page_number: int,
    total_pages: int,
) -> None:
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
        summary.project_dir.name,
        fontsize=8.3,
        color="#455A64",
        va="top",
    )
    ax.plot([0.065, 0.935], [0.910, 0.910], color="#607D8B", linewidth=1.0)

    left_sections = summary.sections[:4]
    right_sections = summary.sections[4:]
    y_left = 0.885
    y_right = 0.885
    for section in left_sections:
        y_left = _draw_section(
            ax,
            title=section.title,
            lines=section.lines,
            x=0.065,
            y=y_left,
            width=0.405,
            max_lines=None,
        )
    for section in right_sections:
        y_right = _draw_section(
            ax,
            title=section.title,
            lines=section.lines,
            x=0.530,
            y=y_right,
            width=0.405,
            max_lines=None,
        )

    _draw_content_section(
        ax,
        rows=_content_rows(toc_sections),
        x=0.065,
        y=0.265,
        width=0.870,
    )
    ax.text(
        0.065,
        0.038,
        "Generated from project YAML, project logs, and results/plots/perf/project_perf_metrics.csv when available.",
        fontsize=6.8,
        color="#607D8B",
        va="bottom",
    )
    _draw_page_number(fig, page_number=page_number, total_pages=total_pages)
    _save_pdf_page(pdf, fig)
    plt.close(fig)


def write_project_pdf_plan(plan: ProjectPdfPlan, output: Path) -> Path:
    if plan.missing_paths:
        logger.info(
            "Project report omitting {} missing optional artifact(s)",
            len(plan.missing_paths),
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matplotlib.use("Agg", force=True)
    total_pages = plan.page_count
    toc_sections = _project_pdf_sections(plan)
    page_number = 1
    with PdfPages(output) as pdf:
        _write_project_summary_page(
            pdf,
            collect_project_report_summary(plan.project_dir),
            toc_sections=toc_sections,
            page_number=page_number,
            total_pages=total_pages,
        )
        page_number += 1
        for item in plan.front_items:
            _write_single_image_page(pdf, item, page_number=page_number, total_pages=total_pages)
            page_number += 1
        if plan.station_snow_depth_items:
            _write_image_group_page(
                pdf,
                plan.station_snow_depth_items,
                page_number=page_number,
                total_pages=total_pages,
            )
            page_number += 1
        if plan.performance_scores_item is not None:
            _write_single_image_page(
                pdf,
                plan.performance_scores_item,
                page_number=page_number,
                total_pages=total_pages,
            )
            page_number += 1
        if plan.project_perf_item is not None:
            _write_single_image_page(
                pdf,
                plan.project_perf_item,
                page_number=page_number,
                total_pages=total_pages,
            )
            page_number += 1
        page_number = _write_da_steps_pages(
            pdf,
            plan.da_steps,
            start_page_number=page_number,
            total_pages=total_pages,
        )
        for item in plan.appendix_items:
            _write_single_image_page(pdf, item, page_number=page_number, total_pages=total_pages)
            page_number += 1
    if plan.page_count < 1:
        raise ValueError("No project PNG artifacts found for PDF collection")
    logger.info("Wrote project plots/maps collection PDF {} ({} page(s))", output, plan.page_count)
    return output


def build_project_collection_pdf(*, project_dir: Path, output: Path | None = None) -> Path:
    project_dir = Path(project_dir)
    output_path = Path(output) if output is not None else project_plots_maps_collection_pdf_path(project_dir)
    plan = collect_project_pdf_items(project_dir)
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
        help="Output PDF path (default: <project>/results/reports/project_report.pdf)",
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
