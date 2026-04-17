"""Project-level scientific benchmarking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from loguru import logger

from openamundsen_da.benchmark.aggregate import aggregate_scores, build_case_scores, enrich_case_scores, reliability_rows
from openamundsen_da.benchmark.extract import (
    benchmark_supported_variables,
    benchmark_variable_spec,
    extract_analysis_cases,
    extract_continuous_cases,
)
from openamundsen_da.methods.viz.plots.benchmark import write_plots
from openamundsen_da.benchmark.render.tables import (
    write_case_tables,
    write_manifest,
    write_score_tables,
    write_summary_markdown,
    write_summary_tables,
)
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    find_project_yaml,
    infer_setup_dir_from_project,
    list_member_dirs,
    list_steps_sorted,
    project_plot_assim_scores_dir,
)
from openamundsen_da.methods.h_of_x.model_scf import compute_step_scf_daily_for_all_members
from openamundsen_da.methods.wet_snow.area import compute_step_wet_snow_daily_for_all_members
from openamundsen_da.methods.wet_snow.classify import classify_step_wet_snow
from openamundsen_da.util.da_events import load_assimilation_events
from openamundsen_da.util.landcover_mask import resolve_landcover_mask
from openamundsen_da.util.parallel import pick_max_workers
from openamundsen_da.util.roi_grid import ensure_setup_roi_vector


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path
    plots: bool
    independent_variables: tuple[str, ...]
    performance_scores_exclude_variables: tuple[str, ...]
    score_station_sigma_threshold: float | None


def _normalize_variable(raw: object) -> str:
    value = str(raw).strip().lower()
    if value == "wet_snow_fraction":
        value = "wet_snow"
    benchmark_variable_spec(value)
    return value


def load_benchmark_config(project_dir: Path) -> BenchmarkConfig:
    cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    da_cfg = cfg.get("data_assimilation")
    if not isinstance(da_cfg, dict):
        return BenchmarkConfig(
            output_dir=project_dir / "results" / "benchmark",
            plots=True,
            independent_variables=(),
            performance_scores_exclude_variables=(),
            score_station_sigma_threshold=None,
        )
    bench_cfg = da_cfg.get("benchmark")
    if bench_cfg is None:
        return BenchmarkConfig(
            output_dir=project_dir / "results" / "benchmark",
            plots=True,
            independent_variables=(),
            performance_scores_exclude_variables=(),
            score_station_sigma_threshold=None,
        )
    if not isinstance(bench_cfg, dict):
        raise ValueError("project.data_assimilation.benchmark must be a mapping")

    independent_raw = bench_cfg.get("independent_variables") or []
    if not isinstance(independent_raw, list):
        raise ValueError("project.data_assimilation.benchmark.independent_variables must be a list")
    independent_variables = tuple(sorted({_normalize_variable(v) for v in independent_raw}))

    exclude_raw = bench_cfg.get("performance_scores_exclude_variables") or []
    if not isinstance(exclude_raw, list):
        raise ValueError("project.data_assimilation.benchmark.performance_scores_exclude_variables must be a list")
    performance_scores_exclude_variables = tuple(sorted({_normalize_variable(v) for v in exclude_raw}))

    threshold_raw = bench_cfg.get("score_station_sigma_threshold")
    score_station_sigma_threshold: float | None = None
    if threshold_raw is not None and str(threshold_raw).strip() != "":
        try:
            score_station_sigma_threshold = float(threshold_raw)
        except Exception as exc:
            raise ValueError("project.data_assimilation.benchmark.score_station_sigma_threshold must be numeric") from exc
        if score_station_sigma_threshold <= 0.0:
            raise ValueError("project.data_assimilation.benchmark.score_station_sigma_threshold must be > 0")

    plots_raw = bench_cfg.get("plots", True)
    plots = bool(plots_raw)

    output_dir_raw = str(bench_cfg.get("output_dir", "results/benchmark")).strip()
    if not output_dir_raw:
        raise ValueError("project.data_assimilation.benchmark.output_dir must not be empty")
    output_dir = Path(output_dir_raw)
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    return BenchmarkConfig(
        output_dir=output_dir,
        plots=plots,
        independent_variables=independent_variables,
        performance_scores_exclude_variables=performance_scores_exclude_variables,
        score_station_sigma_threshold=score_station_sigma_threshold,
    )


def automatic_benchmark_variables(project_dir: Path) -> list[str]:
    variables = sorted({ev.variable for ev in load_assimilation_events(project_dir)})
    return [_normalize_variable(v) for v in variables]


def selected_benchmark_variables(
    project_dir: Path,
    *,
    cli_variables: Iterable[str] | None = None,
) -> list[str]:
    if cli_variables is not None:
        selected = sorted({_normalize_variable(v) for v in cli_variables})
        if not selected:
            raise ValueError("Benchmark variable filter resolved to an empty set")
        return selected

    cfg = load_benchmark_config(project_dir)
    selected = sorted(set(automatic_benchmark_variables(project_dir)) | set(cfg.independent_variables))
    if not selected:
        raise ValueError("No benchmark variables resolved from assimilation events or benchmark config")
    return selected


def _load_wet_snow_threshold(project_dir: Path) -> float:
    cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    da_cfg = cfg.get("data_assimilation")
    if not isinstance(da_cfg, dict):
        raise ValueError("project.data_assimilation is required for benchmark wet-snow preparation")
    wet_cfg = da_cfg.get("wet_snow")
    if not isinstance(wet_cfg, dict):
        raise ValueError("project.data_assimilation.wet_snow is required for benchmark wet-snow preparation")
    raw = wet_cfg.get("classification_threshold_percent")
    try:
        return float(raw)
    except Exception as exc:
        raise ValueError("project.data_assimilation.wet_snow.classification_threshold_percent must be numeric") from exc


def _prior_member_point_series_paths(step_dir: Path, filename: str) -> list[Path]:
    base = Path(step_dir) / "ensembles" / "prior"
    targets = [base / "open_loop" / "results" / filename]
    targets.extend(member_dir / "results" / filename for member_dir in list_member_dirs(base.parent, "prior"))
    return targets


def _prior_member_point_series_complete(step_dir: Path, filename: str) -> bool:
    targets = _prior_member_point_series_paths(step_dir, filename)
    return bool(targets) and all(path.is_file() for path in targets)


def ensure_benchmark_prerequisites(
    *,
    project_dir: Path,
    setup_dir: Path,
    variables: Iterable[str],
    max_workers: int | None = None,
    overwrite: bool = False,
    reuse_existing_prerequisites: bool = False,
) -> None:
    required = {benchmark_variable_spec(v).variable for v in variables}
    if not required.intersection({"scf", "wet_snow"}):
        return

    roi_path = ensure_setup_roi_vector(setup_dir)
    landcover_cfg = resolve_landcover_mask(setup_dir, project_dir)
    workers = pick_max_workers(max_workers, fallback=4)
    wet_threshold = _load_wet_snow_threshold(project_dir) if "wet_snow" in required else None
    effective_overwrite = bool(overwrite and not reuse_existing_prerequisites)

    for step_dir in list_steps_sorted(project_dir):
        if "scf" in required:
            if not effective_overwrite and _prior_member_point_series_complete(step_dir, "point_scf_roi.csv"):
                logger.info(
                    "SCF benchmark prerequisites already present for {} -> reusing existing outputs",
                    step_dir.name,
                )
            else:
                compute_step_scf_daily_for_all_members(
                    setup_dir=setup_dir,
                    project_dir=project_dir,
                    step_dir=step_dir,
                    aoi_path=roi_path,
                    landcover_cfg=landcover_cfg,
                    max_workers=workers,
                    overwrite=effective_overwrite,
                )
        if "wet_snow" in required:
            if not effective_overwrite and _prior_member_point_series_complete(step_dir, "point_wet_snow_roi.csv"):
                logger.info(
                    "Wet-snow benchmark prerequisites already present for {} -> reusing existing outputs",
                    step_dir.name,
                )
                continue
            assert wet_threshold is not None
            classify_step_wet_snow(
                step_dir=step_dir,
                members=None,
                threshold_percent=wet_threshold,
                output_subdir="wet_snow",
                mask_prefix="wet_snow_mask",
                fraction_prefix="lwc_fraction",
                write_fraction=False,
                overwrite=effective_overwrite,
                max_workers=workers,
            )
            compute_step_wet_snow_daily_for_all_members(
                setup_dir=setup_dir,
                project_dir=project_dir,
                step_dir=step_dir,
                aoi_path=roi_path,
                landcover_cfg=landcover_cfg,
                max_workers=workers,
                overwrite=effective_overwrite,
                mask_subdir="wet_snow",
                mask_prefix="wet_snow_mask",
            )


def run_project_benchmark(
    *,
    project_dir: Path,
    setup_dir: Path | None = None,
    variables: Iterable[str] | None = None,
    output_dir: Path | None = None,
    plots: bool | None = None,
    max_workers: int | None = None,
    overwrite: bool = False,
    reuse_existing_prerequisites: bool = False,
) -> dict[str, Path]:
    project_dir = Path(project_dir)
    resolved_setup_dir = Path(setup_dir) if setup_dir is not None else infer_setup_dir_from_project(project_dir)
    cfg = load_benchmark_config(project_dir)
    benchmark_variables = selected_benchmark_variables(project_dir, cli_variables=variables)
    independent_variables = list(cfg.independent_variables)

    results_dir = Path(output_dir) if output_dir is not None else cfg.output_dir
    if not results_dir.is_absolute():
        results_dir = project_dir / results_dir
    plots_enabled = cfg.plots if plots is None else bool(plots)
    plots_dir = project_plot_assim_scores_dir(project_dir)

    logger.info(
        "Running project benchmark for {} variable(s): {}",
        len(benchmark_variables),
        ", ".join(benchmark_variables),
    )
    ensure_benchmark_prerequisites(
        project_dir=project_dir,
        setup_dir=resolved_setup_dir,
        variables=benchmark_variables,
        max_workers=max_workers,
        overwrite=overwrite,
        reuse_existing_prerequisites=reuse_existing_prerequisites,
    )

    continuous_cases = extract_continuous_cases(
        project_dir=project_dir,
        setup_dir=resolved_setup_dir,
        variables=benchmark_variables,
    )
    analysis_cases = extract_analysis_cases(
        project_dir=project_dir,
        setup_dir=resolved_setup_dir,
        variables=benchmark_variables,
    )
    raw_cases = [*continuous_cases, *analysis_cases]
    if not raw_cases:
        raise ValueError("Benchmark stage found no usable observation/model cases in the project window")

    case_scores = build_case_scores(raw_cases)
    case_scores = enrich_case_scores(
        case_scores,
        project_dir=project_dir,
        setup_dir=resolved_setup_dir,
        score_station_sigma_threshold=cfg.score_station_sigma_threshold,
    )
    event_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream", "step_name", "timestamp", "date"),
    )
    project_scores = aggregate_scores(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )
    reliability = reliability_rows(
        case_scores,
        group_cols=("score_set", "variable", "stream"),
    )

    outputs: dict[str, Path] = {}
    outputs.update(write_case_tables(results_dir, case_scores))
    outputs.update(
        write_score_tables(
            results_dir,
            event_scores=event_scores,
            project_scores=project_scores,
            reliability=reliability,
        )
    )
    table_outputs, _tables = write_summary_tables(
        results_dir,
        event_scores=event_scores,
        project_scores=project_scores,
        reliability=reliability,
    )
    outputs.update(table_outputs)
    summary_path = write_summary_markdown(
        results_dir,
        project_dir=project_dir,
        benchmark_variables=benchmark_variables,
        independent_variables=independent_variables,
        project_summary=_tables["project_summary"],
        update_summary=_tables["update_summary"],
    )
    outputs["summary"] = summary_path
    if plots_enabled:
        outputs.update(
            write_plots(
                plots_dir,
                case_scores=case_scores,
                event_scores=event_scores,
                reliability=reliability,
                project_dir=project_dir,
                exclude_variables=cfg.performance_scores_exclude_variables,
            )
        )
    manifest_path = write_manifest(
        results_dir,
        project_dir=project_dir,
        benchmark_variables=benchmark_variables,
        independent_variables=independent_variables,
        outputs=outputs,
        case_scores=case_scores,
        event_scores=event_scores,
        project_scores=project_scores,
    )
    outputs["manifest"] = manifest_path
    return outputs


def cli(argv: list[str] | None = None) -> int:
    import argparse

    from openamundsen_da.util.loguru_utils import configure_cli_logger

    parser = argparse.ArgumentParser(
        prog="oa-da-benchmark",
        description="Run scientific benchmarking on an existing openAMUNDSEN-DA project.",
    )
    parser.add_argument("--project-dir", required=True, type=Path)
    parser.add_argument("--setup-dir", type=Path, help="Optional setup directory; inferred from project when omitted.")
    parser.add_argument(
        "--variables",
        nargs="+",
        help=f"Optional benchmark variables ({', '.join(benchmark_supported_variables())})",
    )
    parser.add_argument("--output-dir", type=Path, help="Optional benchmark results directory override.")
    parser.add_argument("--no-plots", action="store_true", help="Disable benchmark plot writing.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    run_project_benchmark(
        project_dir=args.project_dir,
        setup_dir=args.setup_dir,
        variables=args.variables,
        output_dir=args.output_dir,
        plots=False if args.no_plots else None,
        max_workers=args.max_workers,
        overwrite=args.overwrite,
    )
    return 0
