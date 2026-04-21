from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from openamundsen_da.core.env import (
    apply_env_from_project,
    apply_numeric_thread_defaults,
    ensure_gdal_proj_from_conda,
)
from openamundsen_da.io.paths import (
    find_project_yaml,
    infer_setup_dir_from_project,
    list_steps_sorted,
    project_fraction_envelope_path,
)
from openamundsen_da.pipeline.plot_tasks import (
    aggregate_fraction_envelopes,
    build_fraction_overlay_task,
    build_post_run_plot_tasks,
    custom_overview_needs_benchmark_scores,
    run_plot_tasks_parallel,
)
from openamundsen_da.util.loguru_utils import configure_cli_logger


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def default_project_plots_rerun_command(project_dir: Path) -> str:
    return (
        "python -m openamundsen_da.methods.viz.plots.runner "
        f"--project-dir {Path(project_dir).resolve()}"
    )


def render_project_plots(
    *,
    project_dir: Path,
    plot_workers: int | None = None,
    max_workers: int | None = None,
) -> list[str]:
    project_dir = Path(project_dir).resolve()
    project_yaml = find_project_yaml(project_dir)
    apply_env_from_project(project_yaml)
    ensure_gdal_proj_from_conda()
    apply_numeric_thread_defaults()

    steps = list_steps_sorted(project_dir)
    if not steps:
        raise FileNotFoundError(f"No step directories found under {project_dir / 'steps'}")

    cfg = type(
        "ProjectPlotsConfig",
        (),
        {
            "project_dir": project_dir,
            "setup_dir": infer_setup_dir_from_project(project_dir),
            "log_level": "INFO",
            "plot_workers": plot_workers,
            "max_workers": max_workers,
        },
    )()

    logger.info("Aggregating fraction envelopes for {}", project_dir)
    aggregate_fraction_envelopes(
        project_dir=project_dir,
        project_fraction_envelope_path=project_fraction_envelope_path,
    )

    include_fraction_overlay = not custom_overview_needs_benchmark_scores(project_dir)
    tasks = build_post_run_plot_tasks(
        cfg,
        steps,
        include_fraction_overlay=include_fraction_overlay,
    )
    logger.info("Rendering {} project plot task(s) ...", len(tasks))
    run_plot_tasks_parallel(tasks, plot_workers, max_workers)

    outputs = [task.name for task in tasks]
    if not include_fraction_overlay:
        logger.info("Rendering deferred benchmark-dependent overview plot ...")
        overlay_task = build_fraction_overlay_task(cfg)
        run_plot_tasks_parallel([overlay_task], plot_workers, max_workers)
        outputs.append(overlay_task.name)

    logger.info("Project plot rendering complete -> {} task(s)", len(outputs))
    return outputs


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="oa-da-plot-project-plots",
        description="Render all project plots from existing run outputs without rerunning the DA pipeline.",
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory")
    parser.add_argument(
        "--plot-workers",
        type=_positive_int,
        help="Maximum plot-task workers (default: auto within the task runner)",
    )
    parser.add_argument(
        "--max-workers",
        type=_positive_int,
        help="Optional global worker ceiling passed to the plot-task runner",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")
    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)
    try:
        outputs = render_project_plots(
            project_dir=args.project_dir,
            plot_workers=args.plot_workers,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        logger.error("Project plot rendering failed: {}", exc)
        logger.error(
            "Rerun project plots with: {}",
            default_project_plots_rerun_command(args.project_dir),
        )
        return 1
    logger.info("Project plot rendering complete -> {} task(s)", len(outputs))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
