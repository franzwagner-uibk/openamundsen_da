"""Project helper for wet-snow observations from project summaries."""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from openamundsen_da.observer.fraction_obs import (
    prepare_project_obs_from_summary,
)
from openamundsen_da.observer.summary_paths import record_fraction_summary_path, resolve_fraction_summary_path
from openamundsen_da.util.loguru_utils import configure_cli_logger


def generate_project_from_summary(
    project_dir: Path,
    summary_csv: Path,
    *,
    product: str | None,
    overwrite: bool,
) -> None:
    """Extract per-step obs CSVs from a project-wide ``wet_snow_summary.csv``."""
    prepare_project_obs_from_summary(
        project_dir=project_dir,
        summary_csv=summary_csv,
        variable="wet_snow",
        value_col="wet_snow_fraction",
        accepted_event_variables=("wet_snow", "wet_snow_fraction"),
        product=product,
        overwrite=overwrite,
        include_product_tag=True,
        use_step_start_time=True,
        summary_date_col="date",
        log_prefix="Wet-snow project summary prep",
    )
    prepare_project_obs_from_summary(
        project_dir=project_dir,
        summary_csv=summary_csv,
        variable="wet_snow_line",
        value_col="wet_snow_line",
        accepted_event_variables=("wet_snow_line",),
        product=product,
        overwrite=overwrite,
        include_product_tag=True,
        use_step_start_time=True,
        summary_date_col="date",
        log_prefix="Wet-snow-line project summary prep",
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: fill per-step obs CSVs from wet_snow_summary.csv for a project."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-wet-snow-s1-setup",
        description=(
            "Copy wet-snow rows from wet_snow_summary.csv into per-step "
            "obs_wet_snow_<PRODUCT>_YYYYMMDD.csv files for a project."
        ),
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory (setup/projects/project_YYYY_YYYY)")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help=(
            "Path to wet_snow_summary.csv. When provided, the path is recorded "
            "in obs.wetsnow.summary_csv for maps and benchmarking."
        ),
    )
    parser.add_argument("--product", help="Product tag to use in obs filename (default: obs.wetsnow.product_tag)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing obs_wet_snow_*.csv files")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(argv)

    configure_cli_logger(args.log_level)

    project_dir = args.project_dir
    if args.summary_csv is not None:
        summary_path = args.summary_csv
    else:
        setup_root = project_dir.parent.parent
        summary_path = resolve_fraction_summary_path(
            setup_root,
            project_dir,
            "wet_snow_summary.csv",
        )

    try:
        generate_project_from_summary(
            project_dir=project_dir,
            summary_csv=summary_path,
            product=str(args.product) if args.product else None,
            overwrite=args.overwrite,
        )
        record_fraction_summary_path(
            setup_dir=project_dir.parent.parent,
            project_dir=project_dir,
            filename="wet_snow_summary.csv",
            summary_csv=summary_path,
        )
        return 0
    except Exception as exc:
        logger.error("Wet-snow project summary prep failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
