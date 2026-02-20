"""Project helper for snow-cover observations backed by ``scf_summary.csv``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.observer.fraction_obs import (
    prepare_project_obs_from_summary,
)


def generate_project_from_summary(
    project_dir: Path,
    summary_csv: Path,
    *,
    product: str | None,
    overwrite: bool,
) -> None:
    """Extract per-step obs CSVs from a project-wide ``scf_summary.csv``."""
    prepare_project_obs_from_summary(
        project_dir=project_dir,
        summary_csv=summary_csv,
        variable="scf",
        value_col="scf",
        accepted_event_variables=("scf",),
        product=product,
        overwrite=overwrite,
        include_product_tag=True,
        use_step_start_time=False,
        summary_date_col="date",
        log_prefix="SCF project summary prep",
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: fill per-step obs CSVs from scf_summary.csv for a project."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-scf",
        description=(
            "Copy SCF rows from scf_summary.csv into per-step "
            "obs_scf_<PRODUCT>_YYYYMMDD.csv files for a project."
        ),
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory (setup/projects/project_YYYY_YYYY)")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to scf_summary.csv (default: <setup>/obs/<project>/scf_summary.csv)",
    )
    parser.add_argument("--product", help="Product tag to use in obs filename (default: obs.snowcover.product_tag)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing obs_scf_*.csv files")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    project_dir = args.project_dir
    if args.summary_csv is not None:
        summary_path = args.summary_csv
    else:
        setup_root = project_dir.parent.parent
        summary_path = setup_root / "obs" / project_dir.name / "scf_summary.csv"

    try:
        generate_project_from_summary(
            project_dir=project_dir,
            summary_csv=summary_path,
            product=str(args.product) if args.product else None,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        logger.error("Project summary prep failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())



