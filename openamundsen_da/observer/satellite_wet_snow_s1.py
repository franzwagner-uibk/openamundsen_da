"""Project helper for wet-snow observations from project summaries."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.observer.fraction_obs import (
    list_steps_sorted,
    read_fraction_summary,
    resolve_obs_product_tag,
    write_obs_from_summary_row,
)
from openamundsen_da.io.paths import read_step_config
from openamundsen_da.util.da_events import load_assimilation_events


def _parse_dt_opt(text: str | None) -> datetime | None:
    if not text:
        return None
    t = str(text).strip().replace("_", "-")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(t)
    except Exception:
        return None


def generate_project_from_summary(
    project_dir: Path,
    summary_csv: Path,
    *,
    product: str | None,
    overwrite: bool,
) -> None:
    """Extract per-step obs CSVs from a project-wide ``wet_snow_summary.csv``."""

    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    summary = read_fraction_summary(summary_csv, date_col="date")
    events = load_assimilation_events(project_dir)

    steps = list_steps_sorted(project_dir)
    if len(steps) < 2:
        raise FileNotFoundError(f"Not enough steps to derive assimilation dates under {project_dir}")

    setup_dir = project_dir.parent.parent if project_dir.parent.parent.is_dir() else None
    n = min(len(events), len(steps) - 1)
    if n < len(events):
        logger.warning("Only {} steps (excluding final) available for {} assimilation events; extra events will be ignored.", n, len(events))
    if n < len(steps) - 1:
        logger.warning("Only {} assimilation events available for {} steps; later steps will not receive obs CSVs.", len(events), len(steps) - 1)

    prod_tag = str(product).strip().upper() if product else resolve_obs_product_tag("wet_snow", setup_dir=setup_dir, project_dir=project_dir)

    written = skipped_missing = skipped_existing = 0
    for i in range(n):
        step = steps[i]
        ev = events[i]
        # Only handle wet-snow assimilation events here; skip others silently.
        if str(ev.variable).lower() not in ("wet_snow", "wet_snow_fraction"):
            continue
        cfg = read_step_config(step) or {}
        start_dt = _parse_dt_opt(str(cfg.get("start_date")))
        end_dt = _parse_dt_opt(str(cfg.get("end_date")))
        if start_dt and end_dt:
            if not (start_dt.date() <= ev.date <= end_dt.date()):
                logger.warning(
                    "Assimilation date {} is outside step {} window ({} .. {})",
                    ev.date,
                    step.name,
                    start_dt.date(),
                    end_dt.date(),
                )

        row = summary.by_date.get(ev.date)
        if row is None:
            logger.debug("No wet-snow summary entry for assimilation date {}; skipping {}", ev.date, step.name)
            skipped_missing += 1
            continue

        assim_dt = datetime.combine(ev.date, (start_dt or datetime.min).time())
        out_csv = write_obs_from_summary_row(
            step_dir=step,
            date=assim_dt,
            row=row,
            value_col="wet_snow_fraction",
            product=prod_tag,
            variable="wet_snow",
            overwrite=overwrite,
        )
        written += 1
        logger.info("Wrote wet-snow obs {} -> {} ({})", assim_dt.strftime("%Y-%m-%d"), step.name, out_csv.name)

    logger.info(
        "Wet-snow project summary prep complete: written={} skipped_missing={} skipped_existing={}",
        written,
        skipped_missing,
        skipped_existing,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """CLI: fill per-step obs CSVs from wet_snow_summary.csv for a project."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="oa-da-wet-snow-s1-setup",
        description=(
            "Copy wet-snow rows from wet_snow_summary.csv into per-step "
            "obs_wet_snow_S1_YYYYMMDD.csv files for a project."
        ),
    )
    parser.add_argument("--project-dir", required=True, type=Path, help="Project directory (setup/projects/project_YYYY_YYYY)")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Path to wet_snow_summary.csv (default: <setup>/obs/<project>/wet_snow_summary.csv)",
    )
    parser.add_argument("--product", help="Product tag to use in obs filename (default: obs.wetsnow.product_tag)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing obs_wet_snow_*.csv files")
    parser.add_argument("--log-level", default="INFO", help="Log level (default: INFO)")

    args = parser.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    project_dir = args.project_dir
    if args.summary_csv is not None:
        summary_path = args.summary_csv
    else:
        setup_root = project_dir.parent.parent
        summary_path = setup_root / "obs" / project_dir.name / "wet_snow_summary.csv"

    try:
        generate_project_from_summary(
            project_dir=project_dir,
            summary_csv=summary_path,
            product=str(args.product) if args.product else None,
            overwrite=args.overwrite,
        )
        return 0
    except Exception as exc:
        logger.error("Wet-snow project summary prep failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())



