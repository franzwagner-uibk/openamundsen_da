"""Helpers for converting project-wide fraction summaries into per-step obs CSVs.

Both MODIS SCF and Sentinel-1 wet-snow summaries share the same pattern:

- A project-level summary CSV, preferably in ``obs/summaries/<project>/``.
  Legacy ``obs/<project>/`` paths are still supported when explicitly used.
- Per-step observation CSVs in ``step_XX_*/obs`` that contain one row for the
  assimilation date of that step.

This module provides small utilities that satellite-specific observers can use
to avoid duplicating CSV handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import pandas as pd
from loguru import logger

from openamundsen_da.core.constants import OBS_DIR_NAME
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import (
    find_project_yaml,
    find_setup_yaml,
    list_steps_sorted as io_list_steps_sorted,
    read_step_config,
)
from openamundsen_da.util.ts import parse_datetime_opt


@dataclass(frozen=True)
class SummaryIndex:
    by_date: Dict[date, pd.Series]


def read_fraction_summary(summary_csv: Path, *, date_col: str = "date") -> SummaryIndex:
    """Read a project-level summary CSV and index rows by date."""

    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")
    df = pd.read_csv(summary_csv, parse_dates=[date_col])
    by_date: Dict[date, pd.Series] = {}
    for _, row in df.iterrows():
        datum = row[date_col]
        if not pd.notna(datum):
            continue
        by_date[datum.to_pydatetime().date()] = row
    return SummaryIndex(by_date=by_date)


def list_steps_sorted(project_dir: Path) -> List[Path]:
    """Return step_* directories sorted by their start_date."""
    return io_list_steps_sorted(project_dir)


def build_obs_filename(
    *,
    variable: str,
    date: datetime,
    product: str | None,
    include_product_tag: bool = False,
) -> str:
    tag = str(product).strip() if product else ""
    filename = f"obs_{variable}_{date.strftime('%Y%m%d')}.csv"
    if include_product_tag and tag:
        filename = f"obs_{variable}_{tag}_{date.strftime('%Y%m%d')}.csv"
    return filename


def build_obs_csv_path(
    *,
    step_dir: Path,
    variable: str,
    date: datetime,
    product: str | None,
    include_product_tag: bool = False,
) -> Path:
    return step_dir / OBS_DIR_NAME / build_obs_filename(
        variable=variable,
        date=date,
        product=product,
        include_product_tag=include_product_tag,
    )


def build_obs_candidate_paths(
    *,
    step_dir: Path,
    variable: str,
    date: datetime,
    product: str | None,
) -> list[Path]:
    """Return default observation filename candidates (untagged first, then tagged)."""
    candidates = [
        build_obs_csv_path(
            step_dir=step_dir,
            variable=variable,
            date=date,
            product=None,
            include_product_tag=False,
        )
    ]
    if str(product or "").strip():
        candidates.append(
            build_obs_csv_path(
                step_dir=step_dir,
                variable=variable,
                date=date,
                product=product,
                include_product_tag=True,
            )
        )
    return candidates


def write_obs_from_summary_row(
    *,
    step_dir: Path,
    date: datetime,
    row: Mapping[str, object],
    value_col: str,
    product: str | None,
    variable: str,
    overwrite: bool,
    include_product_tag: bool = False,
) -> Path:
    """Write a one-row obs CSV for a given date and summary row."""

    out_dir = step_dir / OBS_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / build_obs_filename(
        variable=variable,
        date=date,
        product=product,
        include_product_tag=include_product_tag,
    )
    if out_csv.exists() and not overwrite:
        logger.info("Skipping existing obs CSV for {} (step {})", date.strftime("%Y-%m-%d"), step_dir.name)
        return out_csv

    payload: Dict[str, object] = {}
    for col, val in row.items():
        if pd.isna(val):
            continue
        payload[col] = val
    payload["date"] = date.strftime("%Y-%m-%d")

    # Ensure the primary value column is present under its variable-specific name.
    if value_col in row:
        payload[value_col] = row[value_col]

    df = pd.DataFrame({k: [v] for k, v in payload.items()})
    df.to_csv(out_csv, index=False)
    logger.info("Wrote obs {} -> {} ({})", date.strftime("%Y-%m-%d"), step_dir.name, out_csv.name)
    return out_csv


def _setup_from_project(project_dir: Path | None) -> Path | None:
    if project_dir is None:
        return None
    for candidate in (project_dir.parent, project_dir.parent.parent):
        if candidate and candidate.is_dir():
            try:
                return find_setup_yaml(candidate).parent
            except Exception:
                continue
    return None


def _product_tag_key_for_variable(variable: str) -> str:
    if variable == "scf":
        return "snowcover"
    if variable == "wet_snow":
        return "wetsnow"
    raise ValueError(f"Unsupported assimilation variable for product tag resolution: {variable!r}")


def _require_product_tag(cfg: object, *, key: str, source: str) -> str:
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping at {source}")
    obs_cfg = cfg.get("obs")
    if not isinstance(obs_cfg, dict):
        raise ValueError(f"Expected mapping at {source}.obs")
    section = obs_cfg.get(key)
    if not isinstance(section, dict):
        raise ValueError(f"Expected mapping at {source}.obs.{key}")
    if "product_tag" not in section:
        raise ValueError(f"Missing required configuration key: {source}.obs.{key}.product_tag")
    tag = str(section["product_tag"]).strip()
    if not tag:
        raise ValueError(f"Configuration value must not be empty: {source}.obs.{key}.product_tag")
    return tag.upper()


def resolve_obs_product_tag(
    variable: str,
    *,
    project_dir: Path | None = None,
    setup_dir: Path | None = None,
) -> str:
    """Resolve product tag for obs filenames from project YAML first, then setup YAML."""
    key = _product_tag_key_for_variable(variable)

    if project_dir:
        cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
        return _require_product_tag(cfg, key=key, source="project")

    setup_root = setup_dir or _setup_from_project(project_dir)
    if setup_root is None:
        raise ValueError("Could not resolve setup directory for product tag lookup")
    cfg = _read_yaml_file(find_setup_yaml(setup_root)) or {}
    return _require_product_tag(cfg, key=key, source="setup")


def prepare_project_obs_from_summary(
    project_dir: Path,
    summary_csv: Path,
    *,
    variable: str,
    value_col: str,
    accepted_event_variables: Iterable[str],
    product: str | None,
    overwrite: bool,
    include_product_tag: bool = True,
    use_step_start_time: bool = False,
    summary_date_col: str = "date",
    log_prefix: str = "Project summary prep",
) -> tuple[int, int, int]:
    """Create per-step observation CSVs from a project-level summary CSV."""
    from openamundsen_da.util.da_events import load_assimilation_events

    if not project_dir.is_dir():
        raise FileNotFoundError(f"Project directory not found: {project_dir}")
    if not summary_csv.is_file():
        raise FileNotFoundError(f"Summary CSV not found: {summary_csv}")

    summary = read_fraction_summary(summary_csv, date_col=summary_date_col)
    events = load_assimilation_events(project_dir)
    steps = list_steps_sorted(project_dir)
    if len(steps) < 2:
        raise FileNotFoundError(f"Not enough steps to derive assimilation dates under {project_dir}")

    allowed_vars = {str(v).strip().lower() for v in accepted_event_variables if str(v).strip()}
    if not allowed_vars:
        raise ValueError("accepted_event_variables must contain at least one non-empty variable")

    setup_dir = project_dir.parent.parent if project_dir.parent.parent.is_dir() else None
    expected_events = len(steps) - 1
    if len(events) != expected_events:
        raise ValueError(
            f"{log_prefix}: expected exactly {expected_events} assimilation_events for {len(steps)} steps "
            f"(one event per step except final), found {len(events)}"
        )

    prod_tag = str(product).strip().upper() if product else resolve_obs_product_tag(
        variable,
        setup_dir=setup_dir,
        project_dir=project_dir,
    )

    written = 0
    skipped_missing = 0
    skipped_existing = 0
    for i in range(expected_events):
        step = steps[i]
        ev = events[i]
        if str(ev.variable).strip().lower() not in allowed_vars:
            continue

        cfg = read_step_config(step) or {}
        start_dt = parse_datetime_opt(str(cfg.get("start_date")))
        end_dt = parse_datetime_opt(str(cfg.get("end_date")))
        if start_dt and end_dt and not (start_dt.date() <= ev.date <= end_dt.date()):
            raise ValueError(
                f"{log_prefix}: assimilation date {ev.date} is outside step {step.name} window "
                f"({start_dt.date()} .. {end_dt.date()})"
            )

        row = summary.by_date.get(ev.date)
        if row is None:
            raise ValueError(
                f"{log_prefix}: missing summary row for variable {variable} at assimilation date {ev.date} "
                f"(step {step.name})"
            )

        obs_dt = datetime.combine(
            ev.date,
            (start_dt or datetime.min).time() if use_step_start_time else datetime.min.time(),
        )
        out_csv = build_obs_csv_path(
            step_dir=step,
            variable=variable,
            date=obs_dt,
            product=prod_tag,
            include_product_tag=include_product_tag,
        )
        if out_csv.exists() and not overwrite:
            logger.info("Skipping existing obs CSV for {} (step {})", obs_dt.strftime("%Y-%m-%d"), step.name)
            skipped_existing += 1
            continue

        write_obs_from_summary_row(
            step_dir=step,
            date=obs_dt,
            row=row,
            value_col=value_col,
            product=prod_tag,
            variable=variable,
            overwrite=overwrite,
            include_product_tag=include_product_tag,
        )
        written += 1

    logger.info(
        "{} complete: written={} skipped_missing={} skipped_existing={}",
        log_prefix,
        written,
        skipped_missing,
        skipped_existing,
    )
    return written, skipped_missing, skipped_existing

