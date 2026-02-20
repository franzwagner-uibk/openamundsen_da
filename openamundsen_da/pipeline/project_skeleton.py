"""openamundsen_da.pipeline.project_skeleton

Helper to create an empty project step layout from high-level config.

Inputs
------
- setup YAML at setup root (named like the setup, template fallback: ``setup.yml``)
  - Uses the ``timestep`` field (e.g. ``"3H"``) to infer the model time step.
- project YAML at ``<setup>/projects/<project>/`` (named like the project folder)
  - ``start_date`` and ``end_date`` (dates or ISO datetimes).
  - Structured ``data_assimilation.assimilation_events`` block with per-date
    variable/product metadata. Only the dates are used to derive step boundaries.

Behavior
--------
- Creates one project directory of steps under ``<project_dir>/steps/step_*``:
  - ``step_00_init`` for the initial cold-start window.
  - ``step_01_*``, ..., ``step_N_*`` for subsequent windows.
- For each assimilation date ``D_i``:
  - Step i ends at calendar date ``D_i`` with the same time-of-day as
    its own ``start_date``. This guarantees that the run produces a
    daily grid for ``D_i`` (assuming daily outputs are written at a
    fixed clock time per step, as in the examples).
  - The following step (i+1) starts exactly one model time step after
    that end instant. Its calendar date therefore equals ``D_i``, which
    matches the convention used by the setup pipeline (it assimilates
    on the next step's ``start_date`` date).
- The final step's ``end_date`` is aligned with the project YAML ``end_date``.

This module does not touch ensembles or observations; it only creates
step folders and minimal step YAMLs (start_date, end_date, results_dir).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from loguru import logger

from openamundsen_da.io.paths import find_setup_yaml, find_project_yaml
from openamundsen_da.util.loguru_utils import configure_cli_logger
from openamundsen_da.util.yaml_utils import read_yaml_mapping


@dataclass(frozen=True)
class ProjectConfig:
    start: datetime
    end: datetime
    assimilation_event_dates: List[datetime]


def _parse_datetime(text: str | None) -> datetime:
    if not text:
        raise ValueError("Empty datetime string")
    t = str(text).strip()
    # Accept plain date or full ISO datetime.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(t)
    except Exception as exc:
        raise ValueError(f"Invalid datetime value: {text}") from exc


def _parse_date(text: str | None) -> datetime:
    """Parse a calendar date (YYYY-MM-DD) as midnight."""
    if not text:
        raise ValueError("Empty date string")
    t = str(text).strip()
    try:
        d = datetime.strptime(t, "%Y-%m-%d")
        return d.replace(hour=0, minute=0, second=0, microsecond=0)
    except Exception as exc:
        raise ValueError(f"Invalid date value (expected YYYY-MM-DD): {text}") from exc


def _parse_timestep(freq: str | None) -> timedelta:
    """Parse a simple timestep like '3H' or '1D' into a timedelta.

    This intentionally supports only the compact forms used in the examples.
    """
    if not freq:
        raise ValueError("timestep is missing in setup YAML")
    s = str(freq).strip()
    m = re.match(r"^\s*(\d+)?\s*([HhDd])\s*$", s)
    if not m:
        raise ValueError(f"Unsupported timestep format: {freq!r}")
    num_txt, unit = m.groups()
    n = int(num_txt) if num_txt is not None else 1
    if unit.lower() == "h":
        return timedelta(hours=n)
    if unit.lower() == "d":
        return timedelta(days=n)
    raise ValueError(f"Unsupported timestep unit in {freq!r}")


def _load_project_config(setup_dir: Path, project_dir: Path) -> tuple[timedelta, ProjectConfig]:
    setup_yaml = find_setup_yaml(setup_dir)
    setup_cfg = read_yaml_mapping(setup_yaml, error_cls=RuntimeError, context="Setup YAML root")
    ts = setup_cfg.get("timestep")
    dt = _parse_timestep(ts)

    project_yaml = find_project_yaml(project_dir)
    cfg = read_yaml_mapping(project_yaml, error_cls=RuntimeError, context="Project YAML root")
    start = _parse_datetime(str(cfg.get("start_date")))
    end = _parse_datetime(str(cfg.get("end_date")))

    assim_dates: List[datetime] = []
    da_cfg = cfg.get("data_assimilation") or {}
    events_cfg = da_cfg.get("assimilation_events") or []
    if not isinstance(events_cfg, list) or not events_cfg:
        raise ValueError(f"'data_assimilation.assimilation_events' missing or empty in {project_yaml}")
    for entry in events_cfg:
        if not isinstance(entry, dict):
            continue
        date_txt = entry.get("date")
        if not date_txt:
            continue
        assim_dates.append(_parse_date(str(date_txt)))

    assim: List[datetime] = assim_dates
    assim.sort(key=lambda d: d)

    if start >= end:
        raise ValueError(f"Setup start_date {start} must be before end_date {end}")
    if assim[0] <= start:
        logger.warning("First assimilation date {} is not after setup start {}", assim[0].date(), start.date())
    if assim[-1] >= end:
        logger.warning("Last assimilation date {} is not before setup end {}", assim[-1].date(), end.date())

    return dt, ProjectConfig(start=start, end=end, assimilation_event_dates=assim)


def _step_dir_name(index: int, label: str) -> str:
    if index == 0:
        return "step_00_init"
    return f"step_{index:02d}_{label}"


def _write_step_yaml(step_dir: Path, start: datetime, end: datetime, *, overwrite: bool) -> None:
    step_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = step_dir / f"{step_dir.name.split('_')[1]}.yml" if step_dir.name.startswith("step_") else (step_dir / "step.yml")

    if yaml_path.exists() and not overwrite:
        raise FileExistsError(f"Step YAML already exists and overwrite=False: {yaml_path}")

    data = {
        "start_date": start.strftime("%Y-%m-%d %H:%M:%S"),
        "end_date": end.strftime("%Y-%m-%d %H:%M:%S"),
        "results_dir": "results",
    }

    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML()
        y.default_flow_style = False
        with yaml_path.open("w", encoding="utf-8") as f:
            y.dump(data, f)
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to write step YAML {yaml_path}: {exc}") from exc


def create_project_skeleton(setup_dir: Path, project_dir: Path, *, overwrite: bool = False) -> None:
    """Create step_* directories (under project_dir/steps) and minimal step YAMLs for a project.

    Steps are defined such that:
    - Step 0 starts at project.start_date.
    - For each assimilation date D_i, step i includes D_i and ends so
      that the model produces a daily raster for that date. For sub-daily
      timesteps we extend to cover the whole day, then back off one
      timestep so the next step starts exactly one timestep later.
    - The next step (i+1) starts exactly one model time step after the
      end instant (no duplicated timesteps).
    - The last step ends at project.end_date.
    """
    dt, project = _load_project_config(setup_dir, project_dir)

    assim = project.assimilation_event_dates
    n_steps = len(assim) + 1

    steps_root = project_dir / "steps"
    steps_root.mkdir(parents=True, exist_ok=True)

    # Step 0
    step_start = project.start
    for idx in range(n_steps):
        if idx < len(assim):
            # Assimilation date for this window (calendar date)
            a = assim[idx]
            # Assimilation datetime at this step's time-of-day
            step_time = step_start.time()
            assim_dt = a.replace(
                hour=step_time.hour,
                minute=step_time.minute,
                second=step_time.second,
                microsecond=step_time.microsecond,
            )
            # Ensure the step covers the assimilation date's daily outputs.
            # Extend to the end of that calendar day (at least one full day),
            # then back off one timestep so the next step can start at +dt.
            extra = dt if dt >= timedelta(days=1) else timedelta(days=1)
            step_end = assim_dt + (extra - dt)
            if step_end > project.end and idx == len(assim) - 1:
                step_end = project.end
            next_start = step_end + dt
        else:
            # Final step: run until setup end
            next_start = None
            step_end = project.end

        if step_end < step_start:
            raise ValueError(f"Computed end_date {step_end} is before start_date {step_start} for step {idx}")

        if idx == 0:
            label = "init"
        elif idx < len(assim):
            label = f"{assim[idx-1].strftime('%Y%m%d')}-{assim[idx].strftime('%Y%m%d')}"
        else:
            label = f"{assim[-1].strftime('%Y%m%d')}-{project.end.strftime('%Y%m%d')}"

        step_name = _step_dir_name(idx, label)
        step_dir = steps_root / step_name
        if step_dir.exists() and not overwrite:
            raise FileExistsError(f"Step directory already exists and overwrite=False: {step_dir}")

        logger.info("Defining {}: {} -> {}", step_name, step_start, step_end)
        _write_step_yaml(step_dir, start=step_start, end=step_end, overwrite=overwrite)

        if next_start is None:
            break
        step_start = next_start


def cli(argv: list[str] | None = None) -> int:
    """CLI entry point: build a project step skeleton from setup/project config."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-project-skeleton",
        description="Create empty step_* directories and step YAMLs for a project based on assimilation dates.",
    )
    p.add_argument("--setup-dir", required=True, type=Path, help="Setup root directory (contains setup YAML).")
    p.add_argument("--project-dir", required=True, type=Path, help="Project directory under setup/projects (contains project YAML).")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing step YAMLs and reuse existing dirs")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    configure_cli_logger(args.log_level)

    try:
        create_project_skeleton(
            setup_dir=args.setup_dir,
            project_dir=args.project_dir,
            overwrite=bool(args.overwrite),
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI error surface
        logger.error("Project skeleton creation failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())




