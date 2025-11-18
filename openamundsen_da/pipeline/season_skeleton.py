"""openamundsen_da.pipeline.season_skeleton

Helper to create an empty season step layout from high-level config.

Inputs
------
- project.yml
  - Uses the ``timestep`` field (e.g. ``"3H"``) to infer the model time step.
- propagation/season_YYYY-YYYY/season.yml
  - ``start_date`` and ``end_date`` (dates or ISO datetimes).
  - ``assimilation_dates``: list of calendar dates (YYYY-MM-DD) at which
    SCF assimilation should occur.

Behavior
--------
- Creates one season directory of steps under ``<season_dir>/step_*``:
  - ``step_00_init`` for the initial cold-start window.
  - ``step_01_*``, ..., ``step_N_*`` for subsequent windows.
- For each assimilation date ``D_i``:
  - Step i's end_date is set such that the *next* step starts one model
    time step after the assimilation time.
  - The following step (i+1) receives a ``start_date`` whose calendar
    date equals ``D_i``; this matches the convention used by the season
    pipeline, which assimilates on the next step's ``start_date`` date.
- The final step's ``end_date`` is aligned with ``season.yml.end_date``.

This module does not touch ensembles or observations; it only creates
step folders and minimal step YAMLs (start_date, end_date, results_dir).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from loguru import logger

from openamundsen_da.core.constants import LOGURU_FORMAT
from openamundsen_da.io.paths import find_project_yaml, find_season_yaml


@dataclass(frozen=True)
class SeasonConfig:
    start: datetime
    end: datetime
    assimilation_dates: List[datetime]


def _read_yaml(path: Path) -> dict:
    try:
        import ruamel.yaml as _yaml

        y = _yaml.YAML(typ="safe")
        with path.open("r", encoding="utf-8") as f:
            data = y.load(f) or {}
        return data
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Could not read YAML from {path}: {exc}") from exc


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
        raise ValueError("timestep is missing in project.yml")
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


def _load_season_config(project_dir: Path, season_dir: Path) -> tuple[timedelta, SeasonConfig]:
    proj_yaml = find_project_yaml(project_dir)
    proj = _read_yaml(proj_yaml)
    ts = proj.get("timestep")
    dt = _parse_timestep(ts)

    season_yaml = find_season_yaml(season_dir)
    cfg = _read_yaml(season_yaml)
    start = _parse_datetime(str(cfg.get("start_date")))
    end = _parse_datetime(str(cfg.get("end_date")))

    raw_dates = cfg.get("assimilation_dates")
    if not raw_dates:
        raise ValueError(f"'assimilation_dates' missing or empty in {season_yaml}")

    assim: List[datetime] = [_parse_date(str(d)) for d in raw_dates]
    assim.sort(key=lambda d: d)

    if start >= end:
        raise ValueError(f"Season start_date {start} must be before end_date {end}")
    if assim[0] <= start:
        logger.warning("First assimilation date {} is not after season start {}", assim[0].date(), start.date())
    if assim[-1] >= end:
        logger.warning("Last assimilation date {} is not before season end {}", assim[-1].date(), end.date())

    return dt, SeasonConfig(start=start, end=end, assimilation_dates=assim)


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


def create_season_skeleton(project_dir: Path, season_dir: Path, *, overwrite: bool = False) -> None:
    """Create step_* directories and minimal step YAMLs for a season.

    Steps are defined such that:
    - Step 0 starts at season.start_date.
    - For each assimilation date D_i, the next step (i+1) starts at
      D_i + timestep, and step i ends at that start minus one timestep.
    - The last step ends at season.end_date.
    """
    dt, season = _load_season_config(project_dir, season_dir)

    assim = season.assimilation_dates
    n_steps = len(assim) + 1

    # Step 0
    step_start = season.start
    for idx in range(n_steps):
        if idx < len(assim):
            # Assimilation date for this window
            a = assim[idx]
            next_start = a + dt
            step_end = next_start - dt
        else:
            # Final step: run until season end
            next_start = None
            step_end = season.end

        if step_end < step_start:
            raise ValueError(f"Computed end_date {step_end} is before start_date {step_start} for step {idx}")

        if idx == 0:
            label = "init"
        elif idx < len(assim):
            label = f"{assim[idx-1].strftime('%Y%m%d')}-{assim[idx].strftime('%Y%m%d')}"
        else:
            label = f"{assim[-1].strftime('%Y%m%d')}-{season.end.strftime('%Y%m%d')}"

        step_name = _step_dir_name(idx, label)
        step_dir = season_dir / step_name
        if step_dir.exists() and not overwrite:
            raise FileExistsError(f"Step directory already exists and overwrite=False: {step_dir}")

        logger.info("Defining {}: {} -> {}", step_name, step_start, step_end)
        _write_step_yaml(step_dir, start=step_start, end=step_end, overwrite=overwrite)

        if next_start is None:
            break
        step_start = next_start


def cli(argv: list[str] | None = None) -> int:
    """CLI entry point: build a season step skeleton from project/season config."""
    import argparse

    p = argparse.ArgumentParser(
        prog="oa-da-season-skeleton",
        description="Create empty step_* directories and step YAMLs for a season based on assimilation dates.",
    )
    p.add_argument("--project-dir", required=True, type=Path)
    p.add_argument("--season-dir", required=True, type=Path)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing step YAMLs and reuse existing dirs")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)

    logger.remove()
    logger.add(sys.stdout, level=args.log_level.upper(), colorize=True, enqueue=True, format=LOGURU_FORMAT)

    try:
        create_season_skeleton(
            project_dir=args.project_dir,
            season_dir=args.season_dir,
            overwrite=bool(args.overwrite),
        )
        return 0
    except Exception as exc:  # pragma: no cover - CLI error surface
        logger.error("Season skeleton creation failed: {}", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())
