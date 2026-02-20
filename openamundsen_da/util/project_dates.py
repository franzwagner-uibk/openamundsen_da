"""Shared helper for resolving project start/end dates from project YAML."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from openamundsen_da.core.env import _read_yaml_file


def resolve_project_dates(setup_dir: Path, project_label: str) -> dict[str, datetime] | None:
    """Load start/end dates from ``<setup>/projects/<label>/<label>.yml`` if available."""
    project_yml = Path(setup_dir) / "projects" / str(project_label) / f"{project_label}.yml"
    if not project_yml.exists():
        return None
    try:
        data = _read_yaml_file(project_yml) or {}
        start = datetime.fromisoformat(str(data.get("start_date")))
        end = datetime.fromisoformat(str(data.get("end_date")))
        return {"start": start, "end": end}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse project dates from {}: {}", project_yml, exc)
        return None
