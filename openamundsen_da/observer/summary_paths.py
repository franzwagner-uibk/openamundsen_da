"""Project-aware observation summary path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ruamel.yaml

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml


_SUMMARY_CONFIG: dict[str, tuple[str, str]] = {
    "scf_summary.csv": ("snowcover", "summary_csv"),
    "wet_snow_summary.csv": ("wetsnow", "summary_csv"),
    "wet_snow_line_diagnostics.csv": ("wetsnow", "wet_snow_line_diagnostics_csv"),
}


def default_fraction_obs_path(setup_dir: Path, project_name: str, filename: str) -> Path:
    """Return the legacy/default obs summary path for one fraction summary CSV."""
    setup_dir = Path(setup_dir)
    candidates = [
        setup_dir / "obs" / project_name / filename,
        setup_dir / "obs" / "summaries" / project_name / filename,
    ]
    if "-" in project_name:
        candidates.append(setup_dir / "obs" / project_name.replace("-", "_") / filename)
    elif "_" in project_name:
        candidates.append(setup_dir / "obs" / project_name.replace("_", "-") / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _project_obs_section(project_dir: Path) -> dict[str, Any]:
    cfg = _read_yaml_file(find_project_yaml(project_dir)) or {}
    obs = cfg.get("obs") or {}
    return obs if isinstance(obs, dict) else {}


def _resolve_configured_path(setup_dir: Path, raw: object) -> Path | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else Path(setup_dir) / path


def resolve_fraction_summary_path(setup_dir: Path, project_dir: Path, filename: str) -> Path:
    """Resolve a project observation summary path from project config or defaults.

    Project YAML can pin the source used by all downstream consumers:

    - ``obs.snowcover.summary_csv`` for ``scf_summary.csv``
    - ``obs.wetsnow.summary_csv`` for ``wet_snow_summary.csv``
    - ``obs.wetsnow.wet_snow_line_diagnostics_csv`` for WSLA diagnostics

    Relative configured paths are resolved against the setup root.
    """
    setup_dir = Path(setup_dir)
    project_dir = Path(project_dir)
    filename = str(filename)
    obs = _project_obs_section(project_dir)

    section_name, key = _SUMMARY_CONFIG.get(filename, ("", ""))
    section = obs.get(section_name) if section_name else None
    if isinstance(section, dict):
        configured = _resolve_configured_path(setup_dir, section.get(key))
        if configured is not None:
            return configured
        if filename == "wet_snow_line_diagnostics.csv":
            summary = _resolve_configured_path(setup_dir, section.get("summary_csv"))
            if summary is not None:
                return summary.parent / filename

    return default_fraction_obs_path(setup_dir, project_dir.name, filename)


def _path_for_yaml(setup_dir: Path, summary_csv: Path) -> str:
    summary_csv = Path(summary_csv)
    if not summary_csv.is_absolute():
        return summary_csv.as_posix()
    try:
        return summary_csv.relative_to(setup_dir).as_posix()
    except ValueError:
        return summary_csv.as_posix()


def record_fraction_summary_path(
    *,
    setup_dir: Path,
    project_dir: Path,
    filename: str,
    summary_csv: Path,
) -> None:
    """Persist the summary CSV path in project YAML for downstream stages."""
    if filename not in _SUMMARY_CONFIG:
        raise ValueError(f"Unsupported fraction summary filename: {filename}")

    project_yaml = find_project_yaml(project_dir)
    yaml = ruamel.yaml.YAML()
    yaml.default_flow_style = False
    with project_yaml.open("r", encoding="utf-8") as f:
        data = yaml.load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project YAML root must be a mapping: {project_yaml}")

    obs = data.setdefault("obs", {})
    if not isinstance(obs, dict):
        raise ValueError(f"Project YAML obs section must be a mapping: {project_yaml}")

    section_name, key = _SUMMARY_CONFIG[filename]
    section = obs.setdefault(section_name, {})
    if not isinstance(section, dict):
        raise ValueError(f"Project YAML obs.{section_name} section must be a mapping: {project_yaml}")
    section[key] = _path_for_yaml(Path(setup_dir), Path(summary_csv))

    with project_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(data, f)


__all__ = [
    "default_fraction_obs_path",
    "record_fraction_summary_path",
    "resolve_fraction_summary_path",
]
