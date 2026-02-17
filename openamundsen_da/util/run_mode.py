from __future__ import annotations

"""Helpers for project execution mode markers.

The project YAML persists a top-level `run_mode` marker to guard against
accidentally running a project with the wrong workflow entrypoint.

Legacy compatibility: `data_assimilation.run_mode` is still read when the
top-level key is missing.
"""

from pathlib import Path

from openamundsen_da.core.constants import DA_BLOCK
from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml

_RUN_MODE_KEY = "run_mode"
_VALID_RUN_MODES = {"single", "subdomain"}


def _normalize_mode(run_mode: str) -> str:
    mode = str(run_mode or "").strip().lower()
    if mode not in _VALID_RUN_MODES:
        raise ValueError(f"Invalid run_mode {run_mode!r}. Expected one of: {', '.join(sorted(_VALID_RUN_MODES))}")
    return mode


def read_run_mode(project_dir: Path) -> str | None:
    """Return normalized run_mode from project YAML, or None when missing."""
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    raw = cfg.get(_RUN_MODE_KEY)
    if raw is None:
        da_cfg = cfg.get(DA_BLOCK) or {}
        raw = da_cfg.get(_RUN_MODE_KEY)
    if raw is None:
        return None
    return _normalize_mode(str(raw))


def write_run_mode(project_dir: Path, run_mode: str) -> str:
    """Persist run_mode in project YAML and return the normalized mode."""
    mode = _normalize_mode(run_mode)
    project_yaml = find_project_yaml(project_dir)
    cfg = _read_yaml_file(project_yaml) or {}
    cfg[_RUN_MODE_KEY] = mode
    da_cfg = cfg.get(DA_BLOCK)
    if isinstance(da_cfg, dict) and _RUN_MODE_KEY in da_cfg:
        da_cfg = dict(da_cfg)
        da_cfg.pop(_RUN_MODE_KEY, None)
        cfg[DA_BLOCK] = da_cfg

    import ruamel.yaml as _yaml

    yaml = _yaml.YAML()
    with project_yaml.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f)
    return mode


def ensure_run_mode(
    project_dir: Path,
    *,
    expected: str,
    write_if_missing: bool,
) -> str:
    """Ensure a project uses the expected run_mode.

    If the marker is missing and `write_if_missing` is true, it is written.
    Otherwise, a ValueError is raised.
    """
    normalized_expected = _normalize_mode(expected)
    current = read_run_mode(project_dir)
    if current is None:
        if write_if_missing:
            return write_run_mode(project_dir, normalized_expected)
        raise ValueError(
            f"Project {project_dir} has no '{_RUN_MODE_KEY}' marker. "
            f"Expected '{normalized_expected}'."
        )
    if current != normalized_expected:
        raise ValueError(
            f"Project {project_dir} is marked as run_mode='{current}', "
            f"but this command requires run_mode='{normalized_expected}'."
        )
    return current


__all__ = ["ensure_run_mode", "read_run_mode", "write_run_mode"]
