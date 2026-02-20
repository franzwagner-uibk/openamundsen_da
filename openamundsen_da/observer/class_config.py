"""Shared strict loaders for observer class mappings."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml


def _require_mapping(raw: object, *, path: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at {path}")
    return raw


def _require_ints(
    values: object,
    *,
    path: str,
    allow_empty: bool = False,
) -> list[int]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{path} must be a list of integers")
    out: list[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception as exc:
            raise ValueError(f"{path} contains non-integer value: {value!r}") from exc
    if not out and not allow_empty:
        raise ValueError(f"{path} must contain at least one integer")
    return out


def load_wetsnow_classes(project_dir: Path) -> tuple[list[int], list[int], list[int]]:
    """Load strict wet-snow class mapping from project YAML."""
    cfg = _require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = _require_mapping(cfg.get("obs"), path="project.obs")
    wet_cfg = _require_mapping(obs_cfg.get("wetsnow"), path="project.obs.wetsnow")
    classes = _require_mapping(wet_cfg.get("classes"), path="project.obs.wetsnow.classes")
    if "wet" not in classes:
        raise ValueError("Missing required configuration key: project.obs.wetsnow.classes.wet")
    if "valid" not in classes:
        raise ValueError("Missing required configuration key: project.obs.wetsnow.classes.valid")
    if "exclude" not in classes:
        raise ValueError("Missing required configuration key: project.obs.wetsnow.classes.exclude")
    wet = _require_ints(classes["wet"], path="project.obs.wetsnow.classes.wet")
    valid = _require_ints(classes["valid"], path="project.obs.wetsnow.classes.valid")
    exclude = _require_ints(classes["exclude"], path="project.obs.wetsnow.classes.exclude", allow_empty=True)
    return wet, valid, exclude

