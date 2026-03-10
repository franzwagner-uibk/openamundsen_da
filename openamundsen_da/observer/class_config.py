"""Strict loaders for observation class lists from project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from openamundsen_da.core.env import _read_yaml_file
from openamundsen_da.io.paths import find_project_yaml
from openamundsen_da.util.config_validators import require_mapping


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


def load_observation_classes(project_dir: Path, *, obs_key: str) -> dict[str, list[int]]:
    """Load class lists from project.obs.<obs_key>.classes."""
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = require_mapping(cfg.get("obs"), path="project.obs")
    obs_product_cfg = require_mapping(obs_cfg.get(obs_key), path=f"project.obs.{obs_key}")
    classes = require_mapping(obs_product_cfg.get("classes"), path=f"project.obs.{obs_key}.classes")
    out: dict[str, list[int]] = {}
    for key, values in classes.items():
        out[str(key)] = _require_ints(values, path=f"project.obs.{obs_key}.classes.{key}", allow_empty=True)
    return out


def load_wetsnow_classes(project_dir: Path) -> tuple[list[int], list[int], list[int]]:
    """Load strict wet-snow class mapping from project YAML."""
    classes = load_observation_classes(project_dir, obs_key="wetsnow")
    if "wet" not in classes:
        raise ValueError("Missing required configuration key: project.obs.wetsnow.classes.wet")
    if "valid" not in classes:
        raise ValueError("Missing required configuration key: project.obs.wetsnow.classes.valid")
    if "exclude" not in classes:
        raise ValueError("Missing required configuration key: project.obs.wetsnow.classes.exclude")
    wet = [int(v) for v in classes["wet"]]
    valid = [int(v) for v in classes["valid"]]
    exclude = [int(v) for v in classes["exclude"]]
    return wet, valid, exclude
