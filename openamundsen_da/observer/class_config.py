"""Shared strict loaders and resolvers for observer class mappings."""

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


def _require_strings(
    values: object,
    *,
    path: str,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{path} must be a list of strings")
    out: list[str] = []
    for value in values:
        s = str(value).strip()
        if not s:
            raise ValueError(f"{path} contains an empty string")
        out.append(s)
    if not out and not allow_empty:
        raise ValueError(f"{path} must contain at least one string")
    return out


def load_observation_class_groups(project_dir: Path, *, obs_key: str) -> dict[str, tuple[int, ...]]:
    """Load generic class groups from project.obs.<obs_key>.classes."""
    cfg = require_mapping(_read_yaml_file(find_project_yaml(project_dir)) or {}, path="project")
    obs_cfg = require_mapping(cfg.get("obs"), path="project.obs")
    obs_product_cfg = require_mapping(obs_cfg.get(obs_key), path=f"project.obs.{obs_key}")
    classes = require_mapping(obs_product_cfg.get("classes"), path=f"project.obs.{obs_key}.classes")
    out: dict[str, tuple[int, ...]] = {}
    for key, values in classes.items():
        out[str(key)] = tuple(_require_ints(values, path=f"project.obs.{obs_key}.classes.{key}", allow_empty=True))
    return out


def resolve_class_values(
    *,
    path: str,
    class_groups: dict[str, tuple[int, ...]] | None,
    raw_classes: object,
    groups: object,
    allow_empty: bool = False,
) -> tuple[int, ...]:
    """Resolve class IDs from either raw class list, named groups, or both."""
    resolved: list[int] = []

    if raw_classes is not None:
        resolved.extend(_require_ints(raw_classes, path=f"{path}.classes", allow_empty=True))

    if groups is not None:
        if class_groups is None:
            raise ValueError(f"{path}.groups is not supported for this rule/source")
        names = _require_strings(groups, path=f"{path}.groups", allow_empty=True)
        for name in names:
            if name not in class_groups:
                raise ValueError(
                    f"{path}.groups references unknown class group '{name}'. "
                    "Available groups: "
                    + ", ".join(sorted(class_groups.keys()))
                )
            resolved.extend(class_groups[name])

    uniq = sorted(set(int(v) for v in resolved))
    if not uniq and not allow_empty:
        raise ValueError(f"{path} must define at least one class via '.classes' and/or '.groups'")
    return tuple(uniq)


def load_wetsnow_classes(project_dir: Path) -> tuple[list[int], list[int], list[int]]:
    """Load strict wet-snow class mapping from project YAML."""
    classes = load_observation_class_groups(project_dir, obs_key="wetsnow")
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
