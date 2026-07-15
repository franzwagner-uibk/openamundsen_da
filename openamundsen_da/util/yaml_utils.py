from __future__ import annotations

from pathlib import Path
from typing import Any

import ruamel.yaml


def read_yaml_mapping(
    path: Path,
    *,
    error_cls: type[Exception] = RuntimeError,
    context: str = "YAML root",
) -> dict[str, Any]:
    """Read YAML and require a mapping at document root."""
    yaml = ruamel.yaml.YAML(typ="safe")
    yaml.allow_duplicate_keys = False
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.load(f) or {}
    except Exception as exc:
        raise error_cls(f"Could not read YAML from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise error_cls(f"{context} must be a mapping in {path}")
    return data
