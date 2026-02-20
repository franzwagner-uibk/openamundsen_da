from __future__ import annotations

from pathlib import Path
from typing import Any

import ruamel.yaml


_YAML = ruamel.yaml.YAML(typ="safe")


def read_yaml_mapping(
    path: Path,
    *,
    error_cls: type[Exception] = RuntimeError,
    context: str = "YAML root",
) -> dict[str, Any]:
    """Read YAML and require a mapping at document root."""
    try:
        with Path(path).open("r", encoding="utf-8") as f:
            data = _YAML.load(f) or {}
    except Exception as exc:
        raise error_cls(f"Could not read YAML from {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise error_cls(f"{context} must be a mapping in {path}")
    return data
